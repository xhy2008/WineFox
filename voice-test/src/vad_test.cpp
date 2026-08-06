// voice-test/src/vad_test.cpp
//
// VAD benchmark subcommand. Uses TEN-VAD-GGML.cpp (pure C++17 port of
// ten-vad, GGML model format) and adds a segment state machine on top,
// since the C API only exposes a per-frame speech probability
// (no min_speech / min_silence / max_speech).
//
// Metrics reported:
//   - Detected segments (start_s, end_s, dur_s)
//   - Total processing time, RTF (real-time factor)
//   - Per-frame latency: min / max / avg / p50 / p95
//   - Speech ratio
//   - Optional comparison vs. ground-truth reference file
//
// Usage:
//   voice_test vad <wav> [--model <ten-vad-ggml.bin>] [--threshold 0.5] [--hop 256]
//                        [--min-speech 0.25] [--min-silence 0.30]
//                        [--max-speech 30.0]
//                        [--reference <vad_reference.txt>]
//                        [--out <result.txt>]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "ten_vad/ten_vad.h"

#include "common.h"

namespace {

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct VadArgs {
    std::string wav_path;
    std::string model_path = "models/vad/ten-vad-ggml.bin";
    std::string reference_path;   // optional ground-truth file
    std::string out_path;         // optional output file
    float  threshold       = 0.5f;
    int    hop_size        = 256;       // samples per frame (16ms @ 16kHz)
    float  min_speech_s    = 0.25f;     // min speech duration to confirm a segment
    float  min_silence_s   = 0.30f;     // trailing silence to finalize a segment
    float  max_speech_s    = 30.0f;     // cap very long segments
};

bool parse_args(const std::vector<std::string>& args, VadArgs& out) {
    if (args.empty()) {
        std::fprintf(stderr, "vad: missing <wav> argument\n");
        return false;
    }
    out.wav_path = args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "vad: %s requires a value\n", name);
                return std::string();
            }
            return args[++i];
        };
        if (a == "--model")          out.model_path      = next("--model");
        else if (a == "--threshold") out.threshold      = std::stof(next("--threshold"));
        else if (a == "--hop")         out.hop_size       = std::stoi(next("--hop"));
        else if (a == "--min-speech")  out.min_speech_s   = std::stof(next("--min-speech"));
        else if (a == "--min-silence") out.min_silence_s  = std::stof(next("--min-silence"));
        else if (a == "--max-speech")  out.max_speech_s   = std::stof(next("--max-speech"));
        else if (a == "--reference")   out.reference_path = next("--reference");
        else if (a == "--out")         out.out_path       = next("--out");
        else {
            std::fprintf(stderr, "vad: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (out.hop_size <= 0) {
        std::fprintf(stderr, "vad: --hop must be positive\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Segment state machine
//
// ten-vad-ggml emits a per-frame speech probability; we threshold it into a
// binary flag, then wrap it with a small state machine that consolidates
// frames into speech segments using configurable
// min_speech_duration / min_silence_duration / max_speech_duration, matching
// the conventions used by sherpa-onnx / webrtc-vad style front-ends.
// ---------------------------------------------------------------------------
struct Segment {
    double start_s;
    double end_s;
};

struct Segmenter {
    // Configuration (in frames).
    int min_speech_frames;
    int min_silence_frames;
    int max_speech_frames;

    // State.
    enum class State { SILENCE, PENDING_SPEECH, IN_SPEECH };
    State  state = State::SILENCE;

    int    pending_start_frame = 0;   // where the current speech run began (pre-confirmation)
    int    confirmed_start     = 0;   // confirmed segment start (after min_speech)
    int    last_speech_frame   = 0;   // last frame with flag==1 within current segment
    int    consec_speech       = 0;   // consecutive flag==1 frames
    int    consec_silence      = 0;   // consecutive flag==0 frames within IN_SPEECH
    int    speech_frame_count  = 0;   // frames since confirmed_start (for max_speech cap)

    std::vector<Segment> segments;

    explicit Segmenter(int min_sp, int min_si, int max_sp)
        : min_speech_frames(min_sp), min_silence_frames(min_si), max_speech_frames(max_sp) {}

    void feed(int frame_idx, int flag) {
        switch (state) {
        case State::SILENCE:
            if (flag) {
                if (consec_speech == 0) pending_start_frame = frame_idx;
                ++consec_speech;
                consec_silence = 0;
                if (consec_speech >= min_speech_frames) {
                    confirmed_start = pending_start_frame;
                    last_speech_frame = frame_idx;
                    speech_frame_count = consec_speech;
                    state = State::IN_SPEECH;
                }
            } else {
                consec_speech = 0;
            }
            break;

        case State::IN_SPEECH:
            if (flag) {
                ++speech_frame_count;
                last_speech_frame = frame_idx;
                consec_silence = 0;
                if (speech_frame_count >= max_speech_frames) {
                    // Force-finalize on max_speech cap.
                    segments.push_back({double(confirmed_start), double(last_speech_frame + 1)});
                    state = State::SILENCE;
                    consec_speech = 0;
                    consec_silence = 0;
                }
            } else {
                ++consec_silence;
                if (consec_silence >= min_silence_frames) {
                    segments.push_back({double(confirmed_start), double(last_speech_frame + 1)});
                    state = State::SILENCE;
                    consec_speech = 0;
                    consec_silence = 0;
                }
            }
            break;
        }
    }

    void finish() {
        if (state == State::IN_SPEECH) {
            segments.push_back({double(confirmed_start), double(last_speech_frame + 1)});
            state = State::SILENCE;
            consec_speech = 0;
            consec_silence = 0;
        }
    }
};

// ---------------------------------------------------------------------------
// Reference file parsing
//   Format (one segment per line, comma-separated):
//     start_s,end_s,duration_s,text
//   Lines starting with '#' are comments.
// ---------------------------------------------------------------------------
std::vector<Segment> load_reference(const std::string& path) {
    std::vector<Segment> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Replace commas with spaces for easy parsing.
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        double s, e, d;
        if (ss >> s >> e >> d) {
            out.push_back({s, e});
        }
    }
    return out;
}

// Intersection-over-union between two time intervals.
double iou(const Segment& a, const Segment& b) {
    double s = std::max(a.start_s, b.start_s);
    double e = std::min(a.end_s, b.end_s);
    double inter = std::max(0.0, e - s);
    double union_ = (a.end_s - a.start_s) + (b.end_s - b.start_s) - inter;
    return union_ > 0.0 ? inter / union_ : 0.0;
}

// Frame-level precision/recall/F1 by sampling at fine granularity.
struct FrameMetrics {
    double precision;
    double recall;
    double f1;
};

FrameMetrics frame_level_metrics(const std::vector<Segment>& pred,
                                  const std::vector<Segment>& ref,
                                  double total_duration_s) {
    // 10ms granularity is fine enough for VAD comparison.
    const double dt = 0.01;
    int n = std::max(1, int(total_duration_s / dt) + 1);
    std::vector<char> is_pred(n, 0), is_ref(n, 0);
    auto mark = [&](std::vector<char>& v, const std::vector<Segment>& segs) {
        for (const auto& s : segs) {
            int a = std::max(0, int(s.start_s / dt));
            int b = std::min(n, int(s.end_s / dt + 0.5));
            for (int i = a; i < b; ++i) v[i] = 1;
        }
    };
    mark(is_pred, pred);
    mark(is_ref, ref);
    int tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < n; ++i) {
        if (is_pred[i] && is_ref[i]) ++tp;
        else if (is_pred[i] && !is_ref[i]) ++fp;
        else if (!is_pred[i] && is_ref[i]) ++fn;
    }
    double p = tp + fp > 0 ? double(tp) / (tp + fp) : 0.0;
    double r = tp + fn > 0 ? double(tp) / (tp + fn) : 0.0;
    double f = p + r > 0 ? 2.0 * p * r / (p + r) : 0.0;
    return {p, r, f};
}

// Percentile helper (assumes sorted input).
double percentile(std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double idx = q * (sorted.size() - 1);
    size_t lo = size_t(std::floor(idx));
    size_t hi = size_t(std::min<size_t>(lo + 1, sorted.size() - 1));
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

}  // namespace

int run_vad(const std::vector<std::string>& args) {
    VadArgs va;
    if (!parse_args(args, va)) return 1;

    Pcm pcm;
    if (!load_wav(va.wav_path, pcm)) {
        std::fprintf(stderr, "vad: failed to load %s\n", va.wav_path.c_str());
        return 1;
    }
    if (pcm.sample_rate != kSampleRate) {
        std::fprintf(stderr,
                     "vad: %s has sample_rate=%d, expected %d. Resample first.\n",
                     va.wav_path.c_str(), pcm.sample_rate, kSampleRate);
        return 1;
    }

    const double audio_dur_s = double(pcm.num_samples) / kSampleRate;
    const int    hop         = va.hop_size;
    const int    n_frames    = int((pcm.num_samples + hop - 1) / hop);

    // Frame thresholds derived from seconds.
    const double frame_dur_s = double(hop) / kSampleRate;
    const int min_speech_frames  = std::max(1, int(va.min_speech_s  / frame_dur_s + 0.5));
    const int min_silence_frames = std::max(1, int(va.min_silence_s / frame_dur_s + 0.5));
    const int max_speech_frames  = std::max(min_speech_frames, int(va.max_speech_s / frame_dur_s + 0.5));

    std::printf("voice_test vad\n");
    std::printf("  vad model       : %s\n", va.model_path.c_str());
    std::printf("  input           : %s\n", va.wav_path.c_str());
    std::printf("  sample_rate     : %d\n", pcm.sample_rate);
    std::printf("  num_samples     : %lld (%.3fs)\n",
                (long long)pcm.num_samples, audio_dur_s);
    std::printf("  hop_size        : %d samples (%.1f ms)\n", hop, frame_dur_s * 1000.0);
    std::printf("  threshold       : %.3f\n", va.threshold);
    std::printf("  min_speech      : %.3fs (%d frames)\n", va.min_speech_s, min_speech_frames);
    std::printf("  min_silence     : %.3fs (%d frames)\n", va.min_silence_s, min_silence_frames);
    std::printf("  max_speech      : %.3fs (%d frames)\n", va.max_speech_s, max_speech_frames);
    std::printf("  total_frames    : %d\n", n_frames);
    std::printf("\n");

    // Initialize ten-vad (TEN-VAD-GGML).
    ten_vad_ctx* handle = ten_vad_create(va.model_path.c_str());
    if (!handle) {
        std::fprintf(stderr, "vad: ten_vad_create failed (model=%s)\n", va.model_path.c_str());
        return 1;
    }

    Segmenter seg(min_speech_frames, min_silence_frames, max_speech_frames);
    std::vector<double> frame_ms;
    frame_ms.reserve(size_t(n_frames));

    // Process frame-by-frame and time each call.
    std::vector<int16_t> frame_buf(size_t(hop), 0);
    auto t_total_start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_frames; ++i) {
        int64_t off = int64_t(i) * hop;
        int64_t remaining = pcm.num_samples - off;
        if (remaining <= 0) break;
        int copy = int(std::min<int64_t>(remaining, hop));
        std::memcpy(frame_buf.data(), pcm.samples.data() + off, size_t(copy) * sizeof(int16_t));
        if (copy < hop) {
            // Zero-pad the trailing partial frame.
            std::memset(frame_buf.data() + copy, 0, size_t(hop - copy) * sizeof(int16_t));
        }

        auto t0 = std::chrono::steady_clock::now();
        float prob = ten_vad_process(handle, frame_buf.data(), size_t(hop));
        int   flag = (prob >= va.threshold) ? 1 : 0;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        frame_ms.push_back(ms);
        seg.feed(i, flag);
    }
    seg.finish();
    auto t_total_end = std::chrono::steady_clock::now();
    ten_vad_destroy(handle);

    const double total_proc_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
    const double total_proc_s  = total_proc_ms / 1000.0;
    const double rtf           = total_proc_s / audio_dur_s;

    // Per-frame latency stats.
    double min_ms = 0, max_ms = 0, avg_ms = 0, p50_ms = 0, p95_ms = 0;
    if (!frame_ms.empty()) {
        std::vector<double> sorted = frame_ms;
        std::sort(sorted.begin(), sorted.end());
        min_ms  = sorted.front();
        max_ms  = sorted.back();
        avg_ms  = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        p50_ms  = percentile(sorted, 0.50);
        p95_ms  = percentile(sorted, 0.95);
    }

    // Convert segment frame indices to seconds.
    std::vector<Segment> segs_s;
    double total_speech_s = 0.0;
    for (const auto& s : seg.segments) {
        double st = s.start_s * frame_dur_s;
        double en = s.end_s   * frame_dur_s;
        segs_s.push_back({st, en});
        total_speech_s += (en - st);
    }

    // ---- Print detected segments ----
    std::printf("Detected segments (%zu):\n", segs_s.size());
    for (size_t i = 0; i < segs_s.size(); ++i) {
        std::printf("  [#%zu] start=%7.3fs  end=%7.3fs  dur=%6.3fs\n",
                    i + 1, segs_s[i].start_s, segs_s[i].end_s,
                    segs_s[i].end_s - segs_s[i].start_s);
    }
    std::printf("\n");

    // ---- Print performance metrics ----
    std::printf("Performance:\n");
    std::printf("  audio duration    : %7.3f s\n", audio_dur_s);
    std::printf("  processing time   : %7.3f s\n", total_proc_s);
    std::printf("  RTF               : %7.4f  (1 / %.2fx realtime)\n", rtf, rtf > 0 ? 1.0 / rtf : 0.0);
    std::printf("  frames processed  : %d\n", int(frame_ms.size()));
    std::printf("  frame latency ms  : min=%.4f  avg=%.4f  p50=%.4f  p95=%.4f  max=%.4f\n",
                min_ms, avg_ms, p50_ms, p95_ms, max_ms);
    std::printf("  speech ratio      : %.2f%%  (%.3fs speech / %.3fs total)\n",
                total_speech_s / std::max(0.001, audio_dur_s) * 100.0,
                total_speech_s, audio_dur_s);
    std::printf("\n");

    // ---- Optional: compare with reference ----
    int rc = 0;
    if (!va.reference_path.empty()) {
        auto ref = load_reference(va.reference_path);
        if (ref.empty()) {
            std::fprintf(stderr, "vad: could not parse reference %s\n", va.reference_path.c_str());
            rc = 1;
        } else {
            auto fm = frame_level_metrics(segs_s, ref, audio_dur_s);

            // Segment-level matching: a predicted segment is a TP if it has
            // IoU >= 0.3 with some reference segment.
            const double iou_match = 0.3;
            int tp = 0, fp = 0, fn = 0;
            std::vector<char> ref_matched(ref.size(), 0);
            for (const auto& p : segs_s) {
                bool matched = false;
                for (size_t j = 0; j < ref.size(); ++j) {
                    if (iou(p, ref[j]) >= iou_match) {
                        ref_matched[j] = 1;
                        matched = true;
                    }
                }
                if (matched) ++tp; else ++fp;
            }
            for (size_t j = 0; j < ref.size(); ++j) if (!ref_matched[j]) ++fn;
            double seg_p = tp + fp > 0 ? double(tp) / (tp + fp) : 0.0;
            double seg_r = tp + fn > 0 ? double(tp) / (tp + fn) : 0.0;
            double seg_f = seg_p + seg_r > 0 ? 2.0 * seg_p * seg_r / (seg_p + seg_r) : 0.0;

            std::printf("Comparison vs reference (%s):\n", va.reference_path.c_str());
            std::printf("  reference segments : %zu\n", ref.size());
            std::printf("  predicted segments : %zu\n", segs_s.size());
            std::printf("  segment-level (IoU>=%.1f): TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f\n",
                        iou_match, tp, fp, fn, seg_p, seg_r, seg_f);
            std::printf("  frame-level  (10ms)     : P=%.3f R=%.3f F1=%.3f\n",
                        fm.precision, fm.recall, fm.f1);

            // Per-segment IoU table.
            std::printf("  per-segment IoU:\n");
            for (size_t i = 0; i < ref.size(); ++i) {
                double best = 0.0; size_t best_j = 0;
                for (size_t j = 0; j < segs_s.size(); ++j) {
                    double v = iou(ref[i], segs_s[j]);
                    if (v > best) { best = v; best_j = j; }
                }
                std::printf("    ref[%zu] [%.3f, %.3f] <-> pred[%zu] [%.3f, %.3f]  IoU=%.3f\n",
                            i, ref[i].start_s, ref[i].end_s,
                            best_j, best < 0.001 ? 0.0 : segs_s[best_j].start_s,
                            best < 0.001 ? 0.0 : segs_s[best_j].end_s, best);
            }
            std::printf("\n");
        }
    }

    // ---- Optional: write machine-readable output ----
    if (!va.out_path.empty()) {
        std::ofstream of(va.out_path);
        if (!of) {
            std::fprintf(stderr, "vad: could not write %s\n", va.out_path.c_str());
            return 1;
        }
        of << "# voice_test vad output\n";
        of << "# input=" << va.wav_path
           << "  sr=" << kSampleRate
           << "  samples=" << pcm.num_samples
           << "  dur=" << audio_dur_s << "s\n";
        of << "# hop=" << hop
           << "  threshold=" << va.threshold
           << "  min_speech=" << va.min_speech_s
           << "  min_silence=" << va.min_silence_s << "\n";
        of << "# rtf=" << rtf
           << "  proc_s=" << total_proc_s
           << "  frame_avg_ms=" << avg_ms
           << "  frame_p95_ms=" << p95_ms << "\n";
        of << "# segments: start_s,end_s,duration_s\n";
        for (const auto& s : segs_s) {
            of << s.start_s << "," << s.end_s << "," << (s.end_s - s.start_s) << "\n";
        }
        std::printf("Wrote %s\n", va.out_path.c_str());
    }

    std::printf("vad: done.\n");
    return rc;
}
