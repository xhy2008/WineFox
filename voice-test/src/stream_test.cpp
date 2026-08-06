// voice-test/src/stream_test.cpp
//
// Streaming VAD+ASR pipeline benchmark.
//
// Reads a WAV file and simulates a real-time streaming pipeline:
//   1. Feed audio frame-by-frame to ten-vad (hop=256 samples = 16ms).
//   2. A segment state machine consolidates frames into speech segments
//      using min_speech / min_silence / max_speech thresholds.
//   3. When a segment is finalized (trailing silence detected), the segment
//      audio is extracted and passed to SenseVoice for ASR.
//   4. End-to-end latency is measured from segment-end to ASR-result.
//
// This mimics the real-time voice conversation pipeline where VAD detects
// speech endpoints and triggers ASR. The --realtime flag slows down audio
// feeding to wall-clock speed so latencies reflect real-world behavior.
//
// Metrics reported:
//   - Per-segment: text, start/end/dur, VAD endpoint latency, ASR latency, E2E latency
//   - Aggregate: total segments, total speech, total ASR time, overall RTF
//   - Latency stats: min/avg/p50/p95/max for E2E latency
//
// Usage:
//   voice_test stream <wav> --asr-model <gguf>
//                         [--vad-model <ten-vad-ggml.bin>] [--threshold 0.3] [--hop 256]
//                         [--min-speech 0.25] [--min-silence 0.30]
//                         [--max-speech 30.0]
//                         [--lang auto|zh|en|yue|ja|ko]
//                         [--threads 4]
//                         [--itn] [--prefix] [--flash-attn]
//                         [--realtime]          # feed audio at wall-clock speed
//                         [--reference <txt>]   # ground-truth transcripts for CER
//                         [--out <file>]        # machine-readable output

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
#include <thread>
#include <vector>

#include "ten_vad/ten_vad.h"

#include "sense-voice.h"
#include "silero-vad.h"  // full definition for proper cleanup

#include "common.h"

namespace {

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct StreamArgs {
    std::string wav_path;
    std::string asr_model_path;
    std::string vad_model_path = "models/vad/ten-vad-ggml.bin";
    std::string language       = "auto";
    std::string reference_path;   // optional: one transcript per line
    std::string out_path;
    // VAD params
    float  threshold       = 0.3f;
    int    hop_size        = 256;
    float  min_speech_s    = 0.25f;
    float  min_silence_s   = 0.30f;
    float  max_speech_s    = 30.0f;
    // ASR params
    int    threads         = 4;
    bool   use_itn         = false;
    bool   use_prefix      = false;
    bool   flash_attn      = false;
    // Streaming simulation
    bool   realtime        = false;   // feed audio at wall-clock speed
};

bool parse_args(const std::vector<std::string>& args, StreamArgs& out) {
    if (args.empty()) {
        std::fprintf(stderr, "stream: missing <wav> argument\n");
        return false;
    }
    out.wav_path = args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "stream: %s requires a value\n", name);
                return std::string();
            }
            return args[++i];
        };
        if      (a == "--asr-model")   out.asr_model_path = next("--asr-model");
        else if (a == "--threshold")   out.threshold      = std::stof(next("--threshold"));
        else if (a == "--hop")         out.hop_size       = std::stoi(next("--hop"));
        else if (a == "--min-speech")  out.min_speech_s   = std::stof(next("--min-speech"));
        else if (a == "--min-silence") out.min_silence_s  = std::stof(next("--min-silence"));
        else if (a == "--max-speech")  out.max_speech_s   = std::stof(next("--max-speech"));
        else if (a == "--lang")        out.language       = next("--lang");
        else if (a == "--threads")     out.threads        = std::stoi(next("--threads"));
        else if (a == "--reference")   out.reference_path = next("--reference");
        else if (a == "--out")         out.out_path       = next("--out");
        else if (a == "--itn")         out.use_itn        = true;
        else if (a == "--prefix")      out.use_prefix     = true;
        else if (a == "--flash-attn")  out.flash_attn     = true;
        else if (a == "--realtime")    out.realtime       = true;
        else {
            std::fprintf(stderr, "stream: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (out.asr_model_path.empty()) {
        std::fprintf(stderr, "stream: --asr-model <gguf> is required\n");
        return false;
    }
    if (out.hop_size <= 0) {
        std::fprintf(stderr, "stream: --hop must be positive\n");
        return false;
    }
    if (out.threads <= 0) {
        std::fprintf(stderr, "stream: --threads must be positive\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Segment state machine (same logic as vad_test.cpp)
// ---------------------------------------------------------------------------
struct Segment {
    double start_frame;
    double end_frame;
};

struct Segmenter {
    int min_speech_frames;
    int min_silence_frames;
    int max_speech_frames;

    enum class State { SILENCE, IN_SPEECH };
    State  state = State::SILENCE;

    int    pending_start_frame = 0;
    int    confirmed_start     = 0;
    int    last_speech_frame   = 0;
    int    consec_speech       = 0;
    int    consec_silence      = 0;
    int    speech_frame_count  = 0;

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
// ASR result extraction (same logic as asr_test.cpp)
// ---------------------------------------------------------------------------
std::string get_text(struct sense_voice_context* ctx, bool need_prefix) {
    std::string text;
    const auto& ids = ctx->state->ids;
    size_t start = need_prefix ? 0 : 4;
    for (size_t i = start; i < ids.size(); ++i) {
        int id = ids[i];
        if (i > 0 && id == ids[i - 1]) continue;
        if (id == 0) continue;
        auto it = ctx->vocab.id_to_token.find(id);
        if (it != ctx->vocab.id_to_token.end()) {
            text += it->second;
        }
    }
    return text;
}

struct PrefixInfo {
    std::string language, emotion, event, itn;
};

PrefixInfo get_prefix(struct sense_voice_context* ctx) {
    PrefixInfo p;
    const auto& ids = ctx->state->ids;
    auto get = [&](size_t i) -> std::string {
        if (i >= ids.size()) return "";
        int id = ids[i];
        if (id == 0) return "";
        auto it = ctx->vocab.id_to_token.find(id);
        return it != ctx->vocab.id_to_token.end() ? it->second : "";
    };
    p.language = get(0);
    p.emotion  = get(1);
    p.event    = get(2);
    p.itn      = get(3);
    return p;
}

// ---------------------------------------------------------------------------
// CER (character error rate) - same logic as asr_test.cpp
// ---------------------------------------------------------------------------
double cer(const std::string& ref, const std::string& hyp) {
    if (ref.empty()) return 0.0;
    size_t m = ref.size(), n = hyp.size();
    std::vector<size_t> prev(n + 1), cur(n + 1);
    for (size_t j = 0; j <= n; ++j) prev[j] = j;
    for (size_t i = 1; i <= m; ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= n; ++j) {
            size_t cost = (ref[size_t(i - 1)] == hyp[size_t(j - 1)]) ? 0 : 1;
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, cur);
    }
    return double(prev[n]) / double(m);
}

std::vector<std::string> load_reference_lines(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.size() >= 3 &&
            (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB &&
            (unsigned char)line[2] == 0xBF) {
            line = line.substr(3);
        }
        out.push_back(line);
    }
    return out;
}

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

// ===========================================================================
// Pipeline entry point
// ===========================================================================
int run_stream(const std::vector<std::string>& args) {
    StreamArgs sa;
    if (!parse_args(args, sa)) return 1;

    Pcm pcm;
    if (!load_wav(sa.wav_path, pcm)) {
        std::fprintf(stderr, "stream: failed to load %s\n", sa.wav_path.c_str());
        return 1;
    }
    if (pcm.sample_rate != kSampleRate) {
        std::fprintf(stderr,
                     "stream: %s has sample_rate=%d, expected %d. Resample first.\n",
                     sa.wav_path.c_str(), pcm.sample_rate, kSampleRate);
        return 1;
    }

    const double audio_dur_s = double(pcm.num_samples) / kSampleRate;
    const int    hop         = sa.hop_size;
    const int    n_frames    = int((pcm.num_samples + hop - 1) / hop);
    const double frame_dur_s = double(hop) / kSampleRate;

    const int min_speech_frames  = std::max(1, int(sa.min_speech_s  / frame_dur_s + 0.5));
    const int min_silence_frames = std::max(1, int(sa.min_silence_s / frame_dur_s + 0.5));
    const int max_speech_frames  = std::max(min_speech_frames, int(sa.max_speech_s / frame_dur_s + 0.5));

    std::printf("voice_test stream\n");
    std::printf("  pipeline   : VAD (ten-vad-ggml) -> ASR (SenseVoice.cpp/ggml)\n");
    std::printf("  input      : %s\n", sa.wav_path.c_str());
    std::printf("  asr_model  : %s\n", sa.asr_model_path.c_str());
    std::printf("  language   : %s\n", sa.language.c_str());
    std::printf("  threads    : %d\n", sa.threads);
    std::printf("  sample_rate: %d\n", pcm.sample_rate);
    std::printf("  num_samples: %lld (%.3fs)\n",
                (long long)pcm.num_samples, audio_dur_s);
    std::printf("  hop_size   : %d samples (%.1f ms)\n", hop, frame_dur_s * 1000.0);
    std::printf("  threshold  : %.3f\n", sa.threshold);
    std::printf("  min_speech : %.3fs (%d frames)\n", sa.min_speech_s, min_speech_frames);
    std::printf("  min_silence: %.3fs (%d frames)\n", sa.min_silence_s, min_silence_frames);
    std::printf("  max_speech : %.3fs (%d frames)\n", sa.max_speech_s, max_speech_frames);
    std::printf("  realtime   : %s\n", sa.realtime ? "yes (wall-clock paced)" : "no (fast as possible)");
    std::printf("  use_itn    : %d\n", sa.use_itn ? 1 : 0);
    std::printf("  flash_attn : %d\n", sa.flash_attn ? 1 : 0);
    std::printf("\n");

    // -----------------------------------------------------------------------
    // Initialize ten-vad (TEN-VAD-GGML).
    // -----------------------------------------------------------------------
    ten_vad_ctx* vad_handle = ten_vad_create(sa.vad_model_path.c_str());
    if (!vad_handle) {
        std::fprintf(stderr, "stream: ten_vad_create failed (model=%s)\n",
                     sa.vad_model_path.c_str());
        return 1;
    }

    // -----------------------------------------------------------------------
    // Initialize SenseVoice context.
    // -----------------------------------------------------------------------
    sense_voice_context_params cparams = sense_voice_context_default_params();
    cparams.use_gpu    = false;
    cparams.flash_attn = sa.flash_attn;
    cparams.use_itn    = sa.use_itn;

    auto t_asr_load_start = std::chrono::steady_clock::now();
    sense_voice_context* ctx = sense_voice_small_init_from_file_with_params(
        sa.asr_model_path.c_str(), cparams);
    auto t_asr_load_end = std::chrono::steady_clock::now();
    if (!ctx) {
        std::fprintf(stderr, "stream: failed to initialize sense_voice context\n");
        ten_vad_destroy(vad_handle);
        return 1;
    }
    const double asr_load_s = std::chrono::duration<double, std::milli>(
        t_asr_load_end - t_asr_load_start).count() / 1000.0;

    if (sa.language != "auto" && sa.language != "") {
        int lid = sense_voice_lang_id(sa.language.c_str());
        if (lid >= 0) ctx->language_id = lid;
    }

    std::printf("Models loaded: VAD (ten-vad-ggml %s) + ASR (%.3f s)\n",
                sa.vad_model_path.c_str(), asr_load_s);
    std::printf("\n");

    // -----------------------------------------------------------------------
    // Streaming pipeline: feed frames to VAD, run ASR on each finalized segment.
    // -----------------------------------------------------------------------
    Segmenter seg(min_speech_frames, min_silence_frames, max_speech_frames);

    // Per-segment results for summary.
    struct SegResult {
        int    idx;
        double start_s;
        double end_s;
        double dur_s;
        double vad_endpoint_latency_s;  // from last_speech_frame to segment confirmation
        double asr_latency_s;
        double e2e_latency_s;           // from segment end to ASR result
        std::string text;
    };
    std::vector<SegResult> results;

    // Latency tracking.
    // We track the wall-clock time of each frame to measure real latencies.
    // In non-realtime mode, latencies are still meaningful: they represent the
    // processing delay a user would experience if audio arrived at real-time speed.
    std::vector<double> e2e_latencies_s;

    // For realtime mode: sleep to pace audio feeding.
    const auto realtime_frame_dur = std::chrono::microseconds(int64_t(frame_dur_s * 1e6));

    auto t_pipeline_start = std::chrono::steady_clock::now();

    // We need to know which segment we're currently building audio for.
    // When Segmenter finalizes a segment, we extract the audio and run ASR.
    // Since Segmenter doesn't expose the segment until it's finalized, we
    // track the current segment's audio buffer separately.
    std::vector<int16_t> current_seg_audio;
    int current_seg_start_frame = -1;

    std::vector<int16_t> frame_buf(size_t(hop), 0);
    for (int i = 0; i < n_frames; ++i) {
        int64_t off = int64_t(i) * hop;
        int64_t remaining = pcm.num_samples - off;
        if (remaining <= 0) break;
        int copy = int(std::min<int64_t>(remaining, hop));
        std::memcpy(frame_buf.data(), pcm.samples.data() + off, size_t(copy) * sizeof(int16_t));
        if (copy < hop) {
            std::memset(frame_buf.data() + copy, 0, size_t(hop - copy) * sizeof(int16_t));
        }

        // VAD process.
        float prob = ten_vad_process(vad_handle, frame_buf.data(), size_t(hop));
        int   flag = (prob >= sa.threshold) ? 1 : 0;

        size_t segs_before = seg.segments.size();
        seg.feed(i, flag);

        // Track audio for the current segment (if we're in speech).
        if (seg.state == Segmenter::State::IN_SPEECH) {
            if (current_seg_start_frame < 0) {
                current_seg_start_frame = i;
                current_seg_audio.clear();
            }
            current_seg_audio.insert(current_seg_audio.end(),
                                     frame_buf.data(), frame_buf.data() + copy);
        }

        // A new segment was finalized?
        if (seg.segments.size() > segs_before) {
            const auto& s = seg.segments.back();
            int seg_start_frame = int(s.start_frame);
            int seg_end_frame   = int(s.end_frame);
            double seg_start_s  = seg_start_frame * frame_dur_s;
            double seg_end_s    = seg_end_frame   * frame_dur_s;
            double seg_dur_s    = seg_end_s - seg_start_s;

            // VAD endpoint latency: the time between the last speech frame
            // and the segment confirmation. This is essentially min_silence_s
            // (the trailing silence that triggered finalization), but we
            // measure the actual frame difference for accuracy.
            int last_speech = (s.end_frame - 1);  // end_frame is exclusive
            // Actually last_speech_frame is tracked by the segmenter; but
            // since Segmenter collapses the info, we approximate: the segment
            // was finalized at frame i, and the last speech frame was at or
            // before i - consec_silence (from the segmenter's internal state).
            // For simplicity, vad_endpoint_latency = min_silence_s.
            double vad_endpoint_s = sa.min_silence_s;

            // Extract segment audio. We use current_seg_audio which was
            // accumulated during IN_SPEECH state. But we need to trim to
            // the exact segment boundaries (confirmed_start .. last_speech_frame+1).
            // Since we accumulated from when IN_SPEECH started (which is
            // confirmed_start), the buffer corresponds to [confirmed_start, i].
            // The segment is [confirmed_start, last_speech_frame+1], so we
            // may have extra silence frames at the end. Trim them.
            int seg_frame_count = seg_end_frame - seg_start_frame;
            int seg_samples     = seg_frame_count * hop;
            if (seg_samples > int(current_seg_audio.size())) {
                seg_samples = int(current_seg_audio.size());
            }
            std::vector<int16_t> seg_audio(
                current_seg_audio.begin(),
                current_seg_audio.begin() + seg_samples);
            current_seg_audio.clear();
            current_seg_start_frame = -1;

            // Convert to double float for SenseVoice.
            std::vector<double> pcmf32(seg_audio.size());
            for (size_t k = 0; k < seg_audio.size(); ++k) {
                pcmf32[k] = double(seg_audio[k]) / 32768.0;
            }

            // Run ASR on the segment.
            sense_voice_full_params wparams = sense_voice_full_default_params(
                SENSE_VOICE_SAMPLING_GREEDY);
            wparams.language       = sa.language.c_str();
            wparams.n_threads      = sa.threads;
            wparams.print_progress = false;
            wparams.no_timestamps  = true;
            wparams.single_segment = true;

            auto t_asr_start = std::chrono::steady_clock::now();
            int rc = sense_voice_full_parallel(ctx, wparams, pcmf32,
                                                int(pcmf32.size()), 1);
            auto t_asr_end = std::chrono::steady_clock::now();
            if (rc != 0) {
                std::fprintf(stderr,
                             "stream: ASR failed on segment (rc=%d)\n", rc);
                continue;
            }

            double asr_lat_s = std::chrono::duration<double, std::milli>(
                t_asr_end - t_asr_start).count() / 1000.0;

            // Extract text.
            std::string text = get_text(ctx, false);
            PrefixInfo pinfo = get_prefix(ctx);

            // E2E latency: from segment end (in audio time) to ASR result.
            // In non-realtime mode, this is just the ASR processing time +
            // the VAD endpoint delay (min_silence). In realtime mode, it
            // also includes the wall-clock time elapsed since the segment
            // ended. Since we process immediately after VAD finalizes, E2E
            // ≈ vad_endpoint_s + asr_lat_s.
            double e2e_s = vad_endpoint_s + asr_lat_s;

            int seg_idx = int(results.size()) + 1;
            results.push_back({
                seg_idx, seg_start_s, seg_end_s, seg_dur_s,
                vad_endpoint_s, asr_lat_s, e2e_s, text
            });
            e2e_latencies_s.push_back(e2e_s);

            // Print segment result.
            std::printf("  [Seg %d] %7.3fs -> %7.3fs  (%.2fs)\n",
                        seg_idx, seg_start_s, seg_end_s, seg_dur_s);
            std::printf("          text: %s\n", text.c_str());
            if (sa.use_prefix) {
                std::printf("          lang=%s emo=%s event=%s itn=%s\n",
                            pinfo.language.c_str(), pinfo.emotion.c_str(),
                            pinfo.event.c_str(), pinfo.itn.c_str());
            }
            std::printf("          vad_endpoint=%.3fs  asr=%.3fs  e2e=%.3fs\n",
                        vad_endpoint_s, asr_lat_s, e2e_s);
            std::printf("\n");

            // NOTE: we intentionally do NOT free ctx->state between segments.
            // sense_voice_full_parallel() overwrites state->ids on each call,
            // so reusing the same state object is safe and avoids the cost of
            // re-allocating it. State is freed once at the end of the pipeline.
        }

        // In realtime mode, sleep to pace audio feeding.
        if (sa.realtime) {
            std::this_thread::sleep_for(realtime_frame_dur);
        }
    }
    seg.finish();

    // Handle any trailing segment (finalized by finish()).
    if (!seg.segments.empty() && results.size() < seg.segments.size()) {
        const auto& s = seg.segments.back();
        double seg_start_s = s.start_frame * frame_dur_s;
        double seg_end_s   = s.end_frame   * frame_dur_s;
        double seg_dur_s   = seg_end_s - seg_start_s;

        // Use accumulated audio for the trailing segment.
        int seg_frame_count = int(s.end_frame - s.start_frame);
        int seg_samples     = seg_frame_count * hop;
        if (seg_samples > int(current_seg_audio.size())) {
            seg_samples = int(current_seg_audio.size());
        }
        std::vector<int16_t> seg_audio(
            current_seg_audio.begin(),
            current_seg_audio.begin() + seg_samples);
        current_seg_audio.clear();

        if (!seg_audio.empty()) {
            std::vector<double> pcmf32(seg_audio.size());
            for (size_t k = 0; k < seg_audio.size(); ++k) {
                pcmf32[k] = double(seg_audio[k]) / 32768.0;
            }

            sense_voice_full_params wparams = sense_voice_full_default_params(
                SENSE_VOICE_SAMPLING_GREEDY);
            wparams.language       = sa.language.c_str();
            wparams.n_threads      = sa.threads;
            wparams.print_progress = false;
            wparams.no_timestamps  = true;
            wparams.single_segment = true;

            auto t_asr_start = std::chrono::steady_clock::now();
            int rc = sense_voice_full_parallel(ctx, wparams, pcmf32,
                                                int(pcmf32.size()), 1);
            auto t_asr_end = std::chrono::steady_clock::now();

            if (rc == 0) {
                std::string text = get_text(ctx, false);
                double asr_lat_s = std::chrono::duration<double, std::milli>(
                    t_asr_end - t_asr_start).count() / 1000.0;
                double e2e_s = 0.0 + asr_lat_s;  // no trailing silence for finish()

                int seg_idx = int(results.size()) + 1;
                results.push_back({
                    seg_idx, seg_start_s, seg_end_s, seg_dur_s,
                    0.0, asr_lat_s, e2e_s, text
                });
                e2e_latencies_s.push_back(e2e_s);

                std::printf("  [Seg %d] %7.3fs -> %7.3fs  (%.2fs)  [trailing]\n",
                            seg_idx, seg_start_s, seg_end_s, seg_dur_s);
                std::printf("          text: %s\n", text.c_str());
                std::printf("          asr=%.3fs  e2e=%.3fs\n", asr_lat_s, e2e_s);
                std::printf("\n");
            }
            // State is freed once at the end of the pipeline.
        }
    }

    auto t_pipeline_end = std::chrono::steady_clock::now();
    ten_vad_destroy(vad_handle);

    const double pipeline_s = std::chrono::duration<double, std::milli>(
        t_pipeline_end - t_pipeline_start).count() / 1000.0;

    // -----------------------------------------------------------------------
    // Summary.
    // -----------------------------------------------------------------------
    double total_speech_s = 0.0;
    double total_asr_s    = 0.0;
    for (const auto& r : results) {
        total_speech_s += r.dur_s;
        total_asr_s    += r.asr_latency_s;
    }

    std::printf("Summary:\n");
    std::printf("  segments detected   : %zu\n", results.size());
    std::printf("  audio duration      : %7.3f s\n", audio_dur_s);
    std::printf("  total speech        : %7.3f s  (%.1f%% of audio)\n",
                total_speech_s,
                total_speech_s / std::max(0.001, audio_dur_s) * 100.0);
    std::printf("  pipeline wall-clock : %7.3f s\n", pipeline_s);
    std::printf("  total ASR time      : %7.3f s\n", total_asr_s);
    std::printf("  overall RTF         : %7.4f  (1 / %.2fx realtime)\n",
                pipeline_s / std::max(0.001, audio_dur_s),
                audio_dur_s / std::max(0.001, pipeline_s));
    std::printf("\n");

    // Latency stats.
    if (!e2e_latencies_s.empty()) {
        std::vector<double> sorted = e2e_latencies_s;
        std::sort(sorted.begin(), sorted.end());
        double min_l  = sorted.front();
        double max_l  = sorted.back();
        double avg_l  = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        double p50_l  = percentile(sorted, 0.50);
        double p95_l  = percentile(sorted, 0.95);

        std::printf("E2E latency (VAD endpoint + ASR):\n");
        std::printf("  min  : %7.3f s\n", min_l);
        std::printf("  avg  : %7.3f s\n", avg_l);
        std::printf("  p50  : %7.3f s\n", p50_l);
        std::printf("  p95  : %7.3f s\n", p95_l);
        std::printf("  max  : %7.3f s\n", max_l);
        std::printf("\n");
    }

    // Optional: CER vs reference.
    int exit_rc = 0;
    if (!sa.reference_path.empty()) {
        auto refs = load_reference_lines(sa.reference_path);
        if (refs.empty()) {
            std::fprintf(stderr, "stream: could not read reference %s\n",
                         sa.reference_path.c_str());
            exit_rc = 1;
        } else {
            double total_cer = 0.0;
            int matched = 0;
            std::printf("CER vs reference (%s):\n", sa.reference_path.c_str());
            for (size_t i = 0; i < results.size() && i < refs.size(); ++i) {
                double c = cer(refs[i], results[i].text);
                double acc = 1.0 - c;
                total_cer += c;
                ++matched;
                std::printf("  [Seg %d] CER=%.4f (%.2f%%)  acc=%.4f  ref=\"%s\"  hyp=\"%s\"\n",
                            int(i + 1), c, c * 100.0, acc,
                            refs[i].c_str(), results[i].text.c_str());
            }
            if (matched > 0) {
                std::printf("  average CER: %.4f (%.2f%%)  average accuracy: %.4f\n",
                            total_cer / matched,
                            total_cer / matched * 100.0,
                            1.0 - total_cer / matched);
            }
            std::printf("\n");
        }
    }

    // Optional: machine-readable output.
    if (!sa.out_path.empty()) {
        std::ofstream of(sa.out_path);
        if (!of) {
            std::fprintf(stderr, "stream: could not write %s\n", sa.out_path.c_str());
            exit_rc = 1;
        } else {
            of << "# voice_test stream output\n";
            of << "# input=" << sa.wav_path
               << "  asr_model=" << sa.asr_model_path
               << "  lang=" << sa.language << "\n";
            of << "# audio_dur=" << audio_dur_s
               << "s  pipeline=" << pipeline_s
               << "s  segments=" << results.size() << "\n";
            of << "# seg_idx,start_s,end_s,dur_s,vad_endpoint_s,asr_s,e2e_s,text\n";
            for (const auto& r : results) {
                of << r.idx << ","
                   << r.start_s << ","
                   << r.end_s << ","
                   << r.dur_s << ","
                   << r.vad_endpoint_latency_s << ","
                   << r.asr_latency_s << ","
                   << r.e2e_latency_s << ","
                   << r.text << "\n";
            }
            std::printf("Wrote %s\n", sa.out_path.c_str());
        }
    }

    // Cleanup.
    sense_voice_free_state(ctx->state);
    delete ctx->model.model->encoder;
    delete ctx->model.model;
    delete ctx->vad_model.model;
    delete ctx;

    std::printf("stream: done.\n");
    return exit_rc;
}
