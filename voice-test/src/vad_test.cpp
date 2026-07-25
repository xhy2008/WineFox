// VAD subcommand: run TEN-VAD over a WAV file and list speech segments.
//
// Mirrors the VoiceFrontend VAD configuration from PLAN.md 3.2:
//   frame_size       = 256 samples (~16ms at 16kHz; TEN-VAD uses 256)
//   min_speech       = 0.25s
//   min_silence      = 0.30s
//   max_speech       = 10s (split long segments)
// The output lists each detected segment's [start, end] in seconds and
// the total speech duration, which is useful for tuning thresholds.

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

namespace voice_test {

struct VadOpts {
    float threshold       = 0.5f;
    int   window          = 256;      // TEN-VAD default window size
    float min_silence     = 0.30f;
    float min_speech      = 0.25f;
    float max_speech      = 10.0f;
};

static std::string parse_vad_args(const std::vector<std::string>& args, VadOpts& o) {
    std::string wav;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next_str = [&]() -> std::string {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "error: %s expects a value\n", a.c_str());
                return {};
            }
            return args[++i];
        };
        if (a == "--threshold") {
            o.threshold = static_cast<float>(std::atof(next_str().c_str()));
        } else if (a == "--window") {
            o.window = std::atoi(next_str().c_str());
        } else if (a == "--min-silence") {
            o.min_silence = static_cast<float>(std::atof(next_str().c_str()));
        } else if (a == "--min-speech") {
            o.min_speech = static_cast<float>(std::atof(next_str().c_str()));
        } else if (a == "--max-speech") {
            o.max_speech = static_cast<float>(std::atof(next_str().c_str()));
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown VAD flag %s\n", a.c_str());
            return {};
        } else {
            if (wav.empty()) wav = a;
            else {
                std::fprintf(stderr, "error: unexpected positional arg %s\n", a.c_str());
                return {};
            }
        }
    }
    return wav;
}

int run_vad(const std::vector<std::string>& args, const ModelPaths& mp) {
    VadOpts o;
    std::string wav = parse_vad_args(args, o);
    if (wav.empty()) {
        std::fprintf(stderr,
            "usage: voice_test vad <wav> [--threshold 0.5] [--window 256] "
            "[--min-silence 0.3] [--min-speech 0.25] [--max-speech 10]\n");
        return 2;
    }

    const SherpaOnnxWave* wave = SherpaOnnxReadWave(wav.c_str());
    if (!wave) {
        std::fprintf(stderr, "error: failed to read WAV: %s\n", wav.c_str());
        return 1;
    }
    std::printf("loaded %s: sample_rate=%d, num_samples=%d (%.2fs)\n",
                wav.c_str(), wave->sample_rate, wave->num_samples,
                wave->num_samples / static_cast<float>(wave->sample_rate));

    SherpaOnnxVadModelConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.ten_vad.model              = mp.vad_model.c_str();
    cfg.ten_vad.threshold          = o.threshold;
    cfg.ten_vad.min_silence_duration = o.min_silence;
    cfg.ten_vad.min_speech_duration  = o.min_speech;
    cfg.ten_vad.max_speech_duration  = o.max_speech;
    cfg.ten_vad.window_size        = o.window;
    cfg.sample_rate                = 16000;
    cfg.num_threads                = 1;
    cfg.provider                   = "cpu";
    cfg.debug                      = 0;

    // buffer_size_in_seconds = 30s holds a rolling window; long files are
    // streamed in via AcceptWaveform below.
    const SherpaOnnxVoiceActivityDetector* vad =
        SherpaOnnxCreateVoiceActivityDetector(&cfg, 30.0f);
    if (!vad) {
        std::fprintf(stderr,
            "error: SherpaOnnxCreateVoiceActivityDetector failed (check model: %s)\n",
            mp.vad_model.c_str());
        SherpaOnnxFreeWave(wave);
        return 1;
    }

    // Stream the audio through the VAD in window-sized chunks.
    const int win = o.window > 0 ? o.window : 256;
    const float* p = wave->samples;
    int32_t remaining = wave->num_samples;
    while (remaining >= win) {
        SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, p, win);
        p += win;
        remaining -= win;
        // Drain any complete segments.
        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
            const SherpaOnnxSpeechSegment* seg =
                SherpaOnnxVoiceActivityDetectorFront(vad);
            if (!seg) break;
            float start = seg->start / static_cast<float>(wave->sample_rate);
            float end   = (seg->start + seg->n) / static_cast<float>(wave->sample_rate);
            std::printf("  segment: [%6.2fs, %6.2fs]  dur=%.2fs  n=%d\n",
                        start, end, end - start, seg->n);
            SherpaOnnxDestroySpeechSegment(seg);
            SherpaOnnxVoiceActivityDetectorPop(vad);
        }
    }
    // Flush the tail to release any in-progress segment.
    SherpaOnnxVoiceActivityDetectorFlush(vad);
    while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
        const SherpaOnnxSpeechSegment* seg =
            SherpaOnnxVoiceActivityDetectorFront(vad);
        if (!seg) break;
        float start = seg->start / static_cast<float>(wave->sample_rate);
        float end   = (seg->start + seg->n) / static_cast<float>(wave->sample_rate);
        std::printf("  segment: [%6.2fs, %6.2fs]  dur=%.2fs  n=%d  (flushed)\n",
                    start, end, end - start, seg->n);
        SherpaOnnxDestroySpeechSegment(seg);
        SherpaOnnxVoiceActivityDetectorPop(vad);
    }

    SherpaOnnxDestroyVoiceActivityDetector(vad);
    SherpaOnnxFreeWave(wave);
    std::printf("VAD done.\n");
    return 0;
}

}  // namespace voice_test
