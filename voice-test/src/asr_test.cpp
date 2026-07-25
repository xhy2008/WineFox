// ASR subcommand: transcribe a 16kHz mono WAV with SenseVoice-Small.
//
// Verifies the SenseVoice ONNX model + tokens.txt load and that the
// offline recognizer produces non-empty text. SenseVoice also emits an
// emotion label, which we surface for later integration with the
// EmotionDriver (PLAN.md 3.6 / 4.5).

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

namespace voice_test {

// Parse ASR-specific flags (--language, --no-itn) from args. Returns the
// positional <wav> path, or empty on error.
static std::string parse_asr_args(const std::vector<std::string>& args,
                                  std::string& language, int& use_itn) {
    std::string wav;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--language") {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "error: --language expects a value\n");
                return {};
            }
            language = args[++i];
        } else if (a == "--no-itn") {
            use_itn = 0;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown ASR flag %s\n", a.c_str());
            return {};
        } else {
            if (wav.empty()) {
                wav = a;
            } else {
                std::fprintf(stderr, "error: unexpected positional arg %s\n", a.c_str());
                return {};
            }
        }
    }
    return wav;
}

int run_asr(const std::vector<std::string>& args, const ModelPaths& mp) {
    std::string language = "auto";
    int use_itn = 1;
    std::string wav = parse_asr_args(args, language, use_itn);
    if (wav.empty()) {
        std::fprintf(stderr,
            "usage: voice_test asr <wav> [--language auto|zh|en|ja|ko|yue] [--no-itn]\n");
        return 2;
    }

    // Load WAV (mono 16-bit PCM, normalized to [-1, 1] float).
    const SherpaOnnxWave* wave = SherpaOnnxReadWave(wav.c_str());
    if (!wave) {
        std::fprintf(stderr, "error: failed to read WAV: %s\n", wav.c_str());
        return 1;
    }
    std::printf("loaded %s: sample_rate=%d, num_samples=%d (%.2fs)\n",
                wav.c_str(), wave->sample_rate, wave->num_samples,
                wave->num_samples / static_cast<float>(wave->sample_rate));

    // Build SenseVoice config. memset first — the C API expects zero-init.
    SherpaOnnxOfflineRecognizerConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));

    cfg.feat_config.sample_rate = 16000;
    cfg.feat_config.feature_dim = 80;
    cfg.model_config.sense_voice.model    = mp.asr_model.c_str();
    cfg.model_config.sense_voice.language = language.c_str();
    cfg.model_config.sense_voice.use_itn  = use_itn;
    cfg.model_config.tokens               = mp.asr_tokens.c_str();
    cfg.model_config.num_threads          = 1;
    cfg.model_config.provider             = "cpu";
    cfg.model_config.debug                = 0;
    cfg.decoding_method                   = "greedy_search";

    const SherpaOnnxOfflineRecognizer* rec = SherpaOnnxCreateOfflineRecognizer(&cfg);
    if (!rec) {
        std::fprintf(stderr,
            "error: SherpaOnnxCreateOfflineRecognizer failed (check model path: %s)\n",
            mp.asr_model.c_str());
        SherpaOnnxFreeWave(wave);
        return 1;
    }

    const SherpaOnnxOfflineStream* stream = SherpaOnnxCreateOfflineStream(rec);
    SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate,
                                    wave->samples, wave->num_samples);
    SherpaOnnxDecodeOfflineStream(rec, stream);

    const SherpaOnnxOfflineRecognizerResult* result =
        SherpaOnnxGetOfflineStreamResult(stream);
    if (!result) {
        std::fprintf(stderr, "error: SherpaOnnxGetOfflineStreamResult returned NULL\n");
        SherpaOnnxDestroyOfflineStream(stream);
        SherpaOnnxDestroyOfflineRecognizer(rec);
        SherpaOnnxFreeWave(wave);
        return 1;
    }

    std::printf("\n--- ASR result ---\n");
    std::printf("text:    %s\n", result->text ? result->text : "(null)");
    if (result->lang)    std::printf("lang:    %s\n", result->lang);
    if (result->emotion) std::printf("emotion: %s\n", result->emotion);
    if (result->event)   std::printf("event:   %s\n", result->event);
    std::printf("------------------\n");

    SherpaOnnxDestroyOfflineRecognizerResult(result);
    SherpaOnnxDestroyOfflineStream(stream);
    SherpaOnnxDestroyOfflineRecognizer(rec);
    SherpaOnnxFreeWave(wave);
    return 0;
}

}  // namespace voice_test
