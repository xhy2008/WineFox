// TTS subcommand: placeholder until the winefox timbre model is trained
// (PLAN.md Phase 5). When a VITS/kokoro ONNX model becomes available,
// this file will mirror the structure of asr_test.cpp: build a
// SherpaOnnxOfflineTtsConfig, call SherpaOnnxCreateOfflineTts, generate
// audio with SherpaOnnxOfflineTtsGenerateWithConfig, and write it to a
// WAV via SherpaOnnxWriteWave.
//
// For now we just print what would happen, so users get a clear message
// instead of a silent no-op.

#include "common.h"

#include <cstdio>
#include <string>
#include <vector>

namespace voice_test {

int run_tts(const std::vector<std::string>& args, const ModelPaths& mp) {
    std::string text;
    std::string out_wav;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (text.empty()) {
            text = a;
        } else if (out_wav.empty()) {
            out_wav = a;
        } else {
            std::fprintf(stderr, "error: unexpected positional arg %s\n", a.c_str());
            return 2;
        }
    }
    if (text.empty() || out_wav.empty()) {
        std::fprintf(stderr,
            "usage: voice_test tts <text> <out.wav> [--tts-model P] "
            "[--tts-tokens P] [--tts-lexicon P]\n");
        return 2;
    }

    if (mp.tts_model.empty()) {
        std::printf(
            "[TTS placeholder] No TTS model configured yet.\n"
            "\n"
            "The winefox timbre model is produced in Phase 5\n"
            "(CosyVoice teacher -> VITS-Tiny student distillation -> ONNX export).\n"
            "See PLAN.md section 5.2 and TODO.md Phase 5.\n"
            "\n"
            "Once a model is available, pass it explicitly:\n"
            "  voice_test tts \"%s\" %s --tts-model ./vits-winefox.onnx "
            "--tts-tokens ./tokens.txt --tts-lexicon ./lexicon.txt\n",
            text.c_str(), out_wav.c_str());
        return 0;
    }

    // TODO(Phase 5): wire SherpaOnnxCreateOfflineTts + Generate + WriteWave
    // here once a trained model exists. Skeleton:
    //
    //   SherpaOnnxOfflineTtsConfig cfg;
    //   std::memset(&cfg, 0, sizeof(cfg));
    //   cfg.model.vits.model    = mp.tts_model.c_str();
    //   cfg.model.vits.lexicon  = mp.tts_lexicon.c_str();
    //   cfg.model.tokens        = mp.tts_tokens.c_str();
    //   cfg.model.num_threads   = 1;
    //   const SherpaOnnxOfflineTts* tts = SherpaOnnxCreateOfflineTts(&cfg);
    //   const SherpaOnnxGeneratedAudio* audio =
    //       SherpaOnnxOfflineTtsGenerate(tts, text.c_str(), 0, 1.0f);
    //   SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate,
    //                       out_wav.c_str());
    //   SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    //   SherpaOnnxDestroyOfflineTts(tts);

    std::fprintf(stderr,
        "error: TTS model path given (%s) but generation code is not wired yet.\n"
        "Implement the Phase 5 block in voice-test/src/tts_test.cpp.\n",
        mp.tts_model.c_str());
    return 1;
}

}  // namespace voice_test
