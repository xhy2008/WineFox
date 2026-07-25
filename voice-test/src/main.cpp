// voice_test: standalone sherpa-onnx integration sandbox.
//
// Usage:
//   voice_test smoke                       # version + link check (no models)
//   voice_test asr  <wav> [--asr-model P] [--asr-tokens P] [--language auto|zh|en|ja|ko|yue]
//   voice_test vad  <wav> [--vad-model P] [--threshold 0.5] [--window 256]
//   voice_test tts  <text> <out.wav> [--tts-model P] [--tts-tokens P] [--tts-lexicon P]
//   voice_test help
//
// Default model paths assume the binary is launched from the winefox repo
// root (see common.h). Override per-invocation with the flags above.

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#endif

using namespace voice_test;

static void print_usage() {
    std::printf(
        "voice_test — sherpa-onnx integration sandbox\n"
        "\n"
        "USAGE:\n"
        "  voice_test smoke                 Print sherpa-onnx version (no models)\n"
        "  voice_test asr <wav> [opts]      Transcribe a 16kHz mono WAV\n"
        "  voice_test vad <wav> [opts]      Detect speech segments in a WAV\n"
        "  voice_test tts <text> <out.wav>  (placeholder, no model yet)\n"
        "  voice_test help                  Show this message\n"
        "\n"
        "ASR options:\n"
        "  --asr-model <path>     SenseVoice ONNX model (default: models/asr/sense-voice-small-int8.onnx)\n"
        "  --asr-tokens <path>    tokens.txt (default: models/asr/tokens.txt)\n"
        "  --language <lang>      auto|zh|en|ja|ko|yue (default: auto)\n"
        "  --no-itn               Disable inverse text normalization\n"
        "\n"
        "VAD options:\n"
        "  --vad-model <path>     TEN-VAD ONNX model (default: models/vad/ten-vad.onnx)\n"
        "  --threshold <f>        Speech probability threshold (default 0.5)\n"
        "  --window <n>           Frame size in samples (default 256)\n"
        "  --min-silence <f>      Min silence duration in seconds (default 0.3)\n"
        "  --min-speech <f>       Min speech duration in seconds (default 0.25)\n"
        "\n"
        "TTS options (Phase 5, placeholder):\n"
        "  --tts-model <path>     VITS/kokoro ONNX model\n"
        "  --tts-tokens <path>    tokens.txt\n"
        "  --tts-lexicon <path>   lexicon.txt\n");
}

// Apply global --asr-model / --asr-tokens / --vad-model / --tts-* flags to the
// ModelPaths struct. Returns the remaining positional args.
static std::vector<std::string> apply_model_overrides(
        const std::vector<std::string>& args, ModelPaths& mp) {
    std::vector<std::string> positional;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&]() -> const std::string& {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "error: %s expects a value\n", a.c_str());
                std::exit(2);
            }
            return args[++i];
        };
        if      (a == "--asr-model")   mp.asr_model  = next();
        else if (a == "--asr-tokens")  mp.asr_tokens = next();
        else if (a == "--vad-model")   mp.vad_model  = next();
        else if (a == "--tts-model")   mp.tts_model  = next();
        else if (a == "--tts-tokens")  mp.tts_tokens = next();
        else if (a == "--tts-lexicon") mp.tts_lexicon = next();
        else positional.push_back(a);
    }
    return positional;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // UTF-8 console so Chinese ASR output is readable on Windows.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string sub = argv[1];
    std::vector<std::string> rest;
    for (int i = 2; i < argc; ++i) rest.emplace_back(argv[i]);

    if (sub == "help" || sub == "--help" || sub == "-h") {
        print_usage();
        return 0;
    }

    ModelPaths mp;
    auto positional = apply_model_overrides(rest, mp);

    if (sub == "smoke") {
        return run_smoke();
    } else if (sub == "asr") {
        return run_asr(positional, mp);
    } else if (sub == "vad") {
        return run_vad(positional, mp);
    } else if (sub == "tts") {
        return run_tts(positional, mp);
    }

    std::fprintf(stderr, "unknown subcommand: %s\n", sub.c_str());
    print_usage();
    return 1;
}
