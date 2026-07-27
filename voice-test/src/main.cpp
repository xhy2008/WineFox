// voice-test/src/main.cpp
//
// Entry point. Dispatches to subcommands based on argv[1].
//
// Usage:
//   voice_test smoke
//   voice_test vad  <wav> [--threshold 0.5] [--hop 256]
//   voice_test asr  <wav> [--model <gguf>] [--lang auto|zh|en|...]
//   voice_test tts  <text> --model <onnx> --voices <bin> [--out <wav>]
//   voice_test stream <wav> [--realtime|--no-realtime]
//
// Subcommands vad/asr/tts/stream are only available when the
// corresponding VOICE_TEST_HAS_* macro is defined at build time.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"

// ---------------------------------------------------------------------------
// Windows: get UTF-8 argv from the wide-character command line.
//
// On Chinese Windows the default system code page is GBK, so argv[] from
// main() contains GBK-encoded strings. Jieba/PinyinFinder dictionaries
// are UTF-8, so any non-ASCII input (Chinese text, paths with CJK chars)
// would fail to match. We use GetCommandLineW + CommandLineToArgvW to
// get proper Unicode argv, then convert to UTF-8 for the rest of the
// program.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <shellapi.h>

// CommandLineToArgvW is in shell32.lib; link it here for convenience.
#  pragma comment(lib, "shell32.lib")

static std::vector<std::string> wargv_to_utf8(int argc, wchar_t** wargv) {
    std::vector<std::string> out;
    out.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                      nullptr, 0, nullptr, nullptr);
        std::string s(len > 0 ? len - 1 : 0, '\0');
        if (len > 1) {
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                &s[0], len, nullptr, nullptr);
        }
        out.push_back(std::move(s));
    }
    return out;
}
#endif  // _WIN32

// Helper macros that evaluate to 1/0 regardless of whether the
// corresponding VOICE_TEST_HAS_* is defined. Used by print_usage().
#if defined(VOICE_TEST_HAS_VAD)
#  define VOICE_TEST_HAS_VAD_DEFINED 1
#else
#  define VOICE_TEST_HAS_VAD_DEFINED 0
#endif
#if defined(VOICE_TEST_HAS_ASR)
#  define VOICE_TEST_HAS_ASR_DEFINED 1
#else
#  define VOICE_TEST_HAS_ASR_DEFINED 0
#endif
#if defined(VOICE_TEST_HAS_TTS)
#  define VOICE_TEST_HAS_TTS_DEFINED 1
#else
#  define VOICE_TEST_HAS_TTS_DEFINED 0
#endif
#if defined(VOICE_TEST_HAS_STREAM)
#  define VOICE_TEST_HAS_STREAM_DEFINED 1
#else
#  define VOICE_TEST_HAS_STREAM_DEFINED 0
#endif

static void print_usage() {
    std::printf(
        "voice_test - voice front-end benchmark sandbox\n"
        "\n"
        "Usage:\n"
        "  voice_test smoke\n"
        "      Run a quick self-check; verifies WAV I/O and prints build info.\n"
        "\n");
#if defined(VOICE_TEST_HAS_VAD)
    std::printf(
        "  voice_test vad <wav> [--threshold 0.5] [--hop 256]\n"
        "      Run ten-vad on <wav>, report detected speech segments and latency.\n"
        "\n");
#endif
#if defined(VOICE_TEST_HAS_ASR)
    std::printf(
        "  voice_test asr <wav> [--model <gguf>] [--lang auto]\n"
        "      Run SenseVoice.cpp on <wav>, report transcript and RTF.\n"
        "\n");
#endif
#if defined(VOICE_TEST_HAS_TTS)
    std::printf(
        "  voice_test tts <text> --model <onnx> --voices <bin> [--out <wav>]\n"
        "      Run Kokoro TTS on <text>, write synthesized audio to <wav>.\n"
        "\n");
#endif
#if defined(VOICE_TEST_HAS_STREAM)
    std::printf(
        "  voice_test stream <wav> [--realtime|--no-realtime]\n"
        "      Run full streaming VAD+ASR pipeline on <wav>.\n"
        "\n");
#endif
    std::printf(
        "Build-time component availability (controlled by CMake options):\n"
        "  VOICE_TEST_HAS_VAD        = %d\n"
        "  VOICE_TEST_HAS_ASR        = %d\n"
        "  VOICE_TEST_HAS_TTS        = %d\n"
        "  VOICE_TEST_HAS_STREAM     = %d\n",
        VOICE_TEST_HAS_VAD_DEFINED,
        VOICE_TEST_HAS_ASR_DEFINED,
        VOICE_TEST_HAS_TTS_DEFINED,
        VOICE_TEST_HAS_STREAM_DEFINED);
}

int main(int argc, char** argv) {
#if defined(_WIN32)
    // On Windows, re-parse the command line as UTF-16 then convert to UTF-8
    // so Chinese text and CJK paths work regardless of the system code page.
    int wargc = 0;
    wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr || wargc < 1) {
        std::fprintf(stderr, "Failed to parse command line\n");
        return 1;
    }
    auto utf8_args = wargv_to_utf8(wargc, wargv);
    LocalFree(wargv);

    if (utf8_args.size() < 2) {
        print_usage();
        return 1;
    }

    const std::string sub = utf8_args[1];
    std::vector<std::string> positional;
    for (size_t i = 2; i < utf8_args.size(); ++i) {
        positional.emplace_back(utf8_args[i]);
    }
#else
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const std::string sub = argv[1];
    std::vector<std::string> positional;
    for (int i = 2; i < argc; ++i) {
        positional.emplace_back(argv[i]);
    }
#endif

    if (sub == "smoke" || sub == "-smoke" || sub == "--smoke") {
        return run_smoke(positional);
    }
#  if defined(VOICE_TEST_HAS_VAD)
    if (sub == "vad") {
        return run_vad(positional);
    }
#  endif
#  if defined(VOICE_TEST_HAS_ASR)
    if (sub == "asr") {
        return run_asr(positional);
    }
#  endif
#  if defined(VOICE_TEST_HAS_TTS)
    if (sub == "tts") {
        return run_tts(positional);
    }
#  endif
#  if defined(VOICE_TEST_HAS_STREAM)
    if (sub == "stream") {
        return run_stream(positional);
    }
#  endif

    std::fprintf(stderr, "Unknown or unavailable subcommand: %s\n", sub.c_str());
    print_usage();
    return 1;
}
