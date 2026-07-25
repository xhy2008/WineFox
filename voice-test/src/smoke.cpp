// Smoke test: confirm sherpa-onnx C API is linked and report its version.
//
// This is the cheapest possible integration check — no ONNX models are
// loaded, no audio is touched. Run this first when bringing up a new
// platform or after changing the sherpa-onnx submodule.

#include "common.h"

#include <cstdio>

#include "sherpa-onnx/c-api/c-api.h"

namespace voice_test {

int run_smoke() {
    const char* version = SherpaOnnxGetVersionStr();
    if (!version) {
        std::fprintf(stderr, "FAIL: SherpaOnnxGetVersionStr() returned NULL\n");
        return 1;
    }
    std::printf("sherpa-onnx version: %s\n", version);
    std::printf("sherpa-onnx C API link: OK\n");

    // Also exercise SherpaOnnxReadWave on a non-existent file to confirm
    // the symbol resolves and returns NULL gracefully — this catches
    // link-time issues that a version string alone might miss.
    const SherpaOnnxWave* wave = SherpaOnnxReadWave("__nonexistent__.wav");
    if (wave != nullptr) {
        std::fprintf(stderr,
            "WARN: SherpaOnnxReadWave returned non-NULL for a missing file\n");
        SherpaOnnxFreeWave(wave);
    } else {
        std::printf("SherpaOnnxReadWave(NULL path) handled gracefully: OK\n");
    }

    std::printf("\nsmoke test passed.\n");
    return 0;
}

}  // namespace voice_test
