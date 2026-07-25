// voice-test common helpers shared across subcommands.
//
// All subcommands return 0 on success and non-zero on failure. Argument
// parsing is intentionally minimal (no external dep) — each subcommand
// accepts a small fixed set of flags.
#pragma once

#include <string>
#include <vector>

namespace voice_test {

// Default model paths (relative to the winefox repo root). Override at
// runtime via --asr-model / --asr-tokens / --vad-model.
struct ModelPaths {
    std::string asr_model  = "models/asr/sense-voice-small-int8.onnx";
    std::string asr_tokens = "models/asr/tokens.txt";
    std::string vad_model  = "models/vad/ten-vad.onnx";
    // TTS model is not available yet (Phase 5 will train the winefox timbre).
    std::string tts_model;
    std::string tts_tokens;
    std::string tts_lexicon;
};

// Print the sherpa-onnx version and confirm the C API is linked. No models
// required — this is the cheapest smoke test.
int run_smoke();

// Load SenseVoice-Small + tokens, transcribe <wav_path>, print text + emotion.
int run_asr(const std::vector<std::string>& args, const ModelPaths& mp);

// Load TEN-VAD, run VAD on <wav_path>, print detected speech segments.
int run_vad(const std::vector<std::string>& args, const ModelPaths& mp);

// TTS subcommand. Currently a placeholder — no trained winefox model yet.
// Prints instructions for when a model becomes available (Phase 5).
int run_tts(const std::vector<std::string>& args, const ModelPaths& mp);

}  // namespace voice_test
