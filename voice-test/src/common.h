// voice-test/src/common.h
//
// Shared declarations for the voice-test benchmark sandbox.
//
// New architecture (sherpa-onnx-free):
//   VAD  : ten-vad  (onnxruntime)
//   ASR  : SenseVoice.cpp (ggml)
//   TTS  : Kokoro  (onnxruntime)
//
// Each component is gated behind a compile-time macro
// (VOICE_TEST_HAS_VAD / _ASR / _TTS / _STREAM) so the binary can be
// built incrementally as deps are wired in.

#ifndef VOICE_TEST_COMMON_H
#define VOICE_TEST_COMMON_H

#include <cstdint>
#include <string>
#include <vector>

// 16 kHz mono int16 PCM is the canonical format across VAD/ASR/TTS.
inline constexpr int kSampleRate = 16000;
inline constexpr int kHopSize    = 256;  // ten-vad-ggml fixed hop (16ms)

// Loaded WAV file (16-bit mono PCM; resampled to 16kHz by caller if needed).
struct Pcm {
    int sample_rate = 0;
    std::vector<int16_t> samples;  // 16-bit signed samples
    int64_t num_samples = 0;
};

// Load a 16-bit mono WAV file. Returns false on failure.
bool load_wav(const std::string& path, Pcm& out);

// Write a 16-bit mono WAV file. Returns false on failure.
bool write_wav(const std::string& path, const Pcm& pcm);

// Subcommand entry points. Each returns a process exit code (0 = success).
int run_smoke(const std::vector<std::string>& args);

#if defined(VOICE_TEST_HAS_VAD)
int run_vad(const std::vector<std::string>& args);
#endif

#if defined(VOICE_TEST_HAS_ASR)
int run_asr(const std::vector<std::string>& args);
#endif

#if defined(VOICE_TEST_HAS_TTS)
int run_tts(const std::vector<std::string>& args);
#endif

#if defined(VOICE_TEST_HAS_STREAM)
int run_stream(const std::vector<std::string>& args);
#endif

#endif  // VOICE_TEST_COMMON_H
