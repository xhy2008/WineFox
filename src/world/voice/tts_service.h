// tts_service.h — Text-to-Speech service (Kokoro ONNX).
//
// Wraps Kokoro (ONNX Runtime) with streaming synthesis support.
// Audio format: 24kHz mono float32, converted to int16 for playback.
//
// Streaming: synth_stream() calls on_audio for each phoneme batch,
// enabling low-TTFB playback (first audio chunk arrives before the
// full sentence is synthesized).

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace winefox {
namespace world {

class TtsService {
public:
    // Audio format: 24kHz mono (Kokoro's native rate).
    static constexpr int kSampleRate = 24000;

    using AudioCallback = std::function<bool(const int16_t* samples, int n)>;

    TtsService();
    ~TtsService();

    // Split mode (recommended): separate encoder (FP32) + decoder (INT8).
    bool init(const std::string& encoder_path,
              const std::string& decoder_path,
              const std::string& voices_path,
              const std::string& vocab_path,
              int n_threads = 0);

    // Stream synthesis: calls on_audio for each chunk. Return false from
    // the callback to stop early (hard interrupt). The callback receives
    // int16 samples at 24kHz.
    void synth_stream(const std::string& text, float speed,
                      const std::string& voice_name,
                      AudioCallback on_audio);

    // Non-streaming: synthesize the full text at once.
    std::vector<int16_t> synth(const std::string& text, float speed,
                                const std::string& voice_name);

    bool ready() const { return kokoro_ != nullptr; }

private:
    // Kokoro engine instance (opaque to avoid leaking onnxruntime headers).
    void* kokoro_ = nullptr;
};

} // namespace world
} // namespace winefox
