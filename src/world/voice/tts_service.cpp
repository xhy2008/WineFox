// tts_service.cpp — Text-to-Speech service (Kokoro ONNX).
//
// Implementation of the TtsService declared in tts_service.h. Wraps the
// Kokoro ONNX Runtime engine in split mode (INT8 encoder + FP32 decoder)
// and exposes both streaming and one-shot synthesis. Audio is delivered
// as 16-bit PCM at Kokoro's native 24 kHz sample rate.

#include "tts_service.h"

#include "Kokoro.h"

#include <cstdio>
#include <vector>

namespace winefox {
namespace world {

// Convert float32 samples in [-1.0, 1.0] to int16 PCM.
// Uses the same conversion as voice-test: clamp to [-1,1] first, then scale.
static inline std::vector<int16_t> float_to_int16(const std::vector<float>& f) {
    std::vector<int16_t> out;
    out.reserve(f.size());
    for (float v : f) {
        float clamped = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
        out.push_back(static_cast<int16_t>(clamped * 32767.0f));
    }
    return out;
}

TtsService::TtsService() : kokoro_(nullptr) {}

TtsService::~TtsService() {
    if (kokoro_) {
        delete static_cast<Kokoro*>(kokoro_);
        kokoro_ = nullptr;
    }
}

bool TtsService::init(const std::string& encoder_path,
                      const std::string& decoder_path,
                      const std::string& voices_path,
                      const std::string& vocab_path,
                      int n_threads) {
    if (kokoro_) {
        delete static_cast<Kokoro*>(kokoro_);
        kokoro_ = nullptr;
    }

    try {
        kokoro_ = new Kokoro(encoder_path, decoder_path,
                             voices_path, vocab_path, n_threads);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "TtsService::init failed: %s\n", e.what());
        kokoro_ = nullptr;
        return false;
    } catch (...) {
        std::fprintf(stderr, "TtsService::init failed: unknown exception\n");
        kokoro_ = nullptr;
        return false;
    }

    return kokoro_ != nullptr;
}

void TtsService::synth_stream(const std::string& text, float speed,
                              const std::string& voice_name,
                              AudioCallback on_audio) {
    if (!kokoro_ || !on_audio) {
        return;
    }

    auto* k = static_cast<Kokoro*>(kokoro_);

    // Stream each phoneme-batch chunk to the audio callback immediately
    // as it is synthesized, so playback starts before the full sentence
    // is done (low TTFB). Collecting all chunks first would defeat the
    // purpose of streaming synthesis.
    k->create_stream(
        text, voice_name,
        [&on_audio](const std::vector<float>& chunk, int /*sr*/) -> bool {
            if (chunk.empty()) return true;
            std::vector<int16_t> pcm = float_to_int16(chunk);
            return on_audio(pcm.data(), static_cast<int>(pcm.size()));
        },
        speed);
}

std::vector<int16_t> TtsService::synth(const std::string& text, float speed,
                                       const std::string& voice_name) {
    if (!kokoro_) {
        return {};
    }

    auto* k = static_cast<Kokoro*>(kokoro_);

    std::pair<std::vector<float>, int> result;
    try {
        result = k->create(text, voice_name, speed);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "TtsService::synth failed: %s\n", e.what());
        return {};
    } catch (...) {
        std::fprintf(stderr, "TtsService::synth failed: unknown exception\n");
        return {};
    }

    return float_to_int16(result.first);
}

} // namespace world
} // namespace winefox
