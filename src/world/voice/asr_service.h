// asr_service.h — Automatic Speech Recognition service (SenseVoice.cpp).
//
// Wraps SenseVoice.cpp (ggml-based ASR) into a synchronous recognize() call.
// Audio format: 16kHz mono, passed as int16 samples.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace winefox {
namespace world {

class AsrService {
public:
    // Audio format: 16kHz mono.
    static constexpr int kSampleRate = 16000;

    bool init(const std::string& model_path, const std::string& language,
              int threads, bool use_itn, bool flash_attn);
    ~AsrService();

    // Recognize a segment of speech. Returns the transcript text.
    // Empty string on error.
    std::string recognize(const int16_t* samples, int n);

    bool ready() const { return ctx_ != nullptr; }
    void close();

private:
    void* ctx_ = nullptr;  // sense_voice_context*

    std::string language_  = "auto";
    int         threads_   = 4;
    bool        use_itn_   = true;
    bool        flash_attn_ = true;
};

} // namespace world
} // namespace winefox
