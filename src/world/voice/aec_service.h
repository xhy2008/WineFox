// aec_service.h — Acoustic Echo Cancellation (NLMS adaptive filter).
//
// Removes the TTS output (far-end reference) from the microphone capture,
// preventing the fox from hearing itself.
//
// Pipeline:
//   mic (16kHz) + far-end ref (24kHz TTS → resampled to 16kHz)
//     → NLMS → cleaned mic → VAD/ASR
//
// The far-end reference is the same audio being sent to the speaker.
// When TTS is not playing, pass silence as the far-end reference.
//
// Note: AEC is most effective with speaker output. When using headphones,
// set aec.enabled=false in config to bypass AEC entirely (no echo to cancel).

#pragma once

#include <cstdint>
#include <vector>

namespace winefox {
namespace world {

class AecService {
public:
    // Audio format: 16kHz mono, 10ms frames (160 samples).
    static constexpr int kSampleRate = 16000;
    static constexpr int kFrameSize  = 160;  // 10ms @ 16kHz

    bool init(int level);
    ~AecService();

    // Process one 10ms frame.
    //   near:  microphone capture (16kHz, kFrameSize samples)
    //   far:   far-end reference (what's playing through the speaker,
    //          resampled to 16kHz; pass NULL or silence when TTS is idle)
    //   out:   echo-cancelled output (16kHz, kFrameSize samples)
    void process_frame(const int16_t* near, const int16_t* far, int16_t* out);

    // Convenience: process a variable-length buffer by chunking into 10ms frames.
    void process(const int16_t* near, const int16_t* far, int n,
                 std::vector<int16_t>& out);

    bool ready() const { return handle_ != nullptr; }
    bool enabled() const { return enabled_; }

private:
    void* handle_  = nullptr;  // NlmsState*
    bool  enabled_ = false;
    int   level_   = 1;

    // Internal float buffers (NLMS works in float internally).
    std::vector<float> near_f_;
    std::vector<float> far_f_;
    std::vector<float> out_f_;
};

} // namespace world
} // namespace winefox
