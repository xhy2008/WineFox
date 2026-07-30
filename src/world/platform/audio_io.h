// audio_io.h — SDL_audio bidirectional I/O.
//
// Two separate audio streams:
//   Input  (capture):  16kHz mono int16 — fed to VAD/ASR (and AEC if enabled)
//   Output (playback): 24kHz mono int16 — TTS output to speaker
//
// The TTS output is also tapped as the far-end reference for AEC. When TTS
// is idle, silence is fed as the far-end reference.
//
// Threading: SDL audio callbacks run on SDL's audio thread. Input samples
// are pushed into a lock-free ring buffer consumed by the AI thread.
// Output samples are pulled from another ring buffer filled by the TTS
// synthesis callback. Both buffers are SPSC (SDL audio thread = producer
// for input, consumer for output; AI thread = the other side).

#pragma once

#include <SDL3/SDL.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace winefox {
namespace world {

// Lock-free ring buffer for audio samples (SPSC).
// Capacity is a power of 2 for fast masking.
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacity);

    ~AudioRingBuffer();

    // Producer: write samples. Returns number actually written (may be < n
    // if buffer is full).
    size_t write(const int16_t* data, size_t n);

    // Consumer: read samples. Returns number actually read.
    size_t read(int16_t* out, size_t n);

    // How many samples are currently buffered.
    size_t available() const;

    // Clear all buffered samples.
    void clear();

private:
    std::vector<int16_t> buf_;
    size_t               capacity_;
    size_t               mask_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

class AudioIo {
public:
    // Input callback: called when a new chunk of 16kHz mic audio is ready.
    // The callback receives AEC-processed samples (if AEC is enabled).
    using InputCallback = std::function<void(const int16_t* samples, int n)>;

    // Far-end callback: called with the same samples being sent to the
    // speaker, for AEC reference. May be null if AEC is disabled.
    using FarEndCallback = std::function<void(const int16_t* samples, int n)>;

    AudioIo();
    ~AudioIo();

    bool init(InputCallback on_input,
              FarEndCallback on_far_end = nullptr);
    void shutdown();

    // Push TTS audio (24kHz S16 mono) to the playback stream.
    // Called from the AI thread. SDL internally handles resampling
    // and format conversion to the device's actual format.
    void push_output(const int16_t* data, int n);

    // Clear all queued audio from the output stream (hard interrupt).
    void flush_output();

    // Returns true if there is still audio data queued in the SDL stream
    // (i.e., playback has not finished yet).
    bool output_has_data() const;

    // Static constants for sample rates.
    static constexpr int kInputRate  = 16000;  // VAD/ASR
    static constexpr int kOutputRate = 24000;  // TTS/Kokoro

private:
    SDL_AudioStream* in_stream_  = nullptr;
    SDL_AudioStream* out_stream_ = nullptr;
    SDL_AudioDeviceID in_dev_   = 0;
    SDL_AudioDeviceID out_dev_  = 0;

    InputCallback   on_input_;
    FarEndCallback  on_far_end_;

    // Resampling buffer for AEC far-end reference (24kHz → 16kHz).
    std::vector<int16_t> in_buf_;

    // SDL audio callback for input only (output uses push model).
    static void SDLCALL input_callback_(void* userdata, SDL_AudioStream* stream,
                                         int additional_amount, int total_amount);
};

} // namespace world
} // namespace winefox
