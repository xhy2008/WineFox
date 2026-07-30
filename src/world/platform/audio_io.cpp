#include "audio_io.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace winefox {
namespace world {

// ===========================================================================
// AudioRingBuffer — SPSC lock-free ring buffer for int16 samples
// ===========================================================================

AudioRingBuffer::AudioRingBuffer(size_t capacity) {
    size_t c = 1;
    while (c < capacity) c <<= 1;
    capacity_ = c;
    mask_     = c - 1;
    buf_.resize(c);
}

AudioRingBuffer::~AudioRingBuffer() = default;

size_t AudioRingBuffer::write(const int16_t* data, size_t n) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    const size_t used = (h - t) & mask_;
    const size_t free_space = capacity_ - used - 1;  // -1 to distinguish full/empty
    const size_t to_write = std::min(n, free_space);

    for (size_t i = 0; i < to_write; ++i) {
        buf_[(h + i) & mask_] = data[i];
    }
    head_.store((h + to_write) & mask_, std::memory_order_release);
    return to_write;
}

size_t AudioRingBuffer::read(int16_t* out, size_t n) {
    const size_t t = tail_.load(std::memory_order_relaxed);
    const size_t h = head_.load(std::memory_order_acquire);
    const size_t available = (h - t) & mask_;
    const size_t to_read = std::min(n, available);

    for (size_t i = 0; i < to_read; ++i) {
        out[i] = buf_[(t + i) & mask_];
    }
    tail_.store((t + to_read) & mask_, std::memory_order_release);
    return to_read;
}

size_t AudioRingBuffer::available() const {
    const size_t h = head_.load(std::memory_order_acquire);
    const size_t t = tail_.load(std::memory_order_acquire);
    return (h - t) & mask_;
}

void AudioRingBuffer::clear() {
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
}

// ===========================================================================
// AudioIo
// ===========================================================================

AudioIo::AudioIo() = default;

AudioIo::~AudioIo() { shutdown(); }

bool AudioIo::init(InputCallback on_input,
                   FarEndCallback on_far_end) {
    on_input_   = std::move(on_input);
    on_far_end_ = std::move(on_far_end);

    // --- Input stream: 16kHz mono int16 ---
    SDL_AudioSpec in_spec;
    SDL_zero(in_spec);
    in_spec.freq     = kInputRate;
    in_spec.format   = SDL_AUDIO_S16;
    in_spec.channels = 1;

    in_stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_RECORDING,
                                           &in_spec, input_callback_, this);
    if (!in_stream_) {
        std::fprintf(stderr, "[audio] failed to open input stream: %s\n",
                     SDL_GetError());
        return false;
    }
    in_dev_ = SDL_GetAudioStreamDevice(in_stream_);

    // --- Output stream: 24kHz mono int16, NO callback (push model) ---
    // The AI thread pushes TTS audio via push_output(). SDL internally
    // handles resampling and format conversion to the device's actual
    // format (typically 48kHz F32 stereo).
    SDL_AudioSpec out_spec;
    SDL_zero(out_spec);
    out_spec.freq     = kOutputRate;
    out_spec.format   = SDL_AUDIO_S16;
    out_spec.channels = 1;

    out_stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
                                            &out_spec, nullptr, nullptr);
    if (!out_stream_) {
        std::fprintf(stderr, "[audio] failed to open output stream: %s\n",
                     SDL_GetError());
        SDL_DestroyAudioStream(in_stream_);
        in_stream_ = nullptr;
        return false;
    }
    out_dev_ = SDL_GetAudioStreamDevice(out_stream_);

    // Log actual device format for diagnostics.
    SDL_AudioSpec src_spec, dst_spec;
    if (SDL_GetAudioStreamFormat(out_stream_, &src_spec, &dst_spec)) {
        std::fprintf(stderr,
            "[audio] output stream: src=%dHz fmt=%u ch=%u → dst=%dHz fmt=%u ch=%u\n",
            src_spec.freq, src_spec.format, src_spec.channels,
            dst_spec.freq, dst_spec.format, dst_spec.channels);
    }

    // Pre-allocate input buffer for callback use.
    in_buf_.resize(kInputRate / 10);   // 100ms max per callback

    // Start both streams.
    SDL_ResumeAudioDevice(in_dev_);
    SDL_ResumeAudioDevice(out_dev_);

    std::fprintf(stderr, "[audio] input=%dHz output=%dHz started (push model)\n",
                 kInputRate, kOutputRate);
    return true;
}

void AudioIo::push_output(const int16_t* data, int n) {
    if (!out_stream_ || n <= 0) return;

    // Tap the far-end reference for AEC (if enabled).
    if (on_far_end_) {
        on_far_end_(data, n);
    }

    if (!SDL_PutAudioStreamData(out_stream_, data, n * sizeof(int16_t))) {
        std::fprintf(stderr, "[audio] SDL_PutAudioStreamData failed: %s (n=%d)\n",
                     SDL_GetError(), n);
    }
}

bool AudioIo::output_has_data() const {
    if (!out_stream_) return false;
    // SDL_GetAudioStreamAvailable returns bytes still queued in the stream.
    // If > 0, there's audio waiting to be played.
    return SDL_GetAudioStreamAvailable(out_stream_) > 0;
}

void AudioIo::flush_output() {
    if (!out_stream_) return;
    SDL_ClearAudioStream(out_stream_);
}

void AudioIo::shutdown() {
    if (in_stream_) {
        SDL_DestroyAudioStream(in_stream_);
        in_stream_ = nullptr;
    }
    if (out_stream_) {
        SDL_DestroyAudioStream(out_stream_);
        out_stream_ = nullptr;
    }
}

void SDLCALL AudioIo::input_callback_(void* userdata, SDL_AudioStream* stream,
                                       int additional_amount, int total_amount) {
    auto* self = static_cast<AudioIo*>(userdata);
    if (!self || !self->on_input_) return;

    // Read available captured audio from the stream.
    // additional_amount = bytes that SDL wants us to read right now.
    if (additional_amount <= 0) return;

    int samples = additional_amount / sizeof(int16_t);
    if (samples > static_cast<int>(self->in_buf_.size())) {
        self->in_buf_.resize(samples);
    }

    int got = SDL_GetAudioStreamData(stream, self->in_buf_.data(),
                                     samples * sizeof(int16_t));
    if (got <= 0) return;

    int got_samples = got / sizeof(int16_t);
    self->on_input_(self->in_buf_.data(), got_samples);
}

} // namespace world
} // namespace winefox
