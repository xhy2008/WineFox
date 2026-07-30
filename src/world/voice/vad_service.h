// vad_service.h — Voice Activity Detection service (ten-vad + segmenter).
//
// Wraps the official ten-vad prebuilt lib (self-contained: ONNX model +
// onnxruntime baked into ten_vad.dll) and adds a segment state machine on
// top, since the ten-vad C API only exposes per-frame probability/flag.
//
// The segmenter consolidates frames into speech segments using configurable
// min_speech / min_silence / max_speech thresholds, matching the conventions
// used by sherpa-onnx / webrtc-vad style front-ends.
//
// Usage:
//   VadService vad;
//   vad.init(threshold, hop_size, min_speech, min_silence, max_speech);
//   // feed audio frame-by-frame:
//   for (each frame of hop_size samples) {
//       vad.feed(frame, n_samples);
//   }
//   // when a segment is finalized, on_segment callback fires.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace winefox {
namespace world {

struct VadSegment {
    std::vector<int16_t> samples;  // 16kHz mono PCM
    double start_s;
    double end_s;
};

class VadService {
public:
    using SegmentCallback = std::function<void(const VadSegment&)>;

    // Audio format: 16kHz mono int16.
    static constexpr int kSampleRate = 16000;

    bool init(float threshold, int hop_size,
              float min_speech_s, float min_silence_s, float max_speech_s,
              SegmentCallback on_segment);
    ~VadService();

    // Feed a frame of hop_size samples. When a segment is finalized
    // (trailing silence detected or max_speech cap hit), the SegmentCallback
    // is invoked with the segment audio.
    void feed(const int16_t* samples, int n);

    // Finalize any in-progress segment (call at shutdown or interrupt).
    void flush();

    // Reset state machine without finalizing (used on hard interrupt).
    void reset();

    bool ready() const { return handle_ != nullptr; }
    int  hop_size() const { return hop_size_; }
    const char* version() const;

private:
    void* handle_ = nullptr;  // ten_vad_handle_t
    int   hop_size_ = 256;

    // Segmenter state machine (ported from voice-test/src/vad_test.cpp)
    enum class State { SILENCE, PENDING_SPEECH, IN_SPEECH };
    State  state_ = State::SILENCE;

    int    pending_start_frame_ = 0;
    int    confirmed_start_     = 0;
    int    last_speech_frame_   = 0;
    int    consec_speech_       = 0;
    int    consec_silence_      = 0;
    int    speech_frame_count_  = 0;
    int    frame_idx_           = 0;

    int    min_speech_frames_   = 0;
    int    min_silence_frames_  = 0;
    int    max_speech_frames_   = 0;

    std::vector<int16_t> segment_audio_;  // accumulates samples for current segment
    SegmentCallback      on_segment_;
};

} // namespace world
} // namespace winefox
