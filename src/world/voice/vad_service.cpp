// vad_service.cpp — Voice Activity Detection service (ten-vad + segmenter).
//
// Implementation of VadService declared in vad_service.h. Wraps the official
// ten-vad prebuilt lib (self-contained ONNX + onnxruntime) and ports the
// segmenter state machine from voice-test/src/vad_test.cpp (Segmenter struct)
// to a streaming incremental-audio model.
//
// The reference Segmenter operated on a full in-memory audio buffer and
// recorded segment boundaries as frame indices, slicing the audio at the end.
// Here, audio arrives frame-by-frame, so segment_audio_ is accumulated
// incrementally while in PENDING_SPEECH / IN_SPEECH and trimmed to the exact
// speech range [confirmed_start_, last_speech_frame_+1) on finalization.

#include "vad_service.h"
#include "ten_vad.h"

#include <algorithm>
#include <utility>

namespace winefox {
namespace world {

bool VadService::init(float threshold, int hop_size,
                      float min_speech_s, float min_silence_s, float max_speech_s,
                      SegmentCallback on_segment) {
    // Release any previously created handle before re-creating.
    if (handle_) {
        ten_vad_destroy(&handle_);
        handle_ = nullptr;
    }

    hop_size_ = hop_size;
    on_segment_ = std::move(on_segment);

    if (ten_vad_create(&handle_, static_cast<size_t>(hop_size), threshold) != 0 ||
        !handle_) {
        handle_ = nullptr;
        return false;
    }

    // Frame thresholds derived from seconds — same formula as vad_test.cpp:
    //   frame_dur_s = hop / kSampleRate
    //   frames = max(1, int(seconds / frame_dur_s + 0.5))
    const double frame_dur_s = static_cast<double>(hop_size) / kSampleRate;
    min_speech_frames_  = std::max(1, static_cast<int>(min_speech_s  / frame_dur_s + 0.5));
    min_silence_frames_ = std::max(1, static_cast<int>(min_silence_s / frame_dur_s + 0.5));
    max_speech_frames_  = std::max(min_speech_frames_,
                                   static_cast<int>(max_speech_s / frame_dur_s + 0.5));

    // Reset segmenter state to a clean SILENCE.
    state_ = State::SILENCE;
    pending_start_frame_ = 0;
    confirmed_start_ = 0;
    last_speech_frame_ = 0;
    consec_speech_ = 0;
    consec_silence_ = 0;
    speech_frame_count_ = 0;
    frame_idx_ = 0;
    segment_audio_.clear();

    return true;
}

VadService::~VadService() {
    if (handle_) {
        ten_vad_destroy(&handle_);
        handle_ = nullptr;
    }
}

void VadService::feed(const int16_t* samples, int n) {
    if (!handle_ || n <= 0) return;

    float prob = 0.0f;
    int   flag = 0;
    if (ten_vad_process(handle_, samples, static_cast<size_t>(n),
                        &prob, &flag) != 0) {
        return;
    }

    // Finalize and emit the current segment. Trims any trailing silence
    // frames accumulated after last_speech_frame_ so the emitted audio spans
    // exactly [confirmed_start_, last_speech_frame_+1), matching the
    // reference's segment boundaries.
    auto emit_segment = [&] {
        const int segment_frames = last_speech_frame_ + 1 - confirmed_start_;
        const size_t target_samples =
            static_cast<size_t>(segment_frames) * static_cast<size_t>(hop_size_);
        if (segment_audio_.size() > target_samples) {
            segment_audio_.resize(target_samples);
        }

        if (on_segment_ && !segment_audio_.empty()) {
            const double frame_dur_s =
                static_cast<double>(hop_size_) / kSampleRate;
            VadSegment seg;
            seg.samples = segment_audio_;
            seg.start_s = confirmed_start_ * frame_dur_s;
            seg.end_s   = (last_speech_frame_ + 1) * frame_dur_s;
            on_segment_(seg);
        }

        segment_audio_.clear();
        state_ = State::SILENCE;
        consec_speech_ = 0;
        consec_silence_ = 0;
    };

    switch (state_) {
    case State::SILENCE:
        if (flag) {
            // First speech frame after silence — begin pending speech and
            // start accumulating audio from this frame onward.
            pending_start_frame_ = frame_idx_;
            consec_speech_ = 1;
            consec_silence_ = 0;
            segment_audio_.insert(segment_audio_.end(), samples, samples + n);
            if (consec_speech_ >= min_speech_frames_) {
                // min_speech_frames_ == 1: confirm immediately.
                confirmed_start_ = pending_start_frame_;
                last_speech_frame_ = frame_idx_;
                speech_frame_count_ = consec_speech_;
                state_ = State::IN_SPEECH;
            } else {
                state_ = State::PENDING_SPEECH;
            }
        }
        break;

    case State::PENDING_SPEECH:
        if (flag) {
            ++consec_speech_;
            consec_silence_ = 0;
            segment_audio_.insert(segment_audio_.end(), samples, samples + n);
            if (consec_speech_ >= min_speech_frames_) {
                // Speech confirmed — promote to IN_SPEECH. The pending audio
                // already covers [pending_start_frame_, frame_idx_] which
                // becomes [confirmed_start_, last_speech_frame_].
                confirmed_start_ = pending_start_frame_;
                last_speech_frame_ = frame_idx_;
                speech_frame_count_ = consec_speech_;
                state_ = State::IN_SPEECH;
            }
        } else {
            // Pending speech failed (silence before reaching min_speech_frames)
            // — discard the accumulated audio and return to SILENCE.
            consec_speech_ = 0;
            segment_audio_.clear();
            state_ = State::SILENCE;
        }
        break;

    case State::IN_SPEECH:
        // Accumulate every frame while in IN_SPEECH (speech + interspersed
        // silence). Trailing silence is trimmed on finalization.
        segment_audio_.insert(segment_audio_.end(), samples, samples + n);
        if (flag) {
            ++speech_frame_count_;
            last_speech_frame_ = frame_idx_;
            consec_silence_ = 0;
            if (speech_frame_count_ >= max_speech_frames_) {
                // Force-finalize on max_speech cap.
                emit_segment();
            }
        } else {
            ++consec_silence_;
            if (consec_silence_ >= min_silence_frames_) {
                emit_segment();
            }
        }
        break;
    }

    ++frame_idx_;
}

void VadService::flush() {
    // Finalize any in-progress segment (reference Segmenter::finish()).
    if (state_ == State::IN_SPEECH && !segment_audio_.empty()) {
        // Trim trailing silence: segment is [confirmed_start_, last_speech_frame_+1).
        const int segment_frames = last_speech_frame_ + 1 - confirmed_start_;
        const size_t target_samples =
            static_cast<size_t>(segment_frames) * static_cast<size_t>(hop_size_);
        if (segment_audio_.size() > target_samples) {
            segment_audio_.resize(target_samples);
        }

        if (on_segment_) {
            const double frame_dur_s =
                static_cast<double>(hop_size_) / kSampleRate;
            VadSegment seg;
            seg.samples = segment_audio_;
            seg.start_s = confirmed_start_ * frame_dur_s;
            seg.end_s   = (last_speech_frame_ + 1) * frame_dur_s;
            on_segment_(seg);
        }
    }
    // Reset segmenter state.
    segment_audio_.clear();
    state_ = State::SILENCE;
    consec_speech_ = 0;
    consec_silence_ = 0;
}

void VadService::reset() {
    // Hard reset of the segmenter state machine — no segment is emitted.
    // The ten-vad handle is left intact (used for hard interrupt).
    state_ = State::SILENCE;
    pending_start_frame_ = 0;
    confirmed_start_ = 0;
    last_speech_frame_ = 0;
    consec_speech_ = 0;
    consec_silence_ = 0;
    speech_frame_count_ = 0;
    frame_idx_ = 0;
    segment_audio_.clear();
}

} // namespace world
} // namespace winefox
