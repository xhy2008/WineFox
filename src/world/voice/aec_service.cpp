// aec_service.cpp — Acoustic Echo Cancellation.
//
// Default implementation: Normalized Least Mean Squares (NLMS) adaptive filter.
// Self-contained, zero external dependencies, extremely lightweight (~50 lines
// of core logic). Provides real echo cancellation suitable for the
// speaker-output scenario.
//
// To swap in WebRTC AEC3:
//   1. Vendor webrtc-audio-processing under third_party/webrtc
//   2. Replace the NLMS core in process_frame() with EchoCanceller3 calls
//   3. Add webrtc to CMakeLists.txt link chain
// The AecService interface stays identical — only the implementation changes.

#include "aec_service.h"

#include <cstdio>
#include <cstring>

namespace winefox {
namespace world {

namespace {

// NLMS adaptive filter parameters.
//   filter_len:  number of taps (must cover echo tail; 1024 taps = 64ms @ 16kHz)
//   step_size:   adaptation rate (0..1; higher = faster convergence but noisier)
//   regularization: prevents division by zero when far-end energy is ~0
constexpr int   kFilterLen     = 1024;  // 64ms echo tail @ 16kHz
constexpr float kStepSize     = 0.5f;
constexpr float kRegularization = 1e-6f;

// Simple ring buffer for the far-end reference history.
struct NlmsState {
    float    w[kFilterLen] = {};       // adaptive filter coefficients
    float    far_buf[kFilterLen] = {}; // far-end reference history (ring)
    int      far_pos = 0;              // ring buffer write position
};

} // namespace

bool AecService::init(int level) {
    enabled_ = (level > 0);
    level_ = level;
    if (!enabled_) return true;  // pass-through mode

    handle_ = new NlmsState();

    near_f_.resize(kFrameSize);
    far_f_.resize(kFrameSize);
    out_f_.resize(kFrameSize);

    std::fprintf(stderr, "[aec] NLMS AEC initialised (filter_len=%d, step=%.2f)\n",
                 kFilterLen, kStepSize);
    return true;
}

AecService::~AecService() {
    if (handle_) {
        delete static_cast<NlmsState*>(handle_);
        handle_ = nullptr;
    }
}

void AecService::process_frame(const int16_t* near, const int16_t* far, int16_t* out) {
    if (!enabled_ || !handle_) {
        // Pass-through: copy near directly to out.
        if (near && out) std::memcpy(out, near, kFrameSize * sizeof(int16_t));
        return;
    }

    auto* s = static_cast<NlmsState*>(handle_);

    // Convert int16 → float.
    for (int i = 0; i < kFrameSize; ++i) {
        near_f_[i] = float(near[i]);
        far_f_[i]  = far ? float(far[i]) : 0.0f;
    }

    // NLMS: for each sample, estimate echo from far-end history, subtract
    // from near, then adapt the filter coefficients.
    for (int i = 0; i < kFrameSize; ++i) {
        // Push far-end sample into ring buffer.
        s->far_buf[s->far_pos] = far_f_[i];

        // Compute echo estimate: dot product of filter coeffs and far-end history.
        float echo = 0.0f;
        int idx = s->far_pos;
        for (int j = 0; j < kFilterLen; ++j) {
            echo += s->w[j] * s->far_buf[idx];
            idx = (idx > 0) ? idx - 1 : kFilterLen - 1;
        }

        // Error signal = near - echo_estimate (the echo-cancelled output).
        float error = near_f_[i] - echo;
        out_f_[i] = error;

        // Compute far-end energy for normalization.
        float far_energy = kRegularization;
        idx = s->far_pos;
        for (int j = 0; j < kFilterLen; ++j) {
            far_energy += s->far_buf[idx] * s->far_buf[idx];
            idx = (idx > 0) ? idx - 1 : kFilterLen - 1;
        }

        // Adapt filter coefficients: w += step * error * far / energy.
        float adapt = kStepSize * error / far_energy;
        idx = s->far_pos;
        for (int j = 0; j < kFilterLen; ++j) {
            s->w[j] += adapt * s->far_buf[idx];
            idx = (idx > 0) ? idx - 1 : kFilterLen - 1;
        }

        // Advance ring buffer position.
        s->far_pos = (s->far_pos + 1) % kFilterLen;
    }

    // Convert float → int16.
    for (int i = 0; i < kFrameSize; ++i) {
        float v = out_f_[i];
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        out[i] = static_cast<int16_t>(v);
    }
}

void AecService::process(const int16_t* near, const int16_t* far, int n,
                         std::vector<int16_t>& out) {
    if (!enabled_ || !handle_) {
        // Pass-through.
        out.assign(near, near + n);
        return;
    }

    out.resize(n);
    // Process in 10ms (160-sample) frames. The last partial frame is
    // zero-padded internally.
    int processed = 0;
    int16_t near_frame[kFrameSize];
    int16_t far_frame[kFrameSize];
    int16_t out_frame[kFrameSize];

    while (processed < n) {
        int remaining = n - processed;
        int chunk = (remaining < kFrameSize) ? remaining : kFrameSize;

        std::memcpy(near_frame, near + processed, chunk * sizeof(int16_t));
        if (chunk < kFrameSize) {
            std::memset(near_frame + chunk, 0, (kFrameSize - chunk) * sizeof(int16_t));
        }

        if (far) {
            std::memcpy(far_frame, far + processed, chunk * sizeof(int16_t));
            if (chunk < kFrameSize) {
                std::memset(far_frame + chunk, 0, (kFrameSize - chunk) * sizeof(int16_t));
            }
        } else {
            std::memset(far_frame, 0, kFrameSize * sizeof(int16_t));
        }

        process_frame(near_frame, far ? far_frame : nullptr, out_frame);

        int copy = (remaining < kFrameSize) ? remaining : kFrameSize;
        std::memcpy(out.data() + processed, out_frame, copy * sizeof(int16_t));
        processed += copy;
    }
}

} // namespace world
} // namespace winefox
