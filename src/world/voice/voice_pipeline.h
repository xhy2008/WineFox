// voice_pipeline.h — Voice conversation pipeline state machine.
//
// Orchestrates the full voice loop: VAD → ASR → LLM → TTS → playback,
// with hard interrupt support.
//
// State machine:
//   IDLE → (VAD detects speech) → LISTENING
//   LISTENING → (VAD detects silence/end) → RECOGNIZING
//   RECOGNIZING → (ASR returns text) → THINKING
//   THINKING → (LLM streams first token) → SPEAKING
//   SPEAKING → (TTS playback finished OR interrupt) → IDLE
//
// Hard interrupt: when VAD detects speech during SPEAKING state, the
// pipeline immediately:
//   1. Stops TTS synthesis (abort callback returns false)
//   2. Aborts LLM generation (winefox_chat callback returns 0)
//   3. Flushes the output audio buffer
//   4. Transitions to LISTENING with the new speech segment
//
// The pipeline runs on the AI thread. Audio I/O callbacks (from SDL audio
// thread) push mic samples into a ring buffer; the AI thread pulls them,
// runs VAD, and when a segment is finalized, runs ASR → LLM → TTS.
// TTS output is pushed into an output ring buffer consumed by the SDL
// audio playback callback.
//
// Events (state changes, subtitles, emotion) are pushed to the render
// thread via an SpscQueue for UI display.

#pragma once

#include "../config/world_config.h"
#include "../ipc/message_queue.h"
#include "../platform/audio_io.h"
#include "aec_service.h"
#include "asr_service.h"
#include "tts_service.h"
#include "vad_service.h"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// Forward declaration of the winefox_core C ABI handle
struct WineFoxCore;

namespace winefox {
namespace world {

enum class PipelineState : int {
    IDLE        = 0,
    LISTENING   = 1,
    RECOGNIZING = 2,
    THINKING    = 3,
    SPEAKING    = 4,
};

class VoicePipeline {
public:
    VoicePipeline();
    ~VoicePipeline();

    // --- Async init (for fast startup) ---
    // Step 1: Load voice models (TTS, VAD, ASR, AEC). Does NOT require core.
    //         Can run in parallel with winefox_init on another thread.
    bool init_models(const WorldConfig& cfg,
                     ipc::SpscQueue<ipc::RenderEvent, 256>* render_events);
    // Step 2: Set the AI core handle (after winefox_init completes).
    void set_core(WineFoxCore* core);
    // Step 3: Initialize audio I/O (fast — just opens SDL audio devices).
    //         Must be called after init_models and set_core.
    bool init_audio_io();

    // Start the AI thread. Returns false if already running.
    bool start();

    // Signal the AI thread to stop and wait for it to finish.
    void stop();

    // Push a command from the render thread (interrupt/quit).
    void push_command(const ipc::AiCommand& cmd) {
        commands_.try_push(cmd);
    }

private:
    // The AI thread main loop.
    void run_();

    // Audio I/O callbacks (called from SDL audio thread).
    void on_mic_input_(const int16_t* samples, int n);
    void on_far_end_(const int16_t* samples, int n);

    // VAD segment callback (called from AI thread during feed_).
    void on_vad_segment_(const VadSegment& seg);

    // Process one full conversation turn: ASR → LLM → TTS.
    void process_turn_(const std::vector<int16_t>& speech);

    // Helpers
    void set_state_(PipelineState s);
    void emit_subtitle_(const std::string& text);
    void emit_emotion_(const std::string& tag);
    void emit_perf_(int n_eval, double tps);

    // Push TTS audio to the output ring buffer (called from AI thread).
    void enqueue_tts_audio_(const int16_t* samples, int n);

    // Resample 24kHz TTS audio to 16kHz for AEC far-end reference.
    void resample_tts_to_16k_(const int16_t* in24k, int n,
                               std::vector<int16_t>& out16k);

    // --- Config ---
    WorldConfig cfg_;

    // --- External handles ---
    WineFoxCore* core_ = nullptr;
    ipc::SpscQueue<ipc::RenderEvent, 256>* render_events_ = nullptr;

    // --- Voice services ---
    VadService vad_;
    AsrService asr_;
    TtsService tts_;
    AecService aec_;

    // --- Audio I/O ---
    AudioIo audio_io_;
    AudioRingBuffer mic_buffer_;      // 16kHz, AI thread consumes
    AudioRingBuffer far_end_buffer_;  // 16kHz (resampled TTS), for AEC

    // --- AI thread ---
    std::thread       ai_thread_;
    std::atomic<bool> running_{false};

    // --- Commands (render → AI) ---
    ipc::SpscQueue<ipc::AiCommand, 64> commands_;

    // --- State ---
    std::atomic<PipelineState> state_{PipelineState::IDLE};
    std::atomic<bool> interrupt_requested_{false};

    // --- TTS streaming: accumulate sentence for synthesis ---
    std::string sentence_buffer_;

    // --- Resampling scratch ---
    std::vector<int16_t> resample_buf_;
};

} // namespace world
} // namespace winefox
