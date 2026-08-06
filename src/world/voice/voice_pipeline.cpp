#include "voice_pipeline.h"

#include "winefox_api.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

namespace winefox {
namespace world {

// Check if the string ends with a sentence boundary character.
// Handles ASCII punctuation and UTF-8 encoded Chinese punctuation.
static bool ends_with_sentence_boundary(const std::string& s) {
    if (s.empty()) return false;

    // ASCII punctuation — single byte.
    char last = s.back();
    if (last == '.' || last == '!' || last == '?' || last == '\n') return true;

    // UTF-8 Chinese punctuation — 3-byte sequences:
    //   。= E3 80 82   ！= EF BC 81   ？= EF BC 9F   ，= EF BC 8C
    static const char* boundaries[] = {
        "\xE3\x80\x82",  // 。
        "\xEF\xBC\x81",  // ！
        "\xEF\xBC\x9F",  // ？
        "\xEF\xBC\x8C",  // ，(comma — also a good TTS split point)
    };
    const size_t seq_len = 3;
    if (s.size() < seq_len) return false;
    for (const char* b : boundaries) {
        if (s.compare(s.size() - seq_len, seq_len, b) == 0) return true;
    }
    return false;
}

// Ring buffer capacities: ~1 second of audio at each rate.
static constexpr size_t kMicBufSize     = 16000 * 2;  // 16kHz × 2s
static constexpr size_t kFarEndBufSize  = 16000 * 2;  // 16kHz × 2s

VoicePipeline::VoicePipeline()
    : mic_buffer_(kMicBufSize),
      far_end_buffer_(kFarEndBufSize) {}

VoicePipeline::~VoicePipeline() { stop(); }

bool VoicePipeline::init_models(const WorldConfig& cfg,
                                 ipc::SpscQueue<ipc::RenderEvent, 256>* render_events) {
    cfg_ = cfg;
    render_events_ = render_events;

    auto t0 = std::chrono::steady_clock::now();
    auto ms_since = [&](const std::chrono::steady_clock::time_point& prev) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prev).count();
    };

    // --- TTS ---
    if (!tts_.init(cfg_.tts.encoder_path, cfg_.tts.decoder_path,
                   cfg_.tts.voices_path, cfg_.tts.vocab_path, cfg_.tts.threads)) {
        std::fprintf(stderr, "[pipeline] TTS init failed\n");
        return false;
    }
    std::fprintf(stderr, "[pipeline] TTS loaded: %.0f ms\n", ms_since(t0));

    // --- VAD ---
    auto t1 = std::chrono::steady_clock::now();
    if (!vad_.init(cfg_.vad.model_path, cfg_.vad.threshold, cfg_.vad.hop_size,
                   cfg_.vad.min_speech, cfg_.vad.min_silence, cfg_.vad.max_speech,
                   [this](const VadSegment& seg) { on_vad_segment_(seg); })) {
        std::fprintf(stderr, "[pipeline] VAD init failed\n");
        return false;
    }
    std::fprintf(stderr, "[pipeline] VAD loaded: %.0f ms\n", ms_since(t1));

    // --- ASR ---
    auto t2 = std::chrono::steady_clock::now();
    if (!asr_.init(cfg_.asr.model_path, cfg_.asr.language,
                   cfg_.asr.threads, cfg_.asr.use_itn, cfg_.asr.flash_attn)) {
        std::fprintf(stderr, "[pipeline] ASR init failed\n");
        return false;
    }
    std::fprintf(stderr, "[pipeline] ASR loaded: %.0f ms\n", ms_since(t2));

    // --- AEC ---
    if (cfg_.aec.enabled) {
        if (!aec_.init(cfg_.aec.level)) {
            std::fprintf(stderr, "[pipeline] AEC init failed (disabling)\n");
        }
    }

    std::fprintf(stderr, "[pipeline] voice models loaded: %.0f ms total\n", ms_since(t0));
    return true;
}

void VoicePipeline::set_core(WineFoxCore* core) {
    core_ = core;
}

bool VoicePipeline::init_audio_io() {
    if (!audio_io_.init(
            [this](const int16_t* s, int n) { on_mic_input_(s, n); },
            [this](const int16_t* s, int n) { on_far_end_(s, n); })) {
        std::fprintf(stderr, "[pipeline] audio I/O init failed\n");
        return false;
    }
    std::fprintf(stderr, "[pipeline] audio I/O ready\n");
    return true;
}

bool VoicePipeline::start() {
    if (running_.load()) return false;
    running_.store(true);
    ai_thread_ = std::thread([this] { run_(); });
    return true;
}

void VoicePipeline::stop() {
    if (!running_.load()) return;
    running_.store(false);
    ipc::AiCommand quit_cmd;
    quit_cmd.kind = ipc::AiCommandKind::Quit;
    commands_.try_push(quit_cmd);
    if (ai_thread_.joinable()) ai_thread_.join();
}

// ===========================================================================
// AI thread main loop
// ===========================================================================

void VoicePipeline::run_() {
    std::fprintf(stderr, "[pipeline] AI thread started\n");

    std::vector<int16_t> mic_chunk;
    mic_chunk.resize(vad_.hop_size());

    while (running_.load()) {
        // Check for commands from render thread.
        ipc::AiCommand cmd;
        while (commands_.try_pop(cmd)) {
            if (cmd.kind == ipc::AiCommandKind::Quit) {
                running_.store(false);
                goto done;
            }
            if (cmd.kind == ipc::AiCommandKind::Interrupt) {
                interrupt_requested_.store(true);
            }
        }

        // Pull mic samples from ring buffer and feed VAD.
        size_t available = mic_buffer_.available();
        if (available >= (size_t)vad_.hop_size()) {
            size_t got = mic_buffer_.read(mic_chunk.data(), vad_.hop_size());
            if (got > 0) {
                // Check for interrupt during SPEAKING state.
                PipelineState s = state_.load();
                if (s == PipelineState::SPEAKING && interrupt_requested_.load()) {
                    // Hard interrupt: stop TTS + LLM, flush output, go to LISTENING.
                    std::fprintf(stderr, "[pipeline] HARD INTERRUPT\n");
                    interrupt_requested_.store(false);
                    audio_io_.flush_output();
                    vad_.reset();
                    set_state_(PipelineState::LISTENING);
                }

                // Feed VAD. The on_vad_segment_ callback fires when a segment
                // is finalized, which triggers process_turn_.
                vad_.feed(mic_chunk.data(), (int)got);
            }
        } else {
            // No audio available — yield to avoid busy-spinning.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

done:
    set_state_(PipelineState::IDLE);
    std::fprintf(stderr, "[pipeline] AI thread stopped\n");
}

// ===========================================================================
// Audio I/O callbacks (SDL audio thread)
// ===========================================================================

void VoicePipeline::on_mic_input_(const int16_t* samples, int n) {
    // If AEC is enabled, we need the far-end reference. For simplicity,
    // we apply AEC here using the far-end buffer that was filled by
    // on_far_end_. The far-end buffer holds the 24kHz TTS output resampled
    // to 16kHz.
    if (aec_.enabled() && aec_.ready()) {
        // Pull matching amount of far-end samples (may be silence if TTS idle).
        std::vector<int16_t> far(n, 0);
        size_t far_available = far_end_buffer_.available();
        size_t to_read = std::min((size_t)n, far_available);
        if (to_read > 0) {
            far_end_buffer_.read(far.data(), to_read);
        }
        std::vector<int16_t> cleaned;
        aec_.process(samples, far.data(), n, cleaned);
        mic_buffer_.write(cleaned.data(), cleaned.size());
    } else {
        mic_buffer_.write(samples, n);
    }
}

void VoicePipeline::on_far_end_(const int16_t* samples, int n) {
    // samples is 24kHz (speaker output). Resample to 16kHz for AEC reference.
    if (aec_.enabled()) {
        resample_tts_to_16k_(samples, n, resample_buf_);
        far_end_buffer_.write(resample_buf_.data(), resample_buf_.size());
    }
}

// ===========================================================================
// VAD segment callback — triggers ASR → LLM → TTS
// ===========================================================================

void VoicePipeline::on_vad_segment_(const VadSegment& seg) {
    // Only process segments when in IDLE or LISTENING state.
    PipelineState s = state_.load();
    if (s == PipelineState::THINKING || s == PipelineState::SPEAKING) {
        // We're mid-response. If this is a new segment during SPEAKING,
        // it's an interrupt (handled in the main loop). Ignore here.
        return;
    }

    if (seg.samples.empty()) return;

    // Move segment audio into a local vector for processing.
    std::vector<int16_t> speech = seg.samples;
    process_turn_(speech);
}

// ===========================================================================
// Process one conversation turn: ASR → LLM → TTS
// ===========================================================================

void VoicePipeline::process_turn_(const std::vector<int16_t>& speech) {
    // --- ASR ---
    set_state_(PipelineState::RECOGNIZING);
    std::string text = asr_.recognize(speech.data(), (int)speech.size());
    if (text.empty()) {
        std::fprintf(stderr, "[pipeline] ASR returned empty text\n");
        set_state_(PipelineState::IDLE);
        return;
    }

    emit_subtitle_(std::string("[user] ") + text);
    std::fprintf(stderr, "[pipeline] ASR: %s\n", text.c_str());

    // --- LLM + TTS ---
    set_state_(PipelineState::THINKING);

    sentence_buffer_.clear();

    // Streaming TTS: accumulate LLM tokens into sentence_buffer_. When a
    // sentence boundary (。！？，.!?newline) is reached, synthesize that
    // sentence immediately and push audio to playback. This gives low TTFB
    // (time-to-first-byte of audio) while the LLM is still generating.
    const char* reply = winefox_chat(core_, text.c_str(),
        [](const char* token, void* user) -> int {
            auto* self = static_cast<VoicePipeline*>(user);

            // Check for interrupt.
            if (self->interrupt_requested_.load()) {
                return 0;  // abort generation
            }

            // Transition to SPEAKING on first token.
            if (self->state_.load() == PipelineState::THINKING) {
                self->set_state_(PipelineState::SPEAKING);
            }

            self->sentence_buffer_ += token;

            // If we hit a sentence boundary, synthesize the accumulated sentence.
            if (ends_with_sentence_boundary(self->sentence_buffer_)) {
                std::string sentence = std::move(self->sentence_buffer_);
                self->sentence_buffer_.clear();
                self->tts_.synth_stream(sentence, self->cfg_.tts.speed,
                    self->cfg_.tts.voice_name,
                    [self](const int16_t* samples, int n) -> bool {
                        if (self->interrupt_requested_.load()) return false;
                        self->enqueue_tts_audio_(samples, n);
                        return true;
                    });
            }
            return 1;  // continue
        }, this);

    // Synthesize any remaining text in the buffer (last sentence without
    // trailing punctuation).
    if (!sentence_buffer_.empty() && !interrupt_requested_.load()) {
        std::string sentence = std::move(sentence_buffer_);
        sentence_buffer_.clear();
        tts_.synth_stream(sentence, cfg_.tts.speed, cfg_.tts.voice_name,
            [this](const int16_t* samples, int n) -> bool {
                if (interrupt_requested_.load()) return false;
                enqueue_tts_audio_(samples, n);
                return true;
            });
    }

    // Emit the full reply as subtitle.
    if (reply) {
        emit_subtitle_(std::string("[fox] ") + reply);
        winefox_free_string(reply);
    }

    // Emit emotion.
    const char* emotion = winefox_last_emotion(core_);
    if (emotion) {
        emit_emotion_(emotion);
    }

    // Emit perf data.
    WineFoxPerf perf;
    winefox_last_perf(core_, &perf);
    emit_perf_(perf.n_eval, perf.tokens_per_sec);

    // Wait for playback to drain (SDL audio stream finishes playing).
    if (state_.load() == PipelineState::SPEAKING) {
        while (audio_io_.output_has_data() && running_.load() &&
               !interrupt_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    interrupt_requested_.store(false);
    set_state_(PipelineState::IDLE);
}

// ===========================================================================
// Helpers
// ===========================================================================

void VoicePipeline::set_state_(PipelineState s) {
    state_.store(s);
    ipc::RenderEvent e;
    e.kind     = ipc::RenderEventKind::StateChange;
    e.int_val  = static_cast<int32_t>(s);
    if (render_events_) render_events_->try_push(e);
}

void VoicePipeline::emit_subtitle_(const std::string& text) {
    ipc::RenderEvent e;
    e.kind = ipc::RenderEventKind::Subtitle;
    std::strncpy(e.text, text.c_str(), sizeof(e.text) - 1);
    e.text[sizeof(e.text) - 1] = '\0';
    if (render_events_) render_events_->try_push(e);
}

void VoicePipeline::emit_emotion_(const std::string& tag) {
    ipc::RenderEvent e;
    e.kind = ipc::RenderEventKind::Emotion;
    std::strncpy(e.text, tag.c_str(), sizeof(e.text) - 1);
    e.text[sizeof(e.text) - 1] = '\0';
    if (render_events_) render_events_->try_push(e);
}

void VoicePipeline::emit_perf_(int n_eval, double tps) {
    ipc::RenderEvent e;
    e.kind      = ipc::RenderEventKind::Perf;
    e.int_val   = n_eval;
    e.float_val = tps;
    if (render_events_) render_events_->try_push(e);
}

void VoicePipeline::enqueue_tts_audio_(const int16_t* samples, int n) {
    // TTS outputs 24kHz S16 mono — push directly to SDL's audio stream.
    // SDL internally handles resampling to the device's actual format.
    // The far-end reference for AEC is tapped inside push_output().
    audio_io_.push_output(samples, n);
}

void VoicePipeline::resample_tts_to_16k_(const int16_t* in24k, int n,
                                          std::vector<int16_t>& out16k) {
    // Simple linear interpolation downsampling 24kHz → 16kHz.
    // Ratio = 24/16 = 3/2, so for every 3 input samples we produce 2 output samples.
    // This is not ideal quality but is extremely lightweight (no external deps).
    // For production quality, use a proper resampler (e.g., speex, libsamplerate).
    out16k.clear();
    out16k.reserve(n * 2 / 3 + 1);

    const double ratio = 16.0 / 24.0;
    int out_len = (int)(n * ratio);
    for (int i = 0; i < out_len; ++i) {
        double src_pos = (double)i / ratio;
        int    src_idx = (int)src_pos;
        double frac    = src_pos - src_idx;
        if (src_idx + 1 >= n) {
            out16k.push_back(in24k[n - 1]);
        } else {
            double s = (double)in24k[src_idx] * (1.0 - frac) +
                       (double)in24k[src_idx + 1] * frac;
            out16k.push_back(static_cast<int16_t>(s));
        }
    }
}

} // namespace world
} // namespace winefox
