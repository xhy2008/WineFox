#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>

// Forward declarations or placeholder for dependencies
class Tokenizer;
struct KoKoroConfig;

// Constants from config
const int MAX_PHONEME_LENGTH = 510; // Example value
const int SAMPLE_RATE = 24000;      // Example value

// Kokoro TTS engine.
//
// Two construction modes:
//   1. Split mode (recommended): separate encoder (FP32) + decoder (INT8)
//      ONNX sessions. Enables INT8 quantization on the conv-heavy decoder
//      while keeping the BERT-based encoder in FP32 for accuracy. Each
//      session can be tuned with its own thread count.
//   2. Merged mode (legacy): single ONNX session holding the full model.
//      Kept for backwards compatibility; cannot leverage INT8 decoder.
//
// Streaming: create_stream() invokes the callback per phoneme batch
// (typically per sentence), enabling low-TTFB playback.
class Kokoro {
public:
    // Split mode constructor (recommended).
    //   encoder_path: kokoro-encoder.onnx (FP32)
    //   decoder_path: kokoro-decoder-int8-static.onnx (INT8) or kokoro-decoder.onnx (FP32)
    //   n_threads: 0 = auto (physical cores), >0 = explicit thread count
    Kokoro(const std::string& encoder_path,
           const std::string& decoder_path,
           const std::string& voices_path,
           const std::string& vocab_path,
           int n_threads = 0);

    // Merged mode constructor (legacy).
    Kokoro(const std::string& model_path,
           const std::string& voices_path,
           const std::string& vocab_path);

    ~Kokoro();

    std::vector<float> get_voice_style(const std::string& name);

    // Non-streaming synthesis: returns full audio.
    std::pair<std::vector<float>, int> create(
        const std::string& text,
        const std::string& voice_name,
        float speed = 1.0f,
        bool is_phonemes = false,
        bool trim = true
    );

    // Overload for passing voice style directly.
    std::pair<std::vector<float>, int> create(
        const std::string& text,
        const std::vector<float>& voice_style,
        float speed = 1.0f,
        bool is_phonemes = false,
        bool trim = true
    );

    // Streaming synthesis: calls on_audio_chunk for each phoneme batch.
    // The callback receives (audio_chunk, sample_rate) and should return
    // false to stop early, true to continue.
    void create_stream(
        const std::string& text,
        const std::string& voice_name,
        const std::function<bool(const std::vector<float>&, int)>& on_audio_chunk,
        float speed = 1.0f,
        bool is_phonemes = false
    );

    // Report which mode the engine is running in.
    bool is_split_mode() const { return split_mode_; }

private:
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Merged mode: single session. Split mode: nullptr.
    Ort::Session merged_session_{nullptr};

    // Split mode: separate sessions. Merged mode: both nullptr.
    Ort::Session enc_session_{nullptr};
    Ort::Session dec_session_{nullptr};

    bool split_mode_ = false;
    int n_threads_ = 0;

    std::map<std::string, std::vector<float>> voices_;
    std::unique_ptr<Tokenizer> tokenizer_;

    // Reusable buffers for split-mode inference. Avoids re-allocating
    // tokens/style/speed vectors on every chunk in streaming mode.
    // NOTE: not thread-safe; create_stream/create must be called from one
    // thread (the audio pipeline's TTS worker).
    Ort::MemoryInfo mem_info_ = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> tokens_buf_;        // [BOS, ...tokens, EOS]
    std::vector<float>   style_buf_;         // 256 floats, indexed by token length
    std::vector<float>   speed_buf_ = {1.0f};
    std::vector<int64_t> ids_shape_  = {1, 0};
    std::vector<int64_t> ref_shape_  = {1, 256};
    std::vector<int64_t> speed_shape_ = {1};

    // Static input/output name arrays — ORT requires char** but the names
    // themselves never change, so make them static const.
    static const char* kEncInputNames[3];
    static const char* kEncOutputNames[5];
    static const char* kDecInputNames[4];
    static const char* kDecOutputNames[1];

    void load_voices(const std::string& voices_path);

    // Split-mode encoder+decoder path. Returns audio for one phoneme batch.
    std::pair<std::vector<float>, int> _create_audio_split(
        const std::string& phonemes,
        const std::vector<float>& voice,
        float speed
    );

    // Merged-mode path (legacy).
    std::pair<std::vector<float>, int> _create_audio_merged(
        const std::string& phonemes,
        const std::vector<float>& voice,
        float speed
    );

    // Dispatcher used by create() and create_stream().
    std::pair<std::vector<float>, int> _create_audio(
        const std::string& phonemes,
        const std::vector<float>& voice,
        float speed
    ) {
        return split_mode_ ? _create_audio_split(phonemes, voice, speed)
                           : _create_audio_merged(phonemes, voice, speed);
    }

    std::vector<std::string> _split_phonemes(const std::string& phonemes);

    // Run a short dummy forward to prime ORT kernels & memory arena.
    void warmup_split();
};
