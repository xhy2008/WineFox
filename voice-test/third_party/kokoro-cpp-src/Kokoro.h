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
    //   encoder_path: kokoro-encoder.onnx (FP32) or kokoro-encoder-int8.onnx
    //                 (INT8 QDQ, ~25% smaller, Conv-only quantization)
    //   decoder_path: kokoro-decoder-int8-static.onnx (INT8) or kokoro-decoder.onnx (FP32)
    //   n_threads: 0 = auto (physical cores), >0 = explicit thread count
    //   dict_dir: G2P dictionary directory (default "dict/")
    Kokoro(const std::string& encoder_path,
           const std::string& decoder_path,
           const std::string& voices_path,
           const std::string& vocab_path,
           int n_threads = 0,
           const std::string& dict_dir = "");

    // Merged mode constructor (legacy).
    Kokoro(const std::string& model_path,
           const std::string& voices_path,
           const std::string& vocab_path);

    ~Kokoro();

    std::vector<float> get_voice_style(const std::string& name);

    // Preload specific voices into the in-memory cache. Call this if you
    // know which voices will be used and want to avoid file I/O during
    // synthesis. Without this, voices are loaded on first use.
    void preload_voices(const std::vector<std::string>& names);

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

    // Debug: phonemize text without synthesizing (used by --phonemize).
    std::string phonemize_debug(const std::string& text);

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

    // Voice storage with lazy loading.
    //
    // At construction, load_voices() scans voices.bin once to build a
    // name → (file_offset, dim) index but does NOT load the actual style
    // data. Voices are loaded on demand in get_voice_style() and cached
    // in voices_cache_. This reduces steady-state memory from ~51MB
    // (all 54 voices) to ~1MB (1 voice) for the typical single-voice
    // use case.
    struct VoiceIndexEntry {
        std::streamoff offset;  // byte offset of style data in voices.bin
        uint32_t dim;           // number of floats in the style vector
    };
    std::map<std::string, VoiceIndexEntry> voice_index_;
    std::string voices_file_path_;
    std::map<std::string, std::vector<float>> voices_cache_;

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
