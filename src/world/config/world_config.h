// world_config.h — World-specific configuration (voice/AEC/window params).
//
// Reads the "voice" and "window" sections from winefox.json. The LLM/memory
// sections are handled by winefox_core.dll internally.

#pragma once

#include <string>

namespace winefox {
namespace world {

struct WorldConfig {
    // --- VAD ---
    struct Vad {
        std::string model_path;     // ten-vad-ggml.bin path
        float threshold      = 0.5f;
        int   hop_size       = 256;    // samples per frame (16ms @ 16kHz)
        float min_speech     = 0.25f;  // seconds
        float min_silence    = 0.30f;
        float max_speech     = 30.0f;
    } vad;

    // --- ASR ---
    struct Asr {
        std::string model_path;
        std::string language  = "auto";
        int         threads   = 4;
        bool        use_itn   = true;
        bool        flash_attn = true;
    } asr;

    // --- TTS ---
    struct Tts {
        std::string encoder_path;
        std::string decoder_path;
        std::string voices_path;
        std::string vocab_path  = "dict/vocab.txt";
        std::string dict_dir    = "dict";
        std::string voice_name  = "winefox";
        float       speed       = 1.0f;
        int         threads     = 0;      // 0 = auto
    } tts;

    // --- AEC ---
    struct Aec {
        bool enabled = false;
        int  level   = 1;   // 0=off, 1=mild, 2=aggressive
    } aec;

    // --- Window ---
    struct Window {
        int         width  = 1280;
        int         height = 720;
        std::string title  = "WineFox";
    } window;

    // Load voice + window sections from a JSON config file.
    // Returns false if the file cannot be opened or parsed.
    bool load(const std::string& path);

    // Generate a default config JSON string (for --gen-config equivalent).
    static WorldConfig defaults();
};

} // namespace world
} // namespace winefox
