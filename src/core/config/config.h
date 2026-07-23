#pragma once

// Central configuration for WineFox.
//
// All configurable parameters live here. The config is loaded from a JSON file
// (default: ./winefox.json, overridable via --config). CLI arguments act as
// overrides on top of the loaded config.
//
// To add a new parameter:
//   1. Add the field to the appropriate nested struct below.
//   2. Add parsing in config.cpp (Config::load).
//   3. Add serialisation in config.cpp (Config::save).
//   4. Consume it where needed.

#include <cstdint>
#include <string>

namespace winefox {
namespace config {

struct Config {
    // --- LLM backend ---
    struct Llm {
        std::string model_path;
        std::string lora_path;
        float       lora_scale    = 1.0f;
        int         n_ctx         = 4096;
        int         n_batch       = 2048;
        int         n_threads     = 0;      // 0 = auto (physical cores)
        bool        use_mmap      = true;
        bool        enable_thinking = false;  // false = -rea off, suppress <think>
        bool        flash_attention_enabled = true;  // LLAMA_FLASH_ATTN_TYPE_ENABLED
        std::string kv_cache_dtype = "f16";  // "f16", "q8_0", "q4_0"
    } llm;

    // --- Sampling parameters ---
    struct Sampling {
        float     temp           = 0.7f;
        int       top_k          = 40;
        float     top_p          = 0.9f;
        float     repeat_penalty = 1.10f;
        int       penalty_last_n = 64;
        int       max_tokens     = 512;
        uint32_t  seed           = 0xC0FFEE;
    } sampling;

    // --- Embedder ---
    struct Embedder {
        std::string model_path;
    } embedder;

    // --- Memory subsystem ---
    struct Memory {
        std::string db_path           = "winefox.db";
        int         recall_top_k      = 3;
        int         distill_keep_turns = 4;
    } memory;

    // --- Misc ---
    std::string system_prompt_path = "llm-finetune/system_prompt.txt";
    bool        no_lora            = false;

    // Load config from a JSON file. Missing keys keep their defaults.
    // Returns false only if the file cannot be opened or is not valid JSON.
    bool load(const std::string& path);

    // Save current config to a JSON file (used to generate defaults).
    bool save(const std::string& path) const;

    // Return a config with all fields set to their defaults.
    static Config defaults();
};

} // namespace config
} // namespace winefox
