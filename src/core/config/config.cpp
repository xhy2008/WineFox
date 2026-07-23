#include "config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

using json = nlohmann::json;

namespace winefox {
namespace config {

// ---------------------------------------------------------------------------
// defaults
// ---------------------------------------------------------------------------

Config Config::defaults() {
    Config c;
    // All fields already have their default values from the struct definitions.
    return c;
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

bool Config::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return false;

    json j;
    try {
        j = json::parse(ifs);
    } catch (const json::parse_error&) {
        return false;
    }

    // --- llm ---
    if (j.contains("llm") && j["llm"].is_object()) {
        const auto& l = j["llm"];
        if (l.contains("model_path"))      llm.model_path      = l["model_path"].get<std::string>();
        if (l.contains("lora_path"))       llm.lora_path       = l["lora_path"].get<std::string>();
        if (l.contains("lora_scale"))      llm.lora_scale      = l["lora_scale"].get<float>();
        if (l.contains("n_ctx"))           llm.n_ctx           = l["n_ctx"].get<int>();
        if (l.contains("n_batch"))         llm.n_batch         = l["n_batch"].get<int>();
        if (l.contains("n_ubatch"))        llm.n_ubatch        = l["n_ubatch"].get<int>();
        if (l.contains("n_threads"))       llm.n_threads       = l["n_threads"].get<int>();
        if (l.contains("n_threads_batch")) llm.n_threads_batch = l["n_threads_batch"].get<int>();
        if (l.contains("use_mmap"))        llm.use_mmap        = l["use_mmap"].get<bool>();
        if (l.contains("use_mlock"))       llm.use_mlock       = l["use_mlock"].get<bool>();
        if (l.contains("enable_thinking")) llm.enable_thinking = l["enable_thinking"].get<bool>();
        if (l.contains("flash_attention_enabled"))
            llm.flash_attention_enabled = l["flash_attention_enabled"].get<bool>();
        if (l.contains("kv_cache_dtype"))  llm.kv_cache_dtype  = l["kv_cache_dtype"].get<std::string>();
    }

    // --- sampling ---
    if (j.contains("sampling") && j["sampling"].is_object()) {
        const auto& s = j["sampling"];
        if (s.contains("temp"))           sampling.temp           = s["temp"].get<float>();
        if (s.contains("top_k"))          sampling.top_k          = s["top_k"].get<int>();
        if (s.contains("top_p"))          sampling.top_p          = s["top_p"].get<float>();
        if (s.contains("repeat_penalty")) sampling.repeat_penalty = s["repeat_penalty"].get<float>();
        if (s.contains("penalty_last_n")) sampling.penalty_last_n = s["penalty_last_n"].get<int>();
        if (s.contains("max_tokens"))     sampling.max_tokens     = s["max_tokens"].get<int>();
        if (s.contains("seed"))           sampling.seed           = s["seed"].get<uint32_t>();
    }

    // --- embedder ---
    if (j.contains("embedder") && j["embedder"].is_object()) {
        const auto& e = j["embedder"];
        if (e.contains("model_path")) embedder.model_path = e["model_path"].get<std::string>();
    }

    // --- memory ---
    if (j.contains("memory") && j["memory"].is_object()) {
        const auto& m = j["memory"];
        if (m.contains("db_path"))            memory.db_path            = m["db_path"].get<std::string>();
        if (m.contains("recall_top_k"))       memory.recall_top_k       = m["recall_top_k"].get<int>();
        if (m.contains("distill_keep_turns")) memory.distill_keep_turns = m["distill_keep_turns"].get<int>();
    }

    // --- misc ---
    if (j.contains("system_prompt_path"))
        system_prompt_path = j["system_prompt_path"].get<std::string>();
    if (j.contains("no_lora"))
        no_lora = j["no_lora"].get<bool>();

    return true;
}

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

bool Config::save(const std::string& path) const {
    json j;

    j["llm"] = {
        {"model_path",              llm.model_path},
        {"lora_path",               llm.lora_path},
        {"lora_scale",              llm.lora_scale},
        {"n_ctx",                   llm.n_ctx},
        {"n_batch",                 llm.n_batch},
        {"n_ubatch",                llm.n_ubatch},
        {"n_threads",               llm.n_threads},
        {"n_threads_batch",         llm.n_threads_batch},
        {"use_mmap",                llm.use_mmap},
        {"use_mlock",               llm.use_mlock},
        {"enable_thinking",         llm.enable_thinking},
        {"flash_attention_enabled", llm.flash_attention_enabled},
        {"kv_cache_dtype",          llm.kv_cache_dtype},
    };

    j["sampling"] = {
        {"temp",           sampling.temp},
        {"top_k",          sampling.top_k},
        {"top_p",          sampling.top_p},
        {"repeat_penalty", sampling.repeat_penalty},
        {"penalty_last_n", sampling.penalty_last_n},
        {"max_tokens",     sampling.max_tokens},
        {"seed",           sampling.seed},
    };

    j["embedder"] = {
        {"model_path", embedder.model_path},
    };

    j["memory"] = {
        {"db_path",            memory.db_path},
        {"recall_top_k",       memory.recall_top_k},
        {"distill_keep_turns", memory.distill_keep_turns},
    };

    j["system_prompt_path"] = system_prompt_path;
    j["no_lora"]            = no_lora;

    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << j.dump(2) << std::endl;
    return true;
}

} // namespace config
} // namespace winefox
