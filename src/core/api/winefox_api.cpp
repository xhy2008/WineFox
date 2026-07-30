// winefox_api.cpp — C ABI implementation for winefox_core.dll
//
// Wraps the entire CLI initialization sequence (from cli_main.cpp) into an
// opaque handle. The world exe calls winefox_init() once, then winefox_chat()
// per turn, then winefox_shutdown() at exit.

#include "winefox_api.h"

#include "../config/config.h"
#include "../embedder/embedder_service.h"
#include "../llm/llm_service.h"
#include "../memory/distiller.h"
#include "../memory/recall.h"
#include "../memory/short_term.h"
#include "../pipeline/conversation_manager.h"
#include "../storage/migration.h"
#include "../storage/sqlite_db.h"
#include "../util/strings.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Internal state bundle
// ---------------------------------------------------------------------------
struct WineFoxCore {
    winefox::config::Config                 cfg;
    std::unique_ptr<winefox::storage::SqliteDb> db;
    std::unique_ptr<winefox::embedder::EmbedderService> embedder;
    std::unique_ptr<winefox::llm::LlmService>          llm;
    std::unique_ptr<winefox::memory::RecallService>    recall;
    std::unique_ptr<winefox::memory::ShortTermMemory>  short_term;
    std::unique_ptr<winefox::memory::Distiller>        distiller;
    std::unique_ptr<winefox::pipeline::ConversationManager> conv;

    std::string system_prompt;
    std::string last_emotion = "neutral";
    std::string last_reply;       // owned buffer for winefox_chat return value
    std::string memory_info_buf;  // owned buffer for winefox_get_memory_info
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string read_file(const char* path) {
    if (!path) return {};
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// API implementation
// ---------------------------------------------------------------------------

WF_API WineFoxCore* winefox_init(const char* config_path,
                                  const char* system_prompt_path) {
    // Initialise the llama.cpp backend and suppress its verbose log output
    // (print_info, create_tensor, sched_reserve, ...) BEFORE any model is
    // loaded. Without this, load_base()/embedder->init() flood the console
    // with INFO-level logs on every model load. This also pairs with the
    // shutdown_backend() call in winefox_shutdown() to balance init/free.
    winefox::llm::init_backend();

    auto* c = new WineFoxCore;

    // --- Config ---
    c->cfg = winefox::config::Config::defaults();
    std::string cpath = config_path ? config_path : "winefox.json";
    if (!c->cfg.load(cpath)) {
        std::fprintf(stderr, "[winefox] config not loaded (%s), using defaults\n",
                     cpath.c_str());
    }

    // --- System prompt ---
    std::string spath = system_prompt_path ? system_prompt_path
                                           : c->cfg.system_prompt_path;
    c->system_prompt = read_file(spath.c_str());
    if (c->system_prompt.empty()) {
        std::fprintf(stderr, "[winefox] cannot read system prompt: %s\n", spath.c_str());
        delete c;
        return nullptr;
    }
    c->system_prompt = winefox::strings::trim(c->system_prompt);

    // --- SQLite ---
    c->db = std::make_unique<winefox::storage::SqliteDb>();
    if (!c->db->open(c->cfg.memory.db_path)) {
        std::fprintf(stderr, "[winefox] cannot open database: %s\n",
                     c->cfg.memory.db_path.c_str());
        delete c;
        return nullptr;
    }
    if (!winefox::storage::Migration::migrate(*c->db)) {
        std::fprintf(stderr, "[winefox] database migration failed\n");
        delete c;
        return nullptr;
    }

    // --- Embedder ---
    c->embedder = std::make_unique<winefox::embedder::EmbedderService>();
    if (!c->embedder->init(c->cfg.embedder.model_path)) {
        std::fprintf(stderr, "[winefox] cannot load embedder: %s\n",
                     c->cfg.embedder.model_path.c_str());
        delete c;
        return nullptr;
    }

    // --- LLM ---
    c->llm = std::make_unique<winefox::llm::LlmService>();
    winefox::llm::LlmOptions lopts;
    lopts.n_ctx                   = c->cfg.llm.n_ctx;
    lopts.n_batch                 = c->cfg.llm.n_batch;
    lopts.n_ubatch                = c->cfg.llm.n_ubatch;
    lopts.n_threads               = c->cfg.llm.n_threads;
    lopts.n_threads_batch         = c->cfg.llm.n_threads_batch;
    lopts.use_mmap                = c->cfg.llm.use_mmap;
    lopts.use_mlock               = c->cfg.llm.use_mlock;
    lopts.enable_thinking         = c->cfg.llm.enable_thinking;
    lopts.flash_attention_enabled = c->cfg.llm.flash_attention_enabled;
    lopts.kv_cache_dtype          = c->cfg.llm.kv_cache_dtype;
    if (!c->llm->load_base(c->cfg.llm.model_path, lopts)) {
        std::fprintf(stderr, "[winefox] cannot load LLM: %s\n",
                     c->cfg.llm.model_path.c_str());
        delete c;
        return nullptr;
    }

    // --- Vision (optional) ---
    if (!c->cfg.llm.mmproj_path.empty()) {
        winefox::llm::VisionOptions vopts;
        vopts.use_gpu = false;
        vopts.n_threads = c->cfg.llm.n_threads;
        vopts.flash_attn = c->cfg.llm.flash_attention_enabled;
        vopts.image_min_tokens = 64;
        vopts.image_max_tokens = -1;
        if (!c->llm->load_vision(c->cfg.llm.mmproj_path, vopts)) {
            std::fprintf(stderr, "[winefox] vision model load failed (vision disabled)\n");
        }
    }

    // --- Memory services ---
    c->recall = std::make_unique<winefox::memory::RecallService>();
    c->recall->init(c->db.get(), c->embedder.get());

    c->short_term = std::make_unique<winefox::memory::ShortTermMemory>();

    c->distiller = std::make_unique<winefox::memory::Distiller>();
    c->distiller->init(c->llm.get(), c->recall.get());

    // --- Conversation manager ---
    winefox::pipeline::ConversationManager::Config cmcfg;
    cmcfg.system_prompt       = c->system_prompt;
    cmcfg.lora_path           = c->cfg.no_lora ? "" : c->cfg.llm.lora_path;
    cmcfg.lora_scale          = c->cfg.llm.lora_scale;
    cmcfg.recall_top_k        = c->cfg.memory.recall_top_k;
    cmcfg.distill_keep_turns  = c->cfg.memory.distill_keep_turns;
    cmcfg.sampling.temp           = c->cfg.sampling.temp;
    cmcfg.sampling.top_k          = c->cfg.sampling.top_k;
    cmcfg.sampling.top_p          = c->cfg.sampling.top_p;
    cmcfg.sampling.repeat_penalty = c->cfg.sampling.repeat_penalty;
    cmcfg.sampling.penalty_last_n = c->cfg.sampling.penalty_last_n;
    cmcfg.sampling.max_tokens     = c->cfg.sampling.max_tokens;
    cmcfg.sampling.seed           = c->cfg.sampling.seed;

    c->conv = std::make_unique<winefox::pipeline::ConversationManager>();
    if (!c->conv->init(c->llm.get(), c->recall.get(), c->short_term.get(),
                       c->distiller.get(), cmcfg)) {
        std::fprintf(stderr, "[winefox] ConversationManager init failed\n");
        delete c;
        return nullptr;
    }

    std::fprintf(stderr, "[winefox] core initialised (n_ctx=%u)\n", c->llm->n_ctx());
    return c;
}

WF_API void winefox_shutdown(WineFoxCore* c) {
    if (!c) return;
    if (c->conv && c->conv->session_id() >= 0) {
        c->recall->close_session(c->conv->session_id());
    }
    // Order: conv → distiller → recall → short_term → llm → embedder → db
    c->conv.reset();
    c->distiller.reset();
    c->recall.reset();
    c->short_term.reset();
    c->llm.reset();
    c->embedder.reset();
    c->db.reset();
    winefox::llm::shutdown_backend();
    delete c;
}

WF_API const char* winefox_chat(WineFoxCore* c,
                                 const char* user_input,
                                 WineFoxTokenCallback on_token,
                                 void* user) {
    if (!c || !c->conv) return nullptr;

    // Bridge C callback → C++ std::function
    c->last_reply = c->conv->chat(user_input ? user_input : "",
        [on_token, user](const std::string& piece) -> bool {
            if (on_token) {
                return on_token(piece.c_str(), user) != 0;
            }
            return true;
        });

    c->last_emotion = c->conv->last_emotion();
    return c->last_reply.c_str();
}

WF_API const char* winefox_last_emotion(WineFoxCore* c) {
    return c ? c->last_emotion.c_str() : "neutral";
}

WF_API void winefox_last_perf(WineFoxCore* c, WineFoxPerf* out) {
    if (!c || !out) return;
    auto perf = c->llm->last_perf();
    out->n_eval         = perf.n_eval;
    out->tokens_per_sec = perf.tokens_per_sec();
    out->t_prefill_ms   = perf.t_prefill_ms;
}

WF_API const char* winefox_get_memory_info(WineFoxCore* c) {
    if (!c || !c->conv) return nullptr;
    c->memory_info_buf = c->conv->get_memory_info();
    return c->memory_info_buf.c_str();
}

WF_API void winefox_reset(WineFoxCore* c) {
    if (c && c->conv) c->conv->reset();
}

WF_API long long winefox_force_distill(WineFoxCore* c) {
    if (!c || !c->conv) return -1;
    return c->conv->force_distill();
}

WF_API long long winefox_session_id(WineFoxCore* c) {
    return (c && c->conv) ? c->conv->session_id() : -1;
}

WF_API void winefox_close_session(WineFoxCore* c, long long session_id) {
    if (c && c->recall && session_id >= 0) {
        c->recall->close_session(session_id);
    }
}

WF_API void winefox_free_string(const char* s) {
    // Strings are stored in WineFoxCore member buffers and freed on
    // shutdown or next call. This is a no-op but kept for ABI compatibility
    // in case we switch to per-call allocation later.
    (void)s;
}
