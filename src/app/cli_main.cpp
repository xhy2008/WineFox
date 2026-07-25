// WineFox CLI entry point.
//
// Configuration is loaded from a JSON file (default: ./winefox.json, or
// specified via --config). CLI arguments act as overrides on top of the
// loaded config.
//
// Usage:
//   winefox [--config <path>] [overrides...]
//
// Overrides (override config file values):
//   -m, --model <path>        LLM model (.gguf)
//   -e, --embedder <path>     Embedding model (.gguf)
//   -l, --lora <path>         LoRA adapter (.gguf)
//   -d, --db <path>           SQLite database path
//   -s, --system-prompt <path>  Persona prompt file
//       --n-ctx <n>           Context window size
//       --n-batch <n>         Logical batch size
//       --n-threads <n>       CPU threads (0=auto)
//       --temp <f>            Sampling temperature
//       --top-k <n>           Top-K sampling
//       --top-p <f>           Top-P sampling
//       --repeat-penalty <f>  Repeat penalty
//       --max-tokens <n>      Max generation length
//       --flash-attn <on|off> Enable/disable Flash Attention
//       --kv-dtype <type>     KV cache dtype (f16|q8_0|q4_0)
//       --thinking <on|off>   Enable/disable <think> mode
//       --no-lora             Disable LoRA
//       --mmproj <path>       Multimodal projector (.gguf) for vision input
//
// Commands (typed at the prompt):
//   /quit          Exit
//   /memory        Show memory status
//   /reset         Clear short-term context
//   /distill       Force distillation of the short-term window
//   /image <path>  Attach image(s) to the next message

#include "../core/config/config.h"
#include "../core/embedder/embedder_service.h"
#include "../core/llm/llm_service.h"
#include "../core/log/log.h"
#include "../core/memory/distiller.h"
#include "../core/memory/recall.h"
#include "../core/memory/short_term.h"
#include "../core/pipeline/conversation_manager.h"
#include "../core/storage/migration.h"
#include "../core/storage/sqlite_db.h"
#include "../core/util/strings.h"
#include "../platform/console.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifdef DEBUG
  #define WF_CLI_LOG_INFO(...) WF_LOG_INFO(__VA_ARGS__)
#else
  #define WF_CLI_LOG_INFO(...) ((void)0)
#endif

namespace {

void print_usage() {
    std::fprintf(stderr,
        "WineFox AI - Local offline companion\n"
        "Usage: winefox [--config <path>] [overrides...]\n"
        "\n"
        "Config:\n"
        "      --config <path>         Config file [default: ./winefox.json]\n"
        "\n"
        "Overrides:\n"
        "  -m, --model <path>          LLM model (.gguf)\n"
        "  -e, --embedder <path>       Embedding model (.gguf)\n"
        "  -l, --lora <path>           LoRA adapter (.gguf)\n"
        "  -d, --db <path>             SQLite database\n"
        "  -s, --system-prompt <path>  Persona file\n"
        "      --n-ctx <n>             Context window\n"
        "      --n-batch <n>           Logical batch size\n"
        "      --n-threads <n>         CPU threads (0=auto)\n"
        "      --temp <f>              Temperature\n"
        "      --top-k <n>             Top-K\n"
        "      --top-p <f>             Top-P\n"
        "      --repeat-penalty <f>    Repeat penalty\n"
        "      --max-tokens <n>        Max generation\n"
        "      --flash-attn <on|off>   Flash Attention\n"
        "      --kv-dtype <type>       KV cache dtype (f16|q8_0|q4_0)\n"
        "      --thinking <on|off>     <think> mode\n"
        "      --no-lora               Disable LoRA\n"
        "      --mmproj <path>         Multimodal projector (.gguf)\n"
        "      --gen-config <path>     Write default config to path and exit\n");
}

bool parse_bool(const std::string& s) {
    return s == "on" || s == "true" || s == "1" || s == "yes";
}

// Parse CLI args, applying overrides on top of the loaded Config.
// Returns false on error.
bool parse_args(int argc, char** argv, winefox::config::Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { print_usage(); std::exit(1); }
            return argv[++i];
        };

        // --config was already consumed in main(); skip it here.
        if      (a == "--config")                      next();
        else if (a == "-m" || a == "--model")          cfg.llm.model_path = next();
        else if (a == "-e" || a == "--embedder")       cfg.embedder.model_path = next();
        else if (a == "-l" || a == "--lora")           cfg.llm.lora_path = next();
        else if (a == "-d" || a == "--db")             cfg.memory.db_path = next();
        else if (a == "-s" || a == "--system-prompt")  cfg.system_prompt_path = next();
        else if (a == "--n-ctx")         cfg.llm.n_ctx       = std::atoi(next().c_str());
        else if (a == "--n-batch")       cfg.llm.n_batch     = std::atoi(next().c_str());
        else if (a == "--n-threads")     cfg.llm.n_threads   = std::atoi(next().c_str());
        else if (a == "--temp")          cfg.sampling.temp   = std::strtof(next().c_str(), nullptr);
        else if (a == "--top-k")         cfg.sampling.top_k  = std::atoi(next().c_str());
        else if (a == "--top-p")         cfg.sampling.top_p  = std::strtof(next().c_str(), nullptr);
        else if (a == "--repeat-penalty") cfg.sampling.repeat_penalty = std::strtof(next().c_str(), nullptr);
        else if (a == "--max-tokens")    cfg.sampling.max_tokens = std::atoi(next().c_str());
        else if (a == "--flash-attn")    cfg.llm.flash_attention_enabled = parse_bool(next());
        else if (a == "--kv-dtype")      cfg.llm.kv_cache_dtype = next();
        else if (a == "--thinking")      cfg.llm.enable_thinking = parse_bool(next());
        else if (a == "--no-lora")       cfg.no_lora = true;
        else if (a == "--mmproj")        cfg.llm.mmproj_path = next();
        else if (a == "--gen-config") {
            std::string path = next();
            if (cfg.save(path)) {
                std::printf("Default config written to: %s\n", path.c_str());
            } else {
                std::fprintf(stderr, "Error: cannot write config to: %s\n", path.c_str());
            }
            std::exit(0);
        }
        else if (a == "-h" || a == "--help") { print_usage(); std::exit(0); }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
            print_usage();
            return false;
        }
    }

    if (cfg.llm.model_path.empty()) {
        std::fprintf(stderr, "Error: llm.model_path is required (set in config or via --model)\n");
        return false;
    }
    if (cfg.embedder.model_path.empty()) {
        std::fprintf(stderr, "Error: embedder.model_path is required (set in config or via --embedder)\n");
        return false;
    }
    return true;
}

std::string read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

} // namespace

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
    winefox::platform::init_console_utf8();

    // Initialise llama.cpp backend (silent in Release, verbose in Debug).
    winefox::llm::init_backend();

    // --- Load config file ---
    winefox::config::Config cfg = winefox::config::Config::defaults();

    // Find --config arg first, or default to ./winefox.json
    std::string config_path = "winefox.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    bool config_loaded = cfg.load(config_path);
    if (config_loaded) {
        std::printf("[OK] 配置文件: %s\n", config_path.c_str());
    } else {
        std::printf("[INFO] 未加载配置文件 (%s)，使用默认值\n", config_path.c_str());
    }

    // --- Apply CLI overrides ---
    if (!parse_args(argc, argv, cfg)) return 1;

    // --- Load system prompt ---
    std::string system_prompt = read_file(cfg.system_prompt_path);
    if (system_prompt.empty()) {
        std::fprintf(stderr, "Error: cannot read system prompt: %s\n",
                     cfg.system_prompt_path.c_str());
        return 1;
    }
    system_prompt = winefox::strings::trim(system_prompt);

    std::printf("\nWineFox 启动中...\n\n");

    // --- SQLite ---
    winefox::storage::SqliteDb db;
    if (!db.open(cfg.memory.db_path)) {
        std::fprintf(stderr, "Error: cannot open database: %s\n", cfg.memory.db_path.c_str());
        return 1;
    }
    if (!winefox::storage::Migration::migrate(db)) {
        std::fprintf(stderr, "Error: database migration failed\n");
        return 1;
    }
    std::printf("[OK] 数据库: %s\n", cfg.memory.db_path.c_str());

    // --- Embedder ---
    winefox::embedder::EmbedderService embedder;
    if (!embedder.init(cfg.embedder.model_path)) {
        std::fprintf(stderr, "Error: cannot load embedder: %s\n", cfg.embedder.model_path.c_str());
        return 1;
    }
    std::printf("[OK] Embedding 模型: %s (dim=%d)\n", cfg.embedder.model_path.c_str(), embedder.dim());

    // --- LLM ---
    winefox::llm::LlmService llm;
    winefox::llm::LlmOptions lopts;
    lopts.n_ctx                   = cfg.llm.n_ctx;
    lopts.n_batch                 = cfg.llm.n_batch;
    lopts.n_ubatch                = cfg.llm.n_ubatch;
    lopts.n_threads               = cfg.llm.n_threads;
    lopts.n_threads_batch         = cfg.llm.n_threads_batch;
    lopts.use_mmap                = cfg.llm.use_mmap;
    lopts.use_mlock               = cfg.llm.use_mlock;
    lopts.enable_thinking         = cfg.llm.enable_thinking;
    lopts.flash_attention_enabled = cfg.llm.flash_attention_enabled;
    lopts.kv_cache_dtype          = cfg.llm.kv_cache_dtype;
    if (!llm.load_base(cfg.llm.model_path, lopts)) {
        std::fprintf(stderr, "Error: cannot load LLM: %s\n", cfg.llm.model_path.c_str());
        return 1;
    }
    std::printf("[OK] LLM 模型: %s (n_ctx=%u, flash_attn=%s, kv_dtype=%s)\n",
                cfg.llm.model_path.c_str(), llm.n_ctx(),
                cfg.llm.flash_attention_enabled ? "on" : "off",
                cfg.llm.kv_cache_dtype.c_str());

    // --- Vision (optional) ---
    if (!cfg.llm.mmproj_path.empty()) {
        winefox::llm::VisionOptions vopts;
        vopts.use_gpu = false;
        vopts.n_threads = cfg.llm.n_threads;
        vopts.flash_attn = cfg.llm.flash_attention_enabled;
        // Qwen3VL mmproj defaults produce only ~9 tokens for small images,
        // but the model warns "require at minimum 1024 image tokens to
        // function correctly on grounding tasks". Force a higher min for
        // better visual acuity (color recognition, detail). 64 balances
        // accuracy and encode latency.
        vopts.image_min_tokens = 32;
        vopts.image_max_tokens = -1;
        if (llm.load_vision(cfg.llm.mmproj_path, vopts)) {
            std::printf("[OK] 视觉模型: %s\n", cfg.llm.mmproj_path.c_str());
        } else {
            std::fprintf(stderr, "[WARN] 视觉模型加载失败: %s (视觉功能不可用)\n",
                         cfg.llm.mmproj_path.c_str());
        }
    }

    // --- Memory services ---
    winefox::memory::RecallService recall;
    recall.init(&db, &embedder);

    winefox::memory::ShortTermMemory short_term;

    winefox::memory::Distiller distiller;
    distiller.init(&llm, &recall);

    // --- Conversation manager ---
    winefox::pipeline::ConversationManager::Config cmcfg;
    cmcfg.system_prompt       = system_prompt;
    cmcfg.lora_path           = cfg.no_lora ? "" : cfg.llm.lora_path;
    cmcfg.lora_scale          = cfg.llm.lora_scale;
    cmcfg.recall_top_k        = cfg.memory.recall_top_k;
    cmcfg.distill_keep_turns  = cfg.memory.distill_keep_turns;
    cmcfg.sampling.temp           = cfg.sampling.temp;
    cmcfg.sampling.top_k          = cfg.sampling.top_k;
    cmcfg.sampling.top_p          = cfg.sampling.top_p;
    cmcfg.sampling.repeat_penalty = cfg.sampling.repeat_penalty;
    cmcfg.sampling.penalty_last_n = cfg.sampling.penalty_last_n;
    cmcfg.sampling.max_tokens     = cfg.sampling.max_tokens;
    cmcfg.sampling.seed           = cfg.sampling.seed;

    winefox::pipeline::ConversationManager conv;
    if (!conv.init(&llm, &recall, &short_term, &distiller, cmcfg)) {
        std::fprintf(stderr, "Error: ConversationManager init failed\n");
        return 1;
    }

    std::printf("\n");
    std::printf("========================================\n");
    std::printf("  酒狐已就绪！输入消息开始对话。\n");
    std::printf("  命令: /quit  /memory  /reset  /distill  /image\n");
    std::printf("========================================\n\n");

    // --- REPL ---
    std::string line;
    std::vector<std::string> pending_images;  // images queued by /image
    while (true) {
        // Show a visual hint when images are queued
        if (!pending_images.empty()) {
            std::printf("主人> [%zu 张图片已附加] ", pending_images.size());
        } else {
            std::printf("主人> ");
        }
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;  // EOF

        line = winefox::strings::trim(line);
        if (line.empty()) continue;

        // Commands
        if (line == "/quit" || line == "/exit") break;
        if (line == "/memory") {
            std::printf("%s\n", conv.get_memory_info().c_str());
            continue;
        }
        if (line == "/reset") {
            conv.reset();
            pending_images.clear();
            std::printf("[短期记忆已清空]\n\n");
            continue;
        }
        if (line == "/distill") {
            std::printf("[蒸馏中... (detach LoRA → 基座提取 → embed → commit)]\n");
            std::fflush(stdout);
            long long file_id = conv.force_distill();
            if (file_id < 0) {
                std::printf("[蒸馏失败或短期记忆为空]\n\n");
            } else {
                std::printf("[蒸馏完成, recall_file_id=%lld]\n\n", file_id);
            }
            continue;
        }
        if (line.rfind("/image", 0) == 0) {
            // /image <path>        — attach one image
            // /image <p1> <p2>     — attach multiple images
            // /image clear         — clear pending images
            std::string args = winefox::strings::trim(line.substr(6));
            if (args == "clear" || args.empty()) {
                pending_images.clear();
                std::printf("[已清除待发送图片]\n\n");
            } else {
                // Split by spaces (simple split; paths with spaces need quoting)
                std::istringstream iss(args);
                std::string p;
                while (iss >> p) {
                    pending_images.push_back(p);
                }
                std::printf("[已附加 %zu 张图片，输入消息发送]\n\n", pending_images.size());
            }
            continue;
        }
        if (line[0] == '/') {
            std::printf("未知命令: %s  (可用: /quit /memory /reset /distill /image)\n", line.c_str());
            continue;
        }

        // Chat
        std::printf("酒狐: ");
        std::fflush(stdout);

        std::string reply = conv.chat(line, [](const std::string& piece) -> bool {
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
            return true;  // never stop early
        }, pending_images);

        // Clear pending images after they've been consumed by this turn.
        pending_images.clear();

        auto perf = llm.last_perf();
        std::printf("\n[perf: %d tokens, %.1f tok/s, prefill %.0f ms]\n\n",
                    perf.n_eval, perf.tokens_per_sec(), perf.t_prefill_ms);
    }

    // --- Cleanup ---
    if (conv.session_id() >= 0) {
        recall.close_session(conv.session_id());
    }

    winefox::llm::shutdown_backend();

    std::printf("酒狐: 主人再见～\n");
    return 0;
}
