// WineFox memory import tool.
//
// Reads the SFT dataset (JSONL or JSON array of {"messages": [...]}) and
// extracts every tool-message "[相关记忆]" payload, embeds them with the
// embedder model, and commits them into the long-term memory DB as
// recall_files. This pre-seeds the deployment environment with all the
// persona facts covered by the training data, so conversations can recall
// them via the normal [相关记忆] injection path.
//
// Usage:
//   winefox-memory-import [--config <path>] [--db <path>] --embedder <path>
//                         --input <dataset.jsonl> [--input <...> ...]
//                         [--reset] [--dry-run]
//                         [--recall "<query>"] [--top-k <n>]
//
// Notes:
//   - Idempotent: segments whose content already exists in the DB are skipped.
//   - --reset clears all recall_files (and their segments) first.
//   - Embedding model: same bge-small-zh GGUF used by the running service.
//   - --recall runs an end-to-end retrieval check after importing (or against
//     an existing DB when only --recall is given).

#include "../core/config/config.h"
#include "../core/embedder/embedder_service.h"
#include "../core/llm/llm_service.h"
#include "../core/log/log.h"
#include "../core/memory/message.h"
#include "../core/memory/recall.h"
#include "../core/storage/migration.h"
#include "../core/storage/sqlite_db.h"
#include "../core/util/strings.h"
#include "../platform/console.h"

#if defined(_WIN32)
  #include <shellapi.h>
#endif

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// json alias comes from chat.h (included via llm_service.h).

using winefox::memory::RecallFile;
using winefox::memory::RecallSegment;

namespace {

struct Args {
    std::string              config_path = "winefox.json";
    std::string              db_path;
    std::string              embedder_path;
    std::vector<std::string> inputs;
    std::string              recall_query;
    int                      top_k = 3;
    bool                     reset   = false;
    bool                     dry_run = false;
};

void print_usage() {
    std::fprintf(stderr,
        "WineFox memory import tool\n"
        "Reads SFT dataset JSONL/JSON and pre-seeds the long-term memory DB.\n"
        "\n"
        "Usage: winefox-memory-import [options]\n"
        "  --config <path>       Config file [default: ./winefox.json]\n"
        "  --db <path>           SQLite database path\n"
        "  --embedder <path>     Embedding model (.gguf, required)\n"
        "  --input <path>        Dataset JSONL/JSON file (repeatable)\n"
        "  --recall \"<query>\"    Verify retrieval for <query> after import\n"
        "  --top-k <n>           Hits to print with --recall [default: 3]\n"
        "  --reset               Clear existing recall_files before importing\n"
        "  --dry-run             Only count/extract, do not write\n"
        "  -h, --help            Show this help\n");
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { print_usage(); std::exit(1); }
            return argv[++i];
        };
        if      (arg == "--config")   a.config_path = next();
        else if (arg == "--db")       a.db_path     = next();
        else if (arg == "--embedder") a.embedder_path = next();
        else if (arg == "--input")    a.inputs.push_back(next());
        else if (arg == "--recall")   a.recall_query = next();
        else if (arg == "--top-k")    a.top_k = std::atoi(next().c_str());
        else if (arg == "--reset")    a.reset = true;
        else if (arg == "--dry-run")  a.dry_run = true;
        else if (arg == "-h" || arg == "--help") { print_usage(); std::exit(0); }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage();
            return false;
        }
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

// Extract the "[相关记忆]" payload from a tool message. Returns "" when the
// tool carries no usable memory (e.g. "（无相关往事记忆）").
std::string extract_memory(const std::string& tool_content) {
    const std::string kTag = "[相关记忆]";
    size_t pos = tool_content.find(kTag);
    if (pos == std::string::npos) return "";
    pos += kTag.size();
    size_t end = tool_content.find("\n", pos);       // "[当前时间]..." starts a new line
    if (end == std::string::npos) end = tool_content.size();
    std::string mem = winefox::strings::trim(tool_content.substr(pos, end - pos));
    if (mem.empty() ||
        mem.find("无相关往事记忆") != std::string::npos ||
        mem.find("无相关记忆") != std::string::npos) {
        return "";
    }
    return mem;
}

void collect_dialog(const json& d, std::set<std::string>& seen,
                    std::vector<std::string>& out, int& tool_count) {
    if (!d.contains("messages") || !d["messages"].is_array()) return;
    for (const auto& m : d["messages"]) {
        if (!m.is_object()) continue;
        if (m.value("role", "") != "tool") continue;
        ++tool_count;
        std::string mem = extract_memory(m.value("content", ""));
        if (mem.empty()) continue;
        if (seen.insert(mem).second) out.push_back(mem);
    }
}

// Accepts either a JSON array of dialogs, a single dialog object, or a JSONL
// stream with one dialog per line.
void collect_input(const std::string& content, std::set<std::string>& seen,
                   std::vector<std::string>& out, int& tool_count) {
    try {
        json root = json::parse(content);
        if (root.is_array()) {
            for (const auto& d : root) collect_dialog(d, seen, out, tool_count);
            return;
        }
        collect_dialog(root, seen, out, tool_count);
        return;
    } catch (...) {
        // fall through to line-by-line JSONL parsing
    }
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        if (winefox::strings::trim(line).empty()) continue;
        try {
            collect_dialog(json::parse(line), seen, out, tool_count);
        } catch (...) {
            std::fprintf(stderr, "[warn] skipping unparseable line: %.60s...\n", line.c_str());
        }
    }
}

// UTF-8-safe truncation: never cut in the middle of a multi-byte character.
std::string truncate_utf8(const std::string& s, size_t max_bytes) {
    if (s.size() <= max_bytes) return s;
    size_t end = max_bytes;
    // Step back while we're inside a UTF-8 continuation byte (10xxxxxx).
    while (end > 0 && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) --end;
    return s.substr(0, end);
}

// Title for a recall_file: a short prefix of the memory text.
std::string make_title(const std::string& mem) {
    return truncate_utf8(mem, 24) + "…";
}

} // namespace

int main(int argc, char** argv) {
    winefox::platform::init_console_utf8();

#if defined(_WIN32)
    // Windows main() receives argv in the ANSI codepage (GBK on Chinese
    // systems), which corrupts UTF-8 CJK arguments. Rebuild argv as UTF-8
    // from the wide command line so paths/query text round-trip correctly.
    int argc_u8 = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc_u8);
    if (!wargv) {
        std::fprintf(stderr, "Error: CommandLineToArgvW failed\n");
        return 1;
    }
    std::vector<std::string> argv_storage;
    std::vector<char*>       argv_u8;
    argv_storage.reserve(argc_u8);
    argv_u8.reserve(argc_u8);
    for (int i = 0; i < argc_u8; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(len > 1 ? len - 1 : 0, '\0');
        if (len > 1)
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), len, nullptr, nullptr);
        argv_storage.push_back(std::move(s));
    }
    LocalFree(wargv);
    for (auto& s : argv_storage) argv_u8.push_back(s.data());
    argc = argc_u8;
    argv = argv_u8.data();
#endif

    winefox::llm::init_backend();

    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    // Load config for defaults, CLI args override.
    winefox::config::Config cfg = winefox::config::Config::defaults();
    bool loaded = cfg.load(args.config_path);
    if (loaded) {
        std::printf("[OK] 配置文件: %s\n", args.config_path.c_str());
    } else {
        std::printf("[INFO] 未加载配置文件 (%s)，使用命令行参数\n", args.config_path.c_str());
    }
    if (args.embedder_path.empty()) args.embedder_path = cfg.embedder.model_path;
    if (args.db_path.empty())       args.db_path       = cfg.memory.db_path;

    if (args.embedder_path.empty() ||
        (args.inputs.empty() && args.recall_query.empty())) {
        std::fprintf(stderr,
            "Error: --embedder (or a config with embedder.model_path) is required,\n"
            "       plus at least one --input and/or --recall.\n");
        print_usage();
        return 1;
    }

    // --- Collect memories from all inputs ---
    std::set<std::string> seen;
    std::vector<std::string> memories;
    int tool_count = 0;
    for (const auto& in : args.inputs) {
        std::string content = read_file(in);
        if (content.empty()) {
            std::fprintf(stderr, "[warn] cannot read input: %s\n", in.c_str());
            continue;
        }
        collect_input(content, seen, memories, tool_count);
        std::printf("[OK] 输入: %s\n", in.c_str());
    }
    if (!args.inputs.empty())
        std::printf("tool 消息总数: %d，去重后有效记忆: %zu 条\n", tool_count, memories.size());
    if (args.dry_run) {
        std::printf("[dry-run] 未写入数据库。\n");
        winefox::llm::shutdown_backend();
        return 0;
    }

    // --- DB + embedder + recall ---
    winefox::storage::SqliteDb db;
    if (!db.open(args.db_path)) {
        std::fprintf(stderr, "Error: cannot open database: %s\n", args.db_path.c_str());
        return 1;
    }
    if (!winefox::storage::Migration::migrate(db)) {
        std::fprintf(stderr, "Error: database migration failed\n");
        return 1;
    }
    std::printf("[OK] 数据库: %s\n", args.db_path.c_str());

    winefox::embedder::EmbedderService embedder;
    if (!embedder.init(args.embedder_path)) {
        std::fprintf(stderr, "Error: cannot load embedder: %s\n", args.embedder_path.c_str());
        return 1;
    }
    std::printf("[OK] Embedding 模型: %s (dim=%d)\n", args.embedder_path.c_str(), embedder.dim());

    winefox::memory::RecallService recall;
    recall.init(&db, &embedder);

    if (args.reset) {
        db.exec("DELETE FROM recall_files");   // cascades to segments
        std::printf("[reset] 已清空现有记忆 (recall_files)\n");
    }

    // --- Commit (idempotent: skip content already present) ---
    int committed = 0, skipped = 0, failed = 0;
    if (!args.inputs.empty()) {
        for (const auto& mem : memories) {
            bool exists = false;
            db.query("SELECT 1 FROM recall_segments WHERE content = ? LIMIT 1", {mem},
                     [&](sqlite3_stmt* stmt) { exists = true; return false; });
            if (exists) { ++skipped; continue; }

            RecallFile f;
            f.title   = make_title(mem);
            f.content = mem;
            RecallSegment seg;
            seg.content = mem;
            f.segments.push_back(std::move(seg));

            long long id = recall.commit_recall_file(f);
            if (id >= 0) ++committed;
            else         ++failed;
        }

        std::printf("\n=== 导入完成 ===\n");
        std::printf("写入: %d 条，跳过(已存在): %d 条，失败: %d 条\n", committed, skipped, failed);
        std::printf("当前记忆库: %d 个 recall_file, %d 个 recall_segment\n",
                    recall.count_recall_files(), recall.count_recall_segments());
    }

    // --- End-to-end retrieval check ---
    if (!args.recall_query.empty()) {
        auto hits = recall.recall(args.recall_query, args.top_k);
        std::printf("\n=== 召回验证: %s ===\n", args.recall_query.c_str());
        if (hits.empty()) {
            std::printf("未命中任何记忆。\n");
        } else {
            for (size_t i = 0; i < hits.size(); ++i) {
                std::printf("[%zu] score=%.3f | %s\n",
                            i + 1, hits[i].score, hits[i].content.c_str());
            }
        }
    }

    winefox::llm::shutdown_backend();
    return failed ? 1 : 0;
}
