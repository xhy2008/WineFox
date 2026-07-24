#pragma once

// Shared data structures for the memory subsystem.
//
// Message       - one chat turn (role + content + emotion + timestamp)
// RecallSegment - a retrievable chunk under a recall_file
// RecallFile    - a distilled memory episode (one distillation run)

#include <cstdint>
#include <string>
#include <vector>

namespace winefox {
namespace memory {

struct Message {
    std::string role;       // "user" | "assistant" | "system" | "tool"
    std::string content;
    std::string emotion;    // assistant replies carry one of the 6 emotion tags
    std::string recall;     // user turns: recall text injected as a tool msg after the user msg
    int64_t     timestamp = 0; // Unix epoch seconds

    // assistant replies only: the exact token IDs produced by the model
    // (including the trailing EOG token). Stored so that the next turn can
    // reconstruct the prompt token sequence without re-tokenising the
    // assistant content, which avoids BPE merge-boundary mismatches between
    // model-generated tokens and re-tokenised text — the root cause of KV
    // cache PARTIAL matches.
    std::vector<int32_t> gen_tokens;

    // user turns only: image file paths attached to this message. Vision
    // service loads these into bitmaps; build_tokens_ is bypassed and the
    // prompt is rendered via the jinja template with a media marker so
    // mtmd_tokenize can splice image embeddings into the token sequence.
    std::vector<std::string> image_paths;

    bool is_user()      const { return role == "user"; }
    bool is_assistant() const { return role == "assistant"; }
    bool is_system()    const { return role == "system"; }
    bool is_tool()      const { return role == "tool"; }
};

struct RecallSegment {
    long long              id = 0;
    long long              recall_file_id = 0;
    std::string            content;
    std::vector<float>     embedding;
    int64_t                created_at = 0;
};

struct RecallFile {
    long long              id = 0;
    std::string            title;
    std::string            content;     // full episode summary
    std::string            summary;
    std::vector<float>     embedding;   // file-level embedding (of `content`)
    int64_t                created_at = 0;
    int64_t                last_recalled_at = 0;
    int                    recall_count = 0;

    std::vector<RecallSegment>      segments;
    std::vector<std::string>       resources;   // resource_path list
};

// A single recall hit returned to the conversation manager.
struct Recall {
    long long              file_id = 0;
    std::string            title;
    std::string            content;     // aggregated text used for prompt injection
    float                  score = 0.0f;
    int64_t                created_at = 0;
};

} // namespace memory
} // namespace winefox
