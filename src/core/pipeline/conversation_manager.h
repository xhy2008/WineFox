#pragma once

// ConversationManager: the orchestration layer that ties together the LLM
// service, short-term memory, long-term recall, and the distiller.
//
// Each user turn:
//   1. Recall long-term memory for the user input
//   2. Build messages: system(persona+profile) + short-term window
//      (each user msg followed by its recall as a separate system msg)
//      + current user msg + current recall
//   3. Stream the LLM response, parsing the leading [emotion] tag
//   4. Save the user + assistant messages (with recall) to short-term memory
//   5. If the window exceeds 70% of n_ctx, trigger the distiller
//
// KV cache reuse: because recall is injected as separate system messages
// (not appended to the system prompt), the system prompt stays stable across
// turns. LlmService detects the common prefix and only prefills the suffix.
//
// PLAN.md 3.1 / 3.4 / TODO 1.7

#include "../llm/llm_service.h"
#include "../memory/distiller.h"
#include "../memory/message.h"
#include "../memory/recall.h"
#include "../memory/short_term.h"

#include <functional>
#include <string>

namespace winefox {
namespace pipeline {

class ConversationManager {
public:
    struct Config {
        std::string system_prompt;        // persona text (system_prompt.txt)
        std::string lora_path;            // path to winefox-lora, empty = no LoRA
        float       lora_scale       = 1.0f;
        int         recall_top_k      = 3;       // memories to inject
        int         distill_keep_turns = 4;      // turns kept after distillation
        llm::SamplingParams sampling;
    };

    bool init(llm::LlmService* llm,
              memory::RecallService* recall,
              memory::ShortTermMemory* short_term,
              memory::Distiller* distiller,
              const Config& cfg);

    // Process one user turn. Streams the fox's reply (text only, without the
    // [emotion] tag) via on_token. Returns the full reply text.
    // If image_paths is non-empty, the images are attached to this user turn
    // and processed via the vision path (requires mmproj loaded in LlmService).
    std::string chat(const std::string& user_input,
                     const std::function<bool(const std::string&)>& on_token,
                     const std::vector<std::string>& image_paths = {});

    // Run distillation if the short-term window has overflowed.
    // Returns the new recall_files.id, or -1 if nothing was distilled.
    long long maybe_distill();

    // Force distillation of the entire short-term window, ignoring the
    // 70% overflow threshold. Drains ALL messages (keep_turns=0). Used by
    // the /distill CLI command for testing.
    long long force_distill();

    // --- Commands for the CLI ---
    std::string get_memory_info() const;   // /memory
    void        reset();                    // /reset

    long long session_id() const { return session_id_; }

    // Emotion tag parsed from the last chat() reply.
    const std::string& last_emotion() const { return last_emotion_; }

    // Long-term memories recalled for the last chat() turn (empty if none).
    // Used by the frontend to log/debug what was injected as [相关记忆].
    const std::vector<memory::Recall>& last_recalls() const { return last_recalls_; }

private:
    // Build just the system message (persona + profile). Used by both
    // build_messages_ and the init() warmup to avoid code duplication.
    memory::Message build_system_message_() const;

    void build_messages_(std::vector<memory::Message>& messages,
                         const std::string& user_input,
                         const std::string& current_recall);

    llm::LlmService*           llm_        = nullptr;
    memory::RecallService*     recall_     = nullptr;
    memory::ShortTermMemory*   short_term_ = nullptr;
    memory::Distiller*         distiller_  = nullptr;

    Config   cfg_;
    long long session_id_ = -1;
    std::string last_emotion_ = "neutral";
    std::vector<memory::Recall> last_recalls_;   // 上次对话召回的记忆
};

} // namespace pipeline
} // namespace winefox
