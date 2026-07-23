#pragma once

// Memory distiller: runs when the user is idle or the short-term window
// overflows. It takes the drained conversation prefix, asks the base model
// (LoRA detached) to extract structured facts + episodes, embeds the results,
// and persists them through RecallService.
//
// PLAN.md 2.3 / TODO 1.6: the distillation prompt includes BOTH user and
// assistant messages (with emotion tags) so that recall can reconstruct not
// just what happened but how the fox reacted.

#include "../llm/llm_service.h"
#include "message.h"
#include "recall.h"

#include <string>
#include <vector>

namespace winefox {
namespace memory {

class Distiller {
public:
    void init(llm::LlmService* llm, RecallService* recall);

    // Distill a batch of messages into long-term memory.
    // `messages` are the drained conversation prefix (user + assistant turns).
    // `session_id` is the current session for raw_message logging.
    // Returns the new recall_files.id on success, -1 on failure.
    long long distill(const std::vector<Message>& messages, long long session_id);

    // Convenience: distill + log the raw messages first.
    long long distill_and_log(const std::vector<Message>& messages, long long session_id);

private:
    // Build the extraction prompt from the conversation.
    std::string build_prompt_(const std::vector<Message>& messages) const;

    // Parse the JSON response and persist to RecallService.
    long long parse_and_commit_(const std::string& json_text,
                                const std::vector<Message>& messages);

    llm::LlmService*  llm_    = nullptr;
    RecallService*    recall_ = nullptr;
};

} // namespace memory
} // namespace winefox
