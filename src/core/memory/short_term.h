#pragma once

// Short-term context: the rolling window of recent chat turns that is fed
// directly to the LLM each turn. Truncation is governed by an approximate
// token budget; when the window exceeds 70% of n_ctx the distiller is expected
// to run and then truncate_keep_recent() is called.

#include "message.h"

#include <vector>

namespace winefox {
namespace memory {

class ShortTermMemory {
public:
    void append(const Message& msg);

    // Return the last `n` messages (oldest-first).
    std::vector<Message> get_recent_messages(int n) const;

    // Return the last `n_turns` complete turns (a turn = user + assistant).
    std::vector<Message> get_recent_turns(int n_turns) const;

    const std::vector<Message>& all() const { return messages_; }

    void clear();

    // Rough token estimate: CJK chars count as ~1.5 tokens, ASCII as ~0.25.
    // Good enough for truncation decisions without a tokeniser round-trip.
    size_t approx_tokens() const;

    // True when approx_tokens() exceeds 70% of n_ctx.
    bool needs_summary(int n_ctx) const;

    // Drop everything except the most recent `k_turns` turns. Called by the
    // distiller after it has persisted the evicted messages.
    void truncate_keep_recent(int k_turns);

    // Remove and return messages that were truncated away (for the distiller).
    // If needs_summary() is true, calling this drains everything except the
    // last `k_turns` turns and returns the drained prefix.
    std::vector<Message> drain_for_distill(int k_turns);

private:
    std::vector<Message> messages_;
};

} // namespace memory
} // namespace winefox
