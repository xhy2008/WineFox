#include "short_term.h"

#include "../util/utf8.h"

namespace winefox {
namespace memory {

void ShortTermMemory::append(const Message& msg) {
    messages_.push_back(msg);
}

std::vector<Message> ShortTermMemory::get_recent_messages(int n) const {
    if (n <= 0 || messages_.empty()) return {};
    if (n >= static_cast<int>(messages_.size())) return messages_;
    return std::vector<Message>(messages_.end() - n, messages_.end());
}

std::vector<Message> ShortTermMemory::get_recent_turns(int n_turns) const {
    if (n_turns <= 0 || messages_.empty()) return {};
    // Walk backwards counting assistant turns (each turn ends with assistant).
    int turns = 0;
    int end_idx = static_cast<int>(messages_.size());
    int start_idx = end_idx;
    for (int i = end_idx - 1; i >= 0; --i) {
        if (messages_[i].is_assistant()) {
            ++turns;
            if (turns >= n_turns) { start_idx = i; break; }
        }
        start_idx = i;
    }
    return std::vector<Message>(messages_.begin() + start_idx, messages_.end());
}

void ShortTermMemory::clear() { messages_.clear(); }

size_t ShortTermMemory::approx_tokens() const {
    size_t toks = 0;
    for (const auto& m : messages_) {
        // Count CJK codepoints as ~1.5 tokens (BPE tends to split them),
        // ASCII bytes as ~0.25 tokens (4 chars per token heuristic).
        size_t cjk = 0, ascii = 0;
        for (size_t i = 0; i < m.content.size();) {
            unsigned char c = static_cast<unsigned char>(m.content[i]);
            int cl = utf8::char_len(c);
            if (cl <= 1) {
                ++ascii;
                ++i;
            } else {
                ++cjk;
                i += cl;
            }
        }
        toks += static_cast<size_t>(cjk * 1.5 + ascii * 0.25);
        // Role tag + formatting overhead per message.
        toks += 4;
    }
    return toks;
}

bool ShortTermMemory::needs_summary(int n_ctx) const {
    return approx_tokens() > static_cast<size_t>(n_ctx * 0.7);
}

void ShortTermMemory::truncate_keep_recent(int k_turns) {
    if (k_turns <= 0) { clear(); return; }
    auto kept = get_recent_turns(k_turns);
    messages_ = std::move(kept);
}

std::vector<Message> ShortTermMemory::drain_for_distill(int k_turns) {
    std::vector<Message> drained;
    if (k_turns <= 0) {
        drained = std::move(messages_);
        messages_.clear();
        return drained;
    }
    // Find the boundary: keep the last k_turns (ending at an assistant reply).
    int turns = 0;
    int start_idx = static_cast<int>(messages_.size());
    for (int i = static_cast<int>(messages_.size()) - 1; i >= 0; --i) {
        if (messages_[i].is_assistant()) {
            ++turns;
            if (turns >= k_turns) { start_idx = i; break; }
        }
        start_idx = i;
    }
    if (start_idx > 0) {
        drained.assign(messages_.begin(), messages_.begin() + start_idx);
        messages_.erase(messages_.begin(), messages_.begin() + start_idx);
    }
    return drained;
}

} // namespace memory
} // namespace winefox
