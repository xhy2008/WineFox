#include "conversation_manager.h"

#include "../log/log.h"
#include "../util/time.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <sstream>

namespace winefox {
namespace pipeline {

namespace {

// The six emotion tags defined in the fox persona (system_prompt.txt).
const std::set<std::string>& valid_emotions() {
    static const std::set<std::string> s = {
        "joy", "neutral", "surprise", "sadness", "anger", "fear"
    };
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

bool ConversationManager::init(llm::LlmService* llm,
                                memory::RecallService* recall,
                                memory::ShortTermMemory* short_term,
                                memory::Distiller* distiller,
                                const Config& cfg) {
    if (!llm || !recall || !short_term || !distiller) return false;
    llm_        = llm;
    recall_     = recall;
    short_term_ = short_term;
    distiller_  = distiller;
    cfg_        = cfg;

    // Attach LoRA if a path was provided.
    if (!cfg_.lora_path.empty()) {
        if (!llm_->attach_lora(cfg_.lora_path, cfg_.lora_scale)) {
            WF_LOG_ERROR("ConvMgr: failed to attach LoRA: %s", cfg_.lora_path.c_str());
            return false;
        }
    }

    // Start a new session for raw_message logging.
    session_id_ = recall_->start_session();
    if (session_id_ < 0) {
        WF_LOG_ERROR("ConvMgr: failed to start session");
    }

    WF_LOG_INFO("ConvMgr: initialised (session=%lld, lora=%s)",
                session_id_,
                cfg_.lora_path.empty() ? "off" : "on");

    // --- Pre-warm the KV cache with the system prompt ---
    // This prefills the static system message (persona + profile) immediately
    // at startup, so the first user message only needs to prefill the user
    // input + generation prompt (suffix), not the entire system prompt.
    // Profile is empty at startup (no distillation has happened yet), so the
    // system message is just the persona text — identical to what
    // build_messages_ produces on the first turn.
    auto sys_msg = build_system_message_();
    double warmup_ms = llm_->warmup_prefill({sys_msg});
    if (warmup_ms >= 0) {
        WF_LOG_INFO("ConvMgr: system prompt warmup %.0f ms (%zu tokens)",
                    warmup_ms, 0);  // token count logged inside warmup_prefill
    }

    return true;
}

// ---------------------------------------------------------------------------
// build_system_message_
// ---------------------------------------------------------------------------

memory::Message ConversationManager::build_system_message_() const {
    std::string sys = cfg_.system_prompt;

    auto profile = recall_->get_all_profile();
    if (!profile.empty()) {
        sys += "\n\n[档案]\n";
        for (const auto& kv : profile) {
            sys += kv.first + ": " + kv.second + "\n";
        }
    }

    int64_t now = winefox::time::now_ms() / 1000;
    return {"system", sys, "", "", now};
}

// ---------------------------------------------------------------------------
// build_messages_
// ---------------------------------------------------------------------------

void ConversationManager::build_messages_(std::vector<memory::Message>& messages,
                                           const std::string& user_input,
                                           const std::string& current_recall) {
    // --- System message: persona + [档案] (static, KV-cacheable) ---
    messages.push_back(build_system_message_());

    // --- Short-term window (recent turns) ---
    // Recall is emitted as a role:tool message immediately AFTER the user
    // message that triggered it, NOT merged into user content. This mirrors
    // the training dataset layout and prevents the model from conflating
    // recalled memory with what the user actually said. The tool message
    // is structurally isolated by <|im_start|>tool ... <|im_end|> so the
    // model can clearly distinguish recall context from user input.
    for (const auto& m : short_term_->all()) {
        if (m.is_user()) {
            messages.push_back({"user", m.content, m.emotion, "", m.timestamp});
            if (!m.recall.empty()) {
                messages.push_back({"tool", m.recall, "", "", m.timestamp});
            }
        } else {
            messages.push_back(m);
        }
    }

    // --- Current user turn ---
    int64_t now = winefox::time::now_ms() / 1000;
    messages.push_back({"user", user_input, "", "", now});
    if (!current_recall.empty()) {
        messages.push_back({"tool", current_recall, "", "", now});
    }
}

// ---------------------------------------------------------------------------
// chat
// ---------------------------------------------------------------------------

std::string ConversationManager::chat(const std::string& user_input,
                                       const std::function<bool(const std::string&)>& on_token,
                                       const std::vector<std::string>& image_paths) {
    if (!llm_ || !llm_->ready()) {
        WF_LOG_ERROR("ConvMgr: LLM not ready");
        return {};
    }

    // --- Recall long-term memory for this turn ---
    std::string current_recall;
    if (!user_input.empty() && recall_->embedder_ready()) {
        auto recalls = recall_->recall(user_input, cfg_.recall_top_k);
#ifndef NDEBUG
        // Debug build: dump the raw recall hits so we can inspect what the
        // long-term memory actually surfaced for this query.
        std::fprintf(stderr, "[recall] query=\"%s\" hits=%zu\n",
                     user_input.c_str(), recalls.size());
        for (size_t i = 0; i < recalls.size(); ++i) {
            std::fprintf(stderr, "  [%zu] score=%.4f title=%s | %s\n",
                         i + 1, recalls[i].score,
                         recalls[i].title.c_str(), recalls[i].content.c_str());
        }
        std::fflush(stderr);
#endif
        if (!recalls.empty()) {
            current_recall = "[相关记忆]\n";
            int idx = 1;
            for (const auto& r : recalls) {
                current_recall += std::to_string(idx) + ". " + r.content + "\n";
                ++idx;
            }
        }
    }

    std::vector<memory::Message> messages;
    build_messages_(messages, user_input, current_recall);

    // --- Stream with <think> filtering and emotion-tag buffering ---
    // raw_buffer:  model output EXCLUDING the <think>...</think> block and
    //              the newline immediately after </think>. The chat template
    //              with enable_thinking=false strips <think> blocks from
    //              assistant content, so raw_buffer must match what the
    //              template produces. Saved to short-term memory for KV
    //              cache alignment.
    // display_text: text shown to the user (no <think>, no [emotion] tag).
    std::string raw_buffer;
    std::string display_text;
    std::string emotion = "neutral";
    std::string emo_buffer;
    bool emotion_parsed = false;
    bool in_think = false;
    bool skip_think_newline = false;

    auto stream_cb = [&](const std::string& piece) -> bool {
        // Filter <think>...</think> block from both raw_buffer and display.
        // With special=true in generate_loop_, <think> and </think> arrive
        // as complete text pieces.
        if (piece == "<think>") { in_think = true; return true; }
        if (in_think) {
            if (piece == "</think>") {
                in_think = false;
                skip_think_newline = true;
            }
            return true;
        }
        // Skip the single \n right after </think>.
        if (skip_think_newline) {
            skip_think_newline = false;
            if (piece == "\n") return true;
        }

        raw_buffer += piece;

        // Parse leading [emotion] tag.
        if (!emotion_parsed) {
            emo_buffer += piece;
            size_t first = emo_buffer.find_first_not_of(" \n\t\r");
            if (first == std::string::npos) return true;  // all whitespace

            if (emo_buffer[first] == '[') {
                size_t close = emo_buffer.find(']', first);
                if (close != std::string::npos) {
                    std::string tag = emo_buffer.substr(first + 1, close - first - 1);
                    if (valid_emotions().count(tag)) {
                        emotion = tag;
                    }
                    std::string rest = emo_buffer.substr(close + 1);
                    display_text = rest;
                    if (!rest.empty()) {
                        if (!on_token(rest)) return false;
                    }
                    emotion_parsed = true;
                } else if (emo_buffer.size() - first > 40) {
                    emotion = "neutral";
                    display_text = emo_buffer.substr(first);
                    if (!on_token(display_text)) return false;
                    emotion_parsed = true;
                }
            } else if (emo_buffer.size() - first > 40) {
                emotion = "neutral";
                display_text = emo_buffer.substr(first);
                if (!on_token(display_text)) return false;
                emotion_parsed = true;
            }
        } else {
            display_text += piece;
            if (!on_token(piece)) return false;
        }
        return true;
    };

    // Receive the model-generated token IDs so we can store them in
    // short-term memory. On the next turn, build_tokens_ uses these IDs
    // directly instead of re-tokenising the assistant content, which
    // eliminates BPE merge-boundary mismatches and enables INCREMENTAL
    // KV cache reuse across turns.
    std::vector<llama_token> gen_tokens;
    llm_->chat_stream(messages, stream_cb, cfg_.sampling, &gen_tokens, image_paths);

    if (!emotion_parsed) {
        emotion = "neutral";
    }
    last_emotion_ = emotion;

    auto perf = llm_->last_perf();
    WF_LOG_INFO("ConvMgr: %d tokens, %.1f tok/s, prefill %.0f ms, emotion=[%s]",
                perf.n_eval, perf.tokens_per_sec(), perf.t_prefill_ms,
                emotion.c_str());

    // --- Save to short-term memory ---
    // assistant content = raw_buffer (excludes <think> block, includes
    // [emotion] tag) matching what build_tokens_ produces for assistant
    // history. gen_tokens holds the exact model-generated token IDs
    // (including trailing EOG) for BPE-stable KV cache reuse.
    int64_t now = winefox::time::now_ms() / 1000;
    memory::Message user_msg{"user", user_input, "", current_recall, now};
    user_msg.image_paths = image_paths;
    short_term_->append(user_msg);
    memory::Message asst_msg{"assistant", raw_buffer, emotion, "", now};
    asst_msg.gen_tokens = std::move(gen_tokens);
    short_term_->append(asst_msg);

    // --- Trigger distillation if the window has grown too large ---
    if (short_term_->needs_summary(static_cast<int>(llm_->n_ctx()))) {
        maybe_distill();
    }

    return display_text;
}

// ---------------------------------------------------------------------------
// maybe_distill
// ---------------------------------------------------------------------------

long long ConversationManager::maybe_distill() {
    if (!distiller_ || !short_term_ || !llm_) return -1;

    int n_ctx = static_cast<int>(llm_->n_ctx());
    if (!short_term_->needs_summary(n_ctx)) return -1;

    WF_LOG_INFO("ConvMgr: short-term window overflow, triggering distillation");

    std::vector<memory::Message> drained = short_term_->drain_for_distill(cfg_.distill_keep_turns);
    if (drained.empty()) {
        WF_LOG_INFO("ConvMgr: nothing to distill");
        return -1;
    }

    long long file_id = distiller_->distill(drained, session_id_);
    if (file_id < 0) {
        WF_LOG_ERROR("ConvMgr: distillation failed");
    } else {
        WF_LOG_INFO("ConvMgr: distillation complete (recall_file=%lld)", file_id);
    }
    // Distillation modifies profile/recall DB and drains short-term memory.
    // Re-warm the system prompt (profile may have changed) so the next turn
    // is INCREMENTAL instead of a full re-prefill.
    {
        auto sys_msg = build_system_message_();
        llm_->warmup_prefill({sys_msg});
    }
    return file_id;
}

// ---------------------------------------------------------------------------
// force_distill  (bypasses the 70% threshold; drains ALL messages)
// ---------------------------------------------------------------------------

long long ConversationManager::force_distill() {
    if (!distiller_ || !short_term_ || !llm_) return -1;

    // keep_turns=0 → drain everything.
    std::vector<memory::Message> drained = short_term_->drain_for_distill(0);
    if (drained.empty()) {
        return -1;
    }

    long long file_id = distiller_->distill(drained, session_id_);
    // Re-warm with the (possibly updated) system prompt.
    auto sys_msg = build_system_message_();
    llm_->warmup_prefill({sys_msg});
    return file_id;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

std::string ConversationManager::get_memory_info() const {
    std::ostringstream oss;
    oss << "=== 记忆状态 ===\n";

    // Short-term
    oss << "短期上下文: " << short_term_->all().size() << " 条消息"
        << " (~" << short_term_->approx_tokens() << " tokens)\n";
    if (llm_) {
        oss << "上下文窗口: " << llm_->n_ctx() << " tokens"
            << " (70% 阈值: " << static_cast<int>(llm_->n_ctx() * 0.7) << ")\n";
    }

    // Profile
    auto profile = recall_->get_all_profile();
    oss << "\n[档案] (" << profile.size() << " 条):\n";
    for (const auto& kv : profile) {
        oss << "  " << kv.first << ": " << kv.second << "\n";
    }

    // Long-term recall: files + segments
    int n_files = recall_->count_recall_files();
    int n_segs  = recall_->count_recall_segments();
    oss << "\n[长期记忆] recall_files: " << n_files
        << ", recall_segments: " << n_segs << "\n";

    if (n_files > 0) {
        auto files = recall_->list_recall_files();
        for (const auto& f : files) {
            oss << "  [file " << f.id << "] " << f.title;
            if (!f.summary.empty()) {
                oss << " - " << f.summary;
            }
            oss << " (recalled=" << f.recall_count << ")\n";
        }
        if (n_segs > 0) {
            auto segs = recall_->list_recall_segments();
            oss << "\n  [segments]\n";
            for (const auto& s : segs) {
                oss << "    [seg " << s.id << " (file " << s.recall_file_id << ")]: ";
                // Truncate long segments for readability.
                std::string c = s.content;
                if (c.size() > 80) c = c.substr(0, 80) + "...";
                oss << c << "\n";
            }
        }
    }

    // Session + LoRA status
    oss << "\n会话ID: " << session_id_ << "\n";
    oss << "LoRA: " << (llm_ && llm_->lora_attached() ? "已加载" : "未加载") << "\n";

    return oss.str();
}

void ConversationManager::reset() {
    short_term_->clear();
    // Re-warm the system prompt so the next turn is INCREMENTAL (only user
    // input suffix needs prefill) instead of a full re-prefill.
    auto sys_msg = build_system_message_();
    llm_->warmup_prefill({sys_msg});
    WF_LOG_INFO("ConvMgr: short-term memory cleared, KV cache re-warmed");
}

} // namespace pipeline
} // namespace winefox
