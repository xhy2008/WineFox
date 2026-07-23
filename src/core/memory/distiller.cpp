#include "distiller.h"

#include "../log/log.h"
#include "../util/time.h"

#include <sstream>

namespace winefox {
namespace memory {

void Distiller::init(llm::LlmService* llm, RecallService* recall) {
    llm_    = llm;
    recall_ = recall;
}

// ---------------------------------------------------------------------------
// build_prompt_
// ---------------------------------------------------------------------------

std::string Distiller::build_prompt_(const std::vector<Message>& messages) const {
    std::string convo;
    convo.reserve(1024);
    for (const auto& m : messages) {
        if (m.is_user()) {
            convo += "用户: ";
            convo += m.content;
            convo += "\n";
        } else if (m.is_assistant()) {
            // Include the emotion tag so the base model can see the fox's
            // emotional reaction alongside the text response.
            convo += "酒狐: ";
            if (!m.emotion.empty()) {
                convo += "[";
                convo += m.emotion;
                convo += "]";
            }
            convo += m.content;
            convo += "\n";
        }
    }

    // Line-based format: one record per line, "type:content".
    // Designed for small (0.8B) Instruct models — much easier to produce
    // correctly than JSON. Greedy decoding (temp=0, top_k=1) keeps the
    // output deterministic and well-structured.
    //
    // Recognised line types:
    //   title:对话主题
    //   summary:对话摘要
    //   profile:key:value     (e.g. profile:name:李四)
    //   segment:事件描述
    std::string prompt =
        "请从以下对话中提取关键信息，每行一条，格式为\"类型:内容\"。\n"
        "只输出提取结果，不要输出其他内容。\n\n"
        "可用类型：\n"
        "  title:对话主题（简短）\n"
        "  summary:对话摘要（一句话）\n"
        "  profile:属性名:属性值（如 profile:name:李四）\n"
        "  segment:事件描述（包含用户说了什么和酒狐的反应）\n\n"
        "注意：profile只记录稳定持久的用户信息（姓名、年龄、住址、偏好等）。\n"
        "segment记录具体事件，每件事一行。\n\n"
        "对话内容：\n" +
        convo +
        "\n请开始提取：";

    return prompt;
}

// ---------------------------------------------------------------------------
// distill
// ---------------------------------------------------------------------------

long long Distiller::distill(const std::vector<Message>& messages, long long session_id) {
    if (!llm_ || !recall_) {
        WF_LOG_ERROR("Distiller: not initialised");
        return -1;
    }
    if (messages.empty()) {
        WF_LOG_INFO("Distiller: empty message list, skipping");
        return -1;
    }

    WF_LOG_INFO("Distiller: distilling %zu messages (session=%lld)",
                messages.size(), session_id);

    // Log raw messages first so the durable record exists even if extraction
    // fails downstream.
    for (const auto& m : messages) {
        recall_->log_raw_message(m, session_id);
    }

    std::string prompt = build_prompt_(messages);

    // Detach LoRA → run extraction on the base model → re-attach.
    bool was_attached = llm_->lora_attached();
    if (was_attached) llm_->detach_lora();

    std::string raw_out;
    llm::SamplingParams sp;
    sp.temp       = 0.0f;    // greedy decoding for deterministic output
    sp.top_k      = 1;       // only pick the top-1 token
    sp.top_p      = 1.0f;
    sp.max_tokens = 1024;
    sp.seed       = 0xD15111;

    bool ok = llm_->complete(prompt, raw_out, sp);

    if (was_attached) llm_->attach_lora("");

    if (!ok) {
        WF_LOG_ERROR("Distiller: LLM complete() failed");
        return -1;
    }

    auto perf = llm_->last_perf();
    WF_LOG_INFO("Distiller: extraction %d tokens, %.1f tok/s, prefill %.0f ms",
                perf.n_eval, perf.tokens_per_sec(), perf.t_prefill_ms);

    return parse_and_commit_(raw_out, messages);
}

long long Distiller::distill_and_log(const std::vector<Message>& messages, long long session_id) {
    return distill(messages, session_id);
}

// ---------------------------------------------------------------------------
// parse_and_commit_
// ---------------------------------------------------------------------------

long long Distiller::parse_and_commit_(const std::string& text,
                                        const std::vector<Message>& messages) {
    // Line-based parser. Each line is "type:content". For profile lines,
    // the format is "profile:key:value" (two colons).
    RecallFile file;
    int profile_count = 0;
    int segment_count = 0;

    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        // Strip trailing \r (in case model outputs CRLF).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Skip empty lines.
        if (line.empty()) continue;

        // Find the first colon to split type from content.
        size_t colon1 = line.find(':');
        if (colon1 == std::string::npos) continue;

        std::string type    = line.substr(0, colon1);
        std::string content = line.substr(colon1 + 1);

        if (type == "title") {
            file.title = content;
        } else if (type == "summary") {
            file.summary = content;
        } else if (type == "profile") {
            // content is "key:value" — split on the first colon again.
            size_t colon2 = content.find(':');
            if (colon2 != std::string::npos) {
                std::string key   = content.substr(0, colon2);
                std::string value = content.substr(colon2 + 1);
                if (!key.empty() && !value.empty()) {
                    recall_->upsert_profile(key, value);
                    ++profile_count;
                }
            }
        } else if (type == "segment") {
            if (!content.empty()) {
                RecallSegment rs;
                rs.content = content;
                file.segments.push_back(std::move(rs));
                ++segment_count;
            }
        }
        // Unrecognised types are silently skipped.
    }

    // Fallbacks for title/summary.
    if (file.title.empty()) {
        for (const auto& m : messages) {
            if (m.is_user()) {
                file.title = m.content.substr(0, std::min<size_t>(m.content.size(), 20));
                break;
            }
        }
        if (file.title.empty()) file.title = "未命名对话";
    }
    if (file.summary.empty()) {
        file.summary = file.title;
    }

    // file.content is what gets embedded at the file level for retrieval.
    file.content = file.summary;

    // Fallback: if no segments were extracted, create one per user message.
    if (file.segments.empty()) {
        for (const auto& m : messages) {
            if (m.is_user()) {
                RecallSegment rs;
                rs.content = "用户说：" + m.content;
                file.segments.push_back(std::move(rs));
            }
        }
    }

    long long file_id = recall_->commit_recall_file(file);
    if (file_id < 0) {
        WF_LOG_ERROR("Distiller: commit_recall_file failed");
        return -1;
    }

    WF_LOG_INFO("Distiller: committed recall_file %lld (%d profiles, %d segments, title='%s')",
                file_id, profile_count, segment_count, file.title.c_str());
    return file_id;
}

} // namespace memory
} // namespace winefox
