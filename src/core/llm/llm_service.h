#pragma once

// LLM service: wraps llama.cpp to provide LoRA hot-attach/detach, streaming
// chat generation, and a non-streaming complete() for the distiller.
//
// KV cache reuse: chat_stream() compares the new prompt tokens with the
// previous turn's token sequence. If the new prompt starts with the same
// prefix, only the suffix is prefilled — the KV cache from prior turns is
// reused. This requires recall to be injected as separate system messages
// (not appended to the system prompt), so the system prompt stays stable
// across turns.

#include "../memory/message.h"

#include <chat.h>
#include <functional>
#include <llama.h>
#include <string>
#include <vector>

namespace winefox {
namespace llm {

// Initialise the llama.cpp backend. Must be called once before any LlmService
// or EmbedderService. In Release builds, suppresses all llama.cpp/ggml log
// output. In Debug builds, logs go to stderr (default behaviour).
void init_backend();

// Shut down the backend. Call once at program exit.
void shutdown_backend();

struct LlmOptions {
    int  n_ctx       = 4096;
    int  n_batch     = 0;     // 0 = auto (= n_ctx, avoids manual chunking)
    int  n_ubatch    = 512;   // micro-batch for prefill (smaller = better cache locality)
    int  n_threads   = 0;     // 0 = auto (physical cores) — for decode (memory-bound)
    int  n_threads_batch = 0; // 0 = auto (2x n_threads) — for prefill (compute-bound)
    bool use_mmap    = true;
    bool use_mlock   = false; // force model to stay in RAM (avoids page faults)
    // When false, passes enable_thinking=false to the model's jinja chat
    // template, equivalent to `llama-cli -rea off`. Qwen3.5 then skips the
    // <think> reasoning block entirely.
    bool enable_thinking = false;
    // Flash Attention: LLAMA_FLASH_ATTN_TYPE_ENABLED (true) or DISABLED (false).
    bool flash_attention_enabled = true;
    // KV cache data type: "f16" (default), "q8_0", "q4_0".
    std::string kv_cache_dtype = "f16";
};

struct SamplingParams {
    float    temp            = 0.7f;
    int      top_k           = 40;
    float    top_p           = 0.9f;
    float    repeat_penalty  = 1.10f;
    int      penalty_last_n  = 64;
    int      max_tokens      = 512;
    uint32_t seed            = 0xC0FFEE;
};

struct PerfData {
    int32_t n_eval      = 0;   // generated tokens this call
    double  t_eval_ms   = 0.0; // decode time
    double  t_prefill_ms = 0.0;
    double  lora_attach_ms = 0.0;  // last attach latency
    double  lora_detach_ms = 0.0;  // last detach latency

    double tokens_per_sec() const {
        return t_eval_ms > 0 ? n_eval / (t_eval_ms / 1000.0) : 0.0;
    }
};

class LlmService {
public:
    bool load_base(const std::string& model_path, const LlmOptions& opts);
    bool attach_lora(const std::string& lora_path, float scale = 1.0f);
    void detach_lora();
    bool lora_attached() const { return lora_attached_; }

    // Stream a chat completion. `on_token` returns false to stop early.
    // If `out_gen_tokens` is non-null, it receives the model-generated token
    // IDs (including the trailing EOG token) so the caller can store them for
    // BPE-stable KV cache reuse on the next turn.
    void chat_stream(const std::vector<memory::Message>& messages,
                     const std::function<bool(const std::string&)>& on_token,
                     const SamplingParams& sp = {},
                     std::vector<llama_token>* out_gen_tokens = nullptr);

    // Non-streaming completion (used by the distiller with LoRA detached).
    bool complete(const std::string& prompt, std::string& out,
                  const SamplingParams& sp = {});

    PerfData last_perf() const { return last_perf_; }
    bool     ready() const { return model_ != nullptr && ctx_ != nullptr; }
    uint32_t n_ctx() const { return ctx_ ? llama_n_ctx(ctx_) : 0; }

    // Mark the KV cache as invalid so the next chat_stream() does a full
    // prefill instead of incremental. Call after reset(), distillation,
    // or any event that changes the prompt prefix.
    void invalidate_cache() { cache_valid_ = false; }

    void close();
    ~LlmService();

private:
    std::string apply_chat_template_(const std::vector<memory::Message>& messages,
                                     bool add_ass);
    std::vector<llama_token> tokenize_(const std::string& text);
    // Tokenise text without adding BOS (add_special=false) but with
    // parse_special=true so that <|im_start|> etc. are recognised.
    std::vector<llama_token> tokenize_text_(const std::string& text);

    // Build the full prompt token sequence directly (bypassing the jinja
    // template) so that assistant messages use their stored gen_tokens
    // instead of being re-tokenised. This guarantees that the token sequence
    // matches cached_tokens_ from the previous turn, enabling INCREMENTAL
    // KV cache reuse across turns.
    //
    // Layout per message:  <|im_start|>{role}\n {content} <|im_end|>\n
    //   - assistant content comes from m.gen_tokens (excluding trailing EOG)
    //   - other content is tokenised via tokenize_text_
    // Generation prompt: <|im_start|>assistant\n[<think>\n\n</think>\n\n]
    std::vector<llama_token> build_tokens_(const std::vector<memory::Message>& messages);

    // Prefill tokens[start..end) in n_batch-sized chunks. KV cache must be
    // cleared beforehand if doing a full prefill; for incremental prefill,
    // leave KV cache intact and llama_batch_get_one auto-sets positions.
    bool prefill_tokens_(const std::vector<llama_token>& tokens, size_t start);
    void generate_loop_(const std::function<bool(const std::string&)>& on_token,
                        const SamplingParams& sp,
                        std::vector<llama_token>* generated_tokens);

    llama_model*         model_  = nullptr;
    llama_context*       ctx_    = nullptr;
    const llama_vocab*   vocab_  = nullptr;
    llama_adapter_lora*  lora_   = nullptr;
    bool                 lora_attached_ = false;
    float                lora_scale_ = 1.0f;

    common_chat_templates_ptr chat_templates_;

    // Cached special token IDs (looked up once in load_base).
    llama_token tok_im_start_ = LLAMA_TOKEN_NULL;
    llama_token tok_im_end_   = LLAMA_TOKEN_NULL;  // also the EOG token for Qwen3.5

    // KV cache reuse: full token sequence from the previous chat_stream call
    // (prefill tokens + generated tokens). Used to find a common prefix.
    std::vector<llama_token> cached_tokens_;
    bool                     cache_valid_ = false;

    LlmOptions opts_;
    PerfData   last_perf_;
};

} // namespace llm
} // namespace winefox
