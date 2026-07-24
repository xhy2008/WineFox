#pragma once

// LLM service: wraps llama.cpp to provide LoRA hot-attach/detach, streaming
// chat generation, and a non-streaming complete() for the distiller.
//
// KV cache reuse: instead of prefix-matching token IDs, we track two
// position pointers:
//   n_cached_tokens_ — number of text tokens already cached (matches
//                      build_tokens_ output length, excluding image embeddings)
//   n_cached_kv_     — number of KV positions already cached (includes
//                      image embedding positions, which have no text token)
// In the text path, n_cached_tokens_ == n_cached_kv_ (1 token = 1 KV pos).
// In the vision path, n_cached_kv_ > n_cached_tokens_ (images occupy KV
// positions without corresponding text tokens).
//
// Incremental prefill uses token_start=n_cached_tokens_ (skip count in
// build_tokens_ output) and pos_start=n_cached_kv_ (KV position to continue
// from). This works across text/vision alternation because build_tokens_
// produces identical text token sequences regardless of whether images were
// present — images only shift KV positions, not text token indices.

#include "../memory/message.h"
#include "vision_service.h"

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

    // Load the multimodal projector (mmproj) for vision input. Must be
    // called after load_base. Returns false if mmproj is invalid or does
    // not support vision.
    bool load_vision(const std::string& mmproj_path, const VisionOptions& vopts = {});
    bool vision_ready() const { return vision_.ready(); }

    // Stream a chat completion. `on_token` returns false to stop early.
    // If `out_gen_tokens` is non-null, it receives the model-generated token
    // IDs (including the trailing EOG token) so the caller can store them for
    // BPE-stable KV cache reuse on the next turn.
    // If `image_paths` is non-empty and vision is loaded, the last user
    // message's content gets a media marker appended and the prompt suffix
    // is rendered via mtmd. The text prefix (system + history) is still
    // built via build_tokens_ for KV cache reuse.
    void chat_stream(const std::vector<memory::Message>& messages,
                     const std::function<bool(const std::string&)>& on_token,
                     const SamplingParams& sp = {},
                     std::vector<llama_token>* out_gen_tokens = nullptr,
                     const std::vector<std::string>& image_paths = {});

    // Non-streaming completion (used by the distiller with LoRA detached).
    bool complete(const std::string& prompt, std::string& out,
                  const SamplingParams& sp = {});

    PerfData last_perf() const { return last_perf_; }
    bool     ready() const { return model_ != nullptr && ctx_ != nullptr; }
    uint32_t n_ctx() const { return ctx_ ? llama_n_ctx(ctx_) : 0; }

    // Reset the KV cache pointer so the next chat_stream() does a full
    // prefill instead of incremental. Call after reset(), distillation,
    // or any event that changes the prompt prefix.
    void invalidate_cache() {
        n_cached_tokens_ = 0;
        n_cached_kv_     = 0;
    }

    // Pre-warm the KV cache with a static prefix (e.g. system prompt) so
    // the first chat_stream() only needs to prefill the user input suffix.
    // Returns prefill time in milliseconds, or -1 on failure.
    double warmup_prefill(const std::vector<memory::Message>& messages);

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
    // is a deterministic extension of the previous turn's cached prefix,
    // enabling incremental KV cache reuse across turns.
    //
    // Layout per message:  <|im_start|>{role}\n {content} <|im_end|>[\n]
    //   - assistant content comes from m.gen_tokens (excluding trailing EOG)
    //   - other content is tokenised via tokenize_text_
    //   - assistant footer omits \n (model stops at EOG, no \n in KV cache)
    // Generation prompt (when add_gen_prompt=true):
    //   <|im_start|>assistant\n[<think>\n\n</think>\n\n]
    std::vector<llama_token> build_tokens_(const std::vector<memory::Message>& messages,
                                            bool add_gen_prompt = true);

    // Split result for vision path: text tokens before/after the image
    // insertion point. before + after == build_tokens_(messages, true).
    // The image embeddings are eval'd between before and after.
    struct TokenSplit {
        std::vector<llama_token> before;  // tokens up to (not incl.) image marker
        std::vector<llama_token> after;   // tokens from marker's footer onward
    };

    // Like build_tokens_, but splits the token sequence at the image marker
    // position in the last user message. The caller evals image embeddings
    // between `before` and `after`. This ensures the text token sequence
    // matches build_tokens_ output exactly (enabling cross-turn reuse even
    // when images were present in prior turns).
    TokenSplit build_tokens_split_(const std::vector<memory::Message>& messages,
                                    int image_user_idx);

    // Prefill tokens[token_start..end) at KV positions [pos_start, pos_start+n).
    // When logits_last=true, the final token gets logits (needed for sampling).
    bool prefill_tokens_(const std::vector<llama_token>& tokens,
                         size_t token_start,
                         llama_pos pos_start,
                         bool logits_last = true);

    void generate_loop_(const std::function<bool(const std::string&)>& on_token,
                        const SamplingParams& sp,
                        std::vector<llama_token>* generated_tokens);

    // Image-capable chat path: builds the text prefix (system + history) via
    // build_tokens_ for KV cache reuse, then tokenises the suffix (last user
    // message with media marker + gen prompt) via mtmd. Evals text prefix +
    // image chunks + gen prompt, then runs generate_loop_.
    void chat_stream_with_images_(const std::vector<memory::Message>& messages,
                                   const std::vector<std::string>& image_paths,
                                   const std::function<bool(const std::string&)>& on_token,
                                   const SamplingParams& sp,
                                   std::vector<llama_token>* out_gen_tokens);

    llama_model*         model_  = nullptr;
    llama_context*       ctx_    = nullptr;
    const llama_vocab*   vocab_  = nullptr;
    llama_adapter_lora*  lora_   = nullptr;
    bool                 lora_attached_ = false;
    float                lora_scale_ = 1.0f;

    common_chat_templates_ptr chat_templates_;

    VisionService vision_;

    // Cached special token IDs (looked up once in load_base).
    llama_token tok_im_start_ = LLAMA_TOKEN_NULL;
    llama_token tok_im_end_   = LLAMA_TOKEN_NULL;  // also the EOG token for Qwen3.5

    // KV cache reuse: two position pointers.
    // n_cached_tokens_: text tokens already cached (skip count in
    //   build_tokens_ output). Used as token_start for incremental prefill.
    // n_cached_kv_: KV positions already cached (includes image embedding
    //   positions). Used as pos_start for incremental prefill.
    // In text path: n_cached_tokens_ == n_cached_kv_.
    // In vision path: n_cached_kv_ = n_cached_tokens_ + total image KV positions.
    llama_pos n_cached_tokens_ = 0;
    llama_pos n_cached_kv_     = 0;

    LlmOptions opts_;
    PerfData   last_perf_;
};

} // namespace llm
} // namespace winefox
