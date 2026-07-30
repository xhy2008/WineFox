#include "llm_service.h"

#include "../log/log.h"
#include "../util/time.h"

#include <algorithm>
#include <cstdio>

#include <mtmd-helper.h>

namespace winefox {
namespace llm {

// ---------------------------------------------------------------------------
// Backend lifecycle + log control
// ---------------------------------------------------------------------------

// Swallow all llama.cpp/ggml log output except genuine errors, which are
// still forwarded to stderr. Applied in BOTH Debug and Release builds: the
// verbose INFO-level logs (print_info, create_tensor, sched_reserve, ...)
// are emitted on every load and dominate loading time in Debug due to
// unoptimised code paths + stderr I/O. They are not useful for everyday
// debugging, so we suppress them everywhere.
static void silent_log_cb(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

void init_backend() {
    llama_log_set(silent_log_cb, nullptr);
    // Suppress mtmd/clip verbose logs (clip_model_loader, load_hparams,
    // get_dummy_batch, etc.) the same way we suppress llama.cpp logs.
    mtmd_helper_log_set(silent_log_cb, nullptr);
    llama_backend_init();
    // NUMA: distribute memory across sockets. Safe no-op on single-socket
    // systems; helps multi-socket servers with large models.
    llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
}

void shutdown_backend() {
    llama_backend_free();
}

// ---------------------------------------------------------------------------
// load_base
// ---------------------------------------------------------------------------

bool LlmService::load_base(const std::string& model_path, const LlmOptions& opts) {
    close();
    opts_ = opts;

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap  = opts.use_mmap;
    mparams.use_mlock = opts.use_mlock;
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
        WF_LOG_ERROR("LlmService: failed to load model: %s", model_path.c_str());
        return false;
    }

    vocab_ = llama_model_get_vocab(model_);

    uint32_t n_ctx = static_cast<uint32_t>(opts.n_ctx > 0 ? opts.n_ctx : 4096);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx;
    // n_batch = n_ctx: avoids manual chunking in prefill_tokens_. llama.cpp
    // internally splits by n_ubatch automatically, so one llama_decode call
    // suffices for the entire prompt.
    cparams.n_batch   = opts.n_batch > 0 ? static_cast<uint32_t>(opts.n_batch) : n_ctx;
    // n_ubatch: smaller values improve L2/L3 cache locality during prefill.
    // Default 512 (matches llama.cpp common.h default).
    cparams.n_ubatch  = static_cast<uint32_t>(opts.n_ubatch > 0 ? opts.n_ubatch : 512);
    cparams.n_seq_max = 1;
    cparams.no_perf   = true;                       // we measure perf ourselves

    // Flash Attention
    cparams.flash_attn_type = opts.flash_attention_enabled
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    // KV cache data type
    if (opts.kv_cache_dtype == "q8_0") {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
    } else if (opts.kv_cache_dtype == "q4_0") {
        cparams.type_k = GGML_TYPE_Q4_0;
        cparams.type_v = GGML_TYPE_Q4_0;
    } else {
        // "f16" or anything else → default f16
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
    }

    // Threading: decode is memory-bound (fewer threads = less sync overhead);
    // prefill is compute-bound (more threads = better throughput).
    // 0 = auto (physical cores).
    if (opts.n_threads > 0) {
        cparams.n_threads = opts.n_threads;
    }
    if (opts.n_threads_batch > 0) {
        cparams.n_threads_batch = opts.n_threads_batch;
    }

    ctx_ = llama_new_context_with_model(model_, cparams);
    if (!ctx_) {
        WF_LOG_ERROR("LlmService: failed to create context");
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
        return false;
    }

    // Load the jinja chat template baked into the model metadata so we can
    // pass enable_thinking=false (equivalent to `llama-cli -rea off`).
    chat_templates_ = common_chat_templates_init(model_, /*chat_template_override=*/"");
    if (!chat_templates_) {
        WF_LOG_ERROR("LlmService: failed to init chat templates");
        return false;
    }

    // Cache the special token IDs used by build_tokens_. These are single
    // tokens in the Qwen3.5 vocab; tokenize_text_ with parse_special=true
    // returns a 1-element vector for each.
    auto tok_single = [&](const std::string& s) -> llama_token {
        auto v = tokenize_text_(s);
        return v.size() == 1 ? v[0] : LLAMA_TOKEN_NULL;
    };
    tok_im_start_ = tok_single("<|im_start|>");
    tok_im_end_   = tok_single("<|im_end|>");
    if (tok_im_start_ == LLAMA_TOKEN_NULL || tok_im_end_ == LLAMA_TOKEN_NULL) {
        WF_LOG_ERROR("LlmService: required special tokens <|im_start|>/<|im_end|> not found in vocab");
        return false;
    }

    // --- Warmup: trigger graph build and buffer allocation ---
    // The first llama_decode call builds the compute graph and allocates
    // backend buffers. Running a dummy decode here (with BOS token) avoids
    // paying this cost on the first user message. The KV cache is cleared
    // afterwards so it doesn't interfere with real prefill.
    {
        llama_token bos = llama_vocab_bos(vocab_);
        if (bos != LLAMA_TOKEN_NULL) {
            llama_batch warmup = llama_batch_init(1, 0, 1);
            warmup.token[0]    = bos;
            warmup.pos[0]      = 0;
            warmup.n_seq_id[0] = 1;
            warmup.seq_id[0][0] = 0;
            warmup.logits[0]   = 1;
            warmup.n_tokens    = 1;
            llama_decode(ctx_, warmup);
            llama_batch_free(warmup);
            llama_memory_clear(llama_get_memory(ctx_), true);
            llama_synchronize(ctx_);
        }
    }

    WF_LOG_INFO("LlmService: loaded %s (n_ctx=%u, n_batch=%u, n_ubatch=%u, threads=%d/%d, thinking=%s, flash_attn=%s, kv_dtype=%s)",
                model_path.c_str(),
                llama_n_ctx(ctx_),
                llama_n_batch(ctx_),
                cparams.n_ubatch,
                llama_n_threads(ctx_),
                llama_n_threads_batch(ctx_),
                opts.enable_thinking ? "on" : "off",
                opts.flash_attention_enabled ? "on" : "off",
                opts.kv_cache_dtype.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// LoRA hot-attach / detach  (PLAN.md 11.2.8: measure latency)
// ---------------------------------------------------------------------------

bool LlmService::attach_lora(const std::string& lora_path, float scale) {
    if (!ctx_) return false;

    auto t0 = winefox::time::now_us();

    if (!lora_) {
        lora_ = llama_adapter_lora_init(model_, lora_path.c_str());
        if (!lora_) {
            WF_LOG_ERROR("LlmService: llama_adapter_lora_init failed: %s", lora_path.c_str());
            return false;
        }
    }

    llama_adapter_lora* adapters[1] = { lora_ };
    float scales[1] = { scale };
    if (llama_set_adapters_lora(ctx_, adapters, 1, scales) != 0) {
        WF_LOG_ERROR("LlmService: llama_set_adapters_lora failed");
        return false;
    }

    lora_attached_ = true;

    auto t1 = winefox::time::now_us();
    last_perf_.lora_attach_ms = (t1 - t0) / 1000.0;
    WF_LOG_INFO("LlmService: LoRA attached (scale=%.2f) in %.1f ms",
                scale, last_perf_.lora_attach_ms);
    return true;
}

void LlmService::detach_lora() {
    if (!ctx_ || !lora_attached_) return;

    auto t0 = winefox::time::now_us();

    // Passing nullptr / 0 clears all active LoRA adapters on the context.
    llama_set_adapters_lora(ctx_, nullptr, 0, nullptr);
    lora_attached_ = false;

    auto t1 = winefox::time::now_us();
    last_perf_.lora_detach_ms = (t1 - t0) / 1000.0;
    WF_LOG_INFO("LlmService: LoRA detached in %.1f ms", last_perf_.lora_detach_ms);
}

// ---------------------------------------------------------------------------
// apply_chat_template_
// ---------------------------------------------------------------------------

std::string LlmService::apply_chat_template_(const std::vector<memory::Message>& messages,
                                              bool add_ass) {
    if (!chat_templates_) {
        WF_LOG_ERROR("LlmService: chat templates not initialised");
        return {};
    }

    // Convert our internal Message type into common_chat_msg.
    std::vector<common_chat_msg> chat;
    chat.reserve(messages.size());
    for (const auto& m : messages) {
        common_chat_msg msg;
        msg.role    = m.role;
        msg.content = m.content;
        chat.push_back(std::move(msg));
    }

    common_chat_templates_inputs inputs;
    inputs.messages           = std::move(chat);
    inputs.add_generation_prompt = add_ass;
    inputs.enable_thinking    = opts_.enable_thinking;

    common_chat_params params = common_chat_templates_apply(chat_templates_.get(), inputs);
    return params.prompt;
}

// ---------------------------------------------------------------------------
// tokenize_
// ---------------------------------------------------------------------------

std::vector<llama_token> LlmService::tokenize_(const std::string& text) {
    std::vector<llama_token> tokens(128);
    int n = llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                           tokens.data(), static_cast<int>(tokens.size()),
                           /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                           tokens.data(), static_cast<int>(tokens.size()),
                           true, true);
    }
    if (n <= 0) return {};
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

// ---------------------------------------------------------------------------
// tokenize_text_  (add_special=false, parse_special=true)
// ---------------------------------------------------------------------------

std::vector<llama_token> LlmService::tokenize_text_(const std::string& text) {
    if (text.empty()) return {};
    std::vector<llama_token> tokens(128);
    int n = llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                           tokens.data(), static_cast<int>(tokens.size()),
                           /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                           tokens.data(), static_cast<int>(tokens.size()),
                           false, true);
    }
    if (n <= 0) return {};
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

// ---------------------------------------------------------------------------
// build_tokens_  (BPE-stable prompt construction for KV cache reuse)
// ---------------------------------------------------------------------------

std::vector<llama_token> LlmService::build_tokens_(const std::vector<memory::Message>& messages,
                                                     bool add_gen_prompt) {
    std::vector<llama_token> tokens;

    // The <think> bridge: when enable_thinking=false, the Qwen3.5 generation
    // prompt ends with <think>\n\n</think>\n\n. The model generates content
    // AFTER this bridge. By including the bridge in BOTH the generation prompt
    // AND historical assistant messages, n_cached_kv_ (which covers
    // bridge + generated) aligns exactly with the next turn's build_tokens_
    // output (where the historical assistant is rendered as bridge + gen_tokens
    // + <|im_end|>). Since EOG == <|im_end|>, the positions match perfectly.
    std::vector<llama_token> bridge_toks;
    if (!opts_.enable_thinking) {
        bridge_toks = tokenize_text_("<think>\n\n</think>\n\n");
    }

    for (const auto& m : messages) {
        // --- Header: <|im_start|>{role}\n ---
        std::string header = "<|im_start|>" + m.role + "\n";
        auto hdr_toks = tokenize_text_(header);
        tokens.insert(tokens.end(), hdr_toks.begin(), hdr_toks.end());

        // --- Content ---
        if (m.is_assistant() && !m.gen_tokens.empty()) {
            // Insert the bridge before generated tokens so the layout matches
            // the generation prompt. The model naturally produces content
            // after <think>\n\n</think>\n\n, so this is semantically correct.
            if (!bridge_toks.empty()) {
                tokens.insert(tokens.end(), bridge_toks.begin(), bridge_toks.end());
            }
            // Use the model-generated token IDs directly. The trailing EOG
            // token (== <|im_end|>) is excluded because we append <|im_end|>
            // as part of the footer below.
            if (llama_vocab_is_eog(vocab_, m.gen_tokens.back())) {
                tokens.insert(tokens.end(), m.gen_tokens.begin(), m.gen_tokens.end() - 1);
            } else {
                tokens.insert(tokens.end(), m.gen_tokens.begin(), m.gen_tokens.end());
            }
        } else {
            auto content_toks = tokenize_text_(m.content);
            tokens.insert(tokens.end(), content_toks.begin(), content_toks.end());
        }

        // --- Footer: <|im_end|>[\n] ---
        // For assistant messages with gen_tokens, the model stops at EOG
        // (== <|im_end|>) — there is no trailing \n in the KV cache. We
        // omit the \n so build_tokens_ output matches the cached sequence
        // exactly. Other messages (user/system/tool) are prefilled with
        // their full text including the trailing \n.
        tokens.push_back(tok_im_end_);
        if (!(m.is_assistant() && !m.gen_tokens.empty())) {
            auto nl_toks = tokenize_text_("\n");
            tokens.insert(tokens.end(), nl_toks.begin(), nl_toks.end());
        }
    }

    // --- Generation prompt (optional): <|im_start|>assistant\n [<think>...] ---
    if (add_gen_prompt) {
        auto gen_hdr = tokenize_text_("<|im_start|>assistant\n");
        tokens.insert(tokens.end(), gen_hdr.begin(), gen_hdr.end());
        if (!bridge_toks.empty()) {
            tokens.insert(tokens.end(), bridge_toks.begin(), bridge_toks.end());
        }
    }

    return tokens;
}

// ---------------------------------------------------------------------------
// build_tokens_split_  (split token sequence at image marker for vision path)
// ---------------------------------------------------------------------------

LlmService::TokenSplit LlmService::build_tokens_split_(
    const std::vector<memory::Message>& messages,
    int image_user_idx) {

    TokenSplit result;
    auto& before = result.before;
    auto& after  = result.after;

    std::vector<llama_token> bridge_toks;
    if (!opts_.enable_thinking) {
        bridge_toks = tokenize_text_("<think>\n\n</think>\n\n");
    }

    auto nl_toks = tokenize_text_("\n");

    for (int msg_idx = 0; msg_idx < static_cast<int>(messages.size()); ++msg_idx) {
        const auto& m = messages[msg_idx];

        // Tokens before the image marker go to `before`; after go to `after`.
        // The split point is at the image_user_idx message's footer: header
        // + content → before, footer → after.
        std::vector<llama_token>* target = &before;
        if (msg_idx > image_user_idx) {
            target = &after;
        }

        // --- Header ---
        std::string header = "<|im_start|>" + m.role + "\n";
        auto hdr_toks = tokenize_text_(header);
        target->insert(target->end(), hdr_toks.begin(), hdr_toks.end());

        // --- Content ---
        if (m.is_assistant() && !m.gen_tokens.empty()) {
            if (!bridge_toks.empty()) {
                target->insert(target->end(), bridge_toks.begin(), bridge_toks.end());
            }
            if (llama_vocab_is_eog(vocab_, m.gen_tokens.back())) {
                target->insert(target->end(), m.gen_tokens.begin(), m.gen_tokens.end() - 1);
            } else {
                target->insert(target->end(), m.gen_tokens.begin(), m.gen_tokens.end());
            }
        } else {
            auto content_toks = tokenize_text_(m.content);
            target->insert(target->end(), content_toks.begin(), content_toks.end());
        }

        // --- Footer ---
        // At the split point (image_user_idx), footer goes to `after`.
        // This is where image embeddings will be inserted (between before's
        // last token and after's first token).
        std::vector<llama_token>* footer_target = target;
        if (msg_idx == image_user_idx) {
            footer_target = &after;
        }

        footer_target->push_back(tok_im_end_);
        // Assistant messages with gen_tokens: no \n (matches KV cache).
        // Other messages: include \n.
        if (!(m.is_assistant() && !m.gen_tokens.empty())) {
            footer_target->insert(footer_target->end(), nl_toks.begin(), nl_toks.end());
        }
    }

    // --- Generation prompt → after ---
    auto gen_hdr = tokenize_text_("<|im_start|>assistant\n");
    after.insert(after.end(), gen_hdr.begin(), gen_hdr.end());
    if (!bridge_toks.empty()) {
        after.insert(after.end(), bridge_toks.begin(), bridge_toks.end());
    }

    return result;
}
// ---------------------------------------------------------------------------

bool LlmService::prefill_tokens_(const std::vector<llama_token>& tokens,
                                  size_t token_start,
                                  llama_pos pos_start,
                                  bool logits_last) {
    int n = static_cast<int>(tokens.size() - token_start);
    if (n <= 0) return true;

    const uint32_t n_batch = llama_n_batch(ctx_);

    // Use llama_batch with explicit pos/seq_id to avoid the auto-generation
    // overhead of llama_batch_get_one (which calls memory->seq_pos_max() and
    // builds pos/seq_id arrays on every decode call).
    //
    // Position = pos_start + i: for full prefill (pos_start=0, KV cleared)
    // positions are 0,1,2,...; for incremental prefill positions continue
    // from the existing KV cache prefix.
    for (int i = 0; i < n; i += static_cast<int>(n_batch)) {
        int32_t chunk = std::min(static_cast<int>(n_batch), n - i);

        llama_batch batch = llama_batch_init(chunk, /*embd=*/0, /*n_seq_max=*/1);
        for (int32_t j = 0; j < chunk; ++j) {
            batch.token[j]    = tokens[token_start + i + j];
            batch.pos[j]      = pos_start + i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            // Only request logits for the last token (needed for sampling).
            batch.logits[j] = (logits_last && i + j == n - 1) ? 1 : 0;
        }
        batch.n_tokens = chunk;

        int32_t rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            WF_LOG_ERROR("LlmService: llama_decode failed during prefill (rc=%d, offset=%d)", rc, i);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// generate_loop_
// ---------------------------------------------------------------------------

void LlmService::generate_loop_(const std::function<bool(const std::string&)>& on_token,
                                 const SamplingParams& sp,
                                 std::vector<llama_token>* generated_tokens) {
    // Build the sampler chain: penalties → top_k → top_p → temp → dist.
    llama_sampler_chain_params scparams = llama_sampler_chain_default_params();
    scparams.no_perf = true;
    llama_sampler* smpl = llama_sampler_chain_init(scparams);

    if (sp.penalty_last_n != 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            sp.penalty_last_n, sp.repeat_penalty, 1.0f, 1.0f));
    }
    if (sp.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(sp.top_k));
    }
    if (sp.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(sp.top_p, 1));
    }
    // temp == 0 means greedy decoding. With top_k=1 this is already greedy,
    // so we skip the temp sampler entirely (temp=0 would zero out logits
    // and break argmax). temp < 1 (but > 0) sharpens the distribution.
    if (sp.temp > 0.0f && sp.temp != 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(sp.temp));
    }
    // When temp == 0 (greedy), skip the stochastic dist sampler and use
    // greedy argmax instead. This makes output fully deterministic.
    if (sp.temp == 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(sp.seed));
    }

    char piece_buf[128];
    int32_t n_eval = 0;
    double  t_eval_ms = 0.0;

    while (n_eval < sp.max_tokens) {
        // Sample from the logits of the last token in the context.
        llama_token id = llama_sampler_sample(smpl, ctx_, -1);

        // End-of-generation token → stop.
        if (llama_vocab_is_eog(vocab_, id)) {
            // Include the EOG token in generated_tokens so the cache stays
            // aligned with the next turn's prompt (which includes <|im_end|>).
            if (generated_tokens) {
                generated_tokens->push_back(id);
            }
            break;
        }

        if (generated_tokens) {
            generated_tokens->push_back(id);
        }

        // Convert token → UTF-8 piece.
        // Use special=true so that special tokens like <think>/</think>
        // are rendered as text. This is critical for KV cache reuse:
        // the model may emit an empty <think>\n</think>\n block even
        // with enable_thinking=false. With special=false these tokens
        // produce empty strings, so the cached token sequence diverges
        // from the next turn's re-tokenised prompt. With special=true
        // the caller can filter them from display while preserving the
        // full token sequence in short-term memory.
        int32_t pn = llama_token_to_piece(vocab_, id,
                                          piece_buf, sizeof(piece_buf),
                                          /*lstrip=*/0, /*special=*/true);
        if (pn < 0) {
            // Buffer too small; re-allocate and retry.
            std::vector<char> bigbuf(static_cast<size_t>(-pn));
            pn = llama_token_to_piece(vocab_, id,
                                      bigbuf.data(), static_cast<int32_t>(bigbuf.size()),
                                      0, true);
            if (pn > 0) {
                std::string piece(bigbuf.data(), static_cast<size_t>(pn));
                if (!on_token(piece)) break;
            }
        } else if (pn > 0) {
            std::string piece(piece_buf, static_cast<size_t>(pn));
            if (!on_token(piece)) break;
        }

        // Feed the sampled token back for the next decode step.
        llama_batch batch = llama_batch_get_one(&id, 1);
        auto t0 = winefox::time::now_us();
        int32_t rc = llama_decode(ctx_, batch);
        auto t1 = winefox::time::now_us();
        t_eval_ms += (t1 - t0) / 1000.0;

        if (rc != 0) {
            WF_LOG_ERROR("LlmService: llama_decode failed during generation (rc=%d)", rc);
            break;
        }

        ++n_eval;
    }

    llama_sampler_free(smpl);

    last_perf_.n_eval      = n_eval;
    last_perf_.t_eval_ms   = t_eval_ms;
}

// ---------------------------------------------------------------------------
// warmup_prefill  (pre-warm KV cache with static prefix, e.g. system prompt)
// ---------------------------------------------------------------------------

double LlmService::warmup_prefill(const std::vector<memory::Message>& messages) {
    if (!ready()) {
        WF_LOG_ERROR("LlmService: warmup_prefill called before load_base");
        return -1.0;
    }

    // Build tokens WITHOUT generation prompt — this is just the static prefix
    // (system message). The first chat_stream() will append user messages +
    // generation prompt, and the prefix will match these tokens.
    std::vector<llama_token> new_tokens = build_tokens_(messages, /*add_gen_prompt=*/false);
    if (new_tokens.empty()) {
        WF_LOG_ERROR("LlmService: warmup build_tokens_ returned 0 tokens");
        return -1.0;
    }

    // Full prefill (clear any existing KV cache from the BOS warmup in load_base).
    llama_memory_clear(llama_get_memory(ctx_), true);

    auto t0 = winefox::time::now_us();
    if (!prefill_tokens_(new_tokens, 0, 0)) {
        WF_LOG_ERROR("LlmService: warmup prefill failed");
        return -1.0;
    }
    auto t1 = winefox::time::now_us();
    double prefill_ms = (t1 - t0) / 1000.0;

    // Track cached positions for incremental reuse. No images in warmup,
    // so n_cached_tokens_ == n_cached_kv_.
    n_cached_tokens_ = static_cast<llama_pos>(new_tokens.size());
    n_cached_kv_     = n_cached_tokens_;

    WF_LOG_INFO("LlmService: warmup prefill %zu tokens in %.0f ms",
                new_tokens.size(), prefill_ms);

    return prefill_ms;
}

// ---------------------------------------------------------------------------
// chat_stream  (with KV cache reuse)
// ---------------------------------------------------------------------------

void LlmService::chat_stream(const std::vector<memory::Message>& messages,
                              const std::function<bool(const std::string&)>& on_token,
                              const SamplingParams& sp,
                              std::vector<llama_token>* out_gen_tokens,
                              const std::vector<std::string>& image_paths) {
    if (!ready()) {
        WF_LOG_ERROR("LlmService: chat_stream called before load_base");
        return;
    }

    // --- Vision path: images present and mmproj loaded ---
    if (!image_paths.empty() && vision_.ready()) {
        chat_stream_with_images_(messages, image_paths, on_token, sp, out_gen_tokens);
        return;
    }

    last_perf_.n_eval       = 0;
    last_perf_.t_eval_ms    = 0.0;
    last_perf_.t_prefill_ms = 0.0;

    // Build the prompt token sequence directly (bypassing the jinja template)
    // so that assistant messages use their stored gen_tokens. This guarantees
    // the sequence is a deterministic extension of the previous turn's cached
    // prefix, enabling position-pointer-based KV cache reuse.
    std::vector<llama_token> new_tokens = build_tokens_(messages);
    if (new_tokens.empty()) {
        WF_LOG_ERROR("LlmService: build_tokens_ returned 0 tokens");
        return;
    }

    // --- Incremental prefill using dual position pointers ---
    // n_cached_tokens_: how many text tokens of build_tokens_ output are
    //   already in the KV cache (skip count for token_start).
    // n_cached_kv_: how many KV positions are cached (pos_start). This may
    //   be > n_cached_tokens_ if prior turns had images (image embeddings
    //   occupy KV positions without text tokens).
    //
    // build_tokens_ produces identical text token sequences across turns
    // (assistant messages use stored gen_tokens), so the first
    // n_cached_tokens_ tokens of new_tokens are guaranteed to match the
    // cached KV prefix. We only prefill the suffix.
    const bool can_reuse = (n_cached_tokens_ > 0
                            && n_cached_tokens_ < static_cast<llama_pos>(new_tokens.size()));
    const char* cache_status = can_reuse ? "INCREMENTAL"
                                  : (n_cached_tokens_ > 0 ? "FULL(reset)" : "FULL(first)");

    auto t0 = winefox::time::now_us();

    if (can_reuse) {
        // Remove KV positions beyond the cached prefix (from previous
        // generation), then prefill only the new suffix tokens.
        llama_memory_seq_rm(llama_get_memory(ctx_), 0, n_cached_kv_, -1);
        if (!prefill_tokens_(new_tokens, n_cached_tokens_, n_cached_kv_)) {
            WF_LOG_ERROR("LlmService: incremental prefill failed");
            return;
        }
    } else {
        // Full prefill: clear everything and start fresh.
        llama_memory_clear(llama_get_memory(ctx_), true);
        if (!prefill_tokens_(new_tokens, 0, 0)) {
            WF_LOG_ERROR("LlmService: full prefill failed");
            return;
        }
    }

    auto t1 = winefox::time::now_us();
    last_perf_.t_prefill_ms = (t1 - t0) / 1000.0;

    // --- Generate ---
    std::vector<llama_token> generated;
    generate_loop_(on_token, sp, &generated);

    // --- Update cache pointers for next turn ---
    // n_cached_tokens_: text token count (new_tokens + generated).
    // n_cached_kv_: actual KV position count. Must be computed from the
    //   real prefill/generate positions, NOT reset to n_cached_tokens_.
    //   Prior vision turns leave image embeddings in the KV cache occupying
    //   positions without text tokens, so n_cached_kv_ >= n_cached_tokens_.
    //   The difference (image offset) is preserved across text turns:
    //     new_offset = (K_old + (N - T_old) + G) - (N + G) = K_old - T_old.
    //   Resetting n_cached_kv_ = n_cached_tokens_ would discard this offset
    //   and cause the next turn to prefill at a position behind the KV max,
    //   triggering "X < Y required" M-RoPE position errors.
    if (can_reuse) {
        n_cached_kv_ = n_cached_kv_
                       + static_cast<llama_pos>(new_tokens.size() - n_cached_tokens_)
                       + static_cast<llama_pos>(generated.size());
    } else {
        // Full prefill cleared the KV cache, so the image offset is gone.
        n_cached_kv_ = static_cast<llama_pos>(new_tokens.size() + generated.size());
    }
    n_cached_tokens_ = static_cast<llama_pos>(new_tokens.size() + generated.size());

    if (out_gen_tokens) {
        *out_gen_tokens = std::move(generated);
    }

    WF_LOG_PERF("LlmService: %s prefill %.0f ms (new=%zu cached_toks=%d cached_kv=%d), %d tokens in %.0f ms (%.1f tok/s)",
                cache_status,
                last_perf_.t_prefill_ms,
                new_tokens.size(), static_cast<int>(n_cached_tokens_),
                static_cast<int>(n_cached_kv_),
                last_perf_.n_eval,
                last_perf_.t_eval_ms,
                last_perf_.tokens_per_sec());
}

// ---------------------------------------------------------------------------
// load_vision  (mmproj for image input)
// ---------------------------------------------------------------------------

bool LlmService::load_vision(const std::string& mmproj_path, const VisionOptions& vopts) {
    if (!ready()) {
        WF_LOG_ERROR("LlmService: load_vision called before load_base");
        return false;
    }
    return vision_.load(mmproj_path, model_, vopts);
}

// ---------------------------------------------------------------------------
// chat_stream_with_images_  (vision path: build_tokens_split_ + dual-pointer)
// ---------------------------------------------------------------------------

void LlmService::chat_stream_with_images_(const std::vector<memory::Message>& messages,
                                            const std::vector<std::string>& image_paths,
                                            const std::function<bool(const std::string&)>& on_token,
                                            const SamplingParams& sp,
                                            std::vector<llama_token>* out_gen_tokens) {
    last_perf_.n_eval       = 0;
    last_perf_.t_eval_ms    = 0.0;
    last_perf_.t_prefill_ms = 0.0;

    // --- Find the last user message (where images will be attached) ---
    int last_user_idx = -1;
    for (int i = static_cast<int>(messages.size()) - 1; i >= 0; --i) {
        if (messages[i].is_user()) {
            last_user_idx = i;
            break;
        }
    }
    if (last_user_idx < 0) {
        WF_LOG_ERROR("LlmService: vision path requires a user message");
        return;
    }

    // --- Split text tokens at the image insertion point ---
    // before: all tokens up to (not including) the last user msg's footer.
    // after:  last user msg footer + following msgs + generation prompt.
    // Image embeddings are eval'd between before and after.
    // before + after (concatenated) == build_tokens_(messages, true), so the
    // text token sequence is identical to the text path. This enables
    // cross-turn KV cache reuse even when images were present in prior turns.
    TokenSplit split = build_tokens_split_(messages, last_user_idx);

    // --- Load bitmaps from image files ---
    std::vector<mtmd_bitmap*> bitmaps;
    bitmaps.reserve(image_paths.size());
    for (const auto& path : image_paths) {
        mtmd_bitmap* bm = vision_.load_bitmap_from_file(path);
        if (!bm) {
            WF_LOG_ERROR("LlmService: failed to load image: %s", path.c_str());
            for (auto* b : bitmaps) mtmd_bitmap_free(b);
            return;
        }
        bitmaps.push_back(bm);
    }

    // --- Tokenize images into mtmd chunks ---
    // Use a marker-only prompt so mtmd produces only image chunks (the empty
    // text chunks before/after each marker are no-ops during eval). This
    // keeps text tokenization in build_tokens_split_ (BPE-stable) and image
    // tokenization in mtmd, completely separated.
    const char* marker = vision_.media_marker();
    std::string marker_prompt;
    for (size_t i = 0; i < bitmaps.size(); ++i) {
        marker_prompt += marker;
    }
    mtmd_input_chunks* chunks = vision_.tokenize(marker_prompt, bitmaps,
                                                   /*add_special=*/false,
                                                   /*parse_special=*/true);
    // Free bitmaps — mtmd has consumed pixel data during tokenize.
    for (auto* b : bitmaps) mtmd_bitmap_free(b);
    if (!chunks) {
        WF_LOG_ERROR("LlmService: vision tokenize failed");
        return;
    }

    // Log chunk count and types for vision perf telemetry
    {
        size_t n_chunks = mtmd_input_chunks_size(chunks);
        std::string chunk_info = "chunks=" + std::to_string(n_chunks);
        for (size_t i = 0; i < n_chunks; ++i) {
            auto* c = mtmd_input_chunks_get(chunks, i);
            auto t = mtmd_input_chunk_get_type(c);
            chunk_info += " [";
            chunk_info += (t == 0 ? "TEXT" : (t == 1 ? "IMAGE" : "AUDIO"));
            chunk_info += " n_tok=" + std::to_string(mtmd_input_chunk_get_n_tokens(c));
            chunk_info += " n_pos=" + std::to_string(mtmd_input_chunk_get_n_pos(c));
            chunk_info += "]";
        }
        WF_LOG_PERF("LlmService: VISION mtmd %s mrope=%d marker=%s",
                    chunk_info.c_str(),
                    vision_.use_mrope() ? 1 : 0,
                    marker);
    }

    // --- Incremental prefill of 'before' tokens ---
    // n_cached_tokens_ counts text tokens already in the KV cache (in
    // build_tokens_ output space). n_cached_kv_ counts KV positions already
    // cached (includes image embedding positions from prior vision turns).
    // The first n_cached_tokens_ tokens of split.before are guaranteed to
    // match the cached prefix (same BPE-stable tokenization as build_tokens_).
    const bool can_reuse = (n_cached_tokens_ > 0
                            && n_cached_tokens_ < static_cast<llama_pos>(split.before.size()));
    const char* cache_status = can_reuse ? "INCREMENTAL"
                                  : (n_cached_tokens_ > 0 ? "FULL(reset)" : "FULL(first)");

    auto t0 = winefox::time::now_us();

    if (can_reuse) {
        // Remove KV positions beyond the cached prefix (from previous
        // generation), then prefill only the new suffix of 'before'.
        llama_memory_seq_rm(llama_get_memory(ctx_), 0, n_cached_kv_, -1);
        if (!prefill_tokens_(split.before, n_cached_tokens_, n_cached_kv_,
                             /*logits_last=*/false)) {
            WF_LOG_ERROR("LlmService: vision before incremental prefill failed");
            mtmd_input_chunks_free(chunks);
            return;
        }
    } else {
        // Full prefill: clear everything and start fresh.
        llama_memory_clear(llama_get_memory(ctx_), true);
        if (!split.before.empty()) {
            if (!prefill_tokens_(split.before, 0, 0, /*logits_last=*/false)) {
                WF_LOG_ERROR("LlmService: vision before full prefill failed");
                mtmd_input_chunks_free(chunks);
                return;
            }
        }
    }

    auto t_before = winefox::time::now_us();

    // KV position after prefilling 'before':
    // - Full:       before.size()
    // - Incremental: n_cached_kv_ + (before.size() - n_cached_tokens_)
    //              = before.size() + (n_cached_kv_ - n_cached_tokens_)
    // where (n_cached_kv_ - n_cached_tokens_) = total image KV positions
    // from all prior vision turns (always >= 0).
    llama_pos image_offset = can_reuse ? (n_cached_kv_ - n_cached_tokens_) : 0;
    llama_pos n_past = image_offset + static_cast<llama_pos>(split.before.size());

    WF_LOG_PERF("LlmService: VISION prefill before=%zu after=%zu n_cached_toks=%d n_cached_kv=%d image_offset=%d n_past_before_image=%d",
                split.before.size(), split.after.size(),
                static_cast<int>(n_cached_tokens_),
                static_cast<int>(n_cached_kv_),
                static_cast<int>(image_offset),
                static_cast<int>(n_past));

    // --- Eval image chunks ---
    // mtmd_helper_eval_chunks places image embeddings at positions
    // [n_past, n_past + image_kv_pos). Empty text chunks (from the
    // marker-only prompt) are no-ops. logits_last=false: we need logits on
    // the last token of 'after', not on an image.
    llama_pos new_n_past = 0;
    int32_t n_batch = static_cast<int32_t>(llama_n_batch(ctx_));
    if (!vision_.eval_chunks(ctx_, chunks,
                              /*n_past=*/n_past, /*seq_id=*/0, n_batch,
                              /*logits_last=*/false, &new_n_past)) {
        WF_LOG_ERROR("LlmService: vision eval_chunks failed");
        mtmd_input_chunks_free(chunks);
        return;
    }
    mtmd_input_chunks_free(chunks);

    auto t_image = winefox::time::now_us();

    // new_n_past = n_past + total image KV positions
    n_past = new_n_past;

    // --- Prefill 'after' tokens (logits_last=true for sampling) ---
    if (!split.after.empty()) {
        if (!prefill_tokens_(split.after, 0, n_past, /*logits_last=*/true)) {
            WF_LOG_ERROR("LlmService: vision after prefill failed");
            return;
        }
        n_past += static_cast<llama_pos>(split.after.size());
    }

    auto t1 = winefox::time::now_us();
    last_perf_.t_prefill_ms = (t1 - t0) / 1000.0;

    WF_LOG_PERF("LlmService: VISION timing: before=%.0fms image=%.0fms after=%.0fms total=%.0fms",
                (t_before - t0) / 1000.0,
                (t_image - t_before) / 1000.0,
                (t1 - t_image) / 1000.0,
                last_perf_.t_prefill_ms);

    // --- Generate ---
    std::vector<llama_token> generated;
    generate_loop_(on_token, sp, &generated);

    // --- Update cache pointers ---
    // n_cached_tokens_: text token count (before + after + generated).
    //   On the next turn, build_tokens_ produces the same text token
    //   sequence, so the first n_cached_tokens_ tokens match the cache.
    // n_cached_kv_: KV position count (before + images + after + generated).
    //   On the next turn, incremental prefill continues from this position.
    //   The difference (n_cached_kv_ - n_cached_tokens_) accumulates image
    //   KV positions across all vision turns.
    n_cached_tokens_ = static_cast<llama_pos>(split.before.size()
                                              + split.after.size()
                                              + generated.size());
    n_cached_kv_     = n_past + static_cast<llama_pos>(generated.size());

    if (out_gen_tokens) {
        *out_gen_tokens = std::move(generated);
    }

    WF_LOG_PERF("LlmService: VISION %s prefill %.0f ms (%zu images, before=%zu after=%zu n_past=%d cached_toks=%d cached_kv=%d), %d tokens in %.0f ms (%.1f tok/s)",
                cache_status,
                last_perf_.t_prefill_ms,
                image_paths.size(),
                split.before.size(), split.after.size(),
                static_cast<int>(n_past),
                static_cast<int>(n_cached_tokens_),
                static_cast<int>(n_cached_kv_),
                last_perf_.n_eval,
                last_perf_.t_eval_ms,
                last_perf_.tokens_per_sec());
}

// ---------------------------------------------------------------------------
// complete  (non-streaming, used by the distiller)
// ---------------------------------------------------------------------------

bool LlmService::complete(const std::string& prompt, std::string& out,
                           const SamplingParams& sp) {
    if (!ready()) {
        WF_LOG_ERROR("LlmService: complete called before load_base");
        return false;
    }

    last_perf_.n_eval       = 0;
    last_perf_.t_eval_ms    = 0.0;
    last_perf_.t_prefill_ms = 0.0;

    // Wrap the raw prompt into a single-turn chat so the Instruct model
    // produces well-formed output.
    std::vector<memory::Message> msgs = {
        { "user", prompt, "", "", 0 }
    };
    std::string formatted = apply_chat_template_(msgs, /*add_ass=*/true);
    if (formatted.empty()) {
        WF_LOG_ERROR("LlmService: complete() chat template failed");
        return false;
    }

    std::vector<llama_token> tokens = tokenize_(formatted);
    if (tokens.empty()) {
        WF_LOG_ERROR("LlmService: complete() tokenisation failed");
        return false;
    }

    // Distiller uses a different prompt prefix, so always full prefill.
    llama_memory_clear(llama_get_memory(ctx_), true);

    auto t0 = winefox::time::now_us();
    if (!prefill_tokens_(tokens, 0, 0)) {
        WF_LOG_ERROR("LlmService: complete() prefill failed");
        return false;
    }
    auto t1 = winefox::time::now_us();
    last_perf_.t_prefill_ms = (t1 - t0) / 1000.0;

    out.clear();
    generate_loop_([&out](const std::string& piece) -> bool {
        out += piece;
        return true;
    }, sp, nullptr);

    // Distiller changes the prompt context; invalidate cache so the next
    // chat_stream() does a full re-prefill. (warmup_prefill is called
    // afterwards to re-warm the system prompt.)
    invalidate_cache();

    WF_LOG_INFO("LlmService: complete() %d tokens in %.0f ms (%.1f tok/s)",
                last_perf_.n_eval,
                last_perf_.t_eval_ms,
                last_perf_.tokens_per_sec());
    return true;
}

// ---------------------------------------------------------------------------
// close / destructor
// ---------------------------------------------------------------------------

void LlmService::close() {
    vision_.close();
    lora_attached_ = false;
    if (lora_) {
        llama_adapter_lora_free(lora_);
        lora_ = nullptr;
    }
    chat_templates_.reset();
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
    }
}

LlmService::~LlmService() { close(); }

} // namespace llm
} // namespace winefox
