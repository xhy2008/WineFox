#include "llm_service.h"

#include "../log/log.h"
#include "../util/time.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace winefox {
namespace llm {

// ---------------------------------------------------------------------------
// Backend lifecycle + log control
// ---------------------------------------------------------------------------

#ifdef NDEBUG
// Release: silently swallow all llama.cpp/ggml log output.
static void silent_log_cb(ggml_log_level level, const char* text, void* /*user_data*/) {
    // Still forward genuine errors to stderr so failures are visible.
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}
#endif

void init_backend() {
#ifdef NDEBUG
    llama_log_set(silent_log_cb, nullptr);
#endif
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
    lora_scale_    = scale;

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
    // AND historical assistant messages, cached_tokens_ (which ends with
    // bridge + generated) aligns exactly with the next turn's new_tokens
    // (where the historical assistant is rendered as bridge + gen_tokens +
    // <|im_end|>). This eliminates the need to remove the bridge and re-prefill
    // generated tokens every turn.
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

        // --- Footer: <|im_end|>\n ---
        tokens.push_back(tok_im_end_);
        auto nl_toks = tokenize_text_("\n");
        tokens.insert(tokens.end(), nl_toks.begin(), nl_toks.end());
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
// prefill_tokens_
// ---------------------------------------------------------------------------

bool LlmService::prefill_tokens_(const std::vector<llama_token>& tokens, size_t start) {
    int n = static_cast<int>(tokens.size() - start);
    if (n <= 0) return true;

    const uint32_t n_batch = llama_n_batch(ctx_);

    // Use llama_batch with explicit pos/seq_id to avoid the auto-generation
    // overhead of llama_batch_get_one (which calls memory->seq_pos_max() and
    // builds pos/seq_id arrays on every decode call).
    //
    // Position = start + i: for full prefill (start=0, KV cleared) positions
    // are 0,1,2,...; for incremental prefill (start=common) positions are
    // common,common+1,... which matches the existing KV cache prefix.
    for (int i = 0; i < n; i += static_cast<int>(n_batch)) {
        int32_t chunk = std::min(static_cast<int>(n_batch), n - i);

        llama_batch batch = llama_batch_init(chunk, /*embd=*/0, /*n_seq_max=*/1);
        for (int32_t j = 0; j < chunk; ++j) {
            batch.token[j]    = tokens[start + i + j];
            batch.pos[j]      = static_cast<int32_t>(start + i + j);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            // Only request logits for the last token in the final chunk
            // (needed for sampling). This reduces output buffer usage.
            batch.logits[j] = (i + j == n - 1) ? 1 : 0;
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
    // generation prompt, and the common prefix will match these tokens.
    std::vector<llama_token> new_tokens = build_tokens_(messages, /*add_gen_prompt=*/false);
    if (new_tokens.empty()) {
        WF_LOG_ERROR("LlmService: warmup build_tokens_ returned 0 tokens");
        return -1.0;
    }

    // Full prefill (clear any existing KV cache from the BOS warmup in load_base).
    llama_memory_clear(llama_get_memory(ctx_), true);

    auto t0 = winefox::time::now_us();
    if (!prefill_tokens_(new_tokens, 0)) {
        WF_LOG_ERROR("LlmService: warmup prefill failed");
        return -1.0;
    }
    auto t1 = winefox::time::now_us();
    double prefill_ms = (t1 - t0) / 1000.0;

    // Store in cached_tokens_ so the next chat_stream() does INCREMENTAL reuse.
    cached_tokens_ = std::move(new_tokens);
    cache_valid_   = true;

    WF_LOG_INFO("LlmService: warmup prefill %zu tokens in %.0f ms",
                cached_tokens_.size(), prefill_ms);
    return prefill_ms;
}

// ---------------------------------------------------------------------------
// chat_stream  (with KV cache reuse)
// ---------------------------------------------------------------------------

void LlmService::chat_stream(const std::vector<memory::Message>& messages,
                              const std::function<bool(const std::string&)>& on_token,
                              const SamplingParams& sp,
                              std::vector<llama_token>* out_gen_tokens) {
    if (!ready()) {
        WF_LOG_ERROR("LlmService: chat_stream called before load_base");
        return;
    }

    last_perf_.n_eval       = 0;
    last_perf_.t_eval_ms    = 0.0;
    last_perf_.t_prefill_ms = 0.0;

    // Build the prompt token sequence directly (bypassing the jinja template)
    // so that assistant messages use their stored gen_tokens. This guarantees
    // BPE-stable alignment with cached_tokens_ from the previous turn.
    std::vector<llama_token> new_tokens = build_tokens_(messages);
    if (new_tokens.empty()) {
        WF_LOG_ERROR("LlmService: build_tokens_ returned 0 tokens");
        return;
    }

    // --- Find common prefix with cached tokens ---
    size_t common = 0;
    if (cache_valid_) {
        size_t min_len = std::min(new_tokens.size(), cached_tokens_.size());
        while (common < min_len && new_tokens[common] == cached_tokens_[common]) {
            ++common;
        }
    }

    const char* cache_status = cache_valid_
        ? (common == cached_tokens_.size() ? "INCREMENTAL" : "PARTIAL")
        : "FIRST";

    auto t0 = winefox::time::now_us();

    if (cache_valid_ && common > 0) {
        if (common < cached_tokens_.size()) {
            llama_memory_seq_rm(llama_get_memory(ctx_), 0,
                                static_cast<int32_t>(common), -1);
        }
        if (common < new_tokens.size()) {
            if (!prefill_tokens_(new_tokens, common)) {
                WF_LOG_ERROR("LlmService: incremental prefill failed");
                return;
            }
        }
    } else {
        llama_memory_clear(llama_get_memory(ctx_), true);
        if (!prefill_tokens_(new_tokens, 0)) {
            WF_LOG_ERROR("LlmService: full prefill failed");
            return;
        }
    }

    auto t1 = winefox::time::now_us();
    last_perf_.t_prefill_ms = (t1 - t0) / 1000.0;

    // --- Generate ---
    std::vector<llama_token> generated;
    generate_loop_(on_token, sp, &generated);

    // --- Update cache for next turn ---
    // No bridge removal needed! build_tokens_ renders historical assistant
    // messages WITH the <think> bridge prefix, so cached_tokens_ (which ends
    // with bridge + generated) aligns exactly with the next turn's new_tokens
    // (where the historical assistant is bridge + gen_tokens + <|im_end|>).
    // The EOG token (= <|im_end|>) at the end of generated matches the
    // footer <|im_end|> in the next turn's rendering.
    cached_tokens_ = new_tokens;
    cached_tokens_.insert(cached_tokens_.end(), generated.begin(), generated.end());
    cache_valid_ = true;

    // Hand the generated tokens to the caller for BPE-stable storage.
    if (out_gen_tokens) {
        *out_gen_tokens = std::move(generated);
    }

    WF_LOG_INFO("LlmService: %s prefill %.0f ms (new=%zu cached=%zu common=%zu), %d tokens in %.0f ms (%.1f tok/s)",
                cache_status,
                last_perf_.t_prefill_ms,
                new_tokens.size(), cached_tokens_.size(), common,
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
    if (!prefill_tokens_(tokens, 0)) {
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

    // Distiller changes the prompt context; invalidate chat cache so the
    // next chat_stream() does a full re-prefill.
    cache_valid_ = false;

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
