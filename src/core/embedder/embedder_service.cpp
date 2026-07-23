#include "embedder_service.h"

#include "../log/log.h"

#include <algorithm>
#include <cmath>

namespace winefox {
namespace embedder {

namespace {

// L2-normalise a vector in place. Cosine similarity == dot product when both
// sides are unit length, so we normalise once at embedding time.
void l2_normalize(std::vector<float>& v) {
    float sq = 0.0f;
    for (float x : v) sq += x * x;
    float n = std::sqrt(sq);
    if (n > 1e-9f) {
        float inv = 1.0f / n;
        for (float& x : v) x *= inv;
    }
}

} // namespace

bool EmbedderService::init(const std::string& model_path) {
    close();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
        WF_LOG_ERROR("Embedder: failed to load model: %s", model_path.c_str());
        return false;
    }

    // bge-small-zh is a BERT-style encoder: mean-pool the token embeddings and
    // request the embeddings output path (instead of logits).
    llama_context_params cparams = llama_context_default_params();
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cparams.embeddings   = true;
    cparams.n_ctx        = 512;   // embedding inputs are short
    cparams.n_batch      = 512;
    cparams.n_ubatch     = 512;
    cparams.no_perf      = true;
    ctx_ = llama_new_context_with_model(model_, cparams);
    if (!ctx_) {
        WF_LOG_ERROR("Embedder: failed to create context");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    dim_ = llama_n_embd(model_);
    WF_LOG_INFO("Embedder: loaded %s (dim=%d)", model_path.c_str(), dim_);
    return true;
}

void EmbedderService::close() {
    if (ctx_)   { llama_free(ctx_);          ctx_ = nullptr; }
    if (model_) { llama_model_free(model_);  model_ = nullptr; }
    dim_ = 0;
}

EmbedderService::~EmbedderService() { close(); }

std::vector<float> EmbedderService::embed_one_(const std::string& text) {
    if (!ready()) return {};

    // Tokenise with BOS (bge-small expects it). n_tokens_max first probe;
    // grow buffer if needed.
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(64);
    int n = llama_tokenize(vocab, text.c_str(), static_cast<int>(text.size()),
                           tokens.data(), static_cast<int>(tokens.size()),
                           /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.c_str(), static_cast<int>(text.size()),
                           tokens.data(), static_cast<int>(tokens.size()),
                           true, true);
    }
    if (n <= 0) return {};
    tokens.resize(n);

    // Truncate to context window just in case.
    if (n > 511) {
        tokens.resize(511);
        n = 511;
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    llama_batch batch = llama_batch_get_one(tokens.data(), n);
    if (llama_encode(ctx_, batch) != 0) {
        WF_LOG_ERROR("Embedder: llama_encode failed");
        return {};
    }

    float* emb = llama_get_embeddings_seq(ctx_, 0);
    if (!emb) {
        // Fall back to per-token embeddings[0] (some models do not pool).
        emb = llama_get_embeddings_ith(ctx_, 0);
    }
    if (!emb) {
        WF_LOG_ERROR("Embedder: no embedding returned");
        return {};
    }

    std::vector<float> vec(emb, emb + dim_);
    l2_normalize(vec);
    return vec;
}

std::vector<float> EmbedderService::embed(const std::string& text) {
    return embed_one_(text);
}

std::vector<std::vector<float>> EmbedderService::embed_batch(const std::vector<std::string>& texts) {
    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto& t : texts) {
        out.push_back(embed_one_(t));
    }
    return out;
}

} // namespace embedder
} // namespace winefox
