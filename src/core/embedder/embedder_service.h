#pragma once

// Embedder service: wraps llama.cpp to produce L2-normalised float embeddings
// from text. Default model is bge-small-zh-v1.5 (Q8_0). Used by the long-term
// memory retriever and the distiller.
//
// PLAN.md 3.5 / 9: must expose embed_batch() so the distiller can amortise
// backend context switches.

#include <llama.h>

#include <string>
#include <vector>

namespace winefox {
namespace embedder {

class EmbedderService {
public:
    EmbedderService() = default;
    bool init(const std::string& model_path);
    void close();
    ~EmbedderService();
    EmbedderService(const EmbedderService&) = delete;
    EmbedderService& operator=(const EmbedderService&) = delete;

    // Embed a single piece of text. Returns a L2-normalised vector of length dim().
    // On failure returns an empty vector.
    std::vector<float> embed(const std::string& text);

    // Embed multiple texts. Default implementation loops over embed(); the
    // contract is that batch callers do not need to worry about backend state.
    std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts);

    int  dim() const { return dim_; }
    bool ready() const { return model_ != nullptr && ctx_ != nullptr; }

private:
    std::vector<float> embed_one_(const std::string& text);

    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    int            dim_   = 0;
};

} // namespace embedder
} // namespace winefox
