#pragma once

// VisionService: thin wrapper around libmtmd (llama.cpp's multimodal library).
//
// Responsibilities:
//   1. Load the mmproj (.gguf) projector alongside an already-loaded text model.
//   2. Load image files into mtmd_bitmap (via stb_image, handled by mtmd-helper).
//   3. Tokenize a prompt text + bitmaps into mtmd_input_chunks (text chunks +
//      image chunks with media marker splicing).
//   4. Eval all chunks into a llama_context (text via llama_decode, images via
//      vision encoder + llama_decode), tracking n_past for the caller.
//
// KV cache interaction: image chunks occupy KV positions via embeddings (not
// token IDs). The vision path in LlmService uses build_tokens_split_ to
// separate text tokens (BPE-stable, cache-reusable) from image chunks (mtmd).
// Dual position pointers (n_cached_tokens_ for text, n_cached_kv_ for KV
// positions including images) enable incremental prefill across text/vision
// alternation without full re-prefill.

#include "../memory/message.h"

#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <string>
#include <vector>

namespace winefox {
namespace llm {

struct VisionOptions {
    bool use_gpu          = false;  // mmproj on CPU (WineFox is CPU-only)
    int  n_threads        = 0;      // 0 = auto
    int  image_min_tokens = -1;     // -1 = read from mmproj metadata
    int  image_max_tokens = -1;
    bool warmup           = true;
    bool flash_attn       = true;   // match text model's flash attention setting
};

class VisionService {
public:
    bool load(const std::string& mmproj_path,
              llama_model* text_model,
              const VisionOptions& opts);
    void close();

    bool ready()           const { return ctx_ != nullptr; }

    // Media marker used in prompt text to indicate where the image should be
    // spliced (default "<__media__>"). The prompt must contain exactly
    // n_bitmaps occurrences of this marker.
    const char* media_marker() const;

    // Load an image file (jpg/png/bmp/...) into a bitmap. Returns nullptr
    // wrapper on failure.
    mtmd_bitmap* load_bitmap_from_file(const std::string& path);

    // Tokenize prompt text (containing media markers) + bitmaps into chunks.
    // Returns the chunks (caller frees via mtmd_input_chunks_free) or nullptr
    // on failure.
    mtmd_input_chunks* tokenize(const std::string& prompt_text,
                                const std::vector<mtmd_bitmap*>& bitmaps,
                                bool add_special = false,
                                bool parse_special = true);

    // Eval all chunks into the llama context, advancing n_past. Handles both
    // text chunks (llama_decode) and image chunks (vision encoder + decode),
    // including non-causal attention and M-RoPE setup automatically.
    // Returns total n_pos consumed (written to new_n_past), or false on error.
    bool eval_chunks(llama_context* lctx,
                     const mtmd_input_chunks* chunks,
                     llama_pos n_past,
                     llama_seq_id seq_id,
                     int32_t n_batch,
                     bool logits_last,
                     llama_pos* new_n_past);

    bool use_mrope() const;

    ~VisionService() { close(); }

private:
    mtmd::context_ptr ctx_;
};

} // namespace llm
} // namespace winefox
