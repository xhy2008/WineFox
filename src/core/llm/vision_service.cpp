#include "vision_service.h"

#include "../log/log.h"
#include "../util/time.h"

#include <mtmd.h>

namespace winefox {
namespace llm {

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

bool VisionService::load(const std::string& mmproj_path,
                          llama_model* text_model,
                          const VisionOptions& opts) {
    close();

    if (!text_model) {
        WF_LOG_ERROR("VisionService: text_model is null");
        return false;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu          = opts.use_gpu;
    mparams.n_threads        = opts.n_threads;
    mparams.warmup           = opts.warmup;
    mparams.image_min_tokens = opts.image_min_tokens;
    mparams.image_max_tokens = opts.image_max_tokens;
    // Flash attention: match the text model's setting. Works on CPU too
    // (compute optimization, not GPU-specific).
    mparams.flash_attn_type  = opts.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    flash_attn_ = mparams.flash_attn_type;

    mtmd_context* raw = mtmd_init_from_file(mmproj_path.c_str(), text_model, mparams);
    if (!raw) {
        WF_LOG_ERROR("VisionService: mtmd_init_from_file failed: %s", mmproj_path.c_str());
        return false;
    }

    ctx_.reset(raw);

    bool vision_ok = mtmd_support_vision(ctx_.get());
    if (!vision_ok) {
        WF_LOG_ERROR("VisionService: mmproj does not support vision input: %s", mmproj_path.c_str());
        close();
        return false;
    }

    WF_LOG_INFO("VisionService: loaded %s (vision=%s, mrope=%s, non_causal=%s, marker=%s)",
                mmproj_path.c_str(),
                mtmd_support_vision(ctx_.get()) ? "on" : "off",
                mtmd_decode_use_mrope(ctx_.get()) ? "on" : "off",
                mtmd_decode_use_non_causal(ctx_.get(), nullptr) ? "on" : "off",
                mtmd_get_marker(ctx_.get()));
    return true;
}

// ---------------------------------------------------------------------------
// close
// ---------------------------------------------------------------------------

void VisionService::close() {
    ctx_.reset();
}

bool VisionService::use_mrope() const {
    return ctx_ && mtmd_decode_use_mrope(ctx_.get());
}

// ---------------------------------------------------------------------------
// supports_vision / media_marker
// ---------------------------------------------------------------------------

bool VisionService::supports_vision() const {
    return ctx_ && mtmd_support_vision(ctx_.get());
}

const char* VisionService::media_marker() const {
    return ctx_ ? mtmd_get_marker(ctx_.get()) : mtmd_default_marker();
}

// ---------------------------------------------------------------------------
// load_bitmap_from_file
// ---------------------------------------------------------------------------

mtmd_bitmap* VisionService::load_bitmap_from_file(const std::string& path) {
    if (!ctx_) return nullptr;

    // placeholder=false: actually load the image data (not just count tokens)
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_file(
        ctx_.get(), path.c_str(), /*placeholder=*/false);

    if (!wrapper.bitmap) {
        WF_LOG_ERROR("VisionService: failed to load image: %s", path.c_str());
        return nullptr;
    }

    // Note: wrapper.video_ctx is null for image files (only set for video).
    // The bitmap ownership is transferred to the caller.
    return wrapper.bitmap;
}

// ---------------------------------------------------------------------------
// tokenize
// ---------------------------------------------------------------------------

mtmd_input_chunks* VisionService::tokenize(const std::string& prompt_text,
                                            const std::vector<mtmd_bitmap*>& bitmaps,
                                            bool add_special,
                                            bool parse_special) {
    if (!ctx_) return nullptr;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (!chunks) {
        WF_LOG_ERROR("VisionService: mtmd_input_chunks_init failed");
        return nullptr;
    }

    mtmd_input_text text{};
    text.text          = prompt_text.c_str();
    text.text_len      = prompt_text.size();
    text.add_special   = add_special;
    text.parse_special = parse_special;

    // Build the C-style array of bitmap pointers.
    std::vector<const mtmd_bitmap*> bm_ptrs;
    bm_ptrs.reserve(bitmaps.size());
    for (auto* bm : bitmaps) {
        bm_ptrs.push_back(bm);
    }

    int32_t rc = mtmd_tokenize(ctx_.get(), chunks, &text,
                               bm_ptrs.data(), bm_ptrs.size());

    if (rc != 0) {
        WF_LOG_ERROR("VisionService: mtmd_tokenize failed (rc=%d): "
                     "bitmap count=%zu, marker count must match",
                     rc, bitmaps.size());
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }

    return chunks;
}

// ---------------------------------------------------------------------------
// eval_chunks
// ---------------------------------------------------------------------------

bool VisionService::eval_chunks(llama_context* lctx,
                                 const mtmd_input_chunks* chunks,
                                 llama_pos n_past,
                                 llama_seq_id seq_id,
                                 int32_t n_batch,
                                 bool logits_last,
                                 llama_pos* new_n_past) {
    if (!ctx_ || !chunks) return false;

    // Follow the mtmd-cli two-stage pattern for media chunks:
    //   TEXT  : mtmd_helper_eval_chunk_single (direct llama_decode)
    //   media : mtmd_batch_encode (clip) then mtmd_helper_decode_image_chunk
    // The batch-encode path is ~75x faster than eval_chunk_single for images
    // (142ms vs 10.6s for 256 tokens), because it uses clip's optimized
    // batched encoding instead of the per-chunk fallback in eval_chunk_single.
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; ++i) {
        auto* chunk = mtmd_input_chunks_get(chunks, i);
        auto type = mtmd_input_chunk_get_type(chunk);
        const char* type_name =
            (type == MTMD_INPUT_CHUNK_TYPE_TEXT) ? "TEXT" :
            (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) ? "IMAGE" : "AUDIO";
        size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk);
        llama_pos n_past_before = n_past;

        auto t0 = winefox::time::now_us();

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // decode text chunk directly
            llama_pos chunk_new_n_past = n_past;
            int32_t rc = mtmd_helper_eval_chunk_single(ctx_.get(), lctx, chunk,
                                                         n_past, seq_id, n_batch,
                                                         /*logits_last=*/(logits_last && i == n_chunks - 1),
                                                         &chunk_new_n_past);
            if (rc != 0) {
                WF_LOG_ERROR("VisionService: eval_chunk_single failed (rc=%d, chunk=%zu type=%s)",
                             rc, i, type_name);
                return false;
            }
            n_past = chunk_new_n_past;
            auto t1 = winefox::time::now_us();
            WF_LOG_PERF("VisionService: chunk[%zu] %s n_tok=%zu %.0f ms (n_past %d->%d)",
                        i, type_name, n_tok, (t1 - t0) / 1000.0,
                        static_cast<int>(n_past_before), static_cast<int>(n_past));
        } else {
            // media chunk: collect consecutive media chunks into one batch,
            // encode them together, then decode each into the llama context.
            mtmd_batch* mbatch = mtmd_batch_init(ctx_.get());
            if (!mbatch) {
                WF_LOG_ERROR("VisionService: mtmd_batch_init failed");
                return false;
            }

            int32_t rc = mtmd_batch_add_chunk(mbatch, chunk);
            if (rc != 0) {
                WF_LOG_ERROR("VisionService: mtmd_batch_add_chunk failed (rc=%d, chunk=%zu)", rc, i);
                mtmd_batch_free(mbatch);
                return false;
            }
            size_t n_added = 1;
            for (size_t j = i + 1; j < n_chunks; ++j) {
                auto* next_chunk = mtmd_input_chunks_get(chunks, j);
                if (mtmd_input_chunk_get_type(next_chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) break;
                int32_t r = mtmd_batch_add_chunk(mbatch, next_chunk);
                if (r != 0) break;
                n_added++;
            }

            auto t_enc0 = winefox::time::now_us();
            rc = mtmd_batch_encode(mbatch);
            auto t_enc1 = winefox::time::now_us();
            if (rc != 0) {
                WF_LOG_ERROR("VisionService: mtmd_batch_encode failed (rc=%d)", rc);
                mtmd_batch_free(mbatch);
                return false;
            }

            // decode each media chunk in the batch sequentially
            for (size_t k = 0; k < n_added; ++k) {
                auto* mchunk = mtmd_input_chunks_get(chunks, i + k);
                float* embd = mtmd_batch_get_output_embd(mbatch, mchunk);
                if (!embd) {
                    WF_LOG_ERROR("VisionService: mtmd_batch_get_output_embd null (chunk=%zu)", i + k);
                    mtmd_batch_free(mbatch);
                    return false;
                }
                llama_pos chunk_new_n_past = n_past;
                rc = mtmd_helper_decode_image_chunk(ctx_.get(), lctx, mchunk, embd,
                                                     n_past, seq_id, n_batch,
                                                     &chunk_new_n_past,
                                                     /*callback=*/nullptr, /*user_data=*/nullptr);
                if (rc != 0) {
                    WF_LOG_ERROR("VisionService: mtmd_helper_decode_image_chunk failed (rc=%d, chunk=%zu)",
                                 rc, i + k);
                    mtmd_batch_free(mbatch);
                    return false;
                }
                n_past = chunk_new_n_past;
            }

            auto t1 = winefox::time::now_us();
            WF_LOG_PERF("VisionService: chunk[%zu] %s n_tok=%zu total=%.0f ms (encode=%.0f ms, n_past %d->%d, batch=%zu)",
                        i, type_name, n_tok, (t1 - t0) / 1000.0,
                        (t_enc1 - t_enc0) / 1000.0,
                        static_cast<int>(n_past_before), static_cast<int>(n_past), n_added);

            mtmd_batch_free(mbatch);
            i += (n_added - 1); // skip media chunks already processed in this batch
        }
    }

    if (new_n_past) *new_n_past = n_past;
    return true;
}

} // namespace llm
} // namespace winefox
