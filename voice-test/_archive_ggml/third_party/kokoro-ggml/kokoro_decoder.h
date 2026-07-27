// kokoro_decoder.h — ggml-based inference for the Kokoro-82M TTS decoder.
//
// This header exposes a minimal C API for loading a Kokoro decoder exported
// as GGUF (see scripts/export_kokoro_decoder_gguf.py) and running a forward
// pass to synthesize audio from encoder outputs.
//
// The encoder (PLBERT + text encoder + duration/F0/N predictors) is kept on
// ONNX Runtime; only the decoder (iSTFTNet) is ported to ggml because:
//   1. The decoder dominates inference time (conv-heavy, ~80% of RTF).
//   2. ggml's hand-tuned AVX2/AVX-512 conv kernels outperform ONNX Runtime's
//      generic Conv1D on CPU for this layer mix.
//   3. The decoder is fully conv+elementwise, no attention — easy to map
//      onto ggml's op set.
//
// Tensor layout convention (ggml):
//   All 3D tensors use ne[0]=time, ne[1]=channels, ne[2]=batch.
//   This matches PyTorch's [B, C, T] in row-major memory order, so a
//   PyTorch tensor [1, C, T] can be memcpy'd into a ggml tensor
//   [T, C, 1] with ne=[T, C, 1].
//
// Tensor naming (matches export_kokoro_decoder_gguf.py, PyTorch state_dict):
//   F0_conv.weight, F0_conv.bias
//   N_conv.weight, N_conv.bias
//   asr_res.0.weight, asr_res.0.bias
//   encode.conv1.weight, encode.conv1.bias, encode.conv2.*,
//   encode.norm1.norm.weight, encode.norm1.norm.bias,
//   encode.norm1.fc.weight, encode.norm1.fc.bias, encode.norm2.*,
//   encode.conv1x1.weight
//   decode.{0..3}.*  (same structure as encode)
//   generator.ups.{0,1}.weight, generator.ups.{0,1}.bias
//   generator.noise_convs.{0,1}.weight, generator.noise_convs.{0,1}.bias
//   generator.resblocks.{0..5}.convs1.{0..2}.{weight,bias}
//   generator.resblocks.{0..5}.convs2.{0..2}.{weight,bias}
//   generator.resblocks.{0..5}.adain1.{0..2}.fc.{weight,bias}
//   generator.resblocks.{0..5}.adain1.{0..2}.norm.{weight,bias}
//   generator.resblocks.{0..5}.adain2.{0..2}.*
//   generator.resblocks.{0..5}.alpha1.{0..2}, alpha2.{0..2}
//   generator.noise_res.{0,1}.*  (same structure as resblocks)
//   generator.conv_post.weight, generator.conv_post.bias
//   generator.m_source.l_linear.weight, generator.m_source.l_linear.bias
//   generator.stft.window,
//   generator.stft.weight_forward_real, generator.stft.weight_forward_imag,
//   generator.stft.weight_backward_real, generator.stft.weight_backward_imag

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct kokoro_decoder;
typedef struct kokoro_decoder kokoro_decoder;

struct kokoro_decoder_params {
    int n_threads;  // number of CPU threads for graph compute (default: 4)
};

// Load a Kokoro decoder from a GGUF file.
// Returns NULL on failure. Caller must free with kokoro_decoder_free().
kokoro_decoder * kokoro_decoder_init_from_file(
    const char * path_model,
    struct kokoro_decoder_params params);

// Release all resources.
void kokoro_decoder_free(kokoro_decoder * dec);

#ifdef __cplusplus
}
#endif

// Verbose per-stage timing toggle (default: off). When on, every forward pass
// prints sine_precompute / graph_build / graph_compute timings to stderr.
extern bool kokoro_decoder_profile;

// Run a forward pass.
//
// Inputs (PyTorch row-major, batch=1):
//   asr       : [T_asr, 512]   — encoder's `asr` output, transposed to (T, C).
//                                 T_asr is the frame rate AFTER the F0_conv
//                                 stride-2 (i.e. half of T_f0).
//   F0_pred   : [T_f0]         — encoder's F0 prediction, full frame rate.
//                                 F0_conv (stride 2) halves this to T_asr.
//   N_pred    : [T_f0]         — encoder's N prediction, same length as F0_pred.
//   style_dec : [128]          — ref_s[:, :128]
//
// Returns audio samples (24 kHz, float32) in a std::vector.
// On failure returns an empty vector.
//
// NOTE: This function has C++ linkage because it returns std::vector<float>.
std::vector<float> kokoro_decoder_forward(
    kokoro_decoder * dec,
    const float * asr,       int T_asr,
    const float * F0_pred,   int T_f0,
    const float * N_pred,
    const float * style_dec);

// Diagnostic: compute only the SineGen + forward STFT (the "har" tensor)
// from F0_pred. Returns a vector of size T_frames_stft * 2 * freq_bins,
// laid out as data[t + c * T_frames_stft] (matching the internal ggml layout).
// On failure returns an empty vector.
//
// This is exposed so callers can compare the C++ SineGen output against
// the Python reference (har.npy from dump_decoder_inputs.py) to verify
// the precompute_sine_source implementation.
std::vector<float> kokoro_decoder_precompute_har(
    kokoro_decoder * dec,
    const float * F0_pred,   int T_f0,
    int * T_frames_stft_out,
    int * freq_bins_out);
