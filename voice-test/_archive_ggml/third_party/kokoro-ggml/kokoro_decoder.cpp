// kokoro_decoder.cpp — ggml-based inference for the Kokoro-82M TTS decoder.
//
// Implements the forward pass of the Kokoro iSTFTNet decoder using ggml
// tensor operations. The encoder (PLBERT + text encoder + duration/F0/N
// predictors) remains on ONNX Runtime; only the decoder is ported to ggml.
//
// The SineGen + forward STFT are precomputed in C++ (they need cumsum and
// atan2 which are not in ggml's op set). Everything else — Conv1d,
// ConvTranspose1d, AdaIN, Snake activation, InstanceNorm, LeakyReLU, exp,
// sin, tanh, iSTFT — is expressed as a single ggml compute graph.
//
// See kokoro_decoder.h for the public C API.

#include "kokoro_decoder.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// kokoro_decoder struct
// ---------------------------------------------------------------------------
struct kokoro_decoder {
    // Weight storage: ggml_context owns all weight tensors loaded from GGUF.
    ggml_context * ctx_weights = nullptr;
    // F16 weight storage: conv1d weights pre-converted to F16 to avoid
    // per-forward ggml_cast nodes in the graph.
    ggml_context * ctx_weights_f16 = nullptr;
    gguf_context * gguf_ctx    = nullptr;

    // Map from PyTorch state_dict key -> ggml tensor.
    // For conv1d weights, the tensor is F16 (stored in ctx_weights_f16).
    // For all other weights, the tensor is F32 (stored in ctx_weights).
    std::unordered_map<std::string, ggml_tensor *> weights;

    // Hyperparameters (mirrors config.json istftnet + top-level keys).
    int dim_in                 = 64;
    int style_dim              = 128;
    int n_mels                 = 80;
    int upsample_initial_channel = 512;
    int gen_istft_n_fft        = 20;
    int gen_istft_hop_size     = 5;
    int num_upsamples          = 2;
    int num_resblocks_per_up   = 3;

    std::vector<int> upsample_rates       = {10, 6};
    std::vector<int> upsample_kernel_sizes = {20, 12};
    std::vector<int> resblock_kernel_sizes = {3, 7, 11};
    std::vector<std::vector<int>> resblock_dilation_sizes = {{1,3,5},{1,3,5},{1,3,5}};

    int n_threads = 4;
};

// ---------------------------------------------------------------------------
// Weight access helpers
// ---------------------------------------------------------------------------
static ggml_tensor * w(const kokoro_decoder * dec, const std::string & name) {
    auto it = dec->weights.find(name);
    if (it == dec->weights.end()) {
        fprintf(stderr, "[kokoro_decoder] weight not found: %s\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// Graph-building helpers
//
// All helpers operate in a ggml_context and return graph nodes (tensors).
// They do NOT execute anything — the graph is built first, then executed
// in one shot by kokoro_decoder_forward.
// ---------------------------------------------------------------------------

// Time-only nearest-neighbor upsample by an integer factor.
// ggml_upscale() scales BOTH ne[0] (time) and ne[1] (channels), which is
// wrong for our use case. We use ggml_upscale_ext() to scale ne[0] only.
// x: [T, C, 1, 1] -> [T*scale, C, 1, 1]
static ggml_tensor * upsample_time(
        ggml_context * ctx,
        ggml_tensor * x,
        int scale) {
    return ggml_upscale_ext(ctx, x,
        int(x->ne[0]) * scale,
        int(x->ne[1]),
        int(x->ne[2]),
        int(x->ne[3]),
        GGML_SCALE_MODE_NEAREST);
}

// Conv1d with optional bias, stride, padding, dilation.
// weight: [K, IC, OC] (PyTorch [OC, IC, K] in row-major), stored as F32 in GGUF
// bias:   [OC] or nullptr
// x:      [T, IC, 1]
// returns: [T_out, OC, 1]
//
// NOTE: ggml's conv_1d uses im2col internally, which requires the weight
// tensor to be F16. We pre-convert conv1d weights to F16 at load time
// (see kokoro_decoder_init_from_file) so we can skip the ggml_cast graph
// node entirely. This eliminates ~60 cast nodes from the graph.
static ggml_tensor * conv1d(
        ggml_context * ctx,
        ggml_tensor * weight,
        ggml_tensor * bias,
        ggml_tensor * x,
        int stride, int padding, int dilation) {
    // Pass weight directly to ggml_conv_1d. ggml_conv_1d now selects the
    // im2col output type based on the weight type: F32 weight -> F32 im2col
    // (pure F32 GEMM with AVX2 SIMD), F16 weight -> F16 im2col.
    ggml_tensor * y = ggml_conv_1d(ctx, weight, x, stride, padding, dilation);
    if (bias) {
        // bias is [OC]; reshape to [1, OC] for broadcasting over time.
        ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

// ConvTranspose1d with padding (ggml's version doesn't support padding,
// so we crop the output by `pad` on each side using a view).
// weight: [K, OC, IC] (PyTorch [IC, OC, K] in row-major)
// x:      [T, IC, 1]
// returns: [T_out, OC, 1] where T_out = T*stride
static ggml_tensor * conv_transpose1d(
        ggml_context * ctx,
        ggml_tensor * weight,
        ggml_tensor * bias,
        ggml_tensor * x,
        int stride, int pad) {
    // ggml_conv_transpose_1d asserts p0==0, d0==1; output length = (T-1)*s + K.
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, weight, x, stride, 0, 1);

    // ggml output: [T_ggml, OC, 1, 1] where T_ggml = (T-1)*s + K
    // PyTorch output (with pad): T_out = (T-1)*s + K - 2*pad = T*stride (when pad=(K-s)/2)
    // Crop pad elements from each side of the time dimension.
    if (pad > 0) {
        int64_t T_ggml = y->ne[0];
        int64_t OC     = y->ne[1];
        int64_t T_out  = T_ggml - 2 * pad;
        if (T_out <= 0) {
            fprintf(stderr, "[kokoro_decoder] conv_transpose1d: T_out=%lld <= 0\n", (long long)T_out);
            return y;
        }
        size_t nb1 = y->nb[1]; // stride for dim 1 (time*4)
        size_t nb2 = y->nb[2]; // stride for dim 2
        size_t offset = (size_t)pad * y->nb[0];
        y = ggml_view_3d(ctx, y, T_out, OC, 1, nb1, nb2, offset);
        y = ggml_cont(ctx, y);
    }

    if (bias) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

// InstanceNorm1d with affine (per-channel mean/var over time, then scale+bias).
// x:      [T, C, 1]
// weight: [C] (gamma)
// bias:   [C] (beta)
// returns: [T, C, 1]
//
// Optimization: use ggml_group_norm with n_groups=C, which is a single
// optimized op (multi-threaded, SIMD) that replaces 7 separate graph nodes
// (mean, sub, sqr, mean, add1, sqrt, div). This cuts the adain node count
// from 22 to ~15 and avoids barrier syncs between normalization sub-ops.
static ggml_tensor * instance_norm1d(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * weight,
        ggml_tensor * bias,
        float eps) {
    // x: [T, C, 1] -> reshape to [T, 1, C] so ggml_group_norm sees ne[2]=C.
    // This is a metadata-only change (data layout is data[t + c*T] in both).
    int64_t T = x->ne[0];
    int64_t C = x->ne[1];
    ggml_tensor * x_r = ggml_reshape_3d(ctx, x, T, 1, C);
    // group_norm with n_groups=C normalizes each channel independently over
    // ne[0]*ne[1] = T*1 = T elements. This is exactly InstanceNorm1d.
    ggml_tensor * normed = ggml_group_norm(ctx, x_r, (int)C, eps);
    // reshape back to [T, C, 1]
    normed = ggml_reshape_3d(ctx, normed, T, C, 1);
    // apply affine: weight * x_norm + bias  (broadcast [1, C] over [T, C])
    if (weight) {
        ggml_tensor * w2 = ggml_reshape_2d(ctx, weight, 1, weight->ne[0]);
        normed = ggml_mul(ctx, normed, w2);
    }
    if (bias) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        normed = ggml_add(ctx, normed, b2);
    }
    return normed;
}

// InstanceNorm1d WITHOUT affine (just normalize, no scale/bias).
// Used by adain1d which fuses the affine with the AdaIN scale/bias.
// x: [T, C, 1] -> normalized [T, C, 1]
static ggml_tensor * instance_norm1d_no_affine(
        ggml_context * ctx,
        ggml_tensor * x,
        float eps) {
    int64_t T = x->ne[0];
    int64_t C = x->ne[1];
    ggml_tensor * x_r = ggml_reshape_3d(ctx, x, T, 1, C);
    ggml_tensor * normed = ggml_group_norm(ctx, x_r, (int)C, eps);
    return ggml_reshape_3d(ctx, normed, T, C, 1);
}

// ---------------------------------------------------------------------------
// Precomputed AdaIN parameters
//
// The style vector is fixed for the entire forward pass, so all 58 AdaIN fc
// layer outputs (h = fc_weight @ style + fc_bias) and the fused affine params
// (combined_weight, combined_bias) can be precomputed in C++ before building
// the ggml graph. This eliminates 58 MUL_MAT nodes + ~290 associated nodes
// (ADD, MUL, VIEW, RESHAPE) from the graph, reducing barrier sync overhead.
//
// For each AdaIN layer with prefix P (e.g., "encode.norm1"):
//   weights: P.fc.weight [style_dim, C*2], P.fc.bias [C*2]
//            P.norm.weight [C], P.norm.bias [C]
//   precomputed:
//     h = fc_weight @ style + fc_bias  -> [C*2]
//     gamma = h[:C], beta = h[C:]
//     combined_weight = (1+gamma) * norm_weight  -> [C]
//     combined_bias   = (1+gamma) * norm_bias + beta  -> [C]
// ---------------------------------------------------------------------------
struct AdainParams {
    std::vector<float> combined_weight;  // [C]
    std::vector<float> combined_bias;    // [C]
};

struct PrecomputedAdain {
    // Map from AdaIN prefix (e.g., "encode.norm1") to precomputed params.
    std::unordered_map<std::string, AdainParams> params;

    // ggml tensors created in the graph context, keyed by prefix.
    // Each entry is {combined_weight_tensor, combined_bias_tensor}.
    std::unordered_map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> tensors;
};

// Precompute all AdaIN fc outputs and fused affine params in C++.
// Call this before building the ggml graph.
static void precompute_adain_params(
        const kokoro_decoder * dec,
        const float * style_dec,
        PrecomputedAdain & out) {
    const int style_dim = dec->style_dim;  // 128

    // Scan all weights for keys ending in ".fc.weight"
    for (const auto & [name, tensor] : dec->weights) {
        const std::string suffix = ".fc.weight";
        if (name.size() <= suffix.size()) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;

        // Extract prefix: e.g., "encode.norm1" from "encode.norm1.fc.weight"
        std::string prefix = name.substr(0, name.size() - suffix.size());

        // Get fc weight and bias
        ggml_tensor * fc_w = tensor;  // [style_dim, C*2]
        ggml_tensor * fc_b = w(dec, prefix + ".fc.bias");  // [C*2]
        if (!fc_w || !fc_b) continue;

        // Get norm weight and bias
        ggml_tensor * norm_w = w(dec, prefix + ".norm.weight");  // [C]
        ggml_tensor * norm_b = w(dec, prefix + ".norm.bias");    // [C]
        if (!norm_w) continue;

        const int C = (int)norm_w->ne[0];
        const int C2 = C * 2;

        // Verify fc weight shape: [style_dim, C*2]
        if ((int)fc_w->ne[0] != style_dim || (int)fc_w->ne[1] != C2) continue;

        // Compute h = fc_w @ style + fc_b  -> [C*2]
        // fc_w data layout: [style_dim, C*2] column-major (ne[0]=style_dim varies fastest)
        // h[c] = sum_d(fc_w[d + c*style_dim] * style[d]) + fc_b[c]
        std::vector<float> h(C2);
        const float * fw = (const float *)fc_w->data;
        const float * fb = (const float *)fc_b->data;
        for (int c = 0; c < C2; c++) {
            float val = fb[c];
            const float * col = fw + c * style_dim;
            for (int d = 0; d < style_dim; d++) {
                val += col[d] * style_dec[d];
            }
            h[c] = val;
        }

        // Compute combined_weight and combined_bias
        AdainParams p;
        p.combined_weight.resize(C);
        p.combined_bias.resize(C);
        const float * nw = (const float *)norm_w->data;
        const float * nb = norm_b ? (const float *)norm_b->data : nullptr;

        for (int c = 0; c < C; c++) {
            float gamma = h[c];
            float beta = h[c + C];
            float one_plus_gamma = 1.0f + gamma;
            p.combined_weight[c] = one_plus_gamma * nw[c];
            p.combined_bias[c] = one_plus_gamma * (nb ? nb[c] : 0.0f) + beta;
        }

        out.params[prefix] = std::move(p);
    }
}

// Create ggml tensors for precomputed AdaIN params in the graph context.
// Must be called after ggml_init(ctx) and before building the graph.
static void create_adain_tensors(
        ggml_context * ctx,
        const PrecomputedAdain & precomputed,
        std::unordered_map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> & tensors) {
    for (const auto & [prefix, params] : precomputed.params) {
        int C = (int)params.combined_weight.size();
        ggml_tensor * cw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        memcpy(cw->data, params.combined_weight.data(), C * sizeof(float));
        ggml_tensor * cb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        memcpy(cb->data, params.combined_bias.data(), C * sizeof(float));
        tensors[prefix] = {cw, cb};
    }
}

// AdaIN1d with precomputed combined_weight and combined_bias.
// This replaces the old adain1d which computed fc(style) in the graph.
//
// x:                [T, C, 1]
// combined_weight:  [C]  (precomputed (1+gamma) * norm_weight)
// combined_bias:    [C]  (precomputed (1+gamma) * norm_bias + beta)
//
// Result: combined_weight * InstanceNorm(x) + combined_bias
// Only 2 broadcast ops (mul + add) + 1 group_norm node = 3 graph nodes total.
// (Old adain1d created ~11 graph nodes.)
static ggml_tensor * adain1d(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * combined_weight,
        ggml_tensor * combined_bias) {
    // InstanceNorm WITHOUT affine (we apply fused affine below)
    ggml_tensor * normed = instance_norm1d_no_affine(ctx, x, 1e-5f);

    // Reshape combined_weight/bias from [C] to [1, C] for broadcasting over [T, C]
    int64_t C = combined_weight->ne[0];
    ggml_tensor * w2 = ggml_reshape_2d(ctx, combined_weight, 1, C);
    ggml_tensor * b2 = ggml_reshape_2d(ctx, combined_bias, 1, C);

    // Apply fused affine: combined_weight * normed + combined_bias
    ggml_tensor * scaled = ggml_mul(ctx, normed, w2);   // broadcast [1, C] over [T, C]
    ggml_tensor * result = ggml_add(ctx, scaled, b2);   // broadcast [1, C] over [T, C]
    return result;
}

// Snake activation: x + (1/alpha) * sin(alpha * x)^2
// x:     [T, C, 1]
// alpha: [1, C, 1]
static ggml_tensor * snake_activation(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * alpha) {
    // alpha * x  (broadcast alpha [1, C] over x [T, C])
    ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    // sin(alpha * x)
    ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    // sin^2
    ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
    // snake_part = sin^2 / alpha  (broadcast alpha [1, C] over sin_sq [T, C])
    // This computes (1/alpha) * sin^2 without needing a separate 1/alpha tensor.
    ggml_tensor * snake_part = ggml_div(ctx, sin_sq, alpha);
    // x + snake_part
    return ggml_add(ctx, x, snake_part);
}

// AdaINResBlock1: 3-iteration residual block with Snake activation + AdaIN.
// See istftnet.py AdaINResBlock1.
//
// weights prefix: e.g. "generator.resblocks.0"
//   {prefix}.convs1.{0,1,2}.{weight,bias}
//   {prefix}.convs2.{0,1,2}.{weight,bias}
//   {prefix}.adain1.{0,1,2}.fc.{weight,bias}
//   {prefix}.adain1.{0,1,2}.norm.{weight,bias}
//   {prefix}.adain2.{0,1,2}.fc.{weight,bias}
//   {prefix}.adain2.{0,1,2}.norm.{weight,bias}
//   {prefix}.alpha1.{0,1,2}
//   {prefix}.alpha2.{0,1,2}
static ggml_tensor * adain_resblock1(
        ggml_context * ctx,
        const kokoro_decoder * dec,
        const std::string & prefix,
        ggml_tensor * x,
        ggml_tensor * style,
        int kernel_size,
        const std::vector<int> & dilations) {
    ggml_tensor * residual = x;

    for (int i = 0; i < 3; i++) {
        // --- First sub-block: adain1 -> snake1 -> convs1 ---
        ggml_tensor * n1 = adain1d(ctx, x, style,
            w(dec, prefix + ".adain1." + std::to_string(i) + ".norm.weight"),
            w(dec, prefix + ".adain1." + std::to_string(i) + ".norm.bias"),
            w(dec, prefix + ".adain1." + std::to_string(i) + ".fc.weight"),
            w(dec, prefix + ".adain1." + std::to_string(i) + ".fc.bias"));
        ggml_tensor * a1 = w(dec, prefix + ".alpha1." + std::to_string(i));
        // alpha is [1, C, 1] in PyTorch, stored as [C, 1, 1] in gguf? Let me check.
        // PyTorch: nn.Parameter(torch.ones(1, channels, 1)) -> shape [1, C, 1]
        // In gguf: ne=[1, C, 1] (reversed dims)
        // For broadcasting with x [T, C, 1], alpha [1, C, 1] broadcasts correctly.
        ggml_tensor * s1 = snake_activation(ctx, n1, a1);
        int p1 = (kernel_size * dilations[i] - dilations[i]) / 2;
        ggml_tensor * c1 = conv1d(ctx,
            w(dec, prefix + ".convs1." + std::to_string(i) + ".weight"),
            w(dec, prefix + ".convs1." + std::to_string(i) + ".bias"),
            s1, 1, p1, dilations[i]);

        // --- Second sub-block: adain2 -> snake2 -> convs2 ---
        ggml_tensor * n2 = adain1d(ctx, c1, style,
            w(dec, prefix + ".adain2." + std::to_string(i) + ".norm.weight"),
            w(dec, prefix + ".adain2." + std::to_string(i) + ".norm.bias"),
            w(dec, prefix + ".adain2." + std::to_string(i) + ".fc.weight"),
            w(dec, prefix + ".adain2." + std::to_string(i) + ".fc.bias"));
        ggml_tensor * a2 = w(dec, prefix + ".alpha2." + std::to_string(i));
        ggml_tensor * s2 = snake_activation(ctx, n2, a2);
        int p2 = (kernel_size - 1) / 2;
        ggml_tensor * c2 = conv1d(ctx,
            w(dec, prefix + ".convs2." + std::to_string(i) + ".weight"),
            w(dec, prefix + ".convs2." + std::to_string(i) + ".bias"),
            s2, 1, p2, 1);

        // x = c2 + x (residual)
        x = ggml_add(ctx, c2, x);
    }
    return x;
}

// Depthwise ConvTranspose1d with kernel=3, stride=2, padding=1, output_padding=1.
// PyTorch: nn.ConvTranspose1d(C, C, kernel_size=3, stride=2, groups=C, padding=1, output_padding=1)
//
// This is a depthwise transposed convolution (each channel processed independently).
// ggml's conv_transpose_1d doesn't support groups, so we implement it manually.
//
// For kernel=3, stride=2, padding=1, output_padding=1, the transposed convolution
// formula gives output length = (T-1)*2 + 3 - 2*1 + 1 = 2*T.
//
// The computation per channel c (with weights w0, w1, w2 = weight[c, 0, 0:3]):
//   out[2*i]     = w1 * in[i]
//   out[2*i+1]   = w0 * in[i] + w2 * in[i+1]   (for i < T-1)
//   out[2*(T-1)+1] = w0 * in[T-1]              (last odd sample, no in[T])
//
// weight: PyTorch [C, 1, 3] -> ggml ne=[3, 1, C]
// x:      [T, C, 1]
// output: [2*T, C, 1]
static ggml_tensor * depthwise_conv_transpose1d_k3_s2(
        ggml_context * ctx,
        ggml_tensor * weight,
        ggml_tensor * x) {
    int T = int(x->ne[0]);
    int C = int(x->ne[1]);

    // weight has ne=[3, 1, C]. Extract individual taps.
    // Each tap is a [1, 1, C] view; we make it contiguous and reshape to
    // [1, C, 1] so it broadcasts against x [T, C, 1] in ggml_repeat.
    // weight->nb[0] = sizeof(float), weight->nb[1] = 3*sizeof(float), weight->nb[2] = 3*sizeof(float)
    auto extract_tap = [&](int tap_idx) -> ggml_tensor * {
        ggml_tensor * v = ggml_view_3d(ctx, weight,
            1, 1, C,
            weight->nb[1], weight->nb[2],
            tap_idx * sizeof(float));
        return ggml_reshape_3d(ctx, ggml_cont(ctx, v), 1, C, 1);  // [1, C, 1]
    };
    ggml_tensor * w0 = extract_tap(0);
    ggml_tensor * w1 = extract_tap(1);
    ggml_tensor * w2 = extract_tap(2);

    // a = w0 * x, b = w1 * x, c = w2 * x  (all [T, C, 1])
    // w0/w1/w2 are [1, C, 1]; use ggml_mul's broadcast (avoid ggml_repeat).
    ggml_tensor * a = ggml_mul(ctx, x, w0);
    ggml_tensor * b = ggml_mul(ctx, x, w1);
    ggml_tensor * c = ggml_mul(ctx, x, w2);

    // For odd output: odd[i] = c[i] + a_shifted[i]
    //   where a_shifted[i] = a[i+1] for i < T-1, a_shifted[T-1] = 0.
    // This comes from the transposed convolution formula:
    //   raw[2i+k] += input[i] * weight[k]
    // After cropping padding=1 from the left:
    //   output[2i]   = input[i] * w1                 (even)
    //   output[2i+1] = input[i] * w2 + input[i+1] * w0 (odd, i < T-1)
    //   output[2T-1] = input[T-1] * w2                (last odd)
    // So odd[i] = c[i] + a[i+1], with a[T]=0.
    ggml_tensor * a_padded = ggml_pad(ctx, a, 1, 0, 0, 0);  // [T+1, C, 1, 1]
    ggml_tensor * a_shifted = ggml_view_3d(ctx, a_padded,
        T, C, 1,
        a_padded->nb[1], a_padded->nb[2],
        1 * sizeof(float));  // [T, C, 1]

    ggml_tensor * odd = ggml_add(ctx, c, a_shifted);  // [T, C, 1]

    // Interleave b and odd: out[2*i] = b[i], out[2*i+1] = odd[i].
    // Reshape b and odd from [T, C, 1, 1] to [1, T, C, 1], concat along dim 0
    // to get [2, T, C, 1], then reshape to [2T, C, 1, 1].
    // Memory layout of [2, T, C, 1]: (k, t, c) at offset k + t*2 + c*2*T
    // -> b[0,c], odd[0,c], b[1,c], odd[1,c], ... (correct interleaving)
    ggml_tensor * b_view = ggml_reshape_4d(ctx, b, 1, T, C, 1);
    ggml_tensor * odd_view = ggml_reshape_4d(ctx, odd, 1, T, C, 1);
    ggml_tensor * interleaved = ggml_concat(ctx, b_view, odd_view, 0);  // [2, T, C, 1]
    ggml_tensor * result = ggml_reshape_3d(ctx, interleaved, 2 * T, C, 1);  // [2T, C, 1]

    return result;
}

// AdainResBlk1d: the encode/decode block (different from AdaINResBlock1).
// See istftnet.py AdainResBlk1d.
//
// weights prefix: e.g. "encode" or "decode.0"
//   {prefix}.conv1.{weight,bias}, conv2.{weight,bias}
//   {prefix}.norm1.{norm.weight, norm.bias, fc.weight, fc.bias}
//   {prefix}.norm2.{norm.weight, norm.bias, fc.weight, fc.bias}
//   {prefix}.conv1x1.{weight} (if learned_sc, no bias)
//   {prefix}.pool.weight (if upsample=True, depthwise ConvTranspose1d)
static ggml_tensor * adain_resblk1d(
        ggml_context * ctx,
        const kokoro_decoder * dec,
        const std::string & prefix,
        ggml_tensor * x,
        ggml_tensor * style,
        bool upsample = false) {
    // Shortcut path: UpSample1d(nearest) + optional conv1x1
    // PyTorch: _shortcut(x) = conv1x1(upsample(x)) if learned_sc else upsample(x)
    ggml_tensor * shortcut = x;
    if (upsample) {
        shortcut = upsample_time(ctx, shortcut, 2);
    }
    ggml_tensor * sc_w = w(dec, prefix + ".conv1x1.weight");
    if (sc_w) {
        // conv1x1: 1x1 conv, no bias. Weight is pre-converted to F16 at load time.
        ggml_tensor * sc_w_f16 = (sc_w->type == GGML_TYPE_F16)
            ? sc_w
            : ggml_cast(ctx, sc_w, GGML_TYPE_F16);
        shortcut = ggml_conv_1d(ctx, sc_w_f16, shortcut, 1, 0, 1);
    }

    // Residual path:
    // norm1 -> actv(leaky_relu 0.2) -> [pool if upsample] -> conv1 -> norm2 -> actv -> conv2
    ggml_tensor * r = adain1d(ctx, x, style,
        w(dec, prefix + ".norm1.norm.weight"),
        w(dec, prefix + ".norm1.norm.bias"),
        w(dec, prefix + ".norm1.fc.weight"),
        w(dec, prefix + ".norm1.fc.bias"));
    r = ggml_leaky_relu(ctx, r, 0.2f, false);
    if (upsample) {
        // pool: depthwise ConvTranspose1d(k=3, s=2, groups=C, p=1, op=1)
        // This doubles the time resolution using a learned transposed conv.
        ggml_tensor * pool_w = w(dec, prefix + ".pool.weight");
        if (pool_w) {
            r = depthwise_conv_transpose1d_k3_s2(ctx, pool_w, r);
        } else {
            // Fallback: nearest-neighbor (should not happen if weights are exported)
            fprintf(stderr, "[kokoro_decoder] WARNING: pool weight not found for %s, using nearest upsample\n",
                    prefix.c_str());
            r = upsample_time(ctx, r, 2);
        }
    }
    r = conv1d(ctx,
        w(dec, prefix + ".conv1.weight"),
        w(dec, prefix + ".conv1.bias"),
        r, 1, 1, 1);
    r = adain1d(ctx, r, style,
        w(dec, prefix + ".norm2.norm.weight"),
        w(dec, prefix + ".norm2.norm.bias"),
        w(dec, prefix + ".norm2.fc.weight"),
        w(dec, prefix + ".norm2.fc.bias"));
    r = ggml_leaky_relu(ctx, r, 0.2f, false);
    r = conv1d(ctx,
        w(dec, prefix + ".conv2.weight"),
        w(dec, prefix + ".conv2.bias"),
        r, 1, 1, 1);

    // out = (r + shortcut) / sqrt(2)
    // Use ggml_scale (scalar multiply, multi-threaded) instead of
    // ggml_new_f32 + ggml_repeat + ggml_mul (single-threaded repeat).
    ggml_tensor * sum = ggml_add(ctx, r, shortcut);
    return ggml_scale(ctx, sum, 1.0f / 1.41421356f);
}

// ---------------------------------------------------------------------------
// SineGen + SourceModuleHnNSF + forward STFT (precomputed in C++)
//
// These need cumsum (not in ggml) and atan2 (not in ggml), so we compute
// them directly in C++ and feed the result (har) into the ggml graph.
// ---------------------------------------------------------------------------

struct SineSource {
    std::vector<float> har;           // [T_frames_stft * 2 * freq_bins] (magnitude + phase interleaved per frame)
    int T_frames_stft;
    int freq_bins;
    int T_audio;
};

// Generate the harmonic source signal from F0, then compute its STFT.
// This mirrors: Generator.forward() lines:
//   f0_upsamp -> m_source(f0) -> stft.transform(har_source) -> cat([spec, phase])
static void precompute_sine_source(
        const kokoro_decoder * dec,
        const float * F0_pred,  // [T_f0] — full frame rate, gets upsampled by 300x to audio rate
        int T_f0,
        SineSource & out) {
    // Hyperparams
    const int sampling_rate = 24000;
    const int harmonic_num = 8;
    const int dim = harmonic_num + 1; // 9
    const float sine_amp = 0.1f;
    const float noise_std = 0.003f;
    const float voiced_threshold = 10.0f;

    const int n_fft = dec->gen_istft_n_fft;       // 20
    const int hop = dec->gen_istft_hop_size;       // 5
    const int freq_bins = n_fft / 2 + 1;           // 11
    const int total_upsample = dec->upsample_rates[0] * dec->upsample_rates[1] * hop; // 10*6*5=300

    const int T_audio = T_f0 * total_upsample;

    // Step 1: Upsample F0 by total_upsample (nearest neighbor, like nn.Upsample default)
    // F0_upsampled: [T_audio]
    std::vector<float> f0_up(T_audio);
    for (int i = 0; i < T_f0; i++) {
        for (int j = 0; j < total_upsample; j++) {
            f0_up[i * total_upsample + j] = F0_pred[i];
        }
    }

    // Step 2: SineGen
    // Build fn = f0 * [1, 2, ..., 9] for each time step -> [T_audio, dim]
    // Then rad = (fn / sampling_rate) % 1
    // Downsample to T_f0 (take every total_upsample-th sample since f0 was upsampled by nearest)
    // cumsum at T_f0
    // Multiply by 2*pi*total_upsample
    // Upsample to T_audio via linear interpolation
    // sin -> sine_waves [T_audio, dim]

    // rad at T_f0.
    //
    // PyTorch SineGen adds rand_ini to rad_values[:, 0, :] at T_audio rate,
    // but then F.interpolate(scale_factor=1/upsample_scale, align_corners=False)
    // downsamples rad_values from T_audio to T_f0 by sampling at position
    // (i+0.5)*60 - 0.5 = i*60 + 29.5, which never hits index 0. So the
    // rand_ini is completely lost after downsampling and we can omit it.
    //
    // Combined with f0_upsamp being nearest-neighbor (so each 60-sample
    // segment is constant), rad_down[i] = (F0_pred[i]/sr) % 1 exactly.
    std::vector<float> rad(T_f0 * dim, 0.0f);
    for (int d = 0; d < dim; d++) {
        for (int i = 0; i < T_f0; i++) {
            float f0_val = F0_pred[i] * (d + 1); // harmonic
            rad[i * dim + d] = fmodf(f0_val / sampling_rate, 1.0f);
        }
    }

    // cumsum at T_f0 per harmonic
    std::vector<float> phase_low(T_f0 * dim, 0.0f);
    for (int d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (int i = 0; i < T_f0; i++) {
            acc += rad[i * dim + d];
            phase_low[i * dim + d] = acc;
        }
    }

    // Scale by 2*pi*total_upsample and upsample to T_audio via linear interpolation.
    // PyTorch uses F.interpolate(phase_low * upsample_scale, scale_factor=upsample_scale,
    // mode="linear", align_corners=False).
    // F.interpolate with align_corners=False uses the formula:
    //   input_pos = (out_idx + 0.5) * (input_size / output_size) - 0.5
    // For input_size=T_f0, output_size=T_audio=T_f0*total_upsample:
    //   input_pos = (t + 0.5) / total_upsample - 0.5
    // This has a half-frame offset compared to the naive t/total_upsample formula.
    std::vector<float> phase(T_audio * dim, 0.0f);
    const float scale_factor = (float)T_f0 / (float)T_audio;  // 1/total_upsample
    for (int d = 0; d < dim; d++) {
        for (int t = 0; t < T_audio; t++) {
            // Match PyTorch F.interpolate(align_corners=False) formula
            float src_pos = (float)(t + 0.5) * scale_factor - 0.5f;
            // Clamp to valid range
            if (src_pos < 0.0f) src_pos = 0.0f;
            if (src_pos > T_f0 - 1.0f) src_pos = T_f0 - 1.0f;
            int idx = (int)src_pos;
            if (idx >= T_f0 - 1) {
                phase[t * dim + d] = phase_low[(T_f0 - 1) * dim + d] * 2.0f * (float)M_PI * total_upsample;
            } else {
                float frac = src_pos - idx;
                float p0 = phase_low[idx * dim + d] * 2.0f * (float)M_PI * total_upsample;
                float p1 = phase_low[(idx + 1) * dim + d] * 2.0f * (float)M_PI * total_upsample;
                phase[t * dim + d] = p0 + (p1 - p0) * frac;
            }
        }
    }

    // sine_waves = sin(phase) * sine_amp -> [T_audio, dim]
    std::vector<float> sine_waves(T_audio * dim);
    for (int i = 0; i < T_audio * dim; i++) {
        sine_waves[i] = sinf(phase[i]) * sine_amp;
    }

    // UV detection and noise
    // uv = (f0 > voiced_threshold)
    // noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    // noise = noise_amp * randn
    // sine_waves = sine_waves * uv + noise
    //
    // NOTE: PyTorch uses torch.randn_like which is non-deterministic. We use a
    // fixed seed here for reproducibility; this contributes to cos_sim
    // differences in unvoiced/noise regions but does not affect the harmonic
    // structure.
    std::mt19937 noise_rng(12345);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    for (int t = 0; t < T_audio; t++) {
        float f0_val = f0_up[t];
        float uv = (f0_val > voiced_threshold) ? 1.0f : 0.0f;
        float n_amp = uv * noise_std + (1.0f - uv) * sine_amp / 3.0f;
        for (int d = 0; d < dim; d++) {
            float noise = n_amp * norm_dist(noise_rng);
            sine_waves[t * dim + d] = sine_waves[t * dim + d] * uv + noise;
        }
    }

    // Step 3: SourceModuleHnNSF
    // sine_merge = tanh(l_linear(sine_waves))  // Linear(dim, 1)
    // noise = randn * sine_amp / 3
    // har_source = sine_merge  // [T_audio]
    // noi_source = noise       // [T_audio]
    ggml_tensor * l_linear_w = w(dec, "generator.m_source.l_linear.weight");
    ggml_tensor * l_linear_b = w(dec, "generator.m_source.l_linear.bias");

    std::vector<float> har_source(T_audio, 0.0f);
    // l_linear weight: PyTorch [1, 9] -> ggml ne=[9, 1]
    // sine_waves: [T_audio, dim] -> for each t, compute dot(weight_row, sine_waves[t])
    // weight[0, d] = l_linear_w->data[d] (since ne=[9, 1], data is [d0, d1, ..., d8])
    const float * llw = (const float *) l_linear_w->data;
    const float * llb = l_linear_b ? (const float *) l_linear_b->data : nullptr;
    for (int t = 0; t < T_audio; t++) {
        float val = 0.0f;
        for (int d = 0; d < dim; d++) {
            val += sine_waves[t * dim + d] * llw[d];
        }
        val += llb ? llb[0] : 0.0f;
        har_source[t] = tanhf(val);
    }

    // Step 4: Forward STFT of har_source
    // CustomSTFT.transform(har_source):
    //   center pad (n_fft//2 on each side, replicate)
    //   conv1d with weight_forward_real (stride=hop) -> real
    //   conv1d with weight_forward_imag (stride=hop) -> imag
    //   magnitude = sqrt(real^2 + imag^2 + 1e-14)
    //   phase = atan2(imag, real)
    //   correction: if imag==0 and real<0, phase = pi
    int pad_len = n_fft / 2;
    int T_padded = T_audio + 2 * pad_len;
    std::vector<float> padded(T_padded);
    // Replicate padding
    for (int i = 0; i < pad_len; i++) {
        padded[i] = har_source[0];
        padded[T_padded - 1 - i] = har_source[T_audio - 1];
    }
    memcpy(padded.data() + pad_len, har_source.data(), T_audio * sizeof(float));

    int T_frames_stft = (T_padded - n_fft) / hop + 1;

    // STFT weights: weight_forward_real, weight_forward_imag
    // PyTorch shape: [freq_bins, 1, n_fft] -> ggml ne=[n_fft, 1, freq_bins]
    ggml_tensor * wfr = w(dec, "generator.stft.weight_forward_real");
    ggml_tensor * wfi = w(dec, "generator.stft.weight_forward_imag");
    const float * wfr_data = (const float *) wfr->data;
    const float * wfi_data = (const float *) wfi->data;

    // For each frame, compute real and imag via dot product.
    //
    // har tensor is stored in ggml layout: ne=[T_frames, 2*freq_bins, 1],
    // so element (t, c) is at data[t + c*T_frames] (time varies fastest).
    // Channel layout: c in [0, freq_bins) = magnitude, c in [freq_bins, 2*freq_bins) = phase.
    out.har.resize(T_frames_stft * 2 * freq_bins);
    out.T_frames_stft = T_frames_stft;
    out.freq_bins = freq_bins;
    out.T_audio = T_audio;

    for (int f = 0; f < T_frames_stft; f++) {
        int start = f * hop;
        for (int k = 0; k < freq_bins; k++) {
            // weight_forward_real[k, 0, :] = wfr_data[k * n_fft + ...]
            // But in ggml, ne=[n_fft, 1, freq_bins], so:
            // wfr_data[(k * 1 + 0) * n_fft + n] for n in 0..n_fft-1
            // = wfr_data[k * n_fft + n]
            float real_val = 0.0f, imag_val = 0.0f;
            for (int n = 0; n < n_fft; n++) {
                float sample = padded[start + n];
                real_val += sample * wfr_data[k * n_fft + n];
                imag_val += sample * wfi_data[k * n_fft + n];
            }
            float mag = sqrtf(real_val * real_val + imag_val * imag_val + 1e-14f);
            float phase_val = atan2f(imag_val, real_val);
            if (imag_val == 0.0f && real_val < 0.0f) phase_val = (float)M_PI;

            // ggml layout: data[t + c*T_frames]
            out.har[f + k * T_frames_stft] = mag;
            out.har[f + (freq_bins + k) * T_frames_stft] = phase_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Build the main decoder forward graph
//
// Inputs (as ggml tensors in the graph context):
//   asr       : [T_asr, 512, 1]      — T_asr = T_f0/2 (encoder output is at half rate)
//   F0_pred   : [T_f0, 1, 1]         — full frame rate; F0_conv halves it to T_asr
//   N_pred    : [T_f0, 1, 1]         — same length as F0_pred
//   style_dec : [128, 1, 1]
//   har       : [T_frames_stft, 2*freq_bins, 1]  (precomputed)
//
// Output: [T_audio, 1, 1] audio samples
//
// The graph is split into two stages for profiling:
//   Stage 1 (encoder/decode): F0_conv, N_conv, encode, decode.0..3
//       Output: x at 2x time resolution, 512 channels (input to generator)
//   Stage 2 (generator): upsample + resblocks + conv_post + iSTFT
//       Output: waveform
//
// When kokoro_decoder_profile is true, the two stages are built and computed
// separately to measure per-stage timing.
// ---------------------------------------------------------------------------

// Stage 1: encoder + decode blocks.
// Returns the intermediate tensor x at 2x time resolution, 512 channels.
static ggml_tensor * build_encoder_decode_graph(
        ggml_context * ctx,
        const kokoro_decoder * dec,
        ggml_tensor * asr,
        ggml_tensor * F0_pred,
        ggml_tensor * N_pred,
        ggml_tensor * style_dec) {

    // ----- Step 1: F0_conv, N_conv -----
    // F0_conv: Conv1d(1, 1, k=3, s=2, p=1) on F0_pred [T_f0, 1, 1]
    // Output: [T_f0/2, 1, 1] which must match asr's T_asr.
    ggml_tensor * F0 = conv1d(ctx,
        w(dec, "F0_conv.weight"), w(dec, "F0_conv.bias"),
        F0_pred, 2, 1, 1);
    // N_conv: same config
    ggml_tensor * N = conv1d(ctx,
        w(dec, "N_conv.weight"), w(dec, "N_conv.bias"),
        N_pred, 2, 1, 1);

    // ----- Step 2: concat [asr, F0, N] along channel dim -----
    // asr: [T_asr, 512, 1], F0: [T_f0/2, 1, 1], N: [T_f0/2, 1, 1]
    // In PyTorch, the encoder outputs asr at T_asr = T_f0/2 (already halved),
    // and F0_conv/N_conv halve F0_pred/N_pred from T_f0 to T_asr.
    // After concat: [T_asr, 514, 1].
    // Sanity: T_asr must equal T_f0/2 (asserted by the caller via tensor shapes).
    ggml_tensor * x = ggml_concat(ctx, asr, F0, 1); // concat along dim 1 (channels)
    x = ggml_concat(ctx, x, N, 1);

    // ----- Step 3: encode (AdainResBlk1d: 514 -> 1024) -----
    x = adain_resblk1d(ctx, dec, "encode", x, style_dec, false);

    // ----- Step 4: asr_res (Conv1d 512->64, k=1) -----
    ggml_tensor * asr_res = conv1d(ctx,
        w(dec, "asr_res.0.weight"), w(dec, "asr_res.0.bias"),
        asr, 1, 0, 1);

    // ----- Step 5: decode blocks (4x AdainResBlk1d) -----
    // decode.0: 1024+2+64 -> 1024 (res=True, concat x with asr_res, F0, N)
    // decode.1: 1024+2+64 -> 1024
    // decode.2: 1024+2+64 -> 1024
    // decode.3: 1024+2+64 -> 512, upsample=True
    bool res = true;
    for (int i = 0; i < 4; i++) {
        if (res) {
            x = ggml_concat(ctx, x, asr_res, 1);
            x = ggml_concat(ctx, x, F0, 1);
            x = ggml_concat(ctx, x, N, 1);
        }
        bool ups = (i == 3);
        x = adain_resblk1d(ctx, dec, "decode." + std::to_string(i), x, style_dec, ups);
        if (ups) {
            // After upsampling, asr_res/F0/N are at half resolution, so stop concatenating
            res = false;
            // Also upsample asr_res, F0, N to match (for completeness, though res=false now)
        }
    }

    // After decode.3 (upsample), x is at 2x time resolution, channels=512
    // This is the input to the generator.
    return x;
}

// Stage 2: generator (upsample + resblocks + conv_post + iSTFT).
// x:  [2*T_asr, 512, 1]  — output of build_encoder_decode_graph
// har: [T_frames_stft, 2*freq_bins, 1]  (precomputed)
// Returns: [T_audio, 1, 1] audio samples
static ggml_tensor * build_generator_graph(
        ggml_context * ctx,
        const kokoro_decoder * dec,
        ggml_tensor * x,
        ggml_tensor * style_dec,
        ggml_tensor * har) {

    // ----- Step 6: Generator -----
    // For each upsample layer:
    //   x = leaky_relu(x, 0.1)
    //   x_source = noise_convs[i](har) + noise_res[i](x_source, s)
    //   x = ups[i](x)  (ConvTranspose1d)
    //   if last: x = reflection_pad(x)
    //   x = x + x_source
    //   x = sum(resblocks[i*num_kernels + j](x, s) for j) / num_kernels
    int num_upsamples = dec->num_upsamples;   // 2
    int num_kernels   = dec->num_resblocks_per_up; // 3

    for (int i = 0; i < num_upsamples; i++) {
        x = ggml_leaky_relu(ctx, x, 0.1f, false);

        // noise_convs[i](har): brings har from STFT frame rate to the rate
        // matching x AFTER ups[i]. Two configs (see istftnet.py Generator):
        //   - non-last i: Conv1d(22, c_cur, kernel=stride_f0*2,
        //                        stride=stride_f0, padding=(stride_f0+1)//2)
        //     where stride_f0 = prod(upsample_rates[i+1:]).
        //   - last i:     Conv1d(22, c_cur, kernel=1, stride=1, padding=0)
        int nc_stride, nc_padding;
        if (i + 1 < num_upsamples) {
            int stride_f0 = 1;
            for (int k = i + 1; k < num_upsamples; k++) {
                stride_f0 *= dec->upsample_rates[k];
            }
            nc_stride  = stride_f0;
            nc_padding = (stride_f0 + 1) / 2;
        } else {
            nc_stride  = 1;
            nc_padding = 0;
        }
        ggml_tensor * x_source = conv1d(ctx,
            w(dec, "generator.noise_convs." + std::to_string(i) + ".weight"),
            w(dec, "generator.noise_convs." + std::to_string(i) + ".bias"),
            har, nc_stride, nc_padding, 1);

        // noise_res[i]: AdaINResBlock1 with specific kernel size and dilations
        int noise_kernel = (i < num_upsamples - 1) ? 7 : 11;
        std::vector<int> noise_dil = {1, 3, 5};
        x_source = adain_resblock1(ctx, dec,
            "generator.noise_res." + std::to_string(i), x_source, style_dec,
            noise_kernel, noise_dil);

        // ups[i]: ConvTranspose1d
        int stride = dec->upsample_rates[i];
        int kernel = dec->upsample_kernel_sizes[i];
        int pad = (kernel - stride) / 2;
        x = conv_transpose1d(ctx,
            w(dec, "generator.ups." + std::to_string(i) + ".weight"),
            w(dec, "generator.ups." + std::to_string(i) + ".bias"),
            x, stride, pad);

        // Reflection pad for last upsample: pad (1, 0) on time dim
        if (i == num_upsamples - 1) {
            // ReflectionPad1d((1, 0)): prepend 1 reflected sample
            // For simplicity, use ggml_pad_reflect_1d if available, or skip.
            // The reflection pad adds 1 sample at the start. We'll use
            // a view that shifts by 1 and includes the boundary.
            // Actually, let's use ggml_pad_reflect_1d(ctx, x, 1, 0)
            x = ggml_pad_reflect_1d(ctx, x, 1, 0);
        }

        // x = x + x_source (may need size matching)
        x = ggml_add(ctx, x, x_source);

        // resblocks
        ggml_tensor * xs = nullptr;
        for (int j = 0; j < num_kernels; j++) {
            int idx = i * num_kernels + j;
            ggml_tensor * rb = adain_resblock1(ctx, dec,
                "generator.resblocks." + std::to_string(idx), x, style_dec,
                dec->resblock_kernel_sizes[j],
                dec->resblock_dilation_sizes[j]);
            if (xs == nullptr) {
                xs = rb;
            } else {
                xs = ggml_add(ctx, xs, rb);
            }
        }
        // x = xs / num_kernels  (use ggml_scale: multi-threaded scalar mul)
        x = ggml_scale(ctx, xs, 1.0f / float(num_kernels));
    }

    // ----- Step 7: conv_post + iSTFT -----
    x = ggml_leaky_relu(ctx, x, 0.1f, false);
    x = conv1d(ctx,
        w(dec, "generator.conv_post.weight"),
        w(dec, "generator.conv_post.bias"),
        x, 1, 3, 1); // k=7, p=3

    // Split into spec and phase
    int n_fft = dec->gen_istft_n_fft;     // 20
    int freq_bins = n_fft / 2 + 1;        // 11
    // x: [T, n_fft+2, 1] = [T, 22, 1]
    // spec = exp(x[:, :freq_bins, :])
    // phase = sin(x[:, freq_bins:, :])
    // In ggml layout (data[t + c*T]), channel c starts at byte c*T*4 = c*nb[1].
    // So the phase channels (c=11..21) start at offset freq_bins*nb[1].
    ggml_tensor * spec_view = ggml_view_3d(ctx, x,
        x->ne[0], freq_bins, 1, x->nb[1], x->nb[2], 0);
    ggml_tensor * phase_view = ggml_view_3d(ctx, x,
        x->ne[0], freq_bins, 1, x->nb[1], x->nb[2],
        freq_bins * x->nb[1]);

    ggml_tensor * spec = ggml_exp(ctx, spec_view);
    ggml_tensor * phase = ggml_sin(ctx, phase_view);

    // iSTFT:
    // real = spec * cos(phase)
    // imag = spec * sin(phase)
    // real_rec = conv_transpose1d(real, weight_backward_real, stride=hop, p=0)
    // imag_rec = conv_transpose1d(imag, weight_backward_imag, stride=hop, p=0)
    // waveform = real_rec - imag_rec
    // Then remove center padding (n_fft//2 from each side)
    ggml_tensor * real_part = ggml_mul(ctx, spec, ggml_cos(ctx, phase));
    ggml_tensor * imag_part = ggml_mul(ctx, spec, ggml_sin(ctx, phase));

    int hop = dec->gen_istft_hop_size;
    ggml_tensor * real_rec = ggml_conv_transpose_1d(ctx,
        w(dec, "generator.stft.weight_backward_real"), real_part, hop, 0, 1);
    ggml_tensor * imag_rec = ggml_conv_transpose_1d(ctx,
        w(dec, "generator.stft.weight_backward_imag"), imag_part, hop, 0, 1);

    ggml_tensor * waveform = ggml_sub(ctx, real_rec, imag_rec);

    // Remove center padding: crop n_fft//2 from each side of the time dim
    int pad = n_fft / 2;
    int64_t T_wav = waveform->ne[0];
    int64_t T_out = T_wav - 2 * pad;
    if (T_out > 0) {
        waveform = ggml_view_3d(ctx, waveform,
            T_out, waveform->ne[1], 1,
            waveform->nb[1], waveform->nb[2],
            pad * sizeof(float));
        waveform = ggml_cont(ctx, waveform);
    }

    return waveform;
}

// ---------------------------------------------------------------------------
// Performance profiling (verbose timing of each forward stage)
// ---------------------------------------------------------------------------
bool kokoro_decoder_profile = false;

// ---------------------------------------------------------------------------
// C API implementation
// ---------------------------------------------------------------------------

kokoro_decoder * kokoro_decoder_init_from_file(
        const char * path_model,
        struct kokoro_decoder_params params) {
    auto * dec = new kokoro_decoder();
    dec->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Load GGUF
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &dec->ctx_weights,
    };
    dec->gguf_ctx = gguf_init_from_file(path_model, gguf_params);
    if (!dec->gguf_ctx) {
        fprintf(stderr, "[kokoro_decoder] failed to load GGUF: %s\n", path_model);
        delete dec;
        return nullptr;
    }

    // Load hyperparameters from KV metadata
    auto get_u32 = [&](const char * key) -> uint32_t {
        int64_t id = gguf_find_key(dec->gguf_ctx, key);
        if (id < 0) { fprintf(stderr, "[kokoro_decoder] missing KV: %s\n", key); return 0; }
        return gguf_get_val_u32(dec->gguf_ctx, id);
    };
    auto get_arr_int = [&](const char * key, std::vector<int> & out) {
        int64_t id = gguf_find_key(dec->gguf_ctx, key);
        if (id < 0) return;
        size_t n = gguf_get_arr_n(dec->gguf_ctx, id);
        const int32_t * data = (const int32_t *) gguf_get_arr_data(dec->gguf_ctx, id);
        out.resize(n);
        for (size_t i = 0; i < n; i++) out[i] = data[i];
    };

    dec->style_dim              = get_u32("kokoro.style_dim");
    dec->upsample_initial_channel = get_u32("kokoro.upsample_initial_channel");
    dec->gen_istft_n_fft        = get_u32("kokoro.gen_istft_n_fft");
    dec->gen_istft_hop_size     = get_u32("kokoro.gen_istft_hop_size");
    dec->num_upsamples          = get_u32("kokoro.num_upsamples");
    dec->num_resblocks_per_up   = get_u32("kokoro.num_resblocks_per_up");

    get_arr_int("kokoro.upsample_rates", dec->upsample_rates);
    get_arr_int("kokoro.upsample_kernel_sizes", dec->upsample_kernel_sizes);
    get_arr_int("kokoro.resblock_kernel_sizes", dec->resblock_kernel_sizes);

    // resblock_dilation_sizes_flat: flattened [1,3,5, 1,3,5, 1,3,5]
    {
        int64_t id = gguf_find_key(dec->gguf_ctx, "kokoro.resblock_dilation_sizes_flat");
        if (id >= 0) {
            size_t n = gguf_get_arr_n(dec->gguf_ctx, id);
            const int32_t * data = (const int32_t *) gguf_get_arr_data(dec->gguf_ctx, id);
            dec->resblock_dilation_sizes.resize(n / 3);
            for (size_t i = 0; i < n / 3; i++) {
                dec->resblock_dilation_sizes[i] = {data[i*3], data[i*3+1], data[i*3+2]};
            }
        }
    }

    // Build weight name -> tensor map
    int64_t n_tensors = gguf_get_n_tensors(dec->gguf_ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(dec->gguf_ctx, i);
        ggml_tensor * t = ggml_get_tensor(dec->ctx_weights, name);
        if (t) {
            dec->weights[name] = t;
        }
    }

    // Conv1d weights are kept as F32. ggml_conv_1d now selects F32 im2col
    // output for F32 weights, enabling pure F32 GEMM with AVX2 SIMD.
    // (Previous F16 pre-conversion was removed because F16 GEMM was slower
    //  than F32 GEMM on this platform due to F16->F32 conversion overhead.)

    fprintf(stderr, "[kokoro_decoder] loaded %lld tensors from %s\n",
            (long long)n_tensors, path_model);
    fprintf(stderr, "[kokoro_decoder] config: n_fft=%d hop=%d upsample_rates=[%d,%d] threads=%d\n",
            dec->gen_istft_n_fft, dec->gen_istft_hop_size,
            dec->upsample_rates[0], dec->upsample_rates[1], dec->n_threads);

    return dec;
}

void kokoro_decoder_free(kokoro_decoder * dec) {
    if (!dec) return;
    if (dec->gguf_ctx) gguf_free(dec->gguf_ctx);
    if (dec->ctx_weights) ggml_free(dec->ctx_weights);
    if (dec->ctx_weights_f16) ggml_free(dec->ctx_weights_f16);
    delete dec;
}

std::vector<float> kokoro_decoder_forward(
        kokoro_decoder * dec,
        const float * asr,       int T_asr,
        const float * F0_pred,   int T_f0,
        const float * N_pred,
        const float * style_dec) {
    if (!dec) return {};

    // Sanity: F0_conv has stride 2, so F0_pred/N_pred must be 2x asr length.
    if (T_f0 != T_asr * 2) {
        fprintf(stderr, "[kokoro_decoder] T_f0 (%d) must equal 2*T_asr (%d)\n",
                T_f0, T_asr * 2);
        return {};
    }

    auto t_start = std::chrono::steady_clock::now();

    // ----- Precompute SineGen + STFT in C++ -----
    SineSource sine;
    precompute_sine_source(dec, F0_pred, T_f0, sine);

    if (sine.har.empty()) {
        fprintf(stderr, "[kokoro_decoder] sine source precomputation failed\n");
        return {};
    }

    auto t_sine = std::chrono::steady_clock::now();

    // ----- Build ggml graph -----
    // Context for graph intermediate tensors. Needs to be large enough for
    // all intermediate tensors + graph nodes. The decoder's largest layers
    // operate at the full 24 kHz audio rate (T ~= 21120 for 4.4s output)
    // with 128 channels and kernel sizes up to 11; ggml's Conv1d uses
    // im2col which materializes a [K_eff, IC, T] tensor per conv.
    // Peak observed: ~4.3 GB for a 4.4s utterance. Budget 8 GB (or 4 GB
    // per stage in profile mode to avoid allocating 12 GB simultaneously).
    size_t graph_mem = kokoro_decoder_profile
        ? size_t(4) * 1024 * 1024 * 1024
        : size_t(8) * 1024 * 1024 * 1024;
    struct ggml_init_params gparams = {
        /*.mem_size   = */ graph_mem,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "[kokoro_decoder] failed to init graph context (%zu MB)\n", graph_mem / (1024*1024));
        return {};
    }

    // Create input tensors.
    // asr is at T_asr resolution (already halved by the encoder).
    // F0_pred/N_pred are at T_f0 = 2*T_asr resolution (F0_conv halves them).
    //
    // The .npy file stores asr in row-major [T, C] (data[t*C + c]).
    // ggml's ne=[T, C, 1] expects data[t + c*T] (time varies fastest).
    // We must transpose during load.
    int asr_channels = 512; // hidden_dim
    ggml_tensor * asr_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_asr, asr_channels, 1);
    {
        float * dst = (float *) asr_t->data;
        for (int t = 0; t < T_asr; t++) {
            for (int c = 0; c < asr_channels; c++) {
                dst[t + c * T_asr] = asr[t * asr_channels + c];
            }
        }
    }
    ggml_set_input(asr_t);

    ggml_tensor * F0_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_f0, 1, 1);
    memcpy(F0_t->data, F0_pred, T_f0 * sizeof(float));
    ggml_set_input(F0_t);

    ggml_tensor * N_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_f0, 1, 1);
    memcpy(N_t->data, N_pred, T_f0 * sizeof(float));
    ggml_set_input(N_t);

    int style_dim = dec->style_dim; // 128
    ggml_tensor * style_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, style_dim, 1, 1);
    memcpy(style_t->data, style_dec, style_dim * sizeof(float));
    ggml_set_input(style_t);

    int har_channels = 2 * sine.freq_bins;
    ggml_tensor * har_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        sine.T_frames_stft, har_channels, 1);
    memcpy(har_t->data, sine.har.data(), sine.har.size() * sizeof(float));
    ggml_set_input(har_t);

    // Build the graph
    auto t_graph_build_start = std::chrono::steady_clock::now();

    ggml_tensor * output;
    ggml_cgraph * gf = nullptr;

    if (kokoro_decoder_profile) {
        // Staged execution: compute stage 1, materialize intermediate, then
        // compute stage 2. This adds a small overhead (memcpy of the
        // intermediate) but lets us measure per-stage compute time.
        ggml_tensor * x_intermediate = build_encoder_decode_graph(ctx, dec, asr_t, F0_t, N_t, style_t);
        if (!x_intermediate) {
            fprintf(stderr, "[kokoro_decoder] stage 1 graph build failed\n");
            ggml_free(ctx);
            return {};
        }
        ggml_set_output(x_intermediate);
        ggml_cgraph * gf1 = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf1, x_intermediate);
        int n_nodes_1 = ggml_graph_n_nodes(gf1);
        auto t_graph_build_end = std::chrono::steady_clock::now();

        auto t_compute1_start = std::chrono::steady_clock::now();
        enum ggml_status status1 = ggml_graph_compute_with_ctx(ctx, gf1, dec->n_threads);
        auto t_compute1_end = std::chrono::steady_clock::now();
        if (status1 != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[kokoro_decoder] stage 1 compute failed: %d\n", (int)status1);
            ggml_free(ctx);
            return {};
        }

        // Materialize intermediate: copy x_intermediate data into a new
        // input tensor in a fresh context, then build stage 2 from it.
        int64_t T_int = x_intermediate->ne[0];
        int64_t C_int = x_intermediate->ne[1];
        size_t int_bytes = ggml_nbytes(x_intermediate);

        // Stage 2 context. Stage 2 (generator) needs less memory than stage 1
        // since it operates at 2x time resolution but with fewer channels in
        // the resblocks (128 vs 1024). Budget 4 GB.
        size_t graph_mem2 = size_t(4) * 1024 * 1024 * 1024;
        struct ggml_init_params gparams2 = {
            /*.mem_size   = */ graph_mem2,
            /*.mem_buffer = */ nullptr,
            /*.no_alloc   = */ false,
        };
        ggml_context * ctx2 = ggml_init(gparams2);
        if (!ctx2) {
            fprintf(stderr, "[kokoro_decoder] failed to init stage 2 context\n");
            ggml_free(ctx);
            return {};
        }

        // Copy intermediate into a new input tensor
        ggml_tensor * x_in = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, T_int, C_int, 1);
        memcpy(x_in->data, x_intermediate->data, int_bytes);
        ggml_set_input(x_in);

        // Re-create style and har inputs in ctx2 (they're needed by stage 2)
        ggml_tensor * style_t2 = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, style_dim, 1, 1);
        memcpy(style_t2->data, style_dec, style_dim * sizeof(float));
        ggml_set_input(style_t2);

        int har_channels = 2 * sine.freq_bins;
        ggml_tensor * har_t2 = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32,
            sine.T_frames_stft, har_channels, 1);
        memcpy(har_t2->data, sine.har.data(), sine.har.size() * sizeof(float));
        ggml_set_input(har_t2);

        auto t_build2_start = std::chrono::steady_clock::now();
        output = build_generator_graph(ctx2, dec, x_in, style_t2, har_t2);
        if (!output) {
            fprintf(stderr, "[kokoro_decoder] stage 2 graph build failed\n");
            ggml_free(ctx);
            ggml_free(ctx2);
            return {};
        }
        ggml_set_output(output);
        ggml_cgraph * gf2 = ggml_new_graph_custom(ctx2, 8192, false);
        ggml_build_forward_expand(gf2, output);
        int n_nodes_2 = ggml_graph_n_nodes(gf2);
        auto t_build2_end = std::chrono::steady_clock::now();

        auto t_compute2_start = std::chrono::steady_clock::now();
        enum ggml_status status2 = ggml_graph_compute_with_ctx(ctx2, gf2, dec->n_threads);
        auto t_compute2_end = std::chrono::steady_clock::now();
        if (status2 != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[kokoro_decoder] stage 2 compute failed: %d\n", (int)status2);
            ggml_free(ctx);
            ggml_free(ctx2);
            return {};
        }

        double sine_ms      = std::chrono::duration<double, std::milli>(t_sine - t_start).count();
        double build1_ms    = std::chrono::duration<double, std::milli>(t_graph_build_end - t_graph_build_start).count();
        double compute1_ms  = std::chrono::duration<double, std::milli>(t_compute1_end - t_compute1_start).count();
        double build2_ms    = std::chrono::duration<double, std::milli>(t_build2_end - t_build2_start).count();
        double compute2_ms  = std::chrono::duration<double, std::milli>(t_compute2_end - t_compute2_start).count();
        double total_ms     = std::chrono::duration<double, std::milli>(t_compute2_end - t_start).count();

        int64_t T_out = output->ne[0];
        std::fprintf(stderr, "[kokoro_decoder] T_asr=%d T_f0=%d T_audio=%lld  stage1_nodes=%d  stage2_nodes=%d\n",
                     T_asr, T_f0, (long long)T_out, n_nodes_1, n_nodes_2);
        std::fprintf(stderr, "[kokoro_decoder] timing: sine=%.1fms  build1=%.1fms  compute1(enc/dec)=%.1fms  build2=%.1fms  compute2(gen)=%.1fms  total=%.1fms\n",
                     sine_ms, build1_ms, compute1_ms, build2_ms, compute2_ms, total_ms);

        // Extract output before freeing contexts
        std::vector<float> audio(T_out);
        memcpy(audio.data(), output->data, T_out * sizeof(float));
        ggml_free(ctx);
        ggml_free(ctx2);
        return audio;
    } else {
        // Combined execution (no staged timing)
        ggml_tensor * x_intermediate = build_encoder_decode_graph(ctx, dec, asr_t, F0_t, N_t, style_t);
        if (!x_intermediate) {
            fprintf(stderr, "[kokoro_decoder] stage 1 graph build failed\n");
            ggml_free(ctx);
            return {};
        }
        output = build_generator_graph(ctx, dec, x_intermediate, style_t, har_t);
        if (!output) {
            fprintf(stderr, "[kokoro_decoder] stage 2 graph build failed\n");
            ggml_free(ctx);
            return {};
        }
        ggml_set_output(output);
        auto t_graph_build_end = std::chrono::steady_clock::now();

        gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, output);

        auto t_compute_start = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, dec->n_threads);
        auto t_compute_end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[kokoro_decoder] graph compute failed: %d\n", (int)status);
            ggml_free(ctx);
            return {};
        }

        int64_t T_out = output->ne[0];
        std::vector<float> audio(T_out);
        memcpy(audio.data(), output->data, T_out * sizeof(float));

        double sine_ms     = std::chrono::duration<double, std::milli>(t_sine - t_start).count();
        double build_ms    = std::chrono::duration<double, std::milli>(t_graph_build_end - t_graph_build_start).count();
        double compute_ms  = std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
        double total_ms    = std::chrono::duration<double, std::milli>(t_compute_end - t_start).count();
        int n_nodes        = ggml_graph_n_nodes(gf);
        std::fprintf(stderr, "[kokoro_decoder] T_asr=%d T_f0=%d T_audio=%lld  nodes=%d\n",
                     T_asr, T_f0, (long long)T_out, n_nodes);
        std::fprintf(stderr, "[kokoro_decoder] timing: sine=%.1fms  graph_build=%.1fms  compute=%.1fms  total=%.1fms\n",
                     sine_ms, build_ms, compute_ms, total_ms);

        ggml_free(ctx);
        return audio;
    }
}

std::vector<float> kokoro_decoder_precompute_har(
    kokoro_decoder * dec,
    const float * F0_pred,   int T_f0,
    int * T_frames_stft_out,
    int * freq_bins_out) {
    if (!dec) return {};
    SineSource sine;
    precompute_sine_source(dec, F0_pred, T_f0, sine);
    if (sine.har.empty()) return {};
    if (T_frames_stft_out) *T_frames_stft_out = sine.T_frames_stft;
    if (freq_bins_out)     *freq_bins_out     = sine.freq_bins;
    return sine.har;
}
