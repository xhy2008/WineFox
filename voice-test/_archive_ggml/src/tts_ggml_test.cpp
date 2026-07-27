// voice-test/src/tts_ggml_test.cpp
//
// ggml-based Kokoro decoder benchmark.
//
// Loads precomputed encoder outputs (asr, F0_pred, N_pred, style_dec) from
// .npy files produced by scripts/dump_decoder_inputs.py, runs the ggml
// decoder forward pass, and reports:
//   - RTF (real-time factor) over N iterations
//   - Audio length / MSE / max abs diff / cosine similarity vs. reference
//   - Optional WAV output for manual listening verification
//
// The reference audio comes from the original PyTorch Kokoro decoder run
// (test-data/decoder_inputs/reference.wav). Because SineGen injects
// Gaussian noise, the ggml output will NOT match sample-by-sample; we
// expect RMS-aligned envelopes and a non-trivial cosine similarity, but
// the real correctness check is listening to the output WAV.
//
// Usage:
//   voice_test tts-ggml --model <kokoro-decoder.gguf>
//                       --inputs <npy_dir>          # contains asr.npy, F0_pred.npy, ...
//                       [--reference <ref.wav>]
//                       [--threads N]               # default: 4
//                       [--iters N]                 # default: 5
//                       [--out <wav>]               # write synthesized audio

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common.h"
#include "kokoro_decoder.h"

namespace {

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------
struct TtsGgmlArgs {
    std::string model_path;
    std::string inputs_dir;       // directory containing asr.npy, F0_pred.npy, ...
    std::string reference_path;   // optional reference WAV
    std::string out_path;         // optional output WAV
    int  threads = 4;
    int  iters   = 5;
    bool profile = false;
};

bool parse_args(const std::vector<std::string>& args, TtsGgmlArgs& out) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "tts-ggml: %s requires a value\n", name);
                return std::string();
            }
            return args[++i];
        };
        if (a == "--model")     { out.model_path     = next("--model");     }
        else if (a == "--inputs"){ out.inputs_dir    = next("--inputs");    }
        else if (a == "--reference") { out.reference_path = next("--reference"); }
        else if (a == "--out")  { out.out_path       = next("--out");       }
        else if (a == "--threads") {
            std::string v = next("--threads");
            if (v.empty()) return false;
            out.threads = std::atoi(v.c_str());
        }
        else if (a == "--iters") {
            std::string v = next("--iters");
            if (v.empty()) return false;
            out.iters = std::atoi(v.c_str());
        }
        else if (a == "--profile") {
            out.profile = true;
        }
        else {
            std::fprintf(stderr, "tts-ggml: unknown arg: %s\n", a.c_str());
            return false;
        }
    }
    if (out.model_path.empty()) {
        std::fprintf(stderr, "tts-ggml: --model is required\n");
        return false;
    }
    if (out.inputs_dir.empty()) {
        std::fprintf(stderr, "tts-ggml: --inputs is required\n");
        return false;
    }
    if (out.threads < 1) out.threads = 1;
    if (out.iters < 1)   out.iters   = 1;
    return true;
}

// ---------------------------------------------------------------------------
// Minimal NumPy .npy reader (v1.0, little-endian float32 only).
//
// Format:
//   6 bytes magic: \x93NUMPY
//   1 byte major version (1)
//   1 byte minor version (0)
//   2 bytes header length (LE)
//   N bytes ASCII header dict
//   raw data (row-major float32)
//
// We extract 'shape' from the header to support 1D and 2D arrays.
// ---------------------------------------------------------------------------
struct NpyArray {
    std::vector<int64_t> shape;  // e.g. [176, 512] or [352]
    std::vector<float>   data;   // row-major float32
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

bool load_npy(const std::string& path, NpyArray& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "load_npy: cannot open %s\n", path.c_str());
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    if (buf.size() < 10) {
        std::fprintf(stderr, "load_npy: %s too small (%zu bytes)\n", path.c_str(), buf.size());
        return false;
    }
    // Magic
    static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (std::memcmp(buf.data(), magic, 6) != 0) {
        std::fprintf(stderr, "load_npy: %s bad magic\n", path.c_str());
        return false;
    }
    uint8_t major = buf[6];
    uint8_t minor = buf[7];
    size_t header_len_off = 8;
    size_t header_len_bytes = 2;
    if (major == 2) {
        // v2.0: 4-byte header length
        header_len_bytes = 4;
    } else if (major != 1) {
        std::fprintf(stderr, "load_npy: %s unsupported version %u.%u\n",
                     path.c_str(), major, minor);
        return false;
    }
    if (buf.size() < header_len_off + header_len_bytes) return false;
    uint32_t header_len = 0;
    for (size_t i = 0; i < header_len_bytes; ++i) {
        header_len |= uint32_t(buf[header_len_off + i]) << (8 * i);
    }
    size_t header_off = header_len_off + header_len_bytes;
    if (buf.size() < header_off + header_len) return false;
    std::string header((const char*)buf.data() + header_off, header_len);

    // Parse shape from "shape': (...)" substring.
    // Header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (176, 512), }
    out.shape.clear();
    size_t pos = header.find("shape");
    if (pos == std::string::npos) {
        std::fprintf(stderr, "load_npy: %s missing 'shape' in header\n", path.c_str());
        return false;
    }
    size_t lp = header.find('(', pos);
    size_t rp = header.find(')', pos);
    if (lp == std::string::npos || rp == std::string::npos || rp <= lp) {
        std::fprintf(stderr, "load_npy: %s malformed shape\n", path.c_str());
        return false;
    }
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    // Split by comma
    size_t s = 0;
    while (s < shape_str.size()) {
        while (s < shape_str.size() && (shape_str[s] == ' ' || shape_str[s] == ',')) s++;
        if (s >= shape_str.size()) break;
        size_t e = s;
        while (e < shape_str.size() && shape_str[e] != ',' && shape_str[e] != ' ') e++;
        std::string num = shape_str.substr(s, e - s);
        if (!num.empty()) {
            out.shape.push_back(std::strtoll(num.c_str(), nullptr, 10));
        }
        s = e;
    }

    // Verify dtype is '<f4' (little-endian float32)
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\":\"<f4\"") == std::string::npos) {
        std::fprintf(stderr, "load_npy: %s unsupported dtype (only '<f4' supported)\n",
                     path.c_str());
        return false;
    }

    // Read raw float32 data
    size_t data_off = header_off + header_len;
    int64_t n = out.numel();
    if (buf.size() < data_off + size_t(n) * sizeof(float)) {
        std::fprintf(stderr, "load_npy: %s truncated: have %zu bytes, need %zu\n",
                     path.c_str(), buf.size() - data_off, size_t(n) * sizeof(float));
        return false;
    }
    out.data.resize(n);
    std::memcpy(out.data.data(), buf.data() + data_off, n * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
struct AudioMetrics {
    int64_t n_samples = 0;
    float   mse       = 0.0f;
    float   max_abs   = 0.0f;
    float   rms_a     = 0.0f;   // reference
    float   rms_b     = 0.0f;   // ggml output
    float   cos_sim   = 0.0f;
};

AudioMetrics compute_metrics(const std::vector<float>& a,
                             const std::vector<float>& b) {
    AudioMetrics m;
    m.n_samples = std::min<int64_t>(a.size(), b.size());
    if (m.n_samples == 0) return m;
    double sum_sq_err = 0.0;
    double sum_sq_a   = 0.0;
    double sum_sq_b   = 0.0;
    double dot        = 0.0;
    for (int64_t i = 0; i < m.n_samples; ++i) {
        float av = a[i];
        float bv = b[i];
        float diff = av - bv;
        sum_sq_err += double(diff) * diff;
        sum_sq_a   += double(av) * av;
        sum_sq_b   += double(bv) * bv;
        dot        += double(av) * bv;
        float ad = std::fabs(diff);
        if (ad > m.max_abs) m.max_abs = ad;
    }
    m.mse   = float(sum_sq_err / m.n_samples);
    m.rms_a = float(std::sqrt(sum_sq_a / m.n_samples));
    m.rms_b = float(std::sqrt(sum_sq_b / m.n_samples));
    double denom = std::sqrt(sum_sq_a) * std::sqrt(sum_sq_b);
    m.cos_sim = denom > 0.0 ? float(dot / denom) : 0.0f;
    return m;
}

// Convert float32 [-1, 1] audio -> 16-bit PCM mono WAV via shared write_wav.
bool write_audio_wav(const std::string& path,
                     const std::vector<float>& audio_f32,
                     int sample_rate) {
    Pcm pcm;
    pcm.sample_rate = sample_rate;
    pcm.num_samples = int64_t(audio_f32.size());
    pcm.samples.resize(audio_f32.size());
    for (size_t i = 0; i < audio_f32.size(); ++i) {
        float v = std::max(-1.0f, std::min(1.0f, audio_f32[i]));
        pcm.samples[i] = int16_t(v * 32767.0f);
    }
    return write_wav(path, pcm);
}

// Load a 16-bit mono WAV into a float32 [-1, 1] vector for metric comparison.
bool load_reference_wav(const std::string& path, std::vector<float>& out) {
    Pcm pcm;
    if (!load_wav(path, pcm)) return false;
    out.resize(pcm.samples.size());
    for (size_t i = 0; i < pcm.samples.size(); ++i) {
        out[i] = float(pcm.samples[i]) / 32768.0f;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Subcommand entry point
// ---------------------------------------------------------------------------
int run_tts_ggml(const std::vector<std::string>& args) {
    TtsGgmlArgs a;
    if (!parse_args(args, a)) return 1;

    std::printf("=== voice_test tts-ggml ===\n");
    std::printf("model     : %s\n", a.model_path.c_str());
    std::printf("inputs    : %s\n", a.inputs_dir.c_str());
    std::printf("reference : %s\n",
                a.reference_path.empty() ? "(none)" : a.reference_path.c_str());
    std::printf("out       : %s\n",
                a.out_path.empty() ? "(none)" : a.out_path.c_str());
    std::printf("threads   : %d\n", a.threads);
    std::printf("iters     : %d\n", a.iters);
    std::printf("profile   : %s\n", a.profile ? "on" : "off");
    std::printf("\n");

    // ----- Load .npy inputs -----
    std::printf("[1/4] Loading encoder inputs from .npy files...\n");
    auto npy_path = [&](const char* name) {
        return a.inputs_dir + "\\" + std::string(name) + ".npy";
    };

    NpyArray asr_npy, f0_npy, n_npy, style_npy;
    if (!load_npy(npy_path("asr"), asr_npy))       return 1;
    if (!load_npy(npy_path("F0_pred"), f0_npy))    return 1;
    if (!load_npy(npy_path("N_pred"), n_npy))      return 1;
    if (!load_npy(npy_path("style_dec"), style_npy)) return 1;

    if (asr_npy.shape.size() != 2 || asr_npy.shape[1] != 512) {
        std::fprintf(stderr, "asr.npy must be 2D [T, 512], got [");
        for (size_t i = 0; i < asr_npy.shape.size(); ++i)
            std::fprintf(stderr, "%s%lld", i ? "," : "", (long long)asr_npy.shape[i]);
        std::fprintf(stderr, "]\n");
        return 1;
    }
    int T_asr = int(asr_npy.shape[0]);
    int T_f0  = int(f0_npy.shape.empty() ? 0 : f0_npy.shape[0]);
    int T_n   = int(n_npy.shape.empty() ? 0 : n_npy.shape[0]);
    int style_dim = int(style_npy.shape.empty() ? 0 : style_npy.shape[0]);
    if (T_f0 != 2 * T_asr) {
        std::fprintf(stderr, "T_f0 (%d) must equal 2*T_asr (%d)\n", T_f0, 2 * T_asr);
        return 1;
    }
    if (T_n != T_f0) {
        std::fprintf(stderr, "N_pred length (%d) must equal F0_pred length (%d)\n", T_n, T_f0);
        return 1;
    }
    if (style_dim != 128) {
        std::fprintf(stderr, "style_dec length must be 128, got %d\n", style_dim);
        return 1;
    }

    std::printf("  asr       : [%d, 512]   (%lld floats)\n",
                T_asr, (long long)asr_npy.numel());
    std::printf("  F0_pred   : [%d]\n", T_f0);
    std::printf("  N_pred    : [%d]\n", T_n);
    std::printf("  style_dec : [%d]\n", style_dim);
    std::printf("\n");

    // ----- Load reference audio (optional) -----
    std::vector<float> ref_audio;
    if (!a.reference_path.empty()) {
        std::printf("[2/4] Loading reference audio: %s\n", a.reference_path.c_str());
        if (!load_reference_wav(a.reference_path, ref_audio)) {
            std::fprintf(stderr, "warning: failed to load reference WAV; skipping comparison\n");
        } else {
            std::printf("  reference: %lld samples (%.3fs @ 24kHz)\n",
                        (long long)ref_audio.size(),
                        ref_audio.size() / 24000.0f);
        }
    } else {
        std::printf("[2/4] No reference audio provided; skipping comparison.\n");
    }
    std::printf("\n");

    // ----- Init ggml decoder -----
    std::printf("[3/4] Loading ggml decoder...\n");
    auto t0 = std::chrono::steady_clock::now();
    kokoro_decoder_params params;
    params.n_threads = a.threads;
    kokoro_decoder * dec = kokoro_decoder_init_from_file(a.model_path.c_str(), params);
    auto t1 = std::chrono::steady_clock::now();
    if (!dec) {
        std::fprintf(stderr, "Failed to init kokoro_decoder from %s\n", a.model_path.c_str());
        return 1;
    }
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  loaded in %.1f ms\n", load_ms);
    std::printf("\n");

    // ----- Warmup + timed iterations -----
    std::printf("[4/4] Running %d iteration(s)...\n", a.iters);
    if (a.profile) kokoro_decoder_profile = true;
    std::vector<float> audio_out;
    std::vector<double> iter_ms;
    std::vector<double> rtf_history;

    for (int it = 0; it < a.iters; ++it) {
        auto s0 = std::chrono::steady_clock::now();
        audio_out = kokoro_decoder_forward(dec,
                                           asr_npy.data.data(), T_asr,
                                           f0_npy.data.data(), T_f0,
                                           n_npy.data.data(),
                                           style_npy.data.data());
        auto s1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
        iter_ms.push_back(ms);

        if (audio_out.empty()) {
            std::fprintf(stderr, "iteration %d: decoder returned empty audio\n", it);
            kokoro_decoder_free(dec);
            return 1;
        }
        double audio_dur_s = audio_out.size() / 24000.0;
        double rtf = (ms / 1000.0) / audio_dur_s;
        rtf_history.push_back(rtf);
        std::printf("  iter %d: %.1f ms  audio=%.3fs  rtf=%.3f\n",
                    it + 1, ms, audio_dur_s, rtf);
    }

    // ----- Aggregate timing -----
    double min_ms = *std::min_element(iter_ms.begin(), iter_ms.end());
    double max_ms = *std::max_element(iter_ms.begin(), iter_ms.end());
    double avg_ms = 0.0;
    for (double v : iter_ms) avg_ms += v;
    avg_ms /= iter_ms.size();
    double min_rtf = *std::min_element(rtf_history.begin(), rtf_history.end());
    double avg_rtf = 0.0;
    for (double v : rtf_history) avg_rtf += v;
    avg_rtf /= rtf_history.size();

    std::printf("\n");
    std::printf("--- timing summary (ggml, threads=%d) ---\n", a.threads);
    std::printf("  decoder time: min=%.1f ms  avg=%.1f ms  max=%.1f ms\n",
                min_ms, avg_ms, max_ms);
    std::printf("  audio length: %.3f s (%lld samples @ 24kHz)\n",
                audio_out.size() / 24000.0, (long long)audio_out.size());
    std::printf("  RTF         : min=%.3f  avg=%.3f\n", min_rtf, avg_rtf);
    std::printf("\n");

    // ----- Compare with reference -----
    if (!ref_audio.empty()) {
        std::printf("--- audio comparison vs reference ---\n");
        AudioMetrics m = compute_metrics(ref_audio, audio_out);
        std::printf("  samples     : ref=%lld  ggml=%lld  (compared %lld)\n",
                    (long long)ref_audio.size(),
                    (long long)audio_out.size(),
                    (long long)m.n_samples);
        std::printf("  rms         : ref=%.4f  ggml=%.4f  ratio=%.3f\n",
                    m.rms_a, m.rms_b,
                    m.rms_a > 0.0f ? m.rms_b / m.rms_a : 0.0f);
        std::printf("  MSE         : %.6f\n", m.mse);
        std::printf("  max|diff|   : %.4f\n", m.max_abs);
        std::printf("  cosine sim  : %.4f\n", m.cos_sim);
        std::printf("\n");
        std::printf("Note: SineGen injects Gaussian noise, so sample-exact match is\n");
        std::printf("      impossible. Cosine sim > 0.5 + audible intelligibility in\n");
        std::printf("      the output WAV indicate a correct port.\n");
        std::printf("\n");
    }

    // ----- Diagnostic: compare precomputed har against Python reference -----
    // har.npy and har_source.npy are produced by dump_decoder_inputs.py and
    // live alongside the other .npy inputs. If we find them, compare.
    {
        std::string har_path = npy_path("har");
        NpyArray har_ref;
        if (load_npy(har_path, har_ref)) {
            std::printf("--- har (SineGen + STFT) comparison vs Python ---\n");
            // har_ref shape: [T_frames, 2*freq_bins] in row-major (t, c) order
            int T_ref = int(har_ref.shape.size() >= 1 ? har_ref.shape[0] : 0);
            int C_ref = int(har_ref.shape.size() >= 2 ? har_ref.shape[1] : 1);
            std::printf("  Python har : [%d, %d]\n", T_ref, C_ref);

            int T_cpp = 0, C_cpp = 0;
            std::vector<float> har_cpp = kokoro_decoder_precompute_har(
                dec, f0_npy.data.data(), T_f0, &T_cpp, &C_cpp);
            // C_cpp is freq_bins; total channels = 2*freq_bins
            int C_cpp_total = C_cpp * 2;
            std::printf("  C++    har : [%d, %d]  (freq_bins=%d)\n", T_cpp, C_cpp_total, C_cpp);

            if (har_cpp.empty()) {
                std::printf("  ERROR: precompute_har returned empty\n\n");
            } else if (T_ref != T_cpp || C_ref != C_cpp_total) {
                std::printf("  SHAPE MISMATCH: Python [%d,%d] vs C++ [%d,%d]\n\n",
                            T_ref, C_ref, T_cpp, C_cpp_total);
            } else {
                // C++ har_cpp is laid out as data[t + c*T_cpp] (ggml layout).
                // Python har_ref is laid out as data[t*C_ref + c] (row-major).
                // Build a comparable view of the C++ data in row-major order.
                std::vector<float> har_cpp_rowmajor(T_cpp * C_cpp_total);
                for (int t = 0; t < T_cpp; ++t) {
                    for (int c = 0; c < C_cpp_total; ++c) {
                        har_cpp_rowmajor[t * C_cpp_total + c] = har_cpp[t + c * T_cpp];
                    }
                }
                AudioMetrics hm = compute_metrics(har_ref.data, har_cpp_rowmajor);
                std::printf("  samples     : %lld\n", (long long)hm.n_samples);
                std::printf("  rms         : py=%.6f  cpp=%.6f  ratio=%.3f\n",
                            hm.rms_a, hm.rms_b,
                            hm.rms_a > 0.0f ? hm.rms_b / hm.rms_a : 0.0f);
                std::printf("  MSE         : %.6f\n", hm.mse);
                std::printf("  max|diff|   : %.6f\n", hm.max_abs);
                std::printf("  cosine sim  : %.6f\n", hm.cos_sim);
                // Per-channel breakdown for first few channels (mag + phase)
                int half_c = C_cpp_total / 2;
                std::printf("  per-channel cosine sim (first 4 mag, first 4 phase):\n");
                int show = std::min(4, half_c);
                for (int c = 0; c < show; ++c) {
                    std::vector<float> a_ch(T_ref), b_ch(T_ref);
                    for (int t = 0; t < T_ref; ++t) {
                        a_ch[t] = har_ref.data[t * C_ref + c];
                        b_ch[t] = har_cpp_rowmajor[t * C_ref + c];
                    }
                    AudioMetrics ch_m = compute_metrics(a_ch, b_ch);
                    std::printf("    mag[%d]: cos=%.4f  rms_py=%.4f  rms_cpp=%.4f\n",
                                c, ch_m.cos_sim, ch_m.rms_a, ch_m.rms_b);
                }
                for (int c = 0; c < show; ++c) {
                    int idx = half_c + c;
                    std::vector<float> a_ch(T_ref), b_ch(T_ref);
                    for (int t = 0; t < T_ref; ++t) {
                        a_ch[t] = har_ref.data[t * C_ref + idx];
                        b_ch[t] = har_cpp_rowmajor[t * C_ref + idx];
                    }
                    AudioMetrics ch_m = compute_metrics(a_ch, b_ch);
                    std::printf("    phs[%d]: cos=%.4f  rms_py=%.4f  rms_cpp=%.4f\n",
                                c, ch_m.cos_sim, ch_m.rms_a, ch_m.rms_b);
                }
                std::printf("\n");
            }
        }
    }

    // ----- Write output WAV -----
    if (!a.out_path.empty()) {
        if (write_audio_wav(a.out_path, audio_out, 24000)) {
            std::printf("Wrote synthesized audio: %s (%.3fs)\n",
                        a.out_path.c_str(), audio_out.size() / 24000.0);
        } else {
            std::fprintf(stderr, "Failed to write output WAV: %s\n", a.out_path.c_str());
        }
    }

    kokoro_decoder_free(dec);
    return 0;
}
