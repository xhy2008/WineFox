// voice-test/src/asr_test.cpp
//
// ASR benchmark subcommand. Wraps SenseVoice.cpp (ggml-based ASR) and
// reports the recognized text plus RTF and timing breakdown.
//
// SenseVoice.cpp's CLI (`sense-voice-main`) uses its own silero-vad to chop
// the audio into segments and runs ASR on each segment. For our benchmark
// we run ASR on the *whole* WAV as a single segment, which is what the
// `asr_01..05.wav` test files were designed for (each is one short sentence).
//
// Metrics reported:
//   - Recognized text (+ optional language/emotion/event/itn prefix)
//   - Model load time
//   - Total ASR time, RTF (real-time factor)
//   - SenseVoice internal timings: feature extraction / encode / decode
//   - Optional CER (character error rate) vs. ground-truth text
//
// Usage:
//   voice_test asr <wav> --model <gguf>
//                       [--lang auto|zh|en|yue|ja|ko]
//                       [--threads N]
//                       [--itn]            # enable inverse text normalization
//                       [--prefix]         # print language/emotion/event/itn prefix
//                       [--gpu]            # enable GPU backend
//                       [--flash-attn]     # enable flash attention
//                       [--beam N]         # use beam search with size N
//                       [--reference <txt>]# ground-truth text for CER
//                       [--ref-line N]     # 1-based line in reference file
//                       [--out <file>]     # write machine-readable result

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "sense-voice.h"
#include "silero-vad.h"  // full definition of silero_vad for proper cleanup

#include "common.h"  // voice-test common (Pcm, load_wav, kSampleRate)

namespace {

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct AsrArgs {
    std::string wav_path;
    std::string model_path;
    std::string language      = "auto";
    std::string reference_path;
    std::string out_path;
    int  threads     = 4;
    int  ref_line    = 0;   // 1-based line number in reference file (0 = use whole file)
    bool use_itn     = false;
    bool use_prefix  = false;
    bool use_gpu     = false;
    bool flash_attn  = false;
    bool beam_search = false;
    int  beam_size   = -1;
};

bool parse_args(const std::vector<std::string>& args, AsrArgs& out) {
    if (args.empty()) {
        std::fprintf(stderr, "asr: missing <wav> argument\n");
        return false;
    }
    out.wav_path = args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "asr: %s requires a value\n", name);
                return std::string();
            }
            return args[++i];
        };
        if      (a == "--model")     out.model_path     = next("--model");
        else if (a == "--lang")      out.language       = next("--lang");
        else if (a == "--threads")   out.threads        = std::stoi(next("--threads"));
        else if (a == "--reference") out.reference_path = next("--reference");
        else if (a == "--ref-line")  out.ref_line       = std::stoi(next("--ref-line"));
        else if (a == "--out")       out.out_path       = next("--out");
        else if (a == "--itn")       out.use_itn        = true;
        else if (a == "--prefix")    out.use_prefix     = true;
        else if (a == "--gpu")       out.use_gpu        = true;
        else if (a == "--flash-attn")out.flash_attn     = true;
        else if (a == "--beam") {
            out.beam_search = true;
            out.beam_size   = std::stoi(next("--beam"));
        }
        else {
            std::fprintf(stderr, "asr: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (out.model_path.empty()) {
        std::fprintf(stderr, "asr: --model <gguf> is required\n");
        return false;
    }
    if (out.threads <= 0) {
        std::fprintf(stderr, "asr: --threads must be positive\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Result extraction
//
// SenseVoice emits 4 prefix tokens before the actual transcript:
//   [0] language  (e.g. <|zh|>)
//   [1] emotion   (e.g. <|NEUTRAL|>)
//   [2] event     (e.g. <|Speech|>)
//   [3] itn       (e.g. <|withitn|> or <|noitn|>)
// When need_prefix is false we skip these and only emit the transcript.
//
// Duplicate consecutive tokens are collapsed (matches the behavior of
// sense_voice_print_output). The padding token (id==0) is skipped.
// ---------------------------------------------------------------------------
std::string get_text(struct sense_voice_context* ctx, bool need_prefix) {
    std::string text;
    const auto& ids = ctx->state->ids;
    size_t start = need_prefix ? 0 : 4;
    int prev_id = -1;
    for (size_t i = start; i < ids.size(); ++i) {
        int id = ids[i];
        if (i > 0 && id == ids[i - 1]) continue;  // collapse duplicates
        if (id == 0) continue;                     // skip padding
        auto it = ctx->vocab.id_to_token.find(id);
        if (it != ctx->vocab.id_to_token.end()) {
            text += it->second;
        }
        prev_id = id;
    }
    return text;
}

struct PrefixInfo {
    std::string language;
    std::string emotion;
    std::string event;
    std::string itn;
};

PrefixInfo get_prefix(struct sense_voice_context* ctx) {
    PrefixInfo p;
    const auto& ids = ctx->state->ids;
    auto get = [&](size_t i) -> std::string {
        if (i >= ids.size()) return "";
        int id = ids[i];
        if (id == 0) return "";
        auto it = ctx->vocab.id_to_token.find(id);
        return it != ctx->vocab.id_to_token.end() ? it->second : "";
    };
    p.language = get(0);
    p.emotion  = get(1);
    p.event    = get(2);
    p.itn      = get(3);
    return p;
}

// ---------------------------------------------------------------------------
// Reference text loading + CER (character error rate)
//
// The reference file is plain text (one or more lines). Lines starting with
// '#' are treated as comments. CER is computed as the Levenshtein distance
// per UTF-8 byte divided by the reference length in bytes. For Chinese text
// (3 bytes per character in UTF-8) this is equivalent to character-level CER
// when both strings are pure Chinese; mixed ASCII/CJK is still a reasonable
// relative metric for benchmark comparison.
// ---------------------------------------------------------------------------
std::string load_reference_text(const std::string& path, int ref_line) {
    std::ifstream f(path);
    if (!f) return "";
    std::string line, all;
    int line_no = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        line_no++;
        // Strip UTF-8 BOM if present.
        if (line.size() >= 3 &&
            (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB &&
            (unsigned char)line[2] == 0xBF) {
            line = line.substr(3);
        }
        if (ref_line > 0) {
            if (line_no == ref_line) return line;
        } else {
            all += line;
        }
    }
    return ref_line > 0 ? std::string() : all;
}

double cer(const std::string& ref, const std::string& hyp) {
    if (ref.empty()) return 0.0;
    size_t m = ref.size(), n = hyp.size();
    std::vector<size_t> prev(n + 1), cur(n + 1);
    for (size_t j = 0; j <= n; ++j) prev[j] = j;
    for (size_t i = 1; i <= m; ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= n; ++j) {
            size_t cost = (ref[size_t(i - 1)] == hyp[size_t(j - 1)]) ? 0 : 1;
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, cur);
    }
    return double(prev[n]) / double(m);
}

}  // namespace

int run_asr(const std::vector<std::string>& args) {
    AsrArgs aa;
    if (!parse_args(args, aa)) return 1;

    Pcm pcm;
    if (!load_wav(aa.wav_path, pcm)) {
        std::fprintf(stderr, "asr: failed to load %s\n", aa.wav_path.c_str());
        return 1;
    }
    if (pcm.sample_rate != kSampleRate) {
        std::fprintf(stderr,
                     "asr: %s has sample_rate=%d, expected %d. Resample first.\n",
                     aa.wav_path.c_str(), pcm.sample_rate, kSampleRate);
        return 1;
    }

    const double audio_dur_s = double(pcm.num_samples) / kSampleRate;

    std::printf("voice_test asr\n");
    std::printf("  backend    : SenseVoice.cpp (ggml)\n");
    std::printf("  input      : %s\n", aa.wav_path.c_str());
    std::printf("  model      : %s\n", aa.model_path.c_str());
    std::printf("  language   : %s\n", aa.language.c_str());
    std::printf("  threads    : %d\n", aa.threads);
    std::printf("  use_gpu    : %d\n", aa.use_gpu ? 1 : 0);
    std::printf("  flash_attn : %d\n", aa.flash_attn ? 1 : 0);
    std::printf("  use_itn    : %d\n", aa.use_itn ? 1 : 0);
    std::printf("  use_prefix : %d\n", aa.use_prefix ? 1 : 0);
    std::printf("  sample_rate: %d\n", pcm.sample_rate);
    std::printf("  num_samples: %lld (%.3fs)\n",
                (long long)pcm.num_samples, audio_dur_s);
    std::printf("\n");

    // -------------------------------------------------------------------
    // Initialize SenseVoice context.
    // -------------------------------------------------------------------
    sense_voice_context_params cparams = sense_voice_context_default_params();
    cparams.use_gpu    = aa.use_gpu;
    cparams.flash_attn = aa.flash_attn;
    cparams.use_itn    = aa.use_itn;

    auto t_load_start = std::chrono::steady_clock::now();
    sense_voice_context* ctx = sense_voice_small_init_from_file_with_params(
        aa.model_path.c_str(), cparams);
    auto t_load_end = std::chrono::steady_clock::now();
    if (!ctx) {
        std::fprintf(stderr, "asr: failed to initialize sense_voice context\n");
        return 1;
    }
    const double load_ms = std::chrono::duration<double, std::milli>(
        t_load_end - t_load_start).count();

    // Set language on the context (also set on params below).
    if (aa.language != "auto" && aa.language != "") {
        int lid = sense_voice_lang_id(aa.language.c_str());
        if (lid < 0) {
            std::fprintf(stderr, "asr: unknown language '%s'\n", aa.language.c_str());
            sense_voice_free_state(ctx->state);
            delete ctx->model.model->encoder;
            delete ctx->model.model;
            delete ctx->vad_model.model;
            delete ctx;
            return 1;
        }
        ctx->language_id = lid;
    }

    // -------------------------------------------------------------------
    // Convert int16 PCM to double float (SenseVoice expects std::vector<double>).
    // -------------------------------------------------------------------
    std::vector<double> pcmf32(size_t(pcm.num_samples));
    for (int64_t i = 0; i < pcm.num_samples; ++i) {
        pcmf32[size_t(i)] = double(pcm.samples[size_t(i)]) / 32768.0;
    }

    // -------------------------------------------------------------------
    // Configure decoding params and run ASR.
    // -------------------------------------------------------------------
    sense_voice_full_params wparams = sense_voice_full_default_params(
        SENSE_VOICE_SAMPLING_GREEDY);
    if (aa.beam_search) {
        wparams.strategy = SENSE_VOICE_SAMPLING_BEAM_SEARCH;
        if (aa.beam_size > 0) wparams.beam_search.beam_size = aa.beam_size;
    }
    wparams.language       = aa.language.c_str();
    wparams.n_threads      = aa.threads;
    wparams.print_progress = false;
    wparams.no_timestamps  = true;
    wparams.single_segment = true;

    auto t_asr_start = std::chrono::steady_clock::now();
    int rc = sense_voice_full_parallel(ctx, wparams, pcmf32,
                                       int(pcmf32.size()), 1);
    auto t_asr_end = std::chrono::steady_clock::now();
    if (rc != 0) {
        std::fprintf(stderr,
                     "asr: sense_voice_full_parallel failed (rc=%d)\n", rc);
        sense_voice_free_state(ctx->state);
        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx->vad_model.model;
        delete ctx;
        return 1;
    }

    const double asr_ms = std::chrono::duration<double, std::milli>(
        t_asr_end - t_asr_start).count();
    const double asr_s  = asr_ms / 1000.0;
    const double rtf    = asr_s / audio_dur_s;

    // -------------------------------------------------------------------
    // Extract results.
    //   The transcript always strips the 4 SenseVoice prefix tokens
    //   (language/emotion/event/itn). When --prefix is set, the prefix
    //   info is printed separately via get_prefix().
    // -------------------------------------------------------------------
    std::string text = get_text(ctx, false);
    PrefixInfo pinfo = get_prefix(ctx);

    // SenseVoice's own internal timings (microseconds -> seconds).
    const double t_feat_s = ctx->state->t_feature_us / 1e6;
    const double t_enc_s  = ctx->state->t_encode_us  / 1e6;
    const double t_dec_s  = ctx->state->t_decode_us  / 1e6;
    const double t_int_s  = t_feat_s + t_enc_s + t_dec_s;

    std::printf("Result:\n");
    std::printf("  text       : %s\n", text.c_str());
    if (aa.use_prefix) {
        std::printf("  language   : %s\n", pinfo.language.c_str());
        std::printf("  emotion    : %s\n", pinfo.emotion.c_str());
        std::printf("  event      : %s\n", pinfo.event.c_str());
        std::printf("  itn        : %s\n", pinfo.itn.c_str());
    }
    std::printf("  tokens     : %zu\n", ctx->state->ids.size());
    std::printf("\n");

    std::printf("Performance:\n");
    std::printf("  audio duration   : %7.3f s\n", audio_dur_s);
    std::printf("  model load       : %7.3f s\n", load_ms / 1000.0);
    std::printf("  total ASR time   : %7.3f s\n", asr_s);
    std::printf("  RTF              : %7.4f  (1 / %.2fx realtime)\n",
                rtf, rtf > 0 ? 1.0 / rtf : 0.0);
    std::printf("  feature extract  : %7.3f s\n", t_feat_s);
    std::printf("  encoder          : %7.3f s\n", t_enc_s);
    std::printf("  decoder          : %7.3f s\n", t_dec_s);
    std::printf("  internal total   : %7.3f s\n", t_int_s);
    std::printf("\n");

    // -------------------------------------------------------------------
    // Optional: compare with reference text.
    // -------------------------------------------------------------------
    int exit_rc = 0;
    if (!aa.reference_path.empty()) {
        std::string ref = load_reference_text(aa.reference_path, aa.ref_line);
        if (ref.empty()) {
            std::fprintf(stderr, "asr: could not read reference %s\n",
                         aa.reference_path.c_str());
            exit_rc = 1;
        } else {
            double c = cer(ref, text);
            double acc = 1.0 - c;
            std::printf("Comparison vs reference (%s):\n",
                        aa.reference_path.c_str());
            std::printf("  reference : %s\n", ref.c_str());
            std::printf("  predicted : %s\n", text.c_str());
            std::printf("  CER       : %.4f  (%.2f%%)\n", c, c * 100.0);
            std::printf("  accuracy  : %.4f  (%.2f%%)\n", acc, acc * 100.0);
            std::printf("\n");
        }
    }

    // -------------------------------------------------------------------
    // Optional: write machine-readable output.
    // -------------------------------------------------------------------
    if (!aa.out_path.empty()) {
        std::ofstream of(aa.out_path);
        if (!of) {
            std::fprintf(stderr, "asr: could not write %s\n",
                         aa.out_path.c_str());
            exit_rc = 1;
        } else {
            of << "# voice_test asr output\n";
            of << "# input=" << aa.wav_path
               << "  model=" << aa.model_path
               << "  lang=" << aa.language
               << "  threads=" << aa.threads << "\n";
            of << "# dur=" << audio_dur_s
               << "s  load_ms=" << load_ms
               << "  asr_ms=" << asr_ms
               << "  rtf=" << rtf
               << "  feat_s=" << t_feat_s
               << "  enc_s=" << t_enc_s
               << "  dec_s=" << t_dec_s << "\n";
            of << "# text:\n";
            of << text << "\n";
            std::printf("Wrote %s\n", aa.out_path.c_str());
        }
    }

    // -------------------------------------------------------------------
    // Cleanup.
    //   main.cc just leaks on exit; we free state + sub-objects explicitly
    //   so the benchmark doesn't accumulate memory across multiple runs
    //   in the same process.
    // -------------------------------------------------------------------
    sense_voice_free_state(ctx->state);
    delete ctx->model.model->encoder;
    delete ctx->model.model;
    delete ctx->vad_model.model;
    delete ctx;

    std::printf("asr: done.\n");
    return exit_rc;
}
