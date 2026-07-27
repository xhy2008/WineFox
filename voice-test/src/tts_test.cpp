// voice-test/src/tts_test.cpp
//
// TTS benchmark subcommand. Wraps Kokoro (ONNX Runtime) and reports
// synthesis time, audio duration, RTF, and writes a WAV file.
//
// Two modes:
//   1. Split mode (recommended): --encoder + --decoder + --voices
//      Encoder FP32, decoder INT8-static, multi-threaded.
//   2. Merged mode (legacy): --model + --voices
//      Single ONNX, single-threaded.
//
// Streaming mode: --stream emits audio per sentence batch via callback,
// reporting per-chunk TTFB and total RTF.
//
// Usage:
//   voice_test tts <text> --encoder <enc.onnx> --decoder <dec.onnx> --voices <bin>
//                       [--vocab <txt>] [--dict-dir <dir>] [--voice <name>]
//                       [--speed 1.0] [--threads N] [--out <wav>] [--stream]
//                       [--text-file <utf8.txt>]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "Kokoro.h"

#include "common.h"  // voice-test common (Pcm, write_wav)

namespace {

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct TtsArgs {
    std::string text;
    std::string text_file;
    // Split mode
    std::string encoder_path;
    std::string decoder_path;
    int         threads = 0;       // 0 = auto
    // Merged mode
    std::string model_path;
    // Shared
    std::string voices_path;
    std::string vocab_path  = "dict/vocab.txt";
    std::string dict_dir    = "dict";
    std::string voice_name  = "zf_001";
    std::string out_path    = "tts_output.wav";
    float       speed       = 1.0f;
    bool        stream      = false;
};

bool parse_args(const std::vector<std::string>& args, TtsArgs& out) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "tts: %s requires a value\n", name);
                return std::string();
            }
            return args[++i];
        };
        if      (a == "--model")     out.model_path   = next("--model");
        else if (a == "--encoder")   out.encoder_path = next("--encoder");
        else if (a == "--decoder")   out.decoder_path = next("--decoder");
        else if (a == "--voices")    out.voices_path  = next("--voices");
        else if (a == "--vocab")     out.vocab_path   = next("--vocab");
        else if (a == "--dict-dir")  out.dict_dir     = next("--dict-dir");
        else if (a == "--voice")     out.voice_name   = next("--voice");
        else if (a == "--out")       out.out_path     = next("--out");
        else if (a == "--speed")     out.speed        = std::stof(next("--speed"));
        else if (a == "--threads")   out.threads      = std::stoi(next("--threads"));
        else if (a == "--stream")    out.stream       = true;
        else if (a == "--text-file") out.text_file    = next("--text-file");
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "tts: unknown option %s\n", a.c_str());
            return false;
        } else {
            if (out.text.empty()) out.text = a;
        }
    }
    if (!out.text_file.empty()) {
        std::ifstream f(out.text_file, std::ios::binary);
        if (!f.is_open()) {
            std::fprintf(stderr, "tts: cannot open --text-file %s\n", out.text_file.c_str());
            return false;
        }
        std::string s((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
        out.text = s;
    }
    if (out.text.empty()) {
        std::fprintf(stderr, "tts: missing <text> argument or --text-file\n");
        return false;
    }
    // Mode determination: split if encoder+decoder given, else merged.
    if (out.encoder_path.empty() && out.decoder_path.empty()) {
        if (out.model_path.empty()) {
            std::fprintf(stderr, "tts: either --model (merged) or --encoder+--decoder (split) is required\n");
            return false;
        }
    } else if (out.encoder_path.empty() || out.decoder_path.empty()) {
        std::fprintf(stderr, "tts: --encoder and --decoder must be given together\n");
        return false;
    }
    if (out.voices_path.empty()) {
        std::fprintf(stderr, "tts: --voices <bin> is required\n");
        return false;
    }
    return true;
}

std::vector<int16_t> float_to_int16(const std::vector<float>& f) {
    std::vector<int16_t> out;
    out.reserve(f.size());
    for (float v : f) {
        float clamped = std::max(-1.0f, std::min(1.0f, v));
        out.push_back(static_cast<int16_t>(clamped * 32767.0f));
    }
    return out;
}

}  // namespace

int run_tts(const std::vector<std::string>& args) {
    TtsArgs ta;
    if (!parse_args(args, ta)) return 1;

    const bool split_mode = !ta.encoder_path.empty();

    std::printf("voice_test tts\n");
    std::printf("  backend    : Kokoro (onnxruntime)\n");
    std::printf("  mode       : %s\n", split_mode ? "split (enc+dec)" : "merged");
    if (split_mode) {
        std::printf("  encoder    : %s\n", ta.encoder_path.c_str());
        std::printf("  decoder    : %s\n", ta.decoder_path.c_str());
        std::printf("  threads    : %d%s\n", ta.threads, ta.threads == 0 ? " (auto)" : "");
    } else {
        std::printf("  model      : %s\n", ta.model_path.c_str());
    }
    std::printf("  voices     : %s\n", ta.voices_path.c_str());
    std::printf("  vocab      : %s\n", ta.vocab_path.c_str());
    std::printf("  dict_dir   : %s\n", ta.dict_dir.c_str());
    std::printf("  voice      : %s\n", ta.voice_name.c_str());
    std::printf("  speed      : %.2f\n", ta.speed);
    std::printf("  stream     : %s\n", ta.stream ? "yes" : "no");
    std::printf("  output     : %s\n", ta.out_path.c_str());
    std::printf("  text       : %s\n", ta.text.c_str());
    std::printf("\n");

    // -------------------------------------------------------------------
    // Initialize Kokoro TTS engine.
    // -------------------------------------------------------------------
    auto t_load_start = std::chrono::steady_clock::now();
    std::unique_ptr<Kokoro> kokoro;
    if (split_mode) {
        kokoro = std::make_unique<Kokoro>(
            ta.encoder_path, ta.decoder_path, ta.voices_path,
            ta.vocab_path, ta.threads);
    } else {
        kokoro = std::make_unique<Kokoro>(
            ta.model_path, ta.voices_path, ta.vocab_path);
    }
    auto t_load_end = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(
        t_load_end - t_load_start).count();

    std::printf("Model loaded in %.3f s\n", load_ms / 1000.0);
    std::printf("\n");

    // -------------------------------------------------------------------
    // Synthesize.
    // -------------------------------------------------------------------
    std::vector<float> audio_f32;
    int sample_rate = SAMPLE_RATE;

    if (ta.stream) {
        // Streaming mode: emit audio per sentence batch, measure TTFB.
        std::printf("Streaming synthesis:\n");
        auto t_first_start = std::chrono::steady_clock::now();
        double ttfb_ms = -1.0;
        int chunk_idx = 0;
        size_t total_samples = 0;

        kokoro->create_stream(
            ta.text, ta.voice_name,
            [&](const std::vector<float>& chunk, int sr) -> bool {
                if (ttfb_ms < 0) {
                    auto now = std::chrono::steady_clock::now();
                    ttfb_ms = std::chrono::duration<double, std::milli>(
                        now - t_first_start).count();
                }
                double chunk_dur = double(chunk.size()) / double(sr);
                std::printf("  chunk %d: %zu samples (%.3f s)\n",
                            chunk_idx++, chunk.size(), chunk_dur);
                audio_f32.insert(audio_f32.end(), chunk.begin(), chunk.end());
                total_samples += chunk.size();
                return true;
            },
            ta.speed);

        sample_rate = SAMPLE_RATE;
        std::printf("  TTFB        : %.1f ms\n", ttfb_ms);
        std::printf("  total chunks: %d\n", chunk_idx);
    } else {
        auto t_synth_start = std::chrono::steady_clock::now();
        auto [a, sr] = kokoro->create(ta.text, ta.voice_name, ta.speed);
        auto t_synth_end = std::chrono::steady_clock::now();
        audio_f32 = std::move(a);
        sample_rate = sr;

        const double synth_ms = std::chrono::duration<double, std::milli>(
            t_synth_end - t_synth_start).count();
        const double synth_s  = synth_ms / 1000.0;
        const double audio_dur_s = double(audio_f32.size()) / double(sample_rate);
        const double rtf = audio_dur_s > 0 ? synth_s / audio_dur_s : 0.0;

        std::printf("Result:\n");
        std::printf("  samples       : %zu\n", audio_f32.size());
        std::printf("  sample_rate   : %d Hz\n", sample_rate);
        std::printf("  audio duration: %.3f s\n", audio_dur_s);
        std::printf("\n");
        std::printf("Performance:\n");
        std::printf("  model load    : %7.3f s\n", load_ms / 1000.0);
        std::printf("  synthesis     : %7.3f s\n", synth_s);
        std::printf("  RTF           : %7.4f  (1 / %.2fx realtime)\n",
                    rtf, rtf > 0 ? 1.0 / rtf : 0.0);
    }
    std::printf("\n");

    // -------------------------------------------------------------------
    // Convert to int16 and write WAV.
    // -------------------------------------------------------------------
    if (!ta.out_path.empty() && !audio_f32.empty()) {
        Pcm pcm;
        pcm.sample_rate = sample_rate;
        pcm.samples = float_to_int16(audio_f32);
        pcm.num_samples = static_cast<int64_t>(pcm.samples.size());

        if (!write_wav(ta.out_path, pcm)) {
            std::fprintf(stderr, "tts: failed to write %s\n", ta.out_path.c_str());
            return 1;
        }
        double audio_dur_s = double(audio_f32.size()) / double(sample_rate);
        std::printf("Wrote %s (%d Hz, %lld samples, %.3f s)\n",
                    ta.out_path.c_str(), pcm.sample_rate,
                    (long long)pcm.num_samples, audio_dur_s);
    }

    std::printf("tts: done.\n");
    return 0;
}
