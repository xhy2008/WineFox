// voice-test/src/smoke.cpp
//
// Self-check subcommand. Verifies WAV I/O and prints build info.
// Also serves as the canonical home for load_wav/write_wav so all
// components share a single implementation.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common.h"

namespace {

// Read a little-endian uint32 from a byte buffer.
uint32_t read_u32_le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint16_t read_u16_le(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

void write_u32_le(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v & 0xff);
    p[1] = uint8_t((v >> 8) & 0xff);
    p[2] = uint8_t((v >> 16) & 0xff);
    p[3] = uint8_t((v >> 24) & 0xff);
}

void write_u16_le(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v & 0xff);
    p[1] = uint8_t((v >> 8) & 0xff);
}

}  // namespace

bool load_wav(const std::string& path, Pcm& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    if (buf.size() < 44) return false;

    // RIFF header
    if (std::memcmp(buf.data(), "RIFF", 4) != 0) return false;
    if (std::memcmp(buf.data() + 8, "WAVE", 4) != 0) return false;

    // Walk chunks.
    size_t off = 12;
    int sample_rate = 0, bits = 0, channels = 0;
    const uint8_t* data_ptr = nullptr;
    uint32_t data_len = 0;
    while (off + 8 <= buf.size()) {
        char id[5] = {0};
        std::memcpy(id, buf.data() + off, 4);
        uint32_t sz = read_u32_le(buf.data() + off + 4);
        const uint8_t* body = buf.data() + off + 8;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (sz < 16 || off + 8 + 16 > buf.size()) return false;
            uint16_t audio_fmt = read_u16_le(body);
            channels           = read_u16_le(body + 2);
            sample_rate        = int(read_u32_le(body + 4));
            bits               = read_u16_le(body + 14);
            if (audio_fmt != 1 || channels != 1 || bits != 16) {
                std::fprintf(stderr,
                             "load_wav: unsupported format (fmt=%u ch=%u bits=%u, "
                             "expected PCM mono 16-bit)\n",
                             audio_fmt, channels, bits);
                return false;
            }
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_ptr = body;
            data_len = sz;
            break;
        }
        off += 8 + sz + (sz & 1);  // chunks are word-aligned
    }
    if (!data_ptr || sample_rate == 0) return false;
    if (off + 8 + data_len > buf.size()) data_len = uint32_t(buf.size() - off - 8);

    out.sample_rate = sample_rate;
    out.num_samples = int64_t(data_len) / 2;
    out.samples.resize(size_t(out.num_samples));
    for (int64_t i = 0; i < out.num_samples; ++i) {
        out.samples[size_t(i)] = int16_t(read_u16_le(data_ptr + i * 2));
    }
    return true;
}

bool write_wav(const std::string& path, const Pcm& pcm) {
    if (pcm.sample_rate <= 0) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t data_len = uint32_t(pcm.samples.size() * 2);
    uint8_t hdr[44];
    std::memcpy(hdr, "RIFF", 4);
    write_u32_le(hdr + 4, 36 + data_len);
    std::memcpy(hdr + 8, "WAVE", 4);
    std::memcpy(hdr + 12, "fmt ", 4);
    write_u32_le(hdr + 16, 16);
    write_u16_le(hdr + 20, 1);  // PCM
    write_u16_le(hdr + 22, 1);  // mono
    write_u32_le(hdr + 24, uint32_t(pcm.sample_rate));
    write_u32_le(hdr + 28, uint32_t(pcm.sample_rate) * 2);  // byte rate
    write_u16_le(hdr + 32, 2);  // block align
    write_u16_le(hdr + 34, 16); // bits per sample
    std::memcpy(hdr + 36, "data", 4);
    write_u32_le(hdr + 40, data_len);
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(pcm.samples.data()),
            std::streamsize(data_len));
    return bool(f);
}

int run_smoke(const std::vector<std::string>& args) {
    std::printf("voice_test smoke - self-check\n");
    std::printf("\n");
    std::printf("Build configuration:\n");
    std::printf("  Sample rate (canonical) : %d Hz\n", kSampleRate);
    std::printf("  VAD hop size            : %d samples (%.2f ms)\n",
                kHopSize, double(kHopSize) / kSampleRate * 1000.0);
#if defined(VOICE_TEST_HAS_VAD)
    std::printf("  VAD    : ENABLED  (ten-vad-ggml)\n");
#else
    std::printf("  VAD    : disabled (build with -DVOICE_TEST_ENABLE_VAD=ON)\n");
#endif
#if defined(VOICE_TEST_HAS_ASR)
    std::printf("  ASR    : ENABLED  (SenseVoice.cpp + ggml)\n");
#else
    std::printf("  ASR    : disabled (build with -DVOICE_TEST_ENABLE_ASR=ON)\n");
#endif
#if defined(VOICE_TEST_HAS_TTS)
    std::printf("  TTS    : ENABLED  (Kokoro + onnxruntime)\n");
#else
    std::printf("  TTS    : disabled (build with -DVOICE_TEST_ENABLE_TTS=ON)\n");
#endif
#if defined(VOICE_TEST_HAS_STREAM)
    std::printf("  STREAM : ENABLED  (VAD + ASR pipeline)\n");
#else
    std::printf("  STREAM : disabled (build with -DVOICE_TEST_ENABLE_STREAM=ON)\n");
#endif
    std::printf("\n");

    // WAV I/O round-trip test: synthesize 1s of 440Hz sine, write, reload.
    Pcm pcm;
    pcm.sample_rate = kSampleRate;
    pcm.num_samples = kSampleRate;
    pcm.samples.resize(kSampleRate);
    for (int i = 0; i < kSampleRate; ++i) {
        double t = double(i) / kSampleRate;
        pcm.samples[i] = int16_t(0.2 * 32767.0 * std::sin(2 * 3.14159265358979 * 440.0 * t));
    }
    const std::string tmp = "smoke_test.wav";
    if (!write_wav(tmp, pcm)) {
        std::fprintf(stderr, "FAIL: write_wav(%s) failed\n", tmp.c_str());
        return 1;
    }
    Pcm reloaded;
    if (!load_wav(tmp, reloaded)) {
        std::fprintf(stderr, "FAIL: load_wav(%s) failed\n", tmp.c_str());
        return 1;
    }
    if (reloaded.sample_rate != kSampleRate ||
        reloaded.num_samples != kSampleRate ||
        reloaded.samples.size() != size_t(kSampleRate)) {
        std::fprintf(stderr,
                     "FAIL: round-trip mismatch (sr=%d n=%lld size=%zu)\n",
                     reloaded.sample_rate,
                     (long long)reloaded.num_samples,
                     reloaded.samples.size());
        return 1;
    }
    // Verify first sample matches within 1 LSB.
    int16_t a = pcm.samples[100], b = reloaded.samples[100];
    if (std::abs(int(a) - int(b)) > 1) {
        std::fprintf(stderr, "FAIL: sample[100] mismatch (%d vs %d)\n", a, b);
        return 1;
    }
    std::printf("WAV I/O round-trip: OK (1s 440Hz sine, %d samples)\n", kSampleRate);
    std::remove(tmp.c_str());

    // If a wav file was passed as an argument, load it and print stats.
    if (!args.empty()) {
        Pcm in;
        if (!load_wav(args[0], in)) {
            std::fprintf(stderr, "FAIL: could not load %s\n", args[0].c_str());
            return 1;
        }
        std::printf("\nLoaded %s:\n", args[0].c_str());
        std::printf("  sample_rate = %d\n", in.sample_rate);
        std::printf("  num_samples = %lld (%.2fs)\n",
                    (long long)in.num_samples,
                    double(in.num_samples) / in.sample_rate);
    }

    std::printf("\nsmoke: PASS\n");
    return 0;
}
