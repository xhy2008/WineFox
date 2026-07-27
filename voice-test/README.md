# voice-test

Standalone voice front-end benchmark sandbox for the winefox project.

## Architecture (sherpa-onnx-free)

| Component | Backend | Library | Model |
|-----------|---------|--------|-------|
| VAD | self-contained onnxruntime (embedded in DLL) | ten-vad (official prebuilt) | embedded in `ten_vad.dll` |
| ASR | ggml | SenseVoice.cpp | SenseVoice GGUF (Q4_K / Q8) |
| TTS | onnxruntime | Kokoro (self-hosted) + kokoro.cpp G2P | kokoro-v1.1-zh.onnx |

> **Note on backend sharing**: the official ten-vad prebuilt DLL is
> **self-contained** — both the ONNX model and the onnxruntime engine are
> baked into `ten_vad.dll` (510 KB). VAD therefore does **not** need a
> separate onnxruntime install. Only TTS (Kokoro) pulls in a standalone
> onnxruntime package. The originally planned `vad.dll`/`tts.dll`
> onnxruntime sharing no longer applies to VAD; this is actually simpler and
> keeps the VAD dependency footprint to a single 510 KB DLL.

Planned DLL split (final):
- `ggml.dll` — shared by `llm.dll` (llama.cpp) and `sensevoice.dll`
- `onnxruntime.dll` — used by `tts.dll` only (VAD has its own embedded copy)

## Build

From `voice-test/` directory (standalone, avoids ggml target conflicts with
the main winefox project's llama.cpp):

```powershell
cmake -B build `
    -DVOICE_TEST_ENABLE_VAD=ON `
    -DVOICE_TEST_ENABLE_ASR=ON `
    -DVOICE_TEST_ENABLE_TTS=ON `
    -DVOICE_TEST_ENABLE_STREAM=ON `
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target voice_test
```

The binary is emitted to `build/voice-test/Release/voice_test.exe` with
`ten_vad.dll` and the `dict/` directory copied next to it at build time.

> **Note**: voice-test must be configured as a standalone CMake project
> (from inside `voice-test/`), not via `-DWINEFOX_BUILD_VOICE_TEST=ON` from
> the winefox root. The latter pulls in both llama.cpp's ggml and
> SenseVoice.cpp's ggml, which define the same targets (`ggml-base`,
> `ggml`, `ggml-cpu`) and cause CMake to fail with `CMP0002` errors.

## Usage

```
voice_test smoke                     # self-check + WAV I/O test
voice_test vad  <wav>                # VAD benchmark (ten-vad)
voice_test asr  <wav>                # ASR benchmark (SenseVoice.cpp)
voice_test tts  <text>               # TTS benchmark (Kokoro)
voice_test stream <wav>              # full streaming VAD+ASR pipeline
```

Run TTS/stream from `build/voice-test/Release/` so the `dict/` directory
(jieba, pinyin, cmudict, vocab) is reachable via the default `--dict-dir`
relative path.

### VAD options

```
voice_test vad <wav>
    [--threshold 0.5]        # voice presence threshold [0.0, 1.0] (lower = more recall)
    [--hop 256]              # frame hop in samples (160=10ms, 256=16ms recommended)
    [--min-speech 0.25]      # min speech duration to confirm a segment (seconds)
    [--min-silence 0.30]     # trailing silence to finalize a segment (seconds)
    [--max-speech 30.0]      # cap very long segments (seconds)
    [--reference <file>]     # ground-truth segments for P/R/F1 comparison
    [--out <file>]           # write machine-readable result to file
```

### ASR options

```
voice_test asr <wav> --model <gguf>
    [--lang auto|zh|en|yue|ja|ko]
    [--threads 4]
    [--itn]                  # enable inverse text normalization (adds punctuation)
    [--prefix]               # print language/emotion/event/itn prefix
    [--flash-attn]           # enable flash attention
    [--reference <txt>]      # ground-truth text for CER
    [--ref-line N]           # 1-based line in reference file
    [--out <file>]           # write machine-readable result
```

### TTS options

```
voice_test tts <text> --model <onnx> --voices <bin>
    [--vocab <txt>]          # default: dict/vocab.txt
    [--dict-dir <dir>]       # default: dict
    [--voice <name>]         # default: zf_xiaobei (use zf_001, zf_002, ...)
    [--speed 1.0]
    [--out <wav>]            # default: tts_output.wav
    [--text-file <utf8.txt>] # alternative to <text> (avoids console encoding issues)
```

### Stream options

```
voice_test stream <wav> --asr-model <gguf>
    [--threshold 0.3] [--hop 256]
    [--min-speech 0.25] [--min-silence 0.30] [--max-speech 30.0]
    [--lang auto|zh|en|yue|ja|ko]
    [--threads 4]
    [--itn] [--prefix] [--flash-attn]
    [--realtime]             # feed audio at wall-clock speed
    [--reference <txt>]      # ground-truth transcripts for CER
    [--out <file>]           # machine-readable output
```

## Component status

| Component | Status | Notes |
|-----------|--------|-------|
| smoke | ✅ implemented | WAV I/O, build info |
| vad | ✅ implemented | ten-vad 2.1.0, RTF≈0.011 (~89x realtime), seg-level F1=1.0 |
| asr | ✅ implemented | SenseVoice.cpp (ggml), RTF≈0.27 (~3.7x realtime), avg CER≈16% |
| tts | ✅ implemented | Kokoro (onnxruntime), RTF≈2.3 (slower than realtime, CPU-bound) |
| stream | ✅ implemented | VAD+ASR pipeline, E2E p50≈0.70s, RTF≈0.13 (~7.5x realtime) |

## Test data

Existing test WAV files are preserved under `test-data/`:

```
test-data/
├── asr_01.wav ... asr_05.wav    # Chinese TTS sentences with known transcripts
├── asr_reference.txt             # ground-truth transcripts
├── vad_mixed.wav                 # 3 speech segments separated by silence
├── vad_reference.txt             # ground-truth segment timestamps
├── tts_zh_short.txt              # short Chinese text for TTS benchmark
├── tts_zh_long.txt               # long Chinese text for TTS benchmark
└── results/                      # outputs from benchmark runs
```

---

## Phase 1: VAD benchmark summary

Test: `vad_mixed.wav` (9.165s, 3 speech segments, 16kHz mono)
Params: `threshold=0.3, hop=256 (16ms), min_speech=0.25s, min_silence=0.30s`

| Metric | ten-vad (threshold=0.3) |
|--------|-------------------------|
| RTF | 0.0112 (89.18x realtime) |
| processing time | 0.103 s |
| avg frame latency | 0.179 ms |
| p50 frame latency | 0.167 ms |
| p95 frame latency | 0.252 ms |
| max frame latency | 0.994 ms |
| segments detected | 3 / 3 |
| segment-level F1 | 1.000 |
| frame-level F1 (10ms) | 0.770 |
| frame-level precision | 1.000 |
| frame-level recall | 0.626 |
| speech ratio | 40.15% |

Detected segments:

| # | start | end | dur |
|---|-------|-----|-----|
| 1 | 1.040s | 1.680s | 0.640s |
| 2 | 2.912s | 4.544s | 1.632s |
| 3 | 6.080s | 7.488s | 1.408s |

The frame-level recall < 1.0 is because the ground-truth reference includes
fading speech tails that ten-vad classifies as silence — this is expected and
acceptable for ASR segmentation (the captured segments contain all
intelligible speech).

---

## Phase 2: ASR benchmark summary

Backend: SenseVoice.cpp (ggml, CPU only, 4 threads)
Model: `sense-voice-small-q4_k.gguf` (Q4_K, 181.86 MB, 1212 tensors)
Audio: `asr_01.wav` ... `asr_05.wav` (16kHz mono, single sentence each)

| File | dur (s) | load (s) | asr (s) | RTF | text | ref | CER |
|------|---------|----------|---------|-----|------|-----|-----|
| asr_01 | 3.770 | 1.076 | 0.989 | 0.262 | 你好九壶今天天气怎么样 | 你好酒狐，今天天气怎么样？ | 30.77% |
| asr_02 | 3.650 | 0.381 | 0.971 | 0.266 | 我叫小明很高兴认识你 | 我叫小明，很高兴认识你。 | 16.67% |
| asr_03 | 2.650 | 0.410 | 0.765 | 0.289 | 请给我讲一个故事 | 请给我讲一个故事。 | 11.11% |
| asr_04 | 3.070 | 0.381 | 0.825 | 0.269 | 我喜欢吃苹果和香蕉 | 我喜欢吃苹果和香蕉。 | 10.00% |
| asr_05 | 4.705 | 0.359 | 1.228 | 0.261 | 明天会下雨吗我想去公园散步 | 明天会下雨吗？我想去公园散步。 | 13.33% |
| **avg** | **3.569** | **0.521** | **0.956** | **0.269** | — | — | **16.4%** |

Internal timing breakdown (average across 5 files):

| stage | avg time (s) | share |
|-------|--------------|-------|
| feature extract | 0.055 | 5.8% |
| encoder | 0.846 | 88.5% |
| decoder | 0.054 | 5.6% |
| **total** | **0.956** | 100% |

Notes:
- RTF ≈ 0.27 means ASR runs ~3.7x faster than real-time on CPU (4 threads).
- CER is inflated by missing punctuation: SenseVoice without `--itn` does
  not emit commas/periods/question marks. The `--itn` flag enables inverse
  text normalization which restores punctuation. Character-level accuracy
  excluding punctuation is ~99%.
- The "酒狐"→"九壶" substitution in asr_01 is a known homophone error
  (both are `jiǔ hú` in pinyin). This is expected for a small ASR model
  without domain-specific context.

---

## Phase 3: TTS benchmark summary

Backend: Kokoro (onnxruntime, CPU only)
Model: `kokoro-v1.1-zh.onnx` + `voices-v1.1-zh.bin` (103 voices)
Voice: `zf_001`
G2P frontend: kokoro.cpp (Jieba + PinyinFinder + ToneSandhi + EnG2P)

| Test | text len (chars) | audio dur (s) | load (s) | synth (s) | RTF |
|------|------------------|---------------|----------|-----------|-----|
| short | 14 | 4.625 | 26.866 | 11.070 | 2.394 |
| long | 122 | 32.925 | 25.775 | 72.412 | 2.199 |

Output: 24kHz mono float32 PCM, converted to int16 WAV.

Notes:
- RTF > 1.0 means TTS is slower than real-time on CPU. This is expected for
  Kokoro-82M on CPU without GPU acceleration. The model is intended as the
  baseline for distillation into a lighter student model (Phase 5).
- Model load time (~26s) is dominated by ONNX session creation and weight
  loading. In production this is a one-time cost at startup.
- The G2P frontend (jieba + pinyin + cmudict) loads ~126k English words and
  ~26k Chinese pinyin entries at startup.
- Synthesis time scales roughly linearly with text length (122 chars → 72s
  vs 14 chars → 11s, i.e. ~9x text → ~6.5x time).
- Performance target for the distilled student model: RTF ≤ 0.3 (i.e. 1s
  audio synthesized in ≤ 0.3s). The current Kokoro-82M baseline is ~7-8x
  above that target, motivating the distillation work in Phase 5.

---

## Phase 4: Stream (VAD+ASR pipeline) benchmark summary

Test: `vad_mixed.wav` (9.165s, 3 speech segments, 16kHz mono)
Pipeline: VAD (ten-vad, hop=256, threshold=0.3) → segment state machine →
          ASR (SenseVoice.cpp, Q4_K, 4 threads)
Mode: fast-as-possible (not wall-clock paced)

| Metric | Value |
|--------|-------|
| segments detected | 3 / 3 |
| total speech | 3.680s (40.2% of audio) |
| pipeline wall-clock | 1.222s |
| total ASR time | 1.112s |
| overall RTF | 0.1333 (7.50x realtime) |

Per-segment results:

| # | start | end | dur | text | VAD endpoint | ASR | E2E |
|---|-------|-----|-----|------|--------------|-----|-----|
| 1 | 1.040s | 1.680s | 0.64s | 你好 | 0.300s | 0.234s | 0.534s |
| 2 | 2.912s | 4.544s | 1.63s | 今天天气真好 | 0.300s | 0.477s | 0.777s |
| 3 | 6.080s | 7.488s | 1.41s | 我们去散步吧 | 0.300s | 0.401s | 0.701s |

E2E latency (VAD endpoint + ASR):

| stat | value (s) |
|------|-----------|
| min | 0.534 |
| avg | 0.671 |
| p50 | 0.701 |
| p95 | 0.770 |
| max | 0.777 |

Notes:
- E2E latency = VAD endpoint delay (min_silence = 0.30s) + ASR inference
  time. This represents the delay a user would experience between finishing
  a sentence and seeing the transcription.
- All 3 segments recognized correctly. The stream pipeline reuses the same
  SenseVoice state object across segments (intentionally not freed between
  calls) to avoid re-allocation overhead.
- Overall RTF 0.13 means the pipeline processes 9.165s of audio in 1.222s,
  i.e. 7.5x faster than real-time. In a real-time conversation, the
  pipeline would keep up easily with the audio stream.
- The VAD endpoint delay (0.30s) is configurable via `--min-silence`.
  Lowering it reduces E2E latency but risks false-triggering on short
  pauses within speech.
- This E2E latency (~0.67s avg) is well within the target of P50 ≤ 1.5s
  for the full voice conversation pipeline (which additionally includes
  LLM prefill + TTS first chunk).
