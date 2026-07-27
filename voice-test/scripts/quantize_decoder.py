"""Quantize Kokoro decoder ONNX to INT8 and benchmark speed.

Strategies (in order of aggressiveness):
  1. Dynamic quantization (weight-only, no calibration needed)
     - Fastest to apply, modest speedup (~1.5-2x)
     - May lose some accuracy
  2. Static quantization (weight + activation, needs calibration data)
     - Better speedup (~2-3x)
     - Requires representative input data
  3. Pruning (structured, 30-50% sparsity)
     - Done in PyTorch, then re-export
     - onnxruntime can leverage sparse weights

We start with (1) dynamic quantization since it's a one-liner and gives
immediate feedback on whether INT8 helps the decoder bottleneck.
"""
import os
import sys
import time
import types
import importlib.util
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

# ---------------------------------------------------------------------------
# Load kokoro.model (same stub trick as test_split_tts.py)
# ---------------------------------------------------------------------------
KOKORO_SRC = r"e:\winefox\voice-test\third_party\kokoro-src"
sys.path.insert(0, KOKORO_SRC)

pkg = types.ModuleType("kokoro")
pkg.__path__ = [os.path.join(KOKORO_SRC, "kokoro")]
sys.modules["kokoro"] = pkg

spec = importlib.util.spec_from_file_location(
    "kokoro.model", os.path.join(KOKORO_SRC, "kokoro", "model.py")
)
_kokoro_model = importlib.util.module_from_spec(spec)
sys.modules["kokoro.model"] = _kokoro_model
spec.loader.exec_module(_kokoro_model)
KModel = _kokoro_model.KModel

# Stub misaki to avoid en/espeak dependency chain
import misaki
if not hasattr(misaki, 'en'):
    en_stub = types.ModuleType("misaki.en")
    en_stub.MToken = object
    sys.modules["misaki.en"] = en_stub
    misaki.en = en_stub
if not hasattr(misaki, 'espeak'):
    espeak_stub = types.ModuleType("misaki.espeak")
    espeak_stub.EspeakWrapper = object
    espeak_stub.EspeakBackend = object
    sys.modules["misaki.espeak"] = espeak_stub
    misaki.espeak = espeak_stub

from kokoro.pipeline import KPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = r"e:\winefox\voice-test\models\kokoro-82M-src\config.json"
CHECKPOINT = r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth"
DEC_ONNX = r"e:\winefox\voice-test\models\kokoro-decoder.onnx"
DEC_QUANT = r"e:\winefox\voice-test\models\kokoro-decoder-int8.onnx"
ENC_ONNX = r"e:\winefox\voice-test\models\kokoro-encoder.onnx"
OUT_DIR = r"e:\winefox\voice-test\test-data\results\quant_tts"
SAMPLE_RATE = 24000

TEST_SENTENCES = [
    ("short", "你好，我是酒狐。", "zf_xiaobei"),
    ("medium", "今天天气真好，我们去公园散步吧。", "zf_xiaobei"),
    ("long", "今天天气真好，我们去公园散步吧。听说那里的樱花开得很美，再不去看就要凋谢了。", "zf_xiaobei"),
]


def normalize_audio(audio):
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        return audio / max_val
    return audio


def save_wav(path, audio_f32):
    import wave
    audio_clamped = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_clamped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio_i16.tobytes())


def quantize_decoder():
    """Apply dynamic INT8 quantization to decoder ONNX."""
    print(f"\n{'='*70}")
    print(f"[1] Dynamic quantization of decoder")
    print(f"{'='*70}")
    print(f"  input:  {DEC_ONNX} ({os.path.getsize(DEC_ONNX)/1024/1024:.2f} MB)")

    # Use QOperator format for best compatibility with Conv ops.
    # QDQ format would insert quantize/dequantize nodes which may slow
    # down Conv-heavy graphs. QOperator replaces Conv with QLinearConv.
    quantize_dynamic(
        model_input=DEC_ONNX,
        model_output=DEC_QUANT,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm"],
        per_channel=True,  # per-output-channel quant for Conv weights
        reduce_range=False,
    )
    print(f"  output: {DEC_QUANT} ({os.path.getsize(DEC_QUANT)/1024/1024:.2f} MB)")
    print(f"  size reduction: {(1 - os.path.getsize(DEC_QUANT)/os.path.getsize(DEC_ONNX))*100:.1f}%")
    return DEC_QUANT


def get_real_inputs(pipeline, text, voice, speed=1.0):
    """Run encoder on real text to get decoder inputs for benchmarking."""
    results = []
    for graphemes, phonemes, _ in pipeline(text, voice=voice, speed=speed, model=False):
        if not phonemes:
            continue
        input_ids = list(filter(
            lambda i: i is not None,
            map(lambda p: pipeline.model.vocab.get(p), phonemes)
        ))
        if len(input_ids) + 2 > pipeline.model.context_length:
            input_ids = input_ids[:510]
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).numpy()

        pack = pipeline.load_voice(voice).cpu()
        ref_s = pack[len(phonemes) - 1]
        while ref_s.dim() > 2:
            ref_s = ref_s.squeeze(0)
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)
        ref_s = ref_s.numpy()

        speed_arr = np.array([speed], dtype=np.float32)

        results.append((input_ids, ref_s, speed_arr))
    return results


def benchmark_decoder(enc_sess, dec_sessions, decoder_inputs, labels):
    """Benchmark each decoder variant on the same inputs.

    dec_sessions: list of (name, session)
    decoder_inputs: list of (input_ids, ref_s, speed) tuples
    labels: list of names for each input
    """
    print(f"\n{'='*70}")
    print(f"[2] Decoder benchmark (CPU, single thread)")
    print(f"{'='*70}")
    print(f"{'Test':<10}{'Audio dur':<12}", end="")
    for name, _ in dec_sessions:
        print(f"{name + ' (ms)':<18}", end="")
    print(f"{'speedup':<10}")
    print("-" * 80)

    all_results = []
    for (input_ids, ref_s, speed_arr), label in zip(decoder_inputs, labels):
        # Run encoder once (shared across all decoder variants)
        enc_out = enc_sess.run(None, {
            "input_ids": input_ids,
            "ref_s": ref_s,
            "speed": speed_arr,
        })
        asr, F0_pred, N_pred, style_dec, _ = enc_out
        audio_dur_s = 0
        dec_inputs = {
            "asr": asr, "F0_pred": F0_pred,
            "N_pred": N_pred, "style_dec": style_dec,
        }

        # Warm up each session once
        for _, sess in dec_sessions:
            sess.run(None, dec_inputs)

        # Measure each decoder
        dec_times = []
        for name, sess in dec_sessions:
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                out = sess.run(None, dec_inputs)
                times.append((time.perf_counter() - t0) * 1000)
            med_ms = np.median(times)
            dec_times.append(med_ms)
            if audio_dur_s == 0:
                audio_dur_s = len(out[0].squeeze()) / SAMPLE_RATE

        # Also get audio from fp32 baseline for quality comparison
        audio_ref = dec_sessions[0][1].run(None, dec_inputs)[0].squeeze()
        speedup = dec_times[0] / dec_times[1] if len(dec_times) > 1 else 1.0
        print(f"{label:<10}{audio_dur_s:<12.3f}", end="")
        for t in dec_times:
            print(f"{t:<18.1f}", end="")
        print(f"{speedup:<10.2f}x")
        all_results.append({
            "label": label, "audio_dur": audio_dur_s,
            "dec_times": dec_times, "audio_ref": audio_ref,
            "dec_inputs": dec_inputs,
        })
    return all_results


def quality_check(dec_sessions, results):
    """Compare INT8 audio vs FP32 audio."""
    print(f"\n{'='*70}")
    print(f"[3] Quality check (INT8 vs FP32)")
    print(f"{'='*70}")
    print(f"{'Test':<10}{'rms_fp32':<12}{'rms_int8':<12}{'ratio':<10}{'cos_sim':<10}{'max_err':<12}")
    print("-" * 80)

    for r in results:
        audio_fp32 = r["audio_ref"]
        # Re-run INT8 with same inputs
        audio_int8 = dec_sessions[1][1].run(None, r["dec_inputs"])[0].squeeze()
        n = min(len(audio_fp32), len(audio_int8))
        a, b = audio_fp32[:n], audio_int8[:n]
        rms_a = np.sqrt(np.mean(a ** 2))
        rms_b = np.sqrt(np.mean(b ** 2))
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        max_err = np.abs(a - b).max()
        ratio = max(rms_a, rms_b) / max(min(rms_a, rms_b), 1e-8)
        print(f"{r['label']:<10}{rms_a:<12.4f}{rms_b:<12.4f}{ratio:<10.3f}{cos:<10.4f}{max_err:<12.4f}")

        # Save audio files for listening
        os.makedirs(OUT_DIR, exist_ok=True)
        save_wav(os.path.join(OUT_DIR, f"{r['label']}_fp32.wav"), normalize_audio(audio_fp32))
        save_wav(os.path.join(OUT_DIR, f"{r['label']}_int8.wav"), normalize_audio(audio_int8))


def try_graph_optimization():
    """Try ORT format conversion + max graph optimization level."""
    print(f"\n{'='*70}")
    print(f"[4] Graph optimization levels")
    print(f"{'='*70}")

    # Available levels: ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    levels = [
        ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]
    # Load a small input for testing
    # Reuse encoder output from benchmark if available
    return levels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Kokoro Decoder INT8 Quantization Benchmark")
    print("=" * 70)

    # Step 1: Quantize
    if not os.path.exists(DEC_QUANT):
        quantize_decoder()
    else:
        print(f"\n[1] Quantized model already exists: {DEC_QUANT}")
        print(f"  ({os.path.getsize(DEC_QUANT)/1024/1024:.2f} MB vs {os.path.getsize(DEC_ONNX)/1024/1024:.2f} MB fp32)")

    # Step 2: Load models
    print(f"\n[load] KModel + KPipeline...")
    kmodel = KModel(config=CONFIG, model=CHECKPOINT, disable_complex=True)
    kmodel.eval().cpu()
    pipeline = KPipeline(lang_code='z', model=kmodel, device='cpu')

    print(f"[load] encoder session...")
    enc_sess = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])

    print(f"[load] decoder fp32 session...")
    so_fp32 = ort.SessionOptions()
    so_fp32.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so_fp32.intra_op_num_threads = 4
    so_fp32.inter_op_num_threads = 1
    dec_fp32 = ort.InferenceSession(DEC_ONNX, so_fp32, providers=["CPUExecutionProvider"])

    print(f"[load] decoder int8 session...")
    so_int8 = ort.SessionOptions()
    so_int8.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so_int8.intra_op_num_threads = 4
    so_int8.inter_op_num_threads = 1
    dec_int8 = ort.InferenceSession(DEC_QUANT, so_int8, providers=["CPUExecutionProvider"])

    dec_sessions = [
        ("fp32", dec_fp32),
        ("int8", dec_int8),
    ]

    # Step 3: Generate real inputs for each test sentence
    print(f"\n[gen] generating encoder inputs for {len(TEST_SENTENCES)} test sentences...")
    all_inputs = []
    labels = []
    for name, text, voice in TEST_SENTENCES:
        inputs = get_real_inputs(pipeline, text, voice)
        for i, inp in enumerate(inputs):
            all_inputs.append(inp)
            labels.append(f"{name}_{i}")
    print(f"  generated {len(all_inputs)} segments")

    # Step 4: Benchmark
    results = benchmark_decoder(enc_sess, dec_sessions, all_inputs, labels)

    # Step 5: Quality check
    quality_check(dec_sessions, results)

    # Step 6: Try different thread counts on int8
    print(f"\n{'='*70}")
    print(f"[5] Thread count sweep (int8 decoder)")
    print(f"{'='*70}")
    print(f"{'Test':<10}{'threads':<10}{'dec (ms)':<12}{'rtf':<10}")
    print("-" * 50)
    for n_threads in [1, 2, 4, 8]:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = n_threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(DEC_QUANT, so, providers=["CPUExecutionProvider"])
        for r in results[:1]:  # just test on first segment
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                sess.run(None, r["dec_inputs"])
                times.append((time.perf_counter() - t0) * 1000)
            med = np.median(times)
            rtf = (med / 1000) / r["audio_dur"]
            print(f"{r['label']:<10}{n_threads:<10}{med:<12.1f}{rtf:<10.3f}")

    print(f"\nWAV files saved to: {OUT_DIR}")
    print(f"  Compare *_fp32.wav vs *_int8.wav to check quality")


if __name__ == "__main__":
    main()
