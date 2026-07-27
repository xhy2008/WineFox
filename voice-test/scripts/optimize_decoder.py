"""Try multiple decoder optimization strategies and pick the winner.

After dynamic INT8 quantization proved 7x SLOWER than FP32 (due to
QLinearConv overhead without AVX-VNNI), we try:

  A. FP16 half-precision (weight-only, no activation conversion)
     - Half the memory bandwidth, no quantization overhead
     - CPU FP16 support via AVX512-FP16 or fallback
  B. Static INT8 quantization (weight + activation, with calibration)
     - Avoids per-run activation quantization
     - Requires representative calibration data
  C. Graph optimization level sweep on FP32
     - ORT_ENABLE_ALL may fuse Conv/Act patterns
  D. Thread count sweep on FP32
     - Find the optimal intra_op_num_threads

The decoder is Conv-heavy (HiFi-GAN style), so memory bandwidth is
likely the bottleneck. FP16 should help most.
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
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization.calibrate import CalibrationMethod

# Load kokoro (same stub trick)
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

# Config
CONFIG = r"e:\winefox\voice-test\models\kokoro-82M-src\config.json"
CHECKPOINT = r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth"
ENC_ONNX = r"e:\winefox\voice-test\models\kokoro-encoder.onnx"
DEC_ONNX = r"e:\winefox\voice-test\models\kokoro-decoder.onnx"
DEC_FP16 = r"e:\winefox\voice-test\models\kokoro-decoder-fp16.onnx"
DEC_INT8_STATIC = r"e:\winefox\voice-test\models\kokoro-decoder-int8-static.onnx"
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


# ---------------------------------------------------------------------------
# A. FP16 conversion
# ---------------------------------------------------------------------------
def convert_to_fp16():
    """Convert decoder ONNX weights from FP32 to FP16.

    Uses onnx.converter with keep_io_types=True so inputs/outputs stay FP32
    (avoids need to change the calling code).
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16

    print(f"\n[A] Converting decoder to FP16...")
    print(f"  input: {DEC_ONNX} ({os.path.getsize(DEC_ONNX)/1024/1024:.2f} MB)")

    model = onnx.load(DEC_ONNX)

    # Convert all float32 weights/activations to float16, but keep
    # graph inputs/outputs as float32 so the calling code doesn't change.
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        min_positive_val=1e-7,    # avoid denormal overflow
        max_finite_val=1e4,
    )

    onnx.save(model_fp16, DEC_FP16)
    print(f"  output: {DEC_FP16} ({os.path.getsize(DEC_FP16)/1024/1024:.2f} MB)")
    print(f"  size reduction: {(1 - os.path.getsize(DEC_FP16)/os.path.getsize(DEC_ONNX))*100:.1f}%")
    return DEC_FP16


# ---------------------------------------------------------------------------
# B. Static INT8 quantization with calibration
# ---------------------------------------------------------------------------
class DecoderCalibReader(CalibrationDataReader):
    """Feeds real decoder inputs (from encoder output) as calibration data."""

    def __init__(self, enc_sess, decoder_input_list, max_items=20):
        self.data = []
        for input_ids, ref_s, speed in decoder_input_list[:max_items]:
            enc_out = enc_sess.run(None, {
                "input_ids": input_ids, "ref_s": ref_s, "speed": speed,
            })
            asr, F0_pred, N_pred, style_dec, _ = enc_out
            self.data.append({
                "asr": asr, "F0_pred": F0_pred,
                "N_pred": N_pred, "style_dec": style_dec,
            })
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        item = self.data[self.idx]
        self.idx += 1
        return item

    def rewind(self):
        self.idx = 0


def quantize_static_int8(enc_sess, calib_inputs):
    """Apply static INT8 quantization with calibration data."""
    print(f"\n[B] Static INT8 quantization (with calibration)...")
    print(f"  input: {DEC_ONNX} ({os.path.getsize(DEC_ONNX)/1024/1024:.2f} MB)")

    calib_reader = DecoderCalibReader(enc_sess, calib_inputs)

    quantize_static(
        model_input=DEC_ONNX,
        model_output=DEC_INT8_STATIC,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,    # QDQ is more reliable than QOperator
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,  # U8 for activations (ReLU outputs)
        op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm"],
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"  output: {DEC_INT8_STATIC} ({os.path.getsize(DEC_INT8_STATIC)/1024/1024:.2f} MB)")


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def get_real_inputs(pipeline, text, voice, speed=1.0):
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


def bench_session(sess, dec_inputs, n_runs=3):
    """Run a session n_runs times and return median ms."""
    # Warm up
    sess.run(None, dec_inputs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, dec_inputs)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), times


def benchmark_all(enc_sess, dec_paths, all_inputs, labels, n_threads=4):
    """Benchmark multiple decoder variants.

    dec_paths: list of (name, path, is_int8) tuples
    """
    print(f"\n{'='*100}")
    print(f"Benchmark: {len(dec_paths)} decoder variants, threads={n_threads}")
    print(f"{'='*100}")
    print(f"{'Test':<10}{'Audio(s)':<10}", end="")
    for name, _, _ in dec_paths:
        print(f"{name+'(ms)':<16}", end="")
    print(f"{'best':<10}")
    print("-" * 100)

    results = []
    for (input_ids, ref_s, speed_arr), label in zip(all_inputs, labels):
        enc_out = enc_sess.run(None, {
            "input_ids": input_ids, "ref_s": ref_s, "speed": speed_arr,
        })
        asr, F0_pred, N_pred, style_dec, _ = enc_out
        dec_inputs = {
            "asr": asr, "F0_pred": F0_pred,
            "N_pred": N_pred, "style_dec": style_dec,
        }

        dec_times = []
        audio_dur = 0
        for name, path, is_int8 in dec_paths:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = n_threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            med_ms, _ = bench_session(sess, dec_inputs)
            dec_times.append(med_ms)
            if audio_dur == 0:
                audio_dur = len(sess.run(None, dec_inputs)[0].squeeze()) / SAMPLE_RATE

        best_idx = int(np.argmin(dec_times))
        best_name = dec_paths[best_idx][0]
        print(f"{label:<10}{audio_dur:<10.3f}", end="")
        for t in dec_times:
            print(f"{t:<16.1f}", end="")
        print(f"{best_name:<10}")

        results.append({
            "label": label, "audio_dur": audio_dur,
            "dec_times": dec_times, "dec_inputs": dec_inputs,
        })
    return results


def quality_check_all(dec_paths, results):
    """Compare each variant's audio vs FP32 baseline."""
    print(f"\n{'='*100}")
    print(f"Quality check (each variant vs FP32)")
    print(f"{'='*100}")
    print(f"{'Test':<10}{'variant':<12}{'rms_ref':<12}{'rms_var':<12}{'ratio':<10}{'cos_sim':<10}{'max_err':<12}")
    print("-" * 100)

    # FP32 session as baseline
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    fp32_sess = ort.InferenceSession(DEC_ONNX, so, providers=["CPUExecutionProvider"])

    for r in results:
        audio_ref = fp32_sess.run(None, r["dec_inputs"])[0].squeeze()
        rms_ref = np.sqrt(np.mean(audio_ref ** 2))

        for name, path, _ in dec_paths:
            if name == "fp32":
                continue
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            audio_var = sess.run(None, r["dec_inputs"])[0].squeeze()
            n = min(len(audio_ref), len(audio_var))
            a, b = audio_ref[:n], audio_var[:n]
            rms_var = np.sqrt(np.mean(b ** 2))
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            max_err = np.abs(a - b).max()
            ratio = max(rms_ref, rms_var) / max(min(rms_ref, rms_var), 1e-8)
            print(f"{r['label']:<10}{name:<12}{rms_ref:<12.4f}{rms_var:<12.4f}{ratio:<10.3f}{cos:<10.4f}{max_err:<12.4f}")

            # Save audio
            os.makedirs(OUT_DIR, exist_ok=True)
            save_wav(os.path.join(OUT_DIR, f"{r['label']}_{name}.wav"), normalize_audio(b))
        save_wav(os.path.join(OUT_DIR, f"{r['label']}_fp32.wav"), normalize_audio(audio_ref))


def thread_sweep(dec_path, dec_inputs, label):
    """Sweep thread count for a given decoder."""
    print(f"\n{'='*70}")
    print(f"Thread count sweep: {os.path.basename(dec_path)}")
    print(f"{'='*70}")
    print(f"{'threads':<10}{'dec (ms)':<12}{'rtf':<10}")
    print("-" * 40)
    best_threads = 1
    best_ms = float('inf')
    for n in [1, 2, 4, 8, 16]:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(dec_path, so, providers=["CPUExecutionProvider"])
        med_ms, _ = bench_session(sess, dec_inputs)
        audio_dur = len(sess.run(None, dec_inputs)[0].squeeze()) / SAMPLE_RATE
        rtf = (med_ms / 1000) / audio_dur
        print(f"{n:<10}{med_ms:<12.1f}{rtf:<10.3f}")
        if med_ms < best_ms:
            best_ms = med_ms
            best_threads = n
    print(f"  best: threads={best_threads}, {best_ms:.1f}ms")
    return best_threads


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 100)
    print("Kokoro Decoder Optimization Sweep")
    print("=" * 100)

    # Load pipeline for input generation
    print("[load] KModel + KPipeline...")
    kmodel = KModel(config=CONFIG, model=CHECKPOINT, disable_complex=True)
    kmodel.eval().cpu()
    pipeline = KPipeline(lang_code='z', model=kmodel, device='cpu')
    enc_sess = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])

    # Generate calibration + benchmark inputs
    print("[gen] generating inputs...")
    all_inputs = []
    labels = []
    for name, text, voice in TEST_SENTENCES:
        inputs = get_real_inputs(pipeline, text, voice)
        for i, inp in enumerate(inputs):
            all_inputs.append(inp)
            labels.append(f"{name}_{i}")
    print(f"  {len(all_inputs)} segments")

    # A. FP16 conversion
    if not os.path.exists(DEC_FP16):
        convert_to_fp16()
    else:
        print(f"\n[A] FP16 already exists: {DEC_FP16} ({os.path.getsize(DEC_FP16)/1024/1024:.2f} MB)")

    # B. Static INT8 quantization
    if not os.path.exists(DEC_INT8_STATIC):
        quantize_static_int8(enc_sess, all_inputs)
    else:
        print(f"\n[B] Static INT8 already exists: {DEC_INT8_STATIC} ({os.path.getsize(DEC_INT8_STATIC)/1024/1024:.2f} MB)")

    # Build variants list
    variants = [
        ("fp32", DEC_ONNX, False),
        ("fp16", DEC_FP16, False),
    ]
    if os.path.exists(DEC_INT8_STATIC):
        variants.append(("int8s", DEC_INT8_STATIC, True))

    # Benchmark all
    results = benchmark_all(enc_sess, variants, all_inputs, labels, n_threads=4)

    # Quality check
    quality_check_all(variants, results)

    # Thread sweep on the best variant (use first test input)
    print("\n[determine best variant by min total time...]")
    total_times = [sum(r["dec_times"][i] for r in results) for i in range(len(variants))]
    best_idx = int(np.argmin(total_times))
    best_name, best_path, _ = variants[best_idx]
    print(f"  best variant: {best_name} (total={total_times[best_idx]:.0f}ms)")

    if results:
        thread_sweep(best_path, results[0]["dec_inputs"], labels[0])

    print(f"\nWAV files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
