"""Quantize Kokoro decoder ONNX with multiple strategies and compare quality.

The decoder is conv-heavy (iSTFTNet). Full INT8-static quantization causes
audio corruption (buzzing/crackling). This script tries several strategies
to find the best quality/speed tradeoff:

  1. Dynamic quantization (weight-only, FP32 activations)
     - Fastest to produce, no calibration needed
     - Only weights are INT8, activations stay FP32
     - Should be faster than FP32 but slower than static INT8

  2. Static INT8 with selective op quantization (QDQ format)
     - Only quantize specific op types (e.g., Conv, but not ConvTranspose)
     - Requires calibration data (uses encoder outputs)

  3. FP16 (if supported by this model)

Strategy: generate candidates, synthesize a test sample with each, and
report RTF + audio stats so the user can pick the best one.

Usage:
    python quantize_decoder.py
"""
import os
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static, QuantType, QuantFormat,
    CalibrationDataReader, CalibrationMethod,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENC_FP32    = r"e:\winefox\voice-test\models\kokoro-encoder.onnx"
DEC_FP32    = r"e:\winefox\voice-test\models\kokoro-decoder.onnx"
DEC_INT8_ST = r"e:\winefox\voice-test\models\kokoro-decoder-int8-static.onnx"
VOICES_BIN  = r"e:\winefox\voice-test\models\voices-v1.1-zh.bin"
VOCAB_TXT   = r"e:\winefox\voice-test\third_party\kokoro-cpp-src\dict\vocab.txt"

OUT_DIR     = r"e:\winefox\voice-test\models"

TEST_TEXT   = "你好，我是小狐狸。今天天气真好。"
VOICE_NAME  = "zf_001"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_voice_style(voices_path, name):
    """Load a single voice style vector from voices.bin."""
    with open(voices_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"VOIC", f"bad magic: {magic}"
        version = int.from_bytes(f.read(4), "little")
        assert version == 1
        num_voices = int.from_bytes(f.read(4), "little")
        for _ in range(num_voices):
            name_len = int.from_bytes(f.read(4), "little")
            vname = f.read(name_len).decode("utf-8")
            dim = int.from_bytes(f.read(4), "little")
            offset = f.tell()
            if vname == name:
                f.seek(offset)
                data = f.read(dim * 4)
                return np.frombuffer(data, dtype=np.float32).copy()
            f.seek(dim * 4, 1)  # skip style data
    raise ValueError(f"voice {name} not found")


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            tab = line.find("\t")
            if tab != -1:
                token = line[:tab]
                id_str = line[tab+1:]
                token = token.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
                try:
                    vocab[token] = int(id_str)
                except ValueError:
                    pass
    return vocab


def run_encoder(enc_sess, tokens, voice_style, speed):
    """Run encoder, return (asr, F0_pred, N_pred, style_dec, pred_dur)."""
    input_ids = np.array([[0] + tokens + [0]], dtype=np.int64)
    # Select style by token length (same logic as Kokoro.cpp)
    STYLE_DIM = 256
    if len(voice_style) > STYLE_DIM:
        idx = len(tokens)
        if idx * STYLE_DIM + STYLE_DIM <= len(voice_style):
            ref_s = voice_style[idx * STYLE_DIM : (idx + 1) * STYLE_DIM].reshape(1, STYLE_DIM)
        else:
            ref_s = voice_style[:STYLE_DIM].reshape(1, STYLE_DIM)
    else:
        ref_s = voice_style.reshape(1, STYLE_DIM)
    speed_arr = np.array([speed], dtype=np.float32)

    outputs = enc_sess.run(
        ["asr", "F0_pred", "N_pred", "style_dec", "pred_dur"],
        {"input_ids": input_ids, "ref_s": ref_s.astype(np.float32), "speed": speed_arr},
    )
    return outputs  # [asr, F0_pred, N_pred, style_dec, pred_dur]


def run_decoder(dec_sess, enc_outputs):
    """Run decoder, return audio array."""
    asr, f0, n_pred, style_dec = enc_outputs[:4]
    outputs = dec_sess.run(
        ["audio"],
        {"asr": asr, "F0_pred": f0, "N_pred": n_pred, "style_dec": style_dec},
    )
    return outputs[0].squeeze()


def phonemize_simple(text):
    """Placeholder: use a simple phoneme mapping for testing.
    In production, use the full G2P pipeline. For quantization comparison,
    we only need consistent inputs across decoder variants, so we use
    pre-computed phonemes."""
    # This function is not used — we call the C++ Kokoro binary to get
    # the encoder outputs via a special mode. Instead, for this script,
    # we'll run the FP32 encoder once and reuse its outputs for all
    # decoder variants. This isolates decoder quality differences.
    pass


def write_wav(path, audio, sr=24000):
    """Write float32 audio as 16-bit PCM WAV."""
    import struct
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm) * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm) * 2))
        f.write(pcm.tobytes())
    print(f"  wrote {path} ({len(pcm)} samples, {len(pcm)/sr:.3f}s)")


# ---------------------------------------------------------------------------
# Calibration data reader for static quantization
# ---------------------------------------------------------------------------

class DecoderCalibrationReader(CalibrationDataReader):
    """Generates calibration data by running the FP32 encoder on varied inputs."""
    def __init__(self, enc_sess, voice_style):
        self.enc_sess = enc_sess
        self.voice_style = voice_style
        # Generate varied token lengths for calibration
        np.random.seed(42)
        self._token_sets = []
        for t_len in [8, 16, 32, 64, 128]:
            # Use valid token IDs (1-177 based on vocab)
            tokens = [(i % 177) + 1 for i in range(t_len)]
            self._token_sets.append(tokens)
        self._idx = 0
        self._data = []

    def _generate(self):
        for tokens in self._token_sets:
            enc_out = run_encoder(self.enc_sess, tokens, self.voice_style, 1.0)
            yield {
                "asr": enc_out[0],
                "F0_pred": enc_out[1],
                "N_pred": enc_out[2],
                "style_dec": enc_out[3],
            }

    def get_next(self):
        if not self._data:
            self._data = list(self._generate())
        if self._idx >= len(self._data):
            return None
        item = self._data[self._idx]
        self._idx += 1
        return item


# ---------------------------------------------------------------------------
# Quantization strategies
# ---------------------------------------------------------------------------

def quantize_dynamic_decoder():
    """Strategy 1: Dynamic quantization (weight-only INT8, FP32 activations)."""
    out_path = os.path.join(OUT_DIR, "kokoro-decoder-int8-dynamic.onnx")
    print(f"\n[Dynamic INT8] Quantizing decoder (weight-only)...")
    print(f"  input:  {DEC_FP32} ({os.path.getsize(DEC_FP32)/1024/1024:.2f} MB)")

    quantize_dynamic(
        model_input=DEC_FP32,
        model_output=out_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    )
    print(f"  output: {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")
    return out_path


def quantize_static_conv_only(enc_sess, voice_style):
    """Strategy 2: Static INT8 QDQ, Conv only (skip ConvTranspose)."""
    out_path = os.path.join(OUT_DIR, "kokoro-decoder-int8-static-conv-only.onnx")
    print(f"\n[Static INT8 Conv-only] Quantizing decoder (QDQ, Conv only)...")

    reader = DecoderCalibrationReader(enc_sess, voice_style)
    quantize_static(
        model_input=DEC_FP32,
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],  # skip ConvTranspose
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"  output: {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")
    return out_path


def quantize_static_gemm_only(enc_sess, voice_style):
    """Strategy 3: Static INT8 QDQ, Gemm only."""
    out_path = os.path.join(OUT_DIR, "kokoro-decoder-int8-static-gemm-only.onnx")
    print(f"\n[Static INT8 Gemm-only] Quantizing decoder (QDQ, Gemm only)...")

    reader = DecoderCalibrationReader(enc_sess, voice_style)
    quantize_static(
        model_input=DEC_FP32,
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Gemm"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"  output: {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Benchmark + compare
# ---------------------------------------------------------------------------

def benchmark_decoder(dec_path, enc_outputs, label, out_wav):
    """Run decoder, measure RTF, write WAV, return stats."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 8
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(dec_path, so, providers=["CPUExecutionProvider"])

    # Warmup
    run_decoder(sess, enc_outputs)

    # Benchmark
    t0 = time.perf_counter()
    audio = run_decoder(sess, enc_outputs)
    t1 = time.perf_counter()

    dur = len(audio) / 24000.0
    rtf = (t1 - t0) / dur if dur > 0 else 0

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    print(f"\n  [{label}]")
    print(f"    file:      {os.path.basename(dec_path)}")
    print(f"    size:      {os.path.getsize(dec_path)/1024/1024:.2f} MB")
    print(f"    samples:   {len(audio)}")
    print(f"    duration:  {dur:.3f}s")
    print(f"    synth:     {(t1-t0)*1000:.1f}ms")
    print(f"    RTF:       {rtf:.4f} ({1/rtf:.2f}x realtime)" if rtf > 0 else "    RTF:       N/A")
    print(f"    RMS:       {rms:.4f}")
    print(f"    peak:      {peak:.4f}")

    write_wav(out_wav, audio)
    return {"label": label, "rtf": rtf, "rms": rms, "peak": peak, "audio": audio}


def main():
    print("=" * 70)
    print("Kokoro Decoder Quantization Comparison")
    print("=" * 70)

    # Load encoder (FP32) to generate test inputs
    print("\nLoading FP32 encoder for test input generation...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    enc_sess = ort.InferenceSession(ENC_FP32, so, providers=["CPUExecutionProvider"])

    voice_style = load_voice_style(VOICES_BIN, VOICE_NAME)

    # We need phonemes for the test text. Since we don't have the full G2P
    # pipeline in Python, use a fixed token sequence that represents
    # "你好，我是小狐狸。今天天气真好。" — these are placeholder tokens.
    # The key point is: same encoder outputs for all decoder variants,
    # so we can isolate decoder quality differences.
    #
    # Use a moderate-length token sequence (matches typical sentence).
    # The actual phonemes don't matter for comparing decoder quantization
    # quality — what matters is that all decoders get the same input.
    test_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    print(f"\nGenerating encoder outputs (tokens={len(test_tokens)})...")
    enc_outputs = run_encoder(enc_sess, test_tokens, voice_style, 1.0)
    print(f"  asr shape:       {enc_outputs[0].shape}")
    print(f"  F0_pred shape:   {enc_outputs[1].shape}")
    print(f"  N_pred shape:    {enc_outputs[2].shape}")
    print(f"  style_dec shape: {enc_outputs[3].shape}")

    results = []

    # 1. FP32 baseline (clean)
    results.append(benchmark_decoder(
        DEC_FP32, enc_outputs, "FP32",
        os.path.join(OUT_DIR, "..", "dec_test_fp32.wav"),
    ))

    # 2. Existing INT8-static (SKIP — known to crash with Reshape error)
    # The INT8-static quantization has a bug: Reshape node gets garbage shape
    # value (-4672999374901377890). This is a quantization tool issue, not
    # fixable from our side. FP32 is the only working decoder.
    if False and os.path.exists(DEC_INT8_ST):
        results.append(benchmark_decoder(
            DEC_INT8_ST, enc_outputs, "INT8-static (existing)",
            os.path.join(OUT_DIR, "..", "dec_test_int8_static.wav"),
        ))

    # 3. Dynamic INT8 (weight-only)
    try:
        dyn_path = quantize_dynamic_decoder()
        results.append(benchmark_decoder(
            dyn_path, enc_outputs, "INT8-dynamic",
            os.path.join(OUT_DIR, "..", "dec_test_int8_dynamic.wav"),
        ))
    except Exception as e:
        print(f"  Dynamic quantization failed: {e}")

    # 4. Static INT8 Conv-only (skip ConvTranspose)
    try:
        conv_path = quantize_static_conv_only(enc_sess, voice_style)
        results.append(benchmark_decoder(
            conv_path, enc_outputs, "INT8-static Conv-only",
            os.path.join(OUT_DIR, "..", "dec_test_int8_conv_only.wav"),
        ))
    except Exception as e:
        print(f"  Static Conv-only quantization failed: {e}")

    # 5. Static INT8 Gemm-only
    try:
        gemm_path = quantize_static_gemm_only(enc_sess, voice_style)
        results.append(benchmark_decoder(
            gemm_path, enc_outputs, "INT8-static Gemm-only",
            os.path.join(OUT_DIR, "..", "dec_test_int8_gemm_only.wav"),
        ))
    except Exception as e:
        print(f"  Static Gemm-only quantization failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  {'Variant':<30} {'RTF':<10} {'RMS':<10} {'Peak':<10}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['label']:<30} {r['rtf']:<10.4f} {r['rms']:<10.4f} {r['peak']:<10.4f}")

    print("\nWAV files written to e:\\winefox\\voice-test\\models\\..\\")
    print("Listen to each dec_test_*.wav and compare with dec_test_fp32.wav (clean baseline).")


if __name__ == "__main__":
    main()
