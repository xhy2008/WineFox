"""Quantize Kokoro encoder ONNX to reduce memory footprint.

The encoder contains PLBERT (BERT-based) + LSTM duration predictor + text
encoder (conv). It is precision-sensitive, so we use a conservative
strategy:

  INT8 static quantization with QDQ format (selective):
    - Quantize only Conv and Gemm weights + activations
    - Skip MatMul (BERT attention stays in FP32)
    - Uses QDQ (Quantize-Dequantize) format which uses standard ONNX ops
      (Conv, DequantizeLinear, QuantizeLinear) — compatible with all ORT
      builds including minimal C++ builds without the quantization EP.
    - Requires calibration data to determine activation ranges.

FP16 conversion was attempted but fails on this model due to internal
Cast nodes that produce type mismatches (tensor(float16) vs tensor(float)).
INT8 static is the supported path.

We verify quality by comparing encoder outputs (asr, F0_pred, N_pred,
style_dec) between FP32 and quantized versions on a dummy input.
"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, QuantType, QuantFormat, CalibrationDataReader,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENC_ONNX   = r"e:\winefox\voice-test\models\kokoro-encoder.onnx"
ENC_INT8   = r"e:\winefox\voice-test\models\kokoro-encoder-int8.onnx"


class DummyCalibrationReader(CalibrationDataReader):
    """Provides calibration data for static quantization.

    Uses a few representative input shapes (short, medium, long) to cover
    the dynamic range of the encoder's activation values.
    """
    def __init__(self):
        self._data_iter = self._generate()

    def _generate(self):
        np.random.seed(42)
        # Vocab has 178 tokens (indices 0-177). Use valid IDs for calibration.
        # 0 = BOS/EOS, so use 1-177 for content tokens.
        for t_len in [16, 48, 128, 256]:
            # Clip to valid vocab range and wrap around
            ids = [(i % 177) + 1 for i in range(t_len)]
            input_ids = np.array([[0] + ids + [0]], dtype=np.int64)
            ref_s = np.random.randn(1, 256).astype(np.float32) * 0.3
            speed = np.array([1.0], dtype=np.float32)
            yield {
                "input_ids": input_ids,
                "ref_s": ref_s,
                "speed": speed,
            }

    def get_next(self):
        return next(self._data_iter, None)


def quantize_int8():
    """Static INT8 quantization with QDQ format (Conv only).

    QDQ format inserts QuantizeLinear/DequantizeLinear nodes around Conv
    ops, keeping the original op type. This is compatible with all ORT
    builds, including minimal C++ builds without the QOperator (QLinearConv).

    Only Conv ops (text encoder conv blocks) are quantized. Gemm and MatMul
    are skipped to preserve precision in:
      - BERT FFN (Gemm) — affects semantic features
      - LSTM duration predictor — affects pred_dur, which controls output
        length; even a ±1 rounding change in any duration flips the
        upsampled length and causes shape mismatch in asr/F0_pred/N_pred.
    """
    print(f"\n[INT8] Static quantization (QDQ, Conv only, skip Gemm/MatMul)")
    print(f"  input:  {ENC_ONNX} ({os.path.getsize(ENC_ONNX)/1024/1024:.2f} MB)")

    reader = DummyCalibrationReader()
    quantize_static(
        model_input=ENC_ONNX,
        model_output=ENC_INT8,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f"  output: {ENC_INT8} ({os.path.getsize(ENC_INT8)/1024/1024:.2f} MB)")
    print(f"  reduction: {(1 - os.path.getsize(ENC_INT8)/os.path.getsize(ENC_ONNX))*100:.1f}%")


def verify():
    """Compare encoder outputs between FP32 and quantized versions."""
    print(f"\n{'='*70}")
    print(f"Verification: FP32 vs quantized encoder outputs")
    print(f"{'='*70}")

    # Dummy input: realistic shape (T_ids=48)
    input_ids = np.array([[0] + list(range(1, 49)) + [0]], dtype=np.int64)
    ref_s = np.random.randn(1, 256).astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)

    feeds = {"input_ids": input_ids, "ref_s": ref_s, "speed": speed}
    output_names = ["asr", "F0_pred", "N_pred", "style_dec", "pred_dur"]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4

    variants = [("fp32", ENC_ONNX)]
    if os.path.exists(ENC_INT8):
        variants.append(("int8", ENC_INT8))

    outputs = {}
    for name, path in variants:
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        outs = sess.run(output_names, feeds)
        outputs[name] = {n: v for n, v in zip(output_names, outs)}
        print(f"  [{name}] loaded ({os.path.getsize(path)/1024/1024:.1f} MB)")

    # Compare
    print(f"\n  {'output':<12} {'fp32 shape':<20} {'int8 err':<15}")
    print(f"  {'-'*47}")
    ref = outputs["fp32"]
    for name in output_names:
        ref_val = ref[name]
        row = f"  {name:<12} {str(ref_val.shape):<20}"
        for qname in ["int8"]:
            if qname in outputs:
                qval = outputs[qname][name]
                if qval.shape == ref_val.shape:
                    err = np.abs(ref_val - qval).max()
                    row += f" {err:<15.6f}"
                else:
                    row += f" {'shape mismatch':<15}"
            else:
                row += f" {'N/A':<15}"
        print(row)

    # Also compare decoder audio if decoder is available
    DEC_ONNX = r"e:\winefox\voice-test\models\kokoro-decoder-int8-static.onnx"
    if os.path.exists(DEC_ONNX):
        print(f"\n  Audio comparison (using INT8-static decoder):")
        dec_sess = ort.InferenceSession(DEC_ONNX, so, providers=["CPUExecutionProvider"])
        for name in ["fp32", "int8"]:
            if name not in outputs:
                continue
            o = outputs[name]
            dec_out = dec_sess.run(None, {
                "asr": o["asr"], "F0_pred": o["F0_pred"],
                "N_pred": o["N_pred"], "style_dec": o["style_dec"],
            })
            audio = dec_out[0].squeeze()
            print(f"    [{name}] audio: shape={audio.shape}, rms={np.sqrt(np.mean(audio**2)):.4f}")


def main():
    print("=" * 70)
    print("Kokoro Encoder Quantization")
    print("=" * 70)

    if not os.path.exists(ENC_ONNX):
        print(f"ERROR: encoder ONNX not found: {ENC_ONNX}")
        sys.exit(1)

    # Step 1: INT8 static quantization (QDQ format)
    if not os.path.exists(ENC_INT8):
        quantize_int8()
    else:
        print(f"\n[INT8] Already exists: {ENC_INT8} ({os.path.getsize(ENC_INT8)/1024/1024:.2f} MB)")

    # Step 2: Verify
    verify()


if __name__ == "__main__":
    main()

