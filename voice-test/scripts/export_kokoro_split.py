"""Export Kokoro-82M as two separate ONNX models for streaming inference.

Split point (from kokoro/model.py forward_with_tokens, line 118):
    audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])

Encoder ONNX:
    inputs:  input_ids [B, T_ids], ref_s [B, 256], speed [B]
    outputs: asr [B, 512, T_frm], F0_pred [B, T_frm], N_pred [B, T_frm],
             style_dec [B, 128], pred_dur [B, T_ids]

Decoder ONNX:
    inputs:  asr [B, 512, T_frm], F0_pred [B, T_frm], N_pred [B, T_frm],
             style_dec [B, 128]
    outputs: audio [B, T_audio]

The split enables:
  1. Running encoder early while LLM is still streaming
  2. Future block-level vocoder chunking (conv chain is splittable,
     SineGen cumsum still requires full-segment F0)
  3. Path to ggml port (clearer boundary between transformer-ish encoder
     and conv-based decoder)
"""
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

# Add kokoro source to path. We bypass kokoro/__init__.py because it imports
# KPipeline -> misaki (G2P frontend), which we don't need for ONNX export.
# Instead we load kokoro.model directly via importlib.
KOKORO_SRC = r"e:\winefox\voice-test\third_party\kokoro-src"
sys.path.insert(0, KOKORO_SRC)

import importlib.util

def _load_kokoro_model_module():
    """Load kokoro.model without triggering kokoro/__init__.py imports."""
    # Pre-create an empty 'kokoro' package in sys.modules so that
    # `from .istftnet import Decoder` inside model.py resolves.
    import types
    pkg = types.ModuleType("kokoro")
    pkg.__path__ = [os.path.join(KOKORO_SRC, "kokoro")]
    sys.modules["kokoro"] = pkg

    # Now import submodules directly. istftnet -> custom_stft, modules.
    # These use absolute imports `from kokoro.custom_stft import CustomSTFT`
    # which will work because 'kokoro' is now in sys.modules.
    spec = importlib.util.spec_from_file_location(
        "kokoro.model",
        os.path.join(KOKORO_SRC, "kokoro", "model.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kokoro.model"] = mod
    spec.loader.exec_module(mod)
    return mod

_kokoro_model = _load_kokoro_model_module()
KModel = _kokoro_model.KModel
KModelForONNX = _kokoro_model.KModelForONNX


# ---------------------------------------------------------------------------
# Encoder wrapper: forward_with_tokens up to (but not including) decoder call
# ---------------------------------------------------------------------------
class KModelEncoder(nn.Module):
    """Wraps the pre-decoder part of KModel.forward_with_tokens.

    Reproduces kokoro/model.py lines 93-117, returning the exact inputs
    that Decoder.forward expects (istftnet.py line 407):
        decoder(asr, F0_curve, N, s)  where s = ref_s[:, :128]
    """
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    @torch.no_grad()
    def forward(self, input_ids, ref_s, speed):
        k = self.kmodel
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1
        ).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(k.device)

        bert_dur = k.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = k.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = k.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = k.predictor.lstm(d)
        duration = k.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=k.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=k.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(k.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = k.predictor.F0Ntrain(en, s)
        t_en = k.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        style_dec = ref_s[:, :128]
        return asr, F0_pred, N_pred, style_dec, pred_dur


# ---------------------------------------------------------------------------
# Decoder wrapper: just kmodel.decoder
# ---------------------------------------------------------------------------
class KModelDecoder(nn.Module):
    """Wraps kmodel.decoder for standalone ONNX export.

    Input:  asr [B, 512, T_frm], F0_pred [B, T_frm], N_pred [B, T_frm],
            style_dec [B, 128]
    Output: audio [B, T_audio]
    """
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.decoder = kmodel.decoder

    @torch.no_grad()
    def forward(self, asr, F0_pred, N_pred, style_dec):
        audio = self.decoder(asr, F0_pred, N_pred, style_dec)
        return audio


def export_encoder(kmodel, out_dir):
    """Export encoder ONNX."""
    out_path = os.path.join(out_dir, "kokoro-encoder.onnx")
    wrapper = KModelEncoder(kmodel).eval().cpu()

    # Dummy inputs: T_ids=48 (typical short sentence)
    input_ids = torch.LongTensor([[0] + list(range(1, 49)) + [0]])  # [1, 50]
    ref_s = torch.randn(1, 256)
    speed = torch.tensor([1.0], dtype=torch.float32)

    print(f"[encoder] exporting to {out_path} ...")
    print(f"[encoder] input_ids shape: {input_ids.shape}")
    print(f"[encoder] ref_s shape:     {ref_s.shape}")
    print(f"[encoder] speed shape:     {speed.shape}")

    torch.onnx.export(
        wrapper,
        args=(input_ids, ref_s, speed),
        f=out_path,
        export_params=True,
        verbose=False,
        input_names=["input_ids", "ref_s", "speed"],
        output_names=["asr", "F0_pred", "N_pred", "style_dec", "pred_dur"],
        opset_version=17,
        dynamic_axes={
            "input_ids":  {0: "batch", 1: "T_ids"},
            "ref_s":      {0: "batch"},
            "speed":      {0: "batch"},
            "asr":        {0: "batch", 2: "T_frm"},
            "F0_pred":    {0: "batch", 1: "T_frm"},
            "N_pred":     {0: "batch", 1: "T_frm"},
            "style_dec":  {0: "batch"},
            "pred_dur":   {0: "batch", 1: "T_ids"},
        },
        do_constant_folding=True,
        dynamo=False,  # use legacy TorchScript trace path (more tolerant of SDPA)
    )
    print(f"[encoder] OK -> {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")

    # Validate
    onnx.checker.check_model(onnx.load(out_path))
    print(f"[encoder] onnx.checker OK")
    return out_path


def export_decoder(kmodel, out_dir):
    """Export decoder ONNX."""
    out_path = os.path.join(out_dir, "kokoro-decoder.onnx")
    wrapper = KModelDecoder(kmodel).eval().cpu()

    # Dummy inputs: T_frm=100 (frame-level, post duration alignment).
    # NOTE: F0_pred and N_pred are 2x asr length, because ProsodyPredictor.F0Ntrain
    # contains an upsample=True AdainResBlk1d that doubles the time dimension.
    # Decoder.F0_conv (stride=2) then halves them back to match asr.T.
    T_frm = 100
    asr = torch.randn(1, 512, T_frm)
    F0_pred = torch.randn(1, T_frm * 2)
    N_pred = torch.randn(1, T_frm * 2)
    style_dec = torch.randn(1, 128)

    print(f"\n[decoder] exporting to {out_path} ...")
    print(f"[decoder] asr shape:       {asr.shape}")
    print(f"[decoder] F0_pred shape:   {F0_pred.shape}")
    print(f"[decoder] N_pred shape:    {N_pred.shape}")
    print(f"[decoder] style_dec shape: {style_dec.shape}")

    torch.onnx.export(
        wrapper,
        args=(asr, F0_pred, N_pred, style_dec),
        f=out_path,
        export_params=True,
        verbose=False,
        input_names=["asr", "F0_pred", "N_pred", "style_dec"],
        output_names=["audio"],
        opset_version=17,
        dynamic_axes={
            "asr":       {0: "batch", 2: "T_frm"},
            "F0_pred":   {0: "batch", 1: "T_frm"},
            "N_pred":    {0: "batch", 1: "T_frm"},
            "style_dec": {0: "batch"},
            "audio":     {0: "batch", 1: "T_audio"},
        },
        do_constant_folding=True,
        dynamo=False,  # use legacy TorchScript trace path
    )
    print(f"[decoder] OK -> {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")

    onnx.checker.check_model(onnx.load(out_path))
    print(f"[decoder] onnx.checker OK")
    return out_path


def verify_split(kmodel, enc_path, dec_path):
    """Verify that encoder + decoder produce the same output as the
    original end-to-end KModel.forward_with_tokens.

    IMPORTANT: Kokoro's SineGen (istftnet.py lines 200-209, 253) uses
    torch.randn_like to inject noise into the sine source. This means
    every forward pass produces slightly different audio even with the
    same inputs. To make verification deterministic, we fix the random
    seed before each call.
    """
    print("\n[verify] comparing split (enc+dec) vs end-to-end ...")
    kmodel.eval().cpu()

    # Use a realistic-ish input
    input_ids = torch.LongTensor([[0] + list(range(1, 49)) + [0]])
    ref_s = torch.randn(1, 256)
    speed = torch.tensor([1.0], dtype=torch.float32)

    # 0) Sanity: run end-to-end twice with fixed seeds to measure
    #    the noise floor (inherent non-determinism of SineGen).
    torch.manual_seed(42)
    audio_ref, dur_ref = kmodel.forward_with_tokens(input_ids, ref_s, speed)
    torch.manual_seed(42)
    audio_ref2, _ = kmodel.forward_with_tokens(input_ids, ref_s, speed)
    noise_floor = (audio_ref - audio_ref2).abs().max().item()
    print(f"[verify] end-to-end noise floor (same seed): {noise_floor:.2e}")

    torch.manual_seed(123)
    audio_a, _ = kmodel.forward_with_tokens(input_ids, ref_s, speed)
    torch.manual_seed(456)
    audio_b, _ = kmodel.forward_with_tokens(input_ids, ref_s, speed)
    noise_diff = (audio_a - audio_b).abs().max().item()
    print(f"[verify] end-to-end noise diff (diff seed): {noise_diff:.2e}")

    # 1) End-to-end reference (fixed seed)
    torch.manual_seed(42)
    audio_ref, dur_ref = kmodel.forward_with_tokens(input_ids, ref_s, speed)
    print(f"[verify] end-to-end audio shape: {audio_ref.shape}")

    # 2) Split via PyTorch (fixed seed for decoder, since SineGen is in decoder)
    enc_wrapper = KModelEncoder(kmodel).eval().cpu()
    asr, F0_pred, N_pred, style_dec, pred_dur = enc_wrapper(input_ids, ref_s, speed)
    dec_wrapper = KModelDecoder(kmodel).eval().cpu()
    torch.manual_seed(42)  # same seed as end-to-end
    audio_split = dec_wrapper(asr, F0_pred, N_pred, style_dec)
    # Squeeze to match end-to-end shape
    audio_split = audio_split.squeeze()
    print(f"[verify] split (torch) audio shape: {audio_split.shape}")

    err = (audio_ref - audio_split).abs().max().item()
    print(f"[verify] torch split vs end-to-end max abs err: {err:.2e}")
    print(f"[verify]   (noise floor was {noise_floor:.2e}, diff-seed was {noise_diff:.2e})")
    if err < max(noise_floor * 2, 1e-5):
        print("[verify] OK - torch split matches end-to-end within noise floor")
    else:
        print(f"[verify] WARNING - err {err:.2e} exceeds noise floor {noise_floor:.2e}")

    # 3) Split via ONNX Runtime
    #    ONNX Runtime uses its own RNG, so we can't match PyTorch's seed.
    #    We expect the ONNX result to differ from PyTorch by ~noise_diff.
    enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])

    enc_out = enc_sess.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "ref_s": ref_s.numpy(),
            "speed": speed.numpy(),
        },
    )
    asr_onnx, F0_onnx, N_onnx, style_dec_onnx, dur_onnx = enc_out

    # Verify encoder outputs match (encoder has no randomness)
    enc_err = np.abs(asr.numpy() - asr_onnx).max()
    print(f"[verify] encoder asr err: {enc_err:.2e}")
    enc_err_f0 = np.abs(F0_pred.numpy() - F0_onnx).max()
    print(f"[verify] encoder F0_pred err: {enc_err_f0:.2e}")

    # Print input ranges for debugging decoder
    print(f"[verify] decoder input ranges:")
    print(f"[verify]   asr:       min={asr_onnx.min():.4f} max={asr_onnx.max():.4f} mean={asr_onnx.mean():.4f}")
    print(f"[verify]   F0_pred:   min={F0_onnx.min():.4f} max={F0_onnx.max():.4f}")
    print(f"[verify]   N_pred:    min={N_onnx.min():.4f} max={N_onnx.max():.4f}")
    print(f"[verify]   style_dec: min={style_dec_onnx.min():.4f} max={style_dec_onnx.max():.4f}")

    # Run PyTorch decoder with the SAME inputs as ONNX (no seed fix, to
    # isolate decoder behavior from encoder).
    torch.manual_seed(42)
    audio_pt_dec = dec_wrapper(
        torch.from_numpy(asr_onnx),
        torch.from_numpy(F0_onnx),
        torch.from_numpy(N_onnx),
        torch.from_numpy(style_dec_onnx),
    ).squeeze().numpy()
    print(f"[verify] pytorch decoder audio: min={audio_pt_dec.min():.4f} max={audio_pt_dec.max():.4f}")

    dec_out = dec_sess.run(
        None,
        {
            "asr": asr_onnx,
            "F0_pred": F0_onnx,
            "N_pred": N_onnx,
            "style_dec": style_dec_onnx,
        },
    )
    audio_onnx = dec_out[0].squeeze()
    print(f"[verify] onnx decoder audio:    min={audio_onnx.min():.4f} max={audio_onnx.max():.4f}")

    # Compare shapes
    print(f"[verify] pytorch dec shape: {audio_pt_dec.shape}")
    print(f"[verify] onnx dec shape:    {audio_onnx.shape}")

    # The error is dominated by SineGen RNG. Compare overall statistics
    # (mean abs, RMS) instead of pointwise max.
    rms_pt = np.sqrt(np.mean(audio_pt_dec ** 2))
    rms_onnx = np.sqrt(np.mean(audio_onnx ** 2))
    mae = np.mean(np.abs(audio_pt_dec - audio_onnx))
    print(f"[verify] pytorch dec RMS: {rms_pt:.4f}, onnx dec RMS: {rms_onnx:.4f}")
    print(f"[verify] mean abs err (pt vs onnx decoder): {mae:.4f}")

    err_onnx = np.abs(audio_ref.numpy() - audio_onnx).max()
    print(f"[verify] onnx split vs end-to-end max abs err: {err_onnx:.2e}")
    print(f"[verify]   (expected ~{noise_diff:.2e} due to SineGen RNG difference)")
    # Use RMS ratio as the real quality metric
    rms_ratio = max(rms_pt, rms_onnx) / max(min(rms_pt, rms_onnx), 1e-8)
    print(f"[verify] RMS ratio (pt/onnx): {rms_ratio:.4f} (1.0 = perfect)")
    if rms_ratio < 1.5:
        print("[verify] OK - ONNX decoder produces audio of similar magnitude")
    else:
        print(f"[verify] WARNING - RMS ratio {rms_ratio:.4f} indicates decoder export issue")

    # Also report audio duration for sanity
    audio_dur_s = audio_onnx.shape[-1] / 24000.0
    print(f"[verify] audio duration: {audio_dur_s:.3f}s ({audio_onnx.shape[-1]} samples @ 24kHz)")
    return err_onnx


def main():
    parser = argparse.ArgumentParser("Export Kokoro split ONNX (encoder + decoder)")
    parser.add_argument(
        "--config", "-c",
        default=r"e:\winefox\voice-test\models\kokoro-82M-src\config.json",
    )
    parser.add_argument(
        "--checkpoint", "-p",
        default=r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=r"e:\winefox\voice-test\models",
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Config:      {args.config}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Output dir:  {args.output_dir}")

    # disable_complex=True is required for clean ONNX export
    # (TorchSTFT uses torch.istft which exports poorly; CustomSTFT is
    # implemented with explicit ops that ONNX can represent)
    kmodel = KModel(config=args.config, model=args.checkpoint, disable_complex=True)
    kmodel.eval().cpu()
    print(f"KModel loaded. device={kmodel.device}")

    enc_path = export_encoder(kmodel, args.output_dir)
    dec_path = export_decoder(kmodel, args.output_dir)

    if not args.skip_verify:
        verify_split(kmodel, enc_path, dec_path)

    print("\n=== EXPORT SUMMARY ===")
    print(f"  encoder: {enc_path}")
    print(f"  decoder: {dec_path}")
    print(f"  Split enables streaming: encoder runs while LLM streams,")
    print(f"  decoder can later be chunked (conv chain is splittable).")


if __name__ == "__main__":
    main()
