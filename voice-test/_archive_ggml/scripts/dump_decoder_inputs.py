"""Dump Kokoro decoder inputs and reference audio for C++ ggml benchmark.

Runs the original Kokoro model in Python, intercepts the decoder.forward()
call to capture its inputs (asr, F0_pred, N_pred, style_dec) and saves them
to a .npz file. Also writes the reference audio as a 24 kHz mono WAV.

The C++ benchmark (src/tts_ggml_test.cpp) loads this .npz, runs the ggml
decoder, and compares its output against the reference audio.

Layout (matches the C API in kokoro_decoder.h):
    asr       : [T_frm, 512]  (PyTorch [1, 512, T_frm] transposed to row-major T x C)
    F0_pred   : [T_frm]
    N_pred    : [T_frm]
    style_dec : [128]
    audio     : [T_audio]     (24 kHz float32 reference output)

Usage:
    python dump_decoder_inputs.py --text "你好，世界" \\
        --voice zf_xiaobei --out-dir test-data/decoder_inputs
"""
import argparse
import os
import sys
import types
import importlib.util
import wave

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Load kokoro.model + KPipeline without triggering the full misaki import
# chain. Same stub trick as test_split_tts.py.
# ---------------------------------------------------------------------------
KOKORO_SRC = r"e:\winefox\voice-test\third_party\kokoro-src"
sys.path.insert(0, KOKORO_SRC)

# Pre-create empty 'kokoro' package so submodules can be loaded piecemeal.
pkg = types.ModuleType("kokoro")
pkg.__path__ = [os.path.join(KOKORO_SRC, "kokoro")]
sys.modules["kokoro"] = pkg

# Load kokoro.model (needed by KPipeline at import time).
spec = importlib.util.spec_from_file_location(
    "kokoro.model", os.path.join(KOKORO_SRC, "kokoro", "model.py")
)
_kokoro_model = importlib.util.module_from_spec(spec)
sys.modules["kokoro.model"] = _kokoro_model
spec.loader.exec_module(_kokoro_model)
KModel = _kokoro_model.KModel

# Stub out misaki.en / misaki.espeak to skip num2words/spacy/phonemizer deps
# when we only need the zh (Chinese) G2P path.
import misaki
if not hasattr(misaki, 'en'):
    en_stub = types.ModuleType("misaki.en")
    en_stub.MToken = object  # placeholder
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
CONFIG_PATH    = r"e:\winefox\voice-test\models\kokoro-82M-src\config.json"
CHECKPOINT     = r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth"
VOICES_DIR     = r"e:\winefox\voice-test\models\kokoro-82M-src\voices"
DEFAULT_VOICE  = "zf_xiaobei"
DEFAULT_TEXT   = "你好，这是一个 ggml 推理 Kokoro 的测试。"
SAMPLE_RATE    = 24000


def save_wav(path, audio_f32, sample_rate=SAMPLE_RATE):
    """Save float32 [-1, 1] audio as 16-bit PCM WAV."""
    audio_clamped = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_clamped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_i16.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",     default=DEFAULT_TEXT,
                        help="Text to synthesize (will be phonemized by KPipeline).")
    parser.add_argument("--voice",    default=DEFAULT_VOICE,
                        help="Voice name (e.g. zf_xiaobei). Must exist as <voices_dir>/<voice>.pt.")
    parser.add_argument("--config",   default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--voices-dir", default=VOICES_DIR)
    parser.add_argument("--out-dir",  default=r"e:\winefox\voice-test\test-data\decoder_inputs",
                        help="Where to write decoder_inputs.npz and reference.wav.")
    parser.add_argument("--speed",    type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    print(f"[dump] loading KModel(config={args.config}, model={args.checkpoint})")
    kmodel = KModel(config=args.config, model=args.checkpoint, disable_complex=True)
    kmodel.eval().cpu()

    # ------------------------------------------------------------------
    # Make SineGen deterministic so the C++ ggml implementation (which
    # has no random initial phase and uses a fixed noise seed) can be
    # compared fairly against this Python reference.
    #
    # 1. Patch SineGen._f02sine to set rand_ini=0 (no random initial
    #    phase for harmonics 1-8).
    # 2. Use torch.manual_seed before forward so torch.randn_like noise
    #    is deterministic.
    # ------------------------------------------------------------------
    sine_gen = kmodel.decoder.generator.m_source.l_sin_gen
    orig_f02sine = sine_gen._f02sine

    def deterministic_f02sine(f0_values):
        # Copy the original logic but with rand_ini = 0
        from kokoro import istftnet as _istftnet_mod
        import torch.nn.functional as _F
        rad_values = (f0_values / sine_gen.sampling_rate) % 1
        # NO rand_ini — deterministic
        if not sine_gen.flag_for_pulse:
            rad_values = _F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=1.0 / sine_gen.upsample_scale,
                mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = _F.interpolate(
                phase.transpose(1, 2) * sine_gen.upsample_scale,
                scale_factor=sine_gen.upsample_scale,
                mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            raise NotImplementedError("flag_for_pulse path not supported in deterministic patch")
        return sines

    sine_gen._f02sine = deterministic_f02sine
    print("[dump] patched SineGen._f02sine for deterministic output (rand_ini=0)")

    # Load voice pack (.pt file is a [510, 1, 256] tensor; we pre-populate
    # the pipeline's voice cache so KPipeline doesn't try to HF-download).
    voice_path = os.path.join(args.voices_dir, f"{args.voice}.pt")
    if not os.path.exists(voice_path):
        print(f"[dump] ERROR: voice pack not found: {voice_path}")
        sys.exit(1)
    print(f"[dump] loading voice pack: {voice_path}")
    pack = torch.load(voice_path, weights_only=True).cpu()
    print(f"[dump] pack shape: {tuple(pack.shape)}  dtype: {pack.dtype}")

    # Build pipeline (lang_code from voice prefix, e.g. 'z' for zf_xiaobei).
    print(f"[dump] phonemizing text: {args.text!r}")
    pipeline = KPipeline(lang_code=args.voice[0], model=kmodel, repo_id=None)

    # Pre-populate the voice cache so KPipeline.load_voice doesn't HF-download.
    # KPipeline.load_voice splits by ',' and calls load_single_voice for each;
    # load_single_voice checks `voice in self.voices` first. We register the
    # pack under the voice name and also under the path for safety.
    pipeline.voices[args.voice] = pack
    pipeline.voices[voice_path] = pack

    # Monkey-patch kmodel.decoder.forward to capture inputs.
    captured = {}
    orig_forward = kmodel.decoder.forward

    # Also patch generator.forward to capture har_source and har (the STFT
    # of the sine source) so the C++ side can verify its precompute_sine_source.
    orig_gen_forward = kmodel.decoder.generator.forward

    def capturing_gen_forward(x, s, f0):
        with torch.no_grad():
            f0_up = kmodel.decoder.generator.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, noi_source, uv = kmodel.decoder.generator.m_source(f0_up)
            har_source_sq = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = kmodel.decoder.generator.stft.transform(har_source_sq)
            har = torch.cat([har_spec, har_phase], dim=1)
        captured["har_source"] = har_source_sq.detach().cpu().contiguous()
        captured["har"]        = har.detach().cpu().contiguous()
        captured["sine_waves"] = kmodel.decoder.generator.m_source.l_sin_gen._sine_waves_cache if hasattr(kmodel.decoder.generator.m_source.l_sin_gen, '_sine_waves_cache') else None
        # Now call the original generator forward (which will recompute har, but that's fine)
        return orig_gen_forward(x, s, f0)

    def capturing_forward(asr, F0_curve, N, s):
        captured["asr"]      = asr.detach().cpu().contiguous()
        captured["F0_curve"] = F0_curve.detach().cpu().contiguous()
        captured["N"]        = N.detach().cpu().contiguous()
        captured["s"]        = s.detach().cpu().contiguous()
        # Set deterministic seed for SineGen noise (torch.randn_like in forward)
        torch.manual_seed(12345)
        out = orig_forward(asr, F0_curve, N, s)
        captured["audio"]    = out.detach().cpu().contiguous()
        return out

    kmodel.decoder.generator.forward = capturing_gen_forward
    kmodel.decoder.forward = capturing_forward

    # Run pipeline (single chunk is enough for the benchmark).
    chunk_count = 0
    for gs, ps, chunk_audio in pipeline(args.text, voice=args.voice, speed=args.speed):
        chunk_count += 1
        print(f"[dump] chunk {chunk_count}: graphemes={gs!r} phonemes={ps!r} "
              f"audio_shape={tuple(chunk_audio.shape)}")
        if chunk_audio.numel() > 0:
            break

    kmodel.decoder.forward = orig_forward

    if "audio" not in captured:
        print(f"[dump] ERROR: decoder.forward was never called.")
        sys.exit(1)

    # Extract captured tensors
    asr      = captured["asr"]       # [B, 512, T_frm]
    F0_curve = captured["F0_curve"]  # [B, T_frm]
    N        = captured["N"]         # [B, T_frm]
    s        = captured["s"]         # [B, 128]
    audio    = captured["audio"]     # [B, 1, T_audio] or [1, T_audio]

    # Squeeze batch dim
    asr      = asr.squeeze(0)      # [512, T_frm]
    F0_curve = F0_curve.squeeze(0) # [T_frm]
    N        = N.squeeze(0)        # [T_frm]
    s        = s.squeeze(0)        # [128]
    audio    = audio.squeeze()     # [T_audio]

    print(f"[dump] asr       shape: {tuple(asr.shape)}  dtype: {asr.dtype}")
    print(f"[dump] F0_curve  shape: {tuple(F0_curve.shape)}")
    print(f"[dump] N         shape: {tuple(N.shape)}")
    print(f"[dump] s         shape: {tuple(s.shape)}")
    print(f"[dump] audio     shape: {tuple(audio.shape)}  "
          f"duration: {audio.numel()/SAMPLE_RATE:.3f}s")

    # Transpose asr to [T_frm, 512] (row-major, matches C++ API expectation).
    # PyTorch asr is [512, T_frm] (channel, time). We want [T_frm, 512].
    asr_tx = asr.transpose(0, 1).contiguous()  # [T_frm, 512]
    print(f"[dump] asr (transposed) shape: {tuple(asr_tx.shape)}")

    # Extract har_source and har for C++ comparison
    har_source = captured.get("har_source")  # [T_audio]
    har        = captured.get("har")         # [B, 2*freq_bins, T_frames] -> [22, T_frames]
    if har_source is not None:
        har_source = har_source.squeeze().cpu().contiguous()
        print(f"[dump] har_source shape: {tuple(har_source.shape)}")
    if har is not None:
        har = har.squeeze(0).cpu().contiguous()  # [22, T_frames]
        print(f"[dump] har        shape: {tuple(har.shape)}")

    # Save as .npz (single file, convenient for Python) AND as individual
    # .npy files (easier to parse from C++ without a ZIP/deflate reader).
    npz_path = os.path.join(args.out_dir, "decoder_inputs.npz")
    savez_kwargs = dict(
        asr=asr_tx.numpy().astype(np.float32),       # [T_frm, 512]
        F0_pred=F0_curve.numpy().astype(np.float32), # [T_frm]
        N_pred=N.numpy().astype(np.float32),         # [T_frm]
        style_dec=s.numpy().astype(np.float32),      # [128]
        audio=audio.numpy().astype(np.float32),      # [T_audio]
        sample_rate=np.int32(SAMPLE_RATE),
    )
    if har_source is not None:
        savez_kwargs["har_source"] = har_source.numpy().astype(np.float32)
    if har is not None:
        # Transpose har to [T_frames, 22] to match C++ ggml layout [T, C, 1]
        har_tx = har.transpose(0, 1).contiguous()  # [T_frames, 22]
        savez_kwargs["har"] = har_tx.numpy().astype(np.float32)
    np.savez(npz_path, **savez_kwargs)
    print(f"[dump] wrote {npz_path} ({os.path.getsize(npz_path)/1024:.1f} KB)")

    # Individual .npy files for C++ consumption.
    npy_dir = os.path.join(args.out_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    npy_items = [
        ("asr",       asr_tx.numpy().astype(np.float32)),
        ("F0_pred",   F0_curve.numpy().astype(np.float32)),
        ("N_pred",    N.numpy().astype(np.float32)),
        ("style_dec", s.numpy().astype(np.float32)),
        ("audio",     audio.numpy().astype(np.float32)),
    ]
    if har_source is not None:
        npy_items.append(("har_source", har_source.numpy().astype(np.float32)))
    if har is not None:
        har_tx = har.transpose(0, 1).contiguous()  # [T_frames, 22]
        npy_items.append(("har", har_tx.numpy().astype(np.float32)))
    for name, arr in npy_items:
        np.save(os.path.join(npy_dir, f"{name}.npy"), arr)
    print(f"[dump] wrote individual .npy files to {npy_dir}")

    # Also save audio as WAV for human verification.
    wav_path = os.path.join(args.out_dir, "reference.wav")
    save_wav(wav_path, audio.numpy().astype(np.float32), SAMPLE_RATE)
    print(f"[dump] wrote {wav_path} ({audio.numel()/SAMPLE_RATE:.3f}s)")

    print(f"[dump] done. Use with:")
    print(f"  voice_test tts-ggml --model models/kokoro-decoder.gguf "
          f"--inputs {npy_dir} --reference {wav_path}")


if __name__ == "__main__":
    main()
