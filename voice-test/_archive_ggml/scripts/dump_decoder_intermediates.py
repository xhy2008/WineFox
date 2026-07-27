"""Dump intermediate decoder outputs for C++ comparison.

Runs the Kokoro decoder step by step, capturing the output of each major
operation (F0_conv, encode, decode blocks, generator ups layers, conv_post).
Saves them as .npy files so the C++ side can compare.

Usage:
    python dump_decoder_intermediates.py --text "你好" --voice zf_xiaobei
"""
import argparse
import os
import sys
import types
import importlib.util

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Load kokoro.model + KPipeline (same stub trick as dump_decoder_inputs.py)
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

CONFIG_PATH    = r"e:\winefox\voice-test\models\kokoro-82M-src\config.json"
CHECKPOINT     = r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth"
VOICES_DIR     = r"e:\winefox\voice-test\models\kokoro-82M-src\voices"
DEFAULT_VOICE  = "zf_xiaobei"
DEFAULT_TEXT   = "你好"
SAMPLE_RATE    = 24000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",     default=DEFAULT_TEXT)
    parser.add_argument("--voice",    default=DEFAULT_VOICE)
    parser.add_argument("--config",   default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--voices-dir", default=VOICES_DIR)
    parser.add_argument("--out-dir",
                        default=r"e:\winefox\voice-test\test-data\decoder_inter")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[dump] loading KModel(config={args.config}, model={args.checkpoint})")
    kmodel = KModel(config=args.config, model=args.checkpoint, disable_complex=True)
    kmodel.eval().cpu()

    # Make SineGen deterministic (same patch as dump_decoder_inputs.py)
    sine_gen = kmodel.decoder.generator.m_source.l_sin_gen
    def deterministic_f02sine(f0_values):
        import torch.nn.functional as _F
        rad_values = (f0_values / sine_gen.sampling_rate) % 1
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
            raise NotImplementedError
        return sines
    sine_gen._f02sine = deterministic_f02sine
    print("[dump] patched SineGen for deterministic output")

    # Load voice pack
    voice_path = os.path.join(args.voices_dir, f"{args.voice}.pt")
    pack = torch.load(voice_path, weights_only=True).cpu()

    pipeline = KPipeline(lang_code=args.voice[0], model=kmodel, repo_id=None)
    pipeline.voices[args.voice] = pack
    pipeline.voices[voice_path] = pack

    # Hook every submodule to capture inputs/outputs
    captured = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            captured[name] = {
                "input": [i.detach().cpu() if isinstance(i, torch.Tensor) else i for i in inp],
                "output": out.detach().cpu() if isinstance(out, torch.Tensor) else out,
            }
        return hook

    # Register hooks on key modules
    dec = kmodel.decoder
    targets = [
        ("F0_conv", dec.F0_conv),
        ("N_conv", dec.N_conv),
        ("asr_res", dec.asr_res),
        ("encode", dec.encode),
        ("decode.0", dec.decode[0]),
        ("decode.1", dec.decode[1]),
        ("decode.2", dec.decode[2]),
        ("decode.3", dec.decode[3]),
        ("gen.ups.0", dec.generator.ups[0]),
        ("gen.ups.1", dec.generator.ups[1]),
        ("gen.noise_convs.0", dec.generator.noise_convs[0]),
        ("gen.noise_convs.1", dec.generator.noise_convs[1]),
        ("gen.conv_post", dec.generator.conv_post),
    ]
    for name, mod in targets:
        hooks.append(mod.register_forward_hook(make_hook(name)))

    # Also capture the conv_post output (pre-exp/sin) and the spec/phase
    orig_gen_forward = dec.generator.forward
    gen_captured = {}
    def capturing_gen_forward(x, s, f0):
        # Run the generator manually to capture intermediates
        import torch.nn.functional as F
        import math
        with torch.no_grad():
            f0_up = dec.generator.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, noi_source, uv = dec.generator.m_source(f0_up)
            har_source_sq = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = dec.generator.stft.transform(har_source_sq)
            har = torch.cat([har_spec, har_phase], dim=1)
        gen_captured["har"] = har.detach().cpu()

        # Run the generator's forward loop manually
        gx = x
        for i in range(dec.generator.num_upsamples):
            gx = F.leaky_relu(gx, 0.1)
            x_source = dec.generator.noise_convs[i](har)
            x_source = dec.generator.noise_res[i](x_source, s)
            gx = dec.generator.ups[i](gx)
            if i == dec.generator.num_upsamples - 1:
                gx = dec.generator.reflection_pad(gx)
            gx = gx + x_source
            xs = None
            for j in range(dec.generator.num_kernels):
                rb = dec.generator.resblocks[i * dec.generator.num_kernels + j](gx, s)
                xs = rb if xs is None else xs + rb
            gx = xs / dec.generator.num_kernels
            gen_captured[f"gen.after_ups.{i}"] = gx.detach().cpu()

        gx = F.leaky_relu(gx)
        conv_post_out = dec.generator.conv_post(gx)
        gen_captured["gen.conv_post_out"] = conv_post_out.detach().cpu()

        post_n_fft = dec.generator.post_n_fft
        spec = torch.exp(conv_post_out[:, :post_n_fft // 2 + 1, :])
        phase = torch.sin(conv_post_out[:, post_n_fft // 2 + 1:, :])
        gen_captured["gen.spec"] = spec.detach().cpu()
        gen_captured["gen.phase"] = phase.detach().cpu()

        audio = dec.generator.stft.inverse(spec, phase)
        gen_captured["gen.audio"] = audio.detach().cpu()
        return audio

    dec.generator.forward = capturing_gen_forward

    # Run pipeline
    for gs, ps, chunk_audio in pipeline(args.text, voice=args.voice, speed=1.0):
        if chunk_audio.numel() > 0:
            break

    # Save all captured intermediates
    print(f"\n[dump] captured {len(captured)} module outputs")
    print(f"[dump] captured {len(gen_captured)} generator intermediates")

    for name, data in captured.items():
        out = data["output"]
        if isinstance(out, torch.Tensor):
            arr = out.numpy().astype(np.float32)
            # Squeeze batch dim if present
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            # Transpose from [C, T] to [T, C] for consistency with C++ layout
            if arr.ndim == 2:
                arr = arr.T.copy()
            print(f"  {name}: shape {arr.shape}  rms={np.sqrt(np.mean(arr**2)):.6f}")
            np.save(os.path.join(args.out_dir, f"{name}.npy"), arr)

    for name, data in gen_captured.items():
        if isinstance(data, torch.Tensor):
            arr = data.numpy().astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2:
                arr = arr.T.copy()
            print(f"  {name}: shape {arr.shape}  rms={np.sqrt(np.mean(arr**2)):.6f}")
            np.save(os.path.join(args.out_dir, f"{name}.npy"), arr)

    print(f"\n[dump] wrote {len(captured) + len(gen_captured)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
