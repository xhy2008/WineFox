"""Export Kokoro-82M decoder weights to GGUF format for ggml inference.

This script extracts the `decoder` sub-module from the Kokoro checkpoint
(`kokoro-v1_0.pth`) and writes its weights to a GGUF file that can be
loaded by the C++ ggml decoder implementation under
`voice-test/third_party/kokoro-ggml/`.

Only the decoder is exported; the encoder (PLBERT + text encoder +
duration predictor + F0/N predictor) is kept on ONNX Runtime because:

  1. The encoder is transformer-heavy and ONNX Runtime already optimizes
     it well (small share of total RTF).
  2. The decoder is conv-heavy with ConvTranspose1d + iSTFT and dominates
     inference time; this is where a hand-tuned ggml implementation wins.

The export preserves the original PyTorch tensor names from
`KModel.decoder.state_dict()` so the C++ loader can map them 1:1.

Hyperparameters (kernel sizes, strides, dilation patterns, STFT window
size, etc.) are written as GGUF key/value metadata so the C++ side can
rebuild the graph without parsing config.json at runtime.
"""
import argparse
import json
import os
import sys
import types
import importlib.util

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Load kokoro.model without triggering kokoro/__init__.py (which pulls in
# misaki G2P deps). Same stub trick as test_split_tts.py / quantize_decoder.py.
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

# gguf import (after sys.path tweaks so the import order doesn't matter)
try:
    import gguf
except ImportError:
    print("ERROR: gguf package not installed. Run: pip install gguf")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = r"e:\winefox\voice-test\models\kokoro-82M-src\config.json"
CHECKPOINT  = r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth"
OUT_GGUF    = r"e:\winefox\voice-test\models\kokoro-decoder.gguf"


# ---------------------------------------------------------------------------
# Tensor naming: we keep the original PyTorch state_dict keys verbatim.
# Examples:
#   F0_conv.weight
#   encode.conv1.weight
#   encode.norm1.fc.weight
#   decode.0.conv1.weight
#   generator.ups.0.weight
#   generator.resblocks.0.convs1.0.weight
#   generator.resblocks.0.adain1.0.fc.weight
#   generator.resblocks.0.alpha1.0
#   generator.conv_post.weight
#   generator.m_source.l_linear.weight
#   generator.stft.weight_forward_real
# ---------------------------------------------------------------------------


def load_kokoro_decoder(config_path: str, checkpoint_path: str):
    """Load KModel and return the decoder sub-module with weights loaded."""
    print(f"[load] config:  {config_path}")
    print(f"[load] checkpoint: {checkpoint_path}")
    kmodel = KModel(config=config_path, model=checkpoint_path, disable_complex=True)
    kmodel.eval().cpu()
    return kmodel.decoder


def materialize_state_dict(decoder):
    """Build a state_dict where weight_norm parametrizations are replaced
    by the materialized `.weight` tensor.

    PyTorch's `torch.nn.utils.parametrizations.weight_norm` stores the
    weight as two parameters: `original0` (direction, same shape as weight)
    and `original1` (magnitude, shape [out_channels, 1, 1] for Conv1d).
    The materialized weight is:
        weight = original0 * (original1 / ||original0||_dim)
    where dim=0 by default (norm over in_channels + kernel_size).

    The state_dict() call returns the parametrized form, so we iterate
    over modules and pull the materialized `.weight` attribute directly.
    """
    state = {}
    for name, m in decoder.named_modules():
        # Collect all weight/bias tensors via the public attribute (this
        # automatically applies the parametrization forward hook).
        if hasattr(m, 'weight') and m.weight is not None:
            try:
                w = m.weight.detach().cpu().contiguous()
                state[f"{name}.weight"] = w
            except Exception:
                pass
        if hasattr(m, 'bias') and m.bias is not None:
            try:
                b = m.bias.detach().cpu().contiguous()
                state[f"{name}.bias"] = b
            except Exception:
                pass
        # Parametrized modules also expose `alpha1`/`alpha2` as parameters
        # (used by AdaINResBlock1's snake activation). These are stored
        # normally in state_dict and we catch them below.
    # Now add any non-weight, non-bias parameters (e.g. alpha1, alpha2)
    # and buffers (e.g. stft.window) from the raw state_dict.
    raw = decoder.state_dict()
    for name, tensor in raw.items():
        # Skip weight_norm internals — already materialized above
        if "parametrizations.weight.original" in name:
            continue
        # Skip weight/bias — already collected from module attributes
        # (this avoids overwriting materialized weights with stale values)
        if name.endswith(".weight") or name.endswith(".bias"):
            if name in state:
                continue
        if tensor is None:
            continue
        state[name] = tensor.detach().cpu().contiguous()
    return state


def export_decoder_gguf(decoder, config: dict, out_path: str):
    """Write decoder weights + config metadata to GGUF."""
    state = materialize_state_dict(decoder)
    print(f"\n[export] {len(state)} tensors after materialization")

    writer = gguf.GGUFWriter(
        out_path,
        "kokoro-decoder",
        endianess=gguf.GGUFEndian.LITTLE,
        use_temp_file=False,
    )

    # ----- hyperparameters (mirrors config.json + istftnet section) -----
    istft = config["istftnet"]
    writer.add_uint32("kokoro.dim_in",                 int(config["dim_in"]))
    writer.add_uint32("kokoro.style_dim",              int(config["style_dim"]))
    writer.add_uint32("kokoro.n_mels",                 int(config["n_mels"]))
    writer.add_uint32("kokoro.n_token",                int(config["n_token"]))
    writer.add_uint32("kokoro.upsample_initial_channel", int(istft["upsample_initial_channel"]))
    writer.add_uint32("kokoro.gen_istft_n_fft",        int(istft["gen_istft_n_fft"]))
    writer.add_uint32("kokoro.gen_istft_hop_size",     int(istft["gen_istft_hop_size"]))
    writer.add_uint32("kokoro.num_upsamples",          len(istft["upsample_rates"]))
    writer.add_uint32("kokoro.num_resblocks_per_up",   len(istft["resblock_kernel_sizes"]))

    # arrays — gguf's add_array expects a plain Python sequence of scalars,
    # not a numpy array. Pass lists of ints.
    writer.add_array("kokoro.upsample_rates",
                     [int(x) for x in istft["upsample_rates"]])
    writer.add_array("kokoro.upsample_kernel_sizes",
                     [int(x) for x in istft["upsample_kernel_sizes"]])
    writer.add_array("kokoro.resblock_kernel_sizes",
                     [int(x) for x in istft["resblock_kernel_sizes"]])
    # resblock_dilation_sizes is a list of lists; flatten as int32.
    dil = istft["resblock_dilation_sizes"]
    flat_dil = [int(d) for block in dil for d in block]
    writer.add_array("kokoro.resblock_dilation_sizes_flat", flat_dil)

    # ----- tensors -----
    keys = list(state.keys())
    print(f"[export] sample keys (first 5): {keys[:5]}")
    print(f"[export] sample keys (last  5): {keys[-5:]}")
    # Sanity: verify no weight_norm internals leaked through
    leftover = [k for k in keys if "parametrizations.weight.original" in k]
    if leftover:
        print(f"[export] WARNING: {len(leftover)} weight_norm internals still present")
        print(f"[export]   example: {leftover[0]}")

    n_skipped = 0
    for name, tensor in state.items():
        if tensor is None:
            n_skipped += 1
            continue
        # All decoder weights are float32; we keep them as F32 for now.
        # A future step can quantize to F16/Q8_0 in-file via ggml quantize.
        arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
        # gguf stores tensors with reversed dims vs numpy/torch:
        # torch (out, in, k) -> gguf/ne[0]=k, ne[1]=in, ne[2]=out
        # The GGUFWriter.add_tensor() handles the axis reversal internally
        # based on the numpy array shape, but to be safe we pass the raw
        # numpy shape (which matches torch shape) and let gguf do its thing.
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
    print(f"[export] wrote {len(state) - n_skipped} tensors ({n_skipped} skipped)")

    # total size
    total_bytes = sum(t.numel() * 4 for t in state.values() if t is not None)
    print(f"[export] total weight bytes: {total_bytes/1024/1024:.2f} MB")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(out_path)
    print(f"[export] GGUF written: {out_path} ({file_size/1024/1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output",   default=OUT_GGUF)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    decoder = load_kokoro_decoder(args.config, args.checkpoint)
    export_decoder_gguf(decoder, config, args.output)


if __name__ == "__main__":
    main()
