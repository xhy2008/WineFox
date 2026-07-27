"""End-to-end TTS test with split Kokoro ONNX models.

Validates that encoder.onnx + decoder.onnx produce audible, intelligible
Chinese speech comparable to the original end-to-end model.

Test cases:
  1. Short Chinese sentence
  2. Long Chinese paragraph (multi-sentence)
  3. Compare split-ONNX vs end-to-end PyTorch audio (RMS, spectrogram similarity)

Outputs:
  - WAV files for each test case (split + reference)
  - Quality metrics printed to stdout
"""
import os
import sys
import wave
import time
import types
import importlib.util
import numpy as np
import torch
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Load kokoro.model without triggering KPipeline -> misaki import chain
# (we'll use KPipeline separately for G2P only)
# ---------------------------------------------------------------------------
KOKORO_SRC = r"e:\winefox\voice-test\third_party\kokoro-src"
sys.path.insert(0, KOKORO_SRC)

# Pre-create empty 'kokoro' package
pkg = types.ModuleType("kokoro")
pkg.__path__ = [os.path.join(KOKORO_SRC, "kokoro")]
sys.modules["kokoro"] = pkg

# Load kokoro.model
spec = importlib.util.spec_from_file_location(
    "kokoro.model", os.path.join(KOKORO_SRC, "kokoro", "model.py")
)
_kokoro_model = importlib.util.module_from_spec(spec)
sys.modules["kokoro.model"] = _kokoro_model
spec.loader.exec_module(_kokoro_model)
KModel = _kokoro_model.KModel

# Now we can import KPipeline.
# We only need the zh (Chinese) G2P, so stub out misaki.en/espeak to avoid
# pulling in num2words/spacy/phonemizer dependency chain.
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

# Import split wrappers from export script
sys.path.insert(0, r"e:\winefox\voice-test\scripts")
from export_kokoro_split import KModelEncoder, KModelDecoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = r"e:\winefox\voice-test\models\kokoro-82M-src\config.json"
CHECKPOINT = r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth"
ENC_ONNX = r"e:\winefox\voice-test\models\kokoro-encoder.onnx"
DEC_ONNX = r"e:\winefox\voice-test\models\kokoro-decoder.onnx"
VOICES_DIR = r"e:\winefox\voice-test\models\kokoro-82M-src\voices"
OUT_DIR = r"e:\winefox\voice-test\test-data\results\split_tts"
SAMPLE_RATE = 24000

# Test sentences
TEST_CASES = [
    ("short_zh", "你好，我是酒狐。", "zf_xiaobei"),
    ("long_zh", "今天天气真好，我们去公园散步吧。听说那里的樱花开得很美，再不去看就要凋谢了。", "zf_xiaobei"),
    ("english", "Hello, my name is Kokoro. I am a text to speech model.", "af_heart"),
]


def save_wav(path, audio_f32, sample_rate=SAMPLE_RATE):
    """Save float32 [-1, 1] audio as 16-bit PCM WAV."""
    # Clamp and convert
    audio_clamped = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_clamped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_i16.tobytes())
    print(f"  saved: {path} ({len(audio_i16)/sample_rate:.3f}s)")


def normalize_audio(audio):
    """Normalize audio to [-1, 1] range based on max abs value."""
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        return audio / max_val
    return audio


def synth_endtoend_pytorch(kmodel, pipeline, text, voice, speed=1.0):
    """Synthesize using original end-to-end KModel."""
    results = []
    for graphemes, phonemes, output in pipeline(text, voice=voice, speed=speed):
        if output is not None:
            # KPipeline may yield KModel.Output (has .audio) or raw Tensor
            if hasattr(output, 'audio'):
                audio = output.audio.numpy()
            elif hasattr(output, 'cpu'):
                audio = output.cpu().numpy()
            else:
                audio = np.asarray(output)
            results.append((graphemes, phonemes, audio))
    return results


def synth_split_onnx(enc_sess, dec_sess, pipeline, text, voice, speed=1.0):
    """Synthesize using split ONNX: encoder + decoder.

    Uses KPipeline for G2P (text -> phonemes -> input_ids) and voice
    loading, then runs encoder.onnx + decoder.onnx for each segment.
    """
    results = []
    # Use pipeline in "quiet" mode first to get phonemes, then run ONNX
    # Actually pipeline with model=False yields (graphemes, phonemes, None)
    for graphemes, phonemes, _ in pipeline(text, voice=voice, speed=speed, model=False):
        if not phonemes:
            continue
        # Get input_ids from phonemes (same logic as KPipeline.infer)
        input_ids = list(filter(
            lambda i: i is not None,
            map(lambda p: pipeline.model.vocab.get(p), phonemes)
        ))
        if len(input_ids) + 2 > pipeline.model.context_length:
            print(f"  [warn] truncating {len(input_ids)} -> 510")
            input_ids = input_ids[:510]
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).numpy()

        # Load voice style for this segment
        # pack shape: [N, 256] where N depends on phoneme count.
        # KModel.forward uses ref_s = pack[len(phonemes)-1] which is [256].
        # The ONNX encoder expects ref_s of shape [batch=1, 256].
        pack = pipeline.load_voice(voice).cpu()
        # Debug: print pack shape on first use
        if not hasattr(synth_split_onnx, "_debug_pack"):
            print(f"  [debug] pack shape: {pack.shape}, dtype: {pack.dtype}")
            print(f"  [debug] pack[len-1] shape: {pack[len(phonemes)-1].shape}")
            synth_split_onnx._debug_pack = True
        ref_s = pack[len(phonemes) - 1]  # [256]
        # Ensure 2D [1, 256]
        while ref_s.dim() > 2:
            ref_s = ref_s.squeeze(0)
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)
        ref_s = ref_s.numpy()  # [1, 256]
        print(f"  [debug] ref_s final shape: {ref_s.shape}")

        speed_arr = np.array([speed], dtype=np.float32)

        # Run encoder
        t0 = time.perf_counter()
        enc_out = enc_sess.run(None, {
            "input_ids": input_ids,
            "ref_s": ref_s,
            "speed": speed_arr,
        })
        asr, F0_pred, N_pred, style_dec, pred_dur = enc_out
        t1 = time.perf_counter()
        enc_ms = (t1 - t0) * 1000

        # Run decoder
        t0 = time.perf_counter()
        dec_out = dec_sess.run(None, {
            "asr": asr,
            "F0_pred": F0_pred,
            "N_pred": N_pred,
            "style_dec": style_dec,
        })
        audio = dec_out[0].squeeze()
        t1 = time.perf_counter()
        dec_ms = (t1 - t0) * 1000

        audio_dur_s = len(audio) / SAMPLE_RATE
        rtf = ((enc_ms + dec_ms) / 1000) / audio_dur_s if audio_dur_s > 0 else 0

        print(f"  seg: enc={enc_ms:.1f}ms dec={dec_ms:.1f}ms "
              f"audio={audio_dur_s:.3f}s rtf={rtf:.3f}")

        # Normalize (Kokoro output can exceed [-1, 1])
        audio = normalize_audio(audio)
        results.append((graphemes, phonemes, audio))

    return results


def compute_metrics(audio_a, audio_b):
    """Compute similarity metrics between two audio arrays."""
    # Align lengths (truncate to shorter)
    n = min(len(audio_a), len(audio_b))
    a = audio_a[:n]
    b = audio_b[:n]
    # RMS
    rms_a = np.sqrt(np.mean(a ** 2))
    rms_b = np.sqrt(np.mean(b ** 2))
    # Mean abs error
    mae = np.mean(np.abs(a - b))
    # Cosine similarity
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return {
        "rms_a": rms_a,
        "rms_b": rms_b,
        "rms_ratio": max(rms_a, rms_b) / max(min(rms_a, rms_b), 1e-8),
        "mae": mae,
        "cosine_sim": cos,
        "n_samples": n,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Kokoro Split ONNX TTS Test")
    print("=" * 70)
    print(f"Config:     {CONFIG}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Encoder:    {ENC_ONNX}")
    print(f"Decoder:    {DEC_ONNX}")
    print(f"Voices dir: {VOICES_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print()

    # Check files exist
    for p in [CONFIG, CHECKPOINT, ENC_ONNX, DEC_ONNX]:
        if not os.path.exists(p):
            print(f"[ERROR] missing: {p}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("[1/4] Loading KModel (PyTorch, end-to-end reference)...")
    t0 = time.perf_counter()
    kmodel = KModel(config=CONFIG, model=CHECKPOINT, disable_complex=True)
    kmodel.eval().cpu()
    print(f"  loaded in {time.perf_counter()-t0:.2f}s")

    print("[2/4] Loading KPipeline (zh, for G2P)...")
    t0 = time.perf_counter()
    pipeline = KPipeline(lang_code='z', model=kmodel, device='cpu')
    print(f"  loaded in {time.perf_counter()-t0:.2f}s")

    print("[3/4] Loading encoder ONNX session...")
    t0 = time.perf_counter()
    enc_sess = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    print(f"  loaded in {time.perf_counter()-t0:.2f}s")

    print("[4/4] Loading decoder ONNX session...")
    t0 = time.perf_counter()
    dec_sess = ort.InferenceSession(DEC_ONNX, providers=["CPUExecutionProvider"])
    print(f"  loaded in {time.perf_counter()-t0:.2f}s")
    print()

    # ------------------------------------------------------------------
    # Run test cases
    # ------------------------------------------------------------------
    all_metrics = []
    for name, text, voice in TEST_CASES:
        print("-" * 70)
        print(f"Test: {name}")
        print(f"  text:  {text}")
        print(f"  voice: {voice}")
        print()

        # End-to-end PyTorch reference
        print("  [end-to-end PyTorch]")
        t0 = time.perf_counter()
        e2e_results = synth_endtoend_pytorch(kmodel, pipeline, text, voice)
        e2e_time = time.perf_counter() - t0
        print(f"  total: {e2e_time:.3f}s, {len(e2e_results)} segments")
        print()

        # Split ONNX
        print("  [split ONNX]")
        t0 = time.perf_counter()
        split_results = synth_split_onnx(enc_sess, dec_sess, pipeline, text, voice)
        split_time = time.perf_counter() - t0
        print(f"  total: {split_time:.3f}s, {len(split_results)} segments")
        print()

        # Save WAV files
        if e2e_results:
            e2e_audio = np.concatenate([r[2] for r in e2e_results])
            save_wav(os.path.join(OUT_DIR, f"{name}_e2e.wav"), normalize_audio(e2e_audio))
        if split_results:
            split_audio = np.concatenate([r[2] for r in split_results])
            save_wav(os.path.join(OUT_DIR, f"{name}_split.wav"), split_audio)

        # Compare (only if same number of segments)
        if len(e2e_results) == len(split_results) and e2e_results:
            print("  [metrics]")
            for i, ((g1, p1, a_e2e), (g2, p2, a_split)) in enumerate(
                zip(e2e_results, split_results)
            ):
                # Normalize both to [-1, 1] for fair comparison
                a_e2e_n = normalize_audio(a_e2e)
                a_split_n = a_split  # already normalized
                m = compute_metrics(a_e2e_n, a_split_n)
                all_metrics.append((name, i, m))
                print(f"  seg {i}: rms_e2e={m['rms_a']:.4f} rms_split={m['rms_b']:.4f} "
                      f"ratio={m['rms_ratio']:.3f} cos_sim={m['cosine_sim']:.4f} "
                      f"mae={m['mae']:.4f}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_metrics:
        cos_sims = [m["cosine_sim"] for _, _, m in all_metrics]
        rms_ratios = [m["rms_ratio"] for _, _, m in all_metrics]
        print(f"  segments tested:    {len(all_metrics)}")
        print(f"  cosine similarity:  mean={np.mean(cos_sims):.4f} "
              f"min={np.min(cos_sims):.4f} max={np.max(cos_sims):.4f}")
        print(f"  RMS ratio:          mean={np.mean(rms_ratios):.3f} "
              f"min={np.min(rms_ratios):.3f} max={np.max(rms_ratios):.3f}")
        print()
        if np.mean(cos_sims) > 0.5:
            print("  [PASS] Split ONNX produces audio similar to end-to-end model")
            print("  (Low cosine similarity is expected due to SineGen random noise,")
            print("   but audio quality/intelligibility should be preserved.)")
        else:
            print("  [WARN] Cosine similarity low - check audio files manually")
    print()
    print(f"WAV files saved to: {OUT_DIR}")
    print("  Listen to *_e2e.wav (reference) vs *_split.wav (split ONNX)")
    print("  to verify audio quality is preserved.")


if __name__ == "__main__":
    main()
