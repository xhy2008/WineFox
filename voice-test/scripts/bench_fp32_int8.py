"""Quick benchmark: FP32 decoder with 8 threads (no quantization)."""
import os, sys, time, types, importlib.util
import numpy as np
import torch
import onnxruntime as ort

KOKORO_SRC = r"e:\winefox\voice-test\third_party\kokoro-src"
sys.path.insert(0, KOKORO_SRC)
pkg = types.ModuleType("kokoro")
pkg.__path__ = [os.path.join(KOKORO_SRC, "kokoro")]
sys.modules["kokoro"] = pkg
spec = importlib.util.spec_from_file_location(
    "kokoro.model", os.path.join(KOKORO_SRC, "kokoro", "model.py")
)
_m = importlib.util.module_from_spec(spec)
sys.modules["kokoro.model"] = _m
spec.loader.exec_module(_m)
KModel = _m.KModel

import misaki
if not hasattr(misaki, 'en'):
    s = types.ModuleType("misaki.en"); s.MToken = object
    sys.modules["misaki.en"] = s; misaki.en = s
if not hasattr(misaki, 'espeak'):
    s = types.ModuleType("misaki.espeak")
    s.EspeakWrapper = object; s.EspeakBackend = object
    sys.modules["misaki.espeak"] = s; misaki.espeak = s

from kokoro.pipeline import KPipeline

CONFIG = r"e:\winefox\voice-test\models\kokoro-82M-src\config.json"
CHECKPOINT = r"e:\winefox\voice-test\models\kokoro-82M-src\kokoro-v1_0.pth"
ENC_ONNX = r"e:\winefox\voice-test\models\kokoro-encoder.onnx"
DEC_ONNX = r"e:\winefox\voice-test\models\kokoro-decoder.onnx"
DEC_INT8_STATIC = r"e:\winefox\voice-test\models\kokoro-decoder-int8-static.onnx"

TEST_SENTENCES = [
    ("short", "你好，我是酒狐。", "zf_xiaobei"),
    ("medium", "今天天气真好，我们去公园散步吧。", "zf_xiaobei"),
    ("long", "今天天气真好，我们去公园散步吧。听说那里的樱花开得很美，再不去看就要凋谢了。", "zf_xiaobei"),
]

def get_inputs(pipeline, text, voice):
    results = []
    for graphemes, phonemes, _ in pipeline(text, voice=voice, model=False):
        if not phonemes: continue
        input_ids = list(filter(lambda i: i is not None,
            map(lambda p: pipeline.model.vocab.get(p), phonemes)))
        if len(input_ids) + 2 > pipeline.model.context_length:
            input_ids = input_ids[:510]
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).numpy()
        pack = pipeline.load_voice(voice).cpu()
        ref_s = pack[len(phonemes) - 1]
        while ref_s.dim() > 2: ref_s = ref_s.squeeze(0)
        if ref_s.dim() == 1: ref_s = ref_s.unsqueeze(0)
        ref_s = ref_s.numpy()
        results.append((input_ids, ref_s, np.array([1.0], dtype=np.float32)))
    return results

def main():
    print("=" * 80)
    print("FP32 vs INT8-static, thread sweep")
    print("=" * 80)

    kmodel = KModel(config=CONFIG, model=CHECKPOINT, disable_complex=True)
    kmodel.eval().cpu()
    pipeline = KPipeline(lang_code='z', model=kmodel, device='cpu')
    enc_sess = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])

    all_inputs, labels = [], []
    for name, text, voice in TEST_SENTENCES:
        for inp in get_inputs(pipeline, text, voice):
            all_inputs.append(inp)
            labels.append(name)

    # Pre-compute encoder outputs
    dec_inputs_list = []
    for input_ids, ref_s, speed in all_inputs:
        enc_out = enc_sess.run(None, {"input_ids": input_ids, "ref_s": ref_s, "speed": speed})
        asr, F0_pred, N_pred, style_dec, _ = enc_out
        dec_inputs_list.append({"asr": asr, "F0_pred": F0_pred, "N_pred": N_pred, "style_dec": style_dec})

    # Sweep: (model_path, threads) combinations
    configs = []
    for threads in [4, 8]:
        configs.append((f"fp32_t{threads}", DEC_ONNX, threads))
        configs.append((f"int8_t{threads}", DEC_INT8_STATIC, threads))

    print(f"\n{'Test':<10}{'Audio(s)':<10}", end="")
    for name, _, _ in configs:
        print(f"{name+'(ms)':<16}", end="")
    print(f"{'int8_speedup':<14}")
    print("-" * 90)

    for dec_inputs, label in zip(dec_inputs_list, labels):
        audio_dur = 0
        times = []
        for name, path, threads in configs:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            sess.run(None, dec_inputs)  # warmup
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                out = sess.run(None, dec_inputs)
                ts.append((time.perf_counter() - t0) * 1000)
            med = np.median(ts)
            times.append(med)
            if audio_dur == 0:
                audio_dur = len(out[0].squeeze()) / 24000

        # int8_t8 vs fp32_t4
        speedup = times[0] / times[3]  # fp32_t4 / int8_t8
        print(f"{label:<10}{audio_dur:<10.3f}", end="")
        for t in times:
            print(f"{t:<16.1f}", end="")
        print(f"{speedup:.2f}x")

    # Detailed RTF for best config
    print(f"\n{'='*80}")
    print(f"Detailed: FP32 t=8 vs INT8 t=8")
    print(f"{'='*80}")
    print(f"{'Test':<10}{'Audio(s)':<10}{'fp32_t8(ms)':<14}{'int8_t8(ms)':<14}{'fp32_rtf':<10}{'int8_rtf':<10}{'speedup':<10}")
    print("-" * 80)
    for dec_inputs, label in zip(dec_inputs_list, labels):
        results_row = {}
        for name, path, threads in [("fp32_t8", DEC_ONNX, 8), ("int8_t8", DEC_INT8_STATIC, 8)]:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            sess.run(None, dec_inputs)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                out = sess.run(None, dec_inputs)
                ts.append((time.perf_counter() - t0) * 1000)
            med = np.median(ts)
            audio_dur = len(out[0].squeeze()) / 24000
            results_row[name] = (med, audio_dur)

        fp32_ms, audio_dur = results_row["fp32_t8"]
        int8_ms, _ = results_row["int8_t8"]
        fp32_rtf = (fp32_ms / 1000) / audio_dur
        int8_rtf = (int8_ms / 1000) / audio_dur
        speedup = fp32_ms / int8_ms
        print(f"{label:<10}{audio_dur:<10.3f}{fp32_ms:<14.1f}{int8_ms:<14.1f}{fp32_rtf:<10.3f}{int8_rtf:<10.3f}{speedup:.2f}x")

if __name__ == "__main__":
    main()
