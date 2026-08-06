"""Check how the encoder assigns duration to punctuation tokens.

Compares pred_dur for text with and without trailing punctuation.
If the model assigns significant duration to punctuation tokens,
that explains the "unrecognizable phoneme" at punctuation positions.
"""
import os
import sys
import numpy as np
import onnxruntime as ort

ENC_ONNX = r"e:\winefox\voice-test\models\kokoro-encoder.onnx"
VOICES_BIN = r"e:\winefox\voice-test\models\winefox_voices.bin"
VOCAB_TXT = r"e:\winefox\voice-test\third_party\kokoro-cpp-src\dict\vocab.txt"


def load_voice_style(voices_path, name):
    with open(voices_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"VOIC"
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
            f.seek(dim * 4, 1)
    raise ValueError(f"voice {name} not found")


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            tab = line.find("\t")
            if tab != -1:
                token = line[:tab]
                token = token.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
                try:
                    vocab[token] = int(line[tab+1:])
                except ValueError:
                    pass
    return vocab


def run_encoder(enc_sess, phonemes, voice_style, vocab, speed=1.0):
    # Tokenize: filter unknown chars, add BOS/EOS
    tokens = [vocab[c] for c in phonemes if c in vocab]
    input_ids = np.array([[0] + tokens + [0]], dtype=np.int64)

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
    return outputs


def main():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    enc_sess = ort.InferenceSession(ENC_ONNX, so, providers=["CPUExecutionProvider"])

    voice_style = load_voice_style(VOICES_BIN, "winefox")
    vocab = load_vocab(VOCAB_TXT)

    # Reverse vocab for display
    id2tok = {v: k for k, v in vocab.items()}

    # Test: "你好，世界。" with and without punctuation
    tests = [
        ("你好，世界。", "with punct"),
        ("你好世界", "no punct"),
        ("你好，", "frag: 你好，"),
        ("世界。", "frag: 世界。"),
    ]

    for phonemes, label in tests:
        outs = run_encoder(enc_sess, phonemes, voice_style, vocab)
        pred_dur = outs[4].flatten()  # (T,)
        tokens = [vocab[c] for c in phonemes if c in vocab]
        token_ids = [0] + tokens + [0]  # with BOS/EOS

        print(f"\n[{label}] phonemes={repr(phonemes)}")
        print(f"  token_ids: {token_ids}")
        print(f"  pred_dur:  {pred_dur.tolist()}")
        print(f"  total_dur: {pred_dur.sum()} ({pred_dur.sum()*600/24000:.2f}s audio)")
        print(f"  per-token:")
        for i, (tid, dur) in enumerate(zip(token_ids, pred_dur)):
            tok_char = id2tok.get(tid, '?')
            print(f"    [{i}] id={tid:3d} {repr(tok_char):>8s}  dur={dur:.1f}")


if __name__ == "__main__":
    main()
