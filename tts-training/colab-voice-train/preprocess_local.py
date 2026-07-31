"""本地 CPU 预处理：把 BERT/text_encoder 输出和音频缓存为 pkl。

Colab 只需加载 pkl 即可开始训练，跳过 BERT 预计算（GPU 利用率为 0 的瓶颈）。

用法：
    python preprocess_local.py
    python preprocess_local.py --limit 100   # 只处理前 100 条（调试用）

输出：
    dataset/preprocessed.pkl  ← 上传到 Colab
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import torch
import torchaudio

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from kokoro import KPipeline, KModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")
OUTPUT_PATH = os.path.join(DATASET_DIR, "preprocessed.pkl")

SAMPLE_RATE = 24000


def main():
    parser = argparse.ArgumentParser(description="本地预处理 BERT + 音频")
    parser.add_argument("--limit", type=int, default=None, help="限制处理样本数")
    args = parser.parse_args()

    print("=" * 60)
    print("本地预处理：BERT + text_encoder + 音频 → pkl")
    print("=" * 60)

    # 1. 加载模型（CPU）
    print("\n[1/3] 加载 Kokoro 模型（CPU）...")
    device = torch.device('cpu')
    model = KModel().to(device)
    pipeline = KPipeline(lang_code='z', model=False)
    # 冻结参数，不调用 eval()（保持 training 模式，不影响 CPU 前向）
    for param in model.parameters():
        param.requires_grad = False
    print("  模型加载完成")

    # 2. 加载 manifest
    print("\n[2/3] 加载 manifest...")
    records = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append({
                "text": rec["text"],
                "audio_path": os.path.join(DATASET_DIR, rec["audio"]),
            })
    if args.limit:
        records = records[:args.limit]
    print(f"  {len(records)} 条数据")

    # 3. 预处理
    print(f"\n[3/3] 预处理（G2P + BERT + text_encoder + 音频）...")
    samples = []
    t0 = time.time()

    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(records) - i - 1)
            print(f"  [{i+1}/{len(records)}] 已用 {elapsed:.0f}s, 预计剩余 {eta:.0f}s")

        try:
            # G2P
            ps, _ = pipeline.g2p(rec["text"])
            if not ps:
                continue
            if len(ps) > 510:
                ps = ps[:510]

            # input_ids
            input_ids_list = list(filter(
                lambda x: x is not None,
                map(lambda p: model.vocab.get(p), ps)
            ))
            if len(input_ids_list) < 2:
                continue
            input_ids = torch.LongTensor([[0, *input_ids_list, 0]]).to(device)

            # BERT + text_encoder 前向（不依赖 style，输出固定）
            with torch.no_grad():
                input_lengths = torch.full(
                    (input_ids.shape[0],), input_ids.shape[-1],
                    device=device, dtype=torch.long
                )
                text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
                    input_lengths.shape[0], -1
                ).type_as(input_lengths)
                text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)
                bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
                d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                t_en = model.text_encoder(input_ids, input_lengths, text_mask)
                n_tokens = input_ids.shape[1]

            # 音频加载 + 重采样
            target_audio, sr = torchaudio.load(rec["audio_path"])
            if sr != SAMPLE_RATE:
                target_audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(target_audio)
            target_audio = target_audio.squeeze(0)

            # 转 numpy 存盘（float32 节省空间，Colab 端转回 tensor）
            samples.append({
                'd_en': d_en.cpu().numpy().astype(np.float32),
                't_en': t_en.cpu().numpy().astype(np.float32),
                'input_lengths': input_lengths.cpu().numpy().astype(np.int64),
                'text_mask': text_mask.cpu().numpy().astype(bool),
                'n_tokens': int(n_tokens),
                'target_audio': target_audio.cpu().numpy().astype(np.float32),
            })
        except Exception as e:
            print(f"  跳过 #{i}: {e}")
            continue

    elapsed = time.time() - t0

    # 保存 pkl
    print(f"\n预处理完成: {len(samples)}/{len(records)} 个样本, 耗时 {elapsed:.1f}s")
    print(f"保存到: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"文件大小: {size_mb:.1f} MB")
    print(f"\n下一步: 把 dataset/preprocessed.pkl 上传到 Colab")
    print(f"        Colab 端运行: python train_colab.py --preprocessed dataset/preprocessed.pkl")


if __name__ == "__main__":
    main()
