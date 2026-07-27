#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice2 跨语言音色克隆数据集生成脚本

用途（PLAN.md 5.2 节）：用 CosyVoice2 教师模型将日语音色克隆到中文，
生成酒狐音色的平行语料，供 Kokoro voice style 微调使用。

两种运行模式：
  1. 本地小样本（CPU 验证）：
     python generate.py --local --batch 20
     生成 20 条样本，用于验证训练流程

  2. Colab 大规模生成（GPU）：
     python generate.py --colab --batch 6727
     生成完整数据集

前置条件：
  - 已运行 tts-training/cosyvoice_setup.py 部署 CosyVoice2-0.5B
  - 有 ref3_clean.wav 日语参考音频
  - 有 prompt_text_ref3_clean.txt 参考音频转写文本
  - 有 corpus.txt 中文语料

输出目录结构：
  dataset/
    wavs/
      000000.wav
      000001.wav
      ...
    metadata.csv   # 格式: id|text|phonemes|duration
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
TTS_TRAINING = HERE.parent
DEPLOY_DIR = TTS_TRAINING / 'CosyVoice-0.5B'
REPO_DIR = DEPLOY_DIR / 'repo'
MODEL_DIR = DEPLOY_DIR / 'pretrained_models' / 'CosyVoice2-0.5B'

DEFAULT_REF = TTS_TRAINING / 'ref3_clean.wav'
DEFAULT_CORPUS = TTS_TRAINING / 'corpus.txt'
DEFAULT_OUT = TTS_TRAINING / 'dataset'


def load_corpus(path, limit=None):
    """加载语料，每行一句。"""
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) >= 4:
                lines.append(line)
                if limit and len(lines) >= limit:
                    break
    return lines


def load_prompt_text(ref_path):
    """加载参考音频的转写文本。"""
    cache = TTS_TRAINING / f'prompt_text_{ref_path.stem}.txt'
    if cache.exists():
        return cache.read_text(encoding='utf-8').strip()
    raise FileNotFoundError(
        f'prompt_text 缓存不存在: {cache}\n'
        f'请先用 SenseVoice 转写 {ref_path} 并保存到 {cache}'
    )


def init_model(model_dir):
    """加载 CosyVoice2 模型。"""
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR / 'third_party' / 'Matcha-TTS'))
    from cosyvoice.cli.cosyvoice import AutoModel
    print(f'加载 CosyVoice2 模型: {model_dir}')
    t0 = time.time()
    model = AutoModel(model_dir=str(model_dir))
    print(f'模型加载耗时: {time.time()-t0:.1f}s')
    return model


def synth_one(model, text, ref_path, prompt_text, sample_rate):
    """合成一条音频，返回 numpy array 或 None。

    使用 inference_zero_shot 模式（完整 prompt_text），这是 cosyvoice_verify.py
    验证效果最好的模式。
    """
    chunks = list(model.inference_zero_shot(text, prompt_text, str(ref_path)))
    if not chunks:
        return None

    import numpy as np
    import torch
    audios = [c['tts_speech'].squeeze().cpu().numpy() for c in chunks]
    audio = np.concatenate(audios) if audios else None
    return audio


def detect_tail_silence(audio, sample_rate):
    """检测后半段静音（从 synthesize_local.py 移植）。

    CosyVoice 偶尔会在句末产生异常静音，这些样本需要剔除。
    """
    import numpy as np
    if len(audio) < sample_rate * 0.5:
        return False
    chunks = np.array_split(audio, 4)
    rms = [np.sqrt(np.mean(c**2)) for c in chunks]
    tail_rms = (rms[2] + rms[3]) / 2
    head_rms = (rms[0] + rms[1]) / 2
    return head_rms > 0.01 and tail_rms < head_rms * 0.05 and tail_rms < 0.01


def main():
    ap = argparse.ArgumentParser(description='CosyVoice2 数据集生成')
    ap.add_argument('--local', action='store_true', help='本地 CPU 模式（小批量验证）')
    ap.add_argument('--colab', action='store_true', help='Colab GPU 模式（大批量）')
    ap.add_argument('--batch', type=int, default=20, help='本次合成的条数')
    ap.add_argument('--ref', type=Path, default=DEFAULT_REF, help='参考音频')
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS, help='语料文件')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help='输出目录')
    ap.add_argument('--model-dir', type=Path, default=MODEL_DIR, help='模型目录')
    args = ap.parse_args()

    if not args.local and not args.colab:
        print('请指定 --local 或 --colab 模式')
        return 1

    # 检查输入
    if not args.ref.exists():
        print(f'参考音频不存在: {args.ref}')
        return 1
    if not args.corpus.exists():
        print(f'语料不存在: {args.corpus}')
        return 1
    if not args.model_dir.is_dir():
        print(f'模型目录不存在: {args.model_dir}')
        print('请先运行: python tts-training/cosyvoice_setup.py')
        return 1

    # 加载
    prompt_text = load_prompt_text(args.ref)
    print(f'prompt_text: {prompt_text}')

    corpus_lines = load_corpus(args.corpus, limit=args.batch)
    print(f'语料: {len(corpus_lines)} 条')

    model = init_model(args.model_dir)
    sample_rate = model.sample_rate
    print(f'模型采样率: {sample_rate}Hz')

    # 输出目录
    wavs_dir = args.out / 'wavs'
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / 'metadata.csv'

    # 断点续合
    done_ids = set()
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter='|'):
                if row:
                    done_ids.add(row[0])

    todo = [(i, text) for i, text in enumerate(corpus_lines)
            if f'{i:06d}' not in done_ids]
    print(f'已完成: {len(done_ids)} 条, 待合成: {len(todo)} 条')

    if not todo:
        print('全部合成完成！')
        return 0

    # 合成
    import soundfile as sf
    import numpy as np

    ok, fail = 0, 0
    meta_f = open(metadata_path, 'a', encoding='utf-8', newline='')
    writer = csv.writer(meta_f, delimiter='|')
    t0 = time.time()

    for idx, text in todo:
        wid = f'{idx:06d}'
        wav_path = wavs_dir / f'{wid}.wav'

        # 断点续合
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            continue

        try:
            audio = synth_one(model, text, args.ref, prompt_text, sample_rate)
            if audio is None or len(audio) == 0:
                print(f'  [{wid}] 空音频, 跳过')
                fail += 1
                continue

            # 检测后半段静音
            if detect_tail_silence(audio, sample_rate):
                print(f'  [{wid}] 后半段静音, 跳过')
                fail += 1
                continue

            # 保存
            sf.write(str(wav_path), audio, sample_rate)
            duration = len(audio) / sample_rate
            writer.writerow([wid, text, '', f'{duration:.3f}'])
            meta_f.flush()
            ok += 1

            elapsed = time.time() - t0
            rate = ok / max(1, elapsed)
            print(f'  [{wid}] OK | {duration:.2f}s | {rate:.2f}条/s | '
                  f'{text[:30]}...')

        except Exception as e:
            print(f'  [{wid}] 失败: {e}')
            fail += 1

    meta_f.close()
    elapsed = time.time() - t0
    print(f'\n完成: 成功 {ok}, 失败 {fail}, 耗时 {elapsed:.0f}s')
    print(f'输出: {args.out}')
    print(f'  wavs/: {ok} 个 wav 文件')
    print(f'  metadata.csv: {metadata_path}')

    if args.local:
        print('\n本地验证完成。下一步：')
        print('  1. 试听 dataset/wavs/*.wav 检查音色')
        print('  2. 运行 voice_finetune/train.py 验证训练流程')
        print('  3. 在 Colab 上运行 --colab 模式生成完整数据集')


if __name__ == '__main__':
    sys.exit(main())
