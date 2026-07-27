#!/usr/bin/env python3
"""生成测试数据集（用 Kokoro 自己合成，仅用于验证训练流程）。

数据集由 Kokoro 使用基础 voice (zf_001) 合成，作为训练目标音频。
这样在 CPU 上验证训练流程时不需要先跑 CosyVoice。
"""
import csv
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline

TTS_TRAINING = Path(__file__).parent.parent
CONFIG = TTS_TRAINING / 'kokoro-model' / 'config.json'
CHECKPOINT = TTS_TRAINING / 'kokoro-model' / 'kokoro-v1_0.pth'
VOICES_BIN = TTS_TRAINING / 'voices-v1.1-zh.bin'
DATASET_DIR = TTS_TRAINING / 'dataset'

# Voice style 维度（与 train.py 保持一致）
STYLE_DIM = 256
MAX_PHONEME_LEN = 510

TEST_TEXTS = [
    '你好，我是酒狐。',
    '今天天气真好，我们去公园散步吧。',
    '主人早上好呀，今天想做什么呢？',
    '嘿嘿，我又给你烤了饼干，快尝尝看嘛。',
    '酒狐一直都会陪着你的，不管发生什么事情。',
]


def load_voice_from_bin(voices_path, voice_name):
    """从 voices.bin 加载指定 voice 的 style matrix，返回 torch.FloatTensor [510, 256]。"""
    with open(voices_path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'VOIC', f'Bad magic: {magic}'
        version = struct.unpack('<I', f.read(4))[0]
        num_voices = struct.unpack('<I', f.read(4))[0]

        for _ in range(num_voices):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            dim = struct.unpack('<I', f.read(4))[0]
            data = f.read(dim * 4)

            if name == voice_name:
                style = np.frombuffer(data, dtype=np.float32)
                assert dim == MAX_PHONEME_LEN * STYLE_DIM, \
                    f'Unexpected dim: {dim} != {MAX_PHONEME_LEN * STYLE_DIM}'
                # 返回 [510, 1, 256]：KPipeline.infer 调用 pack[len(ps)-1] 后
                # 得到 [1, 256]，符合 KModel.forward 对 ref_s 的 2D 期望
                style = style.copy()  # 从 read-only buffer 拷贝为可写
                return torch.FloatTensor(style.reshape(MAX_PHONEME_LEN, 1, STYLE_DIM))

    raise KeyError(f'Voice {voice_name} not found in {voices_path}')


def main():
    print('=' * 60)
    print('生成测试数据集（Kokoro zf_001 合成）')
    print('=' * 60)

    kmodel = KModel(config=str(CONFIG), model=str(CHECKPOINT), disable_complex=True)
    kmodel.eval().cpu()
    pipeline = KPipeline(lang_code='z', model=kmodel, device='cpu')

    # 从本地 voices.bin 加载 voice tensor，避免从 HF 下载
    print(f'\n加载 voice: zf_001 (from {VOICES_BIN.name})')
    voice_tensor = load_voice_from_bin(VOICES_BIN, 'zf_001')
    print(f'  shape: {voice_tensor.shape}')

    wavs_dir = DATASET_DIR / 'wavs'
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = DATASET_DIR / 'metadata.csv'

    print(f'\n开始合成 {len(TEST_TEXTS)} 条音频...')
    with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        for i, text in enumerate(TEST_TEXTS):
            wid = f'{i:06d}'
            wav_path = wavs_dir / f'{wid}.wav'

            # 用 Kokoro 合成（直接传 voice tensor）
            for result in pipeline(text, voice=voice_tensor, speed=1.0):
                audio = result.audio
                if audio is not None:
                    audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                    sf.write(str(wav_path), audio_np, 24000)
                    duration = len(audio_np) / 24000
                    writer.writerow([wid, text, '', f'{duration:.3f}'])
                    print(f'  [{wid}] {duration:.2f}s | {text}')
                    break

    print(f'\n测试数据集生成完成: {DATASET_DIR}')


if __name__ == '__main__':
    main()
