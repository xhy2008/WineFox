#!/usr/bin/env python3
"""验证导出的 zf_winefox voice 能否被 Kokoro 正确加载并合成。"""
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline

HERE = Path(__file__).parent.resolve()
TTS_TRAINING = HERE.parent
CONFIG = TTS_TRAINING / 'kokoro-model' / 'config.json'
CHECKPOINT = TTS_TRAINING / 'kokoro-model' / 'kokoro-v1_0.pth'
VOICES_BIN = TTS_TRAINING / 'voices-winefox.bin'

STYLE_DIM = 256
MAX_PHONEME_LEN = 510

TEST_TEXTS = [
    '主人早上好呀，今天想做什么呢？',
    '嘿嘿，我又给你烤了饼干，快尝尝看嘛。',
    '酒狐一直都会陪着你的，不管发生什么事情。',
]


def load_voice_from_bin(voices_path, voice_name):
    """从 voices.bin 加载 voice，返回 [510, 1, 256] tensor。"""
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
                style = np.frombuffer(data, dtype=np.float32).copy()
                return torch.FloatTensor(style.reshape(MAX_PHONEME_LEN, 1, STYLE_DIM))

    raise KeyError(f'Voice {voice_name} not found in {voices_path}')


def main():
    print('=' * 60)
    print('验证 zf_winefox voice')
    print('=' * 60)

    kmodel = KModel(config=str(CONFIG), model=str(CHECKPOINT), disable_complex=True)
    kmodel.eval().cpu()
    pipeline = KPipeline(lang_code='z', model=kmodel, device='cpu')

    print(f'\n加载 voice: zf_winefox (from {VOICES_BIN.name})')
    voice_tensor = load_voice_from_bin(VOICES_BIN, 'zf_winefox')
    print(f'  shape: {voice_tensor.shape}')

    out_dir = HERE / 'verify_output'
    out_dir.mkdir(exist_ok=True)

    print(f'\n合成 {len(TEST_TEXTS)} 条测试音频...')
    for i, text in enumerate(TEST_TEXTS):
        out_path = out_dir / f'winefox_{i:02d}.wav'
        for result in pipeline(text, voice=voice_tensor, speed=1.0):
            audio = result.audio
            if audio is not None:
                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                sf.write(str(out_path), audio_np, 24000)
                duration = len(audio_np) / 24000
                print(f'  [{i:02d}] {duration:.2f}s | {text} -> {out_path.name}')
                break

    print(f'\n验证完成。音频保存到: {out_dir}')
    print('请人工试听，对比音色是否符合预期。')


if __name__ == '__main__':
    main()
