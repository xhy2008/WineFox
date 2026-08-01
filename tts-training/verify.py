#!/usr/bin/env python3
"""验证训练后的 voice pack 音色效果。

用法:
    python verify.py                          # 验证 winefox_final.pt
    python verify.py --voice checkpoints/winefox_best.pt
    python verify.py --voice zf_001           # 对比官方音色
"""

import os
import sys
import argparse
import torch
import soundfile as sf
import numpy as np
from kokoro import KPipeline

TEST_TEXTS = [
    '你好，我是酒狐，爱喝酒的狐狸你见过吗？',
    '今天天气真好，院子里的花都开了呢。',
    '我虽然爱喝酒，但从不误事，放心吧。',
    '大正女仆就是要端庄贤淑，我可是模范。',
    '书里的故事好精彩，我都看入迷了。',
]


def main():
    parser = argparse.ArgumentParser(description='Verify voice pack')
    parser.add_argument('--voice', default='checkpoints/winefox_final.pt',
                        help='Voice pack path or name (e.g. zf_001)')
    parser.add_argument('--output-dir', default='verify_output')
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading KPipeline (lang=z)...')
    pipeline = KPipeline(lang_code='z')

    # 判断是文件还是官方 voice 名
    voice = args.voice
    is_file = voice.endswith('.pt') and os.path.exists(voice)

    if is_file:
        vp = torch.load(voice, weights_only=True)
        print(f'Voice pack: {voice}')
        print(f'  Shape: {vp.shape}')
        print(f'  Norm: {vp[0].norm().item():.4f}')
        print(f'  decoder_style norm: {vp[0,0,:128].norm().item():.4f}')
        print(f'  predictor_style norm: {vp[0,0,128:].norm().item():.4f}')
    else:
        print(f'Official voice: {voice}')

    print(f'\nGenerating {len(TEST_TEXTS)} test sentences...\n')

    for i, text in enumerate(TEST_TEXTS):
        result = pipeline(text, voice=voice, speed=args.speed)
        for chunk_idx, (g, p, audio) in enumerate(result):
            if audio is not None:
                tag = 'finetuned' if is_file else 'original'
                out_path = os.path.join(
                    args.output_dir, f'{tag}_{i+1:02d}.wav'
                )
                audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio
                sf.write(out_path, audio_np, 24000)

                peak = np.abs(audio_np).max()
                rms = np.sqrt(np.mean(audio_np ** 2))
                rms_db = 20 * np.log10(rms + 1e-10)
                dur = len(audio_np) / 24000
                print(f'  [{i+1}] {out_path}')
                print(f'      dur={dur:.1f}s peak={peak:.3f} rms={rms_db:.1f}dB')
                print(f'      "{text}"')

    print(f'\nDone! Audio files in: {args.output_dir}')


if __name__ == '__main__':
    main()
