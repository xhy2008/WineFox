#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练后的 voice style 导出到 voices.bin。

功能：
  1. 从 voice_style_best.pt 加载训练后的 [510, 256] voice style
  2. 将其作为新 voice "zf_winefox" 添加到 voices.bin
  3. 或替换现有 voice（--replace）

用法：
  # 添加新 voice "zf_winefox"
  python export_voice.py --input output/voice_style_best.pt

  # 替换现有 voice "zf_xiaobei"
  python export_voice.py --input output/voice_style_best.pt --replace zf_xiaobei

  # 指定输出路径（默认不覆盖原文件）
  python export_voice.py --input output/voice_style_best.pt --out voices-winefox.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent.resolve()
TTS_TRAINING = HERE.parent
VOICES_BIN = TTS_TRAINING / 'voices-v1.1-zh.bin'

MAX_PHONEME_LEN = 510
STYLE_DIM = 256
NEW_VOICE_NAME = 'zf_winefox'


def read_all_voices(path):
    """读取 voices.bin 中的所有 voice。返回 [(name, style_flat), ...]。"""
    voices = []
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'VOIC', f'Bad magic: {magic}'
        version = struct.unpack('<I', f.read(4))[0]
        assert version == 1
        num_voices = struct.unpack('<I', f.read(4))[0]

        for _ in range(num_voices):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            dim = struct.unpack('<I', f.read(4))[0]
            data = f.read(dim * 4)
            style = np.frombuffer(data, dtype=np.float32)
            voices.append((name, style))

    return voices


def write_voices(path, voices):
    """将 voice 列表写入 voices.bin。"""
    with open(path, 'wb') as f:
        f.write(b'VOIC')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(voices)))

        for name, style in voices:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(style)))
            f.write(style.astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser(description='导出 voice style 到 voices.bin')
    ap.add_argument('--input', type=Path, required=True,
                    help='训练后的 voice_style .pt 文件')
    ap.add_argument('--voices-bin', type=Path, default=VOICES_BIN,
                    help='原始 voices.bin')
    ap.add_argument('--out', type=Path, default=None,
                    help='输出路径（默认 voices-winefox.bin）')
    ap.add_argument('--name', type=str, default=NEW_VOICE_NAME,
                    help='新 voice 的名称')
    ap.add_argument('--replace', type=str, default=None,
                    help='替换指定 voice（而非添加新 voice）')
    args = ap.parse_args()

    if not args.input.exists():
        print(f'错误: 输入文件不存在: {args.input}')
        return 1
    if not args.voices_bin.exists():
        print(f'错误: voices.bin 不存在: {args.voices_bin}')
        return 1

    out_path = args.out or args.voices_bin.parent / 'voices-winefox.bin'

    # 1. 加载训练后的 voice style
    ckpt = torch.load(str(args.input), map_location='cpu', weights_only=False)
    voice_style = ckpt['voice_style'].numpy()  # [510, 256]
    print(f'加载 voice style: {voice_style.shape}')
    assert voice_style.shape == (MAX_PHONEME_LEN, STYLE_DIM), \
        f'Unexpected shape: {voice_style.shape}'

    # 展平为 1D（与 voices.bin 格式一致）
    style_flat = voice_style.flatten()  # [130560]

    # 2. 读取现有 voices
    voices = read_all_voices(args.voices_bin)
    print(f'原始 voices.bin: {len(voices)} voices')

    # 3. 添加或替换
    if args.replace:
        # 替换现有 voice
        replaced = False
        for i, (name, _) in enumerate(voices):
            if name == args.replace:
                voices[i] = (name, style_flat)
                print(f'替换 voice: {name}')
                replaced = True
                break
        if not replaced:
            print(f'错误: voice "{args.replace}" 不存在')
            return 1
    else:
        # 添加新 voice
        # 检查是否已存在
        existing_names = {name for name, _ in voices}
        if args.name in existing_names:
            print(f'错误: voice "{args.name}" 已存在，请用 --replace 或 --name 指定其他名称')
            return 1
        voices.append((args.name, style_flat))
        print(f'添加新 voice: {args.name}')

    # 4. 写入输出文件
    write_voices(out_path, voices)
    print(f'\n输出: {out_path}')
    print(f'  voices 数量: {len(voices)}')
    print(f'  文件大小: {out_path.stat().st_size / 1024 / 1024:.2f} MB')

    # 5. 验证
    print('\n验证...')
    verify_voices = read_all_voices(out_path)
    print(f'  读取到 {len(verify_voices)} voices')
    if args.replace:
        check_name = args.replace
    else:
        check_name = args.name
    for name, style in verify_voices:
        if name == check_name:
            print(f'  {name}: dim={len(style)}, OK')
            break
    else:
        print(f'  错误: 未找到 {check_name}')
        return 1

    print(f'\n完成！将 {out_path} 复制到 voice-test/models/ 即可使用。')
    print(f'  C++ 代码中调用: get_voice_style("{check_name}")')
    return 0


if __name__ == '__main__':
    sys.exit(main())
