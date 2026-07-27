#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab 大规模训练脚本（一键执行）

在 Google Colab 中运行此脚本，完成完整流程：
  1. 安装依赖（kokoro、cosyvoice、librosa 等）
  2. 下载 Kokoro 模型与 voices.bin
  3. 用 CosyVoice 跨语言克隆生成中文酒狐音色数据集
  4. 用 GPU 训练 voice style matrix
  5. 导出到 voices-winefox.bin
  6. 打包供下载

使用方法：
  方式 A（推荐）：将本文件上传到 Colab，然后：
    !python colab_train.py --step all

  方式 B：分步执行（便于调试）：
    !python colab_train.py --step setup
    !python colab_train.py --step dataset --batch 1000
    !python colab_train.py --step train --epochs 50
    !python colab_train.py --step export

前置条件：
  - Google Colab 已选择 GPU 运行时（Runtime → Change runtime type → T4 GPU）
  - 已将 winefox 仓库克隆到 /content/winefox（脚本会自动 clone）

注意：
  - Colab 免费版 T4 GPU 约 16GB 显存，足够运行 CosyVoice2-0.5B + Kokoro-82M
  - 完整 5000+ 条语料合成约需 4-6 小时（CosyVoice 推理较慢）
  - 训练本身很快（Kokoro 只训 voice style matrix，510×256=130K 参数）
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Colab 默认工作目录
COLAB_WORK = Path('/content')
WINEFOX_DIR = COLAB_WORK / 'winefox'
TTS_TRAINING = WINEFOX_DIR / 'tts-training'


def run(cmd, check=True, cwd=None):
    """执行 shell 命令并实时打印输出。"""
    print(f'$ {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        raise RuntimeError(f'命令失败: {cmd}')
    return result.returncode


def step_setup():
    """Step 1: 环境准备。"""
    print('=' * 60)
    print('Step 1: 环境准备')
    print('=' * 60)

    # 1.1 克隆仓库（如不存在）
    if not WINEFOX_DIR.exists():
        run(f'cd {COLAB_WORK} && git clone https://github.com/your-username/winefox.git')
    else:
        print(f'仓库已存在: {WINEFOX_DIR}')

    # 1.2 安装依赖（Kokoro 训练用，CosyVoice 依赖由 cosyvoice_setup.py 装）
    print('\n安装 Python 依赖（Kokoro 训练用）...')
    run('pip install -q kokoro>=0.8 soundfile librosa huggingface_hub')
    run('pip install -q misaki[zh]')

    # 1.3 检查 GPU
    print('\n检查 GPU...')
    run('nvidia-smi')
    run('python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\')"')

    # 1.4 部署 Kokoro 模型
    print('\n下载 Kokoro 模型...')
    run(f'cd {WINEFOX_DIR} && python tts-training/voice_finetune/setup_kokoro.py --check',
        check=False)
    run(f'cd {WINEFOX_DIR} && python tts-training/voice_finetune/setup_kokoro.py')

    # 1.5 部署 CosyVoice（用于数据集生成）
    # Colab 上无法用 venv（ensurepip 不可用），用 --system-python 直接装到系统 Python
    print('\n部署 CosyVoice2（--system-python 模式，Colab 兼容）...')
    run(f'cd {WINEFOX_DIR} && python tts-training/cosyvoice_setup.py --system-python --check',
        check=False)
    run(f'cd {WINEFOX_DIR} && python tts-training/cosyvoice_setup.py --system-python')

    print('\n✓ 环境准备完成')


def step_dataset(batch=1000):
    """Step 2: 用 CosyVoice 生成数据集。"""
    print('=' * 60)
    print(f'Step 2: 生成数据集 (batch={batch})')
    print('=' * 60)

    # Colab 上用 --system-python 模式，直接用系统 python 运行生成脚本
    # 依赖已由 cosyvoice_setup.py --system-python 装到系统 Python
    run(f'cd {WINEFOX_DIR} && python tts-training/cosyvoice_dataset/generate.py '
        f'--colab --batch {batch}')

    # 检查生成结果
    wavs_dir = TTS_TRAINING / 'dataset' / 'wavs'
    metadata = TTS_TRAINING / 'dataset' / 'metadata.csv'
    if wavs_dir.exists():
        n_wavs = len(list(wavs_dir.glob('*.wav')))
        print(f'\n✓ 数据集生成完成: {n_wavs} wav 文件')
    if metadata.exists():
        with open(metadata) as f:
            n_lines = sum(1 for _ in f)
        print(f'  metadata.csv: {n_lines} 条记录')


def step_train(epochs=50, batch_size=16, lr=1e-3):
    """Step 3: GPU 训练。"""
    print('=' * 60)
    print(f'Step 3: GPU 训练 (epochs={epochs}, batch={batch_size}, lr={lr})')
    print('=' * 60)

    run(f'cd {WINEFOX_DIR} && python tts-training/voice_finetune/train.py '
        f'--gpu --epochs {epochs} --batch-size {batch_size} --lr {lr}')

    print('\n✓ 训练完成')
    output_dir = TTS_TRAINING / 'voice_finetune' / 'output'
    if output_dir.exists():
        for f in output_dir.glob('*.pt'):
            size_kb = f.stat().st_size / 1024
            print(f'  {f.name}: {size_kb:.1f} KB')


def step_export():
    """Step 4: 导出 voice 到 voices.bin。"""
    print('=' * 60)
    print('Step 4: 导出 voice')
    print('=' * 60)

    best_pt = TTS_TRAINING / 'voice_finetune' / 'output' / 'voice_style_best.pt'
    run(f'cd {WINEFOX_DIR} && python tts-training/voice_finetune/export_voice.py '
        f'--input {best_pt}')

    # 验证
    run(f'cd {WINEFOX_DIR} && python tts-training/voice_finetune/verify_voice.py',
        check=False)

    # 打包下载
    print('\n打包结果便于下载...')
    run(f'cd {TTS_TRAINING} && tar -czf /content/winefox-voice.tar.gz '
        f'voices-winefox.bin '
        f'voice_finetune/output/voice_style_best.pt '
        f'voice_finetune/verify_output/')

    print('\n✓ 导出完成')
    print('下载结果: /content/winefox-voice.tar.gz')
    print('  解压后包含:')
    print('    voices-winefox.bin     - 新的 voices.bin（含 zf_winefox voice）')
    print('    voice_style_best.pt    - 训练后的 voice style matrix')
    print('    verify_output/*.wav    - 验证音频（试听检查音色）')
    print('\n部署: 将 voices-winefox.bin 复制到 voice-test/models/')


def step_all(batch=1000, epochs=50):
    """一键执行全部步骤。"""
    step_setup()
    step_dataset(batch=batch)
    step_train(epochs=epochs)
    step_export()
    print('\n' + '=' * 60)
    print('全部完成！')
    print('=' * 60)


def main():
    ap = argparse.ArgumentParser(description='Colab 一键训练脚本')
    ap.add_argument('--step', type=str, default='all',
                    choices=['all', 'setup', 'dataset', 'train', 'export'],
                    help='执行的步骤')
    ap.add_argument('--batch', type=int, default=1000,
                    help='数据集合成条数（dataset 步骤）')
    ap.add_argument('--epochs', type=int, default=50,
                    help='训练轮数（train 步骤）')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='训练 batch size')
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='学习率')
    args = ap.parse_args()

    if args.step == 'all':
        step_all(batch=args.batch, epochs=args.epochs)
    elif args.step == 'setup':
        step_setup()
    elif args.step == 'dataset':
        step_dataset(batch=args.batch)
    elif args.step == 'train':
        step_train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.step == 'export':
        step_export()


if __name__ == '__main__':
    main()
