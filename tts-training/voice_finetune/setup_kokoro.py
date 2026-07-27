#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 Kokoro-82M PyTorch 源码与权重，供 voice style 微调使用。

训练需要 Kokoro 的 PyTorch 模型（KModel + KPipeline），本脚本自动下载：
  1. Kokoro Python 包源码（kokoro/）
  2. 预训练权重 kokoro-v1_0.pth
  3. 配置文件 config.json
  4. 语音包 voices-v1.1-zh.bin（如不存在则从 HF 下载）

用法：
    python setup_kokoro.py            # 下载全部
    python setup_kokoro.py --check    # 检查是否已就绪
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
TTS_TRAINING = HERE.parent
KOKORO_SRC = TTS_TRAINING / 'kokoro-src'        # Python 包源码
KOKORO_MODEL = TTS_TRAINING / 'kokoro-model'     # 权重 + config
VOICES_BIN = TTS_TRAINING / 'voices-v1.1-zh.bin'

REPO = 'hexgrad/Kokoro-82M'


def check():
    """检查是否已就绪。

    Kokoro 可通过两种方式安装：
      1. pip 包（推荐）：pip install kokoro
      2. 源码目录：kokoro-src/kokoro/（已弃用，保留兼容）
    """
    # 检查 pip 包是否可用
    kokoro_pkg_ok = False
    try:
        import kokoro  # noqa: F401
        from kokoro import KModel, KPipeline  # noqa: F401
        kokoro_pkg_ok = True
    except ImportError:
        pass

    # 检查源码目录（兼容旧路径）
    kokoro_src_ok = KOKORO_SRC.is_dir() and (KOKORO_SRC / 'kokoro' / 'model.py').is_file()

    checks = [
        ('kokoro 包（pip 或源码）', kokoro_pkg_ok or kokoro_src_ok),
        ('kokoro 权重', (KOKORO_MODEL / 'kokoro-v1_0.pth').is_file()),
        ('kokoro 配置', (KOKORO_MODEL / 'config.json').is_file()),
        ('voices.bin', VOICES_BIN.is_file()),
    ]
    all_ok = True
    for name, ok in checks:
        flag = 'OK' if ok else 'MISSING'
        print(f'  [{flag}] {name}')
        if not ok:
            all_ok = False
    return 0 if all_ok else 1


def download():
    """从 HuggingFace 下载 Kokoro 源码与权重。"""
    from huggingface_hub import snapshot_download

    # 1. 下载完整仓库（含源码 + 权重 + config）
    print(f'从 HuggingFace 下载 {REPO}...')
    KOKORO_SRC.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        local_dir=str(KOKORO_SRC),
        allow_patterns=[
            'kokoro/**',
            'kokoro-v1_0.pth',
            'config.json',
            'voices/*.bin',
            'voices/*.npy',
            'espeak-ng-data/**',
        ],
    )
    print(f'源码下载到: {KOKORO_SRC}')

    # 2. 权重和 config 放到单独目录（保持路径清晰）
    KOKORO_MODEL.mkdir(parents=True, exist_ok=True)
    import shutil
    for fname in ['kokoro-v1_0.pth', 'config.json']:
        src = KOKORO_SRC / fname
        dst = KOKORO_MODEL / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
    print(f'权重和配置复制到: {KOKORO_MODEL}')

    # 3. voices.bin（如不存在则从 HF 下载）
    if not VOICES_BIN.exists():
        # 从 voice-test/models/ 复制（已有）
        existing = Path(r'e:\winefox\voice-test\models\voices-v1.1-zh.bin')
        if existing.exists():
            shutil.copy2(str(existing), str(VOICES_BIN))
            print(f'voices.bin 从 voice-test 复制到: {VOICES_BIN}')
        else:
            print('警告: voices-v1.1-zh.bin 不存在，请手动从 HF 下载')
            print(f'  https://huggingface.co/{REPO}/resolve/main/voices/v1.1-zh.bin')

    print('\n下载完成。')
    return check()


def main():
    ap = argparse.ArgumentParser(description='Kokoro PyTorch 源码下载')
    ap.add_argument('--check', action='store_true', help='仅检查')
    args = ap.parse_args()

    if args.check:
        return check()

    return download()


if __name__ == '__main__':
    sys.exit(main())
