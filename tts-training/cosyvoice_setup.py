#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice2-0.5B 部署脚本（Windows / Python 3.11，无 conda 依赖）

作用：在 tts-training/CosyVoice-0.5B/ 下准备音色克隆教师模型环境，用于
PLAN.md 5.2 节「CosyVoice 跨语言克隆（教师模型）」：
  - 日语音色参考 + 中文文本 -> 中文酒狐音色音频

本脚本只负责环境准备，不生成数据集（数据生成由 cosyvoice_dataset/generate.py 完成）。
部署目录已加入 .gitignore，不会污染仓库。

用法（在仓库根目录执行）：
    python tts-training/cosyvoice_setup.py            # 全流程：venv + 依赖 + 模型
    python tts-training/cosyvoice_setup.py --skip-venv  # 仅下载模型（venv 已就绪时）
    python tts-training/cosyvoice_setup.py --check      # 检查部署完整性

目录布局：
    tts-training/CosyVoice-0.5B/
        repo/                    # cloned FunAudioLLM/CosyVoice
        venv/                    # Python 3.11 虚拟环境
        pretrained_models/
            CosyVoice2-0.5B/     # 从 ModelScope 下载的 0.5B 模型权重
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 部署根目录（与 .gitignore 中 /tts-training/CosyVoice-0.5B/ 对齐）
HERE = Path(__file__).parent.resolve()
DEPLOY_DIR = HERE / 'CosyVoice-0.5B'
REPO_DIR = DEPLOY_DIR / 'repo'
VENV_DIR = DEPLOY_DIR / 'venv'
MODEL_DIR = DEPLOY_DIR / 'pretrained_models' / 'CosyVoice2-0.5B'

# ModelScope 模型 ID（CosyVoice 2.0 0.5B，支持跨语言零样本克隆）
MODELSCOPE_ID = 'iic/CosyVoice2-0.5B'


def run(cmd, cwd=None, check=True, env=None):
    """执行命令并实时输出，失败抛异常。"""
    print(f'$ {" ".join(str(c) for c in cmd)}')
    result = subprocess.run(cmd, cwd=cwd, check=False, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f'命令失败 (exit={result.returncode}): {cmd}')
    return result.returncode


def step(msg):
    print(f'\n{"="*60}\n  {msg}\n{"="*60}')


def clone_repo():
    """克隆 CosyVoice 仓库（含子模块）。"""
    if REPO_DIR.exists() and (REPO_DIR / 'cosyvoice' / 'cli' / 'cosyvoice.py').exists():
        print(f'[skip] 仓库已存在: {REPO_DIR}')
        return
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    step(f'克隆 CosyVoice 仓库到 {REPO_DIR}')
    # 浅克隆加速；子模块用 --init --recursive 补齐
    run(['git', 'clone', '--depth', '1', 'https://github.com/FunAudioLLM/CosyVoice.git',
         str(REPO_DIR)])
    run(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1'],
        cwd=REPO_DIR)


def _venv_python():
    """返回 venv 内 python 可执行文件路径。"""
    if os.name == 'nt':
        return VENV_DIR / 'Scripts' / 'python.exe'
    return VENV_DIR / 'bin' / 'python'


def create_venv():
    """创建 Python 3.11 虚拟环境。"""
    venv_python = _venv_python()
    if venv_python.exists():
        print(f'[skip] venv 已存在: {VENV_DIR}')
        return str(venv_python)
    step(f'创建虚拟环境: {VENV_DIR}')
    run([sys.executable, '-m', 'venv', str(VENV_DIR)])
    # 升级 pip
    run([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    return str(venv_python)


def install_deps(venv_python):
    """安装依赖。

    官方 requirements.txt 钉死 Python 3.10 + torch 2.3.1。在 Python 3.11 上：
      - 大部分包兼容（numpy/torchaudio/transformers/onnxruntime 等）。
      - deepspeed / tensorrt-cu12 标记了 sys_platform=='linux'，Windows 自动跳过。
      - ttsfrd 是 cp310 wheel，仅用于文本规范化（可选），未安装时回落到 wetext。
      - sox 训练数据预处理才需要，推理路径不依赖，跳过。
    策略：先单独装 CPU 版 torch/torchaudio（~200MB），避免从 cu121 索引拉
    ~2.5GB 的 CUDA wheel（本机为 AMD GPU，CUDA 不可用）。

    官方 requirements.txt 含 `--extra-index-url https://download.pytorch.org/whl/cu121`
    与 `torch==2.3.1`/`torchaudio==2.3.1`，会让 pip 强制从 cu121 索引重装 CUDA 版 torch
    （本机无 NVIDIA GPU）。因此生成一份过滤后的 requirements.filtered.txt：
      - 移除 torch / torchaudio（已单独装 CPU 版）
      - 移除 --extra-index-url 行（避免拉 cu121 索引）
      - 保留 Linux-only 的 deepspeed / tensorrt-* 标记（Windows 自动跳过）
    """
    req = REPO_DIR / 'requirements.txt'
    if not req.exists():
        raise RuntimeError(f'未找到 requirements.txt: {req}')

    step('安装 CPU 版 torch + torchaudio（避免拉取 ~2.5GB CUDA wheel）')
    run([
        str(venv_python), '-m', 'pip', 'install',
        'torch==2.3.1', 'torchaudio==2.3.1',
        '--index-url', 'https://download.pytorch.org/whl/cpu',
    ])

    # 生成过滤后的 requirements（移除 torch/torchaudio/extra-index-url）
    filtered = DEPLOY_DIR / 'requirements.filtered.txt'
    # 跳过的包：
    #   torch / torchaudio  -- 已单独安装 CPU 版
    #   openai-whisper      -- 老式 setup.py 依赖 pkg_resources（new setuptools 移除），
    #                          仅用于 ASR 转写，跨语言音色克隆推理路径不需要
    #   sox                 -- 训练数据预处理才用，推理路径不需要
    #   ttsfrd              -- cp310 wheel，Python 3.11 不可用，回落到 wetext
    skip_pkgs = {'torch', 'torchaudio', 'openai-whisper', 'sox', 'ttsfrd'}
    kept = []
    with req.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('--'):  # 移除所有 --extra-index-url / --index-url
                continue
            # 取包名（== 前的部分，忽略环境标记 ; ...）
            pkg = s.split('==')[0].split('>=')[0].split('<=')[0].split('!=')[0].split('~=')[0].split('>')[0].split('<')[0].strip().lower()
            if pkg in skip_pkgs:
                print(f'  [skip] {s}')
                continue
            kept.append(s)
    with filtered.open('w', encoding='utf-8') as f:
        f.write('# Auto-generated by cosyvoice_setup.py\n')
        f.write('# - torch/torchaudio 已单独安装 CPU 版（避免 CUDA 依赖）\n')
        f.write('# - 移除 --extra-index-url（避免拉取 cu121 索引）\n')
        f.write('# - 跳过 openai-whisper（构建失败，推理路径不需要）\n')
        for line in kept:
            f.write(line + '\n')
    print(f'过滤后依赖写入: {filtered}（共 {len(kept)} 条）')

    step('安装其余依赖（首次约 5-10 分钟）')
    # 使用阿里云镜像（清华镜像缺 conformer==0.3.2）
    run([
        str(venv_python), '-m', 'pip', 'install', '-r', str(filtered),
        '-i', 'https://mirrors.aliyun.com/pypi/simple',
        '--trusted-host', 'mirrors.aliyun.com',
    ])


def download_model(venv_python):
    """通过 ModelScope 下载 CosyVoice2-0.5B 模型权重（~2GB）。"""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f'[skip] 模型已存在: {MODEL_DIR}')
        return
    step(f'下载模型 {MODELSCOPE_ID}（~2GB，从 ModelScope）')
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    # modelscope 已在 requirements.txt 中，直接调用
    script = (
        f"from modelscope import snapshot_download\n"
        f"snapshot_download('{MODELSCOPE_ID}', local_dir=r'{MODEL_DIR}')\n"
        f"print('模型下载完成:', r'{MODEL_DIR}')\n"
    )
    run([str(venv_python), '-c', script], cwd=REPO_DIR)


def check():
    """检查部署完整性。"""
    step('部署完整性检查')
    venv_python = _venv_python()
    checks = [
        ('仓库', REPO_DIR.is_dir() and (REPO_DIR / 'cosyvoice').is_dir()),
        ('venv', venv_python.is_file()),
        ('模型目录', MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir())),
    ]
    all_ok = True
    for name, ok in checks:
        flag = 'OK' if ok else 'MISSING'
        print(f'  [{flag}] {name}')
        if not ok:
            all_ok = False

    if all_ok:
        print('\n导入 CosyVoice2 模块验证...')
        try:
            run([str(venv_python), '-c',
                 'import cosyvoice; print("cosyvoice 包导入成功")'], cwd=REPO_DIR, check=False)
        except Exception as e:
            print(f'  [WARN] 导入测试失败（不影响模型下载）: {e}')
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser(description='CosyVoice2-0.5B 部署脚本')
    ap.add_argument('--skip-venv', action='store_true', help='跳过 venv 创建与依赖安装')
    ap.add_argument('--skip-deps', action='store_true', help='跳过依赖安装（仅 venv + 模型）')
    ap.add_argument('--check', action='store_true', help='仅检查部署完整性')
    args = ap.parse_args()

    if args.check:
        return check()

    clone_repo()

    if args.skip_venv:
        venv_python = str(_venv_python())
        if not Path(venv_python).exists():
            print(f'错误: --skip-venv 但 venv 不存在: {venv_python}')
            return 1
    else:
        venv_python = create_venv()

    if not args.skip_deps:
        install_deps(venv_python)

    download_model(venv_python)

    print('\n部署完成。验证:')
    print(f'  {venv_python} tts-training/cosyvoice_verify.py')
    return check()


if __name__ == '__main__':
    sys.exit(main())
