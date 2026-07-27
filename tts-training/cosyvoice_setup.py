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

Colab 用法（不创建 venv，直接用系统 Python）：
    python tts-training/cosyvoice_setup.py --system-python
    python tts-training/cosyvoice_setup.py --system-python --skip-deps  # 依赖已装时

目录布局：
    tts-training/CosyVoice-0.5B/
        repo/                    # cloned FunAudioLLM/CosyVoice
        venv/                    # Python 3.11 虚拟环境（--system-python 模式下不创建）
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


def create_venv(use_system_python=False):
    """创建 Python 3.11 虚拟环境。

    Args:
        use_system_python: True 时跳过 venv 创建，直接返回系统 Python 路径。
            用于 Colab 等无法使用 venv 的环境（Colab 的 Python 无 ensurepip）。
    """
    if use_system_python:
        print(f'[system-python] 使用系统 Python: {sys.executable}')
        return sys.executable

    venv_python = _venv_python()
    if venv_python.exists():
        print(f'[skip] venv 已存在: {VENV_DIR}')
        return str(venv_python)
    step(f'创建虚拟环境: {VENV_DIR}')
    run([sys.executable, '-m', 'venv', str(VENV_DIR)])
    # 升级 pip
    run([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    return str(venv_python)


def _is_pkg_installed(venv_python, pkg_name):
    """检查指定包是否已安装（通过 importlib.metadata）。"""
    try:
        result = subprocess.run(
            [str(venv_python), '-c',
             f'import importlib.metadata; importlib.metadata.version("{pkg_name}"); print("OK")'],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and 'OK' in result.stdout
    except Exception:
        return False


def _get_installed_version(venv_python, pkg_name):
    """获取已安装包的版本号，未安装返回 None。"""
    try:
        result = subprocess.run(
            [str(venv_python), '-c',
             f'import importlib.metadata; print(importlib.metadata.version("{pkg_name}"))'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# Colab 预装环境（基于 2026-07 实测 req.txt）中已存在的包及其版本。
# 这些包无需重装，重装可能触发依赖冲突（transformers 5.x vs 4.x、numpy 2.x vs 1.x 等）。
# key: 包名（小写），value: Colab 预装版本
COLAB_PREINSTALLED = {
    'torch': '2.3.1+cu121',
    'torchaudio': '2.3.1+cu121',
    'transformers': '5.13.1',
    'numpy': '2.0.2',
    'protobuf': '5.29.6',
    'librosa': '0.11.0',
    'soundfile': '0.14.0',
    'omegaconf': '2.3.1',
    'hydra-core': None,  # 未预装
    'networkx': '3.6.1',
    'matplotlib': '3.10.0',
    'rich': '13.9.4',
    'pydantic': '2.13.4',
    'pyarrow': '18.1.0',
    'tensorboard': '2.20.0',
    'inflect': '7.5.0',
    'gdown': '5.2.2',
    'gradio': '6.20.0',
    'grpcio': '1.82.1',
    'fastapi': '0.139.0',
    'uvicorn': '0.51.0',
    'diffusers': '0.39.0',
    'click': '8.4.2',
    'pillow': '11.3.0',
    'scipy': '1.16.3',
    'scikit-learn': '1.6.1',
    'pandas': '2.2.2',
    'tqdm': '4.67.3',
    'requests': '2.32.4',
    'huggingface_hub': '1.23.0',
    'safetensors': '0.8.0',
    'tokenizers': '0.22.2',
    'sentencepiece': '0.2.2',
}

# CosyVoice 推理路径（cosyvoice_dataset/generate.py）真正需要的包。
# 从 requirements.txt 中筛选出运行时 import 的核心依赖，剔除训练/部署专用包。
# 这些包若 Colab 未预装，则需要安装。
COSYVOICE_RUNTIME_DEPS = [
    # 包名（pip 名）, import 名, 是否必需
    ('conformer', 'conformer', True),
    ('HyperPyYAML', 'hyperpyyaml', True),
    ('hydra-core', 'hydra', True),
    ('modelscope', 'modelscope', True),  # 下载模型必需
    ('wetext', 'wetext', True),  # 中文文本规范化（ttsfrd 替代）
    ('x-transformers', 'x_transformers', True),  # 模型结构
    ('onnxruntime-gpu', 'onnxruntime', False),  # 可选，GPU 加速
    ('pyworld', 'pyworld', False),  # 可选，音频分析
    ('wget', 'wget', False),  # 可选，下载工具
    ('lightning', 'lightning', False),  # 训练才用，推理可跳过
    ('deepspeed', 'deepspeed', False),  # 训练才用
]


def install_deps(venv_python, use_system_python=False):
    """安装依赖。

    两种模式：
      1. 本地 Windows（use_system_python=False）：创建 venv，装 CPU 版 torch，
         按 requirements.filtered.txt 装全部依赖（阿里云镜像）。
      2. Colab（use_system_python=True）：不创建 venv，直接用系统 Python。
         Colab 已预装 torch 2.3.1+cu121 / transformers 5.x / numpy 2.x 等，
         只补装 CosyVoice 推理缺失的小包，避免降级触发依赖冲突。

    关键：Colab 模式不重装 torch/transformers/numpy/protobuf，即使版本与
    CosyVoice requirements.txt 不一致。实测 CosyVoice2-0.5B 推理在
    transformers 5.x + numpy 2.x 下可正常工作（仅 warnings）。
    """
    req = REPO_DIR / 'requirements.txt'
    if not req.exists():
        raise RuntimeError(f'未找到 requirements.txt: {req}')

    if not use_system_python:
        # ===== 本地 Windows 模式：完整 venv 安装（原逻辑） =====
        # 生成过滤后的 requirements（移除 torch/torchaudio/extra-index-url）
        filtered = DEPLOY_DIR / 'requirements.filtered.txt'
        skip_pkgs = {'torch', 'torchaudio', 'openai-whisper', 'sox', 'ttsfrd'}
        kept = []
        with req.open('r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if s.startswith('--'):
                    continue
                pkg = s.split('==')[0].split('>=')[0].split('<=')[0].split('!=')[0].split('~=')[0].split('>')[0].split('<')[0].strip().lower()
                if pkg in skip_pkgs:
                    print(f'  [skip] {s}')
                    continue
                kept.append(s)
        with filtered.open('w', encoding='utf-8') as f:
            f.write('# Auto-generated by cosyvoice_setup.py\n')
            f.write('# - torch/torchaudio 已单独安装 CPU 版\n')
            f.write('# - 移除 --extra-index-url\n')
            f.write('# - 跳过 openai-whisper / sox / ttsfrd\n')
            for line in kept:
                f.write(line + '\n')
        print(f'过滤后依赖写入: {filtered}（共 {len(kept)} 条）')

        step('安装 CPU 版 torch + torchaudio（避免拉取 ~2.5GB CUDA wheel）')
        run([
            str(venv_python), '-m', 'pip', 'install',
            'torch==2.3.1', 'torchaudio==2.3.1',
            '--index-url', 'https://download.pytorch.org/whl/cpu',
        ])
        step('安装其余依赖（首次约 5-10 分钟）')
        run([
            str(venv_python), '-m', 'pip', 'install', '-r', str(filtered),
            '-i', 'https://mirrors.aliyun.com/pypi/simple',
            '--trusted-host', 'mirrors.aliyun.com',
        ])
        return

    # ===== Colab 模式：智能补装缺失包 =====
    step('Colab 模式：检查已预装的包')
    print(f'  Colab 预装 torch={_get_installed_version(venv_python, "torch")}, '
          f'torchaudio={_get_installed_version(venv_python, "torchaudio")}, '
          f'transformers={_get_installed_version(venv_python, "transformers")}, '
          f'numpy={_get_installed_version(venv_python, "numpy")}')
    print('  → 保留 Colab 预装版本，不降级（避免依赖冲突）')

    # 检查每个运行时依赖是否已安装
    to_install = []
    skip_count = 0
    for pip_name, import_name, required in COSYVOICE_RUNTIME_DEPS:
        # 优先用 import 名检查（有些 pip 名和 import 名不同）
        installed = _is_pkg_installed(venv_python, import_name)
        if installed:
            ver = _get_installed_version(venv_python, import_name) or '?'
            print(f'  [OK]    {pip_name} ({ver})')
            skip_count += 1
        elif required:
            print(f'  [MISS]  {pip_name} (必需，将安装)')
            to_install.append(pip_name)
        else:
            print(f'  [MISS]  {pip_name} (可选，跳过)')

    print(f'\n  已预装: {skip_count}/{len(COSYVOICE_RUNTIME_DEPS)}，'
          f'需安装: {len(to_install)}')

    if not to_install:
        print('  所有必需依赖已就绪，无需安装')
        return

    # 安装缺失包。使用 --no-deps 避免连带降级已预装的包（如 transformers/numpy）。
    # conformer / x-transformers 等小包无强依赖，--no-deps 安全。
    # modelscope 依赖较多，不能用 --no-deps，但它的依赖大部分 Colab 已预装。
    step(f'安装 {len(to_install)} 个缺失包（--no-deps 避免降级冲突）')

    # modelscope 需要正常安装（依赖 requests、protobuf 等，Colab 已预装）
    normal_install = [p for p in to_install if p == 'modelscope']
    no_deps_install = [p for p in to_install if p != 'modelscope']

    if normal_install:
        print(f'  正常安装（含依赖）: {normal_install}')
        run([str(venv_python), '-m', 'pip', 'install', '-q'] + normal_install)

    if no_deps_install:
        print(f'  --no-deps 安装: {no_deps_install}')
        run([str(venv_python), '-m', 'pip', 'install', '-q', '--no-deps'] + no_deps_install)

    # 验证安装结果
    print('\n  验证安装:')
    for pip_name, import_name, required in COSYVOICE_RUNTIME_DEPS:
        if not required:
            continue
        ok = _is_pkg_installed(venv_python, import_name)
        flag = 'OK' if ok else 'FAIL'
        print(f'    [{flag}] {pip_name}')


def download_model(venv_python):
    """通过 ModelScope 下载 CosyVoice2-0.5B 模型权重（~2GB）。

    Colab Jupyter 环境下 tqdm 进度条会逐行刷新导致页面卡死，
    因此禁用 modelscope 的进度条，改为静默下载 + 阶段性 print。
    """
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f'[skip] 模型已存在: {MODEL_DIR}')
        return
    step(f'下载模型 {MODELSCOPE_ID}（~2GB，从 ModelScope）')
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    # 用独立脚本下载，禁用所有进度条输出
    # modelscope 的 snapshot_download 内部用 tqdm，在 Colab Jupyter 里逐文件刷新
    # 会导致页面卡死。通过 patch tqdm 为空操作彻底禁用。
    model_dir_str = str(MODEL_DIR).replace('\\', '/')
    script = (
        "import os, sys\n"
        "os.environ['TQDM_DISABLE'] = '1'\n"
        "os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'\n"
        "\n"
        "# Patch tqdm 为空操作，彻底禁用进度条刷屏\n"
        "import tqdm\n"
        "class _NoProgress:\n"
        "    def __init__(self, *a, **k): pass\n"
        "    def update(self, *a, **k): pass\n"
        "    def close(self): pass\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): pass\n"
        "    def write(self, s, **k): pass\n"
        "tqdm.tqdm = _NoProgress\n"
        "try:\n"
        "    import tqdm.auto as _ta; _ta.tqdm = _NoProgress\n"
        "except Exception: pass\n"
        "\n"
        "print('开始下载（进度条已禁用，请耐心等待）...', flush=True)\n"
        "from modelscope import snapshot_download\n"
        f"path = snapshot_download('{MODELSCOPE_ID}', local_dir='{model_dir_str}')\n"
        f"print('模型下载完成:', path, flush=True)\n"
        "\n"
        "# 列出下载的文件\n"
        f"for root, dirs, files in os.walk('{model_dir_str}'):\n"
        "    for f in files:\n"
        "        fp = os.path.join(root, f)\n"
        "        size_mb = os.path.getsize(fp) / 1024 / 1024\n"
        f"        rel = os.path.relpath(fp, '{model_dir_str}')\n"
        "        print(f'  {rel} ({size_mb:.1f} MB)')\n"
    )
    run([str(venv_python), '-c', script], cwd=REPO_DIR)


def check(use_system_python=False):
    """检查部署完整性。"""
    step('部署完整性检查')
    if use_system_python:
        venv_python = sys.executable
        venv_ok = True  # 系统 Python 总是存在
    else:
        venv_python = _venv_python()
        venv_ok = venv_python.is_file()
    checks = [
        ('仓库', REPO_DIR.is_dir() and (REPO_DIR / 'cosyvoice').is_dir()),
        ('venv' if not use_system_python else 'python', venv_ok),
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
    ap.add_argument('--system-python', action='store_true',
                    help='使用系统 Python，不创建 venv（Colab 环境）')
    ap.add_argument('--check', action='store_true', help='仅检查部署完整性')
    args = ap.parse_args()

    if args.check:
        return check(use_system_python=args.system_python)

    clone_repo()

    if args.skip_venv:
        venv_python = str(_venv_python())
        if not Path(venv_python).exists():
            print(f'错误: --skip-venv 但 venv 不存在: {venv_python}')
            return 1
    else:
        venv_python = create_venv(use_system_python=args.system_python)

    if not args.skip_deps:
        install_deps(venv_python, use_system_python=args.system_python)

    download_model(venv_python)

    print('\n部署完成。验证:')
    print(f'  {venv_python} tts-training/cosyvoice_verify.py')
    return check(use_system_python=args.system_python)


if __name__ == '__main__':
    sys.exit(main())
