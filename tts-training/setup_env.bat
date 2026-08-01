#!/usr/bin/env bat
@echo off
chcp 65001 >nul
echo ========================================
echo  Kokoro 本地 GPU 训练环境安装
echo  目标: P106-100 (Pascal, 6GB, torch 2.4.1)
echo ========================================
echo.

echo [1/4] 安装 torch 2.4.1 + CUDA 12.1 (Pascal 兼容)...
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo CUDA 12.1 安装失败，尝试 CUDA 11.8...
    pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
echo.

echo [2/4] 安装 Kokoro 和中文 G2P...
pip install kokoro misaki[zh] soundfile
if errorlevel 1 (
    echo misaki[zh] 安装失败，尝试单独安装...
    pip install misaki
    pip install pypinyin jieba
)
echo.

echo [3/4] 安装其他依赖...
pip install huggingface_hub numpy
echo.

echo [4/4] 验证安装...
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "from kokoro import KModel; print('kokoro: OK')"
python -c "from misaki import zh; print('misaki zh: OK')"
python -c "import soundfile; print('soundfile: OK')"

echo.
echo 安装完成！请运行: python train_local.py
pause
