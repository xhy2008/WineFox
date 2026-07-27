#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kokoro Voice Style 微调训练脚本

目标（PLAN.md 5.2 节）：用 CosyVoice 生成的酒狐音色音频作为教师，
微调 Kokoro voice style matrix，使 Kokoro 输出酒狐音色。

训练方法：
  - 冻结 Kokoro 模型权重（encoder + decoder）
  - 将 voice style matrix [510, 256] 作为可训练参数
  - 损失：mel-spectrogram L1 + L2 正则化（约束不偏离初始 voice 太远）
  - 训练后将 voice style 写入 voices.bin

两种运行模式：
  1. CPU 验证（少量数据，验证流程）：
     python train.py --cpu --epochs 1 --batch-size 2

  2. Colab 大规模训练（GPU）：
     python train.py --gpu --epochs 50 --batch-size 16

前置条件：
  - 已运行 setup_kokoro.py 下载 Kokoro 源码与权重
  - 已运行 gen_test_dataset.py（CPU 验证）或 cosyvoice_dataset/generate.py（大规模训练）
  - 数据集目录：tts-training/dataset/{wavs/, metadata.csv}
"""

import argparse
import csv
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

HERE = Path(__file__).parent.resolve()
TTS_TRAINING = HERE.parent
KOKORO_SRC = TTS_TRAINING / 'kokoro-src'
KOKORO_MODEL = TTS_TRAINING / 'kokoro-model'
VOICES_BIN = TTS_TRAINING / 'voices-v1.1-zh.bin'
DATASET_DIR = TTS_TRAINING / 'dataset'

CONFIG = KOKORO_MODEL / 'config.json'
CHECKPOINT = KOKORO_MODEL / 'kokoro-v1_0.pth'

# Kokoro 采样率
SAMPLE_RATE = 24000
# Voice style 维度
STYLE_DIM = 256
# 最大 phoneme 长度
MAX_PHONEME_LEN = 510


# ---------------------------------------------------------------------------
# Kokoro 模型加载
# ---------------------------------------------------------------------------

def load_kokoro():
    """加载 Kokoro KModel + KPipeline（用于合成/前向传播）。"""
    from kokoro import KModel, KPipeline

    kmodel = KModel(config=str(CONFIG), model=str(CHECKPOINT), disable_complex=True)
    kmodel.eval().cpu()
    pipeline = KPipeline(lang_code='z', model=kmodel, device='cpu')
    return kmodel, pipeline


def load_g2p_pipeline():
    """加载仅用于 G2P 的 KPipeline（不加载模型，避免 voice 加载）。

    用于数据集预处理：文本 → phonemes → input_ids。
    """
    from kokoro import KPipeline
    # model=False 时 self.model=None，pipeline 不会触发合成与 voice 加载
    return KPipeline(lang_code='z', model=False, device='cpu')


# ---------------------------------------------------------------------------
# Voice style 加载与导出
# ---------------------------------------------------------------------------

def load_voice_from_bin(voices_path, voice_name):
    """从 voices.bin 加载指定 voice 的 style matrix [510, 256]。

    voices.bin 中每个 voice 存储为 130560 个 float32（510*256）。
    返回 2D shape [510, 256]，可直接作为 nn.Parameter 训练。
    """
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
                assert dim == MAX_PHONEME_LEN * STYLE_DIM, \
                    f'Unexpected dim: {dim} != {MAX_PHONEME_LEN * STYLE_DIM}'
                return style.reshape(MAX_PHONEME_LEN, STYLE_DIM)

    raise KeyError(f'Voice {voice_name} not found in {voices_path}')


# ---------------------------------------------------------------------------
# Mel spectrogram 计算
# ---------------------------------------------------------------------------

class MelSpectrogram(nn.Module):
    """Mel spectrogram 计算，用于训练损失。

    使用 librosa 预计算 mel filter bank，然后用 torch.stft 计算 spectrogram。
    这样不依赖 torchaudio，且支持 GPU 加速。
    """
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256,
                 n_mels=80, fmin=0, fmax=12000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax
        )
        self.register_buffer('mel_fb', torch.FloatTensor(mel_fb))

    def forward(self, audio):
        # audio: [T] 或 [B, T]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            return_complex=True, window=torch.hann_window(self.n_fft, device=audio.device)
        )
        spec = stft.abs() ** 2  # [B, freq, time]
        mel = torch.matmul(self.mel_fb, spec)  # [B, n_mels, time]
        return torch.log1p(mel)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

class TtsDataset:
    """TTS 训练数据集。

    从 dataset/metadata.csv 加载，每条包含 (id, text, phonemes, duration)。
    音频文件在 dataset/wavs/{id}.wav。

    使用独立的 G2P pipeline（model=False）做文本转 phonemes，不触发 voice 加载。
    """
    def __init__(self, dataset_dir, g2p_pipeline, max_samples=None):
        self.dataset_dir = Path(dataset_dir)
        self.g2p = g2p_pipeline
        self.samples = []

        metadata_path = self.dataset_dir / 'metadata.csv'
        wavs_dir = self.dataset_dir / 'wavs'

        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) < 2:
                    continue
                wid, text = row[0], row[1]
                wav_path = wavs_dir / f'{wid}.wav'
                if not wav_path.exists():
                    continue
                self.samples.append((wid, text, wav_path))
                if max_samples and len(self.samples) >= max_samples:
                    break

        print(f'数据集: {len(self.samples)} 条样本')

    def __len__(self):
        return len(self.samples)

    def get_item(self, idx):
        """返回 (input_ids, phoneme_len, teacher_audio, text)。"""
        wid, text, wav_path = self.samples[idx]

        # 文本 → phonemes → input_ids（G2P only，不合成）
        input_ids = None
        phoneme_len = 0
        for result in self.g2p(text, voice=None, speed=1.0):
            phonemes = result.phonemes
            if not phonemes:
                continue
            ids = list(filter(
                lambda i: i is not None,
                map(lambda p: self.g2p.model.vocab.get(p) if self.g2p.model else None, phonemes)
            ))
            # G2P pipeline 的 self.model 是 None，用全局 vocab 兜底
            if not ids and self.g2p.model is None:
                # 通过 kmodel 的 vocab 转换（在外部传入）
                ids = self._vocab_lookup(phonemes)
            if len(ids) + 2 > MAX_PHONEME_LEN:
                ids = ids[:MAX_PHONEME_LEN - 2]
            ids = [0] + ids + [0]  # BOS + tokens + EOS
            input_ids = torch.LongTensor([ids])
            phoneme_len = len(phonemes)
            break  # 只取第一个 chunk

        if input_ids is None:
            return None

        # 加载教师音频
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype='float32')
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        teacher_audio = torch.FloatTensor(audio)

        return input_ids, phoneme_len, teacher_audio, text

    def _vocab_lookup(self, phonemes):
        """phonemes → input_ids（使用 KModel 的 vocab）。"""
        # 延迟加载，避免在 __init__ 时加载模型
        if not hasattr(self, '_vocab'):
            from kokoro import KModel
            kmodel = KModel(config=str(CONFIG), model=str(CHECKPOINT), disable_complex=True)
            self._vocab = kmodel.vocab
        return list(filter(
            lambda i: i is not None,
            map(lambda p: self._vocab.get(p), phonemes)
        ))


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train(args):
    """主训练函数。"""
    print('=' * 60)
    print('Kokoro Voice Style 微调训练')
    print('=' * 60)

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # 1. 加载 Kokoro 模型
    print('\n[1] 加载 Kokoro 模型...')
    kmodel, pipeline = load_kokoro()
    for param in kmodel.parameters():
        param.requires_grad = False
    kmodel.eval()
    print(f'  Kokoro 模型已冻结')

    # 2. 加载基础 voice style
    print(f'\n[2] 加载基础 voice: {args.base_voice}')
    base_style = load_voice_from_bin(VOICES_BIN, args.base_voice)
    print(f'  shape: {base_style.shape}')

    # 创建可训练的 voice style parameter [510, 256]
    voice_style = nn.Parameter(
        torch.FloatTensor(base_style).to(device)
    )
    print(f'  可训练参数量: {voice_style.numel()} ({voice_style.numel()*4/1024:.1f} KB)')

    # 3. 加载数据集（G2P only pipeline）
    print(f'\n[3] 加载数据集...')
    g2p_pipeline = load_g2p_pipeline()
    dataset = TtsDataset(DATASET_DIR, g2p_pipeline, max_samples=args.max_samples)
    if len(dataset) == 0:
        print('错误: 数据集为空')
        print('请先运行: python voice_finetune/gen_test_dataset.py')
        return 1

    # 4. Mel spectrogram 计算器
    mel_fn = MelSpectrogram(sample_rate=SAMPLE_RATE).to(device)

    # 5. 优化器
    optimizer = torch.optim.Adam([voice_style], lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataset), eta_min=args.lr * 0.01
    )

    # 6. 训练循环
    print(f'\n[4] 开始训练: {args.epochs} epochs, batch_size={args.batch_size}')
    print(f'  lr={args.lr}, l2_reg={args.l2_reg}')

    best_loss = float('inf')
    output_dir = HERE / 'output'
    output_dir.mkdir(exist_ok=True)
    base_style_tensor = torch.FloatTensor(base_style).to(device)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_reg_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        indices = np.random.permutation(len(dataset))
        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            batch_items = [dataset.get_item(i) for i in batch_indices]
            batch_items = [item for item in batch_items if item is not None]

            if not batch_items:
                continue

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_mel_loss = 0.0
            batch_reg_loss = 0.0
            n_ok = 0

            for input_ids, phoneme_len, teacher_audio, text in batch_items:
                input_ids = input_ids.to(device)
                teacher_audio = teacher_audio.to(device)

                # 获取当前文本长度的 style vector: [1, 256]
                style_idx = min(phoneme_len - 1, MAX_PHONEME_LEN - 1)
                ref_s = voice_style[style_idx:style_idx+1]  # [1, 256]

                try:
                    # 使用 forward_with_tokens 直接传 input_ids
                    # 注意：speed 是 float，不是 tensor
                    synth_audio, _ = kmodel.forward_with_tokens(
                        input_ids, ref_s, speed=1.0
                    )
                    if synth_audio.dim() > 1:
                        synth_audio = synth_audio.squeeze(0)

                    # Mel loss
                    synth_mel = mel_fn(synth_audio)
                    teacher_mel = mel_fn(teacher_audio)
                    min_len = min(synth_mel.shape[-1], teacher_mel.shape[-1])
                    mel_loss = F.l1_loss(
                        synth_mel[..., :min_len],
                        teacher_mel[..., :min_len]
                    )

                    # L2 正则化：约束 voice_style 不偏离初始值太远
                    reg_loss = args.l2_reg * torch.mean(
                        (voice_style - base_style_tensor) ** 2
                    )

                    loss = mel_loss + reg_loss

                    batch_loss += loss.item()
                    batch_mel_loss += mel_loss.item()
                    batch_reg_loss += reg_loss.item()
                    loss.backward()
                    n_ok += 1

                except Exception as e:
                    print(f'  跳过样本 (forward 失败): {e}')
                    continue

            if n_ok > 0:
                torch.nn.utils.clip_grad_norm_([voice_style], max_norm=1.0)
                optimizer.step()
                scheduler.step()

                n_batches += 1
                epoch_loss += batch_loss / n_ok
                epoch_mel_loss += batch_mel_loss / n_ok
                epoch_reg_loss += batch_reg_loss / n_ok

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_mel = epoch_mel_loss / n_batches
            avg_reg = epoch_reg_loss / n_batches
            elapsed = time.time() - t_epoch
            print(f'  Epoch {epoch+1}/{args.epochs} | '
                  f'loss={avg_loss:.4f} (mel={avg_mel:.4f}, reg={avg_reg:.4f}) | '
                  f'lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s')

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = output_dir / 'voice_style_best.pt'
                torch.save({
                    'voice_style': voice_style.data.cpu(),
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'base_voice': args.base_voice,
                }, save_path)
                print(f'  → 保存最佳模型: {save_path}')

    # 保存最终模型
    final_path = output_dir / 'voice_style_final.pt'
    torch.save({
        'voice_style': voice_style.data.cpu(),
        'epoch': args.epochs,
        'loss': avg_loss if n_batches > 0 else float('inf'),
        'base_voice': args.base_voice,
    }, final_path)
    print(f'\n训练完成。最终模型: {final_path}')
    print(f'最佳 loss: {best_loss:.4f}')
    print(f'\n下一步: 运行 export_voice.py 将 voice_style 写入 voices.bin')

    return 0


def main():
    ap = argparse.ArgumentParser(description='Kokoro Voice Style 微调训练')
    ap.add_argument('--cpu', action='store_true', help='CPU 模式')
    ap.add_argument('--gpu', action='store_true', help='GPU 模式（Colab）')
    ap.add_argument('--epochs', type=int, default=50, help='训练轮数')
    ap.add_argument('--batch-size', type=int, default=4, help='批量大小')
    ap.add_argument('--lr', type=float, default=1e-3, help='学习率')
    ap.add_argument('--l2-reg', type=float, default=0.01, help='L2 正则化系数')
    ap.add_argument('--base-voice', type=str, default='zf_001',
                    help='基础 voice 名（用于初始化）')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='最大样本数（CPU 验证时可限制）')
    args = ap.parse_args()

    if not args.cpu and not args.gpu:
        print('请指定 --cpu 或 --gpu 模式')
        return 1

    if args.cpu:
        args.max_samples = args.max_samples or 5  # CPU 默认最多 5 条
        args.epochs = min(args.epochs, 2)  # CPU 最多 2 轮
        print('CPU 验证模式: 限制为 5 条样本, 最多 2 轮')

    return train(args)


if __name__ == '__main__':
    sys.exit(main())
