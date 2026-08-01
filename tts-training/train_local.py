#!/usr/bin/env python3
"""Kokoro voice style 本地 GPU 微调脚本

针对 P106-100 (6GB VRAM, Pascal 架构 cc 6.1) 优化:
- batch_size=1, 避免显存溢出
- 不使用混合精度 (Pascal 无 Tensor Core, fp16 不加速)
- 预加载目标 mel 到 GPU 减少重复计算
- 显式释放中间张量
- 支持 checkpoint 恢复

用法:
    python train_local.py --epochs 30 --lr 1e-3
    python train_local.py --resume checkpoints/winefox_ep010.pt
"""

import os
import sys
import json
import time
import argparse
import gc
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from kokoro import KModel
from misaki import zh
from huggingface_hub import hf_hub_download

# Mel 参数 (与 Kokoro/StyleTTS2 一致)
SR = 24000
N_FFT = 2048
WIN_LENGTH = 1200
HOP_LENGTH = 300
N_MELS = 80
MEL_MEAN = -4.0
MEL_STD = 4.0
MAX_TOKENS = 510


def make_mel_transform(device):
    return torchaudio.transforms.MelSpectrogram(
        n_mels=N_MELS, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, sample_rate=SR
    ).to(device)


def waveform_to_mel(wav, mel_transform):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mel = mel_transform(wav)
    mel = (torch.log1p(1e-5 + mel) - MEL_MEAN) / MEL_STD
    return mel


def prepare_dataset(manifest_path, data_dir, g2p, model, max_samples=None):
    samples = []
    skipped = 0
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)
            if not s.get('qc_passed', False):
                continue
            text = s.get('text', '')
            audio_rel = s.get('audio', '')
            audio_path = os.path.join(data_dir, audio_rel)
            if not os.path.exists(audio_path):
                skipped += 1
                continue
            phonemes, _ = g2p(text)
            input_ids = list(filter(
                lambda i: i is not None,
                map(lambda p: model.vocab.get(p), phonemes)
            ))
            if len(input_ids) < 5 or len(input_ids) + 2 > model.context_length:
                skipped += 1
                continue
            dur = s.get('duration_seconds', 0)
            if dur < 2 or dur > 15:
                skipped += 1
                continue
            samples.append({
                'input_ids': input_ids,
                'audio_path': audio_path,
                'text': text[:50],
                'duration': dur,
            })
            if max_samples and len(samples) >= max_samples:
                break
    print(f'  Dataset: {len(samples)} samples, {skipped} skipped')
    return samples


def compute_forward_no_grad(model, input_ids, predictor_style, device):
    """no_grad 下计算 BERT + predictor + text_encoder, 返回 detached 张量。"""
    with torch.no_grad():
        T = input_ids.shape[-1]
        input_lengths = torch.full((1,), T, device=device, dtype=torch.long)
        text_mask = torch.arange(T, device=device).unsqueeze(0)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s = predictor_style
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(
            torch.arange(T, device=device), pred_dur
        )
        total_len = indices.shape[0]
        pred_aln_trg = torch.zeros((T, total_len), device=device)
        pred_aln_trg[indices, torch.arange(total_len, device=device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

    return asr.detach(), F0_pred.detach(), N_pred.detach()


def export_voice_pack(style_vec, output_path):
    """导出 [510, 1, 256] 的 Kokoro voice pack。"""
    voice_pack = style_vec.detach().unsqueeze(0).repeat(MAX_TOKENS, 1, 1)
    torch.save(voice_pack, output_path)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {props.name}')
        print(f'  VRAM: {props.total_memory / 1024**3:.1f} GB')
        print(f'  Compute capability: {props.major}.{props.minor}')
        torch.backends.cudnn.benchmark = True

    # 加载模型
    print('Loading Kokoro model (v1.1-zh)...')
    model = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh').to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f'  Model loaded. Context length: {model.context_length}')

    # 加载初始 voice pack 或恢复 checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        if ckpt.dim() == 3:
            init_style = ckpt[ckpt.shape[0] // 2].clone()
        else:
            init_style = ckpt.clone()
        # 从文件名提取 epoch
        import re
        m = re.search(r'ep(\d+)', args.resume)
        if m:
            start_epoch = int(m.group(1))
        print(f'  Resumed from epoch {start_epoch}')
    else:
        print(f'Loading initial voice pack: {args.init_voice}')
        vp_path = hf_hub_download(
            'hexgrad/Kokoro-82M-v1.1-zh', f'voices/{args.init_voice}.pt'
        )
        init_pack = torch.load(vp_path, weights_only=True).to(device)
        init_style = init_pack[len(init_pack) // 2].clone()
        print(f'  Init style norm: {init_style.norm().item():.4f}')
        print(f'    decoder_style norm: {init_style[0, :128].norm().item():.4f}')
        print(f'    predictor_style norm: {init_style[0, 128:].norm().item():.4f}')

    style = torch.nn.Parameter(init_style.to(device))

    optimizer = torch.optim.Adam([style], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # 跳过已训练的 epoch
    for _ in range(start_epoch):
        scheduler.step()

    mel_transform = make_mel_transform(device)

    # G2P
    print('Loading G2P...')
    g2p = zh.ZHG2P(version='1.1')

    # 准备数据集
    data_dir = os.path.dirname(os.path.abspath(args.manifest))
    samples = prepare_dataset(args.manifest, data_dir, g2p, model, args.max_samples)
    if not samples:
        print('Error: no valid training samples')
        sys.exit(1)

    # 预加载目标 mel 到 GPU
    print('Preloading target mel spectrograms to GPU...')
    total_mel_size = 0
    for s in samples:
        wav, sr = torchaudio.load(s['audio_path'])
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(0)
        s['target_mel'] = waveform_to_mel(wav, mel_transform)
        total_mel_size += s['target_mel'].element_size() * s['target_mel'].nelement()
    print(f'  Preloaded {len(samples)} mel spectrograms ({total_mel_size / 1024**2:.1f} MB)')

    # 检查显存
    if device.type == 'cuda':
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'  GPU memory after preload: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved')

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    print(f'\nStarting training: {args.epochs} epochs (from {start_epoch}), lr={args.lr}')
    print(f'  decoder_style (128 dims): learnable')
    print(f'  predictor_style (128 dims): fixed')

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        t_start = time.time()
        indices = np.random.permutation(len(samples))

        for i, idx in enumerate(indices):
            sample = samples[idx]
            input_ids = torch.LongTensor(
                [[0] + sample['input_ids'] + [0]]
            ).to(device)
            target_mel = sample['target_mel']

            # no_grad 部分
            predictor_style = style.data[:, 128:]
            asr, F0_pred, N_pred = compute_forward_no_grad(
                model, input_ids, predictor_style, device
            )

            # 有梯度部分: decoder
            decoder_style = style[:, :128]
            try:
                audio = model.decoder(asr, F0_pred, N_pred, decoder_style).squeeze()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f'  OOM at sample {idx}, skipping...')
                    torch.cuda.empty_cache()
                    continue
                else:
                    continue

            if audio.dim() == 0 or audio.shape[-1] < SR:
                del audio
                continue

            pred_mel = waveform_to_mel(audio, mel_transform)
            min_t = min(pred_mel.shape[-1], target_mel.shape[-1])
            if min_t < 5:
                del audio, pred_mel
                continue

            loss = F.l1_loss(pred_mel[..., :min_t], target_mel[..., :min_t])

            optimizer.zero_grad()
            loss.backward()

            # 只保留 decoder_style 的梯度
            if style.grad is not None:
                style.grad[:, 128:] = 0

            torch.nn.utils.clip_grad_norm_([style], max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_samples += 1

            # 释放中间张量
            del audio, pred_mel, loss, asr, F0_pred, N_pred

            if (i + 1) % 50 == 0:
                vram_str = ''
                if device.type == 'cuda':
                    vram_str = f' vram={torch.cuda.memory_allocated()/1024**3:.2f}GB'
                print(f'  Ep{epoch+1} [{i+1}/{len(samples)}] '
                      f'loss={loss.item():.4f} '
                      f'dec_norm={style.data[0,:128].norm():.3f}'
                      f'{vram_str}')

        avg_loss = epoch_loss / max(epoch_samples, 1)
        elapsed = time.time() - t_start
        print(f'Epoch {epoch+1}/{args.epochs} avg_loss={avg_loss:.4f} '
              f'lr={optimizer.param_groups[0]["lr"]:.6f} '
              f'time={elapsed:.1f}s')

        scheduler.step()

        # 保存 checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f'winefox_ep{epoch+1:03d}.pt')
            export_voice_pack(style, save_path)
            print(f'  Saved: {save_path}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, 'winefox_best.pt')
                export_voice_pack(style, best_path)
                print(f'  New best: {best_path}')

        # 清理显存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    # 最终保存
    final_path = os.path.join(args.output_dir, 'winefox_final.pt')
    export_voice_pack(style, final_path)
    print(f'\nTraining complete. Final voice pack: {final_path}')
    print(f'Best loss: {best_loss:.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Kokoro voice style on local GPU'
    )
    parser.add_argument(
        '--manifest', default='ref3_tts_dataset_routed_500/manifest.jsonl',
        help='Path to manifest.jsonl'
    )
    parser.add_argument(
        '--init-voice', default='zf_001',
        help='Initial voice pack name (e.g. zf_001)'
    )
    parser.add_argument(
        '--resume', default=None,
        help='Resume from checkpoint .pt file'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Max number of training samples'
    )
    parser.add_argument(
        '--save-every', type=int, default=5,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--output-dir', default='checkpoints',
        help='Output directory for voice packs'
    )
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
