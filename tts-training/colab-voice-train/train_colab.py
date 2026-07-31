"""Kokoro voice style 蒸馏训练（Colab GPU 版）。

模型从 HuggingFace 运行时下载（~313MB），数据集从同目录加载。

目录结构（上传到 Colab）：
  colab-voice-train/
  ├── dataset/
  │   ├── manifest.jsonl
  │   └── wavs/*.wav
  ├── train_colab.py       ← 本文件
  └── verify.py            ← 验证脚本

Colab 用法：
    !pip install kokoro misaki
    %cd /content/colab-voice-train
    !python train_colab.py --epochs 100 --lr 0.02
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from kokoro import KPipeline, KModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")
OUTPUT_DIR = os.path.join(HERE, "output")

STYLE_DIM = 256
SAMPLE_RATE = 24000


def enable_grad_inference(model):
    """去掉 KModel.forward_with_tokens 的 @torch.no_grad()，使前向可微。"""
    import types

    def forward_with_tokens_grad(self, input_ids, ref_s, speed=1):
        input_lengths = torch.full(
            (input_ids.shape[0],), input_ids.shape[-1],
            device=input_ids.device, dtype=torch.long
        )
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1
        ).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=self.device, dtype=torch.long),
            pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=self.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward_with_cached(self, d_en, t_en, input_lengths, text_mask,
                            n_tokens, ref_s, speed=1):
        """用预计算的 d_en（BERT 输出）和 t_en（text_encoder 输出）进行前向。
        跳过 BERT 和 text_encoder，只跑 predictor + decoder。
        这两个模块不依赖 style，输出固定，可安全缓存。"""
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(n_tokens, device=self.device, dtype=torch.long),
            pred_dur
        )
        pred_aln_trg = torch.zeros(
            (n_tokens, indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=self.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    model.forward_with_tokens = types.MethodType(forward_with_tokens_grad, model)
    model.forward_with_cached = types.MethodType(forward_with_cached, model)
    for param in model.parameters():
        param.requires_grad = False
    # 不调用 eval()：cudnn RNN backward 要求 training 模式
    # 参数已冻结，即使 train 模式也不会更新模型权重
    return model


def stft_loss(y_pred, y_target, stft_windows):
    """多分辨率 STFT 损失（与 StyleTTS2 官方 MultiResolutionSTFTLoss 对齐）。
    每个分辨率：spectral convergence + log magnitude loss。
    stft_windows: {fft_size: window_tensor} 预创建的窗口。
    """
    min_len = min(y_pred.shape[-1], y_target.shape[-1])
    y_pred = y_pred[..., :min_len]
    y_target = y_target[..., :min_len]

    losses = []
    for fft_size, window in stft_windows.items():
        hop_size = fft_size // 4
        pred_stft = torch.stft(
            y_pred.squeeze(0), n_fft=fft_size, hop_length=hop_size,
            window=window, return_complex=True
        )
        target_stft = torch.stft(
            y_target.squeeze(0), n_fft=fft_size, hop_length=hop_size,
            window=window, return_complex=True
        )
        pred_mag = pred_stft.abs().clamp(min=1e-7)
        target_mag = target_stft.abs().clamp(min=1e-7)
        # Spectral convergence
        loss_sc = (
            torch.norm(target_mag - pred_mag, p='fro')
            / torch.norm(target_mag, p='fro').clamp(min=1e-7)
        )
        # Log magnitude loss（官方实现）
        loss_log = (torch.log(target_mag) - torch.log(pred_mag)).abs().mean()
        losses.append(loss_sc + loss_log)
    return sum(losses) / len(losses)


def waveform_loss(y_pred, y_target):
    min_len = min(y_pred.shape[-1], y_target.shape[-1])
    return (y_pred[..., :min_len] - y_target[..., :min_len]).abs().mean()


def main():
    parser = argparse.ArgumentParser(description="Kokoro voice style 蒸馏 (Colab GPU)")
    # 与 StyleTTS2 官方微调配置一致：epochs=50, lr=0.0001
    parser.add_argument("--epochs", type=int, default=50)
    # 参考 StyleTTS2 官方微调：lr=0.0001, AdamW, betas=(0.0,0.99), weight_decay=1e-4
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--init_voice", default="zf_xiaobei")
    parser.add_argument("--limit", type=int, default=None)
    # 官方不用 clip_grad_norm（用 slmadv 手动缩放），默认关闭
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="梯度裁剪最大范数（None=不裁剪，与官方一致）")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "winefox_style.pt"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--preprocessed", default=None,
                        help="本地预处理好的 pkl 路径，跳过 G2P 和 BERT 预计算")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("Kokoro Voice Style 蒸馏训练 (Colab)")
    print("=" * 60)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. 加载模型（从 HF 下载，约 313MB）
    print("\n[1/4] 加载 Kokoro 模型（从 HuggingFace 下载）...")
    model = KModel().to(device)
    model = enable_grad_inference(model)
    print("  模型加载完成，参数已冻结")

    # 2. G2P pipeline + 初始 voice
    print(f"\n[2/4] 初始化 G2P pipeline 和 voice: {args.init_voice}")
    pipeline = KPipeline(lang_code='z', model=False)
    voice_pack = pipeline.load_voice(args.init_voice).to(device)
    print(f"  voice pack shape: {voice_pack.shape}")

    init_style = voice_pack[10, 0].clone()
    style = nn.Parameter(init_style.to(device))

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        style.data = ckpt['style'].to(device)
        start_epoch = ckpt.get('epoch', 0)
        print(f"  从 epoch {start_epoch} 恢复")

    print(f"  style shape: {style.shape}, norm: {style.data.norm().item():.4f}")

    # 3. 加载数据
    print(f"\n[3/4] 加载训练数据...")

    if args.preprocessed and os.path.exists(args.preprocessed):
        # 从本地预处理好的 pkl 加载（跳过 G2P 和 BERT 预计算）
        import pickle
        print(f"  从 pkl 加载预处理数据: {args.preprocessed}")
        t_load = time.time()
        with open(args.preprocessed, 'rb') as f:
            raw_samples = pickle.load(f)
        if args.limit:
            raw_samples = raw_samples[:args.limit]

        # numpy → tensor 并移到 GPU
        samples = []
        for s in raw_samples:
            samples.append((
                torch.from_numpy(s['d_en']).to(device),
                torch.from_numpy(s['t_en']).to(device),
                torch.from_numpy(s['input_lengths']).to(device),
                torch.from_numpy(s['text_mask']).to(device),
                s['n_tokens'],
                torch.from_numpy(s['target_audio']).to(device),
            ))
        load_secs = time.time() - t_load
        audio_bytes = sum(s[5].element_size() * s[5].numel() for s in samples)
        cache_bytes = sum(
            s[0].element_size() * s[0].numel() + s[1].element_size() * s[1].numel()
            for s in samples
        )
        print(f"  加载完成: {len(samples)} 个样本, "
              f"音频显存 ~{audio_bytes / 1e6:.1f} MB, "
              f"BERT/text_encoder 缓存 ~{cache_bytes / 1e6:.1f} MB, "
              f"耗时 {load_secs:.1f}s")
    else:
        # 在线预处理（Colab 端 G2P + BERT）
        records = []
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records.append({
                    "text": rec["text"],
                    "audio_path": os.path.join(DATASET_DIR, rec["audio"]),
                })
        if args.limit:
            records = records[:args.limit]
        print(f"  {len(records)} 条训练数据")

        print("  G2P 预处理...")
        processed = []
        for i, rec in enumerate(records):
            try:
                ps, _ = pipeline.g2p(rec["text"])
                if not ps:
                    continue
                if len(ps) > 510:
                    ps = ps[:510]
                processed.append({
                    "text": rec["text"],
                    "audio_path": rec["audio_path"],
                    "phonemes": ps,
                })
            except Exception as e:
                print(f"    跳过 #{i}: {e}")
        print(f"  有效样本: {len(processed)}")

        # 预加载：一次性把音频读入 GPU，预计算 input_ids，并缓存 BERT 和 text_encoder 输出
        print("  预加载音频 + 预计算 BERT/text_encoder 输出...")
        samples = []
        t_preload = time.time()
        for rec in processed:
            try:
                input_ids_list = list(filter(
                    lambda i: i is not None,
                    map(lambda p: model.vocab.get(p), rec["phonemes"])
                ))
                if len(input_ids_list) < 2:
                    continue
                input_ids = torch.LongTensor([[0, *input_ids_list, 0]]).to(device)

                target_audio, sr = torchaudio.load(rec["audio_path"])
                if sr != SAMPLE_RATE:
                    target_audio = torchaudio.transforms.Resample(
                        sr, SAMPLE_RATE
                    )(target_audio)
                target_audio = target_audio.squeeze(0).to(device)

                with torch.no_grad():
                    input_lengths = torch.full(
                        (input_ids.shape[0],), input_ids.shape[-1],
                        device=device, dtype=torch.long
                    )
                    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
                        input_lengths.shape[0], -1
                    ).type_as(input_lengths)
                    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)
                    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
                    n_tokens = input_ids.shape[1]

                samples.append((d_en, t_en, input_lengths, text_mask, n_tokens, target_audio))
            except Exception as e:
                print(f"    预加载跳过: {e}")
        preload_secs = time.time() - t_preload
        audio_bytes = sum(s[5].element_size() * s[5].numel() for s in samples)
        cache_bytes = sum(
            s[0].element_size() * s[0].numel() + s[1].element_size() * s[1].numel()
            for s in samples
        )
        print(f"  预加载完成: {len(samples)} 个样本, "
              f"音频显存 ~{audio_bytes / 1e6:.1f} MB, "
              f"BERT/text_encoder 缓存 ~{cache_bytes / 1e6:.1f} MB, "
              f"耗时 {preload_secs:.1f}s")

    # 4. 训练
    print(f"\n[4/4] 开始训练: epochs={args.epochs}, lr={args.lr}")
    print(f"  混合精度: 开启 (FP16)")
    # 注意：不启用 cudnn.benchmark，因为每个样本 phoneme 长度不同 →
    # tensor 形状变化，cudnn 会对每个样本重新搜索算法，反而变慢 2x

    # 预创建 STFT 窗口（避免每样本重复创建）
    stft_windows = {
        fft_size: torch.hann_window(fft_size, device=device)
        for fft_size in [512, 2048]
    }

    # 参考 StyleTTS2 官方微调：AdamW + OneCycleLR
    # betas=(0.0, 0.99) 是 TTS 微调常用配置（降低动量），weight_decay=1e-4
    optimizer = torch.optim.AdamW(
        [style], lr=args.lr, betas=(0.0, 0.99), eps=1e-9, weight_decay=1e-4
    )
    # OneCycleLR: pct_start=0 表示无预热直接退火，div_factor=1 让初始 lr = max_lr
    # final_div_factor=1 让最终 lr 不会退到 0（保持 max_lr/1 = max_lr，实际靠 cosine 退火）
    # 注意：steps_per_epoch 必须等于实际迭代数，否则 scheduler 会报错
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs - start_epoch,
        steps_per_epoch=len(samples),
        pct_start=0.0,
        div_factor=1.0,
        final_div_factor=1.0,
    )
    scaler = torch.amp.GradScaler('cuda')  # 混合精度

    best_loss = float('inf')
    best_style = None

    for epoch in range(start_epoch, args.epochs):
        loss_acc = torch.zeros((), device=device)  # GPU 上累积，避免 .item() 同步
        n_samples = 0
        t0 = time.time()

        for d_en, t_en, input_lengths, text_mask, n_tokens, target_audio in samples:
            try:
                ref_s = style.unsqueeze(0)

                # 混合精度前向（用缓存的 BERT/text_encoder 输出，只跑 predictor + decoder）
                with torch.amp.autocast('cuda'):
                    pred_audio, _ = model.forward_with_cached(
                        d_en, t_en, input_lengths, text_mask, n_tokens, ref_s, speed=1.0
                    )
                    loss_stft = stft_loss(pred_audio.unsqueeze(0), target_audio.unsqueeze(0), stft_windows)
                    loss_wav = waveform_loss(pred_audio.unsqueeze(0), target_audio.unsqueeze(0))
                    # STFT 5x + wav 1.0：增大波形监督防止 style_norm 漂移过快
                    loss = 5.0 * loss_stft + 1.0 * loss_wav

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # 梯度裁剪（官方不用，仅在用户指定 --grad_clip 时启用）
                if args.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([style], max_norm=args.grad_clip)
                # 混合精度 + scheduler 的正确 pattern：
                # 当梯度溢出时 scaler.step 会跳过 optimizer.step，
                # 此时不应调用 scheduler.step（PyTorch 官方推荐做法）
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                # scale 减小说明发生溢出，跳过 scheduler.step
                if scale_before <= scale_after:
                    scheduler.step()

                # GPU 上累积 loss（避免每样本 .item() 同步）
                loss_acc += loss.detach()
                n_samples += 1
            except Exception as e:
                if n_samples == 0 and epoch == 0:
                    import traceback
                    print(f"    [debug] 异常: {e}")
                    traceback.print_exc()
                continue

        if n_samples == 0:
            print(f"Epoch {epoch+1}: 无有效样本")
            continue

        avg_loss = loss_acc.item() / n_samples
        style_norm = style.data.norm().item()
        elapsed = time.time() - t0
        # 取 epoch 最后一个 batch 的 lr 作为代表
        cur_lr = optimizer.param_groups[0]['lr']

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_style = style.data.clone()
            torch.save({
                'style': best_style, 'epoch': epoch + 1, 'loss': best_loss,
            }, args.output)

        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} "
              f"lr={cur_lr:.6f} "
              f"style_norm={style_norm:.3f} "
              f"time={elapsed:.1f}s "
              f"mem={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"{'★ best' if is_best else ''}")

        # 每 epoch 重置峰值统计，便于观察单 epoch 显存行为
        torch.cuda.reset_peak_memory_stats()

        # 每 epoch 释放 PyTorch caching allocator 的显存碎片
        torch.cuda.empty_cache()

        if (epoch + 1) % 10 == 0:
            torch.save({'style': style.data.clone(), 'epoch': epoch + 1},
                       os.path.join(OUTPUT_DIR, f"style_epoch{epoch+1}.pt"))

    print(f"\n{'='*60}")
    if best_style is not None:
        print(f"训练完成！最佳损失: {best_loss:.4f}")
        print(f"Style 已保存: {args.output}")
        export_path = os.path.join(OUTPUT_DIR, "winefox_voices.npy")
        voice_pack_out = best_style.cpu().unsqueeze(0).repeat(512, 1)
        np.save(export_path, voice_pack_out.numpy())
        print(f"Voice pack 导出: {export_path} shape={voice_pack_out.shape}")
    else:
        print("无有效训练数据")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
