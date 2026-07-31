"""Kokoro voice style 蒸馏训练：只训练一个 256 维音色嵌入向量。

原理：
  Kokoro 的 ref_s（256维）被分成两半：
    - ref_s[:, :128]  → decoder style（音色）
    - ref_s[:, 128:]  → predictor style（韵律）

  我们冻结整个 KModel，只优化一个 256 维的 ref_s 向量，
  使 Kokoro 用这个 style 合成出的音频与 CosyVoice 合成的目标音频匹配。

  整个 pipeline 是可微的（PyTorch），可以直接反向传播。

损失函数：
  多分辨率 STFT 损失（主）+ L1 波形损失（辅）

训练数据：
  - 目标音频：ref3_tts_dataset_routed_500/wavs/*.wav（CosyVoice 合成的酒狐音色）
  - 对应文本：manifest.jsonl 中的 text 字段

用法：
    python train_voice.py                     # 默认训练
    python train_voice.py --epochs 50         # 指定轮数
    python train_voice.py --init_voice zf_xiaobei  # 用现有 voice 初始化
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

# 设置 HF mirror（避免网络问题）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from kokoro import KPipeline, KModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "ref3_tts_dataset_routed_500")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")
OUTPUT_DIR = os.path.join(HERE, "voice_training_output")

STYLE_DIM = 256
SAMPLE_RATE = 24000


def enable_grad_inference(model):
    """去掉 KModel.forward_with_tokens 的 @torch.no_grad() 装饰器，
    使前向传播可微（用于 style 向量反向传播）。"""
    # 保存原始 forward_with_tokens 逻辑
    import types
    import kokoro.model as km

    # 原始 forward_with_tokens 的源码（去掉了 @torch.no_grad()）
    def forward_with_tokens_grad(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple:
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long
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

    # 替换方法
    model.forward_with_tokens = types.MethodType(forward_with_tokens_grad, model)

    # 冻结所有参数（我们只训练 style 向量）
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def stft_loss(y_pred, y_target, n_fft=1024, hop=256):
    """多分辨率 STFT 损失。"""
    # 确保相同长度
    min_len = min(y_pred.shape[-1], y_target.shape[-1])
    y_pred = y_pred[..., :min_len]
    y_target = y_target[..., :min_len]

    losses = []
    for fft_size in [512, 1024, 2048]:
        hop_size = fft_size // 4
        window = torch.hann_window(fft_size, device=y_pred.device)

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

        # L1 on magnitude
        loss_mag = (pred_mag - target_mag).abs().mean()
        # Spectral convergence
        loss_sc = (
            torch.norm(target_mag - pred_mag, p='fro')
            / torch.norm(target_mag, p='fro').clamp(min=1e-7)
        )
        losses.append(loss_mag + loss_sc)

    return sum(losses) / len(losses)


def waveform_loss(y_pred, y_target):
    """L1 波形损失。"""
    min_len = min(y_pred.shape[-1], y_target.shape[-1])
    return (y_pred[..., :min_len] - y_target[..., :min_len]).abs().mean()


def main():
    parser = argparse.ArgumentParser(description="Kokoro voice style 蒸馏训练")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--init_voice", default="zf_xiaobei",
                        help="用现有 voice 初始化（zf_xiaobei / af_heart 等）")
    parser.add_argument("--limit", type=int, default=None, help="限制训练样本数")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "winefox_style.pt"))
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Kokoro Voice Style 蒸馏训练")
    print("=" * 60)

    # 1. 加载 Kokoro 模型
    print("\n[1/4] 加载 Kokoro 模型...")
    device = torch.device('cpu')
    model = KModel().to(device)
    # pipeline 仅用于 G2P（不加载模型）
    pipeline = KPipeline(lang_code='z', model=False)

    # 启用可微推理
    model = enable_grad_inference(model)
    print(f"  模型加载完成，参数已冻结")

    # 2. 加载初始 voice style
    print(f"\n[2/4] 加载初始 voice: {args.init_voice}")
    voice_pack = pipeline.load_voice(args.init_voice).to(device)
    print(f"  voice pack shape: {voice_pack.shape}")  # [N, 1, 256]

    # style 向量是可训练参数
    # voice pack 是 [max_tokens, 1, 256]，取一个位置初始化
    init_style = voice_pack[10, 0].clone()  # [256]
    style = nn.Parameter(init_style.to(device))
    print(f"  style shape: {style.shape}, init norm: {style.data.norm().item():.4f}")

    # 3. 加载训练数据
    print(f"\n[3/4] 加载训练数据...")
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

    # 预处理：文本 → phonemes
    print("  预处理文本（G2P）...")
    processed = []
    for i, rec in enumerate(records):
        try:
            # 直接调用 G2P（不经过完整 pipeline）
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

    # 4. 训练
    print(f"\n[4/4] 开始训练: epochs={args.epochs}, lr={args.lr}")
    print(f"  style 向量维度: {STYLE_DIM}")
    print(f"  损失函数: STFT + 0.1*waveform")

    optimizer = torch.optim.Adam([style], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    best_style = None

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_stft = 0.0
        epoch_wav = 0.0
        n_samples = 0
        t0 = time.time()

        for rec in processed:
            try:
                # 加载目标音频
                target_audio, sr = torchaudio.load(rec["audio_path"])
                if sr != SAMPLE_RATE:
                    target_audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(target_audio)
                target_audio = target_audio.squeeze(0).to(device)

                # 文本 phonemes
                phonemes = rec["phonemes"]

                # 转换为 input_ids
                input_ids_list = list(filter(
                    lambda i: i is not None,
                    map(lambda p: model.vocab.get(p), phonemes)
                ))
                if len(input_ids_list) < 2:
                    continue
                input_ids = torch.LongTensor([[0, *input_ids_list, 0]]).to(device)

                # 可微前向传播
                ref_s = style.unsqueeze(0)  # [1, 256]
                pred_audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, speed=1.0)

                # 计算损失
                loss_stft = stft_loss(pred_audio.unsqueeze(0), target_audio.unsqueeze(0))
                loss_wav = waveform_loss(pred_audio.unsqueeze(0), target_audio.unsqueeze(0))
                loss = loss_stft + 0.1 * loss_wav

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_stft += loss_stft.item()
                epoch_wav += loss_wav.item()
                n_samples += 1

            except Exception as e:
                import traceback
                if n_samples == 0 and epoch == 0:
                    print(f"    [debug] 训练异常: {e}")
                    traceback.print_exc()
                continue

        if n_samples == 0:
            print(f"Epoch {epoch+1}: 无有效样本，跳过")
            continue

        avg_loss = epoch_loss / n_samples
        avg_stft = epoch_stft / n_samples
        avg_wav = epoch_wav / n_samples
        elapsed = time.time() - t0

        scheduler.step()

        # 保存最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_style = style.data.clone()
            torch.save({
                'style': best_style,
                'epoch': epoch + 1,
                'loss': best_loss,
            }, args.output)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"loss={avg_loss:.4f} (stft={avg_stft:.4f}, wav={avg_wav:.4f}) "
              f"lr={scheduler.get_last_lr()[0]:.6f} "
              f"time={elapsed:.1f}s "
              f"{'★ best' if avg_loss < best_loss else ''}")

        # 定期保存 checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"style_epoch{epoch+1}.pt")
            torch.save({'style': style.data.clone(), 'epoch': epoch + 1}, ckpt_path)

    print(f"\n{'='*60}")
    print(f"训练完成！")
    if best_style is not None:
        print(f"  最佳损失: {best_loss:.4f}")
        print(f"  Style 向量已保存到: {args.output}")
        print(f"  style norm: {best_style.norm().item():.4f}")
        print(f"{'='*60}")

        # 导出为 voice pack 格式（兼容 voices.bin）
        export_path = os.path.join(OUTPUT_DIR, "winefox_voices.npy")
        export_voice_pack(best_style, export_path)
        print(f"  Voice pack 已导出到: {export_path}")
    else:
        print(f"  无有效训练数据，未生成 style")
        print(f"{'='*60}")
        return


def export_voice_pack(style_vec, output_path):
    """将 style 向量导出为 voice pack 格式。

    Kokoro 的 voice pack 是 [max_tokens, 256] 的张量，
    每个 token 长度对应一个 style 向量。
    我们用同一个 style 向量填充所有位置。
    """
    max_tokens = 512
    voice_pack = style_vec.unsqueeze(0).repeat(max_tokens, 1)  # [512, 256]
    # 保存为 npy（兼容 kokoro Python）
    np.save(output_path, voice_pack.numpy())
    print(f"  Voice pack shape: {voice_pack.shape}")


if __name__ == "__main__":
    main()
