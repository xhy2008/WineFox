"""验证蒸馏后的 voicepack 合成质量。

对比：
  1. 用蒸馏后的 voicepack 合成音频
  2. 与参考音频（数据集中的原始音频）对比 ECAPA-TDNN 相似度
  3. 生成试听样本

用法：
    python verify_voice.py --voice winefox_voice.pt --ref_dir ..\ref3_tts_dataset_routed_500\wavs
"""
import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf

WAVLM_SR = 16000
KOKORO_SR = 24000

VERIFY_TEXTS = [
    "主人，今天天气真好，我们去公园散步吧。",
    "知道啦知道啦，本小姐正在准备呢。",
    "真正的强者，是不会被这种小事打倒的。",
    "嘿嘿，这个想法不错嘛，本小姐允许你继续说下去。",
    "夜晚的风带着一丝凉意，让人忍不住想要靠近温暖的东西。",
]


def load_ecapa(device):
    # 用 speechbrain.inference 路径：新版 speechbrain 的
    # `speechbrain.pretrained` lazy import 会触发 integrations.k2_fsa
    # （缺 k2 时直接炸），inference 路径不经过该集成。
    from speechbrain.inference.classifiers import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "spkrec-ecapa"),
        run_opts={"device": str(device)},
    )
    # 冻结参数：只作为特征提取器
    for p in enc.mods.embedding_model.parameters():
        p.requires_grad_(False)
    return enc


def ecapa_embedding(enc, audio_16k):
    """ECAPA-TDNN 说话人嵌入（192 维，L2 归一化）。

    audio_16k: (1, T) 16kHz 音频
    """
    emb = enc.encode_batch(audio_16k)   # (1, 1, 192)
    emb = emb.squeeze(0).squeeze(0)     # (192,)
    return torch.nn.functional.normalize(emb, dim=0)


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser(description="验证蒸馏后的 voicepack")
    parser.add_argument("--voice", required=True,
                        help=".pt voicepack 文件路径")
    parser.add_argument("--ref_dir", default=None,
                        help="参考音频目录（用于计算相似度）")
    parser.add_argument("--n_ref", type=int, default=5,
                        help="用于计算相似度的参考音频数量")
    parser.add_argument("--compare", default=None,
                        help="对比的预设 voice 名称（如 zf_001）")
    parser.add_argument("--output_dir", default="verify_output",
                        help="试听样本输出目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --- 加载模型 ---
    from kokoro import KModel, KPipeline
    print("[Kokoro] loading model...")
    model = KModel().to(device).eval()
    pipeline = KPipeline(lang_code="z", model=model, device=device)

    # --- 加载 voicepack ---
    voicepack = torch.load(args.voice, map_location=device, weights_only=True)
    # KPipeline.load_voice 要求 torch.FloatTensor
    voicepack = voicepack.float().to(device)
    # 保持完整的 (510, 1, 256) voicepack：KPipeline.infer 会按 pack[len(ps)-1] 索引取行
    # 梯度反演后所有行相同，但形状必须保持 (510, 1, 256) 以兼容索引
    voice_style = voicepack
    print(f"[Voice] loaded: shape={voicepack.shape}")

    # --- 加载 ECAPA-TDNN ---
    print("[ECAPA] loading speaker embedding model...")
    ecapa = load_ecapa(device)

    # --- 参考音频嵌入 ---
    ref_embeddings = []
    if args.ref_dir:
        all_wavs = sorted([
            os.path.join(args.ref_dir, f)
            for f in os.listdir(args.ref_dir)
            if f.endswith(('.wav', '.flac'))
        ])
        rng = np.random.RandomState(42)
        n = min(args.n_ref, len(all_wavs))
        idx = rng.choice(len(all_wavs), n, replace=False)
        ref_files = [all_wavs[i] for i in idx]

        for rf in ref_files:
            wav, _ = librosa.load(rf, sr=WAVLM_SR)
            wav, _ = librosa.effects.trim(wav, top_db=30)
            wav_t = torch.from_numpy(wav).float().to(device)
            with torch.no_grad():
                emb = ecapa_embedding(ecapa, wav_t.unsqueeze(0))
            ref_embeddings.append(emb)
        print(f"[Ref] {len(ref_embeddings)} reference embeddings computed")

    # --- 对比 voicepack ---
    compare_embeddings = []
    if args.compare:
        cmp_voicepack = pipeline.load_voice(args.compare)
        print(f"[Compare] preset voice: {args.compare}, shape={cmp_voicepack.shape}")

    # --- 合成样本并评估 ---
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"合成验证样本")
    print(f"{'='*60}")

    similarities = []
    for i, text in enumerate(VERIFY_TEXTS):
        # 用蒸馏 voicepack 合成
        # KPipeline.load_voice 支持 torch.FloatTensor 直接传入
        chunks = list(pipeline(text, voice=voice_style, speed=1.0))
        if not chunks:
            print(f"[{i}] SKIP: no chunks for '{text}'")
            continue

        output = chunks[0].output
        if output is None or output.audio is None:
            print(f"[{i}] SKIP: no audio")
            continue
        audio = output.audio

        # 保存音频
        out_path = os.path.join(args.output_dir, f"verify_{i}.wav")
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = np.array(audio)
        sf.write(out_path, audio_np, KOKORO_SR)
        dur = len(audio_np) / KOKORO_SR
        print(f"[{i}] {out_path} ({dur:.1f}s) | {text}")

        # 计算与参考的相似度
        if ref_embeddings:
            audio_16k = torchaudio.functional.resample(
                torch.from_numpy(audio_np).unsqueeze(0).float().to(device),
                KOKORO_SR, WAVLM_SR
            )
            with torch.no_grad():
                syn_emb = ecapa_embedding(ecapa, audio_16k)

            sims = [cosine_sim(syn_emb, ref_emb) for ref_emb in ref_embeddings]
            avg_sim = np.mean(sims)
            similarities.append(avg_sim)
            print(f"     similarity: avg={avg_sim:.4f} "
                  f"(min={min(sims):.4f}, max={max(sims):.4f})")

    # --- 汇总 ---
    print(f"\n{'='*60}")
    print(f"验证报告")
    print(f"{'='*60}")
    print(f"voicepack: {args.voice}")
    print(f"样本数:    {len(VERIFY_TEXTS)}")
    if similarities:
        avg = np.mean(similarities)
        print(f"ECAPA 相似度: avg={avg:.4f}")
        print(f"  (>0.5 优秀, >0.3 可用, <0.2 需优化)")
    if args.compare:
        print(f"对比预设:    {args.compare}")
    print(f"输出目录:    {args.output_dir}")


if __name__ == "__main__":
    main()
