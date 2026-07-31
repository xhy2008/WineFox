"""验证训练后的 voice style：合成样本并保存。"""
import os
import torch
import torchaudio
import numpy as np
from kokoro import KPipeline, KModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
STYLE_PATH = os.path.join(OUTPUT_DIR, "winefox_style.pt")
SAMPLE_RATE = 24000

# 测试文本（含翘舌音、情感词）
TEST_TEXTS = [
    "主人，今天天气真好，我们去公园散步吧。",
    "知道啦知道啦，本小姐正在准备呢。",
    "诶？！真的假的？不要小看我哦。",
    "大正女仆就是要端庄贤淑，这可是基本中的基本。",
]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型和 style
    model = KModel().to(device)
    pipeline = KPipeline(lang_code='z', model=model)

    ckpt = torch.load(STYLE_PATH, map_location=device)
    style = ckpt['style'].to(device)
    print(f"加载 style: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss')}, norm={style.norm().item():.4f}")

    # 构造 voice pack（512 × 256）
    voice_pack = style.unsqueeze(0).repeat(512, 1).unsqueeze(1)  # [512, 1, 256]

    # 合成测试样本
    os.makedirs(os.path.join(OUTPUT_DIR, "verify"), exist_ok=True)
    for i, text in enumerate(TEST_TEXTS):
        results = list(pipeline(text, voice=voice_pack, model=model))
        for j, r in enumerate(results):
            if r.audio is None:
                continue
            out_path = os.path.join(OUTPUT_DIR, "verify", f"sample_{i}_{j}.wav")
            torchaudio.save(out_path, r.audio.unsqueeze(0).cpu(), SAMPLE_RATE)
            print(f"  [{i}.{j}] {text[:30]}... → {out_path} ({r.audio.shape[-1]/SAMPLE_RATE:.1f}s)")

    print(f"\n验证样本已保存到 {OUTPUT_DIR}/verify/")


if __name__ == "__main__":
    main()
