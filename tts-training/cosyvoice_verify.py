#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice2-0.5B 音色克隆验证脚本

验证 PLAN.md 5.2 节「CosyVoice 跨语言克隆（教师模型）」流程：
  日语音色参考 (ref3_clean.wav) + 中文文本 -> 中文酒狐音色音频

支持六种对照模式（用 --mode all 一次跑完全部，或用 --modes 指定子集）：
  zero         inference_zero_shot(text, prompt_text, ref_wav)
               完整 prompt_text（42 字符 3 句日语）
  zero_short   prompt_text 只取首句（15 字符）
               规避「too short」警告，但跨语言对齐会失败（输出变日语）
  zero_2sent   prompt_text 取前两句（约 30 字符，平衡点）
               既规避警告，又给模型足够语言切换信息，预期保持中文输出
  cross        inference_cross_lingual(text, ref_wav)
               仅用参考音频声学特征，不传 prompt_text
  instruct2    inference_instruct2(text, instruct_text, ref_wav)
               用自然语言指令描述语气
  instruct2_zh 同 instruct2，但指令明确强调「用中文」
               针对跨语言场景，强制模型输出中文

成功标准：
  1. 模型能正常加载
  2. 推理产出非空音频
  3. 输出 wav 文件可播放，音色与日语参考相似（人工主观评估）
  4. 输出语言为中文（人工主观评估）

用法（用部署好的 venv 执行）：
    # 默认 zero 模式
    tts-training/CosyVoice-0.5B/venv/Scripts/python.exe tts-training/cosyvoice_verify.py
    # 一次跑完全部对照模式（约 75 分钟，CPU 慢）
    tts-training/CosyVoice-0.5B/venv/Scripts/python.exe tts-training/cosyvoice_verify.py --mode all
    # 只跑指定模式（推荐用于快速对比）
    tts-training/CosyVoice-0.5B/venv/Scripts/python.exe tts-training/cosyvoice_verify.py --modes zero_2sent,instruct2_zh
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
DEPLOY_DIR = HERE / 'CosyVoice-0.5B'
REPO_DIR = DEPLOY_DIR / 'repo'
MODEL_DIR = DEPLOY_DIR / 'pretrained_models' / 'CosyVoice2-0.5B'

# 默认参考音频：已清洗的日语酒狐语音（来自 littlemaid_peco 语音包拼接）
DEFAULT_REF = HERE / 'ref3_clean.wav'

# 验证用中文文本（覆盖日常问候 + 长句，便于人工评估音色与自然度）
TEST_TEXTS = [
    '主人早上好呀，今天想做什么呢？',
    '嘿嘿，我又给你烤了饼干，快尝尝看嘛。',
    '酒狐一直都会陪着你的，不管发生什么事情，都不会离开你半步的。',
]

# instruct2 模式默认语气指令（按模式区分）
DEFAULT_INSTRUCTS = {
    'instruct2':    '用温柔、撒娇、亲昵的语气说话，像是对主人撒娇的女仆',
    'instruct2_zh': '请用中文，以温柔撒娇的语气说话',
}

# 全部对照模式（--mode all 时的执行顺序）
ALL_MODES = ['zero', 'zero_short', 'zero_2sent', 'cross', 'instruct2', 'instruct2_zh']

# 单模式可选值（--mode / --modes 中的合法值）
SINGLE_MODES = ALL_MODES

# 需要 prompt_text 的 zero 系列模式
ZERO_MODES = ('zero', 'zero_short', 'zero_2sent')

# instruct2 系列模式
INSTRUCT_MODES = ('instruct2', 'instruct2_zh')


def _silence(seconds, sample_rate):
    """生成指定时长的静音 numpy 数组"""
    import numpy as np
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _concat(arrays):
    """拼接一维 numpy 数组列表"""
    import numpy as np
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)


def _first_n_sentences(text, n=1):
    """取 prompt_text 的前 n 句（按日语句号/问号切分）

    用于 zero_short / zero_2sent 模式：调整 prompt_text 长度
    - 首句（15字符）：太短，模型无法识别语言切换 -> 输出变日语
    - 前两句（30字符）：平衡点，既能规避「too short」警告，又能保持中文输出
    - 完整（42字符）：太长，合成短文本时语气漂移
    """
    import re
    parts = re.split(r'[。？\?]', text)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        return text
    n = min(n, len(sentences))
    return '。'.join(sentences[:n]) + '。'


def _synth_one(model, mode, text, ref_path, prompt_text, instruct_text):
    """调用对应模式合成一条，返回 chunks list"""
    if mode in ZERO_MODES:
        return list(model.inference_zero_shot(text, prompt_text, str(ref_path)))
    elif mode == 'cross':
        return list(model.inference_cross_lingual(text, str(ref_path)))
    elif mode in INSTRUCT_MODES:
        return list(model.inference_instruct2(text, instruct_text, str(ref_path)))
    raise ValueError(f'未知模式: {mode}')


def _save_audio(chunks, out_path, sample_rate, chunk_silence):
    """拼接 chunks 并写入 wav，返回 (时长, chunk_info)"""
    import torch
    chunk_audios = [c['tts_speech'].squeeze().cpu().numpy() for c in chunks]
    if chunk_silence > 0 and len(chunk_audios) > 1:
        silence = _silence(chunk_silence, sample_rate)
        merged = []
        for k, a in enumerate(chunk_audios):
            if a.ndim > 1:
                a = a[:, 0]
            merged.append(a)
            if k < len(chunk_audios) - 1:
                merged.append(silence)
        audio = _concat(merged)
        chunk_info = f'{len(chunk_audios)} chunks + {len(chunk_audios)-1} silence(×{chunk_silence:.2f}s)'
    else:
        audio = _concat(chunk_audios)
        chunk_info = f'{len(chunk_audios)} chunks'
    import soundfile as sf
    sf.write(str(out_path), audio, sample_rate)
    return len(audio) / sample_rate, chunk_info


def run_mode(model, mode, args, prompt_text, instruct_text, sample_rate, out_root):
    """跑单个模式的全部测试文本"""
    out_dir = out_root / f'verify_out_{mode}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'模式: {mode}')
    if mode in ZERO_MODES:
        print(f'prompt_text ({len(prompt_text)} chars): {prompt_text}')
    elif mode in INSTRUCT_MODES:
        print(f'instruct: {instruct_text}')
    print('=' * 60)

    success = 0
    for i, text in enumerate(TEST_TEXTS):
        print(f'\n[{i+1}/{len(TEST_TEXTS)}] 合成: {text}')
        t1 = time.time()
        try:
            chunks = _synth_one(model, mode, text, args.ref, prompt_text, instruct_text)
        except Exception as e:
            print(f'  合成失败: {e}')
            import traceback; traceback.print_exc()
            continue
        elapsed = time.time() - t1
        if not chunks:
            print('  合成返回空结果')
            continue
        out_path = out_dir / f'verify_{i:02d}.wav'
        dur, chunk_info = _save_audio(chunks, out_path, sample_rate, args.chunk_silence)
        rtf = elapsed / dur if dur > 0 else float('inf')
        print(f'  输出: {out_path.name} | {chunk_info} | 时长 {dur:.2f}s | 合成 {elapsed:.1f}s | RTF {rtf:.2f}')
        success += 1
    print(f'\n[{mode}] 完成: {success}/{len(TEST_TEXTS)} 条成功 -> {out_dir}')
    return success


def main():
    ap = argparse.ArgumentParser(description='CosyVoice2-0.5B 音色克隆验证')
    ap.add_argument('--model-dir', type=Path, default=MODEL_DIR,
                    help='CosyVoice2-0.5B 模型目录')
    ap.add_argument('--ref', type=Path, default=DEFAULT_REF,
                    help='参考音频（日语酒狐音色）')
    ap.add_argument('--mode', choices=SINGLE_MODES + ['all'], default='zero',
                    help='克隆模式（单模式）。可用 --modes 指定多个模式')
    ap.add_argument('--modes', type=str, default=None,
                    help='逗号分隔的多个模式，如 "zero_2sent,instruct2_zh"。'
                         '指定后覆盖 --mode')
    ap.add_argument('--prompt-text', type=str, default=None,
                    help='参考音频对应的转写文本（zero 系列模式必需）。'
                         '默认从 prompt_text_{ref_stem}.txt 读取')
    ap.add_argument('--instruct', type=str, default=None,
                    help='instruct2 系列模式的语气指令文字'
                         '（默认用各自模式 DEFAULT_INSTRUCTS）')
    ap.add_argument('--out', type=Path, default=None,
                    help='输出根目录（默认 verify_out_{mode}/）')
    ap.add_argument('--chunk-silence', type=float, default=0.30,
                    help='句子内多 chunk 之间的静音秒数（默认 0.30，'
                         'CosyVoice 会按标点切分长句，独立合成后拼接'
                         '若无停顿语气会漂移；设 0 关闭）')
    args = ap.parse_args()

    if not args.model_dir.is_dir():
        print(f'错误: 模型目录不存在: {args.model_dir}')
        print('请先运行: python tts-training/cosyvoice_setup.py')
        return 1
    if not args.ref.is_file():
        print(f'错误: 参考音频不存在: {args.ref}')
        return 1

    # 解析要跑的模式列表
    if args.modes:
        modes = [m.strip() for m in args.modes.split(',') if m.strip()]
        for m in modes:
            if m not in SINGLE_MODES:
                print(f'错误: 未知模式 {m}，可选: {SINGLE_MODES}')
                return 1
    else:
        modes = ALL_MODES if args.mode == 'all' else [args.mode]

    # zero 系列模式需要 prompt_text
    prompt_texts = {}
    if any(m in ZERO_MODES for m in modes):
        if args.prompt_text is not None:
            prompt_text_full = args.prompt_text
        else:
            cache = HERE / f'prompt_text_{args.ref.stem}.txt'
            if not cache.is_file():
                print(f'错误: zero 系列模式需要 prompt_text，未找到缓存 {cache}')
                print('请用 SenseVoice 转写参考音频，或用 --prompt-text "..." 指定')
                return 1
            prompt_text_full = cache.read_text(encoding='utf-8').strip()
        prompt_texts['zero'] = prompt_text_full
        prompt_texts['zero_short'] = _first_n_sentences(prompt_text_full, 1)
        prompt_texts['zero_2sent'] = _first_n_sentences(prompt_text_full, 2)
        print(f'prompt_text 完整   ({len(prompt_texts["zero"]):2d} chars): {prompt_texts["zero"]}')
        print(f'prompt_text 首句   ({len(prompt_texts["zero_short"]):2d} chars): {prompt_texts["zero_short"]}')
        print(f'prompt_text 前两句 ({len(prompt_texts["zero_2sent"]):2d} chars): {prompt_texts["zero_2sent"]}')

    # instruct2 系列模式的指令
    instruct_texts = {}
    if any(m in INSTRUCT_MODES for m in modes):
        for m in INSTRUCT_MODES:
            instruct_texts[m] = args.instruct if args.instruct else DEFAULT_INSTRUCTS[m]
        for m in [m for m in modes if m in INSTRUCT_MODES]:
            print(f'instruct[{m}]: {instruct_texts[m]}')

    out_root = args.out if args.out is not None else DEPLOY_DIR

    # 在 CosyVoice 仓库目录下导入，并补齐 Matcha-TTS 子模块路径（对齐 example.py）
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR / 'third_party' / 'Matcha-TTS'))
    print(f'\n加载 CosyVoice2 模型: {args.model_dir}')
    t0 = time.time()
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        model = AutoModel(model_dir=str(args.model_dir))
    except Exception as e:
        print(f'模型加载失败: {e}')
        import traceback; traceback.print_exc()
        return 1
    print(f'模型加载耗时: {time.time()-t0:.1f}s')

    import soundfile as sf
    sample_rate = model.sample_rate
    print(f'模型输出采样率: {sample_rate}Hz')

    print(f'\n参考音频: {args.ref}')
    ref_audio, ref_sr = sf.read(str(args.ref), dtype='float32')
    print(f'  采样率={ref_sr}Hz, 时长={len(ref_audio)/ref_sr:.2f}s')

    total_success = 0
    total_count = len(modes) * len(TEST_TEXTS)
    for mode in modes:
        pt = prompt_texts.get(mode)
        it = instruct_texts.get(mode)
        total_success += run_mode(model, mode, args, pt, it, sample_rate, out_root)

    print(f'\n' + '=' * 60)
    print(f'全部完成: {total_success}/{total_count} 条成功')
    print('=' * 60)
    for mode in modes:
        print(f'  {mode:14s} -> {out_root / f"verify_out_{mode}"}')
    print('\n请人工试听各模式 verify_*.wav，重点对比：')
    print('  - 输出语言是否为中文（zero_short 会变日语，是已知问题）')
    print('  - 音色相似度（与日语参考 ref3_clean.wav 对比）')
    print('  - 语气自然度（尤其 02 号长句）')
    print('  - zero_2sent 是否既能保持中文又有好音色（平衡点）')
    print('  - instruct2_zh 是否能用指令强制中文输出')
    return 0 if total_success == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
