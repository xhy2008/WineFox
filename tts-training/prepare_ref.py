#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 LittleMaid peco 语音包拼接出 GPT-SoVITS 参考音频 ref.wav。

策略：
- 扫描所有 ogg 时长
- 优先选 idle（闲聊语气，最中性适合做参考）类中较长的片段
- 累计 15~25 秒，片段间加 0.3 秒静音
- 输出 ref.wav (22050Hz mono int16)

同时输出 ref_files.txt 记录选了哪些文件（供参考）。
"""

import os
import re
import subprocess
import json
import random

random.seed(42)

SOUNDS_DIR = r'e:\winefox\tts-training\littlemaid_peco\sounds\maid'
GAP = 0.3  # 片段间静音
TARGET_MIN = 15.0
TARGET_MAX = 25.0

# 多组参考音频配置：每组用不同语气类别，便于对比克隆效果
REF_CONFIGS = [
    {
        'name': 'ref',  # 中性闲聊版
        'out_wav': r'e:\winefox\tts-training\ref.wav',
        'out_list': r'e:\winefox\tts-training\ref_files.txt',
        'priority': [
            'mode/idle',      # 闲聊，最中性
            'environment/morning',
            'environment/night',
            'ai/item_get',
            'ai/tamed',
        ],
    },
    {
        'name': 'ref2',  # 柔和版（问候 + 亲昵语气）
        'out_wav': r'e:\winefox\tts-training\ref2.wav',
        'out_list': r'e:\winefox\tts-training\ref2_files.txt',
        'priority': [
            'environment/morning',  # 早安问候，语气温柔
            'environment/night',    # 晚安，语气柔和
            'ai/tamed',             # 被驯服，亲昵撒娇语气
            'mode/idle',            # 闲聊作为补充
        ],
    },
]


def get_duration(path):
    """用 ffprobe 获取音频时长"""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def scan_all_ogg():
    """扫描所有 ogg 文件，返回 [(category, filename, path, duration), ...]
    category = 目录/文件名前缀，如 'mode/idle', 'environment/morning'
    """
    results = []
    for root, dirs, files in os.walk(SOUNDS_DIR):
        for fn in files:
            if not fn.endswith('.ogg'):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, SOUNDS_DIR).replace('\\', '/')
            rel_dir = '/'.join(rel.split('/')[:-1])
            # 文件名前缀：去掉数字和扩展名，如 idle1.ogg -> idle, hurt_fire1.ogg -> hurt_fire
            name_prefix = re.match(r'([a-z_]+)', fn).group(1)
            category = f'{rel_dir}/{name_prefix}'
            dur = get_duration(full)
            if dur > 0.5:
                results.append((category, fn, full, dur))
    return results


def select_segments(all_ogg, priority):
    """按指定优先类别选取片段，累计 15~25 秒"""
    by_cat = {}
    for cat, fn, path, dur in all_ogg:
        by_cat.setdefault(cat, []).append((fn, path, dur))

    selected = []
    total = 0.0

    for prio_cat in priority:
        if prio_cat not in by_cat:
            continue
        # 该类别下按时长降序，优先选较长的片段凑够目标时长
        items = sorted(by_cat[prio_cat], key=lambda x: -x[2])
        for fn, path, dur in items:
            if total >= TARGET_MAX:
                break
            selected.append((prio_cat, fn, path, dur))
            total += dur
        if total >= TARGET_MIN:
            break

    return selected, total


def concat_with_ffmpeg(selected, out_wav):
    """用 ffmpeg concat 拼接，片段间加静音，输出 22050Hz mono int16"""
    # 生成 silent gap 音频
    gap_wav = out_wav + '.gap.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-f', 'lavfi', '-i',
         f'anullsrc=channel_layout=mono:sample_rate=22050',
         '-t', str(GAP), '-c:a', 'pcm_s16le', gap_wav],
        capture_output=True, timeout=30)

    # 构建 concat 列表（每个片段 + gap）
    list_file = out_wav + '.concat.txt'
    with open(list_file, 'w', encoding='utf-8') as f:
        for i, (cat, fn, path, dur) in enumerate(selected):
            # 先转换每个 ogg 为临时 wav
            tmp_wav = out_wav + f'.seg{i}.wav'
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ar', '22050', '-ac', '1',
                 '-c:a', 'pcm_s16le', tmp_wav],
                capture_output=True, timeout=30)
            f.write(f"file '{os.path.basename(tmp_wav)}'\n")
            if i < len(selected) - 1:
                f.write(f"file '{os.path.basename(gap_wav)}'\n")

    # concat
    subprocess.run(
        ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file,
         '-c:a', 'pcm_s16le', out_wav],
        capture_output=True, timeout=60, cwd=os.path.dirname(out_wav))

    # 清理临时文件
    for tmp in [gap_wav, list_file] + [out_wav + f'.seg{i}.wav' for i in range(len(selected))]:
        try:
            os.remove(tmp)
        except OSError:
            pass


def main():
    print('扫描所有 ogg 文件...')
    all_ogg = scan_all_ogg()
    print(f'共 {len(all_ogg)} 个 ogg（>0.5秒）')

    # 类别统计（仅打印一次）
    cats = {}
    for cat, fn, path, dur in all_ogg:
        cats.setdefault(cat, []).append(dur)
    print('\n各类别统计:')
    for cat in sorted(cats.keys()):
        durs = cats[cat]
        print(f'  {cat}: {len(durs)} 个, 总时长 {sum(durs):.1f}s, 平均 {sum(durs)/len(durs):.1f}s')

    for cfg in REF_CONFIGS:
        print(f'\n{"="*60}')
        print(f'生成 {cfg["name"]}.wav（优先类别: {", ".join(cfg["priority"])}）')
        print('='*60)

        selected, total = select_segments(all_ogg, cfg['priority'])
        print(f'选取 {len(selected)} 个片段，累计 {total:.1f} 秒')
        print('选中片段:')
        for cat, fn, path, dur in selected:
            print(f'  {cat}/{fn} = {dur:.2f}s')

        print(f'\n拼接为 {cfg["out_wav"]} ...')
        concat_with_ffmpeg(selected, cfg['out_wav'])

        out_dur = get_duration(cfg['out_wav'])
        out_size = os.path.getsize(cfg['out_wav'])
        print(f'输出: {cfg["out_wav"]}')
        print(f'  时长: {out_dur:.2f} 秒')
        print(f'  大小: {out_size//1024} KB')
        print(f'  格式: 22050Hz mono int16 PCM')

        with open(cfg['out_list'], 'w', encoding='utf-8') as f:
            f.write(f'# {cfg["name"]}.wav 由以下 {len(selected)} 个片段拼接，累计 {total:.1f}s\n')
            f.write(f'# 优先类别: {", ".join(cfg["priority"])}\n')
            f.write(f'# 这些是日语语音，需要用 SenseVoice 转写得到 PROMPT_TEXT\n\n')
            for cat, fn, path, dur in selected:
                f.write(f'{cat}/{fn}\t{dur:.2f}s\n')
        print(f'选中文件列表: {cfg["out_list"]}')

    print('\n' + '='*60)
    print('全部生成完成。对比建议:')
    print('  两个 ref.wav 都上传到 Drive，在笔记本 A.5 cell 中切换 REF_AUDIO 路径')
    print('  分别合成一小批（如 50 条）试听，选音色更自然的一个继续完整合成')
    print('  ref.wav  = 中性闲聊语气')
    print('  ref2.wav = 柔和问候 + 亲昵语气')


if __name__ == '__main__':
    main()
