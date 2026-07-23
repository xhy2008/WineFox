#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 GPT-SoVITS 合成脚本（Windows CPU/GPU 通用）

功能：
1. 自动启动 GPT-SoVITS api_v2 服务（子进程管理）
2. 自动用 SenseVoice 转写 ref.wav 得到 PROMPT_TEXT（首次运行，之后读缓存）
3. 分批合成 corpus.txt，支持断点续合
4. 输出 dataset/wavs/*.wav + dataset/metadata.csv

用法：
    python synthesize_local.py --ref ref2.wav --corpus corpus.txt --batch 500

前置条件：
- 已用 conda 安装 GPT-SoVITS 依赖（见 local_setup_steps）
- 已下载 GPT-SoVITS 预训练模型到 GPT-SoVITS/SoVITS_weights/ 和 GPT_weights/
- pip install sherpa-onnx soundfile librosa requests tqdm（用于转写 + 合成调用）

CPU 模式下每条约 3~6 秒，6727 条约 10~12 小时。
可分多次运行（断点续合），中途 Ctrl+C 安全退出。
"""

import os
import sys
import re
import time
import json
import shutil
import subprocess
import argparse
import requests
from pathlib import Path

# ===== 默认配置 =====
HERE = Path(__file__).parent.resolve()
DEFAULT_GPTSOVITS_DIR = HERE / 'GPT-SoVITS'           # GPT-SoVITS 源码目录
DEFAULT_REF = HERE / 'ref2.wav'
DEFAULT_CORPUS = HERE / 'corpus.txt'
DEFAULT_OUT = HERE / 'dataset'
API_HOST = '127.0.0.1'
API_PORT = 9880
API_URL = f'http://{API_HOST}:{API_PORT}'
# 新版 GPT-SoVITS 权重由 GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml 指定
# 默认 custom 段使用 v2（gsv-v2final-pretrained），无需 -s 参数


def get_duration(path):
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def wait_api(timeout=180):
    """等待 API 就绪"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(API_URL + '/', timeout=2)
            if r.status_code == 200 or r.status_code == 404:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_api(gptsovits_dir):
    """启动 GPT-SoVITS api_v2 服务，返回子进程

    新版 GPT-SoVITS 不再支持 -s 参数，权重通过 tts_infer.yaml 配置：
    - 默认 custom 段用 v2（gsv-v2final-pretrained）
    - 已修改为 device=cpu, is_half=false（CPU-only 适配）
    """
    print(f'启动 GPT-SoVITS API（{gptsovits_dir}）...')
    cfg = gptsovits_dir / 'GPT_SoVITS' / 'configs' / 'tts_infer.yaml'
    if not cfg.exists():
        print(f'⚠ 未找到配置: {cfg}')
        return None
    # 简单校验权重存在
    pm = gptsovits_dir / 'GPT_SoVITS' / 'pretrained_models'
    if not pm.exists():
        print(f'⚠ 未找到预训练模型目录: {pm}')
        return None

    log_path = HERE / 'api.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    cmd = [
        sys.executable, 'api_v2.py',
        '-a', API_HOST, '-p', str(API_PORT),
    ]
    proc = subprocess.Popen(cmd, cwd=str(gptsovits_dir),
                            stdout=log_f, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    print(f'API 进程 PID={proc.pid}，日志: {log_path}')
    print('等待 API 就绪（首次加载模型约 1~3 分钟）...')
    if wait_api():
        print('API 已就绪')
        return proc
    else:
        print('API 启动失败，日志末尾:')
        log_f.flush()
        with open(log_path, encoding='utf-8') as f:
            print(f.read()[-3000:])
        proc.terminate()
        return None


def transcribe_ref(ref_path, gptsovits_dir):
    """用 SenseVoice 转写 ref.wav 得到 PROMPT_TEXT，带缓存"""
    cache_path = HERE / f'prompt_text_{Path(ref_path).stem}.txt'
    if cache_path.exists():
        text = cache_path.read_text(encoding='utf-8').strip()
        print(f'复用缓存 PROMPT_TEXT（{cache_path.name}）: {text}')
        return text

    print(f'首次转写 {ref_path}（需要下载 SenseVoice 模型，约 230MB）...')
    try:
        import sherpa_onnx
        import soundfile as sf
        import librosa
    except ImportError:
        print('安装 sherpa-onnx 用于转写...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                               'sherpa-onnx', 'soundfile', 'librosa'])
        import sherpa_onnx
        import soundfile as sf
        import librosa

    sense_dir = HERE / 'sense-voice'
    if not sense_dir.exists():
        print('下载 SenseVoice 模型...')
        url = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2'
        archive = HERE / 'sense-voice.tar.bz2'
        # 用系统 aria2c 或 curl/wget 下载
        dl_ok = False
        for tool, args in [('aria2c', ['-x', '16', '-s', '16', '-o', archive.name, url]),
                           ('curl', ['-L', '-o', archive.name, url]),
                           ('wget', ['-O', archive.name, url])]:
            try:
                subprocess.check_call([tool] + args, cwd=str(HERE))
                dl_ok = True
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        if not dl_ok:
            print('下载失败，请手动下载 SenseVoice 模型并解压到', sense_dir)
            return None
        import tarfile
        with tarfile.open(archive, 'r:bz2') as t:
            t.extractall(HERE)
        extracted = HERE / 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17'
        extracted.rename(sense_dir)
        archive.unlink()

    asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(sense_dir / 'model.int8.onnx'),
        tokens=str(sense_dir / 'tokens.txt'),
        num_threads=2, use_itn=True, language='auto')
    audio, sr = sf.read(str(ref_path), dtype='float32')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000); sr = 16000
    s = asr.create_stream()
    s.accept_waveform(sr, audio)
    asr.decode_stream(s)
    text = s.result.text.strip()
    cache_path.write_text(text, encoding='utf-8')
    print(f'转写完成，PROMPT_TEXT 已缓存到 {cache_path.name}: {text}')
    return text


def load_done(metadata_path):
    """加载已完成的条目 ID"""
    done = set()
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split('|')
            if parts:
                done.add(parts[0])
    return done


def synth_one(text, ref_path, prompt_text, prompt_lang='ja', text_lang='zh',
              temperature=0.2, top_p=0.6, top_k=20,
              speed_factor=1.0, repetition_penalty=1.35,
              timeout=120, max_retry=3):
    """调用 API 合成一条，返回 bytes 或 None

    v2 模型有一定概率出现「半句后静音」：wav 时长正常，但后半段样本全是 0。
    根因：t2s 在某些 token 上崩溃，vits 解码器收到错误语义 token 后输出零幅度。
    修复策略：
    1. 降 temperature 到 0.2 / top_p 0.6 / top_k 20（减少随机性）
    2. 检测后半段 RMS 能量，若 < 前 1/4 的 5% 则判定为「半句静音」
    3. 失败自动重试 max_retry 次，每次换 seed 绕过随机崩溃
    """
    ref_abs = str(Path(ref_path).resolve()).replace('\\', '/')

    import random
    for attempt in range(max_retry):
        payload = {
            'text': text, 'text_lang': text_lang,
            'ref_audio_path': ref_abs,
            'prompt_text': prompt_text, 'prompt_lang': prompt_lang,
            'media_type': 'wav', 'streaming_mode': False,
            'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
            'speed_factor': speed_factor,
            'repetition_penalty': repetition_penalty,
            'text_split_method': 'cut0',
            'batch_size': 1,
            'seed': random.randint(0, 2**31 - 1),
        }
        try:
            r = requests.post(API_URL + '/tts', json=payload, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 1000:
                # 检测后半段静音：对比前 1/4 与后 3/4 的 RMS 能量
                if _detect_tail_silence(r.content):
                    print(f'    [{text[:20]}] 后半段静音, 重试 {attempt+1}/{max_retry}')
                    continue
                return r.content
            print(f'    [{text[:20]}] HTTP {r.status_code}: {r.text[:200]}')
        except Exception as e:
            print(f'    [{text[:20]}] 请求异常: {e}')
    return None


def _detect_tail_silence(wav_bytes):
    """检测 wav 是否后半段静音。

    判定标准：将音频分成 4 段，计算每段 RMS 能量。
    若最后 1 段 RMS < 第 1 段 RMS 的 5%，且绝对值 < 0.01，则判定为后半段静音。
    """
    import wave, io
    import numpy as np
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
            n = w.getnframes()
            sr = w.getframerate()
            sampwidth = w.getsampwidth()
            raw = w.readframes(n)
        # 转 numpy float32
        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return False  # 未知格式不检测
        if len(audio) < sr * 0.5:
            return False  # 太短不检测
        # 分 4 段计算 RMS
        chunks = np.array_split(audio, 4)
        rms = [np.sqrt(np.mean(c**2)) for c in chunks]
        # 后半段（最后 2 段）平均能量
        tail_rms = (rms[2] + rms[3]) / 2
        head_rms = (rms[0] + rms[1]) / 2
        # 后半段静音：能量远小于前半段 且 绝对值很小
        if head_rms > 0.01 and tail_rms < head_rms * 0.05 and tail_rms < 0.01:
            return True
        return False
    except Exception:
        return False  # 解析失败不阻塞合成


def normalize_wav(wav_path, sample_rate=22050):
    """用 ffmpeg 归一化到指定采样率/单声道/int16"""
    tmp = str(wav_path) + '.tmp.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(wav_path), '-ar', str(sample_rate),
         '-ac', '1', '-c:a', 'pcm_s16le', tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.move(tmp, str(wav_path))


def main():
    ap = argparse.ArgumentParser(description='本地 GPT-SoVITS 合成')
    ap.add_argument('--gptsovits-dir', type=Path, default=DEFAULT_GPTSOVITS_DIR,
                    help='GPT-SoVITS 源码目录')
    ap.add_argument('--ref', type=Path, default=DEFAULT_REF, help='参考音频')
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS, help='语料文件')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help='输出目录')
    ap.add_argument('--batch', type=int, default=500, help='本次合成的条数（CPU 建议 500/次）')
    ap.add_argument('--temperature', type=float, default=0.4)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--no-start-api', action='store_true', help='不自动启动 API（手动已启动时用）')
    args = ap.parse_args()

    # 检查输入
    if not args.ref.exists():
        print(f'参考音频不存在: {args.ref}'); sys.exit(1)
    if not args.corpus.exists():
        print(f'语料不存在: {args.corpus}'); sys.exit(1)
    if not args.gptsovits_dir.exists():
        print(f'GPT-SoVITS 目录不存在: {args.gptsovits_dir}')
        print('请先 clone GPT-SoVITS 到该目录，或用 --gptsovits-dir 指定路径')
        sys.exit(1)

    wavs_dir = args.out / 'wavs'
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / 'metadata.csv'

    # 转写 ref
    prompt_text = transcribe_ref(args.ref, args.gptsovits_dir)
    if not prompt_text:
        print('PROMPT_TEXT 获取失败'); sys.exit(1)

    # 启动 API
    proc = None
    if not args.no_start_api:
        proc = start_api(args.gptsovits_dir)
        if proc is None:
            print('API 启动失败'); sys.exit(1)
    else:
        if not wait_api(timeout=5):
            print('API 未运行，请先启动 api_v2.py 或去掉 --no-start-api'); sys.exit(1)
        print('API 已在运行')

    # 加载语料 + 已完成
    lines = [l.strip() for l in args.corpus.read_text(encoding='utf-8').splitlines() if l.strip()]
    done = load_done(metadata_path)
    todo = [(i, l) for i, l in enumerate(lines) if f'{i:06d}' not in done]
    todo = todo[:args.batch]
    print(f'\n语料总计: {len(lines)} 条')
    print(f'已完成: {len(done)} 条')
    print(f'本次将合成: {len(todo)} 条')
    if not todo:
        print('全部合成完成！可将 dataset/ 上传到 Google Drive。'); return

    # 合成
    try:
        from tqdm import tqdm
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tqdm'])
        from tqdm import tqdm

    ok, fail, skipped = 0, 0, 0
    meta_f = open(metadata_path, 'a', encoding='utf-8')
    t0 = time.time()
    try:
        for idx, text in tqdm(todo, desc='合成'):
            wid = f'{idx:06d}'
            wav_path = wavs_dir / f'{wid}.wav'
            # 断点续合：已存在且非空则跳过
            if wav_path.exists() and wav_path.stat().st_size > 1000:
                meta_f.write(f'{wid}|{text}|{text}\n'); meta_f.flush()
                skipped += 1; continue
            data = synth_one(text, args.ref, prompt_text,
                             temperature=args.temperature, top_p=args.top_p)
            if data:
                wav_path.write_bytes(data)
                normalize_wav(wav_path)
                meta_f.write(f'{wid}|{text}|{text}\n'); meta_f.flush()
                ok += 1
            else:
                fail += 1
                time.sleep(1)
            # 进度预估
            done_count = len(done) + ok + skipped
            if done_count % 20 == 0 and done_count > 0:
                elapsed = time.time() - t0
                rate = (ok + skipped) / max(1, elapsed)
                remain = len(lines) - done_count
                eta_min = remain / max(0.01, rate) / 60
                tqdm.write(f'  进度 {done_count}/{len(lines)} | {rate:.2f}条/秒 | 剩余约 {eta_min:.0f} 分钟')
    except KeyboardInterrupt:
        print('\n用户中断，已保存的进度不会丢失，下次运行自动续合')
    finally:
        meta_f.close()

    total_done = len(done) + ok + skipped
    print(f'\n本次完成: 成功 {ok}, 跳过 {skipped}, 失败 {fail}')
    print(f'累计完成: {total_done} / {len(lines)} 条')
    if total_done < len(lines):
        print(f'剩余 {len(lines)-total_done} 条，可再次运行本脚本继续（断点续合）')
    else:
        print('全部合成完成！')
        print(f'\n下一步: 将 {args.out} 目录上传到 Google Drive 的 winefox_tts/dataset/')

    # 关闭 API
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: proc.kill()


if __name__ == '__main__':
    main()
