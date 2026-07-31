"""数据集质检：用本地 SenseVoice GGUF 模型识别每条音频，与文本对照。

流程：
  1. 读取 manifest.jsonl
  2. 每条音频重采样 24kHz→16kHz（SenseVoice 要求）
  3. 调用 voice_test.exe asr 识别
  4. 解析 ASR 输出（去掉 <|zh|><|NEUTRAL|>... 前缀）
  5. 与 normalized_text 对比，计算字符错误率 CER
  6. 输出质检报告

用法：
    python qa_asr.py                    # 全量质检
    python qa_asr.py --limit 50         # 只检前 50 条
    python qa_asr.py --threshold 0.1    # CER 阈值（默认 0.15）
"""
import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import re
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "ref3_tts_dataset_routed_500")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")
WAVS_DIR = os.path.join(DATASET_DIR, "wavs")

VOICE_TEST_EXE = r"E:\winefox\voice-test\build\voice-test\Release\voice_test.exe"
ASR_MODEL = r"E:\winefox\models\asr\sense-voice-small-q4_k.gguf"


def resample_to_16k(src_path, dst_path):
    """重采样到 16kHz，保存为 PCM 16-bit 格式（voice_test 要求）。"""
    import torchaudio
    wav, sr = torchaudio.load(src_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    # float32 → int16
    wav = (wav.clamp(-1, 1) * 32767).to(torch.int16)
    torchaudio.save(dst_path, wav, 16000)


def run_asr(wav_16k_path):
    """调用 voice_test asr 识别，返回纯文本（已去掉前缀 token 和 emoji）。"""
    result = subprocess.run(
        [VOICE_TEST_EXE, "asr", wav_16k_path,
         "--model", ASR_MODEL,
         "--lang", "auto", "--threads", "4", "--itn"],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60
    )
    output = result.stdout + result.stderr
    # 逐行搜索 "text       : <内容>"
    text = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("text") and ":" in stripped:
            text = stripped.split(":", 1)[1].strip()
            break
    if not text:
        return ""
    # 去掉 SenseVoice 前缀 token（如果用 --prefix 才会有，这里通常没有）
    text = re.sub(r'<\|[^|]+\|>', '', text).strip()
    # 去掉末尾 emoji（精确范围，避免误删中文）
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF'
                  r'\U0001FA70-\U0001FAFF]+$', '', text).strip()
    return text


def normalize_text(text):
    """归一化文本：去掉标点和空白，只保留中英文字符和数字。"""
    # 去掉所有标点、空白
    text = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9]', '', text)
    return text


def cer(reference, hypothesis):
    """计算字符错误率 CER = (S + D + I) / N。

    基于 Levenshtein 距离，逐字符比较。
    """
    ref = list(reference)
    hyp = list(hypothesis)
    n = len(ref)
    if n == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # Levenshtein 距离（替换=1, 删除=1, 插入=1）
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        curr = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cost = 0 if rc == hc else 1
            curr[j] = min(prev[j] + 1,      # 删除
                          curr[j-1] + 1,     # 插入
                          prev[j-1] + cost)  # 替换
        prev = curr
    distance = prev[-1]
    return distance / n


def main():
    parser = argparse.ArgumentParser(description="SenseVoice 数据集质检")
    parser.add_argument("--limit", type=int, default=None, help="限制条数")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="CER 阈值，超过则标记为不合格（默认 0.15）")
    parser.add_argument("--output", default=None,
                        help="报告输出路径（默认打印到终端）")
    args = parser.parse_args()

    # 读取 manifest
    records = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    total = len(records)
    print(f"数据集质检：{total} 条音频")
    print(f"ASR 模型：{ASR_MODEL}")
    print(f"CER 阈值：{args.threshold}")
    print(f"{'='*80}\n")

    # 创建临时目录存放 16kHz wav
    tmp_dir = tempfile.mkdtemp(prefix="qa_asr_")

    results = []
    pass_count = 0
    fail_count = 0
    cer_values = []
    t0 = time.time()

    for i, rec in enumerate(records):
        rec_id = rec["id"]
        audio_rel = rec["audio"]
        ref_text = rec.get("normalized_text", rec["text"])
        ref_norm = normalize_text(ref_text)

        src_wav = os.path.join(DATASET_DIR, audio_rel)
        tmp_wav = os.path.join(tmp_dir, f"{rec_id}.wav")

        try:
            # 重采样
            resample_to_16k(src_wav, tmp_wav)
            # ASR 识别
            asr_text = run_asr(tmp_wav)
            asr_norm = normalize_text(asr_text)
            # 计算 CER
            cer_val = cer(ref_norm, asr_norm)
            cer_values.append(cer_val)

            passed = cer_val <= args.threshold
            if passed:
                pass_count += 1
            else:
                fail_count += 1

            status = "PASS" if passed else "FAIL"
            results.append({
                "id": rec_id,
                "ref": ref_text,
                "asr": asr_text,
                "cer": round(cer_val, 4),
                "passed": passed,
            })

            # 进度输出
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            flag = "" if passed else " ← 不合格"
            print(f"[{i+1}/{total}] {rec_id} CER={cer_val:.3f} {status}{flag}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")
            if not passed:
                print(f"  参考: {ref_text}")
                print(f"  识别: {asr_text}")

        except Exception as e:
            fail_count += 1
            print(f"[{i+1}/{total}] {rec_id} ERROR: {e}")
            results.append({
                "id": rec_id,
                "ref": ref_text,
                "asr": "",
                "cer": 1.0,
                "passed": False,
                "error": str(e),
            })

        # 清理临时文件
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    os.rmdir(tmp_dir)

    # === 汇总报告 ===
    elapsed = time.time() - t0
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0
    max_cer = max(cer_values) if cer_values else 0
    min_cer = min(cer_values) if cer_values else 0
    pass_rate = pass_count / total * 100 if total > 0 else 0

    # CER 分布
    cer_0 = sum(1 for c in cer_values if c == 0)
    cer_5 = sum(1 for c in cer_values if 0 < c <= 0.05)
    cer_10 = sum(1 for c in cer_values if 0.05 < c <= 0.10)
    cer_15 = sum(1 for c in cer_values if 0.10 < c <= 0.15)
    cer_20 = sum(1 for c in cer_values if 0.15 < c <= 0.20)
    cer_30 = sum(1 for c in cer_values if 0.20 < c <= 0.30)
    cer_high = sum(1 for c in cer_values if c > 0.30)

    report = []
    report.append(f"\n{'='*80}")
    report.append(f"质检报告")
    report.append(f"{'='*80}")
    report.append(f"总条数:        {total}")
    report.append(f"合格:          {pass_count} ({pass_rate:.1f}%)")
    report.append(f"不合格:        {fail_count} ({100-pass_rate:.1f}%)")
    report.append(f"耗时:          {elapsed:.0f}s")
    report.append(f"")
    report.append(f"CER 统计:")
    report.append(f"  平均:        {avg_cer:.4f}")
    report.append(f"  最小:        {min_cer:.4f}")
    report.append(f"  最大:        {max_cer:.4f}")
    report.append(f"")
    report.append(f"CER 分布:")
    report.append(f"  =0.000  (完美):        {cer_0:4d} ({cer_0/total*100:.1f}%)")
    report.append(f"  0-0.05  (极好):        {cer_5:4d} ({cer_5/total*100:.1f}%)")
    report.append(f"  0.05-0.10 (良好):      {cer_10:4d} ({cer_10/total*100:.1f}%)")
    report.append(f"  0.10-0.15 (可接受):    {cer_15:4d} ({cer_15/total*100:.1f}%)")
    report.append(f"  0.15-0.20 (需关注):    {cer_20:4d} ({cer_20/total*100:.1f}%)")
    report.append(f"  0.20-0.30 (较差):      {cer_30:4d} ({cer_30/total*100:.1f}%)")
    report.append(f"  >0.30   (不合格):      {cer_high:4d} ({cer_high/total*100:.1f}%)")
    report.append(f"")

    # 列出不合格的案例
    failed = [r for r in results if not r.get("passed", False)]
    if failed:
        report.append(f"不合格案例（{len(failed)} 条）:")
        for r in failed[:20]:  # 最多列20条
            report.append(f"  {r['id']} CER={r['cer']:.3f}")
            report.append(f"    参考: {r['ref']}")
            report.append(f"    识别: {r['asr']}")
        if len(failed) > 20:
            report.append(f"  ... 还有 {len(failed)-20} 条")

    report_text = "\n".join(report)
    print(report_text)

    # 保存报告
    if args.output:
        report_path = args.output
    else:
        report_path = os.path.join(DATASET_DIR, "qa_asr_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n\n=== 详细结果 ===\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
