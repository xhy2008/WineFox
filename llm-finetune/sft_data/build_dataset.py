# -*- coding: utf-8 -*-
"""
合并 sft_data/ 下的批次数据，生成标准 SFT 数据集 winefox_dataset.jsonl

格式: 每行一个 {"messages": [...]}
规则校验:
  - user 后必须紧跟 tool
  - tool 必须包含 [相关记忆] 和 [当前时间]
  - assistant 回复必须以情绪标签开头 ([joy]/[neutral]/[surprise]/[sadness]/[anger]/[fear])
"""
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
OUTPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "winefox_dataset.jsonl")

EMOTION_TAGS = ("[joy]", "[neutral]", "[surprise]", "[sadness]", "[anger]", "[fear]", "[sexual]")
BATCH_FILES = [
    "batch1_first_meet.json",
    "batch2_memory_multiturn.json",
    "batch3_daily.json",
    "batch4_philosophy.json",
    "batch5_intimate.json",
    "batch6_commonsense.json",
]


def main():
    all_dialogs = []
    errors = []

    for fname in BATCH_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            errors.append(f"缺少文件: {fname}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            errors.append(f"{fname}: 顶层必须是数组")
            continue

        for idx, dialog in enumerate(data):
            line_no = idx + 1
            if not isinstance(dialog, dict) or "messages" not in dialog:
                errors.append(f"{fname}:{line_no} 缺少 messages")
                continue
            msgs = dialog["messages"]
            if not isinstance(msgs, list) or len(msgs) == 0:
                errors.append(f"{fname}:{line_no} messages 为空")
                continue

            # 校验消息序列: user -> tool -> assistant 循环
            expect = "user"
            turn_count = 0
            for mi, m in enumerate(msgs):
                role = m.get("role")
                content = m.get("content", "")
                if role != expect:
                    errors.append(f"{fname}:{line_no} 第{mi+1}条消息应为 {expect}，实际 {role}")
                    break
                if expect == "user":
                    if not content.strip():
                        errors.append(f"{fname}:{line_no} user 内容为空")
                    expect = "tool"
                elif expect == "tool":
                    if "[相关记忆]" not in content or "[当前时间]" not in content:
                        errors.append(f"{fname}:{line_no} tool 缺少 [相关记忆] 或 [当前时间]")
                    expect = "assistant"
                elif expect == "assistant":
                    if not content.strip():
                        errors.append(f"{fname}:{line_no} assistant 内容为空")
                    elif not any(content.startswith(t) for t in EMOTION_TAGS):
                        errors.append(f"{fname}:{line_no} assistant 回复未以情绪标签开头: {content[:20]}")
                    expect = "user"
                    turn_count += 1
            if expect != "user":
                errors.append(f"{fname}:{line_no} 消息序列不完整，停留在 {expect}")
                continue
            all_dialogs.append(dialog)

    # 去重
    seen = set()
    unique = []
    for d in all_dialogs:
        key = json.dumps(d, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(d)

    if errors:
        print("=== 校验失败 ===")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for d in unique:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    single = sum(1 for d in unique if len(d["messages"]) == 3)
    multi = len(unique) - single
    print(f"校验通过，共 {len(unique)} 条对话")
    print(f"  单轮: {single} 条")
    print(f"  多轮: {multi} 条")
    print(f"已输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
