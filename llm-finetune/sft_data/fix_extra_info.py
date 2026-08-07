# -*- coding: utf-8 -*-
"""
将 batch3/4/5 中 tool 消息的 [额外信息] 从"场景描写"统一改为"天气+心情"。
程序可获取的信息只有：相关记忆、当前时间、真实天气、酒狐心情——不含场景。
"""
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = ["batch3_daily.json", "batch4_philosophy.json", "batch5_intimate.json"]

MOOD_BY_TAG = {
    "[joy]": ["开心", "美滋滋的", "高兴", "雀跃"],
    "[neutral]": ["平静", "安稳", "安安静静的"],
    "[surprise]": ["好奇", "又惊又喜", "惊讶"],
    "[sadness]": ["有点低落", "闷闷的", "心里酸酸的"],
    "[anger]": ["闹小脾气", "气鼓鼓的", "气呼呼的"],
    "[fear]": ["害羞", "紧张", "心怦怦跳"],
}

# 特判：user 消息 → 天气（assistant 回复中明确提到了天气，必须保持一致）
SPECIAL_WEATHER = {
    "我回来啦！": "小雨",
    "酒狐，今天的早餐是什么？": "晴",
    "现在这么热，你受得了吗？": "晴，炎热",
    "酒狐，来吃块西瓜，冰镇的。": "晴，闷热",
    "酒狐，过来看，外面下雪了！": "小雪",
    "酒狐，你好像一直不太喜欢下雨天？": "阴",
}

FALLBACK_WEATHER = ["晴", "多云", "晴", "阴", "晴"]


def main():
    total = 0
    for fname in FILES:
        path = os.path.join(BASE, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        changed = 0
        for idx, dialog in enumerate(data):
            msgs = dialog["messages"]
            tag = ""
            for m in msgs:
                if m["role"] == "assistant":
                    mobj = re.match(r"^(\[[^\]]+\])", m["content"])
                    tag = mobj.group(1) if mobj else ""
                    break
            moods = MOOD_BY_TAG.get(tag, ["平静"])
            mood = moods[idx % len(moods)]
            first_user = next(m["content"] for m in msgs if m["role"] == "user")
            weather = SPECIAL_WEATHER.get(first_user, FALLBACK_WEATHER[idx % len(FALLBACK_WEATHER)])

            for m in msgs:
                if m["role"] != "tool" or "[额外信息]" not in m["content"]:
                    continue
                new_content = re.sub(
                    r"\[额外信息\][^\n]*",
                    f"[额外信息]天气：{weather}；心情：{mood}",
                    m["content"],
                )
                if new_content != m["content"]:
                    m["content"] = new_content
                    changed += 1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{fname}: 改写 {changed} 处 [额外信息]")
        total += changed
    print(f"共改写 {total} 处")


if __name__ == "__main__":
    main()
