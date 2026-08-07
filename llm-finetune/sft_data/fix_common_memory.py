# -*- coding: utf-8 -*-
"""将 batch6（常识数据）的 [相关记忆] 内容清空为“无相关往事记忆”。
常识应固化进模型参数，不通过 tool 记忆注入。"""
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE, "batch6_commonsense.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

changed = 0
for dialog in data:
    for m in dialog["messages"]:
        if m["role"] != "tool":
            continue
        new_content = re.sub(
            r"\[相关记忆\][^\n]*",
            "[相关记忆]（无相关往事记忆）",
            m["content"],
        )
        if new_content != m["content"]:
            m["content"] = new_content
            changed += 1

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"batch6: 清空 {changed} 处 [相关记忆]")
