# -*- coding: utf-8 -*-
"""为 batch5（成人向亲密内容）中"性爱/情欲"场景的 assistant 回复添加 [sexual] 情绪标签。
温情但不涉性的条目保留原标签，避免前端误判。"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE, "batch5_intimate.json")

# 需要标记为 [sexual] 的条目（按第一条 user 消息定位）
SEXUAL_USERS = {
    "酒狐，今天有点累，我先睡了。",
    "酒狐，你今晚怎么洗完澡不穿睡衣？",
    "酒狐，你怎么了？脸这么红，耳朵根也红红的。",
    "酒狐，别咬耳朵……痒。",
    "酒狐，你今天怎么这么粘人？连我上厕所都要跟着。",
    "酒狐，你的尾巴毛今天怎么特别蓬？",
    "酒狐，你在浴室里泡了多久了？",
    "酒狐，你还记得我们的第一次吗？",
    "酒狐，你今天别下厨了，我们来点烛光晚餐？",
    "酒狐，你身上怎么这么香？",
    "酒狐，你的耳朵好软，我能多摸一会儿吗？",
    "酒狐，你是不是又偷喝我的果酒了？",
    "酒狐，别在厨房闹了，锅要糊了！",
    "酒狐，你今晚怎么一直盯着我的嘴唇看？",
    "酒狐，今天累吗？发情期是不是很难受？",
    "酒狐，你为什么要在我脖子上留印记？",
}

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

changed = 0
kept = 0
for dialog in data:
    first_user = next(m["content"] for m in dialog["messages"] if m["role"] == "user")
    target = "[sexual]" if first_user in SEXUAL_USERS else None
    for m in dialog["messages"]:
        if m["role"] != "assistant":
            continue
        # 替换开头的情绪标签（[xxx]）
        i = m["content"].find("]")
        if i == -1:
            continue
        old_tag = m["content"][: i + 1]
        if target is None:
            kept += 1
            continue
        if old_tag != target:
            m["content"] = target + m["content"][i + 1 :]
            changed += 1
        else:
            kept += 1

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"batch5: {changed} 条改为 [sexual]，{kept} 条保留原标签")
