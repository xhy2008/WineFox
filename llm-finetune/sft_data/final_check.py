# -*- coding: utf-8 -*-
import json
import re

lines = open("../winefox_dataset.jsonl", encoding="utf-8").read().strip().split("\n")
pat = ["user", "tool", "assistant"]
n_extra = 0
for i, ln in enumerate(lines):
    d = json.loads(ln)
    roles = [m["role"] for m in d["messages"]]
    assert roles == pat * (len(roles) // 3), (i, roles)
    for m in d["messages"]:
        c = m["content"]
        assert c.strip(), i
        if m["role"] == "assistant":
            assert c.startswith(
                ("[joy]", "[neutral]", "[surprise]", "[sadness]", "[anger]", "[fear]", "[sexual]")
            ), (i, c[:20])
        if m["role"] == "tool":
            assert "[相关记忆]" in c and "[当前时间]" in c, i
            if "[额外信息]" in c:
                n_extra += 1
                assert re.search(r"\[额外信息\]天气：[^；]+；心情：[^\"\\]+", c), (i, c)
print("总条数", len(lines), "含额外信息条数", n_extra)
print("OK")
