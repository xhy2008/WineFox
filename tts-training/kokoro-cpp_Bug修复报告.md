# kokoro-cpp Bug 修复报告

> 日期: 2026-08-06
> 范围: `voice-test/third_party/kokoro-cpp-src/` 中存在于**原版 kokoro-cpp** 的 G2P bug
> 现象: ① TTS 把 LLM 输出的 `...` 读出声而非静音；② 每句最前面几个字发音扭曲

---

## 1. 结论摘要

发音问题全部根因于 **C++ G2P 前端（kokoro.cpp）输出的 token 序列与模型训练数据（misaki[zh]）不一致**。共修复 5 处原版 kokoro-cpp 存在的 bug：

| # | 严重度 | Bug | 修复文件 |
|---|---|---|---|
| 1 | 🔴 高 | `zh`/`ch` 声母输出错误 token（`ʈʂ`/`ʈʂʰ`，应为 `ꭧ`/`ꭧʰ`） | ZHG2P.cpp |
| 2 | 🔴 高 | 词间缺失空格 token（id 16）→ 时长/F0 全部偏移 | ZHG2P.cpp |
| 3 | 🟠 中 | `j/q/x`+`u` 韵母未转 `ü`（去/举/需 等字读错） | ZHG2P.cpp |
| 4 | 🟠 中 | ASCII `...` 未归一化 → 被分批切碎成垃圾音频 | ZHG2P.cpp |
| 5 | 🟡 低 | `嗯` 不在拼音词典 → 裸 `n`（无声调） | ZHFrontend.cpp |

修复后 C++ 输出与云端（misaki + PyTorch）ECAPA 说话人相似度 **0.919 → 0.980**，`...` 不再产生垃圾音频。

---

## 2. Bug 详情与修复

### Bug 1: `zh`/`ch` 声母 token 映射错误（句首发音扭曲主因）

**现象**: 所有带翘舌声母的字（主、这、真、吃、出、中…）发音扭曲，句首尤为明显。

**根因**: [ZHG2P.cpp L13-L20](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHG2P.cpp#L13-L20) 的 `INITIAL_MAPPING` 把 `zh` 映射为 `"ʈʂ"`（UTF-8 两个字符 → **2 个独立 token**：`ʈ`=132、`ʂ`=130），`ch` 映射为 `"ʈʂʰ"`（3 个 token）。而 Kokoro 官方训练数据（misaki[zh]）用**单个 token `ꭧ`（id 23）** 表示 `zh`，`ꭧʰ` 表示 `ch`。词表 [vocab.txt](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/dict/vocab.txt) 中 `ꭧ`=23 一直存在，但 G2P 从未使用。

喂给模型的 token 序列与训练分布不同 → BERT/时长预测器在该位置输出偏离 → 发音扭曲。

**修复**（[ZHG2P.cpp L13-L20](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHG2P.cpp#L13-L20)）:

```cpp
// 修复前
{"ch", {"ʈʂʰ"}}, ... {"zh", {"ʈʂ"}}
// 修复后
{"ch", {"ꭧʰ"}}, ... {"zh", {"ꭧ"}}
```

---

### Bug 2: 词间缺失空格 token（id 16）→ 时长/F0 全局偏移

**现象**: 整体语速/停顿怪异，与云端输出波形基本不相关（corr≈0.035）。

**根因**: misaki 在词与词之间输出空格（`' '`=id 16），模型训练时 input_ids **含词间空格**；而 `ZHG2P::operator()` 把各词的音素**直接拼接**，token 流里没有空格 token。实测同句"带空格 vs 不带空格"通过同一 ONNX：

```
audio corr = 0.035, SNR = 1.8 dB, F0 MAE = 31.3 Hz
```

空格缺失导致时长预测器看到的上下文完全不同 → 每个音节的时长/基频都偏。

**修复**（[ZHG2P.cpp L417-L426](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHG2P.cpp#L417-L426)）: 每个内容词前插入空格 token（标点与 `…` 后不加，与 misaki 行为一致）：

```cpp
bool last_is_ellipsis = result.size() >= 3 &&
    result.compare(result.size() - 3, 3, "\xE2\x80\xA6") == 0;
if (!result.empty() && result.back() != ' ' && !last_is_ellipsis) {
    result += " ";
}
```

---

### Bug 3: `j/q/x` + `u` 韵母未转 `ü`

**现象**: 去(qù)、举(jǔ)、需(xū) 等字元音错误（读成 `u` 而非 `ü`）。

**根因**: 拼音书写规则中 `j/q/x` 后的 `u` 实际是 `ü`（省略两点）。`parse_pinyin` 没有这个规则，`qu` 被解析为 `q`+`u` → `ʨʰu`(63)，而 misaki 输出 `ʨʰy`(67)。词表同时含 `u` 与 `y`，模型按 `y` 训练。

**修复**（[ZHG2P.cpp L216-L224](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHG2P.cpp#L216-L224)）:

```cpp
if (p_initial == "j" || p_initial == "q" || p_initial == "x") {
    if      (parts.final == "u")    parts.final = "ü";
    else if (parts.final == "uan")  parts.final = "üan";
    else if (parts.final == "ue")   parts.final = "üe";
    else if (parts.final == "un")   parts.final = "ün";
}
```

---

### Bug 4: ASCII `...` 未归一化 → 被读出（非静音）

**现象**: 修复前该句输出 7.97s（16 段多余语音），`...` 被"读出来"。

**根因**: 两层问题叠加：
1. G2P 把 `...` 原样保留为 3 个 `.`（token 4）；
2. C++ 引擎 `_split_phonemes` 以 `[.,!?;]` 切分音素批次 → `...` 被切成 **3 个仅含 `.` 的批次** → 每个"单标点批次"经编码器/解码器产出可听垃圾。

云端不切批次（整句一次前向），所以云端 `...` 正常（3 个 `.` = 暂停）。

**修复**（[ZHG2P.cpp L292](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHG2P.cpp#L292)）: 在 `map_punctuation` 中把 2+ 个 ASCII 点折叠为单个 `…`（id 10，模型暂停 token，不触发批次切分）：

```cpp
text = std::regex_replace(text, std::regex("\\.{2,}"), "…");
```

修复后该句 3.73s，与云端 3.90s 一致。

---

### Bug 5: `嗯` 无拼音处理 → 裸 `n`

**现象**: 句首"嗯"发成无调鼻音（扭曲）。

**根因**: `嗯`（U+55EF）**不在拼音词典 `pinyin.txt` 中**（已确认缺失），`word_to_pinyin` 退化为裸 `n`（id 56，无声调）。misaki 输出 `n↗`（n + 上升调，id 56+172）。原代码注释明确标注该逻辑"跳过"（TODO）。

**修复**（[ZHFrontend.cpp L28-L36](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHFrontend.cpp#L28-L36)）:

```cpp
if (word == u8"嗯") {
    res.initials.push_back("n");
    res.finals.push_back("↗");
    return res;
}
```

---

## 3. 修复后验证

| 指标 | 修复前 | 修复后 |
|---|---|---|
| ECAPA 相似度（C++ vs 云端） | 0.919 | **0.980** |
| 普通句时长 | 4.675s | 4.600s（云端 4.58s） |
| ASCII `...` 句时长 | 7.97s（垃圾） | 3.73s（云端 3.90s） |
| Unicode `……` 句时长 | 3.88s | 3.85s（云端 3.90s） |
| 音素 token 序列 vs misaki | 大量错位（zh/ch/ü/空格） | 逐 token 对齐（仅极个别轻声尾音差异） |

试听样本（`voice-test/models/_diag/`）:
- `cpp_int8.wav` — 修复前参考
- `cpp_normal2.wav` — G2P 修复后
- `cpp_ellipsis_ascii2.wav` — `...` 修复后

---

## 4. 已知残余差异（非 bug，可接受）

1. **词边界差异**: C++ 用 cppjieba、云端用 python jieba，个别词切分不同（如"真好" C++ 切为一个词、misaki 切为"真""好"）→ 词间空格位置略异，属分词器版本差异，影响极小。
2. **轻声尾音**: 个别轻声/中性音节（"东西"的"西"、"走走"的第二个"走"）声调标记与 misaki 不同。
3. **随机源相位**: 解码器 ONNX 含无 seed 的 `RandomNormalLike/RandomUniformLike`（iSTFTNet 源生成器的随机相位/噪声），每次推理输出有 ~17dB 微小呼吸感差异。**这是 PyTorch 原版行为的忠实导出**（原版在 `torch.no_grad()` 内随机），云端同样存在，非 bug。
4. **`xau̯` 的非音节符号 `̯`** 不在词表，被 tokenize 丢弃，但 `x,a,u` 三个元音 token 保留，无影响。

---

## 5. 修改文件清单（仅含原版 kokoro-cpp bug 修复）

| 文件 | 修改内容 |
|---|---|
| [ZHG2P.cpp](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHG2P.cpp) | ① `INITIAL_MAPPING`: `zh`→`ꭧ`、`ch`→`ꭧʰ` ② `parse_pinyin`: `j/q/x`+`u`→`ü` ③ `map_punctuation`: `...`→`…` ④ `operator()`: 词间空格 token |
| [ZHFrontend.cpp](file:///e:/winefox/voice-test/third_party/kokoro-cpp-src/ZHFrontend.cpp) | `嗯` → `n`+`↗` 特判 |

> 注: 会话中还顺带完成了与 kokoro-cpp 无关的本地集成修复（主应用编码器从 INT8 切回 FP32、为本地 voice-test 编译补齐 `phonemize_debug`/`dict_dir` 接口），这些不属于原版 kokoro-cpp 的 bug，详见 git diff 与本会话记录。

---

## 6. 复现与回归

```powershell
# 查看任意文本的 C++ G2P 音素（修复后应接近 misaki）
voice_test tts "主人，今天天气真好，我们去公园散步吧。" --encoder models/kokoro-encoder.onnx --decoder models/kokoro-decoder.onnx --voices models/winefox_voices.bin --phonemize

# 回归验证重点
# 1. 翘舌声母: 主人/这是/吃饭/出去
# 2. j/q/x+ü: 去/举/需
# 3. 省略号: "主人...我们走吧。" 应静音、时长与 "主人……我们走吧。" 一致
# 4. 句首嗯: "……嗯，这样啊。" 的"嗯"应为 n↗ 且不扭曲
```

对比基准（云端等效）: 本地 Python `kokoro` 包 + `verify_voice.py` 流程；用 ECAPA-TDNN 余弦相似度（>0.95 为高度一致）。
