# WineFox AI 情感陪伴程序 - 项目实施计划

> 基于「酒狐」IP 的本地化、跨平台、纯 CPU 推理的 AI 情感陪伴应用

---

## 0. 方案评审与关键改进建议

在正式制定计划之前，先对原始方案中存在的若干问题进行说明，并在后续章节给出修正方案。

### 0.1 模型选型

- **基座**：`Qwen3.5-0.8B`（首选）+ 自训酒狐 LoRA；当 0.8B 在某些场景下能力不足时，备选升级到 `Qwen3.5-2B`。
- 必须使用 Instruct 版本作为 LoRA 基座，Base 版本未对齐会导致 LoRA 失效。当前 `models/Qwen3.5-0.8B.gguf` 需核实是否为 Instruct 版本。
- 量化：默认 Q4_0，提供 Q4_K_M / Q5_K_M 选项。

### 0.2 性能目标的现实性

「十年前的 CPU + 8 token/s + 2GB 内存」的目标需要分级对待：

| CPU 档位 | 0.8B Q4_0 | 2B Q4_0 |
|----------|-----------|---------|
| i5-4590 (2014, DDR3) | 6-9 tok/s | 2-4 tok/s |
| i5-8400 (2017, DDR4) | 14-20 tok/s | 6-9 tok/s |
| 骁龙 855 (2019) | 10-15 tok/s | 4-6 tok/s |
| 骁龙 660 (2017, arm64) | 4-7 tok/s | 1.5-2.5 tok/s |
| 树莓派 4 (arm64, 2019) | 3-5 tok/s | 不可用 |
| 32位 arm 单板机 (arm32) | 1-3 tok/s | 不可用 |

**改进**：
- 默认 0.8B 基座，达到 ≥ 8 tok/s 的目标在 i5-4590 及以上 CPU 可满足。
- 老旧 arm32 Linux 设备（如树莓派 2/3）作为「极限兼容」目标，仅保证可运行（≥ 1 tok/s），不强求 8 tok/s。
- 按设备分级配置 preset，自动选择模型与量化等级。

### 0.3 首字延迟目标

端到端首字延迟典型构成：

```
VAD 端点检测尾延迟:  300-500 ms
ASR 推理:           100-300 ms
LLM prefill:        300-1000 ms（取决于 prompt 长度）
TTS 首音块:         150-400 ms
─────────────────────────────────
合计:               850-2200 ms
```

**改进**：将「首字延迟 <1s」调整为「**P50 ≤ 1.5s，P95 ≤ 2.5s**」，并通过以下手段压低：
- VAD 端点延迟动态调整（语义感知）
- LLM prefill 异步启动（ASR 流式输出 token 时即开始 prefill prompt 部分）
- TTS 首块流式输出，不等整句

### 0.4 平台架构支持范围

- **Android**：仅支持 arm64-v8a。2026 年 arm32 安卓设备性能极差，且 Android 14 起官方已逐步弃用 32 位 ABI，sherp-onnx、llama.cpp 在 arm32 上构建维护成本高，故放弃 Android arm32。
- **Linux**：保留 arm32 支持。树莓派 2/3 等 arm32 单板机仍是常见的低成本家庭服务器平台，Linux arm32 作为「极限兼容」目标，仅保证 CLI 模式可运行（不强求性能达标）。
- 最低 API level：Android 8.0 (API 26)。

### 0.5 TTS 方案细化

- **数据集生成**：使用 CosyVoice 跨语言克隆酒狐日语音色 → 生成中文平行语料（一次性，离线 GPU 或 Colab 完成）。
- **蒸馏训练**：使用 **VITS-Tiny** 作为学生模型，对 CosyVoice 生成的酒狐音色音频做蒸馏学习，得到轻量的酒狐音色 TTS 模型。
- **在线推理**：将蒸馏后的 VITS-Tiny 导出为 ONNX，交 sherpa-onnx 推理。
- **不使用 GPT-SoVITS**：其 autoregressive 部分对 CPU 实时性不友好，仅在前期实验阶段曾用作对比基线，正式方案中移除。
- **VITS-Tiny 优势**：参数量小、CPU 推理快、易导出 ONNX、sherpa-onnx 原生支持，符合极致轻量化目标。

### 0.6 Live2D 替代方案与渲染层统一

- 用户已有 `WineFoxModel/模型文件/人物模型/*.bbmodel`（BlockBench 基岩版模型 + 动画）。
- **改进**：自研轻量 `bbmodel` 渲染器（骨骼动画 + 表情切换），不引入 Live2D / Spine / DragonBones。
  - **渲染层统一为 GLES3**：Windows 和 Android 都用 OpenGL ES 3.0，避免为不同平台维护两套渲染接口（D3D11 + GLES3）。Windows 上通过原生 GLES3 驱动（绝大多数现代 GPU 均支持）或 ANGLE 兜底。
  - Linux 无 GUI，不涉及渲染。
  - 动作集：从 bbmodel 自带动画提取，按 emotion 标签映射。
  - 优势：单一渲染代码路径，降低开发与维护复杂度；D3D11 的开发与调试成本过高，对本项目收益有限。

### 0.7 AEC 简化策略

- WebRTC AEC3 调参复杂，跨平台行为差异大。
- **改进**：
  - 默认检测音频输出路由，若为耳机/外部扬声器 → 跳过 AEC（节省 CPU）。
  - 仅在内置扬声器 + 麦克风场景启用 AEC3。
  - 提供「关闭 AEC」开关，供高级用户调试。

### 0.8 内存预算重新分配

当前模型组合在 2GB 内难以容纳，需分级：

| 模型组件 | 常驻 (MB) | 按需加载 (MB) |
|----------|-----------|---------------|
| LLM Qwen3.5-0.8B Q4_0 | 500 | - |
| LLM KV cache (n_ctx=4096, Q4) | 130 | - |
| LoRA 酒狐 (f16) | 30 | - |
| Embedding Qwen3-Embedding-0.6B Q8 | 600 | - |
| ASR SenseVoice-Small INT8 | 450 | - |
| TTS VITS-Tiny (蒸馏后, INT8) | 80 | - |
| VAD TEN-VAD | 20 | - |
| AEC3 状态 | 10 | - |
| 主程序 + SQLite + UI | 200 | - |
| **合计常驻（默认档）** | **~1430 MB** | - |
| **合计常驻（高端档，2B 基座）** | **~2050 MB** | - |

**改进**：
- 默认档：embedding 改用 `bge-small-zh-v1.5` Q8（~120MB），TTS 用 VITS-Tiny INT8（~80MB），LLM 用 0.8B，总量压到 ~1.4GB。
- 高端档：保留 Qwen3-Embedding-0.6B Q8，LLM 升级到 2B，总量约 2GB。
- LLM 与 Embedding 模型共享 ggml 后端内存池，避免重复分配。

---

## 1. 项目目标

### 1.1 核心目标

- 本地、离线、纯 CPU 推理的酒狐 AI 陪伴应用
- 多平台：Windows (x86/x64/arm64)、Android (arm64)、Linux (x86/x64/arm64/arm32, 无 GUI)
- 实时语音对话 + 情感驱动的酒狐形象动画
- 短期 + 长期记忆系统，能记住用户的关键信息
- 极致轻量化：内存 ≤ 2GB，支持十年前主流 CPU

### 1.2 非目标（明确排除）

- 云端推理 / 联网功能（除模型下载外）
- 多角色 / 多人设切换（仅酒狐）
- 商业化 / 付费功能
- 直播推流 / VTuber 集成

### 1.3 质量指标

| 指标 | 目标值 | 最低可接受 |
|------|--------|------------|
| LLM 生成速度 (Qwen3.5-0.8B, i5-4590) | ≥ 8 tok/s | ≥ 5 tok/s |
| LLM 生成速度 (Qwen3.5-2B, i5-8400) | ≥ 8 tok/s | ≥ 6 tok/s |
| LLM 生成速度 (Qwen3.5-0.8B, arm32 树莓派3) | ≥ 1 tok/s | ≥ 0.5 tok/s |
| 端到端首字延迟 P50 | ≤ 1.5s | ≤ 2.5s |
| 端到端首字延迟 P95 | ≤ 2.5s | ≤ 4s |
| 内存占用（默认档，0.8B） | ≤ 1.5GB | ≤ 2GB |
| 内存占用（高端档，2B） | ≤ 2GB | ≤ 2.2GB |
| 安装包体积 | ≤ 800MB | ≤ 1.2GB |
| 启动到可对话 | ≤ 5s | ≤ 10s |

---

## 2. 系统架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│  UI 层 (平台特定)                                        │
│  - Windows: Win32 + GLES3.0 + WASAPI                   │
│  - Android: NativeActivity + GLES3.0 + AAudio          │
│  - Linux: CLI / TUI (无图形界面)                         │
├─────────────────────────────────────────────────────────┤
│  应用层 (C++ 跨平台)                                     │
│  - ConversationManager (对话状态机)                     │
│  - EmotionDriver (情感标签 → 动画映射)                  │
│  - MemoryOrchestrator (短期/长期记忆协调)               │
│  - PipelineScheduler (流式管线调度)                     │
├─────────────────────────────────────────────────────────┤
│  核心服务层 (C++ 跨平台)                                 │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ LLM      │ ASR      │ TTS      │ VAD+AEC  │         │
│  │ Service  │ Service  │ Service  │ Service  │         │
│  ├──────────┼──────────┼──────────┼──────────┤         │
│  │ Memory   │ SQLite   │ Embedder │ Model    │         │
│  │ Service  │ Storage  │ Service  │ Loader   │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
├─────────────────────────────────────────────────────────┤
│  后端层                                                  │
│  - llama.cpp (LLM + Embedding)                          │
│  - sherpa-onnx (ASR + TTS + VAD)                        │
│  - webrtc-audio-processing (AEC3)                       │
│  - SQLite3 (存储)                                       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
winefox/
├── CMakeLists.txt
├── PLAN.md                       # 本文件
├── src/
│   ├── core/                     # 跨平台核心
│   │   ├── llm/                  # LLM 服务（封装 llama.cpp）
│   │   ├── asr/                  # ASR 服务（封装 sherpa-onnx）
│   │   ├── tts/                  # TTS 服务
│   │   ├── vad/                  # VAD 服务
│   │   ├── aec/                  # AEC 服务
│   │   ├── memory/               # 记忆服务（短期 + 长期）
│   │   ├── embedder/             # 向量嵌入
│   │   ├── pipeline/             # 流式管线
│   │   ├── emotion/              # 情感解析与映射
│   │   └── storage/              # SQLite 封装
│   ├── platform/                 # 平台特定
│   │   ├── windows/
│   │   ├── android/
│   │   └── linux/
│   ├── ui/                       # 渲染层（Windows / Android）
│   │   ├── renderer/             # D3D11 / GLES3
│   │   ├── bbmodel/              # BlockBench 模型加载与动画
│   │   └── input/                # 交互
│   └── app/                      # 应用入口
├── third_party/
│   ├── llama.cpp/                # git submodule
│   ├── sherpa-onnx/              # git submodule
│   ├── webrtc-apm/               # AEC3 提取模块
│   ├── sqlite/                   # sqlite3.c + sqlite3.h
│   └── json/                     # nlohmann/json
├── models/                       # 已有
├── llm-finetune/                 # 已有
├── tts-training/                 # 已有
├── WineFoxModel/                 # 已有
└── tests/
```

### 2.3 关键数据流

#### 对话流（实时）

```
Mic ─► AEC ─► VAD ─► ASR (stream) ─► text
                                          │
                                          ▼
                            ┌──────────────────────────┐
                            │ ConversationManager      │
                            │ - 拼接短期上下文          │
                            │ - 检索长期记忆注入 system │
                            └──────────────────────────┘
                                          │
                                          ▼
                            LLM (stream) ─► [emotion] + text
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    EmotionDriver    SentenceSplitter   (并行)
                    → 动画切换         │
                                      ▼
                                  TTS (stream chunk)
                                      │
                                      ▼
                                  Speaker ◄── 远端参考 ─► AEC
```

#### 记忆整理流（空闲）

```
[触发条件]: 用户 N 分钟无输入

1. 从缓存中加载未整理的对话（程序在实时对话时会将对话文本缓存起来等待整理），含 user + assistant + emotion 标签
2. 卸载酒狐 LoRA（llama.cpp lora unload）
3. 构造摘要 prompt（输入为完整对话，含酒狐的反应）：
   "以下是用户与酒狐的对话（含酒狐的情感标签与回应），请提取：
    1) 用户档案信息（姓名、偏好、习惯、重要日期）
    2) 关键事件（时间、地点、人物、动作、结果）
    3) 酒狐的反应与情感（酒狐对事件的态度、情绪、特殊回应）
    4) 关系进展（亲密度变化、承诺、约定）
    输出 JSON。"
4. 基座模型推理 → 结构化输出
5. Embedder 批量嵌入新记忆（含 user 语义 + assistant 反应语义）
6. 写入 SQLite:
   - profile (更新档案)
   - recall_files (新记忆段，含完整上下文)
   - recall_file_resources (关联资源)
7. 清理短期上下文（保留最近 K 轮）
8. 重新加载酒狐 LoRA
```

**蒸馏要点**：
- 必须同时蒸馏 user 与 assistant 消息，仅蒸馏 user 会丢失酒狐的反应与情感，导致召回时无法还原对话上下文。
- recall_segments 的 content 字段保存「事件 + 酒狐反应」的摘要，而非单纯用户输入。
- 召回时注入的 [相关记忆] 应能让酒狐回忆起「当时发生了什么 + 自己当时怎么回应的」。

---

## 3. 模块设计

### 3.1 LLM 服务

**职责**：封装 llama.cpp，提供 LoRA 热加载/卸载、流式生成、KV cache 复用。

**关键接口**（Phase 1 已实现）：

```cpp
struct LlmOptions {
    int  n_ctx       = 4096;
    int  n_batch     = 2048;
    int  n_threads   = 0;     // 0 = auto (physical cores)
    bool use_mmap    = true;
    // When false, passes enable_thinking=false to the model's jinja chat
    // template, equivalent to `llama-cli -rea off`. Qwen3.5 then skips the
    // <think> reasoning block entirely.
    bool enable_thinking = false;
    // Flash Attention: LLAMA_FLASH_ATTN_TYPE_ENABLED (true) or DISABLED (false).
    bool flash_attention_enabled = true;
    // KV cache data type: "f16" (default), "q8_0", "q4_0".
    std::string kv_cache_dtype = "f16";
};

struct SamplingParams {
    float    temp            = 0.7f;
    int      top_k           = 40;
    float    top_p           = 0.9f;
    float    repeat_penalty  = 1.10f;
    int      penalty_last_n  = 64;
    int      max_tokens      = 512;
    uint32_t seed            = 0xC0FFEE;
};

class LlmService {
public:
    bool load_base(const std::string& model_path, const LlmOptions& opts);
    bool attach_lora(const std::string& lora_path, float scale = 1.0f);
    void detach_lora();
    bool lora_attached() const;

    // 流式生成，回调返回 false 提前停止
    void chat_stream(const std::vector<Message>& messages,
                     const std::function<bool(const std::string&)>& on_token,
                     const SamplingParams& sp = {});

    // 用于记忆整理的非酒狐推理（detach_lora 后调用）
    bool complete(const std::string& prompt, std::string& out,
                  const SamplingParams& sp = {});

    PerfData last_perf() const;
    uint32_t n_ctx() const;
    bool     ready() const;

private:
    llama_model*             model_  = nullptr;
    llama_context*           ctx_    = nullptr;
    const llama_vocab*       vocab_  = nullptr;
    llama_adapter_lora*      lora_   = nullptr;
    common_chat_templates_ptr chat_templates_;  // jinja 模板，支持 enable_thinking
    // ...
};
```

**实现要点**：
- 基座：`Qwen3.5-0.8B`（首选）/ `Qwen3.5-2B`（备选，能力不足时升级）。
- **Chat template**：使用 `common_chat_templates_apply`（llama.cpp common 库），支持 jinja 模板参数 `enable_thinking`。
  - **`enable_thinking=false`（默认）**：等价 `llama-cli -rea off`，从 jinja 模板源头禁止 Qwen3.5 生成 `<think>` 块。模型不进入思考模式，输出直接以 `[emotion]` 开头。
  - 需要 `LLAMA_BUILD_COMMON=ON`，链接 `llama-common` 库。
- **Flash Attention**：`flash_attention_enabled` 映射到 `LLAMA_FLASH_ATTN_TYPE_ENABLED/DISABLED`，默认开启以减少内存占用和提升推理速度。
- **KV cache 数据类型**：`kv_cache_dtype` 支持 `"f16"`（默认，精度最高）、`"q8_0"`（内存减半，精度损失极小）、`"q4_0"`（内存 1/4，精度损失较大），映射到 `llama_context_params.type_k`/`type_v`。
- **KV cache 前缀复用（INCREMENTAL）**（已实现）：
  - System prompt 只含人设 + 档案（静态），recall 作为 `role:tool` 消息插入在每轮 user 消息之后。Qwen3.5 的 chat template 将 tool 消息渲染为 `<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>`，既不破坏模板的 "system must be at the beginning" 约束，又使 system 消息保持静态，最大化 KV cache 前缀复用。
  - 历史轮次的 recall 保存在短期记忆中不变，只有当前轮的 recall 是新增的，因此历史部分（system + 所有历史 user/tool/assistant）的 token 序列完全对齐，实现 INCREMENTAL 匹配。
  - `chat_stream()` 比较新 prompt tokens 与 cached_tokens_ 的公共前缀，通过 `llama_memory_seq_rm` 保留公共前缀的 KV cache，仅 prefill 不匹配的后缀。
  - **`<think>` bridge 移除**：chat template 的 `enable_thinking=false` 在 generation prompt 末尾添加空 `<think>...</think>` 块。这些 tokens 存在于 `cached_tokens_` 但不存在于下一轮 prompt 中（template 从 assistant 历史中剥离 `<think>` 块）。生成完成后，向后搜索 `<think>` token，清除 KV cache 尾部（bridge + generated）并重新 prefill generated tokens，使 `cached_tokens_` 与下一轮 prompt 完全对齐，实现 INCREMENTAL 匹配。
  - `invalidate_cache()` 在 reset / 蒸馏后调用，强制下一轮全量 prefill。
  - 超出上下文限制时，去除最前面的几轮对话（保留 system_prompt），保证留有 30% 上下文空间用于下一轮对话。
- LoRA 通过 `llama_adapter_lora_init` + `llama_set_adapters_lora` 实现，无需重载模型。跟踪 `lora_attached_` 状态，蒸馏时仅在已 attach 时才 re-attach。
- 上下文管理：滑动窗口 + 摘要触发，n_ctx 默认 4096。
- 量化策略：**测试期用 FP16**（Q4_0 出现严重幻觉，弃用），正式构建切换 **Q8_0**。
- 采样链：penalties → top_k → top_p → temp → dist。
- 性能埋点：`PerfData` 记录 n_eval / t_eval_ms / t_prefill_ms / lora_attach_ms，us 级计时。

**性能基准**（Phase 1 实测，Q8_0 量化 + KV cache 前缀复用）：
- Q8_0 模型 + LoRA：**17.3 tok/s**（对比 FP16 9.0 tok/s，提速 ~92%）
- LoRA 加载延迟：ms 级（82.6MB LoRA）
- prefill 首轮（全量）：约 7s（Q8_0 模型，273 tokens）
- prefill 后续轮（增量）：约 1.3s（仅 prefill ~50 tokens，公共前缀 KV cache 复用）
- 生成速度：17.3 tok/s（首字延迟已从 ~10s 降至 ~1.3s + 首token生成时间）

**日志控制**：
- `init_backend()` 在程序启动时调用，`shutdown_backend()` 在退出时调用
- Release 构建中通过 `llama_log_set` 设置静默回调，过滤所有 INFO/DEBUG 日志（仅保留 ERROR 级别转发到 stderr）
- Debug 构建中保持默认行为（全部输出到 stderr）
- 条件编译：`#ifdef NDEBUG` 控制

### 3.2 语音前端（VAD + AEC + ASR）

**职责**：从麦克风获取干净语音，转为文本。

```cpp
class VoiceFrontend {
public:
    bool init(const VoiceFrontendConfig& cfg);
    void start();
    void stop();

    // 流式回调：VAD 触发的语音段
    void on_speech_segment(const std::function<void(const float*, size_t)>& cb);

    // 流式回调：ASR 识别出的部分文本
    void on_partial_text(const std::function<void(const std::string&)>& cb);

    // 流式回调：VAD 端点确认后的完整文本
    void on_final_text(const std::function<void(const std::string&)>& cb);

    // 注入远端参考（TTS 输出），供 AEC 使用
    void feed_far_end(const float* samples, size_t n);

    // 用户打断（VAD 检测到说话时，TTS 应立即停止）
    void on_interrupt(const std::function<void()>& cb);
};
```

**实现要点**：
- **VAD**：TEN-VAD，frame_size=10ms，min_speech_duration=250ms，min_silence_duration=300ms（可动态调整）。
- **AEC3**：仅在内置扬声器路由时启用，耳机时跳过。
- **ASR**：SenseVoice-Small（INT8 ONNX），支持流式（partial + final）。
- **打断逻辑**：VAD 触发 speech 时立即停止 TTS 播放和 LLM 生成。

### 3.3 TTS 服务

**职责**：流式合成酒狐语音。

```cpp
class TtsService {
public:
    bool init(const std::string& model_dir);
    // 流式合成：按句子输入，按音频块回调
    void synthesize_stream(const std::string& text,
                           const std::function<void(const int16_t*, size_t)>& on_audio,
                           const std::function<void()>& on_done);
    void stop();  // 用户打断时调用
};
```

**实现要点**：
- 模型：VITS-Tiny 蒸馏版（中文，酒狐音色），INT8 ONNX。
- 数据集：CosyVoice 跨语言克隆酒狐日语音色生成中文平行语料（一次性，Colab GPU），目标 ≥ 5 小时。
- 蒸馏：VITS-Tiny 作为学生模型，学习 CosyVoice 教师模型的酒狐音色。
- 流式：按句子切分（句号/感叹号/问号/逗号），每段独立合成，前段播放时后段并行合成。

### 3.4 记忆服务

**职责**：短期上下文管理 + 长期记忆存储/检索/整理。

短期上下文记忆原理示例（recall 作为独立 system 消息，实现 KV cache 前缀复用）:

用户第一次输入时，模型的上下文：

system:[system_prompt]

user:酒狐，你还记得那天.........

system:[相关记忆1]

assistant:当然了，那天我.......

用户第二次输入时，模型的上下文：

system:[system_prompt]

user:酒狐，你还记得那天.........			(KV Cache可复用)

system:[相关记忆1]					(KV Cache可复用)

assistant:当然了，那天我.......			(KV Cache可复用)

user:那你应该没忘了我那天答应你要........

system:[相关记忆2]

assistant:怎么可能，我都记得清楚呢，..........

用户第N次输入时，超出上下文限制，去除最前面的几轮对话（保留system_prompt)，保证留有30%上下文空间用于下一轮对话。

**关键设计**：recall 不再附加到 system prompt 中，而是作为独立 system 消息插入在每轮 user 消息之后。这样 system prompt 保持静态（仅人设 + 档案），所有 recall 和对话历史都成为连续 token 序列的一部分，下一轮 prompt 的前缀与上一轮完全对齐，KV cache 可以前缀复用。

**Qwen3.5 ChatML 多 system 消息支持**：ChatML 模板是一个简单的 for 循环（`{% for message in messages %}{{'<|im_start|>' + message['role']}}`），支持任意数量、任意顺序的 system/user/assistant 消息。`common_chat_templates_apply` 正确处理多 system 消息的渲染。

**数据库 schema**（基于已有 sqlite-memory 约定）：

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE recall_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    embedding BLOB,           -- 向量
    created_at INTEGER NOT NULL,
    last_recalled_at INTEGER,
    recall_count INTEGER DEFAULT 0
);

CREATE TABLE recall_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recall_file_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (recall_file_id) REFERENCES recall_files(id) ON DELETE CASCADE
);

CREATE TABLE recall_file_resources (
    recall_file_id INTEGER NOT NULL,
    resource_path TEXT NOT NULL,
    resource_type TEXT,
    PRIMARY KEY (recall_file_id, resource_path),
    FOREIGN KEY (recall_file_id) REFERENCES recall_files(id) ON DELETE CASCADE
);

CREATE TABLE raw_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,       -- user / assistant / system
    content TEXT NOT NULL,
    emotion TEXT,
    created_at INTEGER NOT NULL,
    session_id INTEGER NOT NULL
);

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at INTEGER NOT NULL,
    ended_at INTEGER
);
```

**关键接口**：

```cpp
class MemoryService {
public:
    bool init(const std::string& db_path, EmbedderService* embedder);

    // 短期：当前对话历史（带情感标签）
    void append_message(const Message& msg);
    std::vector<Message> get_recent_messages(int n);

    // 长期：召回与查询相关的历史
    std::vector<Recall> recall(const std::string& query, int top_k = 5);

    // 整理：将一段对话蒸馏为结构化记忆
    DistillResult distill(const std::vector<Message>& messages);

    // 写入新记忆（批量嵌入）
    void commit_recall_file(const RecallFile& file);
};
```

**召回流程**：
1. 取用户最近一句话作为 query。
2. Embedder 嵌入 query → SQLite 存储（BLOB）。
3. 余弦相似度检索 top-k recall_segments。
4. 聚合到所属 recall_files，按 file 拼接内容。
5. 注入到 system prompt 的 `[相关记忆]` 部分。

### 3.5a 配置文件模块

**职责**：集中管理所有可配置参数，提供 JSON 配置文件加载/保存。CLI 参数作为覆盖层。

**关键接口**（Phase 1 已实现）：

```cpp
struct Config {
    struct Llm {
        std::string model_path;
        std::string lora_path;
        float       lora_scale    = 1.0f;
        int         n_ctx         = 4096;
        int         n_batch       = 2048;
        int         n_threads     = 0;      // 0 = auto
        bool        use_mmap      = true;
        bool        enable_thinking = false;
        bool        flash_attention_enabled = true;
        std::string kv_cache_dtype = "f16";  // "f16", "q8_0", "q4_0"
    } llm;

    struct Sampling {
        float    temp           = 0.7f;
        int      top_k          = 40;
        float    top_p          = 0.9f;
        float    repeat_penalty = 1.10f;
        int      penalty_last_n = 64;
        int      max_tokens     = 512;
        uint32_t seed           = 0xC0FFEE;
    } sampling;

    struct Embedder { std::string model_path; } embedder;
    struct Memory   { std::string db_path = "winefox.db";
                      int recall_top_k = 3; int distill_keep_turns = 4; } memory;

    std::string system_prompt_path = "llm-finetune/system_prompt.txt";
    bool        no_lora            = false;

    bool load(const std::string& path);       // JSON → Config
    bool save(const std::string& path) const; // Config → JSON
    static Config defaults();
};
```

**实现要点**：
- 配置文件格式：JSON（复用 nlohmann/json），默认路径 `./winefox.json`，可用 `--config <path>` 指定。
- 加载策略：先加载配置文件（缺失键保持默认值），再应用 CLI 参数覆盖。
- `--gen-config <path>`：将当前配置序列化为 JSON 写入指定路径，用于生成默认配置文件。
- **扩展约定**：新增可配置参数时，在 `config.h` 添加字段 → `config.cpp` 添加 load/save 逻辑 → 消费方读取。所有可配置参数必须在配置文件中有对应项。

### 3.5 Embedder 服务

**职责**：批量化文本向量嵌入。

```cpp
class EmbedderService {
public:
    bool init(const std::string& model_path);
    std::vector<float> embed(const std::string& text);
    std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts);  // 批量
};
```

**实现要点**：
- 默认 `bge-small-zh-v1.5` Q8（~120MB），高端档可切 `Qwen3-Embedding-0.6B` Q8。
- 通过 llama.cpp 加载（与 LLM 共享 ggml 后端内存池）。
- 批量接口减少 LLM 后端切换开销。

### 3.6 UI 渲染层

**职责**：渲染酒狐形象，根据情感驱动动画。

**bbmodel 渲染器**：

```cpp
class BbModelRenderer {
public:
    bool load(const std::string& bbmodel_path);  // JSON 解析
    void set_emotion(Emotion e);                 // 切换表情/动作集
    void set_animation(const std::string& name); // 手动切动画
    void update(float dt);                       // 骨骼动画 tick
    void render(RenderContext* ctx);             // GLES3 (Windows / Android 统一)
};
```

**情感 → 动画映射**：

| Emotion 标签 | bbmodel 动画 | 表情 |
|--------------|--------------|------|
| joy | 摇尾巴、跳跳 | 微笑、眯眼 |
| neutral | 站立呼吸 | 默认 |
| surprise | 后仰、耳朵竖起 | 张嘴 |
| sadness | 低头、尾巴垂下 | 皱眉 |
| anger | 跺脚、炸毛 | 撅嘴 |
| fear | 缩成一团 | 瞳孔收缩 |

每个 emotion 对应 2-3 个动作变体，随机选取避免重复。

**交互**：
- 鼠标/触摸酒狐不同部位触发不同反应（摸头 → joy，摸尾巴 → anger）。
- 长按酒狐 → 弹出菜单（设置、记忆查看等）。

---

## 4. 平台支持矩阵

| 平台 | 架构 | GUI | 音频 IO | 备注 |
|------|------|-----|---------|------|
| Windows | x64 | ✅ GLES3 | WASAPI | 主开发平台 |
| Windows | x86 | ✅ GLES3 | WASAPI | 兼容老设备 |
| Windows | arm64 | ✅ GLES3 | WASAPI | Snapdragon X |
| Android | arm64 | ✅ GLES3 | AAudio | API 26+ |
| Linux | x64 | ❌ CLI | PulseAudio | TUI 对话 |
| Linux | x86 | ❌ CLI | PulseAudio | 兼容老设备 |
| Linux | arm64 | ❌ CLI | PulseAudio | 树莓派 4+ |
| Linux | arm32 | ❌ CLI | ALSA | 极限兼容（树莓派 2/3） |

**Linux 无 GUI 的设计**：
- 编译宏 `WINEFOX_NO_GUI` 禁用 UI 模块。
- CLI 模式：stdin/stdout 文本对话 + 可选 TTS 播放。
- 适合服务器部署（如树莓派家庭服务器）。
- arm32 设备默认禁用 TTS 实时合成（性能不足），仅文本对话；可选预生成音频文件方式。

---

## 5. 训练与数据准备

### 5.1 酒狐 LoRA 训练

**基座**：Qwen3.5-0.8B（首选，Instruct 版本）/ Qwen3.5-2B（备选，能力不足时升级）

**数据**：
- 已有 `llm-finetune/winefox_sft.jsonl`（约 200 条多轮对话）。
- 扩充目标：≥ 2000 条，覆盖：
  - 6 种情感各 ≥ 300 条
  - 带记忆注入对话 ≥ 500 条
  - 长程多轮（≥ 5 轮）≥ 300 条

**训练参数**（建议）：
- LoRA rank: 16, alpha: 32
- target_modules: q_proj, k_proj, v_proj, o_proj
- epochs: 3
- lr: 1e-4
- batch_size: 4 + grad_accum 4

**输出**：转为 gguf 格式（`winefox-lora-f16.gguf`），供 llama.cpp 加载。

### 5.2 TTS 数据集与蒸馏训练

**流程**：

```
1. 准备日语音色参考（已有 WineFoxModel 中的酒狐日语语音包）
2. 文本语料（≥ 5 小时）：
   - llm-finetune 中酒狐的所有 assistant 回复
   - 扩充日常对话语料（朗诵、故事、闲聊）
3. CosyVoice 跨语言克隆（教师模型）：
   - 参考音频：日语酒狐样本
   - 目标文本：中文
   - 输出：中文酒狐音色音频（教师输出）
4. 数据清洗：
   - 人工抽检音色一致性
   - 剔除发音错误/不自然的样本
   - 标注 emotion 标签
5. 切分为 train/val (95/5)
6. VITS-Tiny 蒸馏训练（学生模型）：
   - 输入：文本 + 教师音频（作为蒸馏目标）
   - 损失：mel-spectrogram L1 + KL + adversarial + speaker embedding 一致性
   - 目标：学生模型学习教师模型的酒狐音色，但推理速度更快
7. 导出 ONNX 供 sherpa-onnx 使用
```

**VITS-Tiny 蒸馏训练要点**：
- 学生模型：VITS-Tiny（参数量约 5-15M，远小于标准 VITS）
- 教师模型：CosyVoice（仅用于离线生成数据，不参与线上推理）
- 单说话人（酒狐）
- 目标推理速度：CPU RTF ≤ 0.3（即 1s 音频合成 ≤ 0.3s，远快于实时）
- 训练硬件：单卡 GPU（Colab 免费版可完成）
- 训练数据量：≥ 5 小时平行语料（文本 → 教师音频）

### 5.3 ASR 模型

直接使用 sherpa-onnx 提供的 SenseVoice-Small 预训练 INT8 模型，无需训练。

---

## 6. 关键技术难点与解决方案

### 6.1 流式管线协调

**难点**：LLM、TTS、动画三路并行，需要协调打断、错峰、缓冲。

**方案**：
- `PipelineScheduler` 单例，持有三路任务队列。
- LLM 流式输出 → `SentenceBuffer`（按标点切句）→ 推入 TTS 队列。
- TTS 队列消费：当前句合成 + 下一句预合成。
- 任何时刻用户打断：清空 LLM 队列、停止 TTS、切换 idle 动画。

### 6.2 LLM prefill 与 ASR 并行

**难点**：ASR 流式输出 partial text 时，LLM 是否可以提前 prefill？

**方案**：
- ASR final text 之前不启动 LLM（避免错误启动）。
- 但在 ASR 进行中，可以预热 LLM：加载 prompt 模板 + 长期记忆到 KV cache。
- ASR final 一到，立即追加 user message → decode。

### 6.3 KV cache 前缀复用（已实现，替代原分段方案）

**原方案**（seq 0 静态段 + seq 1 动态段）已废弃。新方案更简单：recall 作为独立 system 消息插入在 user 消息之后，使 system prompt 保持静态，所有对话历史（含 recall）成为连续 token 序列。

**实现**（见 [3.1 节](#L346)）：
- `chat_stream()` 比较新 prompt tokens 与 `cached_tokens_`（上一轮 prompt + generated tokens）的公共前缀
- `llama_memory_seq_rm(mem, 0, common, -1)` 保留公共前缀的 KV cache，清除不匹配尾部
- `prefill_tokens_(new_tokens, common)` 仅 prefill 后缀
- BPE tokenizer 在 prompt/generated 交界处可能产生 token 边界差异，公共前缀通常为 90%+，只需重新 prefill ~50 tokens

**实测效果**（Q8_0 模型）：
- 首轮全量 prefill：~7s（273 tokens）
- 后续轮增量 prefill：~1.3s（仅 ~50 tokens）
- 生成速度：17.3 tok/s
- 首字延迟：从 ~10s 降至 ~1.3s + 首token生成时间（~60ms）

### 6.4 AEC 远端参考对齐

**难点**：TTS 输出 → 扬声器播放 → 麦克风采集，存在延迟，需要对齐。

**方案**：
- TTS 写入播放缓冲时，同步拷贝到 AEC far-end 队列。
- 估算播放延迟（启动时探测，~50-200ms），AEC 内部 delay compensation。
- 提供「AEC 校准」向导：播放 sweep tone，测量环路延迟。

### 6.5 bbmodel 渲染

**难点**：bbmodel 是 BlockBench 格式（基岩版实体），非标准骨骼动画。

**方案**：
- bbmodel 本质是 JSON，含 cubes、bones、animations。
- 实现 JSON 解析器 → 构建骨骼树。
- 动画按关键帧插值（位置、旋转、缩放）。
- 立方体渲染：统一用 GLES3 VBO + 简单 vertex/fragment shader，Windows 与 Android 共享同一套渲染代码。
- 贴图：从 bbmodel 引用的 PNG 加载，UV 映射。
- 已有素材：`WineFoxModel/模型文件/人物模型/大正女仆酒狐.bbmodel`（默认形象）。

### 6.6 跨平台音频 IO 抽象

```cpp
class AudioIo {
public:
    virtual bool init(const AudioIoConfig& cfg) = 0;
    virtual void start_input()  = 0;  // 麦克风
    virtual void start_output() = 0;  // 扬声器
    virtual void on_input_samples(const std::function<void(const float*, size_t)>&) = 0;
    virtual void write_output(const float* samples, size_t n) = 0;
    virtual bool is_headset_connected() = 0;  // 用于 AEC 决策
};
```

- Windows: `WASAPI` 实现
- Android: `AAudio` 实现
- Linux: `PulseAudio` (或 `ALSA`) 实现

---

## 7. 实施阶段

### Phase 1：核心 MVP（Windows x64，文本对话）

**目标**：跑通 LLM + 记忆系统的端到端文本对话。

**实施范围**：MVP + Distiller（最小可行版本 + 记忆蒸馏），跳过单元测试。

- [x] 项目骨架（CMake + llama.cpp b10069 submodule + llama-common + sqlite + json）
- [x] SQLite schema 与 MemoryService（migration 幂等迁移）
- [x] EmbedderService（bge-small-zh Q8_0，mean pooling + L2 归一化）
- [x] LlmService（llama.cpp 封装，LoRA 热加载/卸载，流式生成，采样链，PerfData 埋点）
- [x] Chat template（`common_chat_templates_apply`，`enable_thinking=false` 从源头禁止 `<think>` 块）
- [x] 短期上下文管理（滑动窗口 70% 阈值 + drain_for_distill）+ 长期记忆召回注入（cosine + 时间衰减）
- [x] 记忆蒸馏（detach lora → 基座推理 JSON → 批量嵌入 → 写库 → re-attach）
- [x] CLI 文本对话入口（参数解析 + REPL + /quit /memory /reset 命令 + 流式输出 + perf 显示）

**Phase 1 验收结果**：
- 端到端对话跑通，`<think>` 块从源头消除，emotion 标签正确解析
- **配置文件**：`winefox.json` 加载/覆盖正常，`--gen-config` 可生成默认配置
- 性能：**9.4 tok/s**（对比官方 EXE 10 tok/s 基准，差距 6%）
- winefox.exe = 4.5 MB，LoRA 82.6MB 正常加载
- 待验证：多轮记忆闭环、Linux x64 兼容性

**验收**：能在 Windows x64 上文本对话，记得用户姓名/偏好，长时间对话能触发记忆整理。

### Phase 2：语音前端（VAD + ASR + TTS）

**目标**：实现实时语音对话。

- [ ] AudioIo (WASAPI)
- [ ] VadService（TEN-VAD）
- [ ] AsrService（SenseVoice-Small）
- [ ] TtsService（自训 VITS，先临时用 sherpa-onnx 内置模型）
- [ ] PipelineScheduler 流式管线
- [ ] 用户打断逻辑

**验收**：实时语音对话，P50 首字延迟 ≤ 2.5s。

### Phase 3：AEC 与品质优化

**目标**：扬声器模式下无回声，达到生产级品质。

- [ ] AecService（WebRTC AEC3 提取）
- [ ] 耳机检测自动跳过 AEC
- [ ] VAD 端点延迟动态调整
- [x] LLM prefill 优化（KV cache 前缀复用，已在 Phase 1 实现，见 [6.3](#L757)）
- [ ] TTS 首块流式优化
- [ ] 性能调优：P50 ≤ 1.5s

**验收**：扬声器模式无回声，P50 ≤ 1.5s，长时间运行稳定。

### Phase 4：GUI 与酒狐形象（Windows）

**目标**：图形界面 + 酒狐形象 + 情感驱动动画。

- [ ] Win32 窗口 + GLES3 渲染上下文（含 ANGLE 兜底）
- [ ] bbmodel 解析器
- [ ] 骨骼动画系统
- [ ] 表情切换（贴图 UV 切换或顶点形变）
- [ ] EmotionDriver（LLM 输出解析 → 动画切换）
- [ ] 用户交互（点击/拖拽酒狐）
- [ ] 设置面板

**验收**：完整图形化体验，6 种情感对应动画，可交互。

### Phase 5：TTS 自训模型替换

**目标**：用酒狐音色替换临时 TTS。

- [ ] CosyVoice 跨语言克隆生成数据集（Colab GPU）
- [ ] VITS-Tiny 蒸馏训练（学习 CosyVoice 教师模型的酒狐音色）
- [ ] 导出 ONNX
- [ ] 集成到 TtsService
- [ ] 音质评估与调优

**验收**：TTS 输出酒狐音色，自然度可接受，CPU RTF ≤ 0.3。

### Phase 6：跨平台扩展

**目标**：扩展到所有目标平台。

- [ ] Windows x86 / arm64
- [ ] Linux x64 / x86 / arm64 / arm32（CLI 模式）
- [ ] Android arm64
- [ ] Android GUI（NativeActivity + GLES3）
- [ ] Android 音频（AAudio）
- [ ] CI/CD 多平台构建

**验收**：所有目标平台可运行，性能达标。

### Phase 7：打磨与发布

- [ ] 安装包制作（Windows MSI、Android APK、Linux AppImage）
- [ ] 模型下载器（首次启动按需下载）
- [ ] 文档与用户引导
- [ ] 性能基准测试套件
- [ ] 错误处理与崩溃恢复

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 老旧 CPU 性能不达标 | 高 | 分级 preset，0.6B 起步，文档明确最低配置 |
| TTS 自训音色不自然 | 中 | 保留 sherpa-onnx 内置 VITS 作为 fallback |
| AEC3 跨平台不一致 | 中 | 默认关闭，提供手动校准向导 |
| LoRA 训练效果不佳 | 中 | 扩充数据集到 5000+，调整 rank/alpha |
| 内存超标 | 中 | 模型按需加载，embedding/ASR 可选降级 |
| bbmodel 渲染复杂度 | 中 | 先实现静态模型 + 简单动画，逐步增强 |
| Android 兼容性 | 中 | 最低 API 26，限定 arm64，减少设备碎片 |

---

## 9. 工程约定

继承自项目历史约束（参见 `project_memory.md`）：

- **CMake**：`project(winefox C CXX)`，C++ 标准 C++17。
- **MSVC**：必须 `/utf-8`，源文件统一 UTF-8。
- **llama.cpp**：git submodule 接入 `third_party/llama.cpp`，pinned to release tag b10069。`LLAMA_BUILD_COMMON=ON` 以使用 `common_chat_templates_apply`。
- **SQLite**：放置 `third_party/sqlite/sqlite3.c` + `sqlite3.h`。
- **nlohmann/json**：使用 llama.cpp vendor 目录中的版本（`third_party/llama.cpp/vendor/nlohmann/json.hpp`，3.12.0），避免与 common 库版本冲突。
- **控制台**：Windows 下 `SetConsoleOutputCP(CP_UTF8)`。
- **Chat template**：使用 `common_chat_templates_apply`（非 `llama_chat_apply_template`），支持 jinja 模板参数（如 `enable_thinking`）。
- **思考模式**：`LlmOptions.enable_thinking = false` 默认关闭（等价 `llama-cli -rea off`），从源头禁止 Qwen3.5 生成 `<think>` 块。无需在输出流中过滤。
- **MemoryService**：初始化时 `PRAGMA foreign_keys = ON`，自动创建数据目录。
- **Distiller**：蒸馏时同时处理 user 与 assistant 消息，记录酒狐的反应与情感，仅蒸馏 user 会丢失对话上下文。
- **commit_recall_file**：先批量收集所有新 segment，批量嵌入后一次性插入。
- **Embedder**：必须实现 `embed_batch` 接口以减少后端切换。
- **recall_file_resources**：联表用于关联资源（如对话中提到的图片、文件）。
- **量化策略**：测试期用 FP16（Q4_0 出现严重幻觉，弃用），正式构建切换 Q8_0。
- **日志控制**：`init_backend()` / `shutdown_backend()` 管理 llama.cpp 后端生命周期。Release 构建中通过 `#ifdef NDEBUG` 设置静默日志回调，过滤所有非 ERROR 级别日志。Debug 构建保持默认（stderr 输出）。

代码规范：
- 命名：类 PascalCase，方法/变量 snake_case，成员变量后缀 `_`。
- 头文件：`#pragma once`。
- 错误处理：用 `std::expected` 或返回 bool + out 参数，避免异常。
- **日志**：使用 `#ifdef DEBUG` 包裹日志输出，仅在调试构建中启用；Release 构建完全移除日志代码，避免 I/O 与字符串格式化开销。轻量 spdlog 仅在 DEBUG 构建中链接。
- 测试：GoogleTest，模块单元测试覆盖率 ≥ 60%。

---

## 10. 参考资料

- llama.cpp: https://github.com/ggml-org/llama.cpp
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- TEN-VAD: https://github.com/TEN-framework/ten-vad
- WebRTC AEC3: https://chromium.googlesource.com/chromium/src/+/refs/heads/main/third_party/webrtc/modules/audio_processing/aec3/
- CosyVoice (教师模型): https://github.com/FunAudioLLM/CosyVoice
- VITS-Tiny (学生模型，蒸馏目标): https://github.com/jaywalnut310/vits (基础 VITS，Tiny 为轻量化变体)
- BlockBench 模型格式: https://blockbench.net/wiki/api/project
- Qwen3.5 模型: https://huggingface.co/Qwen
- bge-small-zh: https://huggingface.co/BAAI/bge-small-zh-v1.5
- 已有 SFT 数据集: `llm-finetune/winefox_sft.jsonl`
- 已有酒狐人设: `llm-finetune/system_prompt.txt`
- 已有 bbmodel 素材: `WineFoxModel/模型文件/人物模型/`

---

## 11. 整体复杂度评估与调优建议

### 11.1 模块复杂度矩阵

| 模块 | 复杂度 | 主要风险 | 备注 |
|------|--------|----------|------|
| 跨平台构建（8 平台组合） | 极高 | 第三方依赖跨架构编译、CI 矩阵爆炸 | 最大风险源 |
| 实时语音管线（VAD+AEC+ASR+LLM+TTS） | 高 | AEC3 调参、流式协调、打断一致性 | 核心体验 |
| 自研 bbmodel 渲染器 | 中高 | 骨骼动画、表情系统、GLES3 兼容性 | 已统一 GLES3 降低复杂度 |
| 记忆系统（短期+长期+蒸馏） | 中 | KV cache 分段、召回质量、蒸馏完整性 | 已修正蒸馏范围 |
| TTS 蒸馏（CosyVoice→VITS-Tiny） | 中 | 教师模型音色一致性、蒸馏损失设计 | 依赖 GPU 离线训练 |
| LoRA 训练 | 低中 | 数据集扩充、过拟合 | 流程成熟 |
| AEC3 集成 | 中高 | 远端参考对齐、跨平台行为差异 | 建议默认关闭 |
| Android 平台适配 | 中高 | AAudio 延迟、GLES3 设备碎片、NDK 构建 | 延后到 Phase 6 |

### 11.2 已识别的调优点

#### 11.2.1 跨平台构建应进一步收敛起步范围

**问题**：8 个平台组合（Win x86/x64/arm64 + Android arm64 + Linux x86/x64/arm64/arm32）的 CI 矩阵和本地验证成本极高，第三方依赖（llama.cpp、sherpa-onnx、webrtc-apm）在 arm32/arm64 上的构建脚本调试可能消耗大量时间。

**建议**：
- Phase 1-5 仅锁定 **Windows x64 + Linux x64** 两个主开发平台。
- Phase 6 再扩展其余平台，并按优先级排序：Linux arm64（树莓派4）> Windows x86 > Android arm64 > Windows arm64 > Linux x86 > Linux arm32。
- arm32 Linux 作为「极限兼容」目标，可接受功能裁剪（如禁用 TTS 实时合成、仅文本对话）。

#### 11.2.2 AEC3 应作为可选模块，默认关闭

**问题**：WebRTC AEC3 提取与集成复杂度高，远端参考对齐在不同硬件上行为差异大，容易成为体验杀手。耳机场景下完全不需要 AEC。

**建议**：
- Phase 2 先不集成 AEC，默认推荐用户使用耳机。
- Phase 3 再加入 AEC，但默认关闭，提供「扬声器模式」开关手动启用。
- 提供 AEC 校准向导（播放 sweep tone 测量环路延迟）。

#### 11.2.3 bbmodel 渲染器应分阶段实现

**问题**：完整的骨骼动画 + 表情切换 + 交互系统一次性实现复杂度高。

**建议**分三步：
1. **静态模型渲染**：仅加载 bbmodel，渲染立方体 + 贴图，无动画。验证 GLES3 管线。
2. **基础动画**：实现骨骼关键帧插值，支持 idle 呼吸 + 6 种 emotion 对应的预设动画。
3. **交互增强**：点击/拖拽触发反应、表情切换、长按菜单。

#### 11.2.4 KV cache 复用策略（已解决）

**原问题**：[6.3 节](file:///e:/winefox/PLAN.md#L757) 原提出的「seq 0 静态段 + seq 1 动态段」方案依赖多 sequence API。

**解决方案**：放弃多 sequence 方案，改用前缀复用。recall 作为独立 system 消息插入在 user 消息之后，使 system prompt 保持静态，所有对话历史成为连续 token 序列。`chat_stream()` 通过 `llama_memory_seq_rm` 保留公共前缀 KV cache，仅 prefill 不匹配后缀。实测 prefill 从 ~7s 降至 ~1.3s（见 [6.3 节](file:///e:/winefox/PLAN.md#L757)）。

#### 11.2.5 TTS 蒸馏的音色一致性风险

**问题**：CosyVoice 跨语言克隆（日语→中文）本身可能有音色漂移；VITS-Tiny 蒸馏后可能进一步损失音色特征。两次损失叠加可能导致最终音色与酒狐原音色差距过大。

**建议**：
- 蒸馏前先人工评估 CosyVoice 生成的中文音频音色相似度，不达标则调整参考音频或换用其他克隆方案。
- VITS-Tiny 蒸馏时加入 speaker embedding 一致性损失，强制保留音色。
- 保留 sherpa-onnx 内置中文 TTS 作为 fallback，确保功能可用。

#### 11.2.6 内存预算在 arm32 上仍需关注

**问题**：arm32 Linux 设备（树莓派 2/3）通常只有 1GB 物理内存，即使默认档 ~1.4GB 也无法容纳。

**建议**：
- 为 arm32 单独设计「极简档」：
  - LLM 用 Q3_K_S 量化（~350MB）
  - 禁用 Embedding 模型，长期记忆改用 SQLite FTS5 全文检索（无向量召回）
  - 禁用 TTS，仅文本对话
  - 禁用 AEC
  - 总内存目标 ≤ 700MB
- 通过编译宏 `WINEFOX_MINIMAL` 启用极简配置。

#### 11.2.7 日志与调试构建的工程实现

**问题**：`#ifdef DEBUG` 包裹所有日志会污染业务代码可读性。

**建议**：用宏封装日志调用，Release 构建中宏展开为空：

```cpp
// log.h
#ifdef DEBUG
  #include <spdlog/spdlog.h>
  #define WF_LOG_INFO(fmt, ...)  SPDLOG_INFO(fmt, ##__VA_ARGS__)
  #define WF_LOG_ERROR(fmt, ...) SPDLOG_ERROR(fmt, ##__VA_ARGS__)
#else
  #define WF_LOG_INFO(fmt, ...)  ((void)0)
  #define WF_LOG_ERROR(fmt, ...) ((void)0)
#endif
```

业务代码统一用 `WF_LOG_*` 宏，Release 构建零开销，且不污染代码可读性。

#### 11.2.8 LoRA 热加载/卸载的延迟需测量

**问题**：记忆整理流程需要「卸载 LoRA → 基座推理 → 重新加载 LoRA」，若 LoRA 加载延迟过高（>2s），会阻塞用户对话。

**建议**：
- Phase 1 实现时测量 `llama_lora_adapter_init` + `llama_set_lora_adapter` 的实际延迟。
- 若延迟过高，考虑：保留 LoRA 启用，通过 prompt 工程让酒狐模型做摘要（但可能受人设干扰）；或用第二个 llama_context 加载基座模型专门做摘要（牺牲内存）。

### 11.3 整体复杂度结论

本项目是一个**高复杂度的系统工程**，涉及：
- 5 个独立 AI 模型的协同推理（LLM、Embedding、ASR、TTS、VAD）
- 8 个平台组合的跨平台构建
- 自研图形渲染器（bbmodel）
- 自训模型（LoRA + TTS 蒸馏）
- 实时流式管线与打断处理
- 短期+长期记忆系统

**关键成功因素**：
1. 严格按 Phase 1-7 渐进式交付，避免一次性铺开。
2. Phase 1-3 聚焦 Windows x64 + Linux x64，跑通核心闭环。
3. AEC、bbmodel 高级动画、跨平台扩展等高复杂度模块延后。
4. 每个 Phase 有明确的验收标准和可回退方案（fallback）。

**预估的关键风险点**：
- 跨平台第三方依赖构建（最高风险，建议 Phase 6 专门处理）
- AEC3 调参（建议默认关闭，耳机优先）
- TTS 蒸馏音色一致性（建议保留 fallback）
- arm32 内存约束（建议极简档配置）

通过分阶段交付 + 每阶段可回退的设计，整体风险可控。
