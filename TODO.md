# WineFox 项目实施 TODO

> 基于 [PLAN.md](./PLAN.md) 的可执行任务清单。按依赖顺序排列，每项可独立勾选。
>
> **起步平台**：Windows x64 + Linux x64（Phase 1-5），其余平台 Phase 6 扩展。
>
> **状态约定**：`[ ]` 待办 / `[~]` 进行中 / `[x]` 完成 / `[!]` 阻塞

---

## Phase 0：项目骨架与环境准备

### 0.1 仓库与构建系统

- [x] 创建 `src/`、`third_party/`、`tests/`、`cmake/` 目录结构（对齐 PLAN.md [2.2](./PLAN.md#L170-L208)）
- [x] 编写根 `CMakeLists.txt`：`project(winefox C CXX)`，C++17，MSVC `/utf-8`
- [x] 配置 `cmake/WineFoxOptions.cmake`：选项 `WINEFOX_BUILD_GUI`、`WINEFOX_WITH_AEC`、`WINEFOX_MINIMAL`、`WINEFOX_DEBUG`、`WINEFOX_BUILD_CLI`、`WINEFOX_BUILD_TESTS`、`WINEFOX_VOICE_ENABLED`
- [x] 编写 `.gitignore`（build/、models/*.gguf 大文件、venv/、IDE 缓存）
- [x] 初始化 git 仓库，首次提交

### 0.2 第三方依赖接入

- [x] llama.cpp：git submodule 接入 `third_party/llama.cpp`（pinned to release tag b10069），CMake `add_subdirectory`，验证 Windows x64 Release 构建
- [x] **LLAMA_BUILD_COMMON=ON**：链接 `llama-common` 库（为使用 `common_chat_templates_apply` 支持 `enable_thinking=false`，等价 `llama-cli -rea off`，从源头禁止 Qwen3.5 生成 `<think>` 块）
- [x] **新架构**：抛弃 sherpa-onnx 中间层，改为后端直连：
  - VAD：TEN-VAD-GGML.cpp（纯 C++17 移植，GGML 模型格式，零外部依赖，无 onnxruntime）
  - ASR：SenseVoice.cpp（ggml 后端，与 llama.cpp 共享 ggml）
  - TTS：Kokoro ONNX + kokoro.cpp 的 G2P 前端（onnxruntime 后端，Phase 3 接入）
  - 详见 `voice-test/README.md`
- [x] onnxruntime：预编译包已接入 `voice-test/third_party/onnxruntime-static/`（Win x64，**仅 TTS 使用**）。Android arm64-v8a 待 Phase 6 接入。
- [x] ten-vad-ggml：git submodule 接入 `third_party/ten-vad-ggml`（`https://github.com/xhy2008/TEN-VAD-GGML.cpp`，纯 C++17，自带 1024 点 FFT，峰值内存 5.5MB，加载 4ms，RTF 0.011）。**已删除旧 ONNX 预编译库** `voice-test/third_party/ten-vad/`（ten_vad.dll 510KB 自包含 onnxruntime 方案弃用）
- [x] SenseVoice.cpp：git submodule 接入 `voice-test/third_party/sensevoice-cpp`（自带 ggml，voice-test 独立构建，避免与 llama.cpp 的 ggml target 冲突）
- [x] kokoro.cpp：vendor 其 G2P 前端代码（ZHFrontend/ZHG2P/ToneSandhi/JiebaProcessor/EnG2P/Tokenizer）到 `voice-test/third_party/kokoro-cpp-src/`
- [ ] webrtc-apm：克隆 WebRTC 源码，提取 AEC3 模块到 `third_party/webrtc-apm`（Phase 3 用，本阶段仅占位）
- [x] SQLite：下载 amalgamation，放置 `third_party/sqlite/sqlite3.c` + `sqlite3.h`
- [x] nlohmann/json：使用 llama.cpp vendor 目录中的版本（`third_party/llama.cpp/vendor/nlohmann/json.hpp`，3.12.0），避免与 common 库版本冲突
- [ ] GoogleTest：submodule 占位接入（Phase 1 MVP 跳过单元测试）

### 0.3 模型与素材准备

- [x] 核实 `models/Qwen3.5-0.8B.gguf` 是否为 Instruct 版本；若否，下载 Instruct 版
- [x] 下载 `Qwen3.5-0.8B-Instruct-Q4_0.gguf` 作为默认基座（注：Q4_0 量化出现严重幻觉，弃用；FP16 测试通过；**Q8_0 量化已切换**，生成速度 17.3 tok/s，CPU_REPACK 优化生效）
- [x] 下载 `bge-small-zh-v1.5-Q8_0.gguf` 作为默认 embedding 模型
- [x] 下载 SenseVoice-Small INT8 ONNX 模型到 `models/asr/`
- [x] 下载 TEN-VAD 模型到 `models/vad/`（ONNX 版已弃用删除，改用 GGML 格式 `ten-vad-ggml.bin`）
- [x] 验证 `models/winefox-lora-f16.gguf` 与新基座的兼容性（LoRA 层名匹配）

### 0.4 基础工具模块

- [x] `src/core/log/log.h`：实现 `WF_LOG_INFO/WARN/ERROR/DEBUG` 宏（对齐 [11.2.7](./PLAN.md#L906-L924)）
- [x] `src/core/util/utf8.h`：UTF-8 字符串工具（按 token 切分、按句切分）
- [x] `src/core/util/strings.h`：通用字符串工具（trim、split、join）
- [x] `src/core/util/time.h`：高精度计时器（用于性能埋点）
- [x] Windows 平台：`SetConsoleOutputCP(CP_UTF8)` 初始化封装（`src/platform/console.h`）

**Phase 0 验收**：`cmake -B build && cmake --build build` 在 Windows x64 编译空骨架成功；llama.cpp b10069 + llama-common + sqlite + json 全部链接通过。

---

## Phase 1：核心 MVP（文本对话 + 记忆系统）

> 目标：Windows x64 上跑通 LLM + 短期/长期记忆的端到端文本对话。
>
> **实施范围**：MVP + Distiller（最小可行版本 + 记忆蒸馏），跳过单元测试。

### 1.1 SQLite 存储层

- [x] `src/core/storage/sqlite_db.h/cpp`：SQLite 连接封装（RAII、prepared statement 缓存）
- [x] `src/core/storage/schema.sql`：按 [3.4 节](./PLAN.md#L358-L410) 定义 profile / recall_files / recall_segments / recall_file_resources / raw_messages / sessions 表
- [x] `src/core/storage/migration.h/cpp`：schema 版本管理与自动迁移（幂等，内联 SQL）
- [x] MemoryService 初始化时 `PRAGMA foreign_keys = ON`，自动创建数据目录
- [ ] 单元测试：建表、CRUD、外键级联删除、并发读写（Phase 1 MVP 跳过）

### 1.2 Embedder 服务

- [x] `src/core/embedder/embedder_service.h/cpp`：通过 llama.cpp 加载 embedding 模型
- [x] 实现 `embed(text) -> vector<float>`
- [x] 实现 `embed_batch(texts) -> vector<vector<float>>`（对齐项目约定）
- [x] 向量归一化（余弦相似度前置，L2 normalize）
- [x] 使用 `LLAMA_POOLING_TYPE_MEAN` + `llama_get_embeddings_seq`，`llama_tokenize` 接受 `const llama_vocab*`（b10069 API）
- [ ] 单元测试：嵌入维度正确、批量与单条结果一致（Phase 1 MVP 跳过）

### 1.3 记忆服务 - 短期上下文

- [x] `src/core/memory/message.h`：Message 结构（role、content、emotion、timestamp）
- [x] `src/core/memory/short_term.h/cpp`：短期上下文管理
  - `append(msg)`
  - `all()` 返回最近消息
  - 滑动窗口：`needs_summary(n_ctx)` 70% 阈值触发摘要
  - `drain_for_distill(keep_turns)` 保留最近 N 轮，其余交给蒸馏
  - `approx_tokens()` 估算 token 数
- [ ] 单元测试：滑动窗口边界、多轮追加（Phase 1 MVP 跳过）

### 1.4 记忆服务 - 长期记忆

- [x] `src/core/memory/recall.h/cpp`：长期记忆召回
  - `recall(query, top_k=5)`：query 嵌入 → 余弦相似度检索 recall_segments → 聚合到 recall_files
  - 注入到 system prompt 的 `[相关记忆]` 部分
- [x] 召回结果去重与时间衰减（`0.7*sim + 0.3*recency`，对齐 [3.4 节](./PLAN.md#L358-L410)）
- [x] `embedder_ready()` 检查，嵌入不可用时跳过召回
- [x] `start_session()` / `save_raw_messages()` / `commit_recall_file()` 事务封装
- [ ] 单元测试：写入记忆 → 召回 → 相似度排序正确（Phase 1 MVP 跳过）

### 1.5 LLM 服务

- [x] `src/core/llm/llm_service.h/cpp`：封装 llama.cpp
  - `load_base(model_path, opts)`
  - `attach_lora(lora_path, scale)` / `detach_lora()` / `lora_attached()`
  - `chat_stream(messages, on_token, sp)` 流式回调
  - `complete(prompt, out, sp)` 非流式（用于记忆整理）
  - `last_perf()` 返回 PerfData（n_eval, t_eval_ms, t_prefill_ms, lora_attach_ms, tokens_per_sec）
- [x] Chat template：使用 `common_chat_templates_apply`（jinja 模板，支持 `enable_thinking` 参数）
- [x] **思考模式控制**：`LlmOptions.enable_thinking = false` 默认关闭，等价 `llama-cli -rea off`，模型从源头不生成 `<think>` 块
- [x] **Flash Attention**：`LlmOptions.flash_attention_enabled` 控制 `LLAMA_FLASH_ATTN_TYPE_ENABLED/DISABLED`
- [x] **KV cache 数据类型**：`LlmOptions.kv_cache_dtype` 支持 "f16"（默认）、"q8_0"、"q4_0"，映射到 `type_k`/`type_v`
- [x] 情感标签解析：从 LLM 输出首 token 提取 `[emotion]`（由 ConversationManager 处理）
- [x] **KV cache 前缀复用（INCREMENTAL）**：recall 作为 `role:tool` 消息插入在 user 消息之后（Qwen3.5 模板将 tool 渲染为 `<tool_response>` 块），使 system prompt 保持静态。`chat_stream()` 比较新 prompt tokens 与 `cached_tokens_` 的公共前缀，通过 `llama_memory_seq_rm` 保留公共前缀 KV cache，仅 prefill 不匹配后缀。生成后移除 chat template 添加的 `<think>` bridge tokens（清除 KV cache 尾部并重新 prefill generated tokens），使 `cached_tokens_` 与下一轮 prompt 完全对齐。实测 prefill 从 ~7s（FIRST）降至 ~450ms（INCREMENTAL）
- [x] 测量 LoRA 加载/卸载延迟（对齐 [11.2.8](./PLAN.md#L926-L932)），us 级计时
- [x] 采样链：penalties → top_k → top_p → temp → dist
- [ ] 单元测试：LoRA 加载前后输出对比、流式回调 token 完整（Phase 1 MVP 跳过）

### 1.5a 配置文件模块

- [x] `src/core/config/config.h/cpp`：JSON 配置文件加载/保存
  - `Config::load(path)` 从 JSON 加载，缺失键保持默认值
  - `Config::save(path)` 序列化为 JSON（用于 `--gen-config` 生成默认配置）
  - `Config::defaults()` 返回全默认配置
  - 分区结构：`llm` / `sampling` / `embedder` / `memory` + 顶层 misc
  - **可配置参数**（所有参数均在配置文件中有对应项）：
    - LLM：model_path, lora_path, lora_scale, n_ctx, n_batch, n_threads, use_mmap, enable_thinking, flash_attention_enabled, kv_cache_dtype
    - Sampling：temp, top_k, top_p, repeat_penalty, penalty_last_n, max_tokens, seed
    - Embedder：model_path
    - Memory：db_path, recall_top_k, distill_keep_turns
    - Misc：system_prompt_path, no_lora
- [x] `winefox.json`：默认配置文件模板（项目根目录）
- [x] **扩展约定**：新增可配置参数时，在 config.h 添加字段 → config.cpp 添加 load/save → 消费方读取

### 1.6 记忆蒸馏（Distiller）

- [x] `src/core/memory/distiller.h/cpp`：短期上下文超阈值时触发整理
  - 触发条件：短期上下文超过 n_ctx 70%
  - 保存 raw_messages（含 user + assistant + emotion）
  - `detach_lora()` → 基座推理生成结构化 JSON（对齐 [2.3 节](./PLAN.md#L249-L266)）
  - **蒸馏 prompt 同时处理 user 与 assistant 消息**，记录酒狐反应
  - Embedder 批量嵌入新记忆
  - 写入 profile / recall_files / recall_file_resources
  - `attach_lora()` 恢复（跟踪 LoRA 状态，仅在之前已 attach 时才 re-attach）
- [x] `parse_and_commit_`：从模型输出提取 JSON（找第一个 `{` 和最后一个 `}`），fallback 到每条 user 消息作为一个 segment
- [x] 采样参数：temp=0.3, max_tokens=1024, seed=0xD15111
- [ ] 单元测试：模拟对话 → 触发蒸馏 → 验证记忆写入正确（Phase 1 MVP 跳过）

### 1.7 对话管理器

- [x] `src/core/pipeline/conversation_manager.h/cpp`：对话状态机
  - 组装 messages：system（人设 + [档案] profile，静态）+ 短期窗口（每轮 user 后插入 recall 作为 tool 消息）+ 当前 user + 当前 recall（tool）
  - 调用 LlmService.chat_stream
  - 解析情感标签（缓冲首批 token 直到找到 `]`，6 种 emotion：joy/neutral/surprise/sadness/anger/fear）
  - 流式 token 转发到 UI 回调
  - 保存到短期记忆（含 recall），触发蒸馏
- [x] **recall 作为 tool 消息**（非附加到 system prompt），使 system prompt 保持静态，KV cache 可前缀复用
- [x] assistant 消息恢复 `[emotion]` 标签（保持模型输出模式一致）
- [x] `maybe_distill()`：短期窗口溢出时触发 Distiller，蒸馏后 `invalidate_cache()`
- [x] `reset()`：清空短期记忆 + `invalidate_cache()` 强制下一轮全量 prefill
- [x] `get_memory_info()` 命令支持
- [ ] 单元测试：多轮对话上下文累积、记忆注入正确（Phase 1 MVP 跳过）

### 1.8 CLI 入口

- [x] `src/app/cli_main.cpp`：stdin/stdout 文本对话
  - 启动时从 `winefox.json` 加载配置（可用 `--config <path>` 指定），CLI 参数作为覆盖
  - 初始化链：console UTF-8 → Config → SQLite → migration → Embedder → LLM → RecallService → ShortTermMemory → Distiller → ConversationManager
  - 读入用户输入 → ConversationManager → 打印酒狐回复
  - 支持 `/quit`、`/memory`（查看记忆）、`/reset`（清空短期上下文）命令
  - CLI 覆盖参数：`--model`、`--embedder`、`--lora`、`--db`、`--system-prompt`、`--n-ctx`、`--n-batch`、`--n-threads`、`--temp`、`--top-k`、`--top-p`、`--repeat-penalty`、`--max-tokens`、`--flash-attn`、`--kv-dtype`、`--thinking`、`--no-lora`
  - `--gen-config <path>`：将当前配置（含默认值）序列化为 JSON 写入指定路径并退出
- [x] Windows 下 UTF-8 控制台输出验证
- [x] 流式输出 + perf 数据显示（tokens, tok/s, prefill ms, flash_attn, kv_dtype）

**Phase 1 验收**：
- [x] Windows x64 上能文本对话
- [x] 流式输出正常，`<think>` 块被完全消除（通过 `enable_thinking=false` 从源头禁止）
- [x] emotion 标签正确解析并从输出中移除
- [x] 性能评估：**Q8_0 量化 17.3 tok/s**（对比 FP16 9.0 tok/s，提速 ~92%），KV cache 前缀复用（INCREMENTAL）使 prefill 从 ~7s 降至 ~450ms
- [x] winefox.exe = 4.5 MB，端到端加载+对话+退出流程跑通
- [ ] 告诉酒狐「我叫XXX」，后续对话能正确称呼（待多轮测试验证）
- [ ] 持续对话 20+ 轮后触发记忆整理，能在后续对话中召回（待多轮测试验证）
- [ ] Linux x64 CLI 同样可用（待验证）

---

## Phase 2：语音前端（实时语音对话）

> 目标：实时语音对话，P50 首字延迟 ≤ 2.5s。**本阶段不集成 AEC**，默认推荐耳机。

### 2.1 音频 IO 抽象

- [ ] `src/platform/audio/audio_io.h`：抽象接口（对齐 [6.6 节](./PLAN.md#L665-L636)）
  - `start_input()` / `start_output()`
  - `on_input_samples(cb)` / `write_output(samples)`
  - `is_headset_connected()`
- [ ] Windows WASAPI 实现 `src/platform/windows/wasapi_audio.h/cpp`
- [ ] Linux PulseAudio 实现 `src/platform/linux/pulse_audio.h/cpp`
- [ ] 采样率统一 16kHz 单声道 float32
- [ ] 单元测试：录音回放、设备枚举、耳机检测

### 2.2 VAD 服务

- [x] **voice-test VAD 基准测试已通过**（`voice-test/src/vad_test.cpp`）：
  - TEN-VAD-GGML.cpp 纯 C++17 移植（GGML 模型，零外部依赖，加载 4ms）
  - 段状态机：hop=256 (16ms), min_speech=250ms, min_silence=300ms, max_speech=30s
  - 性能：RTF=0.011 (~89x realtime)，峰值内存 5.5MB，avg 0.21ms/frame
  - 模型：`models/vad/ten-vad-ggml.bin`（GGML 格式，302KB）
  - 详见 `voice-test/README.md` Phase 1 VAD benchmark summary
- [x] **VadService 已迁移到 winefox_world**（`src/world/voice/vad_service.h/cpp`）：
  - 调用 ten-vad-ggml C API（`ten_vad_create(model_path)` / `ten_vad_process` 返回概率 / `ten_vad_reset`）
  - 自实现段状态机：hop=256, min_speech=250ms, min_silence=300ms
  - `on_segment` 流式回调语音段；`feed()` 概率 ≥ threshold 判定 speech
  - 模型路径来自 winefox.json `voice.vad.model_path`（WorldConfig 解析）
- [ ] 单元测试：喂入静音/语音混合音频，端点检测正确

### 2.3 ASR 服务

- [x] **voice-test ASR 基准测试已通过**（`voice-test/src/asr_test.cpp`）：
  - SenseVoice.cpp (ggml) + sense-voice-small-q4_k.gguf (181.86 MB)
  - 5 个中文测试句：avg RTF=0.269 (~3.7x realtime), avg CER=16.4%（含标点缺失误差）
  - 内部时序：encoder 占 88.5%，feature 5.8%，decoder 5.6%
  - 已知问题：「酒狐」→「九壶」同音字替换（无领域上下文时正常）
  - 详见 `voice-test/README.md` Phase 2 ASR benchmark summary
- [ ] `src/core/asr/asr_service.h/cpp`：封装 SenseVoice.cpp（ggml）
  - `on_partial_text(cb)` 流式部分识别
  - `on_final_text(cb)` 端点确认后完整文本
- [ ] 单元测试：喂入中文音频，识别结果与预期一致

### 2.4 TTS 服务（Kokoro）

- [x] **voice-test TTS 基准测试已通过**（`voice-test/src/tts_test.cpp`）：
  - Kokoro (onnxruntime) + kokoro-v1.1-zh.onnx + voices-v1.1-zh.bin (103 voices)
  - G2P 前端：kokoro.cpp (Jieba + PinyinFinder + ToneSandhi + EnG2P)
  - 短文本 (14 字)：RTF=2.394；长文本 (122 字)：RTF=2.199
  - 输出：24kHz mono float32 PCM
  - 性能结论：RTF > 1.0（CPU-bound），需 Phase 5 蒸馏到轻量学生模型（目标 RTF ≤ 0.3）
  - 详见 `voice-test/README.md` Phase 3 TTS benchmark summary
- [ ] `src/core/tts/tts_service.h/cpp`：封装 Kokoro ONNX（onnxruntime）
  - 调用 onnxruntime C API 推理 kokoro-v1.1-zh.onnx
  - G2P 前端使用 kokoro.cpp 的 ZHFrontend/ZHG2P/ToneSandhi/JiebaProcessor
  - `synthesize_stream(text, on_audio, on_done)` 流式合成
  - `stop()` 用户打断时调用
  - 按句切分（句号/感叹号/问号/逗号），前段播放时后段并行合成
- [ ] 单元测试：合成音频长度合理、stop 立即生效

### 2.5 流式管线调度器

- [x] **voice-test Stream 基准测试已通过**（`voice-test/src/stream_test.cpp`）：
  - VAD (ten-vad-ggml) + ASR (SenseVoice.cpp) 流式管线，段状态机驱动
  - 3/3 段正确识别："你好"、"今天天气真好"、"我们去散步吧"
  - E2E 延迟：min=0.534s, avg=0.671s, p50=0.701s, p95=0.770s, max=0.777s
  - 整体 RTF=0.1333 (7.50x realtime)，9.165s 音频 1.222s 处理完
  - E2E 延迟 = VAD 端点延迟 (min_silence=0.30s) + ASR 推理时间
  - 详见 `voice-test/README.md` Phase 4 Stream benchmark summary
- [ ] `src/core/pipeline/pipeline_scheduler.h/cpp`：协调 VAD/ASR/LLM/TTS
  - VAD speech → 触发打断（停止 TTS、停止 LLM）
  - ASR final → 推入 ConversationManager
  - LLM token → SentenceBuffer 按句切分 → 推入 TTS 队列
  - TTS 队列：当前句合成 + 下一句预合成
- [ ] `src/core/pipeline/sentence_buffer.h/cpp`：流式 token 按标点切句
- [ ] 单元测试：模拟流式输入，验证切句与队列行为

### 2.6 语音 CLI 入口

- [ ] 扩展 `src/app/cli_main.cpp`：增加 `--voice` 模式
  - 启动音频 IO、VAD、ASR、TTS
  - 用户说话 → ASR → LLM → TTS 播放
  - 支持 `Space` 键强制打断
- [ ] 端到端延迟埋点：VAD 端点 → ASR final → LLM 首 token → TTS 首音块

**Phase 2 验收**：
- Windows x64 + 耳机模式下实时语音对话
- P50 首字延迟 ≤ 2.5s（埋点数据）
- 用户说话能打断酒狐
- Linux x64 同样可用（PulseAudio）

> **voice-test 基准测试结论**：VAD+ASR 流式管线 E2E 延迟 p50≈0.70s，
> 加上 LLM prefill (~0.45s 增量) + TTS 首块 (~0.4s)，预估对话首字延迟
> p50 ≈ 1.55s，接近 P50 ≤ 1.5s 目标。TTS 是当前瓶颈（RTF > 2），需
> Phase 5 蒸馏后才能达到生产级延迟。

---

## Phase 3：AEC 与品质优化

> 目标：扬声器模式无回声，P50 ≤ 1.5s。

### 3.1 AEC3 集成

- [ ] `third_party/webrtc-apm/`：完成 AEC3 模块提取与 CMake 封装
- [ ] `src/core/aec/aec_service.h/cpp`：封装 AEC3
  - `process(near_end, far_end) -> cleaned`
  - 远端参考队列管理（TTS 输出同步拷贝）
- [ ] AudioIo 集成：扬声器模式时在 VAD 前插入 AEC 处理
- [ ] 耳机检测自动跳过 AEC（对齐 [0.7](./PLAN.md#L76-L80)）
- [ ] AEC 校准向导：播放 sweep tone 测量环路延迟
- [ ] 单元测试：合成回声音频，AEC 后回声抑制 ≥ 20dB

### 3.2 VAD 端点延迟动态调整

- [ ] VAD 参数自适应：检测到长对话时缩短 min_silence_duration（更快响应）
- [ ] 语义感知端点（实验性）：ASR partial 文本以句号结尾时提前触发 final

### 3.3 LLM prefill 优化

- [x] **KV cache 前缀复用（INCREMENTAL）**（已在 Phase 1 实现，对齐 [6.3](./PLAN.md#L757)）：
  - recall 作为 tool 消息，system prompt 保持静态
  - `llama_memory_seq_rm` 保留公共前缀 KV cache，仅 prefill 后缀
  - 生成后移除 `<think>` bridge tokens，使 `cached_tokens_` 与下一轮 prompt 完全对齐
  - 实测：prefill 从 ~7s（FIRST）降至 ~450ms（INCREMENTAL）
- [ ] ASR 进行中预热 LLM：加载 system prompt 到 KV cache（Phase 2 用）
- [ ] 性能对比：全量 prefill vs 前缀复用，记录延迟数据（已有初步数据）

### 3.4 TTS 首块流式优化

- [ ] TTS 首句优先合成（不等 LLM 整句完成）
- [ ] TTS 音频块大小调优（256 / 512 / 1024 samples 对比）
- [ ] 音频播放缓冲区最小化（降低播放延迟）

### 3.5 性能基准

- [ ] `tests/bench/latency_bench.cpp`：端到端延迟基准测试
- [ ] `tests/bench/throughput_bench.cpp`：LLM tok/s 基准
- [ ] `tests/bench/memory_bench.cpp`：内存占用采样
- [ ] 生成性能报告：Windows x64 + Linux x64 各项指标

**Phase 3 验收**：
- 扬声器模式无回声（用户主观评估）
- P50 首字延迟 ≤ 1.5s（基准测试数据）
- 连续对话 1 小时内存稳定（无泄漏）
- 报告归档到 `docs/perf-report-phase3.md`

---

## Phase 3.5：Vulkan 渲染技术验证（vulkan-test 沙盒）

> 目标：在独立沙盒项目中验证 Vulkan 3D 渲染管线，为 Phase 4 GUI 铺路。
> **不依赖** winefox_core.dll 或任何语音组件，仅 SDL3 + Vulkan SDK + glm。

### 3.5.1 基础渲染管线

- [x] `vulkan-test/CMakeLists.txt`：独立构建配置（SDL3 静态链接、Vulkan SDK、glslc 着色器编译）
- [x] `vulkan-test/src/vulkan_app.h/cpp`：完整 Vulkan 渲染管线
  - Instance + debug messenger → surface → physical device → logical device
  - Swapchain → render pass (color + depth) → graphics pipeline
  - Framebuffers → vertex/index buffers (staging) → command buffers
  - Sync objects (MAX_FRAMES_IN_FLIGHT = 2)
- [x] `vulkan-test/shaders/cube.vert` + `cube.frag`：MVP push constant + 环境光/漫反射光照
- [x] 深度缓冲（D32_SFLOAT）+ 深度测试
- [x] **修复崩溃 bug**：`VkPipelineShaderStageCreateInfo stages[2]` 未零初始化导致 `pNext` 野指针，`vkCreateGraphicsPipelines` 访问违规（退出码 0xC0000005）

### 3.5.2 相机与交互

- [x] `vulkan-test/src/camera.h`：FPS 相机（glm 球坐标 → 欧拉角）
  - Y 轴翻转（Vulkan NDC Y 朝下，`p[1][1] *= -1`）
  - 俯仰角 clamp（±1.55 rad，避免万向锁翻转）
  - `forward_flat()`：XZ 平面投影前向（WSAD 走路不受俯仰影响）
- [x] 鼠标捕获：`SDL_SetWindowRelativeMouseMode`（指针锁定窗口中心，ESC 释放/退出）
- [x] WASD 移动 + QE 升降 + 滚轮调速（帧率无关，`dt` 驱动）
- [x] FPS 显示（标题栏，每秒更新）

### 3.5.3 封闭房间场景

- [x] 20×20×20 封闭房间几何体（法线朝内，6 面不同颜色）
- [x] 索引绕序反转（从内部看为 CCW 正面）
- [x] 背面剔除 `VK_CULL_MODE_BACK_BIT`（为后期复杂模型性能准备）
- [x] 垂直同步 `VK_PRESENT_MODE_FIFO_KHR`（防止撕裂）

**Phase 3.5 验收**：
- [x] Vulkan 完整管线初始化无崩溃（AMD Radeon Vega 8 Graphics）
- [x] 封闭房间六面彩色墙正确渲染，背面剔除生效
- [x] FPS 相机自由漫游（WASD + 鼠标视角 + QE 升降）
- [x] V-Sync 60FPS 稳定，标题栏实时显示帧率/坐标/速度

---

## Phase 4：GUI 与酒狐形象（Windows）

> 目标：图形界面 + 酒狐形象 + 情感驱动动画。分三步实现（对齐 [11.2.3](./PLAN.md#L866-L873)）。

### 4.1 GLES3 渲染基础

- [ ] `third_party/`：接入 ANGLE（Windows GLES3 兜底）
- [ ] `src/platform/windows/win32_window.h/cpp`：Win32 窗口创建
- [ ] `src/ui/renderer/gles3_context.h/cpp`：GLES3 上下文初始化
- [ ] `src/ui/renderer/shader.h/cpp`：基础 vertex/fragment shader（立方体 + 贴图）
- [ ] `src/ui/renderer/texture.h/cpp`：PNG 纹理加载（stb_image）
- [ ] 验证：渲染一个旋转的立方体 + 贴图

### 4.2 bbmodel 渲染 - 步骤 1：静态模型

- [ ] `src/ui/bbmodel/bbmodel_parser.h/cpp`：JSON 解析（cubes、bones、textures、uv）
- [ ] `src/ui/bbmodel/bone.h/cpp`：骨骼树构建
- [ ] `src/ui/bbmodel/bbmodel_renderer.h/cpp`：静态渲染（对齐 [3.6](./PLAN.md#L478-L491)）
  - `load(bbmodel_path)`
  - `render(ctx)`
- [ ] 验证：加载 `WineFoxModel/模型文件/人物模型/大正女仆酒狐.bbmodel`，正确渲染静态形象

### 4.3 bbmodel 渲染 - 步骤 2：骨骼动画

- [ ] `src/ui/bbmodel/animation.h/cpp`：关键帧动画解析与插值（位置、旋转、缩放）
- [ ] 预设动画：idle（呼吸）、joy（摇尾巴）、neutral、surprise、sadness、anger、fear
- [ ] 从 bbmodel 自带动画提取，按 emotion 标签映射（对齐 [3.6 表格](./PLAN.md#L493-L487)）
- [ ] `BbModelRenderer::update(dt)` 骨骼动画 tick
- [ ] 验证：切换 emotion 时动画平滑过渡

### 4.4 bbmodel 渲染 - 步骤 3：表情与交互

- [ ] 表情切换：贴图 UV 切换或顶点形变（眼、嘴）
- [ ] 鼠标点击/拖拽酒狐不同部位触发反应（摸头→joy，摸尾巴→anger）
- [ ] 长按酒狐弹出菜单（设置、记忆查看、重置对话）

### 4.5 EmotionDriver

- [ ] `src/core/emotion/emotion_driver.h/cpp`：解析 LLM 输出 `[emotion]` 标签
- [ ] 分发到 BbModelRenderer.set_emotion()
- [ ] 情感平滑过渡（避免硬切）

### 4.6 GUI 主循环与设置面板

- [ ] `src/app/gui_main.cpp`：GUI 入口，整合音频管线 + 渲染
- [ ] 设置面板：模型选择、AEC 开关、性能档位、记忆管理
- [ ] 系统托盘：最小化到托盘、开机自启

**Phase 4 验收**：
- Windows GUI 模式完整可用
- 6 种情感对应动画正确切换
- 可点击交互酒狐
- 设置面板功能完整

---

## Phase 5：TTS 自训模型替换

> 目标：用酒狐音色 VITS-Tiny 替换临时 TTS。需 GPU 离线训练。

### 5.1 CosyVoice 数据集生成（离线，Colab GPU）

- [ ] `tts-training/cosyvoice_dataset/`：数据生成脚本目录
- [ ] 准备酒狐日语音色参考样本（从 WineFoxModel 语料库提取）
- [ ] 文本语料整理（≥ 5 小时）：
  - 从 `llm-finetune/winefox_sft.jsonl` 提取所有 assistant 回复
  - 扩充日常对话语料（朗诵、故事、闲聊）
- [ ] `tts-training/cosyvoice_dataset/generate.py`：CosyVoice 跨语言克隆
  - 输入：日语参考 + 中文文本
  - 输出：中文酒狐音色音频
- [ ] 人工抽检音色一致性（对齐 [11.2.5](./PLAN.md#L884-L891)），不达标则调整参考音频
- [ ] 数据清洗：剔除发音错误/不自然样本，标注 emotion 标签
- [ ] 切分 train/val (95/5)

### 5.2 Kokoro 蒸馏训练（离线，Colab GPU）

- [ ] `tts-training/kokoro_distill/`：训练代码目录
- [ ] Kokoro-82M 学生模型加载（或定义 Kokoro-Tiny 20-40M）
- [ ] 蒸馏损失：mel-spectrogram L1 + KL + adversarial + **speaker embedding 一致性**
- [ ] `tts-training/kokoro_distill/train.py`：训练脚本
- [ ] 训练并评估：自然度、音色相似度、RTF
- [ ] 保留官方 Kokoro-82M 模型作为 fallback（对齐 [11.2.5](./PLAN.md#L884-L891)）

### 5.3 ONNX 导出与集成

- [ ] `tts-training/kokoro_distill/export_onnx.py`：导出 ONNX（INT8 量化）
- [ ] 验证 ONNX 输出与 PyTorch 一致
- [ ] 替换 TtsService 模型路径
- [ ] 端到端测试：语音对话使用酒狐音色
- [ ] 性能测试：CPU RTF ≤ 0.3

**Phase 5 验收**：
- TTS 输出酒狐音色，自然度可接受
- CPU RTF ≤ 0.3
- fallback 机制可用（自训模型加载失败时回退）

---

## Phase 6：跨平台扩展

> 目标：扩展到所有 8 个平台组合。按 [11.2.1](./PLAN.md#L848-L855) 优先级排序。

### 6.1 Linux 平台扩展

- [ ] Linux x64 验证（Phase 1-5 已支持，补全测试）
- [ ] Linux x86：32 位编译适配，模型量化调整
- [ ] Linux arm64（树莓派 4）：交叉编译、PulseAudio 验证、性能基准
- [ ] Linux arm32（树莓派 2/3）：
  - 启用 `WINEFOX_MINIMAL` 编译宏（对齐 [11.2.6](./PLAN.md#L893-L904)）
  - LLM 用 Q3_K_S 量化
  - 禁用 Embedding，长期记忆改用 SQLite FTS5 全文检索
  - 禁用 TTS 实时合成、禁用 AEC
  - ALSA 音频 IO（PulseAudio 过重）
  - 内存目标 ≤ 700MB

### 6.2 Windows 平台扩展

- [ ] Windows x86：32 位编译适配
- [ ] Windows arm64：交叉编译、GLES3/ANGLE 验证

### 6.3 Android 平台

- [ ] Android NDK 项目结构搭建（`src/platform/android/`）
- [ ] NativeActivity + GLES3 窗口
- [ ] AAudio 音频 IO 实现
- [ ] bbmodel 渲染器 Android 适配
- [ ] APK 打包脚本（CMake + Gradle）
- [ ] arm64-v8a 设备测试

### 6.4 CI/CD 多平台构建

- [ ] GitHub Actions 矩阵：8 平台组合
- [ ] 第三方依赖预编译缓存
- [ ] Artifact 上传（每平台独立包）

**Phase 6 验收**：
- 所有 8 平台可构建
- Windows x64/x86/arm64、Linux x64/arm64、Android arm64 功能完整
- Linux x86/arm32、Windows arm32（如有）极简模式可用
- CI 全平台绿

---

## Phase 7：打磨与发布

### 7.1 安装包

- [ ] Windows MSI 安装包（WiX）含模型下载器
- [ ] Android APK 签名发布
- [ ] Linux AppImage（x64/arm64）+ deb/rpm（可选）

### 7.2 模型下载器

- [ ] 首次启动按需下载模型（HuggingFace / 自建镜像）
- [ ] 断点续传、校验、失败重试
- [ ] 模型版本管理

### 7.3 文档与引导

- [ ] 用户手册（快速开始、FAQ、故障排查）
- [ ] 首次启动引导（耳机推荐、AEC 校准、性能档位选择）
- [ ] 开发者文档（构建、测试、贡献）

### 7.4 健壮性

- [ ] 错误处理：模型加载失败、磁盘满、内存不足的友好提示
- [ ] 崩溃恢复：异常退出后对话状态恢复
- [ ] 自动更新检查（可选）

### 7.5 发布前测试

- [ ] 长时间稳定性测试（24 小时连续对话）
- [ ] 多用户场景测试（不同 profile 切换）
- [ ] 极端输入测试（空输入、超长输入、特殊字符）
- [ ] 性能回归测试

**Phase 7 验收**：
- 安装包可正常分发
- 首次启动引导完整
- 24 小时稳定运行无崩溃
- 文档齐全

---

## 持续任务（贯穿所有 Phase）

- [ ] 每完成一个模块补充单元测试（覆盖率 ≥ 60%）
- [ ] 每完成一个 Phase 更新 `docs/perf-report-phaseN.md`
- [ ] 维护 `CHANGELOG.md` 记录变更
- [ ] 定期 review 依赖版本与安全更新
- [ ] 监控 LoRA 训练数据集扩充进度（目标 ≥ 2000 条）

---

## 阻塞与待决问题

> 记录执行过程中发现的需要决策的问题。

- [x] **[已废弃]** llama.cpp 多 sequence KV cache 方案：原 seq 0/seq 1 分段方案已废弃，改用前缀复用方案（recall 作为 tool 消息），无需多 sequence API
- [x] **[已测量]** LoRA 热加载/卸载延迟：attach 约 ms 级（82.6MB LoRA），可接受，不影响记忆整理流程（对齐 [11.2.8](./PLAN.md#L926-L932)）
- [ ] **[待决策]** CosyVoice 跨语言克隆音色是否达标，不达标时的备选方案
- [x] **[已验证]** Qwen3.5-0.8B Q8_0 在当前开发机达到 17.3 tok/s（对比 FP16 9.0 tok/s，提速 ~92%），CPU_REPACK 优化生效，超过 8 tok/s 目标
- [x] **[已解决]** VAD 后端迁移：TEN-VAD 从 ONNX 推理（onnxruntime 嵌入 DLL，加载数百 ms）迁移到用户自研纯 C++17 版（TEN-VAD-GGML.cpp，GGML 模型，加载 4ms，峰值内存 5.5MB，RTF 0.011）。旧 ONNX 预编译库（`voice-test/third_party/ten-vad/`）与 ONNX 模型（`models/vad/ten-vad.onnx`）已删除，voice-test 与新 VadService 均改用 `ten_vad_create(model_path)` / `ten_vad_process` 概率 API
- [ ] **[待决策]** Windows GLES3 是用原生驱动还是强制 ANGLE
- [x] **[已解决]** Qwen3.5 `<think>` 块过滤：通过 `LLAMA_BUILD_COMMON=ON` + `common_chat_templates_apply` 设置 `enable_thinking=false`（等价 `llama-cli -rea off`），从 jinja 模板源头禁止思考模式，模型不再生成 `<think>` 块。无需在输出流中过滤
- [x] **[已解决]** Q4_0 量化出现严重幻觉：测试期改用 FP16，正式构建将切换 Q8_0（弃用 Q4_0）
- [x] **[已解决]** nlohmann/json 版本冲突：统一使用 llama.cpp vendor 中的 3.12.0 版本
- [x] **[已解决]** llama.cpp 日志噪音：Release 构建中通过 `llama_log_set` 设置静默回调，过滤所有 INFO/DEBUG 日志（仅保留 ERROR），Debug 构建保持默认（stderr 输出）
- [x] **[已解决]** prefill 首字延迟：Q8_0 量化（CPU_REPACK 优化生效）+ KV cache 前缀复用（recall 作为 tool 消息 + `<think>` bridge 移除），prefill 从 ~7s（FIRST）降至 ~450ms（INCREMENTAL）

---

## 执行顺序建议

```
Phase 0 (骨架)
    ↓
Phase 1 (文本对话+记忆)  ← 第一个可演示里程碑
    ↓
Phase 2 (语音前端)        ← 第二个可演示里程碑
    ↓
Phase 3 (AEC+优化)        ← 品质里程碑
    ↓
Phase 4 (GUI)  ←─ 可与 Phase 5 并行
    ↓
Phase 5 (TTS 自训)  ←─ 依赖离线 GPU，可提前启动数据生成
    ↓
Phase 6 (跨平台)
    ↓
Phase 7 (发布)
```

**并行机会**：
- Phase 5.1（CosyVoice 数据生成）可在 Phase 2 期间并行启动（Colab 异步训练）
- Phase 4（GUI）与 Phase 5（TTS）可并行
- LoRA 数据集扩充可与任何 Phase 并行进行
