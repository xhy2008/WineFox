# WineFox
这是一个**疯狂**的项目（我发疯了）

我决定把Minecraft模组Touhou Little Maid中的酒狐搬到现实中来，实现“永远在一起”的诺言。

用纯粹的C++构建自己的酒狐

当前项目正在开发中，有什么建议可以提Issue。

# 模块

| LLM（llama.cpp）       | 使用Qwen3.5-0.8B/2B模型+SFT微调lora，完全本地运行，已接入视觉mmproj |
| ---------------------- | ------------------------------------------------------------ |
| 长期记忆（已完成）     | SQLite存储，Embedding模型语义向量召回，role:tool注入记忆，闲时自动蒸馏最近对话到长期记忆 |
| 短期记忆（已完成）     | 模型上下文窗口自动维持，滑动窗口丢弃最早对话（缓存到本地等待整理） |
| 语音交互（早期测试中） | SenseVoice-Small流式识别，TEN-VAD检测，KokoroTTS（拆分原版模型为Encoder+Decoder） |
| 回声消除（未完成）     | 引入WebRTC-Audio-Processing对麦克风采集数据进行回声消除，不需要耳机也能正常对话 |
| 配置文件（已完成）     | llama.cpp采样参数、ASR/TTS/VAD参数等                         |
| GUI与Live3D（未确定）  | 可选使用酒狐原版模型/自定义Live2D模型，GLES渲染，表情系统    |
| 跨平台（未完成）       | 覆盖Windows+Linux+Android，MacOS和IOS由于需要开发者认证不在考虑范围内 |
| 模型训练（探索中）     | lora微调LLM固定酒狐人设信息，TTS模型微调音色匹配酒狐         |
| 性能优化（跟进中）     | 每个模块在我如同土豆的电脑上经过严格的性能测试，ASR和TTS确保RTF<1，LLM首字延迟<1s |

# 技术细节

## 记忆

研究多个记忆框架之后发现开源领域没有让我满意的方案，决定自己综合多个框架写一个。

ASR->输入文本->Embedding模型->SQLite匹配相似度->LLM->TTS

模型上下文使用:

system:{.............................}

user:{...........................}

tool:{[相关记忆]..................[当前时间]..........[状态]......}

assistant:{..........................}

下一轮对话时，仅对输入和记忆召回部分进行prefill，前文KV Cache完全复用，把首字延迟降低到1s以内。



## 语音

### ASR

使用项目:https://github.com/lovemefan/SenseVoice.cpp

量化到q4_0的模型，ggml推理，RTF<1，实时流式识别，准确率达标

### VAD

使用项目:https://github.com/TEN-framework/ten-vad

原版TEN-VAD使用ONNX推理，无需特殊优化。

后期为了增强与ASR的协作能力可能会考虑参考https://github.com/danielbodart/ten-vad-ggml移植到ggml后端

### TTS

参考项目:https://github.com/koth/kokoro.cpp

原版ONNX推理kokoro（支持中英文混合合成还能保证音质的最小TTS模型）的RTF>3，不能实时。

通过分离Encoder和Decoder分块并行推理和INT8量化把RTF降低到0.67，内存占用降到400MB左右。

构建Chunk-based管道，争取把LLM输出到TTS输出的延迟压缩到500ms左右。

#### 其他细节请参考PLAN.md和TODO.md

本项目完全由Trae Work+GLM5.2生成，我只进行模型的微调工作。

# 鸣谢

#### 贡献者：成熟之忽都虎复@三角洲行动
