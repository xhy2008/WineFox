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

