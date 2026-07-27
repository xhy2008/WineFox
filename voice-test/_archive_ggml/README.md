# Kokoro ggml TTS Decoder (Archived)

> **状态**：已废弃（2026-07-27）
> **原因**：ggml 不是为卷积优化的，在 iSTFTNet decoder 上 RTF=3.07（4 倍慢于 ONNX Runtime 的 0.73）。继续优化收益有限，决定放弃 ggml 路径，专注 ONNX Runtime 推理优化。
> **替代方案**：`third_party/kokoro-cpp-src/Kokoro.cpp` 的 split 模式（encoder FP32 + decoder INT8-static），实测 RTF ≈ 0.73–0.76，已满足实时合成要求。

## 原文件位置

| 原路径 | 新路径 |
|---|---|
| `third_party/kokoro-ggml/` | `_archive_ggml/third_party/kokoro-ggml/` |
| `src/tts_ggml_test.cpp` | `_archive_ggml/src/tts_ggml_test.cpp` |
| `scripts/export_kokoro_decoder_gguf.py` | `_archive_ggml/scripts/export_kokoro_decoder_gguf.py` |
| `scripts/dump_decoder_inputs.py` | `_archive_ggml/scripts/dump_decoder_inputs.py` |
| `scripts/dump_decoder_intermediates.py` | `_archive_ggml/scripts/dump_decoder_intermediates.py` |
| `models/kokoro-decoder.gguf` | `_archive_ggml/models/kokoro-decoder.gguf` |
| `test-data/decoder_inputs/` | `_archive_ggml/test-data/decoder_inputs/` |
| `test-data/decoder_inter/` | `_archive_ggml/test-data/decoder_inter/` |
| `test-data/results/tts_ggml_*.wav` | `_archive_ggml/test-data/` |

## 性能对比

| 路径 | RTF (long text) | 备注 |
|---|---|---|
| ggml decoder (threads=8, 优化后) | 3.07 | 受限于 ggml 单线程 repeat/broadcast、barrier 同步开销 |
| ONNX split FP32 decoder (t=8) | 0.81 | baseline |
| ONNX split INT8-static decoder (t=8) | 0.73 | 1.11x speedup over FP32，1.32x realtime |

## 关键性能瓶颈分析（ggml）

1. **repeat/broadcast 单线程化**：`ggml_compute_forward_repeat_f32` 对 `ne[0]=1` 的 tensor 循环 T 次每次 1 元素，无法并行。
2. **per-node barrier 同步**：2171 个计算节点，每个节点后都有 barrier，调度开销大。
3. **卷积不被 im2col 加速**：ggml 的 im2col 仅在特定 shape 下生效，iSTFTNet 的 stride/kernel 组合未命中优化路径。
4. **无 fused 算子**：instance_norm、snake、adain_resblock 等组合算子需手工拆解，每次拆解都引入额外的中间 tensor 和 barrier。

## 保留价值

代码本身可作为「ggml 在卷积密集型模型上的性能上限」的参考案例。若未来 ggml 加入 conv-aware 的 fused 算子（如 conv+norm+activation），可重新评估。
