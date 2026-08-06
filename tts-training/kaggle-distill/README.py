"""Kokoro 音色蒸馏 - Kaggle 使用说明

== 快速开始 ==

1. 上传数据集到 Kaggle Dataset：
   - 将整个 ref3_tts_dataset_routed_500/ 目录（含 wavs/ 与 manifest.jsonl）
     上传为 Kaggle Dataset，命名为 "winefox-dataset"
   - 注意：manifest.jsonl 是训练所必需的 —— 训练文本取自其中的真实文本
     （含标点，与每条音频一一对应）。若只上传 wavs/，训练会回退到
     固定文本，模型将学不到标点静音特征。

2. 创建 Kaggle Notebook：
   - Settings → Internet: ON（需要下载 WavLM 和 Kokoro 模型）
   - Settings → Accelerator: GPU T4 x2（推荐）或 P100
   - Add Data → 添加 "winefox-dataset"
   - 上传 winefox_distill.ipynb 并直接运行全部 cell


== 方案原理 ==

梯度反演（本方案）：
  目标音频 → ECAPA-TDNN → 目标说话人嵌入 (192维)     → 约束 acoustic (前128维)
  官方音色 → 同文本前向 → 参考韵律 (duration, F0, N)     → 约束 prosody (后128维)
                                        ↓ 损失
  style_vec (256维, 可训练) → Kokoro decoder → 合成音频 → ECAPA-TDNN → 合成嵌入
                                        ↓
                                  反向传播更新 style_vec

关键点：
  - ECAPA-TDNN 是 speaker verification 专用模型（spkrec-ecapa-voxceleb）。
    数据集由 qwen3-tts 生成（与 Kokoro 不同源），WavLM 统计量无法区分
    二者音色，故音色匹配改用 ECAPA embedding cosine（可微，梯度流回 decoder）
  - 频谱频带损失（spectral_band_profile）：log 频率分桶的能量占比对比，
    只统计有声帧、归一化与时长无关。把高频占比拉向目标 → 尖细/清脆质感
    （仅靠 ECAPA 会出现"音色像但成熟/机械"）
  - acoustic norm 正则：把 acoustic 锁在初始分布水平，防止涨出 decoder
    分布（norm 过大 → AdaIN 异常 → 大舌头/电流音）
  - 参考韵律来自官方音色对同一文本的前向输出（"这段文本该怎么读"
    的参考节奏）
    → 时间维度韵律 loss：节奏（音素时长）+ 语调（逐音素 F0 轮廓，
      目标 = 官方参考轮廓 × F0_TARGET_RATIO，形状与绝对音高同时
      约束到 TARGET_F0_HZ）+ 能量（逐音素 N，token 级，
      标点处自然静音）
  - differentiable_forward 因 RNN backward 需要 model.train()，但 train 模式
    会激活 dropout 导致 F0/N 随机采样（loss 带噪声基线、早停被干扰），
    已将所有 Dropout 的 p 置 0，保证输出确定性且梯度可用
  - 梯度隔离：speaker loss 只更新 acoustic，韵律 loss 天然只更新 prosody
  - 256 维 vs 82M 参数，反问题而非学习问题
  - 3-30 秒参考音频即可（更多只是降低统计噪声）


== 文件说明 ==

winefox_distill.ipynb  梯度反演训练（Kaggle notebook，核心）
export_voice.py        导出 .pt → voices.bin（C++ 推理引擎格式）
verify_voice.py        合成验证样本 + ECAPA 相似度评估
requirements.txt       Kaggle 依赖


== ipynb 参数调优 ==

在 winefox_distill.ipynb 的 Cell 15/16 中调整：

INIT_VOICE       初始 voicepack（选最接近目标音色的预设）
                 中文预设: zf_001 ~ zf_050（ipynb 默认 zf_xiaoyi）
                 推荐尝试: zf_001 (女声), zf_019, zf_029

TARGET_F0_HZ     目标 F0（F0 均值 loss 的目标，349.8 = 数据集 F0 mean）

LR               学习率（prosody 用 2e-4；论文最优）

LR_AC           acoustic 学习率（1e-3）—— style 拆为 acoustic/prosody
                 两个独立参数。spk 梯度路径长（decoder -> audio ->
                 resample -> ECAPA-TDNN），2e-4 下 1250 步几乎不动，
                 故 acoustic 单独用大步长。若 agrad 非零但 spk 仍不
                 降，可加大 LR_AC（2e-3）；若 agrad≈0 说明梯度断在
                 ECAPA 内部（需换可微特征提取路径）。

MAX_STEPS        最大步数（通常 200-1000 步收敛）

THRESHOLD        停止阈值（0.15 论文推荐）
                 太低：训练不停止
                 太高：过早停止，音色未充分优化

W_DUR / W_F0 / W_N   韵律 loss 权重（节奏/语调/能量）
                 语速或停顿异常时优先调 W_DUR
                 W_F0 的目标是缩放后的参考轮廓（含绝对音高），若 F0
                 (f0m) 未达 F0_TARGET_RATIO，可增大 W_F0（如 0.05）

W_BAND           频谱频带损失权重（5.0）—— 音色质感（尖细度）的主要
                 驱动器。若音色仍"成熟/机械"，增大到 8-15；若出现
                 高频刺耳/齿音过重，减小到 2-3。
                 目标频带分布用 N_REF_BAND=100 条参考统计（更多样本
                 → 平均频谱质感更稳定，20 条受单句内容影响大）。

W_NORM           acoustic norm 正则权重（0.25）—— 防止 acoustic 涨出
                 分布。若合成仍有大舌头/电流音或 norm 仍偏高，继续增大
                 （0.3-0.5）；若音色匹配（spk）损失过大，适当减小。

注：N 能量约束使用纯 token 级（帧级对齐对时长误差敏感，且官方 N
    帧级参考不是目标音色的，实测会导致 n loss 不降反升、prosody 震荡）。

EARLY_STOP_PATIENCE / EMA_ALPHA
                 早停用 EMA 平滑后的 total loss 判定（spk 跨文本波动
                 约 ±0.15，直接比较会被波动干扰、误停）。若 spk 波动
                 大导致早停仍过早，减小 EMA_ALPHA 或加大 patience。


== 失败排查 ==

问题: loss 不下降
原因: 梯度被阻断（检查 differentiable_forward 是否正确绕过 @torch.no_grad）
解决: 确保 style.requires_grad_(True)，且不使用 torch.no_grad() 上下文

问题: spk loss 不下降 / agrad 恒为 0
原因: 梯度隔离误清了 acoustic 梯度。曾错误地在 prosody_loss.backward()
      后执行 acoustic.grad.zero_()（或旧版 style.grad[:, :128]=0），
      把 spk loss 给 acoustic 的梯度一起清掉了 → acoustic 从不更新。
解决: prosody_loss（duration/F0/N）只依赖 s=ref_s[:,128:]，不经过
      acoustic，backward 后无需清 acoustic 梯度，直接 opt.step() 即可。
      只需在 spk_loss.backward() 后清 prosody.grad（spk 经 audio 依赖 F0/N）。

问题: 合成音频是杂音
原因: style vector 漂出 decoder 训练分布
解决: 降低 lr 到 1e-4，减少 steps，或换更接近的 init_voice

问题: 音色不够像
原因: 参考音频太少或有噪声
解决: 增加 N_REF 到 20-30，确保参考音频已去静音

问题: 标点位置出现奇怪音素
原因: 训练文本与音频不对应，韵律 loss 学不到标点静音
解决: 确认上传了 manifest.jsonl（训练文本取自其中），而非回退固定文本

问题: OOM (显存不足)
原因: ECAPA-TDNN + Kokoro decoder 同时在 GPU
解决: 减少 batch（本方案 batch=1），或用 gradient checkpointing
"""
