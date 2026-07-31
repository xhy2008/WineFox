# Colab Voice Style 蒸馏训练

## 上传

将整个 `colab-voice-train/` 目录上传到 Google Drive 或直接 zip 上传到 Colab。

## Colab 操作步骤

```python
# 1. 安装依赖（Colab 自带 torch 2.x+cu130，无需额外装 torch）
!pip install kokoro misaki

# 2. 如果上传了 zip，解压
!unzip colab-voice-train.zip -d /content/

# 3. 进入目录
%cd /content/colab-voice-train

# 4. 开始训练（500 样本 × 100 epochs）
!python train_colab.py --epochs 100 --lr 0.02

# 5. 验证训练结果
!python verify.py

# 6. 下载结果
# 从 output/winefox_style.pt 和 output/winefox_voices.npy 下载
```

## 目录结构

```
colab-voice-train/          # 只需上传这个目录
├── dataset/
│   ├── manifest.jsonl      # 500 条数据集索引
│   └── wavs/               # 500 个 wav 文件（~97MB）
├── train_colab.py          # 训练脚本
├── verify.py               # 验证脚本
└── README.md               # 本文件
```

## 训练参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 0.02 | 学习率 |
| `--init_voice` | zf_xiaobei | 初始音色（小贝） |
| `--limit` | None | 限制样本数（调试用） |
| `--resume` | None | 从 checkpoint 恢复 |

## 预期时间（T4 GPU）

- 500 样本 × 100 epochs
- 每条约 0.1s（GPU），每 epoch 约 50s
- 总计约 1.5 小时

## 输出

- `output/winefox_style.pt`：256 维 style 向量
- `output/winefox_voices.npy`：512×256 的 voice pack（兼容 Kokoro Python）
- `output/style_epoch{N}.pt`：每 10 轮的 checkpoint
