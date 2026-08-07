# -*- coding: utf-8 -*-
"""删除 train_kaggle.ipynb 中基座模型转换 cell（本地已有基座），并同步更新汇总与说明。"""
import json

path = r"e:\winefox\llm-finetune\train_kaggle.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# 校验 index 8 是 Cell 9（基座转换）
assert "# Cell 9" in "".join(cells[8]["source"]), "Cell 9 not found at index 8"
del cells[8]  # 删除基座模型转换 cell

# Cell 10（现 index 8）汇总更新：只保留 LoRA 产物
cells[8]["source"] = [
    "# Cell 9: 汇总产物\n",
    "print('\\n=== 全部产出（/kaggle/working，请在右侧 Output 面板下载） ===')\n",
    "print(f'LoRA GGUF（f16，下载后本地量化）: {OUTPUT_DIR}/winefox-lora-f16.gguf')\n",
    "print(f'LoRA adapter 原始文件（HF 格式）: {OUTPUT_DIR}/winefox_lora_adapter/')\n",
    "print('\\n本地量化示例（需本机已构建 llama.cpp 的 llama-quantize）：')\n",
    "print('  llama-quantize winefox-lora-f16.gguf winefox-lora-Q5_K_M.gguf Q5_K_M')\n",
    "print('  llama-quantize winefox-lora-f16.gguf winefox-lora-Q4_K_M.gguf Q4_K_M')\n",
]

# 头部 markdown 更新：产出只保留 LoRA，量化命令只保留 LoRA
cells[0]["source"] = [
    "# WineFox LoRA 微调（Kaggle）\n",
    "\n",
    "基于 Qwen3.5-0.8B 的酒狐人设 LoRA 微调（基座与 LoRA 分离）。\n",
    "\n",
    "**架构**：\n",
    "- 基座模型（Qwen3.5-0.8B）与 LoRA adapter **分开保存**\n",
    "- 对话时启用 LoRA（带酒狐人设），蒸馏时禁用 LoRA（基座模型）\n",
    "\n",
    "**使用步骤**：\n",
    "1. 新建 Kaggle Notebook，GPU 选择 T4 x2（免费）或 P100\n",
    "2. 右侧 `Add Dataset` → 上传 `winefox_dataset.jsonl`（本仓库 [llm-finetune/winefox_dataset.jsonl](https://github.com/) 已生成）\n",
    "3. 从上到下依次运行所有 Cell\n",
    "4. 训练完成后，到右侧 `Output` 标签下载 `/kaggle/working/` 下的 LoRA f16 GGUF\n",
    "\n",
    "**资源**：\n",
    "- 基座模型：`unsloth/Qwen3.5-0.8B`（HF 在线拉取，仅用于训练）\n",
    "- 数据集：`winefox_dataset.jsonl`（Kaggle Dataset 挂载到 `/kaggle/input/`）\n",
    "- GPU：T4 即可（0.8B + 4bit + LoRA 约需 6GB 显存）\n",
    "\n",
    "**产出**（保存到 `/kaggle/working/`）：\n",
    "- `winefox-lora-f16.gguf`（LoRA adapter，f16，下载后本地量化）\n",
    "- `winefox_lora_adapter/`（HF 格式 LoRA，可下载后在本地自行转换/量化）\n",
    "\n",
    "**量化（在本机执行，需要 llama.cpp 的 `llama-quantize`，配合本地已有基座模型）**：\n",
    "- `llama-quantize winefox-lora-f16.gguf winefox-lora-Q5_K_M.gguf Q5_K_M`\n",
    "- `llama-quantize winefox-lora-f16.gguf winefox-lora-Q4_K_M.gguf Q4_K_M`\n",
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("已删除基座转换 cell，更新完成。剩余 cells:", len(nb["cells"]))
