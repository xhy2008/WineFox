# -*- coding: utf-8 -*-
"""修改 train_kaggle.ipynb：去掉 llama.cpp 编译与本地量化，只做 f16 转换。"""
import json

path = r"e:\winefox\llm-finetune\train_kaggle.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
src = "".join(cells[7]["source"])  # Cell 8 (index 7)
assert "# Cell 8" in src, "Cell 8 not found"

# ---- Cell 8: 去除量化部分 ----
cell8_new = [
    "# Cell 8: 转换 LoRA adapter 为 GGUF 格式（纯 Python，无需编译）\n",
    "# llama.cpp 仓库自带 convert_lora_to_gguf.py，浅克隆只为取脚本，不需要编译任何 C++ 工具。\n",
    "import subprocess\n",
    "\n",
    "LLAMA_CPP_DIR = os.path.join(OUTPUT_DIR, 'llama.cpp')\n",
    "if not os.path.exists(LLAMA_CPP_DIR):\n",
    "    subprocess.run(\n",
    "        ['git', 'clone', '--depth', '1', 'https://github.com/ggml-org/llama.cpp', LLAMA_CPP_DIR],\n",
    "        check=True,\n",
    "    )\n",
    "\n",
    "# 转换 LoRA adapter → GGUF（f16）\n",
    "subprocess.run([\n",
    "    'python', f'{LLAMA_CPP_DIR}/convert_lora_to_gguf.py',\n",
    "    '--outfile', f'{OUTPUT_DIR}/winefox-lora-f16.gguf',\n",
    "    lora_output_dir,\n",
    "], check=True)\n",
    "print('LoRA GGUF 转换完成（f16）: winefox-lora-f16.gguf')\n",
    "print('量化请在本地执行: llama-quantize winefox-lora-f16.gguf winefox-lora-Q5_K_M.gguf Q5_K_M')\n",
]
cells[7]["source"] = cell8_new

# ---- Cell 9: 基座模型转换（f16，不量化）----
cell9_new = [
    "# Cell 9: 转换基座模型 Qwen3.5-0.8B 为 GGUF 格式（纯 Python，无需编译）\n",
    "# convert_hf_to_gguf.py 依赖 gguf 包（Cell 2 已安装）。\n",
    "base_gguf = f'{OUTPUT_DIR}/qwen3.5-0.8b-f16.gguf'\n",
    "if not os.path.exists(base_gguf):\n",
    "    subprocess.run([\n",
    "        'python', f'{LLAMA_CPP_DIR}/convert_hf_to_gguf.py',\n",
    "        'unsloth/Qwen3.5-0.8B',\n",
    "        '--outfile', base_gguf,\n",
    "        '--outtype', 'f16',\n",
    "    ], check=True)\n",
    "    print('基座模型 GGUF 转换完成（f16）: qwen3.5-0.8b-f16.gguf')\n",
    "else:\n",
    "    print(f'基座模型已存在: {base_gguf}')\n",
    "print('量化请在本地执行: llama-quantize qwen3.5-0.8b-f16.gguf qwen3.5-0.8b-Q5_K_M.gguf Q5_K_M')\n",
]
cells[8]["source"] = cell9_new

# ---- Cell 10: 汇总 ----
cell10_new = [
    "# Cell 10: 汇总产物\n",
    "print('\\n=== 全部产出（/kaggle/working，请在右侧 Output 面板下载） ===')\n",
    "print(f'LoRA GGUF（f16，下载后本地量化）: {OUTPUT_DIR}/winefox-lora-f16.gguf')\n",
    "print(f'基座 GGUF（f16，下载后本地量化）: {OUTPUT_DIR}/qwen3.5-0.8b-f16.gguf')\n",
    "print(f'LoRA adapter 原始文件（HF 格式）: {OUTPUT_DIR}/winefox_lora_adapter/')\n",
    "print('\\n本地量化示例（需本机已构建 llama.cpp 的 llama-quantize）：')\n",
    "print('  llama-quantize winefox-lora-f16.gguf winefox-lora-Q5_K_M.gguf Q5_K_M')\n",
    "print('  llama-quantize winefox-lora-f16.gguf winefox-lora-Q4_K_M.gguf Q4_K_M')\n",
    "print('  llama-quantize qwen3.5-0.8b-f16.gguf qwen3.5-0.8b-Q5_K_M.gguf Q5_K_M')\n",
    "print('  llama-quantize qwen3.5-0.8b-f16.gguf qwen3.5-0.8b-Q4_K_M.gguf Q4_K_M')\n",
]
cells[9]["source"] = cell10_new

# ---- 头部 markdown：产出说明 ----
md_new = [
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
    "4. 训练完成后，到右侧 `Output` 标签下载 `/kaggle/working/` 下的 f16 GGUF\n",
    "\n",
    "**资源**：\n",
    "- 基座模型：`unsloth/Qwen3.5-0.8B`（HF 在线拉取）\n",
    "- 数据集：`winefox_dataset.jsonl`（Kaggle Dataset 挂载到 `/kaggle/input/`）\n",
    "- GPU：T4 即可（0.8B + 4bit + LoRA 约需 6GB 显存）\n",
    "\n",
    "**产出**（保存到 `/kaggle/working/`）：\n",
    "- `winefox-lora-f16.gguf`（LoRA adapter，f16，下载后本地量化）\n",
    "- `qwen3.5-0.8b-f16.gguf`（基座模型，f16，下载后本地量化）\n",
    "- `winefox_lora_adapter/`（HF 格式 LoRA，可下载后在本地自行转换/量化）\n",
    "\n",
    "**量化（在本机执行，需要 llama.cpp 的 `llama-quantize`）**：\n",
    "- `llama-quantize winefox-lora-f16.gguf winefox-lora-Q5_K_M.gguf Q5_K_M`\n",
    "- `llama-quantize winefox-lora-f16.gguf winefox-lora-Q4_K_M.gguf Q4_K_M`\n",
    "- `llama-quantize qwen3.5-0.8b-f16.gguf qwen3.5-0.8b-Q5_K_M.gguf Q5_K_M`\n",
    "- `llama-quantize qwen3.5-0.8b-f16.gguf qwen3.5-0.8b-Q4_K_M.gguf Q4_K_M`\n",
]
cells[0]["source"] = md_new

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("train_kaggle.ipynb 更新完成")
