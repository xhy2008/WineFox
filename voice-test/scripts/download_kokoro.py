"""Download Kokoro-82M original PyTorch model from HuggingFace."""
import os
import sys
from huggingface_hub import snapshot_download

OUT_DIR = r"e:\winefox\voice-test\models\kokoro-82M-src"
REPO = "hexgrad/Kokoro-82M"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {REPO} to {OUT_DIR} ...")
    # 只下载必要文件，跳过 .gitattributes 和大文件之外的内容
    path = snapshot_download(
        repo_id=REPO,
        local_dir=OUT_DIR,
        # 不指定 allow_patterns 就下载全部， Kokoro-82M 总共 ~326MB
    )
    print(f"Downloaded to: {path}")
    print("\nFiles:")
    for root, dirs, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            size_mb = os.path.getsize(full) / 1024 / 1024
            rel = os.path.relpath(full, path)
            print(f"  {rel} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
