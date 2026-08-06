"""导出 Kokoro voicepack 为 voices.bin 格式（供 C++ 推理引擎使用）。

voices.bin 格式：
  Header: "VOIC" (4 bytes) + version (uint32) + n_voices (uint32)
  每个 voice:
    name_len (uint32) + name (utf-8 bytes)
    dim (uint32) + data (dim * float32)

voicepack shape: (510, 1, 256) → 展平为 510*256 = 130560 个 float32

用法：
    python export_voice.py winefox_voice.pt --name winefox
    python export_voice.py winefox_voice.pt --name winefox --output voices.bin
"""
import os
import sys
import struct
import argparse
import numpy as np
import torch


def export_voicebin(pt_path: str, name: str, output_path: str):
    """将单个 .pt voicepack 导出为 voices.bin 格式。

    Args:
        pt_path: .pt 文件路径，shape (510, 1, 256)
        name: voice 名称（如 "winefox"）
        output_path: 输出 .bin 路径
    """
    voicepack = torch.load(pt_path, map_location="cpu", weights_only=True)
    print(f"[Load] {pt_path}: shape={voicepack.shape}, dtype={voicepack.dtype}")

    # 展平为 1D float32
    style_arr = np.array(voicepack, dtype=np.float32).flatten()
    print(f"[Export] name={name}, dim={len(style_arr)}")

    with open(output_path, "wb") as f:
        # Header
        f.write(b"VOIC")
        f.write(struct.pack("<I", 1))          # version
        f.write(struct.pack("<I", 1))          # n_voices (单个)

        # Voice entry
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", len(style_arr)))
        f.write(style_arr.tobytes())

    size = os.path.getsize(output_path)
    print(f"[Done] {output_path} ({size} bytes, {size/1024:.1f} KB)")


def append_to_voicebin(pt_path: str, name: str, existing_bin: str,
                       output_path: str = None):
    """将 voicepack 追加到已有的 voices.bin 中。

    Args:
        pt_path: .pt 文件路径
        name: voice 名称
        existing_bin: 已有的 voices.bin 路径
        output_path: 输出路径（默认覆盖 existing_bin）
    """
    if output_path is None:
        output_path = existing_bin

    # 读取已有 voices
    voices = read_voicebin(existing_bin)
    print(f"[Read] {existing_bin}: {len(voices)} voices")

    # 加载新 voice
    voicepack = torch.load(pt_path, map_location="cpu", weights_only=True)
    style_arr = np.array(voicepack, dtype=np.float32).flatten()

    # 替换或追加
    voices[name] = style_arr
    print(f"[Append] name={name}, dim={len(style_arr)}")

    # 写回
    with open(output_path, "wb") as f:
        f.write(b"VOIC")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(voices)))
        for vname, vdata in voices.items():
            name_bytes = vname.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(vdata)))
            f.write(vdata.tobytes())

    size = os.path.getsize(output_path)
    print(f"[Done] {output_path}: {len(voices)} voices ({size/1024:.1f} KB)")


def read_voicebin(bin_path: str):
    """读取 voices.bin，返回 {name: np.ndarray} 字典。"""
    voices = {}
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"VOIC", f"bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            dim = struct.unpack("<I", f.read(4))[0]
            data = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            voices[name] = data
    return voices


def main():
    parser = argparse.ArgumentParser(
        description="导出 Kokoro voicepack 为 voices.bin 格式"
    )
    parser.add_argument("pt_path", help=".pt voicepack 文件路径")
    parser.add_argument("--name", required=True,
                        help="voice 名称（如 winefox）")
    parser.add_argument("--output", default="voices.bin",
                        help="输出 .bin 路径")
    parser.add_argument("--append", action="store_true",
                        help="追加到已有的 voices.bin")
    parser.add_argument("--existing", default=None,
                        help="已有的 voices.bin 路径（--append 时使用）")
    args = parser.parse_args()

    if args.append:
        existing = args.existing or args.output
        if not os.path.exists(existing):
            print(f"[ERROR] {existing} 不存在，无法追加")
            sys.exit(1)
        append_to_voicebin(args.pt_path, args.name, existing, args.output)
    else:
        export_voicebin(args.pt_path, args.name, args.output)


if __name__ == "__main__":
    main()
