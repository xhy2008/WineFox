"""Inspect voices.bin to list all voice names."""
import struct
import sys

def inspect(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        print(f"Magic: {magic}")
        assert magic == b'VOIC', f"Bad magic: {magic}"

        version = struct.unpack('<I', f.read(4))[0]
        print(f"Version: {version}")

        num_voices = struct.unpack('<I', f.read(4))[0]
        print(f"Num voices: {num_voices}")

        voices = []
        for i in range(num_voices):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            dim = struct.unpack('<I', f.read(4))[0]
            # Skip the style data
            f.read(dim * 4)
            voices.append((name, dim))

        # Print all voice names
        print("\nAll voices:")
        for name, dim in voices:
            print(f"  {name} (dim={dim})")

        # Filter for Chinese voices (zf_, zm_)
        print("\nChinese voices (zf_/zm_):")
        for name, dim in voices:
            if name.startswith('zf_') or name.startswith('zm_'):
                print(f"  {name} (dim={dim})")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else r'e:\winefox\voice-test\models\voices-v1.1-zh.bin'
    inspect(path)
