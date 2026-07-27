import numpy as np
import struct
import sys
import os

def export_voices(npy_path, output_path):
    print(f"Loading {npy_path}...")
    try:
        # Allow pickle to load the dictionary containing voices
        voices = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading npy file: {e}")
        print("Make sure the file exists and is a valid numpy file.")
        return

    voice_items = []
    if isinstance(voices, dict):
        voice_items = list(voices.items())
    elif isinstance(voices, np.lib.npyio.NpzFile):
        # Handle .npz files
        voice_items = list(voices.items())
    elif isinstance(voices, np.ndarray):
        if voices.dtype.names:
            # Structured array
            voice_items = [(name, voices[name]) for name in voices.dtype.names]
        elif voices.ndim == 0 and voices.dtype == np.object_:
             # 0-d array wrapping a dict
             try:
                 actual_voices = voices.item()
                 if isinstance(actual_voices, dict):
                     voice_items = list(actual_voices.items())
             except:
                 pass

    if not voice_items:
        print("Error: Expected the npy file to contain a dictionary or structured array of voices.")
        return

    print(f"Found {len(voice_items)} voices. Exporting to {output_path}...")

    with open(output_path, 'wb') as f:
        # 1. Magic Header "VOIC"
        f.write(b'VOIC')
        # 2. Version 1
        f.write(struct.pack('<I', 1))
        # 3. Number of voices
        f.write(struct.pack('<I', len(voice_items)))

        for name, style in voice_items:
            # Ensure style is a flat float32 array
            style_arr = np.array(style, dtype=np.float32).flatten()
            
            name_bytes = name.encode('utf-8')
            
            # 4. Name Length
            f.write(struct.pack('<I', len(name_bytes)))
            # 5. Name Bytes
            f.write(name_bytes)
            # 6. Style Dimension
            f.write(struct.pack('<I', len(style_arr)))
            # 7. Style Data
            f.write(style_arr.tobytes())
            
            print(f"  Exported '{name}' (dim: {len(style_arr)})")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_voices.py <path_to_voices.npy> <output_voices.bin>")
        sys.exit(1)
    
    export_voices(sys.argv[1], sys.argv[2])
