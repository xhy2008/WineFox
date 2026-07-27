"""Inspect kokoro-v1.1-zh.onnx - find mel split point."""
import sys
import onnx
import os
from collections import Counter

MODEL = r"e:\winefox\voice-test\models\kokoro-v1.1-zh.onnx"

def main():
    m = onnx.load(MODEL)
    g = m.graph
    print(f"Nodes: {len(g.node)}, Inputs: {len(g.input)}, Outputs: {len(g.output)}")
    print(f"Opset: {[o.version for o in m.opset_import]}")

    # Inputs
    print("\n=== INPUTS ===")
    for i in g.input:
        shape = []
        tt = i.type.tensor_type
        for d in tt.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        print(f"  {i.name}: dtype={tt.elem_type} shape={shape}")

    # Outputs
    print("\n=== OUTPUTS ===")
    for o in g.output:
        shape = []
        tt = o.type.tensor_type
        for d in tt.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        print(f"  {o.name}: dtype={tt.elem_type} shape={shape}")

    # Op histogram
    ops = Counter(n.op_type for n in g.node)
    print("\n=== OP HISTOGRAM ===")
    for op, cnt in ops.most_common(30):
        print(f"  {op:20s} {cnt}")

    # Find ConvTranspose (vocoder signature)
    ct_nodes = [n for n in g.node if n.op_type == "ConvTranspose"]
    print(f"\n=== ConvTranspose count: {len(ct_nodes)} ===")
    if ct_nodes:
        # The first ConvTranspose usually marks vocoder start
        first_ct = ct_nodes[0]
        ct_idx = list(g.node).index(first_ct)
        print(f"  First ConvTranspose at node index: {ct_idx}/{len(g.node)}")
        print(f"  Name: {first_ct.name}")
        print(f"  Inputs: {list(first_ct.input)}")
        print(f"  Outputs: {list(first_ct.output)}")

    # Find nodes with 'mel' or 'spec' in name
    print("\n=== NODES with mel/spec/decoder/vocoder in name ===")
    for n in g.node:
        nl = n.name.lower()
        if any(k in nl for k in ["mel", "spec", "decoder", "vocoder"]):
            print(f"  {n.op_type:20s} {n.name}")

    # Find value_info with 80 in shape (mel dim)
    print("\n=== value_info with 80 in shape ===")
    for v in g.value_info:
        tt = v.type.tensor_type
        shape = []
        for d in tt.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        if 80 in [s for s in shape if isinstance(s, int)]:
            print(f"  {v.name}: shape={shape}")

    # Print structure around first ConvTranspose
    if ct_nodes:
        ct_idx = list(g.node).index(ct_nodes[0])
        print(f"\n=== NODES around first ConvTranspose (idx {ct_idx-3} to {ct_idx+3}) ===")
        for i in range(max(0, ct_idx-3), min(len(g.node), ct_idx+4)):
            n = g.node[i]
            marker = " <-- ConvTranspose" if n.op_type == "ConvTranspose" else ""
            print(f"  [{i:4d}] {n.op_type:20s} {n.name}{marker}")
            print(f"         in:  {list(n.input)[:3]}{'...' if len(n.input) > 3 else ''}")
            print(f"         out: {list(n.output)}")

if __name__ == "__main__":
    main()
