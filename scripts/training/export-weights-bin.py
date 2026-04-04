#!/usr/bin/env python3
"""Export PyTorch deobfuscation model weights to a simple binary format for Rust.

Binary format per tensor:
    name_len (u32 LE) | name (utf8) | ndim (u32 LE) | shape (ndim * u32 LE) | data (f32 LE...)

Usage:
    python export-weights-bin.py --checkpoint model/best_model.pt --output model/weights.bin
    python export-weights-bin.py --checkpoint model/final_model.pt --output model/weights.bin
"""

import argparse
import struct
import sys

import numpy as np
import torch


def export_weights(checkpoint_path: str, output_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle both full checkpoint dicts and raw state_dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
        print(f"Config: {config}")
    elif isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        print("ERROR: unrecognized checkpoint format", file=sys.stderr)
        sys.exit(1)

    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Tensors: {len(state_dict)}, Parameters: {total_params:,}")

    with open(output_path, "wb") as f:
        for name, param in state_dict.items():
            data = param.detach().cpu().numpy().astype(np.float32)
            name_bytes = name.encode("utf-8")

            # name_len + name
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # ndim + shape
            f.write(struct.pack("<I", len(data.shape)))
            for dim in data.shape:
                f.write(struct.pack("<I", dim))

            # flattened float32 data (C-contiguous = row-major)
            f.write(np.ascontiguousarray(data).tobytes())

            print(f"  {name:50s} {str(list(data.shape)):20s} ({data.size:>8,} floats)")

    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nWrote {output_path} ({size_kb:.1f} KB, {total_params:,} parameters)")


def main() -> None:
    p = argparse.ArgumentParser(description="Export PyTorch weights to binary for Rust")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--output", required=True, help="Output .bin path")
    args = p.parse_args()
    export_weights(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
