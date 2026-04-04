#!/usr/bin/env python3
"""
Export a trained deobfuscation model to GGUF Q4 format and package
it into an RVF container with an OVERLAY segment.

Pipeline:
    1. Load PyTorch checkpoint
    2. Export to ONNX (if not already done)
    3. Quantize weights to INT8 / Q4
    4. Write GGUF Q4 file for RuvLLM inference
    5. Create RVF container with OVERLAY segment containing the weights

Usage:
    python export-to-rvf.py --checkpoint model/best_model.pt --output model/deobfuscator
    python export-to-rvf.py --checkpoint model/best_model.pt --output model/deobfuscator --quantize q4
"""

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match train-deobfuscator.py)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
FFN_DIM = 512
MAX_CONTEXT = 64
MAX_NAME = 32

# GGUF magic and version.
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types.
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_FLOAT32 = 6

# RVF magic bytes.
RVF_MAGIC = b"RVF\x01"
RVF_OVERLAY_TYPE = 0x10  # OVERLAY segment type

# Quantization types.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------


def load_checkpoint(path: str) -> dict:
    """Load a PyTorch checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        return checkpoint
    else:
        # Bare state dict.
        return {"model_state_dict": checkpoint, "config": {}}


# ---------------------------------------------------------------------------
# GGUF Writer
# ---------------------------------------------------------------------------


def quantize_q4(tensor: np.ndarray) -> bytes:
    """Quantize a float32 tensor to Q4_0 format (4-bit quantization).

    Q4_0 format: blocks of 32 values, each block has:
      - 1 x float16 scale factor (2 bytes)
      - 16 x uint8 packed nibbles (16 bytes)
    Total: 18 bytes per 32 values.
    """
    flat = tensor.flatten().astype(np.float32)

    # Pad to multiple of 32.
    remainder = len(flat) % 32
    if remainder != 0:
        flat = np.concatenate([flat, np.zeros(32 - remainder, dtype=np.float32)])

    num_blocks = len(flat) // 32
    result = bytearray()

    for i in range(num_blocks):
        block = flat[i * 32 : (i + 1) * 32]
        abs_max = np.max(np.abs(block))
        scale = abs_max / 7.0 if abs_max > 0 else 1.0

        # Quantize to 4-bit signed integers [-8, 7].
        quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)

        # Pack scale as float16.
        result.extend(struct.pack("<e", np.float16(scale)))

        # Pack pairs of 4-bit values into bytes.
        for j in range(0, 32, 2):
            lo = quantized[j] & 0x0F
            hi = (quantized[j + 1] & 0x0F) << 4
            result.append(lo | hi)

    return bytes(result)


def quantize_q8(tensor: np.ndarray) -> bytes:
    """Quantize a float32 tensor to Q8_0 format (8-bit quantization).

    Q8_0 format: blocks of 32 values, each block has:
      - 1 x float16 scale factor (2 bytes)
      - 32 x int8 quantized values (32 bytes)
    Total: 34 bytes per 32 values.
    """
    flat = tensor.flatten().astype(np.float32)

    remainder = len(flat) % 32
    if remainder != 0:
        flat = np.concatenate([flat, np.zeros(32 - remainder, dtype=np.float32)])

    num_blocks = len(flat) // 32
    result = bytearray()

    for i in range(num_blocks):
        block = flat[i * 32 : (i + 1) * 32]
        abs_max = np.max(np.abs(block))
        scale = abs_max / 127.0 if abs_max > 0 else 1.0

        quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)

        result.extend(struct.pack("<e", np.float16(scale)))
        result.extend(quantized.tobytes())

    return bytes(result)


def write_gguf_string(f, s: str):
    """Write a GGUF string (length-prefixed UTF-8)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def write_gguf_kv_string(f, key: str, value: str):
    """Write a GGUF key-value pair with string value."""
    write_gguf_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    write_gguf_string(f, value)


def write_gguf_kv_uint32(f, key: str, value: int):
    """Write a GGUF key-value pair with uint32 value."""
    write_gguf_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", value))


def write_gguf_kv_float32(f, key: str, value: float):
    """Write a GGUF key-value pair with float32 value."""
    write_gguf_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<f", value))


def export_gguf(state_dict: dict, output_path: str, quant: str = "q4"):
    """Export model weights to GGUF format with quantization."""

    # Prepare tensors.
    tensors = []
    for name, param in state_dict.items():
        arr = param.detach().cpu().numpy()
        tensors.append((name, arr))

    # Metadata KV pairs.
    metadata = [
        ("general.architecture", "deobfuscator"),
        ("general.name", "ruvector-deobfuscator"),
        ("general.file_type", quant.upper()),
        ("deobfuscator.vocab_size", VOCAB_SIZE),
        ("deobfuscator.embed_dim", EMBED_DIM),
        ("deobfuscator.num_heads", NUM_HEADS),
        ("deobfuscator.num_layers", NUM_LAYERS),
        ("deobfuscator.ffn_dim", FFN_DIM),
        ("deobfuscator.max_context", MAX_CONTEXT),
        ("deobfuscator.max_name", MAX_NAME),
    ]

    # Quantize all tensors.
    quantized_data = []
    for name, arr in tensors:
        if quant == "q4":
            data = quantize_q4(arr)
            qtype = GGML_TYPE_Q4_0
        elif quant == "q8":
            data = quantize_q8(arr)
            qtype = GGML_TYPE_Q8_0
        else:
            data = arr.astype(np.float32).tobytes()
            qtype = GGML_TYPE_F32
        quantized_data.append((name, arr.shape, qtype, data))

    with open(output_path, "wb") as f:
        # Header.
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(quantized_data)))  # n_tensors
        f.write(struct.pack("<Q", len(metadata)))  # n_kv

        # Metadata.
        for key, value in metadata:
            if isinstance(value, str):
                write_gguf_kv_string(f, key, value)
            elif isinstance(value, int):
                write_gguf_kv_uint32(f, key, value)
            elif isinstance(value, float):
                write_gguf_kv_float32(f, key, value)

        # Tensor info headers.
        for name, shape, qtype, data in quantized_data:
            write_gguf_string(f, name)
            n_dims = len(shape)
            f.write(struct.pack("<I", n_dims))
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", qtype))
            f.write(struct.pack("<Q", 0))  # offset (filled later)

        # Alignment padding.
        alignment = 32
        pos = f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        f.write(b"\x00" * pad)

        # Tensor data.
        for name, shape, qtype, data in quantized_data:
            f.write(data)
            # Align each tensor.
            pad = (alignment - (len(data) % alignment)) % alignment
            f.write(b"\x00" * pad)

    file_size = os.path.getsize(output_path)
    print(f"Wrote GGUF ({quant.upper()}) to {output_path} ({file_size / 1024 / 1024:.2f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# RVF Container
# ---------------------------------------------------------------------------


def create_rvf_container(gguf_path: str, output_path: str):
    """Wrap GGUF model in an RVF container with OVERLAY segment."""

    gguf_data = open(gguf_path, "rb").read()
    gguf_hash = hashlib.sha256(gguf_data).hexdigest()

    # RVF header.
    header = {
        "magic": "RVF",
        "version": 1,
        "segments": [
            {
                "type": "OVERLAY",
                "type_id": RVF_OVERLAY_TYPE,
                "name": "deobfuscator-model",
                "size": len(gguf_data),
                "hash": gguf_hash,
                "format": "gguf-q4",
                "model": {
                    "architecture": "deobfuscator",
                    "vocab_size": VOCAB_SIZE,
                    "embed_dim": EMBED_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "max_context": MAX_CONTEXT,
                    "max_name": MAX_NAME,
                },
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ],
    }

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        # RVF magic.
        f.write(RVF_MAGIC)
        # Header length (4 bytes, little-endian).
        f.write(struct.pack("<I", len(header_json)))
        # Header JSON.
        f.write(header_json)
        # OVERLAY segment data.
        f.write(gguf_data)

    file_size = os.path.getsize(output_path)
    print(f"Wrote RVF container to {output_path} ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  GGUF hash: {gguf_hash[:16]}...")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export deobfuscation model to GGUF/RVF")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", default="./model/deobfuscator", help="Output path prefix")
    parser.add_argument("--quantize", choices=["q4", "q8", "f32"], default="q4", help="Quantization level")
    parser.add_argument("--skip-rvf", action="store_true", help="Skip RVF container creation")
    args = parser.parse_args()

    # Load checkpoint.
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(args.checkpoint)
    state_dict = checkpoint["model_state_dict"]
    print(f"  Loaded {len(state_dict)} tensors")

    # Export GGUF.
    gguf_path = f"{args.output}.gguf"
    os.makedirs(os.path.dirname(gguf_path) or ".", exist_ok=True)
    export_gguf(state_dict, gguf_path, quant=args.quantize)

    # Create RVF container.
    if not args.skip_rvf:
        rvf_path = f"{args.output}.rvf"
        create_rvf_container(gguf_path, rvf_path)

    print("\nExport complete.")


if __name__ == "__main__":
    main()
