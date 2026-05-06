#!/usr/bin/env python3
"""
compile-csi-encoder-hef.py — compile the RuView CSI contrastive encoder
to a Hailo HEF for NPU inference on the Hailo-8 AI HAT+.

ADR-183 Tier 3: WifiCsi128d variant.

Architecture (from ruv/ruview model.safetensors):
  Input:  [batch, 8]   — 8 aggregate CSI features from sliding window
  FC1:    Linear(8→64) + ReLU
  FC2:    Linear(64→128)
  Output: [batch, 128] — contrastive embedding (L2-normalised by caller)

Usage:
  venv-hailo/bin/python deploy/compile-csi-encoder-hef.py \
      [--weights model.safetensors] [--out csi-encoder.hef]

Deps (all in venv-hailo):
  hailo_dataflow_compiler, safetensors, torch, onnx
"""

import argparse
import struct
import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
import onnx

# ── model architecture ────────────────────────────────────────────────────────

class CsiEncoder(nn.Module):
    """2-layer FC CSI encoder matching ruv/ruview model.safetensors."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # caller applies L2-normalise; Hailo HEF is linear-only


def load_weights_from_safetensors(model: CsiEncoder, path: str):
    """Parse the safetensors file and load weights manually."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        import json
        header = json.loads(f.read(header_size).rstrip(b"\x00").decode("utf-8"))
        data_start = 8 + header_size
        f.seek(data_start)
        data = f.read()

    def get_tensor(key):
        info = header[key]
        start, end = info["data_offsets"]
        dtype_map = {"F32": np.float32, "F16": np.float16, "BF16": np.float32}
        dtype = dtype_map[info["dtype"]]
        arr = np.frombuffer(data[start:end], dtype=dtype).copy()
        return torch.from_numpy(arr.reshape(info["shape"]))

    # encoder.w1 is stored flat [512] → reshape to [64, 8]
    model.fc1.weight.data = get_tensor("encoder.w1").reshape(64, 8)
    model.fc1.bias.data   = get_tensor("encoder.b1")
    # encoder.w2 is stored flat [8192] → reshape to [128, 64]
    model.fc2.weight.data = get_tensor("encoder.w2").reshape(128, 64)
    model.fc2.bias.data   = get_tensor("encoder.b2")

    print(f"  Loaded weights from {path}")
    print(f"    fc1: {list(model.fc1.weight.shape)}, fc2: {list(model.fc2.weight.shape)}")


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, onnx_path: str, batch: int = 1):
    model.eval()
    dummy = torch.zeros(batch, 8)
    torch.onnx.export(
        model, dummy, onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["csi_features"],
        output_names=["embedding"],
        dynamic_axes={"csi_features": {0: "batch"}, "embedding": {0: "batch"}},
    )
    onnx.checker.check_model(onnx_path)
    print(f"  ONNX export OK → {onnx_path}")
    # Print graph nodes
    m = onnx.load(onnx_path)
    print(f"  Nodes: {[n.op_type for n in m.graph.node]}")


# ── Hailo DFC compilation ─────────────────────────────────────────────────────

def compile_hef(onnx_path: str, hef_path: str, hw_arch: str = "hailo8"):
    import os
    # Force CPU-only TF inside DFC — prevent CUDA_ERROR_INVALID_HANDLE
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.exposed_definitions import States

    print(f"  Compiling {onnx_path} → {hef_path} for {hw_arch} ...")
    runner = ClientRunner(hw_arch=hw_arch)

    # Parse ONNX
    runner.translate_onnx_model(
        onnx_path,
        "csi_encoder",
        start_node_names=["csi_features"],
        end_node_names=["embedding"],
    )
    assert runner.state == States.HAILO_MODEL

    # Calibration dataset — representative CSI feature vectors (unit normal)
    rng = np.random.default_rng(42)
    calib = {
        "csi_features": rng.standard_normal((64, 8)).astype(np.float32)
    }
    runner.optimize(calib)
    assert runner.state == States.HAILO_MODEL_OPTIMIZED

    # Compile
    hef_bytes = runner.compile()
    with open(hef_path, "wb") as f:
        f.write(hef_bytes)

    size_kb = os.path.getsize(hef_path) / 1024
    print(f"  HEF compiled → {hef_path} ({size_kb:.1f} KiB)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compile RuView CSI encoder to Hailo HEF")
    ap.add_argument(
        "--weights",
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/models--ruv--ruview/snapshots/"
            "472cfbe8482403256d45f4acb0aed1a5f0db41be/model.safetensors"
        ),
        help="Path to model.safetensors from ruv/ruview",
    )
    ap.add_argument("--out", default="csi-encoder.hef", help="Output HEF path")
    ap.add_argument("--hw-arch", default="hailo8", help="Hailo hardware arch")
    ap.add_argument("--random-weights", action="store_true",
                    help="Skip loading weights (unit test / CI mode)")
    args = ap.parse_args()

    print("=== RuView CSI encoder → Hailo HEF (ADR-183 Tier 3) ===")

    # 1. Build model
    model = CsiEncoder()
    if not args.random_weights:
        if not os.path.exists(args.weights):
            print(f"  weights not found at {args.weights}, use --random-weights for CI")
            sys.exit(1)
        load_weights_from_safetensors(model, args.weights)
    else:
        print("  Using random weights (CI / architecture test mode)")

    # 2. Export ONNX
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name
    try:
        export_onnx(model, onnx_path)

        # 3. Compile HEF
        compile_hef(onnx_path, args.out, hw_arch=args.hw_arch)

        # 4. Print SHA-256 for manifest
        import hashlib
        sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
        print(f"  sha256:{sha}")
        print(f"\nDone. Deploy to cognitum-v0:")
        print(f"  scp {args.out} root@cognitum-v0:/usr/local/share/ruvector/csi-encoder.hef")
    finally:
        os.unlink(onnx_path)


if __name__ == "__main__":
    main()
