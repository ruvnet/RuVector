#!/usr/bin/env python3
"""Convert Google TimesFM-1.0-200M weights to the Rust `timesfm` crate's
VarBuilder key hierarchy and emit a remapped `.safetensors`.

WHAT THIS DOES
==============
The HuggingFace model card `google/timesfm-1.0-200m` ships a PyTorch checkpoint
(`checkpoints/torch_model.ckpt`, a `torch.save` state_dict) that is loaded with
`PatchedTimeSeriesDecoder.load_state_dict(...)` in
`google-research/timesfm/v1/src/timesfm/pytorch_patched_decoder.py`.

The Rust crate (`crates/timesfm/src/{model,config}.rs`) rebuilds the same model
with candle `VarBuilder`, but two key paths differ from the PyTorch module tree:

  1. The `nn.ModuleList` of decoder layers is named `layers` in PyTorch
     (`StackedDecoder.self.layers = nn.ModuleList()`), so its state_dict keys are
     `stacked_transformer.layers.{i}....`. The Rust `StackedDecoder` uses
     `vb.pp(i)` directly under `stacked_transformer`, i.e. NO `layers` segment:
     `stacked_transformer.{i}....`.

  2. `ResidualBlock.hidden_layer` is an `nn.Sequential(Linear, SiLU)` in PyTorch,
     so its Linear params live at `hidden_layer.0.{weight,bias}`. The Rust
     `ResidualBlock` uses a plain `candle_nn::linear` at `hidden_layer`, i.e.
     `hidden_layer.{weight,bias}` (NO `.0.` segment).

Everything else maps 1:1 because the Rust attribute names were chosen to match
the PyTorch attribute names exactly:
  input_ff_layer, output_layer, residual_layer, freq_emb, input_layernorm,
  self_attn, qkv_proj, o_proj, scaling, mlp, layer_norm, gate_proj, down_proj,
  horizon_ff_layer.

The source keys below are derived from the literal `self.<name> = ...`
assignments in `pytorch_patched_decoder.py` (verified against the cloned repo,
master). The target keys are derived from the literal `vb.pp("...")` / `vb.get`
calls in `crates/timesfm/src/model.rs`. No key names are invented.

KEY-BY-KEY MAPPING (exhaustive)
===============================
PyTorch source key (state_dict)                         -> Rust VarBuilder target key
---------------------------------------------------------------------------------------
# --- input ResidualBlock --------------------------------------------------------------
input_ff_layer.hidden_layer.0.weight                    -> input_ff_layer.hidden_layer.weight
input_ff_layer.hidden_layer.0.bias                      -> input_ff_layer.hidden_layer.bias
input_ff_layer.output_layer.weight                      -> input_ff_layer.output_layer.weight
input_ff_layer.output_layer.bias                        -> input_ff_layer.output_layer.bias
input_ff_layer.residual_layer.weight                    -> input_ff_layer.residual_layer.weight
input_ff_layer.residual_layer.bias                      -> input_ff_layer.residual_layer.bias
# --- frequency embedding ---------------------------------------------------------------
freq_emb.weight                                         -> freq_emb.weight
# --- per decoder layer i = 0..num_layers-1 (20) ----------------------------------------
stacked_transformer.layers.{i}.input_layernorm.weight   -> stacked_transformer.{i}.input_layernorm.weight
stacked_transformer.layers.{i}.self_attn.qkv_proj.weight-> stacked_transformer.{i}.self_attn.qkv_proj.weight
stacked_transformer.layers.{i}.self_attn.qkv_proj.bias  -> stacked_transformer.{i}.self_attn.qkv_proj.bias
stacked_transformer.layers.{i}.self_attn.o_proj.weight  -> stacked_transformer.{i}.self_attn.o_proj.weight
stacked_transformer.layers.{i}.self_attn.o_proj.bias    -> stacked_transformer.{i}.self_attn.o_proj.bias
stacked_transformer.layers.{i}.self_attn.scaling        -> stacked_transformer.{i}.self_attn.scaling
stacked_transformer.layers.{i}.mlp.layer_norm.weight    -> stacked_transformer.{i}.mlp.layer_norm.weight
stacked_transformer.layers.{i}.mlp.layer_norm.bias      -> stacked_transformer.{i}.mlp.layer_norm.bias
stacked_transformer.layers.{i}.mlp.gate_proj.weight     -> stacked_transformer.{i}.mlp.gate_proj.weight
stacked_transformer.layers.{i}.mlp.gate_proj.bias       -> stacked_transformer.{i}.mlp.gate_proj.bias
stacked_transformer.layers.{i}.mlp.down_proj.weight     -> stacked_transformer.{i}.mlp.down_proj.weight
stacked_transformer.layers.{i}.mlp.down_proj.bias       -> stacked_transformer.{i}.mlp.down_proj.bias
# --- output ResidualBlock --------------------------------------------------------------
horizon_ff_layer.hidden_layer.0.weight                  -> horizon_ff_layer.hidden_layer.weight
horizon_ff_layer.hidden_layer.0.bias                    -> horizon_ff_layer.hidden_layer.bias
horizon_ff_layer.output_layer.weight                    -> horizon_ff_layer.output_layer.weight
horizon_ff_layer.output_layer.bias                      -> horizon_ff_layer.output_layer.bias
horizon_ff_layer.residual_layer.weight                  -> horizon_ff_layer.residual_layer.weight
horizon_ff_layer.residual_layer.bias                    -> horizon_ff_layer.residual_layer.bias

NOTES / DELIBERATELY NOT MAPPED
===============================
  * RMSNorm (`input_layernorm`) has ONLY a `weight` parameter in both PyTorch
    (`self.weight = nn.Parameter(...)`, no bias) and candle `rms_norm`. No bias key.
  * `mlp.layer_norm` is an `nn.LayerNorm` (weight + bias) in both, so bias IS mapped.
  * `position_emb` is a non-learned sinusoidal embedding in both -> NO parameters,
    nothing to map (correctly absent from the checkpoint).
  * The checkpoint loaded by `load_state_dict` directly into
    `PatchedTimeSeriesDecoder` carries NO top-level module prefix. If a wrapper
    prefix is found (e.g. `model.` or `module.`), `--strip-prefix` removes it.

USAGE
=====
  python convert_weights.py --src torch_model.ckpt --out timesfm_remapped.safetensors
  python convert_weights.py --src model.safetensors --dry-run
  python convert_weights.py --src torch_model.ckpt --num-layers 20 --dry-run

Source format is auto-detected by extension (.safetensors vs .ckpt/.pth/.pt/.bin).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Config: must match crates/timesfm/src/config.rs::timesfm_1p0_200m()
# ---------------------------------------------------------------------------
DEFAULT_NUM_LAYERS = 20  # config.rs num_layers


# ---------------------------------------------------------------------------
# Mapping rules. Each rule is (compiled source-regex, target-template).
# The regex captures the layer index as group 'i' where relevant.
# Order matters: more specific (hidden_layer.0) rules come before generic ones.
# ---------------------------------------------------------------------------
def _build_rules() -> List[Tuple[re.Pattern, str]]:
    rules: List[Tuple[str, str]] = [
        # ResidualBlock hidden_layer Sequential -> plain Linear (drop the ".0").
        # input_ff_layer + horizon_ff_layer share the same shape; one rule each.
        (r"^(?P<blk>input_ff_layer|horizon_ff_layer)\.hidden_layer\.0\.(?P<p>weight|bias)$",
         r"\g<blk>.hidden_layer.\g<p>"),
        # ResidualBlock output_layer / residual_layer: identity (explicit so we
        # don't fall through to a generic catch-all and silently mis-map).
        (r"^(?P<blk>input_ff_layer|horizon_ff_layer)\.(?P<lyr>output_layer|residual_layer)\.(?P<p>weight|bias)$",
         r"\g<blk>.\g<lyr>.\g<p>"),
        # Frequency embedding: identity.
        (r"^freq_emb\.weight$", r"freq_emb.weight"),
        # Decoder layers: drop the ".layers" ModuleList segment.
        # input_layernorm (RMSNorm, weight only).
        (r"^stacked_transformer\.layers\.(?P<i>\d+)\.input_layernorm\.weight$",
         r"stacked_transformer.\g<i>.input_layernorm.weight"),
        # self_attn.qkv_proj / o_proj (Linear: weight+bias).
        (r"^stacked_transformer\.layers\.(?P<i>\d+)\.self_attn\.(?P<lyr>qkv_proj|o_proj)\.(?P<p>weight|bias)$",
         r"stacked_transformer.\g<i>.self_attn.\g<lyr>.\g<p>"),
        # self_attn.scaling (nn.Parameter, no .weight suffix).
        (r"^stacked_transformer\.layers\.(?P<i>\d+)\.self_attn\.scaling$",
         r"stacked_transformer.\g<i>.self_attn.scaling"),
        # mlp.layer_norm (LayerNorm: weight+bias).
        (r"^stacked_transformer\.layers\.(?P<i>\d+)\.mlp\.layer_norm\.(?P<p>weight|bias)$",
         r"stacked_transformer.\g<i>.mlp.layer_norm.\g<p>"),
        # mlp.gate_proj / down_proj (Linear: weight+bias).
        (r"^stacked_transformer\.layers\.(?P<i>\d+)\.mlp\.(?P<lyr>gate_proj|down_proj)\.(?P<p>weight|bias)$",
         r"stacked_transformer.\g<i>.mlp.\g<lyr>.\g<p>"),
    ]
    return [(re.compile(p), t) for p, t in rules]


RULES = _build_rules()


def remap_key(src_key: str) -> str | None:
    """Return the Rust target key for a PyTorch source key, or None if no rule
    matches (caller decides how to flag it)."""
    for pat, tmpl in RULES:
        m = pat.match(src_key)
        if m:
            return m.expand(tmpl)
    return None


# ---------------------------------------------------------------------------
# Expected target keys (derived from model.rs) — used to detect MISSING params.
# ---------------------------------------------------------------------------
def expected_target_keys(num_layers: int) -> List[str]:
    keys: List[str] = []
    for blk in ("input_ff_layer", "horizon_ff_layer"):
        for lyr in ("hidden_layer", "output_layer", "residual_layer"):
            keys.append(f"{blk}.{lyr}.weight")
            keys.append(f"{blk}.{lyr}.bias")
    keys.append("freq_emb.weight")
    for i in range(num_layers):
        base = f"stacked_transformer.{i}"
        keys.append(f"{base}.input_layernorm.weight")
        keys.append(f"{base}.self_attn.qkv_proj.weight")
        keys.append(f"{base}.self_attn.qkv_proj.bias")
        keys.append(f"{base}.self_attn.o_proj.weight")
        keys.append(f"{base}.self_attn.o_proj.bias")
        keys.append(f"{base}.self_attn.scaling")
        keys.append(f"{base}.mlp.layer_norm.weight")
        keys.append(f"{base}.mlp.layer_norm.bias")
        keys.append(f"{base}.mlp.gate_proj.weight")
        keys.append(f"{base}.mlp.gate_proj.bias")
        keys.append(f"{base}.mlp.down_proj.weight")
        keys.append(f"{base}.mlp.down_proj.bias")
    return keys


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_source(path: str):
    """Load a state_dict from .safetensors or a torch checkpoint.
    Returns (tensors_dict, backend) where tensors are torch.Tensor objects."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"source weights not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError(
                "reading .safetensors requires `safetensors` (pip install safetensors torch)"
            ) from e
        return load_file(path), "safetensors"

    # torch checkpoint (.ckpt/.pth/.pt/.bin) — the format HF actually ships.
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "reading a torch checkpoint requires `torch` (pip install torch)"
        ) from e
    obj = torch.load(path, map_location="cpu", weights_only=True)
    # Unwrap common containers.
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"checkpoint at {path} did not deserialize to a state_dict mapping "
            f"(got {type(obj).__name__})"
        )
    return obj, "torch"


def strip_prefix(state: Dict, prefix: str) -> Dict:
    if not prefix:
        return state
    if not prefix.endswith("."):
        prefix = prefix + "."
    out = {}
    for k, v in state.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def autodetect_prefix(state: Dict) -> str:
    """If every key shares a leading `model.`/`module.` segment that the Rust
    tree does not expect, report it so the user can --strip-prefix."""
    for cand in ("model.", "module.", "_model.", "net."):
        if state and all(k.startswith(cand) for k in state):
            return cand
    return ""


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------
def convert(state: Dict, num_layers: int):
    """Return (mapping, remapped_tensors, unmapped_src, missing_targets).
    `mapping` is list of (src, dst). `remapped_tensors` is dst->tensor."""
    mapping: List[Tuple[str, str]] = []
    remapped: Dict = {}
    unmapped_src: List[str] = []

    for src_key in state:
        dst = remap_key(src_key)
        if dst is None:
            unmapped_src.append(src_key)
            continue
        mapping.append((src_key, dst))
        remapped[dst] = state[src_key]

    produced = set(remapped.keys())
    expected = expected_target_keys(num_layers)
    missing_targets = [k for k in expected if k not in produced]
    return mapping, remapped, unmapped_src, missing_targets


def write_safetensors(remapped: Dict, out_path: str):
    try:
        from safetensors.torch import save_file
    except ImportError as e:
        raise RuntimeError(
            "writing .safetensors requires `safetensors` + `torch` "
            "(pip install safetensors torch)"
        ) from e
    import torch

    contiguous = {}
    for k, v in remapped.items():
        t = v
        if hasattr(t, "contiguous"):
            t = t.contiguous()
        # safetensors cannot serialize shared storage; clone to be safe.
        if isinstance(t, torch.Tensor):
            t = t.clone()
        contiguous[k] = t
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    save_file(contiguous, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Remap TimesFM-1.0-200M HF weights to the Rust crate's "
                    "VarBuilder key hierarchy and emit a .safetensors.")
    ap.add_argument("--src", required=True,
                    help="source weights: HF .safetensors OR torch_model.ckpt/.pth/.bin")
    ap.add_argument("--out", default="timesfm_remapped.safetensors",
                    help="output .safetensors path (ignored under --dry-run)")
    ap.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS,
                    help=f"decoder layers to expect (default {DEFAULT_NUM_LAYERS}, "
                         "matches config.rs timesfm_1p0_200m)")
    ap.add_argument("--strip-prefix", default=None,
                    help="strip this leading key prefix (e.g. 'model.') before mapping; "
                         "if omitted, a common prefix is auto-detected and reported")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the mapping and diagnostics WITHOUT writing output")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write output even if some expected target keys are missing")
    args = ap.parse_args(argv)

    # Load.
    try:
        state, backend = load_source(args.src)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Prefix handling.
    prefix = args.strip_prefix
    if prefix is None:
        auto = autodetect_prefix(state)
        if auto:
            print(f"NOTE: every key starts with '{auto}'; stripping it "
                  f"(override with --strip-prefix '').", file=sys.stderr)
            prefix = auto
        else:
            prefix = ""
    state = strip_prefix(state, prefix)

    # Convert.
    mapping, remapped, unmapped_src, missing_targets = convert(state, args.num_layers)
    mapping.sort()

    # Report.
    print("=" * 78)
    print(f"TimesFM weight remap  (source backend: {backend}, {len(state)} source keys)")
    print(f"  expecting {args.num_layers} decoder layers "
          f"({len(expected_target_keys(args.num_layers))} target params)")
    print("=" * 78)
    print(f"\nMAPPED ({len(mapping)}):  source -> target")
    for src, dst in mapping:
        tag = "" if src == dst else "   [remapped]"
        shape = ""
        t = state.get(src)
        if t is not None and hasattr(t, "shape"):
            shape = f"  {tuple(t.shape)}"
        print(f"  {src}\n      -> {dst}{shape}{tag}")

    if unmapped_src:
        print(f"\n!! UNMAPPED SOURCE KEYS ({len(unmapped_src)}) "
              f"— present in checkpoint, NO rule matched (FLAGGED, not guessed):")
        for k in sorted(unmapped_src):
            print(f"   ?? {k}")
    else:
        print("\nUNMAPPED SOURCE KEYS: none — every source key was mapped.")

    if missing_targets:
        print(f"\n!! MISSING TARGET KEYS ({len(missing_targets)}) "
              f"— Rust expects these but no source key produced them (FLAGGED):")
        for k in missing_targets:
            print(f"   !! {k}")
    else:
        print("MISSING TARGET KEYS: none — all expected Rust params were produced.")

    ok = (not unmapped_src) and (not missing_targets)

    if args.dry_run:
        print("\n[dry-run] no file written.")
        return 0 if ok else 1

    if missing_targets and not args.allow_incomplete:
        print(f"\nERROR: {len(missing_targets)} expected target key(s) missing; "
              "refusing to write an incomplete checkpoint. "
              "Re-run with --allow-incomplete to override.", file=sys.stderr)
        return 3

    try:
        write_safetensors(remapped, args.out)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"\nwrote {len(remapped)} tensors -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
