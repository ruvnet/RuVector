#!/usr/bin/env python3
"""Generate a deterministic TimesFM-1.0-200M reference forecast using the
official google-research/timesfm PyTorch PatchedTimeSeriesDecoder + the real
torch_model.ckpt. Saves ref.json (input + reference forecast + intermediates)
and re-emits the state_dict for the candle bridge.

This is the GROUND TRUTH the candle port must reproduce.
"""
import json
import math
import os
import sys

import numpy as np
import torch

# Use the reference decoder straight from the cloned repo, loaded as a
# standalone module (the package __init__ pulls pandas/jax we don't need).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ppd", "/tmp/timesfm-ref/v1/src/timesfm/pytorch_patched_decoder.py")
ppd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppd)
PatchedTimeSeriesDecoder = ppd.PatchedTimeSeriesDecoder
TimesFMConfig = ppd.TimesFMConfig

CKPT = "/tmp/timesfm-torch/torch_model.ckpt"
OUT = "/tmp/timesfm-parity/ref.json"

CONTEXT_LEN = 512   # full model context = 16 patches of 32
HORIZON = 128       # one output patch
FREQ_ID = 0

torch.manual_seed(0)


def build_series(n: int) -> np.ndarray:
    """Deterministic, non-trivial input: sine + slow linear trend.
    Chosen so the forecast is genuinely model-driven (not a constant)."""
    t = np.arange(n, dtype=np.float64)
    series = (
        np.sin(2.0 * math.pi * t / 24.0)          # daily-ish seasonality
        + 0.5 * np.sin(2.0 * math.pi * t / 168.0)  # weekly-ish seasonality
        + 0.01 * t                                  # slow upward trend
    )
    return series.astype(np.float32)


def main() -> int:
    if not os.path.exists(CKPT):
        print(f"ERROR: checkpoint not found: {CKPT}", file=sys.stderr)
        return 2

    cfg = TimesFMConfig()  # all defaults == 1.0 200M
    model = PatchedTimeSeriesDecoder(cfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    # Force float32 everywhere for a clean parity comparison (candle path is f32).
    model = model.float()

    series = build_series(CONTEXT_LEN)
    input_ts = torch.tensor(series, dtype=torch.float32).unsqueeze(0)  # [1, C]
    # paddings shape must be [B, C + H]; all zeros (no padding, fully real).
    paddings = torch.zeros((1, CONTEXT_LEN + HORIZON), dtype=torch.float32)
    freq = torch.tensor([[FREQ_ID]], dtype=torch.long)  # [1, 1]

    with torch.no_grad():
        # --- Capture intermediates for bisection if parity fails. ---
        # Replicate _preprocess_input internals to grab post-embedding tensor.
        bsize = input_ts.shape[0]
        patched_inputs = input_ts.view(bsize, -1, cfg.patch_len)
        patched_pads = torch.zeros_like(patched_inputs)
        mu, sigma = ppd._masked_mean_std(patched_inputs, patched_pads)
        sigma_c = torch.clamp(sigma, min=cfg.tolerance)
        normed = (patched_inputs - mu[:, None, None]) / sigma_c[:, None, None]
        normed = normed * (1.0 - patched_pads)
        concat_inputs = torch.cat([normed, patched_pads], dim=-1)
        model_input = model.input_ff_layer(concat_inputs)  # [1,N,D] pre-pos
        after_input_ff = model_input.clone()
        if cfg.use_positional_embedding:
            pos_emb = model.position_emb(model_input.shape[1]).to(model_input.device)
            pos_emb = torch.cat([pos_emb] * model_input.shape[0], dim=0)
            model_input = model_input + pos_emb  # no padding -> no shift
        f_emb = model.freq_emb(freq)
        model_input = model_input + f_emb
        after_embed = model_input.clone()

        # layer 0 output
        patched_padding = torch.min(patched_pads, dim=-1)[0]
        pad_mask = ppd.convert_paddings_to_mask(patched_padding, model_input.dtype)
        atten = ppd.causal_mask(model_input)
        mask = ppd.merge_masks(pad_mask, atten)
        h = after_embed.clone()
        _, h0 = model.stacked_transformer.layers[0](h, mask, patched_padding)
        after_layer0 = h0.clone()

        # full stack
        h_full = after_embed.clone()
        for layer in model.stacked_transformer.layers:
            _, h_full = layer(h_full, mask, patched_padding)
        after_stack = h_full.clone()

        # --- Official decode for the actual forecast. ---
        point, full = model.decode(
            input_ts=input_ts,
            paddings=paddings,
            freq=freq,
            horizon_len=HORIZON,
            return_forecast_on_context=False,
        )

    point_np = point.squeeze(0).cpu().numpy()        # [H]
    full_np = full.squeeze(0).cpu().numpy()          # [H, 10]

    ref = {
        "config": {
            "context_len": CONTEXT_LEN,
            "horizon": HORIZON,
            "freq_id": FREQ_ID,
            "patch_len": cfg.patch_len,
            "horizon_len": cfg.horizon_len,
            "num_outputs": 1 + len(cfg.quantiles),
        },
        "input_series": series.tolist(),
        "mu": float(mu.item()),
        "sigma": float(sigma_c.item()),
        "point_forecast": point_np.tolist(),          # [H] mean channel
        "full_forecast": full_np.tolist(),            # [H, 10]
        # Intermediates for bisection (single forward, last-patch slice for [N,D]).
        "intermediates": {
            "after_input_ff_meanabs": float(after_input_ff.abs().mean().item()),
            "after_input_ff_lastpatch": after_input_ff[0, -1, :].cpu().numpy().tolist(),
            "after_embed_lastpatch": after_embed[0, -1, :].cpu().numpy().tolist(),
            "after_layer0_lastpatch": after_layer0[0, -1, :].cpu().numpy().tolist(),
            "after_stack_lastpatch": after_stack[0, -1, :].cpu().numpy().tolist(),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(ref, f)
    print(f"wrote {OUT}")
    print(f"  mu={ref['mu']:.6f} sigma={ref['sigma']:.6f}")
    print(f"  point_forecast[:5] = {point_np[:5]}")
    print(f"  point_forecast[-5:] = {point_np[-5:]}")
    print(f"  point min/max/mean = {point_np.min():.4f} / {point_np.max():.4f} / {point_np.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
