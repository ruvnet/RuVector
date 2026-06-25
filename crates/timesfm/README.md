# timesfm

Native Rust / [`candle`](https://github.com/huggingface/candle) port of Google's
**TimesFM 1.0 200M** patched time-series Transformer — zero-shot forecasting inside
RuVector, no Python microservice.

[![Crates.io](https://img.shields.io/crates/v/timesfm)](https://crates.io/crates/timesfm)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

---

## What Is TimesFM?

TimesFM (Time Series Foundation Model) is Google's decoder-only Transformer for
zero-shot time-series forecasting, released at ICML 2024. The 200M variant uses:

| Hyper-parameter | Value |
|---|---|
| Model dim | 1280 |
| Layers | 20 |
| Attention heads | 16 |
| Head dim | 80 |
| Input patch length | 32 |
| Output patch length | 128 |
| Quantile outputs | 10 (p10…p90 + mean) |

---

## Features

- **Zero-shot forecasting** — no fine-tuning required for new time series
- **RevIN normalisation** — per-series instance norm (bug-fixed vs. naive implementation)
- **Autoregressive decode** — arbitrary horizon via multi-step patch decoding
- **Quantile outputs** — `[B, N, 128, 10]` tensor with 10 calibrated quantile levels
- **CPU + GPU** — accelerated via `--features cuda` (RTX) or `--features metal` (Apple)
- **Zero default deps** — `default = []`; candle only activates with `--features candle`

---

## Quick Start

### Add to `Cargo.toml`

```toml
[dependencies]
timesfm = { version = "2.2.3", features = ["candle"] }
```

### Inference

```rust
use timesfm::{TimesfmConfig, TimesfmModel};
use candle_core::{Device, Tensor};

// Load model (dummy weights — see Weight Loading below for real weights)
let cfg = TimesfmConfig::timesfm_1p0_200m();
let device = Device::Cpu;
let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
let model = TimesfmModel::new(&cfg, vb)?;

// Input: [batch=1, n_patches=4, patch_len=32], freq_ids: [1] (hourly)
let context = Tensor::randn(0f32, 1f32, (1, 4, 32), &device)?;
let freq = Tensor::from_slice(&[1u32], (1,), &device)?;

// Forward → [1, 4, 128, 10] (patches × quantiles)
let (output, _) = model.forward(&context, &freq, None)?;

// Autoregressive decode to horizon 256
let horizon = model.decode(&context, &freq, 256)?; // [1, 256, 10]
```

### Weight Loading

Weights are not bundled — download from HuggingFace and convert:

```bash
# 1. Download TimesFM 1.0 200M checkpoint
huggingface-cli download google/timesfm-1.0-200m \
  --local-dir ~/.cache/timesfm

# 2. Convert PyTorch keys → Rust VarBuilder keys
python crates/timesfm/scripts/convert_weights.py \
  --src ~/.cache/timesfm/torch_model.ckpt \
  --out ~/.cache/timesfm/timesfm_candle.safetensors

# 3. Load in Rust
```

```rust
let vb = unsafe {
    candle_nn::VarBuilder::from_mmaped_safetensors(
        &["~/.cache/timesfm/timesfm_candle.safetensors"],
        candle_core::DType::F32,
        &device,
    )?
};
let model = TimesfmModel::new(&cfg, vb)?;
```

---

## Feature Flags

| Flag | Enables |
|------|---------|
| `candle` | Transformer inference (required for `TimesfmModel`) |
| `cuda` | NVIDIA GPU acceleration (requires `candle`) |
| `metal` | Apple Metal GPU acceleration (requires `candle`) |
| `hub` | HuggingFace Hub download helpers |

---

## Architecture

```
Input patches [B, N, 32]
     │
     ▼
ResidualBlock patch embedding  →  [B, N, 1280]
     │
     ├── Sinusoidal pos embedding (non-learned, matches reference)
     ├── Frequency embedding (hourly / daily / weekly / …)
     │
     ▼
20× Decoder Layer:
     ├── RMSNorm → fused QKV proj [1280 → 3840]
     ├── Softplus per-head-dim query scaling (learnable)
     ├── Causal + padding mask
     ├── Output proj [1280 → 1280]
     └── MLP: LayerNorm → Gate → Down (with ReLU residual)
     │
     ▼
RevIN denormalisation (instance norm, first-qualifying-patch stats)
     │
     ▼
Output ResidualBlock  →  [B, N, 128 × 10]   (128 output patches × 10 quantiles)
```

---

## Known Limitations

1. **Weight parity not yet verified** — all tests use `zeros`/`randn` dummy weights.
   End-to-end numerical match against the PyTorch reference is the gating next step.
2. **MLP padding-mask omission** — benign for unpadded / right-aligned inputs.
3. **Positional embedding `_shift_padded_seq`** — not implemented; affects heavily padded sequences.
4. **NaN on fully-padded leading patch** — reference clips to `-0.7×max`; this port uses `-inf`.

---

## References

- Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688) (Das et al., ICML 2024)
- Original: [google-research/timesfm](https://github.com/google-research/timesfm)
- Weights: [google/timesfm-1.0-200m](https://huggingface.co/google/timesfm-1.0-200m) on HuggingFace

---

## License

Apache-2.0 — see [LICENSE](../../LICENSE).

*Part of [ruvnet/ruvector](https://github.com/ruvnet/ruvector)*
