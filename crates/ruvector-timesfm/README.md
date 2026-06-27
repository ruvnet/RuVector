# ruvector-timesfm

RuVector-facing integration for the [`timesfm`](https://crates.io/crates/timesfm)
TimesFM 1.0 200M time-series foundation model. The base crate is a faithful,
parity-validated candle port of the model; this crate wraps it in the things
RuVector and ruflo actually call.

[![Crates.io](https://img.shields.io/crates/v/ruvector-timesfm)](https://crates.io/crates/ruvector-timesfm)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

---

## What you get

| Module | Purpose |
|---|---|
| [`Forecaster`] | Load weights once, `forecast(series, horizon)` → point + calibrated **p10..p90 quantile bands**; `forecast_batch` for throughput. |
| [`anomaly`] | Forecast-band anomaly detection — flag observed points that fall outside their p10/p90 band (host/vector-db telemetry: disk-fill, GPU memory, query load). |
| [`sweep::EarlyStopper`] | TimesFM-driven early stopping for optimization sweeps (ADR-191) — kill doomed ruflo/Darwin runs early, with a `min_history` warm-up + confidence gate. |
| [`rebuild`] | Forecast an index's recall-drift curve and advise *when* to rebuild an HNSW index — just before the conservative (p10) forecast crosses a recall floor. |
| `ruvector-timesfm-forecast` | A JSON-in/JSON-out CLI = the `time_series_forecast` MCP tool entry point. |

## Feature gating

The numeric path is behind the **`candle`** feature (and `cuda`/`metal`, which
imply it), mirroring `timesfm`. Without it, only the plain data types compile, so
a stock `cargo build` stays light.

```toml
[dependencies]
ruvector-timesfm = { version = "2.2", features = ["candle"] }
```

## Quick start

```rust,ignore
use ruvector_timesfm::Forecaster;

// Pick CPU / cuda / metal via the TIMESFM_DEVICE env var.
let f = Forecaster::load("/path/timesfm.safetensors", timesfm::select_device()?)?;

let forecast = f.forecast(&history, 64)?;       // 64-step forecast
let (lo, mid, hi) = (forecast.p10(), forecast.p50(), forecast.p90());

// Forecast-band anomaly detection on an observed window:
let report = f.detect_anomalies(&history, &observed, 0)?;
println!("{} anomalies", report.n_anomalies);
```

## Precision knobs

| API | Use | Tradeoff (measured, real weights) |
|---|---|---|
| `Forecaster::load` (f32) | default | reference accuracy; CPU ~45 ms, cuda ~4 ms / forecast |
| `Forecaster::load_f16` | GPU latency | ~1.6× faster batched on GPU; rel error ~2e-2 (CPU f16 is slower) |
| `Forecaster::load_quantized(Quant::Q8_0)` | edge memory | ~4× smaller (~212 MB), rel error ~3e-3; CPU slower (dequant) |
| `Quant::Q4_0` | tightest memory | ~7× smaller (~112 MB), rel error ~3e-2 |

## MCP tool

`ruvector-timesfm-forecast` reads a JSON request on stdin and writes a forecast
on stdout — the shell-out entry point for the RuVector `time_series_forecast`
MCP tool:

```bash
echo '{"weights":"/path/timesfm.safetensors","series":[...],"horizon":32}' \
  | ruvector-timesfm-forecast
# → {"horizon":32,"device":"cpu","point":[...],"p10":[...],"p50":[...],"p90":[...]}
```

## Weights

Weights are not bundled — download `google/timesfm-1.0-200m` from HuggingFace and
convert with `timesfm`'s `scripts/convert_weights.py` (PyTorch state_dict →
candle safetensors). See the [`timesfm`](https://crates.io/crates/timesfm) crate.

## License

Apache-2.0.
