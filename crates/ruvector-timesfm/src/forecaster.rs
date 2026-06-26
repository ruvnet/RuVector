//! The high-level [`Forecaster`] — load TimesFM weights once, forecast many.

use std::path::Path;

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Tensor};
use timesfm::config::TimesfmConfig;
use timesfm::model::PatchedTimeSeriesDecoder;
use timesfm::prune::{decide_prune, PruneDecision};

use crate::forecast_types::{Forecast, NUM_QUANTILES};
use crate::{Error, Result};

/// Weight quantization for loading. `Q8_0` (int8) shrinks the model ~4×
/// (~212 MB) with good accuracy (rel error ~3e-3); `Q4_0` (int4) ~7× (~112 MB)
/// with more error (~3e-2). These are **memory** wins for edge deployment — on a
/// small-context model like TimesFM-200M, quantized CPU matmuls are *slower*
/// than f32 (dequant overhead dominates), so prefer f32/GPU for latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// int8 weights — ~4× smaller, ~3e-3 relative error.
    Q8_0,
    /// int4 weights — ~7× smaller, ~3e-2 relative error.
    Q4_0,
}

impl Quant {
    fn dtype(self) -> GgmlDType {
        match self {
            Quant::Q8_0 => GgmlDType::Q8_0,
            Quant::Q4_0 => GgmlDType::Q4_0,
        }
    }
}

/// A loaded TimesFM 1.0 200M forecaster bound to a compute device.
///
/// Construct once (weight load + mmap is the expensive part), then call
/// [`Forecaster::forecast`] repeatedly.
pub struct Forecaster {
    model: PatchedTimeSeriesDecoder,
    device: Device,
    /// Activation dtype the forward runs in (F32 by default, F16 for `load_f16`).
    dtype: DType,
}

impl Forecaster {
    /// Load TimesFM 1.0 200M weights (a converted `safetensors` file) onto
    /// `device`. Use [`timesfm::select_device`] to pick CPU/cuda/metal from the
    /// `TIMESFM_DEVICE` env var.
    pub fn load(weights: impl AsRef<Path>, device: Device) -> Result<Self> {
        let cfg = TimesfmConfig::timesfm_1p0_200m();
        Self::load_with_config(weights, cfg, device)
    }

    /// Load with an explicit [`TimesfmConfig`] (e.g. a test/tiny variant).
    pub fn load_with_config(
        weights: impl AsRef<Path>,
        cfg: TimesfmConfig,
        device: Device,
    ) -> Result<Self> {
        let path = weights.as_ref();
        if !path.exists() {
            return Err(Error::Invalid(format!(
                "weights file not found: {}",
                path.display()
            )));
        }
        // SAFETY: from_mmaped_safetensors is unsafe because it mmaps a file the
        // caller asserts is a valid safetensors blob; load() validates existence
        // and candle validates the header/dtypes on read.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[path.to_path_buf()],
                DType::F32,
                &device,
            )?
        };
        let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;
        Ok(Self {
            model,
            device,
            dtype: DType::F32,
        })
    }

    /// Load weights as **f16** and run the forward in f16. On GPU this can cut
    /// latency; on CPU it is typically slower (f16 emulation). Forecasts match
    /// f32 only to ~f16 precision (rel error ~1e-3).
    pub fn load_f16(weights: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = weights.as_ref();
        if !path.exists() {
            return Err(Error::Invalid(format!(
                "weights file not found: {}",
                path.display()
            )));
        }
        let cfg = TimesfmConfig::timesfm_1p0_200m();
        // SAFETY: see load_with_config. Weights are cast to f16 on load.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[path.to_path_buf()],
                DType::F16,
                &device,
            )?
        };
        let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;
        Ok(Self {
            model,
            device,
            dtype: DType::F16,
        })
    }

    /// Load with weights quantized to int8/int4 ([`Quant`]) — a ~4–7× smaller
    /// resident model for memory-constrained (edge/Pi) deployment. See [`Quant`]
    /// for the accuracy/latency tradeoff.
    pub fn load_quantized(weights: impl AsRef<Path>, device: Device, quant: Quant) -> Result<Self> {
        let path = weights.as_ref();
        if !path.exists() {
            return Err(Error::Invalid(format!(
                "weights file not found: {}",
                path.display()
            )));
        }
        let cfg = TimesfmConfig::timesfm_1p0_200m();
        // SAFETY: see load_with_config.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[path.to_path_buf()],
                DType::F32,
                &device,
            )?
        };
        let model = PatchedTimeSeriesDecoder::load_quantized(cfg, vb, quant.dtype())?;
        Ok(Self {
            model,
            device,
            dtype: DType::F32,
        })
    }

    /// The device this forecaster runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The underlying model (for callers that need the raw decode API).
    #[must_use]
    pub fn model(&self) -> &PatchedTimeSeriesDecoder {
        &self.model
    }

    /// Forecast `horizon` steps ahead of `series`, using the default (finest)
    /// frequency bucket. Returns point + quantile bands.
    pub fn forecast(&self, series: &[f32], horizon: usize) -> Result<Forecast> {
        self.forecast_with_freq(series, horizon, 0)
    }

    /// Forecast `horizon` steps with an explicit frequency id (0 = high/fine,
    /// 1 = medium, 2 = low — TimesFM's frequency buckets).
    pub fn forecast_with_freq(
        &self,
        series: &[f32],
        horizon: usize,
        freq_id: u32,
    ) -> Result<Forecast> {
        if series.is_empty() {
            return Err(Error::Invalid("series must be non-empty".into()));
        }
        if horizon == 0 {
            return Err(Error::Invalid("horizon must be > 0".into()));
        }

        let k = series.len();
        let input_ts =
            Tensor::from_vec(series.to_vec(), (1, k), &self.device)?.to_dtype(self.dtype)?;
        let input_padding = Tensor::zeros((1, k), self.dtype, &self.device)?;
        let freq = Tensor::from_vec(vec![freq_id], (1, 1), &self.device)?;

        // (point [1, h], full [1, h, num_outputs]); channel 0 = mean, 1..=9 = p10..p90.
        let (point_t, full_t) = self
            .model
            .decode(&input_ts, &input_padding, &freq, horizon)?;

        // Outputs come back in the forward dtype; surface f32 to callers.
        let point: Vec<f32> = point_t.i(0)?.to_dtype(DType::F32)?.to_vec1()?;
        let full: Vec<Vec<f32>> = full_t.i(0)?.to_dtype(DType::F32)?.to_vec2()?; // [h][num_outputs]

        let quantiles: Vec<[f32; NUM_QUANTILES]> = full
            .iter()
            .map(|row| {
                let mut q = [0f32; NUM_QUANTILES];
                // row[0] is the mean; quantiles live at indices 1..=9.
                for (j, slot) in q.iter_mut().enumerate() {
                    *slot = row.get(j + 1).copied().unwrap_or(row[0]);
                }
                q
            })
            .collect();

        Ok(Forecast { point, quantiles })
    }

    /// Forecast a **batch** of equal-length series in a single model call —
    /// the throughput path. All series must share one length (the common case
    /// for windowed telemetry / multi-series dashboards); returns one
    /// [`Forecast`] per input row. For ragged lengths, call [`Self::forecast`]
    /// per series.
    pub fn forecast_batch(
        &self,
        series_batch: &[Vec<f32>],
        horizon: usize,
        freq_id: u32,
    ) -> Result<Vec<Forecast>> {
        if series_batch.is_empty() {
            return Err(Error::Invalid("series batch is empty".into()));
        }
        if horizon == 0 {
            return Err(Error::Invalid("horizon must be > 0".into()));
        }
        let k = series_batch[0].len();
        if k == 0 {
            return Err(Error::Invalid("series must be non-empty".into()));
        }
        if series_batch.iter().any(|s| s.len() != k) {
            return Err(Error::Invalid(
                "forecast_batch requires equal-length series; use forecast() per series for ragged input".into(),
            ));
        }
        let b = series_batch.len();
        let flat: Vec<f32> = series_batch.iter().flatten().copied().collect();
        let input_ts = Tensor::from_vec(flat, (b, k), &self.device)?.to_dtype(self.dtype)?;
        let input_padding = Tensor::zeros((b, k), self.dtype, &self.device)?;
        let freq = Tensor::from_vec(vec![freq_id; b], (b, 1), &self.device)?;

        let (point_t, full_t) = self
            .model
            .decode(&input_ts, &input_padding, &freq, horizon)?;
        let point_t = point_t.to_dtype(DType::F32)?;
        let full_t = full_t.to_dtype(DType::F32)?;

        let mut out = Vec::with_capacity(b);
        for row in 0..b {
            let point: Vec<f32> = point_t.i(row)?.to_vec1()?;
            let full: Vec<Vec<f32>> = full_t.i(row)?.to_vec2()?;
            let quantiles: Vec<[f32; NUM_QUANTILES]> = full
                .iter()
                .map(|r| {
                    let mut q = [0f32; NUM_QUANTILES];
                    for (j, slot) in q.iter_mut().enumerate() {
                        *slot = r.get(j + 1).copied().unwrap_or(r[0]);
                    }
                    q
                })
                .collect();
            out.push(Forecast { point, quantiles });
        }
        Ok(out)
    }

    /// PRUNE/CONTINUE decision for a partial optimization curve (lower = better),
    /// forecasting toward `target_iters` total against a viability `threshold`.
    /// Thin pass-through to [`timesfm::prune::decide_prune`] using this
    /// forecaster's model + device; see [`crate::sweep`] for the gated wrapper.
    pub fn prune_decision(
        &self,
        curve: &[f32],
        target_iters: usize,
        threshold: f32,
    ) -> Result<PruneDecision> {
        if curve.is_empty() {
            return Err(Error::Invalid("curve must be non-empty".into()));
        }
        Ok(decide_prune(
            &self.model,
            curve,
            target_iters,
            threshold,
            &self.device,
        )?)
    }
}
