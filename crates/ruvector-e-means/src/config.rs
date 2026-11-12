use crate::Error;

/// Final centroid estimator. Slow-EMA finalization is reserved for a future API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum Finalization {
    #[default]
    LastEpochMean,
    Ema,
}

/// Fit options. Overrides are useful for tiny or low-dimensional datasets where
/// the paper's empirically fitted scaling laws are outside their training range.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(rename_all = "camelCase", deny_unknown_fields)
)]
pub struct Config {
    pub clusters: usize,
    #[cfg_attr(feature = "serde", serde(default = "default_epochs"))]
    pub epochs: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    pub batch_size: Option<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub peak_step: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default = "default_seed"))]
    pub seed: u64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub finalization: Finalization,
}

fn default_epochs() -> usize {
    3
}

fn default_seed() -> u64 {
    42
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clusters: 8,
            epochs: default_epochs(),
            batch_size: None,
            peak_step: None,
            seed: default_seed(),
            finalization: Finalization::LastEpochMean,
        }
    }
}

/// Resolved scaling-law parameters, inspectable before allocating a model.
#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub batch_size: usize,
    pub peak_step: f64,
    pub updates: usize,
}

impl Config {
    /// Table 2 and equations (6)-(7) of arXiv:2611.09501v1. Batch law uses
    /// coefficient 0.035, divided by 8 above two million samples, rounded to
    /// the nearest power of two and capped at n. Tail batches are included.
    pub fn parameters(&self, samples: usize) -> Result<Parameters, Error> {
        if samples == 0 || self.clusters == 0 || self.clusters > samples {
            return Err(Error("clusters must be in 1..=number of samples"));
        }
        if self.epochs == 0 {
            return Err(Error("epochs must be positive"));
        }
        let n = samples as f64;
        let k = self.clusters as f64;
        let t = self.epochs as f64;
        let batch_size = match self.batch_size {
            Some(0) => return Err(Error("batch size must be positive")),
            Some(b) => b.min(samples),
            None => {
                let coefficient = if samples > 2_000_000 {
                    0.035 / 8.0
                } else {
                    0.035
                };
                let law = coefficient * k.powf(0.13) * n.powf(0.88) * t.powf(0.81);
                2.0_f64.powf(law.max(1.0).log2().round()).min(n) as usize
            }
        };
        let peak_step = self.peak_step.unwrap_or_else(|| {
            (0.824 * k.powf(-0.46) * (batch_size as f64).powf(0.90) * n.powf(-0.45) * t.powf(-0.11))
                .min(1.0)
        });
        if !peak_step.is_finite() || peak_step <= 0.0 || peak_step > 1.0 {
            return Err(Error("peak step must be finite and in (0, 1]"));
        }
        let batches = samples / batch_size + usize::from(samples % batch_size != 0);
        let updates = batches
            .checked_mul(self.epochs)
            .ok_or(Error("update count overflow"))?;
        Ok(Parameters {
            batch_size,
            peak_step,
            updates,
        })
    }
}

/// Equation (5), zero-based batch index t and total M updates. Its denominator
/// is M as printed in the paper: the last update approaches the 0.1 floor.
pub fn cosine_step(peak_step: f64, update: usize, total_updates: usize) -> f64 {
    let fraction = (update as f64) / (total_updates.max(1) as f64);
    peak_step * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos()))
}
