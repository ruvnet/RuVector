//! Run configuration (`configs/openjev-small-v0.toml`; plan Step 3).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::data::{BANKING77, CLINC150, DATASETS, HWU64, TICKETS};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub max_seq_len: usize,
    pub train: TrainCfg,
    /// Dataset mix weights; a weight of 0 drops the dataset (tickets-only ablation).
    pub mix: BTreeMap<String, f64>,
    pub batch: BatchCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainCfg {
    pub max_steps: usize,
    pub eval_every: usize,
    /// Early stop after this many evals without validation improvement.
    pub patience: usize,
    pub warmup_steps: usize,
    pub encoder_lr: f64,
    pub head_lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub adam_eps: f64,
    pub grad_clip: f64,
    pub label_smoothing: f64,
    pub supcon_temperature: f64,
    pub head_scale: f64,
    /// Outlier-exposure weight on CLINC150 `oos_train` rows.
    pub oe_weight: f64,
    /// Soft target for ambiguous tickets (primary; secondary gets 1 − this).
    pub ambiguous_primary: f32,
    pub eval_batch: usize,
    /// Cap validation rows per dataset (0 = all). Smoke runs only.
    #[serde(default)]
    pub val_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchCfg {
    pub tickets_p: usize,
    pub tickets_k: usize,
    pub public_p: usize,
    pub public_k: usize,
    pub clinc_oos: usize,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let s =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let c: Config = toml::from_str(&s).with_context(|| format!("parse {}", path.display()))?;
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<()> {
        for k in self.mix.keys() {
            if !DATASETS.contains(&k.as_str()) {
                bail!("mix: unknown dataset {k}");
            }
        }
        if self.active().is_empty() {
            bail!("mix: every weight is 0");
        }
        if self.mix.values().any(|w| *w < 0.0 || !w.is_finite()) {
            bail!("mix weights must be finite and >= 0");
        }
        if self.train.eval_every == 0 || self.train.max_steps == 0 {
            bail!("max_steps and eval_every must be > 0");
        }
        if !(0.5..=1.0).contains(&self.train.ambiguous_primary) {
            bail!("ambiguous_primary must be in [0.5, 1]");
        }
        Ok(())
    }

    /// Datasets with a positive mix weight, in canonical order.
    pub fn active(&self) -> Vec<&'static str> {
        DATASETS
            .iter()
            .copied()
            .filter(|d| self.mix.get(*d).copied().unwrap_or(0.0) > 0.0)
            .collect()
    }

    /// (P, K) for one dataset's batch.
    pub fn pk(&self, dataset: &str) -> (usize, usize) {
        if dataset == TICKETS {
            (self.batch.tickets_p, self.batch.tickets_k)
        } else {
            (self.batch.public_p, self.batch.public_k)
        }
    }

    /// The v0 defaults (identical to `configs/openjev-small-v0.toml`).
    pub fn v0() -> Self {
        let mix = [
            (TICKETS, 0.20),
            (BANKING77, 0.25),
            (CLINC150, 0.30),
            (HWU64, 0.25),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        Self {
            max_seq_len: 64,
            train: TrainCfg {
                max_steps: 3000,
                eval_every: 250,
                patience: 4,
                warmup_steps: 180,
                encoder_lr: 3e-5,
                head_lr: 1e-3,
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                adam_eps: 1e-8,
                grad_clip: 1.0,
                label_smoothing: 0.1,
                supcon_temperature: 0.05,
                head_scale: 20.0,
                oe_weight: 0.5,
                ambiguous_primary: 0.75,
                eval_batch: 128,
                val_limit: 0,
            },
            mix,
            batch: BatchCfg {
                tickets_p: 8,
                tickets_k: 4,
                public_p: 32,
                public_k: 2,
                clinc_oos: 16,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_config_equals_v0_defaults() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/openjev-small-v0.toml");
        let c = Config::load(&p).unwrap();
        assert_eq!(
            serde_json::to_value(&c).unwrap(),
            serde_json::to_value(Config::v0()).unwrap()
        );
    }

    #[test]
    fn tickets_only_ablation_is_valid() {
        let mut c = Config::v0();
        for d in [BANKING77, CLINC150, HWU64] {
            c.mix.insert(d.into(), 0.0);
        }
        c.validate().unwrap();
        assert_eq!(c.active(), vec![TICKETS]);
    }
}
