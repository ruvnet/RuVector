//! OpenJev v0 trainer (ADR-008, `docs/research/openjev/v0-plan.md`).
//!
//! `prep` → `train` → `transplant` → `parity`. Rust only; candle for training,
//! the engine's own `ort` path for the parity proof.

pub mod config;
pub mod data;
pub mod embed;
pub mod export;
pub mod gpu;
pub mod leakage;
pub mod loss;
pub mod model;
pub mod norm;
pub mod optim;
pub mod parity;
pub mod pins;
pub mod prep;
pub mod sampler;
pub mod train;

use anyhow::Result;
use candle_core::Device;

/// `cpu`, `cuda`, or `cuda:N`.
pub fn device(spec: &str) -> Result<Device> {
    match spec {
        "cpu" => Ok(Device::Cpu),
        s if s == "cuda" || s.starts_with("cuda:") => {
            let n = s
                .strip_prefix("cuda:")
                .map(str::parse)
                .transpose()?
                .unwrap_or(0);
            Ok(Device::new_cuda(n)?)
        }
        other => anyhow::bail!("unknown device {other:?} (cpu | cuda | cuda:N)"),
    }
}
