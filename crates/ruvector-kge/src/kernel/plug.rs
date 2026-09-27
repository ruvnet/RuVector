//! The GEMM kernel as the recipe lane's [`OneToN`] (plan M1 → M2 seam,
//! `train/one_to_n.rs`). Contract, as that trait specifies:
//!
//! - `grad_q` (`B×D`) is **overwritten** with `scale · dL/dQ`;
//! - `grad_ent` (`N×D`) is **accumulated into** (`+=`) with `scale · dL/dE`
//!   (the grad-E GEMM runs with `beta = 1`: no extra `N×D` buffer or pass);
//! - the return value is the **unscaled** summed loss.
//!
//! Select it with `Trainer::fit_with_kernel(.., &GemmOneToN::new(), ..)`;
//! `Trainer::fit` keeps `NaiveOneToN`. Integration TODO: a `TrainConfig`
//! field (e.g. `one_n_kernel: naive | gemm`) so the bindings/HPO can choose
//! it from JSON — left to the Integrate stage because `train/config.rs` and
//! `train/mod.rs` belong to the recipe lane.

use super::{softmax_ce_core, Workspace};
use crate::train::OneToN;
use crate::Result;
use std::sync::Mutex;

/// The GEMM 1-N kernel behind the `&self` [`OneToN`] interface. The `B×N`
/// logits scratch lives in a `Mutex` so it is reused across batches (58 MB at
/// FB15k-237 scale); one instance serves one batch at a time.
#[derive(Debug, Default)]
pub struct GemmOneToN {
    ws: Mutex<Workspace>,
}

impl GemmOneToN {
    pub fn new() -> Self {
        Self::default()
    }
}

impl OneToN for GemmOneToN {
    fn softmax_ce(
        &self,
        q: &[f32],
        ent: &[f32],
        dim: usize,
        targets: &[u32],
        scale: f32,
        grad_q: &mut [f32],
        grad_ent: &mut [f32],
    ) -> Result<f32> {
        let mut ws = self.ws.lock().unwrap_or_else(|p| p.into_inner());
        let sum = softmax_ce_core(q, ent, dim, targets, scale, true, &mut ws, grad_q, grad_ent)?;
        Ok(sum as f32)
    }
}
