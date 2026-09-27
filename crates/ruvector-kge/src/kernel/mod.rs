//! Batched 1-N (1-vs-all) softmax cross-entropy kernel as three GEMMs
//! (ADR-007 §4, plan M2).
//!
//! Per batch of `B` query rows against the whole entity table `E` (`N` rows):
//!
//! ```text
//! logits = Q·Eᵀ                          [B×N]
//! loss_b = logsumexp(logits_b) − logits_b[target_b]
//! dL     = scale · (softmax(logits) − onehot(target))
//! grad_E = dLᵀ·Q     grad_Q = dL·E
//! ```
//!
//! with `scale = 1` ([`Reduction::Sum`]) or `1/B` ([`Reduction::Mean`]).
//!
//! The core ([`one_n_softmax_ce`]) is scorer-agnostic: it sees only dense
//! `Q` and `E` buffers of a common row width `dim`. Two front-ends map a model
//! onto it and chain the query gradient back to the embeddings:
//!
//! - [`complex`] — ComplEx form, rows `[re; im]` of `k` complex dims
//!   (`dim = 2k`). `E` *is* the entity table, so no transform is needed.
//! - [`hole`] — HolE through its frequency view (ADR-002 §2, HolE ≡ ComplEx):
//!   `E` is the weighted half-spectrum index table (`dim = d + 2`, exactly
//!   [`crate::BatchScorer`]'s rows), `Q` is [`crate::Scorer::query_vector`],
//!   and both gradients are pulled back to the real `d`-vectors with one
//!   inverse FFT per row (the adjoint of the index / query maps).
//!
//! **Determinism.** Fixed-size chunking of output rows/columns only; per-row
//! losses are summed sequentially in row order. Results are bitwise identical
//! across runs and across thread counts (see `gemm.rs`, asserted in tests).
//!
//! **Threads.** Under the `parallel` feature the kernel runs inside the
//! caller's current rayon pool (`ThreadPool::install` to pin a count).
//!
//! **Integration TODO (plan M2 → Integrate stage).** Route
//! `LossKind::OneVsAll` in `train/loss.rs` / `train/mod.rs` to
//! [`complex::complex_one_n_step`] (ComplEx) or [`hole::HolEKernel::step`]
//! (HolE) behind a `TrainConfig` switch, one call per mini-batch with both
//! directions (or the reciprocal tail rows) as query rows, and feed the dense
//! `grad_entities` / `grad_relations` buffers to the dense optimizer state
//! the recipe lane (plan M1, `optim.rs`) adds. The per-triple
//! `one_vs_all_step` stays as the test oracle. Multi-label KvsAll targets and
//! the RP auxiliary loss are follow-ups; this kernel is single-target CE.

pub mod complex;
mod gemm;
pub mod hole;
mod plug;

pub use plug::GemmOneToN;

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_plug;

use crate::{EntityId, KgeError, RelationId, Result, Side};

/// How the per-row losses are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// Sum over rows (what the per-triple `one_vs_all_step` returns).
    #[default]
    Sum,
    /// Mean over rows (ssl-RP `loss_reduction: mean`).
    Mean,
}

/// One 1-N query row: `(anchor, relation, ?)` with the open slot on `side`,
/// whose true answer is `target`. Reciprocal training uses `Side::Tail` rows
/// with the inverse relation id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OneNQuery {
    pub anchor: EntityId,
    pub relation: RelationId,
    pub side: Side,
    pub target: EntityId,
}

/// Reusable scratch for [`one_n_softmax_ce`] (the `B×N` logits buffer is the
/// big one: 58 MB at B=1000, N=14,541). Grows, never shrinks.
#[derive(Debug, Default)]
pub struct Workspace {
    pub(crate) logits: Vec<f32>,
    pub(crate) row_loss: Vec<f32>,
}

impl Workspace {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Rows per softmax work item (fixed — determinism does not depend on it,
/// each row is independent, but keep it constant anyway).
const SOFTMAX_ROWS: usize = 4;

/// The kernel core: softmax cross-entropy of `B` query rows against all `N`
/// entity rows, with gradients.
///
/// - `queries`: `B×dim` row-major; `entities`: `N×dim` row-major.
/// - `targets[b]` is the true entity of row `b` (`< N`).
/// - Writes `grad_queries` (`B×dim`) and `grad_entities` (`N×dim`, dense,
///   overwritten — not accumulated). Returns the reduced loss.
///
/// Errors on shape mismatch, an out-of-range target, or a non-finite loss.
#[allow(clippy::too_many_arguments)]
pub fn one_n_softmax_ce(
    queries: &[f32],
    entities: &[f32],
    dim: usize,
    targets: &[u32],
    reduction: Reduction,
    ws: &mut Workspace,
    grad_queries: &mut [f32],
    grad_entities: &mut [f32],
) -> Result<f32> {
    let scale = match reduction {
        Reduction::Sum => 1.0f32,
        Reduction::Mean => 1.0 / targets.len().max(1) as f32,
    };
    let sum = softmax_ce_core(
        queries,
        entities,
        dim,
        targets,
        scale,
        false,
        ws,
        grad_queries,
        grad_entities,
    )?;
    Ok((sum * scale as f64) as f32)
}

/// Shared core. Gradients carry `scale`; `grad_entities` is overwritten, or
/// accumulated into when `accumulate`. Returns the **unscaled** summed loss.
#[allow(clippy::too_many_arguments)]
pub(crate) fn softmax_ce_core(
    queries: &[f32],
    entities: &[f32],
    dim: usize,
    targets: &[u32],
    scale: f32,
    accumulate: bool,
    ws: &mut Workspace,
    grad_queries: &mut [f32],
    grad_entities: &mut [f32],
) -> Result<f64> {
    if dim == 0 {
        return Err(KgeError::Invalid("kernel: dim must be > 0".into()));
    }
    let b = targets.len();
    if queries.len() != b * dim {
        return Err(KgeError::Dims {
            expected: b * dim,
            got: queries.len(),
        });
    }
    if !entities.len().is_multiple_of(dim) {
        return Err(KgeError::Invalid(
            "kernel: entity buffer is not a whole number of rows".into(),
        ));
    }
    let n = entities.len() / dim;
    if grad_queries.len() != b * dim {
        return Err(KgeError::Dims {
            expected: b * dim,
            got: grad_queries.len(),
        });
    }
    if grad_entities.len() != n * dim {
        return Err(KgeError::Dims {
            expected: n * dim,
            got: grad_entities.len(),
        });
    }
    if let Some(&t) = targets.iter().find(|&&t| t as usize >= n) {
        return Err(KgeError::UnknownEntity(t));
    }
    if !scale.is_finite() {
        return Err(KgeError::Invalid("kernel: non-finite scale".into()));
    }
    if b == 0 {
        if !accumulate {
            grad_entities.fill(0.0);
        }
        return Ok(0.0);
    }

    ws.logits.resize(b * n, 0.0);
    ws.row_loss.resize(b, 0.0);
    let logits = &mut ws.logits[..b * n];

    gemm::logits_nt(queries, entities, b, n, dim, logits);
    softmax_ce_rows(logits, n, targets, scale, &mut ws.row_loss[..b]);

    // Sequential, row-ordered sum: deterministic regardless of threads.
    let mut loss = 0.0f64;
    for &l in &ws.row_loss[..b] {
        loss += l as f64;
    }
    if !loss.is_finite() {
        return Err(KgeError::Invalid("kernel: non-finite loss".into()));
    }

    let dl = &ws.logits[..b * n];
    let beta = if accumulate { 1.0 } else { 0.0 };
    gemm::grad_entities_tn(dl, queries, b, n, dim, beta, grad_entities);
    gemm::grad_queries_nn(dl, entities, b, n, dim, grad_queries);
    Ok(loss)
}

/// Forward only: `out[b×n] = queries · entitiesᵀ` (evaluation / ranking).
pub fn one_n_scores(queries: &[f32], entities: &[f32], dim: usize, out: &mut [f32]) -> Result<()> {
    if dim == 0 || !queries.len().is_multiple_of(dim) || !entities.len().is_multiple_of(dim) {
        return Err(KgeError::Invalid(
            "kernel: buffers are not whole rows".into(),
        ));
    }
    let (b, n) = (queries.len() / dim, entities.len() / dim);
    if out.len() != b * n {
        return Err(KgeError::Dims {
            expected: b * n,
            got: out.len(),
        });
    }
    gemm::logits_nt(queries, entities, b, n, dim, out);
    Ok(())
}

/// In place: each logits row becomes `scale·(softmax − onehot)`, and
/// `row_loss[b] = logsumexp − logit[target]` (unscaled).
fn softmax_ce_rows(
    logits: &mut [f32],
    n: usize,
    targets: &[u32],
    scale: f32,
    row_loss: &mut [f32],
) {
    let row = |lg: &mut [f32], t: usize, out: &mut f32| {
        let mut m = f32::NEG_INFINITY;
        for &x in lg.iter() {
            m = m.max(x);
        }
        let target_logit = lg[t];
        let mut sum = 0.0f32;
        for x in lg.iter_mut() {
            let e = (*x - m).exp();
            *x = e;
            sum += e;
        }
        *out = (m + sum.ln()) - target_logit;
        let inv = scale / sum;
        for x in lg.iter_mut() {
            *x *= inv;
        }
        lg[t] -= scale;
    };
    let work = |lgs: &mut [f32], ts: &[u32], outs: &mut [f32]| {
        for ((lg, &t), out) in lgs.chunks_mut(n).zip(ts).zip(outs.iter_mut()) {
            row(lg, t as usize, out);
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        logits
            .par_chunks_mut(n * SOFTMAX_ROWS)
            .zip(targets.par_chunks(SOFTMAX_ROWS))
            .zip(row_loss.par_chunks_mut(SOFTMAX_ROWS))
            .for_each(|((lg, ts), out)| work(lg, ts, out));
    }
    #[cfg(not(feature = "parallel"))]
    for ((lg, ts), out) in logits
        .chunks_mut(n * SOFTMAX_ROWS)
        .zip(targets.chunks(SOFTMAX_ROWS))
        .zip(row_loss.chunks_mut(SOFTMAX_ROWS))
    {
        work(lg, ts, out);
    }
}
