//! The batched 1-N training boundary (plan M1 → M2).
//!
//! For a [`Bilinear`] scorer with an identity index (ComplEx), 1-vs-all
//! cross-entropy over a batch is: build the `B × D` query matrix `Q`, score
//! `logits = Q · Eᵀ` against the raw entity table `E` (`N × D`), take a
//! row-wise softmax-CE against each query's target entity, and backpropagate
//! `grad_Q = dL · E`, `grad_E = dLᵀ · Q`. Everything except the middle step is
//! scorer-specific and lives here; the middle step is the [`OneToN`] trait —
//! the plug point for the kernel lane's GEMM (`train/kernel.rs`, M2).
//! [`NaiveOneToN`] is the fixed-order reference implementation and oracle.

use super::optim::Grads;
use crate::scorer::Bilinear;
use crate::{KgeError, Result, Side, Tables, Triple};

/// Batched softmax cross-entropy over all entities for `B` queries.
///
/// Shapes (row-major, `D = dim`, `N = ent.len() / D`, `B = targets.len()`):
/// - `q`: `B × D` queries (input).
/// - `ent`: `N × D` candidate matrix (the raw entity table; input).
/// - `targets`: the true entity per query, each `< N`.
/// - `grad_q`: `B × D`, **overwritten** with `scale · dL/dQ`.
/// - `grad_ent`: `N × D`, **accumulated into** (`+=`) with `scale · dL/dE`
///   (it already holds other terms of the batch gradient).
///
/// Returns the **unscaled** summed loss `Σ_b −log softmax(Q_b·Eᵀ)[target_b]`.
/// An implementation must be deterministic for a fixed input (fixed-order
/// reductions), and must agree with [`NaiveOneToN`] within 1e-4 (M2 gate).
pub trait OneToN {
    #[allow(clippy::too_many_arguments)]
    fn softmax_ce(
        &self,
        q: &[f32],
        ent: &[f32],
        dim: usize,
        targets: &[u32],
        scale: f32,
        grad_q: &mut [f32],
        grad_ent: &mut [f32],
    ) -> Result<f32>;
}

/// Check the [`OneToN`] shape contract (for implementations to share).
pub fn check_shapes(
    q: &[f32],
    ent: &[f32],
    dim: usize,
    targets: &[u32],
    grad_q: &[f32],
    grad_ent: &[f32],
) -> Result<usize> {
    let bad = |m: &str| Err(KgeError::Invalid(format!("one_to_n: {m}")));
    if dim == 0 || !ent.len().is_multiple_of(dim) {
        return bad("ent length is not a multiple of dim");
    }
    let n = ent.len() / dim;
    if q.len() != targets.len() * dim || grad_q.len() != q.len() {
        return bad("q / grad_q must be B x dim");
    }
    if grad_ent.len() != ent.len() {
        return bad("grad_ent must match ent");
    }
    if targets.iter().any(|&t| t as usize >= n) {
        return bad("target out of range");
    }
    Ok(n)
}

/// Reference 1-N: plain loops, fixed order (queries, then entities, then
/// coordinates). `O(B·N·D)`, no allocation beyond one `N`-vector.
#[derive(Debug, Default, Clone, Copy)]
pub struct NaiveOneToN;

impl OneToN for NaiveOneToN {
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
        let n = check_shapes(q, ent, dim, targets, grad_q, grad_ent)?;
        let mut logits = vec![0.0f32; n];
        let mut loss = 0.0f32;
        for (b, &target) in targets.iter().enumerate() {
            let qb = &q[b * dim..(b + 1) * dim];
            for (e, l) in logits.iter_mut().enumerate() {
                *l = dot(qb, &ent[e * dim..(e + 1) * dim]);
            }
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for l in logits.iter_mut() {
                *l = (*l - m).exp();
                sum += *l;
            }
            let inv = 1.0 / sum;
            let p_t = logits[target as usize] * inv;
            loss += -(p_t.max(1e-30)).ln();
            let gq = &mut grad_q[b * dim..(b + 1) * dim];
            gq.fill(0.0);
            for (e, &l) in logits.iter().enumerate() {
                let p = l * inv;
                let coeff = scale * (p - if e as u32 == target { 1.0 } else { 0.0 });
                if coeff == 0.0 {
                    continue;
                }
                let row = &ent[e * dim..(e + 1) * dim];
                let ge = &mut grad_ent[e * dim..(e + 1) * dim];
                for j in 0..dim {
                    gq[j] += coeff * row[j];
                    ge[j] += coeff * qb[j];
                }
            }
        }
        Ok(loss)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One batched 1-N step over `examples`. Tail queries `(s, r, ?)` always;
/// also head queries `(?, r, o)` when `both_sides` (non-reciprocal training).
/// `grads` must be dense (see [`Grads::dense_entities_mut`]); its scale is
/// applied. Returns the unscaled summed loss.
pub(crate) fn batched_step(
    tables: &Tables,
    scorer: &dyn Bilinear,
    kernel: &dyn OneToN,
    examples: &[Triple],
    both_sides: bool,
    grads: &mut Grads,
) -> Result<f32> {
    let d = scorer.dims();
    if !scorer.index_is_identity() || scorer.index_dims() != d || tables.dims() != d {
        return Err(KgeError::Invalid(
            "batched 1-N needs an identity-index scorer matching the tables".into(),
        ));
    }
    // (anchor id, relation id, side, target)
    let mut queries: Vec<(u32, u32, Side, u32)> = Vec::with_capacity(examples.len() * 2);
    for &t in examples {
        queries.push((t.s, t.r, Side::Tail, t.o));
        if both_sides {
            queries.push((t.o, t.r, Side::Head, t.s));
        }
    }
    let b = queries.len();
    let mut q = vec![0.0f32; b * d];
    let mut targets = Vec::with_capacity(b);
    for (i, &(a, r, side, target)) in queries.iter().enumerate() {
        tables.entity(target)?;
        scorer.query_into(
            tables.relation(r)?,
            tables.entity(a)?,
            side,
            &mut q[i * d..(i + 1) * d],
        );
        targets.push(target);
    }
    let scale = grads.scale();
    let mut grad_q = vec![0.0f32; b * d];
    let loss = {
        let grad_ent = grads.dense_entities_mut().ok_or_else(|| {
            KgeError::Invalid("batched 1-N needs a dense gradient accumulator".into())
        })?;
        kernel.softmax_ce(
            &q,
            tables.entities_raw(),
            d,
            &targets,
            scale,
            &mut grad_q,
            grad_ent,
        )?
    };
    let (mut d_r, mut d_a) = (vec![0.0f32; d], vec![0.0f32; d]);
    for (i, &(a, r, side, _)) in queries.iter().enumerate() {
        d_r.fill(0.0);
        d_a.fill(0.0);
        scorer.query_backward(
            tables.relation(r)?,
            tables.entity(a)?,
            side,
            &grad_q[i * d..(i + 1) * d],
            &mut d_r,
            &mut d_a,
        );
        grads.add_entity_prescaled(a, &d_a);
        grads.add_relation_prescaled(r, &d_r);
    }
    Ok(loss)
}
