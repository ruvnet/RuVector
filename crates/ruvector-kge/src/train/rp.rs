//! Relation-prediction auxiliary loss (RP; Chen et al. 2021,
//! arXiv:2110.02834): for each training example `(s, r, o)`, classify `r`
//! among **every** relation row given `(s, o)` with softmax cross-entropy.
//! Under reciprocal training the rows are all `2·|R|` (base and inverse), and
//! the reciprocal example `(o, r⁻¹, s)` has target `r⁻¹` — as in ssl-RP,
//! whose batches mix both directions. Weighted by `TrainConfig::rp_weight`
//! (`w_rel`); skipped entirely at weight 0.

use super::grad::Differentiable;
use super::loss::{axpy, scaled, softmax_inplace};
use super::optim::Grads;
use crate::{Result, Tables, Triple};

/// One RP step for example `t`, weighted by `weight`. Accumulates
/// `weight · dCE/dθ` into `s`, `o` and every relation row; returns the
/// weighted loss `weight · CE`.
pub(crate) fn relation_prediction_step(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    weight: f32,
    grads: &mut Grads,
) -> Result<f32> {
    let nr = tables.num_relations();
    let s = tables.entity(t.s)?;
    let o = tables.entity(t.o)?;
    tables.relation(t.r)?; // target must exist

    let mut probs: Vec<f32> = Vec::with_capacity(nr);
    for rid in 0..nr as u32 {
        probs.push(scorer.score(s, tables.relation(rid)?, o));
    }
    softmax_inplace(&mut probs);
    let loss = -(probs[t.r as usize].max(1e-30)).ln();

    let mut gs_acc = vec![0.0f32; scorer.dims()];
    let mut go_acc = vec![0.0f32; scorer.dims()];
    for rid in 0..nr as u32 {
        let coeff = weight * (probs[rid as usize] - if rid == t.r { 1.0 } else { 0.0 });
        if coeff == 0.0 {
            continue;
        }
        let (gs, gr, go) = scorer.grad(s, tables.relation(rid)?, o);
        axpy(&mut gs_acc, coeff, &gs);
        axpy(&mut go_acc, coeff, &go);
        grads.add_relation(rid, &scaled(coeff, &gr));
    }
    grads.add_entity(t.s, &gs_acc);
    grads.add_entity(t.o, &go_acc);
    Ok(weight * loss)
}
