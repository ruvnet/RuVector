//! The two supported training regimes (ADR-003 §1) and N3 regularization
//! (§2), each as a per-positive step that reads embeddings from [`Tables`],
//! accumulates loss gradients into [`Grads`], and returns its scalar loss.
//!
//! Scores are "higher is more plausible". The self-adversarial loss is the
//! RotatE form (arXiv:1902.10197) rewritten for that convention; the
//! 1-vs-all loss is softmax cross-entropy over every entity (Lacroix et al.
//! 2018), run on both sides.

use super::grad::Differentiable;
use super::negatives::sample_corruptions;
use super::optim::Grads;
use crate::data::Rng;
use crate::{Result, Tables, Triple};

/// Numerically stable sigmoid.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `log(sigmoid(x)) = -softplus(-x)`.
fn log_sigmoid(x: f32) -> f32 {
    // softplus(z) = max(z,0) + ln(1 + e^{-|z|}); here z = -x.
    let z = -x;
    -(z.max(0.0) + (1.0 + (-z.abs()).exp()).ln())
}

/// Softmax in place (max-shifted). Returns nothing; `v` becomes probabilities.
fn softmax_inplace(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// N3 (nuclear 3-norm) penalty for one embedding row and its gradient.
/// Penalty `= lambda * sum |x_i|^3`; gradient `= 3 * lambda * x_i * |x_i|`.
/// Returns the penalty and fills `grad_out` (same length as `row`).
pub(crate) fn n3_grad(row: &[f32], lambda: f32, grad_out: &mut Vec<f32>) -> f32 {
    grad_out.clear();
    grad_out.reserve(row.len());
    let mut penalty = 0.0f32;
    for &x in row {
        let ax = x.abs();
        penalty += ax * ax * ax;
        grad_out.push(3.0 * lambda * x * ax);
    }
    lambda * penalty
}

/// One self-adversarial step for a positive triple. Corrupts both the tail and
/// the head with `neg_count` samples each. Returns the loss.
#[allow(clippy::too_many_arguments)]
pub(crate) fn self_adversarial_step(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    neg_count: usize,
    temperature: f32,
    margin: f32,
    rng: &mut Rng,
    grads: &mut Grads,
) -> Result<f32> {
    let mut buf = Vec::new();
    let mut loss = 0.0;
    // Tail corruption: (s, r, o'), open slot = object.
    loss += sided_step(
        tables,
        scorer,
        t,
        Side::Tail,
        neg_count,
        temperature,
        margin,
        rng,
        grads,
        &mut buf,
    )?;
    // Head corruption: (s', r, o), open slot = subject.
    loss += sided_step(
        tables,
        scorer,
        t,
        Side::Head,
        neg_count,
        temperature,
        margin,
        rng,
        grads,
        &mut buf,
    )?;
    Ok(loss)
}

#[derive(Clone, Copy)]
enum Side {
    Head,
    Tail,
}

#[allow(clippy::too_many_arguments)]
fn sided_step(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    side: Side,
    neg_count: usize,
    temperature: f32,
    margin: f32,
    rng: &mut Rng,
    grads: &mut Grads,
    negs: &mut Vec<u32>,
) -> Result<f32> {
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;

    let f_pos = scorer.score(s, r, o);

    sample_corruptions(rng, tables.num_entities(), neg_count, negs);

    // Scores of the negatives, and self-adversarial weights over them.
    let mut f_neg: Vec<f32> = Vec::with_capacity(negs.len());
    for &c in negs.iter() {
        let ce = tables.entity(c)?;
        let fn_i = match side {
            Side::Tail => scorer.score(s, r, ce),
            Side::Head => scorer.score(ce, r, o),
        };
        f_neg.push(fn_i);
    }
    let mut w: Vec<f32> = f_neg.iter().map(|&f| temperature * f).collect();
    if !w.is_empty() {
        softmax_inplace(&mut w);
    }

    // Loss.
    let mut loss = -log_sigmoid(margin + f_pos);
    for (&fn_i, &wi) in f_neg.iter().zip(&w) {
        loss -= wi * log_sigmoid(-(margin + fn_i));
    }

    // Positive gradient: dL/df_pos = sigmoid(margin + f_pos) - 1.
    let coeff_pos = sigmoid(margin + f_pos) - 1.0;
    accumulate_triple_grad(scorer, grads, t.s, t.r, t.o, s, r, o, coeff_pos);

    // Negative gradients: dL/df_neg_i = w_i * sigmoid(margin + f_neg_i).
    for (&c, (&fn_i, &wi)) in negs.iter().zip(f_neg.iter().zip(&w)) {
        let coeff = wi * sigmoid(margin + fn_i);
        let ce = tables.entity(c)?;
        match side {
            Side::Tail => accumulate_triple_grad(scorer, grads, t.s, t.r, c, s, r, ce, coeff),
            Side::Head => accumulate_triple_grad(scorer, grads, c, t.r, t.o, ce, r, o, coeff),
        }
    }
    Ok(loss)
}

/// One 1-vs-all cross-entropy step over every entity, on both the tail and the
/// head side. Returns the summed loss.
pub(crate) fn one_vs_all_step(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    grads: &mut Grads,
) -> Result<f32> {
    let n = tables.num_entities();
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;

    // ---- Tail side: classify the object among all entities. ----
    let mut probs: Vec<f32> = Vec::with_capacity(n);
    for e in 0..n as u32 {
        probs.push(scorer.score(s, r, tables.entity(e)?));
    }
    softmax_inplace(&mut probs);
    let mut loss = -(probs[t.o as usize].max(1e-30)).ln();
    // Gradients: dL/df_e = p_e - 1{e==o}. Accumulate into s, r (summed) and e.
    let mut gs_acc = vec![0.0f32; scorer.dims()];
    let mut gr_acc = vec![0.0f32; scorer.dims()];
    for e in 0..n as u32 {
        let coeff = probs[e as usize] - if e == t.o { 1.0 } else { 0.0 };
        if coeff == 0.0 {
            continue;
        }
        let ee = tables.entity(e)?;
        let (gs, gr, go) = scorer.grad(s, r, ee);
        axpy(&mut gs_acc, coeff, &gs);
        axpy(&mut gr_acc, coeff, &gr);
        grads.add_entity(e, &scaled(coeff, &go));
    }
    grads.add_entity(t.s, &gs_acc);
    grads.add_relation(t.r, &gr_acc);

    // ---- Head side: classify the subject among all entities. ----
    let mut probs_h: Vec<f32> = Vec::with_capacity(n);
    for e in 0..n as u32 {
        probs_h.push(scorer.score(tables.entity(e)?, r, o));
    }
    softmax_inplace(&mut probs_h);
    loss += -(probs_h[t.s as usize].max(1e-30)).ln();
    let mut gr_acc2 = vec![0.0f32; scorer.dims()];
    let mut go_acc = vec![0.0f32; scorer.dims()];
    for e in 0..n as u32 {
        let coeff = probs_h[e as usize] - if e == t.s { 1.0 } else { 0.0 };
        if coeff == 0.0 {
            continue;
        }
        let ee = tables.entity(e)?;
        let (gs, gr, go) = scorer.grad(ee, r, o);
        grads.add_entity(e, &scaled(coeff, &gs));
        axpy(&mut gr_acc2, coeff, &gr);
        axpy(&mut go_acc, coeff, &go);
    }
    grads.add_relation(t.r, &gr_acc2);
    grads.add_entity(t.o, &go_acc);

    Ok(loss)
}

/// Per-batch precomputation for [`one_vs_all_step_batched`]: every entity's
/// index vector (row-major, `num_entities × index_dims`) plus, per scored
/// side, the softmax coefficients and the open-slot gradient, which
/// [`flush`](Self::flush) turns into per-entity gradients.
///
/// Built once per mini-batch — the tables do not change inside a batch
/// (the optimiser applies after it), so the index vectors stay valid.
pub(crate) struct BatchedOneVsAll {
    index: Vec<f32>,
    index_dims: usize,
    dims: usize,
    num_entities: usize,
    /// `(coefficients over all entities, open-slot gradient)` per scored
    /// side, in the order the sides were scored.
    pending: Vec<(Vec<f32>, Vec<f32>)>,
}

/// One scored side of one positive: everything the step contributes to the
/// batch gradient, computed without touching shared state.
pub(crate) struct SideOut {
    anchor_id: u32,
    rel_id: u32,
    g_anchor: Vec<f32>,
    g_rel: Vec<f32>,
    coeffs: Vec<f32>,
    unit: Vec<f32>,
}

/// The loss and both sides of one positive (see [`batched_positive`]).
pub(crate) struct PositiveOut {
    pub(crate) loss: f32,
    sides: [SideOut; 2],
}

impl BatchedOneVsAll {
    pub(crate) fn new(tables: &Tables, scorer: &dyn Differentiable) -> Result<Self> {
        let n = tables.num_entities();
        let dd = scorer.index_dims();
        let mut index = Vec::with_capacity(n * dd);
        for e in 0..n as u32 {
            index.extend_from_slice(&scorer.index_vector(tables.entity(e)?));
        }
        Ok(Self {
            index,
            index_dims: dd,
            dims: scorer.dims(),
            num_entities: n,
            pending: Vec::new(),
        })
    }

    /// Add one positive's anchor and relation gradients to `grads` (same
    /// order as the sequential path) and queue its open-slot terms.
    pub(crate) fn absorb(&mut self, out: PositiveOut, grads: &mut Grads) {
        for side in out.sides {
            grads.add_entity(side.anchor_id, &side.g_anchor);
            grads.add_relation(side.rel_id, &side.g_rel);
            self.pending.push((side.coeffs, side.unit));
        }
    }

    /// Move the accumulated per-entity gradients into `grads`:
    /// `dense[e] = Σ_k coeffs_k[e] · unit_k`, summed in scoring order, so the
    /// result is identical whether the rows are built on one thread or many.
    pub(crate) fn flush(&mut self, grads: &mut Grads) {
        let d = self.dims;
        let pending = std::mem::take(&mut self.pending);
        let row = |e: usize| -> Vec<f32> {
            let mut acc = vec![0.0f32; d];
            for (coeffs, unit) in &pending {
                let c = coeffs[e];
                if c != 0.0 {
                    axpy(&mut acc, c, unit);
                }
            }
            acc
        };
        let rows: Vec<Vec<f32>> = super::par::map_range(self.num_entities, row);
        for (e, acc) in rows.iter().enumerate() {
            if acc.iter().any(|&x| x != 0.0) {
                grads.add_entity(e as u32, acc);
            }
        }
    }
}

/// Score one positive on both sides with the batched 1-vs-all formulation
/// (see [`one_vs_all_step_batched`]). Pure: reads the tables and the batch's
/// index vectors, writes nothing, so positives can be scored in parallel.
pub(crate) fn batched_positive(
    tables: &Tables,
    scorer: &dyn Differentiable,
    batch: &BatchedOneVsAll,
    t: Triple,
) -> Result<PositiveOut> {
    let n = tables.num_entities();
    let d = scorer.dims();
    let dd = batch.index_dims;
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;
    let mut loss = 0.0f32;
    let mut sides = Vec::with_capacity(2);

    for side in [crate::Side::Tail, crate::Side::Head] {
        let (anchor, anchor_id, gold) = match side {
            crate::Side::Tail => (s, t.s, t.o),
            crate::Side::Head => (o, t.o, t.s),
        };
        let q = scorer.query_vector(r, anchor, side);
        let mut logits: Vec<f32> = (0..n)
            .map(|e| {
                batch.index[e * dd..(e + 1) * dd]
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        softmax_inplace(&mut logits);
        loss += -(logits[gold as usize].max(1e-30)).ln();
        // Coefficients (p_e − 1{e = gold}); the weighted open entity is Σ_e c_e · E_e.
        logits[gold as usize] -= 1.0;
        let coeffs = logits;
        let mut weighted = vec![0.0f32; d];
        for (e, &c) in coeffs.iter().enumerate() {
            if c != 0.0 {
                axpy(&mut weighted, c, tables.entity(e as u32)?);
            }
        }
        let (unit, g_rel, g_anchor) = match side {
            crate::Side::Tail => {
                let (gs, gr, go) = scorer.grad(s, r, &weighted);
                (go, gr, gs)
            }
            crate::Side::Head => {
                let (gs, gr, go) = scorer.grad(&weighted, r, o);
                (gs, gr, go)
            }
        };
        // `unit` is ∂score/∂(open slot); it is the same for every open entity.
        sides.push(SideOut {
            anchor_id,
            rel_id: t.r,
            g_anchor,
            g_rel,
            coeffs,
            unit,
        });
    }
    let [tail, head]: [SideOut; 2] = sides.try_into().map_err(|_| {
        crate::KgeError::Invalid("batched 1-vs-all produced the wrong number of sides".into())
    })?;
    Ok(PositiveOut {
        loss,
        sides: [tail, head],
    })
}

/// Batched 1-vs-all cross-entropy for multilinear scorers
/// ([`Differentiable::multilinear`]). Same loss and gradients as
/// [`one_vs_all_step`], computed without a per-entity `score`/`grad` call:
///
/// - every entity is scored with one dot product against the precomputed
///   index vectors (`query · index = score` exactly);
/// - because the score is linear in the open slot, `Σ_e c_e ∇_{s,r} score(s,r,e)`
///   equals `∇_{s,r} score(s, r, Σ_e c_e e)`, so the anchor/relation gradients
///   need one `grad` call on the coefficient-weighted entity;
/// - the open-slot gradient does not depend on the open entity, so entity `e`
///   receives `c_e · g` for a single vector `g` (applied in [`BatchedOneVsAll::flush`]).
///
/// Cost per positive: `O(|E|·d)` multiply-adds plus two `grad` calls, instead
/// of `4·|E|` FFT-based score/grad calls. The trainer calls
/// [`batched_positive`] and [`BatchedOneVsAll::absorb`] directly so the
/// scoring can run in parallel; this sequential form is the reference the
/// tests compare against.
#[cfg(test)]
pub(crate) fn one_vs_all_step_batched(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    batch: &mut BatchedOneVsAll,
    grads: &mut Grads,
) -> Result<f32> {
    let out = batched_positive(tables, scorer, batch, t)?;
    let loss = out.loss;
    batch.absorb(out, grads);
    Ok(loss)
}

/// Accumulate `coeff * grad(score)` into the three rows of a triple.
#[allow(clippy::too_many_arguments)]
fn accumulate_triple_grad(
    scorer: &dyn Differentiable,
    grads: &mut Grads,
    s_id: u32,
    r_id: u32,
    o_id: u32,
    s: &[f32],
    r: &[f32],
    o: &[f32],
    coeff: f32,
) {
    if coeff == 0.0 {
        return;
    }
    let (gs, gr, go) = scorer.grad(s, r, o);
    grads.add_entity(s_id, &scaled(coeff, &gs));
    grads.add_relation(r_id, &scaled(coeff, &gr));
    grads.add_entity(o_id, &scaled(coeff, &go));
}

fn scaled(a: f32, v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| a * x).collect()
}

fn axpy(dst: &mut [f32], a: f32, v: &[f32]) {
    for (d, &x) in dst.iter_mut().zip(v) {
        *d += a * x;
    }
}

#[cfg(test)]
mod batched_tests {
    use super::*;
    use crate::scorer::HolE;
    use crate::train::grad::testing::DistMult;
    use crate::train::optim::Grads;

    fn max_abs_diff(
        a: &std::collections::BTreeMap<u32, Vec<f32>>,
        b: &std::collections::BTreeMap<u32, Vec<f32>>,
    ) -> f32 {
        assert_eq!(
            a.keys().collect::<Vec<_>>(),
            b.keys().collect::<Vec<_>>(),
            "same rows touched"
        );
        a.iter()
            .flat_map(|(k, va)| va.iter().zip(&b[k]).map(|(x, y)| (x - y).abs()))
            .fold(0.0, f32::max)
    }

    fn check(scorer: &dyn Differentiable, dims: usize) {
        let (ne, nr) = (37, 4);
        let tables = Tables::new(ne, nr, dims, 11);
        let triples = [
            Triple::new(0, 1, 5),
            Triple::new(7, 0, 7),
            Triple::new(36, 3, 2),
            Triple::new(12, 2, 30),
        ];
        let mut exact = Grads::new(dims);
        let mut fast = Grads::new(dims);
        let mut batch = BatchedOneVsAll::new(&tables, scorer).unwrap();
        let (mut l_exact, mut l_fast) = (0.0f32, 0.0f32);
        for &t in &triples {
            l_exact += one_vs_all_step(&tables, scorer, t, &mut exact).unwrap();
            l_fast += one_vs_all_step_batched(&tables, scorer, t, &mut batch, &mut fast).unwrap();
        }
        batch.flush(&mut fast);
        assert!(
            (l_exact - l_fast).abs() < 1e-4 * l_exact.abs().max(1.0),
            "loss {l_exact} vs {l_fast}"
        );
        let scale = exact
            .entity_rows()
            .values()
            .flatten()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            scale > 1e-3,
            "gradients are non-trivial (max |g| = {scale})"
        );
        let de = max_abs_diff(exact.entity_rows(), fast.entity_rows());
        let dr = max_abs_diff(exact.relation_rows(), fast.relation_rows());
        assert!(
            de < 1e-4 * scale.max(1.0) && dr < 1e-4 * scale.max(1.0),
            "entity diff {de}, relation diff {dr} (scale {scale})"
        );
    }

    #[test]
    fn batched_matches_exact_for_hole() {
        check(&HolE::new(16).unwrap(), 16);
    }

    #[test]
    fn batched_matches_exact_for_distmult() {
        check(&DistMult::new(8), 8);
    }

    /// Scoring positives out of order (as the parallel path may) and folding
    /// them in batch order gives bit-identical gradients to the sequential step.
    #[test]
    fn scoring_order_does_not_change_the_gradients() {
        let dims = 16;
        let scorer = HolE::new(dims).unwrap();
        let tables = Tables::new(29, 3, dims, 5);
        let triples: Vec<Triple> = (0..9u32)
            .map(|i| Triple::new(i * 3 % 29, i % 3, (i * 7 + 1) % 29))
            .collect();

        let mut seq = Grads::new(dims);
        let mut b1 = BatchedOneVsAll::new(&tables, &scorer).unwrap();
        let mut l_seq = 0.0f32;
        for &t in &triples {
            l_seq += one_vs_all_step_batched(&tables, &scorer, t, &mut b1, &mut seq).unwrap();
        }
        b1.flush(&mut seq);

        let mut b2 = BatchedOneVsAll::new(&tables, &scorer).unwrap();
        let mut outs: Vec<Option<PositiveOut>> = (0..triples.len()).map(|_| None).collect();
        for k in (0..triples.len()).rev() {
            outs[k] = Some(batched_positive(&tables, &scorer, &b2, triples[k]).unwrap());
        }
        let mut par = Grads::new(dims);
        let mut l_par = 0.0f32;
        for out in outs.into_iter().flatten() {
            l_par += out.loss;
            b2.absorb(out, &mut par);
        }
        b2.flush(&mut par);

        assert_eq!(l_seq.to_bits(), l_par.to_bits());
        assert_eq!(seq.entity_rows(), par.entity_rows());
        assert_eq!(seq.relation_rows(), par.relation_rows());
    }
}
