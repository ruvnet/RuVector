//! HPO knobs and the deterministic proposal generator (ADR-004 loops 1–2).
//!
//! The generator runs **successive halving over the HPO grid first**
//! (`dims × lr × loss × n3` — ADR-004's KGE deviation that ranks tuning above
//! architecture, per LibKGE/KGTuner), then emits the two model arms
//! (HolE vs RotatE). Every proposal carries a content-hash id (the hash of its
//! [`Knobs`], reusing typesafe-core's `content_hash`) and a parent link so the
//! receipt chain records lineage.
//!
//! This module is pure: it never touches an [`crate::optimize::Evaluator`]. The
//! campaign drives it, supplying a reward closure so `fit → gate → receipt`
//! stays interleaved (ADR-004 "exhaustion pauses, never lowers the bar").

use crate::ScorerKind;
use ruvector_typesafe_core::receipt::content_hash;
use serde::{Deserialize, Serialize};

/// Optimizer arm (ADR-004 loop 1: Adam vs Adagrad).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Optimizer {
    Adam,
    Adagrad,
}

/// Loss arm (LibKGE's dominant three; ADR-004 loop 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Loss {
    /// Full softmax cross-entropy over all entities (1-vs-all).
    CrossEntropy,
    /// Binary cross-entropy with negative sampling.
    Bce,
    /// Self-adversarial margin ranking.
    Margin,
}

/// Which loop a proposal belongs to. Maps by **loop index** onto typesafe's
/// `ProposalKind` (only [`Self::ModelArm`] is a semantic match — see
/// [`super::campaign`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KgeArm {
    /// Loop 1: hyperparameter optimization (ranked first for KGE).
    Hpo,
    /// Loop 2: model-family arm (HolE vs RotatE).
    ModelArm,
    /// Loop 3: continual update on graph growth (EWC).
    Continual,
}

/// One training configuration. Serde field order is the canonical order the
/// content hash is taken over, so the id is stable across runs and targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Knobs {
    pub dims: usize,
    pub lr: f32,
    pub optimizer: Optimizer,
    pub loss: Loss,
    pub neg_count: usize,
    /// Self-adversarial sampling temperature (RotatE §; ADR-002).
    pub temperature: f32,
    /// N3 tensor-regularizer weight (ADR-004 loop 1: N3 vs L2).
    pub n3_lambda: f32,
    pub epochs: usize,
    pub scorer: ScorerKind,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            dims: 128,
            lr: 0.01,
            optimizer: Optimizer::Adam,
            loss: Loss::CrossEntropy,
            neg_count: 100,
            temperature: 1.0,
            n3_lambda: 0.0,
            epochs: 20,
            scorer: ScorerKind::Hole,
        }
    }
}

impl Knobs {
    /// Content-hash id: first 64 bits of `content_hash` over the canonical
    /// JSON. Two identical `Knobs` share an id; differing epochs (successive
    /// halving rungs) or scorer (model arms) yield distinct ids.
    #[must_use]
    pub fn id(&self) -> u64 {
        let bytes = serde_json::to_vec(self).expect("knobs serialisable");
        let hex = content_hash(&bytes);
        u64::from_str_radix(&hex[..16], 16).expect("16 hex chars")
    }
}

/// A candidate change awaiting the gate. `id`/`parent` are `u64` to match
/// typesafe's `Proposal`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub id: u64,
    pub parent: Option<u64>,
    pub arm: KgeArm,
    pub knobs: Knobs,
}

impl Proposal {
    fn with(arm: KgeArm, knobs: Knobs, parent: Option<u64>) -> Self {
        Self {
            id: knobs.id(),
            parent,
            arm,
            knobs,
        }
    }
    /// The campaign incumbent (default/baseline knobs, no parent).
    #[must_use]
    pub fn baseline(knobs: Knobs) -> Self {
        Self::with(KgeArm::Hpo, knobs, None)
    }
    #[must_use]
    pub fn hpo(knobs: Knobs, parent: Option<u64>) -> Self {
        Self::with(KgeArm::Hpo, knobs, parent)
    }
    #[must_use]
    pub fn model_arm(knobs: Knobs, parent: u64) -> Self {
        Self::with(KgeArm::ModelArm, knobs, Some(parent))
    }
    #[must_use]
    pub fn continual(knobs: Knobs, parent: u64) -> Self {
        Self::with(KgeArm::Continual, knobs, Some(parent))
    }
}

/// The HPO grid (ADR-004 loop 1: `dims × lr × loss × n3`). Each axis is a list;
/// [`Self::expand`] takes the cartesian product over a base [`Knobs`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HpoGrid {
    pub dims: Vec<usize>,
    pub lrs: Vec<f32>,
    pub losses: Vec<Loss>,
    pub n3_lambdas: Vec<f32>,
}

impl Default for HpoGrid {
    fn default() -> Self {
        Self {
            dims: vec![128, 256],
            lrs: vec![0.001, 0.01],
            losses: vec![Loss::CrossEntropy, Loss::Bce],
            n3_lambdas: vec![0.0, 1e-3],
        }
    }
}

impl HpoGrid {
    /// Cartesian product in `dims → lr → loss → n3` order (deterministic).
    #[must_use]
    pub fn expand(&self, base: &Knobs) -> Vec<Knobs> {
        let mut out = Vec::new();
        for &d in &self.dims {
            for &lr in &self.lrs {
                for &loss in &self.losses {
                    for &n3 in &self.n3_lambdas {
                        let mut k = base.clone();
                        k.dims = d;
                        k.lr = lr;
                        k.loss = loss;
                        k.n3_lambda = n3;
                        out.push(k);
                    }
                }
            }
        }
        out
    }
}

/// The model arms to try against the best HPO config (ADR-004 loop 2). The HPO
/// winner already *is* the result of its own scorer arm, so that scorer is
/// skipped: a Hole-scored `base` yields only the Rotate arm, and vice versa.
/// This keeps every receipt's `proposal.id` (a content hash) distinct for exact
/// rollback.
#[must_use]
pub fn model_arms(base: &Knobs, parent: u64) -> Vec<Proposal> {
    [ScorerKind::Hole, ScorerKind::Rotate]
        .into_iter()
        .filter(|&scorer| scorer != base.scorer)
        .map(|scorer| {
            let mut k = base.clone();
            k.scorer = scorer;
            Proposal::model_arm(k, parent)
        })
        .collect()
}

/// Result of a successive-halving sweep: every proposal evaluated (in rung
/// order) and the best by reward (`None` for an empty grid).
#[derive(Debug, Clone)]
pub struct HalvingResult {
    pub evaluated: Vec<Proposal>,
    pub best: Option<Proposal>,
}

struct Cand {
    knobs: Knobs,
    parent: Option<u64>,
}

/// Successive halving (Jamieson & Talwalkar 2016) over `configs`: rung 0
/// evaluates all at their base `epochs`; each rung keeps the top `1/eta` by
/// `reward` and re-runs them with `eta`× the epochs, until one survives. The
/// `reward` closure is the campaign's `fit → gate → receipt` step, so budget
/// exhaustion inside it stops further fitting.
pub fn successive_halving<F>(
    configs: Vec<Knobs>,
    eta: usize,
    base_parent: u64,
    mut reward: F,
) -> HalvingResult
where
    F: FnMut(&Proposal) -> f32,
{
    let eta = eta.max(2);
    let mut cands: Vec<Cand> = configs
        .into_iter()
        .map(|knobs| Cand {
            knobs,
            parent: Some(base_parent),
        })
        .collect();
    let mut evaluated = Vec::new();
    let mut best: Option<(Proposal, f32)> = None;
    let mut rung: u32 = 0;
    while !cands.is_empty() {
        let mult = eta.checked_pow(rung).unwrap_or(usize::MAX);
        let mut scored: Vec<(Cand, Proposal, f32)> = Vec::with_capacity(cands.len());
        for c in cands {
            let mut kk = c.knobs.clone();
            kk.epochs = c.knobs.epochs.saturating_mul(mult);
            let p = Proposal::hpo(kk, c.parent);
            let r = reward(&p);
            evaluated.push(p.clone());
            if best.as_ref().is_none_or(|(_, br)| r > *br) {
                best = Some((p.clone(), r));
            }
            scored.push((c, p, r));
        }
        if scored.len() <= 1 {
            break;
        }
        scored.sort_by(|a, b| b.2.total_cmp(&a.2));
        let keep = (scored.len() / eta).max(1);
        cands = scored
            .into_iter()
            .take(keep)
            .map(|(c, p, _)| Cand {
                knobs: c.knobs,
                parent: Some(p.id),
            })
            .collect();
        rung += 1;
    }
    HalvingResult {
        evaluated,
        best: best.map(|(p, _)| p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_stable_and_distinguish_epochs_and_scorer() {
        let k = Knobs::default();
        assert_eq!(k.id(), k.clone().id());
        let mut e = k.clone();
        e.epochs *= 2;
        assert_ne!(k.id(), e.id());
        let mut r = k.clone();
        r.scorer = ScorerKind::Rotate;
        assert_ne!(k.id(), r.id());
    }

    #[test]
    fn grid_expands_in_declared_order() {
        let g = HpoGrid {
            dims: vec![8, 16],
            lrs: vec![0.1],
            losses: vec![Loss::Bce],
            n3_lambdas: vec![0.0, 1.0],
        };
        let cs = g.expand(&Knobs::default());
        assert_eq!(cs.len(), 4);
        assert_eq!(cs[0].dims, 8);
        assert_eq!(cs[0].n3_lambda, 0.0);
        assert_eq!(cs[1].n3_lambda, 1.0);
        assert_eq!(cs[2].dims, 16);
    }

    #[test]
    fn halving_prunes_to_one_and_picks_best_by_reward() {
        // Reward increases with dims → the largest-dims config must win.
        let g = HpoGrid {
            dims: vec![8, 16, 32, 64],
            lrs: vec![0.1],
            losses: vec![Loss::Bce],
            n3_lambdas: vec![0.0],
        };
        let cs = g.expand(&Knobs::default());
        let res = successive_halving(cs, 2, 0, |p| p.knobs.dims as f32);
        let best = res.best.unwrap();
        assert_eq!(best.knobs.dims, 64);
        // 4 → 2 → 1 candidates = 7 evaluations; final rung ran 4× epochs.
        assert_eq!(res.evaluated.len(), 7);
        assert!(res.evaluated.last().unwrap().knobs.epochs > Knobs::default().epochs);
    }

    #[test]
    fn empty_grid_has_no_best() {
        let res = successive_halving(vec![], 2, 0, |_| 0.0);
        assert!(res.best.is_none());
        assert!(res.evaluated.is_empty());
    }

    #[test]
    fn model_arms_skip_the_base_scorer() {
        // Hole base → only the Rotate arm remains (the Hole result is the base).
        let arms = model_arms(&Knobs::default(), 42);
        assert_eq!(arms.len(), 1);
        assert_eq!(arms[0].knobs.scorer, ScorerKind::Rotate);
        assert_eq!(arms[0].parent, Some(42));
        assert_eq!(arms[0].arm, KgeArm::ModelArm);
        // Rotate base → only the Hole arm remains.
        let rot = Knobs {
            scorer: ScorerKind::Rotate,
            ..Knobs::default()
        };
        let arms = model_arms(&rot, 7);
        assert_eq!(arms.len(), 1);
        assert_eq!(arms[0].knobs.scorer, ScorerKind::Hole);
    }
}
