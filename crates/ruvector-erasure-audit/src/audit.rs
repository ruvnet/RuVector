//! The black-box adversary and the paired distinguisher.
//!
//! The adversary never touches `graph.layers`, `graph.deleted` or
//! `graph.vectors` directly — it only calls [`HnswGraph::search`] and reads
//! back `(id, distance)` pairs, which is what any production vector-DB API
//! returns. Distances are recomputed here from the returned ids purely because
//! this HNSW implementation's `search` returns bare ids; the values are
//! exactly what the API would have handed over.

use ruvector_hnsw_repair::{l2_sq, HnswGraph};
use std::collections::HashSet;

use crate::data::Rng;

/// Names of the scalar features the adversary computes, in feature order.
pub const FEATURE_NAMES: [&str; 6] = [
    "dsum_lo",          // sum of top-k distances at low effort
    "d1_hi",            // nearest-neighbour distance at high effort
    "effort_gap",       // dsum_lo - dsum_hi: how much extra effort buys
    "topk_instability", // 1 - Jaccard(top-k @ ef_lo, top-k @ ef_hi)
    "jitter_churn",     // mean 1 - Jaccard(top-k(q+noise), top-k(q))
    "result_deficit",   // k - |returned| at low effort
];

/// One observation vector: the adversary's view of a single index.
#[derive(Clone, Debug, Default)]
pub struct Features(pub [f64; FEATURE_NAMES.len()]);

/// Parameters of the adversary's probing budget.
#[derive(Clone, Debug)]
pub struct ProbeConfig {
    pub k: usize,
    pub ef_lo: usize,
    pub ef_hi: usize,
    /// Number of jittered repeats of the query used for `jitter_churn`.
    pub jitter_trials: usize,
    /// Per-coordinate Gaussian noise added to jittered queries.
    pub jitter_sigma: f32,
    /// Seed for the jitter stream. Must be identical across a paired A/B run
    /// so the two indexes see byte-identical probe queries.
    pub jitter_seed: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            k: 10,
            ef_lo: 16,
            ef_hi: 64,
            jitter_trials: 4,
            jitter_sigma: 0.05,
            jitter_seed: 0xA5A5_1234,
        }
    }
}

fn top_with_dists(graph: &HnswGraph, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
    let dim = graph.config.dim;
    graph
        .search(q, k, ef)
        .into_iter()
        .map(|id| (id, l2_sq(q, &graph.vectors[id as usize], dim)))
        .collect()
}

fn jaccard(a: &[(u32, f32)], b: &[(u32, f32)]) -> f64 {
    let sa: HashSet<u32> = a.iter().map(|(i, _)| *i).collect();
    let sb: HashSet<u32> = b.iter().map(|(i, _)| *i).collect();
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

/// Run the adversary's probe battery against one index.
pub fn observe(graph: &HnswGraph, q: &[f32], cfg: &ProbeConfig) -> Features {
    let lo = top_with_dists(graph, q, cfg.k, cfg.ef_lo);
    let hi = top_with_dists(graph, q, cfg.k, cfg.ef_hi);

    let dsum_lo: f64 = lo.iter().map(|(_, d)| *d as f64).sum();
    let dsum_hi: f64 = hi.iter().map(|(_, d)| *d as f64).sum();
    let d1_hi = hi.first().map(|(_, d)| *d as f64).unwrap_or(f64::MAX);

    let mut churn = 0.0f64;
    if cfg.jitter_trials > 0 {
        let mut rng = Rng::new(cfg.jitter_seed);
        for _ in 0..cfg.jitter_trials {
            let jq: Vec<f32> = q
                .iter()
                .map(|&x| x + cfg.jitter_sigma * rng.next_normal())
                .collect();
            let jr = top_with_dists(graph, &jq, cfg.k, cfg.ef_lo);
            churn += 1.0 - jaccard(&jr, &lo);
        }
        churn /= cfg.jitter_trials as f64;
    }

    Features([
        dsum_lo,
        d1_hi,
        dsum_lo - dsum_hi,
        1.0 - jaccard(&lo, &hi),
        churn,
        (cfg.k as f64 - lo.len() as f64).max(0.0),
    ])
}

/// Outcome of the split-half feature selection + held-out scoring.
#[derive(Clone, Debug)]
pub struct DistinguisherResult {
    pub feature: &'static str,
    pub feature_idx: usize,
    /// `+1` means "index A tends to score higher", `-1` the reverse.
    pub direction: i8,
    pub select_acc: f64,
    pub holdout_acc: f64,
    pub holdout_n: usize,
    /// Per-feature accuracy over *all* pairs, direction fixed on the selection
    /// half. Reported for transparency about the multiple-comparison caveat.
    pub per_feature_overall: Vec<(&'static str, f64)>,
}

/// Accumulates paired (A, B) observations and scores a distinguisher.
#[derive(Default)]
pub struct PairedDistinguisher {
    pairs: Vec<(Features, Features)>,
}

impl PairedDistinguisher {
    pub fn new() -> Self {
        Self::default()
    }

    /// `a` = index where the target was inserted then erased; `b` = control.
    pub fn push(&mut self, a: Features, b: Features) {
        self.pairs.push((a, b));
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    fn accuracy(pairs: &[(Features, Features)], f: usize, dir: i8) -> f64 {
        if pairs.is_empty() {
            return 0.5;
        }
        let mut wins = 0.0f64;
        for (a, b) in pairs {
            let delta = a.0[f] - b.0[f];
            let signed = if dir >= 0 { delta } else { -delta };
            if signed > 0.0 {
                wins += 1.0;
            } else if signed == 0.0 {
                wins += 0.5;
            }
        }
        wins / pairs.len() as f64
    }

    /// Select the best feature *and* its direction on the first half of the
    /// trials, then score it on the untouched second half.
    ///
    /// This is what keeps the headline number honest: picking the best of six
    /// features on the same data it is scored on would inflate accuracy even
    /// with zero leakage.
    pub fn evaluate(&self) -> DistinguisherResult {
        let n = self.pairs.len();
        let split = n / 2;
        let (sel, hold) = self.pairs.split_at(split);

        let mut best = (0usize, 1i8, 0.0f64);
        for f in 0..FEATURE_NAMES.len() {
            for dir in [1i8, -1i8] {
                let acc = Self::accuracy(sel, f, dir);
                if acc > best.2 {
                    best = (f, dir, acc);
                }
            }
        }

        let per_feature_overall = (0..FEATURE_NAMES.len())
            .map(|f| {
                let d = if Self::accuracy(sel, f, 1) >= 0.5 {
                    1
                } else {
                    -1
                };
                (FEATURE_NAMES[f], Self::accuracy(&self.pairs, f, d))
            })
            .collect();

        DistinguisherResult {
            feature: FEATURE_NAMES[best.0],
            feature_idx: best.0,
            direction: best.1,
            select_acc: best.2,
            holdout_acc: Self::accuracy(hold, best.0, best.1),
            holdout_n: hold.len(),
            per_feature_overall,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(v: f64) -> Features {
        let mut f = Features::default();
        f.0[0] = v;
        f
    }

    #[test]
    fn no_signal_scores_near_half() {
        let mut d = PairedDistinguisher::new();
        // Alternating sign: feature 0 carries no consistent information.
        for i in 0..200 {
            let (a, b) = if i % 2 == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
            d.push(feat(a), feat(b));
        }
        let r = d.evaluate();
        assert!(
            (r.holdout_acc - 0.5).abs() < 0.15,
            "held-out accuracy {} should be near 0.5",
            r.holdout_acc
        );
    }

    #[test]
    fn perfect_signal_scores_one() {
        let mut d = PairedDistinguisher::new();
        for i in 0..100 {
            d.push(feat(1.0 + i as f64), feat(0.0));
        }
        let r = d.evaluate();
        assert_eq!(r.feature_idx, 0);
        assert_eq!(r.direction, 1);
        assert!((r.holdout_acc - 1.0).abs() < 1e-9, "{}", r.holdout_acc);
    }

    #[test]
    fn inverted_signal_is_detected_with_negative_direction() {
        let mut d = PairedDistinguisher::new();
        for _ in 0..100 {
            d.push(feat(0.0), feat(5.0));
        }
        let r = d.evaluate();
        assert_eq!(r.direction, -1);
        assert!(r.holdout_acc > 0.99);
    }

    #[test]
    fn identical_features_tie_at_exactly_half() {
        let mut d = PairedDistinguisher::new();
        for _ in 0..50 {
            d.push(feat(2.0), feat(2.0));
        }
        let r = d.evaluate();
        assert!((r.holdout_acc - 0.5).abs() < 1e-9);
    }
}
