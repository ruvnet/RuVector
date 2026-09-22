//! Shared experiment harness: index construction, the paired leak audit, and
//! the utility/cost measurement.
//!
//! Lives in the library rather than in a binary so that the headline benchmark
//! and the independent replication binary run *identical* code paths with
//! nothing but the seed stream differing.

use crate::audit::{observe, PairedDistinguisher, ProbeConfig};
use crate::data::{ClusteredSource, DatasetConfig};
use crate::erasure::{erase, ErasureMode};
use crate::wilson95;
use ruvector_hnsw_repair::{brute_force_knn_live, HnswConfig, HnswGraph};
use std::collections::HashSet;
use std::time::Instant;

/// Fixed HNSW parameters for every index in this experiment.
pub fn hnsw_config(dim: usize) -> HnswConfig {
    HnswConfig {
        dim,
        m: 8,
        m0: 16,
        ef_construction: 40,
        ml: 1.0 / (8f64.ln()),
    }
}

/// Experiment-wide knobs. `seed_offset` is XORed into every seed, which is how
/// an independent replicate is produced.
#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    pub base_n: usize,
    pub dim: usize,
    pub trials: usize,
    pub ef_rebuild: usize,
    pub seed_offset: u64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            base_n: 2_000,
            dim: 32,
            trials: 600,
            ef_rebuild: 48,
            seed_offset: 0,
        }
    }
}

impl ExperimentConfig {
    fn dataset(&self, stream: u64) -> DatasetConfig {
        DatasetConfig {
            dim: self.dim,
            seed: stream ^ self.seed_offset,
            ..DatasetConfig::default()
        }
    }

    pub fn modes(&self) -> Vec<ErasureMode> {
        vec![
            ErasureMode::Tombstone,
            ErasureMode::EagerRepair,
            ErasureMode::LocalRebuild {
                ef_rebuild: self.ef_rebuild,
            },
        ]
    }

    /// Build the shared base index every trial forks from.
    pub fn build_base(&self) -> HnswGraph {
        let mut src = ClusteredSource::new(self.dataset(0x5EED_0001));
        let mut g = HnswGraph::new(hnsw_config(self.dim));
        for v in src.sample_many(self.base_n) {
            g.insert(v);
        }
        g
    }

    /// Query set for the utility measurement, drawn from its own stream.
    pub fn queries(&self, n: usize) -> Vec<Vec<f32>> {
        ClusteredSource::new(self.dataset(0x9999_0001)).sample_many(n)
    }
}

// ---------------------------------------------------------------------------
// Part 1 — erasure-leak audit
// ---------------------------------------------------------------------------

/// Outcome of one (mode, churn) leak audit.
#[derive(Clone, Debug)]
pub struct LeakResult {
    pub mode: &'static str,
    pub churn: usize,
    pub feature: &'static str,
    pub direction: i8,
    pub select_acc: f64,
    pub holdout_acc: f64,
    pub holdout_n: usize,
    pub ci: (f64, f64),
    pub per_feature: Vec<(&'static str, f64)>,
    pub elapsed_s: f64,
}

impl LeakResult {
    /// True when the 95% CI excludes chance — i.e. a leak was detected.
    pub fn leaks(&self) -> bool {
        self.ci.0 > 0.5
    }
}

/// Paired A/B audit: A has the target inserted then erased, B has an unrelated
/// decoy inserted then erased at the same sequence position. Both then take the
/// same post-erasure churn. The adversary queries both with the target.
pub fn run_leak_audit(
    cfg: &ExperimentConfig,
    mode: ErasureMode,
    churn: usize,
    base: &HnswGraph,
    probe: &ProbeConfig,
) -> LeakResult {
    let t0 = Instant::now();
    let mut tsrc = ClusteredSource::new(cfg.dataset(0x7A46_0001));
    let mut dsrc = ClusteredSource::new(cfg.dataset(0x7A46_0002));
    let mut csrc = ClusteredSource::new(cfg.dataset(0x7A46_0003));
    let churn_pool: Vec<Vec<f32>> = csrc.sample_many(churn.max(1));

    let mut dist = PairedDistinguisher::new();
    for _ in 0..cfg.trials {
        let target = tsrc.sample();
        let decoy = dsrc.sample();

        let mut ga = base.clone();
        let ida = ga.insert(target.clone()) as usize;
        erase(&mut ga, ida, mode);

        let mut gb = base.clone();
        let idb = gb.insert(decoy) as usize;
        erase(&mut gb, idb, mode);

        for v in churn_pool.iter().take(churn) {
            ga.insert(v.clone());
            gb.insert(v.clone());
        }

        dist.push(observe(&ga, &target, probe), observe(&gb, &target, probe));
    }

    let r = dist.evaluate();
    LeakResult {
        mode: mode.label(),
        churn,
        feature: r.feature,
        direction: r.direction,
        select_acc: r.select_acc,
        holdout_acc: r.holdout_acc,
        holdout_n: r.holdout_n,
        ci: wilson95(r.holdout_acc * r.holdout_n as f64, r.holdout_n),
        per_feature: r.per_feature_overall,
        elapsed_s: t0.elapsed().as_secs_f64(),
    }
}

// ---------------------------------------------------------------------------
// Part 2 — utility and cost
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Summary {
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
}

pub fn summarize(mut xs: Vec<f64>) -> Summary {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |p: f64| {
        if xs.is_empty() {
            0.0
        } else {
            xs[(((xs.len() - 1) as f64) * p).round() as usize]
        }
    };
    Summary {
        mean: xs.iter().sum::<f64>() / xs.len().max(1) as f64,
        p50: pick(0.50),
        p95: pick(0.95),
    }
}

#[derive(Clone, Debug)]
pub struct UtilityResult {
    pub mode: &'static str,
    pub recall_before: f64,
    pub recall_after: f64,
    pub delete_us: Summary,
    pub search_us: Summary,
    pub retained_bytes_total: usize,
    pub referrers_mean: f64,
    pub rebuilt_mean: f64,
}

pub fn recall_at_k(graph: &HnswGraph, queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
    let gt = brute_force_knn_live(graph, queries, k);
    let mut total = 0.0;
    for (i, q) in queries.iter().enumerate() {
        let gt_set: HashSet<u32> = gt[i].iter().copied().collect();
        total += graph
            .search(q, k, ef)
            .iter()
            .filter(|id| gt_set.contains(id))
            .count() as f64
            / k as f64;
    }
    total / queries.len() as f64
}

/// Delete `delete_fraction` of the base index under `mode`, measuring recall,
/// erasure latency, post-erasure query latency and retained payload bytes.
///
/// Victims are chosen by a fixed stride so every mode erases the identical id
/// set — the comparison is paired, not sampled.
pub fn run_utility(
    cfg: &ExperimentConfig,
    mode: ErasureMode,
    base: &HnswGraph,
    queries: &[Vec<f32>],
    delete_fraction: f64,
    warmup: usize,
) -> UtilityResult {
    let mut g = base.clone();
    let recall_before = recall_at_k(&g, queries, 10, 64);

    let n_del = ((cfg.base_n as f64) * delete_fraction) as usize;
    let stride = (cfg.base_n / n_del.max(1)).max(1);
    let mut lat = Vec::with_capacity(n_del);
    let (mut retained, mut referrers, mut rebuilt) = (0usize, 0usize, 0usize);
    for i in 0..n_del {
        let s = erase(&mut g, i * stride, mode);
        lat.push(s.elapsed_ns as f64 / 1000.0);
        retained += s.retained_vector_bytes;
        referrers += s.referrers;
        rebuilt += s.rebuilt_lists;
    }

    let recall_after = recall_at_k(&g, queries, 10, 64);

    for q in queries.iter().take(warmup.min(queries.len())) {
        std::hint::black_box(g.search(q, 10, 64));
    }
    let mut slat = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        std::hint::black_box(g.search(q, 10, 64));
        slat.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }

    UtilityResult {
        mode: mode.label(),
        recall_before,
        recall_after,
        delete_us: summarize(lat),
        search_us: summarize(slat),
        retained_bytes_total: retained,
        referrers_mean: referrers as f64 / n_del.max(1) as f64,
        rebuilt_mean: rebuilt as f64 / n_del.max(1) as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_index_is_navigable_and_deterministic() {
        let cfg = ExperimentConfig {
            base_n: 400,
            trials: 4,
            ..Default::default()
        };
        let a = cfg.build_base();
        let b = cfg.build_base();
        assert_eq!(a.live_count(), 400);
        let qs = cfg.queries(20);
        assert_eq!(
            recall_at_k(&a, &qs, 10, 64),
            recall_at_k(&b, &qs, 10, 64),
            "identical configs must produce identical indexes"
        );
    }

    #[test]
    fn seed_offset_produces_a_different_corpus() {
        let a = ExperimentConfig {
            base_n: 300,
            ..Default::default()
        };
        let b = ExperimentConfig {
            base_n: 300,
            seed_offset: 0xABCD,
            ..Default::default()
        };
        assert_ne!(a.build_base().vectors[0], b.build_base().vectors[0]);
    }

    #[test]
    fn summarize_orders_percentiles() {
        let s = summarize((1..=100).map(|x| x as f64).collect());
        assert!(s.p50 <= s.p95);
        assert!((s.mean - 50.5).abs() < 1e-9);
    }

    #[test]
    fn utility_run_deletes_the_requested_fraction() {
        let cfg = ExperimentConfig {
            base_n: 400,
            ..Default::default()
        };
        let base = cfg.build_base();
        let qs = cfg.queries(20);
        let u = run_utility(&cfg, ErasureMode::Tombstone, &base, &qs, 0.20, 0);
        assert!(u.recall_before > 0.5, "recall {} too low", u.recall_before);
        assert_eq!(u.retained_bytes_total, 80 * 32 * 4);
    }
}
