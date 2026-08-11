//! Bandit-Tuned ANN for RuVector
//!
//! A lightweight HNSW index whose `ef_search` beam-width is automatically
//! tuned at runtime by a Multi-Armed Bandit.  The bandit observes a reward
//! signal (recall / latency tradeoff) and converges to the Pareto-optimal
//! operating point for the current workload.
//!
//! Three measurable variants:
//!  - `StaticDefault`  – fixed ef_search = 50 (safe, slow)
//!  - `StaticFast`     – fixed ef_search = 10 (fast, lower recall)
//!  - `BanditTuned`    – UCB1 bandit explores {10,20,30,40,50} and converges

pub mod bandit;
pub mod dataset;
pub mod hnsw;

use std::collections::HashSet;

// ─── core types ──────────────────────────────────────────────────────────────

/// A nearest-neighbour hit returned by any variant.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

impl Eq for Hit {}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── ANN trait ───────────────────────────────────────────────────────────────

/// Common search interface for all three variants.
pub trait AnnVariant: Send + Sync {
    /// Return the `k` approximate nearest neighbours to `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;

    /// Human-readable variant name.
    fn name(&self) -> &str;

    /// Estimated heap bytes used by this index.
    fn memory_bytes(&self) -> usize;
}

// ─── metrics ─────────────────────────────────────────────────────────────────

/// Recall@k: fraction of ground-truth top-k ids in `results`.
pub fn recall_at_k(results: &[Hit], ground_truth: &[Hit], k: usize) -> f32 {
    let res_ids: HashSet<usize> = results.iter().take(k).map(|h| h.id).collect();
    let gt_ids: HashSet<usize> = ground_truth.iter().take(k).map(|h| h.id).collect();
    if gt_ids.is_empty() {
        return 1.0;
    }
    let common = res_ids.intersection(&gt_ids).count();
    common as f32 / k.min(gt_ids.len()) as f32
}

/// Squared L2 distance (no sqrt; monotone for ranking purposes).
#[inline(always)]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ─── StaticDefault variant ────────────────────────────────────────────────────

/// Wraps `Hnsw` with a fixed conservative ef_search = 50.
pub struct StaticDefault {
    pub index: hnsw::Hnsw,
    ef: usize,
}

impl StaticDefault {
    pub fn build(data: &[Vec<f32>], m: usize, ef_construction: usize) -> Self {
        let mut index = hnsw::Hnsw::new(m, ef_construction);
        for (id, vec) in data.iter().enumerate() {
            index.insert(id, vec);
        }
        Self { index, ef: 50 }
    }
}

impl AnnVariant for StaticDefault {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        self.index
            .search(query, k, self.ef)
            .into_iter()
            .map(|(id, dist)| Hit { id, dist })
            .collect()
    }
    fn name(&self) -> &str {
        "StaticDefault(ef=50)"
    }
    fn memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }
}

// ─── StaticFast variant ───────────────────────────────────────────────────────

/// Wraps `Hnsw` with a fixed aggressive ef_search = 10.
pub struct StaticFast {
    pub index: hnsw::Hnsw,
    ef: usize,
}

impl StaticFast {
    pub fn build(data: &[Vec<f32>], m: usize, ef_construction: usize) -> Self {
        let mut index = hnsw::Hnsw::new(m, ef_construction);
        for (id, vec) in data.iter().enumerate() {
            index.insert(id, vec);
        }
        Self { index, ef: 10 }
    }
}

impl AnnVariant for StaticFast {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        self.index
            .search(query, k, self.ef)
            .into_iter()
            .map(|(id, dist)| Hit { id, dist })
            .collect()
    }
    fn name(&self) -> &str {
        "StaticFast(ef=10)"
    }
    fn memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }
}

// ─── BanditTuned variant ─────────────────────────────────────────────────────

/// HNSW whose ef_search is auto-tuned by a UCB1 bandit over candidate arms.
pub struct BanditTuned {
    pub index: hnsw::Hnsw,
    pub bandit: bandit::Ucb1Bandit,
    /// The arms map arm index → ef_search value.
    pub arms: Vec<usize>,
}

impl BanditTuned {
    /// Create from pre-built data.  `arms` = candidate ef values.
    pub fn build(data: &[Vec<f32>], m: usize, ef_construction: usize, arms: Vec<usize>) -> Self {
        let mut index = hnsw::Hnsw::new(m, ef_construction);
        for (id, vec) in data.iter().enumerate() {
            index.insert(id, vec);
        }
        let n_arms = arms.len();
        Self {
            index,
            bandit: bandit::Ucb1Bandit::new(n_arms),
            arms,
        }
    }

    /// Select an ef_search arm, run the query, observe reward, update bandit.
    /// `ground_truth` is used only for the reward signal (not returned).
    pub fn search_with_feedback(
        &mut self,
        query: &[f32],
        k: usize,
        ground_truth: &[Hit],
        latency_us: f64,
    ) -> Vec<Hit> {
        let arm = self.bandit.select();
        let ef = self.arms[arm];
        let results: Vec<Hit> = self
            .index
            .search(query, k, ef)
            .into_iter()
            .map(|(id, dist)| Hit { id, dist })
            .collect();

        // Reward: recall penalised by latency.  Shape: R = recall - alpha*latency_norm
        // latency_norm scales 1µs→0, 200µs→1 so it stays in [0,1].
        let recall = recall_at_k(&results, ground_truth, k);
        let latency_norm = (latency_us / 200.0_f64).min(1.0) as f32;
        let reward = recall - 0.15 * latency_norm;
        self.bandit.update(arm, reward);
        results
    }
}

impl AnnVariant for BanditTuned {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        // Exploitation-only: use the current best arm.
        let ef = self.arms[self.bandit.best_arm()];
        self.index
            .search(query, k, ef)
            .into_iter()
            .map(|(id, dist)| Hit { id, dist })
            .collect()
    }
    fn name(&self) -> &str {
        "BanditTuned(UCB1)"
    }
    fn memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }
}
