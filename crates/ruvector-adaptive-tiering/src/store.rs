//! Generic tiered vector store parameterised by a `Scorer`.

use crate::{
    distance::l2_sq,
    temperature::{TieringConfig, VectorMeta},
    BatchStats, Scorer, SearchResult, Tier,
};

struct Entry {
    id: u64,
    vector: Vec<f32>,
    meta: VectorMeta,
    tier: Tier,
}

/// Generic tiered store.
///
/// * Vectors start in the Cold tier.
/// * `evaluate_tiers(epoch)` re-scores every vector and reassigns tiers.
/// * `search(query, k)` scans all three tiers in order, merging by L2.
pub struct TieredStore<S> {
    entries: Vec<Entry>,
    config: TieringConfig,
    scorer: S,
}

impl<S: Scorer> TieredStore<S> {
    pub fn new(config: TieringConfig, scorer: S) -> Self {
        Self {
            entries: Vec::new(),
            config,
            scorer,
        }
    }

    /// Insert a vector; starts in the Cold tier with default metadata.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.entries.push(Entry {
            id,
            vector,
            meta: VectorMeta::default(),
            tier: Tier::Cold,
        });
    }

    /// Increment access count and update last_access_epoch for `id`.
    pub fn record_access(&mut self, id: u64, epoch: u64) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.meta.access_count += 1;
            e.meta.last_access_epoch = epoch;
        }
    }

    /// Re-score every vector and reassign tiers.
    pub fn evaluate_tiers(&mut self, current_epoch: u64) {
        if self.scorer.needs_coherence() {
            self.recompute_coherence();
        }

        let cfg = &self.config;
        let scorer = &self.scorer;

        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, scorer.score(&e.meta, current_epoch, cfg)))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, (idx, _score)) in scored.iter().enumerate() {
            self.entries[*idx].tier = if rank < cfg.hot_capacity {
                Tier::Hot
            } else if rank < cfg.hot_capacity + cfg.warm_capacity {
                Tier::Warm
            } else {
                Tier::Cold
            };
        }
    }

    /// Brute-force k-nearest-neighbour search across all tiers.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut candidates: Vec<SearchResult> = self
            .entries
            .iter()
            .map(|e| SearchResult {
                id: e.id,
                dist_sq: l2_sq(query, &e.vector),
                tier: e.tier,
            })
            .collect();

        let k = k.min(candidates.len());
        if k < candidates.len() {
            candidates.select_nth_unstable_by(k - 1, |a, b| {
                a.dist_sq
                    .partial_cmp(&b.dist_sq)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| {
            a.dist_sq
                .partial_cmp(&b.dist_sq)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    pub fn tier_of(&self, id: u64) -> Option<Tier> {
        self.entries.iter().find(|e| e.id == id).map(|e| e.tier)
    }

    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let (mut h, mut w, mut c) = (0, 0, 0);
        for e in &self.entries {
            match e.tier {
                Tier::Hot => h += 1,
                Tier::Warm => w += 1,
                Tier::Cold => c += 1,
            }
        }
        (h, w, c)
    }

    pub fn accumulate_stats(&self, results: &[SearchResult], stats: &mut BatchStats) {
        for r in results {
            stats.record_result(r.tier);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -----------------------------------------------------------------------

    fn recompute_coherence(&mut self) {
        let n = self.entries.len();
        let k = self.config.coherence_sample_k;
        let radius_sq = self.config.centrality_radius * self.config.centrality_radius;

        let step = if n <= 4 * k { 1 } else { 4 };

        // Snapshot pool indices and vectors to avoid borrow conflicts.
        let pool: Vec<(usize, Vec<f32>)> = self
            .entries
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(i, e)| (i, e.vector.clone()))
            .collect();

        let mut coherence_scores: Vec<f32> = Vec::with_capacity(n);
        let mut graph_degrees: Vec<u32> = Vec::with_capacity(n);

        for (i, e) in self.entries.iter().enumerate() {
            let v = &e.vector;

            let mut dists: Vec<f32> = pool
                .iter()
                .filter(|(pi, _)| *pi != i)
                .map(|(_, pv)| l2_sq(v, pv))
                .collect();

            if dists.is_empty() {
                coherence_scores.push(0.5);
                graph_degrees.push(0);
                continue;
            }

            let take = k.min(dists.len());
            dists.select_nth_unstable_by(take - 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(take);

            let mean_l2 = dists.iter().map(|d| d.sqrt()).sum::<f32>() / take as f32;
            let coherence = 1.0 / (1.0 + mean_l2);
            let degree = dists.iter().filter(|&&d| d <= radius_sq).count() as u32;

            coherence_scores.push(coherence);
            graph_degrees.push(degree);
        }

        for (e, (coh, deg)) in self
            .entries
            .iter_mut()
            .zip(coherence_scores.into_iter().zip(graph_degrees.into_iter()))
        {
            e.meta.coherence_score = coh;
            e.meta.graph_degree = deg;
        }
    }
}
