use std::time::Instant;

use crate::{
    graph::{build_knn_graph, isolation_scores},
    store::{cosine_sim, CompactionConfig, CompactionResult, MemoryStore},
};

/// Core trait: take a store and configuration, return a compaction plan.
pub trait MemoryCompactor {
    fn compact(&self, store: &MemoryStore, config: &CompactionConfig) -> CompactionResult;
}

fn target_count(n: usize, retain_fraction: f32) -> usize {
    ((n as f32) * retain_fraction.clamp(0.0, 1.0)).round() as usize
}

fn build_result(
    store: &MemoryStore,
    retained_indices: Vec<usize>,
    start: Instant,
) -> CompactionResult {
    let n = store.len();
    let entries = store.entries();
    let retained_ids: Vec<u64> = retained_indices.iter().map(|&i| entries[i].id).collect();
    let retained_set: std::collections::HashSet<u64> = retained_ids.iter().copied().collect();
    let evicted_ids: Vec<u64> = entries
        .iter()
        .filter(|e| !retained_set.contains(&e.id))
        .map(|e| e.id)
        .collect();
    let entries_after = retained_ids.len();
    let retention_pct = if n == 0 {
        100.0
    } else {
        (entries_after as f32 / n as f32) * 100.0
    };
    CompactionResult {
        retained_ids,
        evicted_ids,
        entries_before: n,
        entries_after,
        retention_pct,
        duration_us: start.elapsed().as_micros() as u64,
    }
}

// ---------------------------------------------------------------------------
// Variant 1: Greedy age-based (oldest first)
// ---------------------------------------------------------------------------

/// Baseline compactor — evict the oldest entries by timestamp_ms.
///
/// O(n log n) in sort; O(n) otherwise.  Serves as the accuracy floor.
pub struct GreedyAgeCompactor;

impl MemoryCompactor for GreedyAgeCompactor {
    fn compact(&self, store: &MemoryStore, config: &CompactionConfig) -> CompactionResult {
        let start = Instant::now();
        let entries = store.entries();
        let n = entries.len();
        let keep = target_count(n, config.retain_fraction);

        // Sort indices by descending timestamp (newest first = keep)
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by(|&a, &b| entries[b].timestamp_ms.cmp(&entries[a].timestamp_ms));

        // Also hard-evict entries beyond max_age
        let now_ms = entries.iter().map(|e| e.timestamp_ms).max().unwrap_or(0);

        let retained: Vec<usize> = idx
            .into_iter()
            .filter(|&i| now_ms.saturating_sub(entries[i].timestamp_ms) <= config.max_age_ms)
            .take(keep)
            .collect();

        build_result(store, retained, start)
    }
}

// ---------------------------------------------------------------------------
// Variant 2: Decay + diversity score
// ---------------------------------------------------------------------------

/// Retains entries that score high on a combined recency + uniqueness metric.
///
/// `score(e) = recency(e) * (1 - diversity_weight) + uniqueness(e) * diversity_weight`
///
/// - `recency(e)` = exp(-ln2 * age_ms / half_life_ms)  (1.0 = brand new)
/// - `uniqueness(e)` = 1 - max cosine-sim to already-retained entries (greedy)
///
/// The greedy uniqueness pass ensures the retained set spans the embedding
/// space rather than clustering around one topic.
pub struct DecayScoreCompactor;

impl MemoryCompactor for DecayScoreCompactor {
    fn compact(&self, store: &MemoryStore, config: &CompactionConfig) -> CompactionResult {
        let start = Instant::now();
        let entries = store.entries();
        let n = entries.len();
        let keep = target_count(n, config.retain_fraction);

        let now_ms = entries.iter().map(|e| e.timestamp_ms).max().unwrap_or(0);

        // Recency scores
        let recency: Vec<f32> = entries
            .iter()
            .map(|e| {
                let age = now_ms.saturating_sub(e.timestamp_ms) as f32;
                let hl = config.half_life_ms.max(1) as f32;
                (-std::f32::consts::LN_2 * age / hl).exp()
            })
            .collect();

        // Access bonus: entries with higher access_count get a small boost
        let max_access = entries
            .iter()
            .map(|e| e.access_count)
            .max()
            .unwrap_or(1)
            .max(1) as f32;
        let access_bonus: Vec<f32> = entries
            .iter()
            .map(|e| e.access_count as f32 / max_access * 0.2)
            .collect();

        // Greedy selection: pick the entry with highest combined score,
        // then penalise remaining entries for being similar to what's kept.
        let mut similarity_penalty = vec![0.0f32; n];
        let mut remaining: Vec<usize> = (0..n)
            .filter(|&i| {
                let age = now_ms.saturating_sub(entries[i].timestamp_ms);
                age <= config.max_age_ms
            })
            .collect();

        let mut retained: Vec<usize> = Vec::with_capacity(keep);

        while retained.len() < keep && !remaining.is_empty() {
            let best = remaining
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let sa = recency[a] * (1.0 - config.diversity_weight)
                        + (1.0 - similarity_penalty[a]) * config.diversity_weight
                        + access_bonus[a];
                    let sb = recency[b] * (1.0 - config.diversity_weight)
                        + (1.0 - similarity_penalty[b]) * config.diversity_weight
                        + access_bonus[b];
                    sa.partial_cmp(&sb).unwrap()
                })
                .unwrap();

            retained.push(best);
            remaining.retain(|&i| i != best);

            // Update similarity penalty for remaining entries
            let kept_vec = &entries[best].vector;
            for &i in &remaining {
                let sim = cosine_sim(kept_vec, &entries[i].vector);
                if sim > similarity_penalty[i] {
                    similarity_penalty[i] = sim;
                }
            }
        }

        build_result(store, retained, start)
    }
}

// ---------------------------------------------------------------------------
// Variant 3: MinCut-guided graph compaction
// ---------------------------------------------------------------------------

/// Builds a k-NN similarity graph and evicts the most isolated nodes first.
///
/// Isolation score = 1 − (mean cosine-sim to k nearest neighbours).
///
/// This approximates the min-cut criterion: highly-isolated nodes lie near
/// cut boundaries and contribute least to the global semantic coherence of
/// the retained memory set.  Evicting them preserves the dense cores of
/// each topic cluster.
pub struct MinCutCompactor;

impl MemoryCompactor for MinCutCompactor {
    fn compact(&self, store: &MemoryStore, config: &CompactionConfig) -> CompactionResult {
        let start = Instant::now();
        let entries = store.entries();
        let n = entries.len();
        let keep = target_count(n, config.retain_fraction);

        let now_ms = entries.iter().map(|e| e.timestamp_ms).max().unwrap_or(0);

        // Hard-evict entries that exceed max_age before graph analysis
        let eligible: Vec<usize> = (0..n)
            .filter(|&i| {
                let age = now_ms.saturating_sub(entries[i].timestamp_ms);
                age <= config.max_age_ms
            })
            .collect();

        // Build k-NN graph over eligible entries only
        let vecs: Vec<&[f32]> = eligible
            .iter()
            .map(|&i| entries[i].vector.as_slice())
            .collect();

        let adj = build_knn_graph(&vecs, config.top_k_neighbors, config.similarity_threshold);
        let scores = isolation_scores(&adj);

        // Sort eligible indices by ascending isolation score (keep dense/connected nodes)
        let mut ranked: Vec<(usize, f32)> = eligible
            .iter()
            .copied()
            .zip(scores.iter().copied())
            .collect();
        ranked.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Retain the keep-many least-isolated entries
        let retained: Vec<usize> = ranked.iter().take(keep).map(|&(i, _)| i).collect();

        build_result(store, retained, start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryEntry, MemoryStore};

    fn make_store(n: usize, dim: usize, now_ms: u64) -> MemoryStore {
        let mut store = MemoryStore::new();
        for i in 0u64..n as u64 {
            // Half of entries are "old" (timestamp near 0), half are "new"
            let ts = if i < n as u64 / 2 {
                i * 100
            } else {
                now_ms - i * 100
            };
            let v: Vec<f32> = (0..dim).map(|d| (i as f32 + d as f32).sin()).collect();
            store.insert(MemoryEntry::new(i, v, ts));
        }
        store
    }

    fn quality_centroid_similarity(before: &MemoryStore, after: &MemoryStore) -> f32 {
        let c_before = before.centroid().unwrap();
        let c_after = after.centroid().unwrap();
        cosine_sim(&c_before, &c_after)
    }

    #[test]
    fn greedy_retains_correct_count() {
        let store = make_store(100, 16, 10_000_000);
        let cfg = CompactionConfig {
            retain_fraction: 0.5,
            ..Default::default()
        };
        let result = GreedyAgeCompactor.compact(&store, &cfg);
        assert_eq!(result.entries_after, 50);
        assert_eq!(result.entries_before, 100);
        assert_eq!(result.retained_ids.len() + result.evicted_ids.len(), 100);
    }

    #[test]
    fn decay_retains_correct_count() {
        let store = make_store(100, 16, 10_000_000);
        let cfg = CompactionConfig {
            retain_fraction: 0.5,
            ..Default::default()
        };
        let result = DecayScoreCompactor.compact(&store, &cfg);
        assert_eq!(result.entries_after, 50);
    }

    #[test]
    fn mincut_retains_correct_count() {
        let store = make_store(100, 16, 10_000_000);
        let cfg = CompactionConfig {
            retain_fraction: 0.5,
            ..Default::default()
        };
        let result = MinCutCompactor.compact(&store, &cfg);
        assert_eq!(result.entries_after, 50);
    }

    /// Acceptance test: MinCutCompactor must preserve centroid quality ≥ 90 %.
    #[test]
    fn mincut_preserves_centroid_quality() {
        let mut store = make_store(200, 32, 50_000_000);
        let before_centroid = store.centroid().unwrap();

        let cfg = CompactionConfig {
            retain_fraction: 0.5,
            similarity_threshold: 0.0,
            top_k_neighbors: 6,
            ..Default::default()
        };
        let result = MinCutCompactor.compact(&store, &cfg);
        store.apply_compaction(&result);

        let after_centroid = store.centroid().unwrap();
        let sim = cosine_sim(&before_centroid, &after_centroid);
        assert!(
            sim >= 0.90,
            "centroid similarity {sim:.4} < 0.90 — quality threshold not met"
        );
    }

    /// Acceptance test: DecayScoreCompactor must preserve centroid quality ≥ 85 %.
    #[test]
    fn decay_preserves_centroid_quality() {
        let mut store = make_store(200, 32, 50_000_000);
        let before_centroid = store.centroid().unwrap();

        let cfg = CompactionConfig {
            retain_fraction: 0.5,
            diversity_weight: 0.4,
            ..Default::default()
        };
        let result = DecayScoreCompactor.compact(&store, &cfg);
        store.apply_compaction(&result);

        let after_centroid = store.centroid().unwrap();
        let sim = cosine_sim(&before_centroid, &after_centroid);
        assert!(
            sim >= 0.85,
            "centroid similarity {sim:.4} < 0.85 — quality threshold not met"
        );
    }

    #[test]
    fn all_retained_ids_exist_in_original() {
        let store = make_store(50, 8, 1_000_000);
        let cfg = CompactionConfig {
            retain_fraction: 0.6,
            ..Default::default()
        };
        let all_ids: std::collections::HashSet<u64> =
            store.entries().iter().map(|e| e.id).collect();

        for result in [
            GreedyAgeCompactor.compact(&store, &cfg),
            DecayScoreCompactor.compact(&store, &cfg),
            MinCutCompactor.compact(&store, &cfg),
        ] {
            for id in &result.retained_ids {
                assert!(
                    all_ids.contains(id),
                    "retained id {id} not in original store"
                );
            }
        }
    }
}
