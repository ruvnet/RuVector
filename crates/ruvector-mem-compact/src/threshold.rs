//! Threshold compactor: greedily deduplicate by cosine similarity.
//!
//! Scans entries in order; drops an entry if it has cosine similarity ≥
//! threshold to any already-kept entry.  This is a pointwise greedy approach
//! with no transitivity — it misses the case where A≈B≈C but A and C are
//! moderately different.

use crate::compactor::{cosine_sim, CompactionResult, MemoryCompactor, MemoryStore};
use std::time::Instant;

pub struct ThresholdCompactor {
    /// Drop entry if its max similarity to any kept entry exceeds this.
    pub similarity_threshold: f32,
}

impl ThresholdCompactor {
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
        }
    }
}

impl MemoryCompactor for ThresholdCompactor {
    fn name(&self) -> &'static str {
        "ThresholdMerge"
    }

    fn compact(&self, store: &MemoryStore, target_ratio: f32) -> CompactionResult {
        let t = Instant::now();
        let n = store.entries.len();
        let keep_count = ((n as f32) * target_ratio).ceil() as usize;

        let mut kept_indices: Vec<usize> = Vec::with_capacity(keep_count);

        'outer: for (i, entry) in store.entries.iter().enumerate() {
            for &j in &kept_indices {
                let sim = cosine_sim(&entry.vector, &store.entries[j].vector);
                if sim >= self.similarity_threshold {
                    // Redundant — drop it.
                    continue 'outer;
                }
            }
            kept_indices.push(i);
            if kept_indices.len() >= keep_count {
                break;
            }
        }

        // If we kept too few (threshold was too tight), pad with remaining by age.
        if kept_indices.len() < keep_count {
            let kept_set: std::collections::HashSet<usize> = kept_indices.iter().copied().collect();
            let mut extras: Vec<(usize, u64)> = store
                .entries
                .iter()
                .enumerate()
                .filter(|(i, _)| !kept_set.contains(i))
                .map(|(i, e)| (i, e.age_tick))
                .collect();
            extras.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for (idx, _) in extras.into_iter().take(keep_count - kept_indices.len()) {
                kept_indices.push(idx);
            }
        }

        let kept_ids: Vec<u64> = kept_indices.iter().map(|&i| store.entries[i].id).collect();
        let kept_ratio = kept_ids.len() as f32 / n as f32;
        CompactionResult {
            kept_ids,
            kept_ratio,
            duration: t.elapsed(),
        }
    }
}
