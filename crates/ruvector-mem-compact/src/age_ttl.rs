//! Baseline compactor: drop the oldest entries by age tick.
//!
//! This is the simplest possible strategy — no structural awareness.

use crate::compactor::{CompactionResult, MemoryCompactor, MemoryStore};
use std::time::Instant;

pub struct AgeTtlCompactor;

impl MemoryCompactor for AgeTtlCompactor {
    fn name(&self) -> &'static str {
        "AgeTtl"
    }

    fn compact(&self, store: &MemoryStore, target_ratio: f32) -> CompactionResult {
        let t = Instant::now();
        let n = store.entries.len();
        let keep_count = ((n as f32) * target_ratio).ceil() as usize;

        // Sort by age_tick descending (newest first), keep the newest.
        let mut indexed: Vec<(usize, u64)> = store
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.age_tick))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        indexed.truncate(keep_count);

        let kept_ids: Vec<u64> = indexed.iter().map(|(i, _)| store.entries[*i].id).collect();

        let kept_ratio = kept_ids.len() as f32 / n as f32;
        CompactionResult {
            kept_ids,
            kept_ratio,
            duration: t.elapsed(),
        }
    }
}
