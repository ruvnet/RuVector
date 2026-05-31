//! Baseline: flat linear scan over all vectors, no tiering.

use crate::{l2_sq, SearchResult, Tier, TierStats, TieredMemoryStore};

struct Entry {
    id: u64,
    vector: Vec<f32>,
}

/// Flat memory store — every vector lives in RAM, every search is a full linear scan.
/// Serves as the latency baseline for the tiered variants.
pub struct FlatMemory {
    entries: Vec<Entry>,
    dims: usize,
}

impl FlatMemory {
    pub fn new(dims: usize) -> Self {
        FlatMemory {
            entries: Vec::new(),
            dims,
        }
    }
}

impl TieredMemoryStore for FlatMemory {
    fn name(&self) -> &str {
        "FlatMemory (baseline)"
    }

    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims);
        self.entries.push(Entry { id, vector });
    }

    fn search(&mut self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(f32, u64)> = self
            .entries
            .iter()
            .map(|e| (l2_sq(query, &e.vector), e.id))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored
            .into_iter()
            .take(k)
            .map(|(dist, id)| SearchResult {
                id,
                distance: dist,
                tier: Tier::Hot,
            })
            .collect()
    }

    fn tier_stats(&self) -> TierStats {
        let bytes = self.entries.len() * self.dims * 4;
        TierStats {
            hot_count: self.entries.len(),
            warm_count: 0,
            cold_count: 0,
            hot_bytes: bytes,
            warm_bytes: 0,
            cold_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_finds_nearest() {
        let dims = 4;
        let mut store = FlatMemory::new(dims);
        store.insert(0, vec![1.0, 0.0, 0.0, 0.0]);
        store.insert(1, vec![0.0, 1.0, 0.0, 0.0]);
        store.insert(2, vec![10.0, 10.0, 10.0, 10.0]);
        let results = store.search(&[1.1, 0.0, 0.0, 0.0], 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn flat_returns_k_results() {
        let dims = 4;
        let mut store = FlatMemory::new(dims);
        for i in 0..20u64 {
            store.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }
        let results = store.search(&[5.0, 0.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 5);
    }
}
