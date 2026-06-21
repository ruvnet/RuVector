//! MemTable: the mutable L0 write buffer.
//!
//! Accepts any number of inserts in O(1) amortised time. Searches are O(N·D)
//! brute-force; this is acceptable when the memtable is bounded (≤ l0_max entries).

use crate::{brute_force_knn, sq_dist};

/// Flat, mutable write buffer (L0 tier of the LSM-ANN hierarchy).
#[derive(Default, Clone, Debug)]
pub struct MemTable {
    pub(crate) entries: Vec<(u64, Vec<f32>)>,
}

impl MemTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a vector.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if let Some(pos) = self.entries.iter().position(|(i, _)| *i == id) {
            self.entries[pos].1 = vector;
        } else {
            self.entries.push((id, vector));
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Brute-force k-nearest search.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        brute_force_knn(&self.entries, query, k)
    }

    /// Drain all entries, leaving the memtable empty.
    pub fn drain(&mut self) -> Vec<(u64, Vec<f32>)> {
        std::mem::take(&mut self.entries)
    }

    /// Estimate memory usage in bytes (vectors only, ignoring Vec overhead).
    pub fn memory_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(_, v)| 8 + v.len() * 4)
            .sum::<usize>()
    }

    /// Exact search within this memtable for use in recall validation.
    pub fn exact_search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut results: Vec<(u64, f32)> = self
            .entries
            .iter()
            .map(|(id, v)| (*id, sq_dist(v, query)))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }
}
