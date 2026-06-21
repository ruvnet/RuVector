//! Core memory entry and in-memory store.

use crate::scoring::cosine_sim;

/// A single agent memory record.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Stable identifier.
    pub id: u64,
    /// Dense embedding vector.
    pub vector: Vec<f32>,
    /// Optional human-readable label (for debugging).
    pub label: Option<String>,
    /// Logical clock tick at creation.
    pub created_at: u64,
    /// Logical clock tick at most recent access.
    pub last_accessed_at: u64,
    /// Number of times this entry has been accessed since insertion.
    pub access_count: u64,
}

impl MemoryEntry {
    pub fn new(id: u64, vector: Vec<f32>, now: u64) -> Self {
        Self {
            id,
            vector,
            label: None,
            created_at: now,
            last_accessed_at: now,
            access_count: 0,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Record one access at logical time `now`.
    pub fn touch(&mut self, now: u64) {
        self.last_accessed_at = now;
        self.access_count += 1;
    }
}

/// Search result: (entry id, cosine similarity score).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

/// Flat in-memory vector store with logical-clock tracking.
///
/// All search is exact (brute-force).  This crate's focus is on the
/// *compaction* layer; a production deployment would replace the scan
/// with an HNSW or IVF index.
pub struct MemoryStore {
    entries: Vec<MemoryEntry>,
    clock: u64,
    pub dims: usize,
}

impl MemoryStore {
    pub fn new(dims: usize) -> Self {
        Self {
            entries: Vec::new(),
            clock: 0,
            dims,
        }
    }

    /// Advance the logical clock by one tick and return the new tick.
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Insert a new memory entry.  Returns the assigned id.
    pub fn insert(&mut self, vector: Vec<f32>) -> u64 {
        assert_eq!(vector.len(), self.dims, "dimension mismatch");
        let now = self.tick();
        let id = self.entries.len() as u64;
        self.entries.push(MemoryEntry::new(id, vector, now));
        id
    }

    /// Record an access for the entry at `index` (0-based position).
    pub fn access_by_index(&mut self, index: usize) {
        let now = self.tick();
        if let Some(e) = self.entries.get_mut(index) {
            e.touch(now);
        }
    }

    /// Exact k-nearest-neighbor search using cosine similarity.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine_sim(query, &e.vector)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
            .into_iter()
            .take(k)
            .map(|(i, s)| SearchResult {
                id: self.entries[i].id,
                score: s,
            })
            .collect()
    }

    /// Return all entries as a slice (read-only).
    pub fn entries(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// Replace all entries with the given subset (compaction result).
    pub fn replace_entries(&mut self, new_entries: Vec<MemoryEntry>) {
        self.entries = new_entries;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search() {
        let mut store = MemoryStore::new(3);
        store.insert(vec![1.0, 0.0, 0.0]);
        store.insert(vec![0.0, 1.0, 0.0]);
        store.insert(vec![0.0, 0.0, 1.0]);

        let results = store.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn touch_updates_clock() {
        let mut store = MemoryStore::new(2);
        store.insert(vec![1.0, 0.0]);
        store.access_by_index(0);
        assert_eq!(store.entries()[0].access_count, 1);
        assert!(store.entries()[0].last_accessed_at > store.entries()[0].created_at);
    }
}
