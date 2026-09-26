//! Write-Ahead Log for pending vector inserts.
//!
//! Incoming vectors are buffered here instead of immediately grafted into the
//! main navigable graph. The WAL is always searchable via brute-force linear
//! scan, ensuring no inserted vector is ever invisible to queries.
//!
//! When the merge gate fires the WAL is drained into the main graph in a
//! single batch, amortising the O(|WAL| × ef × M × D) graph-update cost.

/// Single entry in the write-ahead log.
#[derive(Clone, Debug)]
pub struct WalEntry {
    pub id: u64,
    pub vector: Vec<f32>,
}

/// In-memory write-ahead log.
///
/// The log is bounded in size by `capacity`. Callers must check
/// [`VectorWal::is_full`] or inspect the return value of [`VectorWal::push`]
/// to decide whether to trigger a merge before the next push.
pub struct VectorWal {
    entries: Vec<WalEntry>,
    capacity: usize,
    dims: usize,
}

impl VectorWal {
    /// Create a new WAL with the given capacity and vector dimension.
    pub fn new(capacity: usize, dims: usize) -> Self {
        VectorWal {
            entries: Vec::with_capacity(capacity),
            capacity,
            dims,
        }
    }

    /// Append a vector. Returns `true` when the WAL reached capacity after
    /// this push (i.e. the merge gate should now be evaluated).
    pub fn push(&mut self, id: u64, vector: Vec<f32>) -> bool {
        debug_assert_eq!(vector.len(), self.dims, "dimension mismatch on WAL push");
        self.entries.push(WalEntry { id, vector });
        self.entries.len() >= self.capacity
    }

    /// Drain all entries from the WAL (for merging into the main graph).
    pub fn drain(&mut self) -> Vec<WalEntry> {
        std::mem::take(&mut self.entries)
    }

    /// Brute-force nearest-neighbour search over WAL entries.
    ///
    /// Returns up to `k` `(id, l2_squared_distance)` pairs, sorted ascending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut heap: Vec<(u64, f32)> = self
            .entries
            .iter()
            .map(|e| (e.id, l2_sq(query, &e.vector)))
            .collect();
        heap.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        heap.truncate(k);
        heap
    }

    /// Current number of pending entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the WAL contains no pending entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True when the WAL has reached its capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Iterate over WAL entries (used by coherence scoring).
    pub fn iter(&self) -> impl Iterator<Item = &WalEntry> {
        self.entries.iter()
    }

    /// Number of dimensions each vector in this WAL must have.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }
}

/// Squared L2 distance (no sqrt — consistent with graph.rs).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_returns_true_at_capacity() {
        let mut wal = VectorWal::new(4, 2);
        assert!(!wal.push(0, vec![1.0, 0.0]));
        assert!(!wal.push(1, vec![2.0, 0.0]));
        assert!(!wal.push(2, vec![3.0, 0.0]));
        assert!(wal.push(3, vec![4.0, 0.0])); // 4th push hits capacity
    }

    #[test]
    fn drain_clears_wal() {
        let mut wal = VectorWal::new(10, 2);
        wal.push(0, vec![1.0, 2.0]);
        wal.push(1, vec![3.0, 4.0]);
        let drained = wal.drain();
        assert_eq!(drained.len(), 2);
        assert!(wal.is_empty());
    }

    #[test]
    fn search_returns_nearest_first() {
        let mut wal = VectorWal::new(10, 2);
        wal.push(0, vec![0.0, 0.0]);
        wal.push(1, vec![10.0, 10.0]);
        wal.push(2, vec![0.5, 0.5]);

        let q = vec![0.0, 0.0];
        let results = wal.search(&q, 2);
        assert_eq!(results[0].0, 0, "closest should be id=0");
        assert_eq!(results[1].0, 2, "second closest should be id=2");
    }
}
