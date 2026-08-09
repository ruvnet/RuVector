//! Cache statistics with interior-mutability via raw atomics-free counters.
//!
//! Counters are updated through shared-reference `record_*` helpers by
//! wrapping them in `std::cell::Cell`. This is single-threaded; for
//! concurrent use a future production crate would use `AtomicU64`.

use std::cell::Cell;

/// Hit/miss statistics for a cache backend.
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    // Interior-mutability accumulators (single-threaded).
    hits_cell: Cell<u64>,
    misses_cell: Cell<u64>,
    evictions_cell: Cell<u64>,
}

impl CacheStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_hit(&self) {
        self.hits_cell.set(self.hits_cell.get() + 1);
    }

    pub fn record_miss(&self) {
        self.misses_cell.set(self.misses_cell.get() + 1);
    }

    pub fn record_evictions(&self, n: u64) {
        self.evictions_cell.set(self.evictions_cell.get() + n);
    }

    /// Flush cell values into the public fields and return a clone.
    pub fn snapshot(&self) -> CacheStats {
        CacheStats {
            hits: self.hits_cell.get(),
            misses: self.misses_cell.get(),
            evictions: self.evictions_cell.get(),
            hits_cell: Cell::new(0),
            misses_cell: Cell::new(0),
            evictions_cell: Cell::new(0),
        }
    }

    pub fn hit_rate(&self) -> f32 {
        let h = self.hits_cell.get();
        let m = self.misses_cell.get();
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f32 / total as f32
        }
    }
}

impl Clone for CacheStats {
    fn clone(&self) -> Self {
        CacheStats {
            hits: self.hits_cell.get(),
            misses: self.misses_cell.get(),
            evictions: self.evictions_cell.get(),
            hits_cell: Cell::new(self.hits_cell.get()),
            misses_cell: Cell::new(self.misses_cell.get()),
            evictions_cell: Cell::new(self.evictions_cell.get()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_rate_zero_when_empty() {
        let s = CacheStats::new();
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn hit_rate_correct() {
        let s = CacheStats::new();
        s.record_hit();
        s.record_hit();
        s.record_miss();
        // 2 hits / 3 total
        let r = s.hit_rate();
        assert!((r - 2.0 / 3.0).abs() < 1e-5, "hit_rate = {r}");
    }

    #[test]
    fn snapshot_freezes_values() {
        let s = CacheStats::new();
        s.record_hit();
        let snap = s.snapshot();
        s.record_hit(); // after snapshot
        assert_eq!(snap.hits, 1, "snapshot must freeze at 1 hit");
    }
}
