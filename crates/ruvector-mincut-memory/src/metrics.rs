//! Measurement types returned by every compaction run.

/// Outcome of a single compaction pass.
#[derive(Clone, Debug)]
pub struct CompactionResult {
    /// Number of entries before compaction.
    pub entries_before: usize,
    /// Number of entries after compaction.
    pub entries_after: usize,
    /// Graph edges before compaction (non-zero cells / 2).
    pub edges_before: usize,
    /// Graph edges after compaction.
    pub edges_after: usize,
    /// Wall-clock duration of the compaction call, in microseconds.
    pub latency_us: u64,
    /// Name of the strategy used.
    pub strategy: &'static str,
}

impl CompactionResult {
    pub fn entries_removed(&self) -> usize {
        self.entries_before.saturating_sub(self.entries_after)
    }

    /// Fraction of entries removed (0.0 – 1.0).
    pub fn reduction_ratio(&self) -> f32 {
        if self.entries_before == 0 {
            return 0.0;
        }
        self.entries_removed() as f32 / self.entries_before as f32
    }
}

impl std::fmt::Display for CompactionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {}/{} entries kept ({:.1}% removed) | edges {}->{} | {:.0} µs",
            self.strategy,
            self.entries_after,
            self.entries_before,
            self.reduction_ratio() * 100.0,
            self.edges_before,
            self.edges_after,
            self.latency_us,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_ratio_full_eviction() {
        let r = CompactionResult {
            entries_before: 100,
            entries_after: 0,
            edges_before: 50,
            edges_after: 0,
            latency_us: 100,
            strategy: "test",
        };
        assert!((r.reduction_ratio() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn reduction_ratio_no_eviction() {
        let r = CompactionResult {
            entries_before: 100,
            entries_after: 100,
            edges_before: 50,
            edges_after: 50,
            latency_us: 10,
            strategy: "test",
        };
        assert_eq!(r.reduction_ratio(), 0.0);
    }
}
