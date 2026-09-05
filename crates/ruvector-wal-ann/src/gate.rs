//! Merge gate trait and three concrete implementations.
//!
//! A merge gate answers a single question:
//! "Given the current WAL size and the index coherence score, should we flush
//! the WAL into the main navigable graph right now?"
//!
//! ## Implementations
//!
//! | Gate | Trigger condition | Trade-off |
//! |------|-------------------|-----------|
//! | [`EagerGate`] | WAL ≥ threshold (small) | Frequent merges, low WAL overhead |
//! | [`LazyGate`] | WAL ≥ threshold (large) | Infrequent merges, high WAL overhead |
//! | [`CoherenceGate`] | coherence < min OR WAL ≥ max | Quality-driven merges |
//!
//! ## Coherence score semantics
//!
//! The coherence score (0.0 – 1.0) quantifies how well the WAL is represented
//! by the current main graph. Formally it is:
//!
//! ```text
//! isolation = mean( min_dist(wal_vec, graph) for wal_vec in wal )
//! coherence = 1 / (1 + isolation)
//! ```
//!
//! * **High coherence** (→ 1.0): WAL vectors are close to existing graph
//!   nodes — merging soon adds little navigational value.
//! * **Low coherence** (→ 0.0): WAL vectors are isolated from the graph —
//!   merging is urgent to maintain search coverage.

/// Trait implemented by all merge gate strategies.
pub trait MergeGate: Send + Sync {
    /// Return `true` when the WAL should be flushed into the main graph.
    ///
    /// `wal_size`       — current number of pending WAL entries.
    /// `coherence`      — current coherence score (0.0 – 1.0; see module docs).
    fn should_merge(&self, wal_size: usize, coherence: f32) -> bool;

    /// Human-readable variant name (used in benchmark output).
    fn name(&self) -> &'static str;
}

/// Merge after every `threshold` inserts — small threshold → frequent merges.
///
/// Approximates the naive strategy of rebuilding the graph every N inserts.
/// Insert throughput is dominated by merge cost when the threshold is small.
pub struct EagerGate {
    /// Merge when WAL reaches this size. Typical value: 32–64.
    pub threshold: usize,
}

impl MergeGate for EagerGate {
    fn should_merge(&self, wal_size: usize, _coherence: f32) -> bool {
        wal_size >= self.threshold
    }

    fn name(&self) -> &'static str {
        "EagerMerge"
    }
}

/// Merge after `threshold` inserts — large threshold → infrequent merges.
///
/// WAL scan cost grows with WAL size. For very large thresholds, WAL search
/// latency dominates query time. The trade-off is fewer, cheaper graph
/// rebuilds at the cost of higher per-query WAL scan overhead.
pub struct LazyGate {
    /// Merge when WAL reaches this size. Typical value: 256–1024.
    pub threshold: usize,
}

impl MergeGate for LazyGate {
    fn should_merge(&self, wal_size: usize, _coherence: f32) -> bool {
        wal_size >= self.threshold
    }

    fn name(&self) -> &'static str {
        "LazyMerge"
    }
}

/// Coherence-driven merge gate.
///
/// Merges when:
/// 1. **Isolation alarm**: coherence score drops below `min_coherence`, OR
/// 2. **Overflow guard**: WAL reaches `max_wal_size`
///
/// This balances quality and throughput: the gate delays merges when new
/// vectors are close to existing graph nodes (no urgency) but fires early when
/// WAL vectors are far from the graph (risking poor coverage on queries that
/// should hit the WAL region).
pub struct CoherenceGate {
    /// Minimum acceptable coherence. Merge fires below this value.
    pub min_coherence: f32,
    /// Hard cap on WAL size (overflow guard regardless of coherence).
    pub max_wal_size: usize,
}

impl MergeGate for CoherenceGate {
    fn should_merge(&self, wal_size: usize, coherence: f32) -> bool {
        coherence < self.min_coherence || wal_size >= self.max_wal_size
    }

    fn name(&self) -> &'static str {
        "CoherenceGatedMerge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_gate_fires_at_threshold() {
        let g = EagerGate { threshold: 32 };
        assert!(!g.should_merge(31, 0.0));
        assert!(g.should_merge(32, 0.0));
        assert!(g.should_merge(100, 0.0));
    }

    #[test]
    fn lazy_gate_fires_at_threshold() {
        let g = LazyGate { threshold: 256 };
        assert!(!g.should_merge(255, 0.0));
        assert!(g.should_merge(256, 0.5));
    }

    #[test]
    fn coherence_gate_fires_on_low_coherence() {
        let g = CoherenceGate {
            min_coherence: 0.5,
            max_wal_size: 1000,
        };
        assert!(g.should_merge(10, 0.49)); // low coherence → merge
        assert!(!g.should_merge(10, 0.51)); // high coherence, small WAL → wait
    }

    #[test]
    fn coherence_gate_fires_on_overflow() {
        let g = CoherenceGate {
            min_coherence: 0.5,
            max_wal_size: 100,
        };
        assert!(g.should_merge(100, 0.99)); // overflow even with high coherence
    }
}
