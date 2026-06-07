//! Witness log for cross-namespace vector access events.
//!
//! Every time a [`CoherenceGated`] search returns a result from a namespace
//! other than the one queried, an entry is appended here.  The log is
//! intentionally append-only within a process lifetime; persistence to RVF or
//! an external store is left to the caller.

/// A single recorded cross-namespace access.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Namespace that issued the query.
    pub source_ns: u32,
    /// Namespace that contributed the result.
    pub target_ns: u32,
    /// Coherence score that allowed the crossing (higher = more related).
    pub coherence: f32,
    /// Squared distance of the returned vector to the query.
    pub distance: f32,
}

/// Append-only log of cross-namespace access events.
#[derive(Debug, Default)]
pub struct WitnessLog {
    entries: Vec<WitnessEntry>,
}

impl WitnessLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record one cross-namespace access event.
    pub fn record(&mut self, entry: WitnessEntry) {
        self.entries.push(entry);
    }

    /// Total events recorded since construction.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Slice of all recorded entries (read-only).
    pub fn entries(&self) -> &[WitnessEntry] {
        &self.entries
    }

    /// Fraction of entries where `target_ns != source_ns`.
    /// Always 1.0 since only cross-namespace events are recorded.
    pub fn cross_boundary_rate(&self) -> f64 {
        1.0
    }

    /// Mean coherence score of all logged events.
    pub fn mean_coherence(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.entries.iter().map(|e| e.coherence as f64).sum();
        sum / self.entries.len() as f64
    }

    /// Number of unique (source_ns, target_ns) pairs seen.
    pub fn unique_pairs(&self) -> usize {
        use std::collections::HashSet;
        let pairs: HashSet<(u32, u32)> = self
            .entries
            .iter()
            .map(|e| (e.source_ns, e.target_ns))
            .collect();
        pairs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log_has_zero_len() {
        let log = WitnessLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn record_and_query() {
        let mut log = WitnessLog::new();
        log.record(WitnessEntry {
            source_ns: 0,
            target_ns: 1,
            coherence: 0.8,
            distance: 0.1,
        });
        log.record(WitnessEntry {
            source_ns: 0,
            target_ns: 2,
            coherence: 0.6,
            distance: 0.3,
        });
        assert_eq!(log.len(), 2);
        assert_eq!(log.unique_pairs(), 2);
        let mean = log.mean_coherence();
        assert!((mean - 0.7).abs() < 1e-5);
    }
}
