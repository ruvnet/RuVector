//! Temporal buckets keyed by hour.
//!
//! # Example
//!
//! ```
//! use ruvector_field::model::NodeId;
//! use ruvector_field::storage::TemporalBuckets;
//! let mut tb = TemporalBuckets::new();
//! tb.insert(NodeId(1), 3600_000_000_000); // 1 hour in ns
//! let ids = tb.range(0, 7200_000_000_000);
//! assert_eq!(ids, vec![NodeId(1)]);
//! ```

use std::collections::BTreeMap;

use crate::model::NodeId;

/// Nanoseconds per hour.
pub const NS_PER_HOUR: u64 = 3_600 * 1_000_000_000;

/// Bucketed temporal index.
#[derive(Debug, Clone, Default)]
pub struct TemporalBuckets {
    buckets: BTreeMap<u64, Vec<NodeId>>,
}

impl TemporalBuckets {
    /// Empty bucket set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the bucket key for a timestamp.
    pub fn bucket_for(ts_ns: u64) -> u64 {
        ts_ns / NS_PER_HOUR
    }

    /// Insert a node id into the appropriate bucket.
    pub fn insert(&mut self, node: NodeId, ts_ns: u64) -> u64 {
        let key = Self::bucket_for(ts_ns);
        self.buckets.entry(key).or_default().push(node);
        key
    }

    /// All node ids whose bucket falls in `[from_ns, to_ns]` (inclusive on both ends).
    pub fn range(&self, from_ns: u64, to_ns: u64) -> Vec<NodeId> {
        let from = Self::bucket_for(from_ns);
        let to = Self::bucket_for(to_ns);
        let mut out = Vec::new();
        for (_, ids) in self.buckets.range(from..=to) {
            out.extend(ids.iter().copied());
        }
        out
    }

    /// Number of occupied buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Total number of nodes across all buckets.
    pub fn total(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }
}
