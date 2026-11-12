//! Isolate-wide resident-set registry (ADR-351 §6.1): caps resident shard
//! memory across every shard loaded in one isolate, evicting the least
//! recently used shards first. Pure bookkeeping — the host drops the evicted
//! shards' resident state (they cold-load again on next use). Logical ticks,
//! no clock.
//!
//! Shards register as soon as they are loaded (not after a successful op),
//! with their full per-row footprint ([`crate::VectorShard::resident_bytes`]).
//! Registration never fails: the per-shard cap
//! ([`crate::shard::SHARD_RESIDENT_CAP_BYTES`]) keeps a single shard within
//! the isolate cap, and if one were ever larger every other shard is
//! evicted instead of refusing it.

use std::collections::BTreeMap;

/// Default isolate cap: 56 MB of resident shard memory (§6.1, `[U]`), which
/// fits four 14 MB M1 shards; a fifth evicts one.
pub const ISOLATE_RESIDENT_CAP_BYTES: u64 = 56_000_000;

/// LRU registry keyed by DO name.
#[derive(Debug, Clone)]
pub struct ResidentRegistry {
    cap: u64,
    tick: u64,
    total: u64,
    entries: BTreeMap<String, (u64, u64)>,
}

impl Default for ResidentRegistry {
    fn default() -> Self {
        Self::new(ISOLATE_RESIDENT_CAP_BYTES)
    }
}

impl ResidentRegistry {
    /// A registry with `cap` bytes.
    pub fn new(cap: u64) -> Self {
        ResidentRegistry {
            cap,
            tick: 0,
            total: 0,
            entries: BTreeMap::new(),
        }
    }

    /// Record that `name` is resident with `bytes` and was just used.
    /// Returns the names evicted (least recently used first, never `name`)
    /// so the total fits the cap, or so that `name` is alone if it does not
    /// fit by itself.
    pub fn touch(&mut self, name: &str, bytes: u64) -> Vec<String> {
        self.tick += 1;
        if let Some((old, _)) = self.entries.insert(name.to_string(), (bytes, self.tick)) {
            self.total = self.total.saturating_sub(old);
        }
        self.total = self.total.saturating_add(bytes);
        let mut evicted = Vec::new();
        while self.total > self.cap {
            let victim = self
                .entries
                .iter()
                .filter(|(n, _)| n.as_str() != name)
                .min_by_key(|(_, (_, t))| *t)
                .map(|(n, _)| n.clone());
            let Some(v) = victim else { break };
            self.remove(&v);
            evicted.push(v);
        }
        evicted
    }

    /// Forget `name` (dropped or evicted by the host).
    pub fn remove(&mut self, name: &str) {
        if let Some((b, _)) = self.entries.remove(name) {
            self.total = self.total.saturating_sub(b);
        }
    }

    /// The byte cap.
    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// Total resident bytes.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// `true` if `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHARD: u64 = 14_000_000;

    #[test]
    fn four_full_shards_fit_fifth_evicts_lru() {
        let mut r = ResidentRegistry::default();
        for n in ["a", "b", "c", "d"] {
            assert!(r.touch(n, SHARD).is_empty());
        }
        r.touch("a", SHARD); // a is now most recent
        assert_eq!(r.touch("e", SHARD), vec!["b".to_string()]);
        assert_eq!(r.total(), 4 * SHARD);
        assert!(!r.contains("b") && r.contains("a"));
    }

    #[test]
    fn growth_of_touched_shard_evicts_others_not_itself() {
        let mut r = ResidentRegistry::new(100);
        r.touch("a", 40);
        r.touch("b", 40);
        assert_eq!(r.touch("b", 90), vec!["a".to_string()]);
        assert_eq!(r.total(), 90);
        // Oversized: everyone else goes, the shard is still tracked.
        r.touch("c", 10);
        assert_eq!(r.touch("x", 150), vec!["b".to_string(), "c".to_string()]);
        assert!(r.contains("x") && r.total() == 150);
        r.remove("x");
        assert_eq!(r.total(), 0);
    }
}
