//! Shard router — maps an input text to a stable worker for cache-friendly
//! repeats. ADR-167 §8.3: hash-shard means the same input always hits the
//! same worker, so a per-worker LRU cache (Phase 2) gets hits.
//!
//! P2C remains the *fallback* path used when no shard-cache hit exists or
//! when the chosen worker is unhealthy. The router is a *hint*, not a
//! constraint — never break correctness for shard locality.

use crate::transport::WorkerEndpoint;

/// Strategy for picking a worker given an input text.
pub trait ShardRouter {
    /// Pick a worker. Returns the chosen endpoint name. The caller is
    /// responsible for falling back to P2C if the named worker is
    /// unhealthy.
    fn pick(&self, text: &str, workers: &[WorkerEndpoint]) -> Option<String>;
}

/// Stable hash shard. Uses FxHash-style mixing of the text bytes so the
/// same text always picks the same worker (within the same fleet shape).
/// No allocation, no external dep.
pub struct HashShardRouter;

impl ShardRouter for HashShardRouter {
    fn pick(&self, text: &str, workers: &[WorkerEndpoint]) -> Option<String> {
        if workers.is_empty() {
            return None;
        }
        let h = fx_hash(text.as_bytes());
        let i = (h as usize) % workers.len();
        Some(workers[i].name.clone())
    }
}

/// Dead-simple non-cryptographic hash. Good enough for sharding (uniform
/// distribution over English-text input) and avoids pulling `rustc-hash`
/// as a dep for one function.
fn fx_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = h.wrapping_mul(0x100_0000_01b3) ^ (b as u64);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workers(n: usize) -> Vec<WorkerEndpoint> {
        (0..n)
            .map(|i| WorkerEndpoint::new(format!("pi-{}", i), format!("10.0.0.{}:50051", i)))
            .collect()
    }

    #[test]
    fn empty_workers_returns_none() {
        let r = HashShardRouter;
        assert!(r.pick("hello", &[]).is_none());
    }

    #[test]
    fn same_text_always_picks_same_worker() {
        let r = HashShardRouter;
        let ws = workers(4);
        let a = r.pick("hello world", &ws).unwrap();
        let b = r.pick("hello world", &ws).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_texts_distribute_across_workers() {
        let r = HashShardRouter;
        let ws = workers(4);
        let mut buckets: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for i in 0..1000 {
            let t = format!("text number {}", i);
            let pick = r.pick(&t, &ws).unwrap();
            *buckets.entry(pick).or_insert(0) += 1;
        }
        // With 1000 texts × 4 workers, each bucket should be ~250 ± 50.
        // Loose bound — just confirms we're not dumping everything on
        // one worker.
        for (name, count) in &buckets {
            assert!(
                (150..=350).contains(count),
                "worker {} got {} (expected ~250)",
                name,
                count
            );
        }
    }
}
