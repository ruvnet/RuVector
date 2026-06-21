//! Compaction policies: select the `target_size` most important memory entries.
//!
//! Three strategies are implemented and compared in the nightly benchmark:
//!
//! 1. `LruPolicy`  — keep entries with the highest `last_accessed_at` timestamp.
//! 2. `LfuPolicy`  — keep entries with the highest `access_count`.
//! 3. `CoherencePolicy` — keep entries with the highest weighted importance score:
//!    `I = α·recency + β·frequency + γ·coherence`, where *coherence* is the
//!    maximum cosine similarity between the entry and a recent query context window.

use crate::memory::MemoryEntry;
use crate::scoring::coherence_score;

/// Trait implemented by every compaction strategy.
///
/// Returns the indices (into `entries`) of the surviving memories.
pub trait CompactionPolicy {
    fn name(&self) -> &str;

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        context_window: &[Vec<f32>],
    ) -> Vec<usize>;
}

// ────────────────────────────────────────────────────────────────────────────
// LRU: most recently accessed wins
// ────────────────────────────────────────────────────────────────────────────

/// Keep the `target_size` entries with the most recent access timestamp.
pub struct LruPolicy;

impl CompactionPolicy for LruPolicy {
    fn name(&self) -> &str {
        "LRU"
    }

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        _context: &[Vec<f32>],
    ) -> Vec<usize> {
        let mut indexed: Vec<(usize, u64)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.last_accessed_at))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        indexed
            .into_iter()
            .take(target_size)
            .map(|(i, _)| i)
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LFU: most frequently accessed wins
// ────────────────────────────────────────────────────────────────────────────

/// Keep the `target_size` entries with the highest cumulative access count.
pub struct LfuPolicy;

impl CompactionPolicy for LfuPolicy {
    fn name(&self) -> &str {
        "LFU"
    }

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        _context: &[Vec<f32>],
    ) -> Vec<usize> {
        let mut indexed: Vec<(usize, u64)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.access_count))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        indexed
            .into_iter()
            .take(target_size)
            .map(|(i, _)| i)
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Coherence-Weighted Policy (CoW)
// ────────────────────────────────────────────────────────────────────────────

/// Weights for the three importance components.
#[derive(Debug, Clone)]
pub struct CoherenceWeights {
    /// Weight for normalized recency score (0 = oldest, 1 = newest).
    pub alpha: f32,
    /// Weight for normalized frequency score (0 = least accessed, 1 = most).
    pub beta: f32,
    /// Weight for coherence with active context window.
    pub gamma: f32,
}

impl Default for CoherenceWeights {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            beta: 0.35,
            gamma: 0.40,
        }
    }
}

/// Keep entries that maximize a weighted combination of recency, frequency,
/// and semantic coherence with the active query context window.
///
/// This is the novel variant introduced by this nightly research run.
pub struct CoherencePolicy {
    pub weights: CoherenceWeights,
}

impl CoherencePolicy {
    pub fn new(weights: CoherenceWeights) -> Self {
        Self { weights }
    }
}

impl Default for CoherencePolicy {
    fn default() -> Self {
        Self {
            weights: CoherenceWeights::default(),
        }
    }
}

impl CompactionPolicy for CoherencePolicy {
    fn name(&self) -> &str {
        "CoherenceWeighted"
    }

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        context: &[Vec<f32>],
    ) -> Vec<usize> {
        if entries.is_empty() {
            return Vec::new();
        }

        // Normalisation anchors
        let max_time = entries
            .iter()
            .map(|e| e.last_accessed_at)
            .max()
            .unwrap_or(1);
        let min_time = entries
            .iter()
            .map(|e| e.last_accessed_at)
            .min()
            .unwrap_or(0);
        let time_range = (max_time - min_time).max(1) as f32;

        let max_count = entries.iter().map(|e| e.access_count).max().unwrap_or(1);
        let max_count_f = max_count.max(1) as f32;

        let w = &self.weights;

        let mut scored: Vec<(usize, f32)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let recency = (e.last_accessed_at - min_time) as f32 / time_range;
                let frequency = e.access_count as f32 / max_count_f;
                let coherence = if context.is_empty() {
                    0.0
                } else {
                    coherence_score(&e.vector, context)
                };
                let importance = w.alpha * recency + w.beta * frequency + w.gamma * coherence;
                (i, importance)
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
            .into_iter()
            .take(target_size)
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryEntry;

    fn make_entries(n: usize, dims: usize) -> Vec<MemoryEntry> {
        (0..n)
            .map(|i| {
                let mut e = MemoryEntry::new(i as u64, vec![0.0; dims], i as u64);
                e.access_count = i as u64;
                e.last_accessed_at = i as u64;
                e
            })
            .collect()
    }

    #[test]
    fn lru_keeps_most_recent() {
        let entries = make_entries(10, 2);
        let survivors = LruPolicy.select_survivors(&entries, 3, &[]);
        // Indices should be 9, 8, 7 (highest last_accessed_at)
        let ids: Vec<u64> = survivors.iter().map(|&i| entries[i].id).collect();
        assert!(ids.contains(&9));
        assert!(ids.contains(&8));
        assert!(ids.contains(&7));
    }

    #[test]
    fn lfu_keeps_most_frequent() {
        let entries = make_entries(10, 2);
        let survivors = LfuPolicy.select_survivors(&entries, 3, &[]);
        let ids: Vec<u64> = survivors.iter().map(|&i| entries[i].id).collect();
        assert!(ids.contains(&9));
        assert!(ids.contains(&8));
        assert!(ids.contains(&7));
    }

    #[test]
    fn coherence_policy_prefers_contextually_relevant() {
        // Two entries: one aligned with context, one orthogonal.
        let mut e0 = MemoryEntry::new(0, vec![1.0, 0.0], 1);
        e0.access_count = 1;
        let mut e1 = MemoryEntry::new(1, vec![0.0, 1.0], 2);
        e1.access_count = 2; // higher frequency

        let entries = vec![e0, e1];
        let context = vec![vec![1.0, 0.0]]; // context aligns with e0

        // With gamma=1.0, coherence dominates: e0 should win despite lower frequency
        let policy = CoherencePolicy::new(CoherenceWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
        });
        let survivors = policy.select_survivors(&entries, 1, &context);
        assert_eq!(survivors[0], 0, "coherence-aligned entry should be kept");
    }
}
