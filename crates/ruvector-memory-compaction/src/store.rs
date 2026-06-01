use serde::{Deserialize, Serialize};

/// A single timestamped vector entry in agent memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: u64,
    pub vector: Vec<f32>,
    /// Unix-epoch milliseconds — used for age scoring.
    pub timestamp_ms: u64,
    /// How many times this memory was retrieved (used in decay scoring).
    pub access_count: u32,
    pub tag: Option<String>,
}

impl MemoryEntry {
    pub fn new(id: u64, vector: Vec<f32>, timestamp_ms: u64) -> Self {
        Self {
            id,
            vector,
            timestamp_ms,
            access_count: 0,
            tag: None,
        }
    }

    pub fn with_access(mut self, count: u32) -> Self {
        self.access_count = count;
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// Parameters that control how much and how aggressively to compact.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Fraction of entries to keep (0.0–1.0); e.g. 0.5 = keep 50 %.
    pub retain_fraction: f32,
    /// Entries older than this are forcibly evicted regardless of score.
    pub max_age_ms: u64,
    /// Cosine-similarity threshold for k-NN graph edges (MinCutCompactor).
    pub similarity_threshold: f32,
    /// Number of nearest neighbors per node in the graph (MinCutCompactor).
    pub top_k_neighbors: usize,
    /// Exponential-decay half-life in ms (DecayScoreCompactor).
    pub half_life_ms: u64,
    /// Weight of the diversity / uniqueness term vs. recency (DecayScoreCompactor).
    pub diversity_weight: f32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            retain_fraction: 0.5,
            max_age_ms: u64::MAX,
            similarity_threshold: 0.7,
            top_k_neighbors: 8,
            half_life_ms: 3_600_000, // 1 hour
            diversity_weight: 0.4,
        }
    }
}

/// Output of a compaction pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    pub retained_ids: Vec<u64>,
    pub evicted_ids: Vec<u64>,
    pub entries_before: usize,
    pub entries_after: usize,
    /// Percentage of entries retained (0–100).
    pub retention_pct: f32,
    /// Wall-clock duration of the compaction pass in microseconds.
    pub duration_us: u64,
}

impl CompactionResult {
    pub fn compaction_ratio(&self) -> f32 {
        if self.entries_before == 0 {
            return 0.0;
        }
        1.0 - (self.entries_after as f32 / self.entries_before as f32)
    }
}

/// In-memory store of timestamped vector entries.
#[derive(Debug, Default)]
pub struct MemoryStore {
    entries: Vec<MemoryEntry>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entry: MemoryEntry) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// Apply compaction in-place: keep only retained IDs.
    pub fn apply_compaction(&mut self, result: &CompactionResult) {
        let keep: std::collections::HashSet<u64> = result.retained_ids.iter().copied().collect();
        self.entries.retain(|e| keep.contains(&e.id));
    }

    /// Compute the centroid of all stored vectors (used for quality checks).
    pub fn centroid(&self) -> Option<Vec<f32>> {
        if self.entries.is_empty() {
            return None;
        }
        let dim = self.entries[0].dim();
        let mut c = vec![0.0f32; dim];
        for e in &self.entries {
            for (a, b) in c.iter_mut().zip(e.vector.iter()) {
                *a += b;
            }
        }
        let n = self.entries.len() as f32;
        for v in &mut c {
            *v /= n;
        }
        Some(c)
    }
}

/// Cosine similarity between two equal-length slices (returns 0.0 on zero norm).
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_sim_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn store_centroid_single() {
        let mut s = MemoryStore::new();
        s.insert(MemoryEntry::new(0, vec![2.0, 4.0], 0));
        let c = s.centroid().unwrap();
        assert_eq!(c, vec![2.0, 4.0]);
    }

    #[test]
    fn apply_compaction_removes_evicted() {
        let mut s = MemoryStore::new();
        for i in 0u64..5 {
            s.insert(MemoryEntry::new(i, vec![i as f32], i * 100));
        }
        let result = CompactionResult {
            retained_ids: vec![0, 1, 2],
            evicted_ids: vec![3, 4],
            entries_before: 5,
            entries_after: 3,
            retention_pct: 60.0,
            duration_us: 0,
        };
        s.apply_compaction(&result);
        assert_eq!(s.len(), 3);
        assert!(s.entries().iter().all(|e| e.id < 3));
    }
}
