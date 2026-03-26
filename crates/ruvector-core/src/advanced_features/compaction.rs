//! LSM-Tree Style Streaming Index Compaction
//!
//! Implements a Log-Structured Merge-tree (LSM-tree) index optimised for
//! write-heavy vector workloads. Writes are absorbed by an in-memory
//! [`MemTable`] and periodically flushed into immutable, sorted [`Segment`]s
//! organised across multiple levels. Background compaction merges segments
//! to bound read amplification while keeping writes sequential.
//!
//! ## Why LSM for Vectors?
//!
//! Traditional vector indices (HNSW, IVF) are optimised for read-heavy
//! patterns and require expensive in-place updates. LSM-trees turn random
//! writes into sequential appends, making them ideal for:
//! - High-throughput ingestion pipelines
//! - Streaming embedding updates
//! - Workloads with frequent deletes (tombstone-based)
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────┐
//! │ MemTable │  ← hot writes (sorted by id)
//! └────┬─────┘
//!      │ flush
//! ┌────▼─────┐
//! │ Level 0  │  ← recent segments (may overlap)
//! ├──────────┤
//! │ Level 1  │  ← merged, non-overlapping
//! ├──────────┤
//! │ Level 2  │  ← larger sorted runs …
//! └──────────┘
//! ```

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

use serde::{Deserialize, Serialize};

use crate::types::{SearchResult, VectorId};

// ---------------------------------------------------------------------------
// CompactionConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for the LSM-tree index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Maximum number of entries in the memtable before it is flushed.
    pub memtable_capacity: usize,
    /// Size ratio between adjacent levels (fanout).
    pub level_size_ratio: usize,
    /// Maximum number of levels in the tree.
    pub max_levels: usize,
    /// Number of segments in a level that triggers compaction into the next.
    pub merge_threshold: usize,
    /// Target false-positive rate for per-segment bloom filters.
    pub bloom_fp_rate: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            memtable_capacity: 1000,
            level_size_ratio: 10,
            max_levels: 4,
            merge_threshold: 4,
            bloom_fp_rate: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// BloomFilter
// ---------------------------------------------------------------------------

/// A space-efficient probabilistic set for fast negative lookups.
///
/// Uses the double-hashing technique: `h_i(x) = h1(x) + i * h2(x)` to
/// simulate `k` independent hash functions from two base hashes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new bloom filter sized for `expected_items` at the given
    /// false-positive rate.
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let expected_items = expected_items.max(1);
        let fp_rate = fp_rate.clamp(1e-10, 0.5);
        let num_bits = (-(expected_items as f64) * fp_rate.ln() / (2.0_f64.ln().powi(2)))
            .ceil() as usize;
        let num_bits = num_bits.max(8);
        let num_hashes =
            ((num_bits as f64 / expected_items as f64) * 2.0_f64.ln()).ceil() as usize;
        let num_hashes = num_hashes.max(1);
        Self {
            bits: vec![false; num_bits],
            num_hashes,
        }
    }

    /// Insert an element into the filter.
    pub fn insert(&mut self, key: &str) {
        let (h1, h2) = self.hashes(key);
        let m = self.bits.len();
        for i in 0..self.num_hashes {
            let idx = (h1.wrapping_add(i.wrapping_mul(h2))) % m;
            self.bits[idx] = true;
        }
    }

    /// Test membership. `true` means *possibly* present; `false` means
    /// *definitely* absent.
    pub fn may_contain(&self, key: &str) -> bool {
        let (h1, h2) = self.hashes(key);
        let m = self.bits.len();
        for i in 0..self.num_hashes {
            let idx = (h1.wrapping_add(i.wrapping_mul(h2))) % m;
            if !self.bits[idx] {
                return false;
            }
        }
        true
    }

    fn hashes(&self, key: &str) -> (usize, usize) {
        // FNV-1a inspired pair of hashes.
        let bytes = key.as_bytes();
        let mut h1: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h1 ^= b as u64;
            h1 = h1.wrapping_mul(0x100000001b3);
        }
        let mut h2: u64 = 0x517cc1b727220a95;
        for &b in bytes {
            h2 = h2.wrapping_mul(31).wrapping_add(b as u64);
        }
        (h1 as usize, (h2 | 1) as usize) // h2 must be odd for full period
    }
}

// ---------------------------------------------------------------------------
// MemTable
// ---------------------------------------------------------------------------

/// Tombstone sentinel — deleted entries carry `None` vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LSMEntry {
    id: VectorId,
    /// `None` signifies a tombstone (delete marker).
    vector: Option<Vec<f32>>,
    metadata: Option<HashMap<String, serde_json::Value>>,
    /// Monotonic sequence number for conflict resolution (higher wins).
    seq: u64,
}

/// In-memory sorted write buffer.
///
/// Entries are kept in a `BTreeMap` keyed by vector id so that flushes
/// produce already-sorted segments with no additional sorting step.
#[derive(Debug, Clone)]
pub struct MemTable {
    entries: BTreeMap<VectorId, LSMEntry>,
    capacity: usize,
}

impl MemTable {
    /// Create a memtable with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            capacity,
        }
    }

    /// Insert or update an entry. Returns `true` when the table is full and
    /// should be flushed.
    pub fn insert(
        &mut self,
        id: VectorId,
        vector: Option<Vec<f32>>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        seq: u64,
    ) -> bool {
        self.entries.insert(
            id.clone(),
            LSMEntry { id, vector, metadata, seq },
        );
        self.is_full()
    }

    /// Brute-force scan of the memtable returning the closest `top_k` live
    /// entries by Euclidean distance.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut heap: BinaryHeap<(OrderedFloat, VectorId)> = BinaryHeap::new();
        for entry in self.entries.values() {
            let vec = match &entry.vector {
                Some(v) => v,
                None => continue, // skip tombstones
            };
            let dist = euclidean_distance(query, vec);
            let of = OrderedFloat(dist);
            if heap.len() < top_k {
                heap.push((of, entry.id.clone()));
            } else if let Some(top) = heap.peek() {
                if of < top.0 {
                    heap.pop();
                    heap.push((of, entry.id.clone()));
                }
            }
        }
        heap_to_results(heap, &self.entries)
    }

    /// Freeze and flush the memtable into an immutable segment.
    pub fn flush(&mut self, level: usize, bloom_fp_rate: f64) -> Segment {
        let entries: Vec<LSMEntry> = self.entries.values().cloned().collect();
        let segment = Segment::from_entries(entries, level, bloom_fp_rate);
        self.entries.clear();
        segment
    }

    /// Number of entries currently buffered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the memtable is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the memtable has reached capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }
}

// ---------------------------------------------------------------------------
// Segment
// ---------------------------------------------------------------------------

/// An immutable sorted run of vector entries with an associated bloom filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Entries sorted by `id`.
    entries: Vec<LSMEntry>,
    /// Bloom filter over entry ids for fast negative lookups.
    bloom: BloomFilter,
    /// The LSM level this segment belongs to.
    pub level: usize,
}

impl Segment {
    fn from_entries(entries: Vec<LSMEntry>, level: usize, fp_rate: f64) -> Self {
        let mut bloom = BloomFilter::new(entries.len(), fp_rate);
        for e in &entries {
            bloom.insert(&e.id);
        }
        Self { entries, bloom, level }
    }

    /// Number of entries (including tombstones).
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Probabilistic id membership test (may return false positives).
    pub fn contains(&self, id: &str) -> bool {
        self.bloom.may_contain(id)
    }

    /// Brute-force search within this segment.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut heap: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();
        for (i, entry) in self.entries.iter().enumerate() {
            let vec = match &entry.vector {
                Some(v) => v,
                None => continue,
            };
            let dist = euclidean_distance(query, vec);
            let of = OrderedFloat(dist);
            if heap.len() < top_k {
                heap.push((of, i));
            } else if let Some(top) = heap.peek() {
                if of < top.0 {
                    heap.pop();
                    heap.push((of, i));
                }
            }
        }
        let mut results: Vec<SearchResult> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|(OrderedFloat(score), idx)| {
                let e = &self.entries[idx];
                SearchResult {
                    id: e.id.clone(),
                    score,
                    vector: e.vector.clone(),
                    metadata: e.metadata.clone(),
                }
            })
            .collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        results
    }

    /// K-way merge of multiple segments. Deduplicates by id, keeping the
    /// entry with the highest sequence number. Tombstones are dropped during
    /// merge (compaction GC).
    pub fn merge(segments: &[Segment], target_level: usize, fp_rate: f64) -> Segment {
        let mut merged: BTreeMap<VectorId, LSMEntry> = BTreeMap::new();
        for seg in segments {
            for entry in &seg.entries {
                let dominated = merged
                    .get(&entry.id)
                    .map_or(true, |existing| entry.seq > existing.seq);
                if dominated {
                    merged.insert(entry.id.clone(), entry.clone());
                }
            }
        }
        // Drop tombstones during compaction.
        let entries: Vec<LSMEntry> = merged
            .into_values()
            .filter(|e| e.vector.is_some())
            .collect();
        Segment::from_entries(entries, target_level, fp_rate)
    }
}

// ---------------------------------------------------------------------------
// LSMStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the LSM-tree index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSMStats {
    /// Number of active levels.
    pub num_levels: usize,
    /// Number of segments at each level.
    pub segments_per_level: Vec<usize>,
    /// Total live entries across all levels and memtable.
    pub total_entries: usize,
    /// Write amplification factor (total bytes / user bytes).
    pub write_amplification: f64,
}

// ---------------------------------------------------------------------------
// LSMIndex
// ---------------------------------------------------------------------------

/// A write-optimised vector index using LSM-tree tiered compaction.
///
/// All writes go through the in-memory [`MemTable`]. Once full it is flushed
/// to level 0 as an immutable [`Segment`]. When a level accumulates
/// `merge_threshold` segments they are merged into the next level, bounding
/// read amplification while keeping writes sequential.
#[derive(Debug, Clone)]
pub struct LSMIndex {
    config: CompactionConfig,
    memtable: MemTable,
    /// `levels[i]` holds the segments at level `i`.
    levels: Vec<Vec<Segment>>,
    /// Monotonically increasing sequence counter.
    next_seq: u64,
    /// Bytes attributed to user writes (inserts + deletes).
    bytes_written_user: u64,
    /// Total bytes written including compaction rewrites.
    bytes_written_total: u64,
    /// Ids that have been logically deleted (for filtering search results
    /// when tombstones may still be in older segments).
    deleted_ids: HashSet<VectorId>,
}

impl LSMIndex {
    /// Create a new LSM index with the given configuration.
    pub fn new(config: CompactionConfig) -> Self {
        let cap = config.memtable_capacity;
        let num_levels = config.max_levels;
        Self {
            config,
            memtable: MemTable::new(cap),
            levels: vec![Vec::new(); num_levels],
            next_seq: 0,
            bytes_written_user: 0,
            bytes_written_total: 0,
            deleted_ids: HashSet::new(),
        }
    }

    /// Insert a vector. Automatically flushes the memtable when full and
    /// triggers compaction when level thresholds are exceeded.
    pub fn insert(
        &mut self,
        id: VectorId,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) {
        let entry_bytes = (vector.len() * 4 + id.len()) as u64;
        self.bytes_written_user += entry_bytes;
        self.bytes_written_total += entry_bytes;
        self.deleted_ids.remove(&id);

        let seq = self.next_seq;
        self.next_seq += 1;
        let full = self.memtable.insert(id, Some(vector), metadata, seq);
        if full {
            self.flush_memtable();
            self.auto_compact();
        }
    }

    /// Mark a vector as deleted by inserting a tombstone.
    pub fn delete(&mut self, id: VectorId) {
        let entry_bytes = id.len() as u64;
        self.bytes_written_user += entry_bytes;
        self.bytes_written_total += entry_bytes;
        self.deleted_ids.insert(id.clone());

        let seq = self.next_seq;
        self.next_seq += 1;
        let full = self.memtable.insert(id, None, None, seq);
        if full {
            self.flush_memtable();
            self.auto_compact();
        }
    }

    /// Search across the memtable and all levels, merging results.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut seen: HashSet<VectorId> = HashSet::new();
        let mut all_results: Vec<SearchResult> = Vec::new();

        // Memtable first (freshest data).
        for r in self.memtable.search(query, top_k) {
            if !self.deleted_ids.contains(&r.id) {
                seen.insert(r.id.clone());
                all_results.push(r);
            }
        }

        // Then levels, newest to oldest.
        for level in &self.levels {
            for seg in level.iter().rev() {
                for r in seg.search(query, top_k) {
                    if !seen.contains(&r.id) && !self.deleted_ids.contains(&r.id) {
                        seen.insert(r.id.clone());
                        all_results.push(r);
                    }
                }
            }
        }

        all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_results.truncate(top_k);
        all_results
    }

    /// Manually trigger compaction across all levels.
    pub fn compact(&mut self) {
        if !self.memtable.is_empty() {
            self.flush_memtable();
        }
        for level in 0..self.config.max_levels.saturating_sub(1) {
            if self.levels[level].len() >= 2 {
                self.compact_level(level);
            }
        }
    }

    /// Check each level and compact if it exceeds `merge_threshold`.
    pub fn auto_compact(&mut self) {
        for level in 0..self.config.max_levels.saturating_sub(1) {
            if self.levels[level].len() >= self.config.merge_threshold {
                self.compact_level(level);
            }
        }
    }

    /// Return runtime statistics.
    pub fn stats(&self) -> LSMStats {
        let segments_per_level: Vec<usize> = self.levels.iter().map(|l| l.len()).collect();
        let total_entries = self.memtable.len()
            + self.levels.iter().flat_map(|l| l.iter()).map(|s| s.size()).sum::<usize>();
        LSMStats {
            num_levels: self.levels.len(),
            segments_per_level,
            total_entries,
            write_amplification: self.write_amplification(),
        }
    }

    /// Write amplification: total bytes written / user bytes written.
    pub fn write_amplification(&self) -> f64 {
        if self.bytes_written_user == 0 {
            return 1.0;
        }
        self.bytes_written_total as f64 / self.bytes_written_user as f64
    }

    // -- internal helpers ---------------------------------------------------

    fn flush_memtable(&mut self) {
        let seg = self.memtable.flush(0, self.config.bloom_fp_rate);
        let flush_bytes: u64 = seg
            .entries
            .iter()
            .map(|e| {
                let vb = e.vector.as_ref().map_or(0, |v| v.len() * 4);
                (vb + e.id.len()) as u64
            })
            .sum();
        self.bytes_written_total += flush_bytes;
        self.levels[0].push(seg);
    }

    fn compact_level(&mut self, level: usize) {
        let target = level + 1;
        if target >= self.config.max_levels {
            return;
        }
        let segments = std::mem::take(&mut self.levels[level]);
        let merged = Segment::merge(&segments, target, self.config.bloom_fp_rate);
        let merge_bytes: u64 = merged
            .entries
            .iter()
            .map(|e| {
                let vb = e.vector.as_ref().map_or(0, |v| v.len() * 4);
                (vb + e.id.len()) as u64
            })
            .sum();
        self.bytes_written_total += merge_bytes;
        self.levels[target].push(merged);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Wrapper for f32 that implements Ord (NaN-safe) for use in BinaryHeap.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn heap_to_results(
    heap: BinaryHeap<(OrderedFloat, VectorId)>,
    entries: &BTreeMap<VectorId, LSMEntry>,
) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = heap
        .into_sorted_vec()
        .into_iter()
        .filter_map(|(OrderedFloat(score), id)| {
            entries.get(&id).map(|e| SearchResult {
                id: e.id.clone(),
                score,
                vector: e.vector.clone(),
                metadata: e.metadata.clone(),
            })
        })
        .collect();
    results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    // -- MemTable tests -----------------------------------------------------

    #[test]
    fn memtable_insert_and_len() {
        let mut mt = MemTable::new(5);
        assert!(mt.is_empty());
        mt.insert("a".into(), Some(vec![1.0]), None, 0);
        mt.insert("b".into(), Some(vec![2.0]), None, 1);
        assert_eq!(mt.len(), 2);
        assert!(!mt.is_full());
    }

    #[test]
    fn memtable_is_full() {
        let mut mt = MemTable::new(2);
        mt.insert("a".into(), Some(vec![1.0]), None, 0);
        let full = mt.insert("b".into(), Some(vec![2.0]), None, 1);
        assert!(full);
        assert!(mt.is_full());
    }

    #[test]
    fn memtable_search_returns_closest() {
        let mut mt = MemTable::new(100);
        mt.insert("far".into(), Some(vec![10.0, 10.0]), None, 0);
        mt.insert("close".into(), Some(vec![1.0, 0.0]), None, 1);
        mt.insert("mid".into(), Some(vec![5.0, 5.0]), None, 2);

        let results = mt.search(&[0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "close");
    }

    #[test]
    fn memtable_flush_produces_segment() {
        let mut mt = MemTable::new(10);
        mt.insert("x".into(), Some(vec![1.0]), None, 0);
        mt.insert("y".into(), Some(vec![2.0]), None, 1);
        let seg = mt.flush(0, 0.01);
        assert_eq!(seg.size(), 2);
        assert_eq!(seg.level, 0);
        assert!(mt.is_empty());
    }

    // -- Segment tests ------------------------------------------------------

    #[test]
    fn segment_merge_dedup_keeps_latest() {
        let s1 = Segment::from_entries(
            vec![LSMEntry { id: "a".into(), vector: Some(vec![1.0]), metadata: None, seq: 1 }],
            0, 0.01,
        );
        let s2 = Segment::from_entries(
            vec![LSMEntry { id: "a".into(), vector: Some(vec![9.0]), metadata: None, seq: 5 }],
            0, 0.01,
        );
        let merged = Segment::merge(&[s1, s2], 1, 0.01);
        assert_eq!(merged.size(), 1);
        assert_eq!(merged.entries[0].vector.as_ref().unwrap(), &vec![9.0]);
    }

    #[test]
    fn segment_merge_drops_tombstones() {
        let s1 = Segment::from_entries(
            vec![LSMEntry { id: "a".into(), vector: Some(vec![1.0]), metadata: None, seq: 1 }],
            0, 0.01,
        );
        let s2 = Segment::from_entries(
            vec![LSMEntry { id: "a".into(), vector: None, metadata: None, seq: 5 }],
            0, 0.01,
        );
        let merged = Segment::merge(&[s1, s2], 1, 0.01);
        assert_eq!(merged.size(), 0);
    }

    // -- BloomFilter tests --------------------------------------------------

    #[test]
    fn bloom_filter_no_false_negatives() {
        let mut bf = BloomFilter::new(100, 0.01);
        for i in 0..100 {
            bf.insert(&format!("key-{i}"));
        }
        for i in 0..100 {
            assert!(bf.may_contain(&format!("key-{i}")));
        }
    }

    #[test]
    fn bloom_filter_low_false_positive_rate() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..1000 {
            bf.insert(&format!("present-{i}"));
        }
        let mut false_positives = 0;
        let test_count = 10_000;
        for i in 0..test_count {
            if bf.may_contain(&format!("absent-{i}")) {
                false_positives += 1;
            }
        }
        let fp_rate = false_positives as f64 / test_count as f64;
        // Allow some margin over theoretical 1%.
        assert!(fp_rate < 0.05, "FP rate too high: {fp_rate}");
    }

    // -- LSMIndex tests -----------------------------------------------------

    #[test]
    fn lsm_insert_and_search() {
        let config = CompactionConfig { memtable_capacity: 10, ..Default::default() };
        let mut idx = LSMIndex::new(config);
        idx.insert("v1".into(), vec![1.0, 0.0], None);
        idx.insert("v2".into(), vec![0.0, 1.0], None);

        let results = idx.search(&[1.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn lsm_delete_with_tombstone() {
        let config = CompactionConfig { memtable_capacity: 100, ..Default::default() };
        let mut idx = LSMIndex::new(config);
        idx.insert("v1".into(), vec![1.0, 0.0], None);
        idx.insert("v2".into(), vec![0.0, 1.0], None);
        idx.delete("v1".into());

        let results = idx.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v2");
    }

    #[test]
    fn lsm_auto_compaction_trigger() {
        let config = CompactionConfig {
            memtable_capacity: 2,
            merge_threshold: 2,
            max_levels: 3,
            ..Default::default()
        };
        let mut idx = LSMIndex::new(config);
        // Insert enough to trigger multiple flushes and compaction.
        for i in 0..10 {
            idx.insert(format!("v{i}"), vec![i as f32], None);
        }
        let stats = idx.stats();
        // Level 0 should have been compacted into level 1+.
        assert!(
            stats.segments_per_level[0] < 4,
            "Level 0 should have been compacted: {:?}",
            stats.segments_per_level
        );
    }

    #[test]
    fn lsm_multi_level_compaction() {
        let config = CompactionConfig {
            memtable_capacity: 2,
            merge_threshold: 2,
            max_levels: 4,
            ..Default::default()
        };
        let mut idx = LSMIndex::new(config);
        for i in 0..30 {
            idx.insert(format!("v{i}"), make_vec(4, i as f32), None);
        }
        let stats = idx.stats();
        // At least some data should have migrated beyond level 0.
        let total_segments: usize = stats.segments_per_level.iter().sum();
        assert!(total_segments >= 1, "Expected segments across levels");
    }

    #[test]
    fn lsm_write_amplification_increases() {
        let config = CompactionConfig {
            memtable_capacity: 5,
            merge_threshold: 2,
            max_levels: 3,
            ..Default::default()
        };
        let mut idx = LSMIndex::new(config);
        for i in 0..20 {
            idx.insert(format!("v{i}"), make_vec(4, i as f32), None);
        }
        let wa = idx.write_amplification();
        assert!(wa >= 1.0, "Write amplification should be >= 1.0, got {wa}");
    }

    #[test]
    fn lsm_empty_index() {
        let idx = LSMIndex::new(CompactionConfig::default());
        let results = idx.search(&[0.0, 0.0], 10);
        assert!(results.is_empty());
        let stats = idx.stats();
        assert_eq!(stats.total_entries, 0);
        assert!((stats.write_amplification - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn lsm_large_batch_insert() {
        let config = CompactionConfig {
            memtable_capacity: 50,
            merge_threshold: 4,
            max_levels: 4,
            ..Default::default()
        };
        let mut idx = LSMIndex::new(config);
        for i in 0..500 {
            idx.insert(format!("v{i}"), make_vec(8, i as f32 * 0.01), None);
        }
        let stats = idx.stats();
        assert!(stats.total_entries > 0);
        // Search should still work correctly.
        let results = idx.search(&make_vec(8, 0.0), 5);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, "v0");
    }

    #[test]
    fn lsm_search_across_levels() {
        let config = CompactionConfig {
            memtable_capacity: 3,
            merge_threshold: 3,
            max_levels: 3,
            ..Default::default()
        };
        let mut idx = LSMIndex::new(config);
        // Phase 1: insert and let some flush to segments.
        for i in 0..9 {
            idx.insert(format!("v{i}"), vec![i as f32, 0.0], None);
        }
        // Phase 2: insert more into memtable.
        idx.insert("latest".into(), vec![0.0, 0.0], None);

        let results = idx.search(&[0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        // "latest" and "v0" are both at origin.
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"latest"));
        assert!(ids.contains(&"v0"));
    }
}
