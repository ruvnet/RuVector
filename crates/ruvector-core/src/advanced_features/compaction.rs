//! LSM-Tree Style Streaming Index Compaction
//!
//! Implements a Log-Structured Merge-tree (LSM-tree) index for write-heavy
//! vector workloads. Writes are absorbed by an in-memory [`MemTable`] and
//! flushed into immutable, sorted [`Segment`]s across tiered levels.
//! Compaction merges segments to bound read amplification.
//!
//! LSM-trees turn random writes into sequential appends, ideal for
//! high-throughput ingestion, streaming embedding updates, and frequent
//! deletes (tombstone-based).

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::types::{SearchResult, VectorId};

/// Configuration for the LSM-tree index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Max entries in memtable before flush.
    pub memtable_capacity: usize,
    /// Size ratio between adjacent levels.
    pub level_size_ratio: usize,
    /// Maximum number of levels.
    pub max_levels: usize,
    /// Segments per level that triggers compaction.
    pub merge_threshold: usize,
    /// False-positive rate for bloom filters.
    pub bloom_fp_rate: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self { memtable_capacity: 1000, level_size_ratio: 10, max_levels: 4,
               merge_threshold: 4, bloom_fp_rate: 0.01 }
    }
}

/// Probabilistic set using double-hashing: `h_i(x) = h1(x) + i * h2(x)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter { bits: Vec<bool>, num_hashes: usize }

impl BloomFilter {
    /// Create a bloom filter for `n` items at `fp_rate`.
    pub fn new(n: usize, fp_rate: f64) -> Self {
        let n = n.max(1);
        let fp = fp_rate.clamp(1e-10, 0.5);
        let m = (-(n as f64) * fp.ln() / 2.0_f64.ln().powi(2)).ceil() as usize;
        let m = m.max(8);
        let k = ((m as f64 / n as f64) * 2.0_f64.ln()).ceil().max(1.0) as usize;
        Self { bits: vec![false; m], num_hashes: k }
    }

    /// Insert an element.
    pub fn insert(&mut self, key: &str) {
        let (h1, h2) = Self::hashes(key);
        let m = self.bits.len();
        for i in 0..self.num_hashes { self.bits[h1.wrapping_add(i.wrapping_mul(h2)) % m] = true; }
    }

    /// Test membership (may return false positives).
    pub fn may_contain(&self, key: &str) -> bool {
        let (h1, h2) = Self::hashes(key);
        let m = self.bits.len();
        (0..self.num_hashes).all(|i| self.bits[h1.wrapping_add(i.wrapping_mul(h2)) % m])
    }

    fn hashes(key: &str) -> (usize, usize) {
        let (mut h1, mut h2): (u64, u64) = (0xcbf29ce484222325, 0x517cc1b727220a95);
        for &b in key.as_bytes() {
            h1 ^= b as u64; h1 = h1.wrapping_mul(0x100000001b3);
            h2 = h2.wrapping_mul(31).wrapping_add(b as u64);
        }
        (h1 as usize, (h2 | 1) as usize)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LSMEntry {
    id: VectorId,
    vector: Option<Vec<f32>>, // None = tombstone
    metadata: Option<HashMap<String, serde_json::Value>>,
    seq: u64, // higher wins on conflict
}

/// In-memory sorted write buffer backed by `BTreeMap`.
#[derive(Debug, Clone)]
pub struct MemTable { entries: BTreeMap<VectorId, LSMEntry>, capacity: usize }

impl MemTable {
    pub fn new(capacity: usize) -> Self { Self { entries: BTreeMap::new(), capacity } }

    /// Insert/update. Returns `true` when full.
    pub fn insert(&mut self, id: VectorId, vector: Option<Vec<f32>>,
                  metadata: Option<HashMap<String, serde_json::Value>>, seq: u64) -> bool {
        self.entries.insert(id.clone(), LSMEntry { id, vector, metadata, seq });
        self.is_full()
    }

    /// Brute-force nearest-neighbour scan (Euclidean).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut heap: BinaryHeap<(OrdF32, VectorId)> = BinaryHeap::new();
        for e in self.entries.values() {
            let v = match &e.vector { Some(v) => v, None => continue };
            let d = OrdF32(euclid(query, v));
            if heap.len() < top_k { heap.push((d, e.id.clone())); }
            else if d < heap.peek().unwrap().0 { heap.pop(); heap.push((d, e.id.clone())); }
        }
        let mut r: Vec<SearchResult> = heap.into_sorted_vec().into_iter().filter_map(|(OrdF32(s), id)| {
            self.entries.get(&id).map(|e| SearchResult { id: e.id.clone(), score: s,
                vector: e.vector.clone(), metadata: e.metadata.clone() })
        }).collect();
        r.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap()); r
    }

    /// Flush to an immutable segment, clearing the memtable.
    pub fn flush(&mut self, level: usize, fp_rate: f64) -> Segment {
        let entries: Vec<LSMEntry> = self.entries.values().cloned().collect();
        self.entries.clear();
        Segment::from_entries(entries, level, fp_rate)
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn is_full(&self) -> bool { self.entries.len() >= self.capacity }
}

/// Immutable sorted run with bloom filter for point lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment { entries: Vec<LSMEntry>, bloom: BloomFilter, pub level: usize }

impl Segment {
    fn from_entries(entries: Vec<LSMEntry>, level: usize, fp_rate: f64) -> Self {
        let mut bloom = BloomFilter::new(entries.len(), fp_rate);
        for e in &entries { bloom.insert(&e.id); }
        Self { entries, bloom, level }
    }

    pub fn size(&self) -> usize { self.entries.len() }
    pub fn contains(&self, id: &str) -> bool { self.bloom.may_contain(id) }

    /// Brute-force search within this segment.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut heap: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();
        for (i, e) in self.entries.iter().enumerate() {
            let v = match &e.vector { Some(v) => v, None => continue };
            let d = OrdF32(euclid(query, v));
            if heap.len() < top_k { heap.push((d, i)); }
            else if d < heap.peek().unwrap().0 { heap.pop(); heap.push((d, i)); }
        }
        let mut r: Vec<SearchResult> = heap.into_sorted_vec().into_iter().map(|(OrdF32(s), i)| {
            let e = &self.entries[i];
            SearchResult { id: e.id.clone(), score: s, vector: e.vector.clone(), metadata: e.metadata.clone() }
        }).collect();
        r.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap()); r
    }

    /// K-way merge deduplicating by id (highest seq wins). Drops tombstones.
    pub fn merge(segments: &[Segment], target_level: usize, fp_rate: f64) -> Segment {
        let mut merged: BTreeMap<VectorId, LSMEntry> = BTreeMap::new();
        for seg in segments {
            for e in &seg.entries {
                if merged.get(&e.id).map_or(true, |x| e.seq > x.seq) {
                    merged.insert(e.id.clone(), e.clone());
                }
            }
        }
        let entries: Vec<LSMEntry> = merged.into_values().filter(|e| e.vector.is_some()).collect();
        Segment::from_entries(entries, target_level, fp_rate)
    }
}

/// Runtime statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSMStats {
    pub num_levels: usize,
    pub segments_per_level: Vec<usize>,
    pub total_entries: usize,
    pub write_amplification: f64,
}

/// Write-optimised vector index using LSM-tree tiered compaction.
///
/// Writes go to the [`MemTable`]; when full it flushes to level 0. Levels
/// exceeding `merge_threshold` segments are compacted into the next level.
#[derive(Debug, Clone)]
pub struct LSMIndex {
    config: CompactionConfig,
    memtable: MemTable,
    levels: Vec<Vec<Segment>>,
    next_seq: u64,
    bytes_written_user: u64,
    bytes_written_total: u64,
    deleted_ids: HashSet<VectorId>,
}

impl LSMIndex {
    pub fn new(config: CompactionConfig) -> Self {
        let cap = config.memtable_capacity;
        let nl = config.max_levels;
        Self { config, memtable: MemTable::new(cap), levels: vec![Vec::new(); nl],
               next_seq: 0, bytes_written_user: 0, bytes_written_total: 0,
               deleted_ids: HashSet::new() }
    }

    /// Insert a vector. Auto-flushes and compacts as needed.
    pub fn insert(&mut self, id: VectorId, vector: Vec<f32>,
                  metadata: Option<HashMap<String, serde_json::Value>>) {
        let bytes = (vector.len() * 4 + id.len()) as u64;
        self.bytes_written_user += bytes;
        self.bytes_written_total += bytes;
        self.deleted_ids.remove(&id);
        let seq = self.next_seq; self.next_seq += 1;
        if self.memtable.insert(id, Some(vector), metadata, seq) {
            self.flush_memtable(); self.auto_compact();
        }
    }

    /// Mark a vector as deleted (tombstone).
    pub fn delete(&mut self, id: VectorId) {
        let bytes = id.len() as u64;
        self.bytes_written_user += bytes;
        self.bytes_written_total += bytes;
        self.deleted_ids.insert(id.clone());
        let seq = self.next_seq; self.next_seq += 1;
        if self.memtable.insert(id, None, None, seq) {
            self.flush_memtable(); self.auto_compact();
        }
    }

    /// Search across memtable and all levels, merging results.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut seen = HashSet::new();
        let mut all = Vec::new();
        for r in self.memtable.search(query, top_k) {
            if !self.deleted_ids.contains(&r.id) { seen.insert(r.id.clone()); all.push(r); }
        }
        for level in &self.levels {
            for seg in level.iter().rev() {
                for r in seg.search(query, top_k) {
                    if !seen.contains(&r.id) && !self.deleted_ids.contains(&r.id) {
                        seen.insert(r.id.clone()); all.push(r);
                    }
                }
            }
        }
        all.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all.truncate(top_k); all
    }

    /// Manual compaction across all levels.
    pub fn compact(&mut self) {
        if !self.memtable.is_empty() { self.flush_memtable(); }
        for l in 0..self.config.max_levels.saturating_sub(1) {
            if self.levels[l].len() >= 2 { self.compact_level(l); }
        }
    }

    /// Auto-compact levels exceeding `merge_threshold`.
    pub fn auto_compact(&mut self) {
        for l in 0..self.config.max_levels.saturating_sub(1) {
            if self.levels[l].len() >= self.config.merge_threshold { self.compact_level(l); }
        }
    }

    pub fn stats(&self) -> LSMStats {
        let spl: Vec<usize> = self.levels.iter().map(|l| l.len()).collect();
        let total = self.memtable.len()
            + self.levels.iter().flat_map(|l| l.iter()).map(|s| s.size()).sum::<usize>();
        LSMStats { num_levels: self.levels.len(), segments_per_level: spl,
                   total_entries: total, write_amplification: self.write_amplification() }
    }

    pub fn write_amplification(&self) -> f64 {
        if self.bytes_written_user == 0 { 1.0 }
        else { self.bytes_written_total as f64 / self.bytes_written_user as f64 }
    }

    fn flush_memtable(&mut self) {
        let seg = self.memtable.flush(0, self.config.bloom_fp_rate);
        self.bytes_written_total += entry_bytes(&seg.entries);
        self.levels[0].push(seg);
    }

    fn compact_level(&mut self, level: usize) {
        let target = level + 1;
        if target >= self.config.max_levels { return; }
        let segments = std::mem::take(&mut self.levels[level]);
        let merged = Segment::merge(&segments, target, self.config.bloom_fp_rate);
        self.bytes_written_total += entry_bytes(&merged.entries);
        self.levels[target].push(merged);
    }
}

fn entry_bytes(entries: &[LSMEntry]) -> u64 {
    entries.iter().map(|e| {
        (e.vector.as_ref().map_or(0, |v| v.len() * 4) + e.id.len()) as u64
    }).sum()
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn euclid(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(dim: usize, val: f32) -> Vec<f32> { vec![val; dim] }
    fn entry(id: &str, vec: Option<Vec<f32>>, seq: u64) -> LSMEntry {
        LSMEntry { id: id.into(), vector: vec, metadata: None, seq }
    }

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
        assert!(mt.insert("b".into(), Some(vec![2.0]), None, 1));
    }

    #[test]
    fn memtable_search_returns_closest() {
        let mut mt = MemTable::new(100);
        mt.insert("far".into(), Some(vec![10.0, 10.0]), None, 0);
        mt.insert("close".into(), Some(vec![1.0, 0.0]), None, 1);
        mt.insert("mid".into(), Some(vec![5.0, 5.0]), None, 2);
        let r = mt.search(&[0.0, 0.0], 2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].id, "close");
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

    #[test]
    fn segment_merge_dedup_keeps_latest() {
        let s1 = Segment::from_entries(vec![entry("a", Some(vec![1.0]), 1)], 0, 0.01);
        let s2 = Segment::from_entries(vec![entry("a", Some(vec![9.0]), 5)], 0, 0.01);
        let m = Segment::merge(&[s1, s2], 1, 0.01);
        assert_eq!(m.size(), 1);
        assert_eq!(m.entries[0].vector.as_ref().unwrap(), &vec![9.0]);
    }

    #[test]
    fn segment_merge_drops_tombstones() {
        let s1 = Segment::from_entries(vec![entry("a", Some(vec![1.0]), 1)], 0, 0.01);
        let s2 = Segment::from_entries(vec![entry("a", None, 5)], 0, 0.01);
        assert_eq!(Segment::merge(&[s1, s2], 1, 0.01).size(), 0);
    }

    #[test]
    fn bloom_filter_no_false_negatives() {
        let mut bf = BloomFilter::new(100, 0.01);
        for i in 0..100 { bf.insert(&format!("key-{i}")); }
        for i in 0..100 { assert!(bf.may_contain(&format!("key-{i}"))); }
    }

    #[test]
    fn bloom_filter_low_false_positive_rate() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..1000 { bf.insert(&format!("present-{i}")); }
        let fp: usize = (0..10_000).filter(|i| bf.may_contain(&format!("absent-{i}"))).count();
        assert!((fp as f64 / 10_000.0) < 0.05, "FP rate too high: {fp}/10000");
    }

    #[test]
    fn lsm_insert_and_search() {
        let mut idx = LSMIndex::new(CompactionConfig { memtable_capacity: 10, ..Default::default() });
        idx.insert("v1".into(), vec![1.0, 0.0], None);
        idx.insert("v2".into(), vec![0.0, 1.0], None);
        let r = idx.search(&[1.0, 0.0], 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, "v1");
    }

    #[test]
    fn lsm_delete_with_tombstone() {
        let mut idx = LSMIndex::new(CompactionConfig { memtable_capacity: 100, ..Default::default() });
        idx.insert("v1".into(), vec![1.0, 0.0], None);
        idx.insert("v2".into(), vec![0.0, 1.0], None);
        idx.delete("v1".into());
        let r = idx.search(&[1.0, 0.0], 2);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, "v2");
    }

    #[test]
    fn lsm_auto_compaction_trigger() {
        let cfg = CompactionConfig { memtable_capacity: 2, merge_threshold: 2, max_levels: 3, ..Default::default() };
        let mut idx = LSMIndex::new(cfg);
        for i in 0..10 { idx.insert(format!("v{i}"), vec![i as f32], None); }
        assert!(idx.stats().segments_per_level[0] < 4, "L0 should compact");
    }

    #[test]
    fn lsm_multi_level_compaction() {
        let cfg = CompactionConfig { memtable_capacity: 2, merge_threshold: 2, max_levels: 4, ..Default::default() };
        let mut idx = LSMIndex::new(cfg);
        for i in 0..30 { idx.insert(format!("v{i}"), v(4, i as f32), None); }
        let total_seg: usize = idx.stats().segments_per_level.iter().sum();
        assert!(total_seg >= 1);
    }

    #[test]
    fn lsm_write_amplification_increases() {
        let cfg = CompactionConfig { memtable_capacity: 5, merge_threshold: 2, max_levels: 3, ..Default::default() };
        let mut idx = LSMIndex::new(cfg);
        for i in 0..20 { idx.insert(format!("v{i}"), v(4, i as f32), None); }
        assert!(idx.write_amplification() >= 1.0);
    }

    #[test]
    fn lsm_empty_index() {
        let idx = LSMIndex::new(CompactionConfig::default());
        assert!(idx.search(&[0.0, 0.0], 10).is_empty());
        let s = idx.stats();
        assert_eq!(s.total_entries, 0);
        assert!((s.write_amplification - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn lsm_large_batch_insert() {
        let cfg = CompactionConfig { memtable_capacity: 50, merge_threshold: 4, max_levels: 4, ..Default::default() };
        let mut idx = LSMIndex::new(cfg);
        for i in 0..500 { idx.insert(format!("v{i}"), v(8, i as f32 * 0.01), None); }
        assert!(idx.stats().total_entries > 0);
        let r = idx.search(&v(8, 0.0), 5);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0].id, "v0");
    }

    #[test]
    fn lsm_search_across_levels() {
        let cfg = CompactionConfig { memtable_capacity: 3, merge_threshold: 3, max_levels: 3, ..Default::default() };
        let mut idx = LSMIndex::new(cfg);
        for i in 0..9 { idx.insert(format!("v{i}"), vec![i as f32, 0.0], None); }
        idx.insert("latest".into(), vec![0.0, 0.0], None);
        let r = idx.search(&[0.0, 0.0], 3);
        assert_eq!(r.len(), 3);
        let ids: Vec<&str> = r.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"latest"));
        assert!(ids.contains(&"v0"));
    }
}
