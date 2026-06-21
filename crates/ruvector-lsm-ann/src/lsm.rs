//! Three LSM-ANN index variants with increasing sophistication.
//!
//! | Variant     | L0 (MemTable) | L1 (small NSW segments) | L2 (large NSW) | Compaction |
//! |-------------|---------------|-------------------------|----------------|------------|
//! | BaselineLsm | ✓ only        | —                       | —              | —          |
//! | TwoTierLsm  | ✓             | ✓ (one segment at most) | —              | manual     |
//! | FullLsm     | ✓             | ✓ (up to l1_merge_threshold) | ✓         | auto       |
//!
//! All variants implement `LsmIndex`. Queries always merge candidate lists from
//! every populated tier and re-rank by exact squared Euclidean distance.

use crate::{merge_candidates, FrozenSegment, LsmConfig, LsmIndex, MemTable};

// ---------------------------------------------------------------------------
// Variant 1 – Baseline: MemTable only (brute-force, no graph)
// ---------------------------------------------------------------------------

/// Baseline LSM-ANN: a single flat MemTable with brute-force search.
/// Writes are O(1), searches are O(N·D). No compaction, no graph.
/// Establishes the write-throughput ceiling and recall floor.
pub struct BaselineLsm {
    pub(crate) table: MemTable,
    #[allow(dead_code)]
    config: LsmConfig,
}

impl BaselineLsm {
    pub fn new(config: LsmConfig) -> Self {
        Self {
            table: MemTable::new(),
            config,
        }
    }

    /// Total bytes used by vectors (excludes Vec/struct overhead).
    pub fn memory_bytes(&self) -> usize {
        self.table.memory_bytes()
    }
}

impl LsmIndex for BaselineLsm {
    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.table.insert(id, vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.table.search(query, k)
    }

    fn len(&self) -> usize {
        self.table.len()
    }

    fn segment_count(&self) -> usize {
        0
    }

    fn compact(&mut self) {
        // No-op: baseline has no segments to compact.
    }
}

// ---------------------------------------------------------------------------
// Variant 2 – TwoTier: MemTable + one frozen NSW segment
// ---------------------------------------------------------------------------

/// Two-tier LSM-ANN: L0 MemTable (writable) + up to one L1 frozen NSW segment.
///
/// When `compact()` is called (or L0 exceeds `l0_max`), all L0 entries are
/// built into an NSW segment and appended to `segments`. Subsequent queries
/// merge results from L0 and the segment. A second compaction merges all
/// existing segments with the new flush.
pub struct TwoTierLsm {
    pub(crate) table: MemTable,
    pub(crate) segment: Option<FrozenSegment>,
    pub(crate) config: LsmConfig,
}

impl TwoTierLsm {
    pub fn new(config: LsmConfig) -> Self {
        Self {
            table: MemTable::new(),
            segment: None,
            config,
        }
    }

    /// Total bytes used by vectors and graph edges across all tiers.
    pub fn memory_bytes(&self) -> usize {
        self.table.memory_bytes() + self.segment.as_ref().map_or(0, |s| s.memory_bytes())
    }
}

impl LsmIndex for TwoTierLsm {
    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.table.insert(id, vector);
        // Auto-compact when memtable exceeds threshold.
        if self.table.len() >= self.config.l0_max {
            self.compact();
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let l0 = self.table.search(query, k);
        match &self.segment {
            None => l0,
            Some(seg) => {
                let l1 = seg.search(query, k);
                merge_candidates(l0, l1, k)
            }
        }
    }

    fn len(&self) -> usize {
        self.table.len() + self.segment.as_ref().map_or(0, |s| s.len())
    }

    fn segment_count(&self) -> usize {
        if self.segment.is_some() {
            1
        } else {
            0
        }
    }

    fn compact(&mut self) {
        if self.table.is_empty() {
            return;
        }
        let drained = self.table.drain();

        // Merge drained data with any existing segment data, then rebuild.
        let mut combined: Vec<(u64, Vec<f32>)> = drained;
        if let Some(seg) = self.segment.take() {
            combined.extend(seg.vectors);
        }

        if combined.len() >= 2 {
            self.segment = Some(FrozenSegment::build(
                combined,
                self.config.m,
                self.config.ef_construction,
                self.config.ef_search,
            ));
        } else {
            // Too few vectors to build a meaningful graph; put them back.
            for (id, v) in combined {
                self.table.insert(id, v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Variant 3 – FullLsm: MemTable + L1 small segments + L2 large segment
// ---------------------------------------------------------------------------

/// Full three-tier LSM-ANN index with automatic background compaction.
///
/// Write path:
/// 1. Insert → L0 MemTable.
/// 2. When L0 reaches `l0_max`, flush to a new L1 segment.
/// 3. When L1 has `l1_merge_threshold` segments, merge all L1 into L2.
///
/// Query path: merge candidates from L0 (brute-force), all L1 segments
/// (NSW search), and L2 (NSW search), then re-rank.
pub struct FullLsm {
    pub(crate) table: MemTable,
    pub(crate) l1_segments: Vec<FrozenSegment>,
    pub(crate) l2_segment: Option<FrozenSegment>,
    pub(crate) config: LsmConfig,
}

impl FullLsm {
    pub fn new(config: LsmConfig) -> Self {
        Self {
            table: MemTable::new(),
            l1_segments: Vec::new(),
            l2_segment: None,
            config,
        }
    }

    /// Total bytes used by vectors and graph edges across all tiers.
    pub fn memory_bytes(&self) -> usize {
        self.table.memory_bytes()
            + self
                .l1_segments
                .iter()
                .map(|s| s.memory_bytes())
                .sum::<usize>()
            + self.l2_segment.as_ref().map_or(0, |s| s.memory_bytes())
    }

    /// Flush L0 → new L1 segment (always builds a graph if ≥2 vectors).
    fn flush_l0(&mut self) {
        let drained = self.table.drain();
        if drained.len() < 2 {
            for (id, v) in drained {
                self.table.insert(id, v);
            }
            return;
        }
        let seg = FrozenSegment::build(
            drained,
            self.config.m,
            self.config.ef_construction,
            self.config.ef_search,
        );
        self.l1_segments.push(seg);
    }

    /// Merge all L1 segments (+ optional existing L2) into a single L2 segment.
    fn merge_l1_to_l2(&mut self) {
        let mut combined: Vec<(u64, Vec<f32>)> = Vec::new();

        for seg in self.l1_segments.drain(..) {
            combined.extend(seg.vectors);
        }
        if let Some(l2) = self.l2_segment.take() {
            combined.extend(l2.vectors);
        }

        if combined.len() >= 2 {
            self.l2_segment = Some(FrozenSegment::build(
                combined,
                self.config.m,
                self.config.ef_construction,
                self.config.ef_search,
            ));
        }
    }
}

impl LsmIndex for FullLsm {
    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.table.insert(id, vector);
        if self.table.len() >= self.config.l0_max {
            self.flush_l0();
            if self.l1_segments.len() >= self.config.l1_merge_threshold {
                self.merge_l1_to_l2();
            }
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // L0: brute-force
        let mut merged = self.table.search(query, k);

        // L1 segments: NSW search, merge incrementally
        for seg in &self.l1_segments {
            let seg_res = seg.search(query, k);
            merged = merge_candidates(merged, seg_res, k);
        }

        // L2 segment: NSW search
        if let Some(l2) = &self.l2_segment {
            let l2_res = l2.search(query, k);
            merged = merge_candidates(merged, l2_res, k);
        }

        merged
    }

    fn len(&self) -> usize {
        self.table.len()
            + self.l1_segments.iter().map(|s| s.len()).sum::<usize>()
            + self.l2_segment.as_ref().map_or(0, |s| s.len())
    }

    fn segment_count(&self) -> usize {
        self.l1_segments.len() + if self.l2_segment.is_some() { 1 } else { 0 }
    }

    fn compact(&mut self) {
        // Manual compaction: flush L0 then consolidate L1 → L2.
        if !self.table.is_empty() {
            self.flush_l0();
        }
        if !self.l1_segments.is_empty() {
            self.merge_l1_to_l2();
        }
    }
}
