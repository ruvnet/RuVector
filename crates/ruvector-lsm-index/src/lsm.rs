use crate::flat::FlatSegment;
use crate::nsw::NswSegment;

/// Configuration for the three-tier LSM vector index.
#[derive(Clone, Debug)]
pub struct LsmConfig {
    /// Flush hot → warm when hot reaches this size.
    pub hot_capacity: usize,
    /// Flush warm → cold when warm reaches this size.
    pub warm_capacity: usize,
    /// NSW M parameter (target edges per node).
    pub nsw_m: usize,
    /// NSW ef_construction (beam width during graph build).
    pub nsw_ef_build: usize,
    /// Vector dimensionality.
    pub dims: usize,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            hot_capacity: 256,
            warm_capacity: 4096,
            nsw_m: 16,
            nsw_ef_build: 40,
            dims: 128,
        }
    }
}

/// Statistics snapshot from an [`LsmVectorIndex`].
#[derive(Clone, Debug)]
pub struct LsmStats {
    pub hot_size: usize,
    pub warm_size: usize,
    pub cold_size: usize,
    pub total: usize,
    pub flushes_to_warm: u64,
    pub flushes_to_cold: u64,
    pub memory_bytes: usize,
}

/// Three-tier LSM-style vector index.
///
/// Tier layout
/// ───────────
/// hot  (FlatSegment) — newest writes, linear scan, O(1) insert
/// warm (NswSegment)  — recent epochs, NSW graph, O(log n) search
/// cold (NswSegment)  — stable bulk, NSW graph, O(log n) search
///
/// Write path: insert → hot → (flush) → warm → (flush) → cold
/// Read  path: search hot ∪ warm ∪ cold → merge → top-k
///
/// Compaction is synchronous on write (no background thread required),
/// making this suitable for single-threaded agents and WASM targets.
pub struct LsmVectorIndex {
    hot: FlatSegment,
    warm: NswSegment,
    cold: NswSegment,
    cfg: LsmConfig,
    total: usize,
    flushes_to_warm: u64,
    flushes_to_cold: u64,
}

impl LsmVectorIndex {
    pub fn new(cfg: LsmConfig) -> Self {
        let m = cfg.nsw_m;
        let ef = cfg.nsw_ef_build;
        let d = cfg.dims;
        Self {
            hot: FlatSegment::new(d),
            warm: NswSegment::new(d, m, ef),
            cold: NswSegment::new(d, m, ef),
            cfg,
            total: 0,
            flushes_to_warm: 0,
            flushes_to_cold: 0,
        }
    }

    /// Insert one vector. O(1) amortised (hot is a flat append).
    /// Compaction happens inline when tier thresholds are exceeded.
    pub fn insert(&mut self, id: u64, vec: Vec<f32>) {
        self.hot.insert(id, vec);
        self.total += 1;

        if self.hot.len() >= self.cfg.hot_capacity {
            self.flush_hot_to_warm();
        }
        if self.warm.len() >= self.cfg.warm_capacity {
            self.flush_warm_to_cold();
        }
    }

    /// Search all tiers and return the k nearest neighbours.
    /// Results from each tier are merged and deduplicated.
    /// ef_per_tier: beam width per segment search (default: max(k, ef_build * 3)).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(f32, u64)> {
        let ef = k.max(self.cfg.nsw_ef_build * 3);
        let mut all: Vec<(f32, u64)> = Vec::with_capacity(k * 3);
        all.extend_from_slice(&self.hot.search(query, k));
        all.extend_from_slice(&self.warm.search_ef(query, k, ef));
        all.extend_from_slice(&self.cold.search_ef(query, k, ef));

        // Sort by distance, then deduplicate by id (keep first = closest).
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut seen = std::collections::HashSet::new();
        all.retain(|(_, id)| seen.insert(*id));
        all.truncate(k);
        all
    }

    /// Total number of vectors inserted (including those pending flush).
    pub fn len(&self) -> usize {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Current tier occupancy and memory estimate.
    pub fn stats(&self) -> LsmStats {
        LsmStats {
            hot_size: self.hot.len(),
            warm_size: self.warm.len(),
            cold_size: self.cold.len(),
            total: self.total,
            flushes_to_warm: self.flushes_to_warm,
            flushes_to_cold: self.flushes_to_cold,
            memory_bytes: self.hot.memory_bytes()
                + self.warm.memory_bytes()
                + self.cold.memory_bytes(),
        }
    }

    // ─── compaction ────────────────────────────────────────────────────────────

    fn flush_hot_to_warm(&mut self) {
        let hot_entries = self.hot.drain_all();
        let m = self.cfg.nsw_m;
        let ef = self.cfg.nsw_ef_build;
        let d = self.cfg.dims;

        if self.warm.is_empty() {
            self.warm = NswSegment::build_from(hot_entries, d, m, ef);
        } else {
            // Absorb hot into warm by rebuilding the warm graph.
            let mut all = self.warm.drain_all();
            all.extend(hot_entries);
            self.warm = NswSegment::build_from(all, d, m, ef);
        }
        self.flushes_to_warm += 1;
    }

    fn flush_warm_to_cold(&mut self) {
        let warm_entries = self.warm.drain_all();
        let m = self.cfg.nsw_m;
        let ef = self.cfg.nsw_ef_build;
        let d = self.cfg.dims;

        if self.cold.is_empty() {
            self.cold = NswSegment::build_from(warm_entries, d, m, ef);
        } else {
            let mut all = self.cold.drain_all();
            all.extend(warm_entries);
            self.cold = NswSegment::build_from(all, d, m, ef);
        }
        self.flushes_to_cold += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatSegment;

    fn xvecs(n: usize, dims: usize, seed: u32) -> Vec<(u64, Vec<f32>)> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dims)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        (s as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0
                    })
                    .collect();
                (i as u64, v)
            })
            .collect()
    }

    #[test]
    fn lsm_insert_and_search_basic() {
        let cfg = LsmConfig {
            hot_capacity: 32,
            warm_capacity: 256,
            nsw_m: 8,
            nsw_ef_build: 20,
            dims: 8,
        };
        let vecs = xvecs(150, 8, 7);
        let mut idx = LsmVectorIndex::new(cfg);
        for (id, v) in &vecs {
            idx.insert(*id, v.clone());
        }
        assert_eq!(idx.len(), 150);
        let results = idx.search(&vecs[0].1, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Nearest to vecs[0] must be vecs[0] itself (dist ≈ 0).
        assert!(results[0].0 < 1e-5, "nearest dist = {}", results[0].0);
    }

    #[test]
    fn lsm_recall_at_least_60_pct() {
        let n = 1000;
        let dims = 64;
        let k = 10;
        let vecs = xvecs(n, dims, 42);
        let queries = xvecs(100, dims, 12345);

        let cfg = LsmConfig {
            hot_capacity: 128,
            warm_capacity: 1024,
            nsw_m: 16,
            nsw_ef_build: 40,
            dims,
        };
        let mut index = LsmVectorIndex::new(cfg);
        for (id, v) in &vecs {
            index.insert(*id, v.clone());
        }

        let mut flat = FlatSegment::new(dims);
        for (id, v) in &vecs {
            flat.insert(*id, v.clone());
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for (_, q) in &queries {
            let gt: std::collections::HashSet<u64> =
                flat.search(q, k).iter().map(|(_, id)| *id).collect();
            let res: std::collections::HashSet<u64> =
                index.search(q, k).iter().map(|(_, id)| *id).collect();
            hits += gt.intersection(&res).count();
            total += k;
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.60,
            "LSM recall@10 = {:.3}, expected >=0.60",
            recall
        );
    }

    #[test]
    fn stats_track_flushes() {
        let cfg = LsmConfig {
            hot_capacity: 10,
            warm_capacity: 100,
            nsw_m: 4,
            nsw_ef_build: 10,
            dims: 4,
        };
        let mut idx = LsmVectorIndex::new(cfg);
        for i in 0u64..55 {
            idx.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }
        let s = idx.stats();
        assert!(
            s.flushes_to_warm >= 5,
            "expected ≥5 warm flushes, got {}",
            s.flushes_to_warm
        );
        assert_eq!(s.total, 55);
    }
}
