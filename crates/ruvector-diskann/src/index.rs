//! DiskANN index — ties together Vamana graph, PQ, and mmap persistence

// Lint suggests `as_chunks`, stable only since Rust 1.88 — past the declared
// `rust-version = "1.77"`. `unknown_lints` keeps older toolchains quiet.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use crate::distance::{l2_squared, FlatVectors, VisitedSet};
use crate::error::{DiskAnnError, Result};
use crate::graph::VamanaGraph;
use crate::pq::ProductQuantizer;
use memmap2::MmapOptions;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, TryLockError};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_VISITED_POOL_SIZE: usize = 4;

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
}

/// Configuration for DiskANN index
#[derive(Debug, Clone)]
pub struct DiskAnnConfig {
    /// Vector dimension
    pub dim: usize,
    /// Maximum out-degree for Vamana graph (R)
    pub max_degree: usize,
    /// Search beam width during construction (L_build)
    pub build_beam: usize,
    /// Search beam width during query (L_search)
    pub search_beam: usize,
    /// Alpha parameter for robust pruning (>= 1.0)
    pub alpha: f32,
    /// Number of PQ subspaces (M). 0 = no PQ.
    pub pq_subspaces: usize,
    /// PQ training iterations
    pub pq_iterations: usize,
    /// Storage directory for persistence
    pub storage_path: Option<PathBuf>,
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_degree: 64,
            build_beam: 128,
            search_beam: 64,
            alpha: 1.2,
            pq_subspaces: 0,
            pq_iterations: 10,
            storage_path: None,
        }
    }
}

/// DiskANN index with Vamana graph + optional PQ + mmap persistence
pub struct DiskAnnIndex {
    config: DiskAnnConfig,
    /// Flat contiguous vector storage (cache-friendly)
    vectors: FlatVectors,
    /// ID mapping: internal index -> external string ID
    id_map: Vec<String>,
    /// Reverse mapping: external ID -> internal index
    id_reverse: HashMap<String, u32>,
    /// Packed deletion markers. Tombstoned vectors remain available for routing.
    tombstones: Vec<u64>,
    /// Number of tombstoned vectors.
    deleted_count: usize,
    /// Vamana graph
    graph: Option<VamanaGraph>,
    /// Product quantizer (optional)
    pq: Option<ProductQuantizer>,
    /// PQ codes for all vectors
    pq_codes: Vec<Vec<u8>>,
    /// Whether index has been built
    built: bool,
    /// Small pool of reusable visited sets for shared-reference searches.
    visited_pool: Mutex<Vec<VisitedSet>>,
}

impl DiskAnnIndex {
    /// Create a new DiskANN index
    pub fn new(config: DiskAnnConfig) -> Self {
        let dim = config.dim;
        Self {
            config,
            vectors: FlatVectors::new(dim),
            id_map: Vec::new(),
            id_reverse: HashMap::new(),
            tombstones: Vec::new(),
            deleted_count: 0,
            graph: None,
            pq: None,
            pq_codes: Vec::new(),
            built: false,
            visited_pool: Mutex::new(Vec::new()),
        }
    }

    /// Insert a vector with a string ID
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if self.vectors.is_mmap_backed() {
            return Err(DiskAnnError::InvalidConfig(
                "cannot insert into an mmap-loaded index (read-only, read-through storage); use DiskAnnIndex::load() instead of load_mmap() if further inserts are needed".to_string(),
            ));
        }
        if vector.len() != self.config.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dim,
                actual: vector.len(),
            });
        }
        if self.id_reverse.contains_key(&id) {
            return Err(DiskAnnError::InvalidConfig(format!("Duplicate ID: {id}")));
        }

        let idx = self.vectors.len() as u32;
        self.id_reverse.insert(id.clone(), idx);
        self.id_map.push(id);
        if idx as usize / 64 == self.tombstones.len() {
            self.tombstones.push(0);
        }
        self.vectors.push(&vector);
        self.built = false;
        Ok(())
    }

    /// Insert a batch of vectors
    pub fn insert_batch(&mut self, entries: Vec<(String, Vec<f32>)>) -> Result<()> {
        for (id, vector) in entries {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Build the index (must be called after all inserts, before search)
    pub fn build(&mut self) -> Result<()> {
        let n = self.vectors.len();
        if n == 0 {
            return Err(DiskAnnError::Empty);
        }

        // Train PQ if configured
        if self.config.pq_subspaces > 0 {
            // Collect vectors for PQ training
            let vecs: Vec<Vec<f32>> = (0..n).map(|i| self.vectors.get(i).to_vec()).collect();
            let mut pq = ProductQuantizer::new(self.config.dim, self.config.pq_subspaces)?;
            pq.train(&vecs, self.config.pq_iterations)?;

            self.pq_codes = vecs
                .iter()
                .map(|v| pq.encode(v))
                .collect::<Result<Vec<_>>>()?;

            self.pq = Some(pq);
        }

        // Build Vamana graph on flat storage
        let mut graph = VamanaGraph::new(
            n,
            self.config.max_degree,
            self.config.build_beam,
            self.config.alpha,
        );
        graph.build(&self.vectors)?;
        for deleted in 0..n {
            if self.is_tombstoned(deleted) {
                graph.repair_deleted(&self.vectors, deleted as u32, &self.tombstones);
            }
        }
        self.graph = Some(graph);

        // Seed the single-threaded fast path. Concurrent searches allocate a
        // fallback set when every pooled set is checked out.
        *self
            .visited_pool
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = vec![VisitedSet::new(n)];
        self.built = true;

        if let Some(ref path) = self.config.storage_path {
            self.save(path)?;
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.validate_search(query)?;

        let mut visited = match self.visited_pool.try_lock() {
            Ok(mut pool) => pool
                .pop()
                .unwrap_or_else(|| VisitedSet::new(self.vectors.len())),
            Err(TryLockError::Poisoned(poisoned)) => poisoned
                .into_inner()
                .pop()
                .unwrap_or_else(|| VisitedSet::new(self.vectors.len())),
            Err(TryLockError::WouldBlock) => VisitedSet::new(self.vectors.len()),
        };

        let result = self.search_with(query, k, &mut visited);

        let mut pool = self
            .visited_pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if pool.len() < MAX_VISITED_POOL_SIZE {
            pool.push(visited);
        }

        result
    }

    /// Search with caller-owned visited-state storage.
    ///
    /// Reuse the same [`VisitedSet`] for steady-state searches without visited
    /// set allocation. If its size differs from this index, the set is resized
    /// and fully reset before the search.
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        visited: &mut VisitedSet,
    ) -> Result<Vec<SearchResult>> {
        self.validate_search(query)?;

        let graph = self.graph.as_ref().unwrap();
        let beam = self.config.search_beam.max(k);

        let (candidates, _) = graph.greedy_search_fast(&self.vectors, query, beam, visited);

        // Re-rank candidates with exact distance
        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter(|&id| !self.is_tombstoned(id as usize))
            .map(|id| (id, l2_squared(self.vectors.get(id as usize), query)))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(id, dist)| SearchResult {
                id: self.id_map[id as usize].clone(),
                distance: dist,
            })
            .collect())
    }

    fn validate_search(&self, query: &[f32]) -> Result<()> {
        if !self.built {
            return Err(DiskAnnError::NotBuilt);
        }
        if query.len() != self.config.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dim,
                actual: query.len(),
            });
        }

        Ok(())
    }

    /// Get the number of vectors in the index
    pub fn count(&self) -> usize {
        self.vectors.len() - self.deleted_count
    }

    /// Delete a vector by ID and locally repair its live in-neighbors.
    ///
    /// In-neighbor discovery scans the bounded-degree adjacency, making this
    /// O(n * degree) per delete. Unknown and already-deleted IDs return `Ok(false)`
    /// for compatibility with the original delete API.
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let Some(idx) = self.mark_deleted(id) else {
            return Ok(false);
        };
        if let Some(graph) = self.graph.as_mut() {
            graph.repair_deleted(&self.vectors, idx, &self.tombstones);
        }
        Ok(true)
    }

    /// Tombstone a vector without repairing graph edges.
    ///
    /// Searches may still traverse the preserved vector as a routing waypoint,
    /// but the ID is filtered from results. Unknown and repeated deletes return
    /// `Ok(false)`.
    pub fn delete_deferred(&mut self, id: &str) -> Result<bool> {
        Ok(self.mark_deleted(id).is_some())
    }

    fn mark_deleted(&mut self, id: &str) -> Option<u32> {
        let idx = self.id_reverse.remove(id)?;
        let idx = idx as usize;
        self.tombstones[idx / 64] |= 1u64 << (idx % 64);
        self.deleted_count += 1;
        Some(idx as u32)
    }

    fn is_tombstoned(&self, idx: usize) -> bool {
        self.tombstones
            .get(idx / 64)
            .is_some_and(|word| word & (1u64 << (idx % 64)) != 0)
    }

    /// Save index to disk
    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;

        // Save vectors as flat binary (already contiguous — mmap-friendly)
        let vec_path = dir.join("vectors.bin");
        let saving_unchanged_mmap_in_place = self.vectors.is_mmap_backed()
            && self.config.storage_path.as_deref() == Some(dir)
            && vec_path.exists();
        if !saving_unchanged_mmap_in_place {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let vec_temp_path =
                dir.join(format!(".vectors.bin.{}.{nonce}.tmp", std::process::id()));
            let mut f = BufWriter::new(
                File::options()
                    .write(true)
                    .create_new(true)
                    .open(&vec_temp_path)?,
            );
            let n = self.vectors.len() as u64;
            let dim = self.config.dim as u64;
            f.write_all(&n.to_le_bytes())?;
            f.write_all(&dim.to_le_bytes())?;
            if let Some(data) = self.vectors.as_owned_slice() {
                // Owned storage: write the flat slab directly — zero copy.
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                f.write_all(byte_slice)?;
            } else {
                // Mmap-backed storage being copied to a different directory:
                // re-serialize each vector without materializing the full slab.
                for i in 0..self.vectors.len() {
                    for &value in self.vectors.get(i) {
                        f.write_all(&value.to_le_bytes())?;
                    }
                }
            }
            f.flush()?;
            f.get_ref().sync_all()?;
            drop(f);
            if let Err(error) = fs::rename(&vec_temp_path, &vec_path) {
                #[cfg(windows)]
                {
                    if vec_path.exists() {
                        fs::remove_file(&vec_path)?;
                        if fs::rename(&vec_temp_path, &vec_path).is_ok() {
                            // Windows cannot atomically replace an existing file.
                            // The source mmap case is skipped above, so removal is
                            // safe here and only affects a separate destination.
                        } else {
                            let _ = fs::remove_file(&vec_temp_path);
                            return Err(error.into());
                        }
                    } else {
                        let _ = fs::remove_file(&vec_temp_path);
                        return Err(error.into());
                    }
                }
                #[cfg(not(windows))]
                {
                    let _ = fs::remove_file(&vec_temp_path);
                    return Err(error.into());
                }
            }
        }

        // Save graph adjacency
        let graph_path = dir.join("graph.bin");
        let mut f = BufWriter::new(File::create(&graph_path)?);
        if let Some(ref graph) = self.graph {
            f.write_all(&(graph.medoid as u64).to_le_bytes())?;
            f.write_all(&(graph.neighbors.len() as u64).to_le_bytes())?;
            for neighbors in &graph.neighbors {
                f.write_all(&(neighbors.len() as u32).to_le_bytes())?;
                for &n in neighbors {
                    f.write_all(&n.to_le_bytes())?;
                }
            }
        } else {
            // Unbuilt: still write a well-formed, empty header (medoid 0, 0 nodes).
            f.write_all(&[0u8; 16])?;
        }
        f.flush()?;

        // Save ID map
        let ids_path = dir.join("ids.json");
        let ids_json = serde_json::to_string(&self.id_map)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
        fs::write(&ids_path, ids_json)?;

        // Persist tombstones independently. Older indexes without this file are
        // migrated by recognizing the all-NaN rows used by prior delete logic.
        let tombstones: Vec<u8> = self
            .tombstones
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        fs::write(dir.join("tombstones.bin"), tombstones)?;

        // Save PQ if present
        if let Some(ref pq) = self.pq {
            let pq_path = dir.join("pq.bin");
            let pq_bytes = bincode::encode_to_vec(pq, bincode::config::standard())
                .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
            fs::write(&pq_path, pq_bytes)?;

            // Save PQ codes
            let codes_path = dir.join("pq_codes.bin");
            let mut f = BufWriter::new(File::create(&codes_path)?);
            for codes in &self.pq_codes {
                f.write_all(codes)?;
            }
            f.flush()?;
        }

        // Save config
        let config_path = dir.join("config.json");
        let config_json = serde_json::json!({
            "dim": self.config.dim,
            "max_degree": self.config.max_degree,
            "build_beam": self.config.build_beam,
            "search_beam": self.config.search_beam,
            "alpha": self.config.alpha,
            "pq_subspaces": self.config.pq_subspaces,
            "count": self.count(),
            "vector_count": self.vectors.len(),
            "built": self.built,
        });
        fs::write(
            &config_path,
            serde_json::to_string_pretty(&config_json).unwrap(),
        )?;

        Ok(())
    }

    /// Load index from disk with vectors fully materialized into RAM (default,
    /// back-compat behavior — unchanged from prior releases).
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_impl(dir, false)
    }

    /// Load index from disk with vectors served read-through from an mmap: the OS
    /// pages in only accessed vectors, so RSS stays proportional to the query
    /// working set instead of the full dataset. Graph adjacency, PQ codes, and IDs
    /// are still heap-resident (small relative to the vector slab). See #674.
    ///
    /// Opt-in — `load()` remains the default and is unaffected. Further inserts on
    /// the returned index fail with `DiskAnnError::InvalidConfig`; the read-through
    /// map is not writable. Deletes use the same persistent tombstone bitmap as
    /// owned indexes.
    pub fn load_mmap(dir: &Path) -> Result<Self> {
        Self::load_impl(dir, true)
    }

    fn load_impl(dir: &Path, mmap_vectors: bool) -> Result<Self> {
        // Load config
        let config_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)
                .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

        let field = |name: &str| {
            config_json[name]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| {
                    DiskAnnError::Serialization(format!("config.json: missing or invalid `{name}`"))
                })
        };
        let dim = field("dim")?;
        let max_degree = field("max_degree")?;
        let build_beam = field("build_beam")?;
        let search_beam = field("search_beam")?;
        let alpha = config_json["alpha"].as_f64().ok_or_else(|| {
            DiskAnnError::Serialization("config.json: missing or invalid `alpha`".to_string())
        })? as f32;
        let pq_subspaces = config_json["pq_subspaces"].as_u64().unwrap_or(0) as usize;
        // Releases before this field existed only saved built indexes.
        let built = config_json["built"].as_bool().unwrap_or(true);

        let config = DiskAnnConfig {
            dim,
            max_degree,
            build_beam,
            search_beam,
            alpha,
            pq_subspaces,
            storage_path: Some(dir.to_path_buf()),
            ..Default::default()
        };

        // Map the vectors file (needed either way to read the 16-byte header; in
        // mmap mode the map is retained and read through, in owned mode it is
        // copied out below and then dropped).
        let vec_file = File::open(dir.join("vectors.bin"))?;
        let mmap = unsafe { MmapOptions::new().map(&vec_file)? };
        if mmap.len() < 16 {
            return Err(DiskAnnError::Serialization(
                "vectors.bin is shorter than its 16-byte header".to_string(),
            ));
        }

        let n = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let file_dim = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        if file_dim != dim {
            return Err(DiskAnnError::Serialization(format!(
                "vector dimension {file_dim} does not match config dimension {dim}"
            )));
        }
        let expected_vector_bytes = n
            .checked_mul(dim)
            .and_then(|floats| floats.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or_else(|| {
                DiskAnnError::Serialization("vector slab size overflowed usize".to_string())
            })?;
        if mmap.len() != expected_vector_bytes {
            return Err(DiskAnnError::Serialization(format!(
                "vectors.bin has {} bytes; expected {expected_vector_bytes}",
                mmap.len()
            )));
        }

        let data_start = 16;
        let vectors = if mmap_vectors {
            FlatVectors::from_mmap(mmap, data_start, dim, n)?
        } else {
            // Copy the flat slab out of the map into an owned Vec — unchanged
            // from the pre-#674 behavior.
            let total_floats = n * dim;
            let mut flat_data = Vec::with_capacity(total_floats);
            let byte_slice = &mmap[data_start..data_start + total_floats * 4];
            // Safe: f32 from le bytes
            for chunk in byte_slice.chunks_exact(4) {
                flat_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            FlatVectors::from_owned(flat_data, dim, n)
        };

        // Load IDs
        let ids_json = fs::read_to_string(dir.join("ids.json"))?;
        let id_map: Vec<String> = serde_json::from_str(&ids_json)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
        if id_map.len() != n {
            return Err(DiskAnnError::Serialization(format!(
                "ID count {} does not match vector count {n}",
                id_map.len()
            )));
        }

        let tombstone_path = dir.join("tombstones.bin");
        let has_persisted_tombstones = tombstone_path.exists();
        let tombstones = if has_persisted_tombstones {
            let bytes = fs::read(tombstone_path)?;
            let word_count = n.div_ceil(64);
            if bytes.len() != word_count * 8 {
                return Err(DiskAnnError::Serialization(
                    "invalid tombstone data".to_string(),
                ));
            }
            let words: Vec<u64> = bytes
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            if n % 64 != 0
                && words
                    .last()
                    .is_some_and(|word| word & !((1u64 << (n % 64)) - 1) != 0)
            {
                return Err(DiskAnnError::Serialization(
                    "invalid tombstone data".to_string(),
                ));
            }
            words
        } else {
            // Pre-tombstone releases encoded deletion by replacing the entire
            // vector row with NaNs. Recover those rows as tombstones instead of
            // silently resurrecting their IDs during an upgrade.
            let mut words = vec![0; n.div_ceil(64)];
            for idx in 0..n {
                if vectors.get(idx).iter().all(|value| value.is_nan()) {
                    words[idx / 64] |= 1u64 << (idx % 64);
                }
            }
            words
        };
        let deleted_count = tombstones
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum();

        let mut id_reverse = HashMap::new();
        for (i, id) in id_map.iter().enumerate() {
            let is_tombstoned = tombstones[i / 64] & (1u64 << (i % 64)) != 0;
            if !is_tombstoned && id_reverse.insert(id.clone(), i as u32).is_some() {
                return Err(DiskAnnError::Serialization(format!(
                    "duplicate live ID in ids.json: {id}"
                )));
            }
        }

        if let Some(vector_count) = config_json["vector_count"].as_u64() {
            if vector_count as usize != n {
                return Err(DiskAnnError::Serialization(format!(
                    "config vector_count {vector_count} does not match vectors.bin count {n}"
                )));
            }
        }
        if has_persisted_tombstones {
            if let Some(live_count) = config_json["count"].as_u64() {
                let expected_live = n - deleted_count;
                if live_count as usize != expected_live {
                    return Err(DiskAnnError::Serialization(format!(
                        "config count {live_count} does not match tombstone-derived live count {expected_live}"
                    )));
                }
            }
        }

        // Load graph
        // An index saved before build() has no usable graph; load it unbuilt so
        // the caller can build() it, rather than trusting a stale adjacency.
        let graph = if built {
            let graph_bytes = fs::read(dir.join("graph.bin"))?;
            Some(VamanaGraph::from_bytes(
                &graph_bytes,
                n,
                max_degree,
                build_beam,
                alpha,
            )?)
        } else {
            None
        };

        // Load PQ if present
        let pq_path = dir.join("pq.bin");
        let (pq, pq_codes) = if pq_path.exists() {
            let pq_bytes = fs::read(&pq_path)?;
            // Bound bincode's up-front container allocations: without a limit a
            // crafted length prefix requests petabytes and aborts the process.
            let pq_config = bincode::config::standard().with_limit::<{ 1 << 30 }>();
            let (pq, _): (ProductQuantizer, usize) =
                bincode::decode_from_slice(&pq_bytes, pq_config)
                    .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
            let m = pq.m;
            if m == 0 || pq.dim != dim || m.checked_mul(pq.dsub) != Some(dim) {
                return Err(DiskAnnError::Serialization(format!(
                    "pq.bin: m={m}, dsub={}, dim={} inconsistent with index dim {dim}",
                    pq.dsub, pq.dim
                )));
            }

            let codes_bytes = fs::read(dir.join("pq_codes.bin"))?;
            if n.checked_mul(m) != Some(codes_bytes.len()) {
                return Err(DiskAnnError::Serialization(format!(
                    "pq_codes.bin has {} bytes; expected {n} x {m}",
                    codes_bytes.len()
                )));
            }
            let codes = codes_bytes.chunks_exact(m).map(<[u8]>::to_vec).collect();
            (Some(pq), codes)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            config,
            vectors,
            id_map,
            id_reverse,
            tombstones,
            deleted_count,
            graph,
            pq,
            pq_codes,
            built,
            visited_pool: Mutex::new(vec![VisitedSet::new(n)]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn random_vectors(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        use rand::prelude::*;
        // Seeded so tests are deterministic across CI runs — random data made
        // basic-search assertions (nearest of vec-X is vec-X) flake when the
        // ANN graph traversal happened to land on an unrelated near-duplicate.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("vec-{i}"), v)
            })
            .collect()
    }

    fn random_data(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        random_vectors(n, dim)
    }

    #[test]
    fn test_diskann_basic() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 32,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            ..Default::default()
        });

        let data = random_vectors(500, 32);
        let query = data[42].1.clone();

        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "vec-42"); // Should find itself
        assert!(results[0].distance < 1e-6); // Exact match
    }

    #[test]
    fn search_paths_return_identical_ids() {
        use rand::prelude::*;

        let dim = 32;
        let n = 500;
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            ..Default::default()
        });
        index.insert_batch(random_vectors(n, dim)).unwrap();
        index.build().unwrap();

        let mut query_rng = rand::rngs::StdRng::seed_from_u64(0x6775_EA2C);
        let mut visited = VisitedSet::new(n);

        for _ in 0..100 {
            let query: Vec<f32> = (0..dim).map(|_| query_rng.gen()).collect();
            let pooled_ids: Vec<String> = index
                .search(&query, 10)
                .unwrap()
                .into_iter()
                .map(|result| result.id)
                .collect();
            let caller_owned_ids: Vec<String> = index
                .search_with(&query, 10, &mut visited)
                .unwrap()
                .into_iter()
                .map(|result| result.id)
                .collect();

            // This mirrors the pre-fix search shape: allocate a VisitedSet in
            // graph.greedy_search, then perform the same exact-distance rerank.
            let graph = index.graph.as_ref().unwrap();
            let beam = index.config.search_beam.max(10);
            let (candidates, _) = graph.greedy_search(&index.vectors, &query, beam);
            let mut scored: Vec<(u32, f32)> = candidates
                .into_iter()
                .map(|id| (id, l2_squared(index.vectors.get(id as usize), &query)))
                .collect();
            scored.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let old_path_ids: Vec<String> = scored
                .into_iter()
                .take(10)
                .map(|(id, _)| index.id_map[id as usize].clone())
                .collect();

            assert_eq!(pooled_ids, old_path_ids);
            assert_eq!(caller_owned_ids, old_path_ids);
        }

        index.delete("vec-7").unwrap();
        let query = vec![0.5; dim];
        let pooled_ids: Vec<String> = index
            .search(&query, 10)
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect();
        let caller_owned_ids: Vec<String> = index
            .search_with(&query, 10, &mut visited)
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect();
        assert_eq!(pooled_ids, caller_owned_ids);
        assert!(!pooled_ids.iter().any(|id| id == "vec-7"));
    }

    #[test]
    fn concurrent_searches_match_single_threaded_results() {
        let dim = 16;
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            ..Default::default()
        });
        let data = random_vectors(1_000, dim);
        let query = data[123].1.clone();
        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let expected: Vec<(String, u32)> = index
            .search(&query, 10)
            .unwrap()
            .into_iter()
            .map(|result| (result.id, result.distance.to_bits()))
            .collect();

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..100 {
                        let actual: Vec<(String, u32)> = index
                            .search(&query, 10)
                            .unwrap()
                            .into_iter()
                            .map(|result| (result.id, result.distance.to_bits()))
                            .collect();
                        assert_eq!(actual, expected);
                    }
                });
            }
        });
    }

    #[test]
    fn test_diskann_with_pq() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 32,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            pq_subspaces: 4,
            pq_iterations: 5,
            ..Default::default()
        });

        let data = random_vectors(200, 32);
        let query = data[10].1.clone();

        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let results = index.search(&query, 5).unwrap();
        assert_eq!(results[0].id, "vec-10");
    }

    #[test]
    fn test_diskann_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_test");

        let data = random_vectors(100, 16);
        let query = data[7].1.clone();

        // Build and save
        {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim: 16,
                max_degree: 8,
                build_beam: 16,
                search_beam: 16,
                alpha: 1.2,
                storage_path: Some(path.clone()),
                ..Default::default()
            });
            index.insert_batch(data).unwrap();
            index.build().unwrap();
        }

        // Load and search
        let loaded = DiskAnnIndex::load(&path).unwrap();
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results[0].id, "vec-7");
    }

    #[test]
    fn test_diskann_load_mmap_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_mmap_test");
        let data = random_vectors(100, 16);
        let query = data[7].1.clone();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 16,
            max_degree: 8,
            build_beam: 16,
            search_beam: 16,
            storage_path: Some(path.clone()),
            ..Default::default()
        });
        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let loaded = DiskAnnIndex::load_mmap(&path).unwrap();
        assert!(loaded.vectors.is_mmap_backed());
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results[0].id, "vec-7");
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_load_and_load_mmap_return_identical_results() {
        use rand::prelude::*;

        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_parity_test");
        let data = random_vectors(300, 24);
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 24,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            storage_path: Some(path.clone()),
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();
        index.build().unwrap();

        let owned = DiskAnnIndex::load(&path).unwrap();
        let mmapped = DiskAnnIndex::load_mmap(&path).unwrap();
        let mut rng = StdRng::seed_from_u64(0xACED);
        for _ in 0..20 {
            let query = &data[rng.gen_range(0..data.len())].1;
            let owned_results: Vec<(String, f32)> = owned
                .search(query, 5)
                .unwrap()
                .into_iter()
                .map(|result| (result.id, result.distance))
                .collect();
            let mmap_results: Vec<(String, f32)> = mmapped
                .search(query, 5)
                .unwrap()
                .into_iter()
                .map(|result| (result.id, result.distance))
                .collect();
            assert_eq!(owned_results, mmap_results);
        }
    }

    #[test]
    fn test_insert_rejected_on_mmap_loaded_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_mmap_insert_reject_test");
        let data = random_vectors(20, 4);
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            max_degree: 4,
            build_beam: 8,
            search_beam: 8,
            storage_path: Some(path.clone()),
            ..Default::default()
        });
        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let mut loaded = DiskAnnIndex::load_mmap(&path).unwrap();
        assert!(loaded.insert("new-vec".to_string(), vec![1.0; 4]).is_err());
    }

    #[test]
    fn test_delete_filters_exact_match_and_preserves_vector() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 16,
            max_degree: 8,
            build_beam: 24,
            search_beam: 24,
            ..Default::default()
        });
        let data = random_vectors(200, 16);
        let query = data[42].1.clone();
        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let internal_id = index.id_reverse["vec-42"] as usize;
        let before = index.vectors.get(internal_id).to_vec();
        index.delete("vec-42").unwrap();

        assert_eq!(index.vectors.get(internal_id), before);
        let graph = index.graph.as_ref().unwrap();
        assert!(graph.neighbors[internal_id].is_empty());
        assert!(graph
            .neighbors
            .iter()
            .all(|neighbors| !neighbors.contains(&(internal_id as u32))));
        let first = index.search(&query, 10).unwrap();
        let second = index.search(&query, 10).unwrap();
        assert!(first.iter().all(|result| result.id != "vec-42"));
        assert!(first.iter().all(|result| result.distance.is_finite()));
        assert_eq!(
            first
                .iter()
                .map(|result| (&result.id, result.distance.to_bits()))
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|result| (&result.id, result.distance.to_bits()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_delete_count_and_errors() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![0.0; 4]).unwrap();
        index.insert("b".to_string(), vec![1.0; 4]).unwrap();
        assert_eq!(index.count(), 2);

        assert!(!index.delete("missing").unwrap());
        assert!(index.delete_deferred("a").unwrap());
        assert_eq!(index.count(), 1);
        assert!(!index.delete("a").unwrap());
    }

    #[test]
    fn test_tombstones_survive_save_load() {
        let dir = tempdir().unwrap();
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 8,
            max_degree: 8,
            build_beam: 16,
            search_beam: 16,
            ..Default::default()
        });
        let data = random_vectors(100, 8);
        let deleted_query = data[9].1.clone();
        index.insert_batch(data).unwrap();
        index.build().unwrap();
        index.delete_deferred("vec-9").unwrap();
        index.save(dir.path()).unwrap();
        assert_eq!(
            fs::metadata(dir.path().join("tombstones.bin"))
                .unwrap()
                .len(),
            16
        );

        let loaded = DiskAnnIndex::load(dir.path()).unwrap();
        assert_eq!(loaded.count(), 99);
        assert!(loaded
            .search(&deleted_query, 10)
            .unwrap()
            .iter()
            .all(|result| result.id != "vec-9"));
    }

    #[test]
    fn test_legacy_nan_delete_migrates_to_tombstone() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy");
        let data = random_vectors(20, 4);
        let deleted_query = data[3].1.clone();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            max_degree: 4,
            build_beam: 8,
            search_beam: 8,
            ..Default::default()
        });
        index.insert_batch(data).unwrap();
        index.build().unwrap();
        index.save(&path).unwrap();

        // Emulate the pre-tombstone format: row 3 was overwritten with NaNs and
        // there was no tombstones.bin sidecar.
        fs::remove_file(path.join("tombstones.bin")).unwrap();
        let vectors_path = path.join("vectors.bin");
        let mut bytes = fs::read(&vectors_path).unwrap();
        let row_start = 16 + 3 * 4 * 4;
        for offset in (row_start..row_start + 16).step_by(4) {
            bytes[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        fs::write(vectors_path, bytes).unwrap();

        for loaded in [
            DiskAnnIndex::load(&path).unwrap(),
            DiskAnnIndex::load_mmap(&path).unwrap(),
        ] {
            assert_eq!(loaded.count(), 19);
            assert!(loaded
                .search(&deleted_query, 10)
                .unwrap()
                .iter()
                .all(|result| result.id != "vec-3"));
        }
    }

    #[test]
    fn test_mmap_delete_can_save_back_to_same_directory() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mmap-save");
        let data = random_vectors(30, 4);

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            max_degree: 4,
            build_beam: 8,
            search_beam: 8,
            ..Default::default()
        });
        index.insert_batch(data).unwrap();
        index.build().unwrap();
        index.save(&path).unwrap();
        drop(index);

        let mut mmapped = DiskAnnIndex::load_mmap(&path).unwrap();
        assert!(mmapped.delete_deferred("vec-3").unwrap());
        mmapped.save(&path).unwrap();
        drop(mmapped);

        let loaded = DiskAnnIndex::load(&path).unwrap();
        assert_eq!(loaded.count(), 29);
        assert!(!loaded.id_reverse.contains_key("vec-3"));
    }

    #[test]
    #[ignore = "20k-vector recall regression; run explicitly for release validation"]
    fn test_delete_repair_recall_regression_20k() {
        use rand::prelude::*;
        use std::collections::HashSet;

        fn recall_at_10(
            index: &DiskAnnIndex,
            data: &[(String, Vec<f32>)],
            survivors: &HashSet<usize>,
            queries: &[usize],
        ) -> f64 {
            let mut matches = 0usize;
            for &query_idx in queries {
                let query = &data[query_idx].1;
                let mut exact: Vec<(usize, f32)> = survivors
                    .iter()
                    .map(|&idx| (idx, l2_squared(&data[idx].1, query)))
                    .collect();
                exact.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let truth: HashSet<&str> = exact
                    .iter()
                    .take(10)
                    .map(|(idx, _)| data[*idx].0.as_str())
                    .collect();
                matches += index
                    .search(query, 10)
                    .unwrap()
                    .iter()
                    .filter(|result| truth.contains(result.id.as_str()))
                    .count();
            }
            matches as f64 / (queries.len() * 10) as f64
        }

        let n = 20_000;
        let dim = 32;
        let mut rng = StdRng::seed_from_u64(0x679D_15CA);
        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|idx| {
                (
                    format!("v{idx}"),
                    (0..dim).map(|_| rng.gen::<f32>()).collect(),
                )
            })
            .collect();
        let config = DiskAnnConfig {
            dim,
            max_degree: 32,
            build_beam: 64,
            search_beam: 64,
            alpha: 1.2,
            ..Default::default()
        };

        let dir = tempdir().unwrap();
        let mut base = DiskAnnIndex::new(config.clone());
        base.insert_batch(data.clone()).unwrap();
        base.build().unwrap();
        base.save(dir.path()).unwrap();
        let mut repaired = DiskAnnIndex::load(dir.path()).unwrap();
        let mut deferred = DiskAnnIndex::load(dir.path()).unwrap();

        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        let deleted: HashSet<usize> = order[..n / 5].iter().copied().collect();
        let survivors: HashSet<usize> = (0..n).filter(|idx| !deleted.contains(idx)).collect();
        for &idx in &order[..n / 5] {
            repaired.delete(&format!("v{idx}")).unwrap();
            deferred.delete_deferred(&format!("v{idx}")).unwrap();
        }

        let survivor_data: Vec<(String, Vec<f32>)> =
            survivors.iter().map(|&idx| data[idx].clone()).collect();
        let mut fresh = DiskAnnIndex::new(config);
        fresh.insert_batch(survivor_data).unwrap();
        fresh.build().unwrap();
        let queries: Vec<usize> = order[n / 5..n / 5 + 100].to_vec();
        let repaired_recall = recall_at_10(&repaired, &data, &survivors, &queries);
        let deferred_recall = recall_at_10(&deferred, &data, &survivors, &queries);
        let fresh_recall = recall_at_10(&fresh, &data, &survivors, &queries);
        println!(
            "delete recall@10: deferred={deferred_recall:.4} repaired={repaired_recall:.4} fresh={fresh_recall:.4}"
        );
        assert!(
            (repaired_recall - fresh_recall).abs() <= 0.02,
            "repaired recall {repaired_recall:.4} differs from fresh recall {fresh_recall:.4} by more than 0.02"
        );
    }

    #[test]
    fn test_recall_at_10() {
        // Measure recall@10: what fraction of true top-10 neighbors does DiskANN find?
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);
        let n = 2000;
        let dim = 64;
        let k = 10;

        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 32,
            build_beam: 64,
            search_beam: 64,
            alpha: 1.2,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();
        index.build().unwrap();

        // Test 50 random queries
        let num_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let qi = rng.gen_range(0..n);
            let query = &data[qi].1;

            // Brute-force ground truth
            let mut brute: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i, crate::distance::l2_squared(v, query)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: std::collections::HashSet<String> =
                brute[..k].iter().map(|(i, _)| data[*i].0.clone()).collect();

            // DiskANN search
            let results = index.search(query, k).unwrap();
            let found: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            let recall = gt.intersection(&found).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Recall@{k} = {avg_recall:.3} (n={n}, dim={dim}, queries={num_queries})");
        assert!(
            avg_recall >= 0.85,
            "Recall@{k} = {avg_recall:.3}, expected >= 0.85"
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 16,
            ..Default::default()
        });

        // Wrong dimension on insert
        let result = index.insert("bad".to_string(), vec![1.0; 32]);
        assert!(result.is_err());

        // Wrong dimension on search
        index.insert("ok".to_string(), vec![1.0; 16]).unwrap();
        index.build().unwrap();
        let result = index.search(&[1.0; 32], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![1.0; 4]).unwrap();
        let result = index.insert("a".to_string(), vec![2.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_before_build_fails() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![1.0; 4]).unwrap();
        let result = index.search(&[1.0; 4], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_scale_5k() {
        // 5000 vectors, 128-dim — should build in under 5 seconds
        use rand::prelude::*;
        use std::time::Instant;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);

        let n = 5000;
        let dim = 128;
        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 48,
            build_beam: 96,
            search_beam: 48,
            alpha: 1.2,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();

        let t0 = Instant::now();
        index.build().unwrap();
        let build_ms = t0.elapsed().as_millis();
        println!("Build {n} vectors ({dim}d): {build_ms}ms");

        // Search latency
        let query = &data[0].1;
        let t0 = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let _ = index.search(query, 10).unwrap();
        }
        let search_us = t0.elapsed().as_micros() / iters;
        println!("Search latency (k=10): {search_us}µs avg over {iters} queries");

        assert!(
            search_us < 10_000,
            "Search took {search_us}µs, expected <10ms"
        );
    }
}
