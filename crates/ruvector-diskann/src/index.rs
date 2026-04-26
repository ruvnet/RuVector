//! DiskANN index — ties together Vamana graph, PQ, and mmap persistence

use crate::distance::{l2_squared, FlatVectors, VisitedSet};
use crate::error::{DiskAnnError, Result};
use crate::graph::VamanaGraph;
#[cfg(feature = "rabitq")]
use crate::quantize::RabitqQuantizer;
use crate::quantize::{ProductQuantizer, Quantizer};
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Which quantizer backend a [`DiskAnnIndex`] should use during search.
///
/// We use an enum rather than a generic type parameter on the index for two
/// reasons:
///
///   1. The [`Quantizer`] trait has an associated `Query` type, so it isn't
///      object-safe — `Box<dyn Quantizer>` won't compile. We'd have to add
///      another erasure layer to make it work.
///   2. The NAPI binding (`ruvector-diskann-node`) holds `DiskAnnIndex`
///      directly with no type parameter. Generic-ifying the index would
///      cascade through every binding crate.
///
/// An internal enum keeps the public API stable while letting the search
/// path monomorphise on the concrete quantizer at the match arm. The closure
/// passed to [`crate::graph::VamanaGraph::greedy_search_with_codes`] is
/// inlined per arm so the hot loop stays branch-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizerKind {
    /// No quantizer — `search()` falls back to the legacy f32 flat-vector
    /// hot path. This is the default for back-compat with callers who
    /// haven't opted in.
    None,
    /// Product Quantization (M subspaces, 256 centroids each). Activated
    /// when `DiskAnnConfig::pq_subspaces > 0` for back-compat.
    Pq,
    /// 1-bit RaBitQ rotation-quantized codes. Requires the `rabitq` feature.
    #[cfg(feature = "rabitq")]
    Rabitq,
}

/// Type-erased quantizer state held by the index. Each variant owns the
/// concrete impl; the [`Quantizer`] trait methods are dispatched per arm in
/// [`DiskAnnIndex::search`] so the hot-loop closure stays monomorphic.
enum QuantizerBackend {
    None,
    Pq(ProductQuantizer),
    #[cfg(feature = "rabitq")]
    Rabitq(RabitqQuantizer),
}

impl QuantizerBackend {
    #[inline]
    fn is_active(&self) -> bool {
        !matches!(self, QuantizerBackend::None)
    }
}

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

    // ── New knobs introduced by the trait-driven search path. ────────────
    //
    // These are `pub` for symmetry with the rest of the struct, but new
    // callers should prefer the builder methods below
    // ([`Self::with_quantizer_kind`] / [`Self::with_rerank_factor`] /
    // [`Self::with_originals_in_memory`]) so future additions don't
    // require source changes at every construction site. The legacy
    // `pq_subspaces` field still auto-selects [`QuantizerKind::Pq`] for
    // back-compat.
    /// Which quantizer drives the search path. `None` = legacy flat-f32.
    pub quantizer_kind: QuantizerKind,
    /// RaBitQ rotation seed. ADR-154 mandates determinism on this path.
    #[cfg(feature = "rabitq")]
    pub rabitq_seed: u64,
    /// Multiplier on `k` for the rerank pool. The graph traversal still
    /// returns up to `search_beam` candidates; this knob picks how many
    /// of those candidates we re-score with the exact f32 distance
    /// before returning the final top-k.
    pub rerank_factor: usize,
    /// Whether to keep the original f32 vectors resident in DRAM after
    /// build. The `false` path is currently rejected at `build()` time
    /// pending the disk-backed rerank follow-up.
    pub keep_originals_in_memory: bool,
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
            quantizer_kind: QuantizerKind::None,
            #[cfg(feature = "rabitq")]
            rabitq_seed: 0xDEAD_BEEF_CAFE_F00D,
            rerank_factor: 4,
            keep_originals_in_memory: true,
        }
    }
}

impl DiskAnnConfig {
    /// Builder-style override for the rerank pool multiplier.
    pub fn with_rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor.max(1);
        self
    }

    /// Builder-style override for whether to keep f32 originals in DRAM.
    /// Today the rerank path *requires* originals (we don't read them from
    /// disk yet), so `false` is rejected at `build()` time with an error
    /// rather than silently degrading recall. The plumbing is in place so a
    /// follow-up PR can wire mmap-backed reranking without another API
    /// break.
    pub fn with_originals_in_memory(mut self, keep: bool) -> Self {
        self.keep_originals_in_memory = keep;
        self
    }

    /// Builder-style override for the quantizer backend. `Pq` is also
    /// auto-selected when `pq_subspaces > 0` for back-compat.
    pub fn with_quantizer_kind(mut self, kind: QuantizerKind) -> Self {
        self.quantizer_kind = kind;
        self
    }

    /// Builder-style override for the RaBitQ rotation seed.
    #[cfg(feature = "rabitq")]
    pub fn with_rabitq_seed(mut self, seed: u64) -> Self {
        self.rabitq_seed = seed;
        self
    }

    /// Read accessor for the configured quantizer kind.
    pub fn quantizer_kind(&self) -> QuantizerKind {
        self.quantizer_kind
    }

    /// Read accessor for the rerank multiplier.
    pub fn rerank_factor(&self) -> usize {
        self.rerank_factor
    }
}

/// DiskANN index with Vamana graph + optional quantized codes + mmap
/// persistence.
///
/// As of this PR (closing the architectural gap surfaced in PR #383), the
/// graph traversal can consult **either** the flat f32 vectors (legacy path
/// when `quantizer_kind == None`) **or** the quantized codes via the
/// [`Quantizer`] trait. The exact-L2² rerank still uses the original f32
/// vectors — that's intentional, the codes are an approximation.
pub struct DiskAnnIndex {
    config: DiskAnnConfig,
    /// Flat contiguous vector storage (cache-friendly).
    ///
    /// Held in DRAM today even when a quantizer is active, so the rerank
    /// pass can compute exact distances on the candidate pool. The
    /// `keep_originals_in_memory(false)` knob is wired into the config but
    /// rejected at `build()` time pending the disk-backed rerank follow-up.
    vectors: FlatVectors,
    /// ID mapping: internal index -> external string ID
    id_map: Vec<String>,
    /// Reverse mapping: external ID -> internal index
    id_reverse: HashMap<String, u32>,
    /// Vamana graph
    graph: Option<VamanaGraph>,
    /// Active quantizer backend. `None` for the legacy f32 path.
    quantizer: QuantizerBackend,
    /// Quantized codes for all vectors (one entry per vector, in insertion
    /// order). Empty when `quantizer == None`. Each entry has length
    /// `quantizer.code_bytes()`. **This is the field that PR #383 left as
    /// dead storage**; the new `search()` path consumes it via the trait.
    codes: Vec<Vec<u8>>,
    /// Whether index has been built
    built: bool,
    /// Reusable visited set for search (avoids per-query allocation)
    visited: Option<VisitedSet>,
    /// Memory-mapped vector data (for large datasets)
    mmap: Option<Mmap>,
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
            graph: None,
            quantizer: QuantizerBackend::None,
            codes: Vec::new(),
            built: false,
            visited: None,
            mmap: None,
        }
    }

    /// Insert a vector with a string ID
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
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

        // Disk-backed rerank isn't wired yet — bail early rather than
        // silently dropping originals and degrading recall to garbage.
        if !self.config.keep_originals_in_memory {
            return Err(DiskAnnError::InvalidConfig(
                "keep_originals_in_memory=false requires the disk-backed rerank path; \
                 not yet implemented — keep originals in DRAM for this PR"
                    .into(),
            ));
        }

        // Resolve quantizer kind: explicit setting wins, else fall back to
        // the legacy `pq_subspaces > 0` heuristic so old callers keep
        // working without any source change.
        let kind = match self.config.quantizer_kind {
            QuantizerKind::None if self.config.pq_subspaces > 0 => QuantizerKind::Pq,
            other => other,
        };

        match kind {
            QuantizerKind::None => {
                self.quantizer = QuantizerBackend::None;
                self.codes.clear();
            }
            QuantizerKind::Pq => {
                let m = self.config.pq_subspaces;
                if m == 0 {
                    return Err(DiskAnnError::InvalidConfig(
                        "QuantizerKind::Pq requires pq_subspaces > 0".into(),
                    ));
                }
                // Collect vectors for PQ training
                let vecs: Vec<Vec<f32>> = (0..n).map(|i| self.vectors.get(i).to_vec()).collect();
                let mut pq = ProductQuantizer::new(self.config.dim, m)?;
                Quantizer::train(&mut pq, &vecs, self.config.pq_iterations)?;

                self.codes = vecs
                    .iter()
                    .map(|v| Quantizer::encode(&pq, v))
                    .collect::<Result<Vec<_>>>()?;

                self.quantizer = QuantizerBackend::Pq(pq);
            }
            #[cfg(feature = "rabitq")]
            QuantizerKind::Rabitq => {
                let vecs: Vec<Vec<f32>> = (0..n).map(|i| self.vectors.get(i).to_vec()).collect();
                let mut rb = RabitqQuantizer::new(self.config.dim, self.config.rabitq_seed);
                Quantizer::train(&mut rb, &vecs, 0)?;

                self.codes = vecs
                    .iter()
                    .map(|v| Quantizer::encode(&rb, v))
                    .collect::<Result<Vec<_>>>()?;

                self.quantizer = QuantizerBackend::Rabitq(rb);
            }
        }

        // Build Vamana graph on flat storage. Graph construction still
        // walks f32 originals — pruning needs exact distances or it loses
        // the α-robust property. The traversal-time use of codes is a
        // *query-time* optimisation; it doesn't affect graph quality.
        let mut graph = VamanaGraph::new(
            n,
            self.config.max_degree,
            self.config.build_beam,
            self.config.alpha,
        );
        graph.build(&self.vectors)?;
        self.graph = Some(graph);

        // Pre-allocate visited set for search
        self.visited = Some(VisitedSet::new(n));
        self.built = true;

        if let Some(ref path) = self.config.storage_path {
            self.save(path)?;
        }

        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// When a quantizer is active, graph traversal computes per-node
    /// distances by looking up the **codes** (PQ asymmetric LUT or RaBitQ
    /// XNOR-popcount), not the original f32 vectors. Top candidates are
    /// then exact-reranked against the f32 originals — that's the
    /// approximate→exact two-stage pattern from the DiskANN paper, with
    /// the trait abstraction making the approximate side pluggable.
    ///
    /// When `quantizer_kind == None` the old f32 hot path is preserved
    /// verbatim (modulo a single new helper call) so existing callers see
    /// zero recall change.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if !self.built {
            return Err(DiskAnnError::NotBuilt);
        }
        if query.len() != self.config.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dim,
                actual: query.len(),
            });
        }

        let graph = self.graph.as_ref().unwrap();
        let beam = self.config.search_beam.max(k);
        let n = self.vectors.len();

        // Phase 1: graph traversal. Distance source depends on the
        // configured quantizer. Each match arm calls
        // `greedy_search_with_codes` with a closure that's monomorphic to
        // the concrete quantizer — the trait dispatch happens once outside
        // the hot loop, not per node.
        let candidates: Vec<u32> = match &self.quantizer {
            QuantizerBackend::None => {
                // Legacy path — exact f32 distance during traversal. Keeps
                // old benchmarks bit-stable.
                let (cands, _) = graph.greedy_search(&self.vectors, query, beam);
                cands
            }
            QuantizerBackend::Pq(pq) => {
                let prep = pq.prepare_query(query)?;
                let codes = &self.codes;
                let (cands, _) = graph.greedy_search_with_codes(n, beam, |id| {
                    Quantizer::distance(pq, &prep, &codes[id as usize])
                });
                cands
            }
            #[cfg(feature = "rabitq")]
            QuantizerBackend::Rabitq(rb) => {
                let prep = rb.prepare_query(query)?;
                let codes = &self.codes;
                let (cands, _) = graph.greedy_search_with_codes(n, beam, |id| {
                    Quantizer::distance(rb, &prep, &codes[id as usize])
                });
                cands
            }
        };

        // Phase 2: rerank. Quantized distances are biased / noisy, so we
        // exact-rescore at most `rerank_factor * k` candidates against the
        // original f32 vectors. When no quantizer is active the candidates
        // are already exact-distance ordered, but we still re-sort to keep
        // the codepath uniform.
        let rerank_pool = (self.config.rerank_factor.max(1) * k).min(candidates.len());
        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(rerank_pool)
            .map(|id| (id, l2_squared(self.vectors.get(id as usize), query)))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(id, dist)| SearchResult {
                id: self.id_map[id as usize].clone(),
                distance: dist,
            })
            .collect())
    }

    /// Get the number of vectors in the index
    pub fn count(&self) -> usize {
        self.vectors.len()
    }

    /// Delete a vector by ID (marks as deleted, doesn't rebuild graph)
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        if let Some(&idx) = self.id_reverse.get(id) {
            self.vectors.zero_out(idx as usize);
            self.id_reverse.remove(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Save index to disk
    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;

        // Save vectors as flat binary (already contiguous — mmap-friendly)
        let vec_path = dir.join("vectors.bin");
        let mut f = BufWriter::new(File::create(&vec_path)?);
        let n = self.vectors.len() as u64;
        let dim = self.config.dim as u64;
        f.write_all(&n.to_le_bytes())?;
        f.write_all(&dim.to_le_bytes())?;
        // Write flat slab directly — zero copy
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.vectors.data.as_ptr() as *const u8,
                self.vectors.data.len() * 4,
            )
        };
        f.write_all(byte_slice)?;
        f.flush()?;

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
        }
        f.flush()?;

        // Save ID map
        let ids_path = dir.join("ids.json");
        let ids_json = serde_json::to_string(&self.id_map)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
        fs::write(&ids_path, ids_json)?;

        // Save PQ if present. RaBitQ persistence is intentionally a
        // follow-up: the rotation matrix lives in `ruvector-rabitq` and
        // doesn't yet expose a stable on-disk format. For now we only
        // persist PQ-backed indexes; a RaBitQ-backed save returns Ok and
        // skips the codes — `load()` will rebuild from f32 originals.
        match &self.quantizer {
            QuantizerBackend::None => {}
            QuantizerBackend::Pq(pq) => {
                let pq_path = dir.join("pq.bin");
                let pq_bytes = bincode::encode_to_vec(pq, bincode::config::standard())
                    .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
                fs::write(&pq_path, pq_bytes)?;

                // Save PQ codes
                let codes_path = dir.join("pq_codes.bin");
                let mut f = BufWriter::new(File::create(&codes_path)?);
                for codes in &self.codes {
                    f.write_all(codes)?;
                }
                f.flush()?;
            }
            #[cfg(feature = "rabitq")]
            QuantizerBackend::Rabitq(_) => {
                // Disk format for RaBitQ rotation matrices is a follow-up.
                // The graph + originals are still saved above so the index
                // can be reloaded with `quantizer_kind = None` and rebuilt
                // by the caller if needed.
            }
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
            "count": self.vectors.len(),
            "built": self.built,
        });
        fs::write(
            &config_path,
            serde_json::to_string_pretty(&config_json).unwrap(),
        )?;

        Ok(())
    }

    /// Load index from disk with memory-mapped vectors
    pub fn load(dir: &Path) -> Result<Self> {
        // Load config
        let config_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)
                .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

        let dim = config_json["dim"].as_u64().unwrap() as usize;
        let max_degree = config_json["max_degree"].as_u64().unwrap() as usize;
        let build_beam = config_json["build_beam"].as_u64().unwrap() as usize;
        let search_beam = config_json["search_beam"].as_u64().unwrap() as usize;
        let alpha = config_json["alpha"].as_f64().unwrap() as f32;
        let pq_subspaces = config_json["pq_subspaces"].as_u64().unwrap_or(0) as usize;

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

        // Load vectors via mmap
        let vec_file = File::open(dir.join("vectors.bin"))?;
        let mmap = unsafe { MmapOptions::new().map(&vec_file)? };

        let n = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let file_dim = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        assert_eq!(file_dim, dim);

        // Load vectors directly into flat slab from mmap
        let data_start = 16;
        let total_floats = n * dim;
        let mut flat_data = Vec::with_capacity(total_floats);
        let byte_slice = &mmap[data_start..data_start + total_floats * 4];
        // Safe: f32 from le bytes
        for chunk in byte_slice.chunks_exact(4) {
            flat_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let vectors = FlatVectors {
            data: flat_data,
            dim,
            count: n,
        };

        // Load IDs
        let ids_json = fs::read_to_string(dir.join("ids.json"))?;
        let id_map: Vec<String> = serde_json::from_str(&ids_json)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

        let mut id_reverse = HashMap::new();
        for (i, id) in id_map.iter().enumerate() {
            id_reverse.insert(id.clone(), i as u32);
        }

        // Load graph
        let graph_bytes = fs::read(dir.join("graph.bin"))?;
        let medoid = u64::from_le_bytes(graph_bytes[0..8].try_into().unwrap()) as u32;
        let graph_n = u64::from_le_bytes(graph_bytes[8..16].try_into().unwrap()) as usize;

        let mut neighbors = Vec::with_capacity(graph_n);
        let mut offset = 16;
        for _ in 0..graph_n {
            let deg =
                u32::from_le_bytes(graph_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let mut nbrs = Vec::with_capacity(deg);
            for _ in 0..deg {
                let nbr = u32::from_le_bytes(graph_bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                nbrs.push(nbr);
            }
            neighbors.push(nbrs);
        }

        let graph = VamanaGraph {
            neighbors,
            medoid,
            max_degree,
            build_beam,
            alpha,
        };

        // Load PQ if present. RaBitQ persistence is a follow-up — see the
        // matching note in `save()`.
        let pq_path = dir.join("pq.bin");
        let (quantizer, codes) = if pq_path.exists() {
            let pq_bytes = fs::read(&pq_path)?;
            let (pq, _): (ProductQuantizer, usize) =
                bincode::decode_from_slice(&pq_bytes, bincode::config::standard())
                    .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

            let codes_bytes = fs::read(dir.join("pq_codes.bin"))?;
            let m = pq.m;
            let mut codes = Vec::with_capacity(n);
            for i in 0..n {
                codes.push(codes_bytes[i * m..(i + 1) * m].to_vec());
            }
            (QuantizerBackend::Pq(pq), codes)
        } else {
            (QuantizerBackend::None, Vec::new())
        };

        // Mirror the saved quantizer into the runtime config. The legacy
        // `pq_subspaces` field is already populated from JSON above; the
        // explicit `quantizer_kind` is set so callers can introspect it.
        let mut config = config;
        config.quantizer_kind = match &quantizer {
            QuantizerBackend::None => QuantizerKind::None,
            QuantizerBackend::Pq(_) => QuantizerKind::Pq,
            #[cfg(feature = "rabitq")]
            QuantizerBackend::Rabitq(_) => QuantizerKind::Rabitq,
        };

        Ok(Self {
            config,
            vectors,
            id_map,
            id_reverse,
            graph: Some(graph),
            quantizer,
            codes,
            built: true,
            visited: Some(VisitedSet::new(n)),
            mmap: Some(mmap),
        })
    }

    /// Number of bytes the in-memory codes slab consumes. Useful to verify
    /// that a quantizer-backed index is actually compressing memory and not
    /// just disk. Returns 0 when no quantizer is active.
    pub fn codes_memory_bytes(&self) -> usize {
        self.codes.iter().map(|c| c.len()).sum()
    }

    /// Number of bytes the f32 originals consume in DRAM. Pair with
    /// [`Self::codes_memory_bytes`] to compute the compression ratio.
    pub fn originals_memory_bytes(&self) -> usize {
        self.vectors.data.len() * std::mem::size_of::<f32>()
    }

    /// Which quantizer this index is using.
    pub fn quantizer_kind(&self) -> QuantizerKind {
        match &self.quantizer {
            QuantizerBackend::None => QuantizerKind::None,
            QuantizerBackend::Pq(_) => QuantizerKind::Pq,
            #[cfg(feature = "rabitq")]
            QuantizerBackend::Rabitq(_) => QuantizerKind::Rabitq,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn random_vectors(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
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

    /// Regression guard for the trait-driven PQ path.
    ///
    /// Pre-PR: `pq_subspaces > 0` trained PQ, encoded codes, and then…
    /// ignored them. The graph traversal walked f32 originals and recall
    /// equalled the no-PQ baseline.
    ///
    /// Post-PR: `pq_subspaces > 0` activates [`QuantizerKind::Pq`] and the
    /// graph traversal consults PQ codes via the [`Quantizer`] trait. PQ
    /// recall@10 should be **at least as good as before** (and in practice
    /// slightly better, because the rerank pool is now `rerank_factor * k`
    /// instead of the full beam).
    ///
    /// We compare the no-quantizer path (`pq_subspaces = 0`) against the
    /// PQ path on the same dataset + queries. The PQ path is allowed to
    /// trail the f32 baseline by a small margin (PQ's 1-byte-per-subspace
    /// codes are intrinsically lossy), but it must clear the same 0.85
    /// floor as `test_recall_at_10`.
    #[test]
    fn test_pq_recall_no_regression_post_trait_refactor() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 1_000;
        let dim = 64;
        let k = 10;
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        // Baseline: no quantizer (legacy f32 hot path).
        let mut idx_flat = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 32,
            build_beam: 96,
            search_beam: 96,
            alpha: 1.2,
            ..Default::default()
        });
        idx_flat.insert_batch(data.clone()).unwrap();
        idx_flat.build().unwrap();

        // Trait-driven PQ path. Same beam settings — only the traversal
        // distance source changes.
        let mut idx_pq = DiskAnnIndex::new(
            DiskAnnConfig {
                dim,
                max_degree: 32,
                build_beam: 96,
                search_beam: 96,
                alpha: 1.2,
                pq_subspaces: 8,
                pq_iterations: 8,
                ..Default::default()
            }
            .with_rerank_factor(4),
        );
        idx_pq.insert_batch(data.clone()).unwrap();
        idx_pq.build().unwrap();
        assert_eq!(idx_pq.quantizer_kind(), QuantizerKind::Pq);
        assert!(
            idx_pq.codes_memory_bytes() > 0,
            "PQ codes slab is empty — codes aren't actually stored"
        );

        // 30 random queries, recall@10 vs brute-force.
        let num_q = 30;
        let mut flat_recall = 0.0f64;
        let mut pq_recall = 0.0f64;
        for _ in 0..num_q {
            let qi = rng.gen_range(0..n);
            let query = &data[qi].1;

            let mut brute: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i, crate::distance::l2_squared(v, query)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: std::collections::HashSet<String> =
                brute[..k].iter().map(|(i, _)| data[*i].0.clone()).collect();

            let flat_found: std::collections::HashSet<String> = idx_flat
                .search(query, k)
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect();
            let pq_found: std::collections::HashSet<String> = idx_pq
                .search(query, k)
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect();

            flat_recall += gt.intersection(&flat_found).count() as f64 / k as f64;
            pq_recall += gt.intersection(&pq_found).count() as f64 / k as f64;
        }
        flat_recall /= num_q as f64;
        pq_recall /= num_q as f64;
        eprintln!(
            "[regression] flat recall@{k} = {flat_recall:.3}, PQ recall@{k} = {pq_recall:.3}"
        );

        // PQ codes are an approximation, so a small recall drop is
        // acceptable — but it must not collapse and must clear the same
        // 0.85 floor as the no-PQ path. This is the regression guard PR
        // #383 should have shipped with.
        assert!(
            pq_recall >= 0.85,
            "PQ recall@{k} = {pq_recall:.3} < 0.85 (flat path measures {flat_recall:.3})"
        );
    }

    #[test]
    fn test_recall_at_10() {
        // Measure recall@10: what fraction of true top-10 neighbors does DiskANN find?
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
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
        let mut rng = rand::thread_rng();

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
