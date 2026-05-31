//! IVF-PQ index with HAKES filter-refine search architecture.
//!
//! Train:  (1) k-means++ → nlist IVF centroids  (2) PQ k-means → M×ksub codebook
//! Insert: assign centroid → PQ-encode → store (id, codes, raw) in inverted list
//! Search: coarse centroid scan → nprobe lists → ADC filter → exact-L2 rerank
//!
//! References:
//! - Jégou et al., "Product Quantization for Nearest Neighbor Search", TPAMI 2011
//! - Faiss library (arXiv:2401.08281, 2024)
//! - HAKES VLDB 2025 (filter-refine split, 4-bit PQ filter + full-precision refine)

use crate::kmeans::{l2sq, n_nearest, nearest, train as kmeans_train};
use crate::pq::PqCodebook;

/// Configuration for the IVF-PQ index.
#[derive(Debug, Clone)]
pub struct IvfPqConfig {
    /// Number of Voronoi cells (inverted lists).
    pub nlist: usize,
    /// Number of PQ subspaces (M).
    pub m: usize,
    /// Codebook entries per subspace; max 256 for u8 codes.
    pub ksub: usize,
    /// Default number of lists probed at search time.
    pub nprobe: usize,
    /// Default candidate pool size for exact-L2 reranking.
    pub rerank_k: usize,
    /// k-means iteration limit for IVF and PQ training.
    pub max_iter: usize,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        IvfPqConfig {
            nlist: 64,
            m: 8,
            ksub: 256,
            nprobe: 4,
            rerank_k: 50,
            max_iter: 30,
        }
    }
}

/// One nearest-neighbor result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Sequential insertion id.
    pub id: usize,
    /// Exact squared-L2 distance after refine stage.
    pub distance: f32,
}

struct Entry {
    id: usize,
    codes: Vec<u8>,
    raw: Vec<f32>,
}

/// IVF-PQ index.
pub struct IvfPqIndex {
    config: IvfPqConfig,
    centroids: Vec<Vec<f32>>,
    codebook: Option<PqCodebook>,
    lists: Vec<Vec<Entry>>,
    ntotal: usize,
    dim: usize,
}

impl IvfPqIndex {
    /// Construct an untrained index.
    pub fn new(config: IvfPqConfig) -> Self {
        let nlist = config.nlist;
        IvfPqIndex {
            config,
            centroids: Vec::new(),
            codebook: None,
            lists: (0..nlist).map(|_| Vec::new()).collect(),
            ntotal: 0,
            dim: 0,
        }
    }

    /// Train IVF centroids and PQ codebook on representative vectors.
    /// Must be called before `add`.
    ///
    /// Two-phase training (FAISS IVF-PQ standard):
    /// 1. k-means++ on full vectors → IVF centroids
    /// 2. Compute residuals (v − centroid[assign(v)]), train PQ on residuals
    ///    so the codebook captures fine-grained within-cell structure.
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        assert!(!vectors.is_empty(), "training set must not be empty");
        self.dim = vectors[0].len();
        let k = self.config.nlist.min(vectors.len());

        self.centroids = kmeans_train(vectors, k, self.config.max_iter, 0x1337_CAFE);

        // Residuals: v - centroid[nearest_cell(v)]
        let residuals: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                let cell = nearest(v, &self.centroids);
                let c = &self.centroids[cell];
                v.iter().zip(c.iter()).map(|(vi, ci)| vi - ci).collect()
            })
            .collect();

        self.codebook = Some(PqCodebook::train(
            &residuals,
            self.config.m,
            self.config.ksub,
            self.config.max_iter,
        ));
    }

    /// Add one vector with an explicit id.
    /// Encodes the residual (v − IVF centroid) with PQ for accurate ADC scoring.
    pub fn add_with_id(&mut self, id: usize, v: &[f32]) {
        let cell = nearest(v, &self.centroids);
        let residual: Vec<f32> = v
            .iter()
            .zip(self.centroids[cell].iter())
            .map(|(vi, ci)| vi - ci)
            .collect();
        let cb = self.codebook.as_ref().expect("call train() before add()");
        let codes = cb.encode(&residual);
        self.lists[cell].push(Entry { id, codes, raw: v.to_vec() });
        self.ntotal += 1;
    }

    /// Add a slice of vectors; ids are assigned sequentially from current `len()`.
    pub fn add(&mut self, vectors: &[Vec<f32>]) {
        for v in vectors {
            let id = self.ntotal;
            self.add_with_id(id, v);
        }
    }

    /// HAKES filter-refine search.
    ///
    /// * `nprobe = 0` → use `config.nprobe`
    /// * `rerank_k = 0` → use `config.rerank_k`
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_k: usize,
    ) -> Vec<SearchResult> {
        let nprobe = if nprobe == 0 { self.config.nprobe } else { nprobe };
        let rerank_k = if rerank_k == 0 { self.config.rerank_k } else { rerank_k };
        let cb = self.codebook.as_ref().expect("call train() before search()");

        // Stage 1 – coarse scan: find nprobe nearest IVF cells
        let probed = n_nearest(query, &self.centroids, nprobe.min(self.centroids.len()));

        // Stage 2 – ADC filter with per-cell residual query (FAISS IVF-PQ standard).
        // For each probed cell c, the residual query is q − centroid[c].
        // This matches how codes were stored during add(), giving accurate ADC scoring.
        let mut candidates: Vec<(usize, f32, &[f32])> = Vec::new();
        for &cell in &probed {
            let qr: Vec<f32> = query
                .iter()
                .zip(self.centroids[cell].iter())
                .map(|(qi, ci)| qi - ci)
                .collect();
            let lut = cb.build_lut(&qr);
            for entry in &self.lists[cell] {
                candidates.push((entry.id, lut.score(&entry.codes), entry.raw.as_slice()));
            }
        }

        // Keep top-rerank_k by approximate distance
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.truncate(rerank_k);

        // Stage 3 – exact-L2 refine on stored raw vectors
        let mut results: Vec<SearchResult> = candidates
            .iter()
            .map(|c| SearchResult { id: c.0, distance: l2sq(query, c.2) })
            .collect();

        results.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        results.truncate(k);
        results
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.ntotal
    }

    /// True if no vectors have been added.
    pub fn is_empty(&self) -> bool {
        self.ntotal == 0
    }

    /// Approximate index memory in bytes:
    /// IVF centroids + PQ codebook + PQ codes + raw vectors.
    pub fn memory_bytes(&self) -> usize {
        let centroid_b = self.centroids.len() * self.dim * 4;
        let codebook_b = self.codebook.as_ref().map(|c| c.codebook_bytes()).unwrap_or(0);
        let code_b = self.ntotal * self.config.m;
        let raw_b = self.ntotal * self.dim * 4;
        centroid_b + codebook_b + code_b + raw_b
    }

    /// Memory in bytes for the compressed index only (no raw vectors).
    /// In production the raw vectors can be stored on disk and fetched only for rerank.
    pub fn compressed_memory_bytes(&self) -> usize {
        let centroid_b = self.centroids.len() * self.dim * 4;
        let codebook_b = self.codebook.as_ref().map(|c| c.codebook_bytes()).unwrap_or(0);
        let code_b = self.ntotal * self.config.m;
        centroid_b + codebook_b + code_b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::l2sq;

    fn two_cluster_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let half = n / 2;
        let mut vecs: Vec<Vec<f32>> = (0..half)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.01).collect())
            .collect();
        let far: Vec<Vec<f32>> = (0..half)
            .map(|i| (0..dim).map(|d| 100.0 + (i * dim + d) as f32 * 0.01).collect())
            .collect();
        vecs.extend(far);
        vecs
    }

    #[test]
    fn smoke_build_and_search() {
        let data = two_cluster_data(200, 8);
        let mut idx = IvfPqIndex::new(IvfPqConfig {
            nlist: 4,
            m: 2,
            ksub: 16,
            nprobe: 2,
            rerank_k: 20,
            max_iter: 20,
        });
        idx.train(&data);
        idx.add(&data);
        assert_eq!(idx.len(), 200);

        let query: Vec<f32> = (0..8_usize).map(|d| 100.0 + d as f32 * 0.01).collect();
        let results = idx.search(&query, 5, 2, 20);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        for r in &results {
            assert!(r.id >= 100, "expected far-cluster id (≥100), got {}", r.id);
        }
    }

    #[test]
    fn full_scan_achieves_exact_recall() {
        // With nprobe=nlist and rerank_k=N, search is equivalent to brute force
        let data = two_cluster_data(400, 8);
        let mut idx = IvfPqIndex::new(IvfPqConfig {
            nlist: 4,
            m: 2,
            ksub: 16,
            nprobe: 4,
            rerank_k: 400,
            max_iter: 30,
        });
        idx.train(&data);
        idx.add(&data);

        let query: Vec<f32> = (0..8_usize).map(|d| 100.5 + d as f32 * 0.005).collect();
        let results = idx.search(&query, 10, 4, 400);

        let mut bf: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2sq(&query, v)))
            .collect();
        bf.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let gt: std::collections::HashSet<usize> = bf.iter().take(10).map(|(i, _)| *i).collect();
        let found: std::collections::HashSet<usize> = results.iter().map(|r| r.id).collect();
        let hits = gt.intersection(&found).count();
        assert_eq!(hits, 10, "full scan must achieve 100% recall@10, got {hits}/10");
    }

    #[test]
    fn memory_estimate_sane() {
        let data = two_cluster_data(100, 8);
        let mut idx = IvfPqIndex::new(IvfPqConfig {
            nlist: 2,
            m: 2,
            ksub: 8,
            nprobe: 2,
            rerank_k: 10,
            max_iter: 10,
        });
        idx.train(&data);
        idx.add(&data);
        // Full mem >= compressed mem
        assert!(idx.memory_bytes() >= idx.compressed_memory_bytes());
        // Compressed mem > 0
        assert!(idx.compressed_memory_bytes() > 0);
    }
}
