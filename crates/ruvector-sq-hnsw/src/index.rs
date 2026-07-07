//! Concrete ANN index variants built on top of `HnswGraph`.
//!
//! All three implement `AnnIndex` with identical insert/search interfaces;
//! the difference is what is stored and how distances are computed during
//! HNSW graph traversal.

use crate::hnsw::HnswGraph;
use crate::quantizer::ScalarQuantizer;
use crate::{l2_sq, AnnIndex, HnswConfig, SearchResult};

// ---------------------------------------------------------------------------
// Variant 1: full float32 — baseline
// ---------------------------------------------------------------------------

/// Standard HNSW with float32 vectors.  Highest recall, largest memory.
pub struct F32Index {
    graph: HnswGraph,
    data: Vec<Vec<f32>>,
    config: HnswConfig,
    dims: usize,
}

impl F32Index {
    pub fn new(config: HnswConfig, dims: usize) -> Self {
        Self {
            graph: HnswGraph::new(config.m, config.ef_construction),
            data: Vec::new(),
            config,
            dims,
        }
    }
}

impl AnnIndex for F32Index {
    fn add(&mut self, _id: usize, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims);
        let new_idx = self.data.len();
        self.data.push(vector);
        let data = &self.data;
        self.graph
            .insert_node(new_idx, |i, j| l2_sq(&data[i], &data[j]));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let data = &self.data;
        self.graph
            .search(self.config.ef_search, k, |j| l2_sq(query, &data[j]))
            .into_iter()
            .map(|(d, id)| SearchResult { id, distance: d })
            .collect()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn bytes_per_vector(&self) -> usize {
        self.dims * 4 // f32
    }
}

// ---------------------------------------------------------------------------
// Variant 2: SQ8 — int8 throughout
// ---------------------------------------------------------------------------

/// HNSW using int8 scalar-quantized vectors for both storage and traversal.
///
/// Memory: ~4× less than f32.  Speed: ~2× faster distance computation.
/// Recall: typically 1-5% below f32 at the same EF.
///
/// The quantizer is calibrated on the first `calibration_budget` vectors;
/// subsequent vectors use the frozen calibration.
pub struct Sq8Index {
    graph: HnswGraph,
    codes: Vec<Vec<i8>>,
    quant: Option<ScalarQuantizer>,
    /// Buffer for pre-calibration vectors
    calib_buf: Vec<Vec<f32>>,
    calib_budget: usize,
    config: HnswConfig,
    dims: usize,
}

impl Sq8Index {
    /// `calibration_budget`: how many vectors to collect before calibrating.
    /// Use at least a few hundred for stable per-dimension statistics.
    pub fn new(config: HnswConfig, dims: usize, calibration_budget: usize) -> Self {
        Self {
            graph: HnswGraph::new(config.m, config.ef_construction),
            codes: Vec::new(),
            quant: None,
            calib_buf: Vec::new(),
            calib_budget: calibration_budget,
            config,
            dims,
        }
    }

    /// Force calibration now (useful when you know you have enough data).
    pub fn calibrate_now(&mut self) {
        if !self.calib_buf.is_empty() {
            let quant = ScalarQuantizer::calibrate(&self.calib_buf);
            // Encode buffered vectors and flush them into the graph
            let to_insert: Vec<Vec<f32>> = std::mem::take(&mut self.calib_buf);
            self.quant = Some(quant);
            for v in to_insert {
                self.add(0 /* id unused */, v);
            }
        }
    }

    fn add_encoded(&mut self, code: Vec<i8>) {
        let new_idx = self.codes.len();
        self.codes.push(code);
        let codes = &self.codes;
        self.graph.insert_node(new_idx, |i, j| {
            ScalarQuantizer::sq8_l2_sq(&codes[i], &codes[j]) as f32
        });
    }
}

impl AnnIndex for Sq8Index {
    fn add(&mut self, _id: usize, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims);

        if self.quant.is_none() {
            self.calib_buf.push(vector.clone());
            if self.calib_buf.len() >= self.calib_budget {
                let quant = ScalarQuantizer::calibrate(&self.calib_buf);
                let to_insert: Vec<Vec<f32>> = std::mem::take(&mut self.calib_buf);
                self.quant = Some(quant);
                for v in to_insert {
                    let code = self.quant.as_ref().unwrap().encode(&v);
                    self.add_encoded(code);
                }
                // Now insert the current vector too
                let code = self.quant.as_ref().unwrap().encode(&vector);
                self.add_encoded(code);
            }
            // Not calibrated yet — buffered above
            return;
        }

        let code = self.quant.as_ref().unwrap().encode(&vector);
        self.add_encoded(code);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let quant = match &self.quant {
            Some(q) => q,
            None => return vec![], // not yet calibrated
        };
        let q_code = quant.encode(query);
        let codes = &self.codes;
        self.graph
            .search(self.config.ef_search, k, |j| {
                ScalarQuantizer::sq8_l2_sq(&q_code, &codes[j]) as f32
            })
            .into_iter()
            .map(|(d, id)| SearchResult { id, distance: d })
            .collect()
    }

    fn len(&self) -> usize {
        self.codes.len() + self.calib_buf.len()
    }

    fn bytes_per_vector(&self) -> usize {
        self.dims // i8
    }
}

// ---------------------------------------------------------------------------
// Variant 3: SQ8 + float32 rerank
// ---------------------------------------------------------------------------

/// HNSW with int8 graph traversal and float32 exact reranking of the
/// top-`overquery_factor * k` candidates.
///
/// Best of both worlds: fast graph traversal, accurate final results.
/// Memory cost: int8 codes + float32 originals (~5× vs baseline).
pub struct Sq8RerankIndex {
    graph: HnswGraph,
    codes: Vec<Vec<i8>>,
    originals: Vec<Vec<f32>>,
    quant: Option<ScalarQuantizer>,
    calib_buf: Vec<Vec<f32>>,
    calib_budget: usize,
    config: HnswConfig,
    dims: usize,
    /// How many extra candidates to fetch before reranking.
    pub overquery_factor: usize,
}

impl Sq8RerankIndex {
    pub fn new(config: HnswConfig, dims: usize, calibration_budget: usize) -> Self {
        Self {
            graph: HnswGraph::new(config.m, config.ef_construction),
            codes: Vec::new(),
            originals: Vec::new(),
            quant: None,
            calib_buf: Vec::new(),
            calib_budget: calibration_budget,
            config,
            dims,
            overquery_factor: 3,
        }
    }

    fn add_encoded(&mut self, code: Vec<i8>, original: Vec<f32>) {
        let new_idx = self.codes.len();
        self.codes.push(code);
        self.originals.push(original);
        let codes = &self.codes;
        self.graph.insert_node(new_idx, |i, j| {
            ScalarQuantizer::sq8_l2_sq(&codes[i], &codes[j]) as f32
        });
    }
}

impl AnnIndex for Sq8RerankIndex {
    fn add(&mut self, _id: usize, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims);

        if self.quant.is_none() {
            self.calib_buf.push(vector.clone());
            if self.calib_buf.len() >= self.calib_budget {
                let quant = ScalarQuantizer::calibrate(&self.calib_buf);
                let to_insert: Vec<Vec<f32>> = std::mem::take(&mut self.calib_buf);
                self.quant = Some(quant);
                for v in to_insert {
                    let code = self.quant.as_ref().unwrap().encode(&v);
                    self.add_encoded(code, v);
                }
                let code = self.quant.as_ref().unwrap().encode(&vector);
                self.add_encoded(code, vector);
            }
            return;
        }

        let code = self.quant.as_ref().unwrap().encode(&vector);
        self.add_encoded(code, vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let quant = match &self.quant {
            Some(q) => q,
            None => return vec![],
        };
        let q_code = quant.encode(query);
        let codes = &self.codes;
        let originals = &self.originals;

        // Phase 1: graph traversal with int8 distances
        let overquery_k = (k * self.overquery_factor).max(k + 1);
        let candidates =
            self.graph
                .search(self.config.ef_search.max(overquery_k), overquery_k, |j| {
                    ScalarQuantizer::sq8_l2_sq(&q_code, &codes[j]) as f32
                });

        // Phase 2: exact f32 rerank on candidates
        let mut reranked: Vec<(f32, usize)> = candidates
            .into_iter()
            .map(|(_, id)| (l2_sq(query, &originals[id]), id))
            .collect();
        reranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        reranked
            .into_iter()
            .take(k)
            .map(|(d, id)| SearchResult { id, distance: d })
            .collect()
    }

    fn len(&self) -> usize {
        self.codes.len() + self.calib_buf.len()
    }

    fn bytes_per_vector(&self) -> usize {
        self.dims + self.dims * 4 // i8 codes + f32 originals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_vecs(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                (0..dims)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        // Box-Muller would be nicer but simple uniform works for tests
                        (rng as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                    })
                    .collect()
            })
            .collect()
    }

    fn recall_at_k(
        index: &dyn AnnIndex,
        corpus: &[Vec<f32>],
        queries: &[Vec<f32>],
        k: usize,
    ) -> f64 {
        let mut hits = 0usize;
        for q in queries {
            let approx: Vec<usize> = index.search(q, k).into_iter().map(|r| r.id).collect();
            let mut exact: Vec<(f32, usize)> = corpus
                .iter()
                .enumerate()
                .map(|(j, v)| (l2_sq(q, v), j))
                .collect();
            exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let ground: Vec<usize> = exact.into_iter().take(k).map(|(_, id)| id).collect();
            hits += approx.iter().filter(|id| ground.contains(id)).count();
        }
        hits as f64 / (queries.len() * k) as f64
    }

    const N: usize = 2_000;
    const DIMS: usize = 64;
    const K: usize = 10;

    fn make_config() -> HnswConfig {
        HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        }
    }

    #[test]
    fn f32_recall_above_threshold() {
        let corpus = gaussian_vecs(N, DIMS, 42);
        let queries = gaussian_vecs(100, DIMS, 99);
        let mut idx = F32Index::new(make_config(), DIMS);
        for (i, v) in corpus.iter().enumerate() {
            idx.add(i, v.clone());
        }
        let r = recall_at_k(&idx, &corpus, &queries, K);
        assert!(r >= 0.75, "F32Index recall@{K} = {r:.3} < 0.75");
    }

    #[test]
    fn sq8_recall_above_threshold() {
        let corpus = gaussian_vecs(N, DIMS, 42);
        let queries = gaussian_vecs(100, DIMS, 99);
        let mut idx = Sq8Index::new(make_config(), DIMS, N);
        for (i, v) in corpus.iter().enumerate() {
            idx.add(i, v.clone());
        }
        let r = recall_at_k(&idx, &corpus, &queries, K);
        assert!(r >= 0.60, "Sq8Index recall@{K} = {r:.3} < 0.60");
    }

    #[test]
    fn sq8_rerank_recall_above_threshold() {
        let corpus = gaussian_vecs(N, DIMS, 42);
        let queries = gaussian_vecs(100, DIMS, 99);
        let mut idx = Sq8RerankIndex::new(make_config(), DIMS, N);
        for (i, v) in corpus.iter().enumerate() {
            idx.add(i, v.clone());
        }
        let r = recall_at_k(&idx, &corpus, &queries, K);
        assert!(r >= 0.75, "Sq8RerankIndex recall@{K} = {r:.3} < 0.75");
    }

    #[test]
    fn memory_comparison() {
        let f32_idx = F32Index::new(make_config(), DIMS);
        let sq8_idx = Sq8Index::new(make_config(), DIMS, 100);
        let rerank_idx = Sq8RerankIndex::new(make_config(), DIMS, 100);
        assert_eq!(f32_idx.bytes_per_vector(), 256); // 64 * 4
        assert_eq!(sq8_idx.bytes_per_vector(), 64); // 64 * 1
        assert_eq!(rerank_idx.bytes_per_vector(), 320); // 64 + 64*4
    }
}
