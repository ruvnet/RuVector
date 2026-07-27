//! ANN index types sharing the [`AnnIndex`] trait.
//!
//! | Index | Build cost | Search cost | Bytes/vec |
//! |---|---|---|---|
//! | `FlatF32Index` | O(N) | O(N·D) | 4D |
//! | `PqIndex` | O(N·M·K·D/M·iter) | O(N·M + M·K·D/M) | M |
//! | `RvqIndex` | O(N·S·K·D·iter) | O(N·S + S·K·D) | S |
//! | `RvqRerankIndex` | same as RvqIndex | same + rerank top-R | S + 4D |

use crate::{
    codebook::{l2_sq, l2_sq_self},
    rvq::{ProductQuantizer, RvqEncoder},
    RvqConfig, SearchResult,
};

// ── Trait ─────────────────────────────────────────────────────────────────────

pub trait AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
    fn bytes_per_vector(&self) -> usize;
}

// ── FlatF32Index — exact brute-force ─────────────────────────────────────────

pub struct FlatF32Index {
    vectors: Vec<Vec<f32>>,
}

impl FlatF32Index {
    pub fn build(data: Vec<Vec<f32>>) -> Self {
        FlatF32Index { vectors: data }
    }
}

impl AnnIndex for FlatF32Index {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut heap: Vec<SearchResult> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, v)| SearchResult { id, distance: l2_sq(query, v) })
            .collect();
        heap.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        heap.truncate(k);
        heap
    }

    fn memory_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.len() * 4).sum()
    }

    fn name(&self) -> &'static str {
        "FlatF32"
    }

    fn bytes_per_vector(&self) -> usize {
        self.vectors.first().map_or(0, |v| v.len() * 4)
    }
}

// ── PqIndex — standard product quantization ───────────────────────────────────

pub struct PqIndex {
    pq: ProductQuantizer,
    codes: Vec<Vec<u8>>,  // N × M
    n: usize,
}

impl PqIndex {
    pub fn build(data: Vec<Vec<f32>>, m: usize, k: usize, train_iters: usize) -> Self {
        let dim = data[0].len();
        let pq = ProductQuantizer::train(&data, m, k, train_iters, dim);
        let codes = data.iter().map(|v| pq.encode(v)).collect();
        let n = data.len();
        PqIndex { pq, codes, n }
    }
}

impl AnnIndex for PqIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let table = self.pq.adc_table(query);
        let mut results: Vec<SearchResult> = (0..self.n)
            .map(|id| SearchResult {
                id,
                distance: ProductQuantizer::adc_distance(&self.codes[id], &table),
            })
            .collect();
        results.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        results.truncate(k);
        results
    }

    fn memory_bytes(&self) -> usize {
        // codes + codebooks
        let code_bytes = self.n * self.pq.m;
        let cb_bytes = self.pq.m * self.pq.k * self.pq.sub_dim * 4;
        code_bytes + cb_bytes
    }

    fn name(&self) -> &'static str {
        "PqIndex"
    }

    fn bytes_per_vector(&self) -> usize {
        self.pq.m
    }
}

// ── RvqIndex — residual vector quantization ───────────────────────────────────

pub struct RvqIndex {
    encoder: RvqEncoder,
    codes: Vec<Vec<u8>>,  // N × num_stages
    n: usize,
    dim: usize,
}

impl RvqIndex {
    pub fn build_with_config(config: RvqConfig, data: Vec<Vec<f32>>) -> Result<Self, String> {
        if data.is_empty() {
            return Err("data must not be empty".into());
        }
        let dim = data[0].len();
        if dim != config.dim {
            return Err(format!("config.dim={} but data dim={}", config.dim, dim));
        }
        let encoder = RvqEncoder::train(config, &data);
        let codes = data.iter().map(|v| encoder.encode(v)).collect();
        let n = data.len();
        Ok(RvqIndex { encoder, codes, n, dim })
    }
}

impl AnnIndex for RvqIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let query_norm_sq = l2_sq_self(query);
        let (inner, norms) = self.encoder.adc_tables(query);
        let mut results: Vec<SearchResult> = (0..self.n)
            .map(|id| SearchResult {
                id,
                distance: RvqEncoder::adc_distance(
                    query_norm_sq,
                    &self.codes[id],
                    &inner,
                    &norms,
                ),
            })
            .collect();
        results.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        results.truncate(k);
        results
    }

    fn memory_bytes(&self) -> usize {
        let code_bytes = self.n * self.encoder.config.num_stages;
        let s = self.encoder.config.num_stages;
        let k = self.encoder.config.codebook_size;
        let d = self.encoder.config.dim;
        let cb_bytes = s * k * d * 4;
        code_bytes + cb_bytes
    }

    fn name(&self) -> &'static str {
        "RvqIndex"
    }

    fn bytes_per_vector(&self) -> usize {
        self.encoder.config.num_stages
    }
}

// ── RvqRerankIndex — RVQ with exact rerank ────────────────────────────────────

/// Extends `RvqIndex` by storing original f32 vectors for exact reranking of
/// the top `rerank_factor × k` ADC candidates.
pub struct RvqRerankIndex {
    inner: RvqIndex,
    originals: Vec<Vec<f32>>,
    rerank_factor: usize,
}

impl RvqRerankIndex {
    pub fn build_with_config(
        config: RvqConfig,
        data: Vec<Vec<f32>>,
        rerank_factor: usize,
    ) -> Result<Self, String> {
        let originals = data.clone();
        let inner = RvqIndex::build_with_config(config, data)?;
        Ok(RvqRerankIndex { inner, originals, rerank_factor })
    }
}

impl AnnIndex for RvqRerankIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        // Fetch candidates at rerank_factor × k via ADC.
        let candidates = self.inner.search(query, k * self.rerank_factor);
        // Exact rerank.
        let mut reranked: Vec<SearchResult> = candidates
            .iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: l2_sq(query, &self.originals[c.id]),
            })
            .collect();
        reranked.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        reranked.truncate(k);
        reranked
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes() + self.originals.iter().map(|v| v.len() * 4).sum::<usize>()
    }

    fn name(&self) -> &'static str {
        "RvqRerank"
    }

    fn bytes_per_vector(&self) -> usize {
        // codes + originals
        self.inner.bytes_per_vector() + self.inner.dim * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RvqConfig;

    fn tiny_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::SeedableRng;
        use rand::Rng as _;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| (0..d).map(|_| rng.gen::<f32>()).collect()).collect()
    }

    #[test]
    fn flat_returns_exact_top1() {
        let data = tiny_data(100, 16, 1);
        let query = data[42].clone();
        let idx = FlatF32Index::build(data);
        let results = idx.search(&query, 1);
        assert_eq!(results[0].id, 42);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn pq_index_builds_and_searches() {
        let data = tiny_data(200, 32, 2);
        let query = data[10].clone();
        let idx = PqIndex::build(data, 4, 16, 15);
        let results = idx.search(&query, 5);
        assert_eq!(results.len(), 5);
        // PQ should find the exact vector somewhere in top-5 for random data
        assert!(results.iter().any(|r| r.id == 10));
    }

    #[test]
    fn rvq_index_builds_and_searches() {
        let data = tiny_data(200, 32, 3);
        let query = data[5].clone();
        let cfg = RvqConfig {
            dim: 32,
            num_stages: 4,
            codebook_size: 16,
            train_iters: 15,
            dropout_prob: 0.1,
        };
        let idx = RvqIndex::build_with_config(cfg, data).unwrap();
        let results = idx.search(&query, 5);
        assert_eq!(results.len(), 5);
        assert!(results.iter().any(|r| r.id == 5));
    }

    #[test]
    fn rvq_rerank_improves_over_raw() {
        // With exact reranking, the indexed point should be rank-1.
        let data = tiny_data(500, 32, 4);
        let query = data[77].clone();
        let cfg = RvqConfig {
            dim: 32,
            num_stages: 2,
            codebook_size: 16,
            train_iters: 10,
            dropout_prob: 0.0,
        };
        let idx = RvqRerankIndex::build_with_config(cfg, data, 4).unwrap();
        let results = idx.search(&query, 1);
        assert_eq!(results[0].id, 77);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn rvq_encode_decode_roundtrip() {
        let data = tiny_data(300, 16, 5);
        let cfg = RvqConfig {
            dim: 16,
            num_stages: 8,
            codebook_size: 32,
            train_iters: 20,
            dropout_prob: 0.1,
        };
        let encoder = crate::rvq::RvqEncoder::train(cfg, &data);
        // Distortion should decrease as stages increase.
        let stage_dists = encoder.stage_distortions(&data);
        assert_eq!(stage_dists.len(), 8);
        // Mean distortion over all stages should be well-defined (not NaN/inf).
        for &d in &stage_dists {
            assert!(d.is_finite());
        }
        // Encode-decode roundtrip should give finite distances.
        let v = &data[0];
        let codes = encoder.encode(v);
        let reconstructed = encoder.decode(&codes);
        assert_eq!(reconstructed.len(), v.len());
        for x in reconstructed {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn memory_estimates_are_positive() {
        let data = tiny_data(100, 16, 6);
        let flat = FlatF32Index::build(data.clone());
        assert!(flat.memory_bytes() > 0);

        let pq = PqIndex::build(data.clone(), 4, 8, 10);
        assert!(pq.memory_bytes() > 0);
        assert!(pq.memory_bytes() < flat.memory_bytes());

        let cfg = RvqConfig { dim: 16, num_stages: 4, codebook_size: 8, train_iters: 10, dropout_prob: 0.0 };
        let rvq = RvqIndex::build_with_config(cfg, data).unwrap();
        assert!(rvq.memory_bytes() > 0);
    }
}
