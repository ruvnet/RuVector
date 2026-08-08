//! Flat (linear scan) AQ index variants.
//!
//! Both `IsotropicFlat` and `AnisotropicFlat` use the same ADC scan loop;
//! they differ only in how the codebook was trained.

use crate::codebook::{AqCodebook, AqConfig};
use crate::{l2_norm, AqSearch, SearchResult};

/// Flat index backed by an isotropic-trained codebook (standard PQ baseline).
pub struct IsotropicFlat {
    codebook: AqCodebook,
    codes: Vec<Vec<u8>>,
    #[allow(dead_code)]
    dim: usize,
}

impl IsotropicFlat {
    pub fn new(dim: usize, m: usize, k: usize, train_vectors: &[f32]) -> Self {
        let cfg = AqConfig::isotropic(m, k);
        let normalized = normalize_rows(train_vectors, dim);
        let codebook = AqCodebook::train(cfg, &normalized, dim);
        Self {
            codebook,
            codes: Vec::new(),
            dim,
        }
    }
}

impl AqSearch for IsotropicFlat {
    fn insert(&mut self, vector: &[f32]) {
        let norm = l2_norm(vector);
        let nv: Vec<f32> = vector.iter().map(|x| x / norm).collect();
        let code = encode(&self.codebook, &nv);
        self.codes.push(code);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let norm = l2_norm(query);
        let nq: Vec<f32> = query.iter().map(|x| x / norm).collect();
        flat_scan(&self.codebook, &self.codes, &nq, k)
    }

    fn memory_bytes(&self) -> usize {
        self.codebook.memory_bytes() + self.codes.len() * self.codebook.config.m
    }

    fn name(&self) -> &'static str {
        "IsotropicFlat"
    }
}

/// Flat index backed by an anisotropic-trained codebook (AQ).
pub struct AnisotropicFlat {
    codebook: AqCodebook,
    codes: Vec<Vec<u8>>,
    #[allow(dead_code)]
    dim: usize,
}

impl AnisotropicFlat {
    pub fn new(dim: usize, m: usize, k: usize, eta: f32, train_vectors: &[f32]) -> Self {
        let cfg = AqConfig::anisotropic(m, k, eta);
        let normalized = normalize_rows(train_vectors, dim);
        let codebook = AqCodebook::train(cfg, &normalized, dim);
        Self {
            codebook,
            codes: Vec::new(),
            dim,
        }
    }
}

impl AqSearch for AnisotropicFlat {
    fn insert(&mut self, vector: &[f32]) {
        let norm = l2_norm(vector);
        let nv: Vec<f32> = vector.iter().map(|x| x / norm).collect();
        let code = encode(&self.codebook, &nv);
        self.codes.push(code);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let norm = l2_norm(query);
        let nq: Vec<f32> = query.iter().map(|x| x / norm).collect();
        flat_scan(&self.codebook, &self.codes, &nq, k)
    }

    fn memory_bytes(&self) -> usize {
        self.codebook.memory_bytes() + self.codes.len() * self.codebook.config.m
    }

    fn name(&self) -> &'static str {
        "AnisotropicFlat"
    }
}

/// Encode a normalised vector to PQ codes using the AQ assignment.
pub fn encode(cb: &AqCodebook, nv: &[f32]) -> Vec<u8> {
    let m = cb.config.m;
    let sub_dim = cb.sub_dim;
    (0..m)
        .map(|sub| {
            let sv = &nv[sub * sub_dim..(sub + 1) * sub_dim];
            cb.assign(sub, sv, nv)
        })
        .collect()
}

/// Linear ADC scan returning top-k by inner product score.
pub fn flat_scan(cb: &AqCodebook, codes: &[Vec<u8>], query: &[f32], k: usize) -> Vec<SearchResult> {
    let table = cb.build_ip_adc_table(query);
    let mut scored: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| (i, cb.adc_score(&table, code)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
        .into_iter()
        .take(k)
        .map(|(id, score)| SearchResult { id, score })
        .collect()
}

pub fn normalize_rows(vectors: &[f32], dim: usize) -> Vec<f32> {
    let n = vectors.len() / dim;
    let mut out = vectors.to_vec();
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        let norm = l2_norm(row);
        for d in 0..dim {
            out[i * dim + d] = row[d] / norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let raw: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        normalize_rows(&raw, dim)
    }

    #[test]
    fn isotropic_flat_builds_and_searches() {
        let dim = 64;
        let train = make_vecs(200, dim, 10);
        let mut idx = IsotropicFlat::new(dim, 8, 16, &train);
        for i in 0..100usize {
            let v: Vec<f32> = (0..dim).map(|d| ((i * dim + d) as f32).sin()).collect();
            idx.insert(&v);
        }
        let q: Vec<f32> = (0..dim).map(|d| (d as f32).cos()).collect();
        let results = idx.search(&q, 5);
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.score.is_finite()));
    }

    #[test]
    fn anisotropic_flat_builds_and_searches() {
        let dim = 64;
        let train = make_vecs(200, dim, 11);
        let mut idx = AnisotropicFlat::new(dim, 8, 16, 2.0, &train);
        for i in 0..100usize {
            let v: Vec<f32> = (0..dim).map(|d| ((i + d) as f32).cos()).collect();
            idx.insert(&v);
        }
        let q: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.1).sin()).collect();
        let results = idx.search(&q, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn top_result_is_self_similar() {
        let dim = 32;
        let train = make_vecs(100, dim, 20);
        let mut idx = AnisotropicFlat::new(dim, 4, 16, 2.0, &train);
        let mut rng = StdRng::seed_from_u64(42);
        let mut vecs: Vec<Vec<f32>> = (0..50)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let n = l2_norm(&v);
                v.iter().map(|x| x / n).collect()
            })
            .collect();
        for v in &vecs {
            idx.insert(v);
        }
        // Query identical to inserted vector 0 — should be top result.
        let q = vecs[0].clone();
        let results = idx.search(&q, 3);
        assert_eq!(results[0].id, 0, "self should be top result");
    }
}
