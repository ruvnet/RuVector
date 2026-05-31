//! Baseline ANN variants for comparison with CoDEQ.
//!
//! - `FlatL2IndexCoDEQ`: exact brute-force (always correct; slowest)
//! - `StaticPqIndex`: naive k-means Product Quantization (fast build; no streaming)

use crate::dist::l2_sq;
use crate::error::CoDEQError;

// ---------------------------------------------------------------------------
// Variant 1: FlatL2 brute-force
// ---------------------------------------------------------------------------

pub struct FlatL2IndexCoDEQ {
    data: Vec<(u64, Vec<f32>)>,
    dim: usize,
}

impl FlatL2IndexCoDEQ {
    pub fn from_vecs(data: Vec<(u64, Vec<f32>)>) -> Result<Self, CoDEQError> {
        if data.is_empty() { return Err(CoDEQError::EmptyDataset); }
        let dim = data[0].1.len();
        Ok(Self { data, dim })
    }

    pub fn insert(&mut self, id: u64, v: Vec<f32>) -> Result<(), CoDEQError> {
        if v.len() != self.dim {
            return Err(CoDEQError::DimMismatch { expected: self.dim, actual: v.len() });
        }
        self.data.push((id, v));
        Ok(())
    }

    pub fn delete(&mut self, id: u64) -> Result<(), CoDEQError> {
        if let Some(pos) = self.data.iter().position(|(x, _)| *x == id) {
            self.data.swap_remove(pos);
            Ok(())
        } else {
            Err(CoDEQError::IdNotFound(id))
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, CoDEQError> {
        let n = self.data.len();
        if k > n { return Err(CoDEQError::KTooLarge { k, n }); }
        let mut scores: Vec<(u64, f32)> = self.data.iter()
            .map(|(id, v)| (*id, l2_sq(query, v)))
            .collect();
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        scores.truncate(k);
        Ok(scores)
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn memory_bytes(&self) -> usize { self.data.len() * self.dim * 4 }
    pub fn name(&self) -> &'static str { "FlatL2 (exact baseline)" }
}

// ---------------------------------------------------------------------------
// Variant 2: Static PQ (k-means per subspace, frozen after build)
//
// Standard Product Quantization:
//   - Split dim D into m subspaces of d = D/m dims each.
//   - For each subspace, run k-means with k=256 centroids on training data.
//   - Encode each vector as m bytes (nearest centroid per subspace).
//   - ADC: precompute query-subvec → centroid distances; look up per code.
//
// Limitation: codebook is frozen. After significant drift, centroids no longer
// represent the live data distribution — recall degrades.
// ---------------------------------------------------------------------------

pub struct StaticPqIndex {
    /// m subspaces, each of dsub = dim/m dimensions.
    m: usize,
    dsub: usize,
    dim: usize,
    /// centroids[s] = Vec of 256 centroids for subspace s, each of length dsub.
    centroids: Vec<Vec<Vec<f32>>>,
    /// Encoded codes: one Vec<u8> of length m per vector.
    codes: Vec<(u64, Vec<u8>)>,
    /// Raw data kept for reranking (optional; increases memory).
    raw: Vec<(u64, Vec<f32>)>,
    n_centroids: usize,
}

impl StaticPqIndex {
    /// Build from a training set. m subspaces, k_sub centroids each, n_iter k-means.
    pub fn build(
        data: &[(u64, Vec<f32>)],
        m: usize,
        k_sub: usize,
        n_iter: usize,
    ) -> Result<Self, CoDEQError> {
        if data.is_empty() { return Err(CoDEQError::EmptyDataset); }
        let dim = data[0].1.len();
        if dim % m != 0 {
            return Err(CoDEQError::DimMismatch { expected: m, actual: dim % m });
        }
        let dsub = dim / m;
        let k = k_sub.min(data.len());

        // Train codebooks via k-means (one per subspace).
        let mut centroids: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
        for s in 0..m {
            let subvecs: Vec<Vec<f32>> = data.iter()
                .map(|(_, v)| v[s * dsub..(s + 1) * dsub].to_vec())
                .collect();
            centroids.push(kmeans(&subvecs, k, n_iter, s as u64));
        }

        // Encode all training vectors.
        let codes: Vec<(u64, Vec<u8>)> = data.iter().map(|(id, v)| {
            let code: Vec<u8> = (0..m)
                .map(|s| {
                    let sub = &v[s * dsub..(s + 1) * dsub];
                    nearest_centroid(sub, &centroids[s]) as u8
                })
                .collect();
            (*id, code)
        }).collect();

        let raw = data.to_vec();

        Ok(Self { m, dsub, dim, centroids, codes, raw, n_centroids: k })
    }

    pub fn search_adc(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, CoDEQError> {
        let n = self.codes.len();
        if k > n { return Err(CoDEQError::KTooLarge { k, n }); }

        // Precompute lookup table: LUT[s][c] = L2² between query subvec s and centroid c.
        let lut: Vec<Vec<f32>> = (0..self.m)
            .map(|s| {
                let qsub = &query[s * self.dsub..(s + 1) * self.dsub];
                (0..self.n_centroids)
                    .map(|c| l2_sq(qsub, &self.centroids[s][c]))
                    .collect()
            })
            .collect();

        // ADC scan: sum LUT lookups per code.
        let mut scores: Vec<(u64, f32)> = self.codes.iter()
            .map(|(id, code)| {
                let dist: f32 = code.iter().enumerate()
                    .map(|(s, &c)| lut[s][c as usize])
                    .sum();
                (*id, dist)
            })
            .collect();
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Rerank top 8k with exact distance.
        let oversample = (k * 8).min(n);
        scores.truncate(oversample);
        let raw_map: std::collections::HashMap<u64, &Vec<f32>> =
            self.raw.iter().map(|(id, v)| (*id, v)).collect();
        for (id, dist) in &mut scores {
            if let Some(v) = raw_map.get(id) {
                *dist = l2_sq(query, v);
            }
        }
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        scores.truncate(k);
        Ok(scores)
    }

    pub fn len(&self) -> usize { self.codes.len() }
    pub fn dim(&self) -> usize { self.dim }
    pub fn memory_bytes(&self) -> usize {
        let codebook_bytes = self.m * self.n_centroids * self.dsub * 4;
        let code_bytes = self.codes.len() * self.m;
        let raw_bytes = self.raw.len() * self.dim * 4;
        codebook_bytes + code_bytes + raw_bytes
    }
    pub fn name(&self) -> &'static str { "StaticPQ (k-means, frozen)" }
}

// ---------------------------------------------------------------------------
// k-means helpers
// ---------------------------------------------------------------------------

/// Lloyd's k-means on a set of subvectors with `n_iter` iterations.
fn kmeans(data: &[Vec<f32>], k: usize, n_iter: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Initialise centroids by sampling k points without replacement.
    let mut centroids: Vec<Vec<f32>> = data.choose_multiple(&mut rng, k).cloned().collect();
    if centroids.len() < k {
        // Pad by repeating if dataset < k.
        while centroids.len() < k { centroids.push(centroids[0].clone()); }
    }

    let dsub = data[0].len();
    for _ in 0..n_iter {
        // Assign each point to nearest centroid.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dsub]; k];
        let mut counts = vec![0usize; k];
        for v in data {
            let c = nearest_centroid(v, &centroids);
            for (s, x) in sums[c].iter_mut().zip(v) { *s += x; }
            counts[c] += 1;
        }
        // Update centroids.
        for (c, sum) in sums.iter().enumerate() {
            if counts[c] > 0 {
                centroids[c] = sum.iter().map(|&s| s / counts[c] as f32).collect();
            }
        }
    }
    centroids
}

fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids.iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0_f32, 1.0).unwrap();
        (0..n as u64)
            .map(|id| (id, (0..dim).map(|_| dist.sample(&mut rng)).collect()))
            .collect()
    }

    #[test]
    fn flat_returns_exact_top1() {
        let data = make_vecs(100, 16, 1);
        let idx = FlatL2IndexCoDEQ::from_vecs(data.clone()).unwrap();
        let res = idx.search(&data[0].1, 1).unwrap();
        assert_eq!(res[0].0, data[0].0);
        assert!(res[0].1 < 1e-5);
    }

    #[test]
    fn pq_search_returns_k() {
        let data = make_vecs(200, 32, 2);
        let idx = StaticPqIndex::build(&data, 4, 32, 5).unwrap();
        let res = idx.search_adc(&data[0].1, 5).unwrap();
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn pq_recall_on_gaussian() {
        let data = make_vecs(500, 32, 3);
        let queries = make_vecs(50, 32, 4);
        let idx = StaticPqIndex::build(&data, 4, 64, 10).unwrap();
        let mut hits = 0usize;
        let k = 10;
        for (_qid, qv) in &queries {
            let approx: std::collections::HashSet<u64> =
                idx.search_adc(qv, k).unwrap().into_iter().map(|(id,_)| id).collect();
            let exact: std::collections::HashSet<u64> =
                FlatL2IndexCoDEQ::from_vecs(data.clone()).unwrap()
                    .search(qv, k).unwrap().into_iter().map(|(id,_)| id).collect();
            hits += approx.intersection(&exact).count();
        }
        let recall = hits as f64 / (queries.len() * k) as f64;
        assert!(recall >= 0.70, "PQ recall={:.3}", recall);
    }
}
