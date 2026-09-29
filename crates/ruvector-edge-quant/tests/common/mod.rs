//! Shared test fixtures: clustered data, exact ground truth, an in-memory
//! rerank source standing in for the f32 store.
#![allow(dead_code)]

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_edge_quant::{Metric, RerankSource};
use ruvector_edge_store::distance;
use std::collections::HashMap;

/// `n` points around `clusters` Gaussian centres (+ `queries` held-out
/// points from the same mixture).
pub fn clustered(
    n: usize,
    queries: usize,
    dim: usize,
    clusters: usize,
    spread: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let unit = Normal::new(0.0f32, 1.0).unwrap();
    let noise = Normal::new(0.0f32, spread).unwrap();
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| unit.sample(&mut rng)).collect())
        .collect();
    let draw = |rng: &mut rand::rngs::StdRng| -> Vec<f32> {
        let c = &centres[rng.gen_range(0..clusters)];
        c.iter().map(|&x| x + noise.sample(rng)).collect()
    };
    let data = (0..n).map(|_| draw(&mut rng)).collect();
    let qs = (0..queries).map(|_| draw(&mut rng)).collect();
    (data, qs)
}

/// Exact top-k keys (key = row index) under `metric`, ties by key.
pub fn exact_topk(data: &[Vec<f32>], q: &[f32], k: usize, metric: Metric) -> Vec<u64> {
    let qn = distance::norm(q);
    let mut d: Vec<(f64, u64)> = data
        .iter()
        .enumerate()
        .map(|(i, r)| {
            (
                distance::distance(metric, q, qn, r, distance::norm(r)),
                i as u64,
            )
        })
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.into_iter().take(k).map(|x| x.1).collect()
}

/// Rows keyed by `u64`, as the store would hold them.
pub struct MemRows {
    pub rows: HashMap<u64, Vec<f32>>,
    pub fetched: u64,
}

impl MemRows {
    pub fn from_data(data: &[Vec<f32>]) -> Self {
        MemRows {
            rows: data
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, v)| (i as u64, v))
                .collect(),
            fetched: 0,
        }
    }
}

impl RerankSource for MemRows {
    fn fetch(&mut self, keys: &[u64]) -> Result<Vec<Option<Vec<f32>>>, String> {
        self.fetched += keys.len() as u64;
        Ok(keys.iter().map(|k| self.rows.get(k).cloned()).collect())
    }
}

/// Rows as `(key, &[f32])` pairs keyed by index.
pub fn rows(data: &[Vec<f32>]) -> Vec<(u64, &[f32])> {
    data.iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect()
}

/// Mean recall@k of `got` against `truth`.
pub fn recall(truth: &[Vec<u64>], got: &[Vec<u64>]) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (t, g) in truth.iter().zip(got) {
        hit += g.iter().filter(|x| t.contains(x)).count();
        total += t.len();
    }
    hit as f64 / total as f64
}
