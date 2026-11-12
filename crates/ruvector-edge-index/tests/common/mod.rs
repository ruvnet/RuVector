//! Deterministic clustered synthetic data and exact ground truth.
//!
//! Each vector = cluster centroid (N(0,1) per dim) + a point in that
//! cluster's own 64-dim latent subspace (basis N(0, 0.15²)) + isotropic
//! noise N(0, 0.3²), then scaled by a per-vector factor uniform in
//! [0.8, 1.2] (the norm spread that matters for `dot`). 64 clusters.
//! Queries come from the same process with a different stream (held out).
#![allow(dead_code)]

use ruvector_edge_index::{distance, Hit, LevelRng, Metric, SplitMix64};

pub const N: usize = 10_000;
pub const DIM: usize = 384;
pub const QUERIES: usize = 200;
pub const K: usize = 10;
pub const SALT: u64 = 0x5EED_CAFE_F00D_0001;

const CLUSTERS: usize = 64;
const LATENT: usize = 64;

pub struct Gen(SplitMix64);

impl Gen {
    pub fn new(seed: u64) -> Self {
        Self(SplitMix64::new(seed))
    }
    pub fn unit(&mut self) -> f32 {
        ((self.0.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    pub fn gauss(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

pub struct World {
    centroids: Vec<f32>,
    bases: Vec<f32>,
    dim: usize,
}

impl World {
    pub fn new(dim: usize) -> Self {
        let mut g = Gen::new(42);
        let centroids = (0..CLUSTERS * dim).map(|_| g.gauss()).collect();
        let bases = (0..CLUSTERS * LATENT * dim)
            .map(|_| 0.15 * g.gauss())
            .collect();
        Self {
            centroids,
            bases,
            dim,
        }
    }

    pub fn sample(&self, g: &mut Gen, out: &mut Vec<f32>) {
        let d = self.dim;
        let c = (g.0.next_u64() % CLUSTERS as u64) as usize;
        let start = out.len();
        out.extend_from_slice(&self.centroids[c * d..(c + 1) * d]);
        for j in 0..LATENT {
            let z = g.gauss();
            let b = &self.bases[(c * LATENT + j) * d..(c * LATENT + j + 1) * d];
            for (o, x) in out[start..].iter_mut().zip(b) {
                *o += z * x;
            }
        }
        let s = 0.8 + 0.4 * g.unit();
        for o in &mut out[start..] {
            *o = (*o + 0.3 * g.gauss()) * s;
        }
    }

    pub fn points(&self, seed: u64, n: usize) -> Vec<f32> {
        let mut g = Gen::new(seed);
        let mut v = Vec::with_capacity(n * self.dim);
        for _ in 0..n {
            self.sample(&mut g, &mut v);
        }
        v
    }
}

/// `(base, queries)` at the given size.
pub fn data(n: usize, dim: usize, queries: usize) -> (Vec<f32>, Vec<f32>) {
    let w = World::new(dim);
    (w.points(1, n), w.points(2, queries))
}

/// Exact top-k ids, ties by iid.
pub fn exact(metric: Metric, base: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let mut all: Vec<(f32, u32)> = base
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| (distance(metric, q, v), i as u32))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.truncate(k);
    all.into_iter().map(|x| x.1).collect()
}

pub fn truth(metric: Metric, base: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<u32>> {
    queries
        .chunks_exact(dim)
        .map(|q| exact(metric, base, dim, q, k))
        .collect()
}

pub fn recall(truth: &[u32], got: &[Hit]) -> f64 {
    let hit = got.iter().filter(|h| truth.contains(&h.iid)).count();
    hit as f64 / truth.len() as f64
}

/// Per-op stateless level randomness (ADR §6.1).
pub fn op_rng(seq: u64) -> SplitMix64 {
    SplitMix64::new(seq ^ SALT)
}

pub const METRICS: [Metric; 3] = [Metric::Cosine, Metric::L2, Metric::Dot];
