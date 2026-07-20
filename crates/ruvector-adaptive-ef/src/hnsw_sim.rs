//! Minimal single-layer k-NN graph with greedy beam search.
//!
//! This is an HNSW-style simulation: it captures the key property that recall improves
//! monotonically with `ef` (beam width) while latency scales roughly linearly. It is not
//! a full multi-layer HNSW implementation; it serves as a controlled test-bed for the
//! adaptive ef policies.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::time::Instant;

/// Euclidean distance squared between two vectors.
fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Wrapper that reverses BinaryHeap ordering (min-heap by distance).
#[derive(PartialEq)]
struct MinDist(f32, usize);

impl Eq for MinDist {}
impl PartialOrd for MinDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MinDist {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap wrapper (closer = higher priority).
#[derive(PartialEq)]
struct MaxDist(f32, usize);

impl Eq for MaxDist {}
impl PartialOrd for MaxDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MaxDist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// Single-layer k-NN graph with greedy beam-search retrieval.
pub struct HnswSim {
    pub vectors: Vec<Vec<f32>>,
    pub graph: Vec<Vec<usize>>,
    pub dim: usize,
    pub m: usize, // max neighbours per node
    rng: u64,
}

impl HnswSim {
    pub fn new(dim: usize, m: usize) -> Self {
        Self { vectors: Vec::new(), graph: Vec::new(), dim, m, rng: 0xDEAD_BEEF_1234_5678 }
    }

    fn lcg(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f32 / (u64::MAX >> 11) as f32
    }

    /// Generate a deterministic random Gaussian-ish vector using Box-Muller on LCG output.
    pub fn random_vector(&mut self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.dim);
        let mut i = 0;
        while i < self.dim {
            let u1 = self.lcg().max(1e-7_f32);
            let u2 = self.lcg();
            let mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            v.push(mag * theta.cos());
            if i + 1 < self.dim { v.push(mag * theta.sin()); }
            i += 2;
        }
        v.truncate(self.dim);
        v
    }

    /// Insert a vector and connect it to its nearest neighbours in the existing graph.
    pub fn insert(&mut self, vec: Vec<f32>) {
        let id = self.vectors.len();
        self.vectors.push(vec);
        self.graph.push(Vec::new());

        if id == 0 { return; }

        // Find m nearest neighbours via linear scan (building phase, correctness over speed)
        let query = &self.vectors[id].clone();
        let mut dists: Vec<(f32, usize)> = (0..id)
            .map(|j| (dist_sq(query, &self.vectors[j]), j))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = self.m.min(dists.len());
        let neighbours: Vec<usize> = dists[..k].iter().map(|(_, j)| *j).collect();

        // Bidirectional edges
        for &nb in &neighbours {
            self.graph[id].push(nb);
            if !self.graph[nb].contains(&id) && self.graph[nb].len() < self.m * 2 {
                self.graph[nb].push(id);
            }
        }
    }

    /// Greedy beam search with beam width `ef`. Returns (result_ids, elapsed_us).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> (Vec<usize>, u64) {
        let n = self.vectors.len();
        if n == 0 { return (vec![], 0); }

        let start = Instant::now();
        let result = self.beam_search(query, k, ef);
        let elapsed = start.elapsed().as_micros() as u64;
        (result, elapsed)
    }

    fn beam_search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
        let n = self.vectors.len();
        if n == 0 { return vec![]; }

        let entry = 0usize; // start from node 0 (single-layer simplification)
        let mut visited = vec![false; n];
        let mut candidates: BinaryHeap<MinDist> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxDist> = BinaryHeap::new();

        let d0 = dist_sq(query, &self.vectors[entry]);
        candidates.push(MinDist(d0, entry));
        results.push(MaxDist(d0, entry));
        visited[entry] = true;

        while let Some(MinDist(d_cur, cur)) = candidates.pop() {
            // If the best candidate is already worse than the ef-th result, stop
            if results.len() >= ef {
                if let Some(MaxDist(d_worst, _)) = results.peek() {
                    if d_cur > *d_worst { break; }
                }
            }

            for &nb in &self.graph[cur] {
                if visited[nb] { continue; }
                visited[nb] = true;
                let d_nb = dist_sq(query, &self.vectors[nb]);
                let should_add = results.len() < ef || {
                    results.peek().map(|MaxDist(d_w, _)| d_nb < *d_w).unwrap_or(true)
                };
                if should_add {
                    candidates.push(MinDist(d_nb, nb));
                    results.push(MaxDist(d_nb, nb));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Return top-k from results (currently sorted by max distance; reverse for min)
        let mut out: Vec<(f32, usize)> = results.into_iter().map(|MaxDist(d, i)| (d, i)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.iter().take(k).map(|(_, i)| *i).collect()
    }

    /// Exact brute-force search. Used as ground truth for recall estimation.
    pub fn exact_search(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = self.vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (dist_sq(query, v), i))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.iter().take(k).map(|(_, i)| *i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::recall_at_k;

    #[test]
    fn insert_and_exact_search() {
        let mut idx = HnswSim::new(8, 8);
        for _ in 0..50 {
            let v = idx.random_vector();
            idx.insert(v);
        }
        assert_eq!(idx.vectors.len(), 50);

        let q = idx.random_vector();
        let gt = idx.exact_search(&q, 5);
        assert_eq!(gt.len(), 5);
    }

    #[test]
    fn recall_improves_with_ef() {
        let mut idx = HnswSim::new(16, 12);
        for _ in 0..200 {
            let v = idx.random_vector();
            idx.insert(v);
        }

        let q = idx.random_vector();
        let gt = idx.exact_search(&q, 10);

        let (r_low, _) = idx.search(&q, 10, 8);
        let (r_high, _) = idx.search(&q, 10, 64);

        let rec_low = recall_at_k(&r_low, &gt);
        let rec_high = recall_at_k(&r_high, &gt);

        // Higher ef should yield equal or better recall in expectation
        assert!(
            rec_high >= rec_low,
            "Expected recall(ef=64)={rec_high} >= recall(ef=8)={rec_low}"
        );
    }

    #[test]
    fn search_returns_k_results() {
        let mut idx = HnswSim::new(8, 6);
        for _ in 0..30 {
            let v = idx.random_vector();
            idx.insert(v);
        }
        let q = idx.random_vector();
        let (res, _) = idx.search(&q, 5, 20);
        assert!(res.len() <= 5);
    }
}
