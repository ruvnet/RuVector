//! Flat brute-force corpus scan. Used as the fallback (cache miss) path.
//! For corpus sizes ≤ 100 K this runs in 1–10 ms on a modern CPU, which
//! is the window the semantic cache is designed to short-circuit.

use crate::SearchResult;

/// An immutable vector corpus supporting top-k nearest-neighbour queries.
pub struct FlatCorpus {
    /// Row-major: vectors[i*dim .. i*dim+dim] is the i-th vector.
    vectors: Vec<f32>,
    dim: usize,
    count: usize,
}

impl FlatCorpus {
    pub fn new(vectors: Vec<f32>, dim: usize) -> Self {
        assert_eq!(
            vectors.len() % dim,
            0,
            "vectors.len() must be divisible by dim"
        );
        let count = vectors.len() / dim;
        Self {
            vectors,
            dim,
            count,
        }
    }

    /// Generate a deterministic corpus using an LCG seeded from `seed`.
    pub fn generate(count: usize, dim: usize, seed: u64) -> Self {
        let mut rng = LcgRng::new(seed);
        let mut vectors = Vec::with_capacity(count * dim);
        for _ in 0..count * dim {
            vectors.push(rng.next_f32_unit_normal());
        }
        Self::new(vectors, dim)
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the i-th vector as a slice.
    pub fn get(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }

    /// Brute-force top-k search by L2 distance. O(n·d).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim);
        let mut heap: Vec<SearchResult> = Vec::with_capacity(k + 1);

        for i in 0..self.count {
            let v = self.get(i);
            let dist = crate::l2_distance_sq(query, v);
            if heap.len() < k {
                heap.push(SearchResult {
                    id: i as u32,
                    distance: dist,
                });
                if heap.len() == k {
                    // Build max-heap property (largest dist at position 0).
                    heap.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
                }
            } else if dist < heap[0].distance {
                heap[0] = SearchResult {
                    id: i as u32,
                    distance: dist,
                };
                // Sift down to maintain max-heap.
                sift_down(&mut heap);
            }
        }

        heap.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        heap
    }

    /// Bytes of heap memory used by the vectors.
    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * 4
    }
}

fn sift_down(heap: &mut Vec<SearchResult>) {
    // The root (index 0) may not be the max. Restore max-heap by sifting down.
    let n = heap.len();
    let mut i = 0;
    loop {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        let mut largest = i;
        if l < n && heap[l].distance > heap[largest].distance {
            largest = l;
        }
        if r < n && heap[r].distance > heap[largest].distance {
            largest = r;
        }
        if largest == i {
            break;
        }
        heap.swap(i, largest);
        i = largest;
    }
}

/// A minimal linear congruential generator for deterministic test data.
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Standard normal approximation via Box-Muller (needs two uniforms).
    pub fn next_f32_unit_normal(&mut self) -> f32 {
        let u = self.next_f32().max(1e-10);
        let v = self.next_f32();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

/// Generate a mixed workload of query vectors modelling realistic agent query patterns.
///
/// Three tiers of query similarity are blended:
/// - **Exact** (`exact_frac`): identical to a prototype (cosine = 1.0 with cached copy).
///   Models agents re-submitting the same retrieval request verbatim.
/// - **Near-duplicate** (`near_frac`): prototype + small noise (σ = `near_sigma`).
///   Models slightly-rephrased queries; cosine ≈ 1 / (1 + near_sigma² × dim).
///   With near_sigma = 0.02, dim = 128: cosine ≈ 0.95 → hits coarse cache, misses fine cache.
/// - **Diverse** (remaining fraction): prototype + large noise (σ = `far_sigma`).
///   Models genuinely new queries; cosine typically < 0.80 → misses all thresholds.
///
/// All output vectors are L2-normalised for cosine similarity correctness.
pub fn generate_workload_mixed(
    prototypes: &[Vec<f32>],
    n: usize,
    exact_frac: f32,
    near_frac: f32,
    near_sigma: f32,
    far_sigma: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = LcgRng::new(seed);
    let p = prototypes.len();
    let mut out = Vec::with_capacity(n);

    for _ in 0..n {
        let proto_idx = (rng.next_u64() as usize) % p;
        let proto = &prototypes[proto_idx];

        let tier = rng.next_f32();
        let sigma = if tier < exact_frac {
            0.0_f32 // exact repeat
        } else if tier < exact_frac + near_frac {
            near_sigma // near-duplicate
        } else {
            far_sigma // diverse / genuinely new
        };

        let mut q: Vec<f32> = if sigma < 1e-9 {
            proto.clone()
        } else {
            proto
                .iter()
                .map(|&x| x + sigma * rng.next_f32_unit_normal())
                .collect()
        };

        // L2-normalise.
        let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in q.iter_mut() {
            *x /= norm;
        }
        out.push(q);
    }
    out
}

/// Simple uniform-noise workload (kept for tests). See `generate_workload_mixed` for production.
pub fn generate_workload(
    prototypes: &[Vec<f32>],
    n: usize,
    noise_sigma: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    generate_workload_mixed(prototypes, n, 0.0, 1.0, noise_sigma, noise_sigma, seed)
}

/// Generate prototype query vectors (normalized).
pub fn generate_prototypes(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let corpus = FlatCorpus::generate(count, dim, seed.wrapping_add(99_999));
    (0..count)
        .map(|i| {
            let v = corpus.get(i);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter().map(|x| x / norm).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_search_returns_k_results_ordered_by_distance() {
        let corpus = FlatCorpus::generate(1000, 64, 42);
        let query: Vec<f32> = corpus.get(7).to_vec();
        let results = corpus.search(&query, 10);
        assert_eq!(results.len(), 10);
        // The query vector itself should be the nearest neighbour (distance ≈ 0).
        assert!(
            results[0].distance < 1e-4,
            "nn dist={}",
            results[0].distance
        );
        assert_eq!(results[0].id, 7);
        // Results are ordered ascending by distance.
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn workload_queries_are_normalized() {
        let protos = generate_prototypes(10, 32, 1);
        let workload = generate_workload(&protos, 50, 0.05, 2);
        for q in &workload {
            let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
        }
    }
}
