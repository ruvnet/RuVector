//! Deterministic synthetic dataset generation for benchmarks.
//!
//! Uses a minimal xorshift64 PRNG (no external crates) to produce
//! reproducible vectors and query workloads.

/// Minimal xorshift64 PRNG — deterministic, no external deps.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Seed must be non-zero.
        Rng(if seed == 0 { 0xdeadbeef_cafebabe } else { seed })
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform float in (-1, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64();
        let u = (bits >> 33) as f32; // 31-bit mantissa
        u / (0x7FFF_FFFFu32 as f32) - 1.0
    }

    /// Unit-normalised random vector of dimension `dim`.
    pub fn unit_vec(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// A near-duplicate of `base` by adding Gaussian-distributed noise of
    /// magnitude `epsilon` and re-normalising.
    pub fn perturb(&mut self, base: &[f32], epsilon: f32) -> Vec<f32> {
        let mut v: Vec<f32> = base
            .iter()
            .map(|&x| x + epsilon * self.next_f32())
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

/// A synthetic benchmark dataset.
pub struct Dataset {
    pub vectors: Vec<Vec<f32>>,
    pub dim: usize,
    pub n: usize,
}

impl Dataset {
    /// Generate `n` unit-normalised vectors of dimension `dim` using `seed`.
    pub fn random(n: usize, dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let vectors = (0..n).map(|_| rng.unit_vec(dim)).collect();
        Dataset { vectors, dim, n }
    }
}

/// Description of one query in a workload.
pub struct QueryEntry {
    pub vec: Vec<f32>,
    /// True if this is a near-duplicate of an earlier query.
    pub is_near_dup: bool,
    /// Index into the dataset of the expected nearest neighbour (brute-force).
    pub nn_idx: u32,
}

/// Build a mixed workload of unique and near-duplicate queries.
///
/// - `n_unique`  — queries that are new random vectors
/// - `n_near_dup` — near-duplicates of randomly chosen earlier unique queries
/// - `epsilon`   — noise magnitude for near-duplicates
///
/// Shuffle is intentionally disabled: near-duplicates are interleaved immediately
/// after their source to simulate a realistic agent query pattern (topic locality).
pub fn build_workload(
    dataset: &Dataset,
    n_unique: usize,
    n_near_dup: usize,
    epsilon: f32,
    seed: u64,
) -> Vec<QueryEntry> {
    let mut rng = Rng::new(seed);
    let unique_queries: Vec<Vec<f32>> = (0..n_unique).map(|_| rng.unit_vec(dataset.dim)).collect();

    // Build a per-topic list of near-duplicate counts.
    let mut dup_counts = vec![0usize; n_unique];
    for _ in 0..n_near_dup {
        let src_idx = (rng.next_u64() as usize) % n_unique;
        dup_counts[src_idx] += 1;
    }

    // Interleave: for each unique query, immediately emit its near-duplicates.
    // This reflects real agent workloads where the same topic is revisited
    // within a short window rather than arbitrarily later.
    let mut entries: Vec<QueryEntry> = Vec::with_capacity(n_unique + n_near_dup);
    let mut rng2 = Rng::new(seed.wrapping_add(0xDEAD_BEEF));
    for (i, q) in unique_queries.iter().enumerate() {
        let nn = brute_force_nn(q, &dataset.vectors);
        entries.push(QueryEntry {
            vec: q.clone(),
            is_near_dup: false,
            nn_idx: nn,
        });
        for _ in 0..dup_counts[i] {
            let dup = rng2.perturb(q, epsilon);
            let nn_d = brute_force_nn(&dup, &dataset.vectors);
            entries.push(QueryEntry {
                vec: dup,
                is_near_dup: true,
                nn_idx: nn_d,
            });
        }
    }

    entries
}

/// Brute-force nearest neighbour (cosine distance) in the dataset.
pub fn brute_force_nn(q: &[f32], vectors: &[Vec<f32>]) -> u32 {
    let mut best_sim = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for (i, v) in vectors.iter().enumerate() {
        let sim: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        if sim > best_sim {
            best_sim = sim;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Top-k brute-force nearest neighbours (cosine distance).
pub fn brute_force_topk(q: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut sims: Vec<(f32, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let sim: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (sim, i as u32)
        })
        .collect();

    // Partial sort: bring top-k to the front.
    let k = k.min(sims.len());
    sims.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
    sims[..k].iter().map(|(_, id)| *id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn unit_vec_has_unit_norm() {
        let mut rng = Rng::new(1);
        for _ in 0..100 {
            let v = rng.unit_vec(128);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
        }
    }

    #[test]
    fn perturb_is_close_to_base() {
        let mut rng = Rng::new(2);
        let base = rng.unit_vec(64);
        let dup = rng.perturb(&base, 0.05);
        let sim: f32 = base.iter().zip(dup.iter()).map(|(a, b)| a * b).sum();
        assert!(
            sim > 0.95,
            "near-duplicate should have cosine sim > 0.95; got {}",
            sim
        );
    }

    #[test]
    fn brute_force_nn_is_consistent() {
        let ds = Dataset::random(200, 32, 99);
        // Query identical to the first vector must return index 0.
        let nn = brute_force_nn(&ds.vectors[0], &ds.vectors);
        assert_eq!(nn, 0);
    }

    #[test]
    fn workload_contains_near_dups() {
        let ds = Dataset::random(500, 64, 7);
        let workload = build_workload(&ds, 100, 50, 0.05, 13);
        let nd_count = workload.iter().filter(|e| e.is_near_dup).count();
        assert_eq!(nd_count, 50);
    }
}
