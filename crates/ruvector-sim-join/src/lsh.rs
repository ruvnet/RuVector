//! LSH-bucket similarity join.
//!
//! Uses random-hyperplane locality-sensitive hashing to group vectors into
//! buckets. Pairs are compared only when they share a bucket in at least one
//! hash table, reducing the O(n²) pairwise checks to O(n · bucket_size).
//!
//! Reference: Indyk & Motwani, "Approximate Nearest Neighbors: Towards
//! Removing the Curse of Dimensionality" (STOC 1998).

use crate::{cosine_similarity, dataset::Lcg64, Pair, SimJoin};
use std::collections::HashMap;

/// LSH similarity join using random hyperplane families.
///
/// Parameters:
/// - `hash_bits`: Number of random hyperplanes per table (bucket resolution).
/// - `tables`: Number of independent hash tables (collision probability).
/// - `dims`: Dimensionality of input vectors.
pub struct LshJoin {
    tables: Vec<LshTable>,
    hash_bits: usize,
}

struct LshTable {
    /// Random unit hyperplane normals: [hash_bits × dims]
    hyperplanes: Vec<Vec<f32>>,
}

impl LshTable {
    fn new(hash_bits: usize, dims: usize, rng: &mut Lcg64) -> Self {
        let hyperplanes = (0..hash_bits)
            .map(|_| {
                let mut h: Vec<f32> = (0..dims).map(|_| rng.randn()).collect();
                let norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-9 {
                    h.iter_mut().for_each(|x| *x /= norm);
                }
                h
            })
            .collect();
        Self { hyperplanes }
    }

    /// Compute the hash code (u64 bitmask) for vector v.
    fn hash(&self, v: &[f32]) -> u64 {
        let mut code: u64 = 0;
        for (bit, h) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = v.iter().zip(h.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                code |= 1u64 << bit;
            }
        }
        code
    }
}

impl LshJoin {
    /// Create a new LshJoin with the given parameters.
    pub fn new(hash_bits: usize, tables: usize, dims: usize, seed: u64) -> Self {
        let mut rng = Lcg64::new(seed);
        let tables = (0..tables)
            .map(|_| LshTable::new(hash_bits, dims, &mut rng))
            .collect();
        Self { tables, hash_bits }
    }

    fn hash_set<'a>(table: &LshTable, vectors: &'a [Vec<f32>]) -> HashMap<u64, Vec<usize>> {
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, v) in vectors.iter().enumerate() {
            buckets.entry(table.hash(v)).or_default().push(i);
        }
        buckets
    }
}

impl SimJoin for LshJoin {
    fn join(&self, a: &[Vec<f32>], b: &[Vec<f32>], threshold: f32) -> Vec<Pair> {
        // Candidate pair set: (a_idx, b_idx) discovered via any table
        let mut candidate_set: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();

        for table in &self.tables {
            let a_buckets = Self::hash_set(table, a);
            let b_buckets = Self::hash_set(table, b);

            for (code, a_idxs) in &a_buckets {
                if let Some(b_idxs) = b_buckets.get(code) {
                    for &ai in a_idxs {
                        for &bi in b_idxs {
                            candidate_set.insert((ai, bi));
                        }
                    }
                }
            }
        }

        // Verify candidates against threshold
        let mut pairs: Vec<Pair> = candidate_set
            .into_iter()
            .filter_map(|(ai, bi)| {
                let sim = cosine_similarity(&a[ai], &b[bi]);
                if sim >= threshold {
                    Some(Pair {
                        a_idx: ai,
                        b_idx: bi,
                        similarity: sim,
                    })
                } else {
                    None
                }
            })
            .collect();

        pairs.sort_by(|p, q| p.a_idx.cmp(&q.a_idx).then(p.b_idx.cmp(&q.b_idx)));
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute::BruteJoin;
    use crate::dataset::ClusteredDataset;
    use crate::recall;

    #[test]
    fn lsh_finds_majority_of_brute_pairs() {
        let ds = ClusteredDataset::generate(100, 64, 4, 0.05, 99);
        let gt = BruteJoin.join(&ds.a, &ds.b, ds.threshold);
        let lsh = LshJoin::new(8, 6, 64, 123);
        let found = lsh.join(&ds.a, &ds.b, ds.threshold);
        let r = recall(&found, &gt);
        assert!(r >= 0.65, "LSH recall {r:.3} too low");
    }

    #[test]
    fn lsh_no_false_negatives_on_identical() {
        let v = vec![1.0_f32 / 8.0_f32.sqrt(); 8];
        let a = vec![v.clone()];
        let b = vec![v];
        let lsh = LshJoin::new(4, 4, 8, 0);
        let pairs = lsh.join(&a, &b, 0.99);
        assert!(!pairs.is_empty(), "identical vectors must be found");
    }
}
