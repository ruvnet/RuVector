//! IVF-partition similarity join.
//!
//! Partitions both sets A and B into K clusters via Lloyd's k-means.
//! Pairs are compared only within the same cluster and optionally between
//! adjacent probe clusters, reducing average comparisons from O(|A|·|B|)
//! to O(|A|·|B|/K + |A|·|B|·probe/K²).
//!
//! Reference: Jégou et al., "Product Quantization for Nearest Neighbor
//! Search" (TPAMI 2011) for IVF-style candidate filtering.

use crate::{cosine_similarity, dataset::Lcg64, Pair, SimJoin};

/// IVF-partition similarity join.
///
/// Parameters:
/// - `n_clusters`: Number of partition cells K.
/// - `n_probe`: Number of cells to probe per query vector (1 = exact partition only).
pub struct IvfJoin {
    n_clusters: usize,
    n_probe: usize,
    seed: u64,
}

impl IvfJoin {
    pub fn new(n_clusters: usize, n_probe: usize, seed: u64) -> Self {
        Self {
            n_clusters,
            n_probe,
            seed,
        }
    }

    /// Run Lloyd's k-means for `iters` iterations.
    fn kmeans(
        vectors: &[Vec<f32>],
        k: usize,
        iters: usize,
        rng: &mut Lcg64,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        let n = vectors.len();
        let dims = vectors[0].len();
        let k = k.min(n);

        // Initialise centroids via random subset
        let mut centroid_idxs: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + rng.rand_usize(n - i);
            centroid_idxs.swap(i, j);
        }
        let mut centroids: Vec<Vec<f32>> = centroid_idxs[..k]
            .iter()
            .map(|&i| vectors[i].clone())
            .collect();

        let mut assignments = vec![0usize; n];

        for _iter in 0..iters {
            // Assignment step
            for (i, v) in vectors.iter().enumerate() {
                let best = (0..k)
                    .map(|c| (c, cosine_similarity(v, &centroids[c])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(c, _)| c)
                    .unwrap_or(0);
                assignments[i] = best;
            }

            // Update step: recompute centroids as mean of assigned vectors
            let mut new_centroids = vec![vec![0.0_f32; dims]; k];
            let mut counts = vec![0usize; k];
            for (i, &c) in assignments.iter().enumerate() {
                for (d, x) in vectors[i].iter().enumerate() {
                    new_centroids[c][d] += x;
                }
                counts[c] += 1;
            }
            for (c, cent) in new_centroids.iter_mut().enumerate() {
                if counts[c] > 0 {
                    let cnt = counts[c] as f32;
                    cent.iter_mut().for_each(|x| *x /= cnt);
                    // L2-normalise centroid for cosine-space IVF
                    let norm: f32 = cent.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-9 {
                        cent.iter_mut().for_each(|x| *x /= norm);
                    }
                }
            }
            centroids = new_centroids;
        }

        (centroids, assignments)
    }

    /// For a vector v, return the indices of the `n_probe` nearest centroids.
    fn nearest_cells(v: &[f32], centroids: &[Vec<f32>], n_probe: usize) -> Vec<usize> {
        let mut sims: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity(v, c)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.into_iter().take(n_probe).map(|(i, _)| i).collect()
    }
}

impl SimJoin for IvfJoin {
    fn join(&self, a: &[Vec<f32>], b: &[Vec<f32>], threshold: f32) -> Vec<Pair> {
        let mut rng = Lcg64::new(self.seed);

        // Build partitions from A
        let (centroids, a_assignments) =
            Self::kmeans(a, self.n_clusters, /*iters=*/ 10, &mut rng);

        // Group A indices by cluster
        let mut a_cells: Vec<Vec<usize>> = vec![Vec::new(); centroids.len()];
        for (ai, &c) in a_assignments.iter().enumerate() {
            a_cells[c].push(ai);
        }

        let mut pair_set: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();

        // For each b vector, probe its nearest cells and compare against those A vectors
        for (bi, bv) in b.iter().enumerate() {
            let cells = Self::nearest_cells(bv, &centroids, self.n_probe);
            for cell in cells {
                for &ai in &a_cells[cell] {
                    pair_set.insert((ai, bi));
                }
            }
        }

        // Verify candidates
        let mut pairs: Vec<Pair> = pair_set
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
    fn ivf_recall_above_baseline() {
        let ds = ClusteredDataset::generate(100, 64, 4, 0.05, 55);
        let gt = BruteJoin.join(&ds.a, &ds.b, ds.threshold);
        let ivf = IvfJoin::new(5, 2, 42);
        let found = ivf.join(&ds.a, &ds.b, ds.threshold);
        let r = recall(&found, &gt);
        assert!(r >= 0.70, "IVF recall {r:.3} too low");
    }

    #[test]
    fn ivf_does_not_add_false_pairs() {
        let ds = ClusteredDataset::generate(80, 32, 3, 0.05, 77);
        let ivf = IvfJoin::new(4, 2, 88);
        let found = ivf.join(&ds.a, &ds.b, ds.threshold);
        for p in &found {
            let sim = cosine_similarity(&ds.a[p.a_idx], &ds.b[p.b_idx]);
            assert!(
                sim >= ds.threshold - 1e-5,
                "false pair ({},{}): sim={sim:.4}",
                p.a_idx,
                p.b_idx
            );
        }
    }
}
