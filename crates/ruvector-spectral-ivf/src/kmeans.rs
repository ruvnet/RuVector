//! Simple Lloyd's k-means for the IVF baseline variant.

use crate::distance::l2_sq;

/// Run k-means with Lloyd's algorithm.
///
/// Returns `(assignments, centroids)` where `assignments[i]` is the cluster
/// index (0..k) for vector `i`, and `centroids` holds the final centroid
/// positions.
pub fn kmeans(
    vectors: &[Vec<f32>],
    k: usize,
    iters: usize,
    seed: u64,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let n = vectors.len();
    let dim = vectors[0].len();
    assert!(k <= n, "k ({k}) must be ≤ n ({n})");

    // Deterministic initialisation: evenly spaced indices.
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|c| {
            // Use a simple LCG to select seed indices.
            let idx = lcg(seed.wrapping_add(c as u64), n);
            vectors[idx].clone()
        })
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..iters {
        // Assignment step.
        let mut changed = false;
        for i in 0..n {
            let best = nearest_centroid(&vectors[i], &centroids);
            if best != assignments[i] {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += vectors[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
            // If a centroid is empty, re-seed from the vector with maximum
            // distance to its current centroid.
            else {
                let farthest = (0..n)
                    .max_by(|&a, &b| {
                        let da = l2_sq(&vectors[a], &centroids[assignments[a]]);
                        let db = l2_sq(&vectors[b], &centroids[assignments[b]]);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                centroids[c] = vectors[farthest].clone();
            }
        }
    }

    (assignments, centroids)
}

fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Very simple LCG for seeded index selection (no external dep).
fn lcg(seed: u64, n: usize) -> usize {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (x >> 33) as usize % n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_two_clusters() {
        // Ten vectors near (1,0) and ten near (0,1) → k=2 should recover them.
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for i in 0..10usize {
            vecs.push(vec![1.0 + i as f32 * 0.01, 0.0]);
        }
        for i in 0..10usize {
            vecs.push(vec![0.0, 1.0 + i as f32 * 0.01]);
        }
        let (asgn, _) = kmeans(&vecs, 2, 50, 42);
        // All first 10 should share one cluster; all last 10 the other.
        let a_label = asgn[0];
        let b_label = asgn[10];
        assert_ne!(a_label, b_label, "two clusters should get different labels");
        assert!(asgn[0..10].iter().all(|&x| x == a_label));
        assert!(asgn[10..20].iter().all(|&x| x == b_label));
    }

    #[test]
    fn all_assignments_in_range() {
        let vecs: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32; 4]).collect();
        let (asgn, _) = kmeans(&vecs, 4, 20, 7);
        for a in asgn {
            assert!(a < 4, "assignment {a} out of range");
        }
    }
}
