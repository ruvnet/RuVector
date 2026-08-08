//! K-means clustering with cohesion scoring.
//!
//! Cohesion: mean cosine similarity of cluster members to their centroid.
//! Higher cohesion ⟹ tighter cluster ⟹ more reliable neighbourhood.

use crate::{cosine_sim, l2_sq};

/// Result of a k-means run.
pub struct KMeansResult {
    /// Centroid vectors (length = k).
    pub centroids: Vec<Vec<f32>>,
    /// Cluster id for each input vector (length = n).
    pub assignments: Vec<usize>,
    /// Per-cluster cohesion ∈ [−1, 1]; higher is tighter (length = k).
    pub cohesion: Vec<f32>,
    /// Number of members per cluster (length = k).
    pub cluster_sizes: Vec<usize>,
}

/// Run Lloyd's k-means for `iters` iterations.
pub fn kmeans(vectors: &[Vec<f32>], k: usize, iters: usize) -> KMeansResult {
    let n = vectors.len();

    if n == 0 {
        assert_eq!(k, 0, "k must be 0 when clustering an empty dataset");
        return KMeansResult {
            centroids: Vec::new(),
            assignments: Vec::new(),
            cohesion: Vec::new(),
            cluster_sizes: Vec::new(),
        };
    }

    assert!(k > 0, "k must be greater than 0 for a non-empty dataset");
    assert!(k <= n, "k must not exceed the number of vectors");
    let dim = vectors[0].len();
    assert!(
        vectors.iter().all(|vector| vector.len() == dim),
        "all vectors must have the same dimension"
    );

    // Initialise centroids from distinct points (deterministic k-means++ max-dist).
    let init_ids = crate::dataset::initial_centroid_indices(vectors, k);
    let mut centroids: Vec<Vec<f32>> = init_ids.iter().map(|&i| vectors[i].clone()).collect();
    let mut assignments;

    for _iter in 0..iters {
        // Assignment step.
        assignments = assign_to_nearest(vectors, &centroids);

        // Update step: recompute centroids as member means.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, vec) in vectors.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (d, x) in vec.iter().enumerate() {
                sums[c][d] += x;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                centroids[c] = sums[c].iter().map(|s| s / cnt).collect();
            }
        }
    }

    // The final Lloyd update moves centroids after the loop's assignment step.
    // Reassign once more so every returned membership, cohesion value, and
    // inverted-list entry is consistent with the returned centroids. This also
    // gives zero-iteration runs valid assignments against initial centroids.
    assignments = assign_to_nearest(vectors, &centroids);

    // Compute per-cluster cohesion from final-centroid assignments.
    let cohesion = compute_cohesion(vectors, &assignments, &centroids, k);
    let cluster_sizes: Vec<usize> = (0..k)
        .map(|c| assignments.iter().filter(|&&a| a == c).count())
        .collect();

    KMeansResult {
        centroids,
        assignments,
        cohesion,
        cluster_sizes,
    }
}

/// Assign every vector to its nearest centroid with deterministic tie-breaking.
fn assign_to_nearest(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    debug_assert!(!centroids.is_empty());
    vectors
        .iter()
        .map(|vector| {
            let mut best = 0usize;
            let mut best_dist = l2_sq(vector, &centroids[0]);
            for (cluster, centroid) in centroids.iter().enumerate().skip(1) {
                let dist = l2_sq(vector, centroid);
                if dist < best_dist {
                    best = cluster;
                    best_dist = dist;
                }
            }
            best
        })
        .collect()
}

/// Mean cosine similarity of each cluster's members to their centroid.
fn compute_cohesion(
    vectors: &[Vec<f32>],
    assignments: &[usize],
    centroids: &[Vec<f32>],
    k: usize,
) -> Vec<f32> {
    let mut sums = vec![0.0f32; k];
    let mut counts = vec![0usize; k];
    for (i, vec) in vectors.iter().enumerate() {
        let c = assignments[i];
        sums[c] += cosine_sim(vec, &centroids[c]);
        counts[c] += 1;
    }
    (0..k)
        .map(|c| {
            if counts[c] > 0 {
                sums[c] / counts[c] as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Build per-cluster member lists (vector ids) from assignments.
pub fn build_inverted_lists(assignments: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut lists = vec![Vec::new(); k];
    for (i, &c) in assignments.iter().enumerate() {
        lists[c].push(i);
    }
    lists
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::generate_vectors;

    #[test]
    fn kmeans_partitions_all_vectors() {
        let vecs = generate_vectors(200, 16, 1);
        let result = kmeans(&vecs, 5, 10);
        assert_eq!(result.assignments.len(), 200);
        let total: usize = result.cluster_sizes.iter().sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn cohesion_in_range() {
        let vecs = generate_vectors(200, 16, 2);
        let result = kmeans(&vecs, 5, 10);
        for &c in &result.cohesion {
            assert!((-1.0..=1.0).contains(&c), "cohesion out of range: {c}");
        }
    }

    #[test]
    fn tight_clusters_have_higher_cohesion() {
        // Generate two tight clusters (low-variance within, high-variance between).
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        // Cluster A: near [1, 0, 0, ...]
        for i in 0..100usize {
            let mut v = vec![0.0f32; 32];
            v[0] = 1.0 + (i as f32) * 0.001;
            vecs.push(v);
        }
        // Cluster B: near [-1, 0, 0, ...]
        for i in 0..100usize {
            let mut v = vec![0.0f32; 32];
            v[0] = -1.0 - (i as f32) * 0.001;
            vecs.push(v);
        }
        let result = kmeans(&vecs, 2, 20);
        // Both clusters should have very high cohesion (> 0.9).
        for &c in &result.cohesion {
            assert!(
                c > 0.9,
                "expected high cohesion for tight clusters, got {c}"
            );
        }
    }

    #[test]
    fn final_assignments_are_nearest_to_final_centroids() {
        // After one update the centroids are 2.5 and 7.0. The point at 5.0
        // belonged to centroid 0 before that update, but is nearest centroid 1
        // afterwards. This specifically catches stale final assignments.
        let vecs = vec![
            vec![0.0],
            vec![5.0],
            vec![6.0],
            vec![6.0],
            vec![6.0],
            vec![10.0],
        ];
        let result = kmeans(&vecs, 2, 1);

        assert_eq!(result.assignments[1], 1, "point 5.0 must be reassigned");
        for (vector, &assigned) in vecs.iter().zip(&result.assignments) {
            let assigned_dist = l2_sq(vector, &result.centroids[assigned]);
            for centroid in &result.centroids {
                assert!(
                    assigned_dist <= l2_sq(vector, centroid),
                    "assignment {assigned} is not nearest for {vector:?}"
                );
            }
        }
    }

    #[test]
    fn empty_dataset_with_zero_clusters_is_supported() {
        let result = kmeans(&[], 0, 10);
        assert!(result.centroids.is_empty());
        assert!(result.assignments.is_empty());
        assert!(result.cohesion.is_empty());
        assert!(result.cluster_sizes.is_empty());
    }

    #[test]
    fn zero_iterations_still_assigns_to_initial_centroids() {
        let vecs = vec![vec![0.0], vec![2.0], vec![10.0]];
        let result = kmeans(&vecs, 2, 0);

        assert_eq!(result.assignments, vec![0, 0, 1]);
        assert_eq!(result.cluster_sizes, vec![2, 1]);
    }
}
