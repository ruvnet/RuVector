//! Spectral bisection via the Fiedler vector.
//!
//! The Fiedler vector is the eigenvector corresponding to the **second smallest**
//! eigenvalue of the graph Laplacian L = D − W. Partitioning by the sign of the
//! Fiedler vector is the continuous relaxation of the minimum balanced graph cut
//! (Cheeger inequality). It produces two groups that minimise inter-group edges
//! relative to intra-group edge weight.
//!
//! ## Algorithm
//!
//! 1. Represent the random-walk matrix T = D⁻¹W (transition probabilities).
//! 2. Run power iteration to find the eigenvector with the **second largest**
//!    eigenvalue of T (≡ second smallest of L/D).
//! 3. Deflate against the stationary distribution π_i = d_i / Σd to remove the
//!    leading eigenvector (all-ones in the unweighted case).
//! 4. Bisect at the median of the resulting vector for balanced partitions.
//!
//! For recursive partitioning, apply `fiedler_bisect()` repeatedly on each sub-group.

use crate::graph::KnnGraph;

const POWER_ITERS: usize = 150;

/// Compute the Fiedler vector of the graph Laplacian.
/// Returns a vector of length `graph.n`.
pub fn fiedler_vector(graph: &KnnGraph) -> Vec<f32> {
    let n = graph.n;
    if n <= 1 {
        return vec![0.0; n];
    }

    let d_sum: f32 = graph.degree.iter().sum();

    // Initialise with a deterministic non-constant vector.
    let mut v: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 23) as f32 - 11.0).collect();

    // Deflate initial vector against stationary distribution and normalise.
    deflate_stationary(&mut v, &graph.degree, d_sum);
    normalize(&mut v);

    for _ in 0..POWER_ITERS {
        // Random-walk step: v_new[i] = Σ_j W[i,j] * v[j] / d[i]
        let mut v_new = vec![0.0f32; n];
        for i in 0..n {
            for &(j, w) in &graph.adj[i] {
                v_new[i] += w * v[j];
            }
            v_new[i] /= graph.degree[i];
        }

        // Deflate to isolate the second eigenvector (remove λ₁ = 1 component).
        deflate_stationary(&mut v_new, &graph.degree, d_sum);

        // Normalise to prevent numerical drift.
        let norm = normalize(&mut v_new);
        if norm < 1e-12 {
            break;
        }

        v = v_new;
    }

    v
}

/// Bisect all `graph.n` nodes into two groups using the Fiedler vector.
/// Returns a Vec<usize> of length `graph.n` with values 0 or 1.
/// Splits at the **median** for a balanced partition.
pub fn fiedler_bisect(graph: &KnnGraph) -> Vec<usize> {
    let v = fiedler_vector(graph);
    let mut sorted = v.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[graph.n / 2];
    v.iter().map(|&x| if x >= median { 1 } else { 0 }).collect()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Remove the projection of `v` onto the constant vector in the π-weighted
/// inner product (π_i = d_i / d_sum).  After this, `v ⊥ 1` w.r.t. ⟨·,·⟩_π.
fn deflate_stationary(v: &mut [f32], degree: &[f32], d_sum: f32) {
    // π-weighted mean: ⟨v, 1⟩_π = Σ_i (d_i / d_sum) v_i
    let mean: f32 = v
        .iter()
        .zip(degree.iter())
        .map(|(vi, di)| vi * di / d_sum)
        .sum();
    for vi in v.iter_mut() {
        *vi -= mean;
    }
}

/// Normalise `v` in place; return the original L2 norm.
fn normalize(v: &mut [f32]) -> f32 {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::KnnGraph;

    /// Two tight clusters separated in feature space — Fiedler bisect must
    /// assign all members of each cluster to the same side.
    #[test]
    fn fiedler_bisects_two_clusters() {
        // Cluster A: vectors concentrated near (1, 0, 0, …)
        // Cluster B: vectors concentrated near (0, 1, 0, …)
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for i in 0..4usize {
            let mut v = vec![0.0f32; 8];
            v[0] = 1.0 + i as f32 * 0.05;
            v[1] = 0.05;
            vecs.push(v);
        }
        for i in 0..4usize {
            let mut v = vec![0.0f32; 8];
            v[1] = 1.0 + i as f32 * 0.05;
            v[0] = 0.05;
            vecs.push(v);
        }
        let g = KnnGraph::build(&vecs, 3);
        let labels = fiedler_bisect(&g);
        assert_eq!(labels.len(), 8);

        let a_uniform = labels[0..4].windows(2).all(|w| w[0] == w[1]);
        let b_uniform = labels[4..8].windows(2).all(|w| w[0] == w[1]);
        let clusters_differ = labels[0] != labels[4];

        assert!(
            a_uniform,
            "cluster A labels not uniform: {:?}",
            &labels[0..4]
        );
        assert!(
            b_uniform,
            "cluster B labels not uniform: {:?}",
            &labels[4..8]
        );
        assert!(clusters_differ, "both clusters got same label");
    }

    #[test]
    fn fiedler_vector_correct_length() {
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v = vec![0.1f32; 8];
                v[i % 8] += 1.0;
                v
            })
            .collect();
        let g = KnnGraph::build(&vecs, 3);
        let fv = fiedler_vector(&g);
        assert_eq!(fv.len(), g.n);
    }

    #[test]
    fn deflate_removes_mean() {
        let degree = vec![1.0f32; 4];
        let d_sum = 4.0;
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        deflate_stationary(&mut v, &degree, d_sum);
        let mean: f32 = v.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "mean after deflation should be ~0, got {mean}"
        );
    }
}
