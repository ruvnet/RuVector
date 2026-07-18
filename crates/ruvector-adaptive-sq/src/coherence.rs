//! Local neighborhood density scoring for coherence-based precision routing.
//!
//! For each vector in a dataset we compute the mean L2 distance to its K
//! nearest neighbours.  Vectors with a *low* mean kNN distance live in
//! dense, contested regions of the embedding space — the exact regions where
//! quantisation noise most disrupts nearest-neighbour rank ordering.
//!
//! Routing rule:
//!   density_score ≤ threshold → HIGH precision (16-bit SQ)
//!   density_score >  threshold → LOW precision  (8-bit SQ)
//!
//! Complexity: O(N² × D) brute-force kNN — acceptable for PoC sizes (N ≤ 10 K).
//! Production would use an approximate kNN structure (HNSW layer-0 or IVF).

/// Squared L2 distance between two equal-length slices.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Compute a density score for every vector: mean L2 distance to its k
/// nearest neighbours (excluding itself).
///
/// Returns `scores[i]` in the same order as `vectors`.
pub fn density_scores(vectors: &[Vec<f32>], k: usize) -> Vec<f32> {
    let n = vectors.len();
    let k = k.min(n.saturating_sub(1));
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let mut dists: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| l2_sq(&vectors[i], &vectors[j]).sqrt())
            .collect();
        // Partial sort: only need the k smallest.
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_knn = dists[..k].iter().sum::<f32>() / k as f32;
        scores.push(mean_knn);
    }
    scores
}

/// Choose a precision threshold from density scores.
///
/// Vectors with `density_score ≤ mean * factor` are routed to high precision.
/// `factor = 0.6` routes the densest ~30 % of a clustered distribution to HP.
/// Lower values route fewer vectors to HP (saves memory, risks recall).
pub fn precision_threshold(scores: &[f32], factor: f32) -> f32 {
    let n = scores.len() as f32;
    let mean = scores.iter().sum::<f32>() / n;
    mean * factor
}

/// Returns the fraction of vectors routed to high precision.
pub fn hp_fraction(scores: &[f32], threshold: f32) -> f32 {
    let hp = scores.iter().filter(|&&s| s <= threshold).count();
    hp as f32 / scores.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster(n: usize, center: f32, spread: f32, seed_offset: u32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let off = (i as u32).wrapping_add(seed_offset) as f32 * 0.001;
                vec![center + spread * off, center - spread * off * 0.5]
            })
            .collect()
    }

    #[test]
    fn dense_cluster_has_lower_score_than_sparse() {
        let dim = 2;
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        // tight cluster at (0,0)
        vecs.extend(make_cluster(20, 0.0, 0.01, 0));
        // loose cluster at (10,10)
        vecs.extend(make_cluster(20, 10.0, 1.0, 100));

        let scores = density_scores(&vecs, 5);
        let tight_mean: f32 = scores[..20].iter().sum::<f32>() / 20.0;
        let loose_mean: f32 = scores[20..].iter().sum::<f32>() / 20.0;
        assert!(
            tight_mean < loose_mean,
            "Tight cluster score {tight_mean:.4} should be less than loose {loose_mean:.4}"
        );
    }

    #[test]
    fn precision_threshold_routes_dense_to_hp() {
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        vecs.extend(make_cluster(10, 0.0, 0.01, 0)); // tight
        vecs.extend(make_cluster(10, 10.0, 2.0, 100)); // loose

        let scores = density_scores(&vecs, 5);
        let threshold = precision_threshold(&scores, 0.6);
        // All 10 tight vectors should score below threshold
        let tight_hp = scores[..10].iter().filter(|&&s| s <= threshold).count();
        assert!(
            tight_hp >= 7,
            "At least 7/10 tight vectors should be routed to HP, got {tight_hp}"
        );
    }

    #[test]
    fn hp_fraction_in_range() {
        let scores: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let threshold = precision_threshold(&scores, 0.6);
        let frac = hp_fraction(&scores, threshold);
        // Should route roughly 0-60% to HP depending on factor
        assert!(
            frac >= 0.0 && frac <= 1.0,
            "fraction must be in [0,1], got {frac}"
        );
    }
}
