//! Brute-force exact kNN ground truth + recall, and the shared L2 helper.
//!
//! The triangle-inequality lower bound the kernel relies on holds for the **metric** L2, not
//! its square — so radii, centroid distances, and member distances all use true L2 (`sqrt`).
//! Keeping one `l2` here guarantees the bound and the ranking use an identical metric.

/// Euclidean (L2) distance between two equal-length vectors.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Exact top-`k` neighbour ids of `q` over `corpus` under L2 (ascending distance).
///
/// `q` may itself be a corpus point; self (distance 0) is **not** excluded — it lands in both
/// the oracle set and any contender's result, so it cancels and does not bias recall.
pub fn brute_force_topk(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (l2(q, v), i))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// recall@k = |truth_k ∩ got_k| / k. Tolerant of tie-reshuffling (set intersection, not order).
pub fn recall_at_k(truth: &[usize], got: &[usize], k: usize) -> f64 {
    let t: std::collections::HashSet<usize> = truth.iter().take(k).copied().collect();
    let hits = got.iter().take(k).filter(|g| t.contains(g)).count();
    hits as f64 / k.max(1) as f64
}
