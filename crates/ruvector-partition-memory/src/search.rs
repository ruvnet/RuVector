//! Shared brute-force cosine top-k search.
//!
//! N in this crate's benchmarks stays in the low thousands, so brute force
//! is the right tool: it is the ground-truth oracle everything else is
//! measured against, and introducing an approximate index here would let
//! the evaluator's own approximation error leak into the acceptance test.

/// Cosine similarity between two equal-length vectors. Returns 0.0 for a
/// zero vector rather than panicking; inputs in this crate are always
/// normalized on construction so this only guards degenerate edges.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Brute-force top-k by cosine similarity. `candidates` is `(id, embedding)`.
/// Ties broken by ascending id for determinism.
pub fn top_k_by_cosine(query: &[f32], candidates: &[(u64, &[f32])], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = candidates
        .iter()
        .map(|(id, emb)| (*id, cosine(query, emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Recall@k of `retrieved` against `ground_truth`, both treated as sets.
pub fn recall_at_k(retrieved: &[u64], ground_truth: &[u64]) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let retrieved_set: std::collections::HashSet<_> = retrieved.iter().collect();
    let hits = ground_truth
        .iter()
        .filter(|id| retrieved_set.contains(id))
        .count();
    hits as f64 / ground_truth.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![0.6, 0.8];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_k_orders_by_similarity() {
        let q = vec![1.0, 0.0];
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![0.9, 0.1];
        let cands = vec![
            (1u64, a.as_slice()),
            (2u64, b.as_slice()),
            (3u64, c.as_slice()),
        ];
        let top = top_k_by_cosine(&q, &cands, 2);
        assert_eq!(top, vec![1, 3]);
    }

    #[test]
    fn recall_full_overlap_is_one() {
        assert_eq!(recall_at_k(&[1, 2, 3], &[3, 2, 1]), 1.0);
    }

    #[test]
    fn recall_no_overlap_is_zero() {
        assert_eq!(recall_at_k(&[4, 5, 6], &[1, 2, 3]), 0.0);
    }
}
