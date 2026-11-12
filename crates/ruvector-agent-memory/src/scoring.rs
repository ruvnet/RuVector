//! Vector scoring primitives: cosine similarity, L2 distance, normalization.

/// Rank finite scores first, then use a total order for exceptional values.
/// This preserves descending order for valid cosine/importance scores without
/// allowing a non-finite input to panic a search or compaction sort.
pub(crate) fn compare_scores_desc(a: f32, b: f32) -> std::cmp::Ordering {
    (!a.is_finite())
        .cmp(&(!b.is_finite()))
        .then_with(|| b.total_cmp(&a))
}

/// Compute the dot product of two equal-length slices.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity in [-1, 1]; returns 0.0 for zero vectors.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

/// Return a unit-length copy of `v`, or the zero vector.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n < 1e-9 {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Coherence of a memory vector against a context window.
///
/// Returns the *maximum* cosine similarity between `v` and any query in `context`.
/// An empty context window returns 0.0.
pub fn coherence_score(v: &[f32], context: &[Vec<f32>]) -> f32 {
    context
        .iter()
        .map(|q| cosine_sim(v, q))
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn coherence_picks_max() {
        let v = vec![1.0, 0.0];
        let ctx = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let s = coherence_score(&v, &ctx);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_order_puts_finite_values_before_nonfinite_values() {
        let mut scores = [f32::NAN, f32::INFINITY, -0.5, 0.5, f32::NEG_INFINITY];
        scores.sort_by(|a, b| compare_scores_desc(*a, *b));
        assert_eq!(scores[..2], [0.5, -0.5]);
        assert!(scores[2..].iter().all(|s| !s.is_finite()));
    }
}
