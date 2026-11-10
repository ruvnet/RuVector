//! Traversal-direction coherence scoring, reused from `ruvector-coherence-hnsw`.
//!
//! Coherence of candidate C relative to entry E toward query Q is the cosine
//! similarity between the direction E->C (where traversal moved) and E->Q
//! (where it should be heading). Computed on the exact f32 vectors: direction
//! is a geometric notion that u8 quantization would distort too coarsely to
//! be a useful gate signal.

#[inline]
pub fn traversal_coherence(entry: &[f32], candidate: &[f32], query: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut nec = 0.0f32;
    let mut neq = 0.0f32;
    for ((&e, &c), &q) in entry.iter().zip(candidate.iter()).zip(query.iter()) {
        let ec = c - e;
        let eq = q - e;
        dot += ec * eq;
        nec += ec * ec;
        neq += eq * eq;
    }
    let denom = nec.sqrt() * neq.sqrt();
    if denom < 1e-8 {
        1.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_direction_is_one() {
        let entry = vec![0.0f32; 4];
        let candidate = vec![1.0, 0.0, 0.0, 0.0];
        let query = vec![2.0, 0.0, 0.0, 0.0];
        let c = traversal_coherence(&entry, &candidate, &query);
        assert!((c - 1.0).abs() < 1e-5, "expected 1.0, got {c}");
    }

    #[test]
    fn opposite_direction_is_negative() {
        let entry = vec![0.0f32; 4];
        let candidate = vec![1.0, 0.0, 0.0, 0.0];
        let query = vec![-1.0, 0.0, 0.0, 0.0];
        let c = traversal_coherence(&entry, &candidate, &query);
        assert!(c < -0.9, "expected near -1.0, got {c}");
    }
}
