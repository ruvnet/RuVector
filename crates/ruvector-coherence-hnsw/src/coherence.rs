//! Traversal-direction coherence scoring.
//!
//! The **traversal coherence** of a candidate node C with respect to an entry
//! point E and a query Q is the cosine similarity between:
//!   - the direction E → C  (where we moved)
//!   - the direction E → Q  (where we want to go)
//!
//! When coherence ≈ 1.0 the candidate lies directly toward the query.
//! When coherence ≈ 0.0 the movement is perpendicular.
//! When coherence < 0.0 we moved away from the query.
//!
//! This is orthogonal to the distance metric: a node can be close to the
//! query (small L2) but have low coherence if the path to reach it was
//! circuitous. Skipping the neighborhood expansion of low-coherence nodes
//! prunes traversal without necessarily discarding the node as a result.

/// Cosine similarity between two vectors. Returns 0.0 if either has zero norm.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Traversal-direction coherence of candidate C relative to entry E toward Q.
///
/// Subtracts E from both C and Q to get displacement vectors, then computes
/// the cosine of the angle between them.
pub fn traversal_coherence(entry: &[f32], candidate: &[f32], query: &[f32]) -> f32 {
    // ec = candidate − entry
    // eq = query − entry
    // We avoid allocating by computing dot products directly.
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
        // Entry and candidate are coincident, or entry and query are coincident.
        // Treat as maximally coherent so we don't accidentally prune.
        1.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Adaptive threshold update rule.
///
/// Raises the threshold when progress is made (new best distance < prev best),
/// lowers it slightly when no progress is made. The threshold drifts toward
/// a value where the gate is aggressive when the beam is converging and
/// permissive when the beam is exploring.
#[inline]
pub fn update_adaptive_threshold(
    threshold: f32,
    prev_best: f32,
    new_best: f32,
    adaptation_rate: f32,
    max_threshold: f32,
) -> f32 {
    if new_best < prev_best {
        (threshold + adaptation_rate).min(max_threshold)
    } else {
        (threshold - adaptation_rate * 0.5).max(0.0)
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
    fn perpendicular_is_zero() {
        let entry = vec![0.0f32; 4];
        let candidate = vec![1.0, 0.0, 0.0, 0.0];
        let query = vec![0.0, 1.0, 0.0, 0.0];
        let c = traversal_coherence(&entry, &candidate, &query);
        assert!(c.abs() < 1e-5, "expected 0.0, got {c}");
    }

    #[test]
    fn opposite_direction_is_negative() {
        let entry = vec![0.0f32; 4];
        let candidate = vec![1.0, 0.0, 0.0, 0.0];
        let query = vec![-1.0, 0.0, 0.0, 0.0];
        let c = traversal_coherence(&entry, &candidate, &query);
        assert!(c < -0.9, "expected near -1.0, got {c}");
    }

    #[test]
    fn coincident_entry_returns_one() {
        let entry = vec![0.5f32; 4];
        let candidate = vec![0.5, 0.5, 0.5, 0.5]; // same as entry
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let c = traversal_coherence(&entry, &candidate, &query);
        assert_eq!(c, 1.0);
    }

    #[test]
    fn adaptive_threshold_rises_on_progress() {
        let t = update_adaptive_threshold(0.2, 10.0, 8.0, 0.05, 0.8);
        assert!((t - 0.25).abs() < 1e-5);
    }

    #[test]
    fn adaptive_threshold_falls_on_stagnation() {
        let t = update_adaptive_threshold(0.2, 10.0, 10.0, 0.05, 0.8);
        assert!((t - 0.175).abs() < 1e-5);
    }
}
