//! Query difficulty estimation from initial search candidates.
//!
//! A query is "easy" when the nearest neighbour is clearly separated from
//! the rest (small d₁/d_k ratio).  A query is "hard" when many candidates
//! are equidistant (ratio approaches 1.0).
//!
//! The distance-ratio score is inexpensive to compute: it uses only the
//! first and last elements of the sorted candidate list returned by an
//! initial low-ef beam search, requiring zero additional distance computations.

use crate::AnnResult;

/// Estimate query difficulty from sorted search candidates.
///
/// Returns a score in `[0.0, 1.0]`:
/// - `0.0` → easy query (d₁ ≪ d_k, clear nearest neighbour)
/// - `1.0` → hard query (d₁ ≈ d_k, many equidistant neighbours)
///
/// Formula: `difficulty = d_nearest / d_farthest_in_k`
///
/// Rationale: when all k candidates are at similar distances, we may have
/// missed true nearest neighbours just outside the beam.  A larger ef is
/// warranted.  When d₁ ≪ d_k, the first result is a clear winner and a
/// small ef is sufficient.
pub fn distance_ratio_score(results: &[AnnResult]) -> f32 {
    if results.len() < 2 {
        return 0.0; // single result: trivially easy
    }
    let d_near = results[0].distance;
    let d_far = results[results.len() - 1].distance;
    if d_far < 1e-8 {
        return 1.0; // degenerate: all at origin, treat as hard
    }
    // Clamp to [0, 1] in case of floating-point noise
    (d_near / d_far).clamp(0.0, 1.0)
}

/// Predict ef from a difficulty score.
///
/// Maps `[0.0, 1.0]` → `[ef_min, ef_max]` with a linear schedule.
/// Easy queries (`difficulty ≈ 0`) get the minimum ef.
/// Hard queries (`difficulty ≈ 1`) get the maximum ef.
pub fn predict_ef(difficulty: f32, ef_min: usize, ef_max: usize) -> usize {
    let span = (ef_max - ef_min) as f32;
    let raw = ef_min as f32 + span * difficulty.clamp(0.0, 1.0);
    raw.round() as usize
}

/// A confidence-style complement: 1 − difficulty_score.
///
/// Useful for logging: `retrieval_confidence = 1 − difficulty`.
#[inline]
pub fn retrieval_confidence(results: &[AnnResult]) -> f32 {
    1.0 - distance_ratio_score(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnnResult;

    fn make(distances: &[f32]) -> Vec<AnnResult> {
        distances
            .iter()
            .enumerate()
            .map(|(i, &d)| AnnResult { id: i, distance: d })
            .collect()
    }

    #[test]
    fn easy_query_low_score() {
        // d₁ = 0.1, d_k = 10.0 → ratio = 0.01
        let r = make(&[0.1, 0.5, 1.0, 3.0, 10.0]);
        assert!(distance_ratio_score(&r) < 0.02, "expected near 0.01");
    }

    #[test]
    fn hard_query_high_score() {
        // d₁ ≈ d_k → ratio ≈ 1.0
        let r = make(&[1.00, 1.01, 1.02, 1.03, 1.04]);
        assert!(distance_ratio_score(&r) > 0.95, "expected near 1.0");
    }

    #[test]
    fn single_result_easy() {
        let r = make(&[5.0]);
        assert_eq!(distance_ratio_score(&r), 0.0);
    }

    #[test]
    fn predict_ef_bounds() {
        assert_eq!(predict_ef(0.0, 16, 64), 16);
        assert_eq!(predict_ef(1.0, 16, 64), 64);
        let mid = predict_ef(0.5, 16, 64);
        assert!(mid >= 38 && mid <= 42, "expected ~40, got {}", mid);
    }

    #[test]
    fn confidence_complement() {
        let easy = make(&[0.1, 10.0]);
        let hard = make(&[1.0, 1.05]);
        assert!(retrieval_confidence(&easy) > 0.95);
        assert!(retrieval_confidence(&hard) < 0.10);
    }
}
