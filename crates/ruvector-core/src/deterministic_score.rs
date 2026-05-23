//! Fixed-point integer scoring for cross-platform deterministic ranking.
//!
//! Converts f64 to i64 with 2^32 scaling. NaN → ZERO, +Inf → MAX, -Inf → NEG_INF.
//! All arithmetic uses i128 intermediates with saturating semantics.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

use crate::types::{DistanceMetric, SearchResult, VectorId};

// ---------------------------------------------------------------------------
// DeterministicScore
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct DeterministicScore(i64);

impl DeterministicScore {
    const SCALE: f64 = 4_294_967_296.0; // 2^32

    pub const MAX: Self = Self(i64::MAX);
    pub const NEG_INF: Self = Self(i64::MIN + 1);
    pub const ZERO: Self = Self(0);

    #[inline]
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn to_raw(self) -> i64 {
        self.0
    }

    #[inline]
    pub fn from_f64(val: f64) -> Self {
        if val.is_nan() {
            return Self::ZERO;
        }
        if val.is_infinite() {
            return if val.is_sign_positive() {
                Self::MAX
            } else {
                Self::NEG_INF
            };
        }
        let scaled = (val * Self::SCALE).round();
        Self::from_rounded_arithmetic(scaled)
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        Self::from_f64(val as f64)
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        if self.0 == Self::MAX.0 {
            return f64::INFINITY;
        }
        if self.0 == Self::NEG_INF.0 {
            return f64::NEG_INFINITY;
        }
        self.0 as f64 / Self::SCALE
    }

    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.0 == i64::MAX || self.0 == Self::NEG_INF.0
    }

    /// Convert an f32 distance (lower = closer) to a similarity DeterministicScore
    /// (higher = better), taking the distance metric into account.
    #[inline]
    pub fn similarity_from_distance(distance: f32, metric: DistanceMetric) -> Self {
        let similarity = match metric {
            DistanceMetric::Cosine => 1.0 - distance as f64,
            DistanceMetric::DotProduct => -(distance as f64),
            DistanceMetric::Euclidean | DistanceMetric::Manhattan => {
                1.0 / (1.0 + distance as f64)
            }
        };
        Self::from_f64(similarity)
    }

    #[inline]
    fn from_arithmetic_raw(raw: i128) -> Self {
        if raw >= i64::MAX as i128 {
            Self::MAX
        } else if raw <= Self::NEG_INF.0 as i128 {
            Self::NEG_INF
        } else {
            Self(raw as i64)
        }
    }

    #[inline]
    fn from_rounded_arithmetic(raw: f64) -> Self {
        if raw.is_nan() {
            Self::ZERO
        } else if raw.is_sign_positive() && !raw.is_finite() {
            Self::MAX
        } else if !raw.is_finite() {
            Self::NEG_INF
        } else if raw >= i64::MAX as f64 {
            Self::MAX
        } else if raw <= i64::MIN as f64 {
            Self::NEG_INF
        } else {
            Self(raw as i64)
        }
    }
}

impl Ord for DeterministicScore {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for DeterministicScore {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for DeterministicScore {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Default for DeterministicScore {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Add for DeterministicScore {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_arithmetic_raw(self.0 as i128 + rhs.0 as i128)
    }
}

impl Sub for DeterministicScore {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_arithmetic_raw(self.0 as i128 - rhs.0 as i128)
    }
}

impl Mul<i64> for DeterministicScore {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i64) -> Self::Output {
        let result = (self.0 as i128).saturating_mul(rhs as i128);
        Self::from_arithmetic_raw(result)
    }
}

impl Mul<f64> for DeterministicScore {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        if rhs.is_nan() {
            return Self::ZERO;
        }
        let product = (self.0 as f64) * rhs;
        Self::from_rounded_arithmetic(product.round())
    }
}

impl Div<i64> for DeterministicScore {
    type Output = Self;
    #[inline]
    fn div(self, rhs: i64) -> Self::Output {
        if rhs == 0 {
            return if self.0 == 0 {
                Self::ZERO
            } else if self.0 > 0 {
                Self::MAX
            } else {
                Self::NEG_INF
            };
        }
        Self::from_arithmetic_raw(self.0.saturating_div(rhs) as i128)
    }
}

impl fmt::Debug for DeterministicScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::MAX {
            write!(f, "DeterministicScore(+Inf)")
        } else if *self == Self::NEG_INF {
            write!(f, "DeterministicScore(-Inf)")
        } else {
            write!(f, "DeterministicScore({:.9})", self.to_f64())
        }
    }
}

impl fmt::Display for DeterministicScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::MAX {
            write!(f, "+Inf")
        } else if *self == Self::NEG_INF {
            write!(f, "-Inf")
        } else {
            write!(f, "{:.6}", self.to_f64())
        }
    }
}

impl From<f64> for DeterministicScore {
    fn from(val: f64) -> Self {
        Self::from_f64(val)
    }
}

impl From<f32> for DeterministicScore {
    fn from(val: f32) -> Self {
        Self::from_f32(val)
    }
}

impl From<DeterministicScore> for f64 {
    fn from(score: DeterministicScore) -> Self {
        score.to_f64()
    }
}

// ---------------------------------------------------------------------------
// DeterministicSearchResult
// ---------------------------------------------------------------------------

/// Search result with deterministic fixed-point scoring.
///
/// Parallel to `SearchResult` but uses `DeterministicScore` instead of f32.
/// Score semantics: higher = more relevant (similarity, not distance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicSearchResult {
    pub id: VectorId,
    pub score: DeterministicScore,
    pub vector: Option<Vec<f32>>,
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl SearchResult {
    /// Convert a distance-based SearchResult into a deterministic similarity result.
    pub fn to_deterministic(&self, metric: DistanceMetric) -> DeterministicSearchResult {
        DeterministicSearchResult {
            id: self.id.clone(),
            score: DeterministicScore::similarity_from_distance(self.score, metric),
            vector: self.vector.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic Fusion
// ---------------------------------------------------------------------------

/// Deterministic Reciprocal Rank Fusion over DeterministicScore.
///
/// Uses BTreeMap (not HashMap) for iteration-order determinism.
/// Accumulates in i128 to prevent overflow. Ties broken by id (ascending).
/// Default k=15 per Lean4 proof `ret010_k60_consensus_bias`.
pub fn deterministic_rrf(
    sources: &[Vec<DeterministicSearchResult>],
    k_param: usize,
    top_k: usize,
) -> Vec<DeterministicSearchResult> {
    let mut totals: BTreeMap<&VectorId, i128> = BTreeMap::new();

    for source in sources {
        for (rank_0, result) in source.iter().enumerate() {
            let rank_1 = rank_0 + 1;
            let contrib = DeterministicScore::from_f64(1.0 / (k_param + rank_1) as f64);
            *totals.entry(&result.id).or_default() += contrib.to_raw() as i128;
        }
    }

    let mut results: Vec<DeterministicSearchResult> = totals
        .into_iter()
        .map(|(id, raw)| {
            let clamped = raw.clamp(DeterministicScore::NEG_INF.0 as i128, i64::MAX as i128);
            DeterministicSearchResult {
                id: id.clone(),
                score: DeterministicScore::from_raw(clamped as i64),
                vector: None,
                metadata: None,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    results.truncate(top_k);
    results
}

/// Deterministic weighted linear combination.
///
/// Each source gets a weight. Scores are accumulated in i128.
pub fn deterministic_weighted(
    sources: &[(f64, Vec<DeterministicSearchResult>)],
    top_k: usize,
) -> Vec<DeterministicSearchResult> {
    let mut totals: BTreeMap<&VectorId, i128> = BTreeMap::new();

    for (weight, results) in sources {
        for result in results {
            let weighted = result.score * *weight;
            *totals.entry(&result.id).or_default() += weighted.to_raw() as i128;
        }
    }

    let mut results: Vec<DeterministicSearchResult> = totals
        .into_iter()
        .map(|(id, raw)| {
            let clamped = raw.clamp(DeterministicScore::NEG_INF.0 as i128, i64::MAX as i128);
            DeterministicSearchResult {
                id: id.clone(),
                score: DeterministicScore::from_raw(clamped as i64),
                vector: None,
                metadata: None,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    results.truncate(top_k);
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Core type ------------------------------------------------------------

    #[test]
    fn roundtrip_f64() {
        let s = DeterministicScore::from_f64(0.5);
        assert!((s.to_f64() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn nan_maps_to_zero() {
        let s = DeterministicScore::from_f64(f64::NAN);
        assert_eq!(s, DeterministicScore::ZERO);
    }

    #[test]
    fn infinity() {
        assert_eq!(
            DeterministicScore::from_f64(f64::INFINITY),
            DeterministicScore::MAX
        );
        assert_eq!(
            DeterministicScore::from_f64(f64::NEG_INFINITY),
            DeterministicScore::NEG_INF
        );
    }

    #[test]
    fn total_ordering() {
        let a = DeterministicScore::from_f64(0.1);
        let b = DeterministicScore::from_f64(0.5);
        let c = DeterministicScore::from_f64(f64::NAN);
        assert!(a < b);
        assert_eq!(c, DeterministicScore::ZERO);
        // NaN-as-zero sorts deterministically
        assert!(c < b);
        assert!(c < a || c == a || c > a); // total order, no panic
    }

    #[test]
    fn saturating_add() {
        assert_eq!(
            DeterministicScore::MAX + DeterministicScore::from_raw(1),
            DeterministicScore::MAX
        );
    }

    #[test]
    fn saturating_sub() {
        assert_eq!(
            DeterministicScore::NEG_INF - DeterministicScore::from_raw(1),
            DeterministicScore::NEG_INF
        );
    }

    #[test]
    fn arithmetic_add() {
        let a = DeterministicScore::from_f64(0.3);
        let b = DeterministicScore::from_f64(0.4);
        assert!((a + b).to_f64() - 0.7 < 1e-9);
    }

    #[test]
    fn div_by_zero() {
        assert_eq!(
            DeterministicScore::from_f64(0.5) / 0,
            DeterministicScore::MAX
        );
        assert_eq!(
            DeterministicScore::from_f64(-0.5) / 0,
            DeterministicScore::NEG_INF
        );
        assert_eq!(DeterministicScore::ZERO / 0, DeterministicScore::ZERO);
    }

    #[test]
    fn raw_scale() {
        assert_eq!(
            DeterministicScore::from_f64(1.0).to_raw(),
            4_294_967_296_i64
        );
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", DeterministicScore::MAX), "+Inf");
        assert_eq!(format!("{}", DeterministicScore::NEG_INF), "-Inf");
        let s = format!("{}", DeterministicScore::from_f64(0.1234567));
        assert_eq!(s, "0.123457");
    }

    // -- Boundary conversion --------------------------------------------------

    // f32→f64 cast introduces ~1e-7 error, so tolerance must be > f32 epsilon
    const F32_TOL: f64 = 1e-7;

    #[test]
    fn cosine_distance_to_similarity() {
        // cosine distance 0.2 → similarity 0.8
        let s = DeterministicScore::similarity_from_distance(0.2, DistanceMetric::Cosine);
        assert!((s.to_f64() - 0.8).abs() < F32_TOL);
    }

    #[test]
    fn euclidean_distance_to_similarity() {
        // euclidean distance 0.0 → similarity 1.0
        let s = DeterministicScore::similarity_from_distance(0.0, DistanceMetric::Euclidean);
        assert!((s.to_f64() - 1.0).abs() < F32_TOL);
        // euclidean distance 1.0 → similarity 0.5
        let s = DeterministicScore::similarity_from_distance(1.0, DistanceMetric::Euclidean);
        assert!((s.to_f64() - 0.5).abs() < F32_TOL);
    }

    #[test]
    fn dot_product_distance_to_similarity() {
        // dot_product_distance stores -dot, so distance = -5.0 means dot = 5.0
        let s = DeterministicScore::similarity_from_distance(-5.0, DistanceMetric::DotProduct);
        assert!((s.to_f64() - 5.0).abs() < F32_TOL);
    }

    #[test]
    fn search_result_to_deterministic() {
        let result = SearchResult {
            id: "doc1".to_string(),
            score: 0.1, // cosine distance
            vector: None,
            metadata: None,
        };
        let det = result.to_deterministic(DistanceMetric::Cosine);
        assert_eq!(det.id, "doc1");
        assert!((det.score.to_f64() - 0.9).abs() < F32_TOL);
    }

    // -- f32→i64 order preservation (Lean4: score_003_order_preservation) -----

    #[test]
    fn f32_to_i64_order_preservation() {
        let distances: Vec<f32> = vec![0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 1.5];
        let scores: Vec<DeterministicScore> = distances
            .iter()
            .map(|&d| DeterministicScore::similarity_from_distance(d, DistanceMetric::Cosine))
            .collect();
        // Cosine: smaller distance → higher similarity → higher score
        for i in 0..scores.len() - 1 {
            assert!(
                scores[i] > scores[i + 1],
                "Order violation at index {}: {:?} should be > {:?}",
                i,
                scores[i],
                scores[i + 1]
            );
        }
    }

    // -- Deterministic RRF ----------------------------------------------------

    #[test]
    fn rrf_basic() {
        let source_a = vec![
            DeterministicSearchResult {
                id: "a".into(),
                score: DeterministicScore::from_f64(0.9),
                vector: None,
                metadata: None,
            },
            DeterministicSearchResult {
                id: "b".into(),
                score: DeterministicScore::from_f64(0.8),
                vector: None,
                metadata: None,
            },
        ];
        let source_b = vec![
            DeterministicSearchResult {
                id: "b".into(),
                score: DeterministicScore::from_f64(0.95),
                vector: None,
                metadata: None,
            },
            DeterministicSearchResult {
                id: "c".into(),
                score: DeterministicScore::from_f64(0.7),
                vector: None,
                metadata: None,
            },
        ];
        let results = deterministic_rrf(&[source_a, source_b], 15, 10);
        // "b" appears at rank 1 in both → highest RRF
        assert_eq!(results[0].id, "b");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn rrf_commutative() {
        // Source order should not affect results (Lean4: rrf_sum_comm)
        let s1 = vec![DeterministicSearchResult {
            id: "x".into(),
            score: DeterministicScore::from_f64(0.9),
            vector: None,
            metadata: None,
        }];
        let s2 = vec![DeterministicSearchResult {
            id: "y".into(),
            score: DeterministicScore::from_f64(0.8),
            vector: None,
            metadata: None,
        }];

        let forward = deterministic_rrf(&[s1.clone(), s2.clone()], 15, 10);
        let reverse = deterministic_rrf(&[s2, s1], 15, 10);

        assert_eq!(forward.len(), reverse.len());
        for (a, b) in forward.iter().zip(reverse.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.score, b.score);
        }
    }

    #[test]
    fn rrf_tie_broken_by_id() {
        let s1 = vec![DeterministicSearchResult {
            id: "beta".into(),
            score: DeterministicScore::from_f64(0.5),
            vector: None,
            metadata: None,
        }];
        let s2 = vec![DeterministicSearchResult {
            id: "alpha".into(),
            score: DeterministicScore::from_f64(0.5),
            vector: None,
            metadata: None,
        }];
        let results = deterministic_rrf(&[s1, s2], 15, 10);
        // Same RRF score (both rank 1 in one source), tie broken by id ascending
        assert_eq!(results[0].id, "alpha");
        assert_eq!(results[1].id, "beta");
    }

    // -- Deterministic weighted -----------------------------------------------

    #[test]
    fn weighted_basic() {
        let dense = vec![DeterministicSearchResult {
            id: "a".into(),
            score: DeterministicScore::from_f64(0.8),
            vector: None,
            metadata: None,
        }];
        let sparse = vec![DeterministicSearchResult {
            id: "a".into(),
            score: DeterministicScore::from_f64(0.6),
            vector: None,
            metadata: None,
        }];
        let results = deterministic_weighted(&[(0.7, dense), (0.3, sparse)], 10);
        assert_eq!(results.len(), 1);
        // 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert!((results[0].score.to_f64() - 0.74).abs() < 1e-6);
    }
}
