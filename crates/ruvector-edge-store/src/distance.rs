//! Distance kernels for the M1 exact flat scan.
//!
//! Local, SIMD-free and `std::arch`-free so they compile unchanged for
//! `wasm32-unknown-unknown`: `rvf-index`'s kernels live in the excluded
//! `crates/rvf` workspace and carry AVX2/NEON paths. Four independent
//! accumulators let LLVM auto-vectorise (simd128 when enabled) without
//! changing results between runs on one target.
//!
//! Inputs are finite `f32`; every sum is accumulated in `f64`, where a
//! product of two `f32` is exact and a sum over `dim ≤ 1536` terms cannot
//! overflow (≤ 1536 · (2·f32::MAX)² ≪ f64::MAX). So pairwise results never
//! overflow or lose subnormal norms, whatever the per-vector magnitudes.
//! [`to_wire`] maps a distance to a finite `f32` for the response.
//!
//! Every kernel returns a **distance: lower is closer**.

use serde::{Deserialize, Serialize};

/// Collection metric (ADR §7: `cosine | l2 | dot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// `1 - cos(a, b)`.
    Cosine,
    /// Euclidean distance `‖a - b‖₂`.
    L2,
    /// `-(a · b)`.
    Dot,
}

impl Metric {
    /// Wire form.
    pub fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::L2 => "l2",
            Metric::Dot => "dot",
        }
    }
    /// Strict parse of the wire form.
    pub fn parse(s: &str) -> Option<Metric> {
        match s {
            "cosine" => Some(Metric::Cosine),
            "l2" => Some(Metric::L2),
            "dot" => Some(Metric::Dot),
            _ => None,
        }
    }
}

fn fold4(a: &[f32], b: &[f32], term: impl Fn(f64, f64) -> f64) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..4 {
            acc[k] += term(f64::from(x[k]), f64::from(y[k]));
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += term(f64::from(*x), f64::from(*y));
    }
    s
}

/// Dot product (f64 accumulation).
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    fold4(a, b, |x, y| x * y)
}

/// Squared Euclidean distance (f64 accumulation).
pub fn l2_sq(a: &[f32], b: &[f32]) -> f64 {
    fold4(a, b, |x, y| (x - y) * (x - y))
}

/// Euclidean norm.
pub fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

/// Distance between `query` and `row` under `metric`. For cosine the caller
/// passes both norms (precomputed per row at upsert, once per query); a zero
/// norm is rejected at validation, so the division is defined. A
/// non-finite result (impossible for validated inputs) ranks last.
pub fn distance(metric: Metric, query: &[f32], q_norm: f64, row: &[f32], row_norm: f64) -> f64 {
    let d = match metric {
        Metric::Cosine => 1.0 - dot(query, row) / (q_norm * row_norm),
        Metric::L2 => l2_sq(query, row).sqrt(),
        Metric::Dot => -dot(query, row),
    };
    if d.is_finite() {
        d
    } else {
        f64::INFINITY
    }
}

/// Response form of a distance: the nearest finite `f32` (JSON has no
/// infinities; `serde_json` would emit `null`).
pub fn to_wire(d: f64) -> f32 {
    if d.is_nan() {
        return f32::MAX;
    }
    d.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_match_naive_on_odd_lengths() {
        let a: Vec<f32> = (0..7).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b: Vec<f32> = (0..7).map(|i| 2.0 - i as f32 * 0.25).collect();
        let nd: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
        let nl: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from((x - y) * (x - y)))
            .sum();
        assert!((dot(&a, &b) - nd).abs() < 1e-9);
        assert!((l2_sq(&a, &b) - nl).abs() < 1e-9);
        let d = distance(Metric::Cosine, &a, norm(&a), &a, norm(&a));
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn extreme_magnitudes_stay_finite_and_ordered() {
        let big = vec![3.0e38f32; 1536];
        let neg = vec![-3.0e38f32; 1536];
        let tiny = vec![1.0e-45f32, 0.0];
        // Norms beyond f32::MAX and below f32's normal range are exact in f64.
        assert!(norm(&big).is_finite() && norm(&big) > f64::from(f32::MAX));
        assert!(norm(&tiny) > 0.0);
        let cos = distance(Metric::Cosine, &big, norm(&big), &neg, norm(&neg));
        assert!((cos - 2.0).abs() < 1e-9, "opposite vectors: {cos}");
        let near = distance(Metric::L2, &big, 0.0, &big, 0.0);
        let far = distance(Metric::L2, &big, 0.0, &neg, 0.0);
        assert!(near < far && far.is_finite());
        assert_eq!(to_wire(far), f32::MAX);
        assert_eq!(to_wire(-far), -f32::MAX);
        assert_eq!(to_wire(f64::NAN), f32::MAX);
        let t = distance(Metric::Cosine, &tiny, norm(&tiny), &tiny, norm(&tiny));
        assert!(t.abs() < 1e-12, "subnormal cosine: {t}");
    }

    #[test]
    fn metric_wire_round_trip() {
        for m in [Metric::Cosine, Metric::L2, Metric::Dot] {
            assert_eq!(Metric::parse(m.as_str()), Some(m));
        }
        assert_eq!(Metric::parse("Cosine"), None);
    }
}
