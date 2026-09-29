//! Metrics and exact f32 kernels (used for rerank and ground truth).
//!
//! Local and `std::arch`-free so they compile unchanged for wasm32; eight
//! independent accumulators let LLVM auto-vectorise. Every kernel returns a
//! **distance: lower is closer**.

use crate::error::IndexError;

/// Collection metric. Wire forms match `ruvector-edge-store`
/// (`cosine | l2 | dot`) so the store can bridge by string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    pub(crate) fn code(self) -> u8 {
        match self {
            Metric::Cosine => 1,
            Metric::L2 => 2,
            Metric::Dot => 3,
        }
    }

    pub(crate) fn from_code(c: u8) -> Option<Metric> {
        match c {
            1 => Some(Metric::Cosine),
            2 => Some(Metric::L2),
            3 => Some(Metric::Dot),
            _ => None,
        }
    }
}

#[inline]
fn fold8(a: &[f32], b: &[f32], term: impl Fn(f32, f32) -> f32) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += term(x[k], y[k]);
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += term(*x, *y);
    }
    s
}

/// Dot product.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    fold8(a, b, |x, y| x * y)
}

/// Squared Euclidean distance.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    fold8(a, b, |x, y| (x - y) * (x - y))
}

/// Euclidean norm.
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Exact distance under `metric` (lower is closer). A zero-norm operand
/// under cosine yields `1.0` (orthogonal); callers reject such vectors at
/// the boundary with [`validate_vector`].
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::Cosine => {
            let den = norm(a) * norm(b);
            if den > 0.0 {
                1.0 - dot(a, b) / den
            } else {
                1.0
            }
        }
        Metric::L2 => l2_sq(a, b).sqrt(),
        Metric::Dot => -dot(a, b),
    }
}

/// Boundary validation: exact dimension, all values finite, and a non-zero
/// norm for cosine.
pub fn validate_vector(metric: Metric, dim: usize, v: &[f32]) -> Result<(), IndexError> {
    if v.len() != dim {
        return Err(IndexError::DimMismatch {
            expected: dim,
            got: v.len(),
        });
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(IndexError::NonFinite);
    }
    let n = norm(v);
    if !n.is_finite() {
        // Squared norm overflowed f32: distances would be inf/NaN.
        return Err(IndexError::NonFinite);
    }
    if metric == Metric::Cosine && n <= 0.0 {
        return Err(IndexError::ZeroNorm);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_match_naive() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.11).cos()).collect();
        let nd: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let nl: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((dot(&a, &b) - nd).abs() < 1e-4);
        assert!((l2_sq(&a, &b) - nl).abs() < 1e-4);
        assert!(distance(Metric::Cosine, &a, &a).abs() < 1e-5);
        assert_eq!(distance(Metric::Dot, &a, &b), -dot(&a, &b));
    }

    #[test]
    fn wire_forms_and_validation() {
        for m in [Metric::Cosine, Metric::L2, Metric::Dot] {
            assert_eq!(Metric::parse(m.as_str()), Some(m));
            assert_eq!(Metric::from_code(m.code()), Some(m));
        }
        assert!(Metric::parse("COSINE").is_none());
        assert!(validate_vector(Metric::L2, 2, &[1.0]).is_err());
        assert!(validate_vector(Metric::L2, 2, &[1.0, f32::NAN]).is_err());
        assert!(validate_vector(Metric::Cosine, 2, &[0.0, 0.0]).is_err());
        assert!(validate_vector(Metric::Dot, 2, &[0.0, 0.0]).is_ok());
    }
}
