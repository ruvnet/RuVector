//! Minimal f32 vector math — cosine similarity is the only distance used
//! throughout this crate, matching the convention set by other RuVector
//! nightly ANN benchmarks (`ruvector-retrieval-receipt`, `ruvector-acorn`).

pub type Vector = Vec<f32>;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

pub fn normalize_in_place(a: &mut [f32]) {
    let n = norm(a);
    if n > 1e-12 {
        for x in a.iter_mut() {
            *x /= n;
        }
    }
}

/// Cosine similarity. Every vector generated in this crate is
/// unit-normalized at construction time (`normalize_in_place`), so in
/// practice this reduces to a dot product; the explicit division below is
/// kept as a defensive fallback for any non-normalized input.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_cosine_one() {
        let v = vec![1.0, 2.0, 3.0, -1.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_vectors_have_cosine_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn normalize_produces_unit_norm() {
        let mut v = vec![3.0, 4.0];
        normalize_in_place(&mut v);
        assert!((norm(&v) - 1.0).abs() < 1e-5);
    }
}
