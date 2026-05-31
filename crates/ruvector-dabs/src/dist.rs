//! Distance functions. All operate on f32 slices.
//! L2-squared is used throughout; the squared form avoids a sqrt while
//! preserving total ordering, which is sufficient for nearest-neighbor ranking.

/// Squared Euclidean distance over the full slice.
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Squared Euclidean distance over first `dims` elements only.
#[inline(always)]
pub fn l2_sq_partial(a: &[f32], b: &[f32], dims: usize) -> f32 {
    debug_assert!(dims <= a.len() && dims <= b.len());
    a[..dims]
        .iter()
        .zip(b[..dims].iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Inner product (dot product), returned as f32.
#[inline(always)]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector in-place to unit L2 norm.
pub fn normalize(v: &mut [f32]) {
    let n = l2_norm(v);
    if n > 1e-9 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_identity() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(l2_sq(&v, &v), 0.0);
    }

    #[test]
    fn l2_sq_known() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 2.0];
        assert!((l2_sq(&a, &b) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn l2_sq_partial_prefix() {
        let a = vec![1.0_f32, 2.0, 100.0];
        let b = vec![1.0_f32, 2.0, 0.0];
        assert_eq!(l2_sq_partial(&a, &b, 2), 0.0);
        assert!((l2_sq(&a, &b) - 10000.0).abs() < 1.0);
    }

    #[test]
    fn normalize_unit() {
        let mut v = vec![3.0_f32, 4.0];
        normalize(&mut v);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
    }
}
