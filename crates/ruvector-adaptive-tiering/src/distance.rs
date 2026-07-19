//! Distance primitives — no external dependencies.

/// Squared L2 distance.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 magnitude.
#[inline]
pub fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity in [-1, 1].
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let ma = magnitude(a);
    let mb = magnitude(b);
    if ma < 1e-9 || mb < 1e-9 {
        return 0.0;
    }
    (d / (ma * mb)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_zero_distance() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((l2_sq(&v, &v)).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }
}
