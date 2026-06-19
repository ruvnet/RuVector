/// Squared L2 distance. Monotone-equivalent to L2 for nearest-neighbour ranking.
#[inline(always)]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine distance in [0, 2].
pub fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 1.0;
    }
    (1.0 - dot / (na * nb)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2sq_identical_is_zero() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert!(l2sq(&a, &a) < 1e-9);
    }

    #[test]
    fn cosine_identical_is_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        assert!(cosine_dist(&a, &a) < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let d = cosine_dist(&a, &b);
        assert!((d - 1.0).abs() < 1e-6, "expected 1.0, got {}", d);
    }
}
