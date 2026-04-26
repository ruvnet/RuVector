/// Squared Euclidean (L2²) distance — avoids sqrt for comparison-only paths.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// Euclidean distance (for reporting, not inner-loop comparison).
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_sq(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_self_distance() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(l2_sq(&v, &v), 0.0);
    }

    #[test]
    fn known_l2() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        assert!((l2(&a, &b) - 5.0).abs() < 1e-5);
    }
}
