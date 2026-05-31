/// Squared Euclidean distance — the inner loop of every ANN search.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product (used in ADC asymmetric distance estimation).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_known() {
        assert!((l2_sq(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-5);
    }

    #[test]
    fn dot_orthogonal() {
        assert!(dot(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }
}
