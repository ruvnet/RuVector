/// Squared Euclidean distance.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Unit-tests for the distance kernel.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_zero() {
        let v = [1.0_f32, 2.0, 3.0];
        assert_eq!(l2_sq(&v, &v), 0.0);
    }

    #[test]
    fn l2_sq_known() {
        // [0,0] vs [3,4] → 9 + 16 = 25
        assert!((l2_sq(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-5);
    }
}
