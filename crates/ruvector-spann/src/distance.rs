//! Distance computation for SPANN partition index.

/// Compute L2 squared distance between two f32 slices.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute cosine similarity (in [0, 2] range, lower = more similar).
/// Returns 1 - dot(a, b) / (|a| * |b|), scaled to [0, 2].
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 1.0;
    }
    1.0 - dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_zero_self() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(l2_squared(&v, &v) < 1e-9);
    }

    #[test]
    fn l2_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        assert!((l2_squared(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!(cosine_distance(&v, &v) < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }
}
