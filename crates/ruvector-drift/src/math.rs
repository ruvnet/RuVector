/// Returns cosine similarity in [-1, 1]; 0 when either vector is near-zero.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Normalizes a vector in-place; no-op if near-zero.
pub fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Squared L2 distance.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Mean of a slice.
pub fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Variance of a slice.
pub fn variance(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / xs.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn normalize_unit() {
        let mut v = vec![3.0f32, 4.0, 0.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn variance_constant() {
        let xs = vec![2.0f32; 10];
        assert!(variance(&xs) < 1e-6);
    }
}
