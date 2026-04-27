/// Squared L2 distance. Manually unrolled 4× for auto-vectorization.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;
    let mut acc = 0.0f32;
    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Dot product. Manually unrolled 4× for auto-vectorization.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;
    let mut acc = 0.0f32;
    for i in 0..chunks {
        let base = i * 4;
        acc += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }
    for i in (chunks * 4)..len {
        acc += a[i] * b[i];
    }
    acc
}

/// Subtract b from a, storing result in out: out[i] = a[i] - b[i].
pub fn sub_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai - bi;
    }
}

/// Saxpy: out[i] += scale * v[i].
pub fn saxpy(out: &mut [f32], v: &[f32], scale: f32) {
    for (o, x) in out.iter_mut().zip(v.iter()) {
        *o += scale * x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_known() {
        let a = [3.0f32, 0.0];
        let b = [0.0f32, 4.0];
        assert!((l2_sq(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn dot_orthogonal() {
        assert!((dot(&[1.0f32, 0.0], &[0.0, 1.0])).abs() < 1e-9);
    }

    #[test]
    fn sub_into_basic() {
        let a = [3.0f32, 1.0];
        let b = [1.0f32, 2.0];
        let mut out = [0.0f32; 2];
        sub_into(&a, &b, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-9);
        assert!((out[1] - (-1.0)).abs() < 1e-9);
    }
}
