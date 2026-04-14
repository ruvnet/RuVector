//! Core EML operators.
//!
//! The EML operator `eml(x, y) = exp(x) - ln(y)` is the continuous-mathematics
//! analog of the NAND gate: combined with the constant 1, it can reconstruct
//! all elementary functions (Odrzywolel 2026).

/// The EML universal operator: `eml(x, y) = exp(x) - ln(y)`.
///
/// Combined with the constant 1, this single operator can reconstruct
/// all elementary functions.
#[inline]
pub fn eml(x: f64, y: f64) -> f64 {
    x.exp() - y.ln()
}

/// Numerically safe EML: clamps exp input to [-20, 20] and ensures
/// a positive ln argument.
///
/// Use this instead of [`eml`] in evaluation trees where inputs may
/// be out of range.
#[inline]
pub fn eml_safe(x: f64, y: f64) -> f64 {
    let ex = x.clamp(-20.0, 20.0).exp();
    let ly = if y > 0.0 {
        y.ln()
    } else {
        f64::MIN_POSITIVE.ln()
    };
    ex - ly
}

/// Softmax over 3 values so that `alpha + beta + gamma = 1`.
///
/// Used as a mixing function in EML tree levels.
#[inline]
pub fn softmax3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let max = a.max(b).max(c);
    let ea = (a - max).exp();
    let eb = (b - max).exp();
    let ec = (c - max).exp();
    let sum = ea + eb + ec;
    (ea / sum, eb / sum, ec / sum)
}

/// Generate random parameters in [-1, 1] using a simple LCG.
///
/// This is a deterministic PRNG suitable for random restarts during
/// coordinate descent training. Not cryptographically secure.
pub(crate) fn random_params(state: &mut u64, count: usize) -> Vec<f64> {
    let mut params = vec![0.0f64; count];
    for p in params.iter_mut() {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *p = (*state >> 33) as f64 / (u32::MAX as f64 / 2.0) - 1.0;
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eml_identity() {
        // eml(0, 1) = exp(0) - ln(1) = 1 - 0 = 1
        let result = eml(0.0, 1.0);
        assert!(
            (result - 1.0).abs() < 1e-12,
            "eml(0, 1) should be 1.0, got {result}"
        );
    }

    #[test]
    fn eml_exp_only() {
        // eml(1, 1) = exp(1) - ln(1) = e
        let result = eml(1.0, 1.0);
        assert!(
            (result - std::f64::consts::E).abs() < 1e-12,
            "eml(1, 1) should be e, got {result}"
        );
    }

    #[test]
    fn eml_ln_only() {
        // eml(0, e) = exp(0) - ln(e) = 1 - 1 = 0
        let result = eml(0.0, std::f64::consts::E);
        assert!(
            result.abs() < 1e-12,
            "eml(0, e) should be 0.0, got {result}"
        );
    }

    #[test]
    fn eml_safe_does_not_panic() {
        let _ = eml_safe(100.0, 0.0);
        let _ = eml_safe(-100.0, -5.0);
        let _ = eml_safe(0.0, f64::MIN_POSITIVE);
        let _ = eml_safe(f64::NAN, 1.0);
    }

    #[test]
    fn eml_safe_clamps_large_exp() {
        let result = eml_safe(100.0, 1.0);
        // Should use exp(20) not exp(100)
        assert!(result.is_finite(), "eml_safe(100, 1) should be finite");
        let expected = 20.0_f64.exp(); // ln(1) = 0
        assert!(
            (result - expected).abs() < 1e-6,
            "eml_safe(100, 1) should be exp(20), got {result}"
        );
    }

    #[test]
    fn softmax3_sums_to_one() {
        let (a, b, c) = softmax3(1.0, 2.0, 3.0);
        let sum = a + b + c;
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "softmax3 should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn softmax3_equal_inputs() {
        let (a, b, c) = softmax3(0.0, 0.0, 0.0);
        assert!((a - 1.0 / 3.0).abs() < 1e-12);
        assert!((b - 1.0 / 3.0).abs() < 1e-12);
        assert!((c - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn softmax3_dominated_input() {
        let (a, _b, _c) = softmax3(100.0, 0.0, 0.0);
        assert!(a > 0.99, "dominant input should get nearly all weight");
    }

    #[test]
    fn random_params_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        let p1 = random_params(&mut s1, 10);
        let p2 = random_params(&mut s2, 10);
        assert_eq!(p1, p2, "same seed should produce same params");
    }

    #[test]
    fn random_params_in_range() {
        let mut s = 0xDEAD_BEEF_u64;
        let params = random_params(&mut s, 100);
        for &p in &params {
            assert!(
                p >= -1.0 && p <= 1.0,
                "param {p} out of [-1, 1] range"
            );
        }
    }
}
