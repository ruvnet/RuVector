//! Input validation for the JS boundary.
//!
//! Every size that arrives from JavaScript is untrusted: products are
//! computed with checked arithmetic (usize is 32-bit on wasm32, so a
//! wrapping product can make a too-short buffer look correctly sized) and
//! the kernels below are only reached once lengths are known to match.
//! Errors are plain `String`s so these checks are unit-testable natively.

/// Largest embedding dimension accepted from JS (bounds the allocation).
pub const MAX_EMBEDDING_DIM: usize = 65_536;

/// Checked product of a list of dimensions; rejects zero and overflow.
pub fn checked_size(what: &str, dims: &[usize]) -> Result<usize, String> {
    let mut total: usize = 1;
    for &d in dims {
        if d == 0 {
            return Err(format!("{what}: dimensions must be non-zero"));
        }
        total = total
            .checked_mul(d)
            .ok_or_else(|| format!("{what}: dimension product overflows"))?;
    }
    Ok(total)
}

/// Require `actual == expected` for a buffer length.
pub fn expect_len(what: &str, expected: usize, actual: usize) -> Result<(), String> {
    if expected != actual {
        return Err(format!(
            "{what}: invalid length, expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

/// Validate a configured embedding dimension.
pub fn embedding_dim(dim: u32) -> Result<usize, String> {
    let dim = dim as usize;
    if dim == 0 || dim > MAX_EMBEDDING_DIM {
        return Err(format!(
            "embedding_dim must be in 1..={MAX_EMBEDDING_DIM}, got {dim}"
        ));
    }
    Ok(dim)
}

/// Validate a strictly positive, finite scalar (e.g. temperature).
pub fn positive_finite(what: &str, v: f32) -> Result<f64, String> {
    if !(v.is_finite() && v > 0.0) {
        return Err(format!("{what} must be a positive finite number"));
    }
    Ok(v as f64)
}

/// Validate a non-negative, finite scalar (e.g. margin).
pub fn non_negative_finite(what: &str, v: f32) -> Result<f64, String> {
    if !(v.is_finite() && v >= 0.0) {
        return Err(format!("{what} must be a non-negative finite number"));
    }
    Ok(v as f64)
}

/// Validate batch-norm parameters against the input; returns channel count.
pub fn batch_norm(
    input: usize,
    gamma: usize,
    beta: usize,
    mean: usize,
    var: usize,
) -> Result<usize, String> {
    if gamma == 0 {
        return Err("batch_norm: at least one channel is required".into());
    }
    if beta != gamma || mean != gamma || var != gamma {
        return Err(format!(
            "batch_norm: parameter lengths must match (gamma {gamma}, beta {beta}, mean {mean}, var {var})"
        ));
    }
    if input % gamma != 0 {
        return Err(format!(
            "batch_norm: input length {input} is not a multiple of channels {gamma}"
        ));
    }
    Ok(gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_size_rejects_zero_and_overflow() {
        assert!(checked_size("x", &[0, 3]).is_err());
        assert!(checked_size("x", &[usize::MAX, 2]).is_err());
        assert_eq!(checked_size("x", &[4, 5, 3]), Ok(60));
    }

    #[test]
    fn checked_size_catches_u32_wrap() {
        // 2 * 715827883 * 3 == 2^32 + 2: wraps to 2 in u32 arithmetic.
        let r = checked_size("x", &[2, 715_827_883, 3]);
        if usize::BITS == 32 {
            assert!(r.is_err());
        } else {
            assert_eq!(r.map(|v| v as u64), Ok((1u64 << 32) + 2));
        }
        let w: u32 = 2u32.wrapping_mul(715_827_883).wrapping_mul(3);
        assert_eq!(w, 2);
    }

    #[test]
    fn scalars() {
        assert!(positive_finite("t", 0.0).is_err());
        assert!(positive_finite("t", f32::NAN).is_err());
        assert!(positive_finite("t", f32::INFINITY).is_err());
        assert!(positive_finite("t", 0.1).is_ok());
        assert!(non_negative_finite("m", -1.0).is_err());
        assert!(non_negative_finite("m", f32::NAN).is_err());
        assert!(non_negative_finite("m", 0.0).is_ok());
    }

    #[test]
    fn embedding_dim_bounds() {
        assert!(embedding_dim(0).is_err());
        assert!(embedding_dim(u32::MAX).is_err());
        assert_eq!(embedding_dim(512), Ok(512));
    }

    #[test]
    fn batch_norm_params() {
        assert!(batch_norm(8, 0, 0, 0, 0).is_err());
        assert!(batch_norm(8, 2, 1, 1, 1).is_err());
        assert!(batch_norm(7, 2, 2, 2, 2).is_err());
        assert_eq!(batch_norm(8, 2, 2, 2, 2), Ok(2));
    }
}
