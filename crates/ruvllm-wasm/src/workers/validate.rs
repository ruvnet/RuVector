//! Shape checks for `ParallelInference` inputs.
//!
//! Dimensions arrive from JS as `usize` (32-bit on wasm32). Products are
//! computed with `checked_mul`, because a wrapped product can match a short
//! buffer's length and send the kernels indexing past it.

/// Validate `C[m x n] = A[m x k] * B[k x n]`.
pub(crate) fn matmul_dims(
    a_len: usize,
    b_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    let a_expected = m.checked_mul(k).ok_or("m * k overflows")?;
    let b_expected = k.checked_mul(n).ok_or("k * n overflows")?;
    m.checked_mul(n)
        .and_then(|mn| mn.checked_mul(k))
        .ok_or("m * n * k overflows")?;
    if a_len != a_expected {
        return Err(format!(
            "Matrix A size mismatch: expected {} ({}x{}), got {}",
            a_expected, m, k, a_len
        ));
    }
    if b_len != b_expected {
        return Err(format!(
            "Matrix B size mismatch: expected {} ({}x{}), got {}",
            b_expected, k, n, b_len
        ));
    }
    Ok(())
}

/// Validate Q/K/V of shape `(num_heads, seq_len, head_dim)`; returns the
/// element count of one tensor.
pub(crate) fn attention_dims(
    q_len: usize,
    k_len: usize,
    v_len: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<usize, String> {
    let expected = num_heads
        .checked_mul(seq_len)
        .and_then(|x| x.checked_mul(head_dim))
        .ok_or("num_heads * seq_len * head_dim overflows")?;
    if expected == 0 {
        // a zero head_dim would still size a seq_len x seq_len score matrix
        return Err("num_heads, seq_len and head_dim must be non-zero".into());
    }
    // the score matrix is seq_len x seq_len
    seq_len
        .checked_mul(seq_len)
        .ok_or("seq_len * seq_len overflows")?;
    if q_len != expected || k_len != expected || v_len != expected {
        return Err(format!(
            "Tensor size mismatch: expected {}, got Q={}, K={}, V={}",
            expected, q_len, k_len, v_len
        ));
    }
    Ok(expected)
}

/// Validate layer-norm parameters: `gamma`/`beta` share the hidden size and
/// the input is a whole number of rows.
pub(crate) fn layer_norm_params(
    input_len: usize,
    gamma_len: usize,
    beta_len: usize,
) -> Result<(), String> {
    if gamma_len == 0 {
        return Err("gamma must not be empty".into());
    }
    if beta_len != gamma_len {
        return Err(format!(
            "beta length {} must equal gamma length {}",
            beta_len, gamma_len
        ));
    }
    if input_len % gamma_len != 0 {
        return Err(format!(
            "input length {} is not a multiple of hidden size {}",
            input_len, gamma_len
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_rejects_wrapped_products() {
        // On wasm32 65536 * 65536 wraps to 0, which matched an empty A.
        assert!(matmul_dims(0, 65536, 65536, 1, 65536).is_err());
        assert!(matmul_dims(0, 0, usize::MAX, 0, 2).is_err());
        assert!(matmul_dims(6, 6, 2, 2, 3).is_ok());
        assert!(matmul_dims(5, 6, 2, 2, 3).is_err());
    }

    #[test]
    fn attention_rejects_overflow_and_mismatch() {
        assert!(attention_dims(0, 0, 0, usize::MAX, 2, 2).is_err());
        assert_eq!(attention_dims(8, 8, 8, 2, 2, 2), Ok(8));
        assert!(attention_dims(8, 7, 8, 2, 2, 2).is_err());
    }

    #[test]
    fn layer_norm_rejects_bad_params() {
        assert!(layer_norm_params(10, 0, 0).is_err());
        assert!(layer_norm_params(8, 4, 3).is_err());
        assert!(layer_norm_params(10, 4, 4).is_err());
        assert!(layer_norm_params(8, 4, 4).is_ok());
    }
}
