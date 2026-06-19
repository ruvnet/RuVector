//! Rotary position embedding helpers shared by the attention variants.

use candle_core::{DType, Device, Tensor, D};

use crate::error::{Result, RuvLLMError};

pub(crate) fn cand(e: candle_core::Error) -> RuvLLMError {
    RuvLLMError::Model(format!("candle (openmythos): {e}"))
}

/// Precompute `(cos, sin)` RoPE tables of shape `[seq, head_dim]` for absolute
/// positions `offset .. offset + seq`.
pub(crate) fn rope_tables(
    seq: usize,
    offset: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let theta = theta as f64;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device).map_err(cand)?;
    let positions: Vec<f32> = (0..seq).map(|p| (p + offset) as f32).collect();
    let positions = Tensor::from_vec(positions, (seq, 1), device).map_err(cand)?;
    let freqs = positions.matmul(&inv_freq).map_err(cand)?;
    let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1).map_err(cand)?;
    let cos = freqs.cos().map_err(cand)?.to_dtype(dtype).map_err(cand)?;
    let sin = freqs.sin().map_err(cand)?.to_dtype(dtype).map_err(cand)?;
    Ok((cos, sin))
}

/// Apply RoPE to `[b, n, seq, head_dim]` using `[seq, head_dim]` tables.
pub(crate) fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _n, seq, hd) = x.dims4().map_err(cand)?;
    let cos = cos
        .narrow(0, 0, seq)
        .map_err(cand)?
        .reshape((1, 1, seq, hd))
        .map_err(cand)?;
    let sin = sin
        .narrow(0, 0, seq)
        .map_err(cand)?
        .reshape((1, 1, seq, hd))
        .map_err(cand)?;
    let rot = rotate_half(x)?;
    (x.broadcast_mul(&cos).map_err(cand)? + rot.broadcast_mul(&sin).map_err(cand)?).map_err(cand)
}

/// `rotate_half([x1, x2]) = [-x2, x1]` along the last dimension.
pub(crate) fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let hd = x.dim(D::Minus1).map_err(cand)?;
    let half = hd / 2;
    let x1 = x.narrow(D::Minus1, 0, half).map_err(cand)?;
    let x2 = x.narrow(D::Minus1, half, hd - half).map_err(cand)?;
    Tensor::cat(&[&x2.neg().map_err(cand)?, &x1], D::Minus1).map_err(cand)
}

/// Repeat KV heads `n_rep` times for grouped-query attention.
pub(crate) fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv, seq, hd) = x.dims4().map_err(cand)?;
    x.unsqueeze(2)
        .map_err(cand)?
        .expand((b, kv, n_rep, seq, hd))
        .map_err(cand)?
        .reshape((b, kv * n_rep, seq, hd))
        .map_err(cand)
}

/// Additive causal mask of shape `[q_len, kv_len]`. Query position `i` (absolute
/// `offset + i`) may attend to key positions `<= offset + i`.
pub(crate) fn causal_mask(
    q_len: usize,
    kv_len: usize,
    offset: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let allowed = offset + i; // last key index this query may see
        for j in 0..kv_len {
            if j > allowed {
                data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (q_len, kv_len), device)
        .map_err(cand)?
        .to_dtype(dtype)
        .map_err(cand)
}
