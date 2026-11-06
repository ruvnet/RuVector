//! Input validation for the NAPI boundary.
//!
//! Every size and vector that arrives from JavaScript is checked here before it
//! reaches the Rust attention code, so bad input becomes a JavaScript `Error`
//! instead of a panic. Sync bindings additionally use `#[napi(catch_unwind)]` as a
//! backstop for anything these checks miss.

use napi::bindgen_prelude::*;

/// Largest accepted embedding dimension.
pub const MAX_DIM: u32 = 1 << 16;
/// Largest accepted head / expert / feature / position count.
pub const MAX_COUNT: u32 = 1 << 20;
/// Largest number of `f32` elements a constructor may preallocate (256 MiB).
pub const MAX_ELEMENTS: u64 = 1 << 26;

fn invalid(message: String) -> Error {
    Error::new(Status::InvalidArg, message)
}

/// `value` must be in `1..=MAX_DIM`.
pub fn dim(name: &str, value: u32) -> Result<usize> {
    if value == 0 || value > MAX_DIM {
        return Err(invalid(format!(
            "{name} must be between 1 and {MAX_DIM}, got {value}"
        )));
    }
    Ok(value as usize)
}

/// `value` must be in `1..=MAX_COUNT`.
pub fn count(name: &str, value: u32) -> Result<usize> {
    if value == 0 || value > MAX_COUNT {
        return Err(invalid(format!(
            "{name} must be between 1 and {MAX_COUNT}, got {value}"
        )));
    }
    Ok(value as usize)
}

/// `factors` multiplied together must not exceed [`MAX_ELEMENTS`].
pub fn elements(what: &str, factors: &[usize]) -> Result<()> {
    let total = factors
        .iter()
        .try_fold(1u64, |acc, &f| acc.checked_mul(f as u64));
    match total {
        Some(n) if n <= MAX_ELEMENTS => Ok(()),
        _ => Err(invalid(format!(
            "{what} would need more than {MAX_ELEMENTS} elements"
        ))),
    }
}

/// `dim` must be divisible by a non-zero `num_heads`.
pub fn heads(dim: usize, num_heads: u32) -> Result<usize> {
    let h = count("numHeads", num_heads)?;
    if dim % h != 0 {
        return Err(invalid(format!(
            "Dimension {dim} must be divisible by number of heads {h}"
        )));
    }
    Ok(h)
}

/// A finite number, optionally required to be strictly positive.
pub fn finite(name: &str, value: f64, positive: bool) -> Result<f32> {
    if !value.is_finite() || (positive && value <= 0.0) {
        let need = if positive {
            "a positive finite"
        } else {
            "a finite"
        };
        return Err(invalid(format!(
            "{name} must be {need} number, got {value}"
        )));
    }
    Ok(value as f32)
}

/// `vector` must have exactly `expected` elements.
pub fn len(name: &str, vector: &[f32], expected: usize) -> Result<()> {
    if vector.len() != expected {
        return Err(invalid(format!(
            "{name} must have length {expected}, got {}",
            vector.len()
        )));
    }
    Ok(())
}

/// Checks a query / keys / values triple.
///
/// The query and every key must have length `dim`, keys and values must pair up,
/// and every value must have the same length — `dim` when `value_dim` is
/// `Some(dim)`, otherwise the length of the first value.
pub fn qkv(
    dim: usize,
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    value_dim: Option<usize>,
) -> Result<()> {
    len("query", query, dim)?;
    if keys.is_empty() {
        return Err(invalid("keys must not be empty".to_string()));
    }
    if keys.len() != values.len() {
        return Err(invalid(format!(
            "keys and values must have the same count, got {} and {}",
            keys.len(),
            values.len()
        )));
    }
    for (i, key) in keys.iter().enumerate() {
        len(&format!("keys[{i}]"), key, dim)?;
    }
    let expected = value_dim.unwrap_or(values[0].len());
    for (i, value) in values.iter().enumerate() {
        len(&format!("values[{i}]"), value, expected)?;
    }
    Ok(())
}

/// Same as [`qkv`] for keys only (no values).
pub fn qk(dim: usize, query: &[f32], keys: &[Vec<f32>]) -> Result<()> {
    len("query", query, dim)?;
    for (i, key) in keys.iter().enumerate() {
        len(&format!("keys[{i}]"), key, dim)?;
    }
    Ok(())
}

/// Copies JS typed arrays into owned vectors.
pub fn to_vecs(arrays: Vec<Float32Array>) -> Vec<Vec<f32>> {
    arrays.into_iter().map(|a| a.to_vec()).collect()
}

/// Borrows owned vectors as slices.
pub fn as_slices(vectors: &[Vec<f32>]) -> Vec<&[f32]> {
    vectors.iter().map(|v| v.as_slice()).collect()
}
