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
    keys: &[&[f32]],
    values: &[&[f32]],
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
pub fn qk(dim: usize, query: &[f32], keys: &[&[f32]]) -> Result<()> {
    len("query", query, dim)?;
    for (i, key) in keys.iter().enumerate() {
        len(&format!("keys[{i}]"), key, dim)?;
    }
    Ok(())
}

/// Copies JS typed arrays into owned vectors (for work that leaves the JS
/// thread, e.g. async tasks).
pub fn to_vecs(arrays: Vec<Float32Array>) -> Vec<Vec<f32>> {
    arrays.into_iter().map(|a| a.to_vec()).collect()
}

/// Borrows owned vectors as slices.
pub fn as_slices(vectors: &[Vec<f32>]) -> Vec<&[f32]> {
    vectors.iter().map(|v| v.as_slice()).collect()
}

/// A JavaScript `Float32Array[]` borrowed for the duration of one synchronous
/// call.
///
/// `Vec<Float32Array>` makes napi-rs create (and later delete) a persistent
/// reference, an `Arc` and a finalizer box for every element, and the old
/// bindings then copied every element into a `Vec<f32>`. For 1,000 keys and
/// 1,000 values that was ~2,000 reference round trips plus 2,000 copies per
/// call, which made the native binding slower than a plain JS loop. This type
/// only reads each element's backing-store pointer (`napi_get_element` +
/// `napi_get_typedarray_info`) and never copies.
///
/// Soundness: the pointers are only used while the synchronous native call
/// runs. The outer array is an argument of that call, so every element stays
/// reachable, and V8 typed-array backing stores are not moved by the GC. Do
/// not hold an `F32Rows` across an `await` or send it to another thread (it is
/// neither `Send` nor `Sync`).
pub struct F32Rows {
    rows: Vec<(*const f32, usize)>,
}

impl F32Rows {
    /// Length of the first row, if any.
    pub fn first_len(&self) -> Option<usize> {
        self.rows.first().map(|r| r.1)
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Borrows every row as a slice.
    pub fn slices(&self) -> Vec<&[f32]> {
        self.rows
            .iter()
            .map(|&(ptr, len)| {
                if len == 0 || ptr.is_null() {
                    &[][..]
                } else {
                    // SAFETY: see the type-level comment; `ptr`/`len` come
                    // from napi_get_typedarray_info for a Float32Array that
                    // outlives this synchronous call.
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                }
            })
            .collect()
    }
}

impl TypeName for F32Rows {
    fn type_name() -> &'static str {
        "Float32Array[]"
    }

    fn value_type() -> ValueType {
        ValueType::Object
    }
}

impl ValidateNapiValue for F32Rows {}

impl FromNapiValue for F32Rows {
    unsafe fn from_napi_value(env: sys::napi_env, value: sys::napi_value) -> Result<Self> {
        let mut is_array = false;
        check_status!(
            sys::napi_is_array(env, value, &mut is_array),
            "Failed to check for an array"
        )?;
        if !is_array {
            return Err(invalid("expected an array of Float32Array".to_string()));
        }
        let mut length = 0u32;
        check_status!(
            sys::napi_get_array_length(env, value, &mut length),
            "Failed to read array length"
        )?;
        let mut rows = Vec::with_capacity(length as usize);
        for i in 0..length {
            let mut element = std::ptr::null_mut();
            check_status!(
                sys::napi_get_element(env, value, i, &mut element),
                "Failed to read array element"
            )?;
            let mut kind = 0;
            let mut len = 0usize;
            let mut data = std::ptr::null_mut();
            let mut buffer = std::ptr::null_mut();
            let mut offset = 0usize;
            let status = sys::napi_get_typedarray_info(
                env,
                element,
                &mut kind,
                &mut len,
                &mut data,
                &mut buffer,
                &mut offset,
            );
            if status != sys::Status::napi_ok || kind != sys::TypedarrayType::float32_array {
                return Err(invalid(format!("element {i} must be a Float32Array")));
            }
            rows.push((data as *const f32, len));
        }
        Ok(Self { rows })
    }
}

/// Borrows a flat row-major `Float32Array` as `rows` slices of equal length.
///
/// `what` names the argument in errors. The length must be a non-zero
/// multiple of `row_len`.
pub fn flat_rows<'a>(what: &str, data: &'a [f32], row_len: usize) -> Result<Vec<&'a [f32]>> {
    if row_len == 0 || data.is_empty() || data.len() % row_len != 0 {
        return Err(invalid(format!(
            "{what} must be a non-empty Float32Array whose length is a multiple of {row_len}, got {}",
            data.len()
        )));
    }
    Ok(data.chunks_exact(row_len).collect())
}
