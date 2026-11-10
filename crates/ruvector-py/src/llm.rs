//! Bounded full precision KV cache backed by the real ruvllm runtime.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ruvllm::kv_cache::{KvCacheConfig, KvMemoryPool, TwoTierKvCache};
use ruvllm::types::Precision;
use std::sync::Arc;

const MAX_ELEMENTS: usize = 8_388_608;
const MAX_MAGNITUDE: f32 = 1.0e10;

fn validate_vector(data: &[f32]) -> PyResult<()> {
    if data
        .iter()
        .any(|v| !v.is_finite() || v.abs() > MAX_MAGNITUDE)
    {
        return Err(PyValueError::new_err(
            "coordinates must be finite with absolute value <= 1e10",
        ));
    }
    Ok(())
}

/// Full precision, bounded CPU key/value cache using ruvllm's native runtime.
///
/// This is an inference building block, not a text generation engine. Inputs
/// are flat sequences of float32 values, token-major. Attention treats each
/// token as one vector (not independent multi-head attention). No model is
/// loaded, no network access occurs, and no quantization or eviction occurs.
/// Capacity exhaustion raises ValueError before modifying the cache.
#[pyclass(name = "KVCache", module = "ruvector")]
pub struct PyKVCache {
    inner: TwoTierKvCache,
    dimensions: usize,
    max_tokens: usize,
}

#[pymethods]
impl PyKVCache {
    /// Allocate an empty cache. Limits: dimensions <= 4096, max_tokens <=
    /// 65536, and dimensions * max_tokens <= 8388608 (64 MiB raw K/V).
    #[new]
    #[pyo3(signature = (dimensions, max_tokens = 4096))]
    fn new(dimensions: usize, max_tokens: usize) -> PyResult<Self> {
        if dimensions == 0
            || dimensions > 4096
            || max_tokens == 0
            || max_tokens > 65536
            || dimensions
                .checked_mul(max_tokens)
                .is_none_or(|n| n > MAX_ELEMENTS)
        {
            return Err(PyValueError::new_err("require dimensions in 1..=4096, max_tokens in 1..=65536, and their product <= 8388608"));
        }
        // Staying entirely within the tail excludes quantization and eviction.
        // A lazy pool avoids the default constructor's eager prewarm allocations.
        let config = KvCacheConfig {
            tail_length: max_tokens,
            tail_precision: Precision::FP32,
            store_precision: Precision::FP32,
            max_tokens,
            num_kv_heads: 1,
            head_dim: dimensions,
            migration_batch: 1,
        };
        let pool = Arc::new(KvMemoryPool::new(dimensions, 4));
        Ok(Self {
            inner: TwoTierKvCache::with_pool(config, pool),
            dimensions,
            max_tokens,
        })
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    #[getter]
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn __len__(&self) -> usize {
        self.inner.stats().total_tokens
    }

    /// Append one or more tokens. Equal nonempty flat arrays must contain a
    /// multiple of dimensions values. Invalid calls leave the cache unchanged.
    fn append(&mut self, keys: Vec<f32>, values: Vec<f32>) -> PyResult<()> {
        if keys.is_empty()
            || keys.len() != values.len()
            || (keys.len())
                .checked_rem(self.dimensions)
                .expect("attempt to calculate the remainder with a divisor of zero")
                != 0
        {
            return Err(PyValueError::new_err(
                "keys and values must have equal nonzero lengths divisible by dimensions",
            ));
        }
        let incoming = keys.len() / self.dimensions;
        if incoming > self.max_tokens - self.__len__() {
            return Err(PyValueError::new_err(
                "cache capacity exceeded; clear the cache before appending",
            ));
        }
        validate_vector(&keys)?;
        validate_vector(&values)?;
        self.inner
            .append(&keys, &values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return copied flat (keys, values), in insertion order.
    fn get_all(&self) -> (Vec<f32>, Vec<f32>) {
        self.inner.get_all_kv()
    }

    /// Compute softmax(query dot keys * scale) weighted values in native Rust.
    /// Default scale is 1/sqrt(dimensions). An empty cache returns zeros.
    #[pyo3(signature = (query, scale = None))]
    fn attend(&self, query: Vec<f32>, scale: Option<f32>) -> PyResult<Vec<f32>> {
        if query.len() != self.dimensions {
            return Err(PyValueError::new_err("query length must equal dimensions"));
        }
        validate_vector(&query)?;
        let scale = scale.unwrap_or_else(|| 1.0 / (self.dimensions as f32).sqrt());
        if !scale.is_finite() || scale <= 0.0 || scale > MAX_MAGNITUDE {
            return Err(PyValueError::new_err(
                "scale must be finite, positive and <= 1e10",
            ));
        }
        let output = self
            .inner
            .attend(&query, scale)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if output.iter().any(|v| !v.is_finite()) {
            return Err(PyRuntimeError::new_err(
                "native attention produced a non-finite value",
            ));
        }
        Ok(output)
    }

    /// Remove every token while preserving cache configuration.
    fn clear(&mut self) {
        self.inner.clear();
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKVCache>()
}
