//! `GnnLayer` (GNN forward-pass rerank) and `AttentionReranker`
//! (attention-based rerank) Python classes — wrap
//! `ruvector_gnn::layer::RuvectorLayer` and
//! `ruvector_attention::attention::ScaledDotProductAttention`.
//!
//! ## Honesty note (read before using `GnnLayer`)
//!
//! `RuvectorLayer::new` initialises every weight matrix with Xavier/Glorot
//! random values (see `crates/ruvector-gnn/src/layer.rs`'s `Linear::new`).
//! There is no training step anywhere in this binding or in
//! `ruvector_gnn` as currently wired here — `forward()` runs immediately,
//! but on a freshly-constructed layer it is a **random projection**, not a
//! learned reranking signal. Calling `GnnLayer(...).forward(...)` does NOT
//! make search results better by itself; it only becomes a quality
//! improvement once the weights are actually trained (out of scope for
//! this binding) or loaded from a trained checkpoint via `from_json`.
//! `AttentionReranker` is different: `softmax(QK^T/√d)V` is a fixed,
//! trainless, legitimate reranking primitive (no weights to train), so its
//! output is meaningful from the first call.
//!
//! ## Why both primitives live in one file
//!
//! Per the orchestrating session's plan (ADR-352 capability-expansion
//! slice): both are "rerank" primitives, both are small, and splitting
//! them would just add a second `mod` line to `lib.rs` for no benefit.
//! This file stays well under the 500-line budget.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use ruvector_attention::attention::ScaledDotProductAttention;
use ruvector_attention::traits::Attention;
use ruvector_gnn::layer::RuvectorLayer;

use crate::error::{to_pyerr_attention, to_pyerr_gnn, RuVectorError};

/// `(blended, weights)` — one `AttentionReranker.rerank` result.
type RerankResult<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>);

/// GNN forward-pass rerank layer — wraps `ruvector_gnn::layer::RuvectorLayer`
/// (message passing + multi-head attention aggregation + GRU update +
/// layer norm, operating on HNSW-topology-shaped inputs: one node, its
/// neighbor embeddings, and per-edge weights). See the module docstring
/// above for why an untrained instance's `forward()` is not a quality
/// improvement by itself.
///
/// Not `unsendable`: `RuvectorLayer` (`crates/ruvector-gnn/src/layer.rs`)
/// is composed entirely of `ndarray::Array1<f32>`/`Array2<f32>` fields
/// (via its `Linear`/`LayerNorm`/`GRUCell`/`MultiHeadAttention` members)
/// plus plain `f32`/`usize` — no `Rc`, `RefCell`, or raw pointers anywhere
/// in the type. Same check `rabitq.rs`/`hnsw.rs` already do: a type built
/// only from `Send + Sync` leaves is `Send + Sync` itself, so pyo3's
/// auto-derive gives a real `Send + Sync` `#[pyclass]` for free, and
/// marking it `unsendable` would only add the exact cross-thread panic
/// risk ADR-352 flags elsewhere.
///
/// `input_dim`/`hidden_dim`/`heads`/`dropout` are tracked redundantly on
/// this wrapper (not available as getters on `RuvectorLayer` itself)
/// purely so this binding can validate shapes *before* calling into Rust:
/// `RuvectorLayer::forward` calls straight into `ndarray`'s `Array2::dot`,
/// which **panics** (not a `Result`) on a shape mismatch. Pre-validating
/// here, the same way `hnsw.rs`/`rabitq.rs` pre-validate shapes before
/// touching their backends, turns that panic into a clean `ValueError`.
#[pyclass(name = "GnnLayer", module = "ruvector._native")]
pub struct GnnLayer {
    inner: RuvectorLayer,
    input_dim: usize,
    hidden_dim: usize,
    heads: usize,
    dropout: f32,
}

#[pymethods]
impl GnnLayer {
    /// `RuvectorLayer::new(input_dim, hidden_dim, heads, dropout)`.
    ///
    /// `heads` must divide `hidden_dim` and `dropout` must be in
    /// `[0.0, 1.0]` — both checked by the real constructor and surfaced
    /// as `RuVectorError` via `to_pyerr_gnn` (`GnnError::LayerConfig`).
    /// `input_dim`/`hidden_dim`/`heads` of `0` are rejected directly as
    /// `ValueError` here, before ever calling into Rust, since a `0`
    /// dimension would otherwise degrade into confusing downstream shape
    /// errors rather than a clear message at the boundary.
    #[new]
    #[pyo3(signature = (input_dim, hidden_dim, heads, dropout = 0.0))]
    fn new(input_dim: usize, hidden_dim: usize, heads: usize, dropout: f32) -> PyResult<Self> {
        if input_dim == 0 {
            return Err(PyValueError::new_err("input_dim must be > 0"));
        }
        if hidden_dim == 0 {
            return Err(PyValueError::new_err("hidden_dim must be > 0"));
        }
        if heads == 0 {
            return Err(PyValueError::new_err("heads must be > 0"));
        }
        let inner =
            RuvectorLayer::new(input_dim, hidden_dim, heads, dropout).map_err(to_pyerr_gnn)?;
        Ok(Self {
            inner,
            input_dim,
            hidden_dim,
            heads,
            dropout,
        })
    }

    /// Forward pass for one node.
    ///
    /// `node`: 1D float32 array of length `input_dim`.
    /// `neighbors`: 2D float32 array of shape `(n, input_dim)` — may have
    /// `n == 0` (no neighbors; the layer falls back to a normalized
    /// projection of `node` alone, per `RuvectorLayer::forward`).
    /// `weights`: optional 1D float32 array of length `n` (per-neighbor
    /// edge weight, e.g. an HNSW distance-derived score). Defaults to a
    /// uniform `[1.0, ...]` vector when omitted — `RuvectorLayer`
    /// normalizes weights to sum to 1 internally regardless, so a
    /// uniform default and an explicit all-equal array behave
    /// identically; the default is just the ergonomic no-argument case.
    ///
    /// Returns a 1D float32 array of length `hidden_dim`.
    #[pyo3(signature = (node, neighbors, weights = None))]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        node: PyReadonlyArray1<'_, f32>,
        neighbors: PyReadonlyArray2<'_, f32>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if !node.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "node must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        if node.len() != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "node dimension mismatch: expected {}, got {}",
                self.input_dim,
                node.len()
            )));
        }
        if !neighbors.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "neighbors must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        let shape = neighbors.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "neighbors must be 2D, got {}D",
                shape.len()
            )));
        }
        let (n, dim) = (shape[0], shape[1]);
        if n > 0 && dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "neighbors dimension mismatch: expected (*, {}), got (*, {})",
                self.input_dim, dim
            )));
        }
        if let Some(w) = &weights {
            if !w.is_c_contiguous() {
                return Err(PyTypeError::new_err(
                    "weights must be C-contiguous; pass np.ascontiguousarray(...) first",
                ));
            }
            if w.len() != n {
                return Err(PyValueError::new_err(format!(
                    "weights length ({}) must match neighbors row count ({})",
                    w.len(),
                    n
                )));
            }
        }

        let node_vec = node.as_slice()?.to_vec();
        let neighbor_rows: Vec<Vec<f32>> = if n == 0 {
            Vec::new()
        } else {
            neighbors
                .as_slice()?
                .chunks_exact(dim)
                .map(|row| row.to_vec())
                .collect()
        };
        let weight_vec: Vec<f32> = match &weights {
            Some(w) => w.as_slice()?.to_vec(),
            None => vec![1.0; n],
        };

        // Heavy work (several Linear/GRU/attention forward passes) — drop
        // the GIL, same calculus as `HnswIndex::search`/`RabitqIndex::search`.
        let output = py.detach(|| self.inner.forward(&node_vec, &neighbor_rows, &weight_vec));
        Ok(PyArray1::from_vec(py, output))
    }

    /// Serialize to JSON. `RuvectorLayer` derives `Serialize`/`Deserialize`
    /// already (`crates/ruvector-gnn/src/layer.rs`), but it exposes no
    /// getters for the constructor's own `input_dim`/`hidden_dim`/`heads`/
    /// `dropout` — this binding needs those back on `from_json` to
    /// pre-validate `forward()` shapes (see the class doc comment), so
    /// the JSON envelope carries them alongside the serialized layer
    /// rather than serializing `RuvectorLayer` bare.
    fn to_json(&self) -> PyResult<String> {
        let layer_value = serde_json::to_value(&self.inner)
            .map_err(|e| RuVectorError::new_err(format!("serialize GnnLayer: {e}")))?;
        let envelope = serde_json::json!({
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "dropout": self.dropout,
            "layer": layer_value,
        });
        serde_json::to_string(&envelope)
            .map_err(|e| RuVectorError::new_err(format!("serialize GnnLayer: {e}")))
    }

    /// Deserialize a value previously produced by `to_json`.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let mut envelope: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| RuVectorError::new_err(format!("deserialize GnnLayer: {e}")))?;
        let input_dim = envelope
            .get("input_dim")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| RuVectorError::new_err("GnnLayer JSON missing \"input_dim\""))?
            as usize;
        let hidden_dim = envelope
            .get("hidden_dim")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| RuVectorError::new_err("GnnLayer JSON missing \"hidden_dim\""))?
            as usize;
        let heads = envelope
            .get("heads")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| RuVectorError::new_err("GnnLayer JSON missing \"heads\""))?
            as usize;
        let dropout = envelope
            .get("dropout")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| RuVectorError::new_err("GnnLayer JSON missing \"dropout\""))?
            as f32;
        let layer_value = envelope
            .get_mut("layer")
            .ok_or_else(|| RuVectorError::new_err("GnnLayer JSON missing \"layer\""))?
            .take();
        let inner: RuvectorLayer = serde_json::from_value(layer_value)
            .map_err(|e| RuVectorError::new_err(format!("deserialize GnnLayer: {e}")))?;
        Ok(Self {
            inner,
            input_dim,
            hidden_dim,
            heads,
            dropout,
        })
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[getter]
    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    #[getter]
    fn heads(&self) -> usize {
        self.heads
    }

    #[getter]
    fn dropout(&self) -> f32 {
        self.dropout
    }

    fn __repr__(&self) -> String {
        format!(
            "GnnLayer(input_dim={}, hidden_dim={}, heads={}, dropout={})",
            self.input_dim, self.hidden_dim, self.heads, self.dropout
        )
    }
}

/// Attention-based rerank — wraps
/// `ruvector_attention::attention::ScaledDotProductAttention`
/// (`softmax(QK^T/√d)V`). Unlike `GnnLayer`, this has no trainable weights
/// at all, so its output is a meaningful, deterministic reranking signal
/// from the very first call — see the module docstring's honesty note.
///
/// Both the blended output vector *and* the raw per-candidate attention
/// weights are returned from `rerank()`. For a RAG-style reranking caller,
/// the weights (not the blended vector) are usually the actually useful
/// part — they give a direct score per candidate that can be used to
/// re-sort the original candidate list, whereas the blended vector is a
/// single new point that doesn't by itself tell you which candidate
/// "won". Returning both costs nothing extra (the weights are a
/// by-product of computing the blend) and lets callers use whichever
/// they need.
///
/// Not `unsendable`: `ScaledDotProductAttention` is a single `usize`
/// field (`dim`) with no interior mutability — trivially `Send + Sync`,
/// same reasoning as `GnnLayer` above.
#[pyclass(name = "AttentionReranker", module = "ruvector._native")]
pub struct AttentionReranker {
    inner: ScaledDotProductAttention,
    dim: usize,
}

#[pymethods]
impl AttentionReranker {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        Ok(Self {
            inner: ScaledDotProductAttention::new(dim),
            dim,
        })
    }

    /// Rerank `candidates` (shape `(n, dim)`, used as both keys and values
    /// per the softmax-attention formula) against `query` (shape `(dim,)`).
    ///
    /// Returns `(blended, weights)`:
    /// - `blended`: the attention-weighted blend of `candidates`, computed
    ///   by the real `ScaledDotProductAttention::compute` — this is the
    ///   library's own, authoritative output.
    /// - `weights`: the per-candidate softmax attention weight, in the
    ///   same order as `candidates`' rows, summing to ~1.0. These are
    ///   **recomputed locally** using the identical formula
    ///   `softmax(QK^T/√dim)` that `ScaledDotProductAttention` uses
    ///   internally (`crates/ruvector-attention/src/attention/
    ///   scaled_dot_product.rs`'s `compute_scores`/`softmax`), because
    ///   those helper methods are private to that crate and the `Attention`
    ///   trait only returns the blended vector, never the intermediate
    ///   weights. The two are deterministically linked — `blended` is
    ///   exactly `weights @ candidates` — and this binding's own test
    ///   suite pins that relationship down (`tests/test_gnn.py`), so if
    ///   the upstream formula ever drifts from this duplicate, the test
    ///   catches it.
    fn rerank<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'_, f32>,
        candidates: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<RerankResult<'py>> {
        if !query.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "query must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        if query.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            )));
        }
        if !candidates.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "candidates must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        let shape = candidates.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "candidates must be 2D, got {}D",
                shape.len()
            )));
        }
        let (n, dim) = (shape[0], shape[1]);
        if n == 0 {
            // Pre-validate before any local softmax math: an empty
            // candidate set would otherwise divide by a zero `sum_exp`
            // below, well before `ScaledDotProductAttention::compute`
            // gets a chance to raise its own `EmptyInput` error.
            return Err(PyValueError::new_err(
                "candidates must contain at least 1 row",
            ));
        }
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "candidates dimension mismatch: expected (*, {}), got (*, {})",
                self.dim, dim
            )));
        }

        let query_vec = query.as_slice()?.to_vec();
        let candidate_rows: Vec<Vec<f32>> = candidates
            .as_slice()?
            .chunks_exact(dim)
            .map(|row| row.to_vec())
            .collect();

        let (blended, weights) = py.detach(|| -> PyResult<(Vec<f32>, Vec<f32>)> {
            let key_refs: Vec<&[f32]> = candidate_rows.iter().map(|r| r.as_slice()).collect();
            let value_refs: Vec<&[f32]> = key_refs.clone();

            // Authoritative blended output, via the real crate code.
            let blended = self
                .inner
                .compute(&query_vec, &key_refs, &value_refs)
                .map_err(to_pyerr_attention)?;

            // Weights: duplicated formula — see the doc comment above for
            // why (`compute_scores`/`softmax` are private upstream).
            let scale = (self.dim as f32).sqrt();
            let scores: Vec<f32> = candidate_rows
                .iter()
                .map(|c| {
                    query_vec
                        .iter()
                        .zip(c.iter())
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        / scale
                })
                .collect();
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

            Ok((blended, weights))
        })?;

        Ok((
            PyArray1::from_vec(py, blended),
            PyArray1::from_vec(py, weights),
        ))
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __repr__(&self) -> String {
        format!("AttentionReranker(dim={})", self.dim)
    }
}

/// Module registration — called from `lib.rs`'s `_native` module init.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GnnLayer>()?;
    m.add_class::<AttentionReranker>()?;
    Ok(())
}
