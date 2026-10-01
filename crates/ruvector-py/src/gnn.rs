//! `RuvectorLayer` (GNN forward-pass rerank) and attention-rerank Python
//! classes — wrap `ruvector_gnn::layer::RuvectorLayer` and
//! `ruvector_attention`'s `ScaledDotProductAttention` (or a similar
//! primitive).
//!
//! Stub pre-wired by the orchestrating session (ADR-352) so a parallel
//! fork can fill in the module body without touching `Cargo.toml` or
//! `lib.rs` — see those files' comments for why. `register()` is a no-op
//! placeholder until the fork lands real `#[pyclass]` types here.

use pyo3::prelude::*;

pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
