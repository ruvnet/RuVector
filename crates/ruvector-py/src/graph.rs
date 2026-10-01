//! `GraphDB` Python class — wraps `ruvector_graph::GraphDB`.
//!
//! Stub pre-wired by the orchestrating session (ADR-352) so a parallel
//! fork can fill in the module body without touching `Cargo.toml` or
//! `lib.rs` — see those files' comments for why. `register()` is a no-op
//! placeholder until the fork lands real `#[pyclass]` types here.

use pyo3::prelude::*;

pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
