//! k-means clustering Python binding — wraps `ruvector_cluster_rag`'s
//! `cluster::kmeans` (the real ML clustering algorithm in the workspace;
//! NOT `crates/ruvector-cluster`, which is distributed-sharding
//! coordination infra, a premise mismatch for "clustering" — see
//! ADR-352).
//!
//! Stub pre-wired by the orchestrating session (ADR-352) so a parallel
//! fork can fill in the module body without touching `Cargo.toml` or
//! `lib.rs` — see those files' comments for why. `register()` is a no-op
//! placeholder until the fork lands real `#[pyclass]`/`#[pyfunction]`
//! items here.

use pyo3::prelude::*;

pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
