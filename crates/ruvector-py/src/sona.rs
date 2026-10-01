//! SONA (inference-only) Python binding — wraps `ruvector_sona::SonaEngine`
//! (crate name `ruvector-sona`, package directory `crates/sona`).
//!
//! Scope per ADR-352's capability inventory: `apply_micro_lora`/
//! `apply_base_lora` are stateless-per-call transforms that work the
//! instant the engine is constructed (an untrained LoRA is close to an
//! identity transform) — the usable inference-only entry point. The
//! online-learning half (`begin_trajectory`/`TrajectoryBuilder::add_step`/
//! `end_trajectory`/`tick`/`force_learn`/`find_patterns`) is real but only
//! does something once fed genuine reward/quality signal; exposing the API
//! is in scope, but be honest that a fresh engine's rerank is close to a
//! no-op until trained — do not oversell forward-pass-only as a quality
//! improvement (ADR-352 §"Be honest about what GNN forward on a
//! random-init layer buys the user" — the same caveat applies here).
//!
//! Stub pre-wired by the orchestrating session so a parallel fork can fill
//! in the module body without touching `Cargo.toml` or `lib.rs` — see
//! those files' comments for why. `register()` is a no-op placeholder
//! until the fork lands real `#[pyclass]` types here.

use pyo3::prelude::*;

pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
