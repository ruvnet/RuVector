//! SONA (inference-only) Python binding — wraps `ruvector_sona::SonaEngine`
//! (crate name `ruvector-sona`, package directory `crates/sona`).
//!
//! Scope per ADR-352's capability inventory: `apply_micro_lora`/
//! `apply_base_lora` are stateless-per-call transforms that work the
//! instant the engine is constructed — but be precise about what that
//! means, not just "close to identity": `MicroLoRA`'s up-projection is
//! zero-initialised at construction (`crates/sona/src/lora.rs`'s
//! `MicroLoRA::new`, `up_proj = vec![0.0f32; ...]`) and so is both of
//! `BaseLoRA`'s projections for every layer. Since both LoRA forward
//! passes are *residual* (`output` is seeded with the input and the
//! learned delta is added on top — see this module's own
//! `seed_residual_output` and the upstream NAPI binding's doc comment,
//! which states this outright), a freshly constructed engine's
//! `apply_micro_lora`/`apply_base_lora` is an **exact identity
//! transform**, not merely "close to one" — until the online-learning
//! half (which this binding deliberately does not expose — see below)
//! has actually updated those weights. Do not oversell forward-pass-only
//! as a quality improvement (ADR-352 §"Be honest about what GNN forward on
//! a random-init layer buys the user" — the same caveat applies here).
//!
//! **Out of scope, deliberately**: `begin_trajectory`/
//! `TrajectoryBuilder::add_step`/`end_trajectory`/`submit_trajectory`/
//! `tick`/`force_learn`/`find_patterns`. These are real but only do
//! something once fed a genuine reward/quality signal a test can't
//! meaningfully fabricate, and `end_trajectory` consumes its
//! `TrajectoryBuilder` by value — awkward to bind from a shared
//! `#[pyclass]` reference (would need `Option<T>` + `.take()` on the
//! Python-side wrapper). Left for a separate follow-up commit per the
//! task's own sequencing, not attempted here.
//!
//! Also out of scope: `export_lora_state` (safetensors) — not part of
//! this binding's required surface (`new`, `apply_micro_lora`,
//! `apply_base_lora`, `stats`, `save_state`/`load_state`); skipped to keep
//! this slice inference-only and minimal.

use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ruvector_sona::SonaEngine as RustSonaEngine;

use crate::error::RuVectorError;

/// Python-visible SONA engine. Backed by `ruvector_sona::SonaEngine`.
///
/// Not `unsendable`: every field `RustSonaEngine` holds under the hood is
/// `Arc<parking_lot::RwLock<_>>`-guarded (see `crates/sona/src/
/// loops/coordinator.rs`'s `LoopCoordinator`) — `Arc<RwLock<T>>` is
/// `Send + Sync` whenever `T: Send + Sync`, and every `T` here
/// (`MicroLoRA`, `BaseLoRA`, `ReasoningBank`, `EwcPlusPlus`) is a plain
/// data struct of `Vec<f32>`/`usize`/etc with no interior mutability of
/// its own and no thread-affine handle (no `Rc`, no raw pointer; checked
/// `crates/sona/src/lora.rs` directly — `MicroLoRA`'s weight
/// initialisation is a deterministic golden-ratio formula, not even
/// `rand`-backed, so there's no stored RNG handle of any kind to worry
/// about). This is the same check
/// `rabitq.rs`/`hnsw.rs` make before dropping `unsendable`, and it is also
/// verified structurally here: the crate compiles `SonaEngine` into a
/// non-`unsendable` `#[pyclass]`, which pyo3 0.29 only accepts for a type
/// that is actually `Send + Sync` — the compiler itself is the proof.
#[pyclass(name = "SonaEngine", module = "ruvector._native")]
pub struct SonaEngine {
    inner: RustSonaEngine,
}

/// Seed the LoRA output buffer with the input (residual semantics — both
/// `MicroLoRA::forward` and `BaseLoRA::forward_layer` compute `output +=
/// delta`, never `output = delta`; starting from zeros would silently
/// report only the learned delta, which is the identity vector on a fresh
/// engine). Mirrors `crates/sona/src/napi.rs`'s `apply_micro_lora`/
/// `apply_base_lora`, which do exactly this and say why in their comments.
fn seed_residual_output(input: &[f32]) -> Vec<f32> {
    input.to_vec()
}

#[pymethods]
impl SonaEngine {
    /// Create a new engine with default config, sized for `hidden_dim`.
    ///
    /// `hidden_dim` must be `> 0` — a zero-width engine has no valid LoRA
    /// shape and every forward pass would be a vacuous no-op; rejected
    /// here with `ValueError` rather than silently constructing a useless
    /// engine.
    #[new]
    fn new(hidden_dim: usize) -> PyResult<Self> {
        if hidden_dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "hidden_dim must be > 0",
            ));
        }
        Ok(Self {
            inner: RustSonaEngine::new(hidden_dim),
        })
    }

    /// Apply the micro-LoRA transform to `input` (length must equal
    /// `hidden_dim`) and return the transformed output.
    ///
    /// On a freshly constructed engine this is an **exact identity
    /// transform** — see this module's doc comment for why (zero-init
    /// up-projection + residual forward pass), not an approximation. It
    /// only starts doing something once the engine's online-learning loop
    /// (not exposed by this binding) has updated the LoRA weights.
    fn apply_micro_lora<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if !input.is_c_contiguous() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "input must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        let expected = self.inner.config().hidden_dim;
        let input_slice = input.as_slice()?;
        if input_slice.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input length ({}) must equal hidden_dim ({})",
                input_slice.len(),
                expected
            )));
        }
        let mut output = seed_residual_output(input_slice);
        self.inner.apply_micro_lora(input_slice, &mut output);
        Ok(PyArray1::from_vec(py, output))
    }

    /// Apply the base-LoRA transform for layer `layer_idx` to `input`
    /// (length must equal `hidden_dim`) and return the transformed output.
    ///
    /// `layer_idx` must be `< num_layers` (see the `num_layers` getter —
    /// fixed at 12 by `LoopCoordinator::with_config`, see
    /// `crates/sona/src/loops/coordinator.rs`). The underlying
    /// `BaseLoRA::forward_layer` silently no-ops on an out-of-range
    /// `layer_idx` instead of erroring; this binding deliberately does
    /// NOT mirror that silent behaviour, since a Python caller passing a
    /// bad layer index almost certainly wants to know, not get back an
    /// unexplained identity vector.
    ///
    /// Same "exact identity on a fresh engine" honesty note as
    /// `apply_micro_lora` applies here too (zero-init projections).
    fn apply_base_lora<'py>(
        &self,
        py: Python<'py>,
        layer_idx: usize,
        input: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if !input.is_c_contiguous() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "input must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        let expected = self.inner.config().hidden_dim;
        let input_slice = input.as_slice()?;
        if input_slice.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input length ({}) must equal hidden_dim ({})",
                input_slice.len(),
                expected
            )));
        }
        let n_layers = self.inner.coordinator().base_lora().read().num_layers();
        if layer_idx >= n_layers {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "layer_idx ({layer_idx}) must be < num_layers ({n_layers})"
            )));
        }
        let mut output = seed_residual_output(input_slice);
        self.inner
            .apply_base_lora(layer_idx, input_slice, &mut output);
        Ok(PyArray1::from_vec(py, output))
    }

    /// Engine statistics as a plain dict — mirrors every field of the
    /// Rust `CoordinatorStats` (`crates/sona/src/loops/coordinator.rs`):
    /// `trajectories_recorded`, `trajectories_buffered`,
    /// `trajectories_dropped`, `buffer_success_rate`, `patterns_stored`,
    /// `patterns_learned`, `ewc_tasks`, `instant_enabled`,
    /// `background_enabled`. Built by hand rather than via a general JSON
    /// converter (`hnsw.rs`'s `json_to_py` is private to that module and
    /// out of scope to touch here per this task's file list).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("trajectories_recorded", s.trajectories_recorded)?;
        dict.set_item("trajectories_buffered", s.trajectories_buffered)?;
        dict.set_item("trajectories_dropped", s.trajectories_dropped)?;
        dict.set_item("buffer_success_rate", s.buffer_success_rate)?;
        dict.set_item("patterns_stored", s.patterns_stored)?;
        dict.set_item("patterns_learned", s.patterns_learned)?;
        dict.set_item("ewc_tasks", s.ewc_tasks)?;
        dict.set_item("instant_enabled", s.instant_enabled)?;
        dict.set_item("background_enabled", s.background_enabled)?;
        Ok(dict)
    }

    /// Serialize engine state to a JSON string for persistence.
    ///
    /// Honest scope: this persists learned **patterns** (the
    /// ReasoningBank) plus the EWC task count and the instant/background
    /// enabled flags — see `LoopCoordinator::serialize_state`
    /// (`crates/sona/src/loops/coordinator.rs`). It does **not** persist
    /// LoRA weights (that's `export_lora_state`, out of scope here — see
    /// this module's doc comment). A fresh, untrained engine therefore
    /// round-trips an empty-patterns state.
    fn save_state(&self) -> String {
        self.inner.coordinator().serialize_state()
    }

    /// Restore patterns from a JSON string produced by `save_state`.
    /// Returns the number of patterns restored.
    ///
    /// Unlike the upstream NAPI binding (`crates/sona/src/napi.rs`'s
    /// `load_state`, which swallows the error via `eprintln!` and returns
    /// `0`), this binding propagates a malformed-JSON error as
    /// `RuVectorError` instead of silently reporting zero patterns
    /// restored — a Python caller should be able to tell "nothing to
    /// restore" apart from "your state string is corrupt".
    fn load_state(&self, state_json: &str) -> PyResult<usize> {
        self.inner
            .coordinator()
            .load_state(state_json)
            .map_err(RuVectorError::new_err)
    }

    /// Number of base-LoRA layers (fixed at engine-construction time).
    #[getter]
    fn num_layers(&self) -> usize {
        self.inner.coordinator().base_lora().read().num_layers()
    }

    /// The hidden dimension this engine was constructed with.
    #[getter]
    fn hidden_dim(&self) -> usize {
        self.inner.config().hidden_dim
    }

    /// Whether the engine is enabled (learning/application active).
    #[getter]
    fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }

    /// Enable or disable the engine.
    ///
    /// Named `set_is_enabled` (not `set_enabled`) deliberately: pyo3's
    /// `#[setter]` strips a literal `set_` prefix to infer the property
    /// name when none is given explicitly, so `set_enabled` would bind to
    /// a *different* property (`enabled`) than the `is_enabled` getter
    /// above — two independent, mismatched attributes instead of one
    /// read/write property. `set_is_enabled` strips to `is_enabled`,
    /// matching the getter.
    #[setter]
    fn set_is_enabled(&mut self, enabled: bool) {
        self.inner.set_enabled(enabled);
    }

    /// Diagnostic-friendly repr.
    fn __repr__(&self) -> String {
        format!(
            "SonaEngine(hidden_dim={}, num_layers={}, enabled={})",
            self.inner.config().hidden_dim,
            self.inner.coordinator().base_lora().read().num_layers(),
            self.inner.is_enabled(),
        )
    }
}

/// Convenience exporter — the module init in `lib.rs` calls this to add
/// the class. Mirrors `rabitq.rs`/`hnsw.rs`'s `register()` convention.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SonaEngine>()?;
    Ok(())
}
