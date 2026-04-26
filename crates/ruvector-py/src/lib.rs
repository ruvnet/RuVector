// pyo3 0.22's `create_exception!` and `#[pymethods]` macros emit cfg(feature = "gil-refs")
// gates and identity-`?` conversions on `PyResult<T>` returns. Both are
// known false-positives against current rustc/clippy and are tracked
// upstream — silence them at the crate root rather than littering the
// source with #[allow] attrs.
#![allow(unexpected_cfgs)]
#![allow(clippy::useless_conversion)]

//! ruvector — Python bindings (M1).
//!
//! Single PyO3 extension module exposed as `ruvector._native`. The maturin
//! `pyproject.toml` `module-name = "ruvector._native"` setting wires the
//! cdylib into the `ruvector` package at install time; the pure-Python
//! `python/ruvector/__init__.py` re-exports from `ruvector._native` so
//! end users only ever type `import ruvector`.
//!
//! M1 surface (per `docs/sdk/04-milestones.md`):
//!   - `RabitqIndex` class (RaBitQ+ with rerank)
//!   - `RuVectorError` exception
//!   - `__version__` string mirroring the Cargo crate version
//!
//! Subsequent milestones add `RuLake`, `Embedder`, and `A2aClient` as
//! additional `register()` calls in this same `_native` module — no new
//! extensions, no separate wheels.

use pyo3::prelude::*;

mod error;
mod rabitq;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Class + exception registrations.
    rabitq::register(m)?;
    m.add("RuVectorError", py.get_type_bound::<error::RuVectorError>())?;

    // Version mirrors the Cargo crate version. The pure-Python
    // `__init__.py` also re-exports it so `ruvector.__version__` works
    // without import-time gymnastics.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
