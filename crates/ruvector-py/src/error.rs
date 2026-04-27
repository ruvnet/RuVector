//! Python exception hierarchy for `ruvector`.
//!
//! M1 ships a single user-visible exception, `RuVectorError`, plus the
//! `to_pyerr` mapper that converts every `RabitqError` variant into it.
//! Subclasses (`DimensionMismatch`, `EmptyIndex`, `PersistError`, …) are
//! reserved for M2/M3/M4 expansions — see `docs/sdk/03-api-surface.md`
//! § "Error hierarchy". For now the message string is the wire format.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// `create_exception!` injects a unit-struct `RuVectorError` and a
// `RuVectorError::type_object_bound(py)` method we use in `lib.rs` when
// adding the symbol to the module.
pyo3::create_exception!(
    ruvector._native,
    RuVectorError,
    PyException,
    "Base class for every error raised by the ruvector extension."
);

/// Map a `ruvector_rabitq::RabitqError` into a `PyErr` carrying
/// `RuVectorError`. The Display impl on `RabitqError` is already
/// human-readable so we forward it verbatim — no double-formatting.
pub fn to_pyerr(err: ruvector_rabitq::RabitqError) -> PyErr {
    RuVectorError::new_err(err.to_string())
}
