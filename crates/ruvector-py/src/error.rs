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

/// Same mapping for `ruvector_core::error::RuvectorError` (the HNSW/VectorDB
/// backend added for the generic `HnswIndex` pyclass, ADR-352 M2 slice).
/// Two source error types, one Python exception — `RuVectorError` stays the
/// single base class for every backend per `docs/sdk/03-api-surface.md`.
pub fn to_pyerr_core(err: ruvector_core::error::RuvectorError) -> PyErr {
    RuVectorError::new_err(err.to_string())
}

/// Same mapping for `ruvector_gnn::error::GnnError` (added for the GNN
/// forward-pass rerank binding, `gnn.rs`'s `GnnLayer`).
pub fn to_pyerr_gnn(err: ruvector_gnn::error::GnnError) -> PyErr {
    RuVectorError::new_err(err.to_string())
}

/// Same mapping for `ruvector_attention::error::AttentionError` (added for
/// the attention-rerank binding, `gnn.rs`'s `AttentionReranker`).
pub fn to_pyerr_attention(err: ruvector_attention::error::AttentionError) -> PyErr {
    RuVectorError::new_err(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both mapper fns need an attached, initialized interpreter the moment
    /// a `PyErr` is actually inspected (formatting a `PyErr` — even via
    /// `Display`/`to_string()` — calls `Python::attach` internally in pyo3
    /// 0.29; see `pyo3::err::PyErr`'s `Display` impl). `Python::initialize`
    /// is idempotent/`Once`-guarded, so calling it per-test is safe and
    /// matches the one-off nature of each `#[test]` fn (no shared fixture
    /// needed).
    fn attach_py() {
        Python::initialize();
    }

    #[test]
    fn to_pyerr_forwards_rabitq_dimension_mismatch_verbatim() {
        attach_py();
        let src = ruvector_rabitq::RabitqError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        let expected_msg = src.to_string();
        let err = to_pyerr(src);
        Python::attach(|py| {
            assert!(err.is_instance_of::<RuVectorError>(py));
            assert_eq!(err.value(py).to_string(), expected_msg);
            assert_eq!(expected_msg, "dimension mismatch: expected 128, got 64");
        });
    }

    #[test]
    fn to_pyerr_forwards_rabitq_empty_index_verbatim() {
        attach_py();
        let src = ruvector_rabitq::RabitqError::EmptyIndex;
        let expected_msg = src.to_string();
        let err = to_pyerr(src);
        Python::attach(|py| {
            assert!(err.is_instance_of::<RuVectorError>(py));
            assert_eq!(err.value(py).to_string(), expected_msg);
        });
    }

    #[test]
    fn to_pyerr_forwards_rabitq_invalid_parameter_verbatim() {
        attach_py();
        let src =
            ruvector_rabitq::RabitqError::InvalidParameter("rerank_factor must be > 0".to_string());
        let expected_msg = src.to_string();
        let err = to_pyerr(src);
        Python::attach(|py| {
            assert_eq!(err.value(py).to_string(), expected_msg);
            assert_eq!(expected_msg, "invalid parameter: rerank_factor must be > 0");
        });
    }

    #[test]
    fn to_pyerr_core_forwards_dimension_mismatch_verbatim() {
        attach_py();
        let src = ruvector_core::error::RuvectorError::DimensionMismatch {
            expected: 32,
            actual: 16,
        };
        let expected_msg = src.to_string();
        let err = to_pyerr_core(src);
        Python::attach(|py| {
            assert!(err.is_instance_of::<RuVectorError>(py));
            assert_eq!(err.value(py).to_string(), expected_msg);
            assert_eq!(expected_msg, "Dimension mismatch: expected 32, got 16");
        });
    }

    #[test]
    fn to_pyerr_core_forwards_vector_not_found_verbatim() {
        attach_py();
        let src = ruvector_core::error::RuvectorError::VectorNotFound("abc-123".to_string());
        let expected_msg = src.to_string();
        let err = to_pyerr_core(src);
        Python::attach(|py| {
            assert_eq!(err.value(py).to_string(), expected_msg);
            assert_eq!(expected_msg, "Vector not found: abc-123");
        });
    }

    #[test]
    fn to_pyerr_core_forwards_invalid_input_verbatim() {
        attach_py();
        let src = ruvector_core::error::RuvectorError::InvalidInput("k must be > 0".to_string());
        let expected_msg = src.to_string();
        let err = to_pyerr_core(src);
        Python::attach(|py| {
            assert_eq!(err.value(py).to_string(), expected_msg);
        });
    }

    #[test]
    fn to_pyerr_gnn_forwards_layer_config_verbatim() {
        attach_py();
        let src = ruvector_gnn::error::GnnError::layer_config("dropout must be in [0, 1]");
        let expected_msg = src.to_string();
        let err = to_pyerr_gnn(src);
        Python::attach(|py| {
            assert!(err.is_instance_of::<RuVectorError>(py));
            assert_eq!(err.value(py).to_string(), expected_msg);
        });
    }

    #[test]
    fn to_pyerr_attention_forwards_dimension_mismatch_verbatim() {
        attach_py();
        let src = ruvector_attention::error::AttentionError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        let expected_msg = src.to_string();
        let err = to_pyerr_attention(src);
        Python::attach(|py| {
            assert!(err.is_instance_of::<RuVectorError>(py));
            assert_eq!(err.value(py).to_string(), expected_msg);
        });
    }

    /// Both error families must land on the *same* Python exception class —
    /// the whole point of having a single `RuVectorError` base per
    /// `docs/sdk/03-api-surface.md` § "Error hierarchy", rather than one
    /// exception type per Rust backend.
    #[test]
    fn both_mappers_raise_the_same_exception_type() {
        attach_py();
        let a = to_pyerr(ruvector_rabitq::RabitqError::EmptyIndex);
        let b = to_pyerr_core(ruvector_core::error::RuvectorError::InvalidInput(
            "x".to_string(),
        ));
        Python::attach(|py| {
            assert_eq!(
                a.get_type(py).qualname().unwrap().to_string(),
                b.get_type(py).qualname().unwrap().to_string()
            );
        });
    }
}
