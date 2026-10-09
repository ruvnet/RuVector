//! k-means clustering Python binding — wraps `ruvector_cluster_rag`'s
//! `cluster::kmeans` (the real ML clustering algorithm in the workspace;
//! NOT `crates/ruvector-cluster`, which is distributed-sharding
//! coordination infra, a premise mismatch for "clustering" — see
//! ADR-352).
//!
//! `ruvector_cluster_rag::cluster::kmeans` validates its own invariants
//! with `assert!`/`assert_eq!`, which — unlike a `Result` — unwinds as a
//! Rust panic. PyO3 converts an unhandled panic into
//! `pyo3_runtime.PanicException`, a `BaseException` subclass that bypasses
//! `except Exception` (and therefore `except RuVectorError`) entirely —
//! unacceptable for a library binding, and a real crash risk in a
//! threaded host (e.g. an MCP server dispatching tool calls on a pool).
//! Every invariant `kmeans` (and the `initial_centroid_indices` helper it
//! calls internally) asserts on is therefore re-checked here first, with a
//! clear `ValueError`/`TypeError`, before the Rust call — same boundary
//! discipline as `rabitq.rs`/`hnsw.rs`.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use ruvector_cluster_rag::cluster::kmeans as rust_kmeans;

/// `(assignments, centroids, cohesion, cluster_sizes)` — see `kmeans`'s doc
/// comment for what each element is. Factored into a type alias (rather
/// than a 4-tuple return type inline) per clippy's `type_complexity` lint —
/// same convention as `hnsw.rs`'s `SearchHit`/`ExportedItem`.
type KMeansPyResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i64>>,
);

/// Run Lloyd's k-means over `vectors` (shape `(n, dim)`, `dtype=float32`).
///
/// Returns a 4-tuple:
///   - `assignments`: `int64[n]` — cluster id of each input row.
///   - `centroids`: `float32[k, dim]` — final cluster centroids.
///   - `cohesion`: `float32[k]` — mean cosine similarity of each cluster's
///     members to their centroid (∈ [-1, 1]; higher is tighter). This is
///     the crate's distinctive output (used upstream to prioritise tight,
///     relevant clusters at query time) and isn't cheaply recoverable in
///     Python without re-walking every vector, so it's exposed rather than
///     dropped.
///   - `cluster_sizes`: `int64[k]` — member count per cluster.
///
/// A 4-tuple (rather than a dedicated result class) was chosen because
/// every field is a plain NumPy array with no behaviour attached — a class
/// would only add getter boilerplate for the same four reads.
///
/// `iters` defaults to 20 (an arbitrary-but-reasonable Lloyd's-iteration
/// budget; the upstream Rust tests use 10-20).
///
/// Raises `ValueError` for `k == 0`, `k > n`, an empty `vectors` array, a
/// zero-width `dim`, or any non-finite (`NaN`/`inf`) coordinate — every one
/// of these is an `assert!` inside `ruvector_cluster_rag::cluster::kmeans`
/// (or the `initial_centroid_indices` seeding helper it calls), so this
/// function re-validates them all up front rather than let the Rust side
/// panic across the FFI boundary. Raises `TypeError` for a non-C-contiguous
/// input, matching `rabitq.rs`/`hnsw.rs`'s convention of refusing an
/// implicit O(n·dim) copy rather than performing one silently.
#[pyfunction]
#[pyo3(signature = (vectors, k, *, iters = 20))]
pub fn kmeans<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'_, f32>,
    k: usize,
    iters: usize,
) -> PyResult<KMeansPyResult<'py>> {
    if !vectors.is_c_contiguous() {
        return Err(PyTypeError::new_err(
            "vectors must be C-contiguous; pass np.ascontiguousarray(...) first",
        ));
    }
    let shape = vectors.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "vectors must be 2D, got {}D",
            shape.len()
        )));
    }
    let (n, dim) = (shape[0], shape[1]);
    if dim == 0 {
        return Err(PyValueError::new_err("dim must be > 0"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("vectors must contain at least 1 row"));
    }
    if k == 0 {
        return Err(PyValueError::new_err("k must be > 0"));
    }
    if k > n {
        return Err(PyValueError::new_err(format!(
            "k ({k}) must not exceed the number of vectors ({n})"
        )));
    }

    let slice = vectors.as_slice()?; // contiguous view, len = n*dim
    if !slice.iter().all(|x| x.is_finite()) {
        return Err(PyValueError::new_err(
            "vectors must contain only finite (non-NaN, non-infinite) coordinates",
        ));
    }

    // Materialise owned rows — the Rust API takes `&[Vec<f32>]`.
    let rows: Vec<Vec<f32>> = (0..n)
        .map(|i| slice[i * dim..(i + 1) * dim].to_vec())
        .collect();

    // Heavy work: drop the GIL, same calculus as `RabitqIndex::build`.
    let result = py.detach(|| rust_kmeans(&rows, k, iters));

    let assignments: Vec<i64> = result.assignments.iter().map(|&a| a as i64).collect();
    let cluster_sizes: Vec<i64> = result.cluster_sizes.iter().map(|&s| s as i64).collect();

    let assignments_arr = PyArray1::from_vec(py, assignments);
    let centroids_arr = PyArray2::from_vec2(py, &result.centroids)
        .map_err(|e| PyValueError::new_err(format!("failed to build centroids array: {e}")))?;
    let cohesion_arr = PyArray1::from_vec(py, result.cohesion);
    let cluster_sizes_arr = PyArray1::from_vec(py, cluster_sizes);

    Ok((
        assignments_arr,
        centroids_arr,
        cohesion_arr,
        cluster_sizes_arr,
    ))
}

/// Convenience exporter — the module init in `lib.rs` calls this to add
/// the function. Mirrors `rabitq.rs::register`/`hnsw.rs::register`'s
/// convention for `#[pyclass]`es, adapted for a free `#[pyfunction]`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    Ok(())
}
