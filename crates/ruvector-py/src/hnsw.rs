//! `HnswIndex` Python class — wraps `ruvector_core::vector_db::VectorDB`.
//!
//! Added in ADR-352's M2 slice ("make the default Collection the fast
//! path"). Per the capability inventory in that ADR: `ruvector-core`'s
//! `VectorIndex` trait (and therefore `VectorDB`, `Box<dyn VectorIndex>`,
//! and the patched `hnsw_rs::Hnsw` it wraps) is `Send + Sync` all the way
//! down — same check this crate already does in `rabitq.rs` before
//! dropping `unsendable`, so this class isn't `unsendable` either.
//!
//! Two things this backend gets "for free" that `RabitqIndex` doesn't:
//! metadata is stored natively in Rust (`VectorEntry::metadata`), and
//! `VectorDB::search`'s `SearchQuery::filter` applies an equality filter
//! *inside* the Rust call, not in a Python-side post-filter loop — this is
//! the literal fix for "metadata filtering done in Rust, not Python".
//! It is still a bolt-on post-ANN-search filter (retain-after-search, not
//! pushed into the HNSW graph traversal) — `ruvector-core`'s own ADR-352
//! inventory pass found no native in-traversal filter wired up today
//! either (the lower-level `hnsw_rs::Hnsw::search_filter` exists but
//! `HnswIndex`/`VectorDB` don't surface it) — so this is an upgrade from
//! "Python filters," not a claim of best-possible filtered-ANN recall.
//!
//! Always constructed with a `memory://` storage path (see `create()`):
//! persistence goes through `export_items()` + Python's existing
//! save/load sidecar pattern (same idiom `RabitqIndex` uses), not
//! `VectorDB`'s own redb-backed persistent mode — keeps one persistence
//! story across both backends instead of two.

use std::collections::HashMap;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use ruvector_core::types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry};
use ruvector_core::vector_db::VectorDB;

use crate::error::to_pyerr_core;

/// `(id, score, metadata)` — one `HnswIndex.search` hit.
type SearchHit<'py> = (String, f32, Option<Bound<'py, PyDict>>);
/// `(id, vector, metadata)` — one `HnswIndex.export_items` row.
type ExportedItem<'py> = (String, Bound<'py, numpy::PyArray1<f32>>, Option<Bound<'py, PyDict>>);

fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "dot" | "dot_product" => Ok(DistanceMetric::DotProduct),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        other => Err(PyValueError::new_err(format!(
            "unknown metric {other:?}; expected one of: cosine, euclidean, dot, manhattan"
        ))),
    }
}

/// Convert a Python value into `serde_json::Value`. Hand-rolled rather than
/// pulling in the `pythonize` crate: metadata dicts in practice are a flat
/// mix of str/int/float/bool/None/list/dict, and a small explicit converter
/// means no surprise behavior from a general-purpose serializer (e.g. how
/// it'd handle a numpy scalar, which we deliberately reject here instead of
/// silently coercing).
fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = value.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(serde_json::json!(f));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let items: PyResult<Vec<serde_json::Value>> = list.iter().map(|v| py_to_json(&v)).collect();
        return Ok(serde_json::Value::Array(items?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return Ok(serde_json::Value::Object(py_dict_to_json_map(dict)?.into_iter().collect()));
    }
    Err(PyTypeError::new_err(format!(
        "metadata values must be str/int/float/bool/None/list/dict, got {}",
        value.get_type().name()?
    )))
}

fn py_dict_to_json_map(
    dict: &Bound<'_, PyDict>,
) -> PyResult<HashMap<String, serde_json::Value>> {
    let mut map = HashMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract().map_err(|_| {
            PyTypeError::new_err("metadata keys must be strings")
        })?;
        map.insert(key, py_to_json(&v)?);
    }
    Ok(map)
}

/// Convert `serde_json::Value` back into a Python object.
fn json_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    match value {
        serde_json::Value::Null => Ok(py.None().into_bound(py)),
        serde_json::Value::Bool(b) => b.into_bound_py_any(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_bound_py_any(py)
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_bound_py_any(py)
            }
        }
        serde_json::Value::String(s) => s.into_bound_py_any(py),
        serde_json::Value::Array(items) => {
            let converted: PyResult<Vec<_>> = items.iter().map(|v| json_to_py(py, v)).collect();
            PyList::new(py, converted?)?.into_bound_py_any(py)
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.into_bound_py_any(py)
        }
    }
}

fn json_map_to_py<'py>(
    py: Python<'py>,
    map: HashMap<String, serde_json::Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, json_to_py(py, &v)?)?;
    }
    Ok(dict)
}

/// HNSW-backed generic vector index. Metadata-aware, filter-in-Rust,
/// true (non-tombstone-in-Python) delete — the default backend for
/// `ruvector.Collection` as of ADR-352's M2 slice.
#[pyclass(name = "HnswIndex", module = "ruvector._native")]
pub struct HnswIndex {
    inner: VectorDB,
}

#[pymethods]
impl HnswIndex {
    /// Create an empty index. `metric` is one of `cosine` (default),
    /// `euclidean`/`l2`, `dot`/`dot_product`, `manhattan`/`l1`.
    #[staticmethod]
    #[pyo3(signature = (dim, *, metric = "cosine", m = 16, ef_construction = 200, ef_search = 50))]
    fn create(dim: usize, metric: &str, m: usize, ef_construction: usize, ef_search: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        let metric_enum = parse_metric(metric)?;
        let opts = DbOptions {
            dimensions: dim,
            distance_metric: metric_enum,
            // Always in-memory — see module docstring. `memory://` is the
            // sentinel VectorDB::new checks for before touching disk.
            storage_path: "memory://ruvector-py".to_string(),
            hnsw_config: Some(HnswConfig {
                m,
                ef_construction,
                ef_search,
                ..HnswConfig::default()
            }),
            quantization: None,
        };
        let inner = VectorDB::new(opts).map_err(to_pyerr_core)?;
        Ok(Self { inner })
    }

    /// Insert one vector under `id` (any string). Returns the id actually
    /// stored (VectorDB would auto-generate one if `id` were `None`, but
    /// this binding always supplies an explicit id so the return value is
    /// just `id` echoed back — kept as a return value for symmetry with
    /// the underlying Rust API rather than silently dropped).
    #[pyo3(signature = (id, vector, metadata = None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: PyReadonlyArray1<'_, f32>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        if !vector.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "vector must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        let v = vector.as_slice()?.to_vec();
        let meta = metadata.map(py_dict_to_json_map).transpose()?;
        let entry = VectorEntry {
            id: Some(id.to_string()),
            vector: v,
            metadata: meta,
        };
        py.detach(|| self.inner.insert(entry)).map_err(to_pyerr_core)
    }

    /// Insert many vectors at once. Releases the GIL around the loop
    /// (same calculus as `RabitqIndex.add_batch` — see
    /// `docs/sdk/02-strategy.md` § "GIL story").
    #[pyo3(signature = (ids, vectors, metadatas = None))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: PyReadonlyArray2<'_, f32>,
        metadatas: Option<Vec<Option<Bound<'_, PyDict>>>>,
    ) -> PyResult<Vec<String>> {
        if !vectors.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "vectors must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        let shape = vectors.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!("vectors must be 2D, got {}D", shape.len())));
        }
        let (n, dim) = (shape[0], shape[1]);
        if ids.len() != n {
            return Err(PyValueError::new_err(format!(
                "ids length ({}) must match vectors row count ({})",
                ids.len(),
                n
            )));
        }
        if let Some(m) = &metadatas {
            if m.len() != n {
                return Err(PyValueError::new_err(format!(
                    "metadatas length ({}) must match vectors row count ({})",
                    m.len(),
                    n
                )));
            }
        }
        let slice = vectors.as_slice()?;
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let row = slice[i * dim..(i + 1) * dim].to_vec();
            let meta = metadatas
                .as_ref()
                .and_then(|m| m[i].as_ref())
                .map(py_dict_to_json_map)
                .transpose()?;
            entries.push(VectorEntry {
                id: Some(ids[i].clone()),
                vector: row,
                metadata: meta,
            });
        }
        py.detach(|| self.inner.insert_batch(entries)).map_err(to_pyerr_core)
    }

    /// Search for the `k` nearest neighbours of `query`, optionally
    /// restricted by an exact-match `filter` dict (applied in Rust —
    /// see this module's docstring for the "bolt-on, not in-traversal"
    /// caveat). Returns `(id, score, metadata)` tuples.
    ///
    /// No per-call `ef_search` override: `SearchQuery.ef_search` exists on
    /// the Rust struct but `VectorDB::search` never reads it — it calls
    /// the generic `VectorIndex::search(&self, query, k)` trait method,
    /// which has no ef parameter at all; `HnswIndex`'s impl always uses
    /// `self.config.ef_search`, fixed at construction (`HnswIndex::create`'s
    /// `ef_search` kwarg). Found this by reading `vector_db.rs::search`
    /// after a benchmark result looked suspicious — exposing a per-call
    /// kwarg that silently does nothing would be worse than not having
    /// one, so it isn't here. Tune `ef_search` by constructing with the
    /// value you want.
    #[pyo3(signature = (query, k, *, filter = None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'_, f32>,
        k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<SearchHit<'py>>> {
        if !query.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "query must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        if k == 0 {
            return Err(PyValueError::new_err("k must be > 0"));
        }
        let q = query.as_slice()?.to_vec();
        let filt = filter.map(py_dict_to_json_map).transpose()?;
        let sq = SearchQuery {
            vector: q,
            k,
            filter: filt,
            ef_search: None,
        };
        let results = py.detach(|| self.inner.search(sq)).map_err(to_pyerr_core)?;
        results
            .into_iter()
            .map(|r| -> PyResult<SearchHit<'py>> {
                let meta = r.metadata.map(|m| json_map_to_py(py, m)).transpose()?;
                Ok((r.id, r.score, meta))
            })
            .collect()
    }

    /// Delete by id. Returns whether an entry with that id existed. Real
    /// delete at the storage/metadata layer (not a Python tombstone) —
    /// `ruvector-core`'s own ADR-352 inventory notes the underlying HNSW
    /// *graph node* is not physically removed (`hnsw_rs` has no live-delete),
    /// so memory isn't reclaimed until a rebuild, but the id is gone from
    /// every result/count/get immediately.
    fn delete(&self, id: &str) -> PyResult<bool> {
        self.inner.delete(id).map_err(to_pyerr_core)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner.len().map_err(to_pyerr_core)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.options().dimensions
    }

    /// Export every `(id, vector, metadata)` triple currently held, for
    /// `Collection.save()`'s rebuild-into-persistent-form path (this
    /// backend is always in-memory — see the module docstring).
    fn export_items<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<ExportedItem<'py>>> {
        let ids = self.inner.keys().map_err(to_pyerr_core)?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(entry) = self.inner.get(&id).map_err(to_pyerr_core)? {
                let meta = entry.metadata.map(|m| json_map_to_py(py, m)).transpose()?;
                out.push((id, numpy::PyArray1::from_vec(py, entry.vector), meta));
            }
        }
        Ok(out)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "HnswIndex(n={}, dim={})",
            self.inner.len().unwrap_or(0),
            self.inner.options().dimensions
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HnswIndex>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_metric`'s Ok branches are pure string-matching — no PyErr is
    /// ever constructed, so (unlike the Err branch below) these don't need
    /// an attached interpreter at all. Kept as a separate, GIL-free test so
    /// it's unmistakable which half of this function needs what.
    #[test]
    fn parse_metric_accepts_every_documented_alias() {
        assert!(matches!(parse_metric("cosine"), Ok(DistanceMetric::Cosine)));
        assert!(matches!(parse_metric("euclidean"), Ok(DistanceMetric::Euclidean)));
        assert!(matches!(parse_metric("l2"), Ok(DistanceMetric::Euclidean)));
        assert!(matches!(parse_metric("dot"), Ok(DistanceMetric::DotProduct)));
        assert!(matches!(parse_metric("dot_product"), Ok(DistanceMetric::DotProduct)));
        assert!(matches!(parse_metric("manhattan"), Ok(DistanceMetric::Manhattan)));
        assert!(matches!(parse_metric("l1"), Ok(DistanceMetric::Manhattan)));
    }

    /// The error path constructs a `PyValueError` (via `PyValueError::new_err`),
    /// whose lazy exception closure references `PyExc_ValueError` — that
    /// needs a real, attached interpreter the same way `error.rs`'s mappers
    /// do (see the long comment on `crates/ruvector-py/Cargo.toml`'s pyo3
    /// dependency for why this links at all under `cargo test`).
    #[test]
    fn parse_metric_rejects_unknown_string() {
        Python::initialize();
        let result = parse_metric("hamming");
        Python::attach(|py| match result {
            Err(e) => {
                assert!(e.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                let msg = e.value(py).to_string();
                assert!(msg.contains("hamming"), "message was: {msg}");
            }
            Ok(_) => panic!("expected Err for an unrecognised metric string"),
        });
    }

    #[test]
    fn parse_metric_rejects_empty_string() {
        Python::initialize();
        assert!(parse_metric("").is_err());
    }

    /// Round-trip a `serde_json::Value` through `json_to_py` then back
    /// through `py_to_json`, asserting the result is byte-identical to the
    /// input. Exercises nested object/array structure, every scalar kind
    /// these converters claim to support, unicode text, and negative
    /// numbers — none of which the Python-side pytest suite checks in
    /// isolation (it only ever sees these functions through a full
    /// `insert`/`search` round trip with whatever metadata that test
    /// happens to pass).
    fn round_trip(value: serde_json::Value) {
        Python::initialize();
        Python::attach(|py| {
            let py_obj = json_to_py(py, &value).expect("json_to_py failed");
            let back = py_to_json(&py_obj).expect("py_to_json failed");
            assert_eq!(back, value, "round trip changed the value");
        });
    }

    #[test]
    fn round_trip_null() {
        round_trip(serde_json::Value::Null);
    }

    #[test]
    fn round_trip_bool_true_and_false() {
        round_trip(serde_json::json!(true));
        round_trip(serde_json::json!(false));
    }

    #[test]
    fn round_trip_negative_integer() {
        round_trip(serde_json::json!(-42));
    }

    #[test]
    fn round_trip_float() {
        round_trip(serde_json::json!(-3.5));
    }

    #[test]
    fn round_trip_unicode_string() {
        round_trip(serde_json::json!("héllo wörld — 日本語 🦀"));
    }

    #[test]
    fn round_trip_empty_string() {
        round_trip(serde_json::json!(""));
    }

    #[test]
    fn round_trip_empty_list_and_dict() {
        round_trip(serde_json::json!([]));
        round_trip(serde_json::json!({}));
    }

    #[test]
    fn round_trip_flat_list_of_mixed_scalars() {
        round_trip(serde_json::json!([1, "two", 3.0, true, serde_json::Value::Null]));
    }

    #[test]
    fn round_trip_deeply_nested_structure() {
        round_trip(serde_json::json!({
            "a": [1, 2, {"b": [3, 4, {"c": "deep"}]}],
            "d": {"e": {"f": {"g": [true, false, null]}}},
            "h": -7.25,
        }));
    }

    /// `py_dict_to_json_map` / `json_map_to_py` are the `HashMap`-flavoured
    /// siblings `insert`/`search`/`export_items` actually call (metadata is
    /// always a dict at the Python boundary, never an arbitrary top-level
    /// JSON value) — round-trip those directly too, not just the more
    /// general `Value` converters above.
    #[test]
    fn metadata_map_round_trip() {
        Python::initialize();
        Python::attach(|py| {
            let mut map = HashMap::new();
            map.insert("name".to_string(), serde_json::json!("widget"));
            map.insert("count".to_string(), serde_json::json!(3));
            map.insert("tags".to_string(), serde_json::json!(["a", "b"]));
            map.insert("nested".to_string(), serde_json::json!({"active": true}));

            let py_dict = json_map_to_py(py, map.clone()).expect("json_map_to_py failed");
            let back = py_dict_to_json_map(&py_dict).expect("py_dict_to_json_map failed");
            assert_eq!(back, map, "metadata map round trip changed the value");
        });
    }

    /// `py_dict_to_json_map` rejects non-string keys with a clear
    /// `TypeError` rather than silently stringifying them.
    #[test]
    fn metadata_map_rejects_non_string_keys() {
        Python::initialize();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item(1, "value").unwrap();
            let err = py_dict_to_json_map(&dict).expect_err("expected TypeError for int key");
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    /// `py_to_json` rejects a Python type it doesn't know how to represent
    /// (e.g. an arbitrary object) instead of silently coercing it.
    #[test]
    fn py_to_json_rejects_unsupported_type() {
        Python::initialize();
        Python::attach(|py| {
            // `set` is not one of the str/int/float/bool/None/list/dict
            // cases `py_to_json` matches on.
            let set = py.eval(c"set([1, 2, 3])", None, None).unwrap();
            let err = py_to_json(&set).expect_err("expected TypeError for a set");
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    /// `bool` must be checked before `int`/`float` in `py_to_json` — in
    /// Python, `bool` is a subclass of `int`, so `True.extract::<i64>()`
    /// would otherwise succeed and silently turn a bool into `1`/`0`.
    #[test]
    fn py_to_json_keeps_bool_distinct_from_int() {
        Python::initialize();
        Python::attach(|py| {
            let value = json_to_py(py, &serde_json::json!(true)).unwrap();
            let back = py_to_json(&value).unwrap();
            assert_eq!(back, serde_json::json!(true));
            assert_ne!(back, serde_json::json!(1));
        });
    }
}

/// Regression tests pinning down surprising-but-confirmed current behaviour
/// found while writing the round-trip tests above. Not assertions of
/// *desired* behaviour — see this session's summary for the bug report;
/// fixing these is explicitly left to the orchestrating session.
#[cfg(test)]
mod known_limitations {
    use super::*;

    /// A Python `int` that doesn't fit in `i64` (metadata value, e.g. a
    /// user-supplied timestamp-as-nanos or a hash) silently loses precision:
    /// `py_to_json` falls through `extract::<i64>()` to `extract::<f64>()`,
    /// so `2**64` (`18446744073709551616`) comes back as the double-rounded
    /// `18446744073709551616.0` -> `1.8446744073709552e19`, with no error
    /// raised anywhere. Confirmed bug — reported in this session's summary,
    /// not fixed here per the task's "report, don't silently patch" rule.
    #[test]
    fn py_to_json_large_python_int_loses_precision_silently() {
        Python::initialize();
        Python::attach(|py| {
            let big = py.eval(c"2**64", None, None).unwrap();
            assert_eq!(big.repr().unwrap().to_string(), "18446744073709551616");
            let back = py_to_json(&big).expect("does not error; silently lossy instead");
            // Documents the data loss: the round-tripped value is not an
            // exact integer any more (it prints with scientific notation /
            // a trailing `.0`-equivalent), unlike the exact input.
            assert!(matches!(back, serde_json::Value::Number(_)));
            assert_ne!(back.to_string(), "18446744073709551616");
        });
    }

    /// A Python `str` containing a lone UTF-16 surrogate (constructible via
    /// `surrogateescape`, e.g. from a malformed-filename-derived string)
    /// cannot `extract::<String>()` (it isn't valid UTF-8/UTF-32 text), so
    /// `py_to_json` falls through every branch and hits the final
    /// `TypeError` whose message claims `"got str"` — true as far as it
    /// goes, but potentially confusing (the type IS str, the problem is its
    /// *content*, not its type) when someone's metadata value fails this
    /// way in production.
    #[test]
    fn py_to_json_lone_surrogate_str_hits_generic_type_error() {
        Python::initialize();
        Python::attach(|py| {
            let bad = py
                .eval(
                    c"'abc'.encode('utf-8', 'surrogateescape') + bytes([0xed, 0xa0, 0x80])",
                    None,
                    None,
                )
                .unwrap();
            // bytes, not str — decode with surrogateescape to get an
            // actual lone-surrogate `str` object the way real code would
            // (e.g. decoding a malformed OS path).
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("b", bad).unwrap();
            let lone_surrogate_str = py
                .eval(c"b.decode('utf-8', 'surrogateescape')", None, Some(&locals))
                .unwrap();
            let err = py_to_json(&lone_surrogate_str).expect_err("lone surrogates aren't valid text");
            assert!(err.is_instance_of::<PyTypeError>(py));
            let msg = err.value(py).to_string();
            assert!(msg.contains("got str"), "message was: {msg}");
        });
    }

    /// `float('nan')` passed as metadata is also silently accepted and
    /// converted to JSON `null` with no error (this one traces back to
    /// `serde_json`'s own `Serialize` impl for `f64`, which maps every
    /// non-finite float to `null` by design — not a bug unique to this
    /// crate, but worth pinning down since `py_to_json`'s doc comment
    /// otherwise promises "no surprise behavior").
    #[test]
    fn py_to_json_nan_float_becomes_null_silently() {
        Python::initialize();
        Python::attach(|py| {
            let nan = py.eval(c"float('nan')", None, None).unwrap();
            let result = py_to_json(&nan).expect("does not error; becomes null instead");
            assert_eq!(result, serde_json::Value::Null);
        });
    }
}
