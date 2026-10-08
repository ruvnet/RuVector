use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use ruvector_core::types::DbOptions;
use ruvector_core::{RuvectorError, SearchQuery, VectorDB, VectorEntry};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Mutex;

create_exception!(_native, RuVectorError, PyException);
create_exception!(_native, InvalidVectorError, RuVectorError);
create_exception!(_native, DimensionError, InvalidVectorError);
create_exception!(_native, DuplicateIDError, RuVectorError);
create_exception!(_native, StorageError, RuVectorError);
create_exception!(_native, IndexError, RuVectorError);
create_exception!(_native, ClosedError, RuVectorError);

fn error(e: RuvectorError) -> PyErr {
    let message = e.to_string();
    match e {
        RuvectorError::DimensionMismatch { .. } => DimensionError::new_err(message),
        RuvectorError::InvalidInput(_)
        | RuvectorError::InvalidParameter(_)
        | RuvectorError::InvalidDimension(_) => InvalidVectorError::new_err(message),
        RuvectorError::StorageError(_)
        | RuvectorError::DatabaseError(_)
        | RuvectorError::IoError(_)
        | RuvectorError::InvalidPath(_) => StorageError::new_err(message),
        RuvectorError::IndexError(_) => IndexError::new_err(message),
        _ => RuVectorError::new_err(message),
    }
}

fn decode<T: serde::de::DeserializeOwned>(value: Value) -> PyResult<T> {
    serde_json::from_value(value).map_err(|e| InvalidVectorError::new_err(e.to_string()))
}

fn validate(db: &VectorDB, vector: &[f32]) -> PyResult<()> {
    if vector.len() != db.options().dimensions {
        return Err(DimensionError::new_err(format!(
            "expected {} dimensions, got {}",
            db.options().dimensions,
            vector.len()
        )));
    }
    if !vector.iter().all(|x| x.is_finite()) {
        return Err(InvalidVectorError::new_err(
            "vectors must contain finite float32 values",
        ));
    }
    Ok(())
}

fn preflight(db: &VectorDB, entries: &[VectorEntry]) -> PyResult<()> {
    let mut ids = HashSet::new();
    for entry in entries {
        validate(db, &entry.vector)?;
        if let Some(id) = &entry.id {
            if id.is_empty() {
                return Err(InvalidVectorError::new_err("id must not be empty"));
            }
            if !ids.insert(id) || db.get(id).map_err(error)?.is_some() {
                return Err(DuplicateIDError::new_err(format!(
                    "id already exists: {id}"
                )));
            }
        }
    }
    Ok(())
}

#[pyclass]
struct NativeDB {
    // Serializes operations including close. No lock is held while Python has the GIL.
    inner: Mutex<Option<VectorDB>>,
}

#[pymethods]
impl NativeDB {
    #[new]
    fn new(py: Python<'_>, config: String) -> PyResult<Self> {
        let options: DbOptions = serde_json::from_str(&config)
            .map_err(|e| InvalidVectorError::new_err(e.to_string()))?;
        let db = py.allow_threads(move || VectorDB::new(options).map_err(error))?;
        if db.options().distance_metric == ruvector_core::DistanceMetric::DotProduct {
            return Err(InvalidVectorError::new_err(
                "dot-product HNSW ranking is not supported at the pinned upstream commit",
            ));
        }
        Ok(Self {
            inner: Mutex::new(Some(db)),
        })
    }

    fn execute(&self, py: Python<'_>, operation: String, args: String) -> PyResult<String> {
        let args: Value =
            serde_json::from_str(&args).map_err(|e| InvalidVectorError::new_err(e.to_string()))?;
        py.allow_threads(|| {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| RuVectorError::new_err("database lock poisoned"))?;
            if operation == "close" {
                guard.take();
                return Ok("null".to_owned());
            }
            let db = guard
                .as_ref()
                .ok_or_else(|| ClosedError::new_err("database is closed"))?;
            let result = match operation.as_str() {
                "options" => json!(db.options()),
                "len" => json!(db.len().map_err(error)?),
                "keys" => json!(db.keys().map_err(error)?),
                "get" => {
                    let id: String = decode(args)?;
                    json!(db.get(&id).map_err(error)?)
                }
                "insert" => {
                    let entry: VectorEntry = decode(args)?;
                    preflight(db, std::slice::from_ref(&entry))?;
                    json!(db.insert(entry).map_err(error)?)
                }
                "insert_batch" => {
                    let entries: Vec<VectorEntry> = decode(args)?;
                    preflight(db, &entries)?;
                    json!(db.insert_batch(entries).map_err(error)?)
                }
                "search" => {
                    let query: SearchQuery = decode(args)?;
                    validate(db, &query.vector)?;
                    if query.k == 0 {
                        return Err(InvalidVectorError::new_err("k must be positive"));
                    }
                    json!(db.search(query).map_err(error)?)
                }
                "search_batch" => {
                    let queries: Vec<SearchQuery> = decode(args)?;
                    for query in &queries {
                        validate(db, &query.vector)?;
                        if query.k == 0 {
                            return Err(InvalidVectorError::new_err("k must be positive"));
                        }
                    }
                    let results = queries
                        .into_iter()
                        .map(|q| db.search(q).map_err(error))
                        .collect::<PyResult<Vec<_>>>()?;
                    json!(results)
                }
                "delete_batch" => {
                    let ids: Vec<String> = decode(args)?;
                    let results = ids
                        .iter()
                        .map(|id| db.delete(id).map_err(error))
                        .collect::<PyResult<Vec<_>>>()?;
                    json!(results)
                }
                _ => return Err(RuVectorError::new_err("unknown native operation")),
            };
            serde_json::to_string(&result).map_err(|e| RuVectorError::new_err(e.to_string()))
        })
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeDB>()?;
    m.add(
        "UPSTREAM_COMMIT",
        "5a93328f2fceb0307c25929ed38cd7a0911fdf00",
    )?;
    m.add("UPSTREAM_VERSION", "2.3.1")?;
    m.add("RuVectorError", m.py().get_type::<RuVectorError>())?;
    m.add(
        "InvalidVectorError",
        m.py().get_type::<InvalidVectorError>(),
    )?;
    m.add("DimensionError", m.py().get_type::<DimensionError>())?;
    m.add("DuplicateIDError", m.py().get_type::<DuplicateIDError>())?;
    m.add("StorageError", m.py().get_type::<StorageError>())?;
    m.add("IndexError", m.py().get_type::<IndexError>())?;
    m.add("ClosedError", m.py().get_type::<ClosedError>())?;
    Ok(())
}
