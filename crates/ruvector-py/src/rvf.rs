//! Bounded Python surface for trusted local RVF files. No signature verification claims.
use pyo3::exceptions::{PyFileExistsError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore as Store};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Mutex,
};

#[pyclass(name = "RvfStore", module = "ruvector._native")]
pub struct RvfStore {
    inner: Mutex<Option<Store>>,
}

fn failure(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
fn check_vector(vector: &[f32], dimension: usize) -> PyResult<()> {
    if vector.len() != dimension || vector.iter().any(|x| !x.is_finite() || x.abs() > 1.0e10) {
        return Err(PyValueError::new_err(
            "vector must match dimension and contain finite float32 values with magnitude <= 1e10",
        ));
    }
    Ok(())
}
impl RvfStore {
    fn access<T>(&self, action: impl FnOnce(&mut Store) -> PyResult<T>) -> PyResult<T> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| failure("store lock poisoned"))?;
        action(guard.as_mut().ok_or_else(|| failure("store is closed"))?)
    }
}
#[pymethods]
impl RvfStore {
    #[staticmethod]
    fn create(path: &str, dimension: u16) -> PyResult<Self> {
        if dimension == 0 || dimension > 4096 {
            return Err(PyValueError::new_err(
                "dimension must be between 1 and 4096",
            ));
        }
        if Path::new(path).symlink_metadata().is_ok() {
            return Err(PyFileExistsError::new_err(
                "create refuses to overwrite an existing path",
            ));
        }
        let options = RvfOptions {
            dimension,
            ..Default::default()
        };
        let store = Store::create(Path::new(path), options).map_err(failure)?;
        Ok(Self {
            inner: Mutex::new(Some(store)),
        })
    }
    #[staticmethod]
    #[pyo3(signature = (path, read_only=true))]
    fn open(path: &str, read_only: bool) -> PyResult<Self> {
        let store = if read_only {
            Store::open_readonly(Path::new(path))
        } else {
            Store::open(Path::new(path))
        }
        .map_err(failure)?;
        if store.dimension() == 0 || store.dimension() > 4096 {
            return Err(PyValueError::new_err("unsupported dimension"));
        }
        Ok(Self {
            inner: Mutex::new(Some(store)),
        })
    }
    #[getter]
    fn dimension(&self) -> PyResult<u16> {
        self.access(|s| Ok(s.dimension()))
    }
    fn ingest(&self, vectors: Vec<Vec<f32>>, ids: Vec<u64>) -> PyResult<u64> {
        if vectors.is_empty() || vectors.len() > 10000 || vectors.len() != ids.len() {
            return Err(PyValueError::new_err(
                "provide 1 to 10000 vectors and matching ids",
            ));
        }
        if ids.iter().copied().collect::<HashSet<_>>().len() != ids.len() {
            return Err(PyValueError::new_err("ids must be unique within a batch"));
        }
        self.access(|s| {
            for vector in &vectors {
                check_vector(vector, s.dimension() as usize)?;
            }
            let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
            let result = s.ingest_batch(&refs, &ids, None).map_err(failure)?;
            if result.rejected != 0 {
                return Err(failure(format!(
                    "ingest partially accepted {} vectors and rejected {}",
                    result.accepted, result.rejected
                )));
            }
            Ok(result.accepted)
        })
    }
    #[pyo3(signature = (vector, k=10))]
    fn query(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        if k == 0 || k > 10000 {
            return Err(PyValueError::new_err("k must be between 1 and 10000"));
        }
        self.access(|s| {
            check_vector(&vector, s.dimension() as usize)?;
            let opts = QueryOptions {
                force_exact: true,
                ..Default::default()
            };
            let results = s.query(&vector, k, &opts).map_err(failure)?;
            if results.iter().any(|r| !r.distance.is_finite()) {
                return Err(failure("native query returned non-finite distance"));
            }
            Ok(results.into_iter().map(|r| (r.id, r.distance)).collect())
        })
    }
    fn delete(&self, ids: Vec<u64>) -> PyResult<u64> {
        if ids.is_empty() || ids.len() > 10000 {
            return Err(PyValueError::new_err("provide 1 to 10000 ids"));
        }
        self.access(|s| Ok(s.delete(&ids).map_err(failure)?.deleted))
    }
    fn status(&self) -> PyResult<HashMap<String, u64>> {
        self.access(|s| {
            let v = s.status();
            Ok(HashMap::from([
                ("total_vectors".into(), v.total_vectors),
                ("file_size".into(), v.file_size),
                ("epoch".into(), v.current_epoch as u64),
                ("read_only".into(), u64::from(v.read_only)),
            ]))
        })
    }
    fn close(&self) -> PyResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| failure("store lock poisoned"))?;
        if let Some(store) = guard.take() {
            store.close().map_err(failure)?;
        }
        Ok(())
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RvfStore>()
}
