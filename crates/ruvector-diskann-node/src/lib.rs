//! NAPI-RS bindings for ruvector-diskann

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_diskann::{DiskAnnConfig, DiskAnnIndex as CoreIndex};
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;

#[napi(object)]
pub struct DiskAnnOptions {
    pub dim: u32,
    pub max_degree: Option<u32>,
    pub build_beam: Option<u32>,
    pub search_beam: Option<u32>,
    pub alpha: Option<f64>,
    pub pq_subspaces: Option<u32>,
    pub pq_iterations: Option<u32>,
    pub storage_path: Option<String>,
}

#[napi(object)]
pub struct DiskAnnSearchResult {
    pub id: String,
    pub distance: f64,
}

#[napi]
pub struct DiskAnn {
    inner: Arc<RwLock<CoreIndex>>,
}

#[napi]
impl DiskAnn {
    #[napi(constructor)]
    pub fn new(options: DiskAnnOptions) -> Result<Self> {
        let config = DiskAnnConfig {
            dim: options.dim as usize,
            max_degree: options.max_degree.unwrap_or(64) as usize,
            build_beam: options.build_beam.unwrap_or(128) as usize,
            search_beam: options.search_beam.unwrap_or(64) as usize,
            alpha: options.alpha.unwrap_or(1.2) as f32,
            pq_subspaces: options.pq_subspaces.unwrap_or(0) as usize,
            pq_iterations: options.pq_iterations.unwrap_or(10) as usize,
            storage_path: options.storage_path.map(PathBuf::from),
        };
        let index = CoreIndex::new(config);
        Ok(Self {
            inner: Arc::new(RwLock::new(index)),
        })
    }

    /// Insert a vector with a string ID
    #[napi]
    pub fn insert(&self, id: String, vector: Float32Array) -> Result<()> {
        let v: Vec<f32> = vector.to_vec();
        self.inner
            .write()
            .insert(id, v)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Insert multiple vectors: ids[] and flat Float32Array (n * dim)
    #[napi]
    pub fn insert_batch(&self, ids: Vec<String>, vectors: Float32Array, dim: u32) -> Result<()> {
        let d = dim as usize;
        let data: Vec<f32> = vectors.to_vec();
        if data.len() != ids.len() * d {
            return Err(Error::from_reason(format!(
                "Expected {} floats ({} ids x {} dim), got {}",
                ids.len() * d,
                ids.len(),
                d,
                data.len()
            )));
        }
        let mut batch = Vec::with_capacity(ids.len());
        for (i, id) in ids.into_iter().enumerate() {
            batch.push((id, data[i * d..(i + 1) * d].to_vec()));
        }
        self.inner
            .write()
            .insert_batch(batch)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Build the index (must be called after inserts, before search)
    #[napi]
    pub fn build(&self) -> Result<()> {
        self.inner
            .write()
            .build()
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Build the index asynchronously
    #[napi]
    pub async fn build_async(&self) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            inner
                .write()
                .build()
                .map_err(|e| Error::from_reason(e.to_string()))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Search for k nearest neighbors
    #[napi]
    pub fn search(&self, query: Float32Array, k: u32) -> Result<Vec<DiskAnnSearchResult>> {
        let q: Vec<f32> = query.to_vec();
        let results = self
            .inner
            .read()
            .search(&q, k as usize)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| DiskAnnSearchResult {
                id: r.id,
                distance: r.distance as f64,
            })
            .collect())
    }

    /// Search asynchronously
    #[napi]
    pub async fn search_async(
        &self,
        query: Float32Array,
        k: u32,
    ) -> Result<Vec<DiskAnnSearchResult>> {
        let inner = self.inner.clone();
        let q: Vec<f32> = query.to_vec();

        tokio::task::spawn_blocking(move || {
            let results = inner
                .read()
                .search(&q, k as usize)
                .map_err(|e| Error::from_reason(e.to_string()))?;

            Ok(results
                .into_iter()
                .map(|r| DiskAnnSearchResult {
                    id: r.id,
                    distance: r.distance as f64,
                })
                .collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Delete a vector by ID
    #[napi]
    pub fn delete(&self, id: String) -> Result<bool> {
        self.inner
            .write()
            .delete(&id)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Get the number of vectors
    #[napi]
    pub fn count(&self) -> u32 {
        self.inner.read().count() as u32
    }

    /// Save index to disk
    #[napi]
    pub fn save(&self, dir: String) -> Result<()> {
        self.inner
            .read()
            .save(std::path::Path::new(&dir))
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Load index from disk
    #[napi(factory)]
    pub fn load(dir: String) -> Result<Self> {
        let index = CoreIndex::load(std::path::Path::new(&dir))
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(index)),
        })
    }
}
