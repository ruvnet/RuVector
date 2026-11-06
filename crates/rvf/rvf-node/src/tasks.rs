//! Store operations shared by the synchronous methods and their
//! Promise-returning `*Async` variants.
//!
//! The async variants are napi `AsyncTask`s: `compute` runs on the libuv
//! threadpool, so the JS event loop keeps turning while a query or a
//! compaction runs. The store sits behind an `RwLock`: queries take the read
//! lock (and run in parallel, up to `UV_THREADPOOL_SIZE` at once), while
//! ingest, delete and compaction take the write lock. Arguments are copied
//! out of JS values before the task is queued and parsed inside `compute`,
//! so every failure -- bad filter JSON, closed store, runtime error --
//! surfaces as a rejected promise rather than a synchronous throw. A task
//! that runs after `close()` rejects with "Store is closed".

use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi::{Env, Task};

use rvf_runtime::options::{MetadataEntry as RustMetadataEntry, QueryOptions as RustQueryOptions};
use rvf_runtime::RvfStore;

use crate::{
    map_rvf_err, parse_filter_json, parse_metadata_entry, RvfCompactionResult, RvfDeleteResult,
    RvfIngestResult, RvfMetadataEntry, RvfQueryOptions, RvfSearchResult,
};

/// The store handle shared between a `RvfDatabase` and its queued tasks.
pub(crate) type SharedStore = Arc<RwLock<Option<RvfStore>>>;

fn poisoned() -> napi::Error {
    napi::Error::from_reason("Lock poisoned")
}

fn closed() -> napi::Error {
    napi::Error::from_reason("Store is closed")
}

/// Run `f` with shared (read) access to the open store.
pub(crate) fn with_store<R>(
    inner: &SharedStore,
    f: impl FnOnce(&RvfStore) -> Result<R>,
) -> Result<R> {
    let guard = inner.read().map_err(|_| poisoned())?;
    f(guard.as_ref().ok_or_else(closed)?)
}

/// Run `f` with exclusive (write) access to the open store.
pub(crate) fn with_store_mut<R>(
    inner: &SharedStore,
    f: impl FnOnce(&mut RvfStore) -> Result<R>,
) -> Result<R> {
    let mut guard = inner.write().map_err(|_| poisoned())?;
    f(guard.as_mut().ok_or_else(closed)?)
}

fn query_options(options: Option<&RvfQueryOptions>) -> Result<RustQueryOptions> {
    let opts = match options {
        Some(opts) => opts,
        None => return Ok(RustQueryOptions::default()),
    };
    let filter = match &opts.filter {
        Some(json_str) => Some(parse_filter_json(json_str)?),
        None => None,
    };
    Ok(RustQueryOptions {
        ef_search: opts.ef_search.unwrap_or(100) as u16,
        filter,
        timeout_ms: opts.timeout_ms.unwrap_or(0),
        rabitq: opts.rabitq.unwrap_or(false),
        force_exact: opts.force_exact.unwrap_or(false),
        ..RustQueryOptions::default()
    })
}

pub(crate) fn run_query(
    store: &RvfStore,
    vector: &[f32],
    k: u32,
    options: Option<&RvfQueryOptions>,
) -> Result<Vec<RvfSearchResult>> {
    let rust_opts = query_options(options)?;
    let results = store
        .query(vector, k as usize, &rust_opts)
        .map_err(map_rvf_err)?;
    Ok(results
        .into_iter()
        .map(|r| RvfSearchResult {
            id: r.id as i64,
            distance: r.distance as f64,
            metadata: crate::metadata::record_to_js(store, r.id),
        })
        .collect())
}

pub(crate) fn run_ingest(
    store: &mut RvfStore,
    vectors: &[f32],
    ids: &[i64],
    metadata: Option<&[RvfMetadataEntry]>,
) -> Result<RvfIngestResult> {
    let n = ids.len();
    if n == 0 {
        return Ok(RvfIngestResult {
            accepted: 0,
            rejected: 0,
            epoch: 0,
        });
    }
    let total_floats = vectors.len();
    if !total_floats.is_multiple_of(n) {
        return Err(napi::Error::from_reason(format!(
            "vectors length ({total_floats}) must be divisible by ids length ({n})"
        )));
    }
    let dim = total_floats / n;
    let vec_slices: Vec<&[f32]> = (0..n).map(|i| &vectors[i * dim..(i + 1) * dim]).collect();
    let rust_ids: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
    let rust_metadata: Option<Vec<RustMetadataEntry>> = match metadata {
        Some(entries) => Some(
            entries
                .iter()
                .map(parse_metadata_entry)
                .collect::<Result<_>>()?,
        ),
        None => None,
    };
    let result = store
        .ingest_batch(&vec_slices, &rust_ids, rust_metadata.as_deref())
        .map_err(map_rvf_err)?;
    Ok(RvfIngestResult {
        accepted: result.accepted as i64,
        rejected: result.rejected as i64,
        epoch: result.epoch,
    })
}

pub(crate) fn run_delete(store: &mut RvfStore, ids: &[i64]) -> Result<RvfDeleteResult> {
    let rust_ids: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
    let result = store.delete(&rust_ids).map_err(map_rvf_err)?;
    Ok(RvfDeleteResult {
        deleted: result.deleted as i64,
        epoch: result.epoch,
    })
}

pub(crate) fn run_compact(store: &mut RvfStore) -> Result<RvfCompactionResult> {
    let result = store.compact().map_err(map_rvf_err)?;
    Ok(RvfCompactionResult {
        segments_compacted: result.segments_compacted,
        bytes_reclaimed: result.bytes_reclaimed as i64,
        epoch: result.epoch,
    })
}

/// `queryAsync` work item.
pub struct QueryTask {
    pub(crate) inner: SharedStore,
    pub(crate) vector: Vec<f32>,
    pub(crate) k: u32,
    pub(crate) options: Option<RvfQueryOptions>,
}

impl Task for QueryTask {
    type Output = Vec<RvfSearchResult>;
    type JsValue = Vec<RvfSearchResult>;

    fn compute(&mut self) -> Result<Self::Output> {
        with_store(&self.inner, |s| {
            run_query(s, &self.vector, self.k, self.options.as_ref())
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// `ingestBatchAsync` work item.
pub struct IngestTask {
    pub(crate) inner: SharedStore,
    pub(crate) vectors: Vec<f32>,
    pub(crate) ids: Vec<i64>,
    pub(crate) metadata: Option<Vec<RvfMetadataEntry>>,
}

impl Task for IngestTask {
    type Output = RvfIngestResult;
    type JsValue = RvfIngestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        with_store_mut(&self.inner, |s| {
            run_ingest(s, &self.vectors, &self.ids, self.metadata.as_deref())
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// `deleteAsync` work item.
pub struct DeleteTask {
    pub(crate) inner: SharedStore,
    pub(crate) ids: Vec<i64>,
}

impl Task for DeleteTask {
    type Output = RvfDeleteResult;
    type JsValue = RvfDeleteResult;

    fn compute(&mut self) -> Result<Self::Output> {
        with_store_mut(&self.inner, |s| run_delete(s, &self.ids))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// `compactAsync` work item.
pub struct CompactTask {
    pub(crate) inner: SharedStore,
}

impl Task for CompactTask {
    type Output = RvfCompactionResult;
    type JsValue = RvfCompactionResult;

    fn compute(&mut self) -> Result<Self::Output> {
        with_store_mut(&self.inner, run_compact)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}
