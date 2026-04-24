//! Backend-adapter trait. Every supported data lake (Parquet-on-S3,
//! BigQuery, Snowflake, Delta, Iceberg, …) implements this.
//!
//! ## The minimum surface
//!
//! - `id()` — a stable string identifier unique per-backend instance.
//! - `list_collections()` — what vector collections live in this backend.
//! - `pull_vectors(collection)` — stream all vectors in the collection.
//!   Called on cache miss / coherence bump.
//! - `generation(collection)` — an opaque coherence token (Parquet file
//!   mtime, Iceberg snapshot id, BQ `last_modified_time`, …). Used to
//!   decide whether the cache for this collection is still fresh.
//! - `supports_pushdown()` — optional; defaults to `false`. When `true`,
//!   the router may choose to push top-k ANN search into the backend
//!   instead of pulling all vectors. v1 does not actually call push-down;
//!   the flag is the forward-compatibility hook.
//!
//! `LocalBackend` is the in-memory reference implementation — used by
//! tests, demos, and the "does the federation path round-trip correctly"
//! smoke.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{Result, RuLakeError};

/// Stable identifier for a registered backend.
pub type BackendId = String;

/// Collection name inside a given backend — globally unique only in
/// conjunction with its `BackendId`.
pub type CollectionId = String;

/// One pull from a backend. Vectors are returned by value — we assume
/// the caller (the cache) is the only consumer and will compress them
/// into 1-bit codes immediately.
#[derive(Debug, Clone)]
pub struct PulledBatch {
    /// Collection this batch belongs to.
    pub collection: CollectionId,
    /// Parallel arrays: `ids[i]` and `vectors[i]` describe vector i.
    pub ids: Vec<u64>,
    /// Each vector must have length `dim`.
    pub vectors: Vec<Vec<f32>>,
    /// Dimensionality reported by the backend.
    pub dim: usize,
    /// Coherence token — when this bumps the cache is stale.
    pub generation: u64,
}

/// Hard caps enforced on any `PulledBatch` returned from a backend.
/// Protects `RuLake` from a hostile or corrupt backend returning an
/// impossibly-large batch that would OOM the host. Operators with
/// legitimate use cases beyond these bounds can raise them at the
/// call site after explicit review.
///
/// Values chosen to cover every realistic collection size seen in
/// production vector workloads (100M vectors × 8192 dim is ~8 TB
/// of f32, more than any single serving process handles).
pub const MAX_PULLED_VECTORS: usize = 100_000_000;
pub const MAX_PULLED_DIM: usize = 8192;
pub const MAX_PULLED_BYTES: usize = 16 * 1024 * 1024 * 1024; // 16 GiB

/// Validate a `PulledBatch` against the hard caps above. Called by
/// `VectorCache::prime` before allocating. Returns an error rather
/// than panicking because a hostile backend reaching this check is
/// expected — fail the one search, keep the process alive.
pub(crate) fn validate_pulled_batch(batch: &PulledBatch) -> Result<()> {
    if batch.ids.len() != batch.vectors.len() {
        return Err(RuLakeError::InvalidParameter(format!(
            "PulledBatch: ids.len={} != vectors.len={}",
            batch.ids.len(),
            batch.vectors.len()
        )));
    }
    if batch.ids.len() > MAX_PULLED_VECTORS {
        return Err(RuLakeError::InvalidParameter(format!(
            "PulledBatch: {} vectors exceeds cap {MAX_PULLED_VECTORS}",
            batch.ids.len()
        )));
    }
    if batch.dim == 0 || batch.dim > MAX_PULLED_DIM {
        return Err(RuLakeError::InvalidParameter(format!(
            "PulledBatch: dim={} outside (0, {MAX_PULLED_DIM}]",
            batch.dim
        )));
    }
    // checked_mul catches the 32-bit-usize overflow case the security
    // audit flagged: a 64-bit count field that truncates when cast.
    let bytes = batch
        .ids
        .len()
        .checked_mul(batch.dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| {
            RuLakeError::InvalidParameter(format!(
                "PulledBatch: size overflow (n={}, dim={})",
                batch.ids.len(),
                batch.dim
            ))
        })?;
    if bytes > MAX_PULLED_BYTES {
        return Err(RuLakeError::InvalidParameter(format!(
            "PulledBatch: {bytes} bytes exceeds cap {MAX_PULLED_BYTES}"
        )));
    }
    Ok(())
}

pub trait BackendAdapter: Send + Sync {
    fn id(&self) -> &str;

    fn list_collections(&self) -> Result<Vec<CollectionId>>;

    fn pull_vectors(&self, collection: &str) -> Result<PulledBatch>;

    fn generation(&self, collection: &str) -> Result<u64>;

    /// Produce a [`RuLakeBundle`](crate::RuLakeBundle) describing the
    /// *current* state of this collection. Default impl synthesizes a
    /// bundle from `id() + collection + generation()`; real backends
    /// (Parquet, BigQuery, Iceberg, …) should override with their
    /// authoritative `data_ref` so two deployments reading the same
    /// underlying bytes produce the same witness and share the cache.
    fn current_bundle(
        &self,
        collection: &str,
        rotation_seed: u64,
        rerank_factor: usize,
    ) -> Result<crate::RuLakeBundle> {
        // Needs the dim; default impl does a single pull to get it.
        // Real backends override to avoid the pull on a hot path.
        let batch = self.pull_vectors(collection)?;
        Ok(crate::RuLakeBundle::new(
            format!("{}://{}", self.id(), collection),
            batch.dim,
            rotation_seed,
            rerank_factor,
            crate::Generation::Num(batch.generation),
        ))
    }

    fn supports_pushdown(&self) -> bool {
        false
    }
}

// ────────────────────────────────────────────────────────────────────
// LocalBackend — in-memory reference impl
// ────────────────────────────────────────────────────────────────────

/// In-memory backend. Useful as a demo, as the unit-test substrate, and
/// as an example for real-backend implementers (ParquetBackend,
/// BigQueryBackend, …). Thread-safe: the inner collections table is
/// guarded by an `RwLock` so the backend can be shared across threads.
#[derive(Clone)]
pub struct LocalBackend {
    id: String,
    inner: Arc<RwLock<LocalState>>,
}

struct LocalState {
    collections: HashMap<CollectionId, LocalCollection>,
}

struct LocalCollection {
    dim: usize,
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
    generation: u64,
}

impl LocalBackend {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            inner: Arc::new(RwLock::new(LocalState {
                collections: HashMap::new(),
            })),
        }
    }

    /// Insert a collection wholesale. Bumps the generation so any cache
    /// watching this collection sees it as stale on the next check.
    pub fn put_collection(
        &self,
        name: impl Into<String>,
        dim: usize,
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<()> {
        if ids.len() != vectors.len() {
            return Err(RuLakeError::InvalidParameter(format!(
                "put_collection: ids.len={} != vectors.len={}",
                ids.len(),
                vectors.len()
            )));
        }
        for v in &vectors {
            if v.len() != dim {
                return Err(RuLakeError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }
        let mut inner = self.inner.write().unwrap();
        let entry = inner
            .collections
            .entry(name.into())
            .or_insert(LocalCollection {
                dim,
                ids: Vec::new(),
                vectors: Vec::new(),
                generation: 0,
            });
        entry.dim = dim;
        entry.ids = ids;
        entry.vectors = vectors;
        entry.generation = entry.generation.wrapping_add(1);
        Ok(())
    }

    /// Append a single vector. Bumps generation.
    pub fn append(&self, collection: impl Into<String>, id: u64, vector: Vec<f32>) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        let name = collection.into();
        let entry =
            inner
                .collections
                .get_mut(&name)
                .ok_or_else(|| RuLakeError::UnknownCollection {
                    backend: self.id.clone(),
                    collection: name.clone(),
                })?;
        if entry.dim == 0 {
            entry.dim = vector.len();
        }
        if vector.len() != entry.dim {
            return Err(RuLakeError::DimensionMismatch {
                expected: entry.dim,
                actual: vector.len(),
            });
        }
        entry.ids.push(id);
        entry.vectors.push(vector);
        entry.generation = entry.generation.wrapping_add(1);
        Ok(())
    }
}

impl BackendAdapter for LocalBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn list_collections(&self) -> Result<Vec<CollectionId>> {
        Ok(self
            .inner
            .read()
            .unwrap()
            .collections
            .keys()
            .cloned()
            .collect())
    }

    fn pull_vectors(&self, collection: &str) -> Result<PulledBatch> {
        let inner = self.inner.read().unwrap();
        let c =
            inner
                .collections
                .get(collection)
                .ok_or_else(|| RuLakeError::UnknownCollection {
                    backend: self.id.clone(),
                    collection: collection.to_string(),
                })?;
        Ok(PulledBatch {
            collection: collection.to_string(),
            ids: c.ids.clone(),
            vectors: c.vectors.clone(),
            dim: c.dim,
            generation: c.generation,
        })
    }

    fn generation(&self, collection: &str) -> Result<u64> {
        let inner = self.inner.read().unwrap();
        let c =
            inner
                .collections
                .get(collection)
                .ok_or_else(|| RuLakeError::UnknownCollection {
                    backend: self.id.clone(),
                    collection: collection.to_string(),
                })?;
        Ok(c.generation)
    }

    /// Override: `LocalBackend` already knows its dim without pulling
    /// vectors, so synthesize the bundle directly. Tests that rely on
    /// cache sharing across two backends (same data, different id)
    /// can set `data_ref_override` below by wrapping `LocalBackend` in
    /// a thin backend shim; the default uses `"local://{id}/{coll}"`
    /// so two distinct `LocalBackend` instances never share the cache
    /// unless explicitly overridden.
    fn current_bundle(
        &self,
        collection: &str,
        rotation_seed: u64,
        rerank_factor: usize,
    ) -> Result<crate::RuLakeBundle> {
        let inner = self.inner.read().unwrap();
        let c =
            inner
                .collections
                .get(collection)
                .ok_or_else(|| RuLakeError::UnknownCollection {
                    backend: self.id.clone(),
                    collection: collection.to_string(),
                })?;
        Ok(crate::RuLakeBundle::new(
            format!("local://{}/{}", self.id, collection),
            c.dim,
            rotation_seed,
            rerank_factor,
            crate::Generation::Num(c.generation),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(n: usize, dim: usize) -> PulledBatch {
        PulledBatch {
            collection: "c".to_string(),
            ids: (0..n as u64).collect(),
            vectors: vec![vec![0.0; dim]; n],
            dim,
            generation: 1,
        }
    }

    #[test]
    fn pulled_batch_validator_accepts_reasonable_size() {
        // 10k vectors × 128 dim × 4 bytes = 5 MiB — well within caps.
        assert!(validate_pulled_batch(&batch(10_000, 128)).is_ok());
    }

    #[test]
    fn pulled_batch_validator_rejects_dim_zero() {
        let b = PulledBatch {
            collection: "c".to_string(),
            ids: vec![0],
            vectors: vec![vec![]],
            dim: 0,
            generation: 1,
        };
        assert!(validate_pulled_batch(&b).is_err());
    }

    #[test]
    fn pulled_batch_validator_rejects_dim_over_cap() {
        let b = PulledBatch {
            collection: "c".to_string(),
            ids: vec![0],
            vectors: vec![vec![0.0; MAX_PULLED_DIM + 1]],
            dim: MAX_PULLED_DIM + 1,
            generation: 1,
        };
        assert!(validate_pulled_batch(&b).is_err());
    }

    #[test]
    fn pulled_batch_validator_rejects_len_mismatch() {
        let b = PulledBatch {
            collection: "c".to_string(),
            ids: vec![0, 1],
            vectors: vec![vec![0.0; 4]],
            dim: 4,
            generation: 1,
        };
        assert!(validate_pulled_batch(&b).is_err());
    }
}
