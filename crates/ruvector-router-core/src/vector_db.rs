//! Main VectorDB API

use crate::error::{Result, VectorDbError};
use crate::index::{HnswConfig, HnswIndex};
use crate::storage::Storage;
use crate::types::*;
use parking_lot::{Mutex, RwLock};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Main Vector Database
pub struct VectorDB {
    config: VectorDbConfig,
    storage: Arc<Storage>,
    index: Arc<HnswIndex>,
    stats: Arc<RwLock<VectorDbStats>>,
    /// Serializes check-then-insert while `enforce_max_elements` is on, so two
    /// concurrent inserts cannot both pass the capacity check (#1099).
    capacity_guard: Mutex<()>,
}

impl VectorDB {
    /// Create a new vector database with configuration
    pub fn new(config: VectorDbConfig) -> Result<Self> {
        let storage = Arc::new(Storage::new(&config.storage_path)?);

        let hnsw_config = HnswConfig {
            m: config.hnsw_m,
            ef_construction: config.hnsw_ef_construction,
            ef_search: config.hnsw_ef_search,
            metric: config.distance_metric,
            dimensions: config.dimensions,
        };

        let index = Arc::new(HnswIndex::new(hnsw_config));

        // Issue #430: rebuild the in-memory HNSW from persisted vectors. Without
        // this step a fresh `HnswIndex::new` is created on every open, so all
        // previously-inserted vectors are invisible to search after restart
        // (search returns 0 results despite `get_all_ids` listing them).
        let stored_ids = storage.get_all_ids()?;
        let total_vectors = stored_ids.len();
        if !stored_ids.is_empty() {
            let mut entries = Vec::with_capacity(stored_ids.len());
            for id in &stored_ids {
                if let Some(vector) = storage.get(id)? {
                    entries.push((id.clone(), vector));
                }
            }
            index.insert_batch(entries)?;
        }

        let stats = Arc::new(RwLock::new(VectorDbStats {
            total_vectors,
            index_size_bytes: 0,
            storage_size_bytes: 0,
            avg_query_latency_us: 0.0,
            qps: 0.0,
        }));

        Ok(Self {
            config,
            storage,
            index,
            stats,
            capacity_guard: Mutex::new(()),
        })
    }

    /// Create a builder for configuring the database
    pub fn builder() -> VectorDbBuilder {
        VectorDbBuilder::default()
    }

    /// Fail with `CapacityExceeded` if adding `ids` would exceed an enforced
    /// `max_elements`. Ids already present (or repeated within `ids`) replace
    /// rather than grow the database, so they don't count.
    fn check_capacity<'a>(&self, ids: impl Iterator<Item = &'a str>) -> Result<()> {
        if !self.config.enforce_max_elements {
            return Ok(());
        }
        let mut seen = std::collections::HashSet::new();
        let new_ids = ids
            .filter(|id| !self.index.contains(id) && seen.insert(*id))
            .count();
        let attempted = self.index.len() + new_ids;
        if attempted > self.config.max_elements {
            return Err(VectorDbError::CapacityExceeded {
                max_elements: self.config.max_elements,
                attempted,
            });
        }
        Ok(())
    }

    /// Insert a vector entry
    ///
    /// Fails with [`VectorDbError::CapacityExceeded`] when
    /// `enforce_max_elements` is set and the insert would exceed `max_elements`.
    pub fn insert(&self, entry: VectorEntry) -> Result<String> {
        // Validate dimensions
        if entry.vector.len() != self.config.dimensions {
            return Err(VectorDbError::InvalidDimensions {
                expected: self.config.dimensions,
                actual: entry.vector.len(),
            });
        }

        let _guard = self
            .config
            .enforce_max_elements
            .then(|| self.capacity_guard.lock());
        self.check_capacity(std::iter::once(entry.id.as_str()))?;

        // Store in storage layer
        self.storage.insert(&entry)?;

        // Insert into index
        self.index.insert(entry.id.clone(), entry.vector)?;

        // Update stats
        self.stats.write().total_vectors += 1;

        Ok(entry.id)
    }

    /// Insert multiple vectors in batch
    pub fn insert_batch(&self, entries: Vec<VectorEntry>) -> Result<Vec<String>> {
        // Validate all dimensions first
        for entry in &entries {
            if entry.vector.len() != self.config.dimensions {
                return Err(VectorDbError::InvalidDimensions {
                    expected: self.config.dimensions,
                    actual: entry.vector.len(),
                });
            }
        }

        // The whole batch is rejected before anything is written.
        let _guard = self
            .config
            .enforce_max_elements
            .then(|| self.capacity_guard.lock());
        self.check_capacity(entries.iter().map(|e| e.id.as_str()))?;

        // Store in storage layer
        self.storage.insert_batch(&entries)?;

        // Build index entries
        let index_entries: Vec<(String, Vec<f32>)> = entries
            .iter()
            .map(|e| (e.id.clone(), e.vector.clone()))
            .collect();

        // Insert into index
        self.index.insert_batch(index_entries)?;

        // Update stats
        self.stats.write().total_vectors += entries.len();

        Ok(entries.into_iter().map(|e| e.id).collect())
    }

    /// Search for similar vectors
    pub fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let start = Instant::now();

        // Validate query vector dimensions
        if query.vector.len() != self.config.dimensions {
            return Err(VectorDbError::InvalidDimensions {
                expected: self.config.dimensions,
                actual: query.vector.len(),
            });
        }

        // Search index
        let mut results = self.index.search(&query)?;

        // Enrich results with metadata if needed
        for result in &mut results {
            if let Some(metadata) = self.storage.get_metadata(&result.id)? {
                result.metadata = metadata;
            }
        }

        // Apply metadata filters if specified
        if let Some(filters) = &query.filters {
            results.retain(|r| {
                filters
                    .iter()
                    .all(|(key, value)| r.metadata.get(key).map(|v| v == value).unwrap_or(false))
            });
        }

        // Update stats
        let latency_us = start.elapsed().as_micros() as f64;
        let mut stats = self.stats.write();
        stats.avg_query_latency_us = (stats.avg_query_latency_us * 0.9) + (latency_us * 0.1);

        Ok(results)
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let deleted = self.storage.delete(id)?;

        if deleted {
            self.index.remove(id)?;
            let mut stats = self.stats.write();
            stats.total_vectors = stats.total_vectors.saturating_sub(1);
        }

        Ok(deleted)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        if let Some(vector) = self.storage.get(id)? {
            let metadata = self.storage.get_metadata(id)?.unwrap_or_default();

            Ok(Some(VectorEntry {
                id: id.to_string(),
                vector,
                metadata,
                timestamp: chrono::Utc::now().timestamp(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get database statistics
    pub fn stats(&self) -> VectorDbStats {
        self.stats.read().clone()
    }

    /// Get total number of vectors
    pub fn count(&self) -> Result<usize> {
        self.storage.count()
    }

    /// Get all vector IDs
    pub fn get_all_ids(&self) -> Result<Vec<String>> {
        self.storage.get_all_ids()
    }
}

/// Builder for VectorDB configuration
#[derive(Debug, Clone, Default)]
pub struct VectorDbBuilder {
    config: VectorDbConfig,
}

impl VectorDbBuilder {
    /// Set vector dimensions
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.config.dimensions = dimensions;
        self
    }

    /// Set the capacity hint. Not a hard limit unless
    /// [`VectorDbBuilder::enforce_max_elements`] is also set (#1099).
    pub fn max_elements(mut self, max_elements: usize) -> Self {
        self.config.max_elements = max_elements;
        self
    }

    /// Enforce `max_elements` as a hard bound: inserts that would exceed it
    /// fail with `VectorDbError::CapacityExceeded`.
    pub fn enforce_max_elements(mut self, enforce: bool) -> Self {
        self.config.enforce_max_elements = enforce;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Set HNSW M parameter
    pub fn hnsw_m(mut self, m: usize) -> Self {
        self.config.hnsw_m = m;
        self
    }

    /// Set HNSW ef_construction parameter
    pub fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.config.hnsw_ef_construction = ef;
        self
    }

    /// Set HNSW ef_search parameter
    pub fn hnsw_ef_search(mut self, ef: usize) -> Self {
        self.config.hnsw_ef_search = ef;
        self
    }

    /// Set quantization type
    pub fn quantization(mut self, qtype: QuantizationType) -> Self {
        self.config.quantization = qtype;
        self
    }

    /// Set storage path
    pub fn storage_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.storage_path = path.as_ref().to_string_lossy().to_string();
        self
    }

    /// Enable or disable memory mapping
    pub fn mmap_vectors(mut self, mmap: bool) -> Self {
        self.config.mmap_vectors = mmap;
        self
    }

    /// Build the VectorDB instance
    pub fn build(self) -> Result<VectorDB> {
        VectorDB::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn entry(id: &str) -> VectorEntry {
        VectorEntry {
            id: id.to_string(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: std::collections::HashMap::new(),
            timestamp: 0,
        }
    }

    /// #1099: `max_elements` alone is a capacity hint, not a limit. This pins
    /// the documented default so it can't silently change for existing users.
    #[test]
    fn max_elements_without_enforcement_is_only_a_hint() {
        let dir = tempdir().unwrap();
        let db = VectorDB::builder()
            .dimensions(3)
            .max_elements(5)
            .storage_path(dir.path().join("hint.db"))
            .build()
            .unwrap();
        for i in 0..12 {
            db.insert(entry(&format!("v{i}"))).unwrap();
        }
        assert_eq!(db.count().unwrap(), 12);
    }

    /// #1099: with `enforce_max_elements`, the bound is real and typed.
    #[test]
    fn enforced_max_elements_rejects_the_insert_that_would_exceed_it() {
        let dir = tempdir().unwrap();
        let db = VectorDB::builder()
            .dimensions(3)
            .max_elements(5)
            .enforce_max_elements(true)
            .storage_path(dir.path().join("strict.db"))
            .build()
            .unwrap();
        for i in 0..5 {
            db.insert(entry(&format!("v{i}"))).unwrap();
        }
        let err = db.insert(entry("v5")).unwrap_err();
        assert!(
            matches!(
                err,
                VectorDbError::CapacityExceeded {
                    max_elements: 5,
                    attempted: 6
                }
            ),
            "got {err:?}"
        );
        assert_eq!(db.count().unwrap(), 5, "rejected insert must not be stored");

        // Replacing an existing id at capacity does not grow the database.
        db.insert(entry("v2")).unwrap();
        assert_eq!(db.count().unwrap(), 5);

        // Deleting frees a slot.
        assert!(db.delete("v0").unwrap());
        db.insert(entry("v5")).unwrap();
        assert_eq!(db.count().unwrap(), 5);
    }

    /// #1099: concurrent inserts cannot both pass the capacity check.
    #[test]
    fn enforced_max_elements_holds_under_concurrent_inserts() {
        let dir = tempdir().unwrap();
        let db = std::sync::Arc::new(
            VectorDB::builder()
                .dimensions(3)
                .max_elements(5)
                .enforce_max_elements(true)
                .storage_path(dir.path().join("race.db"))
                .build()
                .unwrap(),
        );
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let db = db.clone();
                std::thread::spawn(move || {
                    (0..5)
                        .filter(|i| db.insert(entry(&format!("t{t}-{i}"))).is_ok())
                        .count()
                })
            })
            .collect();
        let accepted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(accepted, 5);
        assert_eq!(db.count().unwrap(), 5);
    }

    /// #1099: a batch that would exceed the bound is rejected as a whole,
    /// before anything is written; duplicate ids inside the batch count once.
    #[test]
    fn enforced_max_elements_rejects_an_over_limit_batch_atomically() {
        let dir = tempdir().unwrap();
        let db = VectorDB::builder()
            .dimensions(3)
            .max_elements(3)
            .enforce_max_elements(true)
            .storage_path(dir.path().join("batch.db"))
            .build()
            .unwrap();
        db.insert(entry("a")).unwrap();
        let err = db
            .insert_batch(vec![entry("b"), entry("c"), entry("d")])
            .unwrap_err();
        assert!(matches!(
            err,
            VectorDbError::CapacityExceeded {
                max_elements: 3,
                attempted: 4
            }
        ));
        assert_eq!(
            db.count().unwrap(),
            1,
            "nothing from a rejected batch is stored"
        );

        // "a" already exists and "b" repeats: only b and c are new -> 3, allowed.
        db.insert_batch(vec![entry("a"), entry("b"), entry("b"), entry("c")])
            .unwrap();
        assert_eq!(db.count().unwrap(), 3);
    }

    #[test]
    fn test_vector_db_basic_operations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let db = VectorDB::builder()
            .dimensions(3)
            .storage_path(&path)
            .build()
            .unwrap();

        // Insert
        let entry = VectorEntry {
            id: "test1".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: std::collections::HashMap::new(),
            timestamp: 0,
        };

        let id = db.insert(entry).unwrap();
        assert_eq!(id, "test1");

        // Search
        let query = SearchQuery {
            vector: vec![0.9, 0.1, 0.0],
            k: 1,
            filters: None,
            threshold: None,
            ef_search: None,
        };

        let results = db.search(query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test1");

        // Delete
        assert!(db.delete("test1").unwrap());
        assert_eq!(db.count().unwrap(), 0);
    }

    /// Issue #430: search must return persisted vectors after the VectorDB is
    /// reopened. Before the fix, `VectorDB::new` always created an empty
    /// in-memory HNSW, so `search` returned 0 results despite the storage
    /// containing the vectors.
    #[test]
    fn test_index_rebuilt_from_storage_on_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rebuild.db");

        // Write a handful of vectors with the first DB instance.
        {
            let db = VectorDB::builder()
                .dimensions(4)
                .storage_path(&path)
                .build()
                .unwrap();
            for i in 0..5u32 {
                let v = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 5) as f32];
                db.insert(VectorEntry {
                    id: format!("v{i}"),
                    vector: v,
                    metadata: std::collections::HashMap::new(),
                    timestamp: 0,
                })
                .unwrap();
            }
        } // drop closes the storage handle.

        // Reopen against the same on-disk path — index must be rebuilt.
        let db = VectorDB::builder()
            .dimensions(4)
            .storage_path(&path)
            .build()
            .unwrap();

        assert_eq!(
            db.count().unwrap(),
            5,
            "storage.count() should report persisted vectors"
        );

        let q = SearchQuery {
            vector: vec![2.0, 4.0, 6.0, 10.0], // matches v2 exactly
            k: 3,
            filters: None,
            threshold: None,
            ef_search: None,
        };
        let results = db.search(q).unwrap();
        assert!(
            !results.is_empty(),
            "regression of #430: search returned 0 results after reopening; \
             index was not rebuilt from storage"
        );
        assert_eq!(
            results[0].id, "v2",
            "exact-match query should return v2 as top hit"
        );
    }
}
