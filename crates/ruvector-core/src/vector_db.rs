//! Main VectorDB interface

use crate::error::Result;
use crate::index::flat::FlatIndex;

#[cfg(feature = "hnsw")]
use crate::index::hnsw::HnswIndex;

use crate::index::VectorIndex;
use crate::types::*;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::storage_memory::MemoryStorage;

#[cfg(feature = "storage")]
enum StorageBackend {
    Persistent(crate::storage::VectorStorage),
    Memory(MemoryStorage),
}

#[cfg(not(feature = "storage"))]
type StorageBackend = MemoryStorage;

#[cfg(feature = "storage")]
impl StorageBackend {
    fn insert(&self, entry: &VectorEntry) -> Result<VectorId> {
        match self {
            Self::Persistent(storage) => storage.insert(entry),
            Self::Memory(storage) => storage.insert(entry),
        }
    }

    fn insert_batch(&self, entries: &[VectorEntry]) -> Result<Vec<VectorId>> {
        match self {
            Self::Persistent(storage) => storage.insert_batch(entries),
            Self::Memory(storage) => storage.insert_batch(entries),
        }
    }

    fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        match self {
            Self::Persistent(storage) => storage.get(id),
            Self::Memory(storage) => storage.get(id),
        }
    }

    fn delete(&self, id: &str) -> Result<bool> {
        match self {
            Self::Persistent(storage) => storage.delete(id),
            Self::Memory(storage) => storage.delete(id),
        }
    }

    fn len(&self) -> Result<usize> {
        match self {
            Self::Persistent(storage) => storage.len(),
            Self::Memory(storage) => storage.len(),
        }
    }

    fn is_empty(&self) -> Result<bool> {
        match self {
            Self::Persistent(storage) => storage.is_empty(),
            Self::Memory(storage) => storage.is_empty(),
        }
    }

    fn all_ids(&self) -> Result<Vec<VectorId>> {
        match self {
            Self::Persistent(storage) => storage.all_ids(),
            Self::Memory(storage) => storage.all_ids(),
        }
    }

    fn save_config_value(&self, key: &str, value: &str) -> Result<()> {
        match self {
            Self::Persistent(storage) => storage.save_config_value(key, value),
            Self::Memory(storage) => storage.save_config_value(key, value),
        }
    }

    fn load_config_value(&self, key: &str) -> Result<Option<String>> {
        match self {
            Self::Persistent(storage) => storage.load_config_value(key),
            Self::Memory(storage) => storage.load_config_value(key),
        }
    }
}

/// Main vector database
pub struct VectorDB {
    storage: Arc<StorageBackend>,
    index: Arc<RwLock<Box<dyn VectorIndex>>>,
    options: DbOptions,
}

impl VectorDB {
    /// Create a new vector database with the given options
    ///
    /// If a storage path is provided and contains persisted vectors,
    /// the HNSW index will be automatically rebuilt from storage.
    /// If opening an existing database, the stored configuration (dimensions,
    /// distance metric, etc.) will be used instead of the provided options.
    #[allow(unused_mut)] // `options` is mutated only when feature = "storage"
    pub fn new(mut options: DbOptions) -> Result<Self> {
        #[cfg(feature = "storage")]
        let storage = {
            if options.storage_path.starts_with("memory://") {
                Arc::new(StorageBackend::Memory(MemoryStorage::new(
                    options.dimensions,
                )?))
            } else {
                // First, try to load existing configuration from the database
                // We create a temporary storage to check for config
                let temp_storage =
                    crate::storage::VectorStorage::new(&options.storage_path, options.dimensions)?;

                let stored_config = temp_storage.load_config()?;

                if let Some(config) = stored_config {
                    // Existing database - use stored configuration
                    tracing::info!(
                        "Loading existing database with {} dimensions",
                        config.dimensions
                    );
                    options = DbOptions {
                        // Keep the provided storage path (may have changed)
                        storage_path: options.storage_path.clone(),
                        // Use stored configuration for everything else
                        dimensions: config.dimensions,
                        distance_metric: config.distance_metric,
                        hnsw_config: config.hnsw_config,
                        quantization: config.quantization,
                    };
                    // Recreate storage with correct dimensions
                    Arc::new(StorageBackend::Persistent(
                        crate::storage::VectorStorage::new(
                            &options.storage_path,
                            options.dimensions,
                        )?,
                    ))
                } else {
                    // New database - save the configuration
                    tracing::info!(
                        "Creating new database with {} dimensions",
                        options.dimensions
                    );
                    temp_storage.save_config(&options)?;
                    Arc::new(StorageBackend::Persistent(temp_storage))
                }
            }
        };

        #[cfg(not(feature = "storage"))]
        let storage = Arc::new(MemoryStorage::new(options.dimensions)?);

        // Choose index based on configuration and available features.
        // Turbo4 quantization (ADR-296) is applied here: with an HNSW config
        // it swaps the f32 index for one that stores only packed 4-bit codes.
        let turbo4_params = match &options.quantization {
            Some(crate::types::QuantizationConfig::Turbo4 {
                rotation_seed,
                rescore_multiplier,
                policy,
                search_quantization,
            }) => Some((
                *rotation_seed,
                *rescore_multiplier,
                *policy,
                *search_quantization,
            )),
            _ => None,
        };

        #[allow(unused_mut)] // `index` is mutated only when feature = "storage"
        let mut index: Box<dyn VectorIndex> = if let Some(hnsw_config) = &options.hnsw_config {
            #[cfg(feature = "hnsw")]
            {
                if let Some((rotation_seed, rescore_multiplier, policy, search_quantization)) =
                    turbo4_params
                {
                    tracing::info!(
                        "Turbo4 quantization active ({policy:?} policy): {} bytes/vector instead of {}",
                        options.dimensions / 2 + 8,
                        options.dimensions * 4
                    );
                    Box::new(crate::index::turbo4::Turbo4HnswIndex::new(
                        options.dimensions,
                        options.distance_metric,
                        hnsw_config.clone(),
                        rotation_seed,
                        rescore_multiplier,
                        policy,
                        search_quantization,
                    )?) as Box<dyn VectorIndex>
                } else {
                    Box::new(HnswIndex::new(
                        options.dimensions,
                        options.distance_metric,
                        hnsw_config.clone(),
                    )?) as Box<dyn VectorIndex>
                }
            }
            #[cfg(not(feature = "hnsw"))]
            {
                // Fall back to flat index if HNSW is not available
                tracing::warn!("HNSW requested but not available (WASM build), using flat index");
                Box::new(FlatIndex::new(options.dimensions, options.distance_metric))
            }
        } else {
            Box::new(FlatIndex::new(options.dimensions, options.distance_metric))
        };

        // The legacy variants of `DbOptions.quantization` are persisted and
        // restored but not applied to the index or storage representation
        // (issue #563). Warn loudly rather than silently ignoring a requested
        // quantization so callers don't assume a memory reduction that isn't
        // happening. `Turbo4` (ADR-296) IS applied — but only on the HNSW
        // path, so warn if it was requested without one.
        match &options.quantization {
            None | Some(crate::types::QuantizationConfig::None) => {}
            Some(crate::types::QuantizationConfig::Turbo4 { .. }) => {
                if options.hnsw_config.is_none() || cfg!(not(feature = "hnsw")) {
                    tracing::warn!(
                        "QuantizationConfig::Turbo4 requires an HNSW index (hnsw_config set \
                         and the `hnsw` feature enabled); falling back to an unquantized \
                         flat index — no compression is happening."
                    );
                }
            }
            other => {
                tracing::warn!(
                    "DbOptions.quantization = {:?} is set but not yet applied — the \
                     index is stored unquantized (no compression / memory reduction). \
                     See issue #563. Use QuantizationConfig::Turbo4 for applied \
                     quantization (ADR-296).",
                    other
                );
            }
        }

        // ADR-297 §7: persist the collection-level provenance record and
        // validate the codec contract on reopen. Codes encoded under one
        // rotation seed are meaningless under another, and a codec_version
        // bump means tables/layout changed — refuse to open rather than
        // silently serve wrong results (a migration re-encodes; ADR-297
        // phase D wires that path).
        #[cfg(feature = "storage")]
        if let Some((rotation_seed, _, _, _)) = turbo4_params {
            use crate::encoding::{CodecKind, VectorProvenance, CODEC_VERSION};
            const PROVENANCE_KEY: &str = "turbo4_provenance";
            let current = VectorProvenance {
                model_id: None,
                codec: CodecKind::Turbo4,
                codec_version: CODEC_VERSION,
                rotation_seed: Some(rotation_seed),
                dim: options.dimensions,
                metric: options.distance_metric,
                source_hash: None,
                lineage: Vec::new(),
            };
            match storage.load_config_value(PROVENANCE_KEY)? {
                Some(stored_json) => {
                    let stored: VectorProvenance =
                        serde_json::from_str(&stored_json).map_err(|e| {
                            crate::error::RuvectorError::SerializationError(format!(
                                "corrupt turbo4 provenance record: {e}"
                            ))
                        })?;
                    if stored.rotation_seed != current.rotation_seed
                        || stored.dim != current.dim
                        || stored.metric != current.metric
                        || stored.codec_version != current.codec_version
                    {
                        return Err(crate::error::RuvectorError::InvalidParameter(format!(
                            "Turbo4 provenance mismatch: stored {stored:?} vs requested \
                             {current:?}. Codes are bound to (rotation_seed, dim, metric, \
                             codec_version); migrate the collection instead of changing \
                             them in place."
                        )));
                    }
                }
                None => {
                    let json = serde_json::to_string(&current).map_err(|e| {
                        crate::error::RuvectorError::SerializationError(e.to_string())
                    })?;
                    storage.save_config_value(PROVENANCE_KEY, &json)?;
                }
            }
        }

        // Rebuild index from persisted vectors if storage is not empty
        // This fixes the bug where search() returns empty results after restart
        #[cfg(feature = "storage")]
        {
            let stored_ids = storage.all_ids()?;
            if !stored_ids.is_empty() {
                tracing::info!(
                    "Rebuilding index from {} persisted vectors",
                    stored_ids.len()
                );

                // Batch load all vectors for efficient index rebuilding
                let mut entries = Vec::with_capacity(stored_ids.len());
                for id in stored_ids {
                    if let Some(entry) = storage.get(&id)? {
                        entries.push((id, entry.vector));
                    }
                }

                // Add all vectors to index in batch for better performance
                index.add_batch(entries)?;

                tracing::info!("Index rebuilt successfully");
            }
        }

        Ok(Self {
            storage,
            index: Arc::new(RwLock::new(index)),
            options,
        })
    }

    /// Create with default options
    pub fn with_dimensions(dimensions: usize) -> Result<Self> {
        let options = DbOptions {
            dimensions,
            ..DbOptions::default()
        };
        Self::new(options)
    }

    /// Insert a vector entry
    pub fn insert(&self, entry: VectorEntry) -> Result<VectorId> {
        let id = self.storage.insert(&entry)?;

        // Add to index
        let mut index = self.index.write();
        index.add(id.clone(), entry.vector)?;

        Ok(id)
    }

    /// Insert multiple vectors in a batch
    pub fn insert_batch(&self, entries: impl AsRef<[VectorEntry]>) -> Result<Vec<VectorId>> {
        let entries = entries.as_ref();
        let ids = self.storage.insert_batch(entries)?;

        // Add to index
        let mut index = self.index.write();
        let index_entries: Vec<_> = ids
            .iter()
            .zip(entries.iter())
            .map(|(id, entry)| (id.clone(), entry.vector.clone()))
            .collect();

        index.add_batch(index_entries)?;

        Ok(ids)
    }

    /// Exact filtered retrieval for the flat backend and completeness fallback.
    /// Filters before distance calculation, so selective queries avoid scoring
    /// ineligible vectors and always return the eligible top-k.
    fn search_filtered_exact(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let Some(filter) = query.filter.as_ref() else {
            unreachable!("exact filtered retrieval requires a predicate");
        };
        if query.k == 0 {
            return Ok(Vec::new());
        }
        if query.vector.len() != self.options.dimensions {
            return Err(crate::error::RuvectorError::DimensionMismatch {
                expected: self.options.dimensions,
                actual: query.vector.len(),
            });
        }
        #[cfg(feature = "storage")]
        if let StorageBackend::Memory(storage) = self.storage.as_ref() {
            return storage.search_filtered_exact(query, self.options.distance_metric);
        }
        #[cfg(not(feature = "storage"))]
        {
            return self
                .storage
                .search_filtered_exact(query, self.options.distance_metric);
        }

        let mut matches = Vec::new();
        for id in self.storage.all_ids()? {
            let Some(entry) = self.storage.get(&id)? else {
                continue;
            };
            let Some(metadata) = &entry.metadata else {
                continue;
            };
            if !filter
                .iter()
                .all(|(key, value)| metadata.get(key) == Some(value))
            {
                continue;
            }
            let score = crate::distance::distance(
                &query.vector,
                &entry.vector,
                self.options.distance_metric,
            )?;
            matches.push(SearchResult {
                id,
                score,
                vector: Some(entry.vector),
                metadata: entry.metadata,
            });
        }
        matches.sort_unstable_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)));
        matches.truncate(query.k);
        Ok(matches)
    }

    /// Search for similar vectors
    pub fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        // Exact flat retrieval can prefilter before scoring; this avoids both
        // top-k underfill and redundant distance work on ineligible vectors.
        if query.filter.is_some() && (self.options.hnsw_config.is_none() || !cfg!(feature = "hnsw"))
        {
            return self.search_filtered_exact(&query);
        }
        if query.k == 0 {
            return Ok(Vec::new());
        }
        let index = self.index.read();
        // Verification tier (ADR-297 §2): with Turbo4 quantization, over-fetch
        // 2k candidates and re-rank them against the stored f32 vectors.
        // Ablation (20k×768-D clustered): this recovers the precision-bound
        // tail ranks that wider quantized traversal cannot — recall@10 went
        // from −7 pp vs the f32 index to +1.4 pp, at ~f32 latency — and it is
        // nearly free here because result enrichment fetches the stored
        // vectors anyway.
        let verify = matches!(
            self.options.quantization,
            Some(crate::types::QuantizationConfig::Turbo4 { .. })
        );
        let total = index.len();
        let mut fetch_k = if query.filter.is_some() {
            query.k.saturating_mul(8).max(16).min(total)
        } else if verify {
            query.k.saturating_mul(2)
        } else {
            query.k
        };
        loop {
            let mut results = index.search_ef(&query.vector, fetch_k, query.ef_search)?;

            // Enrich results with full data if needed
            for result in &mut results {
                if let Ok(Some(entry)) = self.storage.get(&result.id) {
                    result.vector = Some(entry.vector);
                    result.metadata = entry.metadata;
                }
            }

            if verify {
                for r in &mut results {
                    if let Some(v) = &r.vector {
                        r.score = crate::encoding::metric_distance(
                            self.options.distance_metric,
                            &query.vector,
                            v,
                        );
                    }
                }
                results.sort_unstable_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            // Apply metadata filters if specified
            if let Some(filter) = &query.filter {
                results.retain(|r| {
                    if let Some(metadata) = &r.metadata {
                        filter
                            .iter()
                            .all(|(key, value)| metadata.get(key).is_some_and(|v| v == value))
                    } else {
                        false
                    }
                });
            }

            if query.filter.is_some() && results.len() < query.k {
                if fetch_k < total {
                    fetch_k = fetch_k
                        .saturating_mul(2)
                        .max(fetch_k.saturating_add(1))
                        .min(total);
                    continue;
                }
                // HNSW can visit fewer than all records even with k == index.len().
                // Exhaustive predicate-first fallback ensures eligible completeness.
                return self.search_filtered_exact(&query);
            }
            results.truncate(query.k);
            return Ok(results);
        }
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let deleted_storage = self.storage.delete(id)?;

        if deleted_storage {
            let mut index = self.index.write();
            let _ = index.remove(&id.to_string())?;
        }

        Ok(deleted_storage)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        self.storage.get(id)
    }

    /// Get the number of vectors
    pub fn len(&self) -> Result<usize> {
        self.storage.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> Result<bool> {
        self.storage.is_empty()
    }

    /// Get database options
    pub fn options(&self) -> &DbOptions {
        &self.options
    }

    /// Persist a provenance value in the vector store itself, so it cannot be
    /// removed without removing the vectors it describes.
    #[cfg(feature = "storage")]
    pub fn save_config_value(&self, key: &str, value: &str) -> Result<()> {
        self.storage.save_config_value(key, value)
    }

    /// Read a value written by [`save_config_value`](Self::save_config_value).
    #[cfg(feature = "storage")]
    pub fn load_config_value(&self, key: &str) -> Result<Option<String>> {
        self.storage.load_config_value(key)
    }

    /// Get all vector IDs (for iteration/serialization)
    pub fn keys(&self) -> Result<Vec<String>> {
        self.storage.all_ids()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "storage")]
    #[test]
    fn memory_url_selects_memory_backend_when_storage_is_compiled() {
        let options = DbOptions {
            dimensions: 3,
            distance_metric: DistanceMetric::Euclidean,
            storage_path: "memory://feature-unification-regression".to_string(),
            hnsw_config: None,
            quantization: None,
        };
        let db = VectorDB::new(options).unwrap();
        db.insert(VectorEntry {
            id: Some("one".to_string()),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        })
        .unwrap();
        assert_eq!(db.len().unwrap(), 1);
        assert!(!std::path::Path::new("memory:").exists());
    }
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn test_vector_db_creation() -> Result<()> {
        let dir = tempdir().unwrap();
        let mut options = DbOptions::default();
        options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
        options.dimensions = 3;

        let db = VectorDB::new(options)?;
        assert!(db.is_empty()?);

        Ok(())
    }

    #[test]
    fn test_insert_and_search() -> Result<()> {
        let dir = tempdir().unwrap();
        let mut options = DbOptions::default();
        options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
        options.dimensions = 3;
        options.distance_metric = DistanceMetric::Euclidean; // Use Euclidean for clearer test
        options.hnsw_config = None; // Use flat index for testing

        let db = VectorDB::new(options)?;

        // Insert vectors
        db.insert(VectorEntry {
            id: Some("v1".to_string()),
            vector: vec![1.0, 0.0, 0.0],
            metadata: None,
        })?;

        db.insert(VectorEntry {
            id: Some("v2".to_string()),
            vector: vec![0.0, 1.0, 0.0],
            metadata: None,
        })?;

        db.insert(VectorEntry {
            id: Some("v3".to_string()),
            vector: vec![0.0, 0.0, 1.0],
            metadata: None,
        })?;

        // Search for exact match
        let results = db.search(SearchQuery {
            vector: vec![1.0, 0.0, 0.0],
            k: 2,
            filter: None,
            ef_search: None,
        })?;

        assert!(results.len() >= 1);
        assert_eq!(results[0].id, "v1", "First result should be exact match");
        assert!(
            results[0].score < 0.01,
            "Exact match should have ~0 distance"
        );

        Ok(())
    }

    #[test]
    fn test_filtered_top_k_refills_from_eligible_candidates() -> Result<()> {
        let dir = tempdir().unwrap();
        let mut options = DbOptions::default();
        options.storage_path = dir.path().join("filtered.db").to_string_lossy().to_string();
        options.dimensions = 2;
        options.distance_metric = DistanceMetric::Euclidean;
        options.hnsw_config = None;
        let db = VectorDB::new(options)?;
        for i in 0..100 {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert(
                "tenant".to_owned(),
                serde_json::json!(if i >= 90 { "a" } else { "b" }),
            );
            db.insert(VectorEntry {
                id: Some(format!("v{i:03}")),
                vector: vec![i as f32, 0.0],
                metadata: Some(metadata),
            })?;
        }
        let mut filter = std::collections::HashMap::new();
        filter.insert("tenant".to_owned(), serde_json::json!("a"));
        let results = db.search(SearchQuery {
            vector: vec![0.0, 0.0],
            k: 10,
            filter: Some(filter),
            ef_search: None,
        })?;
        assert_eq!(
            results.len(),
            10,
            "filtered search must refill to eligible top-k"
        );
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.id, format!("v{:03}", i + 90));
            assert_eq!(r.score, (i + 90) as f32);
            assert_eq!(r.metadata.as_ref().unwrap()["tenant"], "a");
        }
        Ok(())
    }

    #[test]
    fn test_filtered_sparse_and_empty_eligible() -> Result<()> {
        let dir = tempdir().unwrap();
        let mut options = DbOptions::default();
        options.storage_path = dir
            .path()
            .join("sparse-filter.db")
            .to_string_lossy()
            .to_string();
        options.dimensions = 2;
        options.distance_metric = DistanceMetric::Euclidean;
        options.hnsw_config = None;
        let db = VectorDB::new(options)?;
        for i in 0..30 {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert(
                "tenant".to_string(),
                serde_json::json!(if i >= 27 { "a" } else { "b" }),
            );
            db.insert(VectorEntry {
                id: Some(format!("v{i}")),
                vector: vec![i as f32, 0.0],
                metadata: Some(metadata),
            })?;
        }
        let query_for = |tenant: &str, k| {
            let mut filter = std::collections::HashMap::new();
            filter.insert("tenant".to_string(), serde_json::json!(tenant));
            SearchQuery {
                vector: vec![0.0, 0.0],
                k,
                filter: Some(filter),
                ef_search: None,
            }
        };
        let results = db.search(query_for("a", 10))?;
        assert_eq!(results.len(), 3);
        assert_eq!(
            results.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["v27", "v28", "v29"]
        );
        assert!(db.search(query_for("missing", 10))?.is_empty());
        assert!(db.search(query_for("a", 0))?.is_empty());
        Ok(())
    }

    #[cfg(feature = "hnsw")]
    #[test]
    fn test_hnsw_filtered_top_k_completeness() -> Result<()> {
        let dir = tempdir().unwrap();
        let mut options = DbOptions::default();
        options.storage_path = dir
            .path()
            .join("filtered-hnsw.db")
            .to_string_lossy()
            .to_string();
        options.dimensions = 2;
        options.distance_metric = DistanceMetric::Euclidean;
        options.hnsw_config = Some(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 20,
            max_elements: 1000,
        });
        let db = VectorDB::new(options)?;
        for i in 0..100 {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert(
                "tenant".to_owned(),
                serde_json::json!(if i >= 90 { "a" } else { "b" }),
            );
            db.insert(VectorEntry {
                id: Some(format!("v{i:03}")),
                vector: vec![i as f32, 0.0],
                metadata: Some(metadata),
            })?;
        }
        let mut filter = std::collections::HashMap::new();
        filter.insert("tenant".to_string(), serde_json::json!("a"));
        let results = db.search(SearchQuery {
            vector: vec![0.0, 0.0],
            k: 10,
            filter: Some(filter),
            ef_search: Some(20),
        })?;
        assert_eq!(results.len(), 10);
        let ids: std::collections::HashSet<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids.len(), 10);
        assert!(results
            .iter()
            .all(|r| r.metadata.as_ref().unwrap()["tenant"] == "a"));
        let mut missing = std::collections::HashMap::new();
        missing.insert("tenant".to_owned(), serde_json::json!("missing"));
        assert!(db
            .search(SearchQuery {
                vector: vec![0.0, 0.0],
                k: 10,
                filter: Some(missing),
                ef_search: Some(20),
            })?
            .is_empty());
        Ok(())
    }

    /// Opt-in deterministic smoke benchmark: run with
    /// cargo test -p ruvector-core --lib bench_filtered_contract -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_filtered_contract() -> Result<()> {
        use std::time::Instant;
        let mut options = DbOptions::default();
        options.storage_path = "memory://filtered-contract-bench".to_string();
        options.dimensions = 384;
        options.distance_metric = DistanceMetric::Euclidean;
        options.hnsw_config = None;
        let db = VectorDB::new(options)?;
        for i in 0..10_000usize {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert(
                "ten".to_owned(),
                serde_json::json!(if i % 10 == 0 { "yes" } else { "no" }),
            );
            metadata.insert(
                "hundred".to_owned(),
                serde_json::json!(if i % 100 == 0 { "yes" } else { "no" }),
            );
            let mut v = vec![0.0; 384];
            for (j, x) in v.iter_mut().enumerate() {
                *x = ((i.wrapping_mul(17) + j * 31) % 10007) as f32 / 10007.0;
            }
            db.insert(VectorEntry {
                id: Some(format!("v{i}")),
                vector: v,
                metadata: Some(metadata),
            })?;
        }
        for field in ["ten", "hundred"] {
            let mut elapsed = Vec::new();
            let mut complete = 0;
            for i in 0..40usize {
                let mut filter = std::collections::HashMap::new();
                filter.insert(field.to_owned(), serde_json::json!("yes"));
                let query = vec![(i * 113 % 997) as f32 / 997.0; 384];
                let start = Instant::now();
                let results = db.search(SearchQuery {
                    vector: query,
                    k: 10,
                    filter: Some(filter),
                    ef_search: None,
                })?;
                elapsed.push(start.elapsed().as_micros() as u64);
                complete += usize::from(results.len() == 10);
            }
            elapsed.sort_unstable();
            println!(
                "BENCH field={} complete={}/40 p50_us={} p95_us={}",
                field, complete, elapsed[20], elapsed[38]
            );
        }
        Ok(())
    }

    /// Turbo4 quantization end-to-end: insert, search, restart-rebuild (ADR-296).
    #[test]
    #[cfg(all(feature = "hnsw", feature = "storage"))]
    fn test_turbo4_quantized_db() -> Result<()> {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("turbo4.db").to_string_lossy().to_string();
        let dim = 64usize;

        let mk_options = || {
            let mut options = DbOptions::default();
            options.storage_path = db_path.clone();
            options.dimensions = dim;
            options.distance_metric = DistanceMetric::Euclidean;
            options.quantization = Some(QuantizationConfig::Turbo4 {
                rotation_seed: 42,
                rescore_multiplier: 4,
                policy: SearchPolicy::Balanced,
                search_quantization: SearchQuantization::default(),
            });
            options
        };

        // Deterministic, pairwise-distinct pseudo-random vectors (SplitMix64
        // hash of (i, j) — a modular pattern here can silently produce
        // identical vectors, making "self is nearest" ambiguous).
        let vectors: Vec<Vec<f32>> = (0..80u64)
            .map(|i| {
                (0..dim as u64)
                    .map(|j| {
                        let mut z = (i << 32 | j).wrapping_add(0x9E3779B97F4A7C15);
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                        ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect();

        {
            let db = VectorDB::new(mk_options())?;
            for (i, v) in vectors.iter().enumerate() {
                db.insert(VectorEntry {
                    id: Some(format!("v{i}")),
                    vector: v.clone(),
                    metadata: None,
                })?;
            }
            let results = db.search(SearchQuery {
                vector: vectors[5].clone(),
                k: 3,
                filter: None,
                ef_search: None,
            })?;
            assert_eq!(results[0].id, "v5", "self-query must return itself first");
        }

        // Restart: config (incl. quantization) restored from storage, index
        // rebuilt by re-encoding persisted vectors.
        {
            let db = VectorDB::new(mk_options())?;
            assert!(matches!(
                db.options().quantization,
                Some(QuantizationConfig::Turbo4 {
                    rotation_seed: 42,
                    ..
                })
            ));
            let results = db.search(SearchQuery {
                vector: vectors[12].clone(),
                k: 3,
                filter: None,
                ef_search: None,
            })?;
            assert_eq!(results[0].id, "v12", "self-query after restart");
        }
        Ok(())
    }

    /// Provenance record (ADR-297 §7): written on create, readable, and a
    /// tampered rotation seed refuses to open.
    #[test]
    #[cfg(all(feature = "hnsw", feature = "storage"))]
    fn test_turbo4_provenance_guard() -> Result<()> {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("prov.db").to_string_lossy().to_string();
        let mk_options = || {
            let mut options = DbOptions::default();
            options.storage_path = db_path.clone();
            options.dimensions = 64;
            options.distance_metric = DistanceMetric::Euclidean;
            options.quantization = Some(QuantizationConfig::Turbo4 {
                rotation_seed: 42,
                rescore_multiplier: 4,
                policy: SearchPolicy::Balanced,
                search_quantization: SearchQuantization::default(),
            });
            options
        };

        {
            let db = VectorDB::new(mk_options())?;
            let json = db
                .load_config_value("turbo4_provenance")?
                .expect("provenance record must be written when Turbo4 is active");
            let prov: crate::encoding::VectorProvenance = serde_json::from_str(&json).unwrap();
            assert_eq!(prov.rotation_seed, Some(42));
            assert_eq!(prov.dim, 64);
            assert_eq!(prov.codec_version, crate::encoding::CODEC_VERSION);
        }

        // Clean reopen with the same parameters succeeds.
        {
            let db = VectorDB::new(mk_options())?;
            // Tamper: claim the codes were made with a different seed.
            let mut prov: crate::encoding::VectorProvenance =
                serde_json::from_str(&db.load_config_value("turbo4_provenance")?.unwrap()).unwrap();
            prov.rotation_seed = Some(7);
            db.save_config_value("turbo4_provenance", &serde_json::to_string(&prov).unwrap())?;
        }

        // Reopen must now refuse: stored contract disagrees with the seed
        // the (restored) config asks for.
        let err = VectorDB::new(mk_options());
        assert!(
            err.is_err(),
            "reopen with mismatched provenance must fail, got Ok"
        );
        Ok(())
    }

    /// Test that search works after simulated restart (new VectorDB instance)
    /// This verifies the fix for issue #30: HNSW index not rebuilt from storage
    #[test]
    #[cfg(feature = "storage")]
    fn test_search_after_restart() -> Result<()> {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("persist.db").to_string_lossy().to_string();

        // Phase 1: Create database and insert vectors
        {
            let mut options = DbOptions::default();
            options.storage_path = db_path.clone();
            options.dimensions = 3;
            options.distance_metric = DistanceMetric::Euclidean;
            options.hnsw_config = None;

            let db = VectorDB::new(options)?;

            db.insert(VectorEntry {
                id: Some("v1".to_string()),
                vector: vec![1.0, 0.0, 0.0],
                metadata: None,
            })?;

            db.insert(VectorEntry {
                id: Some("v2".to_string()),
                vector: vec![0.0, 1.0, 0.0],
                metadata: None,
            })?;

            db.insert(VectorEntry {
                id: Some("v3".to_string()),
                vector: vec![0.7, 0.7, 0.0],
                metadata: None,
            })?;

            // Verify search works before "restart"
            let results = db.search(SearchQuery {
                vector: vec![0.8, 0.6, 0.0],
                k: 3,
                filter: None,
                ef_search: None,
            })?;
            assert_eq!(results.len(), 3, "Should find all 3 vectors before restart");
        }
        // db is dropped here, simulating application shutdown

        // Phase 2: Create new database instance (simulates restart)
        {
            let mut options = DbOptions::default();
            options.storage_path = db_path.clone();
            options.dimensions = 3;
            options.distance_metric = DistanceMetric::Euclidean;
            options.hnsw_config = None;

            let db = VectorDB::new(options)?;

            // Verify vectors are still accessible
            assert_eq!(db.len()?, 3, "Should have 3 vectors after restart");

            // Verify get() works
            let v1 = db.get("v1")?;
            assert!(v1.is_some(), "get() should work after restart");

            // Verify search() works - THIS WAS THE BUG
            let results = db.search(SearchQuery {
                vector: vec![0.8, 0.6, 0.0],
                k: 3,
                filter: None,
                ef_search: None,
            })?;

            assert_eq!(
                results.len(),
                3,
                "search() should return results after restart (was returning 0 before fix)"
            );

            // v3 should be closest to query [0.8, 0.6, 0.0]
            assert_eq!(
                results[0].id, "v3",
                "v3 [0.7, 0.7, 0.0] should be closest to query [0.8, 0.6, 0.0]"
            );
        }

        Ok(())
    }
}
