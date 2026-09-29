//! Node.js bindings for Ruvector via NAPI-RS
//!
//! High-performance Rust vector database with zero-copy buffer sharing,
//! async/await support, and complete TypeScript type definitions.

#![allow(clippy::all)]
#![allow(clippy::pedantic)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_core::{
    types::{DbOptions, HnswConfig, QuantizationConfig},
    ConfigCheck, DistanceMetric, SearchQuery, SearchResult, VectorDB as CoreVectorDB, VectorEntry,
};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

// Import new crates
use ruvector_collections::CollectionManager as CoreCollectionManager;
use ruvector_filter::FilterExpression;
use ruvector_metrics::{gather_metrics, HealthChecker, HealthStatus};
use std::path::PathBuf;

/// Distance metric for similarity calculation
#[napi(string_enum)]
#[derive(Debug)]
pub enum JsDistanceMetric {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine similarity (converted to distance)
    Cosine,
    /// Dot product (converted to distance for maximization)
    DotProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

impl From<JsDistanceMetric> for DistanceMetric {
    fn from(metric: JsDistanceMetric) -> Self {
        match metric {
            JsDistanceMetric::Euclidean => DistanceMetric::Euclidean,
            JsDistanceMetric::Cosine => DistanceMetric::Cosine,
            JsDistanceMetric::DotProduct => DistanceMetric::DotProduct,
            JsDistanceMetric::Manhattan => DistanceMetric::Manhattan,
        }
    }
}

/// Quantization configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsQuantizationConfig {
    /// Quantization type: "none", "scalar", "product", "binary"
    pub r#type: String,
    /// Number of subspaces (for product quantization)
    pub subspaces: Option<u32>,
    /// Codebook size (for product quantization)
    pub k: Option<u32>,
}

impl From<JsQuantizationConfig> for QuantizationConfig {
    fn from(config: JsQuantizationConfig) -> Self {
        match config.r#type.as_str() {
            "none" => QuantizationConfig::None,
            "scalar" => QuantizationConfig::Scalar,
            "product" => QuantizationConfig::Product {
                subspaces: config.subspaces.unwrap_or(16) as usize,
                k: config.k.unwrap_or(256) as usize,
            },
            "binary" => QuantizationConfig::Binary,
            // Turbo4 (ADR-296): applied 4-bit quantization. Seed/multiplier
            // use the core defaults; a full JS surface can extend this later.
            "turbo4" => QuantizationConfig::Turbo4 {
                rotation_seed: ruvector_core::types::default_turbo4_rotation_seed(),
                rescore_multiplier: ruvector_core::types::default_turbo4_rescore_multiplier(),
                policy: ruvector_core::types::SearchPolicy::default(),
                search_quantization: ruvector_core::types::SearchQuantization::default(),
            },
            _ => QuantizationConfig::Scalar,
        }
    }
}

/// HNSW index configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsHnswConfig {
    /// Number of connections per layer (M)
    pub m: Option<u32>,
    /// Size of dynamic candidate list during construction
    pub ef_construction: Option<u32>,
    /// Size of dynamic candidate list during search
    pub ef_search: Option<u32>,
    /// Maximum number of elements
    pub max_elements: Option<u32>,
}

impl From<JsHnswConfig> for HnswConfig {
    fn from(config: JsHnswConfig) -> Self {
        HnswConfig {
            m: config.m.unwrap_or(32) as usize,
            ef_construction: config.ef_construction.unwrap_or(200) as usize,
            ef_search: config.ef_search.unwrap_or(100) as usize,
            max_elements: config.max_elements.unwrap_or(10_000_000) as usize,
        }
    }
}

/// Database configuration options
#[napi(object)]
#[derive(Debug)]
pub struct JsDbOptions {
    /// Vector dimensions
    pub dimensions: u32,
    /// Distance metric.
    ///
    /// Omitted: `Cosine` for a new database; an existing database at
    /// `storagePath` keeps its stored metric. Passed: an existing database's
    /// stored metric must equal it, otherwise the constructor throws.
    pub distance_metric: Option<JsDistanceMetric>,
    /// Where the database is persisted.
    ///
    /// - Omitted: an in-memory database private to this instance. Nothing is
    ///   written to disk and nothing is shared with other `VectorDB`
    ///   instances; the data is gone when the instance is garbage-collected.
    ///   (Before issue #1063 this opened a shared `./ruvector.db`; pass
    ///   `storagePath: './ruvector.db'` to keep using such a store.)
    /// - A file path: a persistent database at that path. If a database
    ///   already exists there, its stored `dimensions` (and `distanceMetric`,
    ///   when one is passed) must equal the requested ones, otherwise the
    ///   constructor throws an error naming the path and both values.
    /// - A value starting with `memory://`: explicitly in-memory, same as
    ///   omitting it.
    pub storage_path: Option<String>,
    /// HNSW configuration
    pub hnsw_config: Option<JsHnswConfig>,
    /// Quantization configuration
    pub quantization: Option<JsQuantizationConfig>,
}

/// Storage path used when `storagePath` is omitted: ruvector-core's
/// per-instance in-memory backend (no file is created).
const IN_MEMORY_STORAGE_PATH: &str = "memory://";

/// The file an omitted `storagePath` opened before issue #1063.
const LEGACY_DEFAULT_STORAGE_PATH: &str = "./ruvector.db";

static LEGACY_DEFAULT_STORE_WARNING: std::sync::Once = std::sync::Once::new();

/// An omitted `storagePath` used to mean `./ruvector.db`. If that file exists,
/// an upgrading caller is probably relying on it and would otherwise lose
/// writes to an in-memory store without noticing, so say so once per process.
fn warn_if_legacy_default_store_exists() {
    if std::path::Path::new(LEGACY_DEFAULT_STORAGE_PATH).exists() {
        LEGACY_DEFAULT_STORE_WARNING.call_once(|| {
            eprintln!(
                "ruvector: storagePath was omitted, so this VectorDB is in-memory and its data \
                 is not saved. A {LEGACY_DEFAULT_STORAGE_PATH} exists in the working directory; \
                 older versions used it by default. Pass storagePath: '{LEGACY_DEFAULT_STORAGE_PATH}' \
                 to keep using it (see issue #1063)."
            );
        });
    }
}

impl From<JsDbOptions> for DbOptions {
    fn from(options: JsDbOptions) -> Self {
        DbOptions {
            dimensions: options.dimensions as usize,
            distance_metric: options
                .distance_metric
                .map(Into::into)
                .unwrap_or(DistanceMetric::Cosine),
            // Issue #1063: no storagePath means in-memory. The previous default,
            // a shared `./ruvector.db` in the working directory, made every
            // instance share one store and let its stored dimension override
            // the requested one.
            storage_path: options
                .storage_path
                .unwrap_or_else(|| IN_MEMORY_STORAGE_PATH.to_string()),
            hnsw_config: options.hnsw_config.map(Into::into),
            quantization: options.quantization.map(Into::into),
        }
    }
}

/// Vector entry
#[napi(object)]
pub struct JsVectorEntry {
    /// Optional ID (auto-generated if not provided)
    pub id: Option<String>,
    /// Vector data as Float32Array or array of numbers
    pub vector: Float32Array,
    /// Optional metadata as JSON string (use JSON.stringify on objects)
    pub metadata: Option<String>,
}

impl JsVectorEntry {
    fn to_core(&self) -> Result<VectorEntry> {
        // Parse JSON string to HashMap<String, serde_json::Value>
        let metadata = self.metadata.as_ref().and_then(|s| {
            serde_json::from_str::<std::collections::HashMap<String, serde_json::Value>>(s).ok()
        });

        Ok(VectorEntry {
            id: self.id.clone(),
            vector: self.vector.to_vec(),
            metadata,
        })
    }
}

/// Search query parameters
#[napi(object)]
pub struct JsSearchQuery {
    /// Query vector as Float32Array or array of numbers
    pub vector: Float32Array,
    /// Number of results to return (top-k)
    pub k: u32,
    /// Optional ef_search parameter for HNSW
    pub ef_search: Option<u32>,
    /// Optional metadata filter as JSON string (use JSON.stringify on objects)
    pub filter: Option<String>,
}

impl JsSearchQuery {
    fn to_core(&self) -> Result<SearchQuery> {
        // Parse JSON string to HashMap<String, serde_json::Value>
        let filter = self.filter.as_ref().and_then(|s| {
            serde_json::from_str::<std::collections::HashMap<String, serde_json::Value>>(s).ok()
        });

        Ok(SearchQuery {
            vector: self.vector.to_vec(),
            k: self.k as usize,
            filter,
            ef_search: self.ef_search.map(|v| v as usize),
        })
    }
}

/// Search result with similarity score
#[napi(object)]
#[derive(Clone)]
pub struct JsSearchResult {
    /// Vector ID
    pub id: String,
    /// Distance/similarity score (lower is better for distance metrics)
    pub score: f64,
    /// Vector data (if requested)
    pub vector: Option<Float32Array>,
    /// Metadata as JSON string (use JSON.parse to convert to object)
    pub metadata: Option<String>,
}

impl From<SearchResult> for JsSearchResult {
    fn from(result: SearchResult) -> Self {
        // Convert Vec<f32> to Float32Array
        let vector = result.vector.map(|v| Float32Array::new(v));

        // Convert HashMap to JSON string
        let metadata = result.metadata.and_then(|m| serde_json::to_string(&m).ok());

        JsSearchResult {
            id: result.id,
            score: f64::from(result.score),
            vector,
            metadata,
        }
    }
}

/// High-performance vector database with HNSW indexing
#[napi]
pub struct VectorDB {
    inner: Arc<RwLock<CoreVectorDB>>,
}

#[napi]
impl VectorDB {
    /// Create a new vector database with the given options
    ///
    /// Without `storagePath` the database is in-memory and private to this
    /// instance. With `storagePath`, an existing database at that path must
    /// have the same `dimensions` (and `distanceMetric`, when one is passed),
    /// or this throws.
    ///
    /// # Example
    /// ```javascript
    /// const db = new VectorDB({
    ///   dimensions: 384,
    ///   distanceMetric: 'Cosine',
    ///   storagePath: './vectors.db',
    ///   hnswConfig: {
    ///     m: 32,
    ///     efConstruction: 200,
    ///     efSearch: 100
    ///   }
    /// });
    /// ```
    #[napi(constructor)]
    pub fn new(options: JsDbOptions) -> Result<Self> {
        if options.storage_path.is_none() {
            warn_if_legacy_default_store_exists();
        }
        // Only a metric the caller actually passed is checked against a
        // stored database; an omitted one adopts the stored metric, so
        // reopening a non-Cosine store without restating it keeps working.
        let check = ConfigCheck {
            dimensions: true,
            distance_metric: options.distance_metric.is_some(),
        };
        let core_options: DbOptions = options.into();
        let db = CoreVectorDB::open_checked(core_options, check)
            .map_err(|e| Error::from_reason(format!("Failed to create database: {}", e)))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Create an in-memory vector database with default options
    ///
    /// # Example
    /// ```javascript
    /// const db = VectorDB.withDimensions(384);
    /// ```
    #[napi(factory)]
    pub fn with_dimensions(dimensions: u32) -> Result<Self> {
        warn_if_legacy_default_store_exists();
        let options = DbOptions {
            dimensions: dimensions as usize,
            storage_path: IN_MEMORY_STORAGE_PATH.to_string(),
            ..DbOptions::default()
        };
        let db = CoreVectorDB::new(options)
            .map_err(|e| Error::from_reason(format!("Failed to create database: {}", e)))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Insert a vector entry into the database
    ///
    /// Returns the ID of the inserted vector (auto-generated if not provided)
    ///
    /// # Example
    /// ```javascript
    /// const id = await db.insert({
    ///   vector: new Float32Array([1.0, 2.0, 3.0]),
    ///   metadata: { text: 'example' }
    /// });
    /// ```
    #[napi]
    pub async fn insert(&self, entry: JsVectorEntry) -> Result<String> {
        let core_entry = entry.to_core()?;
        let db = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.insert(core_entry)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Insert failed: {}", e)))
    }

    /// Insert multiple vectors in a batch
    ///
    /// Returns an array of IDs for the inserted vectors
    ///
    /// # Example
    /// ```javascript
    /// const ids = await db.insertBatch([
    ///   { vector: new Float32Array([1, 2, 3]) },
    ///   { vector: new Float32Array([4, 5, 6]) }
    /// ]);
    /// ```
    #[napi]
    pub async fn insert_batch(&self, entries: Vec<JsVectorEntry>) -> Result<Vec<String>> {
        let core_entries: Result<Vec<VectorEntry>> = entries.iter().map(|e| e.to_core()).collect();
        let core_entries = core_entries?;
        let db = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.insert_batch(core_entries)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Batch insert failed: {}", e)))
    }

    /// Search for similar vectors
    ///
    /// Returns an array of search results sorted by similarity
    ///
    /// # Example
    /// ```javascript
    /// const results = await db.search({
    ///   vector: new Float32Array([1, 2, 3]),
    ///   k: 10,
    ///   filter: { category: 'example' }
    /// });
    /// ```
    #[napi]
    pub async fn search(&self, query: JsSearchQuery) -> Result<Vec<JsSearchResult>> {
        let core_query = query.to_core()?;
        let db = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.search(core_query)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Search failed: {}", e)))
        .map(|results| results.into_iter().map(Into::into).collect())
    }

    /// Delete a vector by ID
    ///
    /// Returns true if the vector was deleted, false if not found
    ///
    /// # Example
    /// ```javascript
    /// const deleted = await db.delete('vector-id');
    /// ```
    #[napi]
    pub async fn delete(&self, id: String) -> Result<bool> {
        let db = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.delete(&id)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Delete failed: {}", e)))
    }

    /// Get a vector by ID
    ///
    /// Returns the vector entry if found, null otherwise
    ///
    /// # Example
    /// ```javascript
    /// const entry = await db.get('vector-id');
    /// if (entry) {
    ///   console.log('Found:', entry.metadata);
    /// }
    /// ```
    #[napi]
    pub async fn get(&self, id: String) -> Result<Option<JsVectorEntry>> {
        let db = self.inner.clone();

        let result = tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.get(&id)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Get failed: {}", e)))?;

        Ok(result.map(|entry| {
            // Convert HashMap to JSON string
            let metadata = entry.metadata.and_then(|m| serde_json::to_string(&m).ok());

            JsVectorEntry {
                id: entry.id,
                vector: Float32Array::new(entry.vector),
                metadata,
            }
        }))
    }

    /// Get the number of vectors in the database
    ///
    /// # Example
    /// ```javascript
    /// const count = await db.len();
    /// console.log(`Database contains ${count} vectors`);
    /// ```
    #[napi]
    pub async fn len(&self) -> Result<u32> {
        let db = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.len()
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Len failed: {}", e)))
        .map(|len| len as u32)
    }

    /// Check if the database is empty
    ///
    /// # Example
    /// ```javascript
    /// if (await db.isEmpty()) {
    ///   console.log('Database is empty');
    /// }
    /// ```
    #[napi]
    pub async fn is_empty(&self) -> Result<bool> {
        let db = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.is_empty()
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("IsEmpty failed: {}", e)))
    }
}

/// Get the version of the Ruvector library
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Test function to verify the bindings are working
#[napi]
pub fn hello() -> String {
    "Hello from Ruvector Node.js bindings!".to_string()
}

/// Filter for metadata-based search
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsFilter {
    /// Field name to filter on
    pub field: String,
    /// Operator: "eq", "ne", "gt", "gte", "lt", "lte", "in", "match"
    pub operator: String,
    /// Value to compare against (JSON string)
    pub value: String,
}

impl JsFilter {
    fn to_filter_expression(&self) -> Result<FilterExpression> {
        let value: serde_json::Value = serde_json::from_str(&self.value)
            .map_err(|e| Error::from_reason(format!("Invalid JSON value: {}", e)))?;

        Ok(match self.operator.as_str() {
            "eq" => FilterExpression::eq(&self.field, value),
            "ne" => FilterExpression::ne(&self.field, value),
            "gt" => FilterExpression::gt(&self.field, value),
            "gte" => FilterExpression::gte(&self.field, value),
            "lt" => FilterExpression::lt(&self.field, value),
            "lte" => FilterExpression::lte(&self.field, value),
            "match" => FilterExpression::Match {
                field: self.field.clone(),
                text: value.as_str().unwrap_or("").to_string(),
            },
            _ => FilterExpression::eq(&self.field, value),
        })
    }
}

/// Collection configuration
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsCollectionConfig {
    /// Vector dimensions
    pub dimensions: u32,
    /// Distance metric
    pub distance_metric: Option<JsDistanceMetric>,
    /// HNSW configuration
    pub hnsw_config: Option<JsHnswConfig>,
    /// Quantization configuration
    pub quantization: Option<JsQuantizationConfig>,
}

impl From<JsCollectionConfig> for ruvector_collections::CollectionConfig {
    fn from(config: JsCollectionConfig) -> Self {
        ruvector_collections::CollectionConfig {
            dimensions: config.dimensions as usize,
            distance_metric: config
                .distance_metric
                .map(Into::into)
                .unwrap_or(DistanceMetric::Cosine),
            hnsw_config: config.hnsw_config.map(Into::into),
            quantization: config.quantization.map(Into::into),
            on_disk_payload: true,
        }
    }
}

/// Collection statistics
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsCollectionStats {
    /// Number of vectors in the collection
    pub vectors_count: u32,
    /// Disk space used in bytes
    pub disk_size_bytes: i64,
    /// RAM space used in bytes
    pub ram_size_bytes: i64,
}

impl From<ruvector_collections::CollectionStats> for JsCollectionStats {
    fn from(stats: ruvector_collections::CollectionStats) -> Self {
        JsCollectionStats {
            vectors_count: stats.vectors_count as u32,
            disk_size_bytes: stats.disk_size_bytes as i64,
            ram_size_bytes: stats.ram_size_bytes as i64,
        }
    }
}

/// Collection alias
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsAlias {
    /// Alias name
    pub alias: String,
    /// Collection name
    pub collection: String,
}

impl From<(String, String)> for JsAlias {
    fn from(tuple: (String, String)) -> Self {
        JsAlias {
            alias: tuple.0,
            collection: tuple.1,
        }
    }
}

/// Collection manager for multi-collection support
#[napi]
pub struct CollectionManager {
    inner: Arc<RwLock<CoreCollectionManager>>,
}

#[napi]
impl CollectionManager {
    /// Create a new collection manager
    ///
    /// # Example
    /// ```javascript
    /// const manager = new CollectionManager('./collections');
    /// ```
    #[napi(constructor)]
    pub fn new(base_path: Option<String>) -> Result<Self> {
        let path = PathBuf::from(base_path.unwrap_or_else(|| "./collections".to_string()));
        let manager = CoreCollectionManager::new(path).map_err(|e| {
            Error::from_reason(format!("Failed to create collection manager: {}", e))
        })?;

        Ok(Self {
            inner: Arc::new(RwLock::new(manager)),
        })
    }

    /// Create a new collection
    ///
    /// # Example
    /// ```javascript
    /// await manager.createCollection('my_vectors', {
    ///   dimensions: 384,
    ///   distanceMetric: 'Cosine'
    /// });
    /// ```
    #[napi]
    pub async fn create_collection(&self, name: String, config: JsCollectionConfig) -> Result<()> {
        let core_config: ruvector_collections::CollectionConfig = config.into();
        let manager = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let manager = manager.write().expect("RwLock poisoned");
            manager.create_collection(&name, core_config)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Failed to create collection: {}", e)))
    }

    /// List all collections
    ///
    /// # Example
    /// ```javascript
    /// const collections = await manager.listCollections();
    /// console.log('Collections:', collections);
    /// ```
    #[napi]
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let manager = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let manager = manager.read().expect("RwLock poisoned");
            manager.list_collections()
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))
    }

    /// Delete a collection
    ///
    /// # Example
    /// ```javascript
    /// await manager.deleteCollection('my_vectors');
    /// ```
    #[napi]
    pub async fn delete_collection(&self, name: String) -> Result<()> {
        let manager = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let manager = manager.write().expect("RwLock poisoned");
            manager.delete_collection(&name)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Failed to delete collection: {}", e)))
    }

    /// Get collection statistics
    ///
    /// # Example
    /// ```javascript
    /// const stats = await manager.getStats('my_vectors');
    /// console.log(`Vectors: ${stats.vectorsCount}`);
    /// ```
    #[napi]
    pub async fn get_stats(&self, name: String) -> Result<JsCollectionStats> {
        let manager = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let manager = manager.read().expect("RwLock poisoned");
            manager.collection_stats(&name)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Failed to get stats: {}", e)))
        .map(Into::into)
    }

    /// Create an alias for a collection
    ///
    /// # Example
    /// ```javascript
    /// await manager.createAlias('latest', 'my_vectors_v2');
    /// ```
    #[napi]
    pub async fn create_alias(&self, alias: String, collection: String) -> Result<()> {
        let manager = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let manager = manager.write().expect("RwLock poisoned");
            manager.create_alias(&alias, &collection)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Failed to create alias: {}", e)))
    }

    /// Delete an alias
    ///
    /// # Example
    /// ```javascript
    /// await manager.deleteAlias('latest');
    /// ```
    #[napi]
    pub async fn delete_alias(&self, alias: String) -> Result<()> {
        let manager = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let manager = manager.write().expect("RwLock poisoned");
            manager.delete_alias(&alias)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
        .map_err(|e| Error::from_reason(format!("Failed to delete alias: {}", e)))
    }

    /// List all aliases
    ///
    /// # Example
    /// ```javascript
    /// const aliases = await manager.listAliases();
    /// for (const alias of aliases) {
    ///   console.log(`${alias.alias} -> ${alias.collection}`);
    /// }
    /// ```
    #[napi]
    pub async fn list_aliases(&self) -> Result<Vec<JsAlias>> {
        let manager = self.inner.clone();

        let aliases = tokio::task::spawn_blocking(move || {
            let manager = manager.read().expect("RwLock poisoned");
            manager.list_aliases()
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?;

        Ok(aliases.into_iter().map(Into::into).collect())
    }
}

/// Health response
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsHealthResponse {
    /// Status: "healthy", "degraded", or "unhealthy"
    pub status: String,
    /// Version string
    pub version: String,
    /// Uptime in seconds
    pub uptime_seconds: i64,
}

/// Get Prometheus metrics
///
/// # Example
/// ```javascript
/// const metrics = getMetrics();
/// console.log(metrics);
/// ```
#[napi]
pub fn get_metrics() -> String {
    gather_metrics()
}

/// Get health status
///
/// # Example
/// ```javascript
/// const health = getHealth();
/// console.log(`Status: ${health.status}`);
/// console.log(`Uptime: ${health.uptimeSeconds}s`);
/// ```
#[napi]
pub fn get_health() -> JsHealthResponse {
    let checker = HealthChecker::new();
    let health = checker.health();

    JsHealthResponse {
        status: match health.status {
            HealthStatus::Healthy => "healthy".to_string(),
            HealthStatus::Degraded => "degraded".to_string(),
            HealthStatus::Unhealthy => "unhealthy".to_string(),
        },
        version: health.version,
        uptime_seconds: health.uptime_seconds as i64,
    }
}

#[cfg(test)]
mod tests {
    //! Issue #1063 regression tests for the options conversion and the
    //! constructor. They drive the Rust side directly (no Node runtime).
    use super::*;

    fn js_options(dimensions: u32, storage_path: Option<String>) -> JsDbOptions {
        JsDbOptions {
            dimensions,
            distance_metric: None,
            storage_path,
            hnsw_config: None,
            quantization: None,
        }
    }

    fn insert(db: &VectorDB, id: &str, dimensions: usize) {
        db.inner
            .read()
            .unwrap()
            .insert(VectorEntry {
                id: Some(id.to_string()),
                vector: vec![0.1; dimensions],
                metadata: None,
            })
            .unwrap();
    }

    #[test]
    fn omitted_storage_path_converts_to_in_memory() {
        let options: DbOptions = js_options(8, None).into();
        assert!(options.storage_path.starts_with("memory://"));
        let explicit: DbOptions = js_options(8, Some("/tmp/x.db".to_string())).into();
        assert_eq!(explicit.storage_path, "/tmp/x.db");
    }

    #[test]
    fn instances_without_storage_path_are_isolated_and_create_no_file() {
        let default_file = std::path::Path::new("./ruvector.db");
        let existed_before = default_file.exists();

        let a = VectorDB::new(js_options(256, None)).unwrap();
        let b = VectorDB::new(js_options(4, None)).unwrap();
        insert(&a, "x", 256);
        insert(&b, "y", 4);
        assert_eq!(a.inner.read().unwrap().len().unwrap(), 1);
        assert_eq!(b.inner.read().unwrap().len().unwrap(), 1);

        let c = VectorDB::with_dimensions(16).unwrap();
        insert(&c, "z", 16);

        if !existed_before {
            assert!(
                !default_file.exists(),
                "a VectorDB without storagePath must not create ./ruvector.db"
            );
        }
    }

    #[test]
    fn shared_storage_path_with_different_dimensions_fails_at_construction() {
        let dir = unique_temp_dir("dims");
        let path = dir.join("shared.db").to_string_lossy().to_string();

        {
            let first = VectorDB::new(js_options(256, Some(path.clone()))).unwrap();
            insert(&first, "x", 256);
        }
        let err = match VectorDB::new(js_options(4, Some(path.clone()))) {
            Ok(_) => panic!("opening a 256-dim store with dimensions 4 must fail"),
            Err(e) => e.reason.clone(),
        };
        assert!(err.contains(&path), "error must name the path: {err}");
        assert!(
            err.contains("stored 256"),
            "error must name stored dims: {err}"
        );
        assert!(
            err.contains("requested 4"),
            "error must name requested dims: {err}"
        );

        // Same dimensions still reopen the persisted data.
        let reopened = VectorDB::new(js_options(256, Some(path))).unwrap();
        assert_eq!(reopened.inner.read().unwrap().len().unwrap(), 1);
        drop(reopened);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ruvector-node-1063-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reopen_without_metric_adopts_stored_metric() {
        let dir = unique_temp_dir("metric");
        let path = dir.join("euclid.db").to_string_lossy().to_string();

        {
            let mut options = js_options(128, Some(path.clone()));
            options.distance_metric = Some(JsDistanceMetric::Euclidean);
            let created = VectorDB::new(options).unwrap();
            insert(&created, "x", 128);
        }

        // Omitting distanceMetric must not be read as a request for Cosine.
        let reopened = VectorDB::new(js_options(128, Some(path.clone()))).unwrap();
        {
            let inner = reopened.inner.read().unwrap();
            assert_eq!(inner.options().distance_metric, DistanceMetric::Euclidean);
            assert_eq!(inner.len().unwrap(), 1);
        }
        drop(reopened);

        // An explicitly different metric is still rejected.
        let mut options = js_options(128, Some(path.clone()));
        options.distance_metric = Some(JsDistanceMetric::Cosine);
        let err = match VectorDB::new(options) {
            Ok(_) => panic!("an explicit Cosine must not open a Euclidean store"),
            Err(e) => e.reason.clone(),
        };
        assert!(
            err.contains("distance metric: stored Euclidean, requested Cosine"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
