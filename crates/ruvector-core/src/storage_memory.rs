//! In-memory storage backend for WASM and testing
//!
//! This storage implementation doesn't require file system access,
//! making it suitable for WebAssembly environments.

use crate::error::{Result, RuvectorError};
use crate::types::{DistanceMetric, SearchQuery, SearchResult, VectorEntry, VectorId};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

type PostingMap = HashMap<(String, String), HashMap<String, Arc<Vec<f32>>>>;

/// In-memory storage backend with an exact inverted metadata index
pub struct MemoryStorage {
    vectors: DashMap<String, Arc<Vec<f32>>>,
    metadata: DashMap<String, JsonValue>,
    config: DashMap<String, String>,
    /// Equality postings keyed by field and JSON value.
    postings: RwLock<PostingMap>,
    dimensions: usize,
    counter: AtomicU64,
}

impl MemoryStorage {
    /// Create a new in-memory storage
    pub fn new(dimensions: usize) -> Result<Self> {
        Ok(Self {
            vectors: DashMap::new(),
            metadata: DashMap::new(),
            config: DashMap::new(),
            postings: RwLock::new(HashMap::new()),
            dimensions,
            counter: AtomicU64::new(0),
        })
    }

    /// Generate a new unique ID
    fn generate_id(&self) -> String {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        format!("vec_{}", id)
    }

    /// Remove stale postings when an ID is replaced or deleted.
    fn unindex_id(&self, postings: &mut PostingMap, id: &str) {
        if let Some(raw) = self.metadata.get(id) {
            if let JsonValue::Object(map) = raw.value() {
                for (field, value) in map {
                    let key = (field.clone(), value.to_string());
                    if let Some(ids) = postings.get_mut(&key) {
                        ids.remove(id);
                        if ids.is_empty() {
                            postings.remove(&key);
                        }
                    }
                }
            }
        }
    }

    /// Insert a vector entry
    pub fn insert(&self, entry: &VectorEntry) -> Result<VectorId> {
        if entry.vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: entry.vector.len(),
            });
        }
        if !entry.vector.iter().all(|v| v.is_finite()) {
            return Err(RuvectorError::InvalidInput(
                "vector must contain finite values".into(),
            ));
        }
        // Search and mutation locks serialize allocation with explicit IDs.
        let mut postings = self.postings.write();
        let id = match &entry.id {
            Some(id) => id.clone(),
            None => loop {
                let candidate = self.generate_id();
                if !self.vectors.contains_key(&candidate) {
                    break candidate;
                }
            },
        };
        self.unindex_id(&mut postings, &id);
        let shared_vector = Arc::new(entry.vector.clone());
        self.vectors.insert(id.clone(), shared_vector.clone());
        if let Some(metadata) = &entry.metadata {
            let mut json_map = serde_json::Map::new();
            for (field, value) in metadata {
                postings
                    .entry((field.clone(), value.to_string()))
                    .or_default()
                    .insert(id.clone(), shared_vector.clone());
                json_map.insert(field.clone(), value.clone());
            }
            self.metadata
                .insert(id.clone(), JsonValue::Object(json_map));
        } else {
            self.metadata.remove(&id);
        }
        Ok(id)
    }

    /// Insert multiple vectors in a batch
    pub fn insert_batch(&self, entries: &[VectorEntry]) -> Result<Vec<VectorId>> {
        // Validate the full batch before any mutation.
        for entry in entries {
            if entry.vector.len() != self.dimensions {
                return Err(RuvectorError::DimensionMismatch {
                    expected: self.dimensions,
                    actual: entry.vector.len(),
                });
            }
            if !entry.vector.iter().all(|v| v.is_finite()) {
                return Err(RuvectorError::InvalidInput(
                    "vector must contain finite values".into(),
                ));
            }
        }
        entries.iter().map(|entry| self.insert(entry)).collect()
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        if let Some(vector_ref) = self.vectors.get(id) {
            let vector = vector_ref.value().as_ref().clone();
            let metadata = self.metadata.get(id).and_then(|m| {
                if let serde_json::Value::Object(map) = m.value() {
                    Some(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                } else {
                    None
                }
            });

            Ok(Some(VectorEntry {
                id: Some(id.to_string()),
                vector,
                metadata,
            }))
        } else {
            Ok(None)
        }
    }

    /// Predicate-first exact retrieval without cloning excluded vectors.
    /// This is faster than materializing all entries for selective filters.
    pub(crate) fn search_filtered_exact(
        &self,
        query: &SearchQuery,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        let Some(filter) = query.filter.as_ref() else {
            return Err(RuvectorError::InvalidInput(
                "filtered search requires a filter".into(),
            ));
        };
        if query.k == 0 {
            return Ok(Vec::new());
        }
        if query.vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.vector.len(),
            });
        }
        if !query.vector.iter().all(|v| v.is_finite()) {
            return Err(RuvectorError::InvalidInput(
                "query vector must be finite".into(),
            ));
        }
        // Keep a read guard throughout scoring and materialization, so
        // insertion, replacement and deletion cannot invalidate the snapshot.
        let postings = self.postings.read();
        let mut smallest: Option<&HashMap<String, Arc<Vec<f32>>>> = None;
        let mut criteria = Vec::with_capacity(filter.len());
        for (field, value) in filter {
            let key = (field.clone(), value.to_string());
            let Some(bucket) = postings.get(&key) else {
                return Ok(Vec::new());
            };
            if smallest.is_none_or(|current| bucket.len() < current.len()) {
                smallest = Some(bucket);
            }
            criteria.push(bucket);
        }
        // Borrow IDs and shared vector payloads directly from the posting.
        // This removes per-hit DashMap lookups, metadata reads and ID copies.
        let mut candidates: Vec<(Cow<'_, str>, f32)> = Vec::new();
        if let Some(bucket) = smallest {
            candidates.reserve(bucket.len().min(100_000));
            for (id, vector) in bucket {
                if !criteria.iter().all(|candidate| candidate.contains_key(id)) {
                    continue;
                }
                let score = crate::distance::distance(&query.vector, vector.as_ref(), metric)?;
                candidates.push((Cow::Borrowed(id.as_str()), score));
            }
        } else {
            // Empty predicates include rows without metadata.
            for entry in self.vectors.iter() {
                let score =
                    crate::distance::distance(&query.vector, entry.value().as_ref(), metric)?;
                candidates.push((Cow::Owned(entry.key().clone()), score));
            }
        }
        let rank = |a: &(Cow<'_, str>, f32), b: &(Cow<'_, str>, f32)| {
            a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
        };
        if candidates.len() > query.k {
            candidates.select_nth_unstable_by(query.k, rank);
            candidates.truncate(query.k);
        }
        candidates.sort_unstable_by(rank);
        let mut results = Vec::with_capacity(candidates.len());
        for (id, score) in candidates {
            let Some(entry) = self.vectors.get(id.as_ref()) else {
                continue;
            };
            let metadata = self
                .metadata
                .get(id.as_ref())
                .and_then(|raw| match raw.value() {
                    JsonValue::Object(map) => Some(
                        map.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect::<HashMap<String, JsonValue>>(),
                    ),
                    _ => None,
                });
            if !filter.is_empty()
                && !metadata.as_ref().is_some_and(|map| {
                    filter
                        .iter()
                        .all(|(field, value)| map.get(field) == Some(value))
                })
            {
                continue;
            }
            results.push(SearchResult {
                id: id.into_owned(),
                score,
                vector: Some(entry.value().as_ref().clone()),
                metadata,
            });
        }
        results.sort_unstable_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)));
        Ok(results)
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut postings = self.postings.write();
        self.unindex_id(&mut postings, id);
        let vector_removed = self.vectors.remove(id).is_some();
        self.metadata.remove(id);
        Ok(vector_removed)
    }

    /// Get the number of vectors stored
    pub fn len(&self) -> Result<usize> {
        Ok(self.vectors.len())
    }

    /// Check if the storage is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.vectors.is_empty())
    }

    /// Get all vector IDs (for iteration)
    pub fn keys(&self) -> Vec<String> {
        self.vectors
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get all vector IDs (alias for keys, for API compatibility with VectorStorage)
    pub fn all_ids(&self) -> Result<Vec<String>> {
        Ok(self.keys())
    }

    /// Clear all data
    pub fn clear(&self) -> Result<()> {
        let mut postings = self.postings.write();
        self.vectors.clear();
        self.metadata.clear();
        postings.clear();
        Ok(())
    }

    /// Store an arbitrary configuration value for this in-memory database instance.
    pub fn save_config_value(&self, key: &str, value: &str) -> Result<()> {
        self.config.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Load a configuration value stored on this in-memory database instance.
    pub fn load_config_value(&self, key: &str) -> Result<Option<String>> {
        Ok(self.config.get(key).map(|value| value.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    #[test]
    fn test_postings_upsert_delete_clear_and_empty_filter() {
        let store = MemoryStorage::new(2).unwrap();
        let entry = |id: &str, tenant: Option<&str>, rank: f32| VectorEntry {
            id: Some(id.to_owned()),
            vector: vec![rank, 0.0],
            metadata: tenant.map(|name| {
                HashMap::from([
                    ("tenant".to_owned(), serde_json::json!(name)),
                    ("shard".to_owned(), serde_json::json!(3)),
                ])
            }),
        };
        store.insert(&entry("a", Some("red"), 1.0)).unwrap();
        store.insert(&entry("b", Some("red"), 2.0)).unwrap();
        store.insert(&entry("c", Some("blue"), 3.0)).unwrap();
        store.insert(&entry("missing", None, 4.0)).unwrap();
        let query = |tenant: &str| SearchQuery {
            vector: vec![0.0, 0.0],
            k: 10,
            filter: Some(HashMap::from([(
                "tenant".to_owned(),
                serde_json::json!(tenant),
            )])),
            ef_search: None,
        };
        let search = |q: SearchQuery| {
            store
                .search_filtered_exact(&q, DistanceMetric::Euclidean)
                .unwrap()
        };
        assert_eq!(
            search(query("red"))
                .iter()
                .map(|r| r.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        store.insert(&entry("a", Some("blue"), 0.5)).unwrap();
        assert_eq!(search(query("red")).len(), 1);
        assert_eq!(search(query("blue")).len(), 2);
        store.insert(&entry("a", None, 0.2)).unwrap();
        assert_eq!(search(query("blue")).len(), 1);
        assert!(store.delete("c").unwrap());
        assert!(search(query("blue")).is_empty());
        let mut composite = query("red");
        composite
            .filter
            .as_mut()
            .unwrap()
            .insert("shard".into(), serde_json::json!(4));
        assert!(search(composite).is_empty());
        let all = search(SearchQuery {
            vector: vec![0.0, 0.0],
            k: 10,
            filter: Some(HashMap::new()),
            ef_search: None,
        });
        assert_eq!(all.len(), 3);
        assert!(all
            .iter()
            .any(|r| r.id == "missing" && r.metadata.is_none()));
        store.clear().unwrap();
        assert!(search(query("red")).is_empty());
        assert_eq!(store.postings.read().len(), 0);
    }

    #[test]
    fn test_postings_exact_oracle_across_multi_field_queries() {
        let store = MemoryStorage::new(8).unwrap();
        for i in 0..2000usize {
            let vector: Vec<f32> = (0..8)
                .map(|j| ((i * 37 + j * 19) % 997) as f32 / 997.0)
                .collect();
            let metadata = HashMap::from([
                ("tenant".to_owned(), serde_json::json!(i % 19)),
                ("shard".to_owned(), serde_json::json!(i % 7)),
            ]);
            store
                .insert(&VectorEntry {
                    id: Some(format!("v{i}")),
                    vector,
                    metadata: Some(metadata),
                })
                .unwrap();
        }
        for q in 0..75usize {
            let qvec: Vec<f32> = (0..8)
                .map(|j| ((q * 43 + j * 13) % 983) as f32 / 983.0)
                .collect();
            let mut filter = HashMap::new();
            if q % 11 != 0 {
                filter.insert("tenant".to_owned(), serde_json::json!(q % 19));
            }
            if q % 3 == 0 {
                filter.insert("shard".to_owned(), serde_json::json!(q % 7));
            }
            let k = if q % 7 == 0 { 100 } else { 10 };
            let query = SearchQuery {
                vector: qvec.clone(),
                k,
                filter: Some(filter.clone()),
                ef_search: None,
            };
            let actual = store
                .search_filtered_exact(&query, DistanceMetric::Euclidean)
                .unwrap();
            let mut oracle = Vec::new();
            for entry in store.vectors.iter() {
                let Some(raw) = store.metadata.get(entry.key()) else {
                    continue;
                };
                let JsonValue::Object(map) = raw.value() else {
                    continue;
                };
                if filter
                    .iter()
                    .all(|(key, value)| map.get(key) == Some(value))
                {
                    oracle.push((
                        entry.key().clone(),
                        crate::distance::distance(&qvec, entry.value(), DistanceMetric::Euclidean)
                            .unwrap(),
                    ));
                }
            }
            oracle.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            oracle.truncate(k);
            assert_eq!(
                actual.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
                oracle.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(),
                "query {q}"
            );
            for (r, (_, score)) in actual.iter().zip(oracle.iter()) {
                assert!((r.score - score).abs() <= 1e-6, "score mismatch query {q}");
            }
        }
    }

    #[test]
    fn test_filtered_search_never_leaks_concurrent_namespace_update() {
        let store = std::sync::Arc::new(MemoryStorage::new(2).unwrap());
        let writer_store = store.clone();
        let writer = std::thread::spawn(move || {
            for i in 0..300 {
                let tenant = if i % 2 == 0 { "red" } else { "blue" };
                let metadata = HashMap::from([("tenant".to_owned(), serde_json::json!(tenant))]);
                writer_store
                    .insert(&VectorEntry {
                        id: Some("mutable".to_owned()),
                        vector: vec![i as f32, 0.0],
                        metadata: Some(metadata),
                    })
                    .unwrap();
            }
        });
        for _ in 0..300 {
            let query = SearchQuery {
                vector: vec![0.0, 0.0],
                k: 10,
                filter: Some(HashMap::from([(
                    "tenant".to_owned(),
                    serde_json::json!("red"),
                )])),
                ef_search: None,
            };
            let results = store
                .search_filtered_exact(&query, DistanceMetric::Euclidean)
                .unwrap();
            assert!(results.len() <= 1);
            assert!(results
                .iter()
                .all(|r| r.id == "mutable" && r.metadata.as_ref().unwrap()["tenant"] == "red"));
        }
        writer.join().unwrap();
    }

    #[test]
    fn test_generated_id_does_not_overwrite_explicit_id() {
        let store = MemoryStorage::new(2).unwrap();
        store
            .insert(&VectorEntry {
                id: Some("vec_0".to_string()),
                vector: vec![1.0, 0.0],
                metadata: None,
            })
            .unwrap();
        let generated = store
            .insert(&VectorEntry {
                id: None,
                vector: vec![2.0, 0.0],
                metadata: None,
            })
            .unwrap();
        assert_ne!(generated, "vec_0");
        assert_eq!(store.get("vec_0").unwrap().unwrap().vector, vec![1.0, 0.0]);
        assert_eq!(store.len().unwrap(), 2);
    }

    #[test]
    fn test_insert_and_get() {
        let storage = MemoryStorage::new(128).unwrap();

        let entry = VectorEntry {
            id: Some("test_1".to_string()),
            vector: vec![0.1; 128],
            metadata: Some(HashMap::from([(
                "key".to_string(),
                serde_json::Value::String("value".to_string()),
            )])),
        };

        let id = storage.insert(&entry).unwrap();
        assert_eq!(id, "test_1");

        let retrieved = storage.get("test_1").unwrap().unwrap();
        assert_eq!(retrieved.vector.len(), 128);
        assert!(retrieved.metadata.is_some());
    }

    #[test]
    fn test_batch_insert() {
        let storage = MemoryStorage::new(64).unwrap();

        let entries: Vec<_> = (0..10)
            .map(|i| VectorEntry {
                id: Some(format!("vec_{}", i)),
                vector: vec![i as f32; 64],
                metadata: None,
            })
            .collect();

        let ids = storage.insert_batch(&entries).unwrap();
        assert_eq!(ids.len(), 10);
        assert_eq!(storage.len().unwrap(), 10);
    }

    #[test]
    fn test_delete() {
        let storage = MemoryStorage::new(32).unwrap();

        let entry = VectorEntry {
            id: Some("delete_me".to_string()),
            vector: vec![1.0; 32],
            metadata: None,
        };

        storage.insert(&entry).unwrap();
        assert_eq!(storage.len().unwrap(), 1);

        let deleted = storage.delete("delete_me").unwrap();
        assert!(deleted);
        assert_eq!(storage.len().unwrap(), 0);
    }

    #[test]
    fn test_auto_id_generation() {
        let storage = MemoryStorage::new(16).unwrap();

        let entry = VectorEntry {
            id: None,
            vector: vec![0.5; 16],
            metadata: None,
        };

        let id1 = storage.insert(&entry).unwrap();
        let id2 = storage.insert(&entry).unwrap();

        assert_ne!(id1, id2);
        assert!(id1.starts_with("vec_"));
        assert!(id2.starts_with("vec_"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let storage = MemoryStorage::new(128).unwrap();

        let entry = VectorEntry {
            id: Some("bad".to_string()),
            vector: vec![0.1; 64], // Wrong dimension
            metadata: None,
        };

        let result = storage.insert(&entry);
        assert!(result.is_err());

        if let Err(RuvectorError::DimensionMismatch { expected, actual }) = result {
            assert_eq!(expected, 128);
            assert_eq!(actual, 64);
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }
}
