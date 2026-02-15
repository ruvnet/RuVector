//! `RvfMemoryStore` -- main API wrapping `RvfStore` for Claude-Flow agent memory.
//!
//! Maps Claude-Flow's agent memory model onto the RVF segment model:
//! - Embeddings are stored as vectors via `ingest_batch`
//! - Agent ID, key, value, and namespace are encoded as metadata fields
//! - Searches use `query` with optional namespace filtering
//! - Coordination state and learning patterns are managed by sub-stores
//! - Witness audit trails provide tamper-evident action logging

use std::collections::HashMap;

use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, QueryOptions, RvfOptions,
};
use rvf_runtime::RvfStore;

use crate::config::{ClaudeFlowConfig, ConfigError};
use crate::coordination::SwarmCoordination;
use crate::learning::LearningPatternStore;
use crate::witness::WitnessWriter;

/// Metadata field IDs for memory entries.
const FIELD_AGENT_ID: u16 = 0;
const FIELD_KEY: u16 = 1;
const FIELD_VALUE: u16 = 2;
const FIELD_NAMESPACE: u16 = 3;

/// A memory entry retrieved by ID.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    /// Vector ID in the underlying store.
    pub id: u64,
    /// Agent that stored the memory.
    pub agent_id: String,
    /// Memory key.
    pub key: String,
    /// Memory value.
    pub value: String,
    /// Namespace (if any).
    pub namespace: Option<String>,
}

/// A search result enriched with memory metadata.
#[derive(Clone, Debug)]
pub struct MemoryResult {
    /// Vector ID.
    pub id: u64,
    /// Distance from query.
    pub distance: f32,
    /// Agent ID.
    pub agent_id: String,
    /// Memory key.
    pub key: String,
    /// Memory value.
    pub value: String,
    /// Namespace (if any).
    pub namespace: Option<String>,
}

/// RVF-backed memory store for Claude-Flow agents with coordination and learning.
pub struct RvfMemoryStore {
    store: RvfStore,
    config: ClaudeFlowConfig,
    witness: WitnessWriter,
    coordination: SwarmCoordination,
    learning: LearningPatternStore,
    /// Maps "agent_id/namespace/key" -> vector_id for fast lookup.
    key_index: HashMap<String, u64>,
    /// Maps vector_id -> MemoryEntry for retrieval by ID.
    entry_index: HashMap<u64, MemoryEntry>,
    /// Next vector ID to assign.
    next_id: u64,
}

impl RvfMemoryStore {
    /// Create a new memory store, initializing the RVF file.
    pub fn create(config: ClaudeFlowConfig) -> Result<Self, MemoryStoreError> {
        config.validate().map_err(MemoryStoreError::Config)?;
        config
            .ensure_dirs()
            .map_err(|e| MemoryStoreError::Io(e.to_string()))?;

        let rvf_options = RvfOptions {
            dimension: config.dimension,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let store =
            RvfStore::create(&config.store_path(), rvf_options).map_err(MemoryStoreError::Rvf)?;

        let witness = WitnessWriter::create(&config)
            .map_err(|e| MemoryStoreError::Io(e.to_string()))?;

        Ok(Self {
            store,
            config,
            witness,
            coordination: SwarmCoordination::new(),
            learning: LearningPatternStore::new(),
            key_index: HashMap::new(),
            entry_index: HashMap::new(),
            next_id: 1,
        })
    }

    /// Open an existing memory store.
    pub fn open(config: ClaudeFlowConfig) -> Result<Self, MemoryStoreError> {
        config.validate().map_err(MemoryStoreError::Config)?;

        let store = RvfStore::open(&config.store_path()).map_err(MemoryStoreError::Rvf)?;

        let witness = WitnessWriter::open(&config)
            .map_err(|e| MemoryStoreError::Io(e.to_string()))?;

        // Rebuild next_id from store status.
        let status = store.status();
        let next_id = status.total_vectors + status.current_epoch as u64 + 1;

        Ok(Self {
            store,
            config,
            witness,
            coordination: SwarmCoordination::new(),
            learning: LearningPatternStore::new(),
            key_index: HashMap::new(),
            entry_index: HashMap::new(),
            next_id,
        })
    }

    /// Ingest a memory entry with its embedding.
    ///
    /// If an entry with the same agent_id/namespace/key already exists,
    /// it is soft-deleted and replaced.
    pub fn ingest_memory(
        &mut self,
        key: &str,
        value: &str,
        namespace: Option<&str>,
        embedding: &[f32],
    ) -> Result<u64, MemoryStoreError> {
        if embedding.len() != self.config.dimension as usize {
            return Err(MemoryStoreError::DimensionMismatch {
                expected: self.config.dimension as usize,
                got: embedding.len(),
            });
        }

        let compound_key = if let Some(ns) = namespace {
            format!("{}/{}/{}", self.config.agent_id, ns, key)
        } else {
            format!("{}/{}", self.config.agent_id, key)
        };

        // Soft-delete existing entry with the same compound key.
        if let Some(&old_id) = self.key_index.get(&compound_key) {
            self.store.delete(&[old_id]).map_err(MemoryStoreError::Rvf)?;
            self.entry_index.remove(&old_id);
        }

        let vector_id = self.next_id;
        self.next_id += 1;

        let mut metadata = vec![
            MetadataEntry {
                field_id: FIELD_AGENT_ID,
                value: MetadataValue::String(self.config.agent_id.clone()),
            },
            MetadataEntry {
                field_id: FIELD_KEY,
                value: MetadataValue::String(key.to_string()),
            },
            MetadataEntry {
                field_id: FIELD_VALUE,
                value: MetadataValue::String(value.to_string()),
            },
        ];

        if let Some(ns) = namespace {
            metadata.push(MetadataEntry {
                field_id: FIELD_NAMESPACE,
                value: MetadataValue::String(ns.to_string()),
            });
        }

        self.store
            .ingest_batch(&[embedding], &[vector_id], Some(&metadata))
            .map_err(MemoryStoreError::Rvf)?;

        self.key_index.insert(compound_key, vector_id);
        self.entry_index.insert(
            vector_id,
            MemoryEntry {
                id: vector_id,
                agent_id: self.config.agent_id.clone(),
                key: key.to_string(),
                value: value.to_string(),
                namespace: namespace.map(|s| s.to_string()),
            },
        );

        // Record in witness if enabled
        if self.config.enable_witness {
            let _ = self.witness.record_ingest(key, value, namespace);
        }

        Ok(vector_id)
    }

    /// Search memories by embedding similarity.
    pub fn search_memories(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Vec<MemoryResult> {
        let options = QueryOptions::default();
        let results = match self.store.query(embedding, k, &options) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        results
            .into_iter()
            .filter_map(|r| {
                let entry = self.entry_index.get(&r.id)?;
                Some(MemoryResult {
                    id: r.id,
                    distance: r.distance,
                    agent_id: entry.agent_id.clone(),
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                    namespace: entry.namespace.clone(),
                })
            })
            .collect()
    }

    /// Retrieve a memory entry by its vector ID.
    pub fn get_memory(&self, id: u64) -> Option<MemoryEntry> {
        self.entry_index.get(&id).cloned()
    }

    /// Delete memory entries by their vector IDs.
    pub fn delete_memories(&mut self, ids: &[u64]) -> Result<usize, MemoryStoreError> {
        let existing: Vec<u64> = ids
            .iter()
            .filter(|id| self.entry_index.contains_key(id))
            .copied()
            .collect();

        if existing.is_empty() {
            return Ok(0);
        }

        self.store
            .delete(&existing)
            .map_err(MemoryStoreError::Rvf)?;

        let mut removed = 0;
        for &id in &existing {
            if let Some(entry) = self.entry_index.remove(&id) {
                let compound_key = if let Some(ref ns) = entry.namespace {
                    format!("{}/{}/{}", entry.agent_id, ns, entry.key)
                } else {
                    format!("{}/{}", entry.agent_id, entry.key)
                };
                self.key_index.remove(&compound_key);
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Get a mutable reference to the coordination state tracker.
    pub fn coordination(&mut self) -> &mut SwarmCoordination {
        &mut self.coordination
    }

    /// Get an immutable reference to the coordination state tracker.
    pub fn coordination_ref(&self) -> &SwarmCoordination {
        &self.coordination
    }

    /// Get a mutable reference to the learning pattern store.
    pub fn learning(&mut self) -> &mut LearningPatternStore {
        &mut self.learning
    }

    /// Get an immutable reference to the learning pattern store.
    pub fn learning_ref(&self) -> &LearningPatternStore {
        &self.learning
    }

    /// Get the current store status.
    pub fn status(&self) -> rvf_runtime::StoreStatus {
        self.store.status()
    }

    /// Access the witness writer for audit trails (mutable).
    pub fn witness(&mut self) -> &mut WitnessWriter {
        &mut self.witness
    }

    /// Access the witness writer for audit trails (immutable).
    pub fn witness_ref(&self) -> &WitnessWriter {
        &self.witness
    }

    /// Get the agent ID for this store.
    pub fn agent_id(&self) -> &str {
        &self.config.agent_id
    }

    /// Close the store, releasing locks.
    pub fn close(self) -> Result<(), MemoryStoreError> {
        self.store.close().map_err(MemoryStoreError::Rvf)
    }
}

/// Errors from memory store operations.
#[derive(Debug)]
pub enum MemoryStoreError {
    /// Underlying RVF store error.
    Rvf(rvf_types::RvfError),
    /// Configuration error.
    Config(ConfigError),
    /// I/O error.
    Io(String),
    /// Embedding dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for MemoryStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rvf(e) => write!(f, "RVF store error: {e}"),
            Self::Config(e) => write!(f, "config error: {e}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for MemoryStoreError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(dir: &std::path::Path) -> ClaudeFlowConfig {
        ClaudeFlowConfig::new(dir, "test-agent").with_dimension(4)
    }

    fn make_embedding(seed: f32) -> Vec<f32> {
        vec![seed, seed * 0.5, seed * 0.25, seed * 0.125]
    }

    #[test]
    fn create_and_ingest() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();
        let id = store
            .ingest_memory("pref", "dark-mode", Some("ui"), &make_embedding(1.0))
            .unwrap();
        assert!(id > 0);

        let status = store.status();
        assert_eq!(status.total_vectors, 1);

        store.close().unwrap();
    }

    #[test]
    fn ingest_and_search() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();

        store
            .ingest_memory("a", "val_a", Some("ns1"), &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        store
            .ingest_memory("b", "val_b", Some("ns1"), &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        store
            .ingest_memory("c", "val_c", Some("ns2"), &[0.0, 0.0, 1.0, 0.0])
            .unwrap();

        let results = store.search_memories(&[1.0, 0.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "a");

        store.close().unwrap();
    }

    #[test]
    fn get_memory_by_id() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();
        let id = store
            .ingest_memory("mykey", "myval", Some("ns"), &make_embedding(2.0))
            .unwrap();

        let entry = store.get_memory(id).unwrap();
        assert_eq!(entry.key, "mykey");
        assert_eq!(entry.value, "myval");
        assert_eq!(entry.namespace.as_deref(), Some("ns"));
        assert_eq!(entry.agent_id, "test-agent");

        assert!(store.get_memory(9999).is_none());

        store.close().unwrap();
    }

    #[test]
    fn delete_memories() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();
        let id1 = store
            .ingest_memory("k1", "v1", Some("ns"), &make_embedding(1.0))
            .unwrap();
        let id2 = store
            .ingest_memory("k2", "v2", Some("ns"), &make_embedding(2.0))
            .unwrap();

        let removed = store.delete_memories(&[id1]).unwrap();
        assert_eq!(removed, 1);
        assert!(store.get_memory(id1).is_none());
        assert!(store.get_memory(id2).is_some());

        store.close().unwrap();
    }

    #[test]
    fn replace_existing_key() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();
        let id1 = store
            .ingest_memory("k", "v1", Some("ns"), &make_embedding(1.0))
            .unwrap();
        let id2 = store
            .ingest_memory("k", "v2", Some("ns"), &make_embedding(2.0))
            .unwrap();

        assert_ne!(id1, id2);
        assert!(store.get_memory(id1).is_none());
        let entry = store.get_memory(id2).unwrap();
        assert_eq!(entry.value, "v2");

        let status = store.status();
        assert_eq!(status.total_vectors, 1);

        store.close().unwrap();
    }

    #[test]
    fn dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();
        let result = store.ingest_memory("k", "v", Some("ns"), &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn coordination_state() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();
        store
            .coordination()
            .record_state("agent-1", "status", "active")
            .unwrap();
        store
            .coordination()
            .record_state("agent-2", "status", "idle")
            .unwrap();

        let states = store.coordination_ref().get_all_states();
        assert_eq!(states.len(), 2);

        store.close().unwrap();
    }

    #[test]
    fn learning_patterns() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let mut store = RvfMemoryStore::create(config).unwrap();

        let id = store
            .learning()
            .store_pattern("convergent", "Use batched writes", 0.85)
            .unwrap();

        let pattern = store.learning_ref().get_pattern(id).unwrap();
        assert_eq!(pattern.pattern_type, "convergent");
        assert!((pattern.score - 0.85).abs() < f32::EPSILON);

        store.close().unwrap();
    }

    #[test]
    fn open_existing_store() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        {
            let mut store = RvfMemoryStore::create(config.clone()).unwrap();
            store
                .ingest_memory("k", "v", None, &make_embedding(1.0))
                .unwrap();
            store.close().unwrap();
        }

        {
            let store = RvfMemoryStore::open(config).unwrap();
            let status = store.status();
            assert_eq!(status.total_vectors, 1);
            store.close().unwrap();
        }
    }

    #[test]
    fn empty_store_search() {
        let dir = TempDir::new().unwrap();
        let config = test_config(dir.path());

        let store = RvfMemoryStore::create(config).unwrap();
        let results = store.search_memories(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());

        store.close().unwrap();
    }

    #[test]
    fn invalid_config_rejected() {
        let dir = TempDir::new().unwrap();

        let config = ClaudeFlowConfig::new(dir.path(), "a1").with_dimension(0);
        assert!(RvfMemoryStore::create(config).is_err());

        let config = ClaudeFlowConfig::new(dir.path(), "").with_dimension(4);
        assert!(RvfMemoryStore::create(config).is_err());
    }
}
