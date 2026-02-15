//! RVF-backed memory store for Claude-Flow agents.  
//!  
//! Provides persistent storage for agent memories with metadata and  
//! optional tamper-evident audit trails via WITNESS_SEG. Includes  
//! swarm coordination and learning pattern sub-stores.  
  
use std::collections::HashMap;  
  
use rvf_runtime::options::{  
    DistanceMetric, MetadataEntry, MetadataValue, QueryOptions, RvfOptions,  
};  
use rvf_runtime::{IngestResult, RvfStore, SearchResult, StoreStatus};  
use rvf_types::RvfError;  
  
use crate::config::ClaudeFlowConfig;  
use crate::coordination::SwarmCoordination;  
use crate::error::ClaudeFlowError;  
use crate::learning::LearningPatternStore;  
use crate::witness::WitnessWriter;  
  
/// Metadata field IDs for memory entries.  
pub mod fields {  
    /// Agent identifier.  
    pub const AGENT_ID: u16 = 0;  
    /// Memory key.  
    pub const KEY: u16 = 1;  
    /// Memory value.  
    pub const VALUE: u16 = 2;  
    /// Namespace (optional).  
    pub const NAMESPACE: u16 = 3;  
    /// Timestamp (seconds since epoch).  
    pub const TIMESTAMP_SECS: u16 = 4;  
}  
  
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
    /// Timestamp when stored.  
    pub timestamp_secs: u64,  
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
    /// Timestamp.  
    pub timestamp_secs: u64,  
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
    pub fn create(config: ClaudeFlowConfig) -> Result<Self, ClaudeFlowError> {  
        config.validate().map_err(ClaudeFlowError::Config)?;  
        config  
            .ensure_dirs()  
            .map_err(|e| ClaudeFlowError::Io(e.to_string()))?;  
  
        let rvf_options = RvfOptions {  
            dimension: config.dimension,  
            metric: DistanceMetric::Cosine,  
            ..Default::default()  
        };  
  
        let store = RvfStore::create(&config.store_path(), rvf_options)  
            .map_err(ClaudeFlowError::Rvf)?;  
  
        let witness = WitnessWriter::create(&config)?;  
  
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
    pub fn open(config: ClaudeFlowConfig) -> Result<Self, ClaudeFlowError> {  
        config.validate().map_err(ClaudeFlowError::Config)?;  
  
        let store = RvfStore::open(&config.store_path()).map_err(ClaudeFlowError::Rvf)?;  
  
        let witness = WitnessWriter::open(&config)?;  
  
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
    ) -> Result<u64, ClaudeFlowError> {  
        if embedding.len() != self.config.dimension as usize {  
            return Err(ClaudeFlowError::DimensionMismatch {  
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
            self.store.delete(&[old_id]).map_err(ClaudeFlowError::Rvf)?;  
            self.entry_index.remove(&old_id);  
        }  
  
        let vector_id = self.next_id;  
        self.next_id += 1;  
  
        let metadata = vec![  
            MetadataEntry {  
                field_id: fields::AGENT_ID,  
                value: MetadataValue::String(self.config.agent_id.clone()),  
            },  
            MetadataEntry {  
                field_id: fields::KEY,  
                value: MetadataValue::String(key.to_string()),  
            },  
            MetadataEntry {  
                field_id: fields::VALUE,  
                value: MetadataValue::String(value.to_string()),  
            },  
            MetadataEntry {  
                field_id: fields::NAMESPACE,  
                value: namespace.map(|s| MetadataValue::String(s.to_string()))  
                    .unwrap_or(MetadataValue::Null),  
            },  
            MetadataEntry {  
                field_id: fields::TIMESTAMP_SECS,  
                value: MetadataValue::U64(  
                    std::time::SystemTime::now()  
                        .duration_since(std::time::UNIX_EPOCH)  
                        .map(|d| d.as_secs())  
                        .unwrap_or(0)  
                ),  
            },  
        ];  
  
        self.store  
            .ingest_batch(&[embedding], &[vector_id], Some(&metadata))  
            .map_err(ClaudeFlowError::Rvf)?;  
  
        let timestamp_secs = std::time::SystemTime::now()  
            .duration_since(std::time::UNIX_EPOCH)  
            .map(|d| d.as_secs())  
            .unwrap_or(0);  
  
        self.key_index.insert(compound_key, vector_id);  
        self.entry_index.insert(  
            vector_id,  
            MemoryEntry {  
                id: vector_id,  
                agent_id: self.config.agent_id.clone(),  
                key: key.to_string(),  
                value: value.to_string(),  
                namespace: namespace.map(|s| s.to_string()),  
                timestamp_secs,  
            },  
        );  
  
        // Record in witness if enabled  
        if self.config.enable_witness {  
            let _ = self.witness.record_ingest(key, value, namespace);  
        }  
  
        Ok(vector_id)  
    }  
  
    /// Search memories by embedding similarity.  
    pub fn search_memories(&self, embedding: &[f32], k: usize) -> Result<Vec<MemoryResult>, ClaudeFlowError> {  
        let options = QueryOptions::default();  
        let results = self.store.query(embedding, k, &options)  
            .map_err(ClaudeFlowError::Rvf)?;  
  
        let enriched: Vec<MemoryResult> = results  
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
                    timestamp_secs: entry.timestamp_secs,  
                })  
            })  
            .collect();  
  
        Ok(enriched)  
    }  
  
    /// Retrieve a memory entry by its vector ID.  
    pub fn get_memory(&self, id: u64) -> Option<MemoryEntry> {  
        self.entry_index.get(&id).cloned()  
    }  
  
    /// Delete memory entries by their vector IDs.  
    pub fn delete_memories(&mut self, ids: &[u64]) -> Result<usize, ClaudeFlowError> {  
        let mut removed = 0;  
        for &id in ids {  
            if self.entry_index.remove(&id).is_some() {  
                // Remove from key_index if present  
                self.key_index.retain(|_, &mut v| v != id);  
                removed += 1;  
            }  
        }  
        if removed > 0 {  
            self.store.delete(ids).map_err(ClaudeFlowError::Rvf)?;  
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
    pub fn status(&self) -> StoreStatus {  
        self.store.status()  
    }  
  
    /// Access the witness writer for audit trails.  
    pub fn witness(&mut self) -> &mut WitnessWriter {  
        &mut self.witness  
    }  
  
    /// Close the store, releasing locks.  
    pub fn close(self) -> Result<(), ClaudeFlowError> {  
        self.store.close().map_err(ClaudeFlowError::Rvf)  
    }  
}  
  
#[cfg(test)]  
mod tests {  
    use super::*;  
    use tempfile::TempDir;  
  
    fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {  
        (0..dim).map(|i| seed * (i as f32 + 1.0)).collect()  
    }  
  
    #[test]  
    fn create_and_ingest() {  
        let dir = TempDir::new().unwrap();  
        let config = ClaudeFlowConfig::new(dir.path(), "agent-test").with_dimension(4);  
        let mut store = RvfMemoryStore::create(config).unwrap();  
  
        let embedding = make_embedding(4, 0.5);  
        let id = store  
            .ingest_memory("pref", "dark-mode", Some("ui"), &embedding)  
            .unwrap();  
        assert!(id > 0);  
  
        let status = store.status();  
        assert_eq!(status.total_vectors, 1);  
  
        store.close().unwrap();  
    }  
  
    #[test]  
    fn coordination_and_learning_accessors() {  
        let dir = TempDir::new().unwrap();  
        let config = ClaudeFlowConfig::new(dir.path(), "agent-test").with_dimension(4);  
        let mut store = RvfMemoryStore::create(config).unwrap();  
  
        // Test coordination accessor  
        store.coordination().record_state("agent-1", "status", "active").unwrap();  
        let states = store.coordination_ref().get_all_states();  
        assert_eq!(states.len(), 1);  
  
        // Test learning accessor  
        let pattern_id = store.learning().store_pattern("convergent", "test pattern", 0.8).unwrap();  
        let pattern = store.learning_ref().get_pattern(pattern_id).unwrap();  
        assert_eq!(pattern.pattern_type, "convergent");  
  
        store.close().unwrap();  
    }  
  
    #[test]  
    fn ingest_and_search() {  
        let dir = TempDir::new().unwrap();  
        let config = ClaudeFlowConfig::new(dir.path(), "agent-search").with_dimension(4);  
        let mut store = RvfMemoryStore::create(config).unwrap();  
  
        let e1 = make_embedding(4, 0.1);  
        let e2 = make_embedding(4, 0.2);  
        store  
            .ingest_memory("k1", "v1", Some("ns"), &e1)  
            .unwrap();  
        store  
            .ingest_memory("k2", "v2", Some("ns"), &e2)  
            .unwrap();  
  
        let results = store.search_memories(&e1, 2).unwrap();  
        assert_eq!(results.len(), 2);  
        assert_eq!(results[0].key, "k1");  
        assert_eq!(results[1].key, "k2");  
  
        store.close().unwrap();  
    }  
  
    #[test]  
    fn open_existing_store() {  
        let dir = TempDir::new().unwrap();  
        let config = ClaudeFlowConfig::new(dir.path(), "agent-open").with_dimension(4);  
  
        {  
            let mut store = RvfMemoryStore::create(config.clone()).unwrap();  
            store  
                .ingest_memory("k", "v", None, &make_embedding(4, 0.3))  
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
    fn duplicate_key_replaces() {  
        let dir = TempDir::new().unwrap();  
        let config = ClaudeFlowConfig::new(dir.path(), "agent-dup").with_dimension(4);  
        let mut store = RvfMemoryStore::create(config).unwrap();  
  
        let e1 = make_embedding(4, 0.4);  
        let e2 = make_embedding(4, 0.5);  
        let id1 = store  
            .ingest_memory("dup", "old", Some("ns"), &e1)  
            .unwrap();  
        let id2 = store  
            .ingest_memory("dup", "new", Some("ns"), &e2)  
            .unwrap();  
  
        assert_ne!(id1, id2);  
        let results = store.search_memories(&e2, 1).unwrap();  
        assert_eq!(results[0].value, "new");  
  
        store.close().unwrap();  
    }  
}