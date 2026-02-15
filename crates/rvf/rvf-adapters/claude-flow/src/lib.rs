//! RVF adapter for Claude-Flow AI agent memory with audit trails.  
//!  
//! This crate bridges Claude-Flow's memory store to the RuVector Format (RVF)  
//! segment store, per ADR-029. It provides persistent storage for agent  
//! memories, cross-session persistence, and tamper-evident audit trails.  
//!  
//! # Segment mapping  
//!  
//! - **VEC_SEG + META_SEG**: Memory entries (embeddings + key/value metadata).  
//! - **WITNESS_SEG**: Agent audit trail with hash-chained signatures.  
//! - **MANIFEST_SEG**: Append-only log for cross-session recovery.  
//! - **META_SEG**: Swarm coordination state (agent states, topology changes).  
//! - **SKETCH_SEG**: Agent learning patterns with effectiveness scores.  
//!  
//! # Usage  
//!  
//! ```rust,no_run  
//! use rvf_adapter_claude_flow::{ClaudeFlowConfig, RvfMemoryStore};  
//!  
//! let config = ClaudeFlowConfig::new("/tmp/agent-memory", "agent-42");  
//! let mut store = RvfMemoryStore::create(config).unwrap();  
//!  
//! // Ingest a memory entry  
//! let embedding = vec![0.1f32; 384];  
//! store.ingest_memory("user-preference", "dark-mode", "ui", &embedding).unwrap();  
//!  
//! // Search memories by embedding similarity  
//! let results = store.search_memories(&embedding, 5);  
//!  
//! // Record an audit event  
//! store.witness().record_action("set-preference", &["dark-mode"]).unwrap();  
//!  
//! // Record coordination state  
//! store.coordination().record_state("agent-42", "status", "active").unwrap();  
//!  
//! // Store a learning pattern  
//! store.learning().store_pattern("convergent", "Use batched writes", 0.92).unwrap();  
//!  
//! store.close().unwrap();  
//! ```  
  
pub mod config;  
pub mod coordination;  
pub mod error;  
pub mod learning;  
pub mod memory_store;  
pub mod witness;  
  
pub use config::{ClaudeFlowConfig, ConfigError};  
pub use coordination::{ConsensusVote, StateEntry, SwarmCoordination};  
pub use error::ClaudeFlowError;  
pub use learning::{LearningPatternStore, PatternResult};  
pub use memory_store::{MemoryEntry, MemoryResult, RvfMemoryStore};  
pub use witness::{WitnessEntry, WitnessWriter};