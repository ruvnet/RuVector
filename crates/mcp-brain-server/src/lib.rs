//! mcp-brain-server: Cloud Run backend for RuVector Shared Brain
//!
//! Provides REST API for storing, searching, voting, and managing shared knowledge.
//! Every piece of knowledge is an RVF cognitive container with witness chains,
//! Ed25519 signatures, and differential privacy proofs.

pub mod aggregate;
pub mod auth;
pub mod cognitive;
pub mod drift;
pub mod embeddings;
pub mod gcs;
pub mod graph;
pub mod pipeline;
pub mod ranking;
pub mod rate_limit;
pub mod reputation;
pub mod routes;
pub mod store;
pub mod tests;
pub mod midstream;
pub mod types;
pub mod trainer;
pub mod verify;
pub mod voice;
pub mod symbolic;
pub mod optimizer;
pub mod web_memory;
pub mod web_ingest;
