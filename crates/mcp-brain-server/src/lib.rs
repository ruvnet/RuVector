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
pub mod gist;
pub mod graph;
pub mod midstream;
pub mod notify;
pub mod optimizer;
pub mod pipeline;
pub mod pubmed;
pub mod quantization;
pub mod ranking;
pub mod rate_limit;
pub mod reputation;
pub mod routes;
pub mod store;
pub mod symbolic;
pub mod tests;
pub mod trainer;
pub mod types;
pub mod verify;
pub mod voice;
pub mod web_ingest;
pub mod web_memory;
pub mod web_store;
