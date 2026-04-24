//! P2P Swarm v2 - Production Grade Rust Implementation
//!
//! Features:
//! - Ed25519 identity keys + X25519 ephemeral keys for ECDH
//! - AES-256-GCM authenticated encryption
//! - Message replay protection (nonces, counters, timestamps)
//! - GUN-based signaling (no external PeerServer)
//! - IPFS CID pointers for large payloads
//! - Ed25519 signatures on all messages
//! - Relay health monitoring
//! - Task execution envelope with resource budgets
//! - WASM compatible
//! - Quantization (4-32x compression)
//! - Hyperdimensional Computing for pattern matching

mod advanced;
mod artifact;
mod crypto;
mod envelope;
mod identity;
mod relay;
#[cfg(feature = "native")]
mod swarm;

pub use advanced::{
    // Adaptive compression
    AdaptiveCompressor,
    BinaryQuantized,
    CompressedData,
    HdcMemory,
    // HNSW vector index
    HnswIndex,
    // Post-quantum crypto
    HybridKeyPair,
    HybridPublicKey,
    HybridSignature,
    // Hyperdimensional Computing
    Hypervector,
    // Spiking neural networks
    LIFNeuron,
    LogEntry,
    NetworkCondition,
    // Pattern routing
    PatternRouter,
    RaftAppendEntries,
    RaftAppendEntriesResponse,
    // Raft consensus
    RaftNode,
    RaftState,
    RaftVoteRequest,
    RaftVoteResponse,
    // Quantization
    ScalarQuantized,
    // Semantic embeddings
    SemanticEmbedder,
    SemanticTaskMatcher,
    SpikingNetwork,
    HDC_DIMENSION,
};
pub use artifact::ArtifactStore;
pub use crypto::{CanonicalJson, CryptoV2, EncryptedPayload};
pub use envelope::{ArtifactPointer, SignedEnvelope, TaskEnvelope, TaskReceipt};
pub use identity::{IdentityManager, KeyPair, RegisteredMember};
pub use relay::RelayManager;
#[cfg(feature = "native")]
pub use swarm::{P2PSwarmV2, SwarmStatus};
