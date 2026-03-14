//! # RuVix Proof Engine
//!
//! This crate implements the proof engine for the RuVix Cognition Kernel (ADR-087).
//! It provides 3-tier proof routing with security guarantees including time-bounded
//! validity, single-use nonces, and capability-gated verification.
//!
//! ## Proof Tiers
//!
//! | Tier | Name | Latency | Use Case |
//! |------|------|---------|----------|
//! | 0 | Reflex | <100ns | High-frequency vector updates |
//! | 1 | Standard | <100us | Graph mutations with Merkle witness |
//! | 2 | Deep | <10ms | Full coherence verification with mincut |
//!
//! ## Security Properties (ADR-087 Section 20.4)
//!
//! - **Time-bounded validity**: Proofs expire after a configurable window
//! - **Single-use nonces**: Each nonce can only be consumed once
//! - **Capability-gated**: PROVE rights required on target object
//! - **Cache limits**: Maximum 64 entries with 100ms TTL
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_proof::{ProofEngine, ProofEngineConfig, ProofVerifier, ProofCache};
//! use ruvix_types::{ProofTier, ProofToken};
//!
//! // Create proof engine
//! let mut engine = ProofEngine::new(ProofEngineConfig::default());
//!
//! // Generate a Reflex-tier proof for a vector update
//! let mutation_hash = [0u8; 32];
//! let token = engine.generate_reflex_proof(&mutation_hash).unwrap();
//!
//! // Verify the proof
//! let mut verifier = ProofVerifier::new();
//! assert!(verifier.verify(&token, &mutation_hash, current_time_ns()).is_ok());
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// With std feature, provide Vec compatibility
#[cfg(feature = "std")]
use std::vec::Vec;

mod attestation;
mod cache;
mod engine;
mod error;
mod integration;
mod routing;
mod verifier;
mod witness;

pub use attestation::{AttestationBuilder, WitnessLog};
pub use cache::{CacheEntry, ProofCache, ProofCacheConfig};
pub use engine::{ProofEngine, ProofEngineConfig, ProofEngineStats};
pub use error::{ProofError, ProofResult};
pub use integration::FormallyVerifiable;
pub use routing::{route_proof_tier, MutationType, RoutingContext, RoutingContextBuilder, TierRouter};
pub use verifier::{ProofVerifier, VerificationResult, VerifierConfig};
pub use witness::{MerkleWitness, WitnessBuilder};

#[cfg(feature = "verified")]
pub use integration::VerifiedProofBridge;

// Re-export commonly used types from ruvix-types
pub use ruvix_types::{
    ProofAttestation, ProofPayload, ProofTier, ProofToken, ATTESTATION_SIZE, REFLEX_CACHE_SIZE,
    REFLEX_CACHE_TTL_MS,
};

// Re-export capability types for PROVE rights checking
pub use ruvix_cap::{CapError, CapHandle, CapResult, CapRights, Capability};

/// Default proof validity window in nanoseconds (100ms).
pub const DEFAULT_VALIDITY_WINDOW_NS: u64 = 100_000_000;

/// Maximum number of entries in the proof cache.
pub const MAX_CACHE_ENTRIES: usize = 64;

/// Default TTL for cache entries in nanoseconds (100ms).
pub const DEFAULT_CACHE_TTL_NS: u64 = 100_000_000;

/// Verifier version for attestations (0.1.0).
pub const VERIFIER_VERSION: u32 = 0x00_01_00_00;
