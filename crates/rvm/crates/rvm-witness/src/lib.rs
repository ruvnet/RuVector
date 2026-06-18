//! Witness logging subsystem for the RVM microhypervisor.
//!
//! Implements ADR-134. Two record formats exist:
//!
//! - **v2 (current, write format)**: 96-byte records with 128-bit
//!   keyed-BLAKE3 chain MACs and Merkle segment sealing. One keyed
//!   compression per append; signature cost amortized per segment.
//!   See [`WitnessLogV2`], [`crate::seal`], and `rvm_types::WitnessRecordV2`.
//! - **v1 (legacy, verify-only)**: 64-byte records with 32-bit folded
//!   hash chain links. **Writing new v1 logs is removed**: the v1
//!   format is frozen and retained solely so existing logs keep
//!   verifying ([`verify_chain`], [`verify_chain_v1_with_head`]) and so
//!   in-kernel rings can migrate incrementally. New persisted logs MUST
//!   be v2; a serialized log may contain a v1 prefix followed by v2
//!   records (see [`verify_log_bytes`], which dispatches on the version
//!   byte at offset 19 of each record).
//!
//! # Core Invariant
//!
//! **No witness, no mutation.** Every privileged action emits a witness
//! record before the mutation is committed. If emission fails, the
//! mutation does not proceed.
//!
//! # v1 Record Format (legacy)
//!
//! Each record is exactly 64 bytes, cache-line aligned:
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 8 | sequence (u64) |
//! | 8 | 8 | timestamp_ns (u64) |
//! | 16 | 1 | action_kind (u8) |
//! | 17 | 1 | proof_tier (u8) |
//! | 18 | 2 | flags (u16) |
//! | 20 | 4 | actor_partition_id (u32) |
//! | 24 | 4 | target_object_id (u32) |
//! | 28 | 4 | capability_hash (u32) |
//! | 32 | 8 | payload (u64) |
//! | 40 | 8 | prev_hash (u64) |
//! | 48 | 8 | record_hash (u64) |
//! | 56 | 8 | aux (u64) |
//!
//! The v2 layout (96 bytes, version byte `2` at offset 19, full-width
//! `prev_mac`/`chain_mac`) is documented on `rvm_types::WitnessRecordV2`.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod emit;
mod hash;
mod log;
mod record;
mod replay;
pub mod seal;
mod signer;
mod v2;
mod versioned;

pub use emit::WitnessEmitter;
pub use hash::{fnv1a_64, compute_chain_hash, compute_record_hash};
pub use log::WitnessLog;
pub use record::{ActionKind, WitnessRecord};
pub use replay::{
    ChainIntegrityError, verify_chain, query_by_partition, query_by_action_kind,
    query_by_time_range,
};
pub use seal::{
    Blake3SealSigner, MerkleProof, SealChainError, SealedSegment, SegmentAccumulator,
    SegmentSealSigner, seal_chain_genesis, seal_digest, seal_digest_chained, verify_inclusion,
    verify_seal, verify_seal_chain, verify_seal_chain_binding, verify_seal_chain_binding_from,
    verify_seal_chain_from, DEFAULT_SEGMENT_SIZE, MAX_MERKLE_DEPTH, SEAL_VERSION_CHAINED,
    SEAL_VERSION_UNCHAINED,
};
pub use v2::{
    CHAIN_KEY_CONTEXT, CoverageError, CoveragePolicy, RATCHET_CONTEXT, WitnessLogV2,
    compute_chain_mac_v2, default_chain_key, derive_chain_key, erase_key, ratchet_chain_key,
};
pub use versioned::{
    ChainIntegrityErrorV2, LogVerifyError, LogVerifySummary, v1_head_to_genesis,
    verify_chain_v1_with_head, verify_chain_v2, verify_chain_v2_from,
    verify_chain_v2_ratcheted, verify_log_bytes,
};
#[cfg(any(test, feature = "null-signer"))]
#[allow(deprecated)]
pub use signer::NullSigner;
pub use signer::{DefaultSigner, StrictSigner, WitnessSigner, default_signer};
#[cfg(feature = "crypto-sha256")]
pub use signer::{HmacWitnessSigner, record_to_digest};

/// Default ring buffer capacity: 262,144 records (16 MB / 64 bytes).
pub const DEFAULT_RING_CAPACITY: usize = 262_144;
