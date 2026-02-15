//! Core types for the RuVector Format (RVF).
//!
//! This crate provides the foundational types shared across all RVF crates:
//! segment headers, type enums, flags, error codes, and format constants.
//!
//! All types are `no_std` compatible by default.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
extern crate alloc;

pub mod checksum;
pub mod compression;
pub mod constants;
pub mod cow_map;
pub mod data_type;
pub mod delta;
pub mod ebpf;
pub mod error;
pub mod filter;
pub mod flags;
pub mod kernel;
pub mod kernel_binding;
pub mod manifest;
pub mod membership;
pub mod profile;
pub mod quant_type;
pub mod refcount;
pub mod segment;
pub mod segment_type;
pub mod signature;
pub mod attestation;
pub mod lineage;
pub mod quality;
pub mod qr_seed;
pub mod security;
pub mod wasm_bootstrap;

pub use attestation::{AttestationHeader, AttestationWitnessType, TeePlatform, KEY_TYPE_TEE_BOUND};
pub use ebpf::{
    EbpfAttachType, EbpfHeader, EbpfProgramType, EBPF_MAGIC,
};
pub use kernel::{
    ApiTransport, KernelArch, KernelHeader, KernelType, KERNEL_MAGIC,
    KERNEL_FLAG_SIGNED, KERNEL_FLAG_COMPRESSED, KERNEL_FLAG_REQUIRES_TEE,
    KERNEL_FLAG_MEASURED, KERNEL_FLAG_REQUIRES_KVM, KERNEL_FLAG_REQUIRES_UEFI,
    KERNEL_FLAG_HAS_NETWORKING, KERNEL_FLAG_HAS_QUERY_API, KERNEL_FLAG_HAS_INGEST_API,
    KERNEL_FLAG_HAS_ADMIN_API, KERNEL_FLAG_ATTESTATION_READY, KERNEL_FLAG_RELOCATABLE,
    KERNEL_FLAG_HAS_VIRTIO_NET, KERNEL_FLAG_HAS_VIRTIO_BLK, KERNEL_FLAG_HAS_VSOCK,
};
pub use lineage::{
    DerivationType, FileIdentity, LineageRecord, LINEAGE_RECORD_SIZE,
    WITNESS_DERIVATION, WITNESS_LINEAGE_MERGE, WITNESS_LINEAGE_SNAPSHOT,
    WITNESS_LINEAGE_TRANSFORM, WITNESS_LINEAGE_VERIFY,
};
pub use cow_map::{CowMapEntry, CowMapHeader, MapFormat, COWMAP_MAGIC};
pub use delta::{DeltaEncoding, DeltaHeader, DELTA_MAGIC};
pub use kernel_binding::KernelBinding;
pub use membership::{FilterMode, FilterType, MembershipHeader, MEMBERSHIP_MAGIC};
pub use refcount::{RefcountHeader, REFCOUNT_MAGIC};
pub use checksum::ChecksumAlgo;
pub use compression::CompressionAlgo;
pub use constants::*;
pub use data_type::DataType;
pub use error::{ErrorCode, RvfError};
pub use filter::FilterOp;
pub use flags::SegmentFlags;
pub use manifest::{
    CentroidPtr, EntrypointPtr, HotCachePtr, Level0Root, PrefetchMapPtr, QuantDictPtr, TopLayerPtr,
};
pub use profile::{DomainProfile, ProfileId};
pub use quant_type::QuantType;
pub use segment::SegmentHeader;
pub use segment_type::SegmentType;
pub use signature::{SignatureAlgo, SignatureFooter};
pub use quality::{
    BudgetReport, BudgetType, DegradationReason, DegradationReport, FallbackPath,
    IndexLayersUsed, QualityPreference, ResponseQuality, RetrievalQuality,
    SafetyNetBudget, SearchEvidenceSummary, derive_response_quality,
};
pub use qr_seed::{
    HostEntry, LayerEntry, SeedHeader, SEED_MAGIC, QR_MAX_BYTES,
    SEED_HEADER_SIZE, SEED_HAS_MICROKERNEL, SEED_HAS_DOWNLOAD,
    SEED_SIGNED, SEED_OFFLINE_CAPABLE, SEED_ENCRYPTED, SEED_COMPRESSED,
    SEED_HAS_VECTORS, SEED_STREAM_UPGRADE,
};
pub use security::{HardeningFields, SecurityError, SecurityPolicy};
pub use wasm_bootstrap::{
    WasmHeader, WasmRole, WasmTarget, WASM_MAGIC,
    WASM_FEAT_SIMD, WASM_FEAT_BULK_MEMORY, WASM_FEAT_MULTI_VALUE,
    WASM_FEAT_REFERENCE_TYPES, WASM_FEAT_THREADS, WASM_FEAT_TAIL_CALL,
    WASM_FEAT_GC, WASM_FEAT_EXCEPTION_HANDLING,
};
