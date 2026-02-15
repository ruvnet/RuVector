//! RuVector Format runtime — the main user-facing API.
//!
//! This crate provides [`RvfStore`], the primary interface for creating,
//! opening, querying, and managing RVF vector stores. It ties together
//! the segment model, manifest system, HNSW indexing, quantization, and
//! compaction into a single cohesive runtime.
//!
//! # Architecture
//!
//! - **Append-only writes**: All mutations append new segments; no in-place edits.
//! - **Progressive boot**: Readers see results before the full file is loaded.
//! - **Single-writer / multi-reader**: Advisory lock file enforces exclusivity.
//! - **Background compaction**: Dead space is reclaimed without blocking queries.

pub mod adversarial;
pub mod compaction;
pub mod cow;
pub mod cow_compact;
pub mod cow_map;
pub mod deletion;
pub mod dos;
pub mod filter;
pub mod locking;
pub mod membership;
pub mod options;
pub mod qr_seed;
pub mod read_path;
pub mod safety_net;
pub mod status;
pub mod store;
pub mod write_path;

pub use adversarial::{
    adaptive_n_probe, centroid_distance_cv, combined_effective_n_probe,
    effective_n_probe_with_drift, is_degenerate_distribution, DEGENERATE_CV_THRESHOLD,
};
pub use cow::{CowEngine, CowStats, WitnessEvent};
pub use cow_compact::CowCompactor;
pub use cow_map::CowMap;
pub use dos::{BudgetTokenBucket, NegativeCache, ProofOfWork, QuerySignature};
pub use filter::FilterExpr;
pub use membership::MembershipFilter;
pub use options::{
    CompactionResult, DeleteResult, IngestResult, MetadataEntry, MetadataValue, QueryOptions,
    QualityEnvelope, RvfOptions, SearchResult, WitnessConfig,
};
pub use qr_seed::{
    BootstrapProgress, DownloadManifest, ParsedSeed, SeedBuilder, SeedError,
    make_host_entry,
};
pub use safety_net::{
    selective_safety_net_scan, should_activate_safety_net, Candidate, SafetyNetResult,
};
pub use status::StoreStatus;
pub use store::RvfStore;
