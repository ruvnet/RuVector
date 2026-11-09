//! Configuration types for the RVF runtime.

use crate::filter::FilterExpr;
use rvf_types::quality::{
    BudgetReport, DegradationReport, QualityPreference, ResponseQuality, SafetyNetBudget,
    SearchEvidenceSummary,
};
use rvf_types::security::SecurityPolicy;
use std::sync::{atomic::AtomicBool, Arc};

/// Distance metric used for vector similarity search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Squared Euclidean distance (L2).
    #[default]
    L2,
    /// Inner (dot) product distance (negated).
    InnerProduct,
    /// Cosine distance (1 - cosine_similarity).
    Cosine,
}

impl DistanceMetric {
    /// Encode this metric as a single byte for manifest persistence.
    ///
    /// Encoding: 0 = L2 (default / backward-compatible), 1 = InnerProduct, 2 = Cosine.
    /// Old manifests written before this field existed have 0x00 at that byte
    /// (it was a reserved zero), so they boot correctly as L2.
    pub(crate) fn to_id(self) -> u8 {
        match self {
            DistanceMetric::L2 => 0,
            DistanceMetric::InnerProduct => 1,
            DistanceMetric::Cosine => 2,
        }
    }

    /// Decode a metric from a manifest byte.
    ///
    /// Unknown values fall back to L2 for forward-compatibility: a store
    /// written by a newer version with an unknown metric ID is treated as
    /// L2-distance, which is at least type-safe even if not semantically
    /// correct.
    pub(crate) fn from_id(id: u8) -> Self {
        match id {
            1 => DistanceMetric::InnerProduct,
            2 => DistanceMetric::Cosine,
            _ => DistanceMetric::L2,
        }
    }
}

/// Compression profile for stored vectors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompressionProfile {
    /// No compression — raw fp32 vectors.
    #[default]
    None,
    /// Scalar quantization (int8).
    Scalar,
    /// Product quantization.
    Product,
}

/// Configuration for automatic witness segment generation.
#[derive(Clone, Debug)]
pub struct WitnessConfig {
    /// Append a witness entry after each ingest operation. Default: true.
    pub witness_ingest: bool,
    /// Append a witness entry after each delete operation. Default: true.
    pub witness_delete: bool,
    /// Append a witness entry after each compact operation. Default: true.
    pub witness_compact: bool,
    /// Append a witness entry after each query operation. Default: false.
    /// Enable this for audit-trail compliance; it adds I/O to the hot path.
    pub audit_queries: bool,
}

impl Default for WitnessConfig {
    fn default() -> Self {
        Self {
            witness_ingest: true,
            witness_delete: true,
            witness_compact: true,
            audit_queries: false,
        }
    }
}

/// How a signature-checking open decides whether the signed state is fresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Freshness {
    /// The caller accepts any validly signed state, including an older one
    /// (authenticity only; rollback to an earlier signed state is undetected).
    Unchecked,
    /// Refuse a signed state whose manifest epoch is below this value
    /// (`StreamStaleEpoch`).
    MinEpoch(u32),
}

/// Options for creating a new RVF store.
#[derive(Clone, Debug)]
pub struct RvfOptions {
    /// Vector dimensionality (required).
    pub dimension: u16,
    /// Distance metric for similarity search.
    pub metric: DistanceMetric,
    /// Hardware profile identifier (0=Generic, 1=Core, 2=Hot, 3=Full).
    pub profile: u8,
    /// Domain profile for the file (determines canonical extension).
    pub domain_profile: rvf_types::DomainProfile,
    /// Compression profile for stored vectors.
    pub compression: CompressionProfile,
    /// Whether segment signing is enabled.
    pub signing: bool,
    /// HNSW M parameter: max edges per node per layer.
    pub m: u16,
    /// HNSW ef_construction: beam width during index build.
    pub ef_construction: u16,
    /// Witness auto-generation configuration.
    pub witness: WitnessConfig,
    /// Security policy for manifest signature verification (ADR-033 §4).
    pub security_policy: SecurityPolicy,
    /// Ed25519 public keys the open path trusts for Level-0 root signatures
    /// (ADR-033 section 4). Consulted by `open_with_policy`/`open_with_options`/
    /// `open_signed`; with an empty list every signature-checking policy
    /// refuses signed files as `UnknownSigner`.
    pub trusted_signers: Vec<[u8; 32]>,
    /// Freshness choice (F3-02, F4-04). REQUIRED for every signature-checking
    /// open (`Strict`/`Paranoid`): `None` is refused with
    /// `ErrorCode::FreshnessRequired`. A signed page proves authenticity, not
    /// freshness; only `Freshness::MinEpoch` (or an external pin) detects
    /// rollback to an older validly signed state.
    pub freshness: Option<Freshness>,
    /// Serve the signed state even though a newer parseable manifest exists
    /// beyond the signed page (an unsigned tail: an interrupted commit, a
    /// planted manifest, or a rollback by page swap). ANY bytes between the signed
    /// manifest and the page count, parseable or not (a damaged newer manifest
    /// must not hide itself).
    /// opens and reports the ignored manifest through
    /// `RvfStore::newer_manifest_ignored`.
    pub serve_signed_prefix: bool,
}

impl Default for RvfOptions {
    fn default() -> Self {
        Self {
            dimension: 0,
            metric: DistanceMetric::L2,
            profile: 0,
            domain_profile: rvf_types::DomainProfile::Generic,
            compression: CompressionProfile::None,
            signing: false,
            m: 16,
            ef_construction: 200,
            witness: WitnessConfig::default(),
            security_policy: SecurityPolicy::Strict,
            trusted_signers: Vec::new(),
            freshness: None,
            serve_signed_prefix: false,
        }
    }
}

/// Options controlling a query operation.
#[derive(Clone, Debug)]
pub struct QueryOptions {
    /// HNSW ef_search parameter (beam width during search).
    pub ef_search: u16,
    /// Optional metadata filter expression.
    ///
    /// When the filter matches a large share of the store, a filtered query
    /// runs a predicate-aware HNSW walk, but only if the graph itself
    /// measures at least 0.95 recall@10 at the query's `ef` (a one-off
    /// calibration against the exact scan, cached on the index). Graphs
    /// that measure lower (e.g. high intrinsic-dimension data) and small
    /// match sets are scanned exactly. `QualityPreference::PreferLatency` /
    /// `AcceptDegraded` skip the calibration gate and always take the walk.
    /// Set [`QueryOptions::force_exact`] to always get exact results.
    pub filter: Option<FilterExpr>,
    /// Query timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u32,
    /// Quality vs latency preference (ADR-033).
    pub quality_preference: QualityPreference,
    /// Safety net budget caps. Callers may tighten but not loosen
    /// beyond the mode default (unless PreferQuality, which extends to 4x).
    pub safety_net_budget: SafetyNetBudget,
    /// Force the exact brute-force scan even when an HNSW index is
    /// available. Useful for ground-truth comparison and benchmarking.
    pub force_exact: bool,
    /// Opt in to the RaBitQ two-stage path: a 1-bit-code candidate scan
    /// (~32x smaller than f32) followed by an exact f32 rescore of the
    /// oversampled candidates. Default `false` (full-precision HNSW /
    /// exact scan). Serves L2, cosine and inner product; filtered/COW
    /// queries fall back to the default routing.
    pub rabitq: bool,
    /// Candidate oversampling factor for the RaBitQ first stage: the
    /// binary scan collects `rabitq_oversample * k` candidates (floored
    /// at an internal minimum pool size that keeps recall@10 >= 0.95 on
    /// the 10k x 128 benchmark) before the exact rescore. Values below 1
    /// are treated as 1.
    pub rabitq_oversample: u16,
    /// Maximum records evaluated by linear metadata fallback.
    pub metadata_filter_max_records: u64,
    /// Maximum estimated decoded metadata bytes evaluated by fallback.
    pub metadata_filter_max_bytes: u64,
    /// Cooperative cancellation flag checked between metadata records.
    pub metadata_filter_cancel: Option<Arc<AtomicBool>>,
    /// Return each hit's committed metadata in [`SearchResult::metadata`]
    /// (ADR-280 §6). Default `false`: search returns identifiers and
    /// distance only, avoiding metadata decode and allocation. When `true`,
    /// the metadata is resolved from the same committed snapshot that
    /// produced the hit list, so COW field overrides and record tombstones
    /// are reflected exactly as [`RvfStore::get_metadata`] would report them.
    pub include_metadata: bool,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            ef_search: 100,
            filter: None,
            timeout_ms: 0,
            quality_preference: QualityPreference::Auto,
            safety_net_budget: SafetyNetBudget::LAYER_A,
            force_exact: false,
            rabitq: false,
            rabitq_oversample: 4,
            metadata_filter_max_records: 1_000_000,
            metadata_filter_max_bytes: 256 * 1024 * 1024,
            metadata_filter_cancel: None,
            include_metadata: false,
        }
    }
}

/// A single search result: vector ID and distance.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// The vector's unique identifier.
    pub id: u64,
    /// Distance from the query vector (lower = more similar).
    pub distance: f32,
    /// Per-candidate retrieval quality (ADR-033).
    pub retrieval_quality: rvf_types::quality::RetrievalQuality,
    /// Committed metadata for this hit, populated only when the query set
    /// [`QueryOptions::include_metadata`] (ADR-280 §6). `None` when metadata
    /// was not requested; `Some(vec![])` for a hit that has a present but
    /// empty metadata record. Otherwise `None` also means the hit has no
    /// metadata record at all.
    pub metadata: Option<Vec<MetadataEntry>>,
}

/// The mandatory outer return type for all query APIs (ADR-033 §2.4).
///
/// This is not optional. This is not a nested field.
/// JSON flattening cannot discard it. gRPC serialization cannot drop it.
/// MCP tool responses must include it.
#[derive(Clone, Debug)]
pub struct QualityEnvelope {
    /// The search results.
    pub results: Vec<SearchResult>,
    /// Top-level quality signal. Consumers MUST inspect this.
    pub quality: ResponseQuality,
    /// Structured evidence for why the quality is what it is.
    pub evidence: SearchEvidenceSummary,
    /// Resource consumption report for this query.
    pub budgets: BudgetReport,
    /// If quality is degraded, the structured reason.
    pub degradation: Option<DegradationReport>,
}

/// Result of a batch ingest operation.
#[derive(Clone, Debug)]
pub struct IngestResult {
    /// Number of vectors successfully ingested.
    pub accepted: u64,
    /// Number of vectors rejected.
    pub rejected: u64,
    /// Manifest epoch after the ingest commit.
    pub epoch: u32,
}

/// Result of a delete operation.
#[derive(Clone, Debug)]
pub struct DeleteResult {
    /// Number of vectors soft-deleted.
    pub deleted: u64,
    /// Manifest epoch after the delete commit.
    pub epoch: u32,
}

/// Result of a compaction operation.
#[derive(Clone, Debug)]
pub struct CompactionResult {
    /// Number of segments compacted.
    pub segments_compacted: u32,
    /// Bytes of dead space reclaimed.
    pub bytes_reclaimed: u64,
    /// Manifest epoch after compaction commit.
    pub epoch: u32,
}

/// A single metadata entry for a vector.
#[derive(Clone, Debug, PartialEq)]
pub struct MetadataEntry {
    /// Metadata field identifier.
    pub field_id: u16,
    /// The metadata value.
    pub value: MetadataValue,
}

/// Metadata value types matching the spec.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    Null,
    U64(u64),
    I64(i64),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
    Bool(bool),
    /// Delta-only marker that removes an inherited field.
    DeleteField,
}

/// Metadata associated with one vector. Unlike the legacy flattened
/// `ingest_batch` argument this supports unequal field counts.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorMetadata {
    pub vector_id: u64,
    pub fields: Vec<MetadataEntry>,
    pub delete_record: bool,
}

/// Compatibility state derived from the artifact's embedding-space identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingCompatibility {
    /// No expected identity was supplied by the caller.
    Unchecked,
    /// The expected identity matches the artifact.
    Compatible,
    /// Vector-only reads remain available; corpus mutation is disabled.
    VectorReadOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetadataFilterStats {
    pub scanned_records: u64,
    pub scanned_bytes: u64,
    pub aborted_scans: u64,
}

/// Outcome of replaying the committed `META_SEG` chain at open (ADR-280 §5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetadataRecovery {
    /// Newest metadata generation the served snapshot reflects.
    pub generation: u64,
    /// Generations that could not be replayed and were dropped. Non-zero means
    /// the artifact is damaged and later generations were discarded.
    pub dropped_generations: u64,
    /// Records the truncated replay resurrected -- their vectors are deleted or
    /// absent in the committed vector state, because a dropped generation
    /// carried the deletion. They are excluded from the served metadata.
    pub dropped_records: u64,
}

/// What follows the manifest `open` selected (crash-recovery evidence).
///
/// Open serves the newest valid manifest and ignores later bytes: that is the
/// ADR-280 crash rule, so a torn append recovers to the prior snapshot. The
/// same bytes also result from a file cut short in transit, which then looks
/// like an older -- possibly empty -- generation (#1276). The library cannot
/// tell the two apart, so it reports what it ignored; strict callers use
/// [`crate::RvfStore::open_readonly_strict`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TailRecovery {
    /// Offset of a segment cut off by end-of-file (a header whose payload runs
    /// past EOF, or a partial header), if any. A healthy file never has one.
    pub torn_at: Option<u64>,
    /// Complete non-manifest segments after the selected manifest that no
    /// manifest commits (e.g. data whose commit record was lost). May also be
    /// non-zero in a healthy file (an unsigned audit-query witness).
    pub uncommitted_segments: u32,
    /// Bytes from the end of the selected manifest to the end of the file.
    pub bytes_after_manifest: u64,
    /// A writable open cut the torn tail off (at `torn_at`) before appending,
    /// so the file is clean again. Read-only opens never modify the file.
    pub repaired: bool,
}

impl TailRecovery {
    /// True when the file ends inside a segment: a torn write or truncation.
    pub fn is_torn(&self) -> bool {
        self.torn_at.is_some()
    }
}
