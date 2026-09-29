//! Strict RVF validation over untrusted bytes (ADR-351 §3 rv-registry: "fuzzed
//! streaming `rvf-wire` validator").
//!
//! [`StreamValidator`] accepts the upload in arbitrary chunks (R2 multipart
//! parts, a `ReadableStream`) and never holds more than the limits allow;
//! [`validate`] is the one-shot wrapper. Headers are parsed with
//! `rvf_wire::read_segment_header` and the root page with
//! `rvf_wire::manifest_codec::read_root_manifest`; every length is checked in
//! `u64` (`rvf_wire::read_segment` is deliberately not used: its
//! `payload_length as usize` truncates on wasm32).
//!
//! A registry package is a **vector store**: the stream must carry at least
//! one MANIFEST_SEG in the rvf runtime layout (what the `rvf` CLI writes). The
//! last manifest is authoritative for dimension, metric and the live segment
//! directory; every manifest's directory must name segments seen before it at
//! exactly the recorded offset, id, length and type. VEC_SEGs must match the
//! runtime's flat layout exactly, and live ones must match the manifest
//! dimension. Executable (kernel/eBPF/WASM), compressed and unknown segment
//! types are refused.

mod error;
mod hasher;
mod layout;
mod stream;

pub use error::ValidationError;
pub use layout::VecSegView;
pub use stream::StreamValidator;

use crate::hexser;
use layout::RuntimeManifest;
use ruvector_edge_store::Metric;
use rvf_types::SegmentType;
use serde::{Deserialize, Serialize};

/// Validator limits. Defaults suit a Worker isolate (128 MB) streaming from R2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationLimits {
    /// Maximum object size.
    pub max_total_bytes: u64,
    /// Maximum number of segments (segment-bomb bound).
    pub max_segments: u32,
    /// Maximum declared payload of one segment.
    pub max_segment_payload: u64,
    /// Maximum MANIFEST_SEG payload (buffered in memory).
    pub max_manifest_payload: u32,
    /// Maximum signature length in a `SIGNED` footer.
    pub max_signature_len: u16,
    /// Maximum vector dimension.
    pub max_dimension: u16,
    /// Accept kernel / eBPF / WASM segments (never executed either way).
    pub allow_executable: bool,
}

impl Default for ValidationLimits {
    fn default() -> Self {
        ValidationLimits {
            max_total_bytes: 512 << 20,
            max_segments: 4096,
            max_segment_payload: 256 << 20,
            max_manifest_payload: 4 << 20,
            max_signature_len: 4096,
            max_dimension: 4096,
            allow_executable: false,
        }
    }
}

/// One validated segment, as recorded in the package manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentEntry {
    /// Position in the stream.
    pub index: u32,
    /// Raw `SegmentType` byte.
    pub seg_type: u8,
    /// Header `segment_id`.
    pub segment_id: u64,
    /// Byte offset of the header in the object.
    pub offset: u64,
    /// Declared payload length.
    pub payload_length: u64,
    /// Header + payload + footer (padding excluded).
    pub length: u64,
    /// Header `checksum_algo` (verified).
    pub checksum_algo: u8,
    /// `SIGNED` flag set (footer shape checked; signature not verified).
    pub signed: bool,
    /// SHA-256 of header + payload + footer.
    #[serde(with = "hexser")]
    pub sha256: [u8; 32],
}

impl SegmentEntry {
    /// Byte range of the payload inside the object.
    pub fn payload_range(&self) -> core::ops::Range<u64> {
        let start = self.offset + 64;
        start..start + self.payload_length
    }
}

/// The result of a successful validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedRvf {
    /// Object size in bytes.
    pub total_size: u64,
    /// SHA-256 of the whole object (its content address).
    pub sha256: [u8; 32],
    /// Every segment, in stream order.
    pub segments: Vec<SegmentEntry>,
    /// Vector dimension (last manifest).
    pub dim: u16,
    /// Distance metric (last manifest byte 19).
    pub metric: Metric,
    /// `total_vectors` from the last manifest, checked to be at most the
    /// number of vector records in the live VEC_SEGs.
    pub total_vectors: u64,
    /// Manifest epoch.
    pub epoch: u32,
    /// Indices into `segments` of the live VEC_SEGs (last manifest
    /// directory), strictly ascending: each segment at most once.
    pub live_vec_segments: Vec<u32>,
    /// Deleted vector ids from the last manifest, sorted and deduplicated.
    pub deleted_ids: Vec<u64>,
    /// A trailing Level-0 root page was present and valid.
    pub root_page: bool,
}

impl ValidatedRvf {
    /// Decode every live vector from the full object bytes, **sorted by id**:
    /// live VEC_SEGs are read in directory order, deleted ids removed, and a
    /// later record for the same id replaces an earlier one. For small
    /// packages and tests; a Worker
    /// range-reads each [`SegmentEntry::payload_range`] and uses
    /// [`VecSegView`] directly.
    pub fn live_vectors(&self, object: &[u8]) -> Result<Vec<(u64, Vec<f32>)>, ValidationError> {
        let mut out: std::collections::BTreeMap<u64, Vec<f32>> = Default::default();
        for &i in &self.live_vec_segments {
            let seg = &self.segments[i as usize];
            let r = seg.payload_range();
            let payload = usize::try_from(r.start)
                .ok()
                .zip(usize::try_from(r.end).ok())
                .and_then(|(s, e)| object.get(s..e))
                .ok_or(ValidationError::Truncated {
                    offset: seg.offset,
                    what: "segment payload",
                })?;
            for (id, v) in VecSegView::decode(payload, self.dim)? {
                if self.deleted_ids.binary_search(&id).is_err() {
                    out.insert(id, v);
                }
            }
        }
        Ok(out.into_iter().collect())
    }
}

#[doc(hidden)]
/// Direct entry points into the payload parsers for the second fuzz target
/// (`fuzz/fuzz_targets/rvf_payloads.rs`). Not a stable API.
pub mod fuzz_api {
    use super::layout;
    use super::ValidationError;

    /// What `parse_manifest` accepted, reduced to the checked invariants.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ManifestSummary {
        /// Directory offsets, in directory order.
        pub directory_offsets: Vec<u64>,
        /// Deleted ids listed.
        pub deleted: usize,
        /// Declared dimension.
        pub dim: u16,
        /// Declared `total_vectors`.
        pub total_vectors: u64,
    }

    /// Parse a MANIFEST_SEG payload with a directory bound.
    pub fn parse_manifest(p: &[u8], max_entries: u64) -> Result<ManifestSummary, ValidationError> {
        let m = layout::parse_manifest(p, 0, max_entries)?;
        Ok(ManifestSummary {
            directory_offsets: m.directory.iter().map(|d| d.offset).collect(),
            deleted: m.deleted_ids.len(),
            dim: m.dim,
            total_vectors: m.total_vectors,
        })
    }

    /// Check a VEC_SEG prefix against a payload length.
    pub fn check_vec_head(
        head: &[u8; layout::VEC_HEAD],
        payload_length: u64,
    ) -> Result<(u16, u32), ValidationError> {
        layout::check_vec_head(head, payload_length, 0)
    }
}

/// Validate a complete object in one call.
pub fn validate(bytes: &[u8], limits: ValidationLimits) -> Result<ValidatedRvf, ValidationError> {
    let mut v = StreamValidator::new(limits);
    v.push(bytes)?;
    v.finish()
}

/// Metric byte of the runtime manifest: 0 = L2, 1 = inner product, 2 = cosine.
pub fn metric_from_rvf_id(id: u8) -> Option<Metric> {
    match id {
        0 => Some(Metric::L2),
        1 => Some(Metric::Dot),
        2 => Some(Metric::Cosine),
        _ => None,
    }
}

/// Derive the summary once the stream ended cleanly.
pub(crate) fn finish_summary(
    limits: &ValidationLimits,
    total_size: u64,
    sha256: [u8; 32],
    segments: Vec<SegmentEntry>,
    vec_heads: &[(u32, u16, u32)],
    manifest: Option<(u64, RuntimeManifest)>,
    root_dim: Option<u16>,
) -> Result<ValidatedRvf, ValidationError> {
    if segments.is_empty() {
        return Err(ValidationError::NoSegments);
    }
    let (manifest_offset, m) = manifest.ok_or(ValidationError::NoManifest)?;
    if m.dim == 0 || m.dim > limits.max_dimension {
        return Err(ValidationError::BadDimension { dim: m.dim });
    }
    if let Some(found) = root_dim.filter(|d| *d != 0 && *d != m.dim) {
        return Err(ValidationError::DimensionMismatch {
            expected: m.dim,
            found,
        });
    }
    let metric =
        metric_from_rvf_id(m.metric_id).ok_or(ValidationError::UnknownMetric(m.metric_id))?;
    let mut live_vec_segments = Vec::new();
    let mut live_capacity = 0u64;
    for d in m
        .directory
        .iter()
        .filter(|d| d.seg_type == SegmentType::Vec as u8)
    {
        // The directory was cross-checked against `segments` when parsed.
        let idx = segments
            .binary_search_by_key(&d.offset, |s| s.offset)
            .map_err(|_| ValidationError::DirectoryMismatch {
                offset: d.offset,
                target: d.offset,
            })? as u32;
        // `vec_heads` is in stream order, hence sorted by index.
        let (_, dim, count) = vec_heads
            .binary_search_by_key(&idx, |(i, _, _)| *i)
            .map(|at| vec_heads[at])
            .map_err(|_| ValidationError::VecSegMalformed {
                offset: d.offset,
                reason: "missing prefix",
            })?;
        if dim != m.dim {
            return Err(ValidationError::DimensionMismatch {
                expected: m.dim,
                found: dim,
            });
        }
        live_capacity += u64::from(count);
        live_vec_segments.push(idx);
    }
    // Directory offsets are strictly increasing (checked at parse), so the
    // live set holds each VEC_SEG at most once and `live_capacity` is the
    // number of vector records the object actually carries.
    if m.total_vectors > live_capacity {
        return Err(ValidationError::MalformedManifest {
            offset: manifest_offset,
            reason: "total_vectors exceeds the live vector records",
        });
    }
    let mut deleted_ids = m.deleted_ids;
    deleted_ids.sort_unstable();
    deleted_ids.dedup();
    Ok(ValidatedRvf {
        total_size,
        sha256,
        segments,
        dim: m.dim,
        metric,
        total_vectors: m.total_vectors,
        epoch: m.epoch,
        live_vec_segments,
        deleted_ids,
        root_page: root_dim.is_some(),
    })
}
