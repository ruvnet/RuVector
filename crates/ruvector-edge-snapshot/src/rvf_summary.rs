//! Import limits and the tail summary of an `.rvf` file.
//!
//! The rvf-runtime manifest is written last, so the dimension, metric, live
//! `VEC_SEG` directory and deletions are only known at the tail. Every import
//! starts by range-reading the tail and calling [`inspect_tail`]; the
//! resulting [`RvfSummary`] is **required** by [`crate::RvfImporter`], so
//! dimension, metric and quota are refused before any batch exists.

use crate::error::ImportError;
use crate::rvf_format::{self as fmt, RvfManifest, SEG_MANIFEST, SEG_VEC};
use crate::types::{sha256, Metric};

/// Directory entries a manifest may carry by default: as many 25-byte
/// entries as fit in one default-size segment payload.
pub const DEFAULT_MAX_MANIFEST_ENTRIES: usize = fmt::MAX_SEGMENT_PAYLOAD / 25;

/// Import limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportLimits {
    /// Maximum upload size.
    pub max_file_bytes: u64,
    /// Maximum payload of any single segment.
    pub max_segment_payload: u64,
    /// Upsert operations this job may perform: the tenant's remaining row
    /// quota **at job start** (the job pins it, see [`crate::ImportJob::deliver`]).
    /// Compared with the cumulative [`crate::ImportCursor::rows_done`].
    pub max_rows: u64,
    /// Rows per upsert batch (ADR-351 §7: ≤ 500).
    pub max_batch_rows: usize,
    /// Maximum manifest directory / deletion entries.
    pub max_manifest_entries: usize,
}

impl Default for ImportLimits {
    fn default() -> Self {
        ImportLimits {
            max_file_bytes: 1 << 30,
            max_segment_payload: fmt::MAX_SEGMENT_PAYLOAD as u64,
            max_rows: u64::MAX,
            max_batch_rows: 500,
            max_manifest_entries: DEFAULT_MAX_MANIFEST_ENTRIES,
        }
    }
}

/// What the tail manifest says about a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RvfSummary {
    /// Dimension.
    pub dim: u16,
    /// Metric.
    pub metric: Metric,
    /// Runtime `total_vectors` (the store's live vector count; informative
    /// only — it is not trusted for quota).
    pub total_vectors: u64,
    /// Offset of the final manifest segment.
    pub manifest_offset: u64,
    /// File length.
    pub file_len: u64,
    /// sha256 of the manifest payload.
    pub manifest_hash: [u8; 32],
    /// Live `VEC_SEG`s as `(offset, payload_len)`, sorted by offset. The
    /// importer checks every live segment header against its entry.
    pub vec_segs: Vec<(u64, u64)>,
    /// Exact rows held by the live `VEC_SEG`s (derived from the verified
    /// payload lengths): the upper bound on upsert operations.
    pub planned_rows: u64,
    /// Deleted runtime ids (sorted).
    pub deleted: Vec<u64>,
}

impl RvfSummary {
    /// Estimated upsert operations: `planned_rows − deletions` (exact when
    /// each deleted id occurs once in the live segments). Used for the
    /// up-front quota refusal; the streaming count stays the backstop.
    pub fn estimated_upserts(&self) -> u64 {
        self.planned_rows.saturating_sub(self.deleted.len() as u64)
    }
}

/// Parse the final manifest from the last `tail.len()` bytes of a
/// `file_len`-byte file. The manifest (plus its rvf-wire alignment padding,
/// if any) must end the file.
pub fn inspect_tail(
    tail: &[u8],
    file_len: u64,
    limits: &ImportLimits,
) -> Result<RvfSummary, ImportError> {
    if tail.len() as u64 > file_len || file_len > limits.max_file_bytes {
        return Err(ImportError::FileTooLarge);
    }
    let base = file_len - tail.len() as u64;
    let magic = rvf_types::SEGMENT_MAGIC.to_le_bytes();
    let mut i = tail.len().saturating_sub(fmt::HEADER);
    loop {
        if tail[i..].starts_with(&magic) && tail.get(i + 5) == Some(&SEG_MANIFEST) {
            if let Ok(h) = fmt::parse_header(&tail[i..]) {
                let plen = h.payload_length;
                let end = (i as u64 + fmt::HEADER as u64)
                    .checked_add(plen)
                    .and_then(|e| e.checked_add(u64::from(h.alignment_pad)));
                if end == Some(tail.len() as u64) {
                    // Size refusal before hashing or decoding anything.
                    if plen > limits.max_segment_payload {
                        return Err(ImportError::SegmentTooLarge(plen));
                    }
                    let start = i + fmt::HEADER;
                    let payload = &tail[start..start + plen as usize];
                    if !fmt::payload_hash_ok(&h, payload) {
                        return Err(ImportError::Checksum(base + i as u64));
                    }
                    return summarize(payload, base + i as u64, file_len, limits);
                }
            }
        }
        if i == 0 {
            return Err(ImportError::NoManifest);
        }
        i -= 1;
    }
}

fn summarize(
    payload: &[u8],
    offset: u64,
    file_len: u64,
    limits: &ImportLimits,
) -> Result<RvfSummary, ImportError> {
    let m = RvfManifest::decode(payload, limits.max_manifest_entries)
        .map_err(ImportError::Malformed)?;
    let metric = Metric::from_rvf_id(m.metric_id).ok_or(ImportError::Malformed("metric"))?;
    let stride = 8 + 4 * u64::from(m.dim);
    let mut vec_segs = Vec::new();
    let mut planned_rows = 0u64;
    for e in m.dir.iter().filter(|e| e.seg_type == SEG_VEC) {
        let rows_bytes = e.payload_len.checked_sub(6);
        let ok = e.offset < offset
            && e.payload_len <= limits.max_segment_payload
            && rows_bytes.is_some_and(|b| b % stride == 0);
        if !ok {
            return Err(ImportError::Malformed("directory entry"));
        }
        planned_rows += rows_bytes.unwrap_or(0) / stride;
        vec_segs.push((e.offset, e.payload_len));
    }
    vec_segs.sort_unstable();
    if vec_segs.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(ImportError::Malformed("duplicate directory offset"));
    }
    let mut deleted = m.deleted;
    deleted.sort_unstable();
    Ok(RvfSummary {
        dim: m.dim,
        metric,
        total_vectors: m.total_vectors,
        manifest_offset: offset,
        file_len,
        manifest_hash: sha256(&[payload]),
        vec_segs,
        planned_rows,
        deleted,
    })
}
