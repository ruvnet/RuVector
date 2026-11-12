//! Derivation lineage records (#1289).
//!
//! `FileIdentity` (in every manifest) names one parent and no derivation
//! type. Each derived file therefore also carries one LINEAGE segment: a run
//! of fixed 128-byte `LineageRecord`s (rvf-crypto's `lineage_record_to_bytes`
//! layout), one per parent, in declared parent order. Every record is also
//! chained into the witness log as a `WITNESS_DERIVATION` entry whose action
//! hash is SHAKE-256 of the record bytes, so the readable record can be
//! checked against the witness chain.
//!
//! Format compatibility: the segment type is a new value in a range ADR-030
//! lists as unassigned (0x24). Readers that predate it never look the type up
//! (the directory carries raw type bytes), so they open the file, and their
//! compaction copies unknown segments verbatim. `FileIdentity` is unchanged:
//! for a merge it names `parents[0]`, so an older reader sees a plain chain.

use super::*;
use rvf_crypto::{lineage_record_from_bytes, lineage_record_to_bytes};
use rvf_types::{DerivationType, LineageRecord, LINEAGE_RECORD_SIZE, WITNESS_DERIVATION};

/// Segment type byte of the lineage record segment.
///
/// Not a `SegmentType` variant (that enum is closed; adding one is a breaking
/// change for `rvf-types`). Taken from the 0x24..=0x2F range ADR-030 lists as
/// available, next to the COW/lineage family (CowMap 0x20 .. Delta 0x23).
pub const LINEAGE_SEG_TYPE: u8 = 0x24;

/// Upper bound on parents in one merge record (keeps the segment small and
/// the on-open work bounded).
pub const MAX_MERGE_PARENTS: usize = 64;

impl RvfStore {
    /// Create an empty child that records several parents (`DerivationType::Merge`).
    ///
    /// This is the provenance half of the multi-parent proposal in #1289: the
    /// child's lineage segment holds one `LineageRecord` per parent, in the
    /// order given (that order is the declared read precedence for a future
    /// merged read path), and each record is chained into the witness log.
    /// `FileIdentity` names `parents[0]`, and `lineage_depth` is
    /// `1 + max(parent depth)`.
    ///
    /// The child does **not** read through to its parents yet: like
    /// [`RvfStore::derive`], it starts empty. A merged read path (precedence,
    /// shadowing, search across parents) is still under design in #1289.
    ///
    /// Rejected with `LineageBroken`: fewer than two parents, more than
    /// [`MAX_MERGE_PARENTS`], or the same parent twice. Parents must share
    /// dimension (`DimensionMismatch`) and metric (`MetricUnsupported`).
    pub fn merge(
        parents: &[&RvfStore],
        child_path: &Path,
        child_options: Option<RvfOptions>,
    ) -> Result<Self, RvfError> {
        let (first, rest) = match parents.split_first() {
            Some(split) if parents.len() >= 2 && parents.len() <= MAX_MERGE_PARENTS => split,
            _ => return Err(err(ErrorCode::LineageBroken)),
        };
        let mut seen: HashSet<[u8; 16]> = HashSet::with_capacity(parents.len());
        let mut max_depth = 0u32;
        for parent in parents {
            if !seen.insert(parent.file_identity.file_id) {
                return Err(err(ErrorCode::LineageBroken));
            }
            max_depth = max_depth.max(parent.file_identity.lineage_depth);
        }
        for parent in rest {
            if parent.options.dimension != first.options.dimension {
                return Err(err(ErrorCode::DimensionMismatch));
            }
            if parent.options.metric != first.options.metric {
                return Err(err(ErrorCode::MetricUnsupported));
            }
        }
        let new_depth = max_depth
            .checked_add(1)
            .ok_or_else(|| err(ErrorCode::LineageBroken))?;
        let mut sources = Vec::with_capacity(parents.len());
        for parent in parents {
            sources.push((
                parent.file_identity.file_id,
                parent.compute_own_manifest_hash()?,
            ));
        }
        let mut child = first.derive_child(child_path, sources[0].1, new_depth, child_options)?;
        child.record_lineage(DerivationType::Merge, &sources)?;
        child.write_manifest()?;
        Ok(child)
    }

    /// The lineage records this file carries, in declared parent order: one
    /// for a derived file, one per parent for a merge, none for a root file or
    /// a file derived by a runtime older than 0.5.4.
    ///
    /// Each record must name this file and the first must name the parent in
    /// `FileIdentity`; otherwise `LineageBroken` (for example, a lineage
    /// segment spliced in from another file).
    pub fn lineage_records(&self) -> Result<Vec<LineageRecord>, RvfError> {
        let Some(&(_, offset, _, _)) = self
            .segment_dir
            .iter()
            .rev()
            .find(|entry| entry.3 == LINEAGE_SEG_TYPE)
        else {
            return Ok(Vec::new());
        };
        let payload = {
            let mut reader = BufReader::new(&self.file);
            self.read_seg(&mut reader, offset)
                .map_err(|e| Self::map_read_error(&e, ErrorCode::InvalidChecksum))?
                .1
        };
        if payload.is_empty()
            || payload.len() % LINEAGE_RECORD_SIZE != 0
            || payload.len() / LINEAGE_RECORD_SIZE > MAX_MERGE_PARENTS
        {
            return Err(err(ErrorCode::TruncatedSegment));
        }
        let mut records = Vec::with_capacity(payload.len() / LINEAGE_RECORD_SIZE);
        for chunk in payload.chunks_exact(LINEAGE_RECORD_SIZE) {
            let bytes: &[u8; LINEAGE_RECORD_SIZE] = chunk
                .try_into()
                .map_err(|_| err(ErrorCode::TruncatedSegment))?;
            let record = lineage_record_from_bytes(bytes)?;
            if record.file_id != self.file_identity.file_id {
                return Err(err(ErrorCode::LineageBroken));
            }
            records.push(record);
        }
        let first = &records[0];
        if first.parent_id != self.file_identity.parent_id
            || first.parent_hash != self.file_identity.parent_hash
            || records
                .iter()
                .any(|r| r.derivation_type != first.derivation_type)
        {
            return Err(err(ErrorCode::LineageBroken));
        }
        Ok(records)
    }

    /// How this file was derived from its parent(s), if it records it
    /// (`None` for root files and for files derived before 0.5.4).
    pub fn derivation_type(&self) -> Result<Option<DerivationType>, RvfError> {
        Ok(self
            .lineage_records()?
            .first()
            .map(|record| record.derivation_type))
    }

    /// Write the lineage segment for `parents` (`(file_id, manifest_hash)` in
    /// declared order) and chain one witness entry per record. Does not write
    /// the manifest; the caller commits.
    pub(super) fn record_lineage(
        &mut self,
        derivation_type: DerivationType,
        parents: &[([u8; 16], [u8; 32])],
    ) -> Result<(), RvfError> {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let count = parents.len();
        let mut payload = Vec::with_capacity(count * LINEAGE_RECORD_SIZE);
        for (i, (parent_id, parent_hash)) in parents.iter().enumerate() {
            let description = if count == 1 {
                format!("derive {derivation_type:?}")
            } else {
                format!("merge parent {}/{}", i + 1, count)
            };
            let record = LineageRecord::new(
                self.file_identity.file_id,
                *parent_id,
                *parent_hash,
                derivation_type,
                0,
                timestamp_ns,
                &description,
            );
            payload.extend_from_slice(&lineage_record_to_bytes(&record));
        }

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (seg_id, seg_offset) = {
            let mut buf_writer = Self::append_cursor(&self.file, &self.security)?;
            let written = writer
                .write_typed_seg(&mut buf_writer, LINEAGE_SEG_TYPE, &payload)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
        };
        self.segment_dir
            .push((seg_id, seg_offset, payload.len() as u64, LINEAGE_SEG_TYPE));

        for record in payload.chunks_exact(LINEAGE_RECORD_SIZE) {
            self.append_witness(WITNESS_DERIVATION, record)?;
        }
        // Data before pointer: the records are durable before the manifest
        // that lists them.
        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))
    }
}
