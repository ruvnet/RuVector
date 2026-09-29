//! Restore validation.
//!
//! Check order (ADR-351 §6.3, extended):
//! 1. manifest object size bound (before parsing or allocating),
//! 2. strict parse + known format version,
//! 3. manifest `tenant_key` = caller's tenant → else [`SnapshotError::TenantMismatch`],
//! 4. service / collection uid / shard = restore target,
//! 5. `schema_ver` known,
//! 6. recomputed root = stored root → else [`SnapshotError::ManifestChecksum`],
//! 7. signature policy,
//! 8. witness chain verifies to the ledger's trusted head **and** contains
//!    this root for this collection/shard/epoch → else chain errors,
//! 9. declared rows / bytes vs quota, rows per chunk ≤ `MAX_ROWS_PER_CHUNK`
//!    (bounds the decoded footprint), dimension / metric vs target,
//! 10. chunks one at a time, in order: size, sha256, then per-segment rvf-wire
//!     content hash, segment ids, row-block decode, strictly increasing ids.
//!
//! A relabelled manifest (another tenant's snapshot rewritten to name the
//! caller) passes step 3 by construction; it is caught at step 6 (root), or,
//! if the attacker also recomputes the root, at step 7/8, because the root is
//! not in the caller's chain. Rows are returned per chunk; callers write them
//! to a **staging** shard and swap only after [`RestoreSession::finish`]
//! succeeds, since a later chunk can still be refused.

use crate::error::SnapshotError;
use crate::manifest::{SealedManifest, KNOWN_SCHEMA_VERSIONS, MAX_ROWS_PER_CHUNK};
use crate::rowcodec;
use crate::types::{sha256, Metric, Row, ShardRef};
use crate::witness::{
    genesis, verify_chain_from, verify_signature, SignatureVerifier, WitnessEntry,
};
use rvf_types::SegmentType;

/// The restore target, derived server-side from the caller's context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreTarget {
    /// Caller's shard identity (tenant from the verified token).
    pub shard: ShardRef,
    /// Required dimension (the collection's), if the collection exists.
    pub dim: Option<u16>,
    /// Required metric, if the collection exists.
    pub metric: Option<Metric>,
}

/// Quota available to the restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreQuota {
    /// Maximum rows.
    pub max_rows: u64,
    /// Maximum total chunk bytes.
    pub max_bytes: u64,
}

/// Witness evidence held by the caller's tenant ledger.
#[derive(Debug, Clone, Copy)]
pub struct ChainProof<'a> {
    /// Chain entries from the checkpoint to the head.
    pub entries: &'a [WitnessEntry],
    /// Sequence of the first entry.
    pub start_seq: u64,
    /// `prev` of the first entry (genesis or a trusted checkpoint).
    pub start_prev: [u8; 32],
    /// The ledger's current head.
    pub trusted_head: [u8; 32],
}

impl<'a> ChainProof<'a> {
    /// Full chain from the tenant's genesis.
    pub fn from_genesis(
        tenant_key: &str,
        entries: &'a [WitnessEntry],
        trusted_head: [u8; 32],
    ) -> Self {
        ChainProof {
            entries,
            start_seq: 0,
            start_prev: genesis(tenant_key),
            trusted_head,
        }
    }
}

/// Signature requirement.
///
/// There is deliberately no "verify if present" mode: the signature block
/// is not covered by the root, so anyone able to alter a manifest could
/// also strip its signature, which makes an optional check worthless.
/// Integrity without signatures comes from the root + witness chain.
#[derive(Clone, Copy)]
pub enum SignaturePolicy<'a> {
    /// Signatures are not checked (root + witness chain still are).
    NotRequired,
    /// Must be present and valid under a key authorised for the manifest's tenant.
    Required(&'a dyn SignatureVerifier),
}

/// Outcome of a completed restore.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreSummary {
    /// Rows restored.
    pub rows: u64,
    /// Chunk bytes read.
    pub bytes: u64,
    /// Manifest root.
    pub root: [u8; 32],
    /// Snapshot epoch.
    pub epoch: u64,
}

/// An in-progress, validated restore.
#[derive(Debug)]
pub struct RestoreSession {
    sealed: SealedManifest,
    next_chunk: u32,
    next_seg_id: u64,
    rows_seen: u64,
    bytes_seen: u64,
    last_id: Option<String>,
    poisoned: Option<SnapshotError>,
}

impl RestoreSession {
    /// Validate a manifest object (steps 1–9) before any chunk is fetched.
    pub fn begin(
        manifest_obj: &[u8],
        target: &RestoreTarget,
        quota: RestoreQuota,
        chain: ChainProof<'_>,
        sig: SignaturePolicy<'_>,
    ) -> Result<Self, SnapshotError> {
        let sealed = SealedManifest::from_object(manifest_obj)?;
        let m = &sealed.manifest;
        let t = &target.shard;
        if m.tenant_key != t.tenant_key {
            return Err(SnapshotError::TenantMismatch);
        }
        if m.service != t.service || m.collection_uid != t.collection_uid {
            return Err(SnapshotError::CollectionMismatch);
        }
        if m.shard != t.shard {
            return Err(SnapshotError::ShardMismatch);
        }
        if !KNOWN_SCHEMA_VERSIONS.contains(&m.schema_ver) {
            return Err(SnapshotError::UnknownSchema(m.schema_ver));
        }
        if !sealed.root_matches() {
            return Err(SnapshotError::ManifestChecksum);
        }
        match sig {
            SignaturePolicy::NotRequired => {}
            SignaturePolicy::Required(v) => verify_signature(&sealed, v)?,
        }
        verify_chain_from(
            &t.tenant_key,
            chain.start_seq,
            &chain.start_prev,
            chain.entries,
            &chain.trusted_head,
        )?;
        let witnessed = chain.entries.iter().any(|e| {
            e.manifest_root == sealed.root
                && e.collection_uid == m.collection_uid
                && e.shard == m.shard
                && e.epoch == m.epoch
                && e.audit_head == m.audit_head
        });
        if !witnessed {
            return Err(SnapshotError::NotInChain);
        }
        if m.row_count > quota.max_rows {
            return Err(SnapshotError::QuotaExceeded {
                what: "rows",
                limit: quota.max_rows,
                requested: m.row_count,
            });
        }
        if m.total_bytes() > quota.max_bytes {
            return Err(SnapshotError::QuotaExceeded {
                what: "bytes",
                limit: quota.max_bytes,
                requested: m.total_bytes(),
            });
        }
        // Bound the decoded footprint of any one chunk before fetching it.
        if let Some(c) = m.chunks.iter().find(|c| c.rows > MAX_ROWS_PER_CHUNK) {
            return Err(SnapshotError::QuotaExceeded {
                what: "rows per chunk",
                limit: MAX_ROWS_PER_CHUNK,
                requested: c.rows,
            });
        }
        let declared = m.chunks.iter().try_fold(0u64, |a, c| a.checked_add(c.rows));
        if declared != Some(m.row_count) {
            return Err(SnapshotError::RowCountMismatch);
        }
        if target.dim.is_some_and(|d| d != m.dim) || target.metric.is_some_and(|x| x != m.metric) {
            return Err(SnapshotError::DimensionMismatch);
        }
        Ok(RestoreSession {
            sealed,
            next_chunk: 0,
            next_seg_id: 0,
            rows_seen: 0,
            bytes_seen: 0,
            last_id: None,
            poisoned: None,
        })
    }

    /// The validated manifest.
    pub fn manifest(&self) -> &SealedManifest {
        &self.sealed
    }

    /// Verify and decode chunk `index`, appending its rows to `out`. On error
    /// nothing is appended and the session refuses every later call.
    pub fn accept_chunk(
        &mut self,
        index: u32,
        bytes: &[u8],
        out: &mut Vec<Row>,
    ) -> Result<(), SnapshotError> {
        if let Some(e) = &self.poisoned {
            return Err(e.clone());
        }
        let mark = out.len();
        let r = self.accept_inner(index, bytes, out);
        if let Err(e) = &r {
            out.truncate(mark);
            self.poisoned = Some(e.clone());
        }
        r
    }

    fn accept_inner(
        &mut self,
        index: u32,
        bytes: &[u8],
        out: &mut Vec<Row>,
    ) -> Result<(), SnapshotError> {
        let expected = self.next_chunk;
        let cref = match self.sealed.manifest.chunks.get(index as usize) {
            Some(c) if index == expected => *c,
            _ => {
                return Err(SnapshotError::ChunkOutOfOrder {
                    expected,
                    got: index,
                })
            }
        };
        if bytes.len() as u64 != cref.size {
            return Err(SnapshotError::ChunkSize { index });
        }
        if sha256(&[bytes]) != cref.sha256 {
            return Err(SnapshotError::ChunkChecksum { index });
        }
        let corrupt = |reason| SnapshotError::SegmentCorrupt { index, reason };
        let dim = self.sealed.manifest.dim;
        let start = out.len();
        let mut pos = 0usize;
        while pos < bytes.len() {
            // Bound the declared length before slicing (as `from_object`
            // does): rvf_wire::read_segment adds it to 64 unchecked. This does
            // not rely on the sha256 check above having run first.
            let rest = &bytes[pos..];
            let hdr = rvf_wire::read_segment_header(rest).map_err(|_| corrupt("segment header"))?;
            if hdr.payload_length > (rest.len() - 64) as u64 {
                return Err(corrupt("segment length"));
            }
            let payload = &rest[64..64 + hdr.payload_length as usize];
            if hdr.seg_type != SegmentType::Vec as u8 {
                return Err(corrupt("segment type"));
            }
            if hdr.segment_id != self.next_seg_id {
                return Err(corrupt("segment id"));
            }
            rvf_wire::validate_segment(&hdr, payload).map_err(|_| corrupt("segment hash"))?;
            let padded = rvf_wire::calculate_padded_size(64, payload.len());
            let pad = bytes
                .get(pos + 64 + payload.len()..pos + padded)
                .ok_or(corrupt("segment padding"))?;
            if pad.iter().any(|&b| b != 0) {
                return Err(corrupt("segment padding"));
            }
            let first = out.len();
            // Never decode more rows than the (capped) chunk declares.
            let budget = cref.rows - (first - start) as u64;
            rowcodec::decode_block(payload, dim, budget, out).map_err(corrupt)?;
            for row in &out[first..] {
                if self
                    .last_id
                    .as_deref()
                    .is_some_and(|l| l >= row.id.as_str())
                {
                    return Err(corrupt("ids not strictly increasing"));
                }
                self.last_id = Some(row.id.clone());
            }
            self.next_seg_id += 1;
            pos += padded;
        }
        let n = (out.len() - start) as u64;
        if n != cref.rows {
            return Err(SnapshotError::RowCountMismatch);
        }
        self.rows_seen += n;
        self.bytes_seen += cref.size;
        self.next_chunk += 1;
        Ok(())
    }

    /// Require every chunk and the declared row count.
    pub fn finish(self) -> Result<RestoreSummary, SnapshotError> {
        if let Some(e) = self.poisoned {
            return Err(e);
        }
        let m = &self.sealed.manifest;
        let expected = m.chunks.len() as u32;
        if self.next_chunk != expected {
            return Err(SnapshotError::ChunkMissing {
                expected,
                got: self.next_chunk,
            });
        }
        if self.rows_seen != m.row_count {
            return Err(SnapshotError::RowCountMismatch);
        }
        Ok(RestoreSummary {
            rows: self.rows_seen,
            bytes: self.bytes_seen,
            root: self.sealed.root,
            epoch: m.epoch,
        })
    }
}
