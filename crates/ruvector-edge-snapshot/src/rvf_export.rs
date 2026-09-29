//! Streamed `.rvf` export in the rvf-runtime 0.2.0 layout (see
//! [`crate::rvf_format`]), readable by `rvf inspect` / `rvf query`.
//!
//! Layout: `(PROFILE sidecar, VEC)*`, then (witnessed exports) one `PROFILE`
//! export witness ([`crate::rvf_witness`]), then one `MANIFEST`. Runtime vector ids
//! are the export ordinal (0, 1, 2, …); the sidecar restores the string ids
//! and metadata on import. Output goes to a sink closure piece by piece (a
//! Worker `ReadableStream` controller in production); peak memory is one
//! segment pair plus the segment directory and digests (57 bytes per segment).
//! Redaction of metadata keys (`?redact=`) happens in the gateway before rows
//! reach [`RvfExporter::push`], since this crate does not parse JSON.

use crate::error::SnapshotError;
use crate::manifest::MAX_KEY_ID_LEN;
use crate::rvf_format::{self as fmt, DirEntry, RvfManifest, SEG_MANIFEST, SEG_PROFILE, SEG_VEC};
use crate::rvf_witness::{self, ExportIdentity, ExportWitness};
use crate::types::{Metric, Row, MAX_DIM};
use crate::witness::Signer;

/// Default rows per `VEC_SEG` (≈ 1.5 MiB at 384 dims).
pub const DEFAULT_ROWS_PER_SEGMENT: u32 = 1024;

/// Totals of a finished export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportSummary {
    /// Rows written.
    pub rows: u64,
    /// Bytes written (whole file).
    pub bytes: u64,
    /// Segments written, manifest included.
    pub segments: u64,
}

/// Streaming `.rvf` writer.
#[derive(Debug)]
pub struct RvfExporter {
    dim: u16,
    metric: Metric,
    rows_per_segment: u32,
    vec: Vec<u8>,
    side: Vec<u8>,
    seg_rows: u32,
    offset: u64,
    next_seg_id: u64,
    dir: Vec<DirEntry>,
    seg_hashes: Vec<[u8; 32]>,
    total: u64,
}

impl RvfExporter {
    /// A new export of a `dim`-dimensional collection.
    pub fn new(dim: u16, metric: Metric, rows_per_segment: u32) -> Result<Self, SnapshotError> {
        if dim == 0 || dim > MAX_DIM {
            return Err(SnapshotError::DimensionMismatch);
        }
        Ok(RvfExporter {
            dim,
            metric,
            rows_per_segment: rows_per_segment.clamp(1, 65_536),
            vec: Vec::new(),
            side: Vec::new(),
            seg_rows: 0,
            offset: 0,
            next_seg_id: 1,
            dir: Vec::new(),
            seg_hashes: Vec::new(),
            total: 0,
        })
    }

    /// Append one row; writes a segment pair to `sink` when one fills up.
    pub fn push<F: FnMut(&[u8])>(&mut self, row: &Row, sink: &mut F) -> Result<(), SnapshotError> {
        row.check(self.dim)
            .map_err(|reason| SnapshotError::InvalidRow {
                index: self.total,
                reason,
            })?;
        // Byte cap as well as the row cap, so every segment stays within the
        // importer's default `max_segment_payload` at any dimension.
        let side_row = 2 + row.id.len() + 1 + row.metadata.as_ref().map_or(0, |m| 2 + m.len());
        let vec_row = 8 + row.values.len() * 4;
        if self.seg_rows > 0
            && (self.vec.len() + vec_row > fmt::MAX_SEGMENT_PAYLOAD
                || self.side.len() + side_row > fmt::MAX_SEGMENT_PAYLOAD)
        {
            self.flush(sink);
        }
        if self.seg_rows == 0 {
            self.vec.clear();
            self.side.clear();
            fmt::begin_vec(&mut self.vec, self.dim);
            fmt::begin_sidecar(&mut self.side);
        }
        self.vec.extend_from_slice(&self.total.to_le_bytes());
        crate::bytes::put_f32s(&mut self.vec, &row.values);
        fmt::put_sidecar_row(&mut self.side, &row.id, row.metadata.as_deref());
        self.seg_rows += 1;
        self.total += 1;
        if self.seg_rows >= self.rows_per_segment {
            self.flush(sink);
        }
        Ok(())
    }

    /// Write one segment to `sink` (no directory entry).
    fn write_seg<F: FnMut(&[u8])>(&mut self, seg_type: u8, payload: &[u8], sink: &mut F) -> u64 {
        let seg_id = self.next_seg_id;
        self.next_seg_id += 1;
        let header = fmt::header_bytes(seg_type, seg_id, payload);
        sink(&header);
        sink(payload);
        self.offset += (fmt::HEADER + payload.len()) as u64;
        seg_id
    }

    fn emit<F: FnMut(&[u8])>(&mut self, seg_type: u8, payload: &[u8], sink: &mut F) {
        let offset = self.offset;
        let header_hash = {
            let seg_id = self.next_seg_id;
            crate::types::sha256(&[&fmt::header_bytes(seg_type, seg_id, payload), payload])
        };
        let seg_id = self.write_seg(seg_type, payload, sink);
        self.seg_hashes.push(header_hash);
        self.dir.push(DirEntry {
            seg_id,
            offset,
            payload_len: payload.len() as u64,
            seg_type,
        });
    }

    fn flush<F: FnMut(&[u8])>(&mut self, sink: &mut F) {
        if self.seg_rows == 0 {
            return;
        }
        fmt::seal_vec(&mut self.vec, self.seg_rows);
        fmt::seal_sidecar(&mut self.side, self.seg_rows);
        let side = core::mem::take(&mut self.side);
        let vec = core::mem::take(&mut self.vec);
        self.emit(SEG_PROFILE, &side, sink);
        self.emit(SEG_VEC, &vec, sink);
        // Keep the allocations for the next segment.
        self.side = side;
        self.vec = vec;
        self.seg_rows = 0;
    }

    fn runtime_manifest(&mut self) -> Vec<u8> {
        RvfManifest {
            epoch: 1,
            dim: self.dim,
            total_vectors: self.total,
            profile: 0,
            metric_id: self.metric.rvf_id(),
            dir: core::mem::take(&mut self.dir),
            deleted: Vec::new(),
        }
        .encode()
    }

    fn summary(&self) -> ExportSummary {
        ExportSummary {
            rows: self.total,
            bytes: self.offset,
            segments: self.next_seg_id - 1,
        }
    }

    /// Flush the last pair and write the runtime manifest (no witness: a
    /// plain rvf-runtime file). The gateway's `:export` uses
    /// [`RvfExporter::finish_witnessed`].
    pub fn finish<F: FnMut(&[u8])>(mut self, sink: &mut F) -> ExportSummary {
        self.flush(sink);
        let payload = self.runtime_manifest();
        self.write_seg(SEG_MANIFEST, &payload, sink);
        self.summary()
    }

    /// Flush the last pair, then write the export witness (see
    /// [`crate::rvf_witness`]) and the runtime manifest, which lists the
    /// witness and ends the file. The witness covers every earlier segment,
    /// the identity and audit head, and the manifest payload; it is signed
    /// when a `signer` is given (no key material lives in this crate).
    pub fn finish_witnessed<F: FnMut(&[u8])>(
        mut self,
        sink: &mut F,
        identity: &ExportIdentity,
        signer: Option<&dyn Signer>,
    ) -> Result<ExportSummary, SnapshotError> {
        self.flush(sink);
        let key_len = signer.map(|s| s.key_id().len());
        if key_len.is_some_and(|k| k > MAX_KEY_ID_LEN) {
            return Err(SnapshotError::InvalidIdentifier("key_id"));
        }
        let w_len = rvf_witness::payload_len(identity, self.seg_hashes.len(), key_len);
        // The manifest lists the witness, so its entry is known up front.
        self.dir.push(DirEntry {
            seg_id: self.next_seg_id,
            offset: self.offset,
            payload_len: w_len as u64,
            seg_type: SEG_PROFILE,
        });
        let manifest = self.runtime_manifest();
        let mut w = ExportWitness {
            identity: identity.clone(),
            dim: self.dim,
            metric: self.metric,
            rows: self.total,
            manifest_sha256: crate::types::sha256(&[&manifest]),
            segment_sha256: core::mem::take(&mut self.seg_hashes),
            root: [0; 32],
            signature: None,
        };
        w.root = rvf_witness::root_of(&rvf_witness::encode_body(&w));
        if let Some(s) = signer {
            let sig = s.sign(&rvf_witness::export_signing_message(&w.root))?;
            w.signature = Some((s.key_id().to_string(), sig));
        }
        let payload = rvf_witness::encode_payload(&w);
        debug_assert_eq!(payload.len(), w_len);
        if payload.len() != w_len {
            return Err(SnapshotError::Malformed("witness length"));
        }
        self.write_seg(SEG_PROFILE, &payload, sink);
        self.write_seg(SEG_MANIFEST, &manifest, sink);
        Ok(self.summary())
    }
}
