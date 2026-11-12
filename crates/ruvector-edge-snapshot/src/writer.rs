//! Streaming snapshot writer.
//!
//! Rows arrive from the shard's SQLite cursor in id order and are encoded
//! straight into the current row-block segment; nothing is buffered as
//! [`Row`] values. Segments close at `max_segment_payload` (≤ 4 MiB, ADR-351
//! §6.3) and are packed whole into chunk objects of at most
//! `max_chunk_bytes` (≤ 8 MiB). Each chunk becomes one R2 object
//! (`seg-{n}.rvf`, see [`snapshot_key`]); chunks are not R2 multipart parts,
//! because whole-segment packing cannot give the equal part sizes multipart
//! requires. Peak memory is one segment plus one chunk.
//!
//! Every chunk's final R2 key is known before the chunk exists
//! ([`SnapshotWriter::chunk_key`]): the key prefix hashes the shard identity,
//! epoch and audit head, not the manifest root, so the Worker PUTs each chunk
//! as soon as [`SnapshotWriter::push`] returns it and never buffers or copies
//! objects. Chunks are also capped at [`MAX_ROWS_PER_CHUNK`] rows so that a
//! restore decodes at most ~12 MB of rows at a time at any dimension.

use crate::error::SnapshotError;
use crate::manifest::{
    signing_message, ChunkRef, Manifest, SealedManifest, SignatureBlock, FORMAT_VERSION,
    MAX_CHUNKS, MAX_KEY_ID_LEN, MAX_ROWS_PER_CHUNK, SCHEMA_VERSION,
};
use crate::rowcodec::{self, BLOCK_HEADER};
use crate::types::{sha256, Clock, Metric, Row, ShardRef, MAX_DIM};
use crate::witness::Signer;
use rvf_types::{SegmentFlags, SegmentType};

/// Hard cap on a chunk object (ADR-351 §6.3 / task: ≤ 8 MiB).
pub const MAX_CHUNK_BYTES: usize = 8 << 20;
/// Default segment payload target (ADR-351 §6.3: ≤ 4 MB `VEC_SEG`s).
pub const DEFAULT_SEGMENT_PAYLOAD: usize = 4 << 20;

/// Size limits for one snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotLimits {
    /// Maximum chunk object size (≤ [`MAX_CHUNK_BYTES`]).
    pub max_chunk_bytes: usize,
    /// Maximum row-block payload per segment.
    pub max_segment_payload: usize,
    /// Maximum rows (the shard's quota).
    pub max_rows: u64,
    /// Maximum rows per chunk (≤ [`MAX_ROWS_PER_CHUNK`]).
    pub max_rows_per_chunk: u64,
}

impl Default for SnapshotLimits {
    fn default() -> Self {
        SnapshotLimits {
            max_chunk_bytes: MAX_CHUNK_BYTES,
            max_segment_payload: DEFAULT_SEGMENT_PAYLOAD,
            max_rows: u64::MAX,
            max_rows_per_chunk: MAX_ROWS_PER_CHUNK,
        }
    }
}

/// What is being snapshotted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotSpec {
    /// Owning shard.
    pub shard: ShardRef,
    /// Snapshot epoch (`snapshot_seq`).
    pub epoch: u64,
    /// Vector dimension.
    pub dim: u16,
    /// Metric.
    pub metric: Metric,
    /// Audit-chain head at snapshot time.
    pub audit_head: [u8; 32],
}

/// One finished chunk object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Position (`seg-{index}.rvf`).
    pub index: u32,
    /// Object bytes.
    pub bytes: Vec<u8>,
    /// Rows inside.
    pub rows: u64,
    /// sha256 of `bytes`.
    pub sha256: [u8; 32],
}

/// Streaming writer; see the module docs.
#[derive(Debug)]
pub struct SnapshotWriter {
    spec: SnapshotSpec,
    limits: SnapshotLimits,
    seg: Vec<u8>,
    seg_rows: u32,
    chunk: Vec<u8>,
    chunk_rows: u64,
    chunks: Vec<ChunkRef>,
    next_seg_id: u64,
    rows: u64,
    last_id: Option<String>,
    prefix: String,
}

impl SnapshotWriter {
    /// Validate the spec and limits.
    pub fn new(spec: SnapshotSpec, limits: SnapshotLimits) -> Result<Self, SnapshotError> {
        if spec.dim == 0 || spec.dim > MAX_DIM {
            return Err(SnapshotError::DimensionMismatch);
        }
        let seg_max = BLOCK_HEADER + rowcodec::max_row_size(spec.dim);
        let seg_bytes = rvf_wire::calculate_padded_size(64, limits.max_segment_payload);
        if limits.max_chunk_bytes > MAX_CHUNK_BYTES
            || limits.max_segment_payload < seg_max
            || seg_bytes > limits.max_chunk_bytes
            || limits.max_rows_per_chunk == 0
            || limits.max_rows_per_chunk > MAX_ROWS_PER_CHUNK
        {
            return Err(SnapshotError::QuotaExceeded {
                what: "snapshot limits",
                limit: MAX_CHUNK_BYTES as u64,
                requested: seg_bytes as u64,
            });
        }
        let s = &spec.shard;
        let prefix = crate::manifest::key_prefix(
            &s.tenant_key,
            &s.service,
            &s.collection_uid,
            s.shard,
            spec.epoch,
            &spec.audit_head,
        );
        Ok(SnapshotWriter {
            prefix,
            spec,
            limits,
            seg: Vec::new(),
            seg_rows: 0,
            chunk: Vec::new(),
            chunk_rows: 0,
            chunks: Vec::new(),
            next_seg_id: 0,
            rows: 0,
            last_id: None,
        })
    }

    /// Final R2 key of chunk `n` (`{prefix}/seg-{n}.rvf`), known before the
    /// chunk is produced and equal to [`snapshot_key`] of the finished manifest.
    pub fn chunk_key(&self, n: u32) -> String {
        format!("{}/seg-{n}.rvf", self.prefix)
    }

    /// Final R2 key of the manifest object.
    pub fn manifest_key(&self) -> String {
        format!("{}/manifest.rvf", self.prefix)
    }

    /// Rows accepted so far.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Bytes currently buffered (segment + chunk).
    pub fn buffered_bytes(&self) -> usize {
        self.seg.capacity() + self.chunk.capacity()
    }

    /// Add one row. Ids must be strictly increasing (the cursor's order); this
    /// makes snapshots deterministic and lets restore refuse duplicates.
    /// Returns a finished chunk when one fills up.
    pub fn push(&mut self, row: &Row) -> Result<Option<Chunk>, SnapshotError> {
        let index = self.rows;
        row.check(self.spec.dim)
            .map_err(|reason| SnapshotError::InvalidRow { index, reason })?;
        if self
            .last_id
            .as_deref()
            .is_some_and(|l| l >= row.id.as_str())
        {
            return Err(SnapshotError::InvalidRow {
                index,
                reason: "ids not strictly increasing",
            });
        }
        if self.rows >= self.limits.max_rows {
            return Err(SnapshotError::QuotaExceeded {
                what: "rows",
                limit: self.limits.max_rows,
                requested: self.rows + 1,
            });
        }
        let size = rowcodec::row_size(row);
        let mut emitted = None;
        let seg_full = self.seg.len() + size > self.limits.max_segment_payload
            || u64::from(self.seg_rows) >= self.limits.max_rows_per_chunk;
        if self.seg_rows > 0 && seg_full {
            emitted = self.flush_segment()?;
        }
        if self.seg_rows == 0 {
            self.seg.clear();
            rowcodec::begin_block(&mut self.seg, self.spec.dim);
        }
        rowcodec::put_row(&mut self.seg, row);
        self.seg_rows += 1;
        self.rows += 1;
        self.last_id = Some(row.id.clone());
        Ok(emitted)
    }

    fn flush_segment(&mut self) -> Result<Option<Chunk>, SnapshotError> {
        if self.seg_rows == 0 {
            return Ok(None);
        }
        rowcodec::seal_block(&mut self.seg, self.seg_rows);
        let bytes = rvf_wire::write_segment(
            SegmentType::Vec as u8,
            &self.seg,
            SegmentFlags::empty(),
            self.next_seg_id,
        );
        self.next_seg_id += 1;
        let mut emitted = None;
        let chunk_full = self.chunk.len() + bytes.len() > self.limits.max_chunk_bytes
            || self.chunk_rows + u64::from(self.seg_rows) > self.limits.max_rows_per_chunk;
        if !self.chunk.is_empty() && chunk_full {
            emitted = Some(self.take_chunk()?);
        }
        self.chunk.extend_from_slice(&bytes);
        self.chunk_rows += u64::from(self.seg_rows);
        self.seg_rows = 0;
        self.seg.clear();
        Ok(emitted)
    }

    fn take_chunk(&mut self) -> Result<Chunk, SnapshotError> {
        if self.chunks.len() >= MAX_CHUNKS {
            return Err(SnapshotError::QuotaExceeded {
                what: "chunks",
                limit: MAX_CHUNKS as u64,
                requested: self.chunks.len() as u64 + 1,
            });
        }
        let bytes = core::mem::take(&mut self.chunk);
        let digest = sha256(&[&bytes]);
        let c = Chunk {
            index: self.chunks.len() as u32,
            rows: self.chunk_rows,
            sha256: digest,
            bytes,
        };
        self.chunks.push(ChunkRef {
            size: c.bytes.len() as u64,
            rows: c.rows,
            sha256: digest,
        });
        self.chunk_rows = 0;
        Ok(c)
    }

    /// Close the last segment and chunk, then build, root and optionally sign
    /// the manifest. Returns the remaining chunks (0–2) and the sealed manifest.
    pub fn finish(
        mut self,
        clock: &dyn Clock,
        signer: Option<&dyn Signer>,
    ) -> Result<(Vec<Chunk>, SealedManifest), SnapshotError> {
        let mut tail = Vec::new();
        if let Some(c) = self.flush_segment()? {
            tail.push(c);
        }
        if !self.chunk.is_empty() {
            tail.push(self.take_chunk()?);
        }
        let s = &self.spec;
        let manifest = Manifest {
            format_ver: FORMAT_VERSION,
            schema_ver: SCHEMA_VERSION,
            tenant_key: s.shard.tenant_key.clone(),
            service: s.shard.service.clone(),
            collection_uid: s.shard.collection_uid.clone(),
            shard: s.shard.shard,
            epoch: s.epoch,
            dim: s.dim,
            metric: s.metric,
            row_count: self.rows,
            created_at_ms: clock.now_ms(),
            audit_head: s.audit_head,
            chunks: self.chunks,
        };
        let root = manifest.compute_root();
        let signature = match signer {
            None => None,
            Some(sg) => {
                if sg.key_id().len() > MAX_KEY_ID_LEN {
                    return Err(SnapshotError::InvalidIdentifier("key_id"));
                }
                Some(SignatureBlock {
                    key_id: sg.key_id().to_string(),
                    signature: sg.sign(&signing_message(&root))?,
                })
            }
        };
        Ok((
            tail,
            SealedManifest {
                manifest,
                root,
                signature,
            },
        ))
    }
}

/// R2 key of chunk `n` of a finished snapshot (ADR-351 §6.3,
/// [`crate::manifest::key_prefix`]); restore derives keys from the manifest.
pub fn snapshot_key(m: &Manifest, n: u32) -> String {
    format!("{}/seg-{n}.rvf", m.key_prefix())
}

/// R2 key of the manifest object, next to its chunks.
pub fn manifest_key(m: &Manifest) -> String {
    format!("{}/manifest.rvf", m.key_prefix())
}
