//! Streamed, pull-based `.rvf` import into upsert batches.
//!
//! Flow (the bulk-import job does this): range-read the tail and call
//! [`crate::inspect_tail`]; build an [`RvfImporter`] with that **required**
//! summary — dimension, metric and the estimated upserts vs quota are refused
//! here, before any batch exists; then alternate [`RvfImporter::feed`] (one
//! piece of the body) with [`RvfImporter::next_batch`] until it returns
//! `None`, commit each batch, and call [`RvfImporter::finish`]. Only segments
//! listed in the authoritative manifest are imported, each live `VEC_SEG`
//! header must match its directory entry, and deleted ids are skipped.
//!
//! Memory is bounded by the encoded input, never by what it decodes to:
//! `next_batch` materialises at most `max_batch_rows` rows, reading ids and
//! metadata lazily from the `PROFILE` sidecar in place (a sidecar is
//! validated without allocating before any of its rows is used). Peak is one
//! record (sidecar + vec, ≤ 2 × (`max_segment_payload` + 127) bytes) plus
//! the piece last fed plus one batch — provided the caller drains
//! `next_batch` before feeding again.
//!
//! Batches never span records (sidecar + vec pair), so an [`ImportCursor`]
//! is a record offset plus a row index, and a resumed import reproduces the
//! same batches with the same sequence numbers.
//!
//! Quota unit: `max_rows` meters **upsert operations** (rows emitted). That
//! is an upper bound on the distinct ids the import adds; the shard's usage
//! meter counts distinct stored ids (an upsert of an existing id adds none).
//! [`ImportTotals`] reports both `rows` (ops) and `skipped_deleted`.

use crate::error::ImportError;
use crate::rvf_format::{self as fmt, SEG_MANIFEST, SEG_PROFILE, SEG_VEC};
use crate::rvf_summary::{ImportLimits, RvfSummary};
use crate::types::{check_parts, sha256, Metric, Row};
use rvf_types::SegmentHeader;
use sha2::{Digest, Sha256};

/// Target collection shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportSpec {
    /// Collection dimension.
    pub dim: u16,
    /// Collection metric.
    pub metric: Metric,
    /// Limits.
    pub limits: ImportLimits,
}

/// Resume point: record offset + rows of that record already consumed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImportCursor {
    /// Absolute offset of the next record (stream restarts here).
    pub byte_offset: u64,
    /// Rows of that record already consumed.
    pub row_in_record: u32,
    /// Sequence of the next batch.
    pub batch_seq: u64,
    /// Upsert operations emitted so far.
    pub rows_done: u64,
    /// Rows skipped because their id is deleted in the file.
    pub rows_skipped: u64,
}

/// One upsert batch.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportBatch {
    /// Deterministic sequence (→ `op_id`, see [`crate::ImportJob::op_id`]).
    pub seq: u64,
    /// Rows to upsert.
    pub rows: Vec<Row>,
    /// Cursor to persist once this batch is committed.
    pub cursor_after: ImportCursor,
}

/// Totals of a finished import stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportTotals {
    /// Upsert operations emitted (including those before the resume cursor).
    pub rows: u64,
    /// Rows skipped as deleted.
    pub skipped_deleted: u64,
    /// Next batch sequence (= batches produced overall).
    pub batches: u64,
    /// Final stream offset (= file length).
    pub bytes: u64,
    /// sha256 of the whole file, when this delivery streamed it from byte 0.
    pub sha256: Option<[u8; 32]>,
}

/// The record being turned into batches (offsets relative to `head`).
#[derive(Debug, Clone, Copy)]
struct Active {
    record: u64,
    end: u64,
    len: usize,
    vec_at: usize,
    count: u32,
    next: u32,
    skip: u32,
    side: Option<(usize, usize)>,
}

/// Streaming importer; see the module docs.
#[derive(Debug)]
pub struct RvfImporter {
    spec: ImportSpec,
    summary: RvfSummary,
    buf: Vec<u8>,
    head: usize,
    pos: u64,
    resume: ImportCursor,
    cur: ImportCursor,
    active: Option<Active>,
    live_seen: usize,
    last_manifest: Option<(u64, [u8; 32])>,
    hasher: Option<Sha256>,
    peak_buffered: usize,
    /// Bytes from `head` the current parse is waiting for (a validated
    /// header's full length), so the buffer grows once to the exact record
    /// size instead of by doubling (which would transiently hold ~3×).
    want: usize,
    /// Verified header of a sidecar at `head` whose VEC is still incomplete.
    first_ok: Option<SegmentHeader>,
}

fn seg_len(h: &SegmentHeader) -> usize {
    fmt::HEADER + h.payload_length as usize + h.alignment_pad as usize
}

impl RvfImporter {
    /// Start (or resume at `cursor`); the caller streams bytes from
    /// `cursor.byte_offset`. Refuses dimension / metric mismatches and files
    /// whose estimated upserts exceed `max_rows`, before any batch.
    pub fn new(
        spec: ImportSpec,
        summary: RvfSummary,
        cursor: ImportCursor,
    ) -> Result<Self, ImportError> {
        if summary.dim != spec.dim {
            return Err(ImportError::DimensionMismatch {
                expected: spec.dim,
                got: summary.dim,
            });
        }
        if summary.metric != spec.metric {
            return Err(ImportError::MetricMismatch);
        }
        if summary.estimated_upserts() > spec.limits.max_rows {
            return Err(ImportError::QuotaExceeded {
                limit: spec.limits.max_rows,
            });
        }
        if spec.limits.max_batch_rows == 0 || cursor.byte_offset > summary.file_len {
            return Err(ImportError::Malformed("limits or cursor"));
        }
        Ok(RvfImporter {
            spec,
            summary,
            buf: Vec::new(),
            head: 0,
            pos: cursor.byte_offset,
            resume: cursor,
            cur: cursor,
            active: None,
            live_seen: 0,
            last_manifest: None,
            hasher: (cursor.byte_offset == 0).then(Sha256::new),
            peak_buffered: 0,
            want: 0,
            first_ok: None,
        })
    }

    /// Peak bytes held in the reassembly buffer (capacity).
    pub fn peak_buffered_bytes(&self) -> usize {
        self.peak_buffered
    }

    /// Append the next piece of the stream. Decodes nothing; drain
    /// [`RvfImporter::next_batch`] before feeding again to keep memory bounded.
    pub fn feed(&mut self, data: &[u8]) -> Result<(), ImportError> {
        let end = self.pos + (self.buf.len() - self.head) as u64 + data.len() as u64;
        if end > self.spec.limits.max_file_bytes {
            return Err(ImportError::FileTooLarge);
        }
        if end > self.summary.file_len {
            return Err(ImportError::ManifestMismatch("stream longer than file"));
        }
        let need = self.buf.len() - self.head + data.len();
        let short = self.buf.capacity() - self.head < need;
        if self.head > 0 && (short || self.head * 2 >= self.buf.len()) {
            self.buf.drain(..self.head);
            self.head = 0;
        }
        if self.buf.capacity() < need {
            // Grow once to the awaited record plus one piece of overshoot
            // (drain-then-feed usage); a caller running ahead of the parser
            // gets amortised growth instead.
            let exact = if self.want > 0 {
                self.want + data.len()
            } else {
                0
            };
            if need <= exact {
                self.buf.reserve_exact(exact - self.buf.len());
            } else {
                self.buf.reserve(need - self.buf.len());
            }
        }
        self.buf.extend_from_slice(data);
        self.peak_buffered = self.peak_buffered.max(self.buf.capacity());
        Ok(())
    }

    /// The next complete batch, or `None` when more input is needed.
    pub fn next_batch(&mut self) -> Result<Option<ImportBatch>, ImportError> {
        loop {
            if let Some(a) = self.active {
                if let Some(b) = self.rows_from(a)? {
                    return Ok(Some(b));
                }
                continue;
            }
            if !self.start_record()? {
                return Ok(None);
            }
        }
    }

    /// Header of the segment at `at` (relative to `head`) if fully buffered.
    fn complete_segment(&mut self, at: usize) -> Result<Option<SegmentHeader>, ImportError> {
        let avail = &self.buf[self.head..];
        let Some(rest) = avail.get(at..).filter(|r| r.len() >= fmt::HEADER) else {
            self.want = at + fmt::HEADER;
            return Ok(None);
        };
        let h = fmt::parse_header(rest).map_err(ImportError::Malformed)?;
        if h.payload_length > self.spec.limits.max_segment_payload {
            return Err(ImportError::SegmentTooLarge(h.payload_length));
        }
        if h.alignment_pad > 63 {
            return Err(ImportError::Malformed("alignment pad"));
        }
        if rest.len() < seg_len(&h) {
            // Bounded: the header passed the size checks above.
            self.want = at + seg_len(&h);
            return Ok(None);
        }
        let payload = &rest[fmt::HEADER..fmt::HEADER + h.payload_length as usize];
        if !fmt::payload_hash_ok(&h, payload) {
            return Err(ImportError::Checksum(self.pos + at as u64));
        }
        Ok(Some(h))
    }

    fn payload(&self, at: usize, h: &SegmentHeader) -> &[u8] {
        let s = self.head + at + fmt::HEADER;
        &self.buf[s..s + h.payload_length as usize]
    }

    /// Consume `len` bytes at `head` (hashing them if this delivery hashes).
    fn consume(&mut self, len: usize) {
        self.want = 0;
        self.first_ok = None;
        if let Some(h) = &mut self.hasher {
            h.update(&self.buf[self.head..self.head + len]);
        }
        self.head += len;
        self.pos += len as u64;
    }

    /// Set up (or skip) the record at `head`. `false` = more input needed.
    fn start_record(&mut self) -> Result<bool, ImportError> {
        // A sidecar already verified while its VEC was still arriving is not
        // re-hashed on every call (that would be quadratic in small pieces).
        let h = match self.first_ok {
            Some(h) => h,
            None => match self.complete_segment(0)? {
                Some(h) => h,
                None => return Ok(false),
            },
        };
        let first = seg_len(&h);
        let (side, vec_at, vh) = match h.seg_type {
            SEG_PROFILE if fmt::is_sidecar(self.payload(0, &h)) => {
                let Some(vh) = self.complete_segment(first)? else {
                    self.first_ok = Some(h);
                    return Ok(false);
                };
                self.first_ok = None;
                if vh.seg_type != SEG_VEC {
                    return Err(ImportError::Malformed("sidecar not followed by vec"));
                }
                (Some(h), first, vh)
            }
            SEG_VEC => (None, 0, h),
            SEG_MANIFEST => {
                self.last_manifest = Some((self.pos, sha256(&[self.payload(0, &h)])));
                self.consume(first);
                return Ok(true);
            }
            _ => {
                self.consume(first);
                return Ok(true);
            }
        };
        let len = vec_at + seg_len(&vh);
        let vec_off = self.pos + vec_at as u64;
        let listed = self
            .summary
            .vec_segs
            .binary_search_by_key(&vec_off, |e| e.0)
            .map(|i| self.summary.vec_segs[i].1);
        let Ok(dir_len) = listed else {
            // Superseded segment: not in the final directory.
            self.consume(len);
            return Ok(true);
        };
        if dir_len != vh.payload_length {
            return Err(ImportError::ManifestMismatch("directory length"));
        }
        let (dim, count) =
            fmt::vec_shape(self.payload(vec_at, &vh)).map_err(ImportError::Malformed)?;
        if dim != self.spec.dim {
            return Err(ImportError::DimensionMismatch {
                expected: self.spec.dim,
                got: dim,
            });
        }
        if let Some(sh) = &side {
            fmt::validate_sidecar(self.payload(0, sh), count).map_err(ImportError::Malformed)?;
        }
        let record = self.pos;
        let skip = if record == self.resume.byte_offset {
            self.resume.row_in_record.min(count)
        } else {
            0
        };
        self.live_seen += 1;
        self.active = Some(Active {
            record,
            end: record + len as u64,
            len,
            vec_at,
            count,
            next: 0,
            skip,
            side: side.map(|sh| {
                (
                    fmt::HEADER + fmt::SIDECAR_HEADER,
                    fmt::HEADER + sh.payload_length as usize,
                )
            }),
        });
        Ok(true)
    }

    /// Emit rows of the active record until a batch is full or it ends.
    fn rows_from(&mut self, mut a: Active) -> Result<Option<ImportBatch>, ImportError> {
        let dim = self.spec.dim;
        let stride = 8 + usize::from(dim) * 4;
        let base = self.head;
        let mut batch = Vec::new();
        let mut emitted = None;
        while a.next < a.count {
            let i = a.next;
            a.next += 1;
            let sc = match &mut a.side {
                Some((pos, end)) => {
                    let side = &self.buf[base..base + *end];
                    let row =
                        fmt::sidecar_row(side, pos).ok_or(ImportError::Malformed("sidecar"))?;
                    Some(row)
                }
                None => None,
            };
            if i < a.skip {
                continue;
            }
            let at = base + a.vec_at + fmt::HEADER + 6 + i as usize * stride;
            let rec = &self.buf[at..at + stride];
            let rid = u64::from_le_bytes(rec[..8].try_into().unwrap_or([0; 8]));
            if self.summary.deleted.binary_search(&rid).is_ok() {
                self.cur.rows_skipped += 1;
            } else {
                if self.cur.rows_done >= self.spec.limits.max_rows {
                    return Err(ImportError::QuotaExceeded {
                        limit: self.spec.limits.max_rows,
                    });
                }
                let values: Vec<f32> = rec[8..]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let (id, metadata) = match sc {
                    Some((id, m)) => (id.to_string(), m.map(str::to_string)),
                    None => (rid.to_string(), None),
                };
                check_parts(&id, &values, metadata.as_deref(), dim).map_err(|reason| {
                    ImportError::InvalidRow {
                        index: self.cur.rows_done,
                        reason,
                    }
                })?;
                batch.push(Row {
                    id,
                    values,
                    metadata,
                });
                self.cur.rows_done += 1;
            }
            let last = a.next == a.count;
            if !batch.is_empty() && (batch.len() >= self.spec.limits.max_batch_rows || last) {
                let (byte_offset, row_in_record) =
                    if last { (a.end, 0) } else { (a.record, a.next) };
                self.cur.byte_offset = byte_offset;
                self.cur.row_in_record = row_in_record;
                self.cur.batch_seq += 1;
                emitted = Some(ImportBatch {
                    seq: self.cur.batch_seq - 1,
                    rows: batch,
                    cursor_after: self.cur,
                });
                break;
            }
        }
        if a.next == a.count {
            self.active = None;
            self.consume(a.len);
        } else {
            self.active = Some(a);
        }
        Ok(emitted)
    }

    /// End of stream: every byte consumed, the final manifest is the one the
    /// summary describes, and every live `VEC_SEG` was imported.
    pub fn finish(self) -> Result<ImportTotals, ImportError> {
        if self.active.is_some() || self.buf.len() > self.head {
            return Err(ImportError::Truncated);
        }
        let (offset, hash) = self.last_manifest.ok_or(ImportError::NoManifest)?;
        let s = &self.summary;
        if s.manifest_offset != offset || s.manifest_hash != hash || s.file_len != self.pos {
            return Err(ImportError::ManifestMismatch("tail summary"));
        }
        let expected = s
            .vec_segs
            .iter()
            .filter(|e| e.0 >= self.resume.byte_offset)
            .count();
        if expected != self.live_seen {
            return Err(ImportError::ManifestMismatch("directory offsets"));
        }
        Ok(ImportTotals {
            rows: self.cur.rows_done,
            skipped_deleted: self.cur.rows_skipped,
            batches: self.cur.batch_seq,
            bytes: self.pos,
            sha256: self.hasher.map(|h| h.finalize().into()),
        })
    }
}
