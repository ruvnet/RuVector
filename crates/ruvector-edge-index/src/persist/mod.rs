//! Chunked persistence into `index_chunks(epoch, part, bytes)` rows.
//!
//! **Streaming both ways.** The encoder first measures the payload with a
//! counting pass over the same writer, so every chunk header can carry the
//! total part count and payload length; each ≤ `max_chunk` row is then
//! emitted to the caller's sink as soon as it fills and dropped. The only
//! value that needs every byte, the whole-payload sha256, travels in the
//! header of the **last** part. The decoder pulls rows one at a time,
//! validates each header as it arrives, hashes incrementally and parses
//! into exactly-sized buffers; peak memory is the index plus one chunk.
//!
//! | off | field |
//! |---|---|
//! | 0 | magic `RVEI` |
//! | 4 | format version `u16` (2) |
//! | 6 | index kind `u8`, flags `u8` (bit 0: last part) |
//! | 8 | epoch `u64` |
//! | 16 | part `u32`, total parts `u32` |
//! | 24 | body length `u32` |
//! | 28 | payload length `u64` |
//! | 36 | payload sha256 (last part only, else zero; = `meta.index_sha256`) |
//! | 68 | CRC-32C over bytes `0..68` + body |
//! | 72 | body |
//!
//! Any corruption, truncation, reordering, gap, extra part or digest
//! mismatch is a typed [`DecodeError`]; a parsed index is discarded unless
//! the final digest matches.

mod stream;

use sha2::{Digest, Sha256};

use crate::bytes::{Counter, Sink};
use crate::crc::Crc32c;
use crate::error::{EmitError, EncodeError};

pub(crate) use stream::{decode, decode_slice};

/// Hard cap on one chunk including its header (task cap; ADR §6.1 allows
/// 1.9 MB, the tighter bound wins).
pub const MAX_CHUNK_BYTES: usize = 1 << 20;
/// Header bytes per chunk.
pub const CHUNK_HEADER_LEN: usize = 72;

const MAGIC: [u8; 4] = *b"RVEI";
const VERSION: u16 = 2;
const FLAG_LAST: u8 = 1;
const DIGEST_AT: usize = 36;
const CRC_AT: usize = 68;

/// What a chunk set holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// [`crate::QuantFlatIndex`].
    QuantFlat,
    /// [`crate::HnswIndex`].
    Hnsw,
}

impl IndexKind {
    fn code(self) -> u8 {
        match self {
            IndexKind::QuantFlat => 1,
            IndexKind::Hnsw => 2,
        }
    }
}

/// One `index_chunks` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexChunk {
    /// `index_epoch` the chunk belongs to.
    pub epoch: u64,
    /// Part number, `0..total`.
    pub part: u32,
    /// Header + body (≤ the encoder's `max_chunk`).
    pub bytes: Vec<u8>,
}

/// Summary of one encoded epoch (what the store records in `meta`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexDigest {
    /// Epoch stamped into every chunk.
    pub epoch: u64,
    /// Index kind.
    pub kind: IndexKind,
    /// Number of parts emitted.
    pub parts: u32,
    /// Payload bytes (sum of bodies).
    pub payload_len: u64,
    /// sha256 of the concatenated payload (`meta.index_sha256`).
    pub sha256: [u8; 32],
}

/// Result of encoding one index epoch into memory ([`crate::HnswIndex::to_chunks`]).
#[derive(Debug, Clone)]
pub struct EncodedIndex {
    /// Epoch stamped into every chunk.
    pub epoch: u64,
    /// Index kind.
    pub kind: IndexKind,
    /// Rows to write, in part order.
    pub chunks: Vec<IndexChunk>,
    /// sha256 of the concatenated payload (`meta.index_sha256`).
    pub sha256: [u8; 32],
    /// Payload bytes (sum of bodies).
    pub payload_len: u64,
}

impl EncodedIndex {
    /// Lower-case hex of [`Self::sha256`].
    pub fn sha256_hex(&self) -> String {
        self.sha256.iter().map(|b| format!("{b:02x}")).collect()
    }
}

pub(crate) fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

pub(crate) fn rd_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

pub(crate) fn chunk_crc(b: &[u8]) -> u32 {
    let mut c = Crc32c::new();
    c.update(&b[..CRC_AT]);
    c.update(&b[CHUNK_HEADER_LEN..]);
    c.finish()
}

fn put_crc(b: &mut [u8]) {
    let crc = chunk_crc(b);
    b[CRC_AT..CHUNK_HEADER_LEN].copy_from_slice(&crc.to_le_bytes());
}

/// Fill a complete header (and CRC) into `b`, whose body is already there.
#[allow(clippy::too_many_arguments)]
fn seal(
    b: &mut [u8],
    kind: IndexKind,
    last: bool,
    epoch: u64,
    part: u32,
    total: u32,
    len: u64,
    digest: &[u8; 32],
) {
    let body_len = (b.len() - CHUNK_HEADER_LEN) as u32;
    b[0..4].copy_from_slice(&MAGIC);
    b[4..6].copy_from_slice(&VERSION.to_le_bytes());
    b[6] = kind.code();
    b[7] = if last { FLAG_LAST } else { 0 };
    b[8..16].copy_from_slice(&epoch.to_le_bytes());
    b[16..20].copy_from_slice(&part.to_le_bytes());
    b[20..24].copy_from_slice(&total.to_le_bytes());
    b[24..28].copy_from_slice(&body_len.to_le_bytes());
    b[28..36].copy_from_slice(&len.to_le_bytes());
    b[DIGEST_AT..CRC_AT].copy_from_slice(digest);
    put_crc(b);
}

/// Streaming chunk encoder: each full chunk goes straight to `emit`.
struct ChunkWriter<'e, E> {
    emit: &'e mut dyn FnMut(IndexChunk) -> Result<(), E>,
    err: Option<E>,
    epoch: u64,
    kind: IndexKind,
    body_cap: usize,
    total: u32,
    payload_len: u64,
    part: u32,
    cur: Vec<u8>,
    hasher: Sha256,
    written: u64,
}

impl<E> ChunkWriter<'_, E> {
    /// Fresh chunk buffer sized for what is left, never over `max_chunk`.
    fn fresh(&self) -> Vec<u8> {
        let placed = u64::from(self.part) * self.body_cap as u64;
        let left = self
            .payload_len
            .saturating_sub(placed)
            .min(self.body_cap as u64);
        let mut v = Vec::with_capacity(CHUNK_HEADER_LEN + left as usize);
        v.resize(CHUNK_HEADER_LEN, 0);
        v
    }

    fn emit(&mut self, mut b: Vec<u8>, last: bool, digest: &[u8; 32]) {
        seal(
            &mut b,
            self.kind,
            last,
            self.epoch,
            self.part,
            self.total,
            self.payload_len,
            digest,
        );
        if self.err.is_none() {
            let chunk = IndexChunk {
                epoch: self.epoch,
                part: self.part,
                bytes: b,
            };
            if let Err(e) = (self.emit)(chunk) {
                self.err = Some(e);
            }
        }
        self.part += 1;
    }

    fn finish(mut self) -> Result<IndexDigest, EmitError<E>> {
        if self.written != self.payload_len || self.part + 1 != self.total {
            return Err(EmitError::Encode(EncodeError::Invariant("payload length")));
        }
        let sha256: [u8; 32] = std::mem::take(&mut self.hasher).finalize().into();
        let last = std::mem::take(&mut self.cur);
        self.emit(last, true, &sha256);
        if let Some(e) = self.err {
            return Err(EmitError::Sink(e));
        }
        Ok(IndexDigest {
            epoch: self.epoch,
            kind: self.kind,
            parts: self.total,
            payload_len: self.payload_len,
            sha256,
        })
    }
}

impl<E> Sink for ChunkWriter<'_, E> {
    fn put(&mut self, mut b: &[u8]) {
        self.hasher.update(b);
        self.written += b.len() as u64;
        while !b.is_empty() {
            if self.cur.len() == CHUNK_HEADER_LEN + self.body_cap {
                // More bytes follow, so this full chunk is not the last.
                let full = std::mem::take(&mut self.cur);
                self.emit(full, false, &[0; 32]);
                self.cur = self.fresh();
            }
            let n = (CHUNK_HEADER_LEN + self.body_cap - self.cur.len()).min(b.len());
            self.cur.extend_from_slice(&b[..n]);
            b = &b[n..];
        }
    }
}

/// Encode the payload `write` produces, streaming chunks into `emit`.
pub(crate) fn encode<E>(
    epoch: u64,
    kind: IndexKind,
    max_chunk: usize,
    write: &dyn Fn(&mut dyn Sink),
    emit: &mut dyn FnMut(IndexChunk) -> Result<(), E>,
) -> Result<IndexDigest, EmitError<E>> {
    if max_chunk <= CHUNK_HEADER_LEN || max_chunk > MAX_CHUNK_BYTES {
        return Err(EmitError::Encode(EncodeError::BadChunkSize(max_chunk)));
    }
    let mut count = Counter(0);
    write(&mut count);
    let body_cap = max_chunk - CHUNK_HEADER_LEN;
    let total = count.0.div_ceil(body_cap as u64).max(1);
    let total = u32::try_from(total)
        .map_err(|_| EmitError::Encode(EncodeError::Invariant("too many parts")))?;
    let mut w = ChunkWriter {
        emit,
        err: None,
        epoch,
        kind,
        body_cap,
        total,
        payload_len: count.0,
        part: 0,
        cur: Vec::new(),
        hasher: Sha256::new(),
        written: 0,
    };
    w.cur = w.fresh();
    write(&mut w);
    w.finish()
}

/// [`encode`] into memory.
pub(crate) fn encode_all(
    epoch: u64,
    kind: IndexKind,
    max_chunk: usize,
    write: &dyn Fn(&mut dyn Sink),
) -> Result<EncodedIndex, EncodeError> {
    let mut chunks = Vec::new();
    let d = encode::<std::convert::Infallible>(epoch, kind, max_chunk, write, &mut |c| {
        chunks.push(c);
        Ok(())
    })
    .map_err(|e| match e {
        EmitError::Encode(e) => e,
        EmitError::Sink(never) => match never {},
    })?;
    Ok(EncodedIndex {
        epoch,
        kind,
        chunks,
        sha256: d.sha256,
        payload_len: d.payload_len,
    })
}

/// Recompute the payload length, the last part's digest and every CRC
/// after the bodies were edited. **Test/fuzz helper only**: it lets a
/// mutated payload reach structural validation past the integrity checks.
#[doc(hidden)]
pub fn reseal(chunks: &mut [IndexChunk]) {
    let mut h = Sha256::new();
    let mut len = 0u64;
    for c in chunks.iter() {
        if let Some(body) = c.bytes.get(CHUNK_HEADER_LEN..) {
            h.update(body);
            len += body.len() as u64;
        }
    }
    let digest: [u8; 32] = h.finalize().into();
    let n = chunks.len();
    for (i, c) in chunks.iter_mut().enumerate() {
        let b = &mut c.bytes;
        if b.len() < CHUNK_HEADER_LEN {
            continue;
        }
        let body_len = (b.len() - CHUNK_HEADER_LEN) as u32;
        b[24..28].copy_from_slice(&body_len.to_le_bytes());
        b[28..36].copy_from_slice(&len.to_le_bytes());
        if i + 1 == n {
            b[DIGEST_AT..CRC_AT].copy_from_slice(&digest);
        }
        put_crc(b);
    }
}

#[cfg(test)]
mod tests;
