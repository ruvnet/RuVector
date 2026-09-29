//! Pull-based chunk validation feeding the payload [`Reader`].

use sha2::{Digest, Sha256};

use super::{chunk_crc, rd_u32, rd_u64, IndexChunk, IndexKind};
use super::{CHUNK_HEADER_LEN, CRC_AT, DIGEST_AT, FLAG_LAST, MAGIC, MAX_CHUNK_BYTES, VERSION};
use crate::bytes::{Reader, Source};
use crate::error::DecodeError;

/// Header fields every part must agree on: version, kind, epoch, total
/// parts, payload length.
fn shared(b: &[u8]) -> [u8; 24] {
    let mut s = [0u8; 24];
    s[..3].copy_from_slice(&b[4..7]);
    s[3..11].copy_from_slice(&b[8..16]);
    s[11..15].copy_from_slice(&b[20..24]);
    s[15..23].copy_from_slice(&b[28..36]);
    s
}

/// Self-contained checks of one row: size, magic, version, CRC, and the
/// row's `(epoch, part)` against both its header and the expected part.
fn check_row(c: &IndexChunk, expected: u32) -> Result<(), DecodeError> {
    let (b, part) = (&c.bytes, c.part);
    if b.len() < CHUNK_HEADER_LEN {
        return Err(DecodeError::Truncated { part });
    }
    if b[0..4] != MAGIC {
        return Err(DecodeError::BadMagic { part });
    }
    let ver = u16::from_le_bytes([b[4], b[5]]);
    if ver != VERSION {
        return Err(DecodeError::UnsupportedVersion(ver));
    }
    if (CHUNK_HEADER_LEN as u64) + u64::from(rd_u32(b, 24)) != b.len() as u64 {
        return Err(DecodeError::Truncated { part });
    }
    if chunk_crc(b) != rd_u32(b, CRC_AT) {
        return Err(DecodeError::ChecksumMismatch { part });
    }
    if rd_u64(b, 8) != c.epoch || rd_u32(b, 16) != part {
        return Err(DecodeError::RowMismatch { part });
    }
    if part != expected {
        return Err(DecodeError::PartOutOfOrder {
            expected,
            got: part,
        });
    }
    Ok(())
}

/// Validating chunk stream.
struct ChunkStream<'a> {
    it: &'a mut dyn Iterator<Item = IndexChunk>,
    pending: Option<Vec<u8>>,
    head: [u8; 24],
    total: u32,
    payload_len: u64,
    next: u32,
    body_total: u64,
    hasher: Sha256,
    digest: Option<[u8; 32]>,
}

impl<'a> ChunkStream<'a> {
    fn open(
        it: &'a mut dyn Iterator<Item = IndexChunk>,
        kind: IndexKind,
        max_payload: u64,
    ) -> Result<Self, DecodeError> {
        let first = it.next().ok_or(DecodeError::Empty)?;
        check_row(&first, 0)?;
        let b = &first.bytes;
        if b[6] != kind.code() {
            return Err(DecodeError::WrongKind);
        }
        let total = rd_u32(b, 20);
        let payload_len = rd_u64(b, 28);
        let cap = u64::from(total).saturating_mul((MAX_CHUNK_BYTES - CHUNK_HEADER_LEN) as u64);
        if total == 0 || payload_len > cap {
            return Err(DecodeError::Inconsistent { part: 0 });
        }
        if payload_len > max_payload {
            return Err(DecodeError::TooLarge {
                declared: payload_len,
                limit: max_payload,
            });
        }
        let mut s = Self {
            it,
            pending: None,
            head: shared(b),
            total,
            payload_len,
            next: 0,
            body_total: 0,
            hasher: Sha256::new(),
            digest: None,
        };
        s.accept(&first)?;
        s.pending = Some(first.bytes);
        Ok(s)
    }

    /// Cross-part checks, then hash the body.
    fn accept(&mut self, c: &IndexChunk) -> Result<(), DecodeError> {
        let (b, part) = (&c.bytes, c.part);
        if shared(b) != self.head {
            return Err(DecodeError::Inconsistent { part });
        }
        let last = part + 1 == self.total;
        let flagged = b[7] & FLAG_LAST != 0;
        let digest = &b[DIGEST_AT..CRC_AT];
        if b[7] & !FLAG_LAST != 0 || flagged != last || (!last && digest.iter().any(|&x| x != 0)) {
            return Err(DecodeError::Inconsistent { part });
        }
        let body = &b[CHUNK_HEADER_LEN..];
        self.body_total += body.len() as u64;
        if self.body_total > self.payload_len {
            return Err(DecodeError::Inconsistent { part });
        }
        self.hasher.update(body);
        if last {
            let mut d = [0u8; 32];
            d.copy_from_slice(digest);
            self.digest = Some(d);
        }
        self.next = part + 1;
        Ok(())
    }

    fn pull(&mut self) -> Result<Option<Vec<u8>>, DecodeError> {
        if let Some(b) = self.pending.take() {
            return Ok(Some(b));
        }
        if self.next == self.total {
            return Ok(None);
        }
        let c = self.it.next().ok_or(DecodeError::PartCount {
            expected: self.total,
            got: self.next,
        })?;
        check_row(&c, self.next)?;
        self.accept(&c)?;
        Ok(Some(c.bytes))
    }

    /// All parts present, nothing extra, lengths and digests agree.
    fn finish(mut self, expected: Option<&[u8; 32]>) -> Result<(), DecodeError> {
        self.pending = None;
        while self.pull()?.is_some() {}
        if self.it.next().is_some() {
            return Err(DecodeError::PartCount {
                expected: self.total,
                got: self.total.saturating_add(1),
            });
        }
        let digest: [u8; 32] = std::mem::take(&mut self.hasher).finalize().into();
        if self.body_total != self.payload_len || self.digest != Some(digest) {
            return Err(DecodeError::DigestMismatch);
        }
        if expected.is_some_and(|want| want != &digest) {
            return Err(DecodeError::DigestMismatch);
        }
        Ok(())
    }
}

impl Source for ChunkStream<'_> {
    fn next_body(&mut self) -> Result<Option<(Vec<u8>, usize)>, DecodeError> {
        Ok(self.pull()?.map(|b| (b, CHUNK_HEADER_LEN)))
    }
}

/// Validate and parse a chunk stream. `parse` sees only the payload; its
/// result is returned only if every part and the final digest check out.
pub(crate) fn decode<T>(
    it: &mut dyn Iterator<Item = IndexChunk>,
    kind: IndexKind,
    expected: Option<&[u8; 32]>,
    max_payload: u64,
    parse: impl FnOnce(&mut Reader<'_>) -> Result<T, DecodeError>,
) -> Result<T, DecodeError> {
    let mut s = ChunkStream::open(it, kind, max_payload)?;
    let declared = usize::try_from(s.payload_len).map_err(|_| DecodeError::TooLarge {
        declared: s.payload_len,
        limit: usize::MAX as u64,
    })?;
    let out = {
        let mut r = Reader::new(&mut s, declared);
        let out = parse(&mut r)?;
        r.finish()?;
        out
    };
    s.finish(expected)?;
    Ok(out)
}

/// [`decode`] over rows already in memory. The declared payload may not
/// exceed the bodies actually supplied; when it does, the precise cause
/// (a short row or missing parts) is reported instead of `TooLarge`.
pub(crate) fn decode_slice<T>(
    chunks: &[IndexChunk],
    kind: IndexKind,
    expected: Option<&[u8; 32]>,
    parse: impl FnOnce(&mut Reader<'_>) -> Result<T, DecodeError>,
) -> Result<T, DecodeError> {
    let supplied: u64 = chunks
        .iter()
        .map(|c| c.bytes.len().saturating_sub(CHUNK_HEADER_LEN) as u64)
        .sum();
    match decode(&mut chunks.iter().cloned(), kind, expected, supplied, parse) {
        Err(DecodeError::TooLarge { .. }) => Err(short_rows(chunks)),
        other => other,
    }
}

fn short_rows(chunks: &[IndexChunk]) -> DecodeError {
    for (i, c) in chunks.iter().enumerate() {
        let b = &c.bytes;
        if b.len() < CHUNK_HEADER_LEN
            || (CHUNK_HEADER_LEN as u64) + u64::from(rd_u32(b, 24)) != b.len() as u64
        {
            return DecodeError::Truncated { part: c.part };
        }
        if c.part as usize != i {
            return DecodeError::PartOutOfOrder {
                expected: i as u32,
                got: c.part,
            };
        }
    }
    let total = chunks.first().map_or(0, |c| rd_u32(&c.bytes, 20));
    if total as usize > chunks.len() {
        return DecodeError::PartCount {
            expected: total,
            got: chunks.len() as u32,
        };
    }
    DecodeError::DigestMismatch
}
