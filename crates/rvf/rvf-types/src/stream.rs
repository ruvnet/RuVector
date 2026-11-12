//! RVF Stream Descriptor (RVSD) fixed-layout types (ADR-353).
//!
//! `no_std`, allocation free. The header and the 72-byte entry convert to and
//! from byte arrays only; the table `Vec` lives in `rvf-wire`.
//! All integers are little-endian.

use crate::error::ErrorCode;

/// Descriptor magic, compared as raw bytes (not as a little-endian `u32`).
pub const STREAM_MAGIC: [u8; 4] = *b"RVSD";
/// The only descriptor version.
pub const STREAM_DESC_VERSION: u16 = 1;
/// Size of the fixed descriptor header (offset of the first entry).
pub const STREAM_HEADER_SIZE: usize = 0x90;
/// Size of one table entry.
pub const STREAM_ENTRY_SIZE: usize = 72;
/// Ed25519 signature length.
pub const STREAM_SIG_LEN: usize = 64;
/// Bytes after the table: `sig_len u16` plus the signature.
pub const STREAM_TRAILER_SIZE: usize = 2 + STREAM_SIG_LEN;
/// Domain-separation prefix of the signed message.
pub const STREAM_SIG_DOMAIN: &[u8] = b"RVF-v1-stream-descriptor\0";

/// Flag bit0: chunk tables are present (`chunk_log2 != 0xFF`).
pub const STREAM_FLAG_CHUNKED: u16 = 1;
/// Flag bit1: `rvf_identity` is meaningful and checked at completion.
pub const STREAM_FLAG_IDENTITY_PRESENT: u16 = 2;
/// Flag bit2: nothing may execute before `Complete`.
pub const STREAM_FLAG_REQUIRE_COMPLETE_BEFORE_EXEC: u16 = 4;
/// All defined flag bits.
pub const STREAM_FLAGS_KNOWN: u16 = 7;

/// `hash_alg` value for SHA-256.
pub const STREAM_HASH_SHA256: u8 = 1;
/// `sig_alg` value for Ed25519 (`verify_strict`).
pub const STREAM_SIG_ED25519: u16 = 0;
/// `chunk_log2` value for the segment-granular profile.
pub const STREAM_CHUNK_LOG2_SEGMENT: u8 = 0xFF;
/// Smallest chunked `chunk_log2` (4 KiB).
pub const STREAM_CHUNK_LOG2_MIN: u8 = 12;
/// Largest chunked `chunk_log2` (1 GiB).
pub const STREAM_CHUNK_LOG2_MAX: u8 = 30;
/// Descriptor-local `seg_type` of the trailing Level-0 root page pseudo-entry.
pub const STREAM_SEG_TYPE_ROOT_PAGE: u8 = 0xFE;
/// Number of delivery stages addressable by `requires_mask`.
pub const STREAM_MAX_STAGES: u8 = 32;

/// Segment class used by the per-class size limits (ADR-353 section 2.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamClass {
    /// Wasm, Kernel, eBPF.
    Executable = 0,
    /// Quantization dictionaries and codebooks.
    Model = 1,
    /// Index, hot cache and metadata index segments.
    Index = 2,
    /// Everything else (vectors, journal, manifests, root page, ...).
    State = 3,
}

impl StreamClass {
    /// Number of classes.
    pub const COUNT: usize = 4;

    /// Classify a segment type byte.
    pub const fn of(seg_type: u8) -> Self {
        match seg_type {
            0x0E..=0x10 => Self::Executable,
            0x06 => Self::Model,
            0x02 | 0x08 | 0x0D => Self::Index,
            _ => Self::State,
        }
    }
}

/// Fixed 0x90-byte descriptor header, after static validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamHeader {
    /// Flag bits (`STREAM_FLAG_*`).
    pub flags: u16,
    /// `chunk_log2` (`0xFF` or `12..=30`).
    pub chunk_log2: u8,
    /// Artifact lineage id.
    pub file_id: [u8; 16],
    /// Monotone epoch.
    pub epoch: u64,
    /// Total body bytes.
    pub body_len: u64,
    /// SHA-256 of the body (zero unless `identity_present`).
    pub rvf_identity: [u8; 32],
    /// Capability policy hash.
    pub policy_hash: [u8; 32],
    /// Number of table entries.
    pub seg_count: u32,
    /// Table byte length (`seg_count * 72`).
    pub table_len: u32,
    /// Expiry in ns, `0` = none.
    pub expires_ns: u64,
    /// First 8 bytes of SHA-256(public key).
    pub signer_key_id: [u8; 8],
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}
fn arr<const N: usize>(b: &[u8], o: usize) -> [u8; N] {
    let mut a = [0u8; N];
    a.copy_from_slice(&b[o..o + N]);
    a
}

impl StreamHeader {
    /// Parse and statically validate the first `STREAM_HEADER_SIZE` bytes.
    ///
    /// Checks magic, version, flags, algorithms, `chunk_log2`, reserved bytes
    /// and `table_len == seg_count * 72`. Does not look at the table.
    pub fn parse(b: &[u8]) -> Result<Self, ErrorCode> {
        if b.len() < STREAM_HEADER_SIZE {
            return Err(ErrorCode::StreamDescriptorTruncated);
        }
        if b[0..4] != STREAM_MAGIC {
            return Err(ErrorCode::StreamBadMagic);
        }
        if u16_at(b, 4) != STREAM_DESC_VERSION {
            return Err(ErrorCode::StreamUnsupportedVersion);
        }
        let flags = u16_at(b, 6);
        let chunk_log2 = b[9];
        let chunked_ok = if chunk_log2 == STREAM_CHUNK_LOG2_SEGMENT {
            flags & STREAM_FLAG_CHUNKED == 0
        } else {
            (STREAM_CHUNK_LOG2_MIN..=STREAM_CHUNK_LOG2_MAX).contains(&chunk_log2)
                && flags & STREAM_FLAG_CHUNKED != 0
        };
        if flags & !STREAM_FLAGS_KNOWN != 0
            || b[8] != STREAM_HASH_SHA256
            || u16_at(b, 0x0A) != STREAM_SIG_ED25519
            || u32_at(b, 0x0C) != 0
            || b[0x88..0x90] != [0u8; 8]
            || !chunked_ok
        {
            return Err(ErrorCode::StreamBadFlags);
        }
        let seg_count = u32_at(b, 0x70);
        let table_len = u32_at(b, 0x74);
        if (seg_count as u64) * (STREAM_ENTRY_SIZE as u64) != table_len as u64 {
            return Err(ErrorCode::StreamTableInvalid);
        }
        let rvf_identity: [u8; 32] = arr(b, 0x30);
        if flags & STREAM_FLAG_IDENTITY_PRESENT == 0 && rvf_identity != [0u8; 32] {
            return Err(ErrorCode::StreamBadFlags);
        }
        Ok(Self {
            flags,
            chunk_log2,
            file_id: arr(b, 0x10),
            epoch: u64_at(b, 0x20),
            body_len: u64_at(b, 0x28),
            rvf_identity,
            policy_hash: arr(b, 0x50),
            seg_count,
            table_len,
            expires_ns: u64_at(b, 0x78),
            signer_key_id: arr(b, 0x80),
        })
    }

    /// Serialize to the 0x90-byte wire form.
    pub fn to_bytes(&self) -> [u8; STREAM_HEADER_SIZE] {
        let mut b = [0u8; STREAM_HEADER_SIZE];
        b[0..4].copy_from_slice(&STREAM_MAGIC);
        b[4..6].copy_from_slice(&STREAM_DESC_VERSION.to_le_bytes());
        b[6..8].copy_from_slice(&self.flags.to_le_bytes());
        b[8] = STREAM_HASH_SHA256;
        b[9] = self.chunk_log2;
        b[0x0A..0x0C].copy_from_slice(&STREAM_SIG_ED25519.to_le_bytes());
        b[0x10..0x20].copy_from_slice(&self.file_id);
        b[0x20..0x28].copy_from_slice(&self.epoch.to_le_bytes());
        b[0x28..0x30].copy_from_slice(&self.body_len.to_le_bytes());
        b[0x30..0x50].copy_from_slice(&self.rvf_identity);
        b[0x50..0x70].copy_from_slice(&self.policy_hash);
        b[0x70..0x74].copy_from_slice(&self.seg_count.to_le_bytes());
        b[0x74..0x78].copy_from_slice(&self.table_len.to_le_bytes());
        b[0x78..0x80].copy_from_slice(&self.expires_ns.to_le_bytes());
        b[0x80..0x88].copy_from_slice(&self.signer_key_id);
        b
    }

    /// Chunk size in bytes, `None` for the segment-granular profile.
    pub const fn chunk_size(&self) -> Option<u64> {
        if self.chunk_log2 == STREAM_CHUNK_LOG2_SEGMENT {
            None
        } else {
            Some(1u64 << self.chunk_log2)
        }
    }

    /// Number of chunks of a segment of `stored_len` bytes (always >= 1).
    ///
    /// `None` if the count does not fit a `u32`.
    pub fn chunk_count(&self, stored_len: u64) -> Option<u32> {
        match self.chunk_size() {
            None => Some(1),
            Some(cs) if stored_len <= cs => Some(1),
            Some(cs) => u32::try_from(stored_len.div_ceil(cs)).ok(),
        }
    }

    /// Length of chunk `index` of a segment of `stored_len` bytes.
    pub fn chunk_len(&self, stored_len: u64, index: u32) -> Option<u64> {
        let n = self.chunk_count(stored_len)?;
        if index >= n {
            return None;
        }
        match self.chunk_size() {
            Some(cs) if n > 1 => {
                if index + 1 < n {
                    Some(cs)
                } else {
                    Some(stored_len - cs * (n as u64 - 1))
                }
            }
            _ => Some(stored_len),
        }
    }
}

/// One 72-byte table entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamEntry {
    /// Segment id (unique within the descriptor).
    pub seg_id: u64,
    /// Segment type byte (`0xFE` = root page pseudo-entry).
    pub seg_type: u8,
    /// Delivery stage `0..32`.
    pub stage: u8,
    /// Entry flags (reserved, carried verbatim).
    pub flags: u16,
    /// Bit i set: stage i must be fully verified before release.
    pub requires_mask: u32,
    /// Offset in the body.
    pub body_offset: u64,
    /// Whole stored segment length (header, payload, footer, padding).
    pub stored_len: u64,
    /// Declared uncompressed length (bounded by policy).
    pub uncompressed_len: u64,
    /// Leaf hash (one chunk) or chunk-list hash (several chunks).
    pub chunk_table_hash: [u8; 32],
}

impl StreamEntry {
    /// Decode from exactly 72 bytes.
    pub fn from_bytes(b: &[u8; STREAM_ENTRY_SIZE]) -> Self {
        Self {
            seg_id: u64_at(b, 0),
            seg_type: b[8],
            stage: b[9],
            flags: u16_at(b, 10),
            requires_mask: u32_at(b, 12),
            body_offset: u64_at(b, 16),
            stored_len: u64_at(b, 24),
            uncompressed_len: u64_at(b, 32),
            chunk_table_hash: arr(b, 40),
        }
    }

    /// Encode to 72 bytes.
    pub fn to_bytes(&self) -> [u8; STREAM_ENTRY_SIZE] {
        let mut b = [0u8; STREAM_ENTRY_SIZE];
        b[0..8].copy_from_slice(&self.seg_id.to_le_bytes());
        b[8] = self.seg_type;
        b[9] = self.stage;
        b[10..12].copy_from_slice(&self.flags.to_le_bytes());
        b[12..16].copy_from_slice(&self.requires_mask.to_le_bytes());
        b[16..24].copy_from_slice(&self.body_offset.to_le_bytes());
        b[24..32].copy_from_slice(&self.stored_len.to_le_bytes());
        b[32..40].copy_from_slice(&self.uncompressed_len.to_le_bytes());
        b[40..72].copy_from_slice(&self.chunk_table_hash);
        b
    }
}

/// Exact descriptor length for `seg_count` entries, `None` on overflow.
pub fn stream_descriptor_len(seg_count: u32) -> Option<usize> {
    (seg_count as usize)
        .checked_mul(STREAM_ENTRY_SIZE)?
        .checked_add(STREAM_HEADER_SIZE + STREAM_TRAILER_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StreamHeader {
        StreamHeader {
            flags: STREAM_FLAG_CHUNKED | STREAM_FLAG_IDENTITY_PRESENT,
            chunk_log2: 16,
            file_id: [7; 16],
            epoch: 9,
            body_len: 123,
            rvf_identity: [1; 32],
            policy_hash: [2; 32],
            seg_count: 3,
            table_len: 216,
            expires_ns: 0,
            signer_key_id: [3; 8],
        }
    }

    #[test]
    fn header_roundtrip_and_rejections() {
        let h = sample();
        assert_eq!(StreamHeader::parse(&h.to_bytes()), Ok(h));
        let mut b = h.to_bytes();
        b[0] ^= 1;
        assert_eq!(StreamHeader::parse(&b), Err(ErrorCode::StreamBadMagic));
        let mut b = h.to_bytes();
        b[4] = 2;
        assert_eq!(
            StreamHeader::parse(&b),
            Err(ErrorCode::StreamUnsupportedVersion)
        );
        let mut b = h.to_bytes();
        b[6] |= 0x80;
        assert_eq!(StreamHeader::parse(&b), Err(ErrorCode::StreamBadFlags));
        let mut b = h.to_bytes();
        b[0x0C] = 1;
        assert_eq!(StreamHeader::parse(&b), Err(ErrorCode::StreamBadFlags));
        let mut b = h.to_bytes();
        b[0x74] = 0;
        assert_eq!(StreamHeader::parse(&b), Err(ErrorCode::StreamTableInvalid));
        assert_eq!(
            StreamHeader::parse(&b[..10]),
            Err(ErrorCode::StreamDescriptorTruncated)
        );
    }

    #[test]
    fn entry_roundtrip() {
        let e = StreamEntry {
            seg_id: 5,
            seg_type: 0x10,
            stage: 2,
            flags: 0,
            requires_mask: 3,
            body_offset: 64,
            stored_len: 1000,
            uncompressed_len: 2000,
            chunk_table_hash: [9; 32],
        };
        assert_eq!(StreamEntry::from_bytes(&e.to_bytes()), e);
    }

    #[test]
    fn chunk_geometry() {
        let h = sample();
        assert_eq!(h.chunk_count(65536), Some(1));
        assert_eq!(h.chunk_count(65537), Some(2));
        assert_eq!(h.chunk_len(65537, 1), Some(1));
        assert_eq!(h.chunk_len(65537, 2), None);
        let mut s = h;
        s.chunk_log2 = STREAM_CHUNK_LOG2_SEGMENT;
        s.flags = 0;
        assert_eq!(s.chunk_count(u64::MAX), Some(1));
        assert_eq!(s.chunk_len(77, 0), Some(77));
    }
}
