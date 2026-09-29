//! Persist v2 wire layout: constants, header codec, CRC-32 and the rotation
//! fingerprint. All integers little-endian.
//!
//! ```text
//! header frame (64 B)
//!   0  magic "rbqx0002"          8  version u16 = 2     10 flags u16 = 0
//!   12 dim u32                   16 n u64
//!   24 rotation u8 (0 Haar, 1 Hadamard)   25 metric u8 (0 cos, 1 l2, 2 dot)
//!   26 reserved [0; 2]           28 max_frame_bytes u32 (≤ 1 MiB)
//!   32 seed u64                  40 rotation fingerprint u64
//!   48 data frames u32           52 payload bytes u64
//!   60 crc32(header[0..60])
//! data frame (16 B prefix + payload, whole frame ≤ max_frame_bytes)
//!   0 index u32 (0-based, contiguous)   4 section u8 (1 keys, 2 norms, 3 codes)
//!   5 reserved [0; 3]                   8 payload len u32
//!   12 crc32(prefix[0..12] ‖ payload)
//! footer frame (16 B)
//!   0 magic "rbqx_end"   8 data frames u32   12 crc32(all frame crcs, LE)
//! ```
//! Sections appear in order keys → norms → codes, each split into frames of
//! a fixed payload capacity (the last frame of a section may be short).

use crate::error::{CorruptKind, QuantError, Result};
use ruvector_edge_store::Metric;
use ruvector_rabitq::{RandomRotation, RandomRotationKind};

/// Header magic.
pub const MAGIC: &[u8; 8] = b"rbqx0002";
/// Footer magic.
pub const FOOTER_MAGIC: &[u8; 8] = b"rbqx_end";
/// Format version.
pub const VERSION: u16 = 2;
/// Header frame length.
pub const HEADER_LEN: usize = 64;
/// Data frame prefix length.
pub const FRAME_PREFIX_LEN: usize = 16;
/// Footer frame length.
pub const FOOTER_LEN: usize = 16;
/// Hard ceiling for any frame (ADR-351 chunking: ≤ 1 MiB).
pub const MAX_FRAME_BYTES: usize = 1 << 20;
/// Smallest frame size a writer may choose.
pub const MIN_FRAME_BYTES: usize = 64;

/// Section tags.
pub const SECTION_KEYS: u8 = 1;
/// Norms section.
pub const SECTION_NORMS: u8 = 2;
/// Codes section.
pub const SECTION_CODES: u8 = 3;

/// Decoded header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Dimension.
    pub dim: u32,
    /// Rows.
    pub n: u64,
    /// Rotation kind.
    pub rotation: RandomRotationKind,
    /// Metric.
    pub metric: Metric,
    /// Frame size ceiling the writer used.
    pub max_frame_bytes: u32,
    /// Rotation seed.
    pub seed: u64,
    /// Rotation fingerprint.
    pub fingerprint: u64,
    /// Data frames.
    pub frames: u32,
    /// Sum of data payload lengths.
    pub payload_bytes: u64,
}

/// Payload capacity of a data frame (multiple of 8 so no element straddles).
pub fn payload_cap(max_frame_bytes: usize) -> usize {
    (max_frame_bytes - FRAME_PREFIX_LEN) & !7
}

/// Section byte lengths for `n` rows at `dim`: (keys, norms, codes).
pub fn section_lens(n: u64, dim: usize) -> Option<[u64; 3]> {
    let nw = dim.div_ceil(64) as u64;
    let lens = [
        n.checked_mul(8)?,
        n.checked_mul(4)?,
        n.checked_mul(nw)?.checked_mul(8)?,
    ];
    // The total must fit too (callers sum the sections).
    lens.iter().try_fold(0u64, |a, l| a.checked_add(*l))?;
    Some(lens)
}

/// Data frames needed for the given section lengths.
pub fn frame_count(lens: &[u64; 3], cap: usize) -> u64 {
    lens.iter().map(|l| l.div_ceil(cap as u64)).sum()
}

fn rotation_tag(k: RandomRotationKind) -> u8 {
    match k {
        RandomRotationKind::HaarDense => 0,
        RandomRotationKind::HadamardSigned => 1,
    }
}

fn metric_tag(m: Metric) -> u8 {
    match m {
        Metric::Cosine => 0,
        Metric::L2 => 1,
        Metric::Dot => 2,
    }
}

impl Header {
    /// Encode (64 bytes, CRC last).
    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut h = [0u8; HEADER_LEN];
        h[0..8].copy_from_slice(MAGIC);
        h[8..10].copy_from_slice(&VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&self.dim.to_le_bytes());
        h[16..24].copy_from_slice(&self.n.to_le_bytes());
        h[24] = rotation_tag(self.rotation);
        h[25] = metric_tag(self.metric);
        h[28..32].copy_from_slice(&self.max_frame_bytes.to_le_bytes());
        h[32..40].copy_from_slice(&self.seed.to_le_bytes());
        h[40..48].copy_from_slice(&self.fingerprint.to_le_bytes());
        h[48..52].copy_from_slice(&self.frames.to_le_bytes());
        h[52..60].copy_from_slice(&self.payload_bytes.to_le_bytes());
        let crc = crc32(&h[0..60]);
        h[60..64].copy_from_slice(&crc.to_le_bytes());
        h
    }

    /// Decode and structurally validate (magic, version, CRC, tags, sizes).
    pub fn decode(h: &[u8]) -> Result<Header> {
        let bad = |k| Err(QuantError::corrupt(k));
        if h.len() < 8 || &h[0..8] != MAGIC {
            return if h.len() < 8 {
                bad(CorruptKind::Truncated)
            } else {
                bad(CorruptKind::BadMagic)
            };
        }
        if h.len() < HEADER_LEN {
            return bad(CorruptKind::Truncated);
        }
        if h.len() > HEADER_LEN {
            return bad(CorruptKind::Malformed("header frame length"));
        }
        let version = le_u16(&h[8..10]);
        if version != VERSION {
            return bad(CorruptKind::UnsupportedVersion(version));
        }
        if le_u32(&h[60..64]) != crc32(&h[0..60]) {
            return bad(CorruptKind::HeaderChecksum);
        }
        if le_u16(&h[10..12]) != 0 || h[26] != 0 || h[27] != 0 {
            return bad(CorruptKind::Malformed("reserved header bits"));
        }
        let rotation = match h[24] {
            0 => RandomRotationKind::HaarDense,
            1 => RandomRotationKind::HadamardSigned,
            _ => return bad(CorruptKind::Malformed("rotation kind")),
        };
        let metric = match h[25] {
            0 => Metric::Cosine,
            1 => Metric::L2,
            2 => Metric::Dot,
            _ => return bad(CorruptKind::Malformed("metric")),
        };
        let hd = Header {
            dim: le_u32(&h[12..16]),
            n: le_u64(&h[16..24]),
            rotation,
            metric,
            max_frame_bytes: le_u32(&h[28..32]),
            seed: le_u64(&h[32..40]),
            fingerprint: le_u64(&h[40..48]),
            frames: le_u32(&h[48..52]),
            payload_bytes: le_u64(&h[52..60]),
        };
        let mfb = hd.max_frame_bytes as usize;
        if !(MIN_FRAME_BYTES..=MAX_FRAME_BYTES).contains(&mfb) {
            return bad(CorruptKind::Malformed("max_frame_bytes"));
        }
        if hd.dim == 0 || hd.dim as usize > crate::shard::MAX_DIM {
            return bad(CorruptKind::Malformed("dim"));
        }
        let lens = section_lens(hd.n, hd.dim as usize)
            .ok_or(QuantError::corrupt(CorruptKind::Malformed("row count")))?;
        if lens.iter().sum::<u64>() != hd.payload_bytes {
            return bad(CorruptKind::Malformed("payload bytes"));
        }
        if frame_count(&lens, payload_cap(mfb)) != u64::from(hd.frames) {
            return bad(CorruptKind::Malformed("frame count"));
        }
        Ok(hd)
    }
}

/// Little-endian readers (callers pass exact-length slices).
pub fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
/// `u32` LE.
pub fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
/// `u64` LE.
pub fn le_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_le_bytes(a)
}

/// Slicing-by-8 tables: `CRC_TABLES[0]` is the classic byte table,
/// `CRC_TABLES[k][i]` the CRC of byte `i` followed by `k` zero bytes. Eight
/// bytes per step instead of one: the persist v2 checksums cover every
/// snapshot byte (≈ 3 MB for 50k × 384) on both flush and cold load, and a
/// Worker has ≈ 10 ms per turn on the Free plan. Same CRC-32 values.
const CRC_TABLES: [[u32; 256]; 8] = {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = t[s - 1][i];
            t[s][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            i += 1;
        }
        s += 1;
    }
    t
};

/// Streaming CRC-32 (IEEE 802.3, reflected).
#[derive(Clone, Copy)]
pub struct Crc32(u32);
impl Crc32 {
    /// Fresh state.
    pub fn new() -> Self {
        Crc32(!0)
    }
    /// Absorb bytes.
    pub fn update(&mut self, data: &[u8]) {
        let t = &CRC_TABLES;
        let mut c = self.0;
        let mut words = data.chunks_exact(8);
        for w in words.by_ref() {
            let lo = c ^ u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            c = t[7][(lo & 0xFF) as usize]
                ^ t[6][((lo >> 8) & 0xFF) as usize]
                ^ t[5][((lo >> 16) & 0xFF) as usize]
                ^ t[4][(lo >> 24) as usize]
                ^ t[3][w[4] as usize]
                ^ t[2][w[5] as usize]
                ^ t[1][w[6] as usize]
                ^ t[0][w[7] as usize];
        }
        for &b in words.remainder() {
            c = t[0][((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
        }
        self.0 = c;
    }
    /// Final value.
    pub fn finish(self) -> u32 {
        !self.0
    }
}
impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot CRC-32.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}

/// FNV-1a over the f32 bit patterns of the rotation applied to a fixed
/// probe vector: pins the regenerated rotation to the one the codes were
/// built with (catches a seed/kind mix-up or a platform RNG/float drift).
pub fn rotation_fingerprint(rotation: &RandomRotation) -> u64 {
    let dim = rotation.dim;
    let probe: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 + 1.0) * 0.618_034).fract() - 0.5)
        .collect();
    let out = rotation.apply(&probe);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in out {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    /// Slicing-by-8 equals the bytewise definition at every length and
    /// split point (streaming updates of any size).
    #[test]
    fn crc32_slicing_matches_bytewise() {
        let bytewise = |d: &[u8]| {
            let mut c = !0u32;
            for &b in d {
                c = CRC_TABLES[0][((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
            }
            !c
        };
        let data: Vec<u8> = (0..300u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        for len in 0..data.len() {
            let d = &data[..len];
            assert_eq!(crc32(d), bytewise(d), "len {len}");
            let mut s = Crc32::new();
            let (a, b) = d.split_at(len / 3);
            s.update(a);
            s.update(b);
            assert_eq!(s.finish(), bytewise(d), "split {len}");
        }
    }
}
