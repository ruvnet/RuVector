//! Persist v2 reader. Cold load decodes packed codes straight into the
//! shard's arrays — nothing is re-encoded from f32 originals; the only
//! recomputation is the seeded rotation, verified against its fingerprint.
//!
//! Order of checks (cheapest refusal first, nothing allocated before the
//! budget passes):
//! 1. header: magic, version, CRC, tags, declared sizes self-consistent;
//! 2. budget from the header alone: vectors, resident bytes, load units
//!    → `413` before any row allocation;
//! 3. each frame: contiguous index, section order, length vs. the layout,
//!    payload CRC (prefix included);
//! 4. footer: magic, frame count, rolling CRC over frame CRCs; no trailing
//!    bytes;
//! 5. content: norms finite and ≥ 0, padding bits clear, keys unique;
//! 6. rotation regenerated from `(kind, seed, dim)`, fingerprint compared.

use super::format::*;
use crate::budget::{self, Budget};
use crate::error::{BudgetResource, CorruptKind, QuantError, Result};
use crate::shard::{build_rotation, last_word_mask, QuantConfig, QuantShard};
use std::collections::BTreeMap;
use std::io::{ErrorKind, Read};

fn corrupt<T>(k: CorruptKind) -> Result<T> {
    Err(QuantError::corrupt(k))
}

/// Incremental v2 decoder: feed the header, then data frames, then the
/// footer. Holds exactly the arrays the finished shard will own.
pub struct Decoder {
    header: Header,
    budget: Budget,
    cap: usize,
    lens: [u64; 3],
    /// Bytes already received per section.
    got: [u64; 3],
    next_frame: u32,
    crcs: Crc32,
    keys: Vec<u64>,
    norms: Vec<f32>,
    packed: Vec<u64>,
}

impl Decoder {
    /// Validate the header frame and charge the load against `budget`.
    pub fn new(header_frame: &[u8], budget: Budget) -> Result<Self> {
        let header = Header::decode(header_frame)?;
        let dim = header.dim as usize;
        budget::check(BudgetResource::Vectors, budget.max_vectors, header.n)?;
        let resident = budget::resident_bytes(header.n, dim, header.rotation);
        budget::check(
            BudgetResource::ResidentBytes,
            budget.max_resident_bytes,
            resident,
        )?;
        let units = budget::load_units(header.payload_bytes, header.n, dim, header.rotation);
        budget::check(BudgetResource::LoadUnits, budget.max_load_units, units)?;
        // `section_lens` succeeded in `decode`, and the resident check bounds
        // `n`, so these capacities are small and exact.
        let lens = section_lens(header.n, dim).expect("validated in Header::decode");
        let n = header.n as usize;
        Ok(Decoder {
            cap: payload_cap(header.max_frame_bytes as usize),
            header,
            budget,
            lens,
            got: [0; 3],
            next_frame: 0,
            crcs: Crc32::new(),
            keys: Vec::with_capacity(n),
            norms: Vec::with_capacity(n),
            packed: Vec::with_capacity(n * budget::n_words(dim)),
        })
    }

    /// The validated header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Data frames still expected.
    pub fn frames_left(&self) -> u32 {
        self.header.frames - self.next_frame
    }

    /// The section and payload length the next frame must carry.
    fn expected_next(&self) -> Option<(usize, u64)> {
        (0..3)
            .find(|&s| self.got[s] < self.lens[s])
            .map(|s| (s, (self.lens[s] - self.got[s]).min(self.cap as u64)))
    }

    /// Validate a data frame prefix; returns the payload length it declares.
    pub fn check_prefix(&self, prefix: &[u8]) -> Result<usize> {
        if prefix.len() < FRAME_PREFIX_LEN {
            return corrupt(CorruptKind::Truncated);
        }
        let idx = le_u32(&prefix[0..4]);
        if idx != self.next_frame {
            return corrupt(CorruptKind::FrameSequence {
                expected: self.next_frame,
                got: idx,
            });
        }
        let Some((section, want_len)) = self.expected_next() else {
            return corrupt(CorruptKind::TrailingData);
        };
        if prefix[4] != section as u8 + 1 || prefix[5..8] != [0, 0, 0] {
            return corrupt(CorruptKind::Malformed("frame section"));
        }
        let len = u64::from(le_u32(&prefix[8..12]));
        if len != want_len {
            return corrupt(CorruptKind::Malformed("frame length"));
        }
        Ok(len as usize)
    }

    /// Consume one whole data frame (prefix + payload).
    pub fn push_frame(&mut self, frame: &[u8]) -> Result<()> {
        let len = self.check_prefix(frame)?;
        let payload = &frame[FRAME_PREFIX_LEN.min(frame.len())..];
        if payload.len() < len {
            return corrupt(CorruptKind::Truncated);
        }
        if payload.len() > len {
            return corrupt(CorruptKind::Malformed("frame length"));
        }
        let mut c = Crc32::new();
        c.update(&frame[0..12]);
        c.update(payload);
        let crc = c.finish();
        if crc != le_u32(&frame[12..16]) {
            return corrupt(CorruptKind::FrameChecksum(self.next_frame));
        }
        self.crcs.update(&crc.to_le_bytes());
        let section = (frame[4] - 1) as usize;
        match section {
            0 => self.keys.extend(payload.chunks_exact(8).map(le_u64)),
            1 => self
                .norms
                .extend(payload.chunks_exact(4).map(|b| f32::from_bits(le_u32(b)))),
            _ => self.packed.extend(payload.chunks_exact(8).map(le_u64)),
        }
        self.got[section] += len as u64;
        self.next_frame += 1;
        Ok(())
    }

    /// Consume the footer, validate content, regenerate + verify the
    /// rotation and return the shard.
    pub fn finish(self, footer: &[u8]) -> Result<QuantShard> {
        if self.expected_next().is_some() {
            return corrupt(CorruptKind::Truncated);
        }
        if footer.len() < FOOTER_LEN {
            return corrupt(CorruptKind::Truncated);
        }
        if &footer[0..8] != FOOTER_MAGIC {
            return corrupt(CorruptKind::Malformed("footer magic"));
        }
        if footer.len() > FOOTER_LEN {
            return corrupt(CorruptKind::TrailingData);
        }
        if le_u32(&footer[8..12]) != self.header.frames {
            return corrupt(CorruptKind::Malformed("footer frame count"));
        }
        if le_u32(&footer[12..16]) != self.crcs.finish() {
            return corrupt(CorruptKind::FooterChecksum);
        }
        let h = self.header;
        let dim = h.dim as usize;
        if self.norms.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return corrupt(CorruptKind::InvalidNorm);
        }
        let nw = budget::n_words(dim);
        let pad = !last_word_mask(dim);
        if pad != 0 && self.packed.chunks_exact(nw).any(|c| c[nw - 1] & pad != 0) {
            return corrupt(CorruptKind::PaddingBits);
        }
        let mut index = BTreeMap::new();
        for (pos, &k) in self.keys.iter().enumerate() {
            if index.insert(k, pos as u32).is_some() {
                return corrupt(CorruptKind::DuplicateKey);
            }
        }
        let rotation = build_rotation(dim, h.rotation, h.seed);
        if rotation_fingerprint(&rotation) != h.fingerprint {
            return corrupt(CorruptKind::RotationMismatch);
        }
        let cfg = QuantConfig {
            dim,
            metric: h.metric,
            rotation: h.rotation,
            seed: h.seed,
            budget: self.budget,
        };
        Ok(QuantShard::from_parts(
            cfg,
            rotation,
            self.keys,
            self.norms,
            self.packed,
            index,
        ))
    }
}

/// Load from separately stored frames (`[header, data…, footer]`, e.g. one
/// DO storage row per frame, in order).
pub fn load_frames<'a, I>(frames: I, budget: Budget) -> Result<QuantShard>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut it = frames.into_iter();
    let Some(h) = it.next() else {
        return corrupt(CorruptKind::Truncated);
    };
    let mut dec = Decoder::new(h, budget)?;
    for _ in 0..dec.frames_left() {
        let Some(f) = it.next() else {
            return corrupt(CorruptKind::Truncated);
        };
        dec.push_frame(f)?;
    }
    let Some(footer) = it.next() else {
        return corrupt(CorruptKind::Truncated);
    };
    if it.next().is_some() {
        return corrupt(CorruptKind::TrailingData);
    }
    dec.finish(footer)
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(|e| match e.kind() {
        ErrorKind::UnexpectedEof => QuantError::corrupt(CorruptKind::Truncated),
        _ => QuantError::Io(e.to_string()),
    })
}

/// Load from a byte stream holding the concatenated frames. Reads one
/// frame at a time into a single reusable buffer (peak extra memory: one
/// frame), and requires end-of-input after the footer.
pub fn load_from<R: Read>(r: &mut R, budget: Budget) -> Result<QuantShard> {
    let mut header = [0u8; HEADER_LEN];
    read_exact(r, &mut header[..8])?;
    if &header[..8] != MAGIC {
        return corrupt(CorruptKind::BadMagic);
    }
    read_exact(r, &mut header[8..])?;
    let mut dec = Decoder::new(&header, budget)?;
    let mut frame = Vec::with_capacity(dec.cap + FRAME_PREFIX_LEN);
    for _ in 0..dec.frames_left() {
        frame.resize(FRAME_PREFIX_LEN, 0);
        read_exact(r, &mut frame[..])?;
        // Length is validated against the layout before the buffer grows.
        let len = dec.check_prefix(&frame)?;
        frame.resize(FRAME_PREFIX_LEN + len, 0);
        read_exact(r, &mut frame[FRAME_PREFIX_LEN..])?;
        dec.push_frame(&frame)?;
    }
    let mut footer = [0u8; FOOTER_LEN];
    read_exact(r, &mut footer)?;
    let mut probe = [0u8; 1];
    match r.read(&mut probe) {
        Ok(0) => {}
        Ok(_) => return corrupt(CorruptKind::TrailingData),
        Err(e) => return Err(QuantError::Io(e.to_string())),
    }
    dec.finish(&footer)
}
