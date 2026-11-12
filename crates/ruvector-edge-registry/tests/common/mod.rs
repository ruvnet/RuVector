//! Shared test helpers: RVF builders (rvf-wire writer and the runtime's packed
//! legacy layout) and fixture loading.
#![allow(dead_code)]

use rvf_types::{SegmentFlags, SegmentType};

/// Path to the checked-in rvf CLI example outputs.
pub fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/rvf/output")
}

pub fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture_dir().join(name)).expect("fixture present")
}

/// VEC_SEG payload in the runtime flat layout.
pub fn vec_payload(dim: u16, rows: &[(u64, Vec<f32>)]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&dim.to_le_bytes());
    p.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for (id, v) in rows {
        p.extend_from_slice(&id.to_le_bytes());
        v.iter().for_each(|x| p.extend_from_slice(&x.to_le_bytes()));
    }
    p
}

/// MANIFEST_SEG payload in the runtime layout.
pub fn manifest_payload(
    dim: u16,
    total: u64,
    metric: u8,
    dir: &[(u64, u64, u64, u8)],
    deleted: &[u64],
) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&1u32.to_le_bytes());
    p.extend_from_slice(&dim.to_le_bytes());
    p.extend_from_slice(&total.to_le_bytes());
    p.extend_from_slice(&(dir.len() as u32).to_le_bytes());
    p.extend_from_slice(&[0, metric, 0, 0]);
    for (id, off, len, t) in dir {
        p.extend_from_slice(&id.to_le_bytes());
        p.extend_from_slice(&off.to_le_bytes());
        p.extend_from_slice(&len.to_le_bytes());
        p.push(*t);
    }
    p.extend_from_slice(&(deleted.len() as u32).to_le_bytes());
    deleted
        .iter()
        .for_each(|d| p.extend_from_slice(&d.to_le_bytes()));
    p
}

/// A small store written with the rvf-wire writer (XXH3-128, 64-byte
/// aligned), optionally with a trailing Level-0 root page.
pub fn wire_store(dim: u16, rows: &[(u64, Vec<f32>)], deleted: &[u64], root: bool) -> Vec<u8> {
    let vec_p = vec_payload(dim, rows);
    let mut out = rvf_wire::write_segment(SegmentType::Vec as u8, &vec_p, SegmentFlags::empty(), 1);
    let man = manifest_payload(
        dim,
        rows.len() as u64,
        2,
        &[(1, 0, vec_p.len() as u64, SegmentType::Vec as u8)],
        deleted,
    );
    out.extend(rvf_wire::write_segment(
        SegmentType::Manifest as u8,
        &man,
        SegmentFlags::empty(),
        2,
    ));
    if root {
        let r = rvf_wire::manifest_codec::Level0Root {
            dimension: dim,
            ..Default::default()
        };
        out.extend_from_slice(&rvf_wire::manifest_codec::write_root_manifest(&r));
    }
    out
}

/// Random-ish deterministic rows.
pub fn rows(n: u64, dim: u16, seed: u64) -> Vec<(u64, Vec<f32>)> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|i| {
            let v = (0..dim)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
                })
                .collect();
            (i + 1, v)
        })
        .collect()
}

/// Runtime legacy content hash (IEEE CRC32 in four rotated lanes).
pub fn legacy_hash(data: &[u8]) -> [u8; 16] {
    let c = crc32fast::hash(data);
    let mut h = [0u8; 16];
    for i in 0..4 {
        h[i * 4..i * 4 + 4].copy_from_slice(&c.rotate_left(i as u32 * 8).to_le_bytes());
    }
    h
}

/// A hand-built segment header, every field overridable.
#[derive(Clone, Debug)]
pub struct Seg {
    pub magic: u32,
    pub version: u8,
    pub seg_type: u8,
    pub flags: u16,
    pub id: u64,
    pub payload_len: Option<u64>,
    pub algo: u8,
    pub compression: u8,
    pub reserved0: u16,
    pub reserved1: u32,
    pub hash: Option<[u8; 16]>,
    pub uncompressed: u32,
    pub pad: u32,
}

impl Seg {
    pub fn new(seg_type: SegmentType, id: u64) -> Self {
        Seg {
            magic: rvf_types::SEGMENT_MAGIC,
            version: 1,
            seg_type: seg_type as u8,
            flags: 0,
            id,
            payload_len: None,
            algo: 0,
            compression: 0,
            reserved0: 0,
            reserved1: 0,
            hash: None,
            uncompressed: 0,
            pad: 0,
        }
    }

    /// Header + payload + `pad` zero bytes (`footer` inserted after payload).
    pub fn build(&self, payload: &[u8], footer: &[u8]) -> Vec<u8> {
        let hash = self.hash.unwrap_or_else(|| match self.algo {
            0 => legacy_hash(payload),
            a => rvf_wire::hash::compute_content_hash(a, payload),
        });
        let mut b = Vec::new();
        b.extend_from_slice(&self.magic.to_le_bytes());
        b.push(self.version);
        b.push(self.seg_type);
        b.extend_from_slice(&self.flags.to_le_bytes());
        b.extend_from_slice(&self.id.to_le_bytes());
        b.extend_from_slice(
            &self
                .payload_len
                .unwrap_or(payload.len() as u64)
                .to_le_bytes(),
        );
        b.extend_from_slice(&0u64.to_le_bytes());
        b.push(self.algo);
        b.push(self.compression);
        b.extend_from_slice(&self.reserved0.to_le_bytes());
        b.extend_from_slice(&self.reserved1.to_le_bytes());
        b.extend_from_slice(&hash);
        b.extend_from_slice(&self.uncompressed.to_le_bytes());
        b.extend_from_slice(&self.pad.to_le_bytes());
        assert_eq!(b.len(), 64);
        b.extend_from_slice(payload);
        b.extend_from_slice(footer);
        b.resize(b.len() + self.pad as usize, 0);
        b
    }
}

/// A runtime-shaped (packed, legacy hash) store: VEC_SEG then MANIFEST_SEG.
pub fn packed_store(dim: u16, rows: &[(u64, Vec<f32>)], deleted: &[u64], metric: u8) -> Vec<u8> {
    let vp = vec_payload(dim, rows);
    let mut out = Seg::new(SegmentType::Vec, 1).build(&vp, &[]);
    let man = manifest_payload(
        dim,
        rows.len() as u64,
        metric,
        &[(1, 0, vp.len() as u64, 1)],
        deleted,
    );
    out.extend(Seg::new(SegmentType::Manifest, 2).build(&man, &[]));
    out
}

/// A well-formed signature footer for a `sig_len`-byte signature.
pub fn footer(sig_len: u16) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&1u16.to_le_bytes());
    f.extend_from_slice(&sig_len.to_le_bytes());
    f.resize(4 + sig_len as usize, 0xA5);
    f.extend_from_slice(&(8 + sig_len as u32).to_le_bytes());
    f
}

mod harness;
#[allow(unused_imports)]
pub use harness::*;
