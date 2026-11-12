//! Shared fixture and mutation model for the RVSD fuzz targets.
#![allow(dead_code)]
use arbitrary::Arbitrary;
use ed25519_dalek::SigningKey;
use rvf_types::{SegmentFlags, ROOT_MANIFEST_MAGIC_BYTES};
use rvf_wire::stream::*;
use rvf_wire::writer::{write_segment, write_segment_with_algo};
use std::sync::OnceLock;

pub struct Base {
    pub body: Vec<u8>,
    pub frames: Vec<Vec<u8>>,
    pub keys: Vec<[u8; 32]>,
}

pub fn signing_key() -> SigningKey {
    SigningKey::from_bytes(&[0x2A; 32])
}

pub fn base() -> &'static Base {
    static B: OnceLock<Base> = OnceLock::new();
    B.get_or_init(|| {
        let f = SegmentFlags::empty();
        let p = |s: u8, n: usize| (0..n).map(|i| (i as u8).wrapping_mul(s).wrapping_add(s)).collect::<Vec<_>>();
        let mut body = write_segment(0x07, &p(3, 100), f, 1);
        body.extend(write_segment_with_algo(0x10, &p(5, 5000), f, 2, 2));
        body.extend(write_segment(0x01, &p(7, 4200), f, 3));
        let mut root = vec![0u8; 4096];
        root[..4].copy_from_slice(&ROOT_MANIFEST_MAGIC_BYTES);
        body.extend(root);
        let b = StreamDescriptorBuilder::new(*b"rvsd-fuzz-file01", signing_key())
            .chunk_log2(12)
            .build(&body)
            .unwrap();
        let frames = b.frames_file_order(&body).collect();
        Base { body, frames, keys: vec![signing_key().verifying_key().to_bytes()] }
    })
}

#[derive(Debug, Arbitrary)]
pub enum Op {
    Flip(u16, u16, u8),
    Drop(u8),
    Dup(u8),
    Swap(u8, u8),
    Insert(u8, Vec<u8>),
    Cut(u8, u16),
}

pub fn apply(mut fr: Vec<Vec<u8>>, ops: &[Op]) -> Vec<Vec<u8>> {
    for o in ops {
        let n = fr.len().max(1);
        match o {
            Op::Flip(a, b, m) if !fr.is_empty() => {
                let f = &mut fr[*a as usize % n];
                let l = f.len().max(1);
                if !f.is_empty() {
                    f[*b as usize % l] ^= *m;
                }
            }
            Op::Drop(a) if !fr.is_empty() => {
                fr.remove(*a as usize % n);
            }
            Op::Dup(a) if !fr.is_empty() => fr.insert(*a as usize % n, fr[*a as usize % n].clone()),
            Op::Swap(a, b) if !fr.is_empty() => fr.swap(*a as usize % n, *b as usize % n),
            Op::Insert(a, b) => fr.insert(*a as usize % (fr.len() + 1), b.clone()),
            Op::Cut(a, b) if !fr.is_empty() => {
                let f = &mut fr[*a as usize % n];
                let l = f.len();
                f.truncate(*b as usize % (l + 1));
            }
            _ => {}
        }
    }
    fr
}

pub fn policy(b: &Base, budget: u32) -> StreamPolicy<'_> {
    let mut p = StreamPolicy::new(&b.keys);
    p.fault_budget = budget;
    p
}
