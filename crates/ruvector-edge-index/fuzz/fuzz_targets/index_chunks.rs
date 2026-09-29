//! Persisted index chunks (`index_chunks(epoch, part, bytes)` rows, ADR-351
//! §6.1): the store decodes them on every cold load with
//! `{QuantFlatIndex,HnswIndex}::from_chunk_iter`, and a panic there would
//! reinitialise the whole Worker instance.
//!
//! Input: `mode ‖ rest`. `mode & 3` picks how `rest` becomes rows:
//!
//! * 0 — raw rows: `rest` is split into rows by 2-byte little-endian length
//!   prefixes (row `part` = its position; `mode & 4` drops the epoch check
//!   by stamping row epoch 0 everywhere). Exercises the integrity layer.
//! * 1 — one resealed payload: `rest` is the body of a single last part
//!   whose header, lengths, digest and CRC are made valid, so the payload
//!   parser sees arbitrary bytes (what a buggy or hostile writer could
//!   store).
//! * 2, 3 — multi-part resealed payload: `rest` is split into bodies at
//!   0xFF bytes, each wrapped as consecutive parts and resealed.
//!
//! `mode & 8` selects HNSW, else the quantised flat index. Both the slice
//! decoder and the streaming decoder run (they must agree), with and
//! without the digest pin. Anything that decodes must be safe to search,
//! insert into, update, remove from and re-encode, and must re-decode from
//! its own encoding.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_edge_index::{
    reseal, HnswIndex, IndexChunk, QuantFlatIndex, SplitMix64, CHUNK_HEADER_LEN, MAX_CHUNK_BYTES,
};

const EPOCH: u64 = 7;
/// The store passes its shard memory cap; any bound works here.
const MAX_PAYLOAD: u64 = 1 << 24;

fn header(kind: u8, part: u32, total: u32, last: bool) -> Vec<u8> {
    let mut h = vec![0u8; CHUNK_HEADER_LEN];
    h[0..4].copy_from_slice(b"RVEI");
    h[4..6].copy_from_slice(&2u16.to_le_bytes());
    h[6] = kind;
    h[7] = u8::from(last);
    h[8..16].copy_from_slice(&EPOCH.to_le_bytes());
    h[16..20].copy_from_slice(&part.to_le_bytes());
    h[20..24].copy_from_slice(&total.to_le_bytes());
    h
}

fn sealed(kind: u8, bodies: &[&[u8]]) -> Vec<IndexChunk> {
    let total = bodies.len() as u32;
    let mut chunks: Vec<IndexChunk> = bodies
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut bytes = header(kind, i as u32, total, i as u32 + 1 == total);
            bytes.extend_from_slice(b);
            IndexChunk {
                epoch: EPOCH,
                part: i as u32,
                bytes,
            }
        })
        .collect();
    reseal(&mut chunks);
    chunks
}

fn raw_rows(rest: &[u8], stamp_epoch: bool) -> Vec<IndexChunk> {
    let mut rows = Vec::new();
    let mut r = rest;
    while r.len() >= 2 && rows.len() < 64 {
        let len = usize::from(u16::from_le_bytes([r[0], r[1]])).min(r.len() - 2);
        let bytes = r[2..2 + len].to_vec();
        r = &r[2 + len..];
        let epoch = if stamp_epoch && bytes.len() >= 16 {
            u64::from_le_bytes(bytes[8..16].try_into().unwrap())
        } else {
            0
        };
        rows.push(IndexChunk {
            epoch,
            part: rows.len() as u32,
            bytes,
        });
    }
    rows
}

fn digest(chunks: &[IndexChunk]) -> Option<[u8; 32]> {
    let last = chunks.last()?;
    last.bytes.get(36..68).map(|d| d.try_into().unwrap())
}

/// A valid query for the decoded index's dimension (non-zero for cosine).
fn query(dim: usize, salt: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| 1.0 + ((i as u32 ^ salt) % 7) as f32 * 0.25)
        .collect()
}

fn exercise_flat(mut idx: QuantFlatIndex) {
    let dim = idx.quant().dim();
    let q = query(dim, 1);
    if let Ok(hits) = idx.search(&q, 10) {
        assert!(hits.len() <= 10.min(idx.len()));
        for h in &hits {
            assert!(idx.contains(h.iid), "hit {} not in index", h.iid);
        }
    }
    let _ = idx.memory_bytes();
    let _ = idx.dead_ratio();
    if let Ok(enc) = idx.to_chunks(EPOCH + 1, MAX_CHUNK_BYTES) {
        let back = QuantFlatIndex::from_chunks(&enc.chunks, Some(&enc.sha256))
            .expect("an index re-decodes from its own encoding");
        assert_eq!(back.len(), idx.len());
    }
    let iid = idx.slots();
    let _ = idx.upsert(iid, &q);
    let _ = idx.upsert(iid.saturating_sub(1), &query(dim, 2));
    let _ = idx.remove(iid);
    let _ = idx.compact_ids(0);
    let _ = idx.search(&q, 3);
}

fn exercise_hnsw(mut idx: HnswIndex) {
    let dim = idx.quant().dim();
    let q = query(dim, 3);
    if let Ok(hits) = idx.search(&q, 10, 20) {
        assert!(hits.len() <= 10);
        for h in &hits {
            assert!(idx.contains(h.iid), "hit {} not in index", h.iid);
        }
    }
    let _ = idx.memory_bytes();
    let _ = idx.entry_point();
    if let Ok(enc) = idx.to_chunks(EPOCH + 1, MAX_CHUNK_BYTES) {
        let back = HnswIndex::from_chunks(&enc.chunks, Some(&enc.sha256))
            .expect("an index re-decodes from its own encoding");
        assert_eq!(back.len(), idx.len());
        assert_eq!(back.node_count(), idx.node_count());
    }
    let base = idx.params().iid_base;
    let next = base.saturating_add(idx.node_count());
    let _ = idx.insert(next, &q, &mut SplitMix64::new(5));
    let _ = idx.update(base, &query(dim, 4));
    let _ = idx.delete(base);
    // `compact(true)` requires a completed repair pass.
    let mut cursor = 0;
    for _ in 0..1024 {
        if cursor >= idx.node_count() {
            break;
        }
        cursor = idx.repair_links(cursor, 64);
    }
    let _ = idx.search(&q, 5, 16);
    if cursor >= idx.node_count() {
        let _ = idx.compact(true);
    }
    let _ = idx.search(&q, 5, 16);
}

fn decode(hnsw: bool, chunks: &[IndexChunk]) {
    let pin = digest(chunks);
    if hnsw {
        let a = HnswIndex::from_chunks(chunks, None);
        let b = HnswIndex::from_chunk_iter(chunks.to_vec(), pin.as_ref(), MAX_PAYLOAD);
        assert_eq!(
            a.is_ok(),
            b.is_ok(),
            "slice {:?} vs stream {:?}",
            a.as_ref().err(),
            b.as_ref().err()
        );
        if let Ok(idx) = b {
            exercise_hnsw(idx);
        }
    } else {
        let a = QuantFlatIndex::from_chunks(chunks, None);
        let b = QuantFlatIndex::from_chunk_iter(chunks.to_vec(), pin.as_ref(), MAX_PAYLOAD);
        assert_eq!(
            a.is_ok(),
            b.is_ok(),
            "slice {:?} vs stream {:?}",
            a.as_ref().err(),
            b.as_ref().err()
        );
        if let Ok(idx) = b {
            exercise_flat(idx);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    let hnsw = mode & 8 != 0;
    let kind = if hnsw { 2 } else { 1 };
    let chunks = match mode & 3 {
        0 => raw_rows(rest, mode & 4 == 0),
        1 => sealed(kind, &[rest]),
        _ => {
            let bodies: Vec<&[u8]> = rest.split(|&b| b == 0xFF).take(16).collect();
            sealed(kind, &bodies)
        }
    };
    decode(hnsw, &chunks);
});
