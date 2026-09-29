//! Structural validation past the integrity checks: payloads are mutated
//! and then *resealed* (CRCs and digest recomputed, as a buggy or hostile
//! writer of `index_chunks` would), so decoding must reject or accept them
//! on structure alone — and anything it accepts must be safe to search,
//! insert into, update and re-encode. A panic here would reinitialise the
//! whole Worker instance (ADR §6.1).

mod common;

use common::*;
use ruvector_edge_index::{
    reseal, DecodeError, HnswIndex, HnswParams, IndexChunk, LevelRng, Metric, QuantFlatIndex,
    QuantParams, SplitMix64, CHUNK_HEADER_LEN, MAX_CHUNK_BYTES,
};

const D: usize = 16;
const N: usize = 400;

fn index() -> (HnswIndex, Vec<f32>) {
    let (base, _) = data(N, D, 0);
    let q = QuantParams::train(Metric::Cosine, D, &base, 1).unwrap();
    let p = HnswParams {
        m: 4,
        m0: 8,
        ef_construction: 32,
        ..HnswParams::default()
    };
    let mut idx = HnswIndex::new(p, q).unwrap();
    for (i, v) in base.chunks_exact(D).enumerate() {
        idx.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
    }
    for i in (0..N as u32).step_by(13) {
        idx.delete(i);
    }
    (idx, base)
}

fn body(c: &mut IndexChunk) -> &mut [u8] {
    &mut c.bytes[CHUNK_HEADER_LEN..]
}

/// Whatever decodes must behave.
fn exercise(idx: &mut HnswIndex, base: &[f32]) {
    for q in base.chunks_exact(D).take(5) {
        let _ = idx.search(q, 10, 20).unwrap();
    }
    let next = idx.params().iid_base + idx.node_count();
    let _ = idx.insert(next, &base[..D], &mut op_rng(1));
    let _ = idx.update(idx.params().iid_base, &base[D..2 * D]);
    let _ = idx.to_chunks(1, 4096);
    let _ = idx.search(&base[..D], 10, 20).unwrap();
}

#[test]
fn upper_link_to_a_lower_level_node_is_rejected() {
    let (idx, base) = index();
    let mut c = idx.to_chunks(1, MAX_CHUNK_BYTES).unwrap().chunks;
    assert_eq!(c.len(), 1);
    let m = idx.params().m as usize;
    // Upper rows are the payload tail, in slot order: the last node with a
    // level owns the last `level · m` u32s; its layer-1 row comes first.
    let (last, lvl) = (0..N as u32)
        .rev()
        .find_map(|i| idx.level(i).filter(|&l| l > 0).map(|l| (i, l as usize)))
        .unwrap();
    let b = body(&mut c[0]);
    let at = b.len() - lvl * m * 4;
    let row: Vec<u32> = b[at..at + m * 4]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let low = (0..N as u32)
        .find(|&i| i != last && idx.level(i) == Some(0) && !row.contains(&i))
        .unwrap();
    b[at..at + 4].copy_from_slice(&low.to_le_bytes());
    reseal(&mut c);
    assert_eq!(
        HnswIndex::from_chunks(&c, None).unwrap_err(),
        DecodeError::Malformed("link row")
    );
    // A duplicate id in a row is rejected too.
    let mut c = idx.to_chunks(1, MAX_CHUNK_BYTES).unwrap().chunks;
    let b = body(&mut c[0]);
    if row[1] != u32::MAX {
        b[at..at + 4].copy_from_slice(&row[1].to_le_bytes());
        reseal(&mut c);
        assert_eq!(
            HnswIndex::from_chunks(&c, None).unwrap_err(),
            DecodeError::Malformed("link row")
        );
    }
    let _ = base;
}

#[test]
fn oversized_dims_and_counts_are_malformed_not_wrapped() {
    // Flat payload: metric u8, kind u8, dim u32 at body offset 2.
    let (base, _) = data(100, D, 0);
    let q = QuantParams::train(Metric::L2, D, &base, 1).unwrap();
    let mut f = QuantFlatIndex::new(q, 1 << 16).unwrap();
    for (i, v) in base.chunks_exact(D).enumerate() {
        f.upsert(i as u32, v).unwrap();
    }
    let mut c = f.to_chunks(1, MAX_CHUNK_BYTES).unwrap().chunks;
    body(&mut c[0])[2..6].copy_from_slice(&(1u32 << 16).to_le_bytes());
    reseal(&mut c);
    assert_eq!(
        QuantFlatIndex::from_chunks(&c, None).unwrap_err(),
        DecodeError::Malformed("quant dim")
    );
    // HNSW: max_slots (after quant params and m, m0, efc, max_level) = MAX.
    let (idx, _) = index();
    let mut c = idx.to_chunks(1, MAX_CHUNK_BYTES).unwrap().chunks;
    let at = 1 + 1 + 4 + 8 + 8 * D + 2 + 2 + 2 + 1;
    body(&mut c[0])[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut c);
    assert!(matches!(
        HnswIndex::from_chunks(&c, None),
        Err(DecodeError::Malformed(_))
    ));
}

#[test]
fn resealed_random_mutations_never_panic() {
    let (idx, base) = index();
    let good = idx.to_chunks(1, 2048).unwrap().chunks;
    let mut rng = SplitMix64::new(0xF022);
    let (mut accepted, mut rejected) = (0, 0);
    for round in 0..600 {
        let mut c = good.clone();
        let edits = 1 + (rng.next_u64() % 3) as usize;
        for _ in 0..edits {
            let p = (rng.next_u64() % c.len() as u64) as usize;
            let b = body(&mut c[p]);
            if b.is_empty() {
                continue;
            }
            let at = (rng.next_u64() % b.len() as u64) as usize;
            b[at] = match round % 3 {
                0 => b[at] ^ (1 << (rng.next_u64() % 8)),
                1 => 0xFF,
                _ => rng.next_u64() as u8,
            };
        }
        if round % 50 == 0 {
            let last = c.len() - 1;
            let len = c[last].bytes.len();
            c[last]
                .bytes
                .truncate(len - 1 - (rng.next_u64() % 8) as usize);
        }
        reseal(&mut c);
        match HnswIndex::from_chunks(&c, None) {
            Ok(mut back) => {
                accepted += 1;
                exercise(&mut back, &base);
            }
            Err(_) => rejected += 1,
        }
    }
    eprintln!("resealed mutations: {accepted} accepted (all exercised), {rejected} rejected");
    assert!(accepted > 0 && rejected > 0);
}
