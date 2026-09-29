//! Chunked persistence: bit-identical round trips, typed rejection of every
//! corruption class, replay fallback with identical results, and the
//! gapped-iid encode refusal.

mod common;

use common::*;
use ruvector_edge_index::{
    DecodeError, EncodeError, HnswIndex, HnswParams, IndexChunk, IndexError, Metric,
    QuantFlatIndex, QuantParams, SliceFetch, MAX_CHUNK_BYTES,
};

const D: usize = 64;
const NP: usize = 3000;
const CHUNK: usize = 64 * 1024;

fn params() -> HnswParams {
    HnswParams {
        ef_construction: 64,
        ..HnswParams::default()
    }
}

/// "Replay": rebuild from the op log (insert order + per-op randomness).
fn replay(metric: Metric, base: &[f32]) -> HnswIndex {
    let q = QuantParams::train(metric, D, &base[..500 * D], 7).unwrap();
    let mut idx = HnswIndex::new(params(), q).unwrap();
    for (i, v) in base.chunks_exact(D).enumerate() {
        idx.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
    }
    for i in (0..NP as u32).step_by(97) {
        assert!(idx.delete(i));
    }
    idx
}

fn same_results(a: &HnswIndex, b: &HnswIndex, base: &[f32], queries: &[f32]) {
    let (mut fa, mut fb) = (SliceFetch::new(base, D), SliceFetch::new(base, D));
    for q in queries.chunks_exact(D) {
        let hits = a.search(q, 10, 48).unwrap();
        assert!(
            hits.iter().all(|h| a.contains(h.iid)),
            "tombstoned iid returned"
        );
        assert_eq!(hits, b.search(q, 10, 48).unwrap());
        assert_eq!(
            a.search_rerank(q, 10, 48, &mut fa).unwrap(),
            b.search_rerank(q, 10, 48, &mut fb).unwrap()
        );
    }
}

#[test]
fn hnsw_roundtrip_is_bit_identical() {
    let (base, queries) = data(NP, D, 50);
    for metric in METRICS {
        let idx = replay(metric, &base);
        let enc = idx.to_chunks(11, CHUNK).unwrap();
        assert!(
            enc.chunks.len() > 5,
            "want a multi-part set, got {}",
            enc.chunks.len()
        );
        assert!(enc
            .chunks
            .iter()
            .all(|c| c.bytes.len() <= CHUNK && c.epoch == 11));
        let back = HnswIndex::from_chunks(&enc.chunks, Some(&enc.sha256)).unwrap();
        assert_eq!(
            (back.len(), back.node_count(), back.entry_point()),
            (idx.len(), idx.node_count(), idx.entry_point())
        );
        same_results(&idx, &back, &base, &queries);
        let again = back.to_chunks(11, CHUNK).unwrap();
        assert_eq!(again.chunks, enc.chunks);
        // Replay is deterministic: same op log, same bytes.
        assert_eq!(
            replay(metric, &base).to_chunks(11, CHUNK).unwrap().sha256,
            enc.sha256
        );
    }
}

#[test]
fn flat_roundtrip_is_bit_identical() {
    let (base, queries) = data(NP, D, 20);
    let q = QuantParams::train(Metric::L2, D, &base[..500 * D], 3).unwrap();
    let mut idx = QuantFlatIndex::new(q, 1 << 20).unwrap();
    for (i, v) in base.chunks_exact(D).enumerate() {
        idx.upsert(2 * i as u32, v).unwrap(); // gaps are fine for flat
    }
    assert!(idx.remove(4));
    let enc = idx.to_chunks(2, 32 * 1024).unwrap();
    let back = QuantFlatIndex::from_chunks(&enc.chunks, Some(&enc.sha256)).unwrap();
    assert_eq!((back.len(), back.slots()), (NP - 1, 2 * NP as u32 - 1));
    for qv in queries.chunks_exact(D) {
        assert_eq!(idx.search(qv, 10).unwrap(), back.search(qv, 10).unwrap());
    }
    assert_eq!(back.to_chunks(2, 32 * 1024).unwrap().chunks, enc.chunks);
    assert_eq!(
        HnswIndex::from_chunks(&enc.chunks, None).unwrap_err(),
        DecodeError::WrongKind
    );
}

fn chunks() -> (Vec<IndexChunk>, [u8; 32]) {
    let (base, _) = data(NP, D, 1);
    let enc = replay(Metric::Cosine, &base).to_chunks(5, CHUNK).unwrap();
    (enc.chunks, enc.sha256)
}

fn err(c: &[IndexChunk]) -> DecodeError {
    HnswIndex::from_chunks(c, None).unwrap_err()
}

#[test]
fn every_corruption_class_is_a_typed_error() {
    let (good, sha) = chunks();
    assert!(good.len() >= 4);

    let mut c = good.clone();
    c[3].bytes[500] ^= 0x01; // body bit flip
    assert_eq!(err(&c), DecodeError::ChecksumMismatch { part: 3 });

    let mut c = good.clone();
    c[2].bytes[16] = 9; // header part field
    assert_eq!(err(&c), DecodeError::ChecksumMismatch { part: 2 });

    let mut c = good.clone();
    c[1].bytes[0] = b'X';
    assert_eq!(err(&c), DecodeError::BadMagic { part: 1 });

    let mut c = good.clone();
    let len = c[2].bytes.len();
    c[2].bytes.truncate(len - 7);
    assert_eq!(err(&c), DecodeError::Truncated { part: 2 });
    c[2].bytes.truncate(40);
    assert_eq!(err(&c), DecodeError::Truncated { part: 2 });

    let mut c = good.clone();
    c.swap(1, 2);
    assert_eq!(
        err(&c),
        DecodeError::PartOutOfOrder {
            expected: 1,
            got: 2
        }
    );

    let mut c = good.clone();
    c.remove(1);
    assert_eq!(
        err(&c),
        DecodeError::PartOutOfOrder {
            expected: 1,
            got: 2
        }
    );

    let mut c = good.clone();
    c.pop();
    let total = good.len() as u32;
    assert_eq!(
        err(&c),
        DecodeError::PartCount {
            expected: total,
            got: total - 1
        }
    );

    let mut c = good.clone();
    c[1].part = 1_000;
    assert_eq!(err(&c), DecodeError::RowMismatch { part: 1_000 });
    let mut c = good.clone();
    c[1].epoch = 4;
    assert_eq!(err(&c), DecodeError::RowMismatch { part: 1 });

    // A part from another encode of the same epoch.
    let (base, _) = data(NP, D, 1);
    let other = replay(Metric::L2, &base).to_chunks(5, CHUNK).unwrap();
    let mut c = good.clone();
    c[1] = other.chunks[1].clone();
    // Same epoch, sizes and part count: only the final digest (or the
    // parse) can tell, and it does.
    assert!(matches!(
        err(&c),
        DecodeError::DigestMismatch | DecodeError::Malformed(_) | DecodeError::Inconsistent { .. }
    ));

    assert_eq!(
        HnswIndex::from_chunks(&good, Some(&[7; 32])).unwrap_err(),
        DecodeError::DigestMismatch
    );
    assert_eq!(err(&[]), DecodeError::Empty);
    assert!(HnswIndex::from_chunks(&good, Some(&sha)).is_ok());

    // Exhaustive-ish: any single-byte flip anywhere is rejected.
    for part in 0..good.len() {
        for at in (0..good[part].bytes.len()).step_by(997) {
            let mut c = good.clone();
            c[part].bytes[at] = c[part].bytes[at].wrapping_add(1);
            assert!(
                HnswIndex::from_chunks(&c, None).is_err(),
                "flip {part}:{at} accepted"
            );
        }
    }
}

#[test]
fn corrupted_chunk_falls_back_to_replay_with_identical_results() {
    let (base, queries) = data(NP, D, 30);
    let live = replay(Metric::Dot, &base);
    let mut stored = live.to_chunks(9, CHUNK).unwrap().chunks;
    stored[2].bytes[1234] ^= 0x80;
    let loaded = match HnswIndex::from_chunks(&stored, None) {
        Ok(_) => panic!("corruption accepted"),
        Err(_) => replay(Metric::Dot, &base),
    };
    same_results(&live, &loaded, &base, &queries);
}

#[test]
fn gapped_iid_encode_fails_loudly_and_compaction_fixes_it() {
    let (base, queries) = data(400, D, 10);
    let q = QuantParams::train(Metric::L2, D, &base, 1).unwrap();
    let mut idx = HnswIndex::new(params(), q).unwrap();
    let mut iid = 0u32;
    let mut ids = Vec::new();
    for (i, v) in base.chunks_exact(D).enumerate() {
        iid += if i % 50 == 49 { 3 } else { 1 }; // holes from deleted rows
        idx.insert(iid, v, &mut op_rng(i as u64)).unwrap();
        ids.push(iid);
    }
    assert!(idx.present_count() < idx.node_count());
    match idx.to_chunks(1, CHUNK) {
        Err(EncodeError::GappedIid {
            first_missing,
            missing,
        }) => {
            assert_eq!(first_missing, 0);
            assert_eq!(missing, idx.node_count() - idx.present_count());
        }
        other => panic!("gapped encode accepted: {other:?}"),
    }
    let mut dense = idx.clone();
    let map = dense.compact(false);
    assert_eq!(dense.node_count(), 400);
    let enc = dense.to_chunks(1, CHUNK).unwrap();
    let back = HnswIndex::from_chunks(&enc.chunks, None).unwrap();
    for qv in queries.chunks_exact(D) {
        let a: Vec<(u32, f32)> = idx
            .search(qv, 10, 40)
            .unwrap()
            .iter()
            .map(|h| (map[h.iid as usize], h.distance))
            .collect();
        let b: Vec<(u32, f32)> = back
            .search(qv, 10, 40)
            .unwrap()
            .iter()
            .map(|h| (h.iid, h.distance))
            .collect();
        assert_eq!(a, b);
    }
    assert_eq!(
        idx.insert(ids[3], &base[..D], &mut op_rng(0)),
        Err(IndexError::Occupied(ids[3]))
    );
    assert!(matches!(
        idx.insert(params().max_slots, &base[..D], &mut op_rng(0)),
        Err(IndexError::CapacityExceeded { .. })
    ));
    assert!(matches!(
        idx.to_chunks(1, MAX_CHUNK_BYTES + 1),
        Err(EncodeError::GappedIid { .. })
    ));
    assert!(matches!(
        dense.to_chunks(1, MAX_CHUNK_BYTES + 1),
        Err(EncodeError::BadChunkSize(_))
    ));
    // Request-supplied k / ef never drive an unbounded allocation.
    assert_eq!(
        back.search(&queries[..D], usize::MAX, usize::MAX)
            .unwrap()
            .len(),
        400
    );
}

#[test]
fn streaming_encode_and_decode_hold_one_chunk() {
    let (base, queries) = data(NP, D, 10);
    let idx = replay(Metric::L2, &base);
    let mut rows: Vec<IndexChunk> = Vec::new();
    let mut max_row = 0;
    let digest = idx
        .encode_into::<()>(3, CHUNK, &mut |c| {
            max_row = max_row.max(c.bytes.len());
            rows.push(c); // stands in for one INSERT per row
            Ok(())
        })
        .unwrap();
    assert!(max_row <= CHUNK && digest.parts as usize == rows.len());
    let collected = idx.to_chunks(3, CHUNK).unwrap();
    assert_eq!(collected.chunks, rows);
    assert_eq!(collected.sha256, digest.sha256);
    // Pull rows one at a time (a cursor), bounded by the caller's cap.
    let back = HnswIndex::from_chunk_iter(rows.clone(), Some(&digest.sha256), 64 << 20).unwrap();
    same_results(&idx, &back, &base, &queries);
    assert!(matches!(
        HnswIndex::from_chunk_iter(rows.clone(), None, 1024),
        Err(DecodeError::TooLarge { .. })
    ));
    // A sink failure stops the flush with the sink's own error.
    let r = idx.encode_into(3, CHUNK, &mut |c| {
        if c.part == 2 {
            Err("disk full")
        } else {
            Ok(())
        }
    });
    assert_eq!(
        r.unwrap_err(),
        ruvector_edge_index::EmitError::Sink("disk full")
    );
}

/// Inputs from integer ops only (no `ln`/`cos`, whose libm results differ
/// between native and wasm32), so every platform sees identical floats.
fn int_data(n: usize, d: usize) -> Vec<f32> {
    let mut r = ruvector_edge_index::SplitMix64::new(0xD47A);
    (0..n * d)
        .map(|_| {
            use ruvector_edge_index::LevelRng;
            (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// Replays must be bit-identical on every platform (ADR §6.1: a Worker
/// replays on wasm32, rebuilds may run natively). Golden digests of a
/// build + deletes + updates; this test runs on x86-64 and wasm32.
#[test]
fn replay_bytes_are_identical_across_platforms() {
    const GOLDEN: [&str; 3] = [
        "265e2f6633195929b6889548768e5d91d69e7657aff8ee3caa42d64eea5acb46",
        "8dc5d7a1f711e261ce12c3608056669998a64aff7d834a2948ee503a2b6e4846",
        "f87d81e958849cdd44e99d5d75e6c2a7fcc0580eef5c630bcc5ae976e5d1ce1b",
    ];
    let (d, n) = (32, 1500);
    let base = int_data(n, d);
    let mut got = Vec::new();
    for metric in METRICS {
        let q = QuantParams::train(metric, d, &base[..300 * d], 1).unwrap();
        let mut idx = HnswIndex::new(params(), q).unwrap();
        for (i, v) in base.chunks_exact(d).enumerate() {
            idx.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
        }
        for i in (0..n as u32).step_by(11) {
            idx.delete(i);
        }
        for i in (5..n).step_by(17) {
            idx.update(i as u32, &base[(i - 5) * d..(i - 4) * d])
                .unwrap();
        }
        got.push(idx.to_chunks(1, CHUNK).unwrap().sha256_hex());
    }
    eprintln!("replay digests: {got:?}");
    assert_eq!(got, GOLDEN);
}
