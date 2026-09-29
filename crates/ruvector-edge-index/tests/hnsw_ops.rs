//! Graph mutation paths the store relies on: tombstone-aware search,
//! in-place update, store-shaped iids (`iid_base = 1`), tombstone purge
//! with sliced repair, and bounded growth after a lazy load.

mod common;

use common::*;
use ruvector_edge_index::memory::hnsw_estimate_bytes;
use ruvector_edge_index::{
    distance, EncodeError, HnswIndex, HnswParams, IndexError, Metric, QuantFlatIndex, QuantParams,
    SliceFetch,
};

const D: usize = 64;
const CHUNK: usize = 256 * 1024;

fn params(base: u32) -> HnswParams {
    HnswParams {
        ef_construction: 64,
        iid_base: base,
        ..HnswParams::default()
    }
}

fn build(base: &[f32], iid_base: u32) -> HnswIndex {
    let q = QuantParams::train(Metric::L2, D, &base[..500 * D], 1).unwrap();
    let mut idx = HnswIndex::new(params(iid_base), q).unwrap();
    for (i, v) in base.chunks_exact(D).enumerate() {
        let iid = iid_base + i as u32;
        idx.insert(iid, v, &mut op_rng(u64::from(iid))).unwrap();
    }
    idx
}

/// Exact top-k over live iids (`iid_base = 0`).
fn exact_live(idx: &HnswIndex, base: &[f32], q: &[f32], k: usize) -> Vec<u32> {
    let mut all: Vec<(f32, u32)> = base
        .chunks_exact(D)
        .enumerate()
        .filter(|(i, _)| idx.contains(*i as u32))
        .map(|(i, v)| (distance(Metric::L2, q, v), i as u32))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(k).map(|x| x.1).collect()
}

#[test]
fn tombstones_never_take_result_slots() {
    let (base, queries) = data(2000, D, 20);
    let mut idx = build(&base, 0);
    for q in queries.chunks_exact(D) {
        for iid in exact_live(&idx, &base, q, 5) {
            assert!(idx.delete(iid));
        }
        let hits = idx.search(q, 10, 10).unwrap();
        assert_eq!(hits.len(), 10.min(idx.len()));
        assert!(hits.iter().all(|h| idx.contains(h.iid)));
    }
    // Nine in ten nodes gone: still k live hits from the survivors.
    for iid in 0..2000u32 {
        if iid % 10 != 0 {
            idx.delete(iid);
        }
    }
    let mut fetch = SliceFetch::new(&base, D);
    for q in queries.chunks_exact(D) {
        assert_eq!(idx.search(q, 10, 10).unwrap().len(), 10);
        let rr = idx.search_rerank(q, 10, 10, &mut fetch).unwrap();
        assert_eq!(rr.len(), 10);
        assert!(rr.iter().all(|h| h.iid % 10 == 0));
    }
}

#[test]
fn update_moves_nodes_in_place_and_revives_tombstones() {
    let (base, queries) = data(1500, D, 30);
    let mut idx = build(&base, 0);
    let target = &queries[..D];
    assert!(idx.search(target, 1, 32).unwrap()[0].iid != 5);
    idx.update(5, target).unwrap();
    assert_eq!(idx.search(target, 1, 32).unwrap()[0].iid, 5);

    // Delete then re-upsert the same iid (the store keeps an id's iid).
    assert!(idx.delete(7));
    assert_eq!(
        idx.insert(7, target, &mut op_rng(1)),
        Err(IndexError::Occupied(7))
    );
    idx.upsert(7, &queries[D..2 * D], &mut op_rng(1)).unwrap();
    assert!(idx.contains(7) && idx.len() == 1500);
    assert_eq!(idx.update(4000, target), Err(IndexError::NotFound(4000)));
    idx.upsert(1500, target, &mut op_rng(1500)).unwrap();
    assert_eq!(idx.len(), 1501);

    // Move a fifth of the graph; recall on the *new* data stays high and
    // replaying the same op sequence gives the same bytes.
    let moved = World::new(D).points(4, 300);
    let ops = |idx: &mut HnswIndex| {
        for (j, v) in moved.chunks_exact(D).enumerate() {
            idx.update(
                (j * 5) as u32,
                &v.iter().map(|x| x * 1.01).collect::<Vec<_>>(),
            )
            .unwrap();
        }
    };
    let mut a = build(&base, 0);
    let mut b = build(&base, 0);
    ops(&mut a);
    ops(&mut b);
    assert_eq!(
        a.to_chunks(1, CHUNK).unwrap().sha256,
        b.to_chunks(1, CHUNK).unwrap().sha256
    );
    let mut now = base.clone();
    for (j, v) in moved.chunks_exact(D).enumerate() {
        for (o, x) in now[j * 5 * D..(j * 5 + 1) * D].iter_mut().zip(v) {
            *o = x * 1.01;
        }
    }
    let mut fetch = SliceFetch::new(&now, D);
    let t = truth(Metric::L2, &now, &queries, D, 10);
    let r: f64 = queries
        .chunks_exact(D)
        .zip(&t)
        .map(|(q, t)| recall(t, &a.search_rerank(q, 10, 64, &mut fetch).unwrap()))
        .sum::<f64>()
        / t.len() as f64;
    eprintln!("recall@10 after moving 20% of nodes: {r:.4}");
    assert!(r >= 0.95, "{r}");
    assert!(HnswIndex::from_chunks(&a.to_chunks(1, CHUNK).unwrap().chunks, None).is_ok());
}

#[test]
fn store_shaped_iids_starting_at_one_encode_without_renumbering() {
    let (base, queries) = data(800, D, 10);
    let idx = build(&base, 1);
    assert_eq!(idx.node_count(), 800);
    assert!(idx
        .entry_point()
        .is_some_and(|(e, _)| (1..=800).contains(&e)));
    let enc = idx.to_chunks(1, CHUNK).unwrap();
    let back = HnswIndex::from_chunks(&enc.chunks, Some(&enc.sha256)).unwrap();
    assert_eq!(back.params().iid_base, 1);
    // Rows are fetched by store iid: row i of `base` is iid i + 1.
    let shifted: Vec<f32> = vec![0.0; D]
        .into_iter()
        .chain(base.iter().copied())
        .collect();
    let mut fetch = SliceFetch::new(&shifted, D);
    for q in queries.chunks_exact(D) {
        let hits = back.search_rerank(q, 10, 48, &mut fetch).unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| (1..=800).contains(&h.iid)));
        assert_eq!(hits, idx.search_rerank(q, 10, 48, &mut fetch).unwrap());
    }
    let mut idx = idx;
    assert_eq!(
        idx.insert(0, &base[..D], &mut op_rng(0)),
        Err(IndexError::IidBelowBase { iid: 0, base: 1 })
    );
    // A gap still fails loudly, reported in store iids.
    idx.insert(900, &base[..D], &mut op_rng(900)).unwrap();
    assert!(matches!(
        idx.to_chunks(1, CHUNK),
        Err(EncodeError::GappedIid {
            first_missing: 801,
            missing: 99
        })
    ));
    let map = idx.compact(false);
    assert_eq!((map[0], map[799], map[899]), (1, 800, 801));
    assert!(idx.to_chunks(1, CHUNK).is_ok());
}

#[test]
fn purge_reclaims_tombstones_with_sliced_repair() {
    let (base, queries) = data(3000, D, 40);
    let mut idx = build(&base, 0);
    for iid in (0..3000u32).filter(|i| i % 10 < 3) {
        idx.delete(iid);
    }
    assert!((idx.dead_ratio() - 0.3).abs() < 1e-9);
    let t: Vec<Vec<u32>> = queries
        .chunks_exact(D)
        .map(|q| exact_live(&idx, &base, q, 10))
        .collect();
    let mut cursor = 0;
    while cursor < idx.node_count() {
        cursor = idx.repair_links(cursor, 500); // one alarm slice each
    }
    let before = idx.memory_bytes();
    let map = idx.compact(true);
    assert_eq!((idx.node_count(), idx.len()), (2100, 2100));
    assert_eq!(idx.dead_ratio(), 0.0);
    assert!(map
        .iter()
        .enumerate()
        .all(|(i, &m)| (i % 10 < 3) == (m == u32::MAX)));
    assert_eq!(idx.memory_bytes(), before, "in place: no second copy");
    // Recall over the survivors, translated through the map.
    let mut dense = vec![0f32; 2100 * D];
    for (old, &new) in map.iter().enumerate() {
        if new != u32::MAX {
            dense[new as usize * D..(new as usize + 1) * D]
                .copy_from_slice(&base[old * D..(old + 1) * D]);
        }
    }
    let mut fetch = SliceFetch::new(&dense, D);
    let r: f64 = queries
        .chunks_exact(D)
        .zip(&t)
        .map(|(q, t)| {
            let want: Vec<u32> = t.iter().map(|&i| map[i as usize]).collect();
            recall(&want, &idx.search_rerank(q, 10, 64, &mut fetch).unwrap())
        })
        .sum::<f64>()
        / t.len() as f64;
    eprintln!("recall@10 after purging 30%: {r:.4}");
    assert!(r >= 0.95, "{r}");
    let enc = idx.to_chunks(2, CHUNK).unwrap();
    let mut back = HnswIndex::from_chunks(&enc.chunks, None).unwrap();
    back.insert(2100, &base[..D], &mut op_rng(7)).unwrap();
    // Purging everything leaves a valid empty index.
    for iid in 0..=2100 {
        back.delete(iid);
    }
    back.compact(true);
    assert!(back.is_empty() && back.entry_point().is_none());
    assert!(back.to_chunks(3, CHUNK).is_ok());
}

#[test]
fn lazy_load_then_replay_grows_by_bounded_steps() {
    let (base, _) = data(5000, D, 0);
    let extra = World::new(D).points(3, 200);
    let enc = build(&base, 0).to_chunks(1, CHUNK).unwrap();
    let est = |i: &HnswIndex| hnsw_estimate_bytes(i.node_count() as usize, D, i.params());
    let within = |a: usize, b: usize| (a as f64 - b as f64).abs() <= 0.10 * b as f64;

    let mut idx = HnswIndex::from_chunks(&enc.chunks, None).unwrap();
    assert!(
        within(idx.memory_bytes(), est(&idx)),
        "{} vs {}",
        idx.memory_bytes(),
        est(&idx)
    );
    let (before, preview) = (idx.memory_bytes(), idx.growth_bytes(5000));
    assert!(preview > 0);
    for (j, v) in extra.chunks_exact(D).enumerate() {
        let iid = 5000 + j as u32;
        if j == 0 {
            idx.insert(iid, v, &mut op_rng(u64::from(iid))).unwrap();
            assert!(
                idx.memory_bytes() - before <= preview,
                "preview is an upper bound"
            );
        } else {
            idx.insert(iid, v, &mut op_rng(u64::from(iid))).unwrap();
        }
    }
    eprintln!(
        "5k x {D} decode {before} B -> +200 inserts {} B (estimate {} B)",
        idx.memory_bytes(),
        est(&idx)
    );
    assert!(
        within(idx.memory_bytes(), est(&idx)),
        "{} vs {}",
        idx.memory_bytes(),
        est(&idx)
    );

    // Pre-sizing for the replay: no reallocation of the slot arenas at all.
    let mut idx = HnswIndex::from_chunks(&enc.chunks, None).unwrap();
    idx.reserve(200);
    let reserved = idx.memory_bytes();
    for (j, v) in extra.chunks_exact(D).enumerate() {
        let iid = 5000 + j as u32;
        idx.insert(iid, v, &mut op_rng(u64::from(iid))).unwrap();
    }
    assert!(
        idx.memory_bytes() <= reserved + 4 * 16 * 16 * 16,
        "only upper rows may grow"
    );
}

#[test]
fn flat_compaction_is_in_place_and_starts_at_the_store_base() {
    let (base, queries) = data(2000, D, 10);
    let q = QuantParams::train(Metric::L2, D, &base[..500 * D], 1).unwrap();
    let mut f = QuantFlatIndex::new(q, 1 << 20).unwrap();
    for (i, v) in base.chunks_exact(D).enumerate() {
        f.upsert(1 + i as u32, v).unwrap();
    }
    for iid in (1..=2000u32).filter(|i| i % 2 == 0) {
        f.remove(iid);
    }
    assert!(f.dead_ratio() > 0.49);
    let hits: Vec<_> = queries
        .chunks_exact(D)
        .map(|q| f.search(q, 10).unwrap())
        .collect();
    let map = f.compact_ids(1).unwrap();
    assert_eq!((f.slots(), f.len()), (1001, 1000));
    for (q, h) in queries.chunks_exact(D).zip(&hits) {
        let moved: Vec<u32> = h.iter().map(|h| map[h.iid as usize]).collect();
        let got: Vec<u32> = f.search(q, 10).unwrap().iter().map(|h| h.iid).collect();
        assert_eq!(moved, got);
    }
    assert!(f.compact_ids(5).is_err(), "live iid 1 is below base 5");
}

#[test]
fn slot_caps_that_would_wrap_a_32_bit_usize_are_refused() {
    let q = QuantParams::cosine_fixed(1536, 1).unwrap();
    let big = HnswParams {
        max_slots: 3_000_000, // × 1536 B > isize::MAX on wasm32
        ..HnswParams::default()
    };
    let h = HnswIndex::new(big, q.clone());
    let f = QuantFlatIndex::new(q.clone(), 3_000_000);
    if cfg!(target_pointer_width = "32") {
        assert!(matches!(h, Err(IndexError::InvalidParams(_))));
        assert!(matches!(f, Err(IndexError::InvalidParams(_))));
    } else {
        assert!(h.is_ok() && f.is_ok());
    }
    // The defaults fit every target at the largest dimension.
    assert!(HnswIndex::new(HnswParams::default(), q).is_ok());
}
