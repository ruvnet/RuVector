//! Memory accounting vs measured heap: a counting global allocator (its own
//! test binary, one test, so no parallel test pollutes the count).

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use common::*;
use ruvector_edge_index::memory::{
    flat_estimate_bytes, hnsw_bytes_per_vector, hnsw_estimate_bytes, shards_per_isolate,
};
use ruvector_edge_index::{
    HnswIndex, HnswParams, Metric, QuantFlatIndex, QuantParams, MAX_CHUNK_BYTES,
};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let now = LIVE.fetch_add(l.size() as isize, Ordering::SeqCst) + l.size() as isize;
        PEAK.fetch_max(now, Ordering::SeqCst);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Ordering::SeqCst);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let delta = new as isize - l.size() as isize;
        let now = LIVE.fetch_add(delta, Ordering::SeqCst) + delta;
        PEAK.fetch_max(now, Ordering::SeqCst);
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn live() -> usize {
    LIVE.load(Ordering::SeqCst) as usize
}

fn within(a: usize, b: usize, tol: f64) -> bool {
    (a as f64 - b as f64).abs() <= tol * b as f64
}

#[test]
fn accounting_matches_measured_heap_within_10_percent() {
    let (base, _) = data(N, DIM, 0);
    let quant = QuantParams::train(Metric::Cosine, DIM, &base[..1000 * DIM], 1).unwrap();
    let struct_bytes = std::mem::size_of::<HnswIndex>();

    let before = live();
    let mut h = HnswIndex::with_capacity(HnswParams::default(), quant.clone(), N).unwrap();
    for (i, v) in base.chunks_exact(DIM).enumerate() {
        h.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
    }
    h.shrink_to_fit();
    let measured = live() - before + struct_bytes;
    let reported = h.memory_bytes();
    let estimate = hnsw_estimate_bytes(N, DIM, h.params());
    eprintln!(
        "hnsw 10k x 384: measured {measured} B, memory_bytes {reported} B, estimate {estimate} B ({:.1} B/vector, estimate {:.1})",
        measured as f64 / N as f64,
        hnsw_bytes_per_vector(DIM, h.params())
    );
    assert!(
        within(reported, measured, 0.10),
        "reported {reported} vs measured {measured}"
    );
    assert!(
        within(estimate, measured, 0.10),
        "estimate {estimate} vs measured {measured}"
    );

    let before = live();
    let mut f = QuantFlatIndex::with_capacity(quant, N as u32, N).unwrap();
    for (i, v) in base.chunks_exact(DIM).enumerate() {
        f.upsert(i as u32, v).unwrap();
    }
    let measured = live() - before + std::mem::size_of::<QuantFlatIndex>();
    let reported = f.memory_bytes();
    let estimate = flat_estimate_bytes(N, DIM);
    eprintln!(
        "flat 10k x 384: measured {measured} B, memory_bytes {reported} B, estimate {estimate} B"
    );
    assert!(
        within(reported, measured, 0.10),
        "reported {reported} vs measured {measured}"
    );
    assert!(
        within(estimate, measured, 0.10),
        "estimate {estimate} vs measured {measured}"
    );

    // Isolate budget (56 MB): M2a shards at the M1 cap (7.8k × 384) and
    // M2b shards of this size.
    let flat_shards = shards_per_isolate(7_812, f.memory_bytes() as f64 / N as f64);
    let hnsw_shards = shards_per_isolate(N, h.memory_bytes() as f64 / N as f64);
    eprintln!(
        "per 56 MB isolate: {flat_shards} M2a shards of 7.8k, {hnsw_shards} M2b shards of 10k"
    );
    assert!(flat_shards >= 3 && hnsw_shards >= 2);

    // Flush and lazy load hold one chunk beyond the index (ADR §6.1
    // streaming encoder, `cursor.raw()` rows): measured peak, not intent.
    const SLACK: usize = 64 * 1024;
    let (before, _) = (live(), PEAK.store(live() as isize, Ordering::SeqCst));
    let digest = h
        .encode_into::<()>(1, MAX_CHUNK_BYTES, &mut |row| {
            drop(row); // one INSERT per row
            Ok(())
        })
        .unwrap();
    let enc_peak = PEAK.load(Ordering::SeqCst) as usize - before;
    let rows = h.to_chunks(1, MAX_CHUNK_BYTES).unwrap().chunks;
    let (before, _) = (live(), PEAK.store(live() as isize, Ordering::SeqCst));
    // A cursor: each row materialises only when pulled.
    let back =
        HnswIndex::from_chunk_iter(rows.iter().cloned(), Some(&digest.sha256), 1 << 30).unwrap();
    let dec_peak = PEAK.load(Ordering::SeqCst) as usize - before;
    eprintln!(
        "peak over baseline: encode {enc_peak} B, decode {dec_peak} B (index {} B, payload {} B)",
        back.memory_bytes(),
        digest.payload_len
    );
    assert!(
        enc_peak <= MAX_CHUNK_BYTES + SLACK,
        "encode peak {enc_peak}"
    );
    assert!(
        dec_peak <= back.memory_bytes() + MAX_CHUNK_BYTES + SLACK,
        "decode peak {dec_peak}"
    );
}
