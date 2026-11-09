//! Regression tests for hostile / degenerate inputs reachable through the
//! public API (and therefore through the WASM bindings).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ruvector_rabitq::persist::{load_index, MAGIC, MAX_N, VERSION};
use ruvector_rabitq::{AnnIndex, RabitqAsymIndex, RabitqIndex, RabitqPlusIndex};

/// Records the largest single allocation request so a test can prove that a
/// tiny input never asks the allocator for gigabytes up front.
struct PeakAlloc;
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        PEAK.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;

fn dataset(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.37).sin()).collect())
        .collect()
}

/// A NaN with the sign bit set (`0xFFC0_0000`) — what x86 produces for
/// `inf - inf`, and trivially constructible from JS via a `DataView`.
fn neg_nan() -> f32 {
    f32::from_bits(0xFFC0_0000)
}

fn assert_nan_hits_trail(name: &str, hits: &[ruvector_rabitq::SearchResult], bad_id: usize) {
    assert!(!hits.is_empty(), "{name}: no hits");
    assert_ne!(hits[0].id, bad_id, "{name}: NaN row ranked first: {hits:?}");
    let first_nan = hits.iter().position(|h| h.score.is_nan());
    if let Some(p) = first_nan {
        assert!(
            hits[p..].iter().all(|h| h.score.is_nan()),
            "{name}: NaN score not at the back: {hits:?}"
        );
    }
}

#[test]
fn negative_nan_row_never_outranks_finite_rows() {
    let d = 16;
    let bad = 5;
    let mut data = dataset(50, d);
    data[bad] = vec![neg_nan(); d];
    let q = data[0].clone();

    let mut plus = RabitqPlusIndex::new(d, 42, 20);
    let mut sym = RabitqIndex::new(d, 42);
    let mut asym = RabitqAsymIndex::new(d, 42, 20);
    for (i, v) in data.iter().enumerate() {
        plus.add(i, v.clone()).unwrap();
        sym.add(i, v.clone()).unwrap();
        asym.add(i, v.clone()).unwrap();
    }
    let hits = plus.search(&q, 3).unwrap();
    assert_eq!(hits[0].id, 0, "exact match must rank first: {hits:?}");
    assert_nan_hits_trail("plus", &hits, bad);
    assert_nan_hits_trail("plus-k50", &plus.search(&q, 50).unwrap(), bad);
    assert_nan_hits_trail("sym", &sym.search(&q, 50).unwrap(), bad);
    assert_nan_hits_trail("asym", &asym.search(&q, 50).unwrap(), bad);
}

#[test]
fn persist_header_cannot_force_huge_preallocation() {
    // 32-byte header claiming MAX_N entries at dim=1 and no payload.
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // dim
    buf.extend_from_slice(&7u64.to_le_bytes()); // seed
    buf.extend_from_slice(&1u32.to_le_bytes()); // rerank_factor
    buf.extend_from_slice(&MAX_N.to_le_bytes()); // n
    assert!(load_index(&mut std::io::Cursor::new(&buf)).is_err());
    let peak = PEAK.load(Ordering::Relaxed);
    assert!(
        peak < 256 << 20,
        "truncated 32-byte file triggered a {peak}-byte allocation"
    );
}

#[test]
#[should_panic]
fn scan_rejects_short_packed_buffer() {
    // `scan` is a safe public fn; undersized buffers must panic, not read
    // out of bounds (the old guard was `debug_assert!`, gone in release).
    let mut out = vec![0u32; 1024];
    ruvector_rabitq::scan::scan(&[0u64; 2], 2, 1024, &[0u64; 2], !0, &mut out);
}

#[test]
#[should_panic]
fn scan_scalar_rejects_short_output_buffer() {
    let mut out = vec![0u32; 1];
    ruvector_rabitq::scan::scan_scalar(&[0u64; 2048], 2, 1024, &[0u64; 2], !0, &mut out);
}
