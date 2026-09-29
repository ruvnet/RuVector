//! Resident-byte model vs the real heap (counting global allocator; its own
//! test binary so the allocator affects nothing else).
//!
//! Invariants, for incremental builds, cold load + upsert, and deletes:
//! - heap held by the shard ≤ `QuantShard::resident_bytes()` (capacity
//!   based, what the budget charges);
//! - peak heap during an upsert ≤ `max_resident_bytes` plus the request's
//!   own scratch (O(batch + dim), charged to the request, not the shard).

mod common;

use common::clustered;
use ruvector_edge_quant::persist::{load_frames, save_frames, MAX_FRAME_BYTES};
use ruvector_edge_quant::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Mutex;

struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

// SAFETY: forwards to `System`, only adding byte counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = CUR.fetch_add(l.size(), SeqCst) + l.size();
            PEAK.fetch_max(now, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CUR.fetch_sub(l.size(), SeqCst);
    }
}

#[global_allocator]
static A: Counting = Counting;

/// A shard plus the heap baseline it was created on.
struct Probe {
    base: usize,
    max_held_ratio: f64,
}

impl Probe {
    fn new() -> Self {
        Probe {
            base: CUR.load(SeqCst),
            max_held_ratio: 0.0,
        }
    }
    fn held(&self) -> usize {
        CUR.load(SeqCst) - self.base
    }
    /// Run one upsert and check its peak (call `check` once the caller's
    /// own request buffers are freed).
    fn upsert(&mut self, s: &mut QuantShard, rows: &[(u64, &[f32])]) -> Result<UpsertStats> {
        PEAK.store(CUR.load(SeqCst), SeqCst);
        let r = s.upsert(rows);
        let peak = PEAK.load(SeqCst) - self.base;
        let scratch = rows.len() * 8 + 16 * s.dim() + 4096;
        let cap = s.config().budget.max_resident_bytes as usize;
        assert!(peak <= cap + scratch, "peak {peak} > cap {cap} + {scratch}");
        r
    }
    fn check(&mut self, s: &QuantShard) {
        let (held, model) = (self.held(), s.resident_bytes() as usize);
        assert!(
            held <= model,
            "held {held} > model {model} at {} rows",
            s.len()
        );
        self.max_held_ratio = self.max_held_ratio.max(held as f64 / model as f64);
    }
}

fn pool(dim: usize) -> Vec<Vec<f32>> {
    clustered(512, 0, dim, 8, 0.5, 3).0
}

/// Fill in `batch`-row upserts until the resident budget refuses (413).
fn fill(dim: usize, batch: usize, stop_at: usize) -> usize {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let data = pool(dim);
    let mut p = Probe::new();
    let mut s = QuantShard::new(QuantConfig::new(dim, Metric::L2, 1)).unwrap();
    p.check(&s);
    let mut next = 0u64;
    loop {
        let rows: Vec<(u64, &[f32])> = (0..batch as u64)
            .map(|i| (next + i, data[((next + i) % 512) as usize].as_slice()))
            .collect();
        let r = p.upsert(&mut s, &rows);
        drop(rows);
        p.check(&s);
        match r {
            Ok(_) => next += batch as u64,
            Err(e) => {
                assert_eq!(e.status(), 413);
                break;
            }
        }
        if s.len() >= stop_at {
            break;
        }
    }
    println!(
        "dim {dim} batch {batch}: {} rows, resident {} B, held/model max {:.3}",
        s.len(),
        s.resident_bytes(),
        p.max_held_ratio
    );
    s.len()
}

#[test]
fn incremental_build_heap_is_bounded_by_the_model_dim_384_batch_500() {
    let n = fill(384, 500, usize::MAX);
    // Transient-inclusive growth admits ~100k rows at dim 384; the M4
    // design point (50k) fits with room to spare.
    assert!(n >= 90_000, "{n}");
}

#[test]
fn incremental_build_heap_is_bounded_by_the_model_single_rows() {
    fill(384, 1, 20_000);
}

#[test]
fn incremental_build_heap_is_bounded_by_the_model_dim_1536() {
    fill(1536, 64, usize::MAX);
}

#[test]
fn cold_load_then_one_row_and_deletes_stay_within_the_model() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dim = 384;
    let data = pool(dim);
    let n = 50_000u64;
    let frames = {
        let mut s = QuantShard::new(QuantConfig::new(dim, Metric::Cosine, 9)).unwrap();
        let rows: Vec<(u64, &[f32])> = (0..n)
            .map(|i| (i, data[(i % 512) as usize].as_slice()))
            .collect();
        for chunk in rows.chunks(500) {
            s.upsert(chunk).unwrap();
        }
        save_frames(&s, MAX_FRAME_BYTES).unwrap()
    };
    let mut p = Probe::new();
    let mut s = load_frames(frames.iter().map(|f| f.as_slice()), Budget::default()).unwrap();
    p.check(&s);
    assert_eq!(s.capacity(), n as usize, "load builds exact capacity");
    let loaded = s.resident_bytes();
    // First upsert after a load grows ×1.5, charged at its peak.
    p.upsert(&mut s, &[(n, data[0].as_slice())]).unwrap();
    p.check(&s);
    assert!(s.capacity() > n as usize);
    assert!(s.resident_bytes() <= Budget::default().max_resident_bytes);
    println!(
        "load 50k x 384: {loaded} B; after one-row upsert: {} B (capacity {})",
        s.resident_bytes(),
        s.capacity()
    );
    // Deleting 80% gives the memory back and the model follows it down.
    let doomed: Vec<u64> = (0..40_000).collect();
    assert_eq!(s.delete(&doomed), 40_000);
    drop(doomed);
    p.check(&s);
    assert!(s.capacity() <= 2 * s.len().max(64), "{} rows", s.len());
    assert!(s.resident_bytes() < loaded / 2);
}

#[test]
fn scattered_deletes_and_reinserts_stay_within_the_model() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dim = 128;
    let data = pool(dim);
    let mut p = Probe::new();
    let mut s = QuantShard::new(QuantConfig::new(dim, Metric::L2, 5)).unwrap();
    let mut next = 0u64;
    for round in 0..20u64 {
        let rows: Vec<(u64, &[f32])> = (0..2_000u64)
            .map(|i| (next + i * 7919 % 2_000, data[(i % 512) as usize].as_slice()))
            .collect();
        p.upsert(&mut s, &rows).unwrap();
        drop(rows);
        p.check(&s);
        // Delete every other live key (scattered, keeps the tree sparse).
        let doomed: Vec<u64> = s
            .keys()
            .iter()
            .copied()
            .filter(|k| k % 2 == round % 2)
            .collect();
        s.delete(&doomed);
        drop(doomed);
        p.check(&s);
        next += 2_000;
    }
    println!("churn: held/model max {:.3}", p.max_held_ratio);
}
