//! M4 acceptance at the design point: 50k × 384 clustered vectors.
//!
//! Heavy, so `#[ignore]`d; run in release:
//! `cargo test -p ruvector-edge-quant --release --test scale -- --ignored --nocapture`
//!
//! Reports recall@10 vs exact (with and without rerank), cold-load wall
//! time (total and rotation-only), peak heap during load (counting global
//! allocator), resident bytes, snapshot size, and native ns per work unit.

mod common;

use common::*;
use ruvector_edge_quant::persist::{load_frames, save_frames, MAX_FRAME_BYTES};
use ruvector_edge_quant::*;
use ruvector_edge_store::shard::SHARD_RESIDENT_CAP_BYTES;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to `System`, only adding relaxed byte counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(c, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CUR.fetch_sub(l.size(), Relaxed);
    }
}

#[global_allocator]
static A: Counting = Counting;

const N: usize = 50_000;
const DIM: usize = 384;
const Q: usize = 200;
const K: usize = 10;
const Q100: usize = 50;

fn run(metric: Metric, kind: RandomRotationKind) {
    let (data, qs) = clustered(N, Q, DIM, 100, 0.5, 2026);
    let truth: Vec<Vec<u64>> = qs.iter().map(|q| exact_topk(&data, q, K, metric)).collect();

    let mut cfg = QuantConfig::new(DIM, metric, 0x5EED);
    cfg.rotation = kind;
    let t = Instant::now();
    let mut s = QuantShard::new(cfg).unwrap();
    for chunk in rows(&data).chunks(500) {
        s.upsert(chunk).unwrap();
    }
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    let resident = s.resident_bytes();
    assert!(resident <= SHARD_RESIDENT_CAP_BYTES, "resident {resident}");

    println!("\n== {metric:?} / {kind:?}: n={N} dim={DIM} queries={Q} k={K}");
    println!("build (encode 50k, 100 batches of 500): {build_ms:.1} ms");
    println!(
        "resident bytes (model): {resident} ({:.2} MB of {} MB cap)",
        resident as f64 / 1e6,
        SHARD_RESIDENT_CAP_BYTES / 1_000_000
    );

    let mut src = MemRows::from_data(&data);
    let mut at_default = 0.0;
    for factor in [0u32, 5, 10, 20, 50, 100] {
        let opts = QueryOptions {
            top_k: K as u32,
            rerank_factor: factor,
        };
        let t = Instant::now();
        let mut got = Vec::with_capacity(Q);
        let mut units = 0;
        for q in &qs {
            let out = if factor == 0 {
                s.query(q, &opts, None).unwrap()
            } else {
                s.query(q, &opts, Some(&mut src)).unwrap()
            };
            units = out.stats.units;
            got.push(out.hits.iter().map(|h| h.key).collect::<Vec<_>>());
        }
        let per_q_us = t.elapsed().as_secs_f64() * 1e6 / Q as f64;
        let r = recall(&truth, &got);
        if factor == QueryOptions::default().rerank_factor {
            at_default = r;
        }
        println!(
            "recall@{K} rerank_factor={factor:>3} (cands {:>4}): {r:.4}  {per_q_us:>8.1} us/query  units/query {units}  ns/unit {:.2}",
            if factor == 0 { K as u32 } else { K as u32 * factor },
            per_q_us * 1e3 / units as f64,
        );
    }
    // The gate is the *default* options at one fixed factor (ADR-351
    // vector milestones: recall@10 ≥ 0.95), not the best factor in hindsight.
    assert!(
        at_default >= 0.95,
        "recall@{K} at QueryOptions::default() {at_default:.4} < 0.95"
    );

    // top_k = 100 (ADR-351 §7 maximum): the default factor asks for 5,000
    // candidates, which the default budget caps at `max_rerank_candidates`.
    // The cap is reported in QueryStats, and recall@100 must still hold.
    let truth100: Vec<Vec<u64>> = qs
        .iter()
        .take(Q100)
        .map(|q| exact_topk(&data, q, 100, metric))
        .collect();
    for cap in [200u32, 500, Budget::default().max_rerank_candidates] {
        s.set_budget(Budget {
            max_rerank_candidates: cap,
            ..Budget::default()
        });
        let opts = QueryOptions {
            top_k: 100,
            ..QueryOptions::default()
        };
        let mut got = Vec::with_capacity(Q100);
        let mut last = QueryStats::default();
        for q in qs.iter().take(Q100) {
            let out = s.query(q, &opts, Some(&mut src)).unwrap();
            last = out.stats;
            got.push(out.hits.iter().map(|h| h.key).collect::<Vec<_>>());
        }
        let r = recall(&truth100, &got);
        println!(
            "recall@100 default factor, cap {cap:>4}: {r:.4}  (requested {} candidates, reranked {}, clamped {})",
            last.requested_candidates, last.reranked, last.clamped
        );
        assert!(last.clamped);
        if cap == Budget::default().max_rerank_candidates {
            assert!(r >= 0.95, "recall@100 at the default cap {r:.4} < 0.95");
        }
    }
    s.set_budget(Budget::default());

    // Persist v2 round trip + cold load.
    let t = Instant::now();
    let frames = save_frames(&s, MAX_FRAME_BYTES).unwrap();
    let save_ms = t.elapsed().as_secs_f64() * 1e3;
    let bytes: usize = frames.iter().map(|f| f.len()).sum();
    assert!(frames.iter().all(|f| f.len() <= MAX_FRAME_BYTES));

    let base = CUR.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t = Instant::now();
    let loaded = load_frames(frames.iter().map(|f| f.as_slice()), Budget::default()).unwrap();
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let peak = PEAK.load(Relaxed) - base;
    let t = Instant::now();
    let rot = match kind {
        RandomRotationKind::HaarDense => ruvector_rabitq::RandomRotation::random(DIM, 0x5EED),
        RandomRotationKind::HadamardSigned => {
            ruvector_rabitq::RandomRotation::hadamard(DIM, 0x5EED)
        }
    };
    let rot_ms = t.elapsed().as_secs_f64() * 1e3;
    drop(rot);
    let lu = budget::load_units(bytes as u64, N as u64, DIM, kind);
    println!(
        "persist v2: {} frames, {bytes} bytes ({:.2} MB), save {save_ms:.1} ms",
        frames.len(),
        bytes as f64 / 1e6
    );
    println!(
        "cold load: {load_ms:.1} ms total, rotation regen {rot_ms:.1} ms, codes/validate {:.1} ms; peak heap during load {:.2} MB; load units {lu} ({:.2} ns/unit)",
        load_ms - rot_ms,
        peak as f64 / 1e6,
        load_ms * 1e6 / lu as f64
    );
    assert!(peak < 128_000_000, "isolate is 128 MB");

    // Identical answers after the reload.
    let opts = QueryOptions {
        top_k: K as u32,
        rerank_factor: 20,
    };
    for q in qs.iter().take(50) {
        let a = s.query(q, &opts, Some(&mut src)).unwrap().hits;
        let b = loaded.query(q, &opts, Some(&mut src)).unwrap().hits;
        assert_eq!(a, b);
    }

    // Baseline for comparison: rbpx0001 (stores f32, re-encodes on load).
    let items: Vec<(usize, Vec<f32>)> = data.iter().cloned().enumerate().collect();
    let plus =
        ruvector_rabitq::RabitqPlusIndex::from_vectors_parallel(DIM, 0x5EED, 20, items.clone())
            .unwrap();
    let mut v1 = Vec::new();
    ruvector_rabitq::persist::save_index(&plus, 0x5EED, &items, &mut v1).unwrap();
    drop(items);
    drop(plus);
    let base = CUR.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t = Instant::now();
    let back = ruvector_rabitq::persist::load_index(&mut v1.as_slice()).unwrap();
    let v1_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "baseline rbpx0001: {:.2} MB, load (re-encode from f32) {v1_ms:.1} ms, peak heap {:.2} MB",
        v1.len() as f64 / 1e6,
        (PEAK.load(Relaxed) - base) as f64 / 1e6
    );
    drop(back);
}

#[test]
#[ignore = "50k x 384; run with --release -- --ignored --nocapture"]
fn m4_design_point_cosine_haar() {
    run(Metric::Cosine, RandomRotationKind::HaarDense);
}

#[test]
#[ignore = "50k x 384; run with --release -- --ignored --nocapture"]
fn m4_design_point_l2_haar() {
    run(Metric::L2, RandomRotationKind::HaarDense);
}

#[test]
#[ignore = "50k x 384; run with --release -- --ignored --nocapture"]
fn m4_design_point_cosine_hadamard() {
    run(Metric::Cosine, RandomRotationKind::HadamardSigned);
}

#[test]
#[ignore = "50k x 384; run with --release -- --ignored --nocapture"]
fn m4_design_point_l2_hadamard() {
    run(Metric::L2, RandomRotationKind::HadamardSigned);
}

#[test]
#[ignore = "50k x 384; run with --release -- --ignored --nocapture"]
fn m4_design_point_dot_hadamard() {
    run(Metric::Dot, RandomRotationKind::HadamardSigned);
}
