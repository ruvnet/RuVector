//! Kernel microbenchmarks (plan M2). `#[ignore]`d so they never run in CI;
//! they live in-crate to reach the `pub(crate)` per-triple baseline.
//!
//! ```text
//! CARGO_TARGET_DIR=... cargo test --release -p ruvector-kge --features parallel \
//!     --lib kernel::bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! "Per positive" = two query rows (tail + head, or tail + reciprocal tail),
//! matching `one_vs_all_step`, which does both sides of one triple. Timed
//! region: query construction, all three GEMMs, softmax, and (HolE) the FFT
//! pull-backs — everything but the optimizer update.

use super::complex::{complex_one_n_step, ComplexWorkspace};
use super::hole::HolEKernel;
use super::{OneNQuery, Reduction};
use crate::{HolE, Side, Tables, Triple};
use std::time::Instant;

fn rows(n: usize, nr: usize, positives: usize) -> Vec<OneNQuery> {
    let mut s = 0x5EED_u64;
    let mut next = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as u32
    };
    let mut out = Vec::with_capacity(2 * positives);
    for _ in 0..positives {
        let (h, r, t) = (next(n), next(nr), next(n));
        out.push(OneNQuery {
            anchor: h,
            relation: r,
            side: Side::Tail,
            target: t,
        });
        out.push(OneNQuery {
            anchor: t,
            relation: r,
            side: Side::Head,
            target: h,
        });
    }
    out
}

/// Median of `iters` timed runs after one warm-up, in ms.
fn time_ms<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f();
    let mut v: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn with_threads<R: Send>(t: usize, f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
            .install(f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        assert_eq!(t, 1, "multi-thread timing needs --features parallel");
        f()
    }
}

fn thread_counts() -> Vec<usize> {
    if cfg!(feature = "parallel") {
        vec![1, 16]
    } else {
        vec![1]
    }
}

/// Baseline: the per-triple `one_vs_all_step`, HolE d=256, |E|=2000.
#[test]
#[ignore]
fn bench_baseline_one_vs_all_step_hole_d256_e2000() {
    let (n, nr, d) = (2000usize, 20usize, 256usize);
    let tables = Tables::new(n, nr, d, 1);
    let hole = HolE::new(d).unwrap();
    let trips: Vec<Triple> = rows(n, nr, 20)
        .chunks(2)
        .map(|p| Triple::new(p[0].anchor, p[0].relation, p[0].target))
        .collect();
    let ms = time_ms(3, || {
        let mut g = crate::train::optim::Grads::new(d);
        for &t in &trips {
            crate::train::loss::one_vs_all_step(&tables, &hole, t, &mut g).unwrap();
        }
    });
    println!(
        "BENCH baseline one_vs_all_step hole d=256 |E|=2000: {:.3} ms/pos ({} positives, 1 thread)",
        ms / trips.len() as f64,
        trips.len()
    );
}

/// Like-for-like: HolE d=256, |E|=2000 through the kernel, 500 positives
/// (1000 rows) per batch.
#[test]
#[ignore]
fn bench_kernel_hole_d256_e2000() {
    let (n, nr, d, pos) = (2000usize, 20usize, 256usize, 500usize);
    let tables = Tables::new(n, nr, d, 1);
    let hole = HolE::new(d).unwrap();
    let qs = rows(n, nr, pos);
    for t in thread_counts() {
        let mut k = HolEKernel::new(d).unwrap();
        let (mut ge, mut gr) = (vec![0.0; n * d], vec![0.0; nr * d]);
        let ms = with_threads(t, || {
            time_ms(5, || {
                k.step(&hole, &tables, &qs, Reduction::Mean, &mut ge, &mut gr)
                    .unwrap();
            })
        });
        println!(
            "BENCH kernel hole d=256 |E|=2000 rows={} threads={t}: {:.2} ms/batch, {:.4} ms/pos",
            qs.len(),
            ms,
            ms / pos as f64
        );
    }
}

/// FB15k-237 scale: |E|=14,541, 474 relations (237 + reciprocals), batch
/// 1000 rows, complex_rank k=500 (1000 reals; the lane gate) and k=1000
/// (plan M2 microbench / M4 gate (a) rank).
#[test]
#[ignore]
fn bench_kernel_complex_fb15k237_b1000() {
    for k in [500usize, 1000] {
        bench_complex(k);
    }
}

fn bench_complex(k: usize) {
    let (n, nr, b) = (14_541usize, 474usize, 1000usize);
    let dim = 2 * k;
    let ents = Tables::new(n, 1, dim, 3).entities_raw().to_vec();
    let rels = Tables::new(1, nr, dim, 5).relations_raw().to_vec();
    let qs = rows(n, nr, b / 2);
    let flops = 6.0 * b as f64 * n as f64 * dim as f64;
    for t in thread_counts() {
        let mut ws = ComplexWorkspace::new();
        let (mut ge, mut gr) = (vec![0.0; n * dim], vec![0.0; nr * dim]);
        let ms = with_threads(t, || {
            time_ms(3, || {
                complex_one_n_step(
                    &ents,
                    &rels,
                    k,
                    &qs,
                    Reduction::Mean,
                    &mut ws,
                    &mut ge,
                    &mut gr,
                )
                .unwrap();
            })
        });
        // Full epoch: 272,115 train triples × 2 directions = 544,230 rows.
        let epoch_min = 544_230.0 / b as f64 * ms / 1e3 / 60.0;
        println!(
            "BENCH kernel complex |E|=14541 k={k} B={b} threads={t}: {:.1} ms/batch, {:.4} ms/row, \
             {:.4} ms/pos, {:.1} GFLOP/s, projected FB15k-237 epoch {:.2} min",
            ms,
            ms / b as f64,
            2.0 * ms / b as f64,
            flops / (ms / 1e3) / 1e9,
            epoch_min
        );
    }
}

/// End-to-end: the recipe trainer (`Trainer::fit_with_kernel`, ComplEx-N3-R,
/// reciprocal, dense Adagrad, mean) on synthetic FB15k-237-sized tables with
/// the GEMM kernel — one epoch over 20,000 triples (40,000 rows, 40 batches
/// of 1000), projected to 544,230 rows. Includes everything the trainer does
/// per batch (query build, kernel, pull-back, N3, optimizer), unlike the
/// kernel-only figures above.
#[test]
#[ignore]
fn bench_fit_epoch_complex_fb15k237_k500() {
    use crate::data::TripleStore;
    use crate::scorer::ComplEx;
    use crate::train::{TrainConfig, Trainer};
    let (ne, nr, k, n_trip) = (14_541usize, 237usize, 500usize, 20_000usize);
    let d = 2 * k;
    let mut s = 0xFEEDu64;
    let mut next = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as u32
    };
    let triples: Vec<Triple> = (0..n_trip)
        .map(|_| Triple::new(next(ne), next(nr), next(ne)))
        .collect();
    let store = TripleStore::with_counts(triples, Some(ne), Some(nr)).unwrap();
    let cfg: TrainConfig = serde_json::from_str(&format!(
        r#"{{"loss":{{"kind":"one_vs_all"}},"reciprocal":true,"n3_form":"moduli",
            "loss_reduction":"mean","init":{{"kind":"normal","scale":0.001}},
            "optim_state":"dense","n3_lambda":0.05,"dims":{d},"epochs":1,
            "batch_size":1000,"lr":0.1,"seed":1}}"#
    ))
    .unwrap();
    let sc = ComplEx::new(d).unwrap();
    let kernel = super::GemmOneToN::new();
    for t in thread_counts() {
        let mut tables = Tables::new(ne, 2 * nr, d, 7);
        let start = Instant::now();
        with_threads(t, || {
            Trainer::fit_with_kernel(&mut tables, &sc, &store, &cfg, &kernel, |_| {}).unwrap()
        });
        let secs = start.elapsed().as_secs_f64();
        let rows = 2.0 * n_trip as f64;
        println!(
            "BENCH fit_with_kernel complex |E|=14541 k=500 B=1000 threads={t}: {:.2} s for {} rows, \
             {:.4} ms/pos, projected FB15k-237 epoch {:.2} min",
            secs,
            rows,
            2.0 * secs * 1e3 / rows,
            544_230.0 / rows * secs / 60.0
        );
    }
}
