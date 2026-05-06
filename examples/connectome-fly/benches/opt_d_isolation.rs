//! Criterion benchmark: Opt D (delay-sorted CSR) paired-sample
//! isolation, post-commit-10 (adaptive detect cadence always on).
//!
//! Runs the **same** workload as
//! `benches/lif_throughput.rs::lif_throughput_n_1024` (120 ms simulated,
//! default pulse-train into sensory neurons) across four paired arms:
//!
//!   heap-baseline                     : use_optimized=false,
//!                                       use_delay_sorted_csr=false
//!   wheel-SoA-SIMD                    : use_optimized=true,
//!                                       use_delay_sorted_csr=false
//!   wheel-SoA-SIMD + delay-CSR        : use_optimized=true,
//!                                       use_delay_sorted_csr=true
//!   heap + delay-CSR (diagnostic)     : use_optimized=false,
//!                                       use_delay_sorted_csr=true
//!
//! The adaptive detect cadence landed in commit 10
//! (`feat(observer): adaptive detect cadence …`). It is implemented
//! inside `Observer` and is not gated by any config — every arm here
//! runs with it enabled by construction. All four arms share the same
//! `connectome_seed` and the same `Stimulus`, so samples are paired at
//! the per-sample level: per-sample randomness between arms comes only
//! from the engine's spike-delivery ordering and OS scheduling jitter.
//!
//! The Opt-D-attributable delta in the saturated regime is the median
//! of arm (wheel-SoA-SIMD + delay-CSR) minus the median of arm
//! (wheel-SoA-SIMD). The diagnostic heap + delay-CSR arm exists to
//! check that the delay-sorted CSR is a no-op on the baseline path
//! (Opt D only takes effect when `use_optimized=true`; see
//! `src/lif/types.rs` `EngineConfig::use_delay_sorted_csr`) and so must
//! sit within the heap-baseline sample spread.

use connectome_fly::{Connectome, ConnectomeConfig, Engine, EngineConfig, Observer, Stimulus};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

/// Same N=1024 saturated-regime connectome `delay_csr.rs` and
/// `lif_throughput.rs` use. Seed is hard-coded so every arm sees the
/// same topology; this plus the fixed `Stimulus` is what makes the
/// samples paired.
fn make_connectome() -> Connectome {
    let cfg = ConnectomeConfig {
        num_neurons: 1024,
        avg_out_degree: 48.0,
        seed: 0x51FE_D0FF_CAFE_BABE,
        ..ConnectomeConfig::default()
    };
    Connectome::generate(&cfg)
}

/// Single bench iteration — build the engine, run 120 ms, return the
/// total spike count. `black_box` on the return value keeps LLVM from
/// dead-code-eliminating the spike-delivery path; the engine and
/// observer are freshly constructed per iteration so adaptive-cadence
/// state does not leak between samples.
fn one_run(conn: &Connectome, cfg: EngineConfig, t_end_ms: f32) -> u64 {
    let mut eng = Engine::new(conn, cfg);
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 10.0, t_end_ms - 20.0, 80.0, 100.0);
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, t_end_ms);
    black_box(obs.finalize().total_spikes)
}

fn bench(c: &mut Criterion) {
    // Shared construction outside the per-sample loop — identical
    // across all four arms, which is what makes the samples paired.
    let conn = make_connectome();
    let t_end_ms: f32 = 120.0;

    let mut group = c.benchmark_group("opt_d_isolation_n_1024");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));

    group.bench_function("heap-baseline", |b| {
        b.iter_batched(
            || (),
            |_| {
                one_run(
                    &conn,
                    EngineConfig {
                        use_optimized: false,
                        use_delay_sorted_csr: false,
                        ..EngineConfig::default()
                    },
                    t_end_ms,
                )
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("wheel-SoA-SIMD", |b| {
        b.iter_batched(
            || (),
            |_| {
                one_run(
                    &conn,
                    EngineConfig {
                        use_optimized: true,
                        use_delay_sorted_csr: false,
                        ..EngineConfig::default()
                    },
                    t_end_ms,
                )
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("wheel-SoA-SIMD+delay-csr", |b| {
        b.iter_batched(
            || (),
            |_| {
                one_run(
                    &conn,
                    EngineConfig {
                        use_optimized: true,
                        use_delay_sorted_csr: true,
                        ..EngineConfig::default()
                    },
                    t_end_ms,
                )
            },
            BatchSize::SmallInput,
        )
    });

    // Diagnostic: `use_delay_sorted_csr=true` is ignored on the
    // baseline path (heap + AoS). This arm exists to confirm that the
    // flag is correctly gated — its median should sit inside the
    // heap-baseline sample spread.
    group.bench_function("heap+delay-csr-diag", |b| {
        b.iter_batched(
            || (),
            |_| {
                one_run(
                    &conn,
                    EngineConfig {
                        use_optimized: false,
                        use_delay_sorted_csr: true,
                        ..EngineConfig::default()
                    },
                    t_end_ms,
                )
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
