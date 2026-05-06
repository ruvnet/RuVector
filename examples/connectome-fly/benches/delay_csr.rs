//! Criterion benchmark: Opt D (delay-sorted CSR) saturated-regime
//! throughput at N=1024.
//!
//! Runs the **same** workload as
//! `benches/lif_throughput.rs::lif_throughput_n_1024` (120 ms simulated,
//! default pulse-train into sensory neurons) with three rows:
//!
//!   baseline                : `use_optimized=false` (heap + AoS)
//!   scalar-opt              : `use_optimized=true`, default CSR
//!   scalar-opt + delay-csr  : `use_optimized=true,
//!                              use_delay_sorted_csr=true` — Opt D
//!
//! ADR-154 §3.2 target for Opt D is ≥ 2× over scalar-opt in the saturated
//! regime. The speedup delta is reported by Criterion's median ratio;
//! the commit message captures the measured number.

use connectome_fly::{Connectome, ConnectomeConfig, Engine, EngineConfig, Observer, Stimulus};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

/// Saturated-regime connectome, default SBM seeded deterministically.
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
/// observer are freshly constructed per iteration so state does not
/// leak between samples.
fn one_run(conn: &Connectome, cfg: EngineConfig, t_end_ms: f32) -> u64 {
    let mut eng = Engine::new(conn, cfg);
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 10.0, t_end_ms - 20.0, 80.0, 100.0);
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, t_end_ms);
    black_box(obs.finalize().total_spikes)
}

fn bench(c: &mut Criterion) {
    let conn = make_connectome();
    let t_end_ms: f32 = 120.0;

    let mut group = c.benchmark_group("lif_throughput_n_1024");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));

    group.bench_function("baseline", |b| {
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

    group.bench_function("scalar-opt", |b| {
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

    group.bench_function("scalar-opt+delay-csr", |b| {
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

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
