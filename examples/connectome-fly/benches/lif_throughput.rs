//! Criterion benchmark: LIF throughput (spikes/sec) at N ∈ {100, 1024,
//! 10_000}. Records both the baseline path (`use_optimized: false`,
//! BinaryHeap + AoS) and the optimized path (`use_optimized: true`,
//! timing-wheel + SoA).

use connectome_fly::{Connectome, ConnectomeConfig, Engine, EngineConfig, Observer, Stimulus};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

fn make_connectome(n: u32) -> Connectome {
    let cfg = ConnectomeConfig {
        num_neurons: n,
        avg_out_degree: if n >= 10_000 { 24.0 } else { 48.0 },
        seed: 0x51FE_D0FF_CAFE_BABE,
        ..ConnectomeConfig::default()
    };
    Connectome::generate(&cfg)
}

fn one_run(conn: &Connectome, use_optimized: bool, t_end_ms: f32) -> u64 {
    let mut eng = Engine::new(
        conn,
        EngineConfig {
            use_optimized,
            ..EngineConfig::default()
        },
    );
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 10.0, t_end_ms - 20.0, 80.0, 100.0);
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, t_end_ms);
    black_box(obs.finalize().total_spikes)
}

fn bench(c: &mut Criterion) {
    for &n in &[100u32, 1024, 10_000] {
        let conn = make_connectome(n);
        let t_end_ms: f32 = if n >= 10_000 { 60.0 } else { 120.0 };

        let mut group = c.benchmark_group(format!("lif_throughput_n_{n}"));
        group.sample_size(10);
        group.throughput(Throughput::Elements(1));

        group.bench_function("baseline", |b| {
            b.iter_batched(
                || (),
                |_| one_run(&conn, false, t_end_ms),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("optimized", |b| {
            b.iter_batched(
                || (),
                |_| one_run(&conn, true, t_end_ms),
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
