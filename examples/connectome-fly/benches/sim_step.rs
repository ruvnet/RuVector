//! Criterion benchmark: per-simulated-ms wallclock cost.
//!
//! Measures the wall time to advance one simulated millisecond at
//! N=1024 neurons under both baseline (BinaryHeap + AoS) and optimized
//! (timing-wheel + SoA) paths. Deterministic pulse-train stimulus into
//! sensory neurons keeps spike density realistic.

use connectome_fly::{Connectome, ConnectomeConfig, Engine, EngineConfig, Observer, Stimulus};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

fn bench(c: &mut Criterion) {
    let conn = Connectome::generate(&ConnectomeConfig::default());
    // Fixed stimulus shared across iterations.
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 5.0, 90.0, 85.0, 120.0);
    let steps_per_iter: f32 = 10.0; // 10 simulated milliseconds per iter

    let mut group = c.benchmark_group("sim_step_ms");
    group.sample_size(25);

    group.bench_function("baseline_1ms", |b| {
        b.iter_batched(
            || {
                Engine::new(
                    &conn,
                    EngineConfig {
                        use_optimized: false,
                        ..EngineConfig::default()
                    },
                )
            },
            |mut eng| {
                let mut obs = Observer::new(conn.num_neurons());
                eng.run_with(&stim, &mut obs, steps_per_iter);
                black_box(obs.num_spikes())
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("optimized_1ms", |b| {
        b.iter_batched(
            || {
                Engine::new(
                    &conn,
                    EngineConfig {
                        use_optimized: true,
                        ..EngineConfig::default()
                    },
                )
            },
            |mut eng| {
                let mut obs = Observer::new(conn.num_neurons());
                eng.run_with(&stim, &mut obs, steps_per_iter);
                black_box(obs.num_spikes())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
