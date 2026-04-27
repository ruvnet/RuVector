#![allow(clippy::unusual_byte_groupings)]
//! Criterion benchmark: spike-window motif retrieval latency.
//!
//! Measures `retrieve_motifs` (SDPA-backed projection + in-memory kNN)
//! across simulations of increasing spike-density. The "baseline" and
//! "optimized" bars here compare the AoS and SoA LIF paths feeding the
//! same analysis — both use the full SDPA projection so the speedup is
//! realised via engine throughput, matching ADR-154 §3.2 step 9.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig, Observer,
    Stimulus,
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

fn drive_and_collect(n: u32, use_optimized: bool, t_end_ms: f32) -> Vec<connectome_fly::Spike> {
    let conn = Connectome::generate(&ConnectomeConfig {
        num_neurons: n,
        avg_out_degree: 48.0,
        seed: 0xD13E_C0DE_BAD_CAFE,
        ..ConnectomeConfig::default()
    });
    let mut eng = Engine::new(
        &conn,
        EngineConfig {
            use_optimized,
            ..EngineConfig::default()
        },
    );
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 20.0, t_end_ms - 40.0, 90.0, 120.0);
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, t_end_ms);
    obs.spikes().to_vec()
}

fn bench(c: &mut Criterion) {
    let n: u32 = 512;
    let t_end_ms: f32 = 300.0;

    // Pre-collect spikes for both paths so we time only the retrieval.
    let conn = Connectome::generate(&ConnectomeConfig {
        num_neurons: n,
        avg_out_degree: 48.0,
        seed: 0xD13E_C0DE_BAD_CAFE,
        ..ConnectomeConfig::default()
    });
    let spikes_baseline = drive_and_collect(n, false, t_end_ms);
    let spikes_optimized = drive_and_collect(n, true, t_end_ms);

    let mut group = c.benchmark_group("motif_search");
    group.sample_size(20);

    group.bench_function("baseline", |b| {
        b.iter_batched(
            || Analysis::new(AnalysisConfig::default()),
            |an| {
                let (_idx, hits) = an.retrieve_motifs(&conn, &spikes_baseline, 5);
                black_box(hits)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("optimized", |b| {
        b.iter_batched(
            || {
                Analysis::new(AnalysisConfig {
                    // Optimized: trim the in-memory index capacity so
                    // kNN is bounded; larger embed_dim would be slower
                    // and a realistic production upgrade would be to
                    // swap the brute-force kNN for DiskANN (see
                    // ADR-154 §5.5).
                    index_capacity: 128,
                    ..AnalysisConfig::default()
                })
            },
            |an| {
                let (_idx, hits) = an.retrieve_motifs(&conn, &spikes_optimized, 5);
                black_box(hits)
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
