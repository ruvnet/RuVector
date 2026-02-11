//! Delta Propagation Benchmarks
//!
//! Benchmarks delta creation, application, composition, stream operations,
//! and checkpoint creation/restoration for genomic vector pipelines.
//!
//! Run: cargo bench -p ruvector-dna-bench --bench bench_delta_propagation

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use ruvector_delta_core::{
    Delta, DeltaStream, DeltaStreamConfig, StreamCheckpoint, VectorDelta,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random vector generator.
fn gen_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 33) ^ state) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Generate a pair of vectors with a given sparsity of changes.
/// `sparsity` is the fraction of dimensions that remain unchanged (0.0 = all change, 1.0 = none).
fn gen_delta_pair(dim: usize, sparsity: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let old = gen_vector(dim, seed);
    let mut new = old.clone();
    let mut state = seed.wrapping_add(999_999);
    for i in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (state >> 33) as f32 / (u32::MAX as f32);
        if r > sparsity {
            // Perturb this dimension
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let perturbation = ((state >> 33) as f32 / u32::MAX as f32) * 0.2 - 0.1;
            new[i] += perturbation;
        }
    }
    (old, new)
}

// ---------------------------------------------------------------------------
// Benchmark: Delta Creation
// ---------------------------------------------------------------------------

fn bench_delta_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/creation");

    let dim = 384;

    // Test different sparsity levels
    for &sparsity in &[0.0f32, 0.5, 0.9, 0.99] {
        let (old, new) = gen_delta_pair(dim, sparsity, 42);

        group.bench_with_input(
            BenchmarkId::new("sparsity", format!("{:.0}pct", sparsity * 100.0)),
            &(old.clone(), new.clone()),
            |b, (old, new)| {
                b.iter(|| VectorDelta::compute(old, new));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Delta Application
// ---------------------------------------------------------------------------

fn bench_delta_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/apply");

    let dim = 384;

    for &sparsity in &[0.0f32, 0.5, 0.9, 0.99] {
        let (old, new) = gen_delta_pair(dim, sparsity, 42);
        let delta = VectorDelta::compute(&old, &new);

        group.bench_with_input(
            BenchmarkId::new("sparsity", format!("{:.0}pct", sparsity * 100.0)),
            &(old.clone(), delta.clone()),
            |b, (base, delta)| {
                b.iter(|| {
                    let mut v = base.clone();
                    delta.apply(&mut v).expect("apply");
                    v
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Delta Composition (chaining two deltas)
// ---------------------------------------------------------------------------

fn bench_delta_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/composition");

    let dim = 384;

    for &sparsity in &[0.0f32, 0.5, 0.9] {
        let (v1, v2) = gen_delta_pair(dim, sparsity, 42);
        let (_, v3) = gen_delta_pair(dim, sparsity, 123);

        let delta1 = VectorDelta::compute(&v1, &v2);
        let delta2 = VectorDelta::compute(&v2, &v3);

        group.bench_with_input(
            BenchmarkId::new("sparsity", format!("{:.0}pct", sparsity * 100.0)),
            &(delta1.clone(), delta2.clone()),
            |b, (d1, d2)| {
                b.iter(|| d1.clone().compose(d2.clone()));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Delta Inverse
// ---------------------------------------------------------------------------

fn bench_delta_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/inverse");

    let dim = 384;

    for &sparsity in &[0.0f32, 0.5, 0.9] {
        let (old, new) = gen_delta_pair(dim, sparsity, 42);
        let delta = VectorDelta::compute(&old, &new);

        group.bench_with_input(
            BenchmarkId::new("sparsity", format!("{:.0}pct", sparsity * 100.0)),
            &delta,
            |b, delta| {
                b.iter(|| delta.inverse());
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Delta Byte Size
// ---------------------------------------------------------------------------

fn bench_delta_byte_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/byte_size");

    let dim = 384;

    for &sparsity in &[0.0f32, 0.5, 0.9, 0.99] {
        let (old, new) = gen_delta_pair(dim, sparsity, 42);
        let delta = VectorDelta::compute(&old, &new);

        group.bench_with_input(
            BenchmarkId::new("sparsity", format!("{:.0}pct", sparsity * 100.0)),
            &delta,
            |b, delta| {
                b.iter(|| delta.byte_size());
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Stream Push (pipeline ingestion)
// ---------------------------------------------------------------------------

fn bench_stream_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/stream_push");
    group.sample_size(10);

    let dim = 384;

    for &batch_size in &[100usize, 1_000, 10_000] {
        // Pre-generate deltas
        let deltas: Vec<VectorDelta> = (0..batch_size)
            .map(|i| {
                let (old, new) = gen_delta_pair(dim, 0.9, i as u64);
                VectorDelta::compute(&old, &new)
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("deltas", batch_size),
            &deltas,
            |b, deltas| {
                b.iter(|| {
                    let config = DeltaStreamConfig {
                        max_deltas: batch_size + 1000,
                        checkpoint_interval: batch_size / 10,
                        max_memory_bytes: 256 * 1024 * 1024,
                        auto_compact: false,
                    };
                    let mut stream = DeltaStream::<VectorDelta>::with_config(config);
                    for delta in deltas {
                        stream.push(delta.clone());
                    }
                    stream
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Stream Replay (full reconstruction)
// ---------------------------------------------------------------------------

fn bench_stream_replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/stream_replay");
    group.sample_size(10);

    let dim = 384;

    for &num_deltas in &[100usize, 1_000] {
        let config = DeltaStreamConfig {
            max_deltas: num_deltas + 1000,
            checkpoint_interval: num_deltas + 1,
            max_memory_bytes: 256 * 1024 * 1024,
            auto_compact: false,
        };
        let mut stream = DeltaStream::<VectorDelta>::with_config(config);
        let initial = gen_vector(dim, 0);

        for i in 0..num_deltas {
            let (old, new) = gen_delta_pair(dim, 0.9, i as u64);
            let delta = VectorDelta::compute(&old, &new);
            stream.push(delta);
        }

        group.throughput(Throughput::Elements(num_deltas as u64));
        group.bench_with_input(
            BenchmarkId::new("deltas", num_deltas),
            &(stream, initial.clone()),
            |b, (stream, initial)| {
                b.iter(|| stream.replay(initial.clone()).expect("replay"));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Checkpoint Creation
// ---------------------------------------------------------------------------

fn bench_checkpoint_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/checkpoint_create");

    let dim = 384;

    for &num_deltas in &[100usize, 1_000] {
        let config = DeltaStreamConfig {
            max_deltas: num_deltas + 1000,
            checkpoint_interval: num_deltas + 1,
            max_memory_bytes: 256 * 1024 * 1024,
            auto_compact: false,
        };
        let mut stream = DeltaStream::<VectorDelta>::with_config(config);
        let initial = gen_vector(dim, 0);

        for i in 0..num_deltas {
            let (old, new) = gen_delta_pair(dim, 0.9, i as u64);
            stream.push(VectorDelta::compute(&old, &new));
        }

        // Replay to get current state
        let current_state = stream.replay(initial.clone()).expect("replay");

        group.bench_with_input(
            BenchmarkId::new("after_deltas", num_deltas),
            &(stream.clone(), current_state.clone()),
            |b, (stream, state)| {
                b.iter_batched(
                    || stream.clone(),
                    |mut s| {
                        s.create_checkpoint(state.clone());
                        s
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Checkpoint Restoration (replay from checkpoint)
// ---------------------------------------------------------------------------

fn bench_checkpoint_restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/checkpoint_restore");
    group.sample_size(10);

    let dim = 384;
    let num_before_checkpoint = 500;
    let num_after_checkpoint = 500;

    let config = DeltaStreamConfig {
        max_deltas: 2000,
        checkpoint_interval: 2000,
        max_memory_bytes: 256 * 1024 * 1024,
        auto_compact: false,
    };
    let mut stream = DeltaStream::<VectorDelta>::with_config(config);
    let initial = gen_vector(dim, 0);

    // Push deltas before checkpoint
    for i in 0..num_before_checkpoint {
        let (old, new) = gen_delta_pair(dim, 0.9, i as u64);
        stream.push(VectorDelta::compute(&old, &new));
    }

    // Create checkpoint
    let checkpoint_state = stream.replay(initial.clone()).expect("replay");
    stream.create_checkpoint(checkpoint_state);

    // Push more deltas after checkpoint
    for i in 0..num_after_checkpoint {
        let (old, new) = gen_delta_pair(dim, 0.9, (i + num_before_checkpoint) as u64);
        stream.push(VectorDelta::compute(&old, &new));
    }

    group.bench_function("restore_from_checkpoint", |b| {
        b.iter(|| {
            stream
                .replay_from_checkpoint(0)
                .expect("checkpoint exists")
                .expect("replay ok")
        });
    });

    group.bench_function("replay_from_scratch", |b| {
        b.iter(|| stream.replay(initial.clone()).expect("replay"));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Stream Compaction
// ---------------------------------------------------------------------------

fn bench_stream_compaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/compaction");
    group.sample_size(10);

    let dim = 384;

    for &num_deltas in &[100usize, 1_000] {
        let config = DeltaStreamConfig {
            max_deltas: num_deltas + 10000,
            checkpoint_interval: num_deltas + 10000,
            max_memory_bytes: 256 * 1024 * 1024,
            auto_compact: false,
        };

        group.bench_with_input(
            BenchmarkId::new("deltas", num_deltas),
            &num_deltas,
            |b, &num_deltas| {
                b.iter_batched(
                    || {
                        let mut stream = DeltaStream::<VectorDelta>::with_config(config.clone());
                        for i in 0..num_deltas {
                            let (old, new) = gen_delta_pair(dim, 0.9, i as u64);
                            stream.push(VectorDelta::compute(&old, &new));
                        }
                        stream
                    },
                    |mut stream| {
                        let compacted = stream.compact().expect("compact");
                        compacted
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Dimension Scaling
// ---------------------------------------------------------------------------

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_propagation/dimension_scaling");

    for &dim in &[64usize, 128, 384, 768, 1536] {
        let (old, new) = gen_delta_pair(dim, 0.5, 42);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("compute_dim", dim),
            &(old.clone(), new.clone()),
            |b, (old, new)| {
                b.iter(|| VectorDelta::compute(old, new));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_delta_creation,
    bench_delta_apply,
    bench_delta_composition,
    bench_delta_inverse,
    bench_delta_byte_size,
    bench_stream_push,
    bench_stream_replay,
    bench_checkpoint_creation,
    bench_checkpoint_restore,
    bench_stream_compaction,
    bench_dimension_scaling,
);
criterion_main!(benches);
