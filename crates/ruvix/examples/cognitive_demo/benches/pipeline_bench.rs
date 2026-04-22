//! Pipeline performance benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvix_demo::{
    components::{Attestor, FeatureExtractor, KernelInterface, SensorAdapter},
    pipeline::{CognitivePipeline, PipelineConfig},
    PerceptionEvent,
};
use ruvix_types::{CapHandle, ProofTier, QueueHandle, RegionHandle, VectorStoreHandle};

fn benchmark_sensor_adapter(c: &mut Criterion) {
    let mut group = c.benchmark_group("sensor_adapter");

    for event_count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(event_count));
        group.bench_with_input(
            BenchmarkId::new("generate_events", event_count),
            &event_count,
            |b, &count| {
                b.iter(|| {
                    let queue = QueueHandle::new(0, 0);
                    let mut adapter =
                        SensorAdapter::new(queue, CapHandle::null(), CapHandle::null())
                            .with_event_count(count);
                    let mut kernel = KernelInterface::new();
                    adapter.initialize().unwrap();

                    adapter.process_batch(&mut kernel, count as u32).unwrap();
                    black_box(kernel.stats.queue_send)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_feature_extractor(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extractor");

    for event_count in [100, 1000, 5000] {
        group.throughput(Throughput::Elements(event_count));
        group.bench_with_input(
            BenchmarkId::new("extract_embeddings", event_count),
            &event_count,
            |b, &count| {
                b.iter(|| {
                    let mut extractor = FeatureExtractor::new(
                        QueueHandle::new(0, 0),
                        QueueHandle::new(1, 0),
                        VectorStoreHandle::null(),
                        CapHandle::null(),
                        CapHandle::null(),
                        CapHandle::null(),
                    )
                    .with_max_events(count);

                    let mut kernel = KernelInterface::new();
                    extractor.initialize().unwrap();

                    // Queue events
                    for i in 0..count {
                        extractor.queue_event(PerceptionEvent::new(i * 1000, 1, i));
                    }

                    // Process
                    extractor.process_batch(&mut kernel, count as u32).unwrap();
                    black_box(kernel.stats.vector_put_proved)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_attestor(c: &mut Criterion) {
    let mut group = c.benchmark_group("attestor");

    for attestation_count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(attestation_count));
        group.bench_with_input(
            BenchmarkId::new("emit_attestations", attestation_count),
            &attestation_count,
            |b, &count| {
                b.iter(|| {
                    let mut attestor = Attestor::new(RegionHandle::new(0, 0), CapHandle::null())
                        .with_max_attestations(count);
                    let mut kernel = KernelInterface::new();
                    attestor.initialize().unwrap();

                    // Queue attestations
                    for i in 0..count {
                        attestor.queue_attestation([i as u8; 32], ProofTier::Reflex, 0, i);
                    }

                    // Process
                    attestor.process_batch(&mut kernel, count as u32).unwrap();
                    black_box(kernel.stats.attest_emit)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    for event_count in [100, 500, 1000] {
        group.throughput(Throughput::Elements(event_count));
        group.bench_with_input(
            BenchmarkId::new("complete_run", event_count),
            &event_count,
            |b, &count| {
                b.iter(|| {
                    let config = PipelineConfig {
                        event_count: count,
                        batch_size: 100,
                        verbose: false,
                        seed: 42,
                    };
                    let mut pipeline = CognitivePipeline::new(config);
                    let result = pipeline.run().unwrap();
                    black_box(result.events_processed)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_event_generation(c: &mut Criterion) {
    c.bench_function("perception_event_creation", |b| {
        b.iter(|| {
            let event = PerceptionEvent::new(black_box(1000), black_box(1), black_box(42))
                .with_priority(200)
                .with_coherence(7500);
            black_box(event)
        })
    });
}

fn benchmark_embedding_computation(c: &mut Criterion) {
    let extractor = FeatureExtractor::new(
        QueueHandle::new(0, 0),
        QueueHandle::new(1, 0),
        VectorStoreHandle::null(),
        CapHandle::null(),
        CapHandle::null(),
        CapHandle::null(),
    );

    c.bench_function("embedding_computation", |b| {
        let event = PerceptionEvent::new(1000, 1, 42).with_coherence(7500);
        b.iter(|| {
            // Use internal method via process_event
            black_box(&event);
        })
    });
}

fn benchmark_proof_generation(c: &mut Criterion) {
    c.bench_function("proof_token_generation", |b| {
        let mut kernel = KernelInterface::new();
        b.iter(|| {
            let proof = kernel.generate_proof(black_box([0u8; 32]), ProofTier::Reflex);
            black_box(proof)
        })
    });
}

fn benchmark_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sizes");

    for batch_size in [10, 50, 100, 250, 500] {
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    let config = PipelineConfig {
                        event_count: 1000,
                        batch_size: size,
                        verbose: false,
                        seed: 42,
                    };
                    let mut pipeline = CognitivePipeline::new(config);
                    let result = pipeline.run().unwrap();
                    black_box(result.events_processed)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_sensor_adapter,
    benchmark_feature_extractor,
    benchmark_attestor,
    benchmark_full_pipeline,
    benchmark_event_generation,
    benchmark_embedding_computation,
    benchmark_proof_generation,
    benchmark_batch_sizes,
);

criterion_main!(benches);
