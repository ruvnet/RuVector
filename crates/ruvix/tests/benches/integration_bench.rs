//! Integration benchmarks for the RuVix Cognition Kernel.
//!
//! These benchmarks measure end-to-end performance across subsystems,
//! validating ADR-087 performance targets.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use ruvix_cap::{CapManagerConfig, CapRights, CapabilityManager, ObjectType, TaskHandle};
use ruvix_queue::{KernelQueue, QueueConfig};
use ruvix_region::{
    append_only::AppendOnlyRegion, backing::StaticBacking, immutable::ImmutableRegion,
    slab::SlabAllocator,
};
use ruvix_types::{MsgPriority, RegionHandle};

// ============================================================================
// End-to-End Syscall Path Benchmarks
// ============================================================================

fn bench_capability_gated_region_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_cap_gated_access");

    group.bench_function("read_with_cap_check", |b| {
        // Setup
        let config = CapManagerConfig::default();
        let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
        let task = TaskHandle::new(1, 0);

        let backing = StaticBacking::<4096>::new();
        let region_handle = RegionHandle::new(1, 0);
        let data = vec![0xABu8; 1024];
        let region = ImmutableRegion::new(backing, &data, region_handle).unwrap();

        let cap = cap_manager
            .create_root_capability(1, ObjectType::Region, 0, task)
            .unwrap();

        let mut buf = [0u8; 1024];

        b.iter(|| {
            // Full syscall path: check capability, then read
            let _ = cap_manager.has_rights(black_box(cap), CapRights::READ);
            region.read(0, black_box(&mut buf)).unwrap();
        });
    });

    group.bench_function("write_with_cap_check", |b| {
        b.iter_with_setup(
            || {
                let config = CapManagerConfig::default();
                let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
                let task = TaskHandle::new(1, 0);

                let backing = StaticBacking::<8192>::new();
                let region_handle = RegionHandle::new(1, 0);
                let region = AppendOnlyRegion::new(backing, 8192, region_handle).unwrap();

                let cap = cap_manager
                    .create_root_capability(1, ObjectType::Region, 0, task)
                    .unwrap();

                (cap_manager, region, cap, task)
            },
            |(cap_manager, mut region, cap, _task)| {
                // Full syscall path: check capability, then write
                let _ = cap_manager.has_rights(black_box(cap), CapRights::WRITE);
                region.append(black_box(&[0xABu8; 64])).unwrap();
            },
        );
    });

    group.finish();
}

fn bench_queue_ipc_with_capability(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_queue_ipc");

    group.bench_function("send_recv_cycle", |b| {
        let config = CapManagerConfig::default();
        let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

        let sender = TaskHandle::new(1, 0);
        let receiver = TaskHandle::new(2, 0);

        let send_cap = cap_manager
            .create_root_capability(1, ObjectType::Queue, 0, sender)
            .unwrap();

        let recv_cap = cap_manager
            .grant(send_cap, CapRights::READ, 0, sender, receiver)
            .unwrap();

        let queue_config = QueueConfig::new(64, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

        let msg = [0xABu8; 64];
        let mut recv_buf = [0u8; 256];

        b.iter(|| {
            // Sender path
            let _ = cap_manager.has_rights(send_cap, CapRights::WRITE);
            queue.send(black_box(&msg), MsgPriority::Normal).unwrap();

            // Receiver path
            let _ = cap_manager.has_rights(recv_cap, CapRights::READ);
            black_box(queue.recv(&mut recv_buf))
        });
    });

    group.bench_function("high_priority_send_recv", |b| {
        let queue_config = QueueConfig::new(64, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();
        let msg = [0xABu8; 64];
        let mut recv_buf = [0u8; 256];

        b.iter(|| {
            queue.send(black_box(&msg), MsgPriority::Urgent).unwrap();
            black_box(queue.recv(&mut recv_buf))
        });
    });

    group.finish();
}

// ============================================================================
// Slab + Capability Integration Benchmarks
// ============================================================================

fn bench_slab_with_capability(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_slab_cap");

    group.bench_function("alloc_write_read_free", |b| {
        let config = CapManagerConfig::default();
        let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
        let task = TaskHandle::new(1, 0);

        let backing = StaticBacking::<8192>::new();
        let mut slab = SlabAllocator::new(backing, 64, 64).unwrap();

        let cap = cap_manager
            .create_root_capability(1, ObjectType::Region, 0, task)
            .unwrap();

        let data = [0xABu8; 64];
        let mut buf = [0u8; 64];

        b.iter(|| {
            // Check capability
            let _ = cap_manager.has_rights(cap, CapRights::READ | CapRights::WRITE);

            // Full slab cycle
            let handle = slab.alloc().unwrap();
            slab.write(handle, black_box(&data)).unwrap();
            slab.read(handle, black_box(&mut buf)).unwrap();
            slab.free(handle).unwrap();
        });
    });

    group.finish();
}

// ============================================================================
// Capability Delegation Chain Benchmarks
// ============================================================================

fn bench_delegation_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_delegation");

    for depth in [2, 4, 6, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("grant_chain", depth),
            depth,
            |b, &depth| {
                b.iter_with_setup(
                    || {
                        let config = CapManagerConfig::default();
                        let manager: CapabilityManager<1024> = CapabilityManager::new(config);
                        manager
                    },
                    |mut manager| {
                        let root_task = TaskHandle::new(0, 0);
                        let root_cap = manager
                            .create_root_capability(0x1000, ObjectType::Region, 0, root_task)
                            .unwrap();

                        let mut current_cap = root_cap;
                        let mut current_task = root_task;

                        for i in 0..depth {
                            let next_task = TaskHandle::new(i as u32 + 1, 0);
                            current_cap = manager
                                .grant(
                                    current_cap,
                                    CapRights::READ | CapRights::GRANT,
                                    i as u64,
                                    current_task,
                                    next_task,
                                )
                                .unwrap();
                            current_task = next_task;
                        }

                        black_box(current_cap)
                    },
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Perception Pipeline Benchmarks (ADR-087 Cognition Flow)
// ============================================================================

fn bench_perception_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_perception");

    group.throughput(Throughput::Elements(100));
    group.bench_function("100_events", |b| {
        b.iter_with_setup(
            || {
                let backing = StaticBacking::<65536>::new();
                let handle = RegionHandle::new(1, 0);
                let region = AppendOnlyRegion::new(backing, 65536, handle).unwrap();

                let queue_config = QueueConfig::new(256, 128);
                let (queue, buffer) = KernelQueue::new_heap(queue_config).unwrap();
                (region, queue, buffer)
            },
            |(mut region, mut queue, _buffer)| {
                let mut recv_buf = [0u8; 128];

                for i in 0..100 {
                    // Emit perception event
                    let event_data = [i as u8; 64];
                    region.append(black_box(&event_data)).unwrap();

                    // Queue for processing
                    queue
                        .send(black_box(&event_data), MsgPriority::High)
                        .unwrap();
                }

                // Process all events
                for _ in 0..100 {
                    if let Ok(len) = queue.recv(&mut recv_buf) {
                        black_box(len);
                    }
                }
            },
        );
    });

    group.finish();
}

// ============================================================================
// Checkpoint/Attestation Benchmarks
// ============================================================================

fn bench_checkpoint_attestation(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_checkpoint");

    fn fnv1a_hash(data: &[u8]) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        for byte in data {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    for size in [1024, 4096, 16384].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("checkpoint", size), size, |b, &size| {
            let backing = StaticBacking::<65536>::new();
            let handle = RegionHandle::new(1, 0);
            let mut region = AppendOnlyRegion::new(backing, 65536, handle).unwrap();

            // Fill region
            let chunk_size = 64;
            let chunks = size / chunk_size;
            for i in 0..chunks {
                let data = [i as u8; 64];
                region.append(&data).unwrap();
            }

            let mut buf = vec![0u8; size];

            b.iter(|| {
                // Read state
                region.read(0, black_box(&mut buf)).unwrap();

                // Compute attestation hash
                let hash = fnv1a_hash(&buf);
                black_box(hash)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Concurrent Access Pattern Benchmarks
// ============================================================================

fn bench_multi_task_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_multi_task");

    group.bench_function("producer_consumer", |b| {
        b.iter_with_setup(
            || {
                let queue_config = QueueConfig::new(256, 128);
                KernelQueue::new_heap(queue_config).unwrap()
            },
            |(mut queue, _buffer)| {
                let msg = [0xABu8; 64];
                let mut recv_buf = [0u8; 128];

                // Producer: send 50 messages
                for _ in 0..50 {
                    queue.send(black_box(&msg), MsgPriority::Normal).unwrap();
                }

                // Consumer: receive all
                for _ in 0..50 {
                    black_box(queue.recv(&mut recv_buf));
                }
            },
        );
    });

    group.bench_function("interleaved_send_recv", |b| {
        let queue_config = QueueConfig::new(64, 128);
        let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();
        let msg = [0xABu8; 64];
        let mut recv_buf = [0u8; 128];

        b.iter(|| {
            for _ in 0..10 {
                // Send 3
                for _ in 0..3 {
                    let _ = queue.send(black_box(&msg), MsgPriority::Normal);
                }
                // Receive 2
                black_box(queue.recv(&mut recv_buf));
                black_box(queue.recv(&mut recv_buf));
            }

            // Drain remaining
            while queue.recv(&mut recv_buf).is_ok() {
                // Continue draining
            }
        });
    });

    group.finish();
}

// ============================================================================
// ADR-087 Performance Target Benchmarks
// ============================================================================

fn bench_adr087_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("adr087_targets");
    group.sample_size(1000);

    // Target: capability lookup should be O(1), constant time
    group.bench_function("cap_lookup_o1", |b| {
        let config = CapManagerConfig::default();
        let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
        let task = TaskHandle::new(1, 0);

        // Create many capabilities
        let caps: Vec<_> = (0..100)
            .map(|i| {
                cap_manager
                    .create_root_capability(i as u64, ObjectType::Region, 0, task)
                    .unwrap()
            })
            .collect();

        let target_cap = caps[50];

        b.iter(|| {
            black_box(cap_manager.has_rights(black_box(target_cap), black_box(CapRights::READ)))
        });
    });

    // Target: zero-copy region access
    group.bench_function("zero_copy_access", |b| {
        let data = vec![0xABu8; 4096];
        let backing = StaticBacking::<8192>::new();
        let handle = RegionHandle::new(1, 0);
        let region = ImmutableRegion::new(backing, &data, handle).unwrap();

        b.iter(|| {
            // Zero-copy access via as_slice
            let slice = region.as_slice();
            black_box(slice)
        });
    });

    // Target: queue operations minimal overhead
    group.bench_function("queue_minimal_overhead", |b| {
        let queue_config = QueueConfig::new(64, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();
        let msg = [0xABu8; 64];
        let mut recv_buf = [0u8; 256];

        b.iter(|| {
            queue.send(black_box(&msg), MsgPriority::Normal).unwrap();
            black_box(queue.recv(&mut recv_buf))
        });
    });

    group.finish();
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_throughput");

    // Messages per second through full pipeline
    group.throughput(Throughput::Elements(1000));
    group.bench_function("full_pipeline_1000", |b| {
        b.iter_with_setup(
            || {
                let config = CapManagerConfig::default();
                let cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

                let backing = StaticBacking::<65536>::new();
                let handle = RegionHandle::new(1, 0);
                let region = AppendOnlyRegion::new(backing, 65536, handle).unwrap();

                let queue_config = QueueConfig::new(1024, 64);
                let (queue, buffer) = KernelQueue::new_heap(queue_config).unwrap();

                (cap_manager, region, queue, buffer)
            },
            |(mut cap_manager, mut region, mut queue, _buffer)| {
                let task = TaskHandle::new(1, 0);
                let cap = cap_manager
                    .create_root_capability(1, ObjectType::Region, 0, task)
                    .unwrap();

                let mut recv_buf = [0u8; 64];

                for i in 0..1000 {
                    // Check capability
                    let _ = cap_manager.has_rights(cap, CapRights::WRITE);

                    // Write to region
                    let data = [i as u8; 32];
                    region.append(black_box(&data)).unwrap();

                    // Queue message
                    queue.send(black_box(&data), MsgPriority::Normal).unwrap();
                }

                // Process queue
                for _ in 0..1000 {
                    black_box(queue.recv(&mut recv_buf));
                }
            },
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_capability_gated_region_access,
    bench_queue_ipc_with_capability,
    bench_slab_with_capability,
    bench_delegation_chain,
    bench_perception_pipeline,
    bench_checkpoint_attestation,
    bench_multi_task_patterns,
    bench_adr087_targets,
    bench_throughput,
);

criterion_main!(benches);
