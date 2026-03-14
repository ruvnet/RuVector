//! Benchmarks for ruvix-queue operations.
//!
//! Measures performance of queue operations, ring buffers, and zero-copy messaging.
//! Target performance from ADR-087: zero-copy queue operations with minimal overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvix_queue::{
    descriptor::MessageDescriptor,
    kernel_queue::KernelQueue,
    ring::{RingBuffer, RingEntry},
    Duration, Priority,
};
use ruvix_types::RegionHandle;

// ============================================================================
// Ring Buffer Benchmarks
// ============================================================================

fn bench_ring_enqueue(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_enqueue");

    for capacity in [16, 64, 256].iter() {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("single", capacity),
            capacity,
            |b, &capacity| {
                b.iter_with_setup(
                    || RingBuffer::new(capacity),
                    |mut ring| {
                        let entry = RingEntry::new(1, 64, Priority::Normal);
                        let _ = ring.enqueue(black_box(entry));
                    },
                );
            },
        );
    }

    // Benchmark enqueue when nearly full
    group.bench_function("near_full", |b| {
        b.iter_with_setup(
            || {
                let mut ring = RingBuffer::new(64);
                // Fill to 62/64
                for i in 0..62 {
                    let entry = RingEntry::new(i as u64, 64, Priority::Normal);
                    ring.enqueue(entry).unwrap();
                }
                ring
            },
            |mut ring| {
                let entry = RingEntry::new(100, 64, Priority::Normal);
                let _ = ring.enqueue(black_box(entry));
            },
        );
    });

    group.finish();
}

fn bench_ring_dequeue(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_dequeue");

    group.bench_function("single", |b| {
        b.iter_with_setup(
            || {
                let mut ring = RingBuffer::new(64);
                for i in 0..64 {
                    let entry = RingEntry::new(i as u64, 64, Priority::Normal);
                    ring.enqueue(entry).unwrap();
                }
                ring
            },
            |mut ring| {
                let _ = ring.dequeue();
            },
        );
    });

    group.bench_function("batch_16", |b| {
        b.iter_with_setup(
            || {
                let mut ring = RingBuffer::new(64);
                for i in 0..64 {
                    let entry = RingEntry::new(i as u64, 64, Priority::Normal);
                    ring.enqueue(entry).unwrap();
                }
                ring
            },
            |mut ring| {
                for _ in 0..16 {
                    black_box(ring.dequeue());
                }
            },
        );
    });

    group.finish();
}

fn bench_ring_enqueue_dequeue_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_cycle");

    group.bench_function("single_cycle", |b| {
        let mut ring = RingBuffer::new(64);

        b.iter(|| {
            let entry = RingEntry::new(42, 64, Priority::Normal);
            ring.enqueue(black_box(entry)).unwrap();
            black_box(ring.dequeue())
        });
    });

    group.bench_function("burst_8", |b| {
        let mut ring = RingBuffer::new(64);

        b.iter(|| {
            // Burst enqueue
            for i in 0..8 {
                let entry = RingEntry::new(i as u64, 64, Priority::Normal);
                ring.enqueue(entry).unwrap();
            }
            // Burst dequeue
            for _ in 0..8 {
                black_box(ring.dequeue());
            }
        });
    });

    group.finish();
}

// ============================================================================
// Ring Entry Benchmarks
// ============================================================================

fn bench_ring_entry(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_entry");

    group.bench_function("create", |b| {
        b.iter(|| {
            black_box(RingEntry::new(
                black_box(42),
                black_box(128),
                black_box(Priority::High),
            ))
        });
    });

    group.bench_function("create_with_offset", |b| {
        b.iter(|| {
            black_box(RingEntry::with_offset(
                black_box(42),
                black_box(128),
                black_box(256),
                black_box(Priority::Normal),
            ))
        });
    });

    group.bench_function("is_valid", |b| {
        let entry = RingEntry::new(42, 128, Priority::High);
        b.iter(|| black_box(entry.is_valid()));
    });

    group.finish();
}

// ============================================================================
// Kernel Queue Benchmarks
// ============================================================================

fn bench_kernel_queue_send(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_queue_send");

    for capacity in [16, 64, 256].iter() {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("single", capacity),
            capacity,
            |b, &capacity| {
                b.iter_with_setup(
                    || KernelQueue::new(capacity),
                    |mut queue| {
                        let region = RegionHandle::new(1, 0);
                        let _ = queue.send(black_box(region), black_box(64), black_box(Priority::Normal));
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_kernel_queue_recv(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_queue_recv");

    group.bench_function("single", |b| {
        b.iter_with_setup(
            || {
                let mut queue = KernelQueue::new(64);
                for i in 0..64 {
                    let region = RegionHandle::new(i as u32, 0);
                    queue.send(region, 64, Priority::Normal).unwrap();
                }
                queue
            },
            |mut queue| {
                black_box(queue.recv())
            },
        );
    });

    group.bench_function("with_timeout", |b| {
        b.iter_with_setup(
            || {
                let mut queue = KernelQueue::new(64);
                for i in 0..64 {
                    let region = RegionHandle::new(i as u32, 0);
                    queue.send(region, 64, Priority::Normal).unwrap();
                }
                queue
            },
            |mut queue| {
                let timeout = Duration::from_micros(1000);
                black_box(queue.recv_timeout(timeout))
            },
        );
    });

    group.finish();
}

fn bench_kernel_queue_send_recv_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_queue_cycle");

    group.bench_function("single_cycle", |b| {
        let mut queue = KernelQueue::new(64);

        b.iter(|| {
            let region = RegionHandle::new(1, 0);
            queue.send(black_box(region), black_box(64), black_box(Priority::Normal)).unwrap();
            black_box(queue.recv())
        });
    });

    group.bench_function("high_priority", |b| {
        let mut queue = KernelQueue::new(64);

        b.iter(|| {
            let region = RegionHandle::new(1, 0);
            queue.send(black_box(region), black_box(64), black_box(Priority::High)).unwrap();
            black_box(queue.recv())
        });
    });

    group.bench_function("realtime_priority", |b| {
        let mut queue = KernelQueue::new(64);

        b.iter(|| {
            let region = RegionHandle::new(1, 0);
            queue.send(black_box(region), black_box(64), black_box(Priority::Realtime)).unwrap();
            black_box(queue.recv())
        });
    });

    group.finish();
}

// ============================================================================
// Message Descriptor Benchmarks
// ============================================================================

fn bench_message_descriptor(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_descriptor");

    group.bench_function("create", |b| {
        let region = RegionHandle::new(1, 0);
        b.iter(|| {
            black_box(MessageDescriptor::new(
                black_box(region),
                black_box(0),
                black_box(128),
            ))
        });
    });

    group.bench_function("validate", |b| {
        let region = RegionHandle::new(1, 0);
        let desc = MessageDescriptor::new(region, 0, 128);
        b.iter(|| {
            black_box(desc.validate())
        });
    });

    group.bench_function("is_empty", |b| {
        let region = RegionHandle::new(1, 0);
        let desc = MessageDescriptor::new(region, 0, 128);
        b.iter(|| {
            black_box(desc.is_empty())
        });
    });

    group.finish();
}

// ============================================================================
// Priority Benchmarks
// ============================================================================

fn bench_priority(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority");

    group.bench_function("comparison", |b| {
        let p1 = Priority::High;
        let p2 = Priority::Normal;
        b.iter(|| {
            black_box(p1 > p2)
        });
    });

    group.bench_function("as_u8", |b| {
        let p = Priority::Realtime;
        b.iter(|| {
            black_box(p.as_u8())
        });
    });

    group.finish();
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Measure raw throughput for ring buffer
    group.throughput(Throughput::Elements(1000));
    group.bench_function("ring_1000_ops", |b| {
        b.iter_with_setup(
            || RingBuffer::new(1024),
            |mut ring| {
                for i in 0..1000 {
                    let entry = RingEntry::new(i as u64, 64, Priority::Normal);
                    ring.enqueue(entry).unwrap();
                }
                for _ in 0..1000 {
                    black_box(ring.dequeue());
                }
            },
        );
    });

    // Kernel queue throughput
    group.bench_function("kernel_queue_1000_ops", |b| {
        b.iter_with_setup(
            || KernelQueue::new(1024),
            |mut queue| {
                for i in 0..1000 {
                    let region = RegionHandle::new(i as u32 % 256, 0);
                    queue.send(region, 64, Priority::Normal).unwrap();
                }
                for _ in 0..1000 {
                    black_box(queue.recv());
                }
            },
        );
    });

    group.finish();
}

// ============================================================================
// Latency Benchmarks (for ADR-087 targets)
// ============================================================================

fn bench_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");
    group.sample_size(1000);

    // Target: queue operations should be minimal overhead
    group.bench_function("ring_enqueue_latency", |b| {
        let mut ring = RingBuffer::new(64);

        b.iter(|| {
            if ring.is_full() {
                ring.dequeue();
            }
            let entry = RingEntry::new(42, 64, Priority::Normal);
            ring.enqueue(black_box(entry)).unwrap();
        });
    });

    group.bench_function("ring_dequeue_latency", |b| {
        let mut ring = RingBuffer::new(64);
        // Keep buffer half-full
        for i in 0..32 {
            let entry = RingEntry::new(i as u64, 64, Priority::Normal);
            ring.enqueue(entry).unwrap();
        }

        b.iter(|| {
            if ring.is_empty() {
                for i in 0..32 {
                    let entry = RingEntry::new(i as u64, 64, Priority::Normal);
                    ring.enqueue(entry).unwrap();
                }
            }
            black_box(ring.dequeue())
        });
    });

    group.bench_function("kernel_queue_send_latency", |b| {
        let mut queue = KernelQueue::new(64);

        b.iter(|| {
            if queue.len() >= 60 {
                for _ in 0..30 {
                    queue.recv();
                }
            }
            let region = RegionHandle::new(1, 0);
            queue.send(black_box(region), black_box(64), black_box(Priority::Normal)).unwrap();
        });
    });

    group.bench_function("descriptor_create_latency", |b| {
        let region = RegionHandle::new(1, 0);
        b.iter(|| {
            black_box(MessageDescriptor::new(
                black_box(region),
                black_box(0),
                black_box(128),
            ))
        });
    });

    group.finish();
}

// ============================================================================
// Memory Pattern Benchmarks
// ============================================================================

fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Sequential access pattern
    group.bench_function("sequential_enqueue_dequeue", |b| {
        b.iter_with_setup(
            || RingBuffer::new(256),
            |mut ring| {
                // Sequential enqueue
                for i in 0..256 {
                    let entry = RingEntry::new(i as u64, 64, Priority::Normal);
                    ring.enqueue(entry).unwrap();
                }
                // Sequential dequeue
                for _ in 0..256 {
                    black_box(ring.dequeue());
                }
            },
        );
    });

    // Interleaved pattern (more realistic)
    group.bench_function("interleaved_ops", |b| {
        b.iter_with_setup(
            || RingBuffer::new(256),
            |mut ring| {
                for i in 0..256 {
                    // Enqueue 2, dequeue 1
                    let entry1 = RingEntry::new(i as u64 * 2, 64, Priority::Normal);
                    let entry2 = RingEntry::new(i as u64 * 2 + 1, 64, Priority::Normal);
                    let _ = ring.enqueue(entry1);
                    let _ = ring.enqueue(entry2);
                    black_box(ring.dequeue());
                }
            },
        );
    });

    // Priority mixing
    group.bench_function("mixed_priority", |b| {
        let priorities = [
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Realtime,
        ];

        b.iter_with_setup(
            || KernelQueue::new(256),
            |mut queue| {
                for i in 0..256 {
                    let region = RegionHandle::new(i as u32, 0);
                    let priority = priorities[i % 4];
                    queue.send(region, 64, priority).unwrap();
                }
                for _ in 0..256 {
                    black_box(queue.recv());
                }
            },
        );
    });

    group.finish();
}

// ============================================================================
// Duration Benchmarks
// ============================================================================

fn bench_duration(c: &mut Criterion) {
    let mut group = c.benchmark_group("duration");

    group.bench_function("from_nanos", |b| {
        b.iter(|| {
            black_box(Duration::from_nanos(black_box(1_000_000)))
        });
    });

    group.bench_function("from_micros", |b| {
        b.iter(|| {
            black_box(Duration::from_micros(black_box(1_000)))
        });
    });

    group.bench_function("from_millis", |b| {
        b.iter(|| {
            black_box(Duration::from_millis(black_box(100)))
        });
    });

    group.bench_function("as_nanos", |b| {
        let duration = Duration::from_millis(100);
        b.iter(|| {
            black_box(duration.as_nanos())
        });
    });

    group.bench_function("comparison", |b| {
        let d1 = Duration::from_millis(100);
        let d2 = Duration::from_millis(200);
        b.iter(|| {
            black_box(d1 < d2)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ring_enqueue,
    bench_ring_dequeue,
    bench_ring_enqueue_dequeue_cycle,
    bench_ring_entry,
    bench_kernel_queue_send,
    bench_kernel_queue_recv,
    bench_kernel_queue_send_recv_cycle,
    bench_message_descriptor,
    bench_priority,
    bench_throughput,
    bench_latency,
    bench_memory_patterns,
    bench_duration,
);

criterion_main!(benches);
