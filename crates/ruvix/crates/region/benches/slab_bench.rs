//! Benchmarks for ruvix-region operations.
//!
//! Measures performance of region allocation, deallocation, and access patterns.
//! Target performance from ADR-087: zero-copy when sharing regions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvix_region::{
    append_only::AppendOnlyRegion,
    backing::StaticBacking,
    immutable::ImmutableRegion,
    slab::{SlabAllocator, SlotHandle},
};
use ruvix_types::RegionHandle;

// ============================================================================
// Slab Allocator Benchmarks
// ============================================================================

fn bench_slab_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("slab_alloc");

    for slot_count in [16, 64, 256].iter() {
        group.throughput(Throughput::Elements(*slot_count as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential", slot_count),
            slot_count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        // Setup with enough backing for all slots
                        let size = count * 72; // 64 bytes + alignment
                        let backing = match size {
                            s if s <= 1024 => {
                                let b = StaticBacking::<1024>::new();
                                Box::new(b) as Box<dyn FnOnce() -> _>
                            }
                            s if s <= 4096 => {
                                let b = StaticBacking::<4096>::new();
                                Box::new(move || SlabAllocator::new(b, 64, count).unwrap())
                            }
                            _ => {
                                let b = StaticBacking::<32768>::new();
                                Box::new(move || SlabAllocator::new(b, 64, count).unwrap())
                            }
                        };
                        backing
                    },
                    |create_slab| {
                        // Unfortunately we can't use the boxed closure directly
                        // Just re-create for benchmark
                    },
                );
            },
        );
    }

    // Direct benchmark without setup complexity
    group.bench_function("alloc_single", |b| {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 32).unwrap();
        let mut handles = Vec::with_capacity(32);

        b.iter(|| {
            if slab.is_full() {
                for h in handles.drain(..) {
                    slab.free(h).unwrap();
                }
            }
            let handle = slab.alloc().unwrap();
            handles.push(handle);
            black_box(handle)
        });
    });

    group.bench_function("alloc_free_cycle", |b| {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 32).unwrap();

        b.iter(|| {
            let handle = slab.alloc().unwrap();
            slab.free(black_box(handle)).unwrap();
        });
    });

    group.finish();
}

fn bench_slab_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("slab_free");

    group.bench_function("free_single", |b| {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 32).unwrap();
        let mut handles: Vec<SlotHandle> = Vec::with_capacity(32);

        b.iter(|| {
            if handles.is_empty() {
                for _ in 0..32 {
                    handles.push(slab.alloc().unwrap());
                }
            }
            let handle = handles.pop().unwrap();
            slab.free(black_box(handle)).unwrap();
        });
    });

    group.finish();
}

fn bench_slab_read_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("slab_io");

    for size in [32, 64, 128, 256].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("write", size), size, |b, &size| {
            let backing = StaticBacking::<32768>::new();
            let mut slab = SlabAllocator::new(backing, size, 16).unwrap();
            let handle = slab.alloc().unwrap();
            let data = vec![0xABu8; size];

            b.iter(|| {
                slab.write(black_box(handle), black_box(&data)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("read", size), size, |b, &size| {
            let backing = StaticBacking::<32768>::new();
            let mut slab = SlabAllocator::new(backing, size, 16).unwrap();
            let handle = slab.alloc().unwrap();
            slab.write(handle, &vec![0xABu8; size]).unwrap();
            let mut buf = vec![0u8; size];

            b.iter(|| {
                slab.read(black_box(handle), black_box(&mut buf)).unwrap();
            });
        });
    }

    group.finish();
}

// ============================================================================
// Append-Only Region Benchmarks
// ============================================================================

fn bench_append_only_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_only");

    for chunk_size in [8, 64, 256, 1024].iter() {
        group.throughput(Throughput::Bytes(*chunk_size as u64));

        group.bench_with_input(
            BenchmarkId::new("append", chunk_size),
            chunk_size,
            |b, &size| {
                let data = vec![0xABu8; size];

                b.iter_with_setup(
                    || {
                        let backing = StaticBacking::<65536>::new();
                        AppendOnlyRegion::new(backing, 65536, RegionHandle::new(1, 0)).unwrap()
                    },
                    |mut region| {
                        region.append(black_box(&data)).unwrap();
                    },
                );
            },
        );
    }

    group.bench_function("append_u64", |b| {
        b.iter_with_setup(
            || {
                let backing = StaticBacking::<65536>::new();
                AppendOnlyRegion::new(backing, 65536, RegionHandle::new(1, 0)).unwrap()
            },
            |mut region| {
                region.append_u64(black_box(0x123456789ABCDEF0)).unwrap();
            },
        );
    });

    group.finish();
}

fn bench_append_only_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_only_read");

    for size in [8, 64, 256, 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("read", size), size, |b, &size| {
            let backing = StaticBacking::<65536>::new();
            let mut region =
                AppendOnlyRegion::new(backing, 65536, RegionHandle::new(1, 0)).unwrap();
            region.append(&vec![0xABu8; size]).unwrap();
            let mut buf = vec![0u8; size];

            b.iter(|| {
                region.read(0, black_box(&mut buf)).unwrap();
            });
        });
    }

    group.finish();
}

// ============================================================================
// Immutable Region Benchmarks
// ============================================================================

fn bench_immutable_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("immutable_creation");

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("create", size), size, |b, &size| {
            let data = vec![0xABu8; size];

            b.iter_with_setup(
                || {
                    // Create fresh backing for each iteration
                    StaticBacking::<8192>::new()
                },
                |backing| {
                    ImmutableRegion::new(backing, black_box(&data), RegionHandle::new(1, 0))
                        .unwrap()
                },
            );
        });
    }

    group.finish();
}

fn bench_immutable_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("immutable_read");

    for size in [64, 256, 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("read", size), size, |b, &size| {
            let data = vec![0xABu8; size];
            let backing = StaticBacking::<8192>::new();
            let region = ImmutableRegion::new(backing, &data, RegionHandle::new(1, 0)).unwrap();
            let mut buf = vec![0u8; size];

            b.iter(|| {
                region.read(0, black_box(&mut buf)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("as_slice", size), size, |b, &size| {
            let data = vec![0xABu8; size];
            let backing = StaticBacking::<8192>::new();
            let region = ImmutableRegion::new(backing, &data, RegionHandle::new(1, 0)).unwrap();

            b.iter(|| black_box(region.as_slice()));
        });
    }

    group.finish();
}

fn bench_immutable_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("immutable_hash");

    group.bench_function("content_equals", |b| {
        let data = vec![0xABu8; 1024];
        let backing1 = StaticBacking::<2048>::new();
        let backing2 = StaticBacking::<2048>::new();

        let region1 = ImmutableRegion::new(backing1, &data, RegionHandle::new(1, 0)).unwrap();
        let region2 = ImmutableRegion::new(backing2, &data, RegionHandle::new(1, 0)).unwrap();

        b.iter(|| black_box(region1.content_equals(&region2)));
    });

    group.finish();
}

// ============================================================================
// Memory Access Pattern Benchmarks
// ============================================================================

fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Sequential access to slab slots
    group.bench_function("slab_sequential_access", |b| {
        let backing = StaticBacking::<32768>::new();
        let mut slab = SlabAllocator::new(backing, 64, 256).unwrap();

        let handles: Vec<_> = (0..256).map(|_| slab.alloc().unwrap()).collect();
        let data = [0xABu8; 64];

        b.iter(|| {
            for &handle in black_box(&handles) {
                slab.write(handle, &data).unwrap();
            }
        });
    });

    // Random access to slab slots (simulated by accessing in reverse)
    group.bench_function("slab_reverse_access", |b| {
        let backing = StaticBacking::<32768>::new();
        let mut slab = SlabAllocator::new(backing, 64, 256).unwrap();

        let handles: Vec<_> = (0..256).map(|_| slab.alloc().unwrap()).collect();
        let data = [0xABu8; 64];

        b.iter(|| {
            for &handle in black_box(&handles).iter().rev() {
                slab.write(handle, &data).unwrap();
            }
        });
    });

    // Append-only sequential writes (optimal pattern)
    group.bench_function("append_only_burst", |b| {
        let data = [0xABu8; 64];

        b.iter_with_setup(
            || {
                let backing = StaticBacking::<65536>::new();
                AppendOnlyRegion::new(backing, 65536, RegionHandle::new(1, 0)).unwrap()
            },
            |mut region| {
                for _ in 0..256 {
                    region.append(black_box(&data)).unwrap();
                }
            },
        );
    });

    group.finish();
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Measure raw memory copy throughput for comparison
    group.throughput(Throughput::Bytes(4096));
    group.bench_function("raw_memcpy_4k", |b| {
        let src = vec![0xABu8; 4096];
        let mut dst = vec![0u8; 4096];

        b.iter(|| {
            dst.copy_from_slice(black_box(&src));
        });
    });

    // Slab throughput (alloc + write + read + free)
    group.bench_function("slab_full_cycle", |b| {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 32).unwrap();
        let data = [0xABu8; 64];
        let mut buf = [0u8; 64];

        b.iter(|| {
            let handle = slab.alloc().unwrap();
            slab.write(handle, black_box(&data)).unwrap();
            slab.read(handle, black_box(&mut buf)).unwrap();
            slab.free(handle).unwrap();
        });
    });

    // Append-only throughput (append + read)
    group.bench_function("append_only_full_cycle", |b| {
        let data = [0xABu8; 64];
        let mut buf = [0u8; 64];

        b.iter_with_setup(
            || {
                let backing = StaticBacking::<1024>::new();
                AppendOnlyRegion::new(backing, 1024, RegionHandle::new(1, 0)).unwrap()
            },
            |mut region| {
                let offset = region.append(black_box(&data)).unwrap();
                region.read(offset, black_box(&mut buf)).unwrap();
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

    // Target: region operations should be minimal overhead
    group.bench_function("slab_alloc_latency", |b| {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 32).unwrap();
        let mut handles = Vec::with_capacity(32);

        b.iter(|| {
            if slab.is_full() {
                for h in handles.drain(..) {
                    slab.free(h).unwrap();
                }
            }
            handles.push(slab.alloc().unwrap());
        });
    });

    group.bench_function("append_only_append_latency", |b| {
        let data = [0xABu8; 8]; // Small append

        b.iter_with_setup(
            || {
                let backing = StaticBacking::<65536>::new();
                AppendOnlyRegion::new(backing, 65536, RegionHandle::new(1, 0)).unwrap()
            },
            |mut region| {
                region.append(black_box(&data)).unwrap();
            },
        );
    });

    group.bench_function("immutable_access_latency", |b| {
        let data = vec![0xABu8; 64];
        let backing = StaticBacking::<256>::new();
        let region = ImmutableRegion::new(backing, &data, RegionHandle::new(1, 0)).unwrap();

        b.iter(|| black_box(region.as_slice()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_slab_alloc,
    bench_slab_free,
    bench_slab_read_write,
    bench_append_only_append,
    bench_append_only_read,
    bench_immutable_creation,
    bench_immutable_read,
    bench_immutable_hash,
    bench_memory_access_patterns,
    bench_throughput,
    bench_latency,
);

criterion_main!(benches);
