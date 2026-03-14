//! Benchmarks for the buddy allocator.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvix_physmem::{BuddyAllocator, PhysAddr};

const BASE_ADDR: u64 = 0x1000_0000;

fn bench_single_page_alloc_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_page");
    group.throughput(Throughput::Elements(1));

    group.bench_function("alloc_free", |b| {
        let mut allocator = BuddyAllocator::new(PhysAddr::new(BASE_ADDR), 1024);

        b.iter(|| {
            let addr = allocator.alloc_pages(1).unwrap();
            allocator.dealloc_pages(addr, 1);
            black_box(addr)
        });
    });

    group.finish();
}

fn bench_multi_page_alloc_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_page");

    for pages in [2, 4, 8, 16, 32, 64, 128, 256, 512] {
        group.throughput(Throughput::Elements(pages as u64));

        group.bench_with_input(BenchmarkId::new("alloc_free", pages), &pages, |b, &pages| {
            let mut allocator = BuddyAllocator::new(PhysAddr::new(BASE_ADDR), 4096);

            b.iter(|| {
                let addr = allocator.alloc_pages(pages).unwrap();
                allocator.dealloc_pages(addr, pages);
                black_box(addr)
            });
        });
    }

    group.finish();
}

fn bench_sequential_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential");

    for count in [10, 50, 100, 200] {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("alloc_then_free", count), &count, |b, &count| {
            b.iter(|| {
                let mut allocator = BuddyAllocator::new(PhysAddr::new(BASE_ADDR), 4096);
                let mut addrs = [PhysAddr::NULL; 200];

                for addr in addrs.iter_mut().take(count) {
                    *addr = allocator.alloc_pages(1).unwrap();
                }

                for addr in addrs.iter().take(count) {
                    allocator.dealloc_pages(*addr, 1);
                }

                black_box(&addrs)
            });
        });
    }

    group.finish();
}

fn bench_fragmentation_pattern(c: &mut Criterion) {
    c.bench_function("fragmentation", |b| {
        b.iter(|| {
            let mut allocator = BuddyAllocator::new(PhysAddr::new(BASE_ADDR), 1024);
            let mut addrs = [PhysAddr::NULL; 64];

            // Allocate all
            for addr in &mut addrs {
                *addr = allocator.alloc_pages(1).unwrap();
            }

            // Free every other one (creates fragmentation)
            for (i, addr) in addrs.iter().enumerate() {
                if i % 2 == 0 {
                    allocator.dealloc_pages(*addr, 1);
                }
            }

            // Allocate again
            for (i, addr) in addrs.iter_mut().enumerate() {
                if i % 2 == 0 {
                    *addr = allocator.alloc_pages(1).unwrap();
                }
            }

            // Free all
            for addr in &addrs {
                allocator.dealloc_pages(*addr, 1);
            }

            black_box(&addrs)
        });
    });
}

fn bench_mixed_sizes(c: &mut Criterion) {
    c.bench_function("mixed_sizes", |b| {
        b.iter(|| {
            let mut allocator = BuddyAllocator::new(PhysAddr::new(BASE_ADDR), 4096);

            let a1 = allocator.alloc_pages(1).unwrap();
            let a2 = allocator.alloc_pages(4).unwrap();
            let a3 = allocator.alloc_pages(16).unwrap();
            let a4 = allocator.alloc_pages(64).unwrap();
            let a5 = allocator.alloc_pages(8).unwrap();

            allocator.dealloc_pages(a3, 16);
            allocator.dealloc_pages(a1, 1);
            allocator.dealloc_pages(a5, 8);
            allocator.dealloc_pages(a2, 4);
            allocator.dealloc_pages(a4, 64);

            black_box((a1, a2, a3, a4, a5))
        });
    });
}

fn bench_reset(c: &mut Criterion) {
    c.bench_function("reset", |b| {
        let mut allocator = BuddyAllocator::new(PhysAddr::new(BASE_ADDR), 4096);

        // Fragment memory
        for _ in 0..100 {
            let _ = allocator.alloc_pages(1);
        }

        b.iter(|| {
            allocator.reset();
            black_box(allocator.free_page_count())
        });
    });
}

criterion_group!(
    benches,
    bench_single_page_alloc_free,
    bench_multi_page_alloc_free,
    bench_sequential_alloc,
    bench_fragmentation_pattern,
    bench_mixed_sizes,
    bench_reset,
);

criterion_main!(benches);
