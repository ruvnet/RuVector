//! Benchmarks for type construction in ruvix-types.
//!
//! These benchmarks measure the construction and manipulation of kernel types
//! to ensure they meet the performance targets from ADR-087:
//! - Reflex tier proofs: ~100ns
//! - Type construction: sub-microsecond

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvix_types::*;

/// Benchmark Handle construction
fn bench_handle_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("handle_construction");

    group.bench_function("Handle::new", |b| {
        b.iter(|| Handle::new(black_box(42), black_box(7)))
    });

    group.bench_function("Handle::null", |b| {
        b.iter(|| Handle::null())
    });

    group.bench_function("Handle::to_raw", |b| {
        let h = Handle::new(12345, 67890);
        b.iter(|| black_box(&h).to_raw())
    });

    group.bench_function("Handle::from_raw", |b| {
        let raw = Handle::new(12345, 67890).to_raw();
        b.iter(|| Handle::from_raw(black_box(raw)))
    });

    group.finish();
}

/// Benchmark CapRights operations
fn bench_cap_rights(c: &mut Criterion) {
    let mut group = c.benchmark_group("cap_rights");

    group.bench_function("CapRights::contains", |b| {
        let rights = CapRights::READ | CapRights::WRITE | CapRights::GRANT;
        b.iter(|| black_box(&rights).contains(black_box(CapRights::READ)))
    });

    group.bench_function("CapRights::union", |b| {
        let a = CapRights::READ;
        let b_rights = CapRights::WRITE;
        b.iter(|| black_box(a) | black_box(b_rights))
    });

    group.bench_function("CapRights::intersection", |b| {
        let a = CapRights::READ | CapRights::WRITE;
        let b_rights = CapRights::WRITE | CapRights::GRANT;
        b.iter(|| black_box(a) & black_box(b_rights))
    });

    group.bench_function("CapRights::is_subset_of", |b| {
        let subset = CapRights::READ;
        let superset = CapRights::READ | CapRights::WRITE | CapRights::GRANT;
        b.iter(|| black_box(&subset).is_subset_of(black_box(superset)))
    });

    group.finish();
}

/// Benchmark Capability operations
fn bench_capability(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability");

    group.bench_function("Capability::new", |b| {
        b.iter(|| {
            Capability::new(
                black_box(1),
                black_box(ObjectType::VectorStore),
                black_box(CapRights::ALL),
                black_box(0),
                black_box(1),
            )
        })
    });

    group.bench_function("Capability::has_rights", |b| {
        let cap = Capability::new(1, ObjectType::VectorStore, CapRights::ALL, 0, 1);
        b.iter(|| black_box(&cap).has_rights(black_box(CapRights::PROVE)))
    });

    group.bench_function("Capability::derive", |b| {
        let cap = Capability::new(
            1,
            ObjectType::VectorStore,
            CapRights::READ | CapRights::WRITE | CapRights::GRANT,
            0,
            1,
        );
        b.iter(|| black_box(&cap).derive(black_box(CapRights::READ), black_box(42)))
    });

    group.finish();
}

/// Benchmark RegionPolicy operations
fn bench_region_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("region_policy");

    group.bench_function("RegionPolicy::immutable", |b| {
        b.iter(|| RegionPolicy::immutable())
    });

    group.bench_function("RegionPolicy::append_only", |b| {
        b.iter(|| RegionPolicy::append_only(black_box(4096)))
    });

    group.bench_function("RegionPolicy::slab", |b| {
        b.iter(|| RegionPolicy::slab(black_box(64), black_box(1000)))
    });

    group.bench_function("RegionPolicy::capacity", |b| {
        let policy = RegionPolicy::slab(64, 1000);
        b.iter(|| black_box(&policy).capacity())
    });

    group.bench_function("RegionPolicy::is_writable", |b| {
        let policy = RegionPolicy::slab(64, 1000);
        b.iter(|| black_box(&policy).is_writable())
    });

    group.finish();
}

/// Benchmark ObjectType operations
fn bench_object_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("object_type");

    group.bench_function("ObjectType::from_u8", |b| {
        b.iter(|| ObjectType::from_u8(black_box(4)))
    });

    group.bench_function("ObjectType::as_str", |b| {
        let ot = ObjectType::VectorStore;
        b.iter(|| black_box(&ot).as_str())
    });

    group.finish();
}

/// Benchmark TaskPriority operations
fn bench_task_priority(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_priority");

    group.bench_function("TaskPriority::weight", |b| {
        let p = TaskPriority::RealTime;
        b.iter(|| black_box(&p).weight())
    });

    group.bench_function("TaskPriority::from_u8", |b| {
        b.iter(|| TaskPriority::from_u8(black_box(3)))
    });

    group.bench_function("TaskPriority::comparison", |b| {
        let a = TaskPriority::Normal;
        let b_priority = TaskPriority::High;
        b.iter(|| black_box(&a) < black_box(&b_priority))
    });

    group.finish();
}

/// Benchmark handle type constructions
fn bench_typed_handles(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed_handles");

    group.bench_function("TaskHandle::new", |b| {
        b.iter(|| TaskHandle::new(black_box(1), black_box(2)))
    });

    group.bench_function("RegionHandle::new", |b| {
        b.iter(|| RegionHandle::new(black_box(1), black_box(2)))
    });

    group.bench_function("CapHandle::new", |b| {
        b.iter(|| CapHandle::new(black_box(1), black_box(2)))
    });

    group.bench_function("Handle::is_null", |b| {
        let h = Handle::new(42, 7);
        b.iter(|| black_box(&h).is_null())
    });

    group.finish();
}

/// Benchmark batch capability derivation (simulating proof-gated operations)
fn bench_capability_derivation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability_derivation_throughput");
    group.throughput(Throughput::Elements(1));

    for count in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            count,
            |b, &count| {
                let cap = Capability::new(
                    1,
                    ObjectType::VectorStore,
                    CapRights::ALL,
                    0,
                    1,
                );

                b.iter(|| {
                    for i in 0..count {
                        let _ = black_box(&cap).derive(CapRights::READ, i as u64);
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory layout verification
fn bench_memory_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout");

    group.bench_function("size_of_Handle", |b| {
        b.iter(|| std::mem::size_of::<Handle>())
    });

    group.bench_function("size_of_Capability", |b| {
        b.iter(|| std::mem::size_of::<Capability>())
    });

    group.bench_function("size_of_RegionPolicy", |b| {
        b.iter(|| std::mem::size_of::<RegionPolicy>())
    });

    group.bench_function("size_of_CapRights", |b| {
        b.iter(|| std::mem::size_of::<CapRights>())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_handle_construction,
    bench_cap_rights,
    bench_capability,
    bench_region_policy,
    bench_object_type,
    bench_task_priority,
    bench_typed_handles,
    bench_capability_derivation_throughput,
    bench_memory_layout,
);

criterion_main!(benches);
