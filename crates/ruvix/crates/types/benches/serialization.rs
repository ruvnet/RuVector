//! Benchmarks for serialization operations in ruvix-types.
//!
//! These benchmarks measure the serialization and deserialization overhead
//! for kernel types when crossing the kernel/userspace boundary.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvix_types::*;

/// Benchmark raw byte conversion for handles
fn bench_handle_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("handle_serialization");

    group.bench_function("Handle_to_bytes", |b| {
        let h = Handle::new(12345, 67890);
        b.iter(|| {
            let raw = black_box(&h).to_raw();
            raw.to_le_bytes()
        })
    });

    group.bench_function("Handle_from_bytes", |b| {
        let bytes = Handle::new(12345, 67890).to_raw().to_le_bytes();
        b.iter(|| {
            let raw = u64::from_le_bytes(black_box(bytes));
            Handle::from_raw(raw)
        })
    });

    // Throughput test for batch handle serialization
    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_to_bytes", count),
            count,
            |b, &count| {
                let handles: Vec<Handle> = (0..count)
                    .map(|i| Handle::new(i as u32, (i * 2) as u32))
                    .collect();

                b.iter(|| {
                    let mut buf = Vec::with_capacity(count * 8);
                    for h in black_box(&handles) {
                        buf.extend_from_slice(&h.to_raw().to_le_bytes());
                    }
                    buf
                })
            },
        );
    }

    group.finish();
}

/// Benchmark CapRights bit manipulation (used in syscall boundary)
fn bench_cap_rights_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cap_rights_serialization");

    group.bench_function("CapRights_to_bits", |b| {
        let rights = CapRights::READ | CapRights::WRITE | CapRights::PROVE;
        b.iter(|| black_box(&rights).bits())
    });

    group.bench_function("CapRights_from_bits", |b| {
        let bits = (CapRights::READ | CapRights::WRITE | CapRights::PROVE).bits();
        b.iter(|| CapRights::from_bits(black_box(bits)))
    });

    group.bench_function("CapRights_validation", |b| {
        let rights = CapRights::READ | CapRights::WRITE | CapRights::PROVE;
        let required = CapRights::READ | CapRights::PROVE;
        b.iter(|| black_box(&rights).contains(black_box(required)))
    });

    group.finish();
}

/// Benchmark Capability struct serialization
fn bench_capability_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability_serialization");

    // Manual serialization (no serde, as this is no_std)
    group.bench_function("Capability_to_bytes_manual", |b| {
        let cap = Capability::new(
            0x123456789ABCDEF0,
            ObjectType::VectorStore,
            CapRights::ALL,
            0xFEDCBA9876543210,
            0x0102030405060708,
        );

        b.iter(|| {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&black_box(&cap).object_id.to_le_bytes());
            buf[8] = black_box(&cap).object_type as u8;
            buf[9..13].copy_from_slice(&black_box(&cap).rights.bits().to_le_bytes());
            buf[13..21].copy_from_slice(&black_box(&cap).badge.to_le_bytes());
            buf[21..29].copy_from_slice(&black_box(&cap).epoch.to_le_bytes());
            buf
        })
    });

    group.bench_function("Capability_from_bytes_manual", |b| {
        let mut buf = [0u8; 32];
        let cap = Capability::new(
            0x123456789ABCDEF0,
            ObjectType::VectorStore,
            CapRights::ALL,
            0xFEDCBA9876543210,
            0x0102030405060708,
        );
        buf[0..8].copy_from_slice(&cap.object_id.to_le_bytes());
        buf[8] = cap.object_type as u8;
        buf[9..13].copy_from_slice(&cap.rights.bits().to_le_bytes());
        buf[13..21].copy_from_slice(&cap.badge.to_le_bytes());
        buf[21..29].copy_from_slice(&cap.epoch.to_le_bytes());

        b.iter(|| {
            let object_id = u64::from_le_bytes(black_box(buf)[0..8].try_into().unwrap());
            let object_type = ObjectType::from_u8(black_box(buf)[8]).unwrap();
            let rights = CapRights::from_bits(u32::from_le_bytes(
                black_box(buf)[9..13].try_into().unwrap(),
            ));
            let badge = u64::from_le_bytes(black_box(buf)[13..21].try_into().unwrap());
            let epoch = u64::from_le_bytes(black_box(buf)[21..29].try_into().unwrap());
            Capability::new(object_id, object_type, rights, badge, epoch)
        })
    });

    group.finish();
}

/// Benchmark RegionPolicy serialization
fn bench_region_policy_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("region_policy_serialization");

    group.bench_function("RegionPolicy_immutable_encode", |b| {
        let policy = RegionPolicy::immutable();
        b.iter(|| {
            match black_box(&policy) {
                RegionPolicy::Immutable => 0u8,
                RegionPolicy::AppendOnly { .. } => 1u8,
                RegionPolicy::Slab { .. } => 2u8,
            }
        })
    });

    group.bench_function("RegionPolicy_slab_encode", |b| {
        let policy = RegionPolicy::slab(64, 1000);
        b.iter(|| {
            let mut buf = [0u8; 17];
            match black_box(&policy) {
                RegionPolicy::Immutable => buf[0] = 0,
                RegionPolicy::AppendOnly { max_size } => {
                    buf[0] = 1;
                    buf[1..9].copy_from_slice(&max_size.to_le_bytes());
                }
                RegionPolicy::Slab {
                    slot_size,
                    slot_count,
                } => {
                    buf[0] = 2;
                    buf[1..9].copy_from_slice(&slot_size.to_le_bytes());
                    buf[9..17].copy_from_slice(&slot_count.to_le_bytes());
                }
            }
            buf
        })
    });

    group.finish();
}

/// Benchmark TaskPriority encoding (syscall parameter)
fn bench_task_priority_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_priority_serialization");

    group.bench_function("TaskPriority_encode", |b| {
        let p = TaskPriority::RealTime;
        b.iter(|| black_box(p) as u8)
    });

    group.bench_function("TaskPriority_decode", |b| {
        let byte = TaskPriority::RealTime as u8;
        b.iter(|| TaskPriority::from_u8(black_box(byte)))
    });

    group.bench_function("TaskPriority_weight_lookup", |b| {
        let p = TaskPriority::RealTime;
        b.iter(|| black_box(&p).weight())
    });

    group.finish();
}

/// Benchmark ObjectType encoding
fn bench_object_type_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("object_type_serialization");

    group.bench_function("ObjectType_all_types_encode", |b| {
        let types = [
            ObjectType::Task,
            ObjectType::Region,
            ObjectType::Queue,
            ObjectType::Timer,
            ObjectType::VectorStore,
            ObjectType::GraphStore,
            ObjectType::RvfMount,
            ObjectType::Capability,
            ObjectType::WitnessLog,
            ObjectType::Subscription,
        ];

        b.iter(|| {
            let mut buf = [0u8; 10];
            for (i, ot) in black_box(&types).iter().enumerate() {
                buf[i] = *ot as u8;
            }
            buf
        })
    });

    group.bench_function("ObjectType_all_types_decode", |b| {
        let bytes: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        b.iter(|| {
            let mut result = [None; 10];
            for (i, byte) in black_box(&bytes).iter().enumerate() {
                result[i] = ObjectType::from_u8(*byte);
            }
            result
        })
    });

    group.finish();
}

/// Benchmark syscall parameter packing (simulating real kernel interface)
fn bench_syscall_parameter_packing(c: &mut Criterion) {
    let mut group = c.benchmark_group("syscall_parameter_packing");

    // Simulating cap_grant syscall parameters
    group.bench_function("cap_grant_params", |b| {
        let target = TaskHandle::new(1, 2);
        let cap = CapHandle::new(3, 4);
        let rights = CapRights::READ | CapRights::WRITE;

        b.iter(|| {
            let mut buf = [0u8; 20];
            buf[0..8].copy_from_slice(&black_box(&target).raw().to_raw().to_le_bytes());
            buf[8..16].copy_from_slice(&black_box(&cap).raw().to_raw().to_le_bytes());
            buf[16..20].copy_from_slice(&black_box(&rights).bits().to_le_bytes());
            buf
        })
    });

    // Simulating region_map syscall parameters
    group.bench_function("region_map_params", |b| {
        let size = 4096usize;
        let policy = RegionPolicy::slab(64, 1000);
        let cap = CapHandle::new(5, 6);

        b.iter(|| {
            let mut buf = [0u8; 33];
            buf[0..8].copy_from_slice(&black_box(size).to_le_bytes());
            // Policy discriminant + data
            match black_box(&policy) {
                RegionPolicy::Immutable => buf[8] = 0,
                RegionPolicy::AppendOnly { max_size } => {
                    buf[8] = 1;
                    buf[9..17].copy_from_slice(&max_size.to_le_bytes());
                }
                RegionPolicy::Slab {
                    slot_size,
                    slot_count,
                } => {
                    buf[8] = 2;
                    buf[9..17].copy_from_slice(&slot_size.to_le_bytes());
                    buf[17..25].copy_from_slice(&slot_count.to_le_bytes());
                }
            }
            buf[25..33].copy_from_slice(&black_box(&cap).raw().to_raw().to_le_bytes());
            buf
        })
    });

    group.finish();
}

/// Benchmark memory alignment checks
fn bench_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_alignment");

    group.bench_function("Handle_alignment", |b| {
        b.iter(|| std::mem::align_of::<Handle>())
    });

    group.bench_function("Capability_alignment", |b| {
        b.iter(|| std::mem::align_of::<Capability>())
    });

    group.bench_function("RegionPolicy_alignment", |b| {
        b.iter(|| std::mem::align_of::<RegionPolicy>())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_handle_serialization,
    bench_cap_rights_serialization,
    bench_capability_serialization,
    bench_region_policy_serialization,
    bench_task_priority_serialization,
    bench_object_type_serialization,
    bench_syscall_parameter_packing,
    bench_alignment,
);

criterion_main!(benches);
