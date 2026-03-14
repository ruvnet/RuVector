//! Benchmarks for ruvix-cap operations.
//!
//! Measures performance of capability management, grant/revoke operations,
//! and derivation tree traversal.
//! Target performance from ADR-087: capability lookup should be O(1).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvix_cap::{
    CapManagerConfig, CapabilityManager, CapabilityTable, DerivationNode,
    CapHandle, CapRights, Capability, ObjectType, TaskHandle,
};

// ============================================================================
// Capability Table Benchmarks
// ============================================================================

fn bench_cap_table_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("cap_table_insert");

    for capacity in [64, 256, 1024].iter() {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("single", capacity),
            capacity,
            |b, &capacity| {
                b.iter_with_setup(
                    || CapabilityTable::new(capacity),
                    |mut table| {
                        let cap = Capability::new(
                            0x1000,
                            ObjectType::Region,
                            CapRights::READ | CapRights::WRITE,
                            0,
                            0,
                        );
                        black_box(table.insert(cap))
                    },
                );
            },
        );
    }

    // Benchmark insert when table is nearly full
    group.bench_function("near_full", |b| {
        b.iter_with_setup(
            || {
                let mut table = CapabilityTable::new(256);
                // Fill to 250/256
                for i in 0..250 {
                    let cap = Capability::new(
                        i as u64,
                        ObjectType::Region,
                        CapRights::READ,
                        0,
                        0,
                    );
                    table.insert(cap).unwrap();
                }
                table
            },
            |mut table| {
                let cap = Capability::new(
                    0xFFFF,
                    ObjectType::Region,
                    CapRights::READ,
                    0,
                    0,
                );
                black_box(table.insert(cap))
            },
        );
    });

    group.finish();
}

fn bench_cap_table_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("cap_table_lookup");

    for fill_ratio in [0.25, 0.50, 0.75, 0.90].iter() {
        let capacity = 256;
        let fill_count = (capacity as f64 * fill_ratio) as usize;

        group.bench_with_input(
            BenchmarkId::new("fill_ratio", format!("{:.0}%", fill_ratio * 100.0)),
            &fill_count,
            |b, &fill_count| {
                let mut table = CapabilityTable::new(capacity);
                let mut handles = Vec::new();

                for i in 0..fill_count {
                    let cap = Capability::new(
                        i as u64,
                        ObjectType::Region,
                        CapRights::READ,
                        0,
                        0,
                    );
                    if let Ok(handle) = table.insert(cap) {
                        handles.push(handle);
                    }
                }

                let lookup_handle = handles[fill_count / 2];

                b.iter(|| {
                    black_box(table.get(black_box(lookup_handle)))
                });
            },
        );
    }

    // O(1) lookup verification - should be constant time regardless of table size
    group.bench_function("first_entry", |b| {
        let mut table = CapabilityTable::new(1024);
        let cap = Capability::new(0x1000, ObjectType::Region, CapRights::READ, 0, 0);
        let handle = table.insert(cap).unwrap();

        // Fill rest of table
        for i in 1..1000 {
            let cap = Capability::new(i as u64, ObjectType::Region, CapRights::READ, 0, 0);
            let _ = table.insert(cap);
        }

        b.iter(|| black_box(table.get(black_box(handle))));
    });

    group.bench_function("last_entry", |b| {
        let mut table = CapabilityTable::new(1024);

        // Fill table
        for i in 0..999 {
            let cap = Capability::new(i as u64, ObjectType::Region, CapRights::READ, 0, 0);
            let _ = table.insert(cap);
        }

        // Insert last entry
        let cap = Capability::new(0xFFFF, ObjectType::Region, CapRights::READ, 0, 0);
        let handle = table.insert(cap).unwrap();

        b.iter(|| black_box(table.get(black_box(handle))));
    });

    group.finish();
}

fn bench_cap_table_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("cap_table_remove");

    group.bench_function("single", |b| {
        b.iter_with_setup(
            || {
                let mut table = CapabilityTable::new(256);
                let mut handles = Vec::new();
                for i in 0..256 {
                    let cap = Capability::new(
                        i as u64,
                        ObjectType::Region,
                        CapRights::READ,
                        0,
                        0,
                    );
                    if let Ok(h) = table.insert(cap) {
                        handles.push(h);
                    }
                }
                (table, handles)
            },
            |(mut table, handles)| {
                let handle = handles[128]; // Middle entry
                black_box(table.remove(handle))
            },
        );
    });

    group.finish();
}

// ============================================================================
// Capability Manager Benchmarks
// ============================================================================

fn bench_manager_create_root(c: &mut Criterion) {
    let mut group = c.benchmark_group("manager_create_root");

    group.bench_function("single", |b| {
        b.iter_with_setup(
            || {
                let config = CapManagerConfig::default();
                CapabilityManager::new(config)
            },
            |mut manager| {
                let task = TaskHandle::new(1, 0);
                black_box(manager.create_root_capability(
                    black_box(0x1000),
                    black_box(ObjectType::VectorStore),
                    black_box(0),
                    black_box(task),
                ))
            },
        );
    });

    for object_type in [
        ObjectType::Region,
        ObjectType::Queue,
        ObjectType::Timer,
        ObjectType::VectorStore,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::new("object_type", format!("{:?}", object_type)),
            object_type,
            |b, &obj_type| {
                b.iter_with_setup(
                    || {
                        let config = CapManagerConfig::default();
                        CapabilityManager::new(config)
                    },
                    |mut manager| {
                        let task = TaskHandle::new(1, 0);
                        black_box(manager.create_root_capability(
                            0x1000,
                            obj_type,
                            0,
                            task,
                        ))
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_manager_grant(c: &mut Criterion) {
    let mut group = c.benchmark_group("manager_grant");

    group.bench_function("single_grant", |b| {
        b.iter_with_setup(
            || {
                let config = CapManagerConfig::default();
                let mut manager = CapabilityManager::new(config);
                let task1 = TaskHandle::new(1, 0);
                let root_cap = manager
                    .create_root_capability(0x1000, ObjectType::Region, 0, task1)
                    .unwrap();
                (manager, root_cap, task1)
            },
            |(mut manager, root_cap, task1)| {
                let task2 = TaskHandle::new(2, 0);
                black_box(manager.grant(
                    root_cap,
                    CapRights::READ,
                    42,
                    task1,
                    task2,
                ))
            },
        );
    });

    // Grant with different rights
    for rights in [
        CapRights::READ,
        CapRights::WRITE,
        CapRights::READ | CapRights::WRITE,
        CapRights::READ | CapRights::WRITE | CapRights::GRANT,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::new("rights", format!("{:?}", rights)),
            rights,
            |b, &rights| {
                b.iter_with_setup(
                    || {
                        let config = CapManagerConfig::default();
                        let mut manager = CapabilityManager::new(config);
                        let task1 = TaskHandle::new(1, 0);
                        let all_rights = CapRights::READ
                            | CapRights::WRITE
                            | CapRights::GRANT
                            | CapRights::EXECUTE;
                        let root_cap = manager
                            .create_root_capability(0x1000, ObjectType::Region, 0, task1)
                            .unwrap();
                        (manager, root_cap, task1)
                    },
                    |(mut manager, root_cap, task1)| {
                        let task2 = TaskHandle::new(2, 0);
                        black_box(manager.grant(root_cap, rights, 42, task1, task2))
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_manager_revoke(c: &mut Criterion) {
    let mut group = c.benchmark_group("manager_revoke");

    // Revoke single capability (no derivations)
    group.bench_function("single_no_derivations", |b| {
        b.iter_with_setup(
            || {
                let config = CapManagerConfig::default();
                let mut manager = CapabilityManager::new(config);
                let task = TaskHandle::new(1, 0);
                let cap = manager
                    .create_root_capability(0x1000, ObjectType::Region, 0, task)
                    .unwrap();
                (manager, cap, task)
            },
            |(mut manager, cap, task)| {
                black_box(manager.revoke(cap, task))
            },
        );
    });

    // Revoke with derivation chain
    for chain_length in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("chain_length", chain_length),
            chain_length,
            |b, &chain_length| {
                b.iter_with_setup(
                    || {
                        let config = CapManagerConfig::default();
                        let mut manager = CapabilityManager::new(config);
                        let task1 = TaskHandle::new(1, 0);
                        let root_cap = manager
                            .create_root_capability(0x1000, ObjectType::Region, 0, task1)
                            .unwrap();

                        // Create derivation chain
                        let mut current_cap = root_cap;
                        let mut current_task = task1;
                        for i in 0..chain_length {
                            let next_task = TaskHandle::new(i as u32 + 2, 0);
                            if let Ok(derived) =
                                manager.grant(current_cap, CapRights::READ, i as u64, current_task, next_task)
                            {
                                current_cap = derived;
                                current_task = next_task;
                            }
                        }

                        (manager, root_cap, task1)
                    },
                    |(mut manager, root_cap, task1)| {
                        black_box(manager.revoke(root_cap, task1))
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_manager_has_rights(c: &mut Criterion) {
    let mut group = c.benchmark_group("manager_has_rights");

    group.bench_function("check_read", |b| {
        let config = CapManagerConfig::default();
        let mut manager = CapabilityManager::new(config);
        let task = TaskHandle::new(1, 0);
        let cap = manager
            .create_root_capability(0x1000, ObjectType::Region, 0, task)
            .unwrap();

        b.iter(|| {
            black_box(manager.has_rights(cap, CapRights::READ))
        });
    });

    group.bench_function("check_multiple_rights", |b| {
        let config = CapManagerConfig::default();
        let mut manager = CapabilityManager::new(config);
        let task = TaskHandle::new(1, 0);
        let cap = manager
            .create_root_capability(0x1000, ObjectType::Region, 0, task)
            .unwrap();

        let required_rights = CapRights::READ | CapRights::WRITE | CapRights::EXECUTE;

        b.iter(|| {
            black_box(manager.has_rights(cap, required_rights))
        });
    });

    group.finish();
}

// ============================================================================
// Derivation Tree Benchmarks
// ============================================================================

fn bench_derivation_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivation_tree");

    group.bench_function("create_node", |b| {
        b.iter(|| {
            black_box(DerivationNode::new(
                black_box(CapHandle::new(1, 0)),
                black_box(None),
                black_box(0),
            ))
        });
    });

    group.bench_function("create_derived_node", |b| {
        let parent = CapHandle::new(0, 0);
        b.iter(|| {
            black_box(DerivationNode::new(
                black_box(CapHandle::new(1, 0)),
                black_box(Some(parent)),
                black_box(1),
            ))
        });
    });

    group.finish();
}

// ============================================================================
// Rights Bitmap Benchmarks
// ============================================================================

fn bench_rights_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("rights_ops");

    group.bench_function("combine_rights", |b| {
        let r1 = CapRights::READ;
        let r2 = CapRights::WRITE;
        b.iter(|| {
            black_box(r1 | r2)
        });
    });

    group.bench_function("intersect_rights", |b| {
        let r1 = CapRights::READ | CapRights::WRITE | CapRights::GRANT;
        let r2 = CapRights::READ | CapRights::EXECUTE;
        b.iter(|| {
            black_box(r1 & r2)
        });
    });

    group.bench_function("contains_check", |b| {
        let rights = CapRights::READ | CapRights::WRITE | CapRights::GRANT;
        let required = CapRights::READ | CapRights::WRITE;
        b.iter(|| {
            black_box(rights.contains(required))
        });
    });

    group.bench_function("is_subset", |b| {
        let held = CapRights::READ | CapRights::WRITE | CapRights::GRANT;
        let requested = CapRights::READ | CapRights::WRITE;
        b.iter(|| {
            // Check if requested rights are subset of held rights
            black_box((held & requested) == requested)
        });
    });

    group.finish();
}

// ============================================================================
// Capability Creation Benchmarks
// ============================================================================

fn bench_capability_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability_create");

    group.bench_function("new", |b| {
        b.iter(|| {
            black_box(Capability::new(
                black_box(0x1000),
                black_box(ObjectType::Region),
                black_box(CapRights::READ | CapRights::WRITE),
                black_box(42),
                black_box(0),
            ))
        });
    });

    group.bench_function("with_all_rights", |b| {
        let all_rights = CapRights::READ
            | CapRights::WRITE
            | CapRights::EXECUTE
            | CapRights::GRANT
            | CapRights::REVOKE
            | CapRights::DERIVE;

        b.iter(|| {
            black_box(Capability::new(
                black_box(0x1000),
                black_box(ObjectType::VectorStore),
                black_box(all_rights),
                black_box(0xDEADBEEF),
                black_box(1),
            ))
        });
    });

    group.finish();
}

// ============================================================================
// Handle Benchmarks
// ============================================================================

fn bench_handle_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("handle_ops");

    group.bench_function("cap_handle_new", |b| {
        b.iter(|| {
            black_box(CapHandle::new(black_box(42), black_box(1)))
        });
    });

    group.bench_function("task_handle_new", |b| {
        b.iter(|| {
            black_box(TaskHandle::new(black_box(1), black_box(0)))
        });
    });

    group.bench_function("handle_comparison", |b| {
        let h1 = CapHandle::new(1, 0);
        let h2 = CapHandle::new(1, 0);
        b.iter(|| {
            black_box(h1 == h2)
        });
    });

    group.bench_function("handle_generation_check", |b| {
        let h1 = CapHandle::new(1, 0);
        let h2 = CapHandle::new(1, 1); // Different generation
        b.iter(|| {
            // Generation mismatch detection
            black_box(h1.generation() != h2.generation())
        });
    });

    group.finish();
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Measure capability lookup throughput
    group.throughput(Throughput::Elements(1000));
    group.bench_function("lookup_1000", |b| {
        let mut table = CapabilityTable::new(1024);
        let mut handles = Vec::new();

        for i in 0..1000 {
            let cap = Capability::new(
                i as u64,
                ObjectType::Region,
                CapRights::READ,
                0,
                0,
            );
            if let Ok(h) = table.insert(cap) {
                handles.push(h);
            }
        }

        b.iter(|| {
            for &handle in &handles {
                black_box(table.get(handle));
            }
        });
    });

    // Grant/revoke cycle throughput
    group.bench_function("grant_revoke_cycle", |b| {
        b.iter_with_setup(
            || {
                let config = CapManagerConfig::default();
                let mut manager = CapabilityManager::new(config);
                let task1 = TaskHandle::new(1, 0);
                let root_cap = manager
                    .create_root_capability(0x1000, ObjectType::Region, 0, task1)
                    .unwrap();
                (manager, root_cap, task1)
            },
            |(mut manager, root_cap, task1)| {
                let task2 = TaskHandle::new(2, 0);
                let derived = manager
                    .grant(root_cap, CapRights::READ, 42, task1, task2)
                    .unwrap();
                manager.revoke(derived, task2).unwrap();
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

    // Target: capability lookup should be O(1) constant time
    group.bench_function("cap_lookup_latency", |b| {
        let mut table = CapabilityTable::new(1024);
        let cap = Capability::new(0x1000, ObjectType::Region, CapRights::READ, 0, 0);
        let handle = table.insert(cap).unwrap();

        // Fill table to test under load
        for i in 1..1000 {
            let cap = Capability::new(i as u64, ObjectType::Region, CapRights::READ, 0, 0);
            let _ = table.insert(cap);
        }

        b.iter(|| black_box(table.get(black_box(handle))));
    });

    group.bench_function("rights_check_latency", |b| {
        let config = CapManagerConfig::default();
        let mut manager = CapabilityManager::new(config);
        let task = TaskHandle::new(1, 0);
        let cap = manager
            .create_root_capability(0x1000, ObjectType::Region, 0, task)
            .unwrap();

        b.iter(|| {
            black_box(manager.has_rights(black_box(cap), black_box(CapRights::READ)))
        });
    });

    group.bench_function("grant_latency", |b| {
        b.iter_with_setup(
            || {
                let config = CapManagerConfig::default();
                let mut manager = CapabilityManager::new(config);
                let task1 = TaskHandle::new(1, 0);
                let cap = manager
                    .create_root_capability(0x1000, ObjectType::Region, 0, task1)
                    .unwrap();
                (manager, cap, task1)
            },
            |(mut manager, cap, task1)| {
                let task2 = TaskHandle::new(2, 0);
                black_box(manager.grant(cap, CapRights::READ, 42, task1, task2))
            },
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cap_table_insert,
    bench_cap_table_lookup,
    bench_cap_table_remove,
    bench_manager_create_root,
    bench_manager_grant,
    bench_manager_revoke,
    bench_manager_has_rights,
    bench_derivation_tree,
    bench_rights_operations,
    bench_capability_creation,
    bench_handle_operations,
    bench_throughput,
    bench_latency,
);

criterion_main!(benches);
