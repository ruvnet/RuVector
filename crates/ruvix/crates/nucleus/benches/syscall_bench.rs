//! Benchmarks for RuVix Cognition Kernel syscalls.
//!
//! These benchmarks measure the latency of each syscall to verify
//! compliance with ADR-087 Section 3.2 invariant 3: bounded latency.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, VectorStoreConfig, TaskPriority, ProofTier,
    VectorKey, RegionPolicy, MsgPriority, TimerSpec, SensorDescriptor, QueueHandle,
    GraphMutation, CapHandle, CapRights, RvfMountHandle, RvfComponentId, Duration,
};
use ruvix_types::{TaskHandle, ObjectType};

fn setup_kernel() -> Kernel {
    let mut kernel = Kernel::new(KernelConfig::default());
    kernel.boot(0, [0u8; 32]).expect("Boot failed");
    kernel.set_current_time(1_000_000);
    kernel
}

// ============================================================================
// Syscall Benchmarks
// ============================================================================

fn bench_task_spawn(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    c.bench_function("syscall_task_spawn", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::TaskSpawn {
                entry: RvfComponentId::root(RvfMountHandle::null()),
                caps: Vec::new(),
                priority: TaskPriority::Normal,
                deadline: None,
            }))
        })
    });
}

fn bench_cap_grant(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let root_task = TaskHandle::new(1, 0);
    let cap = kernel.create_root_capability(0, ObjectType::RvfMount, root_task).unwrap();

    c.bench_function("syscall_cap_grant", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::CapGrant {
                target: TaskHandle::new(2, 0),
                cap,
                rights: CapRights::READ,
            }))
        })
    });
}

fn bench_region_map(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let root_task = TaskHandle::new(1, 0);
    let cap = kernel.create_root_capability(0, ObjectType::Region, root_task).unwrap();

    c.bench_function("syscall_region_map", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::RegionMap {
                size: 4096,
                policy: RegionPolicy::Immutable,
                cap,
            }))
        })
    });
}

fn bench_queue_send(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    c.bench_function("syscall_queue_send", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::QueueSend {
                queue: QueueHandle::new(1, 0),
                msg: vec![1, 2, 3, 4],
                priority: MsgPriority::Normal,
            }))
        })
    });
}

fn bench_queue_recv(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    c.bench_function("syscall_queue_recv", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::QueueRecv {
                queue: QueueHandle::new(1, 0),
                buf_size: 4096,
                timeout: Duration::from_millis(0),
            }))
        })
    });
}

fn bench_timer_wait(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    c.bench_function("syscall_timer_wait", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::TimerWait {
                deadline: TimerSpec::from_millis(0),
            }))
        })
    });
}

fn bench_rvf_mount(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let root_task = TaskHandle::new(1, 0);
    let cap = kernel.create_root_capability(0, ObjectType::RvfMount, root_task).unwrap();

    c.bench_function("syscall_rvf_mount", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::RvfMount {
                rvf_data: vec![0u8; 32],
                mount_point: "/test".to_string(),
                cap,
            }))
        })
    });
}

fn bench_vector_get(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    // Setup: create store and insert data
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof,
    }).unwrap();

    c.bench_function("syscall_vector_get", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::VectorGet {
                store,
                key: VectorKey::new(1),
            }))
        })
    });
}

fn bench_vector_put_proved(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    let mut nonce = 0u64;

    c.bench_function("syscall_vector_put_proved", |b| {
        b.iter(|| {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

            kernel.dispatch(black_box(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100) as u64),
                data: vec![1.0, 2.0, 3.0, 4.0],
                proof,
            }))
        })
    });
}

fn bench_graph_apply_proved(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let graph = kernel.create_graph_store().unwrap();

    let mut nonce = 0u64;

    c.bench_function("syscall_graph_apply_proved", |b| {
        b.iter(|| {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, nonce);

            kernel.dispatch(black_box(Syscall::GraphApplyProved {
                graph,
                mutation: GraphMutation::add_node(nonce),
                proof,
            }))
        })
    });
}

fn bench_sensor_subscribe(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let root_task = TaskHandle::new(1, 0);
    let cap = kernel.create_root_capability(0, ObjectType::RvfMount, root_task).unwrap();

    c.bench_function("syscall_sensor_subscribe", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::SensorSubscribe {
                sensor: SensorDescriptor::default(),
                target_queue: QueueHandle::new(1, 0),
                cap,
            }))
        })
    });
}

// ============================================================================
// Proof Tier Comparison
// ============================================================================

fn bench_proof_tiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_verification_by_tier");

    for tier in [ProofTier::Reflex, ProofTier::Standard, ProofTier::Deep] {
        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(4, 100);
        let store = kernel.create_vector_store(config).unwrap();

        let mut nonce = 0u64;

        group.bench_with_input(
            BenchmarkId::new("tier", format!("{:?}", tier)),
            &tier,
            |b, &tier| {
                b.iter(|| {
                    nonce += 1;
                    let mutation_hash = [nonce as u8; 32];
                    let proof = kernel.create_proof(mutation_hash, tier, nonce);

                    kernel.dispatch(black_box(Syscall::VectorPutProved {
                        store,
                        key: VectorKey::new((nonce % 100) as u64),
                        data: vec![1.0, 2.0, 3.0, 4.0],
                        proof,
                    }))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Vector Dimension Scaling
// ============================================================================

fn bench_vector_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_dimensions");

    for dims in [4, 64, 256, 768] {
        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(dims, 100);
        let store = kernel.create_vector_store(config).unwrap();

        let mut nonce = 0u64;
        let data: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();

        group.bench_with_input(
            BenchmarkId::new("dimensions", dims),
            &dims,
            |b, _| {
                b.iter(|| {
                    nonce += 1;
                    let mutation_hash = [nonce as u8; 32];
                    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

                    kernel.dispatch(black_box(Syscall::VectorPutProved {
                        store,
                        key: VectorKey::new((nonce % 100) as u64),
                        data: data.clone(),
                        proof,
                    }))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Checkpoint Operations
// ============================================================================

fn bench_checkpoint(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    // Add some data
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    for i in 0..10 {
        let mutation_hash = [i as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, i + 1);

        kernel.dispatch(Syscall::VectorPutProved {
            store,
            key: VectorKey::new(i as u64),
            data: vec![i as f32; 4],
            proof,
        }).unwrap();
    }

    c.bench_function("checkpoint_creation", |b| {
        b.iter(|| {
            kernel.checkpoint(black_box(ruvix_nucleus::CheckpointConfig::full()))
        })
    });
}

fn bench_checkpoint_verify(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    // Add some data
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    for i in 0..10 {
        let mutation_hash = [i as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, i + 1);

        kernel.dispatch(Syscall::VectorPutProved {
            store,
            key: VectorKey::new(i as u64),
            data: vec![i as f32; 4],
            proof,
        }).unwrap();
    }

    let checkpoint = kernel.checkpoint(ruvix_nucleus::CheckpointConfig::full()).unwrap();

    c.bench_function("checkpoint_verification", |b| {
        b.iter(|| {
            kernel.verify_checkpoint(black_box(&checkpoint))
        })
    });
}

// ============================================================================
// Witness Log Operations
// ============================================================================

fn bench_witness_log_serialization(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    // Add mutations to generate witness records
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    for i in 0..100 {
        let mutation_hash = [i as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, i + 1);

        kernel.dispatch(Syscall::VectorPutProved {
            store,
            key: VectorKey::new(i as u64),
            data: vec![i as f32; 4],
            proof,
        }).unwrap();
    }

    c.bench_function("witness_log_serialize", |b| {
        b.iter(|| {
            kernel.witness_log().to_bytes()
        })
    });
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    syscall_benches,
    bench_task_spawn,
    bench_cap_grant,
    bench_region_map,
    bench_queue_send,
    bench_queue_recv,
    bench_timer_wait,
    bench_rvf_mount,
    bench_vector_get,
    bench_vector_put_proved,
    bench_graph_apply_proved,
    bench_sensor_subscribe,
);

criterion_group!(
    scaling_benches,
    bench_proof_tiers,
    bench_vector_dimensions,
);

criterion_group!(
    checkpoint_benches,
    bench_checkpoint,
    bench_checkpoint_verify,
    bench_witness_log_serialization,
);

criterion_main!(syscall_benches, scaling_benches, checkpoint_benches);
