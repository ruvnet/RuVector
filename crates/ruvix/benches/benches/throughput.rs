//! Criterion benchmarks for throughput analysis.
//!
//! Measures operations per second for various workloads.
//!
//! Run with: cargo bench --bench throughput

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, VectorStoreConfig, ProofTier,
    VectorKey, MsgPriority, QueueHandle, GraphMutation,
};
use std::time::{Duration, Instant};

fn setup_kernel() -> Kernel {
    let mut kernel = Kernel::new(KernelConfig::default());
    kernel.boot(0, [0u8; 32]).expect("Boot failed");
    kernel.set_current_time(1_000_000);
    kernel
}

// ============================================================================
// IPC Throughput
// ============================================================================

fn bench_ipc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_throughput");

    // Message sizes
    for msg_size in [8, 64, 256, 1024, 4096] {
        let mut kernel = setup_kernel();
        let msg: Vec<u8> = (0..msg_size).map(|i| i as u8).collect();

        group.throughput(Throughput::Bytes(msg_size as u64));

        group.bench_with_input(
            BenchmarkId::new("ruvix_queue", msg_size),
            &msg_size,
            |b, _| {
                b.iter(|| {
                    kernel.dispatch(black_box(Syscall::QueueSend {
                        queue: QueueHandle::new(1, 0),
                        msg: msg.clone(),
                        priority: MsgPriority::Normal,
                    }))
                })
            },
        );
    }

    group.finish();
}

#[cfg(unix)]
fn bench_linux_ipc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("linux_ipc_throughput");

    for msg_size in [8, 64, 256, 1024, 4096] {
        // Create pipe
        let mut fds: [libc::c_int; 2] = [0; 2];
        unsafe { libc::pipe(fds.as_mut_ptr()); }

        let write_fd = fds[1];
        let read_fd = fds[0];
        let msg: Vec<u8> = (0..msg_size).map(|i| i as u8).collect();

        let reader = std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            loop {
                let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
                if n <= 0 { break; }
            }
        });

        group.throughput(Throughput::Bytes(msg_size as u64));

        group.bench_with_input(
            BenchmarkId::new("linux_pipe", msg_size),
            &msg_size,
            |b, _| {
                b.iter(|| {
                    black_box(unsafe {
                        libc::write(write_fd, msg.as_ptr() as *const libc::c_void, msg.len())
                    })
                })
            },
        );

        unsafe { libc::close(write_fd); }
        let _ = reader.join();
    }

    group.finish();
}

// ============================================================================
// Vector Store Throughput
// ============================================================================

fn bench_vector_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_throughput");

    for dims in [4, 64, 256, 768] {
        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(dims, 10000);
        let store = kernel.create_vector_store(config).unwrap();
        let data: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();
        let mut nonce = 0u64;

        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("put", dims),
            &dims,
            |b, _| {
                b.iter(|| {
                    nonce += 1;
                    let mutation_hash = [nonce as u8; 32];
                    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

                    kernel.dispatch(black_box(Syscall::VectorPutProved {
                        store,
                        key: VectorKey::new((nonce % 10000) as u64),
                        data: data.clone(),
                        proof,
                    }))
                })
            },
        );
    }

    group.finish();
}

fn bench_vector_get_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_get_throughput");

    for dims in [4, 64, 256, 768] {
        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(dims, 1000);
        let store = kernel.create_vector_store(config).unwrap();
        let data: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();

        // Pre-populate
        for i in 0..100 {
            let mutation_hash = [i as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, i);
            kernel.dispatch(Syscall::VectorPutProved {
                store,
                key: VectorKey::new(i),
                data: data.clone(),
                proof,
            }).unwrap();
        }

        let mut key_idx = 0u64;

        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("get", dims),
            &dims,
            |b, _| {
                b.iter(|| {
                    key_idx = (key_idx + 1) % 100;
                    kernel.dispatch(black_box(Syscall::VectorGet {
                        store,
                        key: VectorKey::new(key_idx),
                    }))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Graph Store Throughput
// ============================================================================

fn bench_graph_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_throughput");

    let mut kernel = setup_kernel();
    let graph = kernel.create_graph_store().unwrap();
    let mut nonce = 0u64;

    group.throughput(Throughput::Elements(1));

    group.bench_function("add_node", |b| {
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

    group.finish();
}

// ============================================================================
// Pipeline Throughput (Perception Pipeline Simulation)
// ============================================================================

fn bench_perception_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("perception_pipeline");

    // Simulate: sensor -> queue -> vector_put -> graph_apply
    let mut kernel = setup_kernel();
    let vector_config = VectorStoreConfig::new(4, 10000);
    let vector_store = kernel.create_vector_store(vector_config).unwrap();
    let graph = kernel.create_graph_store().unwrap();
    let mut nonce = 0u64;

    group.throughput(Throughput::Elements(1));

    group.bench_function("ruvix_full_pipeline", |b| {
        b.iter(|| {
            nonce += 1;

            // Step 1: Queue send (sensor event)
            kernel.dispatch(black_box(Syscall::QueueSend {
                queue: QueueHandle::new(1, 0),
                msg: vec![1, 2, 3, 4],
                priority: MsgPriority::Normal,
            })).ok();

            // Step 2: Vector put (embedding)
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);
            kernel.dispatch(black_box(Syscall::VectorPutProved {
                store: vector_store,
                key: VectorKey::new((nonce % 10000) as u64),
                data: vec![1.0, 2.0, 3.0, 4.0],
                proof,
            })).ok();

            // Step 3: Graph apply (knowledge graph update)
            let graph_hash = [(nonce + 1) as u8; 32];
            let graph_proof = kernel.create_proof(graph_hash, ProofTier::Standard, nonce + 1);
            kernel.dispatch(black_box(Syscall::GraphApplyProved {
                graph,
                mutation: GraphMutation::add_node(nonce),
                proof: graph_proof,
            })).ok();
        })
    });

    group.finish();
}

// ============================================================================
// Burst Throughput (Sustained Load)
// ============================================================================

fn bench_sustained_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("sustained_throughput");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));

    // Measure ops/sec over sustained period
    let mut kernel = setup_kernel();
    let config = VectorStoreConfig::new(4, 100000);
    let store = kernel.create_vector_store(config).unwrap();
    let mut nonce = 0u64;

    group.bench_function("vector_put_sustained", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                nonce += 1;
                let mutation_hash = [nonce as u8; 32];
                let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

                let _ = kernel.dispatch(black_box(Syscall::VectorPutProved {
                    store,
                    key: VectorKey::new((nonce % 100000) as u64),
                    data: vec![1.0, 2.0, 3.0, 4.0],
                    proof,
                }));
            }
            start.elapsed()
        })
    });

    group.finish();
}

// ============================================================================
// Comparison: RuVix vs Linux Pipeline
// ============================================================================

#[cfg(unix)]
fn bench_linux_pipeline_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_comparison");

    // RuVix pipeline
    {
        let mut kernel = setup_kernel();
        let vector_config = VectorStoreConfig::new(4, 10000);
        let vector_store = kernel.create_vector_store(vector_config).unwrap();
        let graph = kernel.create_graph_store().unwrap();
        let mut nonce = 0u64;

        group.throughput(Throughput::Elements(1));

        group.bench_function("ruvix", |b| {
            b.iter(|| {
                nonce += 1;

                // Queue + Vector + Graph
                kernel.dispatch(black_box(Syscall::QueueSend {
                    queue: QueueHandle::new(1, 0),
                    msg: vec![1, 2, 3, 4],
                    priority: MsgPriority::Normal,
                })).ok();

                let mutation_hash = [nonce as u8; 32];
                let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);
                kernel.dispatch(black_box(Syscall::VectorPutProved {
                    store: vector_store,
                    key: VectorKey::new((nonce % 10000) as u64),
                    data: vec![1.0, 2.0, 3.0, 4.0],
                    proof,
                })).ok();

                let graph_hash = [(nonce + 1) as u8; 32];
                let graph_proof = kernel.create_proof(graph_hash, ProofTier::Standard, nonce + 1);
                kernel.dispatch(black_box(Syscall::GraphApplyProved {
                    graph,
                    mutation: GraphMutation::add_node(nonce),
                    proof: graph_proof,
                })).ok();
            })
        });
    }

    // Linux simulation (pipe + file + sync)
    {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut fds: [libc::c_int; 2] = [0; 2];
        unsafe { libc::pipe(fds.as_mut_ptr()); }

        let write_fd = fds[1];
        let read_fd = fds[0];
        let msg = [1u8, 2, 3, 4];

        let reader = std::thread::spawn(move || {
            let mut buf = [0u8; 1024];
            loop {
                let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
                if n <= 0 { break; }
            }
        });

        let temp_path = std::env::temp_dir().join(format!("ruvix_pipeline_{}", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)
            .expect("Failed to create temp file");

        group.bench_function("linux", |b| {
            b.iter(|| {
                // Pipe write (IPC)
                unsafe { libc::write(write_fd, msg.as_ptr() as *const libc::c_void, msg.len()) };

                // File write + sync (durability)
                file.write_all(&[1, 2, 3, 4]).ok();
                file.sync_all().ok();
            })
        });

        unsafe { libc::close(write_fd); }
        let _ = reader.join();
        std::fs::remove_file(&temp_path).ok();
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    ipc_benches,
    bench_ipc_throughput,
);

criterion_group!(
    vector_benches,
    bench_vector_throughput,
    bench_vector_get_throughput,
);

criterion_group!(
    graph_benches,
    bench_graph_throughput,
);

criterion_group!(
    pipeline_benches,
    bench_perception_pipeline,
    bench_sustained_throughput,
);

#[cfg(unix)]
criterion_group!(
    linux_ipc,
    bench_linux_ipc_throughput,
);

#[cfg(unix)]
criterion_group!(
    linux_pipeline,
    bench_linux_pipeline_simulation,
);

#[cfg(unix)]
criterion_main!(ipc_benches, linux_ipc, vector_benches, graph_benches, pipeline_benches, linux_pipeline);

#[cfg(not(unix))]
criterion_main!(ipc_benches, vector_benches, graph_benches, pipeline_benches);
