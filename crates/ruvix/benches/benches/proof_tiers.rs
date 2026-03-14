//! Criterion benchmarks for proof tier overhead analysis.
//!
//! Compares proof verification overhead across Reflex, Standard, and Deep tiers.
//!
//! Run with: cargo bench --bench proof_tiers

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, VectorStoreConfig, ProofTier,
    VectorKey, GraphMutation,
};

fn setup_kernel() -> Kernel {
    let mut kernel = Kernel::new(KernelConfig::default());
    kernel.boot(0, [0u8; 32]).expect("Boot failed");
    kernel.set_current_time(1_000_000);
    kernel
}

// ============================================================================
// Proof Tier Benchmarks
// ============================================================================

fn bench_reflex_tier(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();
    let mut nonce = 0u64;

    c.bench_function("proof/reflex/vector_put", |b| {
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

fn bench_standard_tier(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let graph = kernel.create_graph_store().unwrap();
    let mut nonce = 0u64;

    c.bench_function("proof/standard/graph_apply", |b| {
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

fn bench_deep_tier(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();
    let mut nonce = 0u64;

    c.bench_function("proof/deep/vector_put", |b| {
        b.iter(|| {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Deep, nonce);

            kernel.dispatch(black_box(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100) as u64),
                data: vec![1.0, 2.0, 3.0, 4.0],
                proof,
            }))
        })
    });
}

// ============================================================================
// Tier Comparison with Vector Operations
// ============================================================================

fn bench_proof_tiers_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_tier_vector");
    group.throughput(Throughput::Elements(1));

    for tier in [ProofTier::Reflex, ProofTier::Standard, ProofTier::Deep] {
        let tier_name = match tier {
            ProofTier::Reflex => "Reflex",
            ProofTier::Standard => "Standard",
            ProofTier::Deep => "Deep",
        };

        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(4, 100);
        let store = kernel.create_vector_store(config).unwrap();
        let mut nonce = 0u64;

        group.bench_with_input(
            BenchmarkId::new("put", tier_name),
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
// Tier Comparison with Graph Operations
// ============================================================================

fn bench_proof_tiers_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_tier_graph");
    group.throughput(Throughput::Elements(1));

    for tier in [ProofTier::Reflex, ProofTier::Standard, ProofTier::Deep] {
        let tier_name = match tier {
            ProofTier::Reflex => "Reflex",
            ProofTier::Standard => "Standard",
            ProofTier::Deep => "Deep",
        };

        let mut kernel = setup_kernel();
        let graph = kernel.create_graph_store().unwrap();
        let mut nonce = 0u64;

        group.bench_with_input(
            BenchmarkId::new("add_node", tier_name),
            &tier,
            |b, &tier| {
                b.iter(|| {
                    nonce += 1;
                    let mutation_hash = [nonce as u8; 32];
                    let proof = kernel.create_proof(mutation_hash, tier, nonce);

                    kernel.dispatch(black_box(Syscall::GraphApplyProved {
                        graph,
                        mutation: GraphMutation::add_node(nonce),
                        proof,
                    }))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Proof Creation Overhead
// ============================================================================

fn bench_proof_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_creation");

    for tier in [ProofTier::Reflex, ProofTier::Standard, ProofTier::Deep] {
        let tier_name = match tier {
            ProofTier::Reflex => "Reflex",
            ProofTier::Standard => "Standard",
            ProofTier::Deep => "Deep",
        };

        let kernel = setup_kernel();
        let mut nonce = 0u64;

        group.bench_with_input(
            BenchmarkId::new("create", tier_name),
            &tier,
            |b, &tier| {
                b.iter(|| {
                    nonce += 1;
                    let mutation_hash = [nonce as u8; 32];
                    black_box(kernel.create_proof(mutation_hash, tier, nonce))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Vector Size Impact on Proof Overhead
// ============================================================================

fn bench_proof_with_vector_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_vector_size");

    for dims in [4, 64, 256, 768] {
        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(dims, 100);
        let store = kernel.create_vector_store(config).unwrap();
        let data: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();
        let mut nonce = 0u64;

        // Use Reflex tier for consistent comparison
        group.throughput(Throughput::Bytes((dims * 4) as u64));

        group.bench_with_input(
            BenchmarkId::new("reflex", dims),
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
// Comparison with Linux Security Overhead
// ============================================================================

#[cfg(unix)]
fn bench_linux_security_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("security_overhead_comparison");

    // RuVix Reflex proof
    let mut kernel = setup_kernel();
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();
    let mut nonce = 0u64;

    group.bench_function("ruvix_reflex", |b| {
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

    // Linux SELinux simulation (10 policy checks)
    group.bench_function("linux_selinux_10", |b| {
        b.iter(|| {
            for _ in 0..10 {
                black_box(unsafe { libc::getuid() });
            }
        })
    });

    // Linux seccomp simulation (single check)
    group.bench_function("linux_seccomp_1", |b| {
        b.iter(|| {
            black_box(unsafe { libc::getuid() })
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    tier_benches,
    bench_reflex_tier,
    bench_standard_tier,
    bench_deep_tier,
);

criterion_group!(
    comparison_benches,
    bench_proof_tiers_vector,
    bench_proof_tiers_graph,
    bench_proof_creation,
    bench_proof_with_vector_sizes,
);

#[cfg(unix)]
criterion_group!(
    linux_comparison,
    bench_linux_security_overhead,
);

#[cfg(unix)]
criterion_main!(tier_benches, comparison_benches, linux_comparison);

#[cfg(not(unix))]
criterion_main!(tier_benches, comparison_benches);
