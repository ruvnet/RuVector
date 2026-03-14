//! RuVix syscall benchmarks.
//!
//! This module provides benchmark functions for all 12 RuVix syscalls.

use std::time::{Duration, Instant};

use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, VectorStoreConfig, TaskPriority, ProofTier,
    VectorKey, RegionPolicy, MsgPriority, TimerSpec, SensorDescriptor, QueueHandle,
    GraphMutation, CapHandle, CapRights, RvfMountHandle, RvfComponentId,
};
use ruvix_types::{TaskHandle, ObjectType};

use crate::BenchmarkResult;

/// Sets up a kernel for benchmarking.
pub fn setup_kernel() -> Kernel {
    let mut kernel = Kernel::new(KernelConfig::default());
    kernel.boot(0, [0u8; 32]).expect("Boot failed");
    kernel.set_current_time(1_000_000);
    kernel
}

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Number of measurement iterations.
    pub measure_iterations: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measure_iterations: 10000,
        }
    }
}

/// Runs a benchmark for a given syscall function.
pub fn run_benchmark<F>(
    name: &str,
    mut setup_fn: impl FnMut() -> Kernel,
    syscall_fn: F,
    config: &BenchConfig,
) -> BenchmarkResult
where
    F: Fn(&mut Kernel) -> (),
{
    let mut kernel = setup_fn();

    // Warmup
    for _ in 0..config.warmup_iterations {
        syscall_fn(&mut kernel);
    }

    // Measurement
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        syscall_fn(&mut kernel);
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    let target = crate::target_for(name);
    BenchmarkResult::from_measurements(name, &measurements, target)
}

/// Benchmarks the `task_spawn` syscall.
pub fn bench_task_spawn(config: &BenchConfig) -> BenchmarkResult {
    run_benchmark(
        "task_spawn",
        setup_kernel,
        |kernel| {
            let _ = kernel.dispatch(Syscall::TaskSpawn {
                entry: RvfComponentId::root(RvfMountHandle::null()),
                caps: Vec::new(),
                priority: TaskPriority::Normal,
                deadline: None,
            });
        },
        config,
    )
}

/// Benchmarks the `cap_grant` syscall.
pub fn bench_cap_grant(config: &BenchConfig) -> BenchmarkResult {
    let mut kernel = setup_kernel();
    let root_task = TaskHandle::new(1, 0);
    let cap = kernel.create_root_capability(0, ObjectType::RvfMount, root_task).unwrap();

    // Need to create a kernel for the benchmark
    let cap_copy = cap;
    run_benchmark(
        "cap_grant",
        || {
            let mut k = setup_kernel();
            let rt = TaskHandle::new(1, 0);
            k.create_root_capability(0, ObjectType::RvfMount, rt).unwrap();
            k
        },
        move |kernel| {
            let _ = kernel.dispatch(Syscall::CapGrant {
                target: TaskHandle::new(2, 0),
                cap: cap_copy,
                rights: CapRights::READ,
            });
        },
        config,
    )
}

/// Benchmarks the `region_map` syscall.
pub fn bench_region_map(config: &BenchConfig) -> BenchmarkResult {
    run_benchmark(
        "region_map",
        || {
            let mut k = setup_kernel();
            let rt = TaskHandle::new(1, 0);
            k.create_root_capability(0, ObjectType::Region, rt).unwrap();
            k
        },
        |kernel| {
            let _ = kernel.dispatch(Syscall::RegionMap {
                size: 4096,
                policy: RegionPolicy::Immutable,
                cap: CapHandle::null(),
            });
        },
        config,
    )
}

/// Benchmarks the `queue_send` syscall.
pub fn bench_queue_send(config: &BenchConfig) -> BenchmarkResult {
    run_benchmark(
        "queue_send",
        setup_kernel,
        |kernel| {
            let _ = kernel.dispatch(Syscall::QueueSend {
                queue: QueueHandle::new(1, 0),
                msg: vec![1, 2, 3, 4, 5, 6, 7, 8],
                priority: MsgPriority::Normal,
            });
        },
        config,
    )
}

/// Benchmarks the `queue_recv` syscall.
pub fn bench_queue_recv(config: &BenchConfig) -> BenchmarkResult {
    run_benchmark(
        "queue_recv",
        setup_kernel,
        |kernel| {
            let _ = kernel.dispatch(Syscall::QueueRecv {
                queue: QueueHandle::new(1, 0),
                buf_size: 4096,
                timeout: Duration::from_millis(0),
            });
        },
        config,
    )
}

/// Benchmarks the `timer_wait` syscall.
pub fn bench_timer_wait(config: &BenchConfig) -> BenchmarkResult {
    run_benchmark(
        "timer_wait",
        setup_kernel,
        |kernel| {
            let _ = kernel.dispatch(Syscall::TimerWait {
                deadline: TimerSpec::from_millis(0),
            });
        },
        config,
    )
}

/// Benchmarks the `rvf_mount` syscall.
pub fn bench_rvf_mount(config: &BenchConfig) -> BenchmarkResult {
    run_benchmark(
        "rvf_mount",
        || {
            let mut k = setup_kernel();
            let rt = TaskHandle::new(1, 0);
            k.create_root_capability(0, ObjectType::RvfMount, rt).unwrap();
            k
        },
        |kernel| {
            let _ = kernel.dispatch(Syscall::RvfMount {
                rvf_data: vec![0u8; 32],
                mount_point: "/test".to_string(),
                cap: CapHandle::null(),
            });
        },
        config,
    )
}

/// Benchmarks the `vector_get` syscall.
pub fn bench_vector_get(config: &BenchConfig) -> BenchmarkResult {
    let mut kernel = setup_kernel();

    // Setup: create store and insert data
    let store_config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(store_config).unwrap();

    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof,
    }).unwrap();

    // Now benchmark get
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = kernel.dispatch(Syscall::VectorGet {
            store,
            key: VectorKey::new(1),
        });
    }

    // Measure
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        let _ = kernel.dispatch(Syscall::VectorGet {
            store,
            key: VectorKey::new(1),
        });
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    let target = crate::target_for("vector_get");
    BenchmarkResult::from_measurements("vector_get", &measurements, target)
}

/// Benchmarks the `vector_put_proved` syscall.
pub fn bench_vector_put_proved(config: &BenchConfig) -> BenchmarkResult {
    let mut kernel = setup_kernel();
    let store_config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(store_config).unwrap();

    let mut nonce = 0u64;
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    // Warmup
    for _ in 0..config.warmup_iterations {
        nonce += 1;
        let mutation_hash = [nonce as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

        let _ = kernel.dispatch(Syscall::VectorPutProved {
            store,
            key: VectorKey::new((nonce % 100) as u64),
            data: vec![1.0, 2.0, 3.0, 4.0],
            proof,
        });
    }

    // Measure
    for _ in 0..config.measure_iterations {
        nonce += 1;
        let mutation_hash = [nonce as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

        let start = Instant::now();
        let _ = kernel.dispatch(Syscall::VectorPutProved {
            store,
            key: VectorKey::new((nonce % 100) as u64),
            data: vec![1.0, 2.0, 3.0, 4.0],
            proof,
        });
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    let target = crate::target_for("vector_put_proved");
    BenchmarkResult::from_measurements("vector_put_proved", &measurements, target)
}

/// Benchmarks the `graph_apply_proved` syscall.
pub fn bench_graph_apply_proved(config: &BenchConfig) -> BenchmarkResult {
    let mut kernel = setup_kernel();
    let graph = kernel.create_graph_store().unwrap();

    let mut nonce = 0u64;
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    // Warmup
    for _ in 0..config.warmup_iterations {
        nonce += 1;
        let mutation_hash = [nonce as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, nonce);

        let _ = kernel.dispatch(Syscall::GraphApplyProved {
            graph,
            mutation: GraphMutation::add_node(nonce),
            proof,
        });
    }

    // Measure
    for _ in 0..config.measure_iterations {
        nonce += 1;
        let mutation_hash = [nonce as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, nonce);

        let start = Instant::now();
        let _ = kernel.dispatch(Syscall::GraphApplyProved {
            graph,
            mutation: GraphMutation::add_node(nonce),
            proof,
        });
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    let target = crate::target_for("graph_apply_proved");
    BenchmarkResult::from_measurements("graph_apply_proved", &measurements, target)
}

/// Benchmarks the `sensor_subscribe` syscall.
pub fn bench_sensor_subscribe(config: &BenchConfig) -> BenchmarkResult {
    run_benchmark(
        "sensor_subscribe",
        || {
            let mut k = setup_kernel();
            let rt = TaskHandle::new(1, 0);
            k.create_root_capability(0, ObjectType::RvfMount, rt).unwrap();
            k
        },
        |kernel| {
            let _ = kernel.dispatch(Syscall::SensorSubscribe {
                sensor: SensorDescriptor::default(),
                target_queue: QueueHandle::new(1, 0),
                cap: CapHandle::null(),
            });
        },
        config,
    )
}

/// Runs all RuVix syscall benchmarks.
pub fn bench_all_syscalls(config: &BenchConfig) -> Vec<BenchmarkResult> {
    vec![
        bench_task_spawn(config),
        bench_cap_grant(config),
        bench_region_map(config),
        bench_queue_send(config),
        bench_queue_recv(config),
        bench_timer_wait(config),
        bench_rvf_mount(config),
        bench_vector_get(config),
        bench_vector_put_proved(config),
        bench_graph_apply_proved(config),
        bench_sensor_subscribe(config),
        // Note: attest_emit requires special setup
    ]
}

/// Benchmarks proof verification overhead by tier.
pub fn bench_proof_tiers(config: &BenchConfig) -> Vec<(String, BenchmarkResult)> {
    let mut results = Vec::new();

    for tier in [ProofTier::Reflex, ProofTier::Standard, ProofTier::Deep] {
        let tier_name = match tier {
            ProofTier::Reflex => "Reflex",
            ProofTier::Standard => "Standard",
            ProofTier::Deep => "Deep",
        };

        let mut kernel = setup_kernel();
        let store_config = VectorStoreConfig::new(4, 100);
        let store = kernel.create_vector_store(store_config).unwrap();

        let mut nonce = 0u64;
        let mut measurements = Vec::with_capacity(config.measure_iterations);

        // Warmup
        for _ in 0..config.warmup_iterations {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, tier, nonce);

            let _ = kernel.dispatch(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100) as u64),
                data: vec![1.0, 2.0, 3.0, 4.0],
                proof,
            });
        }

        // Measure
        for _ in 0..config.measure_iterations {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, tier, nonce);

            let start = Instant::now();
            let _ = kernel.dispatch(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100) as u64),
                data: vec![1.0, 2.0, 3.0, 4.0],
                proof,
            });
            let elapsed = start.elapsed();
            measurements.push(elapsed.as_nanos() as f64);
        }

        let result = BenchmarkResult::from_measurements(
            &format!("proof_tier_{}", tier_name),
            &measurements,
            None,
        );
        results.push((tier_name.to_string(), result));
    }

    results
}

/// Benchmarks vector operations at different dimensions.
pub fn bench_vector_dimensions(config: &BenchConfig) -> Vec<(u32, BenchmarkResult)> {
    let dimensions = [4, 64, 256, 768];
    let mut results = Vec::new();

    for dims in dimensions {
        let mut kernel = setup_kernel();
        let store_config = VectorStoreConfig::new(dims, 100);
        let store = kernel.create_vector_store(store_config).unwrap();

        let data: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();
        let mut nonce = 0u64;
        let mut measurements = Vec::with_capacity(config.measure_iterations);

        // Warmup
        for _ in 0..config.warmup_iterations {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

            let _ = kernel.dispatch(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100) as u64),
                data: data.clone(),
                proof,
            });
        }

        // Measure
        for _ in 0..config.measure_iterations {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

            let start = Instant::now();
            let _ = kernel.dispatch(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100) as u64),
                data: data.clone(),
                proof,
            });
            let elapsed = start.elapsed();
            measurements.push(elapsed.as_nanos() as f64);
        }

        let result = BenchmarkResult::from_measurements(
            &format!("vector_put_{}d", dims),
            &measurements,
            None,
        );
        results.push((dims, result));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_kernel() {
        let kernel = setup_kernel();
        assert_eq!(kernel.stats().syscalls_executed, 0);
    }

    #[test]
    fn test_bench_task_spawn() {
        let config = BenchConfig {
            warmup_iterations: 10,
            measure_iterations: 100,
        };
        let result = bench_task_spawn(&config);
        assert_eq!(result.iterations, 100);
        assert!(result.mean_ns > 0.0);
    }
}
