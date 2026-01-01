//! Performance Benchmarks for edge-net
//!
//! Comprehensive benchmarking suite for all critical operations.
//! Run with: `cargo bench --features=bench`

#![cfg(all(test, feature = "bench"))]

use test::Bencher;
use super::*;

// ============================================================================
// Credit Operations Benchmarks
// ============================================================================

#[bench]
fn bench_credit_operation(b: &mut Bencher) {
    let mut ledger = credits::WasmCreditLedger::new("bench-node".to_string()).unwrap();

    b.iter(|| {
        ledger.credit(100, "task").unwrap();
    });
}

#[bench]
fn bench_deduct_operation(b: &mut Bencher) {
    let mut ledger = credits::WasmCreditLedger::new("bench-node".to_string()).unwrap();
    ledger.credit(1_000_000, "initial").unwrap();

    b.iter(|| {
        ledger.deduct(10).unwrap();
    });
}

#[bench]
fn bench_balance_calculation(b: &mut Bencher) {
    let mut ledger = credits::WasmCreditLedger::new("bench-node".to_string()).unwrap();

    // Simulate large history
    for i in 0..1000 {
        ledger.credit(100, &format!("task-{}", i)).unwrap();
    }

    b.iter(|| {
        ledger.balance()
    });
}

#[bench]
fn bench_ledger_merge(b: &mut Bencher) {
    let mut ledger1 = credits::WasmCreditLedger::new("node-1".to_string()).unwrap();
    let mut ledger2 = credits::WasmCreditLedger::new("node-2".to_string()).unwrap();

    for i in 0..100 {
        ledger2.credit(100, &format!("task-{}", i)).unwrap();
    }

    let earned = ledger2.export_earned().unwrap();
    let spent = ledger2.export_spent().unwrap();

    b.iter(|| {
        ledger1.merge(&earned, &spent).unwrap();
    });
}

// ============================================================================
// QDAG Transaction Benchmarks
// ============================================================================

#[bench]
fn bench_qdag_transaction_creation(b: &mut Bencher) {
    use ed25519_dalek::{SigningKey, VerifyingKey};
    use rand::rngs::OsRng;

    let mut ledger = credits::qdag::QDAGLedger::new();
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key: VerifyingKey = (&signing_key).into();
    let pubkey = verifying_key.to_bytes();

    // Create genesis
    ledger.create_genesis(1_000_000_000, &pubkey).unwrap();

    let sender_id = hex::encode(&pubkey);
    let privkey = signing_key.to_bytes();

    b.iter(|| {
        // Note: This will fail after first transaction due to PoW, but measures creation speed
        let _ = ledger.create_transaction(
            &sender_id,
            "recipient",
            1000,
            1, // Transfer
            &privkey,
            &pubkey,
        );
    });
}

#[bench]
fn bench_qdag_balance_query(b: &mut Bencher) {
    let ledger = credits::qdag::QDAGLedger::new();

    b.iter(|| {
        ledger.balance("test-node")
    });
}

#[bench]
fn bench_qdag_tip_selection(b: &mut Bencher) {
    use ed25519_dalek::{SigningKey, VerifyingKey};
    use rand::rngs::OsRng;

    let mut ledger = credits::qdag::QDAGLedger::new();
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key: VerifyingKey = (&signing_key).into();
    let pubkey = verifying_key.to_bytes();

    ledger.create_genesis(1_000_000_000, &pubkey).unwrap();

    b.iter(|| {
        ledger.tip_count()
    });
}

// ============================================================================
// Task Queue Performance Benchmarks
// ============================================================================

#[bench]
fn bench_task_creation(b: &mut Bencher) {
    let queue = tasks::WasmTaskQueue::new().unwrap();
    let identity = identity::WasmNodeIdentity::generate("bench").unwrap();
    let payload = vec![0u8; 1024]; // 1KB payload

    b.iter(|| {
        queue.create_task("vectors", &payload, 100, &identity).unwrap()
    });
}

#[bench]
fn bench_task_queue_operations(b: &mut Bencher) {
    use tokio::runtime::Runtime;

    let rt = Runtime::new().unwrap();
    let mut queue = tasks::WasmTaskQueue::new().unwrap();
    let identity = identity::WasmNodeIdentity::generate("bench").unwrap();

    b.iter(|| {
        rt.block_on(async {
            let payload = vec![0u8; 100];
            let task = queue.create_task("vectors", &payload, 100, &identity).unwrap();
            queue.submit(task).await.unwrap();
        });
    });
}

#[bench]
fn bench_parallel_task_processing(b: &mut Bencher) {
    use tokio::runtime::Runtime;

    let rt = Runtime::new().unwrap();

    b.iter(|| {
        rt.block_on(async {
            let mut queue = tasks::WasmTaskQueue::new().unwrap();
            let identity = identity::WasmNodeIdentity::generate("bench").unwrap();

            // Simulate 10 parallel tasks
            let mut handles = vec![];
            for _ in 0..10 {
                let payload = vec![0u8; 100];
                let task = queue.create_task("vectors", &payload, 100, &identity).unwrap();
                handles.push(queue.submit(task));
            }

            futures::future::join_all(handles).await;
        });
    });
}

// ============================================================================
// Security Operations Benchmarks
// ============================================================================

#[bench]
fn bench_qlearning_decision(b: &mut Bencher) {
    let security = security::AdaptiveSecurity::new();

    b.iter(|| {
        security.choose_action("normal_load", "allow,block,throttle")
    });
}

#[bench]
fn bench_qlearning_update(b: &mut Bencher) {
    let mut security = security::AdaptiveSecurity::new();

    b.iter(|| {
        security.learn("normal_load", "allow", 0.8, "low_attack");
    });
}

#[bench]
fn bench_attack_pattern_matching(b: &mut Bencher) {
    let mut security = security::AdaptiveSecurity::new();

    // Record some attack patterns
    for i in 0..10 {
        let features = vec![i as f32 * 0.1, 0.5, 0.3];
        security.record_attack_pattern("ddos", &features, 0.8);
    }

    let test_features = vec![0.5, 0.5, 0.3];

    b.iter(|| {
        security.detect_attack(&test_features)
    });
}

#[bench]
fn bench_threshold_updates(b: &mut Bencher) {
    let mut security = security::AdaptiveSecurity::new();

    // Generate learning history
    for i in 0..100 {
        security.learn(
            "state",
            if i % 2 == 0 { "allow" } else { "block" },
            if i % 3 == 0 { 0.8 } else { 0.2 },
            "next_state"
        );
    }

    b.iter(|| {
        security.get_rate_limit_window();
        security.get_rate_limit_max();
        security.get_spot_check_probability();
    });
}

#[bench]
fn bench_rate_limiter(b: &mut Bencher) {
    let mut limiter = security::RateLimiter::new(60_000, 100);

    b.iter(|| {
        limiter.check_allowed("test-node")
    });
}

#[bench]
fn bench_reputation_update(b: &mut Bencher) {
    let mut reputation = security::ReputationSystem::new();

    b.iter(|| {
        reputation.record_success("test-node");
    });
}

// ============================================================================
// Network Topology Benchmarks
// ============================================================================

#[bench]
fn bench_node_registration_1k(b: &mut Bencher) {
    b.iter(|| {
        let mut topology = evolution::NetworkTopology::new();
        for i in 0..1_000 {
            topology.register_node(&format!("node-{}", i), &[0.5, 0.3, 0.2]);
        }
    });
}

#[bench]
fn bench_node_registration_10k(b: &mut Bencher) {
    b.iter(|| {
        let mut topology = evolution::NetworkTopology::new();
        for i in 0..10_000 {
            topology.register_node(&format!("node-{}", i), &[0.5, 0.3, 0.2]);
        }
    });
}

#[bench]
fn bench_optimal_peer_selection(b: &mut Bencher) {
    let mut topology = evolution::NetworkTopology::new();

    // Register nodes and create connections
    for i in 0..100 {
        topology.register_node(&format!("node-{}", i), &[0.5, 0.3, 0.2]);
    }

    for i in 0..100 {
        for j in 0..10 {
            topology.update_connection(
                &format!("node-{}", i),
                &format!("node-{}", (i + j + 1) % 100),
                0.8 + (j as f32 * 0.01)
            );
        }
    }

    b.iter(|| {
        topology.get_optimal_peers("node-0", 5)
    });
}

#[bench]
fn bench_cluster_assignment(b: &mut Bencher) {
    let mut topology = evolution::NetworkTopology::new();

    b.iter(|| {
        topology.register_node("test-node", &[0.7, 0.2, 0.1]);
    });
}

// ============================================================================
// Economic Engine Benchmarks
// ============================================================================

#[bench]
fn bench_reward_distribution(b: &mut Bencher) {
    let mut engine = evolution::EconomicEngine::new();

    b.iter(|| {
        engine.process_reward(100, 2.5)
    });
}

#[bench]
fn bench_epoch_processing(b: &mut Bencher) {
    let mut engine = evolution::EconomicEngine::new();

    // Build up some state
    for _ in 0..1000 {
        engine.process_reward(100, 1.0);
    }

    b.iter(|| {
        engine.advance_epoch()
    });
}

#[bench]
fn bench_sustainability_check(b: &mut Bencher) {
    let mut engine = evolution::EconomicEngine::new();

    // Build treasury
    for _ in 0..10000 {
        engine.process_reward(100, 1.0);
    }

    b.iter(|| {
        engine.is_self_sustaining(1000, 5000)
    });
}

// ============================================================================
// Evolution Engine Benchmarks
// ============================================================================

#[bench]
fn bench_performance_recording(b: &mut Bencher) {
    let mut engine = evolution::EvolutionEngine::new();

    b.iter(|| {
        engine.record_performance("node-1", 0.95, 75.0);
    });
}

#[bench]
fn bench_replication_check(b: &mut Bencher) {
    let mut engine = evolution::EvolutionEngine::new();

    // Record high performance
    for _ in 0..10 {
        engine.record_performance("node-1", 0.98, 90.0);
    }

    b.iter(|| {
        engine.should_replicate("node-1")
    });
}

#[bench]
fn bench_evolution_step(b: &mut Bencher) {
    let mut engine = evolution::EvolutionEngine::new();

    b.iter(|| {
        engine.evolve()
    });
}

// ============================================================================
// Optimization Engine Benchmarks
// ============================================================================

#[bench]
fn bench_routing_record(b: &mut Bencher) {
    let mut engine = evolution::OptimizationEngine::new();

    b.iter(|| {
        engine.record_routing("vectors", "node-1", 150, true);
    });
}

#[bench]
fn bench_optimal_node_selection(b: &mut Bencher) {
    let mut engine = evolution::OptimizationEngine::new();

    // Build routing history
    for i in 0..100 {
        engine.record_routing("vectors", &format!("node-{}", i % 10), 100 + i, i % 3 == 0);
    }

    let candidates: Vec<String> = (0..10).map(|i| format!("node-{}", i)).collect();

    b.iter(|| {
        engine.select_optimal_node("vectors", candidates.clone())
    });
}

// ============================================================================
// Network Manager Benchmarks
// ============================================================================

#[bench]
fn bench_peer_registration(b: &mut Bencher) {
    let mut manager = network::WasmNetworkManager::new("bench-node");

    b.iter(|| {
        manager.register_peer(
            "peer-1",
            &[1, 2, 3, 4],
            vec!["vectors".to_string()],
            1000
        );
    });
}

#[bench]
fn bench_worker_selection(b: &mut Bencher) {
    let mut manager = network::WasmNetworkManager::new("bench-node");

    // Register 100 peers
    for i in 0..100 {
        manager.register_peer(
            &format!("peer-{}", i),
            &[1, 2, 3, 4],
            vec!["vectors".to_string()],
            1000
        );
        manager.update_reputation(&format!("peer-{}", i), (i as f32) * 0.005);
    }

    b.iter(|| {
        manager.select_workers("vectors", 5)
    });
}

// ============================================================================
// End-to-End Benchmarks
// ============================================================================

#[bench]
fn bench_full_task_lifecycle(b: &mut Bencher) {
    use tokio::runtime::Runtime;

    let rt = Runtime::new().unwrap();

    b.iter(|| {
        rt.block_on(async {
            let identity = identity::WasmNodeIdentity::generate("bench").unwrap();
            let mut ledger = credits::WasmCreditLedger::new(identity.node_id()).unwrap();
            let mut queue = tasks::WasmTaskQueue::new().unwrap();
            let executor = tasks::WasmTaskExecutor::new(1024 * 1024).unwrap();

            // Initial credits
            ledger.credit(1000, "initial").unwrap();

            // Create and submit task
            let payload = vec![0u8; 256];
            let task = queue.create_task("vectors", &payload, 100, &identity).unwrap();
            queue.submit(task).await.unwrap();

            // Claim and complete (simulated)
            if let Some(claimed_task) = queue.claim_next(&identity).await.unwrap() {
                // Simulated execution
                ledger.credit(10, &format!("task:{}", claimed_task.id)).unwrap();
            }
        });
    });
}

#[bench]
fn bench_network_coordination(b: &mut Bencher) {
    let mut manager = network::WasmNetworkManager::new("coordinator");
    let mut topology = evolution::NetworkTopology::new();
    let mut optimizer = evolution::OptimizationEngine::new();

    // Setup network
    for i in 0..50 {
        let node_id = format!("node-{}", i);
        manager.register_peer(&node_id, &[1, 2, 3, 4], vec!["vectors".to_string()], 1000);
        topology.register_node(&node_id, &[0.5, 0.3, 0.2]);
    }

    b.iter(|| {
        // Select workers
        let workers = manager.select_workers("vectors", 3);

        // Get optimal peers
        for worker in &workers {
            topology.get_optimal_peers(worker, 5);
        }

        // Record routing
        if let Some(worker) = workers.first() {
            optimizer.record_routing("vectors", worker, 120, true);
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn bench_compilation_test() {
        // Ensures benchmarks compile
        assert!(true);
    }
}
