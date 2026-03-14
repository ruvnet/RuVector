//! Integration tests for all 12 syscalls defined in ADR-087.
//!
//! This module tests each syscall independently and verifies:
//!
//! 1. TaskSpawn - Create new tasks
//! 2. CapGrant - Grant capabilities
//! 3. RegionMap - Map memory regions
//! 4. QueueSend - Send messages
//! 5. QueueRecv - Receive messages
//! 6. TimerWait - Wait for timer
//! 7. RvfMount - Mount RVF packages
//! 8. AttestEmit - Emit attestations
//! 9. VectorGet - Get vectors
//! 10. VectorPutProved - Put vectors with proof
//! 11. GraphApplyProved - Apply graph mutations with proof
//! 12. SensorSubscribe - Subscribe to sensors

use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, SyscallResult, VectorStoreConfig,
    TaskPriority, CapHandle, CapRights, ProofTier, VectorKey, RegionPolicy,
    MsgPriority, TimerSpec, SensorDescriptor, QueueHandle, GraphMutation,
    RvfMountHandle, RvfComponentId, Duration,
};
use ruvix_types::{TaskHandle, ObjectType};

// ============================================================================
// Test Fixtures
// ============================================================================

fn setup_kernel() -> Kernel {
    let mut kernel = Kernel::new(KernelConfig::default());
    kernel.boot(0, [0u8; 32]).expect("Boot failed");
    kernel.set_current_time(1_000_000);
    kernel
}

fn create_root_cap(kernel: &mut Kernel) -> CapHandle {
    let root_task = TaskHandle::new(1, 0);
    kernel.create_root_capability(0, ObjectType::RvfMount, root_task)
        .expect("Failed to create root capability")
}

// ============================================================================
// Syscall 0: TaskSpawn
// ============================================================================

#[test]
fn test_syscall_task_spawn_basic() {
    let mut kernel = setup_kernel();

    let result = kernel.dispatch(Syscall::TaskSpawn {
        entry: RvfComponentId::root(RvfMountHandle::null()),
        caps: Vec::new(),
        priority: TaskPriority::Normal,
        deadline: None,
    }).expect("TaskSpawn failed");

    match result {
        SyscallResult::TaskSpawned(handle) => {
            assert!(!handle.is_null(), "Task handle should not be null");
        }
        _ => panic!("Expected TaskSpawned result"),
    }
}

#[test]
fn test_syscall_task_spawn_all_priorities() {
    let mut kernel = setup_kernel();

    for priority in [
        TaskPriority::Background,
        TaskPriority::Normal,
        TaskPriority::High,
        TaskPriority::RealTime,
    ] {
        let result = kernel.dispatch(Syscall::TaskSpawn {
            entry: RvfComponentId::root(RvfMountHandle::null()),
            caps: Vec::new(),
            priority,
            deadline: None,
        });

        assert!(result.is_ok(), "TaskSpawn with {:?} priority should succeed", priority);
    }
}

#[test]
fn test_syscall_task_spawn_with_deadline() {
    let mut kernel = setup_kernel();

    let result = kernel.dispatch(Syscall::TaskSpawn {
        entry: RvfComponentId::root(RvfMountHandle::null()),
        caps: Vec::new(),
        priority: TaskPriority::RealTime,
        deadline: Some(Duration::from_millis(100)),
    }).expect("TaskSpawn with deadline failed");

    assert!(matches!(result, SyscallResult::TaskSpawned(_)));
}

#[test]
fn test_syscall_task_spawn_with_caps() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    let result = kernel.dispatch(Syscall::TaskSpawn {
        entry: RvfComponentId::root(RvfMountHandle::null()),
        caps: vec![cap],
        priority: TaskPriority::Normal,
        deadline: None,
    }).expect("TaskSpawn with caps failed");

    assert!(matches!(result, SyscallResult::TaskSpawned(_)));
}

// ============================================================================
// Syscall 1: CapGrant
// ============================================================================

#[test]
fn test_syscall_cap_grant_basic() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);
    let target = TaskHandle::new(2, 0);

    let result = kernel.dispatch(Syscall::CapGrant {
        target,
        cap,
        rights: CapRights::READ,
    }).expect("CapGrant failed");

    match result {
        SyscallResult::CapGranted(derived) => {
            assert!(!derived.is_null(), "Derived capability should not be null");
        }
        _ => panic!("Expected CapGranted result"),
    }
}

#[test]
fn test_syscall_cap_grant_various_rights() {
    let mut kernel = setup_kernel();
    let target = TaskHandle::new(2, 0);

    let rights_to_test = [
        CapRights::READ,
        CapRights::WRITE,
        CapRights::EXECUTE,
        CapRights::READ | CapRights::WRITE,
    ];

    for rights in rights_to_test {
        let cap = create_root_cap(&mut kernel);

        let result = kernel.dispatch(Syscall::CapGrant {
            target,
            cap,
            rights,
        });

        assert!(result.is_ok(), "CapGrant with {:?} rights should succeed", rights);
    }
}

// ============================================================================
// Syscall 2: RegionMap
// ============================================================================

#[test]
fn test_syscall_region_map_basic() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    let result = kernel.dispatch(Syscall::RegionMap {
        size: 4096,
        policy: RegionPolicy::Immutable,
        cap,
    }).expect("RegionMap failed");

    match result {
        SyscallResult::RegionMapped(handle) => {
            assert!(!handle.is_null(), "Region handle should not be null");
        }
        _ => panic!("Expected RegionMapped result"),
    }
}

#[test]
fn test_syscall_region_map_policies() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    let policies = [
        RegionPolicy::Immutable,
        RegionPolicy::AppendOnly { max_size: 8192 },
        RegionPolicy::Slab { slot_size: 64, slot_count: 128 },
    ];

    for policy in policies {
        let result = kernel.dispatch(Syscall::RegionMap {
            size: 4096,
            policy,
            cap,
        });

        assert!(result.is_ok(), "RegionMap with {:?} policy should succeed", policy);
    }
}

#[test]
fn test_syscall_region_map_various_sizes() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    for size in [256, 4096, 65536, 1048576] {
        let result = kernel.dispatch(Syscall::RegionMap {
            size,
            policy: RegionPolicy::Immutable,
            cap,
        });

        assert!(result.is_ok(), "RegionMap with size {} should succeed", size);
    }
}

// ============================================================================
// Syscall 3: QueueSend
// ============================================================================

#[test]
fn test_syscall_queue_send_basic() {
    let mut kernel = setup_kernel();
    let queue = QueueHandle::new(1, 0);

    let result = kernel.dispatch(Syscall::QueueSend {
        queue,
        msg: vec![1, 2, 3, 4],
        priority: MsgPriority::Normal,
    }).expect("QueueSend failed");

    assert!(matches!(result, SyscallResult::MessageSent));
}

#[test]
fn test_syscall_queue_send_priorities() {
    let mut kernel = setup_kernel();
    let queue = QueueHandle::new(1, 0);

    for priority in [
        MsgPriority::Low,
        MsgPriority::Normal,
        MsgPriority::High,
        MsgPriority::Urgent,
    ] {
        let result = kernel.dispatch(Syscall::QueueSend {
            queue,
            msg: vec![0xDE, 0xAD, 0xBE, 0xEF],
            priority,
        });

        assert!(result.is_ok(), "QueueSend with {:?} priority should succeed", priority);
    }
}

#[test]
fn test_syscall_queue_send_empty_message() {
    let mut kernel = setup_kernel();
    let queue = QueueHandle::new(1, 0);

    let result = kernel.dispatch(Syscall::QueueSend {
        queue,
        msg: Vec::new(),
        priority: MsgPriority::Normal,
    });

    assert!(result.is_ok(), "QueueSend with empty message should succeed");
}

// ============================================================================
// Syscall 4: QueueRecv
// ============================================================================

#[test]
fn test_syscall_queue_recv_basic() {
    let mut kernel = setup_kernel();
    let queue = QueueHandle::new(1, 0);

    let result = kernel.dispatch(Syscall::QueueRecv {
        queue,
        buf_size: 4096,
        timeout: Duration::from_millis(100),
    }).expect("QueueRecv failed");

    match result {
        SyscallResult::MessageReceived { data, priority } => {
            assert!(data.is_empty() || data.len() <= 4096);
            let _ = priority; // Priority is valid
        }
        _ => panic!("Expected MessageReceived result"),
    }
}

#[test]
fn test_syscall_queue_recv_various_timeouts() {
    let mut kernel = setup_kernel();
    let queue = QueueHandle::new(1, 0);

    for timeout_ms in [0, 10, 100, 1000] {
        let result = kernel.dispatch(Syscall::QueueRecv {
            queue,
            buf_size: 1024,
            timeout: Duration::from_millis(timeout_ms),
        });

        assert!(result.is_ok(), "QueueRecv with {}ms timeout should succeed", timeout_ms);
    }
}

// ============================================================================
// Syscall 5: TimerWait
// ============================================================================

#[test]
fn test_syscall_timer_wait_relative() {
    let mut kernel = setup_kernel();

    let result = kernel.dispatch(Syscall::TimerWait {
        deadline: TimerSpec::from_millis(100),
    }).expect("TimerWait failed");

    assert!(matches!(result, SyscallResult::TimerExpired));
}

#[test]
fn test_syscall_timer_wait_absolute() {
    let mut kernel = setup_kernel();

    let result = kernel.dispatch(Syscall::TimerWait {
        deadline: TimerSpec::Absolute { nanos_since_boot: 2_000_000 },
    }).expect("TimerWait failed");

    assert!(matches!(result, SyscallResult::TimerExpired));
}

#[test]
fn test_syscall_timer_wait_various_durations() {
    let mut kernel = setup_kernel();

    for duration_ms in [0, 1, 10, 100, 1000] {
        let result = kernel.dispatch(Syscall::TimerWait {
            deadline: TimerSpec::from_millis(duration_ms),
        });

        assert!(result.is_ok(), "TimerWait with {}ms should succeed", duration_ms);
    }
}

// ============================================================================
// Syscall 6: RvfMount
// ============================================================================

#[test]
fn test_syscall_rvf_mount_basic() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    let result = kernel.dispatch(Syscall::RvfMount {
        rvf_data: vec![0u8; 32], // Minimal RVF data
        mount_point: "/test".to_string(),
        cap,
    }).expect("RvfMount failed");

    match result {
        SyscallResult::RvfMounted(handle) => {
            assert!(!handle.is_null(), "Mount handle should not be null");
        }
        _ => panic!("Expected RvfMounted result"),
    }
}

#[test]
fn test_syscall_rvf_mount_various_paths() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    for mount_point in ["/", "/a", "/test/nested/path", "/pkg/v1.0"] {
        let result = kernel.dispatch(Syscall::RvfMount {
            rvf_data: vec![0u8; 32],
            mount_point: mount_point.to_string(),
            cap,
        });

        assert!(result.is_ok(), "RvfMount at '{}' should succeed", mount_point);
    }
}

// ============================================================================
// Syscall 7: AttestEmit
// ============================================================================

#[test]
fn test_syscall_attest_emit_boot() {
    let mut kernel = setup_kernel();

    let mutation_hash = [0u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    let result = kernel.dispatch(Syscall::AttestEmit {
        operation: ruvix_nucleus::AttestPayload::Boot {
            kernel_hash: [0x42u8; 32],
            boot_time_ns: 0,
        },
        proof,
    }).expect("AttestEmit failed");

    match result {
        SyscallResult::AttestEmitted { sequence } => {
            // Boot record already exists from kernel.boot(), so this is >= 1
            assert!(sequence >= 1, "Attestation sequence should be >= 1");
        }
        _ => panic!("Expected AttestEmitted result"),
    }
}

#[test]
fn test_syscall_attest_emit_checkpoint() {
    let mut kernel = setup_kernel();

    let state_hash = [0x42u8; 32];
    let mutation_hash = state_hash;
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    let result = kernel.dispatch(Syscall::AttestEmit {
        operation: ruvix_nucleus::AttestPayload::Checkpoint {
            sequence: 1,
            state_hash,
        },
        proof,
    }).expect("AttestEmit for checkpoint failed");

    assert!(matches!(result, SyscallResult::AttestEmitted { .. }));
}

// ============================================================================
// Syscall 8: VectorGet
// ============================================================================

#[test]
fn test_syscall_vector_get_basic() {
    let mut kernel = setup_kernel();

    // Create and populate vector store
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    // First put a vector
    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof,
    }).unwrap();

    // Now get it
    let result = kernel.dispatch(Syscall::VectorGet {
        store,
        key: VectorKey::new(1),
    }).expect("VectorGet failed");

    match result {
        SyscallResult::VectorRetrieved { data, coherence } => {
            assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
            assert!((coherence - 1.0).abs() < 0.001);
        }
        _ => panic!("Expected VectorRetrieved result"),
    }
}

#[test]
fn test_syscall_vector_get_not_found() {
    let mut kernel = setup_kernel();

    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    // Try to get non-existent vector
    let result = kernel.dispatch(Syscall::VectorGet {
        store,
        key: VectorKey::new(999),
    });

    assert!(result.is_err(), "VectorGet for non-existent key should fail");
}

// ============================================================================
// Syscall 9: VectorPutProved
// ============================================================================

#[test]
fn test_syscall_vector_put_proved_basic() {
    let mut kernel = setup_kernel();

    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    let result = kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof,
    }).expect("VectorPutProved failed");

    assert!(matches!(result, SyscallResult::VectorStored));
}

#[test]
fn test_syscall_vector_put_proved_all_tiers() {
    let mut kernel = setup_kernel();

    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    for (tier, nonce) in [
        (ProofTier::Reflex, 1),
        (ProofTier::Standard, 2),
        (ProofTier::Deep, 3),
    ] {
        let mutation_hash = [nonce as u8; 32];
        let proof = kernel.create_proof(mutation_hash, tier, nonce);

        let result = kernel.dispatch(Syscall::VectorPutProved {
            store,
            key: VectorKey::new(nonce),
            data: vec![nonce as f32; 4],
            proof,
        });

        assert!(result.is_ok(), "VectorPutProved with {:?} tier should succeed", tier);
    }
}

#[test]
fn test_syscall_vector_put_proved_expired_proof() {
    let mut kernel = setup_kernel();

    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    // Fast-forward time past proof expiry
    kernel.set_current_time(100_000_000_000);

    let result = kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof,
    });

    assert!(result.is_err(), "VectorPutProved with expired proof should fail");
}

#[test]
fn test_syscall_vector_put_proved_replay_attack() {
    let mut kernel = setup_kernel();

    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    // First put succeeds
    let result = kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof: proof.clone(),
    });
    assert!(result.is_ok());

    // Second put with same nonce should fail (replay attack)
    let result = kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(2),
        data: vec![5.0, 6.0, 7.0, 8.0],
        proof,
    });

    assert!(result.is_err(), "Replay attack should be detected");
}

// ============================================================================
// Syscall 10: GraphApplyProved
// ============================================================================

#[test]
fn test_syscall_graph_apply_proved_add_node() {
    let mut kernel = setup_kernel();

    let graph = kernel.create_graph_store().unwrap();

    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, 1);

    let result = kernel.dispatch(Syscall::GraphApplyProved {
        graph,
        mutation: GraphMutation::add_node(1),
        proof,
    }).expect("GraphApplyProved failed");

    assert!(matches!(result, SyscallResult::GraphApplied));
}

#[test]
fn test_syscall_graph_apply_proved_add_edge() {
    let mut kernel = setup_kernel();

    let graph = kernel.create_graph_store().unwrap();

    // First add two nodes
    for (node_id, nonce) in [(1, 1), (2, 2)] {
        let mutation_hash = [nonce as u8; 32];
        let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, nonce);

        kernel.dispatch(Syscall::GraphApplyProved {
            graph,
            mutation: GraphMutation::add_node(node_id),
            proof,
        }).unwrap();
    }

    // Now add an edge
    let mutation_hash = [3u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, 3);

    let result = kernel.dispatch(Syscall::GraphApplyProved {
        graph,
        mutation: GraphMutation::add_edge(1, 2, 1.0),
        proof,
    });

    assert!(result.is_ok(), "Adding edge should succeed");
}

#[test]
fn test_syscall_graph_apply_proved_set_property() {
    let mut kernel = setup_kernel();

    let graph = kernel.create_graph_store().unwrap();

    // Add two nodes first
    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, 1);

    kernel.dispatch(Syscall::GraphApplyProved {
        graph,
        mutation: GraphMutation::add_node(1),
        proof,
    }).unwrap();

    let mutation_hash = [2u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, 2);

    kernel.dispatch(Syscall::GraphApplyProved {
        graph,
        mutation: GraphMutation::add_node(2),
        proof,
    }).unwrap();

    // Add edge between them
    let mutation_hash = [3u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, 3);

    kernel.dispatch(Syscall::GraphApplyProved {
        graph,
        mutation: GraphMutation::add_edge(1, 2, 1.0),
        proof,
    }).unwrap();

    // Update edge weight
    let mutation_hash = [4u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, 4);

    let result = kernel.dispatch(Syscall::GraphApplyProved {
        graph,
        mutation: GraphMutation::update_edge_weight(1, 2, 3.14),
        proof,
    });

    assert!(result.is_ok(), "Updating edge weight should succeed");
}

// ============================================================================
// Syscall 11: SensorSubscribe
// ============================================================================

#[test]
fn test_syscall_sensor_subscribe_basic() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    let result = kernel.dispatch(Syscall::SensorSubscribe {
        sensor: SensorDescriptor::default(),
        target_queue: QueueHandle::new(1, 0),
        cap,
    }).expect("SensorSubscribe failed");

    match result {
        SyscallResult::SensorSubscribed(handle) => {
            assert!(!handle.is_null(), "Subscription handle should not be null");
        }
        _ => panic!("Expected SensorSubscribed result"),
    }
}

#[test]
fn test_syscall_sensor_subscribe_multiple() {
    let mut kernel = setup_kernel();
    let cap = create_root_cap(&mut kernel);

    // Subscribe to multiple sensors
    for i in 0..5 {
        let result = kernel.dispatch(Syscall::SensorSubscribe {
            sensor: SensorDescriptor::default(),
            target_queue: QueueHandle::new(i + 1, 0),
            cap,
        });

        assert!(result.is_ok(), "Subscription {} should succeed", i);
    }
}

// ============================================================================
// Cross-Syscall Integration Tests
// ============================================================================

#[test]
fn test_syscall_integration_full_flow() {
    let mut kernel = setup_kernel();

    // Create necessary resources
    let root_task = TaskHandle::new(1, 0);
    let root_cap = kernel.create_root_capability(0, ObjectType::RvfMount, root_task).unwrap();

    // 1. Spawn a task
    let result = kernel.dispatch(Syscall::TaskSpawn {
        entry: RvfComponentId::root(RvfMountHandle::null()),
        caps: vec![root_cap],
        priority: TaskPriority::Normal,
        deadline: None,
    }).unwrap();
    assert!(matches!(result, SyscallResult::TaskSpawned(_)));

    // 2. Map a region
    let result = kernel.dispatch(Syscall::RegionMap {
        size: 4096,
        policy: RegionPolicy::AppendOnly { max_size: 8192 },
        cap: root_cap,
    }).unwrap();
    assert!(matches!(result, SyscallResult::RegionMapped(_)));

    // 3. Create vector store and insert data
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    let mutation_hash = [1u8; 32];
    let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

    let result = kernel.dispatch(Syscall::VectorPutProved {
        store,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof,
    }).unwrap();
    assert!(matches!(result, SyscallResult::VectorStored));

    // 4. Read back the data
    let result = kernel.dispatch(Syscall::VectorGet {
        store,
        key: VectorKey::new(1),
    }).unwrap();

    match result {
        SyscallResult::VectorRetrieved { data, .. } => {
            assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        }
        _ => panic!("Expected VectorRetrieved"),
    }

    // 5. Verify stats
    let stats = kernel.stats();
    assert!(stats.syscalls_executed >= 4);
    assert_eq!(stats.proofs_verified, 1);
}

#[test]
fn test_syscall_number_mapping() {
    // Verify syscall numbers match ADR-087 specification
    let syscalls: Vec<(Syscall, u8)> = vec![
        (Syscall::TaskSpawn {
            entry: RvfComponentId::root(RvfMountHandle::null()),
            caps: Vec::new(),
            priority: TaskPriority::Normal,
            deadline: None,
        }, 0),
        (Syscall::CapGrant {
            target: TaskHandle::new(0, 0),
            cap: CapHandle::null(),
            rights: CapRights::READ,
        }, 1),
        (Syscall::RegionMap {
            size: 4096,
            policy: RegionPolicy::Immutable,
            cap: CapHandle::null(),
        }, 2),
        (Syscall::QueueSend {
            queue: QueueHandle::null(),
            msg: Vec::new(),
            priority: MsgPriority::Normal,
        }, 3),
        (Syscall::QueueRecv {
            queue: QueueHandle::null(),
            buf_size: 4096,
            timeout: Duration::from_millis(100),
        }, 4),
        (Syscall::TimerWait {
            deadline: TimerSpec::from_millis(100),
        }, 5),
        (Syscall::RvfMount {
            rvf_data: Vec::new(),
            mount_point: String::new(),
            cap: CapHandle::null(),
        }, 6),
        (Syscall::AttestEmit {
            operation: ruvix_nucleus::AttestPayload::Boot {
                kernel_hash: [0u8; 32],
                boot_time_ns: 0,
            },
            proof: Default::default(),
        }, 7),
        (Syscall::VectorGet {
            store: ruvix_nucleus::VectorStoreHandle::null(),
            key: VectorKey::new(0),
        }, 8),
        (Syscall::VectorPutProved {
            store: ruvix_nucleus::VectorStoreHandle::null(),
            key: VectorKey::new(0),
            data: Vec::new(),
            proof: Default::default(),
        }, 9),
        (Syscall::GraphApplyProved {
            graph: ruvix_nucleus::GraphHandle::null(),
            mutation: GraphMutation::add_node(0),
            proof: Default::default(),
        }, 10),
        (Syscall::SensorSubscribe {
            sensor: SensorDescriptor::default(),
            target_queue: QueueHandle::null(),
            cap: CapHandle::null(),
        }, 11),
    ];

    for (syscall, expected_number) in syscalls {
        assert_eq!(
            syscall.number(),
            expected_number,
            "Syscall {} should have number {}",
            syscall.name(),
            expected_number
        );
    }
}

#[test]
fn test_syscall_proof_required_flags() {
    // Verify proof_required flags match ADR-087 specification

    // These syscalls require proofs:
    let proof_required = [
        Syscall::RvfMount {
            rvf_data: Vec::new(),
            mount_point: String::new(),
            cap: CapHandle::null(),
        },
        Syscall::AttestEmit {
            operation: ruvix_nucleus::AttestPayload::Boot {
                kernel_hash: [0u8; 32],
                boot_time_ns: 0,
            },
            proof: Default::default(),
        },
        Syscall::VectorPutProved {
            store: ruvix_nucleus::VectorStoreHandle::null(),
            key: VectorKey::new(0),
            data: Vec::new(),
            proof: Default::default(),
        },
        Syscall::GraphApplyProved {
            graph: ruvix_nucleus::GraphHandle::null(),
            mutation: GraphMutation::add_node(0),
            proof: Default::default(),
        },
    ];

    for syscall in &proof_required {
        assert!(
            syscall.requires_proof(),
            "{} should require proof",
            syscall.name()
        );
    }

    // These syscalls do NOT require proofs:
    let no_proof_required = [
        Syscall::TaskSpawn {
            entry: RvfComponentId::root(RvfMountHandle::null()),
            caps: Vec::new(),
            priority: TaskPriority::Normal,
            deadline: None,
        },
        Syscall::CapGrant {
            target: TaskHandle::new(0, 0),
            cap: CapHandle::null(),
            rights: CapRights::READ,
        },
        Syscall::VectorGet {
            store: ruvix_nucleus::VectorStoreHandle::null(),
            key: VectorKey::new(0),
        },
    ];

    for syscall in &no_proof_required {
        assert!(
            !syscall.requires_proof(),
            "{} should NOT require proof",
            syscall.name()
        );
    }
}
