//! Deterministic replay tests for RuVix Cognition Kernel.
//!
//! This module tests the checkpoint/restore/replay functionality required by ADR-087.
//! The key property being tested is:
//!
//! **Deterministic Replay**: Given a checkpoint and the witness log, the system
//! can be replayed to reach the exact same state as before.
//!
//! This is critical for:
//! - Fault tolerance (recovery from crashes)
//! - Debugging (reproduce exact conditions)
//! - Verification (prove system behaved correctly)

use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, SyscallResult, VectorStoreConfig, CheckpointConfig,
    ProofTier, VectorKey, GraphMutation, WitnessRecord, WitnessRecordKind,
};

// ============================================================================
// Checkpoint Tests
// ============================================================================

#[test]
fn test_checkpoint_creation() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0u8; 32]).unwrap();
    kernel.set_current_time(1_000_000);

    let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();

    assert_eq!(checkpoint.sequence, 1, "First checkpoint should have sequence 1");
    assert_eq!(checkpoint.timestamp_ns, 1_000_000);
}

#[test]
fn test_checkpoint_with_data() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0u8; 32]).unwrap();
    kernel.set_current_time(1_000_000);

    // Create vector store and add data
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

    // Create checkpoint after mutation
    let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();

    // Verify checkpoint captures the state
    assert!(kernel.verify_checkpoint(&checkpoint));
    assert_ne!(checkpoint.state_hash, [0u8; 32], "State hash should not be zero");
}

#[test]
fn test_checkpoint_sequence_increment() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0u8; 32]).unwrap();
    kernel.set_current_time(1_000_000);

    // Create multiple checkpoints
    for i in 1..=5 {
        kernel.set_current_time(i * 1_000_000);
        let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();
        assert_eq!(checkpoint.sequence, i as u64);
    }

    assert_eq!(kernel.stats().checkpoints_created, 5);
}

#[test]
fn test_checkpoint_state_hash_changes() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0u8; 32]).unwrap();
    kernel.set_current_time(1_000_000);

    // Checkpoint before mutation
    let checkpoint1 = kernel.checkpoint(CheckpointConfig::full()).unwrap();

    // Add data
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

    kernel.set_current_time(2_000_000);

    // Checkpoint after mutation
    let checkpoint2 = kernel.checkpoint(CheckpointConfig::full()).unwrap();

    // State hashes should differ
    assert_ne!(
        checkpoint1.state_hash, checkpoint2.state_hash,
        "State hash should change after mutation"
    );
}

// ============================================================================
// Witness Log Tests
// ============================================================================

#[test]
fn test_witness_log_boot_record() {
    let mut kernel = Kernel::with_defaults();
    let kernel_hash = [0x42u8; 32];
    kernel.boot(0, kernel_hash).unwrap();

    let log = kernel.witness_log();
    assert_eq!(log.len(), 1);

    let record = log.get(0).unwrap();
    assert_eq!(record.kind, WitnessRecordKind::Boot);
    assert_eq!(record.mutation_hash, kernel_hash);
}

#[test]
fn test_witness_log_mutation_records() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0u8; 32]).unwrap();
    kernel.set_current_time(1_000_000);

    // Create vector store
    let config = VectorStoreConfig::new(4, 100);
    let store = kernel.create_vector_store(config).unwrap();

    // Perform multiple mutations
    for i in 0..3u64 {
        let mut mutation_hash = [0u8; 32];
        mutation_hash[0..8].copy_from_slice(&i.to_le_bytes());

        let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, i + 1);

        kernel.dispatch(Syscall::VectorPutProved {
            store,
            key: VectorKey::new(i as u64),
            data: vec![i as f32; 4],
            proof,
        }).unwrap();
    }

    // Verify witness log
    let log = kernel.witness_log();
    let mutations: Vec<_> = log.filter_by_kind(WitnessRecordKind::VectorMutation).collect();

    assert_eq!(mutations.len(), 3, "Should have 3 mutation records");

    // Verify sequence numbers are monotonic
    for (i, record) in mutations.iter().enumerate() {
        assert_eq!(record.sequence, (i + 1) as u64); // +1 because boot is sequence 0
    }
}

#[test]
fn test_witness_log_serialization_roundtrip() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0x42u8; 32]).unwrap();
    kernel.set_current_time(1_000_000);

    // Add some data
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

    kernel.checkpoint(CheckpointConfig::full()).unwrap();

    // Serialize witness log
    let bytes = kernel.witness_log().to_bytes();

    // Deserialize
    let restored = ruvix_nucleus::WitnessLog::from_bytes(&bytes).unwrap();

    // Verify
    assert_eq!(restored.len(), kernel.witness_log().len());

    for i in 0..restored.len() {
        let original = kernel.witness_log().get(i as u64).unwrap();
        let restored_record = restored.get(i as u64).unwrap();

        assert_eq!(original.sequence, restored_record.sequence);
        assert_eq!(original.kind, restored_record.kind);
        assert_eq!(original.mutation_hash, restored_record.mutation_hash);
    }
}

// ============================================================================
// Deterministic Replay Tests
// ============================================================================

#[test]
fn test_deterministic_replay_single_mutation() {
    // Original execution
    let mut kernel1 = Kernel::with_defaults();
    kernel1.boot(0, [0x42u8; 32]).unwrap();
    kernel1.set_current_time(1_000_000);

    let config = VectorStoreConfig::new(4, 100);
    let store1 = kernel1.create_vector_store(config).unwrap();

    let mutation_hash = [1u8; 32];
    let proof1 = kernel1.create_proof(mutation_hash, ProofTier::Reflex, 1);

    kernel1.dispatch(Syscall::VectorPutProved {
        store: store1,
        key: VectorKey::new(1),
        data: vec![1.0, 2.0, 3.0, 4.0],
        proof: proof1,
    }).unwrap();

    let checkpoint1 = kernel1.checkpoint(CheckpointConfig::full()).unwrap();
    let log_bytes = kernel1.witness_log().to_bytes();

    // Get final state
    let result1 = kernel1.dispatch(Syscall::VectorGet {
        store: store1,
        key: VectorKey::new(1),
    }).unwrap();

    let (data1, coherence1) = match result1 {
        SyscallResult::VectorRetrieved { data, coherence } => (data, coherence),
        _ => panic!("Expected VectorRetrieved"),
    };

    // Replay execution
    let mut kernel2 = Kernel::with_defaults();
    kernel2.boot(0, [0x42u8; 32]).unwrap();
    kernel2.set_current_time(1_000_000);

    let store2 = kernel2.create_vector_store(config).unwrap();

    // Replay the mutation from witness log
    let restored_log = ruvix_nucleus::WitnessLog::from_bytes(&log_bytes).unwrap();

    for record in restored_log.iter() {
        if record.kind == WitnessRecordKind::VectorMutation {
            // Recreate the mutation
            let proof2 = kernel2.create_proof(
                record.mutation_hash,
                ProofTier::Reflex,
                2, // Different nonce for replay
            );

            kernel2.dispatch(Syscall::VectorPutProved {
                store: store2,
                key: VectorKey::new(1),
                data: vec![1.0, 2.0, 3.0, 4.0], // Same data as original
                proof: proof2,
            }).unwrap();
        }
    }

    // Get replayed state
    let result2 = kernel2.dispatch(Syscall::VectorGet {
        store: store2,
        key: VectorKey::new(1),
    }).unwrap();

    let (data2, coherence2) = match result2 {
        SyscallResult::VectorRetrieved { data, coherence } => (data, coherence),
        _ => panic!("Expected VectorRetrieved"),
    };

    // CRITICAL: Replayed state MUST match original
    assert_eq!(data1, data2, "Replayed data must match original");
    assert!((coherence1 - coherence2).abs() < 0.001, "Replayed coherence must match");

    // Create checkpoint for replayed system
    let checkpoint2 = kernel2.checkpoint(CheckpointConfig::full()).unwrap();

    // State hashes should match
    assert_eq!(
        checkpoint1.state_hash, checkpoint2.state_hash,
        "Replayed state hash must match original"
    );
}

#[test]
fn test_deterministic_replay_multiple_mutations() {
    // Original execution with multiple mutations
    let mut kernel1 = Kernel::with_defaults();
    kernel1.boot(0, [0x42u8; 32]).unwrap();

    let config = VectorStoreConfig::new(4, 100);
    let store1 = kernel1.create_vector_store(config).unwrap();

    // Perform 10 mutations
    let mut mutation_data: Vec<(u32, Vec<f32>)> = Vec::new();

    for i in 0..10u64 {
        kernel1.set_current_time((i + 1) * 1_000_000);

        let mut mutation_hash = [0u8; 32];
        mutation_hash[0..8].copy_from_slice(&i.to_le_bytes());

        let proof = kernel1.create_proof(mutation_hash, ProofTier::Reflex, i + 1);
        let data = vec![(i as f32) * 1.1, (i as f32) * 2.2, (i as f32) * 3.3, (i as f32) * 4.4];

        mutation_data.push((i as u32, data.clone()));

        kernel1.dispatch(Syscall::VectorPutProved {
            store: store1,
            key: VectorKey::new(i as u64),
            data,
            proof,
        }).unwrap();
    }

    kernel1.set_current_time(11_000_000);
    let checkpoint1 = kernel1.checkpoint(CheckpointConfig::full()).unwrap();

    // Collect all final values
    let mut final_values1: Vec<Vec<f32>> = Vec::new();
    for i in 0..10u64 {
        let result = kernel1.dispatch(Syscall::VectorGet {
            store: store1,
            key: VectorKey::new(i as u64),
        }).unwrap();

        if let SyscallResult::VectorRetrieved { data, .. } = result {
            final_values1.push(data);
        }
    }

    // Replay execution
    let mut kernel2 = Kernel::with_defaults();
    kernel2.boot(0, [0x42u8; 32]).unwrap();

    let store2 = kernel2.create_vector_store(config).unwrap();

    // Replay all mutations in order
    for (key, data) in &mutation_data {
        kernel2.set_current_time((*key as u64 + 1) * 1_000_000);

        let mut mutation_hash = [0u8; 32];
        mutation_hash[0..8].copy_from_slice(&(*key as u64).to_le_bytes());

        let proof = kernel2.create_proof(
            mutation_hash,
            ProofTier::Reflex,
            *key as u64 + 11, // Different nonces for replay
        );

        kernel2.dispatch(Syscall::VectorPutProved {
            store: store2,
            key: VectorKey::new(*key as u64),
            data: data.clone(),
            proof,
        }).unwrap();
    }

    kernel2.set_current_time(11_000_000);
    let checkpoint2 = kernel2.checkpoint(CheckpointConfig::full()).unwrap();

    // Collect all replayed values
    let mut final_values2: Vec<Vec<f32>> = Vec::new();
    for i in 0..10u64 {
        let result = kernel2.dispatch(Syscall::VectorGet {
            store: store2,
            key: VectorKey::new(i as u64),
        }).unwrap();

        if let SyscallResult::VectorRetrieved { data, .. } = result {
            final_values2.push(data);
        }
    }

    // CRITICAL: All values must match
    for i in 0..10 {
        assert_eq!(
            final_values1[i], final_values2[i],
            "Replayed value {} must match original",
            i
        );
    }

    // State hashes must match
    assert_eq!(checkpoint1.state_hash, checkpoint2.state_hash);
}

#[test]
fn test_deterministic_replay_graph_mutations() {
    // Original execution with graph mutations
    let mut kernel1 = Kernel::with_defaults();
    kernel1.boot(0, [0u8; 32]).unwrap();
    kernel1.set_current_time(1_000_000);

    let graph1 = kernel1.create_graph_store().unwrap();

    // Add nodes
    for i in 1..=5 {
        let proof = kernel1.create_proof([i as u8; 32], ProofTier::Standard, i);

        kernel1.dispatch(Syscall::GraphApplyProved {
            graph: graph1,
            mutation: GraphMutation::add_node(i as u64),
            proof,
        }).unwrap();
    }

    // Add edges
    for i in 1..5 {
        let proof = kernel1.create_proof([i as u8 + 10; 32], ProofTier::Standard, i + 10);

        kernel1.dispatch(Syscall::GraphApplyProved {
            graph: graph1,
            mutation: GraphMutation::add_edge(i as u64, (i + 1) as u64, 1.0),
            proof,
        }).unwrap();
    }

    kernel1.set_current_time(2_000_000);
    let checkpoint1 = kernel1.checkpoint(CheckpointConfig::full()).unwrap();

    // Replay
    let mut kernel2 = Kernel::with_defaults();
    kernel2.boot(0, [0u8; 32]).unwrap();
    kernel2.set_current_time(1_000_000);

    let graph2 = kernel2.create_graph_store().unwrap();

    // Replay nodes
    for i in 1..=5 {
        let proof = kernel2.create_proof([i as u8; 32], ProofTier::Standard, i + 100);

        kernel2.dispatch(Syscall::GraphApplyProved {
            graph: graph2,
            mutation: GraphMutation::add_node(i as u64),
            proof,
        }).unwrap();
    }

    // Replay edges
    for i in 1..5 {
        let proof = kernel2.create_proof([i as u8 + 10; 32], ProofTier::Standard, i + 110);

        kernel2.dispatch(Syscall::GraphApplyProved {
            graph: graph2,
            mutation: GraphMutation::add_edge(i as u64, (i + 1) as u64, 1.0),
            proof,
        }).unwrap();
    }

    kernel2.set_current_time(2_000_000);
    let checkpoint2 = kernel2.checkpoint(CheckpointConfig::full()).unwrap();

    // State hashes should match
    assert_eq!(checkpoint1.state_hash, checkpoint2.state_hash);

    // Graph structure should match
    let store1 = kernel1.get_graph_store(graph1).unwrap();
    let store2 = kernel2.get_graph_store(graph2).unwrap();

    assert_eq!(store1.node_count(), store2.node_count());
    assert_eq!(store1.edge_count(), store2.edge_count());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_checkpoint_empty_kernel() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0u8; 32]).unwrap();

    // Checkpoint with no data
    let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();

    assert!(kernel.verify_checkpoint(&checkpoint));
}

#[test]
fn test_witness_record_serialization() {
    // Test individual record serialization
    let record = WitnessRecord::new(
        42,
        WitnessRecordKind::VectorMutation,
        1_000_000_000,
        [0xDE; 32],
        [0xAD; 32],
        0x12345678,
    );

    let bytes = record.to_bytes();
    let restored = WitnessRecord::from_bytes(&bytes);

    assert_eq!(record.sequence, restored.sequence);
    assert_eq!(record.kind, restored.kind);
    assert_eq!(record.timestamp_ns, restored.timestamp_ns);
    assert_eq!(record.mutation_hash, restored.mutation_hash);
    assert_eq!(record.attestation_hash, restored.attestation_hash);
    assert_eq!(record.resource_id, restored.resource_id);
}

#[test]
fn test_checkpoint_verification_fails_after_mutation() {
    let mut kernel = Kernel::with_defaults();
    kernel.boot(0, [0u8; 32]).unwrap();
    kernel.set_current_time(1_000_000);

    // Create checkpoint
    let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();

    // Verify checkpoint is valid
    assert!(kernel.verify_checkpoint(&checkpoint));

    // Mutate state
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

    // Old checkpoint should no longer verify (state has changed)
    assert!(
        !kernel.verify_checkpoint(&checkpoint),
        "Old checkpoint should not verify after mutation"
    );
}

#[test]
fn test_replay_order_matters() {
    // Demonstrate that replay order affects final state
    let config = VectorStoreConfig::new(4, 100);

    // Forward order
    let mut kernel1 = Kernel::with_defaults();
    kernel1.boot(0, [0u8; 32]).unwrap();
    kernel1.set_current_time(1_000_000);

    let store1 = kernel1.create_vector_store(config).unwrap();

    // Write to same key multiple times
    for i in 0..3 {
        let mutation_hash = [i as u8; 32];
        let proof = kernel1.create_proof(mutation_hash, ProofTier::Reflex, i + 1);

        kernel1.dispatch(Syscall::VectorPutProved {
            store: store1,
            key: VectorKey::new(1), // Same key
            data: vec![i as f32; 4],
            proof,
        }).unwrap();
    }

    let result1 = kernel1.dispatch(Syscall::VectorGet {
        store: store1,
        key: VectorKey::new(1),
    }).unwrap();

    let data1 = match result1 {
        SyscallResult::VectorRetrieved { data, .. } => data,
        _ => panic!("Expected VectorRetrieved"),
    };

    // Final value should be from last write
    assert_eq!(data1, vec![2.0, 2.0, 2.0, 2.0]);

    // Reverse order
    let mut kernel2 = Kernel::with_defaults();
    kernel2.boot(0, [0u8; 32]).unwrap();
    kernel2.set_current_time(1_000_000);

    let store2 = kernel2.create_vector_store(config).unwrap();

    // Write in reverse order
    for i in (0..3).rev() {
        let mutation_hash = [i as u8; 32];
        let proof = kernel2.create_proof(mutation_hash, ProofTier::Reflex, (2 - i) + 10);

        kernel2.dispatch(Syscall::VectorPutProved {
            store: store2,
            key: VectorKey::new(1), // Same key
            data: vec![i as f32; 4],
            proof,
        }).unwrap();
    }

    let result2 = kernel2.dispatch(Syscall::VectorGet {
        store: store2,
        key: VectorKey::new(1),
    }).unwrap();

    let data2 = match result2 {
        SyscallResult::VectorRetrieved { data, .. } => data,
        _ => panic!("Expected VectorRetrieved"),
    };

    // Final value should be from last write (which is now 0.0)
    assert_eq!(data2, vec![0.0, 0.0, 0.0, 0.0]);

    // Demonstrate that order affects final state
    assert_ne!(data1, data2, "Different order should produce different results");
}
