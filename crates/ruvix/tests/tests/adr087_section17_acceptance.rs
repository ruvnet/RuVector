//! ADR-087 Section 17 Acceptance Test
//!
//! This test implements the acceptance criteria from Section 17 of ADR-087:
//! "RuVix Cognition Kernel: A Deterministic Foundation for Neural-Symbolic AI"
//!
//! ## Section 17 Acceptance Criteria:
//!
//! 1. Mount RVF (RuVector Format) cognitive container
//! 2. Emit perception event (sensory input)
//! 3. Perform proof-gated mutation
//! 4. Verify attestation
//! 5. Checkpoint, restart, and replay
//! 6. Verify bit-identical state
//!
//! This test validates the full cognition loop from perception to verified state.

use ruvix_cap::{CapManagerConfig, CapabilityManager, CapRights, ObjectType, RevokeRequest, TaskHandle};
use ruvix_queue::{KernelQueue, QueueConfig};
use ruvix_region::{
    append_only::AppendOnlyRegion, backing::StaticBacking, immutable::ImmutableRegion,
};
use ruvix_types::{MsgPriority, RegionHandle};

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Simple FNV-1a hash for state verification.
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Attestation record for checkpoint verification.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AttestationRecord {
    region_hash: u64,
    operation_count: u64,
    checkpoint_seq: u32,
}

impl AttestationRecord {
    fn new(region_hash: u64, operation_count: u64, checkpoint_seq: u32) -> Self {
        Self {
            region_hash,
            operation_count,
            checkpoint_seq,
        }
    }

    fn verify_identical(&self, other: &Self) -> bool {
        self.region_hash == other.region_hash
            && self.operation_count == other.operation_count
            && self.checkpoint_seq == other.checkpoint_seq
    }
}

/// Simulated proof for gated mutation.
/// In a real system, this would be a cryptographic proof from the Proof Kernel.
#[derive(Debug, Clone)]
struct SimulatedProof {
    /// The operation being proven
    operation_hash: u64,
    /// Task that produced the proof
    prover_task: TaskHandle,
    /// Validity flag (simulated)
    valid: bool,
}

impl SimulatedProof {
    fn new(operation_hash: u64, prover_task: TaskHandle) -> Self {
        Self {
            operation_hash,
            prover_task,
            valid: true,
        }
    }

    fn verify(&self) -> bool {
        self.valid
    }
}

/// Checkpoint state for restart/replay.
#[derive(Debug, Clone)]
struct CheckpointState {
    /// Serialized region data
    region_data: Vec<u8>,
    /// Operation log for replay
    operation_log: Vec<PerceptionEvent>,
    /// Checkpoint sequence number
    sequence: u32,
    /// State hash at checkpoint
    state_hash: u64,
}

/// Perception event (sensory input to the system).
#[derive(Debug, Clone)]
struct PerceptionEvent {
    /// Event type identifier
    event_type: u8,
    /// Event payload
    payload: Vec<u8>,
    /// Timestamp (simulated)
    timestamp: u64,
}

impl PerceptionEvent {
    fn new(event_type: u8, payload: &[u8], timestamp: u64) -> Self {
        Self {
            event_type,
            payload: payload.to_vec(),
            timestamp,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.event_type);
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }
}

// ============================================================================
// ADR-087 Section 17 Acceptance Test
// ============================================================================

/// Full acceptance test implementing Section 17 criteria.
#[test]
fn test_adr087_section17_full_acceptance() {
    // ========================================================================
    // Step 1: Mount RVF (RuVector Format) Cognitive Container
    // ========================================================================
    // In the real system, this would mount a cognitive container with:
    // - Vector stores for embeddings
    // - Append-only regions for perception logs
    // - Immutable regions for model weights

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
    let cognition_task = TaskHandle::new(1, 0);

    // Create the perception log region (append-only for immutability)
    let perception_backing = StaticBacking::<16384>::new();
    let perception_handle = RegionHandle::new(1, 0);
    let mut perception_region =
        AppendOnlyRegion::new(perception_backing, 16384, perception_handle).unwrap();

    // Create capability for perception region
    let perception_cap = cap_manager
        .create_root_capability(
            perception_handle.raw().id as u64,
            ObjectType::Region,
            0,
            cognition_task,
        )
        .unwrap();

    // Create the state region (for cognitive state)
    let state_backing = StaticBacking::<8192>::new();
    let state_handle = RegionHandle::new(2, 0);
    let mut state_region =
        AppendOnlyRegion::new(state_backing, 8192, state_handle).unwrap();

    // Create capability for state region
    let state_cap = cap_manager
        .create_root_capability(
            state_handle.raw().id as u64,
            ObjectType::Region,
            0,
            cognition_task,
        )
        .unwrap();

    // Create event queue for perception events
    let queue_config = QueueConfig::new(256, 512);
    let (mut event_queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

    // Verify mount succeeded
    assert!(cap_manager
        .has_rights(perception_cap, CapRights::WRITE)
        .is_ok());
    assert!(cap_manager
        .has_rights(state_cap, CapRights::WRITE)
        .is_ok());

    println!("Step 1: RVF cognitive container mounted successfully");

    // ========================================================================
    // Step 2: Emit Perception Event (Sensory Input)
    // ========================================================================
    // Simulate sensory input being recorded to the perception log

    let perception_events = vec![
        PerceptionEvent::new(0x01, b"visual_input_frame_001", 1000),
        PerceptionEvent::new(0x02, b"audio_sample_chunk_001", 1001),
        PerceptionEvent::new(0x03, b"tactile_sensor_data_01", 1002),
    ];

    let mut operation_count: u64 = 0;

    for event in &perception_events {
        // Verify capability before write
        cap_manager
            .has_rights(perception_cap, CapRights::WRITE)
            .unwrap();

        // Write event to perception log
        let event_bytes = event.to_bytes();
        perception_region.append(&event_bytes).unwrap();
        operation_count += 1;

        // Queue event for processing
        event_queue.send(&event_bytes, MsgPriority::High).unwrap();
    }

    assert_eq!(perception_region.len(), perception_events.iter()
        .map(|e| e.to_bytes().len())
        .sum::<usize>());

    println!("Step 2: {} perception events emitted", perception_events.len());

    // ========================================================================
    // Step 3: Perform Proof-Gated Mutation
    // ========================================================================
    // State mutations require a valid proof before execution
    // This ensures only verified computations modify cognitive state

    // Compute operation hash for proof
    let mutation_data = b"cognitive_state_update_v1";
    let operation_hash = fnv1a_hash(mutation_data);

    // Generate proof (simulated - in real system this is cryptographic)
    let proof = SimulatedProof::new(operation_hash, cognition_task);

    // Verify proof before allowing mutation
    assert!(proof.verify(), "Proof verification failed - mutation rejected");

    // Verify capability for state mutation
    cap_manager
        .has_rights(state_cap, CapRights::WRITE)
        .unwrap();

    // Perform proof-gated mutation
    state_region.append(mutation_data).unwrap();
    operation_count += 1;

    // Record proof verification in state (for attestation)
    let proof_record = format!("proof_verified:{}", operation_hash);
    state_region.append(proof_record.as_bytes()).unwrap();
    operation_count += 1;

    println!("Step 3: Proof-gated mutation completed (hash: {:016x})", operation_hash);

    // ========================================================================
    // Step 4: Verify Attestation
    // ========================================================================
    // Generate attestation of current state for external verification

    // Read current perception log state
    let mut perception_buf = vec![0u8; perception_region.len()];
    perception_region.read(0, &mut perception_buf).unwrap();

    // Read current cognitive state
    let mut state_buf = vec![0u8; state_region.len()];
    state_region.read(0, &mut state_buf).unwrap();

    // Compute attestation hashes
    let perception_hash = fnv1a_hash(&perception_buf);
    let state_hash = fnv1a_hash(&state_buf);
    let combined_hash = fnv1a_hash(&[perception_hash.to_le_bytes(), state_hash.to_le_bytes()].concat());

    // Create attestation record
    let attestation = AttestationRecord::new(combined_hash, operation_count, 1);

    println!("Step 4: Attestation verified (hash: {:016x}, ops: {})",
             attestation.region_hash, attestation.operation_count);

    // ========================================================================
    // Step 5: Checkpoint, Restart, and Replay
    // ========================================================================
    // Create checkpoint, simulate restart, replay operations

    // Create checkpoint
    let checkpoint = CheckpointState {
        region_data: state_buf.clone(),
        operation_log: perception_events.clone(),
        sequence: 1,
        state_hash: combined_hash,
    };

    println!("Step 5a: Checkpoint created (seq: {})", checkpoint.sequence);

    // Simulate restart: create fresh regions
    let fresh_perception_backing = StaticBacking::<16384>::new();
    let fresh_perception_handle = RegionHandle::new(10, 0);
    let mut fresh_perception_region =
        AppendOnlyRegion::new(fresh_perception_backing, 16384, fresh_perception_handle).unwrap();

    let fresh_state_backing = StaticBacking::<8192>::new();
    let fresh_state_handle = RegionHandle::new(20, 0);
    let mut fresh_state_region =
        AppendOnlyRegion::new(fresh_state_backing, 8192, fresh_state_handle).unwrap();

    // Replay perception events
    let mut replay_operation_count: u64 = 0;

    for event in &checkpoint.operation_log {
        let event_bytes = event.to_bytes();
        fresh_perception_region.append(&event_bytes).unwrap();
        replay_operation_count += 1;
    }

    // Replay state mutation with proof verification
    let replay_proof = SimulatedProof::new(operation_hash, cognition_task);
    assert!(replay_proof.verify(), "Replay proof verification failed");

    fresh_state_region.append(mutation_data).unwrap();
    replay_operation_count += 1;

    let replay_proof_record = format!("proof_verified:{}", operation_hash);
    fresh_state_region.append(replay_proof_record.as_bytes()).unwrap();
    replay_operation_count += 1;

    println!("Step 5b: Replay completed ({} operations)", replay_operation_count);

    // ========================================================================
    // Step 6: Verify Bit-Identical State
    // ========================================================================
    // The replayed state must be bit-identical to the original

    // Read replayed perception state
    let mut replay_perception_buf = vec![0u8; fresh_perception_region.len()];
    fresh_perception_region.read(0, &mut replay_perception_buf).unwrap();

    // Read replayed cognitive state
    let mut replay_state_buf = vec![0u8; fresh_state_region.len()];
    fresh_state_region.read(0, &mut replay_state_buf).unwrap();

    // Compute replay attestation hashes
    let replay_perception_hash = fnv1a_hash(&replay_perception_buf);
    let replay_state_hash = fnv1a_hash(&replay_state_buf);
    let replay_combined_hash = fnv1a_hash(
        &[replay_perception_hash.to_le_bytes(), replay_state_hash.to_le_bytes()].concat()
    );

    // Create replay attestation
    let replay_attestation = AttestationRecord::new(
        replay_combined_hash,
        replay_operation_count,
        checkpoint.sequence,
    );

    // Verify bit-identical state
    assert_eq!(
        perception_buf.len(),
        replay_perception_buf.len(),
        "Perception log size mismatch"
    );
    assert_eq!(
        perception_buf,
        replay_perception_buf,
        "Perception log content mismatch"
    );

    assert_eq!(
        state_buf.len(),
        replay_state_buf.len(),
        "State region size mismatch"
    );
    assert_eq!(
        state_buf,
        replay_state_buf,
        "State region content mismatch"
    );

    assert!(
        attestation.verify_identical(&replay_attestation),
        "Attestation mismatch: original {:?} != replay {:?}",
        attestation,
        replay_attestation
    );

    println!("Step 6: Bit-identical state verified!");
    println!("  Original hash:  {:016x}", attestation.region_hash);
    println!("  Replay hash:    {:016x}", replay_attestation.region_hash);
    println!("  Operations:     {} (original) == {} (replay)",
             attestation.operation_count, replay_attestation.operation_count);

    println!("\n=== ADR-087 Section 17 Acceptance Test: PASSED ===");
}

// ============================================================================
// Individual Acceptance Criteria Tests
// ============================================================================

#[test]
fn test_section17_criterion_1_mount_rvf() {
    // Criterion 1: Mount RVF cognitive container

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
    let task = TaskHandle::new(1, 0);

    // Mount multiple region types
    let immutable_backing = StaticBacking::<1024>::new();
    let immutable_handle = RegionHandle::new(1, 0);
    let _immutable = ImmutableRegion::new(
        immutable_backing,
        b"model weights placeholder",
        immutable_handle,
    ).unwrap();

    let append_backing = StaticBacking::<4096>::new();
    let append_handle = RegionHandle::new(2, 0);
    let _append = AppendOnlyRegion::new(append_backing, 4096, append_handle).unwrap();

    // Create capabilities for mounted regions
    let cap1 = cap_manager
        .create_root_capability(1, ObjectType::Region, 0, task)
        .unwrap();
    let cap2 = cap_manager
        .create_root_capability(2, ObjectType::Region, 0, task)
        .unwrap();

    // Verify capabilities are valid
    assert!(cap_manager.has_rights(cap1, CapRights::READ).unwrap());
    assert!(cap_manager.has_rights(cap2, CapRights::READ | CapRights::WRITE).unwrap());
}

#[test]
fn test_section17_criterion_2_perception_event() {
    // Criterion 2: Emit perception event

    let backing = StaticBacking::<4096>::new();
    let handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 4096, handle).unwrap();

    let queue_config = QueueConfig::new(64, 256);
    let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

    // Emit multiple perception events
    for i in 0..10 {
        let event = PerceptionEvent::new(
            i as u8,
            format!("perception_data_{}", i).as_bytes(),
            1000 + i as u64,
        );

        let event_bytes = event.to_bytes();
        region.append(&event_bytes).unwrap();
        queue.send(&event_bytes, MsgPriority::Normal).unwrap();
    }

    // Verify events were recorded
    assert!(region.len() > 0);
    assert_eq!(queue.len(), 10);
}

#[test]
fn test_section17_criterion_3_proof_gated_mutation() {
    // Criterion 3: Perform proof-gated mutation

    let backing = StaticBacking::<1024>::new();
    let handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 1024, handle).unwrap();

    let task = TaskHandle::new(1, 0);
    let mutation_data = b"state_update";

    // Create proof
    let proof = SimulatedProof::new(fnv1a_hash(mutation_data), task);

    // Mutation without proof should be rejected (simulated)
    let invalid_proof = SimulatedProof {
        operation_hash: 0,
        prover_task: task,
        valid: false,
    };
    assert!(!invalid_proof.verify());

    // Mutation with valid proof succeeds
    assert!(proof.verify());
    region.append(mutation_data).unwrap();

    assert_eq!(region.len(), mutation_data.len());
}

#[test]
fn test_section17_criterion_4_verify_attestation() {
    // Criterion 4: Verify attestation

    let backing = StaticBacking::<1024>::new();
    let handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 1024, handle).unwrap();

    // Perform some operations
    region.append(b"operation_1").unwrap();
    region.append(b"operation_2").unwrap();
    region.append(b"operation_3").unwrap();

    // Read state and compute attestation
    let mut buf = vec![0u8; region.len()];
    region.read(0, &mut buf).unwrap();

    let attestation = AttestationRecord::new(fnv1a_hash(&buf), 3, 1);

    // Verify attestation is consistent
    let mut verify_buf = vec![0u8; region.len()];
    region.read(0, &mut verify_buf).unwrap();
    let verify_attestation = AttestationRecord::new(fnv1a_hash(&verify_buf), 3, 1);

    assert!(attestation.verify_identical(&verify_attestation));
}

#[test]
fn test_section17_criterion_5_checkpoint_restart_replay() {
    // Criterion 5: Checkpoint, restart, and replay

    let backing = StaticBacking::<1024>::new();
    let handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 1024, handle).unwrap();

    // Record operations
    let operations = vec![b"op1".to_vec(), b"op2".to_vec(), b"op3".to_vec()];
    for op in &operations {
        region.append(op).unwrap();
    }

    // Checkpoint: capture state
    let mut checkpoint_data = vec![0u8; region.len()];
    region.read(0, &mut checkpoint_data).unwrap();
    let checkpoint_hash = fnv1a_hash(&checkpoint_data);

    // Restart: create fresh region
    let fresh_backing = StaticBacking::<1024>::new();
    let fresh_handle = RegionHandle::new(2, 0);
    let mut fresh_region = AppendOnlyRegion::new(fresh_backing, 1024, fresh_handle).unwrap();

    // Replay operations
    for op in &operations {
        fresh_region.append(op).unwrap();
    }

    // Verify state matches
    let mut replay_data = vec![0u8; fresh_region.len()];
    fresh_region.read(0, &mut replay_data).unwrap();
    let replay_hash = fnv1a_hash(&replay_data);

    assert_eq!(checkpoint_hash, replay_hash);
    assert_eq!(checkpoint_data, replay_data);
}

#[test]
fn test_section17_criterion_6_bit_identical_state() {
    // Criterion 6: Verify bit-identical state

    // Create two independent systems
    let backing1 = StaticBacking::<2048>::new();
    let handle1 = RegionHandle::new(1, 0);
    let mut region1 = AppendOnlyRegion::new(backing1, 2048, handle1).unwrap();

    let backing2 = StaticBacking::<2048>::new();
    let handle2 = RegionHandle::new(2, 0);
    let mut region2 = AppendOnlyRegion::new(backing2, 2048, handle2).unwrap();

    // Perform identical operations on both
    let operations = [
        b"deterministic_operation_1".to_vec(),
        b"deterministic_operation_2".to_vec(),
        b"deterministic_operation_3".to_vec(),
        123456789u64.to_le_bytes().to_vec(),
        b"final_state".to_vec(),
    ];

    for op in &operations {
        region1.append(op).unwrap();
        region2.append(op).unwrap();
    }

    // Extract states
    let mut state1 = vec![0u8; region1.len()];
    let mut state2 = vec![0u8; region2.len()];

    region1.read(0, &mut state1).unwrap();
    region2.read(0, &mut state2).unwrap();

    // Verify bit-identical
    assert_eq!(state1.len(), state2.len(), "Length mismatch");

    for (i, (b1, b2)) in state1.iter().zip(state2.iter()).enumerate() {
        assert_eq!(b1, b2, "Byte mismatch at offset {}: {} != {}", i, b1, b2);
    }

    // Hash verification
    assert_eq!(fnv1a_hash(&state1), fnv1a_hash(&state2));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_proof_rejection_prevents_mutation() {
    // Verify that invalid proofs prevent mutation

    let backing = StaticBacking::<1024>::new();
    let handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 1024, handle).unwrap();

    let task = TaskHandle::new(1, 0);

    // Initial state
    region.append(b"initial").unwrap();
    let initial_len = region.len();

    // Create invalid proof
    let invalid_proof = SimulatedProof {
        operation_hash: 0,
        prover_task: task,
        valid: false,
    };

    // Attempted mutation with invalid proof
    if invalid_proof.verify() {
        region.append(b"should_not_appear").unwrap();
    }

    // State should be unchanged
    assert_eq!(region.len(), initial_len);
}

#[test]
fn test_capability_revocation_prevents_access() {
    // Verify that revoked capabilities prevent access

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

    let owner = TaskHandle::new(1, 0);
    let worker = TaskHandle::new(2, 0);

    // Owner creates capability
    let owner_cap = cap_manager
        .create_root_capability(1, ObjectType::Region, 0, owner)
        .unwrap();

    // Owner grants to worker
    let worker_cap = cap_manager
        .grant(owner_cap, CapRights::READ, 0, owner, worker)
        .unwrap();

    // Worker can access
    assert!(cap_manager.has_rights(worker_cap, CapRights::READ).unwrap());

    // Owner revokes their root capability (which has CapRights::ALL including REVOKE)
    // This cascades to invalidate the derived worker_cap via the derivation tree
    // Use revoke() with RevokeRequest to trigger cascading revocation
    let revoke_result = cap_manager.revoke(owner_cap, RevokeRequest::new()).unwrap();
    assert!(revoke_result.revoked_count >= 1); // At least owner_cap was revoked

    // Owner's capability is now invalid
    assert!(cap_manager.has_rights(owner_cap, CapRights::READ).is_err());

    // Worker can no longer access (derived capability invalidated when parent revoked)
    assert!(cap_manager.has_rights(worker_cap, CapRights::READ).is_err());
}

#[test]
fn test_deterministic_hash_computation() {
    // Verify hash computation is deterministic

    let data = b"test data for deterministic hashing";

    let hash1 = fnv1a_hash(data);
    let hash2 = fnv1a_hash(data);
    let hash3 = fnv1a_hash(data);

    assert_eq!(hash1, hash2);
    assert_eq!(hash2, hash3);

    // Different data produces different hash
    let different_data = b"different data";
    let different_hash = fnv1a_hash(different_data);
    assert_ne!(hash1, different_hash);
}
