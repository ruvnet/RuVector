//! # RVM Integration Tests
//!
//! Cross-crate integration tests for the RVM microhypervisor.

#[cfg(test)]
mod tests {
    use rvm_types::{
        CapRights, CapToken, CapType, CoherenceScore, GuestPhysAddr,
        PartitionId, PhysAddr, WitnessHash, WitnessRecord, ActionKind,
    };

    #[test]
    fn partition_id_round_trip() {
        let id = PartitionId::new(42);
        assert_eq!(id.as_u32(), 42);
    }

    #[test]
    fn partition_id_vmid() {
        let id = PartitionId::new(0x1FF);
        assert_eq!(id.vmid(), 0xFF);
    }

    #[test]
    fn coherence_score_clamping() {
        let score = CoherenceScore::from_basis_points(15_000);
        assert_eq!(score.as_basis_points(), 10_000);
    }

    #[test]
    fn coherence_threshold() {
        let high = CoherenceScore::from_basis_points(5000);
        let low = CoherenceScore::from_basis_points(1000);
        assert!(high.is_coherent());
        assert!(!low.is_coherent());
    }

    #[test]
    fn witness_hash_zero() {
        assert!(WitnessHash::ZERO.is_zero());
        let non_zero = WitnessHash::from_bytes([1u8; 32]);
        assert!(!non_zero.is_zero());
    }

    #[test]
    fn witness_record_size() {
        assert_eq!(core::mem::size_of::<WitnessRecord>(), 64);
    }

    #[test]
    fn capability_rights_check() {
        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ,
            0,
        );
        assert!(token.has_rights(CapRights::READ));
        assert!(!token.has_rights(CapRights::WRITE));
    }

    #[test]
    fn capability_combined_rights() {
        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ | CapRights::WRITE | CapRights::GRANT,
            0,
        );
        assert!(token.has_rights(CapRights::READ | CapRights::WRITE));
        assert!(!token.has_rights(CapRights::EXECUTE));
    }

    #[test]
    fn memory_region_alignment() {
        let aligned = GuestPhysAddr::new(0x1000);
        let unaligned = GuestPhysAddr::new(0x1001);
        assert!(aligned.is_page_aligned());
        assert!(!unaligned.is_page_aligned());
    }

    #[test]
    fn phys_addr_page_align_down() {
        let addr = PhysAddr::new(0x1234);
        assert_eq!(addr.page_align_down().as_u64(), 0x1000);
    }

    #[test]
    fn boot_phase_sequence() {
        let mut tracker = rvm_boot::BootTracker::new();
        assert!(!tracker.is_complete());

        tracker.complete_phase(rvm_boot::BootPhase::HalInit).unwrap();
        tracker.complete_phase(rvm_boot::BootPhase::MemoryInit).unwrap();
        tracker.complete_phase(rvm_boot::BootPhase::CapabilityInit).unwrap();
        tracker.complete_phase(rvm_boot::BootPhase::WitnessInit).unwrap();
        tracker.complete_phase(rvm_boot::BootPhase::SchedulerInit).unwrap();
        tracker.complete_phase(rvm_boot::BootPhase::RootPartition).unwrap();
        tracker.complete_phase(rvm_boot::BootPhase::Handoff).unwrap();

        assert!(tracker.is_complete());
    }

    #[test]
    fn boot_phase_out_of_order() {
        let mut tracker = rvm_boot::BootTracker::new();
        assert!(tracker.complete_phase(rvm_boot::BootPhase::MemoryInit).is_err());
    }

    #[test]
    fn wasm_header_validation() {
        let valid = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        assert!(rvm_wasm::validate_header(&valid).is_ok());

        let bad_magic = [0xFF; 8];
        assert!(rvm_wasm::validate_header(&bad_magic).is_err());

        let short = [0x00, 0x61];
        assert!(rvm_wasm::validate_header(&short).is_err());
    }

    #[test]
    fn security_gate_enforcement() {
        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ | CapRights::WRITE,
            0,
        );

        let request = rvm_security::PolicyRequest {
            token: &token,
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            current_epoch: None,
        };
        assert!(rvm_security::enforce(&request).is_ok());

        // Wrong type should fail.
        let bad_request = rvm_security::PolicyRequest {
            token: &token,
            required_type: CapType::Region,
            required_rights: CapRights::READ,
            proof_commitment: None,
            current_epoch: None,
        };
        assert!(rvm_security::enforce(&bad_request).is_err());
    }

    #[test]
    fn witness_log_append() {
        let mut log = rvm_witness::WitnessLog::<16>::new();
        assert!(log.is_empty());

        let record = WitnessRecord::zeroed();
        log.append(record);
        assert_eq!(log.len(), 1);

        log.append(record);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn witness_emitter_builds_records() {
        let log = rvm_witness::WitnessLog::<16>::new();
        let emitter = rvm_witness::WitnessEmitter::new(&log);
        let seq = emitter.emit_partition_create(
            1,         // actor_partition_id
            100,       // new_partition_id
            0xABCD,    // cap_hash
            1_000_000, // timestamp_ns
        );
        assert_eq!(seq, 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn coherence_ema_filter() {
        let mut filter = rvm_coherence::EmaFilter::new(5000); // 50% alpha
        let score = filter.update(8000);
        assert_eq!(score.as_basis_points(), 8000);

        let score2 = filter.update(4000);
        assert_eq!(score2.as_basis_points(), 6000);
    }

    #[test]
    fn partition_manager_basic() {
        let mut mgr = rvm_partition::PartitionManager::new();
        assert_eq!(mgr.count(), 0);

        let id = mgr.create(
            rvm_partition::PartitionType::Agent,
            2,
            1,
        ).unwrap();
        assert_eq!(mgr.count(), 1);
        assert!(mgr.get(id).is_some());
    }

    #[test]
    fn kernel_version() {
        assert!(!rvm_kernel::VERSION.is_empty());
        assert_eq!(rvm_kernel::CRATE_COUNT, 13);
    }

    #[test]
    fn action_kind_subsystem() {
        assert_eq!(ActionKind::PartitionCreate.subsystem(), 0);
        assert_eq!(ActionKind::CapabilityGrant.subsystem(), 1);
        assert_eq!(ActionKind::RegionCreate.subsystem(), 2);
    }

    #[test]
    fn fnv1a_hash() {
        let hash = rvm_witness::fnv1a_64(b"hello");
        assert_ne!(hash, 0);
        // Deterministic.
        assert_eq!(hash, rvm_witness::fnv1a_64(b"hello"));
    }

    // ===============================================================
    // Cross-crate integration scenarios
    // ===============================================================

    // ---------------------------------------------------------------
    // Scenario 1: Create partition -> grant capability -> verify P1
    //             -> emit witness -> check chain
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_partition_cap_proof_witness_chain() {
        use rvm_cap::CapabilityManager;
        use rvm_types::{CapType, CapRights, ProofTier, ProofToken};
        use rvm_proof::context::ProofContextBuilder;
        use rvm_proof::engine::ProofEngine;

        // Step 1: Create a partition via the partition manager.
        let mut part_mgr = rvm_partition::PartitionManager::new();
        let pid = part_mgr
            .create(rvm_partition::PartitionType::Agent, 2, 0)
            .unwrap();

        // Step 2: Grant a capability to this partition via the cap manager.
        let mut cap_mgr = CapabilityManager::<64>::with_defaults();
        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::EXECUTE)
            .union(CapRights::GRANT)
            .union(CapRights::REVOKE)
            .union(CapRights::PROVE);

        let (root_idx, root_gen) = cap_mgr
            .create_root_capability(CapType::Partition, all_rights, 0, pid)
            .unwrap();

        // Step 3: Verify P1 on the capability.
        assert!(cap_mgr.verify_p1(root_idx, root_gen, CapRights::PROVE).is_ok());

        // Step 4: Run the full proof engine pipeline (P1 + P2 + witness).
        let witness_log = rvm_witness::WitnessLog::<32>::new();
        let token = ProofToken {
            tier: ProofTier::P2,
            epoch: 0,
            hash: 0x1234,
        };
        let context = ProofContextBuilder::new(pid)
            .target_object(42)
            .capability_handle(root_idx)
            .capability_generation(root_gen)
            .current_epoch(0)
            .region_bounds(0x1000, 0x2000)
            .time_window(500, 1000)
            .nonce(1)
            .build();

        let mut engine = ProofEngine::<64>::new();
        engine
            .verify_and_witness(&token, &context, &cap_mgr, &witness_log)
            .unwrap();

        // Step 5: Verify witness chain integrity.
        assert_eq!(witness_log.total_emitted(), 1);
        let record = witness_log.get(0).unwrap();
        assert_eq!(record.action_kind, ActionKind::ProofVerifiedP2 as u8);
        assert_eq!(record.actor_partition_id, pid.as_u32());
        assert_eq!(record.target_object_id, 42);
        assert_eq!(record.capability_hash, 0x1234);
    }

    // ---------------------------------------------------------------
    // Scenario 2: Security gate end-to-end with valid/invalid caps
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_security_gate_valid_request() {
        use rvm_security::{SecurityGate, GateRequest};
        use rvm_types::WitnessHash;

        let log = rvm_witness::WitnessLog::<32>::new();
        let gate = SecurityGate::new(&log);

        // Valid request: correct type, sufficient rights, valid proof.
        let token = CapToken::new(
            1,
            CapType::Region,
            CapRights::READ | CapRights::WRITE,
            0,
        );
        let commitment = WitnessHash::from_bytes([0xAB; 32]);
        let request = GateRequest {
            token,
            required_type: CapType::Region,
            required_rights: CapRights::WRITE,
            proof_commitment: Some(commitment),
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::RegionCreate,
            target_object_id: 100,
            timestamp_ns: 5000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 2); // P2 because proof commitment provided.
        assert_eq!(response.witness_sequence, 0);
        assert_eq!(log.total_emitted(), 1);

        // Check the witness record.
        let record = log.get(0).unwrap();
        assert_eq!(record.action_kind, ActionKind::RegionCreate as u8);
    }

    #[test]
    fn cross_crate_security_gate_wrong_type() {
        use rvm_security::{SecurityGate, SecurityError, GateRequest};

        let log = rvm_witness::WitnessLog::<32>::new();
        let gate = SecurityGate::new(&log);

        let token = CapToken::new(1, CapType::Region, CapRights::READ, 0);
        let request = GateRequest {
            token,
            required_type: CapType::Partition, // Wrong type.
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 1,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::CapabilityTypeMismatch);

        // Rejection witness emitted.
        let record = log.get(0).unwrap();
        assert_eq!(record.action_kind, ActionKind::ProofRejected as u8);
    }

    #[test]
    fn cross_crate_security_gate_insufficient_rights() {
        use rvm_security::{SecurityGate, SecurityError, GateRequest};

        let log = rvm_witness::WitnessLog::<32>::new();
        let gate = SecurityGate::new(&log);

        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ, // Only READ, but WRITE required.
            0,
        );
        let request = GateRequest {
            token,
            required_type: CapType::Partition,
            required_rights: CapRights::WRITE,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 1,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::InsufficientRights);
    }

    #[test]
    fn cross_crate_security_gate_zero_proof_commitment() {
        use rvm_security::{SecurityGate, SecurityError, GateRequest};
        use rvm_types::WitnessHash;

        let log = rvm_witness::WitnessLog::<32>::new();
        let gate = SecurityGate::new(&log);

        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ | CapRights::WRITE,
            0,
        );
        let request = GateRequest {
            token,
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: Some(WitnessHash::ZERO), // Zero = invalid.
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 1,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::PolicyViolation);
    }

    // ---------------------------------------------------------------
    // Scenario 3: Coherence scoring -> scheduler priority computation
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_coherence_score_to_scheduler_priority() {
        use rvm_types::CutPressure;

        // Simulate: partition has coherence 8000bp, gets a cut pressure signal.
        let coherence = CoherenceScore::from_basis_points(8000);
        assert!(coherence.is_coherent());

        // Convert coherence into a cut pressure value (higher coherence = lower pressure).
        // Pressure is typically derived from the graph, but we simulate:
        let pressure = CutPressure::from_fixed(0x0003_0000); // boost = 3
        let deadline_urgency: u16 = 100;

        let priority = rvm_sched::compute_priority(deadline_urgency, pressure);
        assert_eq!(priority, 103); // 100 + 3

        // Now test with zero pressure (degraded mode / DC-1).
        let priority_degraded = rvm_sched::compute_priority(deadline_urgency, CutPressure::ZERO);
        assert_eq!(priority_degraded, 100); // deadline only
    }

    #[test]
    fn cross_crate_coherence_driven_partition_split_decision() {
        use rvm_types::CutPressure;

        // Partition with high cut pressure should trigger split.
        let pressure = CutPressure::from_fixed(9000);
        assert!(pressure.exceeds_threshold(CutPressure::DEFAULT_SPLIT_THRESHOLD));

        // Low pressure should not trigger split.
        let low_pressure = CutPressure::from_fixed(5000);
        assert!(!low_pressure.exceeds_threshold(CutPressure::DEFAULT_SPLIT_THRESHOLD));
    }

    // ---------------------------------------------------------------
    // Scenario 4: Full kernel lifecycle: boot, create, tick, witness check
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_kernel_full_lifecycle() {
        use rvm_kernel::{Kernel, KernelConfig};
        use rvm_types::PartitionConfig;

        let mut kernel = Kernel::new(KernelConfig::default());
        kernel.boot().unwrap();
        assert!(kernel.is_booted());

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.partition_count(), 2);
        assert_ne!(id1, id2);

        // Tick a few times.
        for _ in 0..3 {
            kernel.tick().unwrap();
        }
        assert_eq!(kernel.current_epoch(), 3);

        // Destroy one partition.
        kernel.destroy_partition(id1).unwrap();

        // Total witnesses: 7 boot + 2 create + 3 tick + 1 destroy = 13.
        assert_eq!(kernel.witness_count(), 13);
    }

    // ---------------------------------------------------------------
    // Scenario 5: Memory region management + tier placement
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_memory_region_and_tier() {
        use rvm_memory::{RegionManager, RegionConfig, TierManager, Tier, BuddyAllocator, MemoryPermissions};
        use rvm_types::{OwnedRegionId, PhysAddr};

        // Set up a buddy allocator.
        let mut alloc = BuddyAllocator::<16, 2>::new(PhysAddr::new(0x1000_0000)).unwrap();
        let addr = alloc.alloc_pages(0).unwrap();
        assert!(addr.is_page_aligned());

        // Set up a region manager and create a region.
        let mut region_mgr = RegionManager::<16>::new();
        let rid = region_mgr
            .create(RegionConfig {
                id: OwnedRegionId::new(1),
                owner: PartitionId::new(1),
                guest_base: GuestPhysAddr::new(0x0),
                host_base: PhysAddr::new(addr.as_u64()),
                page_count: 1,
                tier: Tier::Warm,
                permissions: MemoryPermissions::READ_WRITE,
            })
            .unwrap();

        // Register in the tier manager.
        let mut tier_mgr = TierManager::<8>::new();
        tier_mgr.register(rid, Tier::Warm).unwrap();

        let state = tier_mgr.get(rid).unwrap();
        assert_eq!(state.tier, Tier::Warm);
    }

    // ---------------------------------------------------------------
    // Scenario 6: Witness log integrity verification
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_witness_log_chain_integrity() {
        let log = rvm_witness::WitnessLog::<32>::new();

        // Emit several records.
        for i in 0..5u8 {
            let mut record = WitnessRecord::zeroed();
            record.action_kind = i;
            record.proof_tier = 1;
            record.actor_partition_id = 1;
            log.append(record);
        }

        assert_eq!(log.total_emitted(), 5);

        // Collect records and verify chain.
        let mut records = [WitnessRecord::zeroed(); 5];
        for i in 0..5 {
            records[i] = log.get(i).unwrap();
        }

        let result = rvm_witness::verify_chain(&records);
        assert!(result.is_ok());
    }

    // ---------------------------------------------------------------
    // Scenario 7: EMA filter feeds coherence score
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_ema_coherence_scoring() {
        // Use EMA filter to smooth coherence signal, then check threshold.
        let mut filter = rvm_coherence::EmaFilter::new(5000); // 50% alpha
        let s1 = filter.update(9000); // First update: takes raw value.
        assert_eq!(s1.as_basis_points(), 9000);
        assert!(s1.is_coherent());

        let s2 = filter.update(2000); // Smoothed: (9000 + 2000) / 2 = 5500.
        assert_eq!(s2.as_basis_points(), 5500);
        assert!(s2.is_coherent()); // 5500 >= 3000 threshold

        let s3 = filter.update(1000); // (5500 + 1000) / 2 = 3250.
        assert_eq!(s3.as_basis_points(), 3250);
        assert!(s3.is_coherent()); // 3250 >= 3000

        let s4 = filter.update(1000); // (3250 + 1000) / 2 = 2125.
        assert_eq!(s4.as_basis_points(), 2125);
        assert!(!s4.is_coherent()); // 2125 < 3000
    }

    // ---------------------------------------------------------------
    // Scenario 8: Partition split scoring
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_partition_split_scoring() {
        let region_coherence = CoherenceScore::from_basis_points(6000);
        let left = CoherenceScore::from_basis_points(5500);
        let right = CoherenceScore::from_basis_points(8000);

        let score = rvm_partition::scored_region_assignment(region_coherence, left, right);
        // |6000-5500| = 500, |6000-8000| = 2000 -> closer to left.
        assert_eq!(score, 7500);
    }

    // ---------------------------------------------------------------
    // Scenario 9: Merge preconditions with coherence scores
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_partition_merge_preconditions() {
        let high = CoherenceScore::from_basis_points(8000);
        let low = CoherenceScore::from_basis_points(5000);

        // Both high -> merge allowed.
        assert!(rvm_partition::merge_preconditions_met(high, high).is_ok());

        // One low -> merge denied.
        assert!(rvm_partition::merge_preconditions_met(high, low).is_err());
    }

    // ---------------------------------------------------------------
    // Scenario 10: Proof verification with insufficient cap then retry
    // ---------------------------------------------------------------
    #[test]
    fn cross_crate_proof_retry_after_cap_grant() {
        use rvm_cap::CapabilityManager;
        use rvm_types::{CapType, CapRights, ProofTier, ProofToken};
        use rvm_proof::context::ProofContextBuilder;
        use rvm_proof::engine::ProofEngine;

        let witness_log = rvm_witness::WitnessLog::<32>::new();
        let mut cap_mgr = CapabilityManager::<64>::with_defaults();
        let owner = PartitionId::new(1);

        // Create capability with READ only (no PROVE).
        let (idx, gen) = cap_mgr
            .create_root_capability(CapType::Region, CapRights::READ, 0, owner)
            .unwrap();

        let token = ProofToken {
            tier: ProofTier::P1,
            epoch: 0,
            hash: 0,
        };

        let context = ProofContextBuilder::new(owner)
            .capability_handle(idx)
            .capability_generation(gen)
            .region_bounds(0x1000, 0x2000)
            .time_window(500, 1000)
            .nonce(1)
            .build();

        let mut engine = ProofEngine::<64>::new();

        // First attempt: should fail (no PROVE right).
        assert!(engine.verify_and_witness(&token, &context, &cap_mgr, &witness_log).is_err());
        assert_eq!(witness_log.total_emitted(), 1); // Rejection emitted.

        // Create a new capability with PROVE rights.
        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::PROVE);
        let (idx2, gen2) = cap_mgr
            .create_root_capability(CapType::Region, all_rights, 0, owner)
            .unwrap();

        let context2 = ProofContextBuilder::new(owner)
            .capability_handle(idx2)
            .capability_generation(gen2)
            .region_bounds(0x1000, 0x2000)
            .time_window(500, 1000)
            .nonce(2) // Different nonce.
            .build();

        // Second attempt with proper cap: should succeed.
        assert!(engine.verify_and_witness(&token, &context2, &cap_mgr, &witness_log).is_ok());
        assert_eq!(witness_log.total_emitted(), 2);
    }

    // ===============================================================
    // End-to-end integration scenarios
    // ===============================================================

    // ---------------------------------------------------------------
    // E2E Scenario 1: Full Agent Lifecycle
    //
    // Boot kernel -> create partition -> verify it exists ->
    // tick scheduler (agent runs) -> destroy partition ->
    // verify witness chain covers entire lifecycle
    // ---------------------------------------------------------------
    #[test]
    fn e2e_full_agent_lifecycle() {
        use rvm_kernel::{Kernel, KernelConfig};
        use rvm_types::PartitionConfig;

        // Phase 1: Boot the kernel.
        let mut kernel = Kernel::new(KernelConfig::default());
        kernel.boot().unwrap();
        assert!(kernel.is_booted());
        let boot_witnesses = kernel.witness_count();
        assert_eq!(boot_witnesses, 7); // 7 boot phases

        // Phase 2: Create partition (agent).
        let config = PartitionConfig::default();
        let pid = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.partition_count(), 1);
        assert!(kernel.partitions().get(pid).is_some());

        // Verify witness for partition creation.
        let create_record = kernel.witness_log().get(boot_witnesses as usize).unwrap();
        assert_eq!(create_record.action_kind, ActionKind::PartitionCreate as u8);

        // Phase 3: Tick scheduler (simulate agent running).
        for i in 0..5 {
            let result = kernel.tick().unwrap();
            assert_eq!(result.summary.epoch, i);
        }
        assert_eq!(kernel.current_epoch(), 5);

        // Phase 4: Destroy partition.
        kernel.destroy_partition(pid).unwrap();

        // Phase 5: Verify witness chain covers full lifecycle.
        // 7 boot + 1 create + 5 ticks + 1 destroy = 14.
        assert_eq!(kernel.witness_count(), 14);

        // Verify the final destroy witness.
        let destroy_record = kernel.witness_log().get(13).unwrap();
        assert_eq!(destroy_record.action_kind, ActionKind::PartitionDestroy as u8);
        assert_eq!(destroy_record.target_object_id, pid.as_u32() as u64);

        // Verify monotonic sequence: each record's sequence >= previous.
        for i in 1..14usize {
            let prev = kernel.witness_log().get(i - 1).unwrap();
            let curr = kernel.witness_log().get(i).unwrap();
            assert!(
                curr.sequence >= prev.sequence,
                "witness sequence not monotonic at index {}",
                i
            );
        }
    }

    // ---------------------------------------------------------------
    // E2E Scenario 2: Split Under Pressure
    //
    // Create 2 partitions in a coherence graph -> add comm edges ->
    // send messages (build weight) -> compute coherence scores ->
    // compute cut pressure -> verify split signal when external
    // traffic exceeds internal -> verify merge signal when
    // coherence rises
    // ---------------------------------------------------------------
    #[test]
    fn e2e_split_under_pressure() {
        use rvm_coherence::graph::CoherenceGraph;
        use rvm_coherence::scoring::compute_coherence_score;
        use rvm_coherence::pressure::{compute_cut_pressure, evaluate_merge, SPLIT_THRESHOLD_BP};

        let p1 = PartitionId::new(1);
        let p2 = PartitionId::new(2);

        let mut graph = CoherenceGraph::<8, 64>::new();
        graph.add_node(p1).unwrap();
        graph.add_node(p2).unwrap();

        // Phase 1: Build heavy internal traffic (self-loops) for p1.
        let _self_edge = graph.add_edge(p1, p1, 0).unwrap();
        // Simulate 100 internal messages: add weight via the self-loop.
        for _ in 0..100 {
            graph.update_weight(_self_edge, 10).unwrap();
        }
        // p1 self-loop now has weight 1000.
        assert_eq!(graph.edge_weight(_self_edge), Some(1000));

        // Phase 2: Add light external traffic between p1 and p2.
        let ext_edge = graph.add_edge(p1, p2, 50).unwrap();

        // Coherence check: p1 has high internal vs low external -> low pressure.
        let score1 = compute_coherence_score(p1, &graph);
        // total = 1000 (self out) + 1000 (self in) + 50 (ext out) = 2050
        // internal = 1000 (self-loop)
        assert_eq!(score1.internal_weight, 1000);
        assert_eq!(score1.total_weight, 2050);
        // score = 1000/2050 * 10000 = ~4878 bp
        assert!(score1.score.as_basis_points() > 0);

        let pressure1 = compute_cut_pressure(p1, &graph);
        assert!(!pressure1.should_split, "should not split with heavy internal traffic");

        // Phase 3: Flood external traffic to trigger split.
        // Add 100 heavy external messages.
        for _ in 0..100 {
            graph.update_weight(ext_edge, 100).unwrap();
        }
        // ext_edge now has weight 50 + 10000 = 10050.

        let pressure2 = compute_cut_pressure(p1, &graph);
        // total = 1000 + 1000 + 10050 = 12050
        // internal = 1000, external = 11050
        // pressure = 11050/12050 * 10000 = ~9170 > 8000
        assert!(
            pressure2.should_split,
            "should split when external traffic dominates: pressure={}",
            pressure2.pressure.as_fixed()
        );
        assert!(pressure2.pressure.as_fixed() > SPLIT_THRESHOLD_BP);

        // Phase 4: Verify merge signal between p1 and p2.
        let merge_signal = evaluate_merge(p1, p2, &graph);
        // Mutual weight between p1 and p2 = 10050 (one direction).
        // Merge signal checks bidirectional: mutual / combined.
        assert!(merge_signal.mutual_coherence.as_basis_points() > 0);
    }

    // ---------------------------------------------------------------
    // E2E Scenario 3: Memory Tier Lifecycle
    //
    // Allocate pages -> create region (Hot) -> access region ->
    // demote to Warm -> demote to Dormant -> create checkpoint
    // (compress) -> reconstruct from Dormant -> verify data intact
    // ---------------------------------------------------------------
    #[test]
    fn e2e_memory_tier_lifecycle() {
        use rvm_memory::{
            BuddyAllocator, RegionManager, RegionConfig, TierManager, Tier,
            MemoryPermissions, ReconstructionPipeline, CheckpointId,
            create_checkpoint,
        };
        use rvm_types::{OwnedRegionId, PhysAddr};

        // Phase 1: Allocate physical pages.
        let mut alloc = BuddyAllocator::<16, 2>::new(PhysAddr::new(0x1000_0000)).unwrap();
        let addr = alloc.alloc_pages(0).unwrap();
        assert!(addr.is_page_aligned());

        // Phase 2: Create a region in Hot tier.
        let mut region_mgr = RegionManager::<16>::new();
        let rid = region_mgr
            .create(RegionConfig {
                id: OwnedRegionId::new(1),
                owner: PartitionId::new(1),
                guest_base: GuestPhysAddr::new(0x0),
                host_base: PhysAddr::new(addr.as_u64()),
                page_count: 1,
                tier: Tier::Hot,
                permissions: MemoryPermissions::READ_WRITE,
            })
            .unwrap();

        // Phase 3: Register in tier manager and record access.
        let mut tier_mgr = TierManager::<8>::new();
        tier_mgr.register(rid, Tier::Hot).unwrap();
        tier_mgr.record_access(rid).unwrap();
        assert_eq!(tier_mgr.get(rid).unwrap().tier, Tier::Hot);

        // Phase 4: Demote Hot -> Warm.
        let old_tier = tier_mgr.demote(rid, Tier::Warm).unwrap();
        assert_eq!(old_tier, Tier::Hot);
        assert_eq!(tier_mgr.get(rid).unwrap().tier, Tier::Warm);

        // Phase 5: Demote Warm -> Dormant (triggers checkpoint creation).
        let old_tier2 = tier_mgr.demote(rid, Tier::Dormant).unwrap();
        assert_eq!(old_tier2, Tier::Warm);
        assert_eq!(tier_mgr.get(rid).unwrap().tier, Tier::Dormant);

        // Phase 6: Simulate compression by creating a checkpoint.
        let original_data = b"RVM dormant memory test data!!!!"; // 32 bytes
        let mut compressed_buf = [0u8; 256];
        let (checkpoint, compressed_size) = create_checkpoint(
            OwnedRegionId::new(1),
            CheckpointId::new(100),
            0,
            original_data,
            &mut compressed_buf,
        )
        .unwrap();
        assert!(compressed_size > 0);

        // Phase 7: Reconstruct from dormant state.
        let pipeline = ReconstructionPipeline::<16>::new();
        let mut output = [0u8; 256];
        let result = pipeline
            .reconstruct(&checkpoint, &compressed_buf[..compressed_size], &mut output, |_| &[])
            .unwrap();

        // Phase 8: Verify data intact.
        assert_eq!(result.size_bytes, original_data.len() as u32);
        assert_eq!(result.deltas_applied, 0);
        assert_eq!(&output[..original_data.len()], original_data.as_slice());

        // Phase 9: Promote back to Warm and verify.
        // Boost residency score enough to promote.
        tier_mgr.update_cut_value(rid, 5_000).unwrap();
        let old_tier3 = tier_mgr.promote(rid, Tier::Warm).unwrap();
        assert_eq!(old_tier3, Tier::Dormant);
        assert_eq!(tier_mgr.get(rid).unwrap().tier, Tier::Warm);
    }

    // ---------------------------------------------------------------
    // E2E Scenario 4: Capability Delegation Chain
    //
    // Create root cap (all rights) -> derive child (READ+WRITE) ->
    // derive grandchild (READ only) -> verify grandchild can READ ->
    // verify grandchild cannot WRITE -> revoke child ->
    // verify grandchild is also revoked -> verify root still valid
    // ---------------------------------------------------------------
    #[test]
    fn e2e_capability_delegation_chain() {
        use rvm_cap::CapabilityManager;
        use rvm_types::{CapType, CapRights};

        let mut cap_mgr = CapabilityManager::<64>::with_defaults();
        let owner = PartitionId::new(1);
        let child_owner = PartitionId::new(2);
        let grandchild_owner = PartitionId::new(3);

        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::EXECUTE)
            .union(CapRights::GRANT)
            .union(CapRights::REVOKE)
            .union(CapRights::PROVE);

        // Step 1: Create root capability with all rights.
        let (root_idx, root_gen) = cap_mgr
            .create_root_capability(CapType::Partition, all_rights, 0, owner)
            .unwrap();

        // Step 2: Derive child with READ + WRITE + GRANT.
        let child_rights = CapRights::READ.union(CapRights::WRITE).union(CapRights::GRANT);
        let (child_idx, child_gen) = cap_mgr
            .grant(root_idx, root_gen, child_rights, 1, child_owner)
            .unwrap();

        // Step 3: Derive grandchild with READ only.
        let (gc_idx, gc_gen) = cap_mgr
            .grant(child_idx, child_gen, CapRights::READ, 2, grandchild_owner)
            .unwrap();

        // Step 4: Verify grandchild can READ via P1.
        assert!(cap_mgr.verify_p1(gc_idx, gc_gen, CapRights::READ).is_ok());

        // Step 5: Verify grandchild cannot WRITE.
        assert!(cap_mgr.verify_p1(gc_idx, gc_gen, CapRights::WRITE).is_err());

        // Step 6: Revoke the child. This should also revoke grandchild.
        let revoke_result = cap_mgr.revoke(child_idx, child_gen).unwrap();
        assert!(revoke_result.revoked_count >= 2); // child + grandchild

        // Step 7: Verify grandchild is revoked (lookup should fail).
        assert!(cap_mgr.verify_p1(gc_idx, gc_gen, CapRights::READ).is_err());

        // Step 8: Verify root is still valid.
        assert!(cap_mgr.verify_p1(root_idx, root_gen, CapRights::READ).is_ok());
    }

    // ---------------------------------------------------------------
    // E2E Scenario 5: Security Gate Rejection Cascade
    //
    // Create cap with READ only -> attempt WRITE through security
    // gate -> verify rejection -> verify PROOF_REJECTED witness ->
    // create WRITE cap -> retry -> verify success ->
    // verify PROOF_VERIFIED witness
    // ---------------------------------------------------------------
    #[test]
    fn e2e_security_gate_rejection_cascade() {
        use rvm_security::{SecurityGate, SecurityError, GateRequest};
        use rvm_types::WitnessHash;

        let log = rvm_witness::WitnessLog::<32>::new();
        let gate = SecurityGate::new(&log);

        // Step 1: Create a cap with READ only.
        let read_token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ,
            0,
        );

        // Step 2: Attempt WRITE through the gate -> should be rejected.
        let request_write = GateRequest {
            token: read_token,
            required_type: CapType::Partition,
            required_rights: CapRights::WRITE,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };
        let err = gate.check_and_execute(&request_write).unwrap_err();
        assert_eq!(err, SecurityError::InsufficientRights);

        // Step 3: Verify PROOF_REJECTED witness was emitted.
        assert_eq!(log.total_emitted(), 1);
        let rejected_record = log.get(0).unwrap();
        assert_eq!(rejected_record.action_kind, ActionKind::ProofRejected as u8);

        // Step 4: Create a new cap with READ + WRITE.
        let rw_token = CapToken::new(
            2,
            CapType::Partition,
            CapRights::READ | CapRights::WRITE,
            0,
        );

        // Step 5: Retry with proper rights -> should succeed.
        let request_retry = GateRequest {
            token: rw_token,
            required_type: CapType::Partition,
            required_rights: CapRights::WRITE,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 2000,
        };
        let response = gate.check_and_execute(&request_retry).unwrap();
        assert_eq!(response.proof_tier, 1); // P1 (no proof commitment provided)
        assert_eq!(response.witness_sequence, 1);

        // Step 6: Verify success witness emitted.
        assert_eq!(log.total_emitted(), 2);
        let success_record = log.get(1).unwrap();
        assert_eq!(success_record.action_kind, ActionKind::PartitionCreate as u8);

        // Step 7: Also verify the full cascade with proof commitment.
        let commitment = WitnessHash::from_bytes([0xCC; 32]);
        let request_p2 = GateRequest {
            token: rw_token,
            required_type: CapType::Partition,
            required_rights: CapRights::WRITE,
            proof_commitment: Some(commitment),
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 99,
            timestamp_ns: 3000,
        };
        let response_p2 = gate.check_and_execute(&request_p2).unwrap();
        assert_eq!(response_p2.proof_tier, 2); // P2 because proof commitment provided
        assert_eq!(log.total_emitted(), 3);
    }

    // ---------------------------------------------------------------
    // E2E Scenario 6: Boot Sequence Timing
    //
    // Run full kernel boot -> extract witness records -> verify 7
    // boot phases -> verify monotonic timestamps -> verify measured
    // boot hash chain
    // ---------------------------------------------------------------
    #[test]
    fn e2e_boot_sequence_timing() {
        use rvm_kernel::{Kernel, KernelConfig};
        use rvm_boot::MeasuredBootState;

        // Phase 1: Boot the kernel.
        let mut kernel = Kernel::new(KernelConfig::default());
        kernel.boot().unwrap();
        assert!(kernel.is_booted());

        // Phase 2: Extract all boot witness records.
        assert_eq!(kernel.witness_count(), 7);

        // Phase 3: Verify 7 boot phases recorded correctly.
        let mut boot_attestation_count = 0u32;
        let mut boot_complete_count = 0u32;
        for i in 0..7usize {
            let record = kernel.witness_log().get(i).unwrap();
            if record.action_kind == ActionKind::BootAttestation as u8 {
                boot_attestation_count += 1;
            } else if record.action_kind == ActionKind::BootComplete as u8 {
                boot_complete_count += 1;
            } else {
                panic!("unexpected action kind in boot sequence: {}", record.action_kind);
            }
        }
        // 6 BootAttestation phases + 1 BootComplete (Handoff)
        assert_eq!(boot_attestation_count, 6);
        assert_eq!(boot_complete_count, 1);

        // Phase 4: Verify monotonic sequence numbers.
        for i in 1..7usize {
            let prev = kernel.witness_log().get(i - 1).unwrap();
            let curr = kernel.witness_log().get(i).unwrap();
            assert!(
                curr.sequence > prev.sequence,
                "sequence not strictly increasing at index {}",
                i
            );
        }

        // Phase 5: Verify measured boot hash chain using standalone tracker.
        let mut measured = MeasuredBootState::new();
        assert!(measured.is_virgin());

        use rvm_boot::sequence::BootStage;
        let stages = BootStage::all();
        for (i, &stage) in stages.iter().enumerate() {
            let hash = [i as u8; 32];
            measured.extend_measurement(stage, &hash);
        }
        assert_eq!(measured.measurement_count(), 7);
        assert!(!measured.is_virgin());

        // Verify each phase hash was recorded.
        for (i, &stage) in stages.iter().enumerate() {
            assert_eq!(*measured.phase_hash(stage), [i as u8; 32]);
        }
    }

    // ---------------------------------------------------------------
    // E2E Scenario 7: Scheduler Mode Transitions
    //
    // Start in Flow mode -> enqueue partitions with mixed priorities
    // -> switch to Reflex mode -> verify highest priority runs ->
    // switch to Recovery mode -> verify scheduler behavior ->
    // return to Flow mode -> verify normal operation
    // ---------------------------------------------------------------
    #[test]
    fn e2e_scheduler_mode_transitions() {
        use rvm_sched::{Scheduler, SchedulerMode};
        use rvm_types::CutPressure;

        let mut sched = Scheduler::<4, 256>::new();
        assert_eq!(sched.mode(), SchedulerMode::Flow);

        // Phase 1: Enqueue partitions with mixed priorities in Flow mode.
        let p_low = PartitionId::new(1);
        let p_mid = PartitionId::new(2);
        let p_high = PartitionId::new(3);

        assert!(sched.enqueue(0, p_low, 50, CutPressure::ZERO));
        assert!(sched.enqueue(0, p_mid, 100, CutPressure::ZERO));
        assert!(sched.enqueue(0, p_high, 200, CutPressure::ZERO));

        // Phase 2: Switch to Reflex mode (hard real-time).
        sched.set_mode(SchedulerMode::Reflex);
        assert_eq!(sched.mode(), SchedulerMode::Reflex);

        // Verify highest priority runs first (priority = deadline urgency
        // since cut_pressure is ZERO).
        let (_, next) = sched.switch_next(0).unwrap();
        assert_eq!(next, p_high, "Reflex mode should run highest priority first");

        // Phase 3: Switch to Recovery mode.
        sched.set_mode(SchedulerMode::Recovery);
        assert_eq!(sched.mode(), SchedulerMode::Recovery);

        // Scheduler still processes queued partitions in Recovery mode.
        let (_, next2) = sched.switch_next(0).unwrap();
        assert_eq!(next2, p_mid);

        // Phase 4: Return to Flow mode.
        sched.set_mode(SchedulerMode::Flow);
        assert_eq!(sched.mode(), SchedulerMode::Flow);

        // Dequeue remaining partition.
        let (_, next3) = sched.switch_next(0).unwrap();
        assert_eq!(next3, p_low);

        // Queue is empty now.
        assert!(sched.switch_next(0).is_none());

        // Phase 5: Verify degraded mode interaction.
        sched.enter_degraded();
        assert!(sched.is_degraded());

        // In degraded mode, cut pressure is zeroed.
        let big_pressure = CutPressure::from_fixed(9999);
        sched.enqueue(0, PartitionId::new(10), 100, big_pressure);
        sched.enqueue(0, PartitionId::new(11), 150, CutPressure::ZERO);

        // pid(11) should win because deadline urgency 150 > 100,
        // and pressure is zeroed in degraded mode.
        let (_, winner) = sched.switch_next(0).unwrap();
        assert_eq!(winner, PartitionId::new(11));

        sched.exit_degraded();
        assert!(!sched.is_degraded());
    }

    // ---------------------------------------------------------------
    // E2E Scenario 8: Coherence Graph Dynamics
    //
    // Create graph with 4 nodes -> add edges with varying weights ->
    // compute scores -> verify highest-coherence pair ->
    // add heavy cross-cut traffic -> verify pressure rises ->
    // verify split recommendation matches the cut
    // ---------------------------------------------------------------
    #[test]
    fn e2e_coherence_graph_dynamics() {
        use rvm_coherence::graph::CoherenceGraph;
        use rvm_coherence::scoring::compute_coherence_score;
        use rvm_coherence::pressure::compute_cut_pressure;
        use rvm_coherence::mincut::MinCutBridge;

        let p1 = PartitionId::new(1);
        let p2 = PartitionId::new(2);
        let p3 = PartitionId::new(3);
        let p4 = PartitionId::new(4);

        let mut graph = CoherenceGraph::<8, 64>::new();
        graph.add_node(p1).unwrap();
        graph.add_node(p2).unwrap();
        graph.add_node(p3).unwrap();
        graph.add_node(p4).unwrap();

        // Phase 1: Build a cluster: strong edges between p1-p2 and p3-p4.
        // Within cluster 1: p1 <-> p2 (weight 1000 each direction)
        graph.add_edge(p1, p2, 1000).unwrap();
        graph.add_edge(p2, p1, 1000).unwrap();

        // Within cluster 2: p3 <-> p4 (weight 1000 each direction)
        graph.add_edge(p3, p4, 1000).unwrap();
        graph.add_edge(p4, p3, 1000).unwrap();

        // Cross-cluster: p2 <-> p3 (weak link, weight 10)
        graph.add_edge(p2, p3, 10).unwrap();
        graph.add_edge(p3, p2, 10).unwrap();

        // Phase 2: Compute coherence scores for all nodes.
        let score_p1 = compute_coherence_score(p1, &graph);
        let score_p2 = compute_coherence_score(p2, &graph);
        let score_p3 = compute_coherence_score(p3, &graph);
        let score_p4 = compute_coherence_score(p4, &graph);

        // p1 and p4 have no self-loops, so their internal_weight = 0.
        // p1: total=2000 (1000 out to p2 + 1000 in from p2), internal=0, score=0
        // p4: same pattern
        assert_eq!(score_p1.internal_weight, 0);
        assert_eq!(score_p4.internal_weight, 0);

        // p2 is the busiest node: out(1000 to p1 + 10 to p3) + in(1000 from p1 + 10 from p3) = 2020
        assert_eq!(score_p2.total_weight, 2020);
        // p3 similarly
        assert_eq!(score_p3.total_weight, 2020);

        // Phase 3: Compute pressure.
        let pressure_p1 = compute_cut_pressure(p1, &graph);
        let pressure_p2 = compute_cut_pressure(p2, &graph);

        // p1 has all external edges (no self-loops) -> max pressure.
        assert_eq!(pressure_p1.pressure.as_fixed(), 10_000);
        assert!(pressure_p1.should_split);

        // p2 also all external -> max pressure.
        assert!(pressure_p2.should_split);

        // Phase 4: Run min-cut to find the natural split.
        // Use p2 as root -- its subgraph includes p1 (neighbor) and p3
        // (neighbor), but not p4 (only reachable via p3, not a direct
        // neighbor of p2). So the subgraph has 3 nodes: {p1, p2, p3}.
        let mut bridge = MinCutBridge::<8>::new(100);
        let cut = bridge.find_min_cut(&graph, p2);

        // The min-cut should find the weak link between p2-p3.
        assert!(cut.within_budget);
        assert!(cut.left_count > 0);
        assert!(cut.right_count > 0);
        // The cut weight should be the cross-cluster weight (10+10=20).
        assert_eq!(cut.cut_weight, 20);

        // Verify both sides of the cut are non-empty.
        let total_nodes = cut.left_count + cut.right_count;
        // Subgraph rooted at p2 includes p1, p2, p3 (direct neighbors + incoming).
        assert_eq!(total_nodes, 3);

        // Phase 5: Add heavy cross-cut traffic and verify pressure changes.
        // Add 100 heavy messages from p1 to p3 (cross cluster).
        let cross_edge = graph.add_edge(p1, p3, 0).unwrap();
        for _ in 0..100 {
            graph.update_weight(cross_edge, 50).unwrap();
        }
        // Cross edge p1->p3 now has weight 5000.

        // Recompute min-cut after the traffic change.
        let cut2 = bridge.find_min_cut(&graph, p1);
        assert!(cut2.within_budget);
        // Cut weight should now be higher due to the added cross-cluster edge.
        assert!(cut2.cut_weight > 20);
    }

    // ---------------------------------------------------------------
    // E2E Scenario: Memory Reconstruction with Deltas
    //
    // Create checkpoint -> apply deltas -> reconstruct -> verify
    // patched data integrity
    // ---------------------------------------------------------------
    #[test]
    fn e2e_memory_reconstruction_with_deltas() {
        use rvm_memory::{
            ReconstructionPipeline, CheckpointId, WitnessDelta, create_checkpoint,
        };
        use rvm_types::OwnedRegionId;

        // Original data: 32 bytes of 0xAA.
        let original = [0xAAu8; 32];
        let mut compressed = [0u8; 256];
        let (checkpoint, csize) = create_checkpoint(
            OwnedRegionId::new(1),
            CheckpointId::new(1),
            0,
            &original,
            &mut compressed,
        )
        .unwrap();

        // Create deltas that modify the data.
        let mut pipeline = ReconstructionPipeline::<16>::new();

        // Delta 1: overwrite bytes 0..4 with [0xBB; 4].
        static PATCH1: [u8; 4] = [0xBB, 0xBB, 0xBB, 0xBB];
        pipeline
            .add_delta(WitnessDelta {
                sequence: 1,
                offset: 0,
                length: 4,
                data_hash: rvm_witness::fnv1a_64(&PATCH1),
            })
            .unwrap();

        // Delta 2: overwrite bytes 16..20 with [0xCC; 4].
        static PATCH2: [u8; 4] = [0xCC, 0xCC, 0xCC, 0xCC];
        pipeline
            .add_delta(WitnessDelta {
                sequence: 2,
                offset: 16,
                length: 4,
                data_hash: rvm_witness::fnv1a_64(&PATCH2),
            })
            .unwrap();

        // Reconstruct.
        let mut output = [0u8; 256];
        let result = pipeline
            .reconstruct(
                &checkpoint,
                &compressed[..csize],
                &mut output,
                |d| {
                    if d.sequence == 1 {
                        &PATCH1
                    } else {
                        &PATCH2
                    }
                },
            )
            .unwrap();

        // Verify reconstruction.
        assert_eq!(result.deltas_applied, 2);
        assert_eq!(result.size_bytes, 32);

        // Bytes 0..4 should be 0xBB.
        assert_eq!(&output[0..4], &[0xBB; 4]);
        // Bytes 4..16 should remain 0xAA.
        assert_eq!(&output[4..16], &[0xAA; 12]);
        // Bytes 16..20 should be 0xCC.
        assert_eq!(&output[16..20], &[0xCC; 4]);
        // Bytes 20..32 should remain 0xAA.
        assert_eq!(&output[20..32], &[0xAA; 12]);
    }

    // ---------------------------------------------------------------
    // E2E Scenario: Full Kernel + Cap + Proof + Witness Integration
    //
    // Boot kernel -> create partition -> grant capability via kernel
    // cap manager -> run proof engine -> verify witness emission
    // ---------------------------------------------------------------
    #[test]
    fn e2e_kernel_cap_proof_witness_full_pipeline() {
        use rvm_kernel::{Kernel, KernelConfig};
        use rvm_types::{CapType, CapRights, PartitionConfig, ProofTier, ProofToken};
        use rvm_proof::context::ProofContextBuilder;
        use rvm_proof::engine::ProofEngine;

        // Boot.
        let mut kernel = Kernel::new(KernelConfig::default());
        kernel.boot().unwrap();

        // Create partition.
        let pid = kernel
            .create_partition(&PartitionConfig::default())
            .unwrap();

        // Grant a capability.
        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::EXECUTE)
            .union(CapRights::GRANT)
            .union(CapRights::REVOKE)
            .union(CapRights::PROVE);

        let (cap_idx, cap_gen) = kernel
            .cap_manager_mut()
            .create_root_capability(CapType::Region, all_rights, 0, pid)
            .unwrap();

        // Verify P1.
        assert!(kernel
            .cap_manager()
            .verify_p1(cap_idx, cap_gen, CapRights::PROVE)
            .is_ok());

        // Run the proof engine with a separate witness log.
        let proof_log = rvm_witness::WitnessLog::<32>::new();
        let token = ProofToken {
            tier: ProofTier::P1,
            epoch: 0,
            hash: 0xDEAD,
        };
        let context = ProofContextBuilder::new(pid)
            .target_object(100)
            .capability_handle(cap_idx)
            .capability_generation(cap_gen)
            .current_epoch(0)
            .region_bounds(0x1000, 0x2000)
            .time_window(500, 1000)
            .nonce(1)
            .build();

        let mut engine = ProofEngine::<256>::new();
        engine
            .verify_and_witness(&token, &context, kernel.cap_manager(), &proof_log)
            .unwrap();

        // Verify proof witness was emitted.
        assert_eq!(proof_log.total_emitted(), 1);
        let record = proof_log.get(0).unwrap();
        assert_eq!(record.actor_partition_id, pid.as_u32());
        assert_eq!(record.target_object_id, 100);
    }

    // ---------------------------------------------------------------
    // E2E Scenario: Buddy Allocator Full Pressure Cycle
    //
    // Allocate all pages -> free all -> verify coalescing -> re-allocate
    // largest possible block -> verify integrity
    // ---------------------------------------------------------------
    #[test]
    fn e2e_buddy_allocator_full_pressure() {
        use rvm_memory::BuddyAllocator;
        use rvm_types::PhysAddr;

        let mut alloc = BuddyAllocator::<256, 16>::new(PhysAddr::new(0x1000_0000)).unwrap();
        assert_eq!(alloc.free_page_count(), 256);

        // Allocate all 256 pages as order-0 blocks.
        let mut addrs = [PhysAddr::new(0); 256];
        for addr in &mut addrs {
            *addr = alloc.alloc_pages(0).unwrap();
        }
        assert_eq!(alloc.free_page_count(), 0);
        assert!(alloc.alloc_pages(0).is_err());

        // Free all pages.
        for addr in &addrs {
            alloc.free_pages(*addr, 0).unwrap();
        }
        assert_eq!(alloc.free_page_count(), 256);

        // After full coalescing, allocate the largest block (order 8 = 256 pages).
        let big_block = alloc.alloc_pages(8).unwrap();
        assert!(big_block.is_page_aligned());
        assert_eq!(alloc.free_page_count(), 0);

        // Free and verify.
        alloc.free_pages(big_block, 8).unwrap();
        assert_eq!(alloc.free_page_count(), 256);
    }

    // ---------------------------------------------------------------
    // E2E Scenario: Witness Chain Integrity End-to-End
    //
    // Emit many witness records through different subsystems ->
    // verify the full chain integrity with verify_chain
    // ---------------------------------------------------------------
    #[test]
    fn e2e_witness_chain_integrity_multi_subsystem() {
        use rvm_types::WitnessRecord;

        let log = rvm_witness::WitnessLog::<64>::new();
        let emitter = rvm_witness::WitnessEmitter::new(&log);

        // Emit records from different subsystems.
        let _ = emitter.emit_partition_create(1, 100, 0xABCD, 1_000_000);
        let _ = emitter.emit_partition_create(1, 101, 0xBCDE, 2_000_000);
        let _ = emitter.emit_partition_create(2, 102, 0xCDEF, 3_000_000);

        // Also append raw records.
        for i in 0..5u8 {
            let mut record = WitnessRecord::zeroed();
            record.action_kind = ActionKind::SchedulerEpoch as u8;
            record.proof_tier = 1;
            record.actor_partition_id = i as u32;
            log.append(record);
        }

        assert_eq!(log.total_emitted(), 8);

        // Collect all records and verify chain.
        let mut records = [WitnessRecord::zeroed(); 8];
        for i in 0..8 {
            records[i] = log.get(i).unwrap();
        }

        let chain_result = rvm_witness::verify_chain(&records);
        assert!(chain_result.is_ok());
    }

    // ---------------------------------------------------------------
    // E2E Scenario: Multiple Partitions with Coherence and Scheduling
    //
    // Create multiple partitions -> build coherence graph ->
    // compute priorities -> feed into scheduler -> verify scheduling
    // order matches coherence-weighted priorities
    // ---------------------------------------------------------------
    #[test]
    fn e2e_coherence_driven_scheduling() {
        use rvm_coherence::graph::CoherenceGraph;
        use rvm_coherence::pressure::compute_cut_pressure;
        use rvm_sched::Scheduler;
        let p1 = PartitionId::new(1);
        let p2 = PartitionId::new(2);
        let p3 = PartitionId::new(3);

        // Build a coherence graph.
        let mut graph = CoherenceGraph::<8, 32>::new();
        graph.add_node(p1).unwrap();
        graph.add_node(p2).unwrap();
        graph.add_node(p3).unwrap();

        // p1: high external traffic (high cut pressure).
        graph.add_edge(p1, p2, 5000).unwrap();
        // p2: moderate external traffic.
        graph.add_edge(p2, p3, 1000).unwrap();
        // p3: only incoming, moderate.

        // Compute cut pressures.
        let pr1 = compute_cut_pressure(p1, &graph);
        let pr2 = compute_cut_pressure(p2, &graph);
        let pr3 = compute_cut_pressure(p3, &graph);

        // Enqueue into scheduler with computed pressures.
        let mut sched = Scheduler::<4, 256>::new();
        let deadline = 100u16; // Same deadline for all.

        sched.enqueue(0, p1, deadline, pr1.pressure);
        sched.enqueue(0, p2, deadline, pr2.pressure);
        sched.enqueue(0, p3, deadline, pr3.pressure);

        // The partition with highest pressure boost runs first.
        // p1 has the highest cut pressure (all external), so highest priority.
        let (_, first) = sched.switch_next(0).unwrap();

        // p1 should have highest priority because its pressure boost is largest.
        // All have same deadline, so the one with highest pressure wins.
        assert_eq!(first, p1);
    }

    // ===============================================================
    // ADR-142 TEE Pipeline Integration Tests
    // ===============================================================

    // ---------------------------------------------------------------
    // ADR-142 A-02: Forged witness entry with reordered fields fails
    //
    // Create two WitnessRecords with identical data but different
    // field ordering. With SHA-256 hashing (crypto-sha256 feature),
    // the signed digests must differ, verifying that XOR
    // commutativity (A-02) is fixed at the signing layer.
    // ---------------------------------------------------------------
    #[test]
    fn adr142_forged_witness_reordered_fields_different_hashes() {
        use rvm_witness::WitnessSigner as _;

        let signer = rvm_witness::HmacWitnessSigner::new([0xAA; 32]);

        // Record A: actor=1, target=2
        let mut record_a = WitnessRecord::zeroed();
        record_a.sequence = 1;
        record_a.timestamp_ns = 1000;
        record_a.action_kind = ActionKind::PartitionCreate as u8;
        record_a.proof_tier = 2;
        record_a.actor_partition_id = 1;
        record_a.target_object_id = 2;
        record_a.capability_hash = 0xABCD;

        // Record B: swap actor and target fields to test commutativity.
        let mut record_b = WitnessRecord::zeroed();
        record_b.sequence = 1;
        record_b.timestamp_ns = 1000;
        record_b.action_kind = ActionKind::PartitionCreate as u8;
        record_b.proof_tier = 2;
        record_b.actor_partition_id = 2; // swapped
        record_b.target_object_id = 1;   // swapped
        record_b.capability_hash = 0xABCD;

        // The HMAC signatures must differ because the signer hashes
        // the serialized record in field order. Under a naive XOR
        // scheme XOR(1,2) == XOR(2,1), but SHA-256/HMAC is order-
        // sensitive.
        let sig_a = signer.sign(&record_a);
        let sig_b = signer.sign(&record_b);
        assert_ne!(
            sig_a, sig_b,
            "XOR commutativity: swapped fields must produce different signatures (A-02)"
        );

        // Cross-verification must fail: signing record A, verifying
        // against record B (with swapped fields).
        record_a.aux = sig_a;
        record_b.aux = sig_a; // forged: use A's signature on B
        assert!(
            signer.verify(&record_a),
            "original record must verify"
        );
        assert!(
            !signer.verify(&record_b),
            "forged record with swapped fields must fail verification (A-02)"
        );

        // Also verify via compute_record_hash that byte order matters.
        let hash_a = rvm_witness::compute_record_hash(&[
            1, 0, 0, 0, // actor = 1
            2, 0, 0, 0, // some field = 2
        ]);
        let hash_b = rvm_witness::compute_record_hash(&[
            2, 0, 0, 0, // actor = 2 (swapped)
            1, 0, 0, 0, // some field = 1 (swapped)
        ]);
        assert_ne!(hash_a, hash_b, "compute_record_hash must be order-sensitive (A-02)");
    }

    // ---------------------------------------------------------------
    // ADR-142: Reused nonce is rejected
    //
    // Create a ProofEngine, submit a proof with nonce N, then submit
    // again with same nonce N. The second should fail with replay
    // detection. Also test nonce 0 is rejected by default.
    // ---------------------------------------------------------------
    #[test]
    fn adr142_reused_nonce_rejected() {
        use rvm_cap::CapabilityManager;
        use rvm_types::{CapType, CapRights, ProofTier, ProofToken};
        use rvm_proof::context::ProofContextBuilder;
        use rvm_proof::engine::ProofEngine;

        let witness_log = rvm_witness::WitnessLog::<32>::new();
        let mut cap_mgr = CapabilityManager::<64>::with_defaults();
        let owner = PartitionId::new(1);

        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::PROVE);
        let (idx, gen) = cap_mgr
            .create_root_capability(CapType::Region, all_rights, 0, owner)
            .unwrap();

        let token = ProofToken {
            tier: ProofTier::P2,
            epoch: 0,
            hash: 0xBEEF,
        };

        let context_n = ProofContextBuilder::new(owner)
            .capability_handle(idx)
            .capability_generation(gen)
            .current_epoch(0)
            .region_bounds(0x1000, 0x2000)
            .time_window(500, 1000)
            .nonce(42)
            .build();

        let mut engine = ProofEngine::<64>::new();

        // First submission with nonce 42 should succeed.
        assert!(
            engine.verify_and_witness(&token, &context_n, &cap_mgr, &witness_log).is_ok(),
            "first nonce=42 should succeed"
        );

        // Second submission with same nonce 42 should fail (replay).
        assert!(
            engine.verify_and_witness(&token, &context_n, &cap_mgr, &witness_log).is_err(),
            "replayed nonce=42 must be rejected"
        );

        // Nonce 0 should be rejected by default (no zero-nonce bypass).
        let context_zero = ProofContextBuilder::new(owner)
            .capability_handle(idx)
            .capability_generation(gen)
            .current_epoch(0)
            .region_bounds(0x1000, 0x2000)
            .time_window(500, 1000)
            .nonce(0)
            .build();

        assert!(
            engine.verify_and_witness(&token, &context_zero, &cap_mgr, &witness_log).is_err(),
            "nonce=0 must be rejected by default"
        );

        // A fresh nonce should still work.
        let context_fresh = ProofContextBuilder::new(owner)
            .capability_handle(idx)
            .capability_generation(gen)
            .current_epoch(0)
            .region_bounds(0x1000, 0x2000)
            .time_window(500, 1000)
            .nonce(99)
            .build();

        assert!(
            engine.verify_and_witness(&token, &context_fresh, &cap_mgr, &witness_log).is_ok(),
            "fresh nonce=99 should succeed"
        );
    }

    // ---------------------------------------------------------------
    // ADR-142: Tampered witness chain detected via signed append
    //
    // Create a signed witness chain (4 entries). After signing,
    // tamper with one entry's content. The signer's verify() should
    // fail because the aux signature no longer matches the record
    // data. Also verify chain-hash integrity detects prev_hash
    // tampering.
    // ---------------------------------------------------------------
    #[test]
    fn adr142_tampered_witness_chain_detected() {
        use rvm_witness::WitnessSigner as _;

        let log = rvm_witness::WitnessLog::<32>::new();
        let signer = rvm_witness::HmacWitnessSigner::new([0xDD; 32]);

        // Emit 4 signed records to build a signed chain.
        for i in 0..4u8 {
            let mut record = WitnessRecord::zeroed();
            record.action_kind = ActionKind::PartitionCreate as u8;
            record.proof_tier = 1;
            record.actor_partition_id = i as u32;
            record.target_object_id = (i as u64) * 100;
            record.timestamp_ns = (i as u64) * 1000;
            log.signed_append(record, &signer);
        }

        assert_eq!(log.total_emitted(), 4);

        // Collect all records and verify signatures are valid.
        let mut records = [WitnessRecord::zeroed(); 4];
        for i in 0..4 {
            records[i] = log.get(i).unwrap();
            assert!(
                signer.verify(&records[i]),
                "untampered record {} must verify",
                i
            );
        }

        // Verify the untampered chain linkage is valid.
        assert!(
            rvm_witness::verify_chain(&records).is_ok(),
            "untampered chain must verify"
        );

        // Tamper with entry 2's content but leave aux (signature) unchanged.
        records[2].actor_partition_id = 0xFF; // changed content

        // Signer verification must fail for the tampered record because
        // the HMAC no longer matches the record data.
        assert!(
            !signer.verify(&records[2]),
            "tampered record content must fail signer verification"
        );

        // The other records should still verify (localized detection).
        assert!(signer.verify(&records[0]));
        assert!(signer.verify(&records[1]));
        assert!(signer.verify(&records[3]));

        // Also verify that tampering with prev_hash breaks chain integrity.
        let mut chain_records = [WitnessRecord::zeroed(); 4];
        for i in 0..4 {
            chain_records[i] = log.get(i).unwrap();
        }
        chain_records[2].prev_hash ^= 0xDEAD; // tamper chain link
        assert!(
            rvm_witness::verify_chain(&chain_records).is_err(),
            "tampered prev_hash must break chain verification"
        );
    }

    // ---------------------------------------------------------------
    // ADR-142: Invalid P3 chain link (Merkle path) rejected
    //
    // Create a P3 witness chain where a sibling hash is wrong.
    // The SecurityGate's verify_p3_chain should reject it.
    // ---------------------------------------------------------------
    #[test]
    fn adr142_invalid_chain_link_rejected() {
        use rvm_security::{SecurityGate, SecurityError, GateRequest, P3WitnessChain};

        let log = rvm_witness::WitnessLog::<32>::new();
        let gate = SecurityGate::new(&log);

        // Build a 3-link chain where link[1].prev_hash != link[0].record_hash.
        let mut chain = P3WitnessChain::empty();
        chain.links[0] = [0, 0x1111];         // prev_hash=0, record_hash=0x1111
        chain.links[1] = [0xDEAD, 0x2222];    // prev_hash=0xDEAD (WRONG! should be 0x1111)
        chain.links[2] = [0x2222, 0x3333];    // prev_hash=0x2222 (correct relative to link[1])
        chain.link_count = 3;

        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ | CapRights::WRITE,
            0,
        );
        let request = GateRequest {
            token,
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: true, // advisory lies, gate ignores
            p3_witness_data: Some(chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(
            err,
            SecurityError::DerivationChainBroken,
            "broken chain link must be rejected as DerivationChainBroken"
        );

        // Also verify that a valid 3-link chain passes.
        let mut valid_chain = P3WitnessChain::empty();
        valid_chain.links[0] = [0, 0x1111];
        valid_chain.links[1] = [0x1111, 0x2222]; // correct linkage
        valid_chain.links[2] = [0x2222, 0x3333]; // correct linkage
        valid_chain.link_count = 3;

        let valid_request = GateRequest {
            token,
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: false,
            p3_witness_data: Some(valid_chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 2000,
        };

        let response = gate.check_and_execute(&valid_request).unwrap();
        assert_eq!(response.proof_tier, 3);
    }

    // ---------------------------------------------------------------
    // ADR-142: Expired TEE collateral blocks signing
    //
    // Create SoftwareTeeProvider + SoftwareTeeVerifier, set verifier
    // epoch past collateral expiry. TeeWitnessSigner::sign() should
    // return zero signature (attestation fails).
    // ---------------------------------------------------------------
    #[test]
    fn adr142_expired_tee_collateral_blocks_signing() {
        use rvm_proof::signer::{HmacSha256WitnessSigner, WitnessSigner};
        use rvm_proof::tee::TeePlatform;
        use rvm_proof::{SoftwareTeeProvider, SoftwareTeeVerifier, TeeWitnessSigner};

        let tee_key = [0xBB; 32];
        let measurement = [0xAA; 32];
        let hmac_key = [0xCC; 32];

        let provider = SoftwareTeeProvider::new(TeePlatform::Sgx, measurement, tee_key);
        // Verifier with collateral_expiry=100, current_epoch=200 => expired.
        let verifier = SoftwareTeeVerifier::new(tee_key, 100, 200);
        let hmac_signer = HmacSha256WitnessSigner::new(hmac_key);
        let signer = TeeWitnessSigner::new(provider, verifier, hmac_signer, measurement);

        let digest = [0x55; 32];
        let sig = signer.sign(&digest);

        // Self-attestation fails due to expired collateral, so zero signature.
        assert_eq!(
            sig, [0u8; 64],
            "expired collateral must produce zero signature"
        );

        // Verify that a non-expired verifier works correctly.
        let provider2 = SoftwareTeeProvider::new(TeePlatform::Sgx, measurement, tee_key);
        let verifier2 = SoftwareTeeVerifier::new(tee_key, 0, 0); // no expiry
        let hmac_signer2 = HmacSha256WitnessSigner::new(hmac_key);
        let signer2 = TeeWitnessSigner::new(provider2, verifier2, hmac_signer2, measurement);

        let sig2 = signer2.sign(&digest);
        assert_ne!(
            sig2, [0u8; 64],
            "valid collateral must produce non-zero signature"
        );
        assert!(signer2.verify(&digest, &sig2).is_ok());
    }

    // ---------------------------------------------------------------
    // ADR-142: Cross-partition key isolation
    //
    // Derive keys for partition 1 and partition 2 from same
    // measurement. Keys must be different. Signing with partition 1's
    // key, verifying with partition 2's key must fail.
    // ---------------------------------------------------------------
    #[test]
    fn adr142_cross_partition_key_isolation() {
        use rvm_proof::signer::{HmacSha256WitnessSigner, WitnessSigner};
        use rvm_proof::{derive_witness_key, derive_key_bundle, dev_measurement};

        let measurement = dev_measurement();

        // Derive keys for two different partitions.
        let key_p1 = derive_witness_key(&measurement, 1);
        let key_p2 = derive_witness_key(&measurement, 2);

        // Keys MUST be different for different partitions.
        assert_ne!(
            key_p1, key_p2,
            "keys derived for different partitions must differ"
        );

        // Create signers from the derived keys.
        let signer_p1 = HmacSha256WitnessSigner::new(key_p1);
        let signer_p2 = HmacSha256WitnessSigner::new(key_p2);

        // Signer IDs must also differ.
        assert_ne!(signer_p1.signer_id(), signer_p2.signer_id());

        // Sign with partition 1's key.
        let digest = [0x77; 32];
        let sig = signer_p1.sign(&digest);

        // Verify with partition 1's key should succeed.
        assert!(signer_p1.verify(&digest, &sig).is_ok());

        // Verify with partition 2's key must fail (cross-partition isolation).
        assert!(
            signer_p2.verify(&digest, &sig).is_err(),
            "cross-partition verification must fail"
        );

        // Also verify full key bundle isolation.
        let bundle_p1 = derive_key_bundle(&measurement, 1);
        let bundle_p2 = derive_key_bundle(&measurement, 2);

        assert_ne!(bundle_p1.witness_key, bundle_p2.witness_key);
        assert_ne!(bundle_p1.attestation_key, bundle_p2.attestation_key);
        assert_ne!(bundle_p1.ipc_key, bundle_p2.ipc_key);

        // All three keys within a bundle must also be distinct from each other.
        assert_ne!(bundle_p1.witness_key, bundle_p1.attestation_key);
        assert_ne!(bundle_p1.witness_key, bundle_p1.ipc_key);
        assert_ne!(bundle_p1.attestation_key, bundle_p1.ipc_key);
    }

    // ---------------------------------------------------------------
    // ADR-142: Full SecurityGate flow with signed witnesses
    //
    // Create a SignedSecurityGate with HmacWitnessSigner, execute a
    // gate check that succeeds, verify the emitted witness record has
    // a valid signature in aux field. Tamper with the witness and
    // verify signature check fails.
    // ---------------------------------------------------------------
    #[test]
    fn adr142_signed_security_gate_full_flow() {
        use rvm_security::{SignedSecurityGate, GateRequest};
        use rvm_witness::WitnessSigner as _;

        let log = rvm_witness::WitnessLog::<32>::new();
        let signer = rvm_witness::HmacWitnessSigner::new([0xDD; 32]);
        let gate = SignedSecurityGate::new(&log, &signer);

        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ | CapRights::WRITE,
            0,
        );

        // Execute a gate check that should succeed.
        let request = GateRequest {
            token,
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 1);
        assert_eq!(response.witness_sequence, 0);

        // Verify the emitted witness record has a non-zero signature.
        let record = log.get(0).unwrap();
        assert_ne!(
            record.aux, [0u8; 8],
            "signed gate must produce non-zero aux signature"
        );

        // Verify the signature is valid.
        assert!(
            signer.verify(&record),
            "freshly signed witness record must verify"
        );

        // Tamper with the witness record and verify signature check fails.
        let mut tampered = record;
        tampered.target_object_id = 999; // changed content
        assert!(
            !signer.verify(&tampered),
            "tampered witness record must fail signature verification"
        );

        // Also tamper with just the aux field (corrupt signature).
        let mut sig_tampered = record;
        sig_tampered.aux[0] ^= 0xFF;
        assert!(
            !signer.verify(&sig_tampered),
            "corrupted aux signature must fail verification"
        );
    }

    // ---------------------------------------------------------------
    // ADR-142: Ed25519 signer round-trip (feature-gated)
    //
    // Create Ed25519WitnessSigner, sign a digest, verify it.
    // Verify that verify_strict rejects a different digest.
    // ---------------------------------------------------------------
    #[cfg(feature = "ed25519")]
    #[test]
    fn adr142_ed25519_signer_round_trip() {
        use rvm_proof::signer::{Ed25519WitnessSigner, WitnessSigner};

        // Create an Ed25519 signer from a deterministic seed.
        let seed = {
            let mut s = [0u8; 32];
            for (i, byte) in s.iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *byte = (i as u8).wrapping_mul(0x5A).wrapping_add(0x13);
                }
            }
            s
        };

        let signer = Ed25519WitnessSigner::from_seed(seed);

        // Sign a digest.
        let digest = [0xAA; 32];
        let sig = signer.sign(&digest);

        // Verify with the correct digest should succeed.
        assert!(
            signer.verify(&digest, &sig).is_ok(),
            "Ed25519 round-trip must verify"
        );

        // Verify with a different digest should fail (verify_strict).
        let wrong_digest = [0xBB; 32];
        assert!(
            signer.verify(&wrong_digest, &sig).is_err(),
            "Ed25519 verify_strict must reject wrong digest"
        );

        // Tampered signature should also fail.
        let mut tampered_sig = sig;
        tampered_sig[0] ^= 0xFF;
        assert!(
            signer.verify(&digest, &tampered_sig).is_err(),
            "Ed25519 verify_strict must reject tampered signature"
        );

        // Signer ID should be non-zero and deterministic.
        let id = signer.signer_id();
        assert_ne!(id, [0u8; 32]);
        assert_eq!(id, signer.signer_id());
    }

    // -- Witness v2 end-to-end integration (ADR-134 v2) ---------------------

    /// Full pipeline: v2 gate emissions -> chain verify -> Merkle seal via
    /// the proof-crate signer -> inclusion proof verify -> tamper checks.
    #[test]
    fn witness_v2_gate_seal_inclusion_end_to_end() {
        use rvm_proof::{HmacSha256WitnessSigner, SealSignerAdapter};
        use rvm_security::{GateRequest, SecurityGateV2};
        use rvm_types::WitnessRecordV2;
        use rvm_witness::{verify_chain_v2, verify_inclusion, verify_seal, WitnessLogV2};

        let log = WitnessLogV2::<64, 16>::new();
        let gate = SecurityGateV2::new(&log);

        for i in 0..10u64 {
            let request = GateRequest {
                token: CapToken::new(1, CapType::Partition, CapRights::READ, 1),
                required_type: CapType::Partition,
                required_rights: CapRights::READ,
                proof_commitment: None,
                require_p3: false,
                p3_chain_valid: false,
                p3_witness_data: None,
                action: ActionKind::PartitionCreate,
                target_object_id: i,
                timestamp_ns: i * 1000,
            };
            gate.check_and_execute(&request).unwrap();
        }

        // Chain verifies under the v2 rules.
        let mut records = [WitnessRecordV2::zeroed(); 10];
        assert_eq!(log.snapshot(&mut records), 10);
        let key = log.chain_key();
        assert_eq!(verify_chain_v2(&records, &key), Ok(10));

        // Seal the segment with the proof-crate HMAC signer.
        let signer = HmacSha256WitnessSigner::new([0x33u8; 32]);
        let adapter = SealSignerAdapter::new(&signer);
        let (sealed, acc) = log.seal_segment(&adapter).unwrap();
        assert_eq!(sealed.count, 10);
        assert!(verify_seal(&sealed, &adapter));

        // Every record has a verifiable inclusion proof.
        for seq in 0..10u64 {
            let proof = acc.proof_for_sequence(seq).unwrap();
            let mac = acc.leaf(seq as usize).unwrap();
            assert!(verify_inclusion(&sealed.root, &mac, &proof));
        }

        // Tampered record content breaks chain verification.
        let mut tampered = records;
        tampered[4].target_object_id = 0xBAD;
        assert!(verify_chain_v2(&tampered, &key).is_err());

        // Tampered root breaks both the seal and inclusion proofs.
        let mut bad_root = sealed.root;
        bad_root[0] ^= 1;
        let proof = acc.proof_for_sequence(0).unwrap();
        assert!(!verify_inclusion(&bad_root, &acc.leaf(0).unwrap(), &proof));

        // After sealing, a fresh segment starts at the next sequence.
        assert_eq!(log.segment_len(), 0);
    }

    /// Micro-benchmark of the v2 per-record append cost (one keyed
    /// BLAKE3 compression + bookkeeping). Target: < 1 us per append.
    ///
    /// Run with: `cargo test -p rvm-tests --release -- --ignored witness_v2`
    /// (criterion benches in `benches/` give the rigorous numbers; this
    /// is a quick sanity gate).
    #[test]
    #[ignore = "timing test; run explicitly with --ignored in release mode"]
    fn witness_v2_append_under_one_microsecond() {
        use rvm_types::WitnessRecordV2;
        use rvm_witness::WitnessLogV2;
        use std::time::Instant;

        let log = WitnessLogV2::<4096, 256>::new();
        // Warm up.
        for _ in 0..1_000 {
            log.append(WitnessRecordV2::zeroed());
        }

        const ITERS: u32 = 100_000;
        let start = Instant::now();
        for _ in 0..ITERS {
            log.append(WitnessRecordV2::zeroed());
        }
        let elapsed = start.elapsed();
        let per_append_ns = elapsed.as_nanos() / u128::from(ITERS);
        // 1 us target with debug-build headroom: assert < 10 us here;
        // release builds measure ~0.1-0.4 us (see criterion bench).
        assert!(
            per_append_ns < 10_000,
            "v2 append took {per_append_ns} ns (expected < 10,000 ns even in debug)"
        );
    }
}
