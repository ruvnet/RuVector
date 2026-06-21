//! Comprehensive RVM performance benchmarks.
//!
//! Benchmarks target the hot-path operations specified in ADR-132/134/135:
//!
//! | Operation                  | Target     |
//! |----------------------------|------------|
//! | Witness emission           | < 500 ns   |
//! | P1 capability verification | < 1 us     |
//! | P2 policy evaluation       | < 100 us   |
//! | Partition switch           | < 10 us — **NOT VALIDATED HERE** (see Benchmark 4) |
//! | Coherence score            | budgeted   |
//! | MinCut (16-node)           | < 50 us    |
//! | Buddy alloc/free           | fast       |
//! | FNV-1a hash                | fast       |

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rvm_types::{
    ActionKind, CapRights, CapToken, CapType, CutPressure,
    PartitionId, PhysAddr, WitnessRecord,
};

// ---------------------------------------------------------------------------
// Benchmark 1: Witness emission throughput
// Target: < 500 ns per record
// ---------------------------------------------------------------------------
fn bench_witness_emit(c: &mut Criterion) {
    c.bench_function("witness_emit_single", |b| {
        let log = rvm_witness::WitnessLog::<4096>::new();
        let emitter = rvm_witness::WitnessEmitter::new(&log);
        let mut ts = 0u64;
        b.iter(|| {
            ts += 1;
            black_box(emitter.emit_partition_create(1, 100, 0xABCD, ts));
        });
    });

    c.bench_function("witness_emit_10000", |b| {
        b.iter(|| {
            let log = rvm_witness::WitnessLog::<16384>::new();
            let emitter = rvm_witness::WitnessEmitter::new(&log);
            for i in 0..10_000u64 {
                let _ = emitter.emit_partition_create(1, 100, 0xABCD, i);
            }
            black_box(log.total_emitted());
        });
    });
}

// ---------------------------------------------------------------------------
// Benchmark 2: P1 capability verification
// Target: < 1 us
// ---------------------------------------------------------------------------
fn bench_p1_verify(c: &mut Criterion) {
    use rvm_cap::CapabilityManager;

    c.bench_function("p1_verify", |b| {
        let mut cap_mgr = CapabilityManager::<256>::with_defaults();
        let owner = PartitionId::new(1);
        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::EXECUTE)
            .union(CapRights::GRANT)
            .union(CapRights::REVOKE)
            .union(CapRights::PROVE);

        let (idx, gen) = cap_mgr
            .create_root_capability(CapType::Region, all_rights, 0, owner)
            .unwrap();

        b.iter(|| {
            black_box(cap_mgr.verify_p1(idx, gen, CapRights::READ).unwrap());
        });
    });

    c.bench_function("p1_verify_10000", |b| {
        let mut cap_mgr = CapabilityManager::<256>::with_defaults();
        let owner = PartitionId::new(1);
        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::EXECUTE)
            .union(CapRights::GRANT)
            .union(CapRights::REVOKE)
            .union(CapRights::PROVE);

        let (idx, gen) = cap_mgr
            .create_root_capability(CapType::Region, all_rights, 0, owner)
            .unwrap();

        b.iter(|| {
            for _ in 0..10_000 {
                black_box(cap_mgr.verify_p1(idx, gen, CapRights::READ).unwrap());
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Benchmark 3: P2 proof engine pipeline (P1 + P2 + witness emission)
// Target: < 100 us
// ---------------------------------------------------------------------------
fn bench_p2_verify(c: &mut Criterion) {
    use rvm_cap::CapabilityManager;
    use rvm_proof::context::ProofContextBuilder;
    use rvm_proof::engine::ProofEngine;
    use rvm_types::{ProofTier, ProofToken};

    c.bench_function("p2_proof_engine_pipeline", |b| {
        let mut cap_mgr = CapabilityManager::<256>::with_defaults();
        let owner = PartitionId::new(1);
        let all_rights = CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::EXECUTE)
            .union(CapRights::GRANT)
            .union(CapRights::REVOKE)
            .union(CapRights::PROVE);

        let (idx, gen) = cap_mgr
            .create_root_capability(CapType::Region, all_rights, 0, owner)
            .unwrap();

        let token = ProofToken {
            tier: ProofTier::P2,
            epoch: 0,
            hash: 0x1234,
        };

        let mut nonce = 0u64;
        b.iter(|| {
            nonce += 1;
            let context = ProofContextBuilder::new(owner)
                .target_object(42)
                .capability_handle(idx)
                .capability_generation(gen)
                .current_epoch(0)
                .region_bounds(0x1000, 0x2000)
                .time_window(500, 1000)
                .nonce(nonce)
                .build();

            let witness_log = rvm_witness::WitnessLog::<256>::new();
            let mut engine = ProofEngine::<256>::new();
            black_box(
                engine
                    .verify_and_witness(&token, &context, &cap_mgr, &witness_log)
                    .unwrap(),
            );
        });
    });
}

// ---------------------------------------------------------------------------
// Benchmark 4: Partition switch — HONESTY NOTE (roadmap item 5)
//
// The < 10 us ADR-132 partition-switch target applies to the AArch64 EL2
// hot path: MRS/MSR register save-restore + VTTBR_EL2 write + TLBI VMALLE1
// + DSB/ISB barriers. That path is NOT implemented
// (`rvm_sched::HARDWARE_SWITCH_IMPLEMENTED == false`), so NOTHING in this
// file validates the < 10 us claim. The previous bench named
// "partition_switch" measured run-queue bookkeeping plus a no-op stub and
// effectively self-certified the target; it has been renamed.
//
// What IS measured, honestly labeled:
//  - sched_switch_next_*: scheduler run-queue selection cost (the software
//    bookkeeping that surrounds a switch).
//  - partition_switch_context_copy: the real register-file save+restore
//    data movement (~2x 288 B) — a LOWER BOUND on switch cost, excluding
//    all privileged hardware operations.
//  - partition_switch_validation_stub: the no-op validation stub, kept
//    only to show what the old "partition_switch" number actually was.
// ---------------------------------------------------------------------------
fn bench_partition_switch(c: &mut Criterion) {
    use rvm_sched::Scheduler;

    // Honesty gate: if the hardware path lands, this bench file must be
    // rewritten to measure it.
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            !rvm_sched::HARDWARE_SWITCH_IMPLEMENTED,
            "hardware switch implemented: rewrite Benchmark 4 to measure it"
        );
    }

    c.bench_function("sched_switch_next_2", |b| {
        let mut sched = Scheduler::<4, 256>::new();
        let pid1 = PartitionId::new(1);
        let pid2 = PartitionId::new(2);

        b.iter(|| {
            sched.enqueue(0, pid1, 100, CutPressure::ZERO);
            sched.enqueue(0, pid2, 200, CutPressure::ZERO);
            let r1 = sched.switch_next(0);
            let r2 = sched.switch_next(0);
            black_box((r1, r2));
        });
    });

    c.bench_function("sched_switch_next_8_with_pressure", |b| {
        let mut sched = Scheduler::<4, 256>::new();

        b.iter(|| {
            for i in 0..8u32 {
                sched.enqueue(
                    0,
                    PartitionId::new(i + 1),
                    (i as u16) * 25,
                    CutPressure::from_fixed(i * 1000),
                );
            }
            for _ in 0..8 {
                black_box(sched.switch_next(0));
            }
        });
    });

    // Honest host-measurable component: the actual save/restore data
    // movement of a context switch (excludes VTTBR/TLBI/barriers).
    c.bench_function("partition_switch_context_copy", |b| {
        use rvm_sched::{partition_switch_save_restore, SwitchContext};

        let mut hw = SwitchContext::new();
        hw.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);
        let mut save_area = SwitchContext::new();
        let mut target = SwitchContext::new();
        target.init(0x8000_0000, 0xF000, 2, 0x0002_0000_0000_0000);

        b.iter(|| {
            black_box(
                partition_switch_save_restore(
                    black_box(&mut hw),
                    black_box(&mut save_area),
                    black_box(&target),
                )
                .unwrap(),
            );
        });
    });

    // The no-op validation stub the old "partition_switch" bench was
    // actually certifying. Kept for transparency; its number is
    // meaningless for the < 10 us target.
    c.bench_function("partition_switch_validation_stub", |b| {
        use rvm_sched::{partition_switch, SwitchContext};

        let mut from = SwitchContext::new();
        from.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);
        let mut to = SwitchContext::new();
        to.init(0x8000_0000, 0xF000, 2, 0x0002_0000_0000_0000);

        b.iter(|| {
            black_box(partition_switch(black_box(&mut from), black_box(&to)).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// Benchmark 5: Coherence score computation
// ---------------------------------------------------------------------------
fn bench_coherence_score(c: &mut Criterion) {
    use rvm_coherence::graph::CoherenceGraph;
    use rvm_coherence::scoring::{compute_coherence_score, recompute_all_scores, PartitionCoherenceResult};

    c.bench_function("coherence_score_single_16node", |b| {
        let mut graph = CoherenceGraph::<16, 128>::new();
        for i in 1..=16u32 {
            graph.add_node(PartitionId::new(i)).unwrap();
        }
        // Add edges to create a connected graph.
        for i in 1..=15u32 {
            graph
                .add_edge(PartitionId::new(i), PartitionId::new(i + 1), 100)
                .unwrap();
            graph
                .add_edge(PartitionId::new(i + 1), PartitionId::new(i), 50)
                .unwrap();
        }
        // Add some self-loops.
        for i in 1..=16u32 {
            graph
                .add_edge(PartitionId::new(i), PartitionId::new(i), 200)
                .unwrap();
        }

        b.iter(|| {
            black_box(compute_coherence_score(PartitionId::new(8), &graph));
        });
    });

    c.bench_function("coherence_recompute_all_16node", |b| {
        let mut graph = CoherenceGraph::<16, 128>::new();
        for i in 1..=16u32 {
            graph.add_node(PartitionId::new(i)).unwrap();
        }
        for i in 1..=15u32 {
            graph
                .add_edge(PartitionId::new(i), PartitionId::new(i + 1), 100)
                .unwrap();
        }

        b.iter(|| {
            let mut output: [Option<PartitionCoherenceResult>; 16] = [None; 16];
            black_box(recompute_all_scores(&graph, &mut output));
        });
    });
}

// ---------------------------------------------------------------------------
// Benchmark 6: MinCut computation
// Target: < 50 us for 16-node graph
// ---------------------------------------------------------------------------
fn bench_mincut(c: &mut Criterion) {
    use rvm_coherence::graph::CoherenceGraph;
    use rvm_coherence::mincut::MinCutBridge;

    c.bench_function("mincut_4node", |b| {
        let mut graph = CoherenceGraph::<8, 32>::new();
        let p1 = PartitionId::new(1);
        let p2 = PartitionId::new(2);
        let p3 = PartitionId::new(3);
        let p4 = PartitionId::new(4);
        graph.add_node(p1).unwrap();
        graph.add_node(p2).unwrap();
        graph.add_node(p3).unwrap();
        graph.add_node(p4).unwrap();
        // Strong cluster: p1-p2.
        graph.add_edge(p1, p2, 1000).unwrap();
        graph.add_edge(p2, p1, 1000).unwrap();
        // Strong cluster: p3-p4.
        graph.add_edge(p3, p4, 1000).unwrap();
        graph.add_edge(p4, p3, 1000).unwrap();
        // Weak link: p2-p3.
        graph.add_edge(p2, p3, 10).unwrap();
        graph.add_edge(p3, p2, 10).unwrap();

        let mut bridge = MinCutBridge::<8>::new(100);

        b.iter(|| {
            black_box(bridge.find_min_cut(&graph, p1));
        });
    });

    c.bench_function("mincut_16node", |b| {
        let mut graph = CoherenceGraph::<16, 128>::new();
        for i in 1..=16u32 {
            graph.add_node(PartitionId::new(i)).unwrap();
        }
        // Create a chain with varying weights.
        for i in 1..=15u32 {
            let weight = if i == 8 { 10u64 } else { 1000 }; // Weak link at node 8.
            graph
                .add_edge(PartitionId::new(i), PartitionId::new(i + 1), weight)
                .unwrap();
            graph
                .add_edge(PartitionId::new(i + 1), PartitionId::new(i), weight)
                .unwrap();
        }

        let mut bridge = MinCutBridge::<16>::new(200);

        b.iter(|| {
            black_box(bridge.find_min_cut(&graph, PartitionId::new(1)));
        });
    });
}

// ---------------------------------------------------------------------------
// Benchmark 7: Buddy allocator alloc/free cycles
// ---------------------------------------------------------------------------
fn bench_buddy_alloc(c: &mut Criterion) {
    use rvm_memory::BuddyAllocator;

    c.bench_function("buddy_alloc_order0_256", |b| {
        b.iter(|| {
            let mut alloc =
                BuddyAllocator::<256, 16>::new(PhysAddr::new(0x1000_0000)).unwrap();
            for _ in 0..256 {
                let addr = alloc.alloc_pages(0).unwrap();
                black_box(addr);
            }
        });
    });

    c.bench_function("buddy_alloc_free_cycle_1000", |b| {
        b.iter(|| {
            let mut alloc =
                BuddyAllocator::<256, 16>::new(PhysAddr::new(0x1000_0000)).unwrap();
            for _ in 0..1000 {
                let addr = alloc.alloc_pages(0).unwrap();
                alloc.free_pages(addr, 0).unwrap();
            }
            black_box(alloc.free_page_count());
        });
    });

    c.bench_function("buddy_alloc_mixed_orders", |b| {
        b.iter(|| {
            let mut alloc =
                BuddyAllocator::<256, 16>::new(PhysAddr::new(0x1000_0000)).unwrap();
            // Allocate a mix of orders.
            let a0 = alloc.alloc_pages(0).unwrap();
            let a1 = alloc.alloc_pages(1).unwrap();
            let a2 = alloc.alloc_pages(2).unwrap();
            let a3 = alloc.alloc_pages(3).unwrap();
            // Free in reverse order.
            alloc.free_pages(a3, 3).unwrap();
            alloc.free_pages(a2, 2).unwrap();
            alloc.free_pages(a1, 1).unwrap();
            alloc.free_pages(a0, 0).unwrap();
            black_box(alloc.free_page_count());
        });
    });
}

// ---------------------------------------------------------------------------
// Benchmark 8: FNV-1a hash (witness chain)
// ---------------------------------------------------------------------------
fn bench_fnv1a_hash(c: &mut Criterion) {
    c.bench_function("fnv1a_64_bytes", |b| {
        let data = [0xABu8; 64];
        b.iter(|| {
            black_box(rvm_witness::fnv1a_64(black_box(&data)));
        });
    });

    c.bench_function("fnv1a_64_bytes_x10000", |b| {
        let data = [0xABu8; 64];
        b.iter(|| {
            let mut acc = 0u64;
            for _ in 0..10_000 {
                acc ^= rvm_witness::fnv1a_64(&data);
            }
            black_box(acc);
        });
    });

    c.bench_function("fnv1a_256_bytes", |b| {
        let data = [0xCDu8; 256];
        b.iter(|| {
            black_box(rvm_witness::fnv1a_64(black_box(&data)));
        });
    });
}

// ---------------------------------------------------------------------------
// Bonus: Security gate throughput
// ---------------------------------------------------------------------------
fn bench_security_gate(c: &mut Criterion) {
    use rvm_security::{SecurityGate, GateRequest};
    use rvm_types::WitnessHash;

    c.bench_function("security_gate_check_p1", |b| {
        let log = rvm_witness::WitnessLog::<4096>::new();
        let gate = SecurityGate::new(&log);
        let token = CapToken::new(
            1,
            CapType::Partition,
            CapRights::READ | CapRights::WRITE,
            0,
        );

        b.iter(|| {
            let request = GateRequest {
                token,
                required_type: CapType::Partition,
                required_rights: CapRights::READ,
                proof_commitment: None,
                require_p3: false,
                p3_chain_valid: false,
                p3_witness_data: None,
                action: ActionKind::PartitionCreate,
                target_object_id: 1,
                timestamp_ns: 1000,
            };
            black_box(gate.check_and_execute(&request).unwrap());
        });
    });

    c.bench_function("security_gate_check_p2", |b| {
        let log = rvm_witness::WitnessLog::<4096>::new();
        let gate = SecurityGate::new(&log);
        let token = CapToken::new(
            1,
            CapType::Region,
            CapRights::READ | CapRights::WRITE,
            0,
        );
        let commitment = WitnessHash::from_bytes([0xAB; 32]);

        b.iter(|| {
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
            black_box(gate.check_and_execute(&request).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// Bonus: Witness chain verification
// ---------------------------------------------------------------------------
fn bench_witness_verify_chain(c: &mut Criterion) {
    c.bench_function("witness_verify_chain_64", |b| {
        let log = rvm_witness::WitnessLog::<64>::new();
        for i in 0..64u8 {
            let mut record = WitnessRecord::zeroed();
            record.action_kind = i % 8;
            record.proof_tier = 1;
            record.actor_partition_id = 1;
            log.append(record);
        }

        let mut records = [WitnessRecord::zeroed(); 64];
        for i in 0..64 {
            records[i] = log.get(i).unwrap();
        }

        b.iter(|| {
            black_box(rvm_witness::verify_chain(black_box(&records)).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// Bonus: Cut pressure computation
// ---------------------------------------------------------------------------
fn bench_cut_pressure(c: &mut Criterion) {
    use rvm_coherence::graph::CoherenceGraph;
    use rvm_coherence::pressure::compute_cut_pressure;

    c.bench_function("cut_pressure_16node", |b| {
        let mut graph = CoherenceGraph::<16, 128>::new();
        for i in 1..=16u32 {
            graph.add_node(PartitionId::new(i)).unwrap();
        }
        for i in 1..=15u32 {
            graph
                .add_edge(PartitionId::new(i), PartitionId::new(i + 1), 100)
                .unwrap();
            graph
                .add_edge(PartitionId::new(i + 1), PartitionId::new(i), 50)
                .unwrap();
        }
        // Add self-loops for some nodes.
        for i in 1..=8u32 {
            graph
                .add_edge(PartitionId::new(i), PartitionId::new(i), 500)
                .unwrap();
        }

        b.iter(|| {
            for i in 1..=16u32 {
                black_box(compute_cut_pressure(PartitionId::new(i), &graph));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_witness_emit,
    bench_p1_verify,
    bench_p2_verify,
    bench_partition_switch,
    bench_coherence_score,
    bench_mincut,
    bench_buddy_alloc,
    bench_fnv1a_hash,
    bench_security_gate,
    bench_witness_verify_chain,
    bench_cut_pressure,
);
criterion_main!(benches);
