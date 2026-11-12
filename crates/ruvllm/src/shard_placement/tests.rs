//! Unit tests for the governed pipeline-shard placement planner
//! (PIR WP20, ruvector ADR-323).
//!
//! The tests exercise the governance property that is the point of ADR-323:
//! hard constraints (residency, trust, hardware capability, cumulative memory
//! capacity) are inviolable and an infeasible request is rejected with a
//! specific reason — never silently placed on a violating node — while the
//! latency/cost objective only optimizes *among* feasible placements.

use super::planner::{
    ObjectiveWeights, PlacementPlanner, PlacementRejection, MAX_NODES, MAX_SHARDS,
};
use super::policy::{ConstraintViolation, GovernancePolicy};
use super::{FleetNode, ModelShard, NodeId, ResidencyZone, ShardId, TrustTier};

/// Assert a finished plan places every shard exactly once and that each
/// placement satisfies all hard constraints against the node it landed on
/// (including cumulative memory capacity per node).
fn assert_plan_respects_constraints(
    plan: &super::planner::PlacementPlan,
    shards: &[ModelShard],
    nodes: &[FleetNode],
) {
    // Exactly one assignment per shard, no shard missing or duplicated.
    assert_eq!(plan.assignments.len(), shards.len());
    let mut seen: Vec<&ShardId> = plan.assignments.iter().map(|a| &a.shard).collect();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), shards.len(), "each shard placed exactly once");

    let node_by_id = |id: &NodeId| nodes.iter().find(|n| &n.id == id).unwrap();
    let shard_by_id = |id: &ShardId| shards.iter().find(|s| &s.id == id).unwrap();

    // Per-node cumulative memory must never exceed the node's capacity.
    for (node_id, used) in &plan.memory_used_mb {
        let node = node_by_id(node_id);
        assert!(
            *used <= node.available_memory_mb,
            "node {node_id} over capacity: used {used} > {}",
            node.available_memory_mb
        );
    }

    // Every individual placement satisfies residency/trust/capability/capacity.
    for a in &plan.assignments {
        let shard = shard_by_id(&a.shard);
        let node = node_by_id(&a.node);
        // Residency.
        if let Some(zone) = &shard.residency {
            assert_eq!(
                &node.residency_zone, zone,
                "residency respected for {}",
                a.shard
            );
        }
        // Trust.
        assert!(
            node.trust_tier >= shard.min_trust_tier,
            "trust respected for {}",
            a.shard
        );
        // Capability.
        assert!(
            shard.required_capabilities.is_subset(&node.capabilities),
            "capability respected for {}",
            a.shard
        );
        // Per-shard fit.
        assert!(
            shard.required_memory_mb <= node.available_memory_mb,
            "capacity respected for {}",
            a.shard
        );
    }
}

/// (1) A satisfiable placement produces a valid assignment respecting all hard
/// constraints.
#[test]
fn satisfiable_placement_produces_valid_assignment() {
    let shards = vec![
        ModelShard::new("stage-0", 0, 4000)
            .with_min_trust(TrustTier(2))
            .with_residency(ResidencyZone::new("eu")),
        ModelShard::new("stage-1", 1, 4000)
            .with_min_trust(TrustTier(1))
            .with_required_capabilities(["int4"]),
    ];
    let nodes = vec![
        FleetNode::new("eu-a", ResidencyZone::new("eu"), TrustTier(3), 8000)
            .with_profile(10.0, 1.0)
            .with_capabilities(["int4", "avx512"]),
        FleetNode::new("eu-b", ResidencyZone::new("eu"), TrustTier(2), 8000)
            .with_profile(20.0, 1.0)
            .with_capabilities(["int4"]),
    ];

    let plan = PlacementPlanner::default().plan(&shards, &nodes).unwrap();
    assert_plan_respects_constraints(&plan, &shards, &nodes);
}

/// (2) A residency-violating candidate is rejected — not silently placed.
#[test]
fn residency_violation_is_rejected() {
    // Shard's data is tagged for `eu`; the whole fleet is in `us`.
    let shards = vec![ModelShard::new("stage-0", 0, 1000).with_residency(ResidencyZone::new("eu"))];
    let nodes = vec![
        FleetNode::new("us-a", ResidencyZone::new("us"), TrustTier(9), 64_000),
        FleetNode::new("us-b", ResidencyZone::new("us"), TrustTier(9), 64_000),
    ];

    let err = PlacementPlanner::default()
        .plan(&shards, &nodes)
        .unwrap_err();
    match err {
        PlacementRejection::NoEligibleNode { shard, violations } => {
            assert_eq!(shard, ShardId::new("stage-0"));
            assert_eq!(violations.len(), 2);
            for v in &violations {
                assert!(
                    matches!(v.violation, ConstraintViolation::Residency { .. }),
                    "expected residency violation, got {:?}",
                    v.violation
                );
            }
        }
        other => panic!("expected NoEligibleNode(residency), got {other:?}"),
    }
}

/// (3) A trust-violating candidate is rejected.
#[test]
fn trust_violation_is_rejected() {
    // Shard needs trust tier >= 5; every node is only tier 1.
    let shards = vec![ModelShard::new("stage-0", 0, 1000).with_min_trust(TrustTier(5))];
    let nodes = vec![
        FleetNode::new("low-a", ResidencyZone::new("us"), TrustTier(1), 64_000),
        FleetNode::new("low-b", ResidencyZone::new("us"), TrustTier(4), 64_000),
    ];

    let err = PlacementPlanner::default()
        .plan(&shards, &nodes)
        .unwrap_err();
    match err {
        PlacementRejection::NoEligibleNode { shard, violations } => {
            assert_eq!(shard, ShardId::new("stage-0"));
            assert_eq!(violations.len(), 2);
            for v in &violations {
                assert!(
                    matches!(
                        v.violation,
                        ConstraintViolation::Trust {
                            required: TrustTier(5),
                            ..
                        }
                    ),
                    "expected trust violation, got {:?}",
                    v.violation
                );
            }
        }
        other => panic!("expected NoEligibleNode(trust), got {other:?}"),
    }
}

/// (4) A capacity-overflowing shard (larger than any single node) is rejected.
#[test]
fn capacity_overflow_is_rejected() {
    // Shard needs 100 GB; no single node has that much.
    let shards = vec![ModelShard::new("stage-0", 0, 100_000)];
    let nodes = vec![
        FleetNode::new("small-a", ResidencyZone::new("us"), TrustTier(9), 16_000),
        FleetNode::new("small-b", ResidencyZone::new("us"), TrustTier(9), 24_000),
    ];

    let err = PlacementPlanner::default()
        .plan(&shards, &nodes)
        .unwrap_err();
    match err {
        PlacementRejection::NoEligibleNode { shard, violations } => {
            assert_eq!(shard, ShardId::new("stage-0"));
            for v in &violations {
                assert!(
                    matches!(
                        v.violation,
                        ConstraintViolation::Capacity {
                            required_mb: 100_000,
                            ..
                        }
                    ),
                    "expected capacity violation, got {:?}",
                    v.violation
                );
            }
        }
        other => panic!("expected NoEligibleNode(capacity), got {other:?}"),
    }
}

/// (5) A hardware-capability-violating candidate is rejected (ADR-323's
/// hardware-capability hard constraint, mirroring the paper's INT4/Lunar-Lake
/// hardware specificity).
#[test]
fn capability_violation_is_rejected() {
    let shards =
        vec![ModelShard::new("stage-0", 0, 1000).with_required_capabilities(["int4", "npu"])];
    let nodes = vec![
        FleetNode::new("cpu-a", ResidencyZone::new("us"), TrustTier(9), 64_000)
            .with_capabilities(["int4"]), // missing "npu"
        FleetNode::new("cpu-b", ResidencyZone::new("us"), TrustTier(9), 64_000)
            .with_capabilities(["avx512"]), // missing both
    ];

    let err = PlacementPlanner::default()
        .plan(&shards, &nodes)
        .unwrap_err();
    match err {
        PlacementRejection::NoEligibleNode { violations, .. } => {
            for v in &violations {
                assert!(
                    matches!(v.violation, ConstraintViolation::Capability { .. }),
                    "expected capability violation, got {:?}",
                    v.violation
                );
            }
        }
        other => panic!("expected NoEligibleNode(capability), got {other:?}"),
    }
}

/// (6) Among multiple feasible placements, the planner picks the lower
/// latency/cost one.
#[test]
fn picks_lowest_latency_cost_among_feasible() {
    let shards = vec![ModelShard::new("stage-0", 0, 4000)];
    // Both nodes are fully eligible; they differ only in latency.
    let nodes = vec![
        FleetNode::new("slow", ResidencyZone::new("us"), TrustTier(5), 8000)
            .with_profile(30.0, 1.0),
        FleetNode::new("fast", ResidencyZone::new("us"), TrustTier(5), 8000).with_profile(5.0, 1.0),
    ];

    let plan = PlacementPlanner::default().plan(&shards, &nodes).unwrap();
    assert_eq!(plan.assignments.len(), 1);
    assert_eq!(
        plan.assignments[0].node,
        NodeId::new("fast"),
        "planner must choose the lower latency/cost node"
    );
    // objective = latency*1 + cost*1 = 5 + 1 = 6 for the "fast" node.
    assert!((plan.total_objective - 6.0).abs() < 1e-9);
}

/// (6b) Objective weighting steers the choice: weighting cost heavily flips the
/// pick to the cheaper-but-slower node, proving latency/cost optimization
/// happens strictly within the feasible set.
#[test]
fn objective_weights_steer_choice_within_feasible_set() {
    let shards = vec![ModelShard::new("stage-0", 0, 4000)];
    let nodes = vec![
        // Fast but expensive.
        FleetNode::new("fast-pricey", ResidencyZone::new("us"), TrustTier(5), 8000)
            .with_profile(5.0, 100.0),
        // Slow but cheap.
        FleetNode::new("slow-cheap", ResidencyZone::new("us"), TrustTier(5), 8000)
            .with_profile(40.0, 1.0),
    ];

    // Latency-only weighting → fast node.
    let latency_plan = PlacementPlanner::new(ObjectiveWeights {
        latency: 1.0,
        cost: 0.0,
    })
    .plan(&shards, &nodes)
    .unwrap();
    assert_eq!(latency_plan.assignments[0].node, NodeId::new("fast-pricey"));

    // Cost-only weighting → cheap node.
    let cost_plan = PlacementPlanner::new(ObjectiveWeights {
        latency: 0.0,
        cost: 1.0,
    })
    .plan(&shards, &nodes)
    .unwrap();
    assert_eq!(cost_plan.assignments[0].node, NodeId::new("slow-cheap"));
}

/// (7) The 4-node/70B feasibility case: a model whose shards fit collectively
/// across the fleet but not on any single node. This is the *separate*
/// 4-node/70B result of arXiv:2608.19147 (single-user interactive serving) —
/// NOT the 2-node/8B 1.79× throughput result. Here we prove only the
/// placement-feasibility shape: no node can hold the whole model, yet a valid
/// governed placement exists that spreads the shards across the fleet.
#[test]
fn four_node_seventy_b_fits_collectively_not_singly() {
    // A ~70B-class model split into 4 pipeline stages of 20 GB each (80 GB
    // total). Each Lunar-Lake-class node advertises 24 GB — enough for one
    // stage, never for two (2 * 20 GB = 40 GB > 24 GB), and never for the
    // whole 80 GB model.
    let stage_mb = 20_000u64;
    let node_mb = 24_000u64;
    let shards: Vec<ModelShard> = (0..4)
        .map(|i| ModelShard::new(format!("stage-{i}"), i, stage_mb).with_min_trust(TrustTier(1)))
        .collect();
    let nodes: Vec<FleetNode> = (0..4)
        .map(|i| {
            FleetNode::new(
                format!("lunar-{i}"),
                ResidencyZone::new("eu"),
                TrustTier(2),
                node_mb,
            )
            .with_profile(10.0 + i as f64, 1.0)
        })
        .collect();

    // Precondition: the whole model does NOT fit on any single node.
    let total_model_mb: u64 = shards.iter().map(|s| s.required_memory_mb).sum();
    assert!(
        nodes.iter().all(|n| n.available_memory_mb < total_model_mb),
        "test premise: no single node can hold the whole model"
    );
    // Precondition: it DOES fit collectively.
    let fleet_mb: u64 = nodes.iter().map(|n| n.available_memory_mb).sum();
    assert!(
        fleet_mb >= total_model_mb,
        "test premise: fleet holds it jointly"
    );

    let plan = PlacementPlanner::default().plan(&shards, &nodes).unwrap();
    assert_plan_respects_constraints(&plan, &shards, &nodes);

    // With one 20 GB stage per 24 GB node and no room for two, the four stages
    // must land on four distinct nodes.
    let mut used_nodes: Vec<&NodeId> = plan.assignments.iter().map(|a| &a.node).collect();
    used_nodes.sort();
    used_nodes.dedup();
    assert_eq!(
        used_nodes.len(),
        4,
        "each 20 GB stage must occupy its own 24 GB node"
    );
}

/// (7b) Cumulative-capacity exhaustion: shards are each individually placeable
/// but collectively exceed the fleet — rejected, never overcommitted.
#[test]
fn cumulative_capacity_exhaustion_is_rejected() {
    // Three 20 GB stages (60 GB) but only two 24 GB nodes (48 GB total). Each
    // stage fits a node individually; all three cannot fit together.
    let shards: Vec<ModelShard> = (0..3)
        .map(|i| ModelShard::new(format!("stage-{i}"), i, 20_000))
        .collect();
    let nodes = vec![
        FleetNode::new("n0", ResidencyZone::new("eu"), TrustTier(2), 24_000),
        FleetNode::new("n1", ResidencyZone::new("eu"), TrustTier(2), 24_000),
    ];

    // Sanity: each shard is individually eligible on each node.
    for s in &shards {
        for n in &nodes {
            assert!(GovernancePolicy::is_eligible(s, n).is_ok());
        }
    }

    let err = PlacementPlanner::default()
        .plan(&shards, &nodes)
        .unwrap_err();
    match err {
        PlacementRejection::CapacityExhausted {
            required_mb,
            fleet_available_mb,
            ..
        } => {
            assert_eq!(required_mb, 20_000);
            assert_eq!(fleet_available_mb, 48_000);
        }
        other => panic!("expected CapacityExhausted, got {other:?}"),
    }
}

/// (8) The 2-node/8B configuration places validly. This is the configuration
/// the paper's 1.79× concurrent-throughput figure applies to — and ONLY that
/// configuration; this test asserts placement feasibility, not throughput.
#[test]
fn two_node_eight_b_places_validly() {
    // An 8B-INT4-class model in two pipeline stages, ~2.4 GB each, on two
    // same-family nodes advertising INT4 support.
    let shards = vec![
        ModelShard::new("stage-0", 0, 2400).with_required_capabilities(["int4"]),
        ModelShard::new("stage-1", 1, 2400).with_required_capabilities(["int4"]),
    ];
    let nodes = vec![
        FleetNode::new("aipc-0", ResidencyZone::new("eu"), TrustTier(2), 8000)
            .with_profile(8.0, 1.0)
            .with_capabilities(["int4"]),
        FleetNode::new("aipc-1", ResidencyZone::new("eu"), TrustTier(2), 8000)
            .with_profile(9.0, 1.0)
            .with_capabilities(["int4"]),
    ];

    let plan = PlacementPlanner::default().plan(&shards, &nodes).unwrap();
    assert_plan_respects_constraints(&plan, &shards, &nodes);
    assert_eq!(plan.assignments.len(), 2);
}

/// Empty shard set is trivially satisfiable (nothing to place).
#[test]
fn empty_shard_set_is_trivially_satisfiable() {
    let nodes = vec![FleetNode::new(
        "n0",
        ResidencyZone::new("eu"),
        TrustTier(1),
        8000,
    )];
    let plan = PlacementPlanner::default().plan(&[], &nodes).unwrap();
    assert!(plan.assignments.is_empty());
    assert_eq!(plan.total_objective, 0.0);
}

/// Defensive input-size bound (PIR WP20, #867): a request whose shard or node
/// count exceeds the documented cap is rejected up front — before the
/// branch-and-bound search runs — so it can neither exhaust the native stack
/// (recursion depth = shard count) nor blow up combinatorially.
#[test]
fn oversized_request_is_rejected_before_search() {
    // One shard over the shard cap on a small fleet.
    let too_many_shards: Vec<ModelShard> = (0..=MAX_SHARDS)
        .map(|i| ModelShard::new(format!("stage-{i}"), i, 1))
        .collect();
    let small_fleet = vec![FleetNode::new(
        "n0",
        ResidencyZone::new("us"),
        TrustTier(9),
        1_000_000,
    )];
    match PlacementPlanner::default()
        .plan(&too_many_shards, &small_fleet)
        .unwrap_err()
    {
        PlacementRejection::InputTooLarge {
            shards,
            nodes,
            max_shards,
            max_nodes,
        } => {
            assert_eq!(shards, MAX_SHARDS + 1);
            assert_eq!(nodes, 1);
            assert_eq!(max_shards, MAX_SHARDS);
            assert_eq!(max_nodes, MAX_NODES);
        }
        other => panic!("expected InputTooLarge (shards), got {other:?}"),
    }

    // One node over the node cap with a single shard.
    let one_shard = vec![ModelShard::new("stage-0", 0, 1)];
    let too_many_nodes: Vec<FleetNode> = (0..=MAX_NODES)
        .map(|i| {
            FleetNode::new(
                format!("n{i}"),
                ResidencyZone::new("us"),
                TrustTier(9),
                1000,
            )
        })
        .collect();
    match PlacementPlanner::default()
        .plan(&one_shard, &too_many_nodes)
        .unwrap_err()
    {
        PlacementRejection::InputTooLarge {
            nodes, max_nodes, ..
        } => {
            assert_eq!(nodes, MAX_NODES + 1);
            assert_eq!(max_nodes, MAX_NODES);
        }
        other => panic!("expected InputTooLarge (nodes), got {other:?}"),
    }
}

/// The bound admits at-cap requests: exactly [`MAX_SHARDS`] shards over
/// [`MAX_NODES`] nodes is within limits and still produces a valid plan.
#[test]
fn at_cap_request_still_plans() {
    let shards: Vec<ModelShard> = (0..MAX_SHARDS)
        .map(|i| ModelShard::new(format!("stage-{i}"), i, 1))
        .collect();
    let nodes: Vec<FleetNode> = (0..MAX_NODES)
        .map(|i| {
            FleetNode::new(
                format!("n{i}"),
                ResidencyZone::new("us"),
                TrustTier(9),
                1_000_000,
            )
        })
        .collect();
    assert_eq!(shards.len(), MAX_SHARDS);
    assert_eq!(nodes.len(), MAX_NODES);

    let plan = PlacementPlanner::default().plan(&shards, &nodes).unwrap();
    assert_plan_respects_constraints(&plan, &shards, &nodes);
    assert_eq!(plan.assignments.len(), MAX_SHARDS);
}
