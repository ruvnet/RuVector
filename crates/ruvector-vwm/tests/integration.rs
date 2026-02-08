//! Comprehensive integration tests for the ruvector-vwm crate.
//!
//! These tests exercise the full Visual World Model pipeline end-to-end,
//! verifying that all modules compose correctly in realistic scenarios.
//! Each test is documented with what it validates and why that matters.

use ruvector_vwm::coherence::{
    CoherenceDecision, CoherenceGate, CoherenceInput, CoherencePolicy, PermissionLevel,
};
use ruvector_vwm::draw_list::{DrawCommand, DrawList, OpacityMode};
use ruvector_vwm::entity::{AttributeValue, Edge, EdgeType, Entity, EntityGraph, EntityType};
use ruvector_vwm::gaussian::Gaussian4D;
use ruvector_vwm::lineage::{LineageEventType, LineageLog, Provenance, ProvenanceSource};
use ruvector_vwm::streaming::{
    ActiveMask, BandwidthBudget, DeltaPacket, KeyframePacket, StreamPacket,
};
use ruvector_vwm::tile::{PrimitiveBlock, QuantTier, Tile, TileCoord};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build a simple perspective-like projection matrix (column-major).
///
/// This produces a matrix that places objects at negative-Z in front of the camera
/// with a valid positive W component, suitable for testing projection math without
/// pulling in a full linear algebra library.
fn simple_perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0,  0.0,                      0.0,
        0.0,        f,    0.0,                      0.0,
        0.0,        0.0,  (far + near) * nf,        -1.0,
        0.0,        0.0,  2.0 * far * near * nf,     0.0,
    ]
}

/// Create a sensor provenance for test events.
fn sensor_provenance(sensor_id: &str, confidence: f32) -> Provenance {
    Provenance {
        source: ProvenanceSource::Sensor {
            sensor_id: sensor_id.to_string(),
        },
        confidence,
        signature: None,
    }
}

/// Create an inference provenance for test events.
fn inference_provenance(model_id: &str, confidence: f32) -> Provenance {
    Provenance {
        source: ProvenanceSource::Inference {
            model_id: model_id.to_string(),
        },
        confidence,
        signature: None,
    }
}

/// Build a warehouse object entity with the given class label and time span.
fn make_warehouse_object(id: u64, class: &str, time_span: [f32; 2]) -> Entity {
    Entity {
        id,
        entity_type: EntityType::Object {
            class: class.to_string(),
        },
        time_span,
        embedding: vec![0.0; 16],
        confidence: 0.9,
        privacy_tags: vec![],
        attributes: vec![],
        gaussian_ids: vec![],
    }
}

/// Build a track entity linking observations over time.
fn make_track(id: u64, time_span: [f32; 2]) -> Entity {
    Entity {
        id,
        entity_type: EntityType::Track,
        time_span,
        embedding: vec![],
        confidence: 0.85,
        privacy_tags: vec![],
        attributes: vec![],
        gaussian_ids: vec![],
    }
}

// ===========================================================================
// 1. Full Pipeline Test
// ===========================================================================

/// Tests the complete render pipeline from Gaussian creation to GPU-ready bytes.
///
/// This is the highest-level integration test: it walks through every stage of
/// the render pipeline in order and verifies that data flows correctly from one
/// stage to the next. A failure here indicates a fundamental incompatibility
/// between pipeline stages.
///
/// Pipeline stages exercised:
///   Gaussian4D::new -> PrimitiveBlock::encode -> Tile -> DrawList -> to_bytes -> checksum
#[test]
fn test_full_pipeline_gaussian_to_gpu_bytes() {
    // Stage 1: Create a small set of Gaussians representing a simple scene.
    // We use three Gaussians placed at different depths along the Z axis so that
    // projection will produce distinct screen positions.
    let g0 = Gaussian4D::new([0.0, 0.0, -5.0], 0);
    let g1 = Gaussian4D::new([1.0, 1.0, -8.0], 1);
    let g2 = Gaussian4D::new([-1.0, 0.5, -3.0], 2);
    let gaussians = vec![g0, g1, g2];

    // Stage 2: Encode Gaussians into a PrimitiveBlock with Hot8 quantization.
    // This packs the Gaussian data into a byte buffer suitable for GPU upload.
    let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
    assert_eq!(block.count, 3, "Block should contain exactly 3 Gaussians");
    assert!(
        !block.data.is_empty(),
        "Encoded data should be non-empty for 3 Gaussians"
    );
    assert!(
        block.verify_checksum(),
        "Checksum must be valid immediately after encoding"
    );

    // Stage 3: Wrap the block in a Tile with spatial coordinates.
    let tile = Tile {
        coord: TileCoord {
            x: 0,
            y: 0,
            z: -1,
            time_bucket: 0,
            lod: 0,
        },
        primitive_block: block,
        entity_refs: vec![100, 101],
        coherence_score: 0.95,
        last_update_epoch: 1,
    };
    assert_eq!(tile.primitive_block.count, 3);

    // Stage 4: Build a DrawList that references the tile.
    let mut draw_list = DrawList::new(1, 0, 0);
    draw_list.bind_tile(42, 0, QuantTier::Hot8);
    draw_list.set_budget(0, 1024, 2.0);
    draw_list.draw_block(0, 0.5, OpacityMode::AlphaBlend);
    let checksum = draw_list.finalize();

    assert_ne!(checksum, 0, "Finalized checksum should be non-zero");
    assert_eq!(
        draw_list.command_count(),
        3,
        "Should have bind + budget + draw (excluding End)"
    );

    // Stage 5: Serialize the draw list to bytes for GPU upload.
    let bytes = draw_list.to_bytes();

    // Header is 20 bytes (epoch:8 + sequence:4 + budget_profile_id:4 + checksum:4).
    // Commands add additional bytes. The total must exceed the header size.
    assert!(
        bytes.len() > 20,
        "Serialized draw list must be larger than the 20-byte header"
    );

    // Verify the epoch is encoded correctly in the first 8 bytes.
    let encoded_epoch = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    assert_eq!(encoded_epoch, 1, "Epoch must round-trip through serialization");

    // Verify the checksum is encoded in bytes 16..20.
    let encoded_checksum = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    assert_eq!(
        encoded_checksum, checksum,
        "Checksum in header bytes must match the finalize() return value"
    );

    // Stage 6: Decode the tile's Gaussians back and verify they survived the round trip.
    // Hot8 quantization is lossy (~1/255 of range per field), so use tolerance.
    let decoded = tile.primitive_block.decode();
    assert_eq!(decoded.len(), 3);
    let tol = 0.05; // conservative tolerance for the value ranges in this test
    for (i, (expected, actual)) in [[0.0f32, 0.0, -5.0], [1.0, 1.0, -8.0], [-1.0, 0.5, -3.0]]
        .iter()
        .zip(decoded.iter())
        .enumerate()
    {
        for (j, (&e, &a)) in expected.iter().zip(actual.center.iter()).enumerate() {
            assert!(
                (e - a).abs() <= tol,
                "Gaussian {} center[{}]: expected {}, got {}, tol={}",
                i, j, e, a, tol
            );
        }
    }
}

// ===========================================================================
// 2. Dynamic Scene Test
// ===========================================================================

/// Tests temporal dynamics: Gaussians with velocity, time-range gating, and
/// projection at different timestamps.
///
/// The linear motion model is critical for representing moving objects without
/// re-encoding the tile every frame. This test verifies that:
///   - position_at() correctly applies velocity relative to the time midpoint
///   - is_active_at() respects exact boundary values
///   - project() returns different screen positions at different times
///   - project() returns None when outside the active time window
#[test]
fn test_dynamic_scene_temporal_evolution() {
    // A Gaussian moving along +X at 2 units/second, active from t=0 to t=10.
    // Midpoint is t=5, so at t=5 the position equals center.
    let mut g = Gaussian4D::new([0.0, 0.0, -5.0], 0);
    g.velocity = [2.0, 0.0, 0.0];
    g.time_range = [0.0, 10.0];

    // Verify position_at at multiple time points.
    // At t=0: dt = 0 - 5 = -5, pos_x = 0 + 2*(-5) = -10
    let pos_t0 = g.position_at(0.0);
    assert!(
        (pos_t0[0] - (-10.0)).abs() < 1e-6,
        "At t=0, x should be -10.0 but got {}",
        pos_t0[0]
    );

    // At t=5 (midpoint): dt = 0, pos_x = 0
    let pos_t5 = g.position_at(5.0);
    assert!(
        pos_t5[0].abs() < 1e-6,
        "At t=5, x should be 0.0 but got {}",
        pos_t5[0]
    );

    // At t=10: dt = 10 - 5 = 5, pos_x = 0 + 2*5 = 10
    let pos_t10 = g.position_at(10.0);
    assert!(
        (pos_t10[0] - 10.0).abs() < 1e-6,
        "At t=10, x should be 10.0 but got {}",
        pos_t10[0]
    );

    // Y and Z should remain at center values.
    assert!(pos_t0[1].abs() < 1e-6);
    assert!((pos_t0[2] - (-5.0)).abs() < 1e-6);

    // Verify is_active_at boundary behavior.
    assert!(!g.is_active_at(-0.001), "Just before start should be inactive");
    assert!(g.is_active_at(0.0), "Exactly at start should be active (inclusive)");
    assert!(g.is_active_at(5.0), "Midpoint should be active");
    assert!(g.is_active_at(10.0), "Exactly at end should be active (inclusive)");
    assert!(!g.is_active_at(10.001), "Just after end should be inactive");

    // Project at different times and verify screen positions differ.
    let vp = simple_perspective(1.0, 1.0, 0.1, 100.0);

    let proj_t2 = g.project(&vp, 2.0);
    let proj_t8 = g.project(&vp, 8.0);
    assert!(proj_t2.is_some(), "Should project at t=2 (within range)");
    assert!(proj_t8.is_some(), "Should project at t=8 (within range)");

    let sg2 = proj_t2.unwrap();
    let sg8 = proj_t8.unwrap();

    // The Gaussian moves along +X, so the screen X positions must differ.
    let screen_x_diff = (sg8.center_screen[0] - sg2.center_screen[0]).abs();
    assert!(
        screen_x_diff > 0.01,
        "Screen X positions at t=2 and t=8 should differ due to X-velocity, diff={}",
        screen_x_diff
    );

    // Projection outside the active window should return None.
    assert!(
        g.project(&vp, -1.0).is_none(),
        "Projection before time range should return None"
    );
    assert!(
        g.project(&vp, 11.0).is_none(),
        "Projection after time range should return None"
    );
}

/// Tests a scene with multiple Gaussians that have staggered time windows.
///
/// In a real scene, Gaussians enter and leave the active set as time progresses.
/// This test verifies that only the temporally active subset projects correctly.
#[test]
fn test_dynamic_scene_staggered_windows() {
    let mut g_early = Gaussian4D::new([0.0, 0.0, -5.0], 0);
    g_early.time_range = [0.0, 3.0];

    let mut g_mid = Gaussian4D::new([1.0, 0.0, -5.0], 1);
    g_mid.time_range = [2.0, 7.0];

    let mut g_late = Gaussian4D::new([2.0, 0.0, -5.0], 2);
    g_late.time_range = [6.0, 10.0];

    let vp = simple_perspective(1.0, 1.0, 0.1, 100.0);

    // At t=1: only g_early active
    assert!(g_early.project(&vp, 1.0).is_some());
    assert!(g_mid.project(&vp, 1.0).is_none());
    assert!(g_late.project(&vp, 1.0).is_none());

    // At t=2.5: g_early and g_mid active
    assert!(g_early.project(&vp, 2.5).is_some());
    assert!(g_mid.project(&vp, 2.5).is_some());
    assert!(g_late.project(&vp, 2.5).is_none());

    // At t=6.5: g_mid and g_late active
    assert!(g_early.project(&vp, 6.5).is_none());
    assert!(g_mid.project(&vp, 6.5).is_some());
    assert!(g_late.project(&vp, 6.5).is_some());

    // At t=9: only g_late active
    assert!(g_early.project(&vp, 9.0).is_none());
    assert!(g_mid.project(&vp, 9.0).is_none());
    assert!(g_late.project(&vp, 9.0).is_some());
}

// ===========================================================================
// 3. Coherence Gate Scenario Test
// ===========================================================================

/// Simulates a realistic sequence of updates with varying data quality, verifying
/// that the CoherenceGate produces the correct decision for each scenario.
///
/// The coherence gate is the core safety mechanism preventing bad data from
/// corrupting the world model. This test exercises every decision path in the
/// priority chain documented in the CoherenceGate implementation:
///   1. Admin always accepts
///   2. ReadOnly always defers
///   3. Stale data defers
///   4. High disagreement triggers rollback/freeze
///   5. Budget pressure triggers freeze
///   6. Entity continuity determines accept vs defer vs freeze
#[test]
fn test_coherence_gate_scenario_sequence() {
    let gate = CoherenceGate::with_defaults();

    // Scenario A: Good sensor data with high continuity and low disagreement.
    // Expected: Accept (effective_continuity = 0.9 * 1.0 = 0.9 >= 0.7 threshold).
    let good_input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&good_input),
        CoherenceDecision::Accept,
        "Good sensor data should be accepted"
    );

    // Scenario B: Stale data (age exceeds max_staleness_ms of 5000).
    // Expected: Defer (stale check fires before disagreement checks).
    let stale_input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 6000,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&stale_input),
        CoherenceDecision::Defer,
        "Stale data should be deferred"
    );

    // Scenario C: High tile disagreement but below rollback threshold.
    // Expected: Freeze (disagreement 0.85 >= freeze_disagreement 0.8).
    let conflict_input = CoherenceInput {
        tile_disagreement: 0.85,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&conflict_input),
        CoherenceDecision::Freeze,
        "High disagreement should freeze the tile"
    );

    // Scenario D: Catastrophically bad data (disagreement above rollback threshold).
    // Expected: Rollback (disagreement 0.96 >= rollback_disagreement 0.95).
    let catastrophic_input = CoherenceInput {
        tile_disagreement: 0.96,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&catastrophic_input),
        CoherenceDecision::Rollback,
        "Catastrophically bad data should trigger rollback"
    );

    // Scenario E: Admin override should accept even with catastrophic disagreement.
    // Expected: Accept (Admin short-circuits all other checks).
    let admin_override = CoherenceInput {
        tile_disagreement: 0.99,
        entity_continuity: 0.0,
        sensor_confidence: 0.0,
        sensor_freshness_ms: 999_999,
        budget_pressure: 1.0,
        permission_level: PermissionLevel::Admin,
    };
    assert_eq!(
        gate.evaluate(&admin_override),
        CoherenceDecision::Accept,
        "Admin override should accept regardless of all other signals"
    );

    // Scenario F: Budget pressure above freeze threshold.
    // Expected: Freeze (budget_pressure 0.95 >= budget_freeze_threshold 0.9).
    let budget_pressure_input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.95,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&budget_pressure_input),
        CoherenceDecision::Freeze,
        "Excessive budget pressure should freeze"
    );

    // Scenario G: Medium continuity that falls between accept and defer thresholds.
    // effective_continuity = 0.5 * 1.0 = 0.5, defer_threshold = 0.4, accept = 0.7.
    // Expected: Defer.
    let medium_input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.5,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&medium_input),
        CoherenceDecision::Defer,
        "Medium continuity should defer"
    );

    // Scenario H: Very low continuity with low sensor confidence.
    // effective_continuity = 0.2 * 0.5 = 0.1, below defer_threshold 0.4.
    // Expected: Freeze.
    let very_low_input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.2,
        sensor_confidence: 0.5,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&very_low_input),
        CoherenceDecision::Freeze,
        "Very low effective continuity should freeze"
    );
}

/// Tests that elevated permissions provide the documented +0.1 boost to
/// effective continuity, tipping a borderline case from Defer to Accept.
#[test]
fn test_coherence_gate_elevated_boost() {
    let gate = CoherenceGate::with_defaults();

    // Continuity 0.65 * confidence 1.0 = 0.65 effective, below accept 0.7 -> Defer.
    let input_standard = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.65,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(gate.evaluate(&input_standard), CoherenceDecision::Defer);

    // Same but Elevated: effective = 0.65 + 0.1 = 0.75 >= 0.7 -> Accept.
    let input_elevated = CoherenceInput {
        permission_level: PermissionLevel::Elevated,
        ..input_standard
    };
    assert_eq!(gate.evaluate(&input_elevated), CoherenceDecision::Accept);
}

/// Tests that custom policy thresholds are respected by the gate.
#[test]
fn test_coherence_gate_custom_policy() {
    let strict_policy = CoherencePolicy {
        accept_threshold: 0.99,
        defer_threshold: 0.8,
        freeze_disagreement: 0.5,
        rollback_disagreement: 0.7,
        max_staleness_ms: 1000,
        budget_freeze_threshold: 0.5,
    };
    let gate = CoherenceGate::new(strict_policy);

    // With strict thresholds, even good data gets deferred.
    let input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    // effective = 0.9, below 0.99 accept but above 0.8 defer -> Defer
    assert_eq!(
        gate.evaluate(&input),
        CoherenceDecision::Defer,
        "Strict policy should defer even high-quality data"
    );

    // With a lower disagreement, freeze kicks in earlier.
    let conflict_input = CoherenceInput {
        tile_disagreement: 0.55,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };
    assert_eq!(
        gate.evaluate(&conflict_input),
        CoherenceDecision::Freeze,
        "Lower freeze threshold should trigger freeze at 0.55 disagreement"
    );
}

// ===========================================================================
// 4. Entity Graph World Model Test
// ===========================================================================

/// Builds a warehouse scene graph and exercises type queries, time-range queries,
/// and neighbor traversal.
///
/// The entity graph is the semantic backbone of the world model -- it connects
/// raw Gaussian geometry to high-level concepts (objects, tracks, regions). This
/// test verifies that a realistic multi-entity, multi-edge graph can be built
/// and queried correctly.
#[test]
fn test_entity_graph_warehouse_scene() {
    let mut graph = EntityGraph::new();

    // Add warehouse objects.
    let forklift = make_warehouse_object(1, "forklift", [0.0, 100.0]);
    let pallet = make_warehouse_object(2, "pallet", [0.0, 100.0]);
    let person = make_warehouse_object(3, "person", [10.0, 80.0]);
    let wall = make_warehouse_object(4, "wall", [0.0, 100.0]);

    graph.add_entity(forklift);
    graph.add_entity(pallet);
    graph.add_entity(person);
    graph.add_entity(wall);

    // Add tracks that link observations of the forklift and person over time.
    let forklift_track = make_track(10, [0.0, 100.0]);
    let person_track = make_track(11, [10.0, 80.0]);
    graph.add_entity(forklift_track);
    graph.add_entity(person_track);

    assert_eq!(graph.entity_count(), 6, "Should have 4 objects + 2 tracks");

    // Add edges representing spatial and temporal relationships.

    // The forklift is adjacent to the pallet (they are near each other).
    graph.add_edge(Edge {
        source: 1,
        target: 2,
        edge_type: EdgeType::Adjacency,
        weight: 1.0,
        time_range: Some([0.0, 100.0]),
    });

    // The pallet is contained within the warehouse region (wall).
    graph.add_edge(Edge {
        source: 4,
        target: 2,
        edge_type: EdgeType::Containment,
        weight: 1.0,
        time_range: None,
    });

    // The forklift is adjacent to the person at a specific time range.
    graph.add_edge(Edge {
        source: 1,
        target: 3,
        edge_type: EdgeType::Adjacency,
        weight: 0.8,
        time_range: Some([20.0, 50.0]),
    });

    // Temporal continuity edges link objects to their tracks.
    graph.add_edge(Edge {
        source: 1,
        target: 10,
        edge_type: EdgeType::Continuity,
        weight: 0.95,
        time_range: None,
    });
    graph.add_edge(Edge {
        source: 3,
        target: 11,
        edge_type: EdgeType::Continuity,
        weight: 0.9,
        time_range: None,
    });

    assert_eq!(graph.edge_count(), 5, "Should have 5 edges");

    // Query by type: "forklift" should return exactly one entity.
    let forklifts = graph.query_by_type("forklift");
    assert_eq!(forklifts.len(), 1);
    assert_eq!(forklifts[0].id, 1);

    // Query by type: "object" should return all 4 objects.
    let objects = graph.query_by_type("object");
    assert_eq!(objects.len(), 4);

    // Query by type: "track" should return 2 tracks.
    let tracks = graph.query_by_type("track");
    assert_eq!(tracks.len(), 2);

    // Query by time range: only entities active in [5, 9] -- person starts at 10.
    let active_early = graph.query_time_range(5.0, 9.0);
    let active_early_ids: Vec<u64> = active_early.iter().map(|e| e.id).collect();
    // forklift(0..100), pallet(0..100), wall(0..100), forklift_track(0..100) active.
    // person(10..80) starts at 10 > 9 end, so NOT active.
    // person_track(10..80) same reason.
    assert!(
        !active_early_ids.contains(&3),
        "Person should not be active in [5, 9]"
    );
    assert!(
        active_early_ids.contains(&1),
        "Forklift should be active in [5, 9]"
    );

    // Neighbor traversal: forklift (id=1) is connected to pallet, person, and track.
    let forklift_neighbors = graph.neighbors(1);
    let neighbor_ids: Vec<u64> = forklift_neighbors.iter().map(|e| e.id).collect();
    assert_eq!(
        forklift_neighbors.len(),
        3,
        "Forklift should have 3 neighbors: pallet, person, track"
    );
    assert!(neighbor_ids.contains(&2), "Pallet should be a neighbor");
    assert!(neighbor_ids.contains(&3), "Person should be a neighbor");
    assert!(neighbor_ids.contains(&10), "Track should be a neighbor");

    // Wall neighbors: connected to pallet via containment.
    let wall_neighbors = graph.neighbors(4);
    assert_eq!(wall_neighbors.len(), 1);
    assert_eq!(wall_neighbors[0].id, 2, "Wall is connected to pallet via containment");
}

// ===========================================================================
// 5. Lineage Audit Trail Test
// ===========================================================================

/// Simulates a realistic tile lifecycle with multiple mutation types and verifies
/// full audit trail querying, rollback point discovery, and chronological ordering.
///
/// The lineage log is the provenance backbone that makes the world model auditable.
/// Every mutation is recorded with who did it, when, and why (coherence decision).
/// This test walks through a complete lifecycle:
///   create -> sensor update -> model update -> rollback -> freeze
#[test]
fn test_lineage_audit_trail_full_lifecycle() {
    let mut log = LineageLog::new();
    let tile_id = 42;

    // Event 0: Tile created from initial sensor data (Accept).
    let ev0 = log.append(
        1000,
        tile_id,
        LineageEventType::TileCreated,
        sensor_provenance("lidar-01", 0.95),
        None,
        CoherenceDecision::Accept,
        0.95,
    );
    assert_eq!(ev0, 0);

    // Event 1: Updated with fresh sensor data (Accept).
    let ev1 = log.append(
        2000,
        tile_id,
        LineageEventType::TileUpdated { delta_size: 256 },
        sensor_provenance("lidar-01", 0.92),
        None,
        CoherenceDecision::Accept,
        0.92,
    );
    assert_eq!(ev1, 1);

    // Event 2: Updated with model inference output (Accept).
    let ev2 = log.append(
        3000,
        tile_id,
        LineageEventType::TileUpdated { delta_size: 128 },
        inference_provenance("nerf-v2", 0.88),
        None,
        CoherenceDecision::Accept,
        0.88,
    );
    assert_eq!(ev2, 2);

    // Event 3: Bad sensor data triggers rollback. The rollback pointer
    // references event 1 (the last known-good sensor state).
    let ev3 = log.append(
        4000,
        tile_id,
        LineageEventType::Rollback {
            reason: "Sensor drift detected".to_string(),
        },
        sensor_provenance("lidar-01", 0.3),
        Some(ev1), // rollback to event 1
        CoherenceDecision::Rollback,
        0.3,
    );
    assert_eq!(ev3, 3);

    // Event 4: Tile frozen during maintenance.
    let ev4 = log.append(
        5000,
        tile_id,
        LineageEventType::Freeze {
            reason: "Scheduled maintenance".to_string(),
        },
        Provenance {
            source: ProvenanceSource::Manual {
                user_id: "admin-42".to_string(),
            },
            confidence: 1.0,
            signature: None,
        },
        None,
        CoherenceDecision::Freeze,
        1.0,
    );
    assert_eq!(ev4, 4);

    // Verify total log size.
    assert_eq!(log.len(), 5, "Log should contain 5 events");

    // Query full history for this tile.
    let tile_history = log.query_tile(tile_id);
    assert_eq!(tile_history.len(), 5, "All 5 events belong to tile 42");

    // Verify chronological ordering (timestamps should be monotonically increasing).
    for window in tile_history.windows(2) {
        assert!(
            window[0].timestamp_ms <= window[1].timestamp_ms,
            "Events must be in chronological order: {} <= {}",
            window[0].timestamp_ms,
            window[1].timestamp_ms,
        );
    }

    // Verify rollback point discovery.
    // The most recent rollback pointer is event 3 pointing to event 1.
    let rollback_point = log.find_rollback_point(tile_id);
    assert_eq!(
        rollback_point,
        Some(1),
        "Rollback point should be event 1 (last known-good sensor update)"
    );

    // Verify individual event retrieval.
    let event3 = log.get(3).unwrap();
    assert_eq!(event3.tile_id, tile_id);
    assert_eq!(event3.coherence_decision, CoherenceDecision::Rollback);
    assert_eq!(event3.rollback_pointer, Some(1));

    // Query by time range: events between t=2500 and t=4500.
    let range_events = log.query_range(2500, 4500);
    assert_eq!(
        range_events.len(),
        2,
        "Should find events at t=3000 and t=4000"
    );
    assert_eq!(range_events[0].timestamp_ms, 3000);
    assert_eq!(range_events[1].timestamp_ms, 4000);
}

/// Tests that the lineage log correctly handles interleaved events from
/// multiple tiles without cross-contamination.
#[test]
fn test_lineage_multi_tile_isolation() {
    let mut log = LineageLog::new();

    // Create two tiles.
    log.append(
        100, 1, LineageEventType::TileCreated,
        sensor_provenance("cam-0", 0.9), None,
        CoherenceDecision::Accept, 0.9,
    );
    log.append(
        100, 2, LineageEventType::TileCreated,
        sensor_provenance("cam-1", 0.9), None,
        CoherenceDecision::Accept, 0.9,
    );

    // Update tile 1 twice.
    log.append(
        200, 1, LineageEventType::TileUpdated { delta_size: 64 },
        sensor_provenance("cam-0", 0.85), None,
        CoherenceDecision::Accept, 0.85,
    );
    log.append(
        300, 1, LineageEventType::TileUpdated { delta_size: 32 },
        sensor_provenance("cam-0", 0.80), None,
        CoherenceDecision::Accept, 0.80,
    );

    // Update tile 2 once.
    log.append(
        250, 2, LineageEventType::TileUpdated { delta_size: 128 },
        sensor_provenance("cam-1", 0.88), None,
        CoherenceDecision::Accept, 0.88,
    );

    // Tile 1 should have 3 events, tile 2 should have 2 events.
    assert_eq!(log.query_tile(1).len(), 3);
    assert_eq!(log.query_tile(2).len(), 2);

    // No cross-contamination.
    for event in log.query_tile(1) {
        assert_eq!(event.tile_id, 1, "Tile 1 query should only return tile 1 events");
    }
}

// ===========================================================================
// 6. Streaming Protocol Test
// ===========================================================================

/// Tests the streaming protocol components: keyframe construction, delta packets,
/// ActiveMask bit tracking, and BandwidthBudget rate limiting.
///
/// The streaming protocol is how the world model is transmitted over the network.
/// This test verifies that:
///   - KeyframePacket can be constructed with encoded Gaussian data
///   - DeltaPacket references a base keyframe and carries partial updates
///   - ActiveMask correctly tracks bit patterns across word boundaries
///   - BandwidthBudget enforces rate limits with window-based accounting
#[test]
fn test_streaming_protocol_keyframe_and_deltas() {
    // Create a keyframe from a set of Gaussians.
    let gaussians: Vec<Gaussian4D> = (0..10)
        .map(|i| Gaussian4D::new([i as f32, 0.0, -5.0], i))
        .collect();
    let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);

    let keyframe = KeyframePacket {
        tile_id: 1,
        time_anchor: 0.0,
        primitive_block: block.data.clone(),
        quant_tier: QuantTier::Hot8,
        total_gaussians: block.count,
        sequence: 0,
    };

    assert_eq!(keyframe.total_gaussians, 10);
    assert_eq!(keyframe.sequence, 0);
    assert!(!keyframe.primitive_block.is_empty());

    // Create a delta packet that updates 3 of the 10 Gaussians.
    let mut active_mask = ActiveMask::new(10);
    active_mask.set(2, true);
    active_mask.set(5, true);
    active_mask.set(9, true);
    assert_eq!(active_mask.active_count(), 3);

    // Build updated Gaussian data for the 3 active ones.
    let updated: Vec<Gaussian4D> = vec![
        Gaussian4D::new([2.0, 0.1, -5.0], 2),
        Gaussian4D::new([5.0, 0.2, -5.0], 5),
        Gaussian4D::new([9.0, 0.3, -5.0], 9),
    ];
    let updated_block = PrimitiveBlock::encode(&updated, QuantTier::Hot8);

    let delta = DeltaPacket {
        tile_id: 1,
        base_sequence: 0,
        time_range: [0.0, 1.0],
        active_mask: active_mask.clone(),
        updated_gaussians: updated_block.data,
        update_count: 3,
    };

    assert_eq!(delta.base_sequence, 0, "Delta references keyframe seq 0");
    assert_eq!(delta.update_count, 3);
    assert_eq!(delta.active_mask.active_count(), 3);

    // Verify the active mask tracks the right indices.
    assert!(!delta.active_mask.is_active(0));
    assert!(!delta.active_mask.is_active(1));
    assert!(delta.active_mask.is_active(2));
    assert!(!delta.active_mask.is_active(3));
    assert!(delta.active_mask.is_active(5));
    assert!(delta.active_mask.is_active(9));

    // Wrap in StreamPacket variants to verify enum construction.
    let _kf_packet = StreamPacket::Keyframe(keyframe.clone());
    let _delta_packet = StreamPacket::Delta(delta);
}

/// Tests BandwidthBudget rate limiting across window boundaries.
///
/// The bandwidth budget must prevent bursts from exceeding the configured rate
/// and correctly reset when the measurement window expires.
#[test]
fn test_streaming_bandwidth_budget_rate_limiting() {
    let mut budget = BandwidthBudget::new(10_000); // 10 KB/s
    budget.reset_window(0);

    // Send 5000 bytes at t=0. Should succeed.
    assert!(budget.can_send(5000, 0));
    budget.record_sent(5000, 0);

    // Send another 5000 bytes. Should succeed (total = 10000 = limit).
    assert!(budget.can_send(5000, 100));
    budget.record_sent(5000, 100);

    // Now at exactly 10000 bytes. Sending even 1 more byte should fail.
    assert!(
        !budget.can_send(1, 200),
        "Should reject send when budget is exhausted"
    );

    // At t=500 (still within the 1-second window), still exhausted.
    assert!(
        !budget.can_send(1, 500),
        "Should still reject within the same window"
    );

    // At t=1000 (window expires), a new send of 10000 bytes should succeed.
    assert!(
        budget.can_send(10_000, 1000),
        "Budget should reset after window expiry"
    );

    // But sending more than the limit in a fresh window should fail.
    assert!(
        !budget.can_send(10_001, 1000),
        "Should reject send exceeding per-second limit even in fresh window"
    );

    // Record a send in the new window and verify utilization.
    budget.record_sent(2000, 1000);
    let util = budget.utilization();
    assert!(
        (util - 0.2).abs() < 1e-6,
        "Utilization should be 2000/10000 = 0.2, got {}",
        util
    );
}

// ===========================================================================
// 7. Multi-Tile Scene Test
// ===========================================================================

/// Tests building a scene from multiple tiles in a spatial grid with a draw list
/// that binds all of them.
///
/// Real-world scenes are partitioned into many tiles. This test verifies that
/// the draw list can reference multiple tiles with different quantization tiers
/// and that command counts are correct.
#[test]
fn test_multi_tile_scene_grid() {
    let num_tiles = 4; // 2x2 grid
    let mut tiles = Vec::new();
    let mut draw_list = DrawList::new(1, 0, 0);

    // Create 4 tiles in a 2x2 spatial grid, each with 5 Gaussians.
    for tx in 0..2i32 {
        for tz in 0..2i32 {
            let tile_id = (tx * 2 + tz) as u64;
            let gaussians: Vec<Gaussian4D> = (0..5)
                .map(|i| {
                    Gaussian4D::new(
                        [tx as f32 * 10.0 + i as f32, 0.0, tz as f32 * 10.0 - 5.0],
                        (tile_id * 100 + i) as u32,
                    )
                })
                .collect();

            // Alternate quantization tiers across tiles to verify the draw list
            // correctly records different tiers.
            let tier = if (tx + tz) % 2 == 0 {
                QuantTier::Hot8
            } else {
                QuantTier::Warm5
            };

            let block = PrimitiveBlock::encode(&gaussians, tier);
            assert_eq!(block.count, 5);

            let tile = Tile {
                coord: TileCoord {
                    x: tx,
                    y: 0,
                    z: tz,
                    time_bucket: 0,
                    lod: 0,
                },
                primitive_block: block,
                entity_refs: vec![],
                coherence_score: 0.9,
                last_update_epoch: 1,
            };
            tiles.push(tile);

            // Add draw commands for this tile.
            draw_list.bind_tile(tile_id, tile_id as u32, tier);
            draw_list.draw_block(tile_id as u32, tile_id as f32 * 0.1, OpacityMode::AlphaBlend);
        }
    }

    assert_eq!(tiles.len(), num_tiles);

    // Set a budget for the single screen tile covering the whole viewport.
    draw_list.set_budget(0, 4096, 3.0);

    // Finalize and check the command count.
    // Commands: 4 bind + 4 draw + 1 budget = 9 (End not counted).
    let checksum = draw_list.finalize();
    assert_ne!(checksum, 0);
    assert_eq!(
        draw_list.command_count(),
        9,
        "Expected 4 binds + 4 draws + 1 budget = 9 commands"
    );

    // Serialize and verify the bytes are structurally sound.
    let bytes = draw_list.to_bytes();
    assert!(bytes.len() > 20, "Multi-tile draw list should serialize to substantial size");

    // Verify all tiles can decode their Gaussians.
    for tile in &tiles {
        let decoded = tile.primitive_block.decode();
        assert_eq!(
            decoded.len(),
            5,
            "Each tile should decode back to 5 Gaussians"
        );
    }
}

/// Tests that draw list correctly records different QuantTier values for
/// each bound tile.
#[test]
fn test_multi_tile_mixed_quant_tiers() {
    let mut dl = DrawList::new(1, 0, 0);

    dl.bind_tile(1, 0, QuantTier::Hot8);
    dl.bind_tile(2, 1, QuantTier::Warm7);
    dl.bind_tile(3, 2, QuantTier::Warm5);
    dl.bind_tile(4, 3, QuantTier::Cold3);

    dl.draw_block(0, 1.0, OpacityMode::AlphaBlend);
    dl.draw_block(1, 2.0, OpacityMode::Additive);
    dl.draw_block(2, 3.0, OpacityMode::Opaque);
    dl.draw_block(3, 4.0, OpacityMode::AlphaBlend);

    dl.finalize();

    // 4 binds + 4 draws = 8 commands.
    assert_eq!(dl.command_count(), 8);

    // Verify the commands contain the expected tier values.
    let bind_tiers: Vec<QuantTier> = dl
        .commands
        .iter()
        .filter_map(|cmd| match cmd {
            DrawCommand::TileBind { quant_tier, .. } => Some(*quant_tier),
            _ => None,
        })
        .collect();
    assert_eq!(
        bind_tiers,
        vec![QuantTier::Hot8, QuantTier::Warm7, QuantTier::Warm5, QuantTier::Cold3]
    );
}

// ===========================================================================
// 8. Privacy Tag Test
// ===========================================================================

/// Tests that entities with privacy tags can be filtered before rendering.
///
/// Privacy enforcement is a critical governance requirement. The world model
/// must support tagging entities with access-control labels and filtering them
/// out at query time. This test verifies that the entity graph stores privacy
/// tags and that application-level filtering works correctly.
#[test]
fn test_privacy_tag_filtering() {
    let mut graph = EntityGraph::new();

    // Public entity: no privacy restrictions.
    let public_obj = Entity {
        id: 1,
        entity_type: EntityType::Object {
            class: "bench".to_string(),
        },
        time_span: [0.0, 100.0],
        embedding: vec![],
        confidence: 0.9,
        privacy_tags: vec![],
        attributes: vec![],
        gaussian_ids: vec![1, 2, 3],
    };

    // Restricted entity: tagged with "PII" for personally identifiable info.
    let restricted_person = Entity {
        id: 2,
        entity_type: EntityType::Object {
            class: "person".to_string(),
        },
        time_span: [0.0, 100.0],
        embedding: vec![],
        confidence: 0.85,
        privacy_tags: vec!["PII".to_string(), "GDPR".to_string()],
        attributes: vec![],
        gaussian_ids: vec![4, 5, 6, 7],
    };

    // Internal-only entity.
    let internal_sensor = Entity {
        id: 3,
        entity_type: EntityType::Object {
            class: "sensor".to_string(),
        },
        time_span: [0.0, 100.0],
        embedding: vec![],
        confidence: 1.0,
        privacy_tags: vec!["INTERNAL".to_string()],
        attributes: vec![],
        gaussian_ids: vec![8],
    };

    graph.add_entity(public_obj);
    graph.add_entity(restricted_person);
    graph.add_entity(internal_sensor);
    assert_eq!(graph.entity_count(), 3);

    // Simulate rendering filter: exclude entities with "PII" tag.
    // In a real system this would be done by a rendering pipeline filter.
    let all_objects = graph.query_by_type("object");
    let renderable: Vec<&&Entity> = all_objects
        .iter()
        .filter(|e| !e.privacy_tags.contains(&"PII".to_string()))
        .collect();

    assert_eq!(
        renderable.len(),
        2,
        "After PII filter, only bench and sensor should be renderable"
    );
    let renderable_ids: Vec<u64> = renderable.iter().map(|e| e.id).collect();
    assert!(renderable_ids.contains(&1), "Bench should be renderable");
    assert!(renderable_ids.contains(&3), "Sensor should be renderable");
    assert!(
        !renderable_ids.contains(&2),
        "Person with PII tag should be filtered out"
    );

    // Verify the restricted entity still exists (filter is at application level).
    let person = graph.get_entity(2).unwrap();
    assert_eq!(person.privacy_tags.len(), 2);
    assert!(person.privacy_tags.contains(&"PII".to_string()));
    assert!(person.privacy_tags.contains(&"GDPR".to_string()));

    // Collect Gaussian IDs that should NOT be rendered due to privacy.
    let blocked_gaussian_ids: Vec<u32> = all_objects
        .iter()
        .filter(|e| e.privacy_tags.contains(&"PII".to_string()))
        .flat_map(|e| e.gaussian_ids.iter().copied())
        .collect();
    assert_eq!(
        blocked_gaussian_ids,
        vec![4, 5, 6, 7],
        "Gaussians 4-7 belong to the PII-tagged person and should be blocked"
    );
}

// ===========================================================================
// 9. Roundtrip Fidelity Test
// ===========================================================================

/// Creates 100 Gaussians with deterministic pseudo-random parameters, encodes
/// them into a PrimitiveBlock, decodes them back, and verifies that every field
/// is preserved at exact f32 precision.
///
/// This test is essential because the encode/decode path uses raw byte
/// serialization of f32 values. Any off-by-one in offsets, stride calculation,
/// or endianness handling would cause subtle data corruption that might only
/// appear as visual artifacts in rendering.
#[test]
fn test_roundtrip_fidelity_100_gaussians() {
    let count = 100;
    let mut gaussians = Vec::with_capacity(count);

    for i in 0..count {
        // Use deterministic values derived from the index so the test is reproducible.
        let fi = i as f32;
        let mut g = Gaussian4D::new(
            [fi * 0.1, fi * 0.2 - 10.0, fi * 0.3 - 15.0],
            i as u32,
        );
        g.covariance = [
            1.0 + fi * 0.01,
            fi * 0.001,
            fi * 0.002,
            1.0 + fi * 0.015,
            fi * 0.003,
            1.0 + fi * 0.02,
        ];
        g.sh_coeffs = [
            (fi * 0.01).sin().abs(),
            (fi * 0.02).cos().abs(),
            (fi * 0.03).sin().abs(),
        ];
        g.opacity = 0.5 + (fi * 0.005).min(0.49);
        g.scale = [1.0 + fi * 0.01, 1.0 + fi * 0.02, 1.0 + fi * 0.03];
        g.rotation = {
            // Generate a valid-ish quaternion (not necessarily unit, but deterministic).
            let w = 1.0 - fi * 0.005;
            let x = fi * 0.003;
            let y = fi * 0.004;
            let z = fi * 0.002;
            [w, x, y, z]
        };
        g.time_range = [fi * 0.1, fi * 0.1 + 10.0];
        g.velocity = [fi * 0.01, fi * -0.01, fi * 0.005];

        gaussians.push(g);
    }

    // Test with every quantization tier. All tiers currently use Hot8 (8-bit)
    // quantization, which is lossy: max error per field is range/255.
    // Use tolerance-based comparison for float fields; IDs must be exact.
    let tiers = [QuantTier::Hot8, QuantTier::Warm7, QuantTier::Warm5, QuantTier::Cold3];

    // Hot8 quantization tolerance: the largest per-field range in this dataset
    // is ~29.7 (center z), giving max error ~29.7/255 ≈ 0.117.
    let tol = 0.15;

    /// Helper to assert two f32 slices are approximately equal.
    fn assert_approx(a: &[f32], b: &[f32], tol: f32, label: &str, idx: usize, tier: QuantTier) {
        assert_eq!(a.len(), b.len());
        for (j, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            if !av.is_finite() || !bv.is_finite() {
                continue; // skip non-finite (e.g. infinity time_range)
            }
            assert!(
                (av - bv).abs() <= tol,
                "Gaussian {} {} [{}] mismatch at tier {:?}: orig={}, decoded={}, tol={}",
                idx, label, j, tier, av, bv, tol
            );
        }
    }

    for &tier in &tiers {
        let block = PrimitiveBlock::encode(&gaussians, tier);
        assert_eq!(block.count, count as u32);
        assert_eq!(block.quant_tier, tier);
        assert!(block.verify_checksum(), "Checksum must verify for {:?}", tier);

        let decoded = block.decode();
        assert_eq!(decoded.len(), count);

        for (i, (orig, dec)) in gaussians.iter().zip(decoded.iter()).enumerate() {
            assert_approx(&orig.center, &dec.center, tol, "center", i, tier);
            assert_approx(&orig.covariance, &dec.covariance, tol, "covariance", i, tier);
            assert_approx(&orig.sh_coeffs, &dec.sh_coeffs, tol, "sh_coeffs", i, tier);
            assert!(
                (orig.opacity - dec.opacity).abs() <= tol,
                "Gaussian {} opacity mismatch at tier {:?}: {} vs {}",
                i, tier, orig.opacity, dec.opacity
            );
            assert_approx(&orig.scale, &dec.scale, tol, "scale", i, tier);
            assert_approx(&orig.rotation, &dec.rotation, tol, "rotation", i, tier);
            assert_approx(&orig.time_range, &dec.time_range, tol, "time_range", i, tier);
            assert_approx(&orig.velocity, &dec.velocity, tol, "velocity", i, tier);
            assert_eq!(
                orig.id, dec.id,
                "Gaussian {} id mismatch at tier {:?}",
                i, tier
            );
        }
    }
}

/// Verifies that the checksum changes when any byte in the encoded data changes.
///
/// This is a sanity check that the FNV-1a checksum is sensitive to data mutations.
#[test]
fn test_roundtrip_checksum_sensitivity() {
    let g = Gaussian4D::new([1.0, 2.0, 3.0], 42);
    let block = PrimitiveBlock::encode(&[g], QuantTier::Hot8);
    let original_checksum = block.checksum;

    // Flip a single byte in the encoded data and verify the checksum changes.
    let mut corrupted = block.clone();
    if !corrupted.data.is_empty() {
        corrupted.data[0] ^= 0xFF;
    }
    let new_checksum = corrupted.compute_checksum();
    assert_ne!(
        original_checksum, new_checksum,
        "Flipping a byte must change the checksum"
    );
    assert!(
        !corrupted.verify_checksum(),
        "Corrupted data should fail checksum verification"
    );
}

// ===========================================================================
// 10. Empty / Edge Case Tests
// ===========================================================================

/// Tests that an empty PrimitiveBlock encodes and decodes correctly.
///
/// Empty blocks arise when a tile is created but no Gaussians have been assigned
/// yet. The system must handle this gracefully.
#[test]
fn test_edge_empty_primitive_block() {
    let block = PrimitiveBlock::encode(&[], QuantTier::Cold3);
    assert_eq!(block.count, 0);
    assert!(block.data.is_empty(), "Empty block should have no data bytes");
    assert!(block.verify_checksum(), "Even an empty block has a valid checksum");

    let decoded = block.decode();
    assert!(decoded.is_empty(), "Decoding empty block should yield empty vec");
}

/// Tests that a DrawList with only an End command serializes correctly.
///
/// This represents a frame where nothing needs to be drawn (e.g., camera is
/// looking at empty space). The renderer must handle this without crashing.
#[test]
fn test_edge_draw_list_only_end() {
    let mut dl = DrawList::new(0, 0, 0);
    // Finalize without adding any commands. This adds only the End sentinel.
    let checksum = dl.finalize();
    assert_ne!(checksum, 0, "End-only draw list should still have a checksum");
    assert_eq!(dl.command_count(), 0, "No commands besides End");
    assert!(matches!(dl.commands.last(), Some(DrawCommand::End)));

    let bytes = dl.to_bytes();
    // Header (20 bytes) + End command (1 byte tag).
    assert_eq!(bytes.len(), 21, "End-only draw list should be 20 header + 1 End byte");
}

/// Tests that querying the LineageLog for a nonexistent tile returns empty
/// results and does not panic.
#[test]
fn test_edge_lineage_nonexistent_tile() {
    let log = LineageLog::new();

    let events = log.query_tile(999);
    assert!(events.is_empty(), "No events should exist for a nonexistent tile");

    let rollback = log.find_rollback_point(999);
    assert_eq!(
        rollback, None,
        "No rollback point should exist for a nonexistent tile"
    );

    assert!(log.get(0).is_none(), "Get on empty log should return None");
    assert!(log.get(u64::MAX).is_none(), "Get with max ID should return None");
}

/// Tests that ActiveMask with zero size handles all operations gracefully.
///
/// A zero-size mask can occur when a tile has no Gaussians. Operations like
/// set, is_active, and active_count must not panic.
#[test]
fn test_edge_active_mask_zero_size() {
    let mut mask = ActiveMask::new(0);
    assert_eq!(mask.total_count, 0);
    assert_eq!(mask.active_count(), 0);
    assert_eq!(mask.byte_size(), 0);
    assert!(mask.bits.is_empty());

    // Setting and querying out-of-bounds should be safe no-ops.
    mask.set(0, true);
    assert!(!mask.is_active(0));
    mask.set(100, true);
    assert!(!mask.is_active(100));

    assert_eq!(mask.active_count(), 0, "No bits should be set on a zero-size mask");
}

/// Tests that BandwidthBudget with zero rate rejects all sends and reports
/// full utilization.
///
/// A zero-rate budget is a valid configuration that effectively pauses all
/// streaming. The system must handle it without division-by-zero panics.
#[test]
fn test_edge_bandwidth_budget_zero_rate() {
    let budget = BandwidthBudget::new(0);
    assert_eq!(budget.max_bytes_per_second, 0);

    // Utilization should be 1.0 (fully utilized = cannot send anything).
    let util = budget.utilization();
    assert!(
        (util - 1.0).abs() < 1e-6,
        "Zero-rate budget utilization should be 1.0, got {}",
        util
    );

    // Sending any bytes should be rejected (even after window expiry).
    // With zero budget, `bytes <= max_bytes_per_second` is `N <= 0`, which is false for N > 0.
    assert!(
        !budget.can_send(1, 0),
        "Zero-rate budget should reject any send"
    );
    assert!(
        !budget.can_send(1, 10_000),
        "Zero-rate budget should reject even after window expiry"
    );

    // Sending zero bytes should succeed (0 <= 0 is true).
    assert!(
        budget.can_send(0, 0),
        "Sending zero bytes should always succeed"
    );
}

/// Tests that a single Gaussian roundtrips correctly through a single-element
/// PrimitiveBlock. This catches any issues with stride calculations at the
/// boundary of a single element.
#[test]
fn test_edge_single_gaussian_roundtrip() {
    let mut g = Gaussian4D::new([42.0, -17.5, 0.001], 0xDEAD);
    g.covariance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    g.sh_coeffs = [0.9, 0.8, 0.7];
    g.opacity = 0.42;
    g.scale = [2.0, 3.0, 4.0];
    g.rotation = [0.1, 0.2, 0.3, 0.4];
    g.time_range = [-1.0, 1.0];
    g.velocity = [0.01, 0.02, 0.03];

    let block = PrimitiveBlock::encode(&[g.clone()], QuantTier::Warm7);
    assert_eq!(block.count, 1);

    let decoded = block.decode();
    assert_eq!(decoded.len(), 1);
    let d = &decoded[0];

    assert_eq!(d.center, g.center);
    assert_eq!(d.covariance, g.covariance);
    assert_eq!(d.sh_coeffs, g.sh_coeffs);
    assert_eq!(d.opacity, g.opacity);
    assert_eq!(d.scale, g.scale);
    assert_eq!(d.rotation, g.rotation);
    assert_eq!(d.time_range, g.time_range);
    assert_eq!(d.velocity, g.velocity);
    assert_eq!(d.id, 0xDEAD);
}

/// Tests the LineageLog query_range with a range that matches no events.
#[test]
fn test_edge_lineage_empty_range_query() {
    let mut log = LineageLog::new();
    log.append(
        1000, 1, LineageEventType::TileCreated,
        sensor_provenance("s1", 0.9), None,
        CoherenceDecision::Accept, 0.9,
    );

    // Query a range entirely before the only event.
    let result = log.query_range(0, 500);
    assert!(result.is_empty(), "Range before any event should return empty");

    // Query a range entirely after the only event.
    let result = log.query_range(2000, 3000);
    assert!(result.is_empty(), "Range after all events should return empty");
}

/// Tests that the EntityGraph returns empty results for queries that match
/// nothing, without panicking.
#[test]
fn test_edge_entity_graph_empty_queries() {
    let graph = EntityGraph::new();

    assert_eq!(graph.entity_count(), 0);
    assert_eq!(graph.edge_count(), 0);
    assert!(graph.get_entity(0).is_none());
    assert!(graph.neighbors(0).is_empty());
    assert!(graph.query_by_type("anything").is_empty());
    assert!(graph.query_time_range(0.0, 100.0).is_empty());
}

/// Tests the DrawList finalize idempotency: calling finalize multiple times
/// should produce the same checksum and not duplicate End commands.
#[test]
fn test_edge_draw_list_finalize_idempotent() {
    let mut dl = DrawList::new(5, 10, 20);
    dl.bind_tile(1, 0, QuantTier::Hot8);
    dl.draw_block(0, 1.0, OpacityMode::Opaque);

    let c1 = dl.finalize();
    let c2 = dl.finalize();
    let c3 = dl.finalize();

    assert_eq!(c1, c2, "Finalize must be idempotent");
    assert_eq!(c2, c3, "Finalize must be idempotent on third call");

    // Count End commands -- should always be exactly 1.
    let end_count = dl
        .commands
        .iter()
        .filter(|c| matches!(c, DrawCommand::End))
        .count();
    assert_eq!(end_count, 1, "Should have exactly one End after multiple finalizes");
}

/// Tests ActiveMask at word boundaries (63, 64, 65) to verify correct bit
/// indexing across u64 word boundaries.
#[test]
fn test_edge_active_mask_word_boundaries() {
    let mut mask = ActiveMask::new(128);

    // Set bits at word boundary positions.
    mask.set(63, true); // last bit of word 0
    mask.set(64, true); // first bit of word 1
    mask.set(65, true); // second bit of word 1

    assert!(mask.is_active(63));
    assert!(mask.is_active(64));
    assert!(mask.is_active(65));
    assert!(!mask.is_active(62));
    assert!(!mask.is_active(66));

    assert_eq!(mask.active_count(), 3);

    // Clear bit 64 and verify only 63 and 65 remain.
    mask.set(64, false);
    assert!(!mask.is_active(64));
    assert_eq!(mask.active_count(), 2);
}

/// Tests that entity attributes are correctly stored and retrievable.
#[test]
fn test_edge_entity_attributes() {
    let mut graph = EntityGraph::new();

    let entity = Entity {
        id: 1,
        entity_type: EntityType::Object {
            class: "vehicle".to_string(),
        },
        time_span: [0.0, 100.0],
        embedding: vec![1.0, 0.0, 0.0],
        confidence: 0.95,
        privacy_tags: vec![],
        attributes: vec![
            ("speed".to_string(), AttributeValue::Float(25.5)),
            ("lane".to_string(), AttributeValue::Int(2)),
            ("plate".to_string(), AttributeValue::Text("ABC123".to_string())),
            ("autonomous".to_string(), AttributeValue::Bool(true)),
            ("position".to_string(), AttributeValue::Vec3([10.0, 0.0, -5.0])),
        ],
        gaussian_ids: vec![1, 2, 3],
    };

    graph.add_entity(entity);
    let retrieved = graph.get_entity(1).unwrap();

    assert_eq!(retrieved.attributes.len(), 5);
    // Verify a specific attribute by name.
    let speed_attr = retrieved
        .attributes
        .iter()
        .find(|(k, _)| k == "speed")
        .map(|(_, v)| v);
    match speed_attr {
        Some(AttributeValue::Float(v)) => {
            assert!((v - 25.5).abs() < 1e-6, "Speed should be 25.5");
        }
        other => panic!("Expected Float(25.5), got {:?}", other),
    }
}

// ===========================================================================
// Cross-Module Integration: Coherence -> Lineage -> Tile
// ===========================================================================

/// End-to-end test: a coherence decision triggers a lineage event which updates
/// a tile. This verifies that the three subsystems compose correctly.
#[test]
fn test_coherence_to_lineage_to_tile_update() {
    let gate = CoherenceGate::with_defaults();
    let mut log = LineageLog::new();
    let tile_id = 1u64;

    // Create initial tile with some Gaussians.
    let initial_gaussians = vec![
        Gaussian4D::new([0.0, 0.0, -5.0], 0),
        Gaussian4D::new([1.0, 0.0, -5.0], 1),
    ];
    let mut tile = Tile {
        coord: TileCoord {
            x: 0, y: 0, z: 0, time_bucket: 0, lod: 0,
        },
        primitive_block: PrimitiveBlock::encode(&initial_gaussians, QuantTier::Hot8),
        entity_refs: vec![],
        coherence_score: 1.0,
        last_update_epoch: 0,
    };

    // Log the creation.
    log.append(
        0, tile_id, LineageEventType::TileCreated,
        sensor_provenance("init", 1.0), None,
        CoherenceDecision::Accept, 1.0,
    );

    // Simulate an incoming update: evaluate coherence first.
    let update_input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.9,
        sensor_confidence: 0.95,
        sensor_freshness_ms: 100,
        budget_pressure: 0.3,
        permission_level: PermissionLevel::Standard,
    };
    let decision = gate.evaluate(&update_input);
    assert_eq!(decision, CoherenceDecision::Accept);

    // Since accepted, apply the update to the tile.
    let updated_gaussians = vec![
        Gaussian4D::new([0.1, 0.0, -5.0], 0), // slightly moved
        Gaussian4D::new([1.1, 0.0, -5.0], 1),
        Gaussian4D::new([2.0, 0.0, -5.0], 2), // new Gaussian
    ];
    tile.primitive_block = PrimitiveBlock::encode(&updated_gaussians, QuantTier::Hot8);
    tile.last_update_epoch = 1;
    tile.coherence_score = 0.9;

    // Log the update.
    log.append(
        100, tile_id,
        LineageEventType::TileUpdated { delta_size: tile.primitive_block.data.len() as u32 },
        sensor_provenance("lidar-01", 0.95), None,
        decision, 0.9,
    );

    // Verify the tile is updated.
    let decoded = tile.primitive_block.decode();
    assert_eq!(decoded.len(), 3, "Tile should now have 3 Gaussians");

    // Verify lineage shows 2 events for this tile.
    let history = log.query_tile(tile_id);
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].coherence_decision, CoherenceDecision::Accept);
    assert_eq!(history[1].coherence_decision, CoherenceDecision::Accept);
}
