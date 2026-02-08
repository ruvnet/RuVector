//! Performance acceptance tests.
//!
//! These verify that operations complete within bounded time.  They are **not**
//! micro-benchmarks -- they assert wall-clock performance envelopes that must
//! hold on any reasonable CI machine.

use std::time::Instant;

use ruvector_vwm::coherence::{CoherenceGate, CoherenceInput, PermissionLevel};
use ruvector_vwm::draw_list::{DrawList, OpacityMode};
use ruvector_vwm::gaussian::Gaussian4D;
use ruvector_vwm::streaming::ActiveMask;
use ruvector_vwm::tile::{PrimitiveBlock, QuantTier};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a realistic Gaussian with non-zero velocity and bounded time range.
fn make_gaussian(i: usize) -> Gaussian4D {
    let fi = i as f32;
    let mut g = Gaussian4D::new(
        [fi * 0.1, (fi * 0.7).sin() * 5.0, -10.0 + (fi * 0.3).cos()],
        i as u32,
    );
    if i % 5 == 0 {
        g.velocity = [0.01 * fi.sin(), 0.02 * fi.cos(), 0.005];
        g.time_range = [0.0, 100.0];
    }
    g.opacity = 0.5 + 0.5 * ((fi * 0.1).sin()).abs();
    g.sh_coeffs = [
        0.3 + 0.2 * (fi * 0.05).sin(),
        0.4 + 0.1 * (fi * 0.07).cos(),
        0.35,
    ];
    g.scale = [
        0.5 + 0.5 * (fi * 0.03).sin().abs(),
        0.5 + 0.5 * (fi * 0.04).cos().abs(),
        0.5 + 0.5 * (fi * 0.05).sin().abs(),
    ];
    g
}

// ---------------------------------------------------------------------------
// Acceptance tests
// ---------------------------------------------------------------------------

#[test]
fn test_encode_decode_throughput() {
    // Encode + decode 100K Gaussians must complete under 1 second.
    let gaussians: Vec<Gaussian4D> = (0..100_000).map(make_gaussian).collect();

    let start = Instant::now();
    let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
    let decoded = block.decode();
    let elapsed = start.elapsed();

    assert_eq!(decoded.len(), 100_000);
    assert!(
        elapsed.as_millis() < 1000,
        "100K encode+decode took {:?}",
        elapsed
    );
}

#[test]
fn test_coherence_gate_throughput() {
    // 1M coherence evaluations must complete under 1 second.
    let gate = CoherenceGate::with_defaults();
    let input = CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 50,
        budget_pressure: 0.2,
        permission_level: PermissionLevel::Standard,
    };

    let start = Instant::now();
    for _ in 0..1_000_000 {
        let _ = gate.evaluate(&input);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 1000,
        "1M evaluations took {:?}",
        elapsed
    );
}

#[test]
fn test_active_mask_throughput() {
    // Set + query 500K bits must complete under 500 ms.
    let mut mask = ActiveMask::new(500_000);

    let start = Instant::now();
    for i in 0..500_000u32 {
        mask.set(i, i % 3 == 0);
    }
    for i in 0..500_000u32 {
        let _ = mask.is_active(i);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "500K mask ops took {:?}",
        elapsed
    );
}

#[test]
fn test_draw_list_build_throughput() {
    // Build a 10K-command draw list and serialize it under 100 ms.
    let start = Instant::now();
    let mut dl = DrawList::new(1, 0, 0);
    for i in 0..10_000u32 {
        dl.bind_tile(i as u64, i, QuantTier::Hot8);
        dl.draw_block(i, i as f32, OpacityMode::AlphaBlend);
    }
    dl.finalize();
    let bytes = dl.to_bytes();
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 100, "draw list build took {:?}", elapsed);
    assert!(!bytes.is_empty());
}

#[test]
fn test_checksum_consistency() {
    // Verify that checksum is deterministic and verify_checksum passes for
    // a range of block sizes.
    for &n in &[0usize, 1, 100, 1_000, 10_000] {
        let gaussians: Vec<Gaussian4D> = (0..n).map(make_gaussian).collect();
        let block = PrimitiveBlock::encode(&gaussians, QuantTier::Warm7);
        assert!(
            block.verify_checksum(),
            "checksum verification failed for block with {} gaussians",
            n
        );
        // Recompute should match stored value.
        assert_eq!(block.compute_checksum(), block.checksum);
    }
}

#[test]
fn test_encode_decode_fidelity() {
    // All fields must survive a round-trip through encode/decode.
    let mut g = Gaussian4D::new([1.5, -2.3, 7.7], 42);
    g.covariance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    g.sh_coeffs = [0.7, 0.8, 0.9];
    g.opacity = 0.65;
    g.scale = [1.1, 2.2, 3.3];
    g.rotation = [0.5, 0.5, 0.5, 0.5];
    g.time_range = [10.0, 90.0];
    g.velocity = [0.01, -0.02, 0.03];

    for &tier in &[QuantTier::Hot8, QuantTier::Warm7, QuantTier::Warm5, QuantTier::Cold3] {
        let block = PrimitiveBlock::encode(&[g.clone()], tier);
        let decoded = block.decode();
        assert_eq!(decoded.len(), 1, "tier {:?}: wrong count", tier);
        let d = &decoded[0];
        assert_eq!(d.center, g.center, "tier {:?}: center mismatch", tier);
        assert_eq!(d.covariance, g.covariance, "tier {:?}: covariance mismatch", tier);
        assert_eq!(d.sh_coeffs, g.sh_coeffs, "tier {:?}: sh_coeffs mismatch", tier);
        assert_eq!(d.opacity, g.opacity, "tier {:?}: opacity mismatch", tier);
        assert_eq!(d.scale, g.scale, "tier {:?}: scale mismatch", tier);
        assert_eq!(d.rotation, g.rotation, "tier {:?}: rotation mismatch", tier);
        assert_eq!(d.time_range, g.time_range, "tier {:?}: time_range mismatch", tier);
        assert_eq!(d.velocity, g.velocity, "tier {:?}: velocity mismatch", tier);
        assert_eq!(d.id, g.id, "tier {:?}: id mismatch", tier);
    }
}

#[test]
fn test_gaussian_projection_throughput() {
    // Project 10K Gaussians through a view-projection matrix under 50 ms.
    let gaussians: Vec<Gaussian4D> = (0..10_000).map(make_gaussian).collect();
    let vp: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.1,
        0.0, 0.0, 0.0, 1.0,
    ];

    let start = Instant::now();
    let mut projected = 0usize;
    for g in &gaussians {
        if g.project(&vp, 50.0).is_some() {
            projected += 1;
        }
    }
    let elapsed = start.elapsed();

    assert!(projected > 0, "no gaussians projected successfully");
    assert!(
        elapsed.as_millis() < 50,
        "10K projections took {:?}",
        elapsed
    );
}

#[test]
fn test_depth_sort_throughput() {
    // Sort 50K screen gaussians by depth under 100 ms.
    let gaussians: Vec<Gaussian4D> = (0..100_000).map(make_gaussian).collect();
    let vp: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.1,
        0.0, 0.0, 0.0, 1.0,
    ];

    let mut screen_gs: Vec<_> = gaussians
        .iter()
        .filter_map(|g| g.project(&vp, 50.0))
        .take(50_000)
        .collect();

    let count = screen_gs.len();
    assert!(count > 1000, "not enough screen gaussians for sort test (got {})", count);

    let start = Instant::now();
    screen_gs.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "depth sort of {} screen gaussians took {:?}",
        count,
        elapsed
    );
    // Verify sorted order.
    for w in screen_gs.windows(2) {
        assert!(w[0].depth <= w[1].depth || w[1].depth.is_nan());
    }
}

#[test]
fn test_coherence_diverse_decisions() {
    // Verify the gate produces all four decision types with varied inputs.
    let gate = CoherenceGate::with_defaults();

    let accept = gate.evaluate(&CoherenceInput {
        tile_disagreement: 0.05,
        entity_continuity: 0.95,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 10,
        budget_pressure: 0.1,
        permission_level: PermissionLevel::Standard,
    });
    assert_eq!(accept, ruvector_vwm::CoherenceDecision::Accept);

    let defer = gate.evaluate(&CoherenceInput {
        tile_disagreement: 0.1,
        entity_continuity: 0.5,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 10,
        budget_pressure: 0.1,
        permission_level: PermissionLevel::Standard,
    });
    assert_eq!(defer, ruvector_vwm::CoherenceDecision::Defer);

    let freeze = gate.evaluate(&CoherenceInput {
        tile_disagreement: 0.85,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 10,
        budget_pressure: 0.1,
        permission_level: PermissionLevel::Standard,
    });
    assert_eq!(freeze, ruvector_vwm::CoherenceDecision::Freeze);

    let rollback = gate.evaluate(&CoherenceInput {
        tile_disagreement: 0.96,
        entity_continuity: 0.9,
        sensor_confidence: 1.0,
        sensor_freshness_ms: 10,
        budget_pressure: 0.1,
        permission_level: PermissionLevel::Standard,
    });
    assert_eq!(rollback, ruvector_vwm::CoherenceDecision::Rollback);
}

#[test]
fn test_active_mask_correctness_at_scale() {
    // Set a pattern on a large mask and verify correctness.
    let n = 200_000u32;
    let mut mask = ActiveMask::new(n);
    for i in 0..n {
        mask.set(i, i % 7 == 0);
    }

    // Verify every bit
    for i in 0..n {
        let expected = i % 7 == 0;
        assert_eq!(
            mask.is_active(i),
            expected,
            "mismatch at index {}",
            i
        );
    }

    // Verify count
    let expected_count = (0..n).filter(|i| i % 7 == 0).count() as u32;
    assert_eq!(mask.active_count(), expected_count);
}
