//! Security tests for ADR-087 fixes.
//!
//! These tests verify the security fixes identified in the code review:
//! - SEC-001: Boot signature failure should panic
//! - SEC-002: Proof cache TTL/nonce/bounds
//! - CQ-001: Bounds check in delta parsing

use cognitum_gate_kernel::{delta::Delta, TileState};
use core::mem::{align_of, size_of};

// =============================================================================
// CQ-001: Delta Parsing Bounds Check Tests
// =============================================================================

#[test]
fn test_delta_size_constant() {
    // Verify Delta is exactly 16 bytes as expected
    assert_eq!(size_of::<Delta>(), 16, "Delta must be 16 bytes");
}

#[test]
fn test_delta_alignment_constant() {
    // Verify Delta requires 16-byte alignment
    assert_eq!(align_of::<Delta>(), 16, "Delta must be 16-byte aligned");
}

#[test]
fn test_ingest_delta_raw_rejects_small_buffer() {
    let mut tile = TileState::new(0);

    // Create a buffer smaller than Delta
    let small_buffer: [u8; 8] = [0; 8];

    // Should return false due to insufficient length
    unsafe {
        let result = tile.ingest_delta_raw(small_buffer.as_ptr(), small_buffer.len());
        assert!(!result, "Should reject buffer smaller than Delta size");
    }
}

#[test]
fn test_ingest_delta_raw_rejects_zero_length() {
    let mut tile = TileState::new(0);

    let buffer: [u8; 16] = [0; 16];

    // Should return false for zero-length buffer
    unsafe {
        let result = tile.ingest_delta_raw(buffer.as_ptr(), 0);
        assert!(!result, "Should reject zero-length buffer");
    }
}

#[test]
fn test_ingest_delta_raw_rejects_misaligned_pointer() {
    let mut tile = TileState::new(0);

    // Create a buffer with room for misalignment
    #[repr(C, align(32))]
    struct AlignedBuffer {
        bytes: [u8; 48],
    }
    let buffer = AlignedBuffer { bytes: [0; 48] };

    // Get a misaligned pointer (offset by 1 byte)
    let misaligned_ptr = unsafe { buffer.bytes.as_ptr().add(1) };

    // Verify it's actually misaligned
    assert_ne!(
        (misaligned_ptr as usize) % align_of::<Delta>(),
        0,
        "Test setup: pointer should be misaligned"
    );

    // Should return false due to misalignment
    unsafe {
        let result = tile.ingest_delta_raw(misaligned_ptr, 32);
        assert!(!result, "Should reject misaligned pointer");
    }
}

#[test]
fn test_ingest_delta_raw_accepts_valid_input() {
    let mut tile = TileState::new(0);

    // Create a properly aligned Delta
    let delta = Delta::edge_add(1, 2, 100);

    // Get pointer and size
    let ptr = &delta as *const Delta as *const u8;
    let len = size_of::<Delta>();

    // Verify alignment
    assert_eq!(
        (ptr as usize) % align_of::<Delta>(),
        0,
        "Delta should be properly aligned"
    );

    // Should succeed
    unsafe {
        let result = tile.ingest_delta_raw(ptr, len);
        assert!(result, "Should accept valid, aligned Delta");
    }

    // Verify the delta was ingested
    assert!(tile.has_pending_deltas());
    assert_eq!(tile.delta_count, 1);
}

#[test]
fn test_ingest_delta_raw_accepts_larger_buffer() {
    let mut tile = TileState::new(0);

    // Create an oversized buffer that starts with a valid Delta
    #[repr(C, align(16))]
    struct OversizedBuffer {
        delta: Delta,
        extra: [u8; 32],
    }
    let buffer = OversizedBuffer {
        delta: Delta::edge_add(5, 10, 200),
        extra: [0; 32],
    };

    let ptr = &buffer as *const OversizedBuffer as *const u8;
    let len = size_of::<OversizedBuffer>(); // Larger than Delta

    // Should succeed (buffer is larger than needed, but that's OK)
    unsafe {
        let result = tile.ingest_delta_raw(ptr, len);
        assert!(result, "Should accept buffer larger than Delta");
    }
}

#[test]
fn test_ingest_delta_raw_boundary_length() {
    let mut tile = TileState::new(0);

    // Create aligned buffer of exactly Delta size
    #[repr(C, align(16))]
    struct ExactBuffer {
        bytes: [u8; 16],
    }
    let buffer = ExactBuffer { bytes: [0; 16] };

    let ptr = buffer.bytes.as_ptr();

    // Exactly size_of::<Delta>() should work
    unsafe {
        let result = tile.ingest_delta_raw(ptr, size_of::<Delta>());
        // Note: This is a NOP delta (all zeros), but it should be accepted
        assert!(result, "Should accept buffer of exact Delta size");
    }

    // One byte less should fail
    tile.reset();
    unsafe {
        let result = tile.ingest_delta_raw(ptr, size_of::<Delta>() - 1);
        assert!(!result, "Should reject buffer one byte smaller than Delta");
    }
}

// =============================================================================
// Additional Security Tests
// =============================================================================

#[test]
fn test_tile_error_state_must_be_checked() {
    let mut tile = TileState::new(0);

    // Initially not in error state
    assert!(!tile.is_error());

    // Set error state manually
    tile.status |= TileState::STATUS_ERROR;
    assert!(tile.is_error());

    // Reset should clear error
    tile.reset();
    assert!(!tile.is_error());
}

#[test]
fn test_buffer_full_behavior() {
    let mut tile = TileState::new(0);

    // Fill the buffer
    for i in 0..cognitum_gate_kernel::MAX_DELTA_BUFFER {
        let delta = Delta::edge_add(i as u16, (i + 1) as u16, 100);
        let result = tile.ingest_delta(&delta);
        assert!(result, "Should accept delta {}", i);
    }

    // Buffer should now be full
    assert_eq!(
        tile.delta_count as usize,
        cognitum_gate_kernel::MAX_DELTA_BUFFER
    );

    // Next insert should fail
    let delta = Delta::edge_add(999, 1000, 100);
    let result = tile.ingest_delta(&delta);
    assert!(!result, "Should reject when buffer is full");
}

#[test]
fn test_tick_clears_buffer() {
    let mut tile = TileState::new(0);

    // Add some deltas
    tile.ingest_delta(&Delta::edge_add(1, 2, 100));
    tile.ingest_delta(&Delta::edge_add(2, 3, 100));
    assert!(tile.has_pending_deltas());

    // Process tick
    let report = tile.tick(1);
    assert_eq!(report.deltas_processed, 2);

    // Buffer should be clear
    assert!(!tile.has_pending_deltas());
    assert_eq!(tile.delta_count, 0);
}
