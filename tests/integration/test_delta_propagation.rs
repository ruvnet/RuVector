//! Integration tests for ruvector-delta-core: VectorDelta, SparseDelta, DeltaStream,
//! DeltaWindow, encoding, and compression.
//!
//! All tests use real types and real computations -- no mocks.

use ruvector_delta_core::delta::{Delta, DeltaOp, DeltaValue, SparseDelta, VectorDelta};
use ruvector_delta_core::encoding::{
    DeltaEncoding, DenseEncoding, HybridEncoding, RunLengthEncoding, SparseEncoding,
};
use ruvector_delta_core::stream::{DeltaStream, DeltaStreamConfig};
use ruvector_delta_core::window::{DeltaWindow, SumAggregator, WindowAggregator, WindowResult};
use smallvec::SmallVec;

// ===========================================================================
// 1. VectorDelta: compute sparse delta and apply
// ===========================================================================
#[test]
fn vector_delta_compute_sparse_and_apply() {
    let old = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let new = vec![1.0f32, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let delta = VectorDelta::compute(&old, &new);

    // Only 1 element changed out of 10 -- should be sparse
    assert!(
        matches!(delta.value, DeltaValue::Sparse(_)),
        "delta should be sparse when only 1/10 elements changed"
    );

    let mut result = old.clone();
    delta
        .apply(&mut result)
        .expect("apply sparse delta should succeed");

    for (i, (&r, &expected)) in result.iter().zip(new.iter()).enumerate() {
        assert!(
            (r - expected).abs() < 1e-6,
            "element {i}: expected {expected}, got {r}"
        );
    }
}

// ===========================================================================
// 2. VectorDelta: compute dense delta when most elements change
// ===========================================================================
#[test]
fn vector_delta_compute_dense() {
    let old = vec![1.0f32, 2.0, 3.0, 4.0];
    let new = vec![5.0f32, 6.0, 7.0, 8.0];

    let delta = VectorDelta::compute(&old, &new);

    assert!(
        matches!(delta.value, DeltaValue::Dense(_)),
        "delta should be dense when all elements changed"
    );

    let mut result = old.clone();
    delta.apply(&mut result).expect("apply dense delta");

    for (i, (&r, &expected)) in result.iter().zip(new.iter()).enumerate() {
        assert!(
            (r - expected).abs() < 1e-6,
            "element {i}: expected {expected}, got {r}"
        );
    }
}

// ===========================================================================
// 3. VectorDelta: identity delta for identical vectors
// ===========================================================================
#[test]
fn vector_delta_identity_when_equal() {
    let v = vec![1.0f32, 2.0, 3.0];
    let delta = VectorDelta::compute(&v, &v);

    assert!(
        delta.is_identity(),
        "delta between identical vectors should be identity"
    );
    assert_eq!(delta.l2_norm(), 0.0, "identity delta should have zero norm");
}

// ===========================================================================
// 4. VectorDelta: compose two deltas
// ===========================================================================
#[test]
fn vector_delta_compose() {
    let initial = vec![0.0f32; 4];
    let mid = vec![1.0f32, 0.0, 0.0, 0.0];
    let final_state = vec![1.0f32, 2.0, 0.0, 0.0];

    let delta1 = VectorDelta::compute(&initial, &mid);
    let delta2 = VectorDelta::compute(&mid, &final_state);
    let composed = delta1.compose(delta2);

    let mut result = initial.clone();
    composed
        .apply(&mut result)
        .expect("apply composed delta");

    for (i, (&r, &expected)) in result.iter().zip(final_state.iter()).enumerate() {
        assert!(
            (r - expected).abs() < 1e-6,
            "composed delta element {i}: expected {expected}, got {r}"
        );
    }
}

// ===========================================================================
// 5. VectorDelta: inverse cancels delta
// ===========================================================================
#[test]
fn vector_delta_inverse_cancels() {
    let old = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let new = vec![1.5f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.5];

    let delta = VectorDelta::compute(&old, &new);
    let inverse = delta.inverse();
    let composed = delta.compose(inverse);

    assert!(
        composed.is_identity(),
        "delta composed with its inverse should be identity"
    );
}

// ===========================================================================
// 6. VectorDelta: l2_norm and l1_norm
// ===========================================================================
#[test]
fn vector_delta_norms() {
    let delta = VectorDelta::from_dense(vec![3.0, 4.0, 0.0, 0.0]);

    let l2 = delta.l2_norm();
    assert!(
        (l2 - 5.0).abs() < 1e-6,
        "L2 norm of [3,4,0,0] should be 5.0, got {l2}"
    );

    let l1 = delta.l1_norm();
    assert!(
        (l1 - 7.0).abs() < 1e-6,
        "L1 norm of [3,4,0,0] should be 7.0, got {l1}"
    );
}

// ===========================================================================
// 7. VectorDelta: scale and clip
// ===========================================================================
#[test]
fn vector_delta_scale_and_clip() {
    let delta = VectorDelta::from_dense(vec![1.0, -2.0, 3.0]);

    let scaled = delta.scale(2.0);
    if let DeltaValue::Dense(values) = &scaled.value {
        assert!((values[0] - 2.0).abs() < 1e-6);
        assert!((values[1] - (-4.0)).abs() < 1e-6);
        assert!((values[2] - 6.0).abs() < 1e-6);
    } else {
        panic!("scaled delta should be dense");
    }

    let clipped = delta.clip(-1.5, 1.5);
    if let DeltaValue::Dense(values) = &clipped.value {
        assert!((values[0] - 1.0).abs() < 1e-6, "1.0 within range");
        assert!((values[1] - (-1.5)).abs() < 1e-6, "-2.0 clipped to -1.5");
        assert!((values[2] - 1.5).abs() < 1e-6, "3.0 clipped to 1.5");
    } else {
        panic!("clipped delta should be dense");
    }
}

// ===========================================================================
// 8. SparseDelta: compute and apply
// ===========================================================================
#[test]
fn sparse_delta_compute_and_apply() {
    let old = vec![1.0f32; 100];
    let mut new = old.clone();
    new[10] = 2.0;
    new[50] = 3.0;

    let delta = SparseDelta::compute(&old, &new);

    assert_eq!(delta.entries.len(), 2, "should have exactly 2 sparse entries");
    assert!(
        delta.sparsity() > 0.9,
        "sparsity should be > 0.9, got {}",
        delta.sparsity()
    );

    let mut result = old.clone();
    delta.apply(&mut result).expect("apply sparse delta");
    assert!(
        (result[10] - 2.0).abs() < 1e-6,
        "index 10 should be 2.0"
    );
    assert!(
        (result[50] - 3.0).abs() < 1e-6,
        "index 50 should be 3.0"
    );
}

// ===========================================================================
// 9. DeltaStream: push, replay, and checkpoint
// ===========================================================================
#[test]
fn delta_stream_push_and_replay() {
    let mut stream = DeltaStream::<VectorDelta>::new();
    let initial = vec![1.0f32, 2.0, 3.0];

    stream.push(VectorDelta::from_dense(vec![0.5, 0.0, 0.5]));
    stream.push(VectorDelta::from_dense(vec![0.0, 1.0, 0.0]));

    assert_eq!(stream.len(), 2, "stream should have 2 deltas");

    let result = stream.replay(initial.clone()).expect("replay");
    assert!((result[0] - 1.5).abs() < 1e-6, "element 0 after replay");
    assert!((result[1] - 3.0).abs() < 1e-6, "element 1 after replay");
    assert!((result[2] - 3.5).abs() < 1e-6, "element 2 after replay");
}

// ===========================================================================
// 10. DeltaStream: replay from checkpoint
// ===========================================================================
#[test]
fn delta_stream_checkpoint_and_replay() {
    let mut stream = DeltaStream::<VectorDelta>::new();
    let initial = vec![0.0f32; 3];

    stream.push(VectorDelta::from_dense(vec![1.0, 1.0, 1.0]));

    let checkpoint_state = stream.replay(initial.clone()).expect("first replay");
    stream.create_checkpoint(checkpoint_state);

    stream.push(VectorDelta::from_dense(vec![2.0, 2.0, 2.0]));

    let from_checkpoint = stream
        .replay_from_checkpoint(0)
        .expect("checkpoint should exist")
        .expect("replay from checkpoint");

    assert!(
        (from_checkpoint[0] - 3.0).abs() < 1e-6,
        "replay from checkpoint should accumulate both deltas, got {}",
        from_checkpoint[0]
    );
}

// ===========================================================================
// 11. DeltaStream: replay to specific sequence
// ===========================================================================
#[test]
fn delta_stream_replay_to_sequence() {
    let mut stream = DeltaStream::<VectorDelta>::new();
    let initial = vec![0.0f32; 3];

    stream.push(VectorDelta::from_dense(vec![1.0, 0.0, 0.0]));
    stream.push(VectorDelta::from_dense(vec![0.0, 1.0, 0.0]));
    stream.push(VectorDelta::from_dense(vec![0.0, 0.0, 1.0]));

    let at_seq_2 = stream.replay_to_sequence(initial, 2).expect("replay to seq 2");
    assert!((at_seq_2[0] - 1.0).abs() < 1e-6);
    assert!((at_seq_2[1] - 1.0).abs() < 1e-6);
    assert!((at_seq_2[2] - 0.0).abs() < 1e-6, "seq 3 delta should not be applied");
}

// ===========================================================================
// 12. Encoding roundtrip: dense, sparse, RLE, hybrid
// ===========================================================================
#[test]
fn encoding_roundtrip_dense() {
    let delta = VectorDelta::from_dense(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let encoding = DenseEncoding::new();

    let bytes = encoding.encode(&delta).expect("encode");
    let decoded = encoding.decode(&bytes).expect("decode");

    assert_eq!(delta.dimensions, decoded.dimensions, "dimensions should match");
}

#[test]
fn encoding_roundtrip_sparse() {
    let mut ops: SmallVec<[DeltaOp<f32>; 8]> = SmallVec::new();
    ops.push(DeltaOp::new(5, 1.5));
    ops.push(DeltaOp::new(50, -2.5));
    let delta = VectorDelta::from_sparse(ops, 100);

    let encoding = SparseEncoding::new();
    let bytes = encoding.encode(&delta).expect("encode sparse");
    let decoded = encoding.decode(&bytes).expect("decode sparse");

    assert_eq!(delta.dimensions, decoded.dimensions);
    assert_eq!(delta.value.nnz(), decoded.value.nnz());
}

#[test]
fn encoding_roundtrip_rle() {
    let values = vec![1.0f32, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let delta = VectorDelta::from_dense(values);

    let encoding = RunLengthEncoding::new();
    let bytes = encoding.encode(&delta).expect("encode RLE");
    let decoded = encoding.decode(&bytes).expect("decode RLE");

    assert_eq!(delta.dimensions, decoded.dimensions);
}

#[test]
fn encoding_roundtrip_hybrid() {
    let delta = VectorDelta::from_dense(vec![1.0, 2.0, 3.0, 4.0]);
    let encoding = HybridEncoding::new();

    let bytes = encoding.encode(&delta).expect("encode hybrid");
    let decoded = encoding.decode(&bytes).expect("decode hybrid");

    assert_eq!(delta.dimensions, decoded.dimensions);
}

// ===========================================================================
// 13. DeltaWindow: count-based emission
// ===========================================================================
#[test]
fn delta_window_count_based() {
    let mut window = DeltaWindow::<VectorDelta>::count_based(3);

    window.add(VectorDelta::from_dense(vec![1.0]), 0);
    window.add(VectorDelta::from_dense(vec![2.0]), 1);
    assert!(window.emit().is_none(), "should not emit with only 2 entries");

    window.add(VectorDelta::from_dense(vec![3.0]), 2);
    let result = window.emit().expect("should emit with 3 entries");
    assert_eq!(result.count, 3, "window result should cover 3 entries");
}

// ===========================================================================
// 14. DeltaStream: dimension mismatch error
// ===========================================================================
#[test]
fn delta_apply_dimension_mismatch() {
    let delta = VectorDelta::from_dense(vec![1.0, 2.0, 3.0]);
    let mut wrong_size = vec![0.0f32; 5]; // wrong dimension

    let result = delta.apply(&mut wrong_size);
    assert!(
        result.is_err(),
        "applying delta to wrong-dimension vector should fail"
    );
}

// ===========================================================================
// 15. VectorDelta: byte_size is non-zero for non-identity
// ===========================================================================
#[test]
fn vector_delta_byte_size() {
    let identity = VectorDelta::new(100);
    let size_identity = identity.byte_size();

    let dense = VectorDelta::from_dense(vec![1.0; 100]);
    let size_dense = dense.byte_size();

    assert!(
        size_dense > size_identity,
        "dense delta should have larger byte_size ({size_dense}) than identity ({size_identity})"
    );
}
