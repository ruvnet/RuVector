use ruvector_apple_core::{DistanceMetric, ExactVectorIndex, IndexConfig, IndexError};

fn populated() -> ExactVectorIndex {
    let mut index = ExactVectorIndex::new(IndexConfig {
        dimensions: 3,
        capacity: 8,
        metric: DistanceMetric::Cosine,
    })
    .unwrap();
    index.upsert(7, &[1.0, 2.0, 3.0]).unwrap();
    index.upsert(2, &[3.0, 2.0, 1.0]).unwrap();
    index
}

#[test]
fn snapshot_is_deterministic_across_insertion_order() {
    let first = populated();
    let mut second = ExactVectorIndex::new(first.config()).unwrap();
    second.upsert(2, &[3.0, 2.0, 1.0]).unwrap();
    second.upsert(7, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(first.snapshot().unwrap(), second.snapshot().unwrap());
}

#[test]
fn round_trip_preserves_configuration_vectors_and_search() {
    let original = populated();
    let snapshot = original.snapshot().unwrap();
    let restored = ExactVectorIndex::from_snapshot(&snapshot).unwrap();
    assert_eq!(restored.config(), original.config());
    assert_eq!(restored.len(), original.len());
    assert_eq!(restored.get(2), original.get(2));
    assert_eq!(
        restored.search(&[1.0, 2.0, 3.0], 2).unwrap(),
        original.search(&[1.0, 2.0, 3.0], 2).unwrap()
    );
}

#[test]
fn every_single_byte_corruption_is_detected() {
    let snapshot = populated().snapshot().unwrap();
    for offset in 0..snapshot.len() {
        let mut corrupted = snapshot.clone();
        corrupted[offset] ^= 0x5a;
        assert!(
            ExactVectorIndex::from_snapshot(&corrupted).is_err(),
            "offset {offset} was accepted"
        );
    }
}

#[test]
fn all_truncations_and_trailing_bytes_are_rejected() {
    let snapshot = populated().snapshot().unwrap();
    for end in 0..snapshot.len() {
        assert!(ExactVectorIndex::from_snapshot(&snapshot[..end]).is_err());
    }
    let mut trailing = snapshot;
    trailing.push(0);
    assert!(matches!(
        ExactVectorIndex::from_snapshot(&trailing),
        Err(IndexError::CorruptSnapshot(_))
    ));
}

#[test]
fn arbitrary_small_inputs_never_panic() {
    for length in 0..256 {
        let bytes: Vec<_> = (0..length)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(length as u8))
            .collect();
        let result = std::panic::catch_unwind(|| ExactVectorIndex::from_snapshot(&bytes));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }
}
