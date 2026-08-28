use ruvector_apple_core::{
    DistanceMetric, ExactVectorIndex, IndexConfig, IndexError, MAX_SEARCH_RESULTS, MAX_TOTAL_VALUES,
};

fn index(metric: DistanceMetric, dimensions: u32, capacity: u32) -> ExactVectorIndex {
    ExactVectorIndex::new(IndexConfig {
        dimensions,
        capacity,
        metric,
    })
    .unwrap()
}

#[test]
fn validates_all_resource_bounds() {
    assert_eq!(
        ExactVectorIndex::new(IndexConfig {
            dimensions: 0,
            capacity: 1,
            metric: DistanceMetric::Dot,
        })
        .unwrap_err(),
        IndexError::InvalidDimensions(0)
    );
    assert_eq!(
        ExactVectorIndex::new(IndexConfig {
            dimensions: 1,
            capacity: 0,
            metric: DistanceMetric::Dot,
        })
        .unwrap_err(),
        IndexError::InvalidCapacity(0)
    );
    assert!(matches!(
        ExactVectorIndex::new(IndexConfig {
            dimensions: 65_536,
            capacity: 262_144,
            metric: DistanceMetric::Dot,
        }),
        Err(IndexError::MemoryBudgetExceeded {
            maximum_values: MAX_TOTAL_VALUES,
            ..
        })
    ));
}

#[test]
fn upsert_validates_before_mutation_and_replaces_at_capacity() {
    let mut index = index(DistanceMetric::Cosine, 2, 1);
    index.upsert(10, &[1.0, 0.0]).unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(
        index.upsert(11, &[0.0, 1.0]),
        Err(IndexError::CapacityExceeded(1))
    );
    assert_eq!(
        index.upsert(10, &[f32::NAN, 1.0]),
        Err(IndexError::NonFiniteValue)
    );
    assert_eq!(index.get(10), Some([1.0, 0.0].as_slice()));
    index.upsert(10, &[0.0, 1.0]).unwrap();
    assert_eq!(index.get(10), Some([0.0, 1.0].as_slice()));
    assert_eq!(
        index.upsert(10, &[0.0, 0.0]),
        Err(IndexError::ZeroNormVector)
    );
    assert_eq!(
        index.upsert(10, &[1.0]),
        Err(IndexError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn cosine_search_is_exact_bounded_and_deterministic() {
    let mut index = index(DistanceMetric::Cosine, 2, 4);
    index.upsert(20, &[1.0, 0.0]).unwrap();
    index.upsert(10, &[1.0, 0.0]).unwrap();
    index.upsert(30, &[0.0, 1.0]).unwrap();
    index.upsert(40, &[-1.0, 0.0]).unwrap();

    let hits = index.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        [10, 20, 30]
    );
    assert_eq!(hits[0].score, 1.0);
    assert_eq!(hits[2].score, 0.0);
    assert!(index.search(&[1.0, 0.0], 0).unwrap().is_empty());
    assert_eq!(
        index.search(&[1.0, 0.0], MAX_SEARCH_RESULTS + 1),
        Err(IndexError::SearchLimitExceeded(MAX_SEARCH_RESULTS + 1))
    );
}

#[test]
fn l2_and_dot_have_uniform_higher_is_better_ordering() {
    let mut l2 = index(DistanceMetric::L2, 2, 3);
    let mut dot = index(DistanceMetric::Dot, 2, 3);
    for target in [&mut l2, &mut dot] {
        target.upsert(1, &[1.0, 0.0]).unwrap();
        target.upsert(2, &[2.0, 0.0]).unwrap();
        target.upsert(3, &[-1.0, 0.0]).unwrap();
    }

    let l2_hits = l2.search(&[1.1, 0.0], 3).unwrap();
    assert_eq!(
        l2_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert!((l2_hits[0].score + 0.01).abs() < 1.0e-6);
    let dot_hits = dot.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(
        dot_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        [2, 1, 3]
    );
}

#[test]
fn remove_repairs_swap_position_map() {
    let mut index = index(DistanceMetric::Dot, 1, 4);
    for id in 1..=4 {
        index.upsert(id, &[id as f32]).unwrap();
    }
    assert!(index.remove(2));
    assert!(!index.remove(2));
    assert_eq!(index.get(4), Some([4.0].as_slice()));
    index.upsert(4, &[8.0]).unwrap();
    assert_eq!(index.get(4), Some([8.0].as_slice()));
    assert_eq!(index.len(), 3);
}

#[test]
fn concurrent_readers_observe_stable_results() {
    let mut value = index(DistanceMetric::Dot, 4, 64);
    for id in 0..64 {
        value.upsert(id, &[id as f32, 1.0, 2.0, 3.0]).unwrap();
    }
    let shared = std::sync::Arc::new(value);
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let shared = shared.clone();
            std::thread::spawn(move || {
                for _ in 0..100 {
                    assert_eq!(shared.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap()[0].id, 63);
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}
