use ruvector_e_means::{Config, IvfPartition, Model};

fn model() -> Model {
    Model {
        centroids: vec![0.0, 10.0],
        dimensions: 1,
        assignments: vec![1, 0, 1, 0],
        mse: 0.0,
        batch_size: 4,
        peak_step: 0.2,
        updates: 1,
        revived_clusters: 0,
    }
}

#[test]
fn stable_postings_group_every_input_row_exactly_once() {
    let partition = IvfPartition::from_model(model()).unwrap();
    assert_eq!(partition.offsets(), [0, 2, 4]);
    assert_eq!(partition.indices(), [1, 3, 0, 2]);
    assert_eq!(partition.posting(0).unwrap(), [1, 3]);
    assert_eq!(partition.posting(1).unwrap(), [0, 2]);
    assert!(partition.posting(2).is_err());
    let restored = IvfPartition::from_parts(
        model(),
        partition.offsets().to_vec(),
        partition.indices().to_vec(),
    )
    .unwrap();
    assert_eq!(restored.indices(), partition.indices());
}

#[test]
fn fit_partitions_final_predictions_in_original_row_order() {
    let data = [10.0, -10.0, 11.0, -11.0, 12.0, -12.0];
    let partition = IvfPartition::fit(
        &data,
        1,
        &Config {
            clusters: 2,
            epochs: 5,
            ..Config::default()
        },
    )
    .unwrap();
    let model = partition.model();
    assert_eq!(model.predict(&data).unwrap(), model.assignments);
    let mut ids = partition.indices().to_vec();
    ids.sort_unstable();
    assert_eq!(ids, (0..data.len()).collect::<Vec<_>>());
    for cluster in 0..2 {
        let posting = partition.posting(cluster).unwrap();
        assert!(posting.windows(2).all(|p| p[0] < p[1]));
        assert!(posting.iter().all(|&id| model.assignments[id] == cluster));
    }
}

#[test]
fn malformed_offsets_labels_and_row_permutations_reject() {
    for (offsets, indices) in [
        (vec![1, 2, 4], vec![1, 3, 0, 2]),
        (vec![0, 3, 4], vec![1, 3, 0, 2]),
        (vec![0, 5, 4], vec![1, 3, 0, 2]),
        (vec![0, 2], vec![1, 3, 0, 2]),
        (vec![0, 2, 4], vec![1, 1, 0, 2]),
        (vec![0, 2, 4], vec![3, 1, 0, 2]),
        (vec![0, 2, 4], vec![1, 3, 0, 9]),
        (vec![0, 2, 4], vec![1, 3, 0]),
    ] {
        assert!(IvfPartition::from_parts(model(), offsets, indices).is_err());
    }
    let mut invalid = model();
    invalid.assignments[0] = 2;
    assert!(IvfPartition::from_model(invalid).is_err());
    let mut invalid = model();
    invalid.centroids[0] = f32::INFINITY;
    assert!(IvfPartition::from_model(invalid).is_err());
    let mut invalid = model();
    invalid.dimensions = 0;
    assert!(IvfPartition::from_model(invalid).is_err());
    let mut invalid = model();
    invalid.batch_size = 0;
    assert!(IvfPartition::from_model(invalid).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn serde_roundtrip_rejects_corrupted_postings_and_models() {
    let partition = IvfPartition::from_model(model()).unwrap();
    let value = serde_json::to_value(&partition).unwrap();
    let restored: IvfPartition = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(restored.offsets(), partition.offsets());
    assert_eq!(restored.indices(), partition.indices());
    let mut invalid = value.clone();
    invalid["indices"] = serde_json::json!([1, 1, 0, 2]);
    assert!(serde_json::from_value::<IvfPartition>(invalid).is_err());
    let mut invalid = value.clone();
    invalid["model"]["assignments"] = serde_json::json!([2, 0, 1, 0]);
    assert!(serde_json::from_value::<IvfPartition>(invalid).is_err());
    let mut invalid = value;
    invalid["offsets"] = serde_json::json!([0, 4, 2]);
    assert!(serde_json::from_value::<IvfPartition>(invalid).is_err());
}
