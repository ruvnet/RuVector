use ruvector_e_means::{Config, ProductQuantizer};

#[test]
fn subspace_layout_codes_and_reconstruction_have_an_independent_oracle() {
    let pq = ProductQuantizer::from_centroids(
        4,
        2,
        2,
        vec![0.0, 0.0, 10.0, 10.0, 100.0, 100.0, 200.0, 200.0],
    )
    .unwrap();
    let data = [0.0, 0.0, 200.0, 200.0, 10.0, 10.0, 100.0, 100.0];
    assert_eq!(pq.encode(&data).unwrap(), [0, 1, 1, 0]);
    assert_eq!(pq.decode(&[0, 1, 1, 0]).unwrap(), data);
    assert_eq!(pq.encode(&[5.0, 5.0, 150.0, 150.0]).unwrap(), [0, 0]);
    assert!(pq.encode(&[]).unwrap().is_empty());
    assert!(pq.decode(&[]).unwrap().is_empty());
}

#[test]
fn fitted_single_codebooks_recover_column_means() {
    let pq = ProductQuantizer::fit(
        &[1.0, 10.0, 3.0, 30.0, 5.0, 50.0],
        2,
        2,
        &Config {
            clusters: 1,
            batch_size: Some(2),
            ..Config::default()
        },
    )
    .unwrap();
    assert_eq!(
        (
            pq.dimensions(),
            pq.subspaces(),
            pq.clusters(),
            pq.subspace_dimensions()
        ),
        (2, 2, 1, 1)
    );
    assert_eq!(pq.centroids(), [3.0, 30.0]);
    assert_eq!(
        pq.decode(&pq.encode(&[2.0, 20.0, 9.0, 90.0]).unwrap())
            .unwrap(),
        [3.0, 30.0, 3.0, 30.0]
    );
}

#[test]
fn checked_pq_boundaries_reject_malformed_and_nonfinite_data() {
    for (dimensions, subspaces, clusters, centroids) in [
        (0, 1, 1, vec![]),
        (2, 0, 1, vec![0.0; 2]),
        (3, 2, 1, vec![0.0; 3]),
        (2, 1, 0, vec![]),
        (1, 1, 257, vec![0.0; 257]),
        (2, 1, 1, vec![0.0]),
        (1, 1, 1, vec![f32::NAN]),
        (usize::MAX, 1, 256, vec![]),
    ] {
        assert!(
            ProductQuantizer::from_centroids(dimensions, subspaces, clusters, centroids).is_err()
        );
    }
    let pq = ProductQuantizer::from_centroids(2, 2, 1, vec![0.0; 2]).unwrap();
    assert!(pq.encode(&[1.0]).is_err());
    assert!(pq.encode(&[f32::INFINITY, 0.0]).is_err());
    assert!(pq.decode(&[0]).is_err());
    assert!(pq.decode(&[0, 1]).is_err());
    assert!(ProductQuantizer::fit(&[], 2, 2, &Config::default()).is_err());
    assert!(ProductQuantizer::fit(
        &[0.0; 8],
        2,
        2,
        &Config {
            clusters: 5,
            ..Config::default()
        }
    )
    .is_err());
}

#[test]
fn all_u8_codes_and_large_finite_distances_remain_valid() {
    let pq =
        ProductQuantizer::from_centroids(1, 1, 256, (0..256).map(|v| v as f32).collect()).unwrap();
    assert_eq!(pq.encode(&[255.0]).unwrap(), [255]);
    assert_eq!(pq.decode(&[255]).unwrap(), [255.0]);
    let magnitude = f32::MAX / 4.0;
    let pq = ProductQuantizer::from_centroids(1, 1, 2, vec![-magnitude, magnitude]).unwrap();
    assert_eq!(pq.encode(&[magnitude * 2.0]).unwrap(), [1]);
}

#[cfg(feature = "serde")]
#[test]
fn serde_roundtrip_validates_the_codebook_before_it_becomes_usable() {
    let pq = ProductQuantizer::from_centroids(2, 2, 2, vec![0.0, 1.0, 10.0, 20.0]).unwrap();
    let value = serde_json::to_value(&pq).unwrap();
    let restored: ProductQuantizer = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(restored.encode(&[1.0, 20.0]).unwrap(), [1, 1]);
    for field in ["dimensions", "subspaces", "clusters"] {
        let mut invalid = value.clone();
        invalid[field] = 0.into();
        assert!(serde_json::from_value::<ProductQuantizer>(invalid).is_err());
    }
    let mut invalid = value.clone();
    invalid["centroids"] = serde_json::json!([0.0]);
    assert!(serde_json::from_value::<ProductQuantizer>(invalid).is_err());
    let mut invalid = value;
    invalid["extra"] = true.into();
    assert!(serde_json::from_value::<ProductQuantizer>(invalid).is_err());
}
