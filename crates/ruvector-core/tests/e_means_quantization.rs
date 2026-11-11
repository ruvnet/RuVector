//! Integration evidence for the shared e-means PQ training path.

use ruvector_core::error::RuvectorError;
use ruvector_core::ProductQuantized;
use ruvector_e_means::Config;

fn config(clusters: usize) -> Config {
    Config {
        clusters,
        epochs: 3,
        seed: 17,
        ..Config::default()
    }
}

#[test]
fn subspace_codebooks_recover_column_means_and_keep_the_existing_format() {
    let vectors = vec![
        vec![1.0, 2.0, 10.0, 20.0],
        vec![3.0, 4.0, 30.0, 40.0],
        vec![5.0, 6.0, 50.0, 60.0],
    ];
    let trained = ProductQuantized::train_e_means(&vectors, 2, &config(1)).unwrap();
    assert!(trained.codes.is_empty());
    assert_eq!(
        trained.codebooks,
        vec![vec![vec![3.0, 4.0]], vec![vec![30.0, 40.0]]]
    );
    assert_eq!(trained.encode(&vectors[0]), vec![0, 0]);

    // Existing persisted PQ consumers use the same serde representation and
    // reconstruct a vector by looking up each subspace's encoded centroid.
    let json = serde_json::to_vec(&trained).unwrap();
    let restored: ProductQuantized = serde_json::from_slice(&json).unwrap();
    let codes = restored.encode(&vectors[2]);
    let reconstructed: Vec<f32> = codes
        .iter()
        .enumerate()
        .flat_map(|(subspace, &code)| restored.codebooks[subspace][code as usize].clone())
        .collect();
    assert_eq!(reconstructed, vec![3.0, 4.0, 30.0, 40.0]);
}

#[test]
fn identical_seed_and_options_produce_identical_codebooks() {
    let vectors: Vec<Vec<f32>> = (0..80)
        .map(|i| vec![i as f32 / 10.0, (i % 7) as f32, -(i as f32), 2.0])
        .collect();
    let mut options = config(4);
    options.batch_size = Some(7); // Exercise the final short batch.
    let first = ProductQuantized::train_e_means(&vectors, 2, &options).unwrap();
    let second = ProductQuantized::train_e_means(&vectors, 2, &options).unwrap();
    assert_eq!(first.codebooks, second.codebooks);
    assert_eq!(first.codebooks.len(), 2);
    assert!(first.codebooks.iter().all(|book| book.len() == 4));
    assert!(first
        .codebooks
        .iter()
        .flatten()
        .all(|centroid| centroid.len() == 2 && centroid.iter().all(|v| v.is_finite())));
}

#[test]
fn maximum_u8_codebook_size_is_usable() {
    let vectors: Vec<Vec<f32>> = (0..256).map(|i| vec![i as f32]).collect();
    let mut options = config(256);
    options.epochs = 1;
    options.batch_size = Some(256);
    let trained = ProductQuantized::train_e_means(&vectors, 1, &options).unwrap();
    assert_eq!(trained.codebooks[0].len(), 256);
    let codes: std::collections::BTreeSet<u8> =
        vectors.iter().map(|v| trained.encode(v)[0]).collect();
    assert_eq!(codes.len(), 256);
}

#[test]
fn large_finite_inputs_keep_the_nearest_centroid_when_squared_f32_distances_overflow() {
    let magnitude = f32::MAX / 4.0;
    let vectors = vec![vec![-magnitude], vec![magnitude]];
    let mut options = config(2);
    options.epochs = 1;
    options.batch_size = Some(2);
    let trained = ProductQuantized::train_e_means(&vectors, 1, &options).unwrap();
    // Pick a query past the second centroid. Both candidate squared distances
    // overflow f32, but the second remains strictly nearer in f64.
    let query = trained.codebooks[0][1][0] * 2.0;
    assert!(query.is_finite());
    for centroid in &trained.codebooks[0] {
        assert!((query - centroid[0]).powi(2).is_infinite());
    }
    assert_eq!(trained.encode(&[query]), vec![1]);
}

#[test]
fn malformed_training_inputs_return_errors() {
    let valid = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0, 5.0]];
    for (vectors, subspaces) in [
        (Vec::new(), 1),
        (vec![Vec::new()], 1),
        (valid.clone(), 0),
        (valid.clone(), 3),
        (valid.clone(), 8),
        (vec![valid[0].clone(), vec![1.0]], 2),
        (vec![vec![f32::NAN, 0.0]], 1),
        (vec![vec![f32::INFINITY, 0.0]], 1),
    ] {
        assert!(ProductQuantized::train_e_means(&vectors, subspaces, &config(1)).is_err());
    }
    let error =
        ProductQuantized::train_e_means(&[vec![1.0, 2.0], vec![1.0]], 1, &config(1)).unwrap_err();
    assert!(matches!(error, RuvectorError::DimensionMismatch { .. }));
}

#[test]
fn invalid_optimizer_settings_return_errors() {
    let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    for clusters in [0, 3, 257] {
        assert!(ProductQuantized::train_e_means(&vectors, 1, &config(clusters)).is_err());
    }
    let mut options = config(1);
    options.epochs = 0;
    assert!(ProductQuantized::train_e_means(&vectors, 1, &options).is_err());
    options = config(1);
    options.batch_size = Some(0);
    assert!(ProductQuantized::train_e_means(&vectors, 1, &options).is_err());
    options = config(1);
    options.peak_step = Some(f64::NAN);
    assert!(ProductQuantized::train_e_means(&vectors, 1, &options).is_err());
}
