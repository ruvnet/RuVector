//! Correctness and validation coverage for `fit_warm_start` and
//! `ProductQuantizer::fit_warm_start`, independent of the drift benchmark in
//! `examples/warm_start_drift_benchmark.rs`.

use ruvector_e_means::{fit, fit_warm_start, Config, ProductQuantizer};

fn three_clouds(offset: f32) -> Vec<f32> {
    let mut data = Vec::new();
    for j in 0..3 {
        for i in 0..48 {
            for d in 0..8 {
                data.push(offset + j as f32 * 40.0 + ((i * 13 + d * 7) % 19) as f32 * 0.01);
            }
        }
    }
    data
}

fn default_three_cloud_config() -> Config {
    Config {
        clusters: 3,
        epochs: 5,
        batch_size: Some(12),
        ..Config::default()
    }
}

#[test]
fn warm_start_rejects_mismatched_or_nonfinite_centroids() {
    let data = three_clouds(0.0);
    let config = default_three_cloud_config();
    assert!(fit_warm_start(&data, 8, &config, &[0.0; 23]).is_err());
    assert!(fit_warm_start(&data, 8, &config, &[0.0; 25]).is_err());
    let mut bad = vec![0.0; 24];
    bad[5] = f32::NAN;
    assert!(fit_warm_start(&data, 8, &config, &bad).is_err());
    bad[5] = f32::INFINITY;
    assert!(fit_warm_start(&data, 8, &config, &bad).is_err());
}

#[test]
fn warm_start_is_deterministic_and_reproduces_its_own_seed() {
    let data = three_clouds(0.0);
    let config = default_three_cloud_config();
    let initial = fit(&data, 8, &config).unwrap().centroids;
    let a = fit_warm_start(&data, 8, &config, &initial).unwrap();
    let b = fit_warm_start(&data, 8, &config, &initial).unwrap();
    assert_eq!(a.centroids, b.centroids);
    assert_eq!(a.assignments, b.assignments);
    assert_eq!(a.mse, b.mse);
    assert_eq!(a.revived_clusters, b.revived_clusters);
}

/// Warm-starting from centroids already close to the true cluster means on
/// the *same* distribution should not need to move far, and should reach a
/// low-MSE solution that separates the three clouds, same as cold `fit`.
#[test]
fn warm_start_from_near_converged_centroids_stays_well_separated() {
    let data = three_clouds(0.0);
    let config = default_three_cloud_config();
    let cold = fit(&data, 8, &config).unwrap();
    assert!(cold.mse < 0.1, "cold mse {}", cold.mse);

    let warm = fit_warm_start(&data, 8, &config, &cold.centroids).unwrap();
    assert!(warm.mse < 0.1, "warm mse {}", warm.mse);
    let leaders: Vec<_> = [0, 48, 96]
        .into_iter()
        .map(|i| warm.assignments[i])
        .collect();
    assert_ne!(leaders[0], leaders[1]);
    assert_ne!(leaders[0], leaders[2]);
    assert_ne!(leaders[1], leaders[2]);
    for cloud in warm.assignments.chunks_exact(48) {
        assert!(cloud.iter().all(|j| *j == cloud[0]));
    }
}

/// A codebook trained on one window, warm-started against a *shifted* window
/// with a smaller epoch budget than a cold restart, should still separate
/// the (shifted) clouds rather than collapsing clusters or diverging.
#[test]
fn warm_start_adapts_to_a_shifted_window_with_fewer_epochs() {
    let old_window = three_clouds(0.0);
    let new_window = three_clouds(5.0); // every cloud center shifts by +5.
    let full = default_three_cloud_config();
    let reduced = Config {
        epochs: 1,
        ..full.clone()
    };

    let base = fit(&old_window, 8, &full).unwrap();
    let warm = fit_warm_start(&new_window, 8, &reduced, &base.centroids).unwrap();
    assert!(warm.mse.is_finite());
    assert!(warm.mse < 5.0, "warm mse after shift {}", warm.mse);
    let leaders: Vec<_> = [0, 48, 96]
        .into_iter()
        .map(|i| warm.assignments[i])
        .collect();
    assert_ne!(leaders[0], leaders[1]);
    assert_ne!(leaders[0], leaders[2]);
    assert_ne!(leaders[1], leaders[2]);
}

#[test]
fn pq_warm_start_requires_matching_shape_and_keeps_previous_cluster_count() {
    let data = [0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0];
    let previous = ProductQuantizer::fit(
        &data,
        4,
        2,
        &Config {
            clusters: 2,
            batch_size: Some(2),
            ..Config::default()
        },
    )
    .unwrap();

    // Wrong dimensions/subspaces are rejected.
    assert!(ProductQuantizer::fit_warm_start(&data, 2, 1, &Config::default(), &previous).is_err());
    assert!(ProductQuantizer::fit_warm_start(&data, 4, 1, &Config::default(), &previous).is_err());

    // config.clusters is ignored in favor of previous.clusters().
    let refreshed = ProductQuantizer::fit_warm_start(
        &data,
        4,
        2,
        &Config {
            clusters: 1,
            batch_size: Some(2),
            ..Config::default()
        },
        &previous,
    )
    .unwrap();
    assert_eq!(refreshed.clusters(), previous.clusters());
    assert_eq!(refreshed.dimensions(), previous.dimensions());
    assert_eq!(refreshed.subspaces(), previous.subspaces());
    assert!(refreshed.centroids().iter().all(|v| v.is_finite()));
}
