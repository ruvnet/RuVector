use ruvector_e_means::{cosine_step, fit, predict, Config, Finalization};

#[test]
fn ema_uses_decayed_sums_and_counts_including_short_batches() {
    let model = fit(
        &[0.0, 1.0, 9.0],
        1,
        &Config {
            clusters: 1,
            epochs: 1,
            batch_size: Some(2),
            peak_step: Some(1.0),
            finalization: Finalization::Ema,
            ..Config::default()
        },
    )
    .unwrap();
    // First step=1, second step=.55. For any shuffled tail x, the
    // correct ratio is (.45*(10-x)+.55*x)/(.45*2+.55*1).
    // Averaging batch centroids directly gives a different answer.
    let possible: Vec<f64> = [0.0, 1.0, 9.0]
        .into_iter()
        .map(|tail| (0.45 * (10.0 - tail) + 0.55 * tail) / 1.45)
        .collect();
    assert!(possible
        .iter()
        .any(|x| (*x - f64::from(model.centroids[0])).abs() < 1e-6));
    assert_eq!(model.updates, 2);
    let center = f64::from(model.centroids[0]);
    let expected_mse = (center.powi(2) + (1.0 - center).powi(2) + (9.0 - center).powi(2)) / 3.0;
    assert!((model.mse - expected_mse).abs() < 1e-12);
}

#[test]
fn last_epoch_mean_matches_independent_mean_and_mse_with_tail_batch() {
    let data = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let config = Config {
        clusters: 1,
        epochs: 2,
        batch_size: Some(2),
        ..Config::default()
    };
    let model = fit(&data, 2, &config).unwrap();
    assert_eq!(model.centroids, [5., 6.]);
    assert_eq!(model.assignments, [0; 5]);
    assert_eq!(model.mse, 16.0);
    assert_eq!(model.updates, 6); // ceil(5/2) * 2, no dropped rows.
    assert_eq!(model.predict(&data).unwrap(), model.assignments);
}

#[test]
fn scaling_laws_match_table_two_and_large_dataset_rule() {
    let params = Config {
        clusters: 1024,
        epochs: 4,
        ..Config::default()
    }
    .parameters(1_000_000)
    .unwrap();
    assert_eq!(params.batch_size, 65536);
    assert_eq!(params.peak_step, 1.0);
    let params = Config {
        clusters: 1024,
        epochs: 4,
        batch_size: Some(256),
        ..Config::default()
    }
    .parameters(1_000_000)
    .unwrap();
    assert!((params.peak_step - 0.008558124524629177).abs() < 1e-14);
    let params = Config {
        clusters: 1000,
        ..Config::default()
    }
    .parameters(3_000_000)
    .unwrap();
    assert_eq!(params.batch_size, 16384);
    assert!((params.peak_step - 0.2299943961117698).abs() < 1e-14);
}

#[test]
fn schedule_matches_equation_five() {
    assert_eq!(cosine_step(0.5, 0, 20), 0.5);
    assert!((cosine_step(0.5, 10, 20) - 0.275).abs() < 1e-15);
    assert_eq!(cosine_step(0.5, 20, 20), 0.05);
    let steps: Vec<f64> = (0..20).map(|i| cosine_step(0.5, i, 20)).collect();
    assert!(steps.windows(2).all(|s| s[1] < s[0]));
    assert!(steps[19] > 0.05);
}

#[test]
fn separates_three_clouds_and_reproduces_seed() {
    let mut data = Vec::new();
    for j in 0..3 {
        for i in 0..48 {
            for d in 0..8 {
                data.push(j as f32 * 40.0 + ((i * 13 + d * 7) % 19) as f32 * 0.01);
            }
        }
    }
    let config = Config {
        clusters: 3,
        epochs: 5,
        batch_size: Some(12),
        ..Config::default()
    };
    let a = fit(&data, 8, &config).unwrap();
    let b = fit(&data, 8, &config).unwrap();
    assert_eq!(a.centroids, b.centroids);
    assert_eq!(a.assignments, b.assignments);
    assert!(a.mse < 0.1, "mse {}", a.mse);
    let leaders: Vec<_> = [0, 48, 96].into_iter().map(|i| a.assignments[i]).collect();
    assert_ne!(leaders[0], leaders[1]);
    assert_ne!(leaders[0], leaders[2]);
    assert_ne!(leaders[1], leaders[2]);
    for cloud in a.assignments.chunks_exact(48) {
        assert!(cloud.iter().all(|j| *j == cloud[0]));
    }
}

#[test]
fn duplicate_and_zero_vectors_remain_finite_with_revival_and_step_one() {
    for value in [0., 12., f32::MAX, -f32::MAX] {
        for finalization in [Finalization::LastEpochMean, Finalization::Ema] {
            let model = fit(
                &[value; 96],
                3,
                &Config {
                    clusters: 8,
                    epochs: 4,
                    batch_size: Some(2),
                    peak_step: Some(1.0),
                    finalization,
                    ..Config::default()
                },
            )
            .unwrap();
            assert!(model.centroids.iter().all(|x| x.is_finite()));
            assert!(model.mse.is_finite());
            assert!(model.revived_clusters > 0);
            assert!(model.assignments.iter().all(|j| *j < 8));
        }
    }
}

#[test]
fn handles_full_batch_k_equals_n_and_large_coordinates() {
    let data = [f32::MAX, -f32::MAX, -f32::MAX, f32::MAX];
    let model = fit(
        &data,
        2,
        &Config {
            clusters: 2,
            batch_size: Some(100),
            ..Config::default()
        },
    )
    .unwrap();
    assert_eq!(model.batch_size, 2);
    assert_eq!(model.mse, 0.0);
    assert_ne!(model.assignments[0], model.assignments[1]);
}

#[test]
fn validates_before_training_and_checks_mutated_models_on_predict() {
    let good = Config {
        clusters: 1,
        ..Config::default()
    };
    for (data, dim) in [
        (vec![], 2),
        (vec![1.], 0),
        (vec![1.], 2),
        (vec![f32::NAN], 1),
        (vec![f32::INFINITY], 1),
    ] {
        assert!(fit(&data, dim, &good).is_err());
    }
    for config in [
        Config {
            clusters: 0,
            ..good.clone()
        },
        Config {
            clusters: 3,
            ..good.clone()
        },
        Config {
            epochs: 0,
            ..good.clone()
        },
        Config {
            batch_size: Some(0),
            ..good.clone()
        },
        Config {
            peak_step: Some(0.),
            ..good.clone()
        },
        Config {
            peak_step: Some(-0.1),
            ..good.clone()
        },
        Config {
            peak_step: Some(1.1),
            ..good.clone()
        },
        Config {
            peak_step: Some(f64::NAN),
            ..good.clone()
        },
        Config {
            epochs: usize::MAX,
            batch_size: Some(1),
            ..good.clone()
        },
    ] {
        assert!(fit(&[1., 2.], 1, &config).is_err());
    }
    let mut model = fit(&[1., 2.], 1, &good).unwrap();
    assert!(model.predict(&[]).unwrap().is_empty());
    model.dimensions = 0;
    assert!(model.predict(&[1.]).is_err());
    assert!(predict(&[1., 2.], 2, &[1.]).is_err());
    assert!(predict(&[1.], 1, &[]).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn serializable_public_api_is_available() {
    fn serializable<T: serde::Serialize + for<'de> serde::Deserialize<'de>>() {}
    serializable::<Config>();
    serializable::<ruvector_e_means::Model>();
}
