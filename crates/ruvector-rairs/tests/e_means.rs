use ruvector_e_means::Config;
use ruvector_rairs::{AnnIndex, IvfFlat, RairsSeil, RairsStrict};

fn corpus() -> Vec<Vec<f32>> {
    (0..40)
        .map(|i| vec![i as f32, i as f32 * 0.1, (i % 3) as f32, -(i as f32)])
        .collect()
}

fn options() -> Config {
    Config {
        clusters: 4,
        epochs: 3,
        batch_size: Some(7),
        seed: 17,
        ..Config::default()
    }
}

fn assert_full_probe_ground_truth(index: &dyn AnnIndex, vectors: &[Vec<f32>]) {
    let query = [17.37, 1.65, 2.13, -17.91];
    let mut truth: Vec<(usize, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(id, row)| {
            (
                id,
                row.iter()
                    .zip(query)
                    .map(|(&x, y)| (f64::from(x) - f64::from(y)).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            )
        })
        .collect();
    truth.sort_by(|a, b| a.1.total_cmp(&b.1));
    let results = index.search(&query, 5, index.num_lists()).unwrap();
    assert_eq!(results.len(), 5);
    for (actual, expected) in results.iter().zip(truth) {
        assert_eq!(actual.id, expected.0);
        assert!((f64::from(actual.distance) - expected.1).abs() < 0.001);
    }
}

#[test]
fn all_rairs_layouts_use_e_means_and_preserve_search_and_training_atomicity() {
    let vectors = corpus();
    let config = options();
    macro_rules! check {
        ($index:expr) => {{
            let mut index = $index;
            index.train_e_means(&vectors, &config).unwrap();
            index.add(&vectors).unwrap();
            assert_eq!(index.len(), vectors.len());
            assert_full_probe_ground_truth(&index, &vectors);
            let before = index.search(&vectors[17], 5, 4).unwrap();
            let mut invalid = vectors.clone();
            invalid[3].pop();
            assert!(index.train_e_means(&invalid, &config).is_err());
            invalid = vectors.clone();
            invalid[3][0] = f32::NAN;
            assert!(index.train_e_means(&invalid, &config).is_err());
            assert!(index.train_e_means(&[], &config).is_err());
            let mut bad_options = config.clone();
            bad_options.clusters = 5;
            assert!(index.train_e_means(&vectors, &bad_options).is_err());
            bad_options = config.clone();
            bad_options.epochs = 0;
            assert!(index.train_e_means(&vectors, &bad_options).is_err());
            assert_eq!(index.len(), vectors.len());
            assert_eq!(index.search(&vectors[17], 5, 4).unwrap(), before);
            index.train_e_means(&vectors, &config).unwrap();
            assert!(index.is_empty());
            index.add(&vectors[10..12]).unwrap();
            assert_eq!(index.len(), 2);
            assert_eq!(index.search(&vectors[10], 1, 4).unwrap()[0].id, 0);
        }};
    }
    check!(IvfFlat::new(4, 4, 25, 42));
    check!(RairsStrict::new(4, 4, 25, 42, 1.0));
    check!(RairsSeil::new(4, 4, 25, 42, 1.0));
}

#[test]
fn invalid_constructor_settings_reject_without_a_training_panic() {
    let vectors = corpus();
    assert!(IvfFlat::new(0, 4, 25, 42)
        .train_e_means(&vectors, &options())
        .is_err());
    assert!(IvfFlat::new(4, 0, 25, 42)
        .train_e_means(
            &vectors,
            &Config {
                clusters: 0,
                ..options()
            }
        )
        .is_err());
    assert!(IvfFlat::new(4, 41, 25, 42)
        .train_e_means(
            &vectors,
            &Config {
                clusters: 41,
                ..options()
            }
        )
        .is_err());
    assert!(RairsStrict::new(4, 4, 25, 42, f32::NAN)
        .train_e_means(&vectors, &options())
        .is_err());
    assert!(RairsSeil::new(4, 4, 25, 42, f32::INFINITY)
        .train_e_means(&vectors, &options())
        .is_err());
}
