use ruvector_e_means::Config;
use ruvector_pq_search::{decode_vector, encode_vector, IvfPqIndex, PqCodebook};

fn data() -> Vec<f32> {
    (0..8).flat_map(|i| [i as f32; 4]).collect()
}

fn options(clusters: usize) -> Config {
    Config {
        clusters,
        epochs: 1,
        batch_size: Some(8),
        seed: 7,
        ..Config::default()
    }
}

#[test]
fn e_means_codebooks_roundtrip_existing_codes_and_adc_search() {
    let data = data();
    let codebook = PqCodebook::train_e_means(2, &data, 4, &options(8)).unwrap();
    assert_eq!(codebook.centroids.len(), 8 * 4);
    for row in data.as_chunks::<4>().0 {
        let codes = encode_vector(&codebook, row);
        assert_eq!(codes.len(), 2);
        assert_eq!(decode_vector(&codebook, &codes), row.as_slice());
        let table = codebook.build_adc_table(row);
        assert_eq!(codebook.adc_distance(&table, &codes), 0.0);
    }
    let mut index = IvfPqIndex::new_e_means(codebook, 2, &data, &options(2)).unwrap();
    assert!(index.try_insert(&[1.0]).is_err());
    assert!(index.try_insert(&[f32::NAN; 4]).is_err());
    for row in data.as_chunks::<4>().0 {
        index.try_insert(row).unwrap();
    }
    let results = index.try_search(&[3.0; 4], 3).unwrap();
    assert_eq!(results[0].id, 3);
    assert_eq!(
        results.iter().map(|r| r.distance).collect::<Vec<_>>(),
        [0.0, 4.0, 4.0]
    );
    let mut ids: Vec<usize> = results.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    assert_eq!(ids, [2, 3, 4]);
    assert!(index.try_search(&[0.0], 1).is_err());
    assert!(index.try_search(&[f32::INFINITY; 4], 1).is_err());
    assert!(index.try_search(&[0.0; 4], 0).unwrap().is_empty());
}

#[test]
fn checked_constructors_reject_invalid_training_and_public_codebook_fields() {
    let data = data();
    assert!(PqCodebook::train_e_means(0, &data, 4, &options(2)).is_err());
    assert!(PqCodebook::train_e_means(3, &data, 4, &options(2)).is_err());
    assert!(PqCodebook::train_e_means(2, &[0.0; 3], 4, &options(2)).is_err());
    assert!(PqCodebook::train_e_means(2, &[f32::NAN; 32], 4, &options(2)).is_err());
    assert!(PqCodebook::train_e_means(2, &data, 4, &options(9)).is_err());
    let codebook = PqCodebook::train_e_means(2, &data, 4, &options(8)).unwrap();
    for probes in [0, 3] {
        assert!(IvfPqIndex::new_e_means(codebook.clone(), probes, &data, &options(2)).is_err());
    }
    let mut corrupt = codebook.clone();
    corrupt.sub_dim = 0;
    assert!(IvfPqIndex::new_e_means(corrupt, 1, &data, &options(2)).is_err());
    let mut corrupt = codebook.clone();
    corrupt.config.m = 0;
    assert!(IvfPqIndex::new_e_means(corrupt, 1, &data, &options(2)).is_err());
    let mut corrupt = codebook.clone();
    corrupt.centroids[0] = f32::NAN;
    assert!(IvfPqIndex::new_e_means(corrupt, 1, &data, &options(2)).is_err());
    assert!(IvfPqIndex::new_e_means(codebook.clone(), 1, &[], &options(2)).is_err());
    assert!(IvfPqIndex::new_e_means(codebook, 1, &[0.0; 3], &options(2)).is_err());
}
