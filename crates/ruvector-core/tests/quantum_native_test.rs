use ruvector_core::types::DbOptions;
use ruvector_core::types::{DistanceMetric, QuantumVector, SearchQuery, VectorEntry};
use ruvector_core::vector_db::VectorDB;
use std::collections::HashMap;

#[test]
fn test_quantum_native_flow() {
    let options = DbOptions {
        dimensions: 4,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: "/tmp/quantum_test.db".to_string(),
        ..Default::default()
    };

    let db = VectorDB::new(options).unwrap();

    // 1. Test Q8 Quantization
    let vec_f32 = vec![0.1, 0.2, 0.3, 0.4];
    // Normally quantization happens in the provider, but we can simulate it
    let q8_vec = QuantumVector::Q8(vec![12, 25, 38, 51], 0.0078);

    db.insert(VectorEntry {
        id: Some("v1".to_string()),
        vector: q8_vec.clone(),
        metadata: None,
    })
    .unwrap();

    // 2. Test Search with Q8
    let results = db
        .search(SearchQuery {
            vector: q8_vec,
            k: 1,
            filter: None,
            ef_search: None,
        })
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "v1");

    // 3. Test NF4 storage (manual insertion)
    let nf4_vec = QuantumVector::NF4 {
        data: vec![0x12, 0x34],
        scale: 1.0,
        orig_len: 4,
    };

    db.insert(VectorEntry {
        id: Some("v2".to_string()),
        vector: nf4_vec,
        metadata: None,
    })
    .unwrap();

    let results_all = db
        .search(SearchQuery {
            vector: QuantumVector::F32(vec_f32),
            k: 2,
            filter: None,
            ef_search: None,
        })
        .unwrap();

    assert_eq!(results_all.len(), 2);
}
