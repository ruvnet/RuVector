//! Integration tests for ruvector-core: HNSW index, FlatIndex, distance metrics, and VectorDB.
//!
//! All tests use real types and real computations -- no mocks.

use ruvector_core::distance::distance;
use ruvector_core::index::flat::FlatIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};
use ruvector_core::VectorDB;

// ---------------------------------------------------------------------------
// Helper: generate a deterministic f32 vector of given dimension
// ---------------------------------------------------------------------------
fn make_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = ((seed as f64 * 0.618033988749895 + i as f64 * 0.414213562373095) % 1.0) as f32;
            x * 2.0 - 1.0
        })
        .collect()
}

/// Normalize a vector in-place to unit length.
fn normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// ===========================================================================
// 1. FlatIndex: basic insertion and search
// ===========================================================================
#[test]
fn flat_index_insert_and_search() {
    let dim = 64;
    let mut index = FlatIndex::new(dim, DistanceMetric::Euclidean);

    for i in 0..100u64 {
        let v = make_vector(dim, i);
        index.add(format!("vec_{i}"), v).expect("insert should succeed");
    }

    assert_eq!(index.len(), 100, "index should contain 100 vectors");

    let query = make_vector(dim, 0);
    let results = index.search(&query, 5).expect("search should succeed");

    assert_eq!(results.len(), 5, "should return exactly k=5 results");
    // The closest result to seed=0 should be itself (vec_0)
    assert_eq!(results[0].id, "vec_0", "nearest neighbor should be vec_0");
    assert!(
        results[0].score < 1e-6,
        "distance to self should be near zero, got {}",
        results[0].score
    );
}

// ===========================================================================
// 2. FlatIndex: empty search returns empty
// ===========================================================================
#[test]
fn flat_index_empty_search() {
    let index = FlatIndex::new(32, DistanceMetric::Cosine);
    let query = vec![1.0; 32];
    let results = index.search(&query, 10).expect("search on empty index should succeed");
    assert!(
        results.is_empty(),
        "search on empty index should return no results"
    );
}

// ===========================================================================
// 3. FlatIndex: remove vector
// ===========================================================================
#[test]
fn flat_index_remove_vector() {
    let dim = 16;
    let mut index = FlatIndex::new(dim, DistanceMetric::Euclidean);

    index
        .add("a".to_string(), vec![1.0; dim])
        .expect("insert a");
    index
        .add("b".to_string(), vec![2.0; dim])
        .expect("insert b");

    assert_eq!(index.len(), 2);

    let removed = index.remove(&"a".to_string()).expect("remove should succeed");
    assert!(removed, "remove should return true for existing vector");
    assert_eq!(index.len(), 1, "one vector should remain after removal");

    let removed_again = index.remove(&"a".to_string()).expect("second remove should succeed");
    assert!(
        !removed_again,
        "removing a non-existent vector should return false"
    );
}

// ===========================================================================
// 4. Distance metrics: Euclidean, Cosine, Manhattan, DotProduct
// ===========================================================================
#[test]
fn distance_euclidean_identity() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let d = distance(&v, &v, DistanceMetric::Euclidean).expect("euclidean distance");
    assert!(
        d.abs() < 1e-6,
        "euclidean distance of a vector to itself should be 0, got {d}"
    );
}

#[test]
fn distance_cosine_orthogonal() {
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    let d = distance(&a, &b, DistanceMetric::Cosine).expect("cosine distance");
    // Cosine distance of orthogonal vectors is 1.0
    assert!(
        (d - 1.0).abs() < 1e-4,
        "cosine distance of orthogonal vectors should be ~1.0, got {d}"
    );
}

#[test]
fn distance_manhattan_known_value() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 6.0, 3.0];
    let d = distance(&a, &b, DistanceMetric::Manhattan).expect("manhattan distance");
    // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
    assert!(
        (d - 7.0).abs() < 1e-4,
        "manhattan distance should be 7.0, got {d}"
    );
}

// ===========================================================================
// 5. HNSW index: construction, insert, and recall
// ===========================================================================
#[cfg(feature = "hnsw")]
#[test]
fn hnsw_index_basic_recall() {
    use ruvector_core::index::hnsw::HnswIndex;

    let dim = 128;
    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        ef_search: 50,
        max_elements: 1000,
    };

    let mut hnsw = HnswIndex::new(dim, DistanceMetric::Euclidean, config)
        .expect("HNSW index creation should succeed");

    let n = 500;
    for i in 0..n {
        let v = make_vector(dim, i as u64);
        hnsw.add(format!("v_{i}"), v)
            .expect("HNSW insert should succeed");
    }

    assert_eq!(hnsw.len(), n, "HNSW should contain {n} vectors");

    // Search for a known vector -- it should appear as the top result
    let query = make_vector(dim, 42);
    let results = hnsw.search(&query, 10).expect("HNSW search should succeed");

    assert!(!results.is_empty(), "HNSW search should return results");
    assert_eq!(
        results[0].id, "v_42",
        "the exact vector should be the top-1 result"
    );
}

// ===========================================================================
// 6. HNSW index: search with k larger than index size
// ===========================================================================
#[cfg(feature = "hnsw")]
#[test]
fn hnsw_search_k_exceeds_size() {
    use ruvector_core::index::hnsw::HnswIndex;

    let dim = 32;
    let config = HnswConfig::default();

    let mut hnsw = HnswIndex::new(dim, DistanceMetric::Euclidean, config)
        .expect("HNSW creation");

    hnsw.add("only".to_string(), vec![0.5; dim])
        .expect("insert single vector");

    let results = hnsw.search(&vec![0.5; dim], 100).expect("search");

    assert!(
        results.len() <= 1,
        "search with k=100 on 1-element index should return at most 1 result, got {}",
        results.len()
    );
}

// ===========================================================================
// 7. FlatIndex: batch add
// ===========================================================================
#[test]
fn flat_index_batch_add() {
    let dim = 16;
    let mut index = FlatIndex::new(dim, DistanceMetric::Euclidean);

    let batch: Vec<(String, Vec<f32>)> = (0..50)
        .map(|i| (format!("batch_{i}"), make_vector(dim, i)))
        .collect();

    index.add_batch(batch).expect("batch add should succeed");
    assert_eq!(index.len(), 50, "batch add should insert all 50 vectors");
}

// ===========================================================================
// 8. VectorDB: end-to-end insert and search
// ===========================================================================
#[test]
fn vector_db_end_to_end() {
    let dim = 64;
    let db = VectorDB::new(dim, DistanceMetric::Euclidean);

    for i in 0..200u64 {
        let v = make_vector(dim, i);
        db.insert(format!("doc_{i}"), v)
            .expect("VectorDB insert should succeed");
    }

    let query = make_vector(dim, 99);
    let results = db.search(&query, 3).expect("VectorDB search should succeed");

    assert_eq!(results.len(), 3, "should return exactly 3 results");
    assert_eq!(
        results[0].id, "doc_99",
        "top result should be the exact match"
    );
}

// ===========================================================================
// 9. VectorDB: search on empty DB
// ===========================================================================
#[test]
fn vector_db_empty_search() {
    let db = VectorDB::new(32, DistanceMetric::Cosine);
    let results = db
        .search(&vec![1.0; 32], 5)
        .expect("search on empty DB should succeed");
    assert!(results.is_empty(), "empty DB search should return nothing");
}

// ===========================================================================
// 10. Large-scale brute-force recall validation
// ===========================================================================
#[test]
fn flat_index_recall_at_10() {
    let dim = 32;
    let n = 1000;
    let k = 10;
    let mut index = FlatIndex::new(dim, DistanceMetric::Euclidean);

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let v = make_vector(dim, i as u64);
        vectors.push(v.clone());
        index
            .add(format!("v_{i}"), v)
            .expect("insert");
    }

    // Compute ground truth for query = vectors[0]
    let query = &vectors[0];
    let mut dists: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ground_truth: Vec<String> = dists[..k].iter().map(|(i, _)| format!("v_{i}")).collect();

    // Search
    let results = index.search(query, k).expect("search");
    let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();

    // All ground truth items must appear in results (for a flat index, recall should be 100%)
    for gt_id in &ground_truth {
        assert!(
            result_ids.contains(gt_id),
            "recall miss: {} not found in top-{k} results",
            gt_id
        );
    }
}
