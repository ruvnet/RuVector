//! Retrieval correctness contract (issues #1310, #1313):
//! - filtered top-k is never underfilled when at least k vectors match (10k corpus, 1% and 10% selectivity);
//! - re-inserting an existing ID is an upsert: one live row, no ghost of the old vector in top-k;
//! - non-finite vectors (NaN / ±Inf) are rejected at the boundary, before anything is stored.

use ruvector_core::types::{DbOptions, HnswConfig};
use ruvector_core::{DistanceMetric, SearchQuery, SearchResult, VectorDB, VectorEntry};
use serde_json::json;
use std::collections::{HashMap, HashSet};

const DIM: usize = 32;
const N: usize = 10_000;

fn vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n).map(|_| (0..DIM).map(|_| next()).collect()).collect()
}

fn db(hnsw: bool) -> VectorDB {
    VectorDB::new(DbOptions {
        dimensions: DIM,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: "memory://".to_string(),
        hnsw_config: hnsw.then(|| HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 64,
            max_elements: 2 * N,
        }),
        quantization: None,
    })
    .unwrap()
}

fn entry(id: String, vector: Vec<f32>, i: usize) -> VectorEntry {
    VectorEntry {
        id: Some(id),
        vector,
        metadata: Some(HashMap::from([
            // 1% selectivity: pct == 0; 10% selectivity: decile == 0.
            ("pct".to_string(), json!(i % 100)),
            ("decile".to_string(), json!(i % 10)),
        ])),
    }
}

fn populate(db: &VectorDB, base: &[Vec<f32>]) {
    let entries: Vec<VectorEntry> = base
        .iter()
        .enumerate()
        .map(|(i, v)| entry(format!("v{i}"), v.clone(), i))
        .collect();
    db.insert_batch(entries).unwrap();
}

fn query(v: &[f32], k: usize, filter: Option<serde_json::Value>) -> SearchQuery {
    SearchQuery {
        vector: v.to_vec(),
        k,
        filter: filter.map(|f| serde_json::from_value(f).unwrap()),
        ef_search: None,
    }
}

fn assert_distinct(got: &[SearchResult]) {
    let mut seen = HashSet::new();
    for r in got {
        assert!(seen.insert(r.id.clone()), "duplicate id {} in top-k", r.id);
    }
}

#[test]
fn filtered_top10_is_full_at_1_and_10_percent_selectivity() {
    let base = vectors(N, 11);
    let queries = vectors(25, 12);
    for hnsw in [true, false] {
        let db = db(hnsw);
        populate(&db, &base);
        for (field, modulo) in [("pct", 100usize), ("decile", 10usize)] {
            for q in &queries {
                let got = db.search(query(q, 10, Some(json!({ field: 0 })))).unwrap();
                assert_eq!(
                    got.len(),
                    10,
                    "hnsw={hnsw} {field}: {} eligible exist, got {}",
                    N / modulo,
                    got.len()
                );
                assert_distinct(&got);
                for r in &got {
                    let i: usize = r.id[1..].parse().unwrap();
                    assert_eq!(i % modulo, 0, "hnsw={hnsw} {field}: {} not eligible", r.id);
                }
            }
        }
    }
}

#[test]
fn duplicate_id_insert_is_an_upsert_without_ghosts() {
    let base = vectors(2_000, 21);
    let far = vectors(200, 22);
    for hnsw in [true, false] {
        let db = db(hnsw);
        populate(&db, &base);
        // Move the first 200 IDs far away from their original position, twice (single and batch paths).
        for (i, v) in far.iter().enumerate() {
            let moved: Vec<f32> = v.iter().map(|x| x + 10.0).collect();
            db.insert(entry(format!("v{i}"), moved, i)).unwrap();
        }
        let again: Vec<VectorEntry> = far
            .iter()
            .enumerate()
            .map(|(i, v)| entry(format!("v{i}"), v.iter().map(|x| x + 20.0).collect(), i))
            .collect();
        db.insert_batch(again).unwrap();
        assert_eq!(
            db.len().unwrap(),
            base.len(),
            "hnsw={hnsw}: upsert changed len"
        );

        for i in 0..200 {
            let got = db.search(query(&base[i], 10, None)).unwrap();
            assert_eq!(got.len(), 10, "hnsw={hnsw}: top-k underfilled after upsert");
            assert_distinct(&got);
            // The old position of v{i} must not answer for v{i} any more.
            if let Some(r) = got.iter().find(|r| r.id == format!("v{i}")) {
                panic!(
                    "hnsw={hnsw}: ghost of v{i} at old position, score {}",
                    r.score
                );
            }
            let stored = db.get(&format!("v{i}")).unwrap().unwrap();
            assert!((stored.vector[0] - (far[i][0] + 20.0)).abs() < 1e-5);
        }
        // Querying the new position finds the row exactly once.
        let target: Vec<f32> = far[3].iter().map(|x| x + 20.0).collect();
        let got = db.search(query(&target, 10, None)).unwrap();
        assert_eq!(got[0].id, "v3", "hnsw={hnsw}");
        assert!(got[0].score < 1e-3);
        assert_distinct(&got);
    }
}

#[test]
fn non_finite_vectors_are_rejected_before_storage() {
    for hnsw in [true, false] {
        let db = db(hnsw);
        populate(&db, &vectors(100, 31));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut v = vec![0.5f32; DIM];
            v[7] = bad;
            assert!(
                db.insert(entry("bad".into(), v.clone(), 0)).is_err(),
                "hnsw={hnsw}: insert accepted {bad}"
            );
            let batch = vec![
                entry("ok".into(), vec![0.1; DIM], 0),
                entry("bad".into(), v.clone(), 0),
            ];
            assert!(
                db.insert_batch(batch).is_err(),
                "hnsw={hnsw}: batch accepted {bad}"
            );
            assert!(
                db.get("ok").unwrap().is_none(),
                "hnsw={hnsw}: batch partially applied"
            );
            assert!(db.get("bad").unwrap().is_none());
            assert!(
                db.search(query(&v, 5, None)).is_err(),
                "hnsw={hnsw}: search accepted {bad} query"
            );
        }
        assert_eq!(db.len().unwrap(), 100);
    }
}
