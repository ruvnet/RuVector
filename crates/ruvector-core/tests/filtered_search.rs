//! VectorDB filtered search: results come from the matching set only and are never fewer than min(k, #matching).

#![cfg(feature = "hnsw")]

use ruvector_core::types::{
    DbOptions, HnswConfig, QuantizationConfig, SearchPolicy, SearchQuantization,
};
use ruvector_core::vector_db::FilterPlan;
use ruvector_core::{DistanceMetric, SearchQuery, SearchResult, VectorDB, VectorEntry};
use serde_json::{json, Value};
use std::collections::HashMap;

const DIM: usize = 16;

fn vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n).map(|_| (0..DIM).map(|_| next()).collect()).collect()
}

fn options(quantization: Option<QuantizationConfig>, hnsw: bool) -> DbOptions {
    DbOptions {
        dimensions: DIM,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: "memory://".to_string(),
        hnsw_config: hnsw.then(|| HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 64,
            max_elements: 100_000,
        }),
        quantization,
    }
}

/// Vector `i` gets `{"bucket": i % 100, "even": i % 2 == 0}`.
fn populate(db: &VectorDB, base: &[Vec<f32>]) {
    let entries: Vec<VectorEntry> = base
        .iter()
        .enumerate()
        .map(|(i, v)| VectorEntry {
            id: Some(format!("v{i}")),
            vector: v.clone(),
            metadata: Some(HashMap::from([
                ("bucket".to_string(), json!(i % 100)),
                ("even".to_string(), json!(i % 2 == 0)),
            ])),
        })
        .collect();
    db.insert_batch(entries).unwrap();
}

fn filter(v: Value) -> Option<HashMap<String, Value>> {
    Some(serde_json::from_value(v).unwrap())
}

fn query(v: &[f32], k: usize, f: Value) -> SearchQuery {
    SearchQuery {
        vector: v.to_vec(),
        k,
        filter: filter(f),
        ef_search: None,
    }
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn truth(base: &[Vec<f32>], q: &[f32], k: usize, pass: impl Fn(usize) -> bool) -> Vec<String> {
    let mut d: Vec<(f32, usize)> = (0..base.len())
        .filter(|&i| pass(i))
        .map(|i| (l2(q, &base[i]), i))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.into_iter().take(k).map(|x| format!("v{}", x.1)).collect()
}

fn ids(r: &[SearchResult]) -> Vec<String> {
    r.iter().map(|x| x.id.clone()).collect()
}

fn recall(got: &[SearchResult], want: &[String]) -> usize {
    got.iter().filter(|r| want.contains(&r.id)).count()
}

#[test]
fn every_plan_returns_k_matching_results_with_high_recall() {
    let base = vectors(5000, 1);
    let queries = vectors(30, 2);
    let db = VectorDB::new(options(None, true)).unwrap();
    populate(&db, &base);
    for (f, pass, min_recall) in [
        (
            json!({"bucket": 7}),
            (|i: usize| i % 100 == 7) as fn(usize) -> bool,
            1.0,
        ),
        (json!({"even": true}), |i: usize| i % 2 == 0, 0.9),
        (
            json!({"even": true, "bucket": 8}),
            |i: usize| i % 100 == 8,
            1.0,
        ),
    ] {
        for plan in [FilterPlan::Auto, FilterPlan::Graph, FilterPlan::Exact] {
            let mut hits = 0;
            for q in &queries {
                let want = truth(&base, q, 10, pass);
                let got = db.search_with_plan(query(q, 10, f.clone()), plan).unwrap();
                assert_eq!(got.len(), 10, "{f} {plan:?} returns k");
                assert!(got.windows(2).all(|w| w[0].score <= w[1].score));
                for r in &got {
                    let i: usize = r.id[1..].parse().unwrap();
                    assert!(pass(i), "{} fails {f}", r.id);
                    assert!(r.vector.is_some() && r.metadata.is_some(), "enriched");
                }
                hits += recall(&got, &want);
            }
            let rec = hits as f64 / (queries.len() * 10) as f64;
            let floor = if plan == FilterPlan::Exact {
                1.0
            } else {
                min_recall
            };
            assert!(rec >= floor, "{f} {plan:?} recall {rec} < {floor}");
        }
    }
}

#[test]
fn k_above_the_match_count_returns_every_match_and_no_match_returns_nothing() {
    let base = vectors(2000, 3);
    let db = VectorDB::new(options(None, true)).unwrap();
    populate(&db, &base);
    let mut got = ids(&db
        .search(query(&base[0], 50, json!({"bucket": 42})))
        .unwrap());
    got.sort();
    let mut want: Vec<String> = (0..2000)
        .filter(|i| i % 100 == 42)
        .map(|i| format!("v{i}"))
        .collect();
    want.sort();
    assert_eq!(got, want);
    for f in [
        json!({"bucket": 1000}),
        json!({"nope": 1}),
        json!({"bucket": "7"}),
    ] {
        assert!(db.search(query(&base[0], 10, f)).unwrap().is_empty());
    }
    // An empty filter is no filter.
    assert_eq!(db.search(query(&base[0], 10, json!({}))).unwrap().len(), 10);
}

#[test]
fn deleted_and_overwritten_ids_are_excluded() {
    let base = vectors(3000, 4);
    let db = VectorDB::new(options(None, true)).unwrap();
    populate(&db, &base);
    let matching: Vec<usize> = (0..3000).filter(|i| i % 2 == 0).collect();
    for i in matching.iter().take(400) {
        assert!(db.delete(&format!("v{i}")).unwrap());
    }
    // Re-insert v1 (odd) with `even: true`, and v2 (even) with `even: false`.
    for (i, even) in [(1usize, true), (2usize, false)] {
        db.insert(VectorEntry {
            id: Some(format!("v{i}")),
            vector: base[i].clone(),
            metadata: Some(HashMap::from([("even".to_string(), json!(even))])),
        })
        .unwrap();
    }
    for plan in [FilterPlan::Auto, FilterPlan::Graph, FilterPlan::Exact] {
        for q in [&base[0], &base[1], &base[2]] {
            let got = db
                .search_with_plan(query(q, 20, json!({"even": true})), plan)
                .unwrap();
            assert_eq!(got.len(), 20, "{plan:?}");
            let mut seen = std::collections::HashSet::new();
            for r in &got {
                assert!(seen.insert(r.id.clone()), "{plan:?} duplicate {}", r.id);
                let i: usize = r.id[1..].parse().unwrap();
                let deleted = i % 2 == 0 && matching[..400].contains(&i);
                assert!(!deleted, "{plan:?} returned deleted {}", r.id);
                assert!(r.id != "v2", "{plan:?} returned overwritten v2");
                assert!(i % 2 == 0 || i == 1, "{plan:?} returned {}", r.id);
            }
        }
        let got = db
            .search_with_plan(query(&base[1], 1, json!({"even": true})), plan)
            .unwrap();
        assert_eq!(ids(&got), vec!["v1"], "{plan:?} finds the re-inserted id");
    }
}

#[test]
fn flat_and_turbo4_indexes_filter_correctly() {
    let base = vectors(3000, 5);
    let turbo4 = QuantizationConfig::Turbo4 {
        rotation_seed: 7,
        rescore_multiplier: 4,
        policy: SearchPolicy::Balanced,
        search_quantization: SearchQuantization::default(),
    };
    for (quant, hnsw) in [(None, false), (Some(turbo4), true)] {
        let db = VectorDB::new(options(quant, hnsw)).unwrap();
        populate(&db, &base);
        for (f, pass) in [
            (
                json!({"bucket": 3}),
                (|i: usize| i % 100 == 3) as fn(usize) -> bool,
            ),
            (json!({"even": false}), |i: usize| i % 2 == 1),
        ] {
            let q = &base[11];
            let got = db.search(query(q, 10, f.clone())).unwrap();
            assert_eq!(got.len(), 10, "hnsw={hnsw} {f}");
            assert!(got.iter().all(|r| pass(r.id[1..].parse().unwrap())));
            let want = truth(&base, q, 10, pass);
            assert!(
                recall(&got, &want) >= 8,
                "hnsw={hnsw} {f}: {:?} vs {want:?}",
                ids(&got)
            );
        }
    }
}

#[test]
fn huge_k_returns_everything_without_aborting() {
    let base = vectors(300, 6);
    let turbo4 = QuantizationConfig::Turbo4 {
        rotation_seed: 7,
        rescore_multiplier: 4,
        policy: SearchPolicy::Balanced,
        search_quantization: SearchQuantization::default(),
    };
    for (quant, hnsw) in [(None, true), (None, false), (Some(turbo4), true)] {
        let db = VectorDB::new(options(quant, hnsw)).unwrap();
        populate(&db, &base);
        for ef_search in [None, Some(usize::MAX)] {
            let q = |f: Option<HashMap<String, Value>>| SearchQuery {
                vector: base[0].clone(),
                k: usize::MAX,
                filter: f,
                ef_search,
            };
            assert_eq!(db.search(q(None)).unwrap().len(), 300, "hnsw={hnsw}");
            for plan in [FilterPlan::Auto, FilterPlan::Graph, FilterPlan::Exact] {
                let got = db
                    .search_with_plan(q(filter(json!({"even": true}))), plan)
                    .unwrap();
                assert_eq!(got.len(), 150, "hnsw={hnsw} {plan:?}");
            }
        }
    }
}

#[test]
#[cfg(feature = "storage")]
fn metadata_view_is_rebuilt_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(None, true);
    opts.storage_path = dir.path().join("f.db").to_string_lossy().to_string();
    let base = vectors(500, 7);
    {
        let db = VectorDB::new(opts.clone()).unwrap();
        populate(&db, &base);
        db.delete("v3").unwrap();
    }
    let db = VectorDB::new(opts).unwrap();
    let mut got = ids(&db
        .search(query(&base[0], 10, json!({"bucket": 3})))
        .unwrap());
    got.sort();
    assert_eq!(got, vec!["v103", "v203", "v303", "v403"]);
}
