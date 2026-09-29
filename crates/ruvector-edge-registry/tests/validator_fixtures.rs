//! Real `.rvf` files produced by the repo's rvf tooling validate and
//! round-trip; the rvf runtime (the `rvf` CLI engine) is the query oracle.

mod common;

use common::*;
use ruvector_edge_registry::validate::{validate, ValidationError, ValidationLimits};
use ruvector_edge_store::Metric;
use rvf_runtime::options::{DistanceMetric, QueryOptions, RvfOptions};
use rvf_runtime::RvfStore;

#[test]
fn every_checked_in_example_validates_or_is_refused_for_executables() {
    let mut names: Vec<_> = std::fs::read_dir(fixture_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "rvf"))
        .collect();
    names.sort();
    assert!(names.len() >= 40, "fixture set shrank: {}", names.len());
    let (mut ok, mut exec) = (0, 0);
    for p in &names {
        let bytes = std::fs::read(p).unwrap();
        match validate(&bytes, ValidationLimits::default()) {
            Ok(v) => {
                ok += 1;
                assert_eq!(v.total_size, bytes.len() as u64);
                let lenient = ValidationLimits {
                    allow_executable: true,
                    ..Default::default()
                };
                assert_eq!(validate(&bytes, lenient).unwrap(), v);
            }
            Err(ValidationError::ExecutableSegment { .. }) => {
                exec += 1;
                let lenient = ValidationLimits {
                    allow_executable: true,
                    ..Default::default()
                };
                validate(&bytes, lenient).unwrap_or_else(|e| panic!("{p:?}: {e}"));
            }
            Err(e) => panic!("{p:?}: {e}"),
        }
    }
    assert!(ok >= 35 && exec >= 5, "ok={ok} exec={exec}");
}

#[test]
fn basic_store_fixture_summary() {
    let bytes = fixture("basic_store.rvf");
    let v = validate(&bytes, ValidationLimits::default()).unwrap();
    assert_eq!(v.dim, 384);
    assert_eq!(v.metric, Metric::L2);
    assert_eq!(v.live_vec_segments.len(), 1);
    let live = v.live_vectors(&bytes).unwrap();
    assert_eq!(live.len() as u64, v.total_vectors);
    assert!(live.iter().all(|(_, x)| x.len() == 384));
    use sha2::Digest;
    assert_eq!(v.sha256, <[u8; 32]>::from(sha2::Sha256::digest(&bytes)));
}

#[test]
fn wire_writer_store_round_trips_with_and_without_root_page() {
    let data = rows(7, 5, 3);
    for root in [false, true] {
        let bytes = wire_store(5, &data, &[3], root);
        let v = validate(&bytes, ValidationLimits::default()).unwrap();
        assert_eq!((v.dim, v.metric, v.root_page), (5, Metric::Cosine, root));
        let live = v.live_vectors(&bytes).unwrap();
        let want: Vec<_> = data.iter().filter(|(id, _)| *id != 3).cloned().collect();
        assert_eq!(live, want);
    }
}

fn dist(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    match metric {
        DistanceMetric::L2 => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum(),
        DistanceMetric::InnerProduct => -dot,
        DistanceMetric::Cosine => {
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            1.0 - dot / (na * nb)
        }
    }
}

/// M5 acceptance, core half: an RVF written by the rvf CLI's engine
/// validates, and the vectors decoded from it answer queries exactly as the
/// engine does (after deletions and a reopen).
#[test]
fn runtime_created_store_validates_and_queries_identically() {
    for (metric, want) in [
        (DistanceMetric::L2, Metric::L2),
        (DistanceMetric::InnerProduct, Metric::Dot),
        (DistanceMetric::Cosine, Metric::Cosine),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvf");
        let opts = RvfOptions {
            dimension: 16,
            metric,
            ..Default::default()
        };
        let data = rows(200, 16, metric as u64 + 11);
        let mut store = RvfStore::create(&path, opts).unwrap();
        let (a, b) = data.split_at(120);
        for chunk in [a, b] {
            let vecs: Vec<&[f32]> = chunk.iter().map(|(_, v)| v.as_slice()).collect();
            let ids: Vec<u64> = chunk.iter().map(|(i, _)| *i).collect();
            store.ingest_batch(&vecs, &ids, None).unwrap();
        }
        store.delete(&[5, 17, 150]).unwrap();
        store.close().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let v = validate(&bytes, ValidationLimits::default()).unwrap();
        assert_eq!((v.dim, v.metric), (16, want));
        let live = v.live_vectors(&bytes).unwrap();
        assert_eq!(live.len(), 197);
        assert!(live.iter().all(|(id, _)| ![5, 17, 150].contains(id)));

        let store = RvfStore::open_readonly(&path).unwrap();
        for q in rows(5, 16, 99) {
            let mut scored: Vec<_> = live
                .iter()
                .map(|(id, x)| (dist(metric, &q.1, x), *id))
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            let ours: Vec<u64> = scored.iter().take(10).map(|s| s.1).collect();
            let theirs: Vec<u64> = store
                .query(
                    &q.1,
                    10,
                    &QueryOptions {
                        force_exact: true,
                        ..Default::default()
                    },
                )
                .unwrap()
                .iter()
                .map(|r| r.id)
                .collect();
            assert_eq!(ours, theirs, "metric {metric:?}");
        }
    }
}
