//! M5 acceptance, core half (ADR-351 §15): an RVF exported by the rvf CLI
//! engine imports via `upload_id` and queries correctly.

mod common;

use common::*;
use ruvector_edge_auth::Capability;
use ruvector_edge_registry::validate::{validate, ValidationLimits, VecSegView};
use ruvector_edge_registry::Visibility;

/// M5 acceptance (core half): an RVF exported by the rvf CLI engine imports
/// via `upload_id`, is pulled from its blob, decodes segment by segment
/// through ranged reads, and answers queries as the engine does.
#[test]
fn rvf_cli_export_imports_via_upload_id_and_queries_correctly() {
    use rvf_runtime::options::{DistanceMetric, QueryOptions, RvfOptions};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("export.rvf");
    let data = rows(300, 32, 5);
    let mut store = rvf_runtime::RvfStore::create(
        &path,
        RvfOptions {
            dimension: 32,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .unwrap();
    for chunk in data.chunks(64) {
        let vecs: Vec<&[f32]> = chunk.iter().map(|(_, v)| v.as_slice()).collect();
        let ids: Vec<u64> = chunk.iter().map(|(i, _)| *i).collect();
        store.ingest_batch(&vecs, &ids, None).unwrap();
    }
    store.delete(&[1, 2, 3]).unwrap();
    store.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();

    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let m = push_with(
        &reg,
        &r2,
        &a,
        "@acme/export",
        "1.0.0",
        Visibility::Tenant,
        &bytes,
        4,
        Declare::default(),
    )
    .unwrap();
    assert_eq!((m.dim, m.metric), (32, ruvector_edge_store::Metric::Cosine));
    let reader = caller('a', "es1_bob", &[Capability::Read]);
    let t = reg
        .pull(&reader, &name("@acme/export"), &ver("1.0.0"))
        .unwrap();
    let blob = r2.get(t.blob.as_str()).unwrap();
    assert_eq!(sha(&blob), m.sha256);

    // The Worker's import: re-validate the pulled object, then decode each
    // live VEC_SEG from a ranged read, dropping deleted ids.
    let v = validate(&blob, ValidationLimits::default()).unwrap();
    let mut imported = Vec::new();
    for &i in &v.live_vec_segments {
        let r = v.segments[i as usize].payload_range();
        for (id, x) in VecSegView::decode(&blob[r.start as usize..r.end as usize], v.dim).unwrap() {
            if v.deleted_ids.binary_search(&id).is_err() {
                imported.push((id, x));
            }
        }
    }
    assert_eq!(imported.len(), 297);

    let engine = rvf_runtime::RvfStore::open_readonly(&path).unwrap();
    let cos = |p: &[f32], q: &[f32]| {
        let d: f32 = p.iter().zip(q).map(|(x, y)| x * y).sum();
        let n = |z: &[f32]| z.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - d / (n(p) * n(q))
    };
    for (_, q) in rows(8, 32, 77) {
        let mut ours: Vec<_> = imported.iter().map(|(id, x)| (cos(&q, x), *id)).collect();
        ours.sort_by(|x, y| x.0.total_cmp(&y.0));
        let ours: Vec<u64> = ours.iter().take(10).map(|s| s.1).collect();
        let theirs: Vec<u64> = engine
            .query(
                &q,
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
        assert_eq!(ours, theirs);
    }
}
