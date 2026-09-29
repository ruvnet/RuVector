//! ADR-351 M4 acceptance: min-cut on 50k edges, served from the persisted
//! chunked edge list, equals the native `ruvector-mincut` result; an
//! over-budget inline query is a 413 and the job path answers it.
mod common;
use ruvector_edge_analytics::*;

use common::{CLUSTERS_50K_DIGEST, SPARSE_50K_DIGEST};
const UID: GraphUid = *b"tenant-graph-uid";

fn roundtrip(edges: &[(u64, u64, f64)], chunk_edges: u32) -> (TenantGraph, Manifest) {
    let g = TenantGraph::from_edges(UID, 42, edges, &GraphLimits::INLINE).unwrap();
    let enc = encode_graph(&g, chunk_edges).unwrap();
    assert_eq!(
        enc.chunks.len(),
        (edges.len() as u32).div_ceil(chunk_edges) as usize
    );
    for c in &enc.chunks {
        assert!(c.len() < 2 << 20, "chunk must fit a 2 MB DO BLOB");
    }
    let m = Manifest::decode(&enc.manifest).unwrap();
    let back = decode_graph(
        &m,
        enc.chunks.iter().map(Vec::as_slice),
        &GraphLimits::INLINE,
    )
    .unwrap();
    assert_eq!(back.edges(), g.edges());
    assert_eq!((back.uid(), back.revision()), (g.uid(), g.revision()));
    assert_eq!(back.snapshot_digest(), Some(&m.digest));
    (back, m)
}

fn assert_equals_native(edges: &[(u64, u64, f64)], r: &CutReport) {
    let (value, partition, cut) = common::native_exact(edges);
    assert_eq!(r.value, Some(value));
    assert_eq!(r.partition.as_ref(), Some(&partition));
    let ours: Vec<(u64, u64, f64)> = r
        .cut_edges
        .as_ref()
        .unwrap()
        .iter()
        .map(|e| (e.u, e.v, e.w))
        .collect();
    assert_eq!(ours, cut);
    let s: f64 = ours.iter().map(|e| e.2).sum();
    assert_eq!(s, value, "cut edges must sum to the value");
}

#[test]
fn sparse_50k_edges_equals_native() {
    let edges = common::sparse_50k();
    let (g, _) = roundtrip(&edges, MAX_CHUNK_EDGES / 4);
    let r = query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap();
    assert_eq!(r.estimate.path, SolverPath::Certified);
    assert_equals_native(&edges, &r);
    println!("sparse digest {}", r.digest_hex());
    assert_eq!(r.digest_hex(), SPARSE_50K_DIGEST);
}

#[test]
fn clusters_50k_edges_stoer_wagner_equals_native() {
    let edges = common::clusters_50k();
    let (g, _) = roundtrip(&edges, MAX_CHUNK_EDGES);
    let r = query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap();
    assert_eq!(r.estimate.path, SolverPath::StoerWagner);
    assert_equals_native(&edges, &r);
    assert_eq!(r.value, Some(5.0));
    println!("clusters digest {}", r.digest_hex());
    assert_eq!(r.digest_hex(), CLUSTERS_50K_DIGEST);
}

#[test]
fn over_budget_inline_is_413_and_job_answers_it() {
    let edges = common::ring_chords(3_000, 9_000, 0xab);
    let (g, m) = roundtrip(&edges, 1_000);
    let err = query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap_err();
    assert!(
        matches!(
            err,
            AnalyticsError::BudgetExceeded {
                resource: "work",
                job_eligible: true,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(err.status(), 413);
    assert_eq!(err.code().as_str(), "budget_exceeded");
    assert!(matches!(
        route(&g, &QueryMode::Exact).unwrap(),
        Route::Job(_)
    ));

    let mut job = JobDescriptor::submit("job-1", &g, QueryMode::Exact, 10).unwrap();
    assert_eq!(job.snapshot_digest, m.digest_hex());
    assert_eq!(job.state, JobState::Queued);
    assert!(job.run(&g, 11).is_err(), "must start first");
    job.start(11).unwrap();
    // Persist and restore the descriptor between alarm invocations.
    let stored = serde_json::to_string(&job).unwrap();
    let mut job: JobDescriptor = serde_json::from_str(&stored).unwrap();
    job.run(&g, 12).unwrap();
    let JobState::Done { report } = &job.state else {
        panic!("{:?}", job.state)
    };
    assert_equals_native(&edges, report);
    assert!(job.is_terminal());
    assert!(job.start(13).is_err());
}

#[test]
fn job_refuses_a_changed_graph_and_bounds_restarts() {
    let edges = common::ring_chords(300, 900, 1);
    let (g, m) = roundtrip(&edges, 1_000);
    let mut job = JobDescriptor::submit("j", &g, QueryMode::Exact, 0).unwrap();
    assert_eq!(job.snapshot_digest, m.digest_hex());
    job.start(1).unwrap();
    let newer = decoded(&edges, 43, 1_000);
    job.run(&newer, 2).unwrap();
    assert!(matches!(&job.state, JobState::Failed { code, .. } if code.status() == 409));

    // Same uid, revision and edges, but a different snapshot (other chunking,
    // so other bytes and digest): the pin is the bytes, not the host's word.
    let mut job = JobDescriptor::submit("r", &g, QueryMode::Exact, 0).unwrap();
    job.start(1).unwrap();
    job.run(&decoded(&edges, 42, 100), 2).unwrap();
    assert!(matches!(&job.state, JobState::Failed { code, .. } if code.status() == 409));

    // A graph that was never persisted cannot be pinned: submit is a 409,
    // and running a pinned job on one fails the same way.
    let fresh = TenantGraph::from_edges(UID, 42, &edges, &GraphLimits::JOB).unwrap();
    assert_eq!(fresh.snapshot_digest(), None);
    let e = JobDescriptor::submit("f", &fresh, QueryMode::Exact, 0).unwrap_err();
    assert_eq!(e.status(), 409);
    let mut job = JobDescriptor::submit("u", &g, QueryMode::Exact, 0).unwrap();
    job.start(1).unwrap();
    job.run(&fresh, 2).unwrap();
    assert!(matches!(&job.state, JobState::Failed { code, .. } if code.status() == 409));

    let mut job = JobDescriptor::submit("k", &g, QueryMode::Exact, 0).unwrap();
    for _ in 0..job::MAX_ATTEMPTS {
        job.start(1).unwrap();
    }
    assert!(job.start(2).is_err());
    assert!(matches!(job.state, JobState::Failed { .. }));
    assert!(JobDescriptor::submit("bad id!", &g, QueryMode::Exact, 0).is_err());
}

/// Encode at `revision` with `chunk_edges`, then decode (as a host would).
fn decoded(edges: &[(u64, u64, f64)], revision: u64, chunk_edges: u32) -> TenantGraph {
    let g = TenantGraph::from_edges(UID, revision, edges, &GraphLimits::INLINE).unwrap();
    let enc = encode_graph(&g, chunk_edges).unwrap();
    let m = Manifest::decode(&enc.manifest).unwrap();
    decode_graph(
        &m,
        enc.chunks.iter().map(Vec::as_slice),
        &GraphLimits::INLINE,
    )
    .unwrap()
}

/// Approximate requests are answered by the exact solver. The heuristic
/// `ApproxMinCut` is not served: on graphs above its 50-edge exact cutoff
/// its estimate has no sound bound (the review probe measured 0.333 against
/// an exact 3.333 on a 300-vertex weighted ring with chords; on the graphs
/// below it returned 1.0 against exact 3, 4 and 5, i.e. 0.20-0.33x). The
/// test prints the ratios it observes.
#[test]
fn approximate_requests_are_answered_exactly() {
    use ruvector_mincut::{ApproxMinCut, ApproxMinCutConfig};
    for (name, edges) in [
        ("ring+chords 300/1.5k", common::ring_chords(300, 1_500, 7)),
        (
            "weighted ring+chords 300/1.5k",
            common::ring_chords(300, 1_500, 7)
                .into_iter()
                .map(|(u, v, _)| (u, v, 1.0 + ((u ^ v) % 3) as f64 / 3.0))
                .collect(),
        ),
        ("two clusters 100/800", common::two_clusters(100, 800, 3, 4)),
    ] {
        let g = TenantGraph::from_edges(UID, 1, &edges, &GraphLimits::INLINE).unwrap();
        let exact = query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap();
        for eps in [0.1, 0.5, 1.0] {
            let a = query(
                &g,
                &QueryMode::Approximate { epsilon: eps },
                &Profile::INLINE,
            )
            .unwrap();
            assert_eq!(
                a.mode,
                QueryMode::Exact,
                "{name}: the report names the solver"
            );
            assert_eq!(a.digest_hex(), exact.digest_hex(), "{name} eps {eps}");
            assert!(a.partition.is_some() && a.cut_edges.is_some());
        }
        let mut ac = ApproxMinCut::new(ApproxMinCutConfig {
            epsilon: 0.5,
            ..Default::default()
        });
        for e in g.edges() {
            ac.insert_edge(e.u, e.v, e.w);
        }
        let h = ac.min_cut().value;
        let x = exact.value.unwrap();
        println!(
            "{name}: exact {x:.4}, ApproxMinCut(eps 0.5) {h:.4} (ratio {:.3})",
            h / x
        );
    }
}

#[test]
fn trivial_graphs_have_no_cut() {
    let g = TenantGraph::from_edges(UID, 1, &[], &GraphLimits::INLINE).unwrap();
    let r = query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap();
    assert_eq!(r.value, None);
    assert_eq!(r.estimate.path, SolverPath::Trivial);
    let enc = encode_graph(&g, 16).unwrap();
    assert!(enc.chunks.is_empty());
    let m = Manifest::decode(&enc.manifest).unwrap();
    let back = decode_graph(&m, std::iter::empty(), &GraphLimits::INLINE).unwrap();
    assert_eq!(back.edges(), g.edges());
    assert_eq!(back.snapshot_digest(), Some(&m.digest));
    // Serializes without NaN/Infinity.
    assert!(serde_json::to_string(&r)
        .unwrap()
        .contains("\"value\":null"));
}
