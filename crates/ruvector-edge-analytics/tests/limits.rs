//! Limits and budgets: every refusal is typed, and size/budget refusals
//! are HTTP 413 (never 500).
mod common;
use ruvector_edge_analytics::*;
use ruvector_edge_store::{ErrorCode, OpError};

const UID: GraphUid = [3; 16];

#[test]
fn manifest_limits_refuse_before_decoding_chunks() {
    let edges = common::sparse_graph(2_000, 6_000, 1);
    let g = TenantGraph::from_edges(UID, 1, &edges, &GraphLimits::INLINE).unwrap();
    let enc = encode_graph(&g, 1_000).unwrap();
    let m = Manifest::decode(&enc.manifest).unwrap();
    let tight = GraphLimits {
        max_vertices: 1_000,
        max_edges: 100_000,
    };
    // No chunks are even offered: the manifest alone decides.
    let e = decode_graph(&m, std::iter::empty(), &tight).unwrap_err();
    assert!(matches!(
        e,
        AnalyticsError::LimitExceeded {
            kind: LimitKind::Vertices,
            ..
        }
    ));
    assert_eq!(e.status(), 413);
    let tight = GraphLimits {
        max_vertices: 100_000,
        max_edges: 5_999,
    };
    let e = decode_graph(&m, std::iter::empty(), &tight).unwrap_err();
    assert!(matches!(
        e,
        AnalyticsError::LimitExceeded {
            kind: LimitKind::Edges,
            ..
        }
    ));
}

#[test]
fn inline_graph_limits_are_413() {
    let edges = common::sparse_graph(1_000, 3_000, 2);
    let g = TenantGraph::from_edges(UID, 1, &edges, &GraphLimits::INLINE).unwrap();
    let small = Profile {
        limits: GraphLimits {
            max_vertices: 999,
            max_edges: 10_000,
        },
        budget: Profile::INLINE.budget,
    };
    let e = query(&g, &QueryMode::Exact, &small).unwrap_err();
    assert_eq!((e.status(), e.code()), (413, ErrorCode::PayloadTooLarge));
}

#[test]
fn memory_budget_is_413_and_job_eligible_flag_is_honest() {
    // ~150k sparse edges: over the inline 64 MB estimate, inside the job's.
    let edges = common::sparse_graph(30_000, 150_000, 3);
    let g = TenantGraph::from_edges(UID, 1, &edges, &GraphLimits::INLINE).unwrap();
    let e = plan(&g, &QueryMode::Exact, &Profile::INLINE).unwrap_err();
    assert!(
        matches!(
            e,
            AnalyticsError::BudgetExceeded {
                resource: "memory_bytes",
                job_eligible: true,
                ..
            }
        ),
        "{e:?}"
    );
    assert_eq!(e.status(), 413);
    assert!(matches!(
        route(&g, &QueryMode::Exact).unwrap(),
        Route::Job(_)
    ));

    // Over even the job budget: 413, not job-eligible, route refuses too.
    let tiny = Profile {
        limits: GraphLimits::JOB,
        budget: Budget {
            max_work: 1,
            max_memory_bytes: 1,
        },
    };
    let e = plan(&g, &QueryMode::Exact, &tiny).unwrap_err();
    assert_eq!(e.status(), 413);
    let big = common::sparse_graph(240_000, 249_000, 4);
    let g = TenantGraph::from_edges(UID, 1, &big, &GraphLimits::JOB).unwrap();
    let e = route(&g, &QueryMode::Exact).unwrap_err();
    assert_eq!(e.status(), 413);
    assert!(
        matches!(
            e,
            AnalyticsError::BudgetExceeded {
                job_eligible: false,
                ..
            } | AnalyticsError::LimitExceeded { .. }
        ),
        "{e:?}"
    );
    let e = JobDescriptor::submit("j", &g, QueryMode::Exact, 0).unwrap_err();
    assert_eq!(e.status(), 413);
}

#[test]
fn approximate_requests_follow_exact_admission_and_validate_epsilon() {
    // No separate (dense-matrix) vertex cap any more: an approximate request
    // is planned, admitted and routed exactly like an exact one.
    for (k, m) in [(600u64, 5_000usize), (400, 3_000)] {
        let edges = common::two_clusters(k, m, 3, 5);
        let g = TenantGraph::from_edges(UID, 1, &edges, &GraphLimits::INLINE).unwrap();
        let approx = plan(
            &g,
            &QueryMode::Approximate { epsilon: 0.1 },
            &Profile::INLINE,
        );
        let exact = plan(&g, &QueryMode::Exact, &Profile::INLINE);
        assert_eq!(format!("{approx:?}"), format!("{exact:?}"));
    }
    // Over the inline budget: the same 413 + job_eligible as exact.
    let g = TenantGraph::from_edges(
        UID,
        1,
        &common::ring_chords(3_000, 9_000, 0xab),
        &GraphLimits::INLINE,
    )
    .unwrap();
    let e = plan(
        &g,
        &QueryMode::Approximate { epsilon: 0.5 },
        &Profile::INLINE,
    )
    .unwrap_err();
    assert_eq!(e.status(), 413);
    assert!(matches!(
        e,
        AnalyticsError::BudgetExceeded {
            job_eligible: true,
            ..
        }
    ));
    // Bad epsilon is a 400.
    for eps in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        let e = plan(
            &g,
            &QueryMode::Approximate { epsilon: eps },
            &Profile::INLINE,
        )
        .unwrap_err();
        assert_eq!(e.status(), 400, "eps {eps}");
    }
}

#[test]
fn errors_convert_to_op_errors_without_echoing_input() {
    let e = AnalyticsError::BudgetExceeded {
        resource: "work",
        estimated: 9,
        budget: 1,
        job_eligible: true,
    };
    let op: OpError = e.into();
    assert_eq!(op.code, ErrorCode::BudgetExceeded);
    assert_eq!(op.code.status(), 413);
    assert!(op.detail.contains("job"));
    let op: OpError = AnalyticsError::Corrupt(CorruptKind::Checksum).into();
    assert_eq!(op.code.status(), 500);
    assert_eq!(
        encode_graph(
            &TenantGraph::from_edges(UID, 1, &[], &GraphLimits::INLINE).unwrap(),
            0
        )
        .unwrap_err()
        .status(),
        400
    );
}
