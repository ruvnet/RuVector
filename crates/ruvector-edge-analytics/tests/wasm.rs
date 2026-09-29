//! wasm32 runtime checks (ADR-351 M4 "rayon wasm test"). Run with
//! `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
//! cargo test -p ruvector-edge-analytics --target wasm32-unknown-unknown
//! --test wasm` (Node; wasm-bindgen-test-runner matching the lockfile's
//! wasm-bindgen). On native targets this file compiles to nothing.
#![cfg(target_arch = "wasm32")]

mod common;
use ruvector_edge_analytics::*;
use wasm_bindgen_test::wasm_bindgen_test;

const UID: GraphUid = *b"tenant-graph-uid";

/// Encode, decode and query exactly as the service would on the edge.
fn served(edges: &[(u64, u64, f64)]) -> CutReport {
    let g = TenantGraph::from_edges(UID, 42, edges, &GraphLimits::INLINE).unwrap();
    let enc = encode_graph(&g, MAX_CHUNK_EDGES).unwrap();
    let m = Manifest::decode(&enc.manifest).unwrap();
    let g = decode_graph(
        &m,
        enc.chunks.iter().map(Vec::as_slice),
        &GraphLimits::INLINE,
    )
    .unwrap();
    query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap()
}

#[wasm_bindgen_test]
fn rayon_falls_back_to_the_calling_thread() {
    // No threads on wasm32-unknown-unknown: rayon-core rebuilds the global
    // pool as one current-thread worker instead of panicking.
    let (threads, sum) = runtime::rayon_probe();
    assert_eq!(threads, 1);
    assert_eq!(sum, 99_990_000);
    assert_eq!(runtime::worker_threads(), 1);
}

#[wasm_bindgen_test]
fn sparse_50k_equals_native_digest() {
    let r = served(&common::sparse_50k());
    assert_eq!(r.estimate.path, SolverPath::Certified);
    assert_eq!(r.digest_hex(), common::SPARSE_50K_DIGEST);
}

#[wasm_bindgen_test]
fn clusters_50k_stoer_wagner_equals_native_digest() {
    let r = served(&common::clusters_50k());
    assert_eq!(r.estimate.path, SolverPath::StoerWagner);
    assert_eq!(r.value, Some(5.0));
    assert_eq!(r.digest_hex(), common::CLUSTERS_50K_DIGEST);
}

#[wasm_bindgen_test]
fn budget_exhaustion_is_413_on_wasm() {
    let g = TenantGraph::from_edges(
        UID,
        1,
        &common::ring_chords(3_000, 9_000, 0xab),
        &GraphLimits::INLINE,
    )
    .unwrap();
    let e = query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap_err();
    assert_eq!(e.status(), 413);
    let e = query(
        &g,
        &QueryMode::Approximate { epsilon: 0.2 },
        &Profile::INLINE,
    )
    .unwrap_err();
    assert_eq!(e.status(), 413);
    // An approximate request is answered by the exact solver.
    let g = TenantGraph::from_edges(
        UID,
        1,
        &common::two_clusters(100, 800, 3, 4),
        &GraphLimits::INLINE,
    )
    .unwrap();
    let a = query(
        &g,
        &QueryMode::Approximate { epsilon: 0.2 },
        &Profile::INLINE,
    )
    .unwrap();
    let x = query(&g, &QueryMode::Exact, &Profile::INLINE).unwrap();
    assert_eq!(a.digest_hex(), x.digest_hex());
}
