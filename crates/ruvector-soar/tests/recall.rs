//! Integration tests — real synthetic data, real recall numbers, no mocks.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_soar::{brute_force_topk, recall, Assignment, IvfIndex};

fn synth(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let anchors: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-5.0..5.0_f32)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let a = &anchors[i % n_clusters];
            (0..dim).map(|d| a[d] + rng.gen_range(-0.6..0.6_f32)).collect()
        })
        .collect()
}

fn measure(assignment: Assignment, db: &[Vec<f32>], queries: &[Vec<f32>], k_centroids: usize, n_probe: usize) -> f32 {
    let idx = IvfIndex::build(db.to_vec(), k_centroids, assignment, 42).unwrap();
    let mut s = 0.0_f32;
    for q in queries {
        let truth = brute_force_topk(db, q, 10);
        let got = idx.search(q, 10, n_probe);
        s += recall(&got, &truth);
    }
    s / queries.len() as f32
}

#[test]
fn soar_beats_or_matches_single_at_equal_probe() {
    let db = synth(4_000, 32, 40, 7);
    let queries: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(100 + i as u64);
            (0..32).map(|_| rng.gen_range(-5.0..5.0_f32)).collect()
        })
        .collect();

    let r_single = measure(Assignment::Single, &db, &queries, 32, 3);
    let r_soar = measure(Assignment::Soar { lambda: 1.5 }, &db, &queries, 32, 3);

    // SOAR pays 2x posting storage, so it should never lose to Single
    // at the same n_probe on this clustered workload.
    assert!(
        r_soar >= r_single - 0.02,
        "SOAR recall {} < Single recall {} at equal n_probe",
        r_soar,
        r_single
    );
}

#[test]
fn soar_orthogonalizes_more_than_spillover() {
    let db = synth(3_000, 32, 30, 11);
    let idx_sp = IvfIndex::build(db.clone(), 24, Assignment::Spillover, 99).unwrap();
    let idx_so = IvfIndex::build(db.clone(), 24, Assignment::Soar { lambda: 2.0 }, 99).unwrap();
    let c_sp = idx_sp.mean_residual_correlation().unwrap();
    let c_so = idx_so.mean_residual_correlation().unwrap();
    // SOAR should produce lower (more orthogonal / more anti-correlated) residual cosine.
    assert!(
        c_so <= c_sp + 1e-3,
        "SOAR residual corr {} not <= Spillover {}",
        c_so,
        c_sp
    );
}

#[test]
fn replication_factors_match_assignment() {
    let db = synth(500, 16, 8, 1);
    let idx_s = IvfIndex::build(db.clone(), 16, Assignment::Single, 1).unwrap();
    let idx_p = IvfIndex::build(db.clone(), 16, Assignment::Spillover, 1).unwrap();
    let idx_o = IvfIndex::build(db.clone(), 16, Assignment::Soar { lambda: 1.0 }, 1).unwrap();
    assert_eq!(idx_s.posting_entries(), 500);
    assert_eq!(idx_p.posting_entries(), 1000);
    assert_eq!(idx_o.posting_entries(), 1000);
}

#[test]
fn search_returns_sorted_unique_topk() {
    let db = synth(800, 24, 10, 3);
    let idx = IvfIndex::build(db.clone(), 16, Assignment::Soar { lambda: 1.0 }, 5).unwrap();
    let q = db[7].clone();
    let res = idx.search(&q, 10, 4);
    assert!(res.len() <= 10);
    // sorted ascending
    for w in res.windows(2) {
        assert!(w[0].1 <= w[1].1, "result not sorted");
    }
    // unique ids
    let mut ids: Vec<u32> = res.iter().map(|(i, _)| *i).collect();
    ids.sort();
    let n = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), n, "duplicate ids in search result");
    // exact-match query: id 7 should be in result with d≈0
    assert!(res.iter().any(|(i, d)| *i == 7 && *d < 1e-5));
}
