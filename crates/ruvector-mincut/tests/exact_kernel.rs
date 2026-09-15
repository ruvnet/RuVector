//! Can also run without workspace dependencies:
//! rustc --edition=2021 --test tests/exact_kernel.rs -o /tmp/mincut-kernel-tests
#[path = "../src/algorithm/exact.rs"]
mod exact;

fn oracle(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
    (1..(1usize << n) - 1)
        .map(|mask| {
            edges
                .iter()
                .filter(|&&(u, v, _)| ((mask >> u) & 1) != ((mask >> v) & 1))
                .map(|&(_, _, w)| w)
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min)
}

fn check(n: usize, edges: &[(usize, usize, f64)]) {
    let (value, side) = exact::minimum_cut(n, edges);
    assert_eq!(value, oracle(n, edges), "n={n}, edges={edges:?}");
    if n >= 2 {
        assert!(!side.is_empty() && side.len() < n);
        let crossing: f64 = edges
            .iter()
            .filter(|&&(u, v, _)| side.contains(&u) != side.contains(&v))
            .map(|&(_, _, w)| w)
            .sum();
        assert_eq!(value, crossing);
    }
}

#[test]
fn all_six_vertex_topologies() {
    let pairs: Vec<_> = (0..6)
        .flat_map(|u| (u + 1..6).map(move |v| (u, v)))
        .collect();
    for topology in 0..1usize << pairs.len() {
        let edges: Vec<_> = pairs
            .iter()
            .enumerate()
            .filter(|(i, _)| topology & (1usize << *i) != 0)
            .map(|(i, &(u, v))| (u, v, (i % 5) as f64 / 2.0))
            .collect();
        check(6, &edges);
    }
}

#[test]
fn trivial_and_large_sparse_graphs() {
    check(0, &[]);
    check(1, &[]);
    check(2, &[]);
    let path: Vec<_> = (0..9999).map(|i| (i, i + 1, 1.0)).collect();
    assert_eq!(exact::minimum_cut(10000, &path).0, 1.0);
    assert_eq!(exact::minimum_cut(10001, &path).0, 0.0);
}

#[test]
fn weighted_bridge_certificate_requires_the_lower_bound() {
    // A heavy bridge must not hide a lighter cut inside a triangle.
    let edges = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 2, 1.0),
        (2, 3, 9.0),
        (3, 4, 4.0),
        (4, 5, 4.0),
        (3, 5, 4.0),
    ];
    let (value, _) = exact::minimum_cut(6, &edges);
    assert_eq!(value, 2.0);
    let mut light_bridge = edges.clone();
    light_bridge[3].2 = 1.5;
    assert_eq!(exact::minimum_cut(6, &light_bridge).0, 1.5);
}

#[test]
fn long_cycle_certificate_does_not_use_recursive_dfs() {
    let n = 100_000;
    let edges: Vec<_> = (0..n).map(|v| (v, (v + 1) % n, 1.0)).collect();
    let (value, side) = exact::minimum_cut(n, &edges);
    assert_eq!(value, 2.0);
    assert_eq!(side.len(), 1);
}
