//! Drift guard: the cost model's certificate predicate must agree with the
//! solver's branch. Where it says "certified", the solver's answer equals
//! the certificate's implied value; where it says Stoer-Wagner, the
//! certificate's conditions really fail (checked independently here).
mod common;
use common::Mix;
use ruvector_edge_analytics::*;

fn random_graph(r: &mut Mix) -> Vec<(u64, u64, f64)> {
    let n = 3 + r.below(40);
    let m = (n - 1 + r.below(n * 3)) as usize;
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Sometimes a spanning path, sometimes not (disconnected cases).
    if r.below(4) != 0 {
        for i in 1..n {
            seen.insert((i - 1, i));
            edges.push((i - 1, i, (r.below(5) + 1) as f64));
        }
    }
    let mut tries = 0;
    while edges.len() < m && tries < 10 * m {
        tries += 1;
        let (a, b) = (r.below(n), r.below(n));
        let k = (a.min(b), a.max(b));
        if a != b && seen.insert(k) {
            // Mix unit, integer and fractional weights.
            let w = match r.below(3) {
                0 => 1.0,
                1 => (r.below(6) + 1) as f64,
                _ => (r.below(1000) + 1) as f64 / 7.0,
            };
            edges.push((k.0, k.1, w));
        }
    }
    edges
}

#[test]
fn certificate_prediction_matches_solver() {
    let mut r = Mix(0xd71f7);
    let (mut certified, mut general) = (0, 0);
    for _ in 0..400 {
        let edges = random_graph(&mut r);
        let g = TenantGraph::from_edges([0; 16], 1, &edges, &GraphLimits::INLINE).unwrap();
        let pre = cost::precheck(&g);
        let (value, _, _) = common::native_exact(&edges);
        if pre.certified {
            certified += 1;
            let implied = pre.certified_value.unwrap();
            // The solver re-sums crossing edges; allow float reassociation.
            assert!(
                (implied - value).abs() <= 1e-9 * value.max(1.0),
                "{implied} vs {value}"
            );
        } else {
            general += 1;
            // Not certified => connected, and (since a bridge <= 2 x lightest
            // would certify) min weighted degree > 2 x lightest edge.
            let mut deg = std::collections::HashMap::new();
            let mut lightest = f64::INFINITY;
            for &(u, v, w) in &edges {
                *deg.entry(u).or_insert(0.0) += w;
                *deg.entry(v).or_insert(0.0) += w;
                lightest = lightest.min(w);
            }
            let min_deg = deg.values().copied().fold(f64::INFINITY, f64::min);
            assert!(value > 0.0, "a disconnected graph must be certified");
            assert!(min_deg > 2.0 * lightest - 1e-9, "{min_deg} vs {lightest}");
            assert!(value <= min_deg + 1e-9);
        }
    }
    assert!(
        certified > 50 && general > 50,
        "coverage: {certified} certified / {general} general"
    );
}
