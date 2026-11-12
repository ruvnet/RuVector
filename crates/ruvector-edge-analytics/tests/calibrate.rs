//! Calibration probe (ignored): prints wall time per estimate unit.
mod common;
use ruvector_edge_analytics::*;
use std::time::Instant;

#[test]
#[ignore]
fn calibrate() {
    for (name, edges) in [
        ("sparse10k/50k", common::sparse_graph(10_000, 50_000, 1)),
        ("sparse50k/200k", common::sparse_graph(50_000, 200_000, 2)),
        ("clusters60/1500", common::two_clusters(60, 1_500, 4, 3)),
        ("clusters150/10k", common::two_clusters(150, 10_000, 4, 3)),
        ("clusters250/50k", common::two_clusters(250, 50_000, 5, 4)),
    ] {
        let g = TenantGraph::from_edges([1; 16], 1, &edges, &GraphLimits::JOB).unwrap();
        let est = cost::estimate_exact(&cost::precheck(&g), g.edge_count());
        let t = Instant::now();
        let big = Profile {
            limits: GraphLimits::JOB,
            budget: Budget {
                max_work: u64::MAX,
                max_memory_bytes: u64::MAX,
            },
        };
        let r = query(&g, &QueryMode::Exact, &big).unwrap();
        let ns = t.elapsed().as_nanos() as f64;
        println!(
            "{name}: path={:?} work={} ns={} ns/unit={:.3} value={:?}",
            est.path,
            est.work,
            ns,
            ns / est.work as f64,
            r.value
        );
    }
}

#[test]
#[ignore]
fn calibrate_sparse_sw() {
    for (n, m) in [
        (500u64, 1_500usize),
        (2_000, 6_000),
        (4_000, 12_000),
        (2_000, 20_000),
    ] {
        let edges = common::ring_chords(n, m, 5);
        let g = TenantGraph::from_edges([1; 16], 1, &edges, &GraphLimits::JOB).unwrap();
        let est = cost::estimate_exact(&cost::precheck(&g), g.edge_count());
        let big = Profile {
            limits: GraphLimits::JOB,
            budget: Budget {
                max_work: u64::MAX,
                max_memory_bytes: u64::MAX,
            },
        };
        let t = Instant::now();
        let r = query(&g, &QueryMode::Exact, &big).unwrap();
        let ns = t.elapsed().as_nanos() as f64;
        println!(
            "ring n={n} m={m}: path={:?} work={} ns={} ns/unit={:.3} value={:?}",
            est.path,
            est.work,
            ns,
            ns / est.work as f64,
            r.value
        );
    }
}
