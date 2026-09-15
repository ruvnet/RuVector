//! Reproducible public-API timings. Run with --release; compare the same machine.
use ruvector_mincut::MinCutBuilder;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    for n in [32u64, 128] {
        let edges: Vec<_> = (0..n - 1).map(|u| (u, u + 1, 1.0)).collect();
        let start = Instant::now();
        for _ in 0..20 {
            let graph = MinCutBuilder::new().with_edges(edges.clone()).build().unwrap();
            assert_eq!(graph.min_cut_value(), 1.0);
            black_box(graph);
        }
        println!("tree_build n={n} iterations=20 mean_us={:.3}", start.elapsed().as_secs_f64() * 1e6 / 20.0);
    }
    // Vertex 63 is separated by a unique weak bridge. All added edges are
    // between existing vertices on the other side of this cut.
    let mut edges: Vec<_> = (0..62).map(|u| (u, u + 1, 10.0)).collect();
    edges.push((62, 63, 0.125));
    let inserts: Vec<_> = (0..20).flat_map(|u| (u + 2..40).map(move |v| (u, v, 10.0))).collect();
    let mut solver = MinCutBuilder::new().with_edges(edges).build().unwrap();
    let start = Instant::now();
    for &(u, v, w) in &inserts {
        assert_eq!(solver.insert_edge(u, v, w).unwrap(), 0.125);
    }
    println!("preserving_inserts n=64 operations={} mean_us={:.3}", inserts.len(), start.elapsed().as_secs_f64() * 1e6 / inserts.len() as f64);
    let start = Instant::now();
    for _ in 0..1000 { black_box(solver.min_cut()); }
    println!("full_result n=64 iterations=1000 mean_us={:.3}", start.elapsed().as_secs_f64() * 1e6 / 1000.0);
    let start = Instant::now();
    for &(u, v, _) in inserts.iter().take(20) {
        assert_eq!(solver.delete_edge(u, v).unwrap(), 0.125);
    }
    println!("deletions n=64 operations=20 mean_us={:.3}", start.elapsed().as_secs_f64() * 1e6 / 20.0);
}
