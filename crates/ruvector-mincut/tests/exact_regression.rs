//! Independent exhaustive-cut oracle for the public dynamic solver.
use ruvector_mincut::{DynamicGraph, DynamicMinCut, MinCutBuilder, MinCutConfig};
use std::collections::HashSet;

fn check(solver: &DynamicMinCut) {
    let shared = solver.graph();
    let graph = shared.read();
    let mut vertices = graph.vertices();
    vertices.sort_unstable();
    let edges = graph.edges();
    let mut expected = f64::INFINITY;
    for mask in 1..(1usize << vertices.len()) - 1 {
        let side: HashSet<_> = vertices.iter().enumerate()
            .filter(|(i, _)| mask & (1usize << *i) != 0).map(|(_, &v)| v).collect();
        let value: f64 = edges.iter()
            .filter(|e| side.contains(&e.source) != side.contains(&e.target))
            .map(|e| e.weight).sum();
        expected = expected.min(value);
    }
    drop(graph);
    let result = solver.min_cut();
    assert_eq!(result.value, expected, "graph: {edges:?}");
    let (s, t) = result.partition.unwrap();
    let s: HashSet<_> = s.into_iter().collect();
    let t: HashSet<_> = t.into_iter().collect();
    assert!(s.is_disjoint(&t));
    assert_eq!(s.union(&t).copied().collect::<HashSet<_>>(), vertices.iter().copied().collect());
    if vertices.len() >= 2 {
        assert!(!s.is_empty() && !t.is_empty());
        let crossing: HashSet<_> = edges.iter()
            .filter(|e| s.contains(&e.source) != s.contains(&e.target)).map(|e| e.id).collect();
        let witness = result.cut_edges.unwrap();
        assert_eq!(witness.iter().map(|e| e.id).collect::<HashSet<_>>(), crossing);
        assert_eq!(witness.iter().map(|e| e.weight).sum::<f64>(), result.value);
    }
}

#[test]
fn all_five_vertex_graphs_match_exhaustive_oracle() {
    let pairs: Vec<_> = (0..5).flat_map(|u| (u + 1..5).map(move |v| (u, v))).collect();
    for topology in 0..1usize << pairs.len() {
        let graph = DynamicGraph::new();
        for v in 0..5 { graph.add_vertex(v); }
        for (i, &(u, v)) in pairs.iter().enumerate() {
            if topology & (1 << i) != 0 {
                graph.insert_edge(u, v, (i % 4 + 1) as f64 / 2.0).unwrap();
            }
        }
        check(&DynamicMinCut::from_graph(graph, MinCutConfig::default()).unwrap());
    }
}

#[test]
fn mixed_updates_match_oracle_after_every_mutation() {
    let mut solver = MinCutBuilder::new().build().unwrap();
    for u in 0..6 {
        for v in u + 1..6 {
            solver.insert_edge(u * 100 + 7, v * 100 + 7, ((u + v) % 5) as f64 / 2.0).unwrap();
            check(&solver);
        }
    }
    for u in (0..6).rev() {
        for v in u + 1..6 {
            solver.delete_edge(v * 100 + 7, u * 100 + 7).unwrap();
            check(&solver);
        }
    }
}

#[test]
fn rejected_weights_do_not_mutate_graph_or_solver() {
    let mut solver = MinCutBuilder::new().with_edges(vec![(1, 2, 1.0)]).build().unwrap();
    for weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(solver.insert_edge(3, 4, weight).is_err());
        assert_eq!(solver.num_vertices(), 2);
        assert_eq!(solver.num_edges(), 1);
        assert_eq!(solver.stats().insertions, 0);
        assert!(solver.graph().read().update_edge_weight(1, 2, weight).is_err());
        check(&solver);
    }
}

#[test]
fn partitions_are_stable_across_input_order() {
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)];
    let a = MinCutBuilder::new().with_edges(edges.clone()).build().unwrap();
    let b = MinCutBuilder::new().with_edges(edges.into_iter().rev().collect()).build().unwrap();
    assert_eq!(a.partition(), b.partition());
    check(&a);
    check(&b);
}

#[test]
fn internal_insertion_skips_full_recomputation() {
    let mut solver = MinCutBuilder::new().with_edges(vec![
        (0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0), (3, 0, 10.0),
        (3, 4, 1.0),
    ]).build().unwrap();
    let before = solver.stats().restructures;
    solver.insert_edge(0, 2, 10.0).unwrap();
    assert_eq!(solver.stats().restructures, before);
    check(&solver);
    solver.insert_edge(4, 5, 0.5).unwrap();
    assert!(solver.stats().restructures > before);
    check(&solver);
}
