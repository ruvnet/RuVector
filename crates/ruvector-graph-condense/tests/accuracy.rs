//! End-to-end accuracy-retention test: a GNN trained on the condensed graph must
//! classify the original graph's held-out nodes nearly as well as one trained on
//! the full graph. This is the graph-condensation field's core success metric.
#![allow(clippy::needless_range_loop)] // index-heavy numeric test code

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_graph_condense::gnn_eval::{accuracy, Gcn, GcnConfig, GcnGraph};
use ruvector_graph_condense::{
    CondenseConfig, CondenseMethod, CondensedGraph, DiffCutConfig, GraphCondenser, NodeFeatures,
};
use ruvector_mincut::DynamicGraph;

fn gen(
    classes: usize,
    per_class: usize,
    dim: usize,
    noise: f64,
    seed: u64,
) -> (DynamicGraph, NodeFeatures, Vec<usize>, usize) {
    let n = classes * per_class;
    let mut rng = StdRng::seed_from_u64(seed);
    let g = DynamicGraph::new();
    let mut f = NodeFeatures::new(dim, classes);
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let cls = i / per_class;
        labels[i] = cls;
        let emb: Vec<f32> = (0..dim)
            .map(|d| {
                let base = if d % classes == cls { 1.5 } else { 0.0 };
                (base + noise * rng.gen_range(-1.0..1.0)) as f32
            })
            .collect();
        f.set(i as u64, emb, cls).unwrap();
        g.add_vertex(i as u64);
    }
    for a in 0..n {
        for b in (a + 1)..n {
            let same = a / per_class == b / per_class;
            let p = if same { 0.15 } else { 0.005 };
            if rng.gen_bool(p) {
                let _ = g.insert_edge(a as u64, b as u64, 1.0);
            }
        }
    }
    (g, f, labels, n)
}

fn full_arrays(g: &DynamicGraph, f: &NodeFeatures, n: usize) -> (GcnGraph, Vec<f64>) {
    let edges: Vec<(usize, usize, f64)> = g
        .edges()
        .iter()
        .map(|e| (e.source as usize, e.target as usize, e.weight))
        .collect();
    let dim = f.dim();
    let mut x = vec![0f64; n * dim];
    for i in 0..n {
        if let Some(emb) = f.embedding(i as u64) {
            for d in 0..dim {
                x[i * dim + d] = emb[d] as f64;
            }
        }
    }
    (GcnGraph::from_edges(n, &edges), x)
}

fn condensed_arrays(c: &CondensedGraph) -> (GcnGraph, Vec<f64>, Vec<usize>) {
    let (cn, dim) = (c.node_count(), c.dim);
    let mut x = vec![0f64; cn * dim];
    let mut labels = vec![0usize; cn];
    for (i, node) in c.nodes.iter().enumerate() {
        for d in 0..dim {
            x[i * dim + d] = node.centroid[d] as f64;
        }
        labels[i] = node.dominant_class().unwrap_or(0);
    }
    let edges: Vec<(usize, usize, f64)> = c
        .edges
        .iter()
        .map(|e| (e.source as usize, e.target as usize, e.weight))
        .collect();
    (GcnGraph::from_edges(cn, &edges), x, labels)
}

#[test]
fn condensed_graph_trains_a_usable_classifier() {
    let classes = 3;
    let (g, f, labels, n) = gen(classes, 24, 12, 1.2, 2026);
    let (full, x_full) = full_arrays(&g, &f, n);

    // Train/test split.
    let mut rng = StdRng::seed_from_u64(7);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for i in 0..n {
        if rng.gen_bool(0.6) {
            train.push(i);
        } else {
            test.push(i);
        }
    }

    let cfg = GcnConfig {
        epochs: 150,
        ..Default::default()
    };
    let base = Gcn::train(&cfg, &full, &x_full, f.dim(), &labels, classes, &train);
    let acc_full = accuracy(&base.predict(&full, &x_full), &labels, &test);
    assert!(
        acc_full > 0.7,
        "baseline too weak to be a fair test: {acc_full}"
    );

    // Condense (DiffMinCut, a few super-nodes per class) and train on it.
    let c = GraphCondenser::new(CondenseConfig {
        method: CondenseMethod::DiffMinCut(DiffCutConfig {
            num_clusters: classes * 3,
            restarts: 2,
            iterations: 300,
            ..Default::default()
        }),
        normalize_centroids: false,
    })
    .condense(&g, &f)
    .unwrap();
    let (cg, x_cond, lab_cond) = condensed_arrays(&c);
    let all: Vec<usize> = (0..c.node_count()).collect();
    let model = Gcn::train(&cfg, &cg, &x_cond, f.dim(), &lab_cond, classes, &all);
    let acc_cond = accuracy(&model.predict(&full, &x_full), &labels, &test);

    let retention = acc_cond / acc_full;
    assert!(
        c.node_count() < n / 4,
        "expected real reduction, got {} of {n}",
        c.node_count()
    );
    assert!(
        retention > 0.8,
        "retention too low: cond {acc_cond:.3} / full {acc_full:.3} = {retention:.3}"
    );
}
