//! Accuracy-retention evaluation — the graph-condensation field's standard
//! protocol: train a GNN on the **condensed** graph, test it on the **original**
//! graph's held-out nodes, and report `accuracy(condensed) / accuracy(full)`.
//!
//! Run: `cargo run --release -p ruvector-graph-condense --example accuracy_eval`
//!
//! Honest scope: this runs on a *controlled synthetic* node-classification task
//! (planted communities as classes, noisy features so the graph actually
//! matters), not the canonical Cora/Citeseer benchmarks — so it is a
//! substantiated *retention* measurement, not a literal "beats GCond on Cora"
//! claim. It closes the gap of having no learning-accuracy validation at all.

#![allow(clippy::needless_range_loop)] // index-heavy numeric example code

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_graph_condense::gnn_eval::{accuracy, Gcn, GcnConfig, GcnGraph};
use ruvector_graph_condense::{
    CondenseConfig, CondenseMethod, CondensedGraph, DiffCutConfig, GraphCondenser, NodeFeatures,
};
use ruvector_mincut::DynamicGraph;

struct Task {
    classes: usize,
    per_class: usize, // nodes per class
    dim: usize,
    p_intra: f64,
    p_inter: f64,
    noise: f64,
    seed: u64,
}

impl Task {
    fn n(&self) -> usize {
        self.classes * self.per_class
    }

    /// Build graph + features + per-node class labels. Node `i` has class
    /// `i / per_class`; features are a class centroid + Gaussian-ish noise so
    /// raw features overlap and the graph carries real signal.
    fn generate(&self) -> (DynamicGraph, NodeFeatures, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let g = DynamicGraph::new();
        let mut f = NodeFeatures::new(self.dim, self.classes);
        let mut labels = vec![0usize; self.n()];

        let centroids: Vec<Vec<f64>> = (0..self.classes)
            .map(|c| {
                (0..self.dim)
                    .map(|d| if d % self.classes == c { 1.5 } else { 0.0 })
                    .collect()
            })
            .collect();

        for i in 0..self.n() {
            let cls = i / self.per_class;
            labels[i] = cls;
            let emb: Vec<f32> = (0..self.dim)
                .map(|d| (centroids[cls][d] + self.noise * rng.gen_range(-1.0..1.0)) as f32)
                .collect();
            f.set(i as u64, emb, cls).unwrap();
            g.add_vertex(i as u64);
        }
        for a in 0..self.n() {
            for b in (a + 1)..self.n() {
                let same = a / self.per_class == b / self.per_class;
                let p = if same { self.p_intra } else { self.p_inter };
                if rng.gen_bool(p) {
                    let _ = g.insert_edge(a as u64, b as u64, 1.0);
                }
            }
        }
        (g, f, labels)
    }
}

/// Extract contiguous `0..n` edge list / feature matrix from the graph.
fn full_arrays(
    g: &DynamicGraph,
    f: &NodeFeatures,
    n: usize,
) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    let edges = g
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
    (edges, x)
}

/// Build the GCN training arrays for a condensed graph: centroids as features,
/// dominant class as label, super-edges as adjacency.
fn condensed_arrays(c: &CondensedGraph) -> (GcnGraph, Vec<f64>, Vec<usize>) {
    let cn = c.node_count();
    let dim = c.dim;
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

fn split(n: usize, train_frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut tr, mut te) = (Vec::new(), Vec::new());
    for i in 0..n {
        if rng.gen_bool(train_frac) {
            tr.push(i);
        } else {
            te.push(i);
        }
    }
    (tr, te)
}

fn main() {
    let task = Task {
        classes: 6,
        per_class: 60,
        dim: 24,
        p_intra: 0.12,
        p_inter: 0.004,
        noise: 1.4,
        seed: 2026,
    };
    let n = task.n();
    let (graph, feats, labels) = task.generate();
    let (full_edges, x_full) = full_arrays(&graph, &feats, n);
    let full_graph = GcnGraph::from_edges(n, &full_edges);
    let (train, test) = split(n, 0.6, 7);
    let cfg = GcnConfig::default();

    println!(
        "Task: {} nodes, {} classes, {} edges, dim {}, noise {} (features overlap; graph matters)",
        n,
        task.classes,
        graph.num_edges(),
        task.dim,
        task.noise
    );
    println!("Protocol: train GNN on condensed graph -> test on original held-out nodes.\n");

    // Baseline: train on the FULL graph's train split.
    let base = Gcn::train(
        &cfg,
        &full_graph,
        &x_full,
        task.dim,
        &labels,
        task.classes,
        &train,
    );
    let acc_full = accuracy(&base.predict(&full_graph, &x_full), &labels, &test);
    println!(
        "Baseline GNN (trained on full graph): test accuracy {:.1}%\n",
        acc_full * 100.0
    );

    for (name, method) in [
        (
            "WeakBoundary",
            CondenseMethod::WeakBoundary {
                relative_threshold: 0.5,
            },
        ),
        (
            "DiffMinCut",
            CondenseMethod::DiffMinCut(DiffCutConfig {
                num_clusters: task.classes * 3, // a few super-nodes per class -> more GNN training signal
                restarts: 3,
                iterations: 500,
                ..Default::default()
            }),
        ),
    ] {
        let c = GraphCondenser::new(CondenseConfig {
            method,
            normalize_centroids: false,
        })
        .condense(&graph, &feats)
        .unwrap();
        let (cg, x_cond, lab_cond) = condensed_arrays(&c);
        let all: Vec<usize> = (0..c.node_count()).collect();
        // Train on condensed, test on the ORIGINAL graph's test split.
        let model = Gcn::train(&cfg, &cg, &x_cond, task.dim, &lab_cond, task.classes, &all);
        let acc_cond = accuracy(&model.predict(&full_graph, &x_full), &labels, &test);
        let retention = if acc_full > 0.0 {
            acc_cond / acc_full
        } else {
            0.0
        };
        println!(
            "{name:>12}: {} -> {} super-nodes ({:.0}x)  | test acc {:.1}%  | retention {:.1}%",
            n,
            c.node_count(),
            c.node_reduction_ratio(),
            acc_cond * 100.0,
            retention * 100.0,
        );
    }

    println!(
        "\nRetention near 100% means a GNN trained on the tiny condensed graph classifies the\n\
         original's held-out nodes about as well as one trained on the full graph — the field's\n\
         core success criterion, here measured on controlled synthetic data."
    );
}
