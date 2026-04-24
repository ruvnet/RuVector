//! QAOA-based Graph-Cut Solver vs Classical Edmonds-Karp Mincut
//!
//! Demonstrates a quantum-classical hybrid approach to the graph-cut coherence
//! gating problem used throughout the RuVector pipeline. The classical approach
//! models the problem as an s-t mincut on a flow network; this example instead
//! encodes the binary labelling problem as a QAOA MaxCut instance and compares
//! solution quality.
//!
//! ## Problem mapping
//!
//! In the graph-cut formulation each node i has a unary cost lambda_i:
//!   - lambda_i > 0  => evidence for label 1 (foreground)
//!   - lambda_i < 0  => evidence for label 0 (background)
//!
//! Pairwise edges (i,j) with weight w_ij penalize assigning different labels
//! to neighboring nodes (smoothness).
//!
//! The s-t mincut solver finds the labelling that minimizes:
//!   sum_i  max(-lambda_i, 0) * x_i  +  max(lambda_i, 0) * (1-x_i)
//!   + sum_{(i,j)} w_ij * |x_i - x_j|
//!
//! We re-encode this as a QAOA MaxCut problem on an augmented graph:
//!   - Auxiliary source (s) and sink (t) nodes are added
//!   - s-i edge with weight |lambda_i| when lambda_i > 0 (prefers i in cut)
//!   - i-t edge with weight |lambda_i| when lambda_i < 0
//!   - Original pairwise edges carry their smoothness weight
//!   - MaxCut on this augmented graph finds the partition that maximizes
//!     total crossing weight, which corresponds to the mincut labelling
//!
//! Because QAOA simulates 2^n amplitudes, we use a small graph (12 nodes)
//! to keep runtime tractable.
//!
//! Run: cargo run --example qaoa_graphcut --release

use ruqu_algorithms::qaoa::{run_qaoa, Graph, QaoaConfig};
use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// LCG deterministic RNG (same as other graph-cut examples)
// ---------------------------------------------------------------------------

fn lcg_next(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}
fn lcg_f64(s: &mut u64) -> f64 {
    lcg_next(s);
    (*s >> 11) as f64 / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_f64(s).max(1e-15);
    let u2 = lcg_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Classical Edmonds-Karp s-t mincut (reference solver)
// ---------------------------------------------------------------------------

/// Solve the graph-cut labelling problem using classical BFS max-flow.
///
/// `lambda[i]` = unary cost for node i (positive => foreground evidence).
/// `edges` = list of (i, j, weight) pairwise smoothness penalties.
/// `num_nodes` = number of problem nodes (excluding s,t).
///
/// Returns a boolean labelling: true = foreground (source-side of cut).
fn solve_mincut_classical(
    lambda: &[f64],
    edges: &[(usize, usize, f64)],
    num_nodes: usize,
) -> Vec<bool> {
    let (s, t, n) = (num_nodes, num_nodes + 1, num_nodes + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();

    let add_edge =
        |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>, u: usize, v: usize, cap: f64| {
            let i = caps.len();
            caps.push(cap);
            caps.push(0.0);
            adj[u].push((v, i));
            adj[v].push((u, i + 1));
        };

    // Unary edges: source/sink connections based on lambda sign
    for i in 0..num_nodes {
        let (p0, p1) = (lambda[i].max(0.0), (-lambda[i]).max(0.0));
        if p0 > 1e-12 {
            add_edge(&mut adj, &mut caps, s, i, p0);
        }
        if p1 > 1e-12 {
            add_edge(&mut adj, &mut caps, i, t, p1);
        }
    }

    // Pairwise edges (bidirectional)
    for &(u, v, w) in edges {
        if w > 1e-12 {
            add_edge(&mut adj, &mut caps, u, v, w);
            add_edge(&mut adj, &mut caps, v, u, w);
        }
    }

    // BFS augmenting-path max-flow
    loop {
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = VecDeque::new();
        vis[s] = true;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t {
                break;
            }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 {
                    vis[v] = true;
                    parent[v] = Some((u, ei));
                    q.push_back(v);
                }
            }
        }
        if !vis[t] {
            break;
        }
        let mut bn = f64::MAX;
        let mut v = t;
        while let Some((u, ei)) = parent[v] {
            bn = bn.min(caps[ei]);
            v = u;
        }
        v = t;
        while let Some((u, ei)) = parent[v] {
            caps[ei] -= bn;
            caps[ei ^ 1] += bn;
            v = u;
        }
    }

    // Source-side reachability = foreground labels
    let mut reach = vec![false; n];
    let mut stk = vec![s];
    reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] {
            if !reach[v] && caps[ei] > 1e-15 {
                reach[v] = true;
                stk.push(v);
            }
        }
    }
    (0..num_nodes).map(|i| reach[i]).collect()
}

/// Compute the total graph-cut energy for a given labelling.
fn graphcut_energy(lambda: &[f64], edges: &[(usize, usize, f64)], labels: &[bool]) -> f64 {
    let mut energy = 0.0;
    // Unary terms
    for (i, &lam) in lambda.iter().enumerate() {
        if labels[i] {
            // Foreground: pay cost for wrong-side unary when lambda < 0
            energy += (-lam).max(0.0);
        } else {
            // Background: pay cost when lambda > 0
            energy += lam.max(0.0);
        }
    }
    // Pairwise terms: penalize label disagreement
    for &(u, v, w) in edges {
        if labels[u] != labels[v] {
            energy += w;
        }
    }
    energy
}

// ---------------------------------------------------------------------------
// QAOA encoding of graph-cut as MaxCut
// ---------------------------------------------------------------------------

/// Encode the graph-cut problem as a QAOA MaxCut instance.
///
/// The key insight: minimizing the graph-cut energy is equivalent to finding
/// a maximum weight cut on an augmented graph that includes source and sink
/// auxiliary nodes.
///
/// We construct an augmented graph with num_nodes + 2 vertices:
///   - Nodes 0..num_nodes are the original problem nodes
///   - Node num_nodes = source (s), always in partition 0
///   - Node num_nodes+1 = sink (t), always in partition 1
///
/// Edge encoding:
///   - For lambda_i > 0: edge (s, i) with weight lambda_i
///     (cutting this edge means i is in partition 1 = foreground, which is
///     correct since positive lambda means foreground evidence)
///   - For lambda_i < 0: edge (i, t) with weight |lambda_i|
///     (cutting this edge means i is NOT with t = i is foreground, but we
///     want i to be background, so NOT cutting is correct)
///   - Pairwise edges (i, j, w) are added directly
///
/// The MaxCut solution on this augmented graph gives us the optimal labelling.
/// Nodes on the same side as s are background; nodes on the opposite side
/// are foreground.
fn encode_graphcut_as_maxcut(
    lambda: &[f64],
    edges: &[(usize, usize, f64)],
    num_nodes: usize,
) -> Graph {
    let total_nodes = num_nodes + 2;
    let s = num_nodes as u32;
    let t = (num_nodes + 1) as u32;

    let mut graph = Graph::new(total_nodes as u32);

    // Unary terms as source/sink edges
    for (i, &lam) in lambda.iter().enumerate() {
        let abs_lam = lam.abs();
        if abs_lam < 1e-12 {
            continue;
        }
        if lam > 0.0 {
            // Positive lambda => foreground evidence => s-i edge
            graph.add_edge(s, i as u32, abs_lam);
        } else {
            // Negative lambda => background evidence => i-t edge
            graph.add_edge(i as u32, t, abs_lam);
        }
    }

    // Pairwise smoothness edges
    for &(u, v, w) in edges {
        if w > 1e-12 {
            graph.add_edge(u as u32, v as u32, w);
        }
    }

    graph
}

/// Extract the graph-cut labelling from a QAOA MaxCut bitstring.
///
/// The source node (s) defines the "background" partition. Any problem node
/// on the opposite side of s is labelled foreground (true).
fn extract_labels_from_maxcut(bitstring: &[bool], num_nodes: usize) -> Vec<bool> {
    let s_partition = bitstring[num_nodes]; // source node partition
    (0..num_nodes)
        .map(|i| bitstring[i] != s_partition) // opposite side of s = foreground
        .collect()
}

// ---------------------------------------------------------------------------
// Synthetic test-case generation
// ---------------------------------------------------------------------------

/// Generate a synthetic graph-cut problem instance.
///
/// Simulates a 1D signal with an embedded anomaly region (nodes fg_start..fg_end
/// have positive lambda, others negative). Pairwise edges connect consecutive
/// nodes with smoothness weight gamma.
struct TestCase {
    num_nodes: usize,
    lambda: Vec<f64>,
    edges: Vec<(usize, usize, f64)>,
    ground_truth: Vec<bool>,
    name: &'static str,
}

fn make_1d_chain(
    num_nodes: usize,
    fg_start: usize,
    fg_end: usize,
    signal_strength: f64,
    gamma: f64,
    noise_sigma: f64,
    seed: u64,
    name: &'static str,
) -> TestCase {
    let mut rng = seed;
    let mut lambda = Vec::with_capacity(num_nodes);
    let mut ground_truth = Vec::with_capacity(num_nodes);

    for i in 0..num_nodes {
        let is_fg = i >= fg_start && i < fg_end;
        ground_truth.push(is_fg);
        let base = if is_fg {
            signal_strength
        } else {
            -signal_strength
        };
        lambda.push(base + lcg_normal(&mut rng) * noise_sigma);
    }

    // Chain edges: i -- i+1
    let mut edges = Vec::with_capacity(num_nodes - 1);
    for i in 0..(num_nodes - 1) {
        edges.push((i, i + 1, gamma));
    }

    TestCase {
        num_nodes,
        lambda,
        edges,
        ground_truth,
        name,
    }
}

fn make_2d_grid(width: usize, height: usize, seed: u64, name: &'static str) -> TestCase {
    let num_nodes = width * height;
    let mut rng = seed;
    let mut lambda = Vec::with_capacity(num_nodes);
    let mut ground_truth = Vec::with_capacity(num_nodes);

    // Create a rectangular foreground region in the center
    let (fx0, fx1) = (width / 4, 3 * width / 4);
    let (fy0, fy1) = (height / 4, 3 * height / 4);

    for y in 0..height {
        for x in 0..width {
            let is_fg = x >= fx0 && x < fx1 && y >= fy0 && y < fy1;
            ground_truth.push(is_fg);
            let base = if is_fg { 2.0 } else { -1.5 };
            lambda.push(base + lcg_normal(&mut rng) * 0.8);
        }
    }

    // 4-connected grid edges
    let mut edges = Vec::new();
    let gamma = 0.5;
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if x + 1 < width {
                edges.push((idx, idx + 1, gamma));
            }
            if y + 1 < height {
                edges.push((idx, idx + width, gamma));
            }
        }
    }

    TestCase {
        num_nodes,
        lambda,
        edges,
        ground_truth,
        name,
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn compute_accuracy(pred: &[bool], truth: &[bool]) -> (f64, f64, f64, f64) {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0, 0, 0);
    for (&p, &t) in pred.iter().zip(truth) {
        match (p, t) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    let accuracy = (tp + tn) as f64 / (tp + tn + fp + fn_) as f64;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (accuracy, precision, recall, f1)
}

// ---------------------------------------------------------------------------
// Exhaustive brute-force optimal (for validation on small graphs)
// ---------------------------------------------------------------------------

fn brute_force_optimal(
    lambda: &[f64],
    edges: &[(usize, usize, f64)],
    num_nodes: usize,
) -> (Vec<bool>, f64) {
    let mut best_labels = vec![false; num_nodes];
    let mut best_energy = f64::MAX;

    for mask in 0..(1u64 << num_nodes) {
        let labels: Vec<bool> = (0..num_nodes).map(|i| (mask >> i) & 1 == 1).collect();
        let e = graphcut_energy(lambda, edges, &labels);
        if e < best_energy {
            best_energy = e;
            best_labels = labels;
        }
    }
    (best_labels, best_energy)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== QAOA Graph-Cut Solver vs Classical Edmonds-Karp ===\n");
    println!("This example encodes the binary graph-cut labelling problem");
    println!("as a QAOA MaxCut instance and compares against the classical");
    println!("Edmonds-Karp BFS max-flow s-t mincut solver.\n");

    // Test cases: small enough for QAOA simulation (n+2 qubits total)
    let test_cases = vec![
        make_1d_chain(8, 3, 6, 2.0, 0.5, 0.3, 42, "1D-chain-8 (clean)"),
        make_1d_chain(10, 4, 8, 1.5, 0.8, 0.6, 99, "1D-chain-10 (noisy)"),
        make_2d_grid(3, 4, 77, "2D-grid-3x4"),
    ];

    println!(
        "  {:>22}  {:>5}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}  {:>9}",
        "Test Case", "Nodes", "Edges", "CL-Enrg", "QA-Enrg", "BF-Enrg", "QA/CL", "CL-F1", "QA-F1"
    );
    println!(
        "  {:->22}  {:->5}  {:->5}  {:->8}  {:->8}  {:->8}  {:->8}  {:->9}  {:->9}",
        "", "", "", "", "", "", "", "", ""
    );

    for tc in &test_cases {
        run_test_case(tc);
    }

    println!("\n=== QAOA Convergence Analysis ===\n");
    run_convergence_analysis();

    println!("\nDone.");
}

fn run_test_case(tc: &TestCase) {
    let num_nodes = tc.num_nodes;

    // --- Classical mincut ---
    let t0 = Instant::now();
    let classical_labels = solve_mincut_classical(&tc.lambda, &tc.edges, num_nodes);
    let classical_time = t0.elapsed();
    let classical_energy = graphcut_energy(&tc.lambda, &tc.edges, &classical_labels);

    // --- Brute-force optimal ---
    let (bf_labels, bf_energy) = brute_force_optimal(&tc.lambda, &tc.edges, num_nodes);

    // --- QAOA MaxCut encoding ---
    let augmented = encode_graphcut_as_maxcut(&tc.lambda, &tc.edges, num_nodes);
    let qaoa_qubits = augmented.num_nodes;

    let t1 = Instant::now();
    let qaoa_result = run_qaoa(&QaoaConfig {
        graph: augmented,
        p: 3, // depth-3 QAOA
        max_iterations: 80,
        learning_rate: 0.15,
        seed: Some(42),
    })
    .expect("QAOA failed");
    let qaoa_time = t1.elapsed();

    let qaoa_labels = extract_labels_from_maxcut(&qaoa_result.best_bitstring, num_nodes);
    let qaoa_energy = graphcut_energy(&tc.lambda, &tc.edges, &qaoa_labels);

    // --- Metrics ---
    let (_, _, _, cl_f1) = compute_accuracy(&classical_labels, &tc.ground_truth);
    let (_, _, _, qa_f1) = compute_accuracy(&qaoa_labels, &tc.ground_truth);
    let (_, _, _, bf_f1) = compute_accuracy(&bf_labels, &tc.ground_truth);

    let ratio = if classical_energy > 1e-12 {
        qaoa_energy / classical_energy
    } else {
        1.0
    };

    println!(
        "  {:>22}  {:>5}  {:>5}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>9.3}  {:>9.3}",
        tc.name,
        num_nodes,
        tc.edges.len(),
        classical_energy,
        qaoa_energy,
        bf_energy,
        ratio,
        cl_f1,
        qa_f1,
    );

    // Detailed per-case output
    println!();
    println!(
        "    Qubits (QAOA): {} ({} problem + 2 auxiliary s,t)",
        qaoa_qubits, num_nodes
    );
    println!("    Classical time: {:?}", classical_time);
    println!("    QAOA time:      {:?}", qaoa_time);
    println!("    QAOA converged: {}", qaoa_result.converged);
    println!(
        "    QAOA iterations: {} (energy trace: {:.3} -> {:.3})",
        qaoa_result.energy_history.len(),
        qaoa_result.energy_history.first().unwrap_or(&0.0),
        qaoa_result.energy_history.last().unwrap_or(&0.0),
    );

    // Label comparison
    print!("    Ground truth:   ");
    for &g in &tc.ground_truth {
        print!("{}", if g { '1' } else { '0' });
    }
    println!();
    print!("    Classical:      ");
    for &l in &classical_labels {
        print!("{}", if l { '1' } else { '0' });
    }
    println!("  (energy={:.3}, F1={:.3})", classical_energy, cl_f1);
    print!("    QAOA:           ");
    for &l in &qaoa_labels {
        print!("{}", if l { '1' } else { '0' });
    }
    println!("  (energy={:.3}, F1={:.3})", qaoa_energy, qa_f1);
    print!("    Brute-force:    ");
    for &l in &bf_labels {
        print!("{}", if l { '1' } else { '0' });
    }
    println!("  (energy={:.3}, F1={:.3})", bf_energy, bf_f1);

    println!(
        "    Quality ratio (QAOA/Classical): {:.3} (1.0 = equal, <1.0 = QAOA better)",
        ratio
    );
    println!();
}

/// Show how QAOA expected cut value evolves with increasing depth p.
fn run_convergence_analysis() {
    println!("  Depth-p sweep on 1D-chain-8:\n");

    let tc = make_1d_chain(8, 3, 6, 2.0, 0.5, 0.3, 42, "sweep");
    let augmented_base = encode_graphcut_as_maxcut(&tc.lambda, &tc.edges, tc.num_nodes);

    // Classical reference
    let cl_labels = solve_mincut_classical(&tc.lambda, &tc.edges, tc.num_nodes);
    let cl_energy = graphcut_energy(&tc.lambda, &tc.edges, &cl_labels);
    println!("  Classical mincut energy: {:.4}", cl_energy);
    println!();
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}",
        "p", "QAOA-Cut", "GC-Energy", "Ratio", "Iters"
    );
    println!(
        "  {:->5}  {:->10}  {:->10}  {:->10}  {:->10}",
        "", "", "", "", ""
    );

    for p in 1..=5 {
        // Rebuild the graph for each depth (run_qaoa takes ownership via ref)
        let graph = Graph {
            num_nodes: augmented_base.num_nodes,
            edges: augmented_base.edges.clone(),
        };

        let result = run_qaoa(&QaoaConfig {
            graph,
            p,
            max_iterations: 100,
            learning_rate: 0.15,
            seed: Some(42),
        })
        .expect("QAOA failed");

        let labels = extract_labels_from_maxcut(&result.best_bitstring, tc.num_nodes);
        let energy = graphcut_energy(&tc.lambda, &tc.edges, &labels);
        let ratio = if cl_energy > 1e-12 {
            energy / cl_energy
        } else {
            1.0
        };

        println!(
            "  {:>5}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10}",
            p,
            result.best_cut_value,
            energy,
            ratio,
            result.energy_history.len(),
        );
    }

    println!("\n  As p increases, QAOA approaches the classical optimal.");
    println!("  At p -> infinity, QAOA is guaranteed to find the exact solution.");
}
