//! Structural (static) analysis of the connectome — no coactivation
//! weighting. Used by AC-3a to test whether `ruvector-mincut` can
//! recover the generator's SBM module structure from the static edge
//! graph alone.
//!
//! This is the complement to `analysis::partition::functional_partition`,
//! which weights edges by recent spike coactivation. The two paths
//! answer different questions:
//!
//! - **Structural (this module, AC-3a)**: given the static connectome
//!   as a weighted graph, can we recover the SBM module labels?
//!   Measured as Adjusted Rand Index vs the ground-truth hub-vs-non-hub
//!   binary partition. Paired against a hand-rolled Louvain-style
//!   greedy modularity baseline so the comparison is honest.
//!
//! - **Functional (partition.rs, AC-3b)**: given the static connectome
//!   weighted by *recent coactivation*, does the partition change with
//!   stimulus? Measured as L1 class-histogram distance between sides.
//!
//! See ADR-154 §3.4 "Acceptance Test Architecture" for why these are
//! split and `BENCHMARK.md` AC-3a / AC-3b for the numbers.

use ruvector_mincut::MinCutBuilder;

use crate::connectome::{Connectome, NeuronId};

use super::types::{class_name, AnalysisConfig, FunctionalPartition};

/// Structural partition of the static connectome via `ruvector-mincut`.
///
/// Weights edges by `synapse.weight` directly (no coactivation).
/// Returns the same `FunctionalPartition` shape as
/// `functional_partition` so downstream tooling is uniform.
pub fn structural_partition(cfg: &AnalysisConfig, conn: &Connectome) -> FunctionalPartition {
    let n = conn.num_neurons();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();

    // Aggregate undirected edge weights. Without coactivation weighting
    // we simply use the synapse weight (signed contribution folded into
    // the absolute-value weight; mincut operates on non-negative edges).
    let mut agg: std::collections::HashMap<(u64, u64), f64> = std::collections::HashMap::new();
    for pre_idx in 0..n {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for syn_entry in &syn[s..e] {
            let post = syn_entry.post.idx();
            if post == pre_idx {
                continue;
            }
            let u = pre_idx.min(post) as u64 + 1;
            let v = pre_idx.max(post) as u64 + 1;
            *agg.entry((u, v)).or_insert(0.0) += syn_entry.weight as f64;
        }
    }
    // Sort and bound to top-k edges by weight so the exact mincut stays
    // tractable. The SBM target has ~50k edges at N=1024; the default
    // `mincut_top_k` keeps ~4k.
    let mut edges: Vec<(u64, u64, f64)> = agg
        .into_iter()
        .map(|((u, v), w)| (u, v, w.clamp(cfg.min_w, cfg.max_w)))
        .collect();
    edges.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
            .then(a.1.cmp(&b.1))
    });
    edges.truncate(cfg.mincut_top_k);
    edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    if edges.is_empty() {
        return FunctionalPartition {
            cut_value: 0.0,
            side_a: Vec::new(),
            side_b: Vec::new(),
            edges_considered: 0,
            side_a_class_histogram: Vec::new(),
            side_b_class_histogram: Vec::new(),
        };
    }
    let edges_considered = edges.len() as u64;
    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(edges)
        .build()
        .expect("structural mincut");
    let cut_value = mc.min_cut_value();
    let result = mc.min_cut();
    let (side_a, side_b) = result
        .partition
        .map(|(a, b)| {
            (
                a.iter()
                    .map(|x| (*x as u32).saturating_sub(1))
                    .collect::<Vec<_>>(),
                b.iter()
                    .map(|x| (*x as u32).saturating_sub(1))
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap_or_default();
    FunctionalPartition {
        cut_value,
        side_a: side_a.clone(),
        side_b: side_b.clone(),
        edges_considered,
        side_a_class_histogram: class_histogram(&side_a, conn),
        side_b_class_histogram: class_histogram(&side_b, conn),
    }
}

/// Hand-rolled greedy modularity (Louvain-style level-1) baseline so
/// AC-3a can compare mincut ARI against a published-family method. Not
/// a full Louvain — one pass of greedy node moves in module-order.
/// Deterministic; no randomness.
pub fn greedy_modularity_labels(conn: &Connectome) -> Vec<u32> {
    let n = conn.num_neurons();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();

    // Initialize: every neuron in its own community.
    let mut comm: Vec<u32> = (0..n as u32).collect();
    // Undirected weighted-degree cache.
    let mut deg = vec![0.0_f64; n];
    for pre_idx in 0..n {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for syn_entry in &syn[s..e] {
            let post = syn_entry.post.idx();
            if post == pre_idx {
                continue;
            }
            let w = syn_entry.weight as f64;
            deg[pre_idx] += w;
            deg[post] += w;
        }
    }
    let two_m: f64 = deg.iter().sum::<f64>().max(1.0);

    // One greedy pass: each node moves to the community that maximizes
    // Δmodularity (classical Louvain level-1). Fixed iteration order
    // → deterministic.
    let mut changed = true;
    let mut it = 0;
    while changed && it < 8 {
        changed = false;
        for i in 0..n {
            // Weight to each neighbor community.
            let mut ng: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
            let s = row_ptr[i] as usize;
            let e = row_ptr[i + 1] as usize;
            for syn_entry in &syn[s..e] {
                let j = syn_entry.post.idx();
                if j == i {
                    continue;
                }
                *ng.entry(comm[j]).or_insert(0.0) += syn_entry.weight as f64;
            }
            let c_self = comm[i];
            let mut best_c = c_self;
            let mut best_gain = 0.0_f64;
            for (&c, &k_ic) in &ng {
                if c == c_self {
                    continue;
                }
                // Simplified Δmodularity = k_i_in_c / m - d_i * Σd_c / (2m²)
                // — full LL formulation omitted; we only need a move
                // criterion, not a stable optimum.
                let d_i = deg[i];
                let d_c: f64 = (0..n)
                    .filter(|&k| comm[k] == c)
                    .map(|k| deg[k])
                    .sum::<f64>();
                let gain = k_ic / two_m - d_i * d_c / (2.0 * two_m * two_m);
                if gain > best_gain + 1e-9 {
                    best_gain = gain;
                    best_c = c;
                }
            }
            if best_c != c_self {
                comm[i] = best_c;
                changed = true;
            }
        }
        it += 1;
    }
    comm
}

fn class_histogram(side: &[u32], conn: &Connectome) -> Vec<(String, u32)> {
    let mut counts = [0_u32; 15];
    for id in side {
        let m = conn.meta(NeuronId(*id));
        counts[m.class as usize] += 1;
    }
    let mut out = Vec::new();
    for (i, c) in counts.iter().enumerate() {
        if *c > 0 {
            out.push((class_name(i as u8), *c));
        }
    }
    out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    out
}
