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

/// Multi-level Louvain (aggregation + re-run until no further gain).
///
/// **Empirical finding on hub-heavy SBMs (ADR-154 §17 item 11):** at
/// the demo's N=1024 SBM with hub modules, this multi-level variant
/// *over-aggregates* — by the second level the whole graph collapses
/// to a single super-community and ARI vs hub-vs-non-hub ground truth
/// drops to 0. The simpler `greedy_modularity_labels` (level-1 only)
/// actually scores higher on the same graph (measured `louvain=0.000`
/// vs `greedy=0.174` on default config). This is the documented
/// failure mode of Louvain without Leiden's refinement phase: the
/// aggregation step can absorb well-connected but structurally
/// distinct communities into one super-node, and there is no
/// mechanism to un-merge. Leiden's refinement phase is what fixes
/// this; it remains named as follow-up in ADR-154 §13.
///
/// Determinism: fixed iteration order, no RNG, fixed tie-break
/// (prefer lower community id). Same input → bit-identical labels.
pub fn louvain_labels(conn: &Connectome) -> Vec<u32> {
    // Build the level-0 undirected-weighted graph from Connectome:
    //   nodes = conn neurons
    //   edges = synapse-weighted undirected, self-loops dropped
    //           (matches greedy_modularity_labels convention)
    let n0 = conn.num_neurons();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    let mut adj0: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n0];
    for pre_idx in 0..n0 {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for syn_entry in &syn[s..e] {
            let post = syn_entry.post.idx();
            if post == pre_idx {
                continue;
            }
            let w = syn_entry.weight as f64;
            adj0[pre_idx].push((post as u32, w));
            adj0[post].push((pre_idx as u32, w));
        }
    }

    // Per-node community label at the *original* level. Initially
    // every neuron is its own community.
    let mut labels_lvl0: Vec<u32> = (0..n0 as u32).collect();

    // Current working graph, initially = level-0 connectome.
    let mut adj: Vec<Vec<(u32, f64)>> = adj0;

    // Loop through aggregation levels. `max_levels` is a safety cap.
    for _level in 0..8 {
        let n = adj.len();
        let labels_this_level = level1_moves(&adj, n);
        // Check if anything changed at this level (every node is its own
        // community after the move pass = no change).
        let mut changed = false;
        for i in 0..n {
            if labels_this_level[i] != i as u32 {
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
        // Project labels_this_level back to the level-0 nodes:
        //   if neuron i at level-0 currently maps to super-node s in
        //   adj (labels_lvl0[i] == s), then its new *level-superior*
        //   super-community label is labels_this_level[s]. That is
        //   still in the OLD label space; renumbering below remaps
        //   it to the new dense super-graph indices.
        for lbl in labels_lvl0.iter_mut() {
            *lbl = labels_this_level[*lbl as usize];
        }
        // Renumber + aggregate adj to produce the new working graph.
        // `renum` maps OLD level-community labels → dense super-node
        // indices in next_adj. `labels_lvl0` must follow that remap
        // so subsequent levels index valid super-graph nodes.
        let (next_adj, renum) = aggregate(&adj, &labels_this_level);
        for lbl in labels_lvl0.iter_mut() {
            *lbl = *renum.get(lbl).expect("super-community must be in renum");
        }
        if next_adj.len() == adj.len() {
            // No aggregation happened (every node is its own community),
            // safe to break.
            break;
        }
        adj = next_adj;
    }

    // Compact label space to a dense 0..k range so downstream
    // `two_way_from_labels` works regardless of intermediate renumbering.
    compact_labels(&labels_lvl0)
}

/// One full sweep of Louvain level-1 moves on `adj` (size `n`). Returns
/// per-node community labels using node indices as initial ids. Same
/// deterministic tie-break as the single-level variant.
fn level1_moves(adj: &[Vec<(u32, f64)>], n: usize) -> Vec<u32> {
    let mut deg = vec![0.0_f64; n];
    for i in 0..n {
        for &(_, w) in &adj[i] {
            deg[i] += w;
        }
    }
    let two_m: f64 = deg.iter().sum::<f64>().max(1.0);
    let mut comm: Vec<u32> = (0..n as u32).collect();
    // Running per-community weighted degree sum, keyed by community id.
    // Initially every node is alone, so `cdeg[i] == deg[i]`.
    let mut cdeg: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for i in 0..n {
        cdeg.insert(i as u32, deg[i]);
    }

    let mut changed = true;
    let mut it = 0;
    while changed && it < 16 {
        changed = false;
        for i in 0..n {
            let mut neigh_w: std::collections::HashMap<u32, f64> =
                std::collections::HashMap::new();
            for &(j, w) in &adj[i] {
                if j as usize == i {
                    continue;
                }
                *neigh_w.entry(comm[j as usize]).or_insert(0.0) += w;
            }
            let c_self = comm[i];
            let mut best_c = c_self;
            let mut best_gain = 0.0_f64;
            let d_i = deg[i];
            for (&c, &k_ic) in &neigh_w {
                if c == c_self {
                    continue;
                }
                let d_c = *cdeg.get(&c).unwrap_or(&0.0);
                let gain = k_ic / two_m - d_i * d_c / (2.0 * two_m * two_m);
                if gain > best_gain + 1e-9 {
                    best_gain = gain;
                    best_c = c;
                }
            }
            if best_c != c_self {
                *cdeg.entry(c_self).or_insert(0.0) -= d_i;
                *cdeg.entry(best_c).or_insert(0.0) += d_i;
                comm[i] = best_c;
                changed = true;
            }
        }
        it += 1;
    }
    comm
}

/// Aggregate `adj` into a super-graph whose nodes are the communities
/// in `labels`. Returns (new_adj, renumber_map) where renumber_map[old]
/// = new_community_index. Edge weights sum inside the super-nodes.
fn aggregate(
    adj: &[Vec<(u32, f64)>],
    labels: &[u32],
) -> (Vec<Vec<(u32, f64)>>, std::collections::HashMap<u32, u32>) {
    // Build dense renumbering old_label → new_index.
    let mut renum: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &lab in labels {
        let k = renum.len() as u32;
        renum.entry(lab).or_insert(k);
    }
    let new_n = renum.len();
    let mut next: Vec<std::collections::HashMap<u32, f64>> =
        (0..new_n).map(|_| std::collections::HashMap::new()).collect();
    for i in 0..adj.len() {
        let ui = *renum.get(&labels[i]).expect("renum");
        for &(j, w) in &adj[i] {
            let uj = *renum.get(&labels[j as usize]).expect("renum");
            if ui == uj {
                continue; // drop intra-community edges (become self-loops)
            }
            *next[ui as usize].entry(uj).or_insert(0.0) += w;
        }
    }
    let new_adj: Vec<Vec<(u32, f64)>> = next
        .into_iter()
        .map(|m| m.into_iter().collect::<Vec<_>>())
        .collect();
    (new_adj, renum)
}

/// Compact arbitrary labels into `0..k` space, preserving grouping.
fn compact_labels(labels: &[u32]) -> Vec<u32> {
    let mut renum: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut out: Vec<u32> = Vec::with_capacity(labels.len());
    for &lab in labels {
        let k = renum.len() as u32;
        let id = *renum.entry(lab).or_insert(k);
        out.push(id);
    }
    out
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
