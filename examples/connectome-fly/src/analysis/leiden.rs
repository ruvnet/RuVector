//! Leiden community detection: multi-level Louvain + Traag's
//! refinement (Traag, Waltman, van Eck 2019, *From Louvain to Leiden:
//! guaranteeing well-connected communities*, *Sci. Rep.* 9:5233).
//!
//! Why this exists (ADR-154 §17 item 11): `structural::louvain_labels`
//! collapses to a single super-community on the demo's N=1024 SBM
//! (`louvain_ari = 0.000`). Refinement splits weakly-connected
//! communities before the next level's moves can collapse them.
//!
//! Each level:
//! 1. Local moves (Louvain-style). `level1_moves` at level 0;
//!    `level1_moves_from` at level ≥ 1 (non-singleton initial:
//!    super-nodes from the same previous coarse community start
//!    grouped — Traag Alg. 1 line 10). Produces coarse `P`.
//! 2. Refinement (Alg. 4). `P_refined ← Singleton`; for each coarse
//!    `C`, greedily merge still-singleton nodes into γ-well-connected
//!    sub-communities (`E(S, C\S) ≥ γ · d(S) · d(C\S) / (2m)`). Once
//!    placed, nodes are frozen (monotonic growth).
//! 3. Aggregate on refined labels. For level k+1,
//!    `initial[new_super] = coarse[old_source]`.
//!
//! Newman-Girvan modularity has a resolution limit (Fortunato &
//! Barthélemy 2007); Leiden's refinement does not fully escape it.
//! We track the best-modularity partition across levels on the base
//! graph and return that.
//!
//! Connectivity defence: `level1_moves_from` with a non-singleton
//! initial can leave same-label super-nodes that share no super-
//! graph edge; `refine` is by construction connectivity-preserving
//! but f64 bookkeeping can leak. We apply
//! `split_into_connected_components` to coarse (level ≥ 1) and
//! refined partitions; splitting only raises modularity.
//!
//! Determinism: ascending-id iteration, lower-sub-id tie-break, no
//! RNG. Same input → bit-identical output.

use std::collections::{HashMap, HashSet};

use crate::connectome::Connectome;

use super::structural::{aggregate, compact_labels, level1_moves};

/// Resolution γ for the well-connectedness check
/// `E(S, C\S) ≥ γ · d(S) · d(C\S) / (2m)`. γ = 1.0 is Traag's canonical
/// choice.
const GAMMA: f64 = 1.0;

/// Safety cap on outer aggregation levels (Leiden terminates in 2–4 in
/// practice).
const MAX_LEVELS: usize = 8;

/// Safety cap on `level1_moves_from` sweeps per level.
const MAX_LOCAL_MOVE_PASSES: usize = 16;

/// Leiden community labels for the static connectome.
///
/// Returns per-neuron labels compacted into `0..k`. Deterministic.
pub fn leiden_labels(conn: &Connectome) -> Vec<u32> {
    let n0 = conn.num_neurons();

    // Build the level-0 undirected-weighted graph. Synapses in either
    // direction between the same pair are summed into a single
    // undirected edge weight.
    let mut agg_edges: HashMap<(u32, u32), f64> = HashMap::new();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    for pre_idx in 0..n0 {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for syn_entry in &syn[s..e] {
            let post = syn_entry.post.idx();
            if post == pre_idx {
                continue;
            }
            let w = syn_entry.weight as f64;
            let (u, v) = if pre_idx < post {
                (pre_idx as u32, post as u32)
            } else {
                (post as u32, pre_idx as u32)
            };
            *agg_edges.entry((u, v)).or_insert(0.0) += w;
        }
    }
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n0];
    for ((u, v), w) in agg_edges {
        adj[u as usize].push((v, w));
        adj[v as usize].push((u, w));
    }

    // Base graph state for modularity scoring (never changes).
    let adj_base: Vec<Vec<(u32, f64)>> = adj.clone();
    let deg_base: Vec<f64> = {
        let mut d = vec![0.0_f64; n0];
        for i in 0..n0 {
            for &(_, w) in &adj_base[i] {
                d[i] += w;
            }
        }
        d
    };
    let two_m_base: f64 = deg_base.iter().sum::<f64>().max(1.0);

    // Current base-node → community mapping, projected through
    // successive aggregation levels.
    let mut labels_lvl0: Vec<u32> = (0..n0 as u32).collect();

    // Input partition to Phase 1 at the current level. Singleton at
    // level 0; at level ≥ 1, super-nodes inherit their previous
    // coarse community (Traag Alg. 1 line 10).
    let mut initial: Vec<u32> = (0..adj.len() as u32).collect();

    // Best-modularity candidate (k ≥ 2) on the base graph.
    let mut best_labels = labels_lvl0.clone();
    let mut best_q = modularity(&adj_base, &labels_lvl0, &deg_base, two_m_base);

    for level in 0..MAX_LEVELS {
        let n = adj.len();

        // Phase 1 — local moves (+ connectivity split at level ≥ 1).
        let raw_coarse = if (0..n).all(|i| initial[i] == i as u32) {
            level1_moves(&adj, n)
        } else {
            level1_moves_from(&adj, &initial)
        };
        let coarse = if level == 0 {
            raw_coarse.clone()
        } else {
            split_into_connected_components(&adj, &raw_coarse)
        };

        // Phase 2 — refinement (+ defensive connectivity split).
        let raw_refined = refine(&adj, &coarse, GAMMA);
        let refined = split_into_connected_components(&adj, &raw_refined);

        // Candidate: project coarse labels to base and score Q.
        let coarse_projected: Vec<u32> = labels_lvl0.iter().map(|&l| coarse[l as usize]).collect();
        consider_candidate(
            &adj_base,
            &coarse_projected,
            &deg_base,
            two_m_base,
            &mut best_labels,
            &mut best_q,
        );

        // Termination (Traag Alg. 1 line 4): MoveNodesFast produced
        // the singleton partition ⇒ nothing left to merge.
        if count_unique(&coarse) == n {
            break;
        }

        // Project refined → labels_lvl0 and score as candidate.
        for lbl in labels_lvl0.iter_mut() {
            *lbl = refined[*lbl as usize];
        }
        consider_candidate(
            &adj_base,
            &labels_lvl0,
            &deg_base,
            two_m_base,
            &mut best_labels,
            &mut best_q,
        );

        // Phase 3 — aggregate on refined labels.
        let (next_adj, renum) = aggregate(&adj, &refined);
        for lbl in labels_lvl0.iter_mut() {
            *lbl = *renum.get(lbl).expect("super-community in renum");
        }
        if next_adj.len() == adj.len() {
            break;
        }

        // Next level's `initial`: new super-nodes inherit the coarse
        // community they were refined out of.
        let new_n = next_adj.len();
        let mut next_initial = vec![0_u32; new_n];
        for i in 0..n {
            let new_sub = *renum.get(&refined[i]).expect("renum");
            next_initial[new_sub as usize] = coarse[i];
        }

        adj = next_adj;
        initial = next_initial;
    }

    let _ = best_q;
    compact_labels(&best_labels)
}

/// Update `best_labels` / `best_q` if `candidate` has k ≥ 2
/// communities and strictly higher modularity than `*best_q`.
fn consider_candidate(
    adj: &[Vec<(u32, f64)>],
    candidate: &[u32],
    deg: &[f64],
    two_m: f64,
    best_labels: &mut Vec<u32>,
    best_q: &mut f64,
) {
    if count_unique(candidate) < 2 {
        return;
    }
    let q = modularity(adj, candidate, deg, two_m);
    if q > *best_q + 1e-12 {
        *best_q = q;
        best_labels.clone_from(&candidate.to_vec());
    }
}

/// Newman-Girvan modularity summed per-community. `adj` double-stores
/// each undirected edge (matches `structural::louvain_labels`).
fn modularity(adj: &[Vec<(u32, f64)>], labels: &[u32], deg: &[f64], two_m: f64) -> f64 {
    if two_m <= 0.0 {
        return 0.0;
    }
    let n = adj.len();
    let mut e_in: HashMap<u32, f64> = HashMap::new();
    let mut d_sum: HashMap<u32, f64> = HashMap::new();
    for i in 0..n {
        *d_sum.entry(labels[i]).or_insert(0.0) += deg[i];
        for &(j, w) in &adj[i] {
            if labels[j as usize] == labels[i] {
                *e_in.entry(labels[i]).or_insert(0.0) += w;
            }
        }
    }
    let mut q = 0.0_f64;
    for c in d_sum.keys() {
        let d = *d_sum.get(c).unwrap_or(&0.0);
        let e = *e_in.get(c).unwrap_or(&0.0);
        q += e / two_m - (d / two_m) * (d / two_m);
    }
    q
}

/// Number of distinct labels in `labels`.
fn count_unique(labels: &[u32]) -> usize {
    let mut s: HashSet<u32> = HashSet::new();
    for &l in labels {
        s.insert(l);
    }
    s.len()
}

/// Split each community in `labels` into its BFS-connected components
/// in the adjacency graph `adj`. Returns new labels where two nodes
/// share a label iff they shared a label in `labels` AND are
/// reachable from each other via `adj` edges whose BOTH endpoints
/// also share that label.
///
/// Output ids are unique within the result and disjoint from input
/// ids (running counter starting above `max(labels)`).
fn split_into_connected_components(adj: &[Vec<(u32, f64)>], labels: &[u32]) -> Vec<u32> {
    let n = adj.len();
    let mut out = vec![u32::MAX; n];
    let mut next_id: u32 = labels.iter().copied().max().unwrap_or(0).saturating_add(1);

    for seed in 0..n {
        if out[seed] != u32::MAX {
            continue;
        }
        let comm = labels[seed];
        let new_id = next_id;
        next_id = next_id.saturating_add(1);
        let mut stack = vec![seed];
        while let Some(v) = stack.pop() {
            if out[v] != u32::MAX {
                continue;
            }
            if labels[v] != comm {
                continue;
            }
            out[v] = new_id;
            for &(u, _) in &adj[v] {
                let u = u as usize;
                if out[u] == u32::MAX && labels[u] == comm {
                    stack.push(u);
                }
            }
        }
    }
    for i in 0..n {
        if out[i] == u32::MAX {
            out[i] = next_id;
            next_id = next_id.saturating_add(1);
        }
    }
    out
}

/// `RefinePartition(G, P)` — Traag 2019 Algorithm 4. Starts with the
/// singleton partition and, within each coarse community in `coarse`,
/// greedily merges singleton nodes into well-connected sub-communities.
fn refine(adj: &[Vec<(u32, f64)>], coarse: &[u32], gamma: f64) -> Vec<u32> {
    let n = adj.len();
    let mut deg = vec![0.0_f64; n];
    for i in 0..n {
        for &(_, w) in &adj[i] {
            deg[i] += w;
        }
    }
    let two_m: f64 = deg.iter().sum::<f64>().max(1.0);

    let mut by_coarse: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, &c) in coarse.iter().enumerate() {
        by_coarse.entry(c).or_default().push(i as u32);
    }
    let mut coarse_keys: Vec<u32> = by_coarse.keys().copied().collect();
    coarse_keys.sort();

    let mut sub: Vec<u32> = (0..n as u32).collect();
    for coarse_id in coarse_keys {
        let mut nodes = by_coarse.remove(&coarse_id).unwrap_or_default();
        nodes.sort();
        if nodes.len() <= 1 {
            continue;
        }
        refine_one_community(&mut sub, adj, &nodes, &deg, two_m, gamma);
    }
    sub
}

/// `MergeNodesSubset(G, P_refined, C)` — Traag 2019 Algorithm 4 for
/// one coarse community. Only singleton nodes move; once v joins a
/// non-singleton sub-community it stays (monotonic growth preserves
/// internal connectivity).
fn refine_one_community(
    sub: &mut [u32],
    adj: &[Vec<(u32, f64)>],
    nodes: &[u32],
    deg: &[f64],
    two_m: f64,
    gamma: f64,
) {
    let mut in_c = vec![false; adj.len()];
    let mut d_total_c = 0.0_f64;
    for &v in nodes {
        in_c[v as usize] = true;
        d_total_c += deg[v as usize];
    }

    // Per-sub-community state in C:
    //   deg_sum[s] = Σ deg(i) for i ∈ s,
    //   e_out[s]   = E(s, C\s) counted once per undirected edge.
    let mut deg_sum: HashMap<u32, f64> = HashMap::with_capacity(nodes.len());
    let mut e_out: HashMap<u32, f64> = HashMap::with_capacity(nodes.len());
    for &v in nodes {
        deg_sum.insert(v, deg[v as usize]);
        let mut ev = 0.0;
        for &(j, w) in &adj[v as usize] {
            if in_c[j as usize] && j != v {
                ev += w;
            }
        }
        e_out.insert(v, ev);
    }

    // Precompute whether each singleton v is well-connected to C.
    let mut v_well: HashMap<u32, bool> = HashMap::with_capacity(nodes.len());
    for &v in nodes {
        let d_v = deg[v as usize];
        let k_v_c = *e_out.get(&v).unwrap_or(&0.0);
        let rhs = gamma * d_v * (d_total_c - d_v) / (2.0 * two_m);
        v_well.insert(v, k_v_c >= rhs - 1e-12);
    }

    let mut moved = vec![false; adj.len()];
    for &v in nodes {
        if moved[v as usize] || !v_well.get(&v).copied().unwrap_or(false) {
            continue;
        }
        let s_v = sub[v as usize];
        debug_assert_eq!(s_v, v);
        let d_v = deg[v as usize];

        // Weight from v into each candidate sub-community within C.
        let mut k_to: HashMap<u32, f64> = HashMap::new();
        for &(j, w) in &adj[v as usize] {
            if !in_c[j as usize] || j == v {
                continue;
            }
            *k_to.entry(sub[j as usize]).or_insert(0.0) += w;
        }
        let mut cand_ids: Vec<u32> = k_to.keys().copied().collect();
        cand_ids.sort();

        let mut best_target: u32 = s_v;
        let mut best_gain: f64 = 0.0;
        for s_t in cand_ids {
            if s_t == s_v {
                continue;
            }
            let d_s = *deg_sum.get(&s_t).unwrap_or(&0.0);
            let e_s_rest = *e_out.get(&s_t).unwrap_or(&0.0);
            // Target well-connectedness (Traag §2.3, weighted form).
            if e_s_rest < gamma * d_s * (d_total_c - d_s) / (2.0 * two_m) {
                continue;
            }
            let k_to_t = *k_to.get(&s_t).unwrap_or(&0.0);
            // Modularity-joining gain (matches level1_moves).
            let gain = k_to_t / two_m - d_v * d_s / (2.0 * two_m * two_m);
            if gain > best_gain + 1e-12 {
                best_gain = gain;
                best_target = s_t;
            }
        }
        if best_target == s_v {
            continue;
        }

        // Move v into best_target. e_out delta (adj double-stores):
        //   (k_v_c − k_to_new)  [v's external-to-best edges added]
        //   − 2·k_to_new        [peer edges both sides become internal]
        let k_to_new = *k_to.get(&best_target).unwrap_or(&0.0);
        let k_v_c: f64 = k_to.values().sum();
        deg_sum.remove(&s_v);
        e_out.remove(&s_v);
        *deg_sum.entry(best_target).or_insert(0.0) += d_v;
        let et = e_out.entry(best_target).or_insert(0.0);
        *et += k_v_c - 2.0 * k_to_new;
        if *et < 0.0 {
            *et = 0.0;
        }
        sub[v as usize] = best_target;
        moved[v as usize] = true;
    }
}

/// `level1_moves` variant that accepts a non-singleton initial
/// partition. Node `i` starts in community `initial[i]`. All other
/// semantics (weighted Δmodularity, deterministic ascending-id
/// iteration, tie-break toward lower community id) match
/// `structural::level1_moves`.
fn level1_moves_from(adj: &[Vec<(u32, f64)>], initial: &[u32]) -> Vec<u32> {
    let n = adj.len();
    debug_assert_eq!(initial.len(), n);

    let mut deg = vec![0.0_f64; n];
    for i in 0..n {
        for &(_, w) in &adj[i] {
            deg[i] += w;
        }
    }
    let two_m: f64 = deg.iter().sum::<f64>().max(1.0);

    let mut comm: Vec<u32> = initial.to_vec();
    let mut cdeg: HashMap<u32, f64> = HashMap::new();
    for i in 0..n {
        *cdeg.entry(comm[i]).or_insert(0.0) += deg[i];
    }

    let mut it = 0;
    let mut changed = true;
    while changed && it < MAX_LOCAL_MOVE_PASSES {
        changed = false;
        for i in 0..n {
            let mut neigh_w: HashMap<u32, f64> = HashMap::new();
            for &(j, w) in &adj[i] {
                if j as usize == i {
                    continue;
                }
                *neigh_w.entry(comm[j as usize]).or_insert(0.0) += w;
            }
            let c_self = comm[i];
            let d_i = deg[i];
            let mut best_c = c_self;
            let mut best_gain = 0.0_f64;
            let mut cands: Vec<u32> = neigh_w.keys().copied().collect();
            cands.sort();
            for c in cands {
                if c == c_self {
                    continue;
                }
                let k_ic = *neigh_w.get(&c).unwrap_or(&0.0);
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
