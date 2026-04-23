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

// -----------------------------------------------------------------
// CPM (Constant Potts Model) variant — ADR §13 follow-up, §17 item 14
// -----------------------------------------------------------------
//
// Modularity has a resolution limit (Fortunato & Barthélemy 2007) —
// on hub-heavy SBMs its landscape rewards merging distinct communities
// into super-communities when 2m grows large, which is why
// multi-level Louvain collapses and modularity-Leiden scores
// ARI = 0.089 on the default SBM. CPM (Traag's own default in
// `leidenalg`) does not have that property — it has a simple
// per-community penalty in `γ * C(n_c, 2)` that makes merging
// strictly net-negative once the inter-community density drops
// below γ. Sweeping γ on real data is the canonical protocol;
// γ = 0.05 is a common starting point on weighted graphs.
//
// The move gain for v moving from S to C (C ≠ S) is
//
//     ΔH = k_{v,C} − k_{v,S\{v}} − γ·(n_C − n_S + 1)
//
// so the per-candidate score we compare is `k_{v,C} − γ·n_C`.
//
// This first cut parallels `level1_moves_from` + the existing
// aggregate loop. It does NOT layer Traag's refinement phase yet;
// `leiden_labels_cpm` is a multi-level Louvain with the CPM
// objective. That's already strictly stronger than modularity-only
// Louvain on resolution-limit-bound graphs; adding CPM-refinement
// is the next lever if the measurement says it's worth it.

/// Leiden-style multi-level driver with the Constant Potts Model
/// quality function at `γ`. **Edges are pre-normalized so the mean
/// edge weight becomes 1.0** — this makes γ dimensionless rather
/// than in raw-synapse-weight units (see §17 item 16 for the
/// non-normalized first-cut that collapsed). Common starting points
/// on normalized weighted SBMs: `γ ∈ [0.1, 0.5]`. Deterministic; no
/// RNG.
pub fn leiden_labels_cpm(conn: &Connectome, gamma: f64) -> Vec<u32> {
    let n0 = conn.num_neurons();

    // Level-0 graph: same undirected aggregation as `leiden_labels`.
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
    // Weight normalization: rescale so the mean undirected edge
    // weight becomes 1.0. This is what Traag's `leidenalg` does
    // implicitly (CPM quality is dimensionless there). Without this
    // rescaling, γ lives in raw-weight units and every reasonable γ
    // is dwarfed by summed synapse weights — see §17 item 16 for the
    // non-normalized failure mode.
    let mean_w = if agg_edges.is_empty() {
        1.0
    } else {
        let sum: f64 = agg_edges.values().sum();
        (sum / agg_edges.len() as f64).max(1e-12)
    };
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n0];
    for ((u, v), w) in agg_edges {
        let wn = w / mean_w;
        adj[u as usize].push((v, wn));
        adj[v as usize].push((u, wn));
    }

    // Multi-level loop: CPM local moves → aggregate → repeat.
    let mut labels_lvl0: Vec<u32> = (0..n0 as u32).collect();
    // Per-level-node count-of-base-nodes inside each node. At level 0
    // every level-node represents 1 base node; at deeper levels each
    // super-node represents the sum of its constituents.
    let mut level_n_per_node: Vec<u64> = vec![1_u64; n0];

    for _level in 0..8 {
        let n = adj.len();
        let comm = level1_moves_cpm(&adj, &level_n_per_node, gamma);

        // Project this level's community map back onto the level-0
        // node ids.
        for lbl in labels_lvl0.iter_mut() {
            *lbl = comm[*lbl as usize];
        }

        // Did anything actually coarsen?
        let unique_new = count_unique(&comm);
        if unique_new == n {
            // Every level-node is its own community → no further aggregation.
            break;
        }

        // Aggregate level-nodes by their community label. New node
        // count per super-node = sum of constituents' counts (so CPM
        // at the next level stays faithful to the base graph).
        let (next_adj, next_n_per_node, renum) =
            aggregate_cpm(&adj, &comm, &level_n_per_node);
        // Re-label the base → level mapping to use the new dense
        // super-node indices.
        for lbl in labels_lvl0.iter_mut() {
            *lbl = *renum.get(lbl).expect("renum must cover every active label");
        }
        adj = next_adj;
        level_n_per_node = next_n_per_node;
    }

    compact_cpm_labels(&labels_lvl0)
}

/// CPM level-1 move pass. Same structure as `level1_moves_from` but
/// per-community accumulator is *count of base nodes* (not weighted
/// degree) and the move gain is `k_{v,C} − γ·n_C`.
fn level1_moves_cpm(
    adj: &[Vec<(u32, f64)>],
    n_per_node: &[u64],
    gamma: f64,
) -> Vec<u32> {
    let n = adj.len();
    let mut comm: Vec<u32> = (0..n as u32).collect();
    // n_c (number of base nodes inside community c). Initialised
    // from the per-level-node counts.
    let mut n_c: HashMap<u32, u64> = HashMap::new();
    for i in 0..n {
        *n_c.entry(comm[i]).or_insert(0) += n_per_node[i];
    }

    let mut it = 0_usize;
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
            let my_n = n_per_node[i] as f64;
            let n_self = *n_c.get(&c_self).unwrap_or(&0) as f64;
            // "stay" score: k_{v,S\{v}} − γ·(n_S − my_n). The
            // k_{v,S\{v}} term = neigh_w[c_self] (sum of edge
            // weights from v to the current community, excluding
            // self-loops which we skipped above).
            let k_self = *neigh_w.get(&c_self).unwrap_or(&0.0);
            let stay_score = k_self - gamma * (n_self - my_n);

            let mut best_c = c_self;
            let mut best_score = stay_score;
            let mut cands: Vec<u32> = neigh_w.keys().copied().collect();
            cands.sort();
            for c in cands {
                if c == c_self {
                    continue;
                }
                let k_ic = *neigh_w.get(&c).unwrap_or(&0.0);
                let n_cand = *n_c.get(&c).unwrap_or(&0) as f64;
                let score = k_ic - gamma * n_cand;
                if score > best_score + 1e-9 {
                    best_score = score;
                    best_c = c;
                }
            }
            if best_c != c_self {
                *n_c.entry(c_self).or_insert(0) -= n_per_node[i];
                *n_c.entry(best_c).or_insert(0) += n_per_node[i];
                comm[i] = best_c;
                changed = true;
            }
        }
        it += 1;
    }
    comm
}

/// Aggregate `adj` by the communities in `labels`. Same shape as the
/// modularity-path's `aggregate` in `structural.rs`, plus it carries
/// the per-super-node base-node count so CPM at the next level has
/// the right n_c. Returns (next_adj, next_n_per_node, renumbering).
fn aggregate_cpm(
    adj: &[Vec<(u32, f64)>],
    labels: &[u32],
    n_per_node: &[u64],
) -> (Vec<Vec<(u32, f64)>>, Vec<u64>, HashMap<u32, u32>) {
    let mut renum: HashMap<u32, u32> = HashMap::new();
    for &lab in labels {
        let k = renum.len() as u32;
        renum.entry(lab).or_insert(k);
    }
    let new_n = renum.len();

    let mut next_n_per_node = vec![0_u64; new_n];
    let mut next_adj_map: Vec<HashMap<u32, f64>> = (0..new_n).map(|_| HashMap::new()).collect();
    for i in 0..adj.len() {
        let ui = *renum.get(&labels[i]).expect("renum");
        next_n_per_node[ui as usize] += n_per_node[i];
        for &(j, w) in &adj[i] {
            let uj = *renum.get(&labels[j as usize]).expect("renum");
            if ui == uj {
                continue; // intra-community edges become self-loops; drop
            }
            *next_adj_map[ui as usize].entry(uj).or_insert(0.0) += w;
        }
    }
    let next_adj: Vec<Vec<(u32, f64)>> = next_adj_map
        .into_iter()
        .map(|m| m.into_iter().collect::<Vec<_>>())
        .collect();
    (next_adj, next_n_per_node, renum)
}

fn compact_cpm_labels(labels: &[u32]) -> Vec<u32> {
    let mut renum: HashMap<u32, u32> = HashMap::new();
    let mut out = Vec::with_capacity(labels.len());
    for &l in labels {
        let k = renum.len() as u32;
        let id = *renum.entry(l).or_insert(k);
        out.push(id);
    }
    out
}
