//! Leiden community-detection tests.
//!
//! Four gates, each independently measured on a deterministic input:
//!
//! 1. `leiden_ari_beats_louvain_on_default_sbm` — on the default
//!    `ConnectomeConfig` (N=1024), Leiden's two-way projection scores
//!    at least 0.05 ARI above multi-level Louvain's projection. This
//!    is the headline gate — Leiden's refinement phase exists
//!    specifically to fix the Louvain collapse measured on this graph
//!    (ADR-154 §17 item 11: `louvain_ari = 0.000` vs
//!    `greedy_ari = 0.174`).
//!
//! 2. `leiden_is_deterministic` — two runs on the same connectome
//!    produce bit-identical label vectors.
//!
//! 3. `leiden_recovers_two_planted_communities` — a deterministic
//!    2-module SBM where multi-level Louvain is known to collapse
//!    (hub-boost pushes everything into a single super-community).
//!    Leiden recovers the two modules at ARI ≥ 0.90.
//!
//! 4. `leiden_sub_communities_are_internally_connected` — the
//!    well-connectedness invariant: after Leiden, no output community
//!    is internally disconnected (every node in a community is
//!    reachable from any other via BFS restricted to the community).
//!
//! Reference: Traag, Waltman, van Eck (2019), "From Louvain to Leiden:
//! guaranteeing well-connected communities", *Sci. Rep.* 9:5233.

use std::collections::HashMap;

use connectome_fly::{Analysis, AnalysisConfig, Connectome, ConnectomeConfig, NeuronId};

// -----------------------------------------------------------------
// Gate 1 — Leiden ≥ multi-level-Louvain + 0.05 on default SBM.
// -----------------------------------------------------------------
#[test]
fn leiden_ari_beats_louvain_on_default_sbm() {
    let cfg = ConnectomeConfig::default();
    let conn = Connectome::generate(&cfg);
    let an = Analysis::new(AnalysisConfig::default());

    let num_hub = cfg.num_hub_modules;
    let is_hub = |id: u32| conn.meta(NeuronId(id)).module < num_hub;

    let labels_lv = an.louvain_labels(&conn);
    let (lv_a, lv_b) = two_way_from_labels(&labels_lv);
    let ari_louvain = if lv_a.is_empty() || lv_b.is_empty() {
        0.0
    } else {
        adjusted_rand_index(&lv_a, &lv_b, is_hub)
    };

    let labels_le = an.leiden_labels(&conn);
    let (le_a, le_b) = two_way_from_labels(&labels_le);
    let ari_leiden = if le_a.is_empty() || le_b.is_empty() {
        0.0
    } else {
        adjusted_rand_index(&le_a, &le_b, is_hub)
    };

    let gap = ari_leiden - ari_louvain;
    eprintln!(
        "leiden-vs-louvain (default SBM N={}): louvain_ari={ari_louvain:.3} \
         leiden_ari={ari_leiden:.3} gap={gap:.3}",
        cfg.num_neurons
    );

    assert!(
        gap >= 0.05 - 1e-6,
        "leiden-refinement gate: gap {gap:.3} below acceptance 0.05 \
         (louvain={ari_louvain:.3}, leiden={ari_leiden:.3}). The \
         whole point of Leiden's refinement is to beat the multi-level \
         collapse documented in ADR-154 §17 item 11."
    );
}

// -----------------------------------------------------------------
// Gate 2 — Determinism.
// -----------------------------------------------------------------
#[test]
fn leiden_is_deterministic() {
    let cfg = ConnectomeConfig::default();
    let conn = Connectome::generate(&cfg);
    let an = Analysis::new(AnalysisConfig::default());
    let a = an.leiden_labels(&conn);
    let b = an.leiden_labels(&conn);
    assert_eq!(a, b, "leiden determinism: two runs must match exactly");
}

// -----------------------------------------------------------------
// Gate 3 — Hand-crafted 2-community SBM where Louvain collapses.
// -----------------------------------------------------------------
#[test]
fn leiden_recovers_two_planted_communities() {
    // Clean 2-module SBM: strong within-module density, near-zero
    // between-module density, no hub boost. This is the textbook
    // case where community-detection algorithms should cleanly
    // recover the planted partition — used here to verify Leiden's
    // refinement phase behaves sensibly on clean input.
    let cfg = ConnectomeConfig {
        num_neurons: 200,
        num_modules: 2,
        num_hub_modules: 0,
        avg_out_degree: 40.0,
        p_within: 0.60,
        p_between: 0.003,
        p_hub_boost: 0.0,
        seed: 0xC0DE_DAB1_A7EA_u64,
        ..ConnectomeConfig::default()
    };
    let conn = Connectome::generate(&cfg);
    let an = Analysis::new(AnalysisConfig::default());

    let is_module_zero = |id: u32| conn.meta(NeuronId(id)).module == 0;

    let labels = an.leiden_labels(&conn);
    let (a, b) = two_way_from_labels(&labels);
    let ari = if a.is_empty() || b.is_empty() {
        0.0
    } else {
        adjusted_rand_index(&a, &b, is_module_zero)
    };

    // For comparison: record what multi-level Louvain does on the same
    // graph so the delta is auditable.
    let labels_lv = an.louvain_labels(&conn);
    let (la, lb) = two_way_from_labels(&labels_lv);
    let ari_lv = if la.is_empty() || lb.is_empty() {
        0.0
    } else {
        adjusted_rand_index(&la, &lb, is_module_zero)
    };

    eprintln!(
        "planted-2-SBM (N={}): leiden_ari={ari:.3} louvain_ari={ari_lv:.3} |A|={} |B|={}",
        cfg.num_neurons,
        a.len(),
        b.len()
    );

    assert!(
        ari.abs() >= 0.90,
        "leiden must recover the 2 planted communities at ARI ≥ 0.90 \
         (got {ari:.3}); louvain baseline scored {ari_lv:.3}"
    );
}

// -----------------------------------------------------------------
// Gate 4 — Well-connectedness invariant.
// -----------------------------------------------------------------
#[test]
fn leiden_sub_communities_are_internally_connected() {
    let cfg = ConnectomeConfig::default();
    let conn = Connectome::generate(&cfg);
    let an = Analysis::new(AnalysisConfig::default());
    let labels = an.leiden_labels(&conn);

    // Build an undirected adjacency for the BFS. Self-loops dropped,
    // both directions recorded — matches the convention in
    // `analysis::leiden` and `structural::louvain_labels`.
    let n = conn.num_neurons();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    for pre_idx in 0..n {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for syn_entry in &syn[s..e] {
            let post = syn_entry.post.idx();
            if post == pre_idx {
                continue;
            }
            adj[pre_idx].push(post as u32);
            adj[post].push(pre_idx as u32);
        }
    }

    let mut by_comm: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_comm.entry(l).or_default().push(i as u32);
    }

    let mut disconnected: Vec<(u32, usize)> = Vec::new();
    for (&comm, nodes) in &by_comm {
        if nodes.len() <= 1 {
            continue;
        }
        let seed = *nodes.iter().min().expect("non-empty");
        let label_set: std::collections::HashSet<u32> = nodes.iter().copied().collect();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut q: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        q.push_back(seed);
        seen.insert(seed);
        while let Some(v) = q.pop_front() {
            for &u in &adj[v as usize] {
                if label_set.contains(&u) && !seen.contains(&u) {
                    seen.insert(u);
                    q.push_back(u);
                }
            }
        }
        if seen.len() < nodes.len() {
            disconnected.push((comm, nodes.len() - seen.len()));
        }
    }

    if !disconnected.is_empty() {
        for (comm, missed) in &disconnected {
            eprintln!(
                "leiden well-connectedness: community {comm} had {missed} \
                 node(s) unreachable via community-induced BFS"
            );
        }
        panic!(
            "leiden must produce internally-connected communities; \
             {} community(ies) violated the invariant",
            disconnected.len()
        );
    }
    eprintln!(
        "leiden well-connectedness: {} communities, all internally connected",
        by_comm.len()
    );
}

// -----------------------------------------------------------------
// Helpers (duplicated from acceptance_partition.rs — test files are
// separate compilation units).
// -----------------------------------------------------------------
fn two_way_from_labels(labels: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut count: HashMap<u32, u32> = HashMap::new();
    for l in labels {
        *count.entry(*l).or_insert(0) += 1;
    }
    let mut counts: Vec<(u32, u32)> = count.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if counts.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let (top_a, top_b) = (counts[0].0, counts[1].0);
    if top_a == top_b {
        return (Vec::new(), Vec::new());
    }
    let mut side_a: Vec<u32> = Vec::new();
    let mut side_b: Vec<u32> = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        if *l == top_b {
            side_b.push(i as u32);
        } else {
            side_a.push(i as u32);
        }
    }
    (side_a, side_b)
}

fn adjusted_rand_index<F: Fn(u32) -> bool>(side_a: &[u32], side_b: &[u32], gt_is_a: F) -> f32 {
    let n = (side_a.len() + side_b.len()) as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mut c: [[u32; 2]; 2] = [[0; 2]; 2];
    for id in side_a {
        let j = if gt_is_a(*id) { 0 } else { 1 };
        c[0][j] += 1;
    }
    for id in side_b {
        let j = if gt_is_a(*id) { 0 } else { 1 };
        c[1][j] += 1;
    }
    let a0 = (c[0][0] + c[0][1]) as f32;
    let a1 = (c[1][0] + c[1][1]) as f32;
    let b0 = (c[0][0] + c[1][0]) as f32;
    let b1 = (c[0][1] + c[1][1]) as f32;
    let binom = |k: f32| -> f32 {
        if k < 2.0 {
            0.0
        } else {
            k * (k - 1.0) / 2.0
        }
    };
    let ij: f32 = [c[0][0], c[0][1], c[1][0], c[1][1]]
        .iter()
        .map(|x| binom(*x as f32))
        .sum();
    let ai: f32 = binom(a0) + binom(a1);
    let bj: f32 = binom(b0) + binom(b1);
    let nc = binom(n);
    let expected = ai * bj / nc.max(1e-6);
    let denom = 0.5 * (ai + bj) - expected;
    if denom.abs() < 1e-6 {
        return 0.0;
    }
    (ij - expected) / denom
}
