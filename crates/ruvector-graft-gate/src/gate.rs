//! Insertion-time gate variants.
//!
//! `NoGate` always admits (the baseline, matching current RuVector ANN
//! insertion behavior). `CoherenceRatio` and `MinCut` each evaluate the
//! candidate's local neighborhood, found via the same graph search the
//! index already performs to link the new node, and may reject.

use crate::graph_index::GraphIndex;
use crate::vector::cosine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateVariant {
    NoGate,
    CoherenceRatio,
    MinCut,
}

#[derive(Clone, Copy, Debug)]
pub struct GateConfig {
    pub k: usize,
    pub peakedness_threshold: f32,
    pub mincut_edge_factor: f32,
    pub mincut_reject_below: usize,
    pub bootstrap_min_index_size: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            k: crate::config::GATE_K,
            peakedness_threshold: crate::config::PEAKEDNESS_THRESHOLD,
            mincut_edge_factor: crate::config::MINCUT_EDGE_FACTOR,
            mincut_reject_below: crate::config::MINCUT_REJECT_BELOW,
            bootstrap_min_index_size: crate::config::BOOTSTRAP_MIN_INDEX_SIZE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GateDecision {
    pub admit: bool,
    pub reason: &'static str,
}

pub fn evaluate(
    variant: GateVariant,
    cfg: &GateConfig,
    index: &GraphIndex,
    search_result: &[(u32, f32)],
) -> GateDecision {
    match variant {
        GateVariant::NoGate => GateDecision {
            admit: true,
            reason: "no_gate",
        },
        GateVariant::CoherenceRatio => coherence_ratio_gate(cfg, index, search_result),
        GateVariant::MinCut => mincut_gate(cfg, index, search_result),
    }
}

fn top_k(search_result: &[(u32, f32)], k: usize) -> &[(u32, f32)] {
    &search_result[..search_result.len().min(k)]
}

/// Peakedness: the candidate's similarity to its single closest existing
/// neighbor, divided by its mean similarity across its k nearest. An
/// organically-clustered point's similarity profile across its k-NN is
/// relatively flat (all drawn from the same covariance); a point crafted
/// to align with one specific target (see `data`'s attack model) tends to
/// have one disproportionately strong match and a weaker tail. O(k), no
/// extra distance computations beyond the search that already ran.
fn coherence_ratio_gate(
    cfg: &GateConfig,
    index: &GraphIndex,
    search_result: &[(u32, f32)],
) -> GateDecision {
    let neigh = top_k(search_result, cfg.k);
    if index.len() < cfg.bootstrap_min_index_size || neigh.len() < 2 {
        return GateDecision {
            admit: true,
            reason: "bootstrap",
        };
    }
    let max_sim = neigh[0].1.max(1e-6);
    let mean_sim = (neigh.iter().map(|&(_, s)| s).sum::<f32>() / neigh.len() as f32).max(1e-6);
    let peakedness = max_sim / mean_sim;
    if peakedness > cfg.peakedness_threshold {
        GateDecision {
            admit: false,
            reason: "peakedness_exceeded",
        }
    } else {
        GateDecision {
            admit: true,
            reason: "coherent",
        }
    }
}

/// Local edge-connectivity gate. Builds the induced graph over the
/// candidate and its k nearest existing neighbors, with an edge between
/// any pair whose cosine similarity clears an adaptive threshold (the
/// neighborhood's own median pairwise similarity, scaled by
/// `mincut_edge_factor`). Finds the neighborhood's "anchor" — the member
/// best embedded among the *other* neighbors — and computes the min
/// edge-cut separating the candidate from that anchor via max-flow
/// (max-flow/min-cut duality). A well-integrated organic point typically
/// has multiple redundant paths to the anchor (direct plus via other
/// mutually-connected neighbors); a single-target attachment typically
/// has one or zero.
fn mincut_gate(cfg: &GateConfig, index: &GraphIndex, search_result: &[(u32, f32)]) -> GateDecision {
    let neigh = top_k(search_result, cfg.k);
    let n = neigh.len();
    if index.len() < cfg.bootstrap_min_index_size || n < 3 {
        return GateDecision {
            admit: true,
            reason: "bootstrap",
        };
    }

    let mut nn_sim = vec![vec![0f32; n]; n];
    let mut all_sims = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &index.vectors[neigh[i].0 as usize];
            let b = &index.vectors[neigh[j].0 as usize];
            let s = cosine(a, b);
            nn_sim[i][j] = s;
            nn_sim[j][i] = s;
            all_sims.push(s);
        }
    }
    all_sims.sort_by(|a, b| a.total_cmp(b));
    let median = if all_sims.is_empty() {
        0.0
    } else {
        all_sims[all_sims.len() / 2]
    };
    let tau = median * cfg.mincut_edge_factor;

    let mut degree = vec![0u32; n];
    #[allow(clippy::needless_range_loop)] // symmetric i,j pairwise scan over a dense matrix
    for i in 0..n {
        for j in 0..n {
            if i != j && nn_sim[i][j] >= tau {
                degree[i] += 1;
            }
        }
    }
    let anchor = (0..n).max_by_key(|&i| degree[i]).unwrap_or(0);

    // Induced subgraph: node 0 = candidate, nodes 1..=n = its k nearest
    // existing neighbors. No virtual sink — connectivity must route
    // through real (thresholded) edges only.
    let num_nodes = n + 1;
    let mut cap = vec![vec![0i32; num_nodes]; num_nodes];
    for i in 0..n {
        if neigh[i].1 >= tau {
            cap[0][i + 1] = 1;
            cap[i + 1][0] = 1;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if nn_sim[i][j] >= tau {
                cap[i + 1][j + 1] = 1;
                cap[j + 1][i + 1] = 1;
            }
        }
    }

    let flow = max_flow(&mut cap, 0, anchor + 1);
    if (flow as usize) < cfg.mincut_reject_below {
        GateDecision {
            admit: false,
            reason: "weak_attachment",
        }
    } else {
        GateDecision {
            admit: true,
            reason: "well_attached",
        }
    }
}

/// Edmonds-Karp max-flow over a dense capacity matrix (== min s-t cut by
/// max-flow/min-cut duality). Deliberately not `ruvector-mincut` — that
/// crate is a general-purpose *dynamic* min-cut engine built for graphs
/// that persist and mutate over time; this graph has at most
/// `GATE_K + 1` nodes (11 here), is rebuilt from scratch on every
/// insertion, and is thrown away immediately after one query. A bespoke
/// O(V*E^2) Edmonds-Karp pass on a graph this small is simpler, easier to
/// audit, and avoids adding a heavy dependency (petgraph, rayon,
/// crossbeam, dashmap, roaring — see `ruvector-mincut`'s Cargo.toml) to
/// the insertion hot path for a problem size where its algorithmic
/// advantages don't apply. See ADR-340 "Alternatives Considered".
fn max_flow(cap: &mut [Vec<i32>], source: usize, sink: usize) -> i32 {
    let n = cap.len();
    let mut total = 0i32;
    loop {
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            if u == sink {
                break;
            }
            for v in 0..n {
                if parent[v] == usize::MAX && cap[u][v] > 0 {
                    parent[v] = u;
                    queue.push_back(v);
                }
            }
        }
        if parent[sink] == usize::MAX {
            break;
        }
        let mut path_flow = i32::MAX;
        let mut v = sink;
        while v != source {
            let u = parent[v];
            path_flow = path_flow.min(cap[u][v]);
            v = u;
        }
        let mut v = sink;
        while v != source {
            let u = parent[v];
            cap[u][v] -= path_flow;
            cap[v][u] += path_flow;
            v = u;
        }
        total += path_flow;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_flow_on_known_triangle() {
        // 0-1-2 triangle, all edges capacity 1: mincut(0,2) should be 2
        // (direct edge 0-2 plus the 0-1-2 detour).
        let mut cap = vec![vec![0i32; 3]; 3];
        cap[0][1] = 1;
        cap[1][0] = 1;
        cap[1][2] = 1;
        cap[2][1] = 1;
        cap[0][2] = 1;
        cap[2][0] = 1;
        assert_eq!(max_flow(&mut cap, 0, 2), 2);
    }

    #[test]
    fn max_flow_on_single_bridge() {
        // 0 -> 1 only (no direct 0-2 edge), 1 <-> 2: mincut(0,2) == 1.
        let mut cap = vec![vec![0i32; 3]; 3];
        cap[0][1] = 1;
        cap[1][0] = 1;
        cap[1][2] = 1;
        cap[2][1] = 1;
        assert_eq!(max_flow(&mut cap, 0, 2), 1);
    }

    #[test]
    fn max_flow_disconnected_is_zero() {
        let cap = vec![vec![0i32; 3]; 3];
        let mut cap = cap;
        assert_eq!(max_flow(&mut cap, 0, 2), 0);
    }

    #[test]
    fn small_index_always_bootstraps_admit() {
        let idx = GraphIndex::new(8, 8);
        let cfg = GateConfig::default();
        let d1 = evaluate(GateVariant::CoherenceRatio, &cfg, &idx, &[]);
        let d2 = evaluate(GateVariant::MinCut, &cfg, &idx, &[]);
        assert!(d1.admit && d1.reason == "bootstrap");
        assert!(d2.admit && d2.reason == "bootstrap");
    }

    #[test]
    fn nogate_always_admits() {
        let idx = GraphIndex::new(8, 8);
        let cfg = GateConfig::default();
        let d = evaluate(GateVariant::NoGate, &cfg, &idx, &[]);
        assert!(d.admit);
    }
}
