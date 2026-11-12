//! Stoer-Wagner global minimum cut on a small dense weighted undirected graph.
//!
//! Unlike an S-T max-flow/min-cut (used by `ruvector-namespace-merge` for
//! query-time routing, where source and sink are fixed by construction), the
//! global min cut has no designated terminals: it finds the cheapest
//! bipartition of *all* nodes, i.e. the single weakest link in the whole
//! graph. That is the right primitive for write-time admission: "is the
//! incoming point (and whichever existing clusters it drags with it) the
//! weakest-attached part of the current memory graph?" has no natural
//! source/sink to fix in advance.
//!
//! Complexity: O(V^3) per invocation via the standard Stoer-Wagner
//! max-adjacency-search + merge phases. Graphs here are `n_clusters + 1`
//! nodes (bounded by the admission policy's cluster cap), so this is cheap.

/// Dense symmetric non-negative weight matrix, row-major `n x n`.
#[derive(Clone)]
pub struct WeightMatrix {
    pub n: usize,
    w: Vec<f64>,
}

impl WeightMatrix {
    pub fn new(n: usize) -> Self {
        WeightMatrix {
            n,
            w: vec![0.0; n * n],
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.w[i * self.n + j]
    }

    #[inline]
    pub fn set_sym(&mut self, i: usize, j: usize, weight: f64) {
        self.w[i * self.n + j] = weight;
        self.w[j * self.n + i] = weight;
    }
}

/// Result of a global min-cut computation.
pub struct MinCutResult {
    /// Total weight of edges crossing the cut.
    pub weight: f64,
    /// Number of edges crossing the cut (used to report an average cut
    /// weight, which is what admission policies threshold on).
    pub crossing_edges: usize,
    /// `side[i] == true` means original node `i` is on the smaller-labelled
    /// partition returned by the last merge phase.
    pub side: Vec<bool>,
}

/// Stoer-Wagner global minimum cut. `graph.n` must be >= 2.
///
/// Returns `None` if `graph.n < 2` (a single node has no cut to find).
pub fn global_min_cut(graph: &WeightMatrix) -> Option<MinCutResult> {
    let n = graph.n;
    if n < 2 {
        return None;
    }

    // `w` is mutated in place by vertex merges; `groups[i]` lists the
    // original node indices currently merged into active vertex `i`.
    let mut w = graph.w_clone();
    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<usize> = (0..n).collect();

    let mut best_weight = f64::INFINITY;
    let mut best_side: Vec<usize> = Vec::new();

    while active.len() > 1 {
        let (order, last_weight, last_two) = min_cut_phase(&w, &active);
        if last_weight < best_weight {
            best_weight = last_weight;
            best_side = groups[last_two].clone();
        }
        // Merge the last two vertices in `order` into the second-to-last.
        let (s, t) = (order[order.len() - 2], order[order.len() - 1]);
        merge_vertices(&mut w, n, s, t);
        let merged_t_group = std::mem::take(&mut groups[t]);
        groups[s].extend(merged_t_group);
        active.retain(|&v| v != t);
    }

    let mut side = vec![false; n];
    for idx in best_side {
        side[idx] = true;
    }
    let crossing_edges = count_crossing_edges(graph, &side);

    Some(MinCutResult {
        weight: best_weight,
        crossing_edges,
        side,
    })
}

fn count_crossing_edges(graph: &WeightMatrix, side: &[bool]) -> usize {
    let n = graph.n;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if side[i] != side[j] && graph.get(i, j) > 0.0 {
                count += 1;
            }
        }
    }
    count.max(1)
}

impl WeightMatrix {
    fn w_clone(&self) -> Vec<f64> {
        self.w.clone()
    }
}

/// One maximum-adjacency-search phase. Returns (visit order, cut weight of
/// last vertex added, index of the last vertex added).
fn min_cut_phase(w: &[f64], active: &[usize]) -> (Vec<usize>, f64, usize) {
    let n = active.len();
    let mut in_a = vec![false; active.len()];
    let mut weights = vec![0f64; active.len()]; // weight from A to each vertex, indexed by position in `active`
    let mut order = Vec::with_capacity(n);

    // Start from the first active vertex.
    in_a[0] = true;
    order.push(active[0]);
    for pos in 1..n {
        weights[pos] = w[active[0] * row_stride(w, active) + active[pos]];
    }

    let mut last = active[0];
    let mut second_last = active[0];
    for _ in 1..n {
        // Pick the not-yet-added vertex with max weight to A.
        let mut sel_pos = usize::MAX;
        let mut sel_w = f64::NEG_INFINITY;
        for pos in 0..n {
            if !in_a[pos] && weights[pos] > sel_w {
                sel_w = weights[pos];
                sel_pos = pos;
            }
        }
        in_a[sel_pos] = true;
        second_last = last;
        last = active[sel_pos];
        order.push(last);

        // Update weights of remaining vertices.
        let stride = row_stride(w, active);
        for pos in 0..n {
            if !in_a[pos] {
                weights[pos] += w[last * stride + active[pos]];
            }
        }
    }

    let cut_weight = weights
        .iter()
        .enumerate()
        .find(|(pos, _)| active[*pos] == last)
        .map(|(_, wgt)| *wgt)
        .unwrap_or(0.0);
    let _ = second_last;
    (order, cut_weight, last)
}

/// The weight matrix is always stored with the *original* `n`, even though
/// only `active` indices are meaningful after merges — this recovers that
/// stride from the flat buffer length.
fn row_stride(w: &[f64], _active: &[usize]) -> usize {
    (w.len() as f64).sqrt().round() as usize
}

/// Fold vertex `t`'s edges into `s` (standard Stoer-Wagner merge) and zero
/// out `t`'s row/column so it is never revisited.
fn merge_vertices(w: &mut [f64], n: usize, s: usize, t: usize) {
    for k in 0..n {
        if k == s || k == t {
            continue;
        }
        let wt = w[t * n + k];
        w[s * n + k] += wt;
        w[k * n + s] += wt;
    }
    for k in 0..n {
        w[t * n + k] = 0.0;
        w[k * n + t] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_node_cut_equals_single_edge() {
        let mut m = WeightMatrix::new(2);
        m.set_sym(0, 1, 3.5);
        let r = global_min_cut(&m).unwrap();
        assert!((r.weight - 3.5).abs() < 1e-9);
        assert_eq!(r.crossing_edges, 1);
    }

    #[test]
    fn two_bridged_pairs_hand_verified() {
        // Two tight pairs {0,1} and {2,3} (internal weight 5 each), joined by
        // two weak bridges 0-2 and 1-3 (weight 1 each). Every possible
        // bipartition of 4 nodes, checked by hand:
        //   {0}|{1,2,3}: 0-1(5)+0-2(1)          = 6
        //   {1}|{0,2,3}: 0-1(5)+1-3(1)          = 6
        //   {2}|{0,1,3}: 2-3(5)+0-2(1)          = 6
        //   {3}|{0,1,2}: 2-3(5)+1-3(1)          = 6
        //   {0,1}|{2,3}: 0-2(1)+1-3(1)          = 2   <- minimum
        //   {0,2}|{1,3}: 0-1(5)+2-3(5)          = 10
        //   {0,3}|{1,2}: 0-1(5)+2-3(5)          = 10
        let mut m = WeightMatrix::new(4);
        m.set_sym(0, 1, 5.0);
        m.set_sym(2, 3, 5.0);
        m.set_sym(0, 2, 1.0);
        m.set_sym(1, 3, 1.0);
        let r = global_min_cut(&m).unwrap();
        assert!(
            (r.weight - 2.0).abs() < 1e-6,
            "expected min cut weight 2.0, got {}",
            r.weight
        );
        assert_eq!(r.crossing_edges, 2);
        assert_eq!(r.side[0], r.side[1], "0 and 1 must be on the same side");
        assert_eq!(r.side[2], r.side[3], "2 and 3 must be on the same side");
        assert_ne!(
            r.side[0], r.side[2],
            "the two pairs must be on opposite sides"
        );
    }

    #[test]
    fn isolated_weak_node_is_found() {
        // Two tight clusters {0,1,2} and {3,4,5}, plus a weakly attached
        // outlier node 6 (single low-weight edge to node 0). The global
        // min cut must isolate node 6.
        let mut m = WeightMatrix::new(7);
        for &(i, j) in &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)] {
            m.set_sym(i, j, 10.0);
        }
        m.set_sym(0, 3, 1.0); // weak bridge between clusters
        m.set_sym(0, 6, 0.1); // very weak outlier
        let r = global_min_cut(&m).unwrap();
        assert!((r.weight - 0.1).abs() < 1e-6);
        // Node 6 must be alone on its side of the cut.
        let side6 = r.side[6];
        let alone = (0..7).filter(|&i| r.side[i] == side6).count();
        assert_eq!(alone, 1, "node 6 should be isolated by the min cut");
    }

    #[test]
    fn complete_graph_uniform_weights() {
        let n = 5;
        let mut m = WeightMatrix::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                m.set_sym(i, j, 1.0);
            }
        }
        let r = global_min_cut(&m).unwrap();
        // Min cut of a uniform-weight complete graph isolates a single
        // vertex: weight = n - 1.
        assert!((r.weight - (n as f64 - 1.0)).abs() < 1e-6);
    }
}
