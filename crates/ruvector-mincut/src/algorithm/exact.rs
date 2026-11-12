//! Sparse Stoer-Wagner global minimum cut with a matching vertex partition.
//!
//! Each phase uses a maximum-adjacency heap. Storage is O(n + m), avoiding
//! a dense n-by-n matrix. Full recomputation is polynomial, not subpolynomial.

use std::collections::BTreeMap;

/// One entry per active vertex. Updates move an existing entry instead of
/// allocating stale candidates; heap memory stays O(n), even on dense graphs.
struct IndexedHeap {
    vertices: Vec<usize>,
    positions: Vec<usize>,
}
impl IndexedHeap {
    fn new(n: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(n),
            positions: vec![usize::MAX; n],
        }
    }
    fn reset(&mut self, active: &[bool]) {
        self.vertices.clear();
        self.positions.fill(usize::MAX);
        for (v, &enabled) in active.iter().enumerate() {
            if enabled {
                self.positions[v] = self.vertices.len();
                self.vertices.push(v);
            }
        }
        // Ascending vertex IDs are already a heap for equal zero weights.
    }
    fn better(a: usize, b: usize, weights: &[f64]) -> bool {
        weights[a]
            .total_cmp(&weights[b])
            .then_with(|| b.cmp(&a))
            .is_gt()
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.vertices.swap(a, b);
        self.positions[self.vertices[a]] = a;
        self.positions[self.vertices[b]] = b;
    }
    fn increase(&mut self, v: usize, weights: &[f64]) {
        let mut i = self.positions[v];
        while i > 0 {
            let parent = (i - 1) / 2;
            if !Self::better(v, self.vertices[parent], weights) {
                break;
            }
            self.swap(i, parent);
            i = parent;
        }
    }
    fn pop(&mut self, weights: &[f64]) -> usize {
        let v = self.vertices[0];
        let last = self.vertices.len() - 1;
        self.swap(0, last);
        self.vertices.pop();
        self.positions[v] = usize::MAX;
        let mut i = 0;
        while 2 * i + 1 < self.vertices.len() {
            let mut child = 2 * i + 1;
            if child + 1 < self.vertices.len()
                && Self::better(self.vertices[child + 1], self.vertices[child], weights)
            {
                child += 1;
            }
            if !Self::better(self.vertices[child], self.vertices[i], weights) {
                break;
            }
            self.swap(i, child);
            i = child;
        }
        v
    }
}

/// Linear preprocessing with an exact lower-bound certificate. In a connected
/// simple graph every cut either removes a bridge or at least two edges, so
/// lambda >= min(lightest_bridge, 2 * lightest_edge). Return only if a concrete
/// bridge or singleton cut meets that bound. Otherwise use the general solver.
/// DFS is iterative to support long paths without overflowing the native/WASM stack.
fn certified_cut(n: usize, edges: &[(usize, usize, f64)]) -> Option<(f64, Vec<usize>)> {
    let mut adjacency = vec![Vec::new(); n];
    let mut degrees = vec![0.0; n];
    let mut lightest = f64::INFINITY;
    for &(u, v, w) in edges {
        adjacency[u].push((v, w));
        adjacency[v].push((u, w));
        degrees[u] += w;
        degrees[v] += w;
        lightest = lightest.min(w);
    }
    let mut enter = vec![usize::MAX; n];
    let mut low = vec![0; n];
    let mut end = vec![0; n];
    let mut parent = vec![usize::MAX; n];
    let mut next = vec![0; n];
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![0];
    enter[0] = 0;
    order.push(0);
    let mut bridge = None;
    let mut bridge_weight = f64::INFINITY;
    while let Some(&v) = stack.last() {
        if next[v] < adjacency[v].len() {
            let (u, _) = adjacency[v][next[v]];
            next[v] += 1;
            if u == parent[v] {
                continue;
            }
            if enter[u] == usize::MAX {
                parent[u] = v;
                enter[u] = order.len();
                low[u] = enter[u];
                order.push(u);
                stack.push(u);
            } else {
                low[v] = low[v].min(enter[u]);
            }
        } else {
            stack.pop();
            end[v] = order.len();
            let p = parent[v];
            if p != usize::MAX {
                low[p] = low[p].min(low[v]);
                if low[v] > enter[p] {
                    let w = adjacency[p][next[p] - 1].1;
                    if bridge.is_none() || w < bridge_weight {
                        bridge_weight = w;
                        bridge = Some(v);
                    }
                }
            }
        }
    }
    if order.len() != n {
        return Some((0.0, order));
    }
    if let Some(v) = bridge {
        if bridge_weight <= 2.0 * lightest {
            return Some((bridge_weight, order[enter[v]..end[v]].to_vec()));
        }
    }
    let bound = bridge_weight.min(2.0 * lightest);
    let v = (0..n)
        .min_by(|&a, &b| degrees[a].total_cmp(&degrees[b]).then(a.cmp(&b)))
        .unwrap();
    if degrees[v] <= bound {
        return Some((degrees[v], vec![v]));
    }
    None
}

/// Simple undirected graph, endpoints index `0..n`, finite nonnegative weights.
/// Returns infinity for fewer than two vertices, matching DynamicMinCut.
pub(super) fn minimum_cut(n: usize, edges: &[(usize, usize, f64)]) -> (f64, Vec<usize>) {
    if n < 2 {
        return (f64::INFINITY, (0..n).collect());
    }
    if let Some(cut) = certified_cut(n, edges) {
        return cut;
    }
    let mut adjacency = vec![BTreeMap::<usize, f64>::new(); n];
    for &(u, v, weight) in edges {
        *adjacency[u].entry(v).or_default() += weight;
        *adjacency[v].entry(u).or_default() += weight;
    }
    let mut active = vec![true; n];
    let mut groups: Vec<Vec<usize>> = (0..n).map(|v| vec![v]).collect();
    let mut best = f64::INFINITY;
    let mut best_side = vec![0];
    // Reuse per-phase work buffers rather than allocating a traversal per cut.
    let mut weights = vec![0.0; n];
    let mut added = vec![false; n];
    let mut heap = IndexedHeap::new(n);

    for remaining in (2..=n).rev() {
        weights.fill(0.0);
        added.fill(false);
        heap.reset(&active);
        let mut previous = 0;
        for step in 0..remaining {
            let v = heap.pop(&weights);
            if step + 1 == remaining {
                if weights[v] < best {
                    best = weights[v];
                    best_side.clone_from(&groups[v]);
                }
                if best == 0.0 {
                    return (best, best_side);
                }
                // Contract the last vertex into the penultimate vertex.
                let neighbors = std::mem::take(&mut adjacency[v]);
                for (neighbor, weight) in neighbors {
                    adjacency[neighbor].remove(&v);
                    if neighbor != previous {
                        *adjacency[previous].entry(neighbor).or_default() += weight;
                        *adjacency[neighbor].entry(previous).or_default() += weight;
                    }
                }
                let merged = std::mem::take(&mut groups[v]);
                groups[previous].extend(merged);
                active[v] = false;
                break;
            }
            added[v] = true;
            for (&neighbor, &weight) in &adjacency[v] {
                if !added[neighbor] {
                    weights[neighbor] += weight;
                    heap.increase(neighbor, &weights);
                }
            }
            previous = v;
        }
    }
    (best, best_side)
}
