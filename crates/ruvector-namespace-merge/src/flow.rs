//! Integer max-flow via Edmonds-Karp (BFS augmentation) for small graphs.
//!
//! Used by [`MinCutRoute`] to find the min S-T cut over the namespace
//! similarity graph. Graph size is O(namespaces) — typically 5–20 nodes —
//! so the O(VE²) complexity is irrelevant in practice.
//!
//! After max-flow, the source-side reachable set from BFS on the residual
//! graph gives the S-side of the min cut (namespaces to search).

use std::collections::VecDeque;

/// A directed capacity graph with integer capacities.
pub struct FlowGraph {
    pub n: usize,
    /// `cap[u * n + v]` = remaining capacity on edge u→v.
    cap: Vec<i64>,
}

impl FlowGraph {
    pub fn new(n: usize) -> Self {
        FlowGraph {
            n,
            cap: vec![0; n * n],
        }
    }

    /// Add directed edge u→v with capacity `c`. Also adds reverse edge v→u
    /// with capacity 0 (for the residual graph).
    pub fn add_edge(&mut self, u: usize, v: usize, c: i64) {
        self.cap[u * self.n + v] += c;
    }

    /// Add undirected edge (bidirectional with capacity `c` in each direction).
    pub fn add_undirected(&mut self, u: usize, v: usize, c: i64) {
        self.cap[u * self.n + v] += c;
        self.cap[v * self.n + u] += c;
    }

    fn cap_at(&self, u: usize, v: usize) -> i64 {
        self.cap[u * self.n + v]
    }

    fn push(&mut self, u: usize, v: usize, f: i64) {
        self.cap[u * self.n + v] -= f;
        self.cap[v * self.n + u] += f;
    }

    /// BFS: find shortest augmenting path from `s` to `t`.
    /// Returns (parent array, flow pushed). 0 if no path found.
    fn bfs(&self, s: usize, t: usize, parent: &mut [usize]) -> i64 {
        let n = self.n;
        parent.iter_mut().for_each(|p| *p = usize::MAX);
        parent[s] = s;
        let mut queue = VecDeque::new();
        queue.push_back((s, i64::MAX));
        while let Some((u, flow)) = queue.pop_front() {
            for (v, parent_v) in parent.iter_mut().enumerate().take(n) {
                if *parent_v == usize::MAX && self.cap_at(u, v) > 0 {
                    *parent_v = u;
                    let new_flow = flow.min(self.cap_at(u, v));
                    if v == t {
                        return new_flow;
                    }
                    queue.push_back((v, new_flow));
                }
            }
        }
        0
    }

    /// Edmonds-Karp max-flow from `s` to `t`. Returns total flow value.
    pub fn max_flow(&mut self, s: usize, t: usize) -> i64 {
        let mut flow = 0i64;
        let mut parent = vec![usize::MAX; self.n];
        loop {
            let f = self.bfs(s, t, &mut parent);
            if f == 0 {
                break;
            }
            flow += f;
            // trace path and push flow
            let mut v = t;
            while v != s {
                let u = parent[v];
                self.push(u, v, f);
                v = u;
            }
        }
        flow
    }

    /// After running max_flow, return the set of nodes reachable from `s`
    /// in the residual graph — these are on the source side of the min cut.
    pub fn source_side(&self, s: usize) -> Vec<bool> {
        let n = self.n;
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        visited[s] = true;
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            for (v, is_visited) in visited.iter_mut().enumerate().take(n) {
                if !*is_visited && self.cap_at(u, v) > 0 {
                    *is_visited = true;
                    queue.push_back(v);
                }
            }
        }
        visited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_max_flow() {
        // 4-node flow: s=0, t=3
        // 0→1 cap 3, 0→2 cap 2, 1→3 cap 2, 2→3 cap 3, 1→2 cap 1
        // Paths: 0→1→3 (2 units) + 0→2→3 (2 units) + 0→1→2→3 (1 unit) = 5
        // Min-cut = cut {0} → caps 3+2=5. Correct answer: 5.
        let mut g = FlowGraph::new(4);
        g.add_edge(0, 1, 3);
        g.add_edge(0, 2, 2);
        g.add_edge(1, 3, 2);
        g.add_edge(2, 3, 3);
        g.add_edge(1, 2, 1);
        let f = g.max_flow(0, 3);
        assert_eq!(f, 5);
    }

    #[test]
    fn test_source_side() {
        // Path: 0 → 1 → 2, capacity 1 each
        let mut g = FlowGraph::new(3);
        g.add_edge(0, 1, 1);
        g.add_edge(1, 2, 1);
        g.max_flow(0, 2);
        let side = g.source_side(0);
        // After saturating the only path, only node 0 is reachable from source
        assert!(side[0]);
        assert!(!side[2]);
    }
}
