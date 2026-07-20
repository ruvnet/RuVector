use crate::{dot, types::Node};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ─── heap wrappers ────────────────────────────────────────────────────────────

/// Max-heap entry: larger score pops first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxE(pub f32, pub usize);
impl Eq for MaxE {}
impl PartialOrd for MaxE {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { self.0.partial_cmp(&o.0) }
}
impl Ord for MaxE {
    fn cmp(&self, o: &Self) -> Ordering { self.partial_cmp(o).unwrap_or(Ordering::Equal) }
}

/// Min-heap entry: smaller score pops first (via reversed comparison).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinE(pub f32, pub usize);
impl Eq for MinE {}
impl PartialOrd for MinE {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { o.0.partial_cmp(&self.0) }
}
impl Ord for MinE {
    fn cmp(&self, o: &Self) -> Ordering { self.partial_cmp(o).unwrap_or(Ordering::Equal) }
}

// ─── NSW graph ────────────────────────────────────────────────────────────────

/// Navigable Small World graph — base navigation structure for TENG.
///
/// Construction is O(n · ef_constr · d).  Search is O(ef_search · d).
pub struct NswGraph {
    /// adj\[i\] = outgoing neighbor ids for node i.
    pub adj: Vec<Vec<usize>>,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl NswGraph {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self { adj: Vec::new(), m, ef_construction, ef_search }
    }

    /// Build NSW by sequential greedy insertion.
    pub fn build(&mut self, nodes: &[Node]) {
        let n = nodes.len();
        self.adj = vec![Vec::new(); n];
        for id in 1..n {
            let ep = 0_usize;
            let candidates = self.search_partial(nodes, id, ep, self.ef_construction);
            for (nbr, _) in candidates.iter().take(self.m) {
                let nbr = *nbr;
                if !self.adj[id].contains(&nbr) { self.adj[id].push(nbr); }
                if !self.adj[nbr].contains(&id) { self.adj[nbr].push(id); }
                if self.adj[nbr].len() > self.m * 2 {
                    self.trim(nodes, nbr);
                }
            }
        }
    }

    fn trim(&mut self, nodes: &[Node], id: usize) {
        let v = &nodes[id].vector;
        let mut scored: Vec<(usize, f32)> = self.adj[id]
            .iter()
            .map(|&j| (j, dot(v, &nodes[j].vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(self.m);
        self.adj[id] = scored.into_iter().map(|(j, _)| j).collect();
    }

    /// Beam search over nodes 0..query_node_id (construction phase).
    fn search_partial(&self, nodes: &[Node], qid: usize, ep: usize, ef: usize) -> Vec<(usize, f32)> {
        let n_active = qid;
        if n_active == 0 { return Vec::new(); }
        let query = &nodes[qid].vector;

        let mut visited = vec![false; n_active];
        let mut cands: BinaryHeap<MaxE> = BinaryHeap::new();
        let mut results: BinaryHeap<MinE> = BinaryHeap::new();

        let init = ep.min(n_active - 1);
        let s = dot(query, &nodes[init].vector);
        visited[init] = true;
        cands.push(MaxE(s, init));
        results.push(MinE(s, init));

        while let Some(MaxE(csim, cid)) = cands.pop() {
            let worst = results.peek().map(|r| r.0).unwrap_or(f32::NEG_INFINITY);
            if results.len() >= ef && csim < worst { break; }
            if cid >= self.adj.len() { continue; }
            for &nbr in &self.adj[cid] {
                if nbr < n_active && !visited[nbr] {
                    visited[nbr] = true;
                    let sim = dot(query, &nodes[nbr].vector);
                    let worst = results.peek().map(|r| r.0).unwrap_or(f32::NEG_INFINITY);
                    if results.len() < ef || sim > worst {
                        cands.push(MaxE(sim, nbr));
                        results.push(MinE(sim, nbr));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results.into_iter().map(|e| (e.1, e.0)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out
    }

    /// Standard beam search over all nodes (query phase).
    pub fn search(&self, nodes: &[Node], query: &[f32], ef: usize) -> Vec<(usize, f32)> {
        let n = nodes.len();
        if n == 0 { return Vec::new(); }

        let mut visited = vec![false; n];
        let mut cands: BinaryHeap<MaxE> = BinaryHeap::new();
        let mut results: BinaryHeap<MinE> = BinaryHeap::new();

        let s = dot(query, &nodes[0].vector);
        visited[0] = true;
        cands.push(MaxE(s, 0));
        results.push(MinE(s, 0));

        while let Some(MaxE(csim, cid)) = cands.pop() {
            let worst = results.peek().map(|r| r.0).unwrap_or(f32::NEG_INFINITY);
            if results.len() >= ef && csim < worst { break; }
            for &nbr in &self.adj[cid] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    let sim = dot(query, &nodes[nbr].vector);
                    let worst = results.peek().map(|r| r.0).unwrap_or(f32::NEG_INFINITY);
                    if results.len() < ef || sim > worst {
                        cands.push(MaxE(sim, nbr));
                        results.push(MinE(sim, nbr));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results.into_iter().map(|e| (e.1, e.0)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out
    }
}
