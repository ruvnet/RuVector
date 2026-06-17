//! Metric-independent nested-dissection ordering (ADR-197, Phase 1).
//!
//! M0 uses a self-contained BFS-layer separator finder: from a pseudo-peripheral
//! start, the BFS frontier at a balanced layer is a valid vertex separator
//! (removing a whole layer disconnects earlier layers from later ones). This is
//! intentionally simple — CCH *correctness* is independent of separator quality;
//! only search-space size depends on it. At M1 scale this finder is swapped for
//! `ruvector-mincut`'s expander/cluster balanced cuts.

use crate::graph::{Graph, NodeId};
use std::collections::VecDeque;

/// Cells smaller than this become leaves (no further dissection).
pub const LEAF: usize = 8;

/// Separator-finding strategy. `BfsLayer` (M0 baseline) takes a whole BFS
/// frontier — fine on grids, degenerate on low-diameter graphs. `Balanced`
/// grows a half-size region and takes only its boundary, giving small
/// separators on sparse graphs (the M1 fix; ADR-197).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeparatorKind {
    BfsLayer,
    Balanced,
}

/// A node of the separator decomposition tree. Every original vertex belongs to
/// exactly one node (either as a separator member or a leaf member), which is
/// what lets POIs be bucketed unambiguously during query (see `query.rs`).
#[derive(Clone, Debug)]
pub struct SepNode {
    /// Separator vertices (original ids) "owned" by this node.
    pub separator: Vec<NodeId>,
    /// Child node indices in `SepTree::nodes`.
    pub children: Vec<usize>,
    /// All vertices in this node's subtree (the cell), separator included.
    pub cell: Vec<NodeId>,
}

#[derive(Clone, Debug)]
pub struct SepTree {
    pub nodes: Vec<SepNode>,
    pub root: usize,
}

/// Result of ordering: `order[rank] = original id` (rank 0 = contracted first =
/// lowest importance) and the separator decomposition tree.
pub struct Ordering {
    pub order: Vec<NodeId>,
    pub sep_tree: SepTree,
}

/// Compute a nested-dissection order over all `n` vertices of `g` using the
/// `Balanced` separator (the default since M1).
#[must_use]
pub fn nested_dissection(g: &Graph) -> Ordering {
    nested_dissection_kind(g, SeparatorKind::Balanced)
}

/// Nested dissection with an explicit separator strategy (for A/B attribution).
#[must_use]
pub fn nested_dissection_kind(g: &Graph, kind: SeparatorKind) -> Ordering {
    let mut builder = NdBuilder {
        g,
        kind,
        order: Vec::with_capacity(g.n),
        nodes: Vec::new(),
    };
    let all: Vec<NodeId> = (0..g.n as NodeId).collect();
    let root = builder.dissect(all);
    Ordering {
        order: builder.order,
        sep_tree: SepTree { nodes: builder.nodes, root },
    }
}

struct NdBuilder<'a> {
    g: &'a Graph,
    kind: SeparatorKind,
    order: Vec<NodeId>,
    nodes: Vec<SepNode>,
}

impl NdBuilder<'_> {
    /// Dissect `verts`; append ranks to `order` (children before separators, so
    /// separators rank higher); return the new node index.
    fn dissect(&mut self, verts: Vec<NodeId>) -> usize {
        // Disconnected cell → recurse per component under an empty-separator node.
        let comps = connected_components(self.g, &verts);
        if comps.len() > 1 {
            let children: Vec<usize> = comps.into_iter().map(|c| self.dissect(c)).collect();
            return self.push_node(Vec::new(), children, verts);
        }

        if verts.len() <= LEAF {
            return self.leaf(verts);
        }

        let sep_result = match self.kind {
            SeparatorKind::BfsLayer => bfs_separator(self.g, &verts),
            SeparatorKind::Balanced => balanced_separator(self.g, &verts),
        };
        match sep_result {
            Some((sep, a, b)) => {
                let ca = self.dissect(a);
                let cb = self.dissect(b);
                // Separators appended AFTER both subtrees → higher rank.
                let mut sorted_sep = sep.clone();
                sorted_sep.sort_unstable();
                self.order.extend_from_slice(&sorted_sep);
                self.push_node(sorted_sep, vec![ca, cb], verts)
            }
            // No usable balanced separator (e.g. clique-like) → treat as leaf.
            None => self.leaf(verts),
        }
    }

    fn leaf(&mut self, verts: Vec<NodeId>) -> usize {
        let mut sorted = verts.clone();
        sorted.sort_unstable();
        self.order.extend_from_slice(&sorted);
        self.push_node(sorted, Vec::new(), verts)
    }

    fn push_node(&mut self, separator: Vec<NodeId>, children: Vec<usize>, cell: Vec<NodeId>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(SepNode { separator, children, cell });
        id
    }
}

/// Connected components of the subgraph induced by `verts`.
fn connected_components(g: &Graph, verts: &[NodeId]) -> Vec<Vec<NodeId>> {
    let in_set = membership(g.n, verts);
    let mut seen = vec![false; g.n];
    let mut comps = Vec::new();
    for &s in verts {
        if seen[s as usize] {
            continue;
        }
        let mut comp = Vec::new();
        let mut q = VecDeque::from([s]);
        seen[s as usize] = true;
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &(v, _) in &g.adj[u as usize] {
                if in_set[v as usize] && !seen[v as usize] {
                    seen[v as usize] = true;
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

/// Find a balanced vertex separator of the connected cell `verts` via BFS layers.
/// Returns `(separator, side_a, side_b)` with both sides non-empty, or `None`.
fn bfs_separator(g: &Graph, verts: &[NodeId]) -> Option<(Vec<NodeId>, Vec<NodeId>, Vec<NodeId>)> {
    let in_set = membership(g.n, verts);
    // Pseudo-peripheral start: farthest vertex from an arbitrary one.
    let start = farthest(g, &in_set, verts[0]);
    let (layer, max_layer) = bfs_layers(g, &in_set, start);
    if max_layer == 0 {
        return None; // single layer (e.g. clique) — no layer separator exists
    }

    // Pick the split layer L (1..max_layer) whose "before" side is closest to half.
    let half = verts.len() / 2;
    let mut counts = vec![0usize; max_layer + 1];
    for &v in verts {
        counts[layer[v as usize] as usize] += 1;
    }
    let mut prefix = 0usize;
    let mut best_l = 1usize;
    let mut best_bal = usize::MAX;
    for l in 1..max_layer {
        prefix += counts[l - 1]; // vertices in layers < l
        let bal = prefix.abs_diff(half);
        if bal < best_bal {
            best_bal = bal;
            best_l = l;
        }
    }
    let l = best_l as u32;

    let mut sep = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &v in verts {
        match layer[v as usize].cmp(&l) {
            std::cmp::Ordering::Less => a.push(v),
            std::cmp::Ordering::Equal => sep.push(v),
            std::cmp::Ordering::Greater => b.push(v),
        }
    }
    if a.is_empty() || b.is_empty() || sep.is_empty() {
        return None;
    }
    Some((sep, a, b))
}

/// Balanced separator: grow a half-size region in BFS order, then take only its
/// boundary (vertices with a neighbour outside the region) as the vertex
/// separator. On sparse graphs the boundary is small; on any graph the two
/// sides are balanced, avoiding the path-like elimination tree that the
/// whole-layer strategy produces on low-diameter graphs.
fn balanced_separator(g: &Graph, verts: &[NodeId]) -> Option<(Vec<NodeId>, Vec<NodeId>, Vec<NodeId>)> {
    let in_set = membership(g.n, verts);
    let start = farthest(g, &in_set, verts[0]);
    let visit = bfs_order(g, &in_set, start);
    if visit.len() < 2 {
        return None;
    }
    let half = verts.len() / 2;
    let mut in_a = vec![false; g.n];
    for &v in &visit[..half] {
        in_a[v as usize] = true;
    }
    let mut sep = Vec::new();
    let mut a = Vec::new();
    for &v in &visit[..half] {
        // Boundary iff some in-cell neighbour lies outside region A.
        let on_boundary = g.adj[v as usize]
            .iter()
            .any(|&(u, _)| in_set[u as usize] && !in_a[u as usize]);
        if on_boundary {
            sep.push(v);
        } else {
            a.push(v);
        }
    }
    let b: Vec<NodeId> = visit[half..].to_vec();
    if a.is_empty() || b.is_empty() || sep.is_empty() {
        return None;
    }
    Some((sep, a, b))
}

/// BFS visitation order within the induced subgraph, starting at `start`.
fn bfs_order(g: &Graph, in_set: &[bool], start: NodeId) -> Vec<NodeId> {
    let mut order = Vec::new();
    let mut seen = vec![false; g.n];
    seen[start as usize] = true;
    let mut q = VecDeque::from([start]);
    while let Some(u) = q.pop_front() {
        order.push(u);
        for &(v, _) in &g.adj[u as usize] {
            if in_set[v as usize] && !seen[v as usize] {
                seen[v as usize] = true;
                q.push_back(v);
            }
        }
    }
    order
}

/// BFS hop-distance layers within the induced subgraph. Returns `(layer, max)`.
fn bfs_layers(g: &Graph, in_set: &[bool], start: NodeId) -> (Vec<u32>, usize) {
    let mut layer = vec![u32::MAX; g.n];
    layer[start as usize] = 0;
    let mut q = VecDeque::from([start]);
    let mut max_layer = 0u32;
    while let Some(u) = q.pop_front() {
        let lu = layer[u as usize];
        for &(v, _) in &g.adj[u as usize] {
            if in_set[v as usize] && layer[v as usize] == u32::MAX {
                layer[v as usize] = lu + 1;
                max_layer = max_layer.max(lu + 1);
                q.push_back(v);
            }
        }
    }
    (layer, max_layer as usize)
}

fn farthest(g: &Graph, in_set: &[bool], from: NodeId) -> NodeId {
    let (layer, _) = bfs_layers(g, in_set, from);
    let mut best = from;
    let mut best_d = 0u32;
    for (v, &d) in layer.iter().enumerate() {
        if d != u32::MAX && d > best_d {
            best_d = d;
            best = v as NodeId;
        }
    }
    best
}

fn membership(n: usize, verts: &[NodeId]) -> Vec<bool> {
    let mut m = vec![false; n];
    for &v in verts {
        m[v as usize] = true;
    }
    m
}
