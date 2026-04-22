//! Backbone (spanning forest) maintenance via union-find.
//!
//! The [`Backbone`] keeps a set of *always-present* edges that guarantee
//! global connectivity in the sparsifier. Non-backbone edges can be freely
//! sampled and dropped; backbone edges are never removed unless the
//! corresponding edge is deleted from the full graph.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::traits::BackboneStrategy;

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

/// Weighted union-find (disjoint-set) with path compression and union by rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    num_components: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            num_components: n,
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        while self.parent.len() < n {
            let id = self.parent.len();
            self.parent.push(id);
            self.rank.push(0);
            self.num_components += 1;
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    /// Union two sets. Returns `true` if they were in different components.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        // Union by rank.
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        self.num_components -= 1;
        true
    }

    fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}

// ---------------------------------------------------------------------------
// Backbone
// ---------------------------------------------------------------------------

/// A spanning-forest backbone that maintains connectivity guarantees.
///
/// When an edge is inserted, it is added to the backbone forest if and only
/// if it connects two previously disconnected components. When a backbone
/// edge is deleted, the backbone attempts to find a replacement edge from
/// the set of known non-backbone edges to restore connectivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backbone {
    /// Union-find for connectivity tracking.
    uf: UnionFind,
    /// Set of edges `(min(u,v), max(u,v))` that are in the backbone.
    backbone_edges: HashSet<(usize, usize)>,
    /// Set of all known non-backbone edges for replacement search.
    non_backbone_edges: HashSet<(usize, usize)>,
}

impl Backbone {
    /// Create a new backbone for up to `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            uf: UnionFind::new(n),
            backbone_edges: HashSet::new(),
            non_backbone_edges: HashSet::new(),
        }
    }

    /// Canonical edge key with `u <= v`.
    #[inline]
    fn edge_key(u: usize, v: usize) -> (usize, usize) {
        if u <= v {
            (u, v)
        } else {
            (v, u)
        }
    }

    /// Rebuild the union-find from the current backbone edges.
    ///
    /// This is O(backbone_edges * alpha(n)) and is used after a backbone
    /// edge deletion when no replacement is found.
    fn rebuild_uf(&mut self) {
        let n = self.uf.parent.len();
        self.uf = UnionFind::new(n);
        for &(u, v) in &self.backbone_edges {
            self.uf.union(u, v);
        }
    }
}

impl BackboneStrategy for Backbone {
    fn insert_edge(&mut self, u: usize, v: usize, _weight: f64) -> bool {
        let key = Self::edge_key(u, v);
        self.ensure_capacity(u.max(v) + 1);

        if !self.uf.connected(u, v) {
            // Bridge: add to backbone.
            self.uf.union(u, v);
            self.backbone_edges.insert(key);
            true
        } else {
            // Not a bridge: track as non-backbone.
            self.non_backbone_edges.insert(key);
            false
        }
    }

    fn delete_edge(&mut self, u: usize, v: usize, _weight: f64) -> bool {
        let key = Self::edge_key(u, v);

        if self.non_backbone_edges.remove(&key) {
            // Non-backbone edge removed; nothing to repair.
            return false;
        }

        if !self.backbone_edges.remove(&key) {
            // Edge was not tracked at all.
            return false;
        }

        // Backbone edge removed. Rebuild UF without it and try to find
        // a replacement from non-backbone edges.
        self.rebuild_uf();

        // Try to find a replacement edge that reconnects the components.
        let mut replacement = None;
        for &(a, b) in &self.non_backbone_edges {
            if !self.uf.connected(a, b) {
                replacement = Some((a, b));
                break;
            }
        }

        if let Some((a, b)) = replacement {
            let rkey = Self::edge_key(a, b);
            self.non_backbone_edges.remove(&rkey);
            self.backbone_edges.insert(rkey);
            self.uf.union(a, b);
        }

        true
    }

    fn is_backbone_edge(&self, u: usize, v: usize) -> bool {
        self.backbone_edges.contains(&Self::edge_key(u, v))
    }

    fn num_components(&self) -> usize {
        self.uf.num_components
    }

    fn connected(&mut self, u: usize, v: usize) -> bool {
        if u >= self.uf.parent.len() || v >= self.uf.parent.len() {
            return false;
        }
        self.uf.connected(u, v)
    }

    fn backbone_edge_count(&self) -> usize {
        self.backbone_edges.len()
    }

    fn ensure_capacity(&mut self, n: usize) {
        self.uf.ensure_capacity(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_connectivity() {
        let mut bb = Backbone::new(4);
        // 0-1-2-3 path
        assert!(bb.insert_edge(0, 1, 1.0)); // bridge
        assert!(bb.insert_edge(1, 2, 1.0)); // bridge
        assert!(bb.insert_edge(2, 3, 1.0)); // bridge
        assert!(!bb.insert_edge(0, 3, 1.0)); // cycle, not bridge

        assert_eq!(bb.num_components(), 1);
        assert_eq!(bb.backbone_edge_count(), 3);
        assert!(bb.is_backbone_edge(0, 1));
        assert!(!bb.is_backbone_edge(0, 3));
    }

    #[test]
    fn test_backbone_deletion_with_replacement() {
        let mut bb = Backbone::new(3);
        bb.insert_edge(0, 1, 1.0); // backbone
        bb.insert_edge(1, 2, 1.0); // backbone
        bb.insert_edge(0, 2, 1.0); // non-backbone (cycle)

        assert_eq!(bb.backbone_edge_count(), 2);

        // Delete backbone edge 0-1; should find replacement 0-2.
        let modified = bb.delete_edge(0, 1, 1.0);
        assert!(modified);
        assert_eq!(bb.num_components(), 1);
        // The replacement edge should now be in backbone.
        assert!(bb.is_backbone_edge(0, 2));
    }

    #[test]
    fn test_backbone_deletion_no_replacement() {
        let mut bb = Backbone::new(3);
        bb.insert_edge(0, 1, 1.0);
        bb.insert_edge(1, 2, 1.0);

        // Delete backbone edge; no replacement available.
        let modified = bb.delete_edge(0, 1, 1.0);
        assert!(modified);
        assert_eq!(bb.num_components(), 2);
    }

    #[test]
    fn test_auto_grow() {
        let mut bb = Backbone::new(0);
        assert!(bb.insert_edge(5, 10, 1.0));
        assert!(bb.connected(5, 10));
        assert_eq!(bb.backbone_edge_count(), 1);
    }
}
