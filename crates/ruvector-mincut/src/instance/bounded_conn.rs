//! Pluggable connectivity backends for [`super::bounded::BoundedInstanceImpl`].
//!
//! Nightly research (2026-09-11-mincut-partition-determinism, "Next
//! Research" item 2) flagged `BoundedInstance`'s ad hoc BFS-based
//! disconnection check as a candidate replacement site for
//! [`crate::connectivity::polylog::PolylogConnectivity`], which was already
//! in-tree but not wired into any `ProperCutInstance`. This module isolates
//! that one variable: the trait below is the only thing that changes
//! between `BoundedInstance` (baseline), `PolylogBoundedInstance` and
//! `EttBoundedInstance` (candidates A and B). Everything else in
//! `bounded.rs` — the LocalKCut search loop, brute-force fallback, boundary
//! caching, certificate tracking — is shared, unmodified code.
//!
//! Each backend owns a private copy of the edge set so its `is_connected`
//! answer never depends on `BoundedInstanceImpl`'s own `adjacency`/`vertices`
//! fields, keeping the three variants structurally comparable.

use crate::connectivity::polylog::PolylogConnectivity;
use crate::connectivity::DynamicConnectivity;
use crate::graph::VertexId;
use std::collections::{HashMap, HashSet, VecDeque};

/// A backend that answers "is the graph currently connected?" incrementally
/// as edges are inserted and deleted, instead of a fresh traversal each time.
pub trait ConnBackend: Send + Sync {
    /// Create a new, empty backend.
    fn new() -> Self
    where
        Self: Sized;
    /// Record an edge insertion.
    fn insert_edge(&mut self, u: VertexId, v: VertexId);
    /// Record an edge deletion.
    fn delete_edge(&mut self, u: VertexId, v: VertexId);
    /// Whether all vertices seen so far are in one connected component.
    fn is_connected(&self) -> bool;
}

/// Baseline: the BFS-from-scratch check `BoundedInstance` used before this
/// experiment, reimplemented here verbatim so the baseline variant exercises
/// the identical asymptotic behavior through the identical trait boundary.
#[derive(Default)]
pub struct BfsConn {
    adjacency: HashMap<VertexId, Vec<VertexId>>,
    vertices: HashSet<VertexId>,
}

impl ConnBackend for BfsConn {
    fn new() -> Self {
        Self::default()
    }

    fn insert_edge(&mut self, u: VertexId, v: VertexId) {
        self.vertices.insert(u);
        self.vertices.insert(v);
        self.adjacency.entry(u).or_default().push(v);
        self.adjacency.entry(v).or_default().push(u);
    }

    fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        if let Some(n) = self.adjacency.get_mut(&u) {
            if let Some(pos) = n.iter().position(|&x| x == v) {
                n.remove(pos);
            }
        }
        if let Some(n) = self.adjacency.get_mut(&v) {
            if let Some(pos) = n.iter().position(|&x| x == u) {
                n.remove(pos);
            }
        }
    }

    fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return true;
        }
        let start = *self.vertices.iter().min().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(cur) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&cur) {
                for &nb in neighbors {
                    if visited.insert(nb) {
                        queue.push_back(nb);
                    }
                }
            }
        }
        visited.len() == self.vertices.len()
    }
}

/// Candidate A: `PolylogConnectivity` (arXiv:2510.08297), the structure the
/// prior nightly named as untested.
pub struct PolylogConn(PolylogConnectivity);

impl ConnBackend for PolylogConn {
    fn new() -> Self {
        Self(PolylogConnectivity::new())
    }

    fn insert_edge(&mut self, u: VertexId, v: VertexId) {
        self.0.insert_edge(u, v);
    }

    fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        self.0.delete_edge(u, v);
    }

    fn is_connected(&self) -> bool {
        self.0.is_connected()
    }
}

/// Candidate B: `DynamicConnectivity` (Euler Tour Tree + union-find), the
/// backend `MinCutWrapper` already uses one layer up — included so the
/// experiment compares PolylogConnectivity against an already-proven
/// in-tree alternative, not only against the naive baseline.
pub struct EttConn(DynamicConnectivity);

impl ConnBackend for EttConn {
    fn new() -> Self {
        Self(DynamicConnectivity::new())
    }

    fn insert_edge(&mut self, u: VertexId, v: VertexId) {
        self.0.insert_edge(u, v);
    }

    fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        self.0.delete_edge(u, v);
    }

    fn is_connected(&self) -> bool {
        self.0.is_connected()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise<C: ConnBackend>() {
        let mut c = C::new();
        // Empty-graph `is_connected()` is deliberately not asserted here:
        // `PolylogConnectivity` treats zero components as vacuously
        // connected (`component_count <= 1`) while `DynamicConnectivity`
        // does not (`component_count == 1`, i.e. 0 != 1). Callers in
        // `BoundedInstanceImpl::query()` never observe this gap because
        // the call site always guards with `&& !self.vertices.is_empty()`.
        c.insert_edge(0, 1);
        c.insert_edge(1, 2);
        assert!(c.is_connected());
        c.insert_edge(3, 4);
        assert!(!c.is_connected());
        c.insert_edge(2, 3);
        assert!(c.is_connected());
        c.delete_edge(2, 3);
        assert!(!c.is_connected());
    }

    #[test]
    fn bfs_conn_matches_contract() {
        exercise::<BfsConn>();
    }

    #[test]
    fn polylog_conn_matches_contract() {
        exercise::<PolylogConn>();
    }

    #[test]
    fn ett_conn_matches_contract() {
        exercise::<EttConn>();
    }
}
