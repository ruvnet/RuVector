//! Tenant graphs: a validated, canonical undirected edge list plus the
//! limits that bound it, and the rebuild into `ruvector-mincut`.
//!
//! `DynamicGraph` is not serializable, so the edge list is the source of
//! truth; the solver is rebuilt from it on load (ADR-351 §3 rv-mincut).

use crate::error::{AnalyticsError, LimitKind, Result};
use ruvector_mincut::{DynamicMinCut, MinCutBuilder, VertexId};
use serde::{Deserialize, Serialize};

/// Opaque graph identity bound into every persisted record (tenant graph
/// uid, e.g. derived from the edge-store `collection_uid`).
pub type GraphUid = [u8; 16];

/// One undirected edge in canonical form: `u < v`, finite `w >= 0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeRecord {
    /// Smaller endpoint.
    pub u: VertexId,
    /// Larger endpoint.
    pub v: VertexId,
    /// Nonnegative finite weight.
    pub w: f64,
}

/// Size limits for a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphLimits {
    /// Maximum distinct vertices.
    pub max_vertices: u64,
    /// Maximum undirected edges.
    pub max_edges: u64,
}

impl GraphLimits {
    /// Inline limits, ADR-351 §10: `<= 200k edges, <= 50k nodes`.
    pub const INLINE: GraphLimits = GraphLimits {
        max_vertices: 50_000,
        max_edges: 200_000,
    };
    /// Async-job limits (a job still runs inside one 128 MB isolate).
    pub const JOB: GraphLimits = GraphLimits {
        max_vertices: 250_000,
        max_edges: 250_000,
    };

    /// Check counts against the limits (used on the manifest before any
    /// chunk is decoded, and again after validation).
    pub fn check(&self, vertices: u64, edges: u64) -> Result<()> {
        if edges > self.max_edges {
            return Err(AnalyticsError::LimitExceeded {
                kind: LimitKind::Edges,
                actual: edges,
                max: self.max_edges,
            });
        }
        if vertices > self.max_vertices {
            return Err(AnalyticsError::LimitExceeded {
                kind: LimitKind::Vertices,
                actual: vertices,
                max: self.max_vertices,
            });
        }
        Ok(())
    }
}

/// A validated tenant graph. Edges are sorted by `(u, v)` and unique, so the
/// encoding and every derived digest are canonical.
#[derive(Debug, Clone, PartialEq)]
pub struct TenantGraph {
    uid: GraphUid,
    revision: u64,
    edges: Vec<EdgeRecord>,
    vertex_count: u64,
    /// Manifest digest of the snapshot this graph was decoded from (set only
    /// by [`crate::decode_graph`]); jobs pin to it.
    snapshot: Option<[u8; 32]>,
}

impl TenantGraph {
    /// Validate and canonicalize caller edges. Rejects self-loops,
    /// non-finite or negative weights and duplicate undirected edges (the
    /// three inputs `DynamicGraph::insert_edge` would refuse), then applies
    /// the limits. The edge limit is checked before any allocation.
    pub fn from_edges(
        uid: GraphUid,
        revision: u64,
        edges: &[(VertexId, VertexId, f64)],
        limits: &GraphLimits,
    ) -> Result<Self> {
        limits.check(0, edges.len() as u64)?;
        let mut out = Vec::with_capacity(edges.len());
        for &(a, b, w) in edges {
            if a == b {
                return Err(AnalyticsError::Invalid("self-loop edge"));
            }
            if !w.is_finite() || w < 0.0 {
                return Err(AnalyticsError::Invalid(
                    "edge weight must be finite and >= 0",
                ));
            }
            let (u, v) = if a < b { (a, b) } else { (b, a) };
            out.push(EdgeRecord { u, v, w });
        }
        out.sort_unstable_by_key(|e| (e.u, e.v));
        if out.windows(2).any(|p| (p[0].u, p[0].v) == (p[1].u, p[1].v)) {
            return Err(AnalyticsError::Invalid("duplicate undirected edge"));
        }
        Self::from_canonical(uid, revision, out, limits)
    }

    /// Build from edges already known to be canonical (strictly increasing,
    /// `u < v`, valid weights) — the decoder guarantees this.
    pub(crate) fn from_canonical(
        uid: GraphUid,
        revision: u64,
        edges: Vec<EdgeRecord>,
        limits: &GraphLimits,
    ) -> Result<Self> {
        let vertex_count = count_vertices(&edges);
        limits.check(vertex_count, edges.len() as u64)?;
        Ok(TenantGraph {
            uid,
            revision,
            edges,
            vertex_count,
            snapshot: None,
        })
    }

    /// Graph identity.
    pub fn uid(&self) -> &GraphUid {
        &self.uid
    }

    /// Manifest digest of the snapshot this graph was decoded from; `None`
    /// for a graph built from caller edges (not yet persisted).
    pub fn snapshot_digest(&self) -> Option<&[u8; 32]> {
        self.snapshot.as_ref()
    }

    pub(crate) fn with_snapshot(mut self, digest: [u8; 32]) -> Self {
        self.snapshot = Some(digest);
        self
    }

    /// Revision this edge list represents.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Canonical edges.
    pub fn edges(&self) -> &[EdgeRecord] {
        &self.edges
    }

    /// Distinct vertices (vertices exist only as edge endpoints).
    pub fn vertex_count(&self) -> u64 {
        self.vertex_count
    }

    /// Undirected edge count.
    pub fn edge_count(&self) -> u64 {
        self.edges.len() as u64
    }

    /// Whether every weight is exactly `1.0` (enables the compact encoding).
    pub fn unit_weights(&self) -> bool {
        self.edges.iter().all(|e| e.w == 1.0)
    }

    /// Edges as solver tuples.
    pub fn solver_edges(&self) -> Vec<(VertexId, VertexId, f64)> {
        self.edges.iter().map(|e| (e.u, e.v, e.w)).collect()
    }

    /// Rebuild the exact solver through `MinCutBuilder`. Validation above
    /// guarantees the builder cannot refuse an edge; a refusal is mapped to
    /// `Solver` rather than a panic. `parallel(false)`: nothing on this
    /// path uses threads, and wasm32 has none.
    pub fn rebuild_exact(&self) -> Result<DynamicMinCut> {
        MinCutBuilder::new()
            .exact()
            .parallel(false)
            .with_edges(self.solver_edges())
            .build()
            .map_err(|_| AnalyticsError::Solver)
    }
}

fn count_vertices(edges: &[EdgeRecord]) -> u64 {
    let mut ids: Vec<VertexId> = Vec::with_capacity(edges.len() * 2);
    for e in edges {
        ids.push(e.u);
        ids.push(e.v);
    }
    ids.sort_unstable();
    ids.dedup();
    ids.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const UID: GraphUid = [7; 16];

    #[test]
    fn canonicalizes_and_rejects_bad_edges() {
        let g = TenantGraph::from_edges(UID, 1, &[(3, 1, 2.0), (1, 2, 1.0)], &GraphLimits::INLINE)
            .unwrap();
        assert_eq!(g.edges()[0], EdgeRecord { u: 1, v: 2, w: 1.0 });
        assert_eq!(g.edges()[1], EdgeRecord { u: 1, v: 3, w: 2.0 });
        assert_eq!(g.vertex_count(), 3);
        let bad: [&[(u64, u64, f64)]; 5] = [
            &[(1, 1, 1.0)],
            &[(1, 2, f64::NAN)],
            &[(1, 2, -1.0)],
            &[(1, 2, f64::INFINITY)],
            &[(1, 2, 1.0), (2, 1, 3.0)],
        ];
        for edges in bad {
            let e = TenantGraph::from_edges(UID, 1, edges, &GraphLimits::INLINE).unwrap_err();
            assert_eq!(e.status(), 400, "{e:?}");
        }
    }

    #[test]
    fn limits_are_413() {
        let lim = GraphLimits {
            max_vertices: 3,
            max_edges: 2,
        };
        let e = TenantGraph::from_edges(UID, 1, &[(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)], &lim)
            .unwrap_err();
        assert!(matches!(
            e,
            AnalyticsError::LimitExceeded {
                kind: LimitKind::Edges,
                ..
            }
        ));
        assert_eq!(e.status(), 413);
        let e = TenantGraph::from_edges(UID, 1, &[(1, 2, 1.0), (3, 4, 1.0)], &lim).unwrap_err();
        assert!(matches!(
            e,
            AnalyticsError::LimitExceeded {
                kind: LimitKind::Vertices,
                ..
            }
        ));
        assert_eq!(e.status(), 413);
    }
}
