//! Edit classes, receipts, and estimates for the dynamic engine.

use crate::operator::{DenseMatrix, RestrictionMap};
use crate::{Endpoint, VertexId};
use sha2::{Digest, Sha256};

/// An edit to the dynamic sheaf (issue #669 edit classes 1–8).
#[derive(Debug, Clone)]
pub enum CohomologyEdit {
    /// Insert a vertex with the given stalk dimension.
    InsertVertex {
        /// Vertex id.
        id: VertexId,
        /// Stalk dimension.
        dim: usize,
    },
    /// Remove a vertex and all incident edges.
    RemoveVertex {
        /// Vertex id.
        id: VertexId,
    },
    /// Insert an edge with both restriction maps, weight, and observation.
    InsertEdge {
        /// One endpoint.
        a: VertexId,
        /// Other endpoint.
        b: VertexId,
        /// Restriction from the stalk of `a`.
        rho_a: RestrictionMap,
        /// Restriction from the stalk of `b`.
        rho_b: RestrictionMap,
        /// Confidence weight (positive).
        weight: f64,
        /// Observed edge relationship `b_e` (edge stalk dimension).
        observation: Vec<f64>,
    },
    /// Remove an edge.
    RemoveEdge {
        /// One endpoint.
        a: VertexId,
        /// Other endpoint.
        b: VertexId,
    },
    /// Replace one endpoint restriction map of an edge.
    UpdateRestriction {
        /// One endpoint.
        a: VertexId,
        /// Other endpoint.
        b: VertexId,
        /// Which endpoint (in canonical tail/head orientation).
        endpoint: Endpoint,
        /// The new map.
        map: RestrictionMap,
    },
    /// Replace the observation on an edge.
    UpdateObservation {
        /// One endpoint.
        a: VertexId,
        /// Other endpoint.
        b: VertexId,
        /// New observation block.
        value: Vec<f64>,
    },
    /// Replace the weight on an edge.
    UpdateWeight {
        /// One endpoint.
        a: VertexId,
        /// Other endpoint.
        b: VertexId,
        /// New positive weight.
        weight: f64,
    },
    /// Orthogonal frame update on a vertex: incident restriction maps are
    /// transported exactly (`ρ ← ρ Rᵀ`, `R = Q_new Q_oldᵀ`) so the syndrome
    /// is invariant.
    UpdateFrame {
        /// The vertex changing frame.
        vertex: VertexId,
        /// Old orthonormal frame.
        q_old: DenseMatrix,
        /// New orthonormal frame.
        q_new: DenseMatrix,
    },
}

/// Receipt for an applied (or rejected) edit.
#[derive(Debug, Clone)]
pub struct EditReceipt {
    /// Whether the edit was accepted.
    pub accepted: bool,
    /// Topology epoch after the edit.
    pub topology_epoch: u64,
    /// Observation epoch after the edit.
    pub observation_epoch: u64,
    /// Cells marked dirty by this edit.
    pub dirty_cells: Vec<usize>,
    /// Deterministic hash of the edit content.
    pub edit_hash: [u8; 32],
    /// Rejection reason, if not accepted.
    pub error: Option<String>,
}

/// Cheap estimate available without a global solve.
#[derive(Debug, Clone)]
pub struct SyndromeEstimate {
    /// Sum of cached syndrome energy over currently-clean cells.
    pub clean_energy: f64,
    /// Number of dirty cells awaiting synchronization.
    pub dirty_cell_count: usize,
    /// `true` when nothing is dirty and the cache matches both epochs — the
    /// estimate then equals the exact flushed energy.
    pub is_exact: bool,
    /// Topology epoch.
    pub topology_epoch: u64,
    /// Observation epoch.
    pub observation_epoch: u64,
}

/// Deterministic SHA-256 hash of an edit's content.
pub fn edit_hash(edit: &CohomologyEdit) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"ruvector-cohomology/edit/v1");
    match edit {
        CohomologyEdit::InsertVertex { id, dim } => {
            h.update([0u8]);
            h.update(id.to_le_bytes());
            h.update((*dim as u64).to_le_bytes());
        }
        CohomologyEdit::RemoveVertex { id } => {
            h.update([1u8]);
            h.update(id.to_le_bytes());
        }
        CohomologyEdit::InsertEdge { a, b, rho_a, rho_b, weight, observation } => {
            use crate::operator::LinearRestriction;
            h.update([2u8]);
            h.update(a.to_le_bytes());
            h.update(b.to_le_bytes());
            h.update(rho_a.canonical_hash());
            h.update(rho_b.canonical_hash());
            h.update(weight.to_bits().to_le_bytes());
            for v in observation {
                h.update(v.to_bits().to_le_bytes());
            }
        }
        CohomologyEdit::RemoveEdge { a, b } => {
            h.update([3u8]);
            h.update(a.to_le_bytes());
            h.update(b.to_le_bytes());
        }
        CohomologyEdit::UpdateRestriction { a, b, endpoint, map } => {
            use crate::operator::LinearRestriction;
            h.update([4u8]);
            h.update(a.to_le_bytes());
            h.update(b.to_le_bytes());
            h.update([matches!(endpoint, Endpoint::Head) as u8]);
            h.update(map.canonical_hash());
        }
        CohomologyEdit::UpdateObservation { a, b, value } => {
            h.update([5u8]);
            h.update(a.to_le_bytes());
            h.update(b.to_le_bytes());
            for v in value {
                h.update(v.to_bits().to_le_bytes());
            }
        }
        CohomologyEdit::UpdateWeight { a, b, weight } => {
            h.update([6u8]);
            h.update(a.to_le_bytes());
            h.update(b.to_le_bytes());
            h.update(weight.to_bits().to_le_bytes());
        }
        CohomologyEdit::UpdateFrame { vertex, q_old, q_new } => {
            h.update([7u8]);
            h.update(vertex.to_le_bytes());
            q_old.hash_into(&mut h);
            q_new.hash_into(&mut h);
        }
    }
    h.finalize().into()
}
