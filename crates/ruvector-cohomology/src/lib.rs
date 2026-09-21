//! # Dynamic Cohomological Error Correction
//!
//! Semantic syndrome extraction, sparse repair, and mincut containment over
//! dynamic cellular sheaves (RFC: ruvnet/RuVector#669).
//!
//! ## Mathematical model
//!
//! A cellular sheaf on a graph `G = (V, E)` assigns a vector space (stalk) to
//! every vertex and edge, and restriction maps `ρ_{v→e}: F(v) → F(e)`. For an
//! oriented edge `e = (u, v)` the coboundary operator is
//!
//! ```text
//! (δx)_e = ρ_{v→e} x_v − ρ_{u→e} x_u
//! ```
//!
//! The correct cohomological semantics on a graph are:
//!
//! - `H⁰(G; F) = ker δ` — the space of **global sections**. For a linear
//!   sheaf the zero section is always global, so `H⁰` is never "empty";
//!   nontrivial `H⁰` means nonzero globally consistent assignments exist.
//! - `H¹(G; F) = C¹ / im δ` — edge data that cannot be written as the
//!   coboundary of any vertex assignment. Nontrivial `H¹` does **not** mean
//!   "no global section exists"; it means some edge observations are
//!   unexplainable by every vertex assignment.
//!
//! Production consistency is therefore an **affine** problem. Given observed
//! edge relationships `b ∈ C¹` and confidence weights `W`, solve
//!
//! ```text
//! x* = argmin_x ‖W^{1/2}(δx − b)‖²
//! ```
//!
//! The irreducible contradiction is the weighted **syndrome**
//!
//! ```text
//! s = b − δ (δᵀWδ)† δᵀW b = b − δx*
//! ```
//!
//! `‖s‖_W = 0` means a globally coherent explanation exists; `‖s‖_W > 0`
//! certifies an irreducible contradiction whose support localizes the
//! responsible constraints.
//!
//! ## Crate layout
//!
//! - [`operator`]: serializable linear restriction operators (no closures)
//! - [`block_sparse`]: deterministic block-sparse coboundary and sheaf Laplacian
//! - [`affine`]: observations, weights, gauge — the affine sheaf system
//! - [`syndrome`]: weighted least-squares syndrome engine with certificates
//! - [`harmonic`]: harmonic (contradiction) subspace and canonicalization
//! - [`repair`]: group-sparse semantic repair with cost and protection policy
//! - [`dynamic`]: bounded-cell dynamic maintenance with exact `flush()`
//! - [`partition`]: deterministic bounded cell partition
//! - [`witness`]: canonical, quantized, SHA-256-hashed contradiction witnesses
//! - [`transport`]: orthogonal frame transport and cycle-consistency guards
//! - [`mincut_bridge`]: syndrome→tension conversion and quarantine boundaries
//! - [`solvers`]: CG, MINRES, LOBPCG, rank-revealing QR, group-sparse ADMM
//!
//! ## Determinism
//!
//! All indexing is derived from sorted identifiers, all solvers are
//! deterministic (fixed iteration order, committed seeds), and witnesses store
//! explicitly quantized values, so repeated runs — and any parallel driver
//! that merges results by canonical key order — produce identical witness
//! hashes.

// Dense numeric kernels use index loops over multiple coupled buffers where
// iterator chains obscure the linear algebra; the bounds are asserted.
#![allow(clippy::needless_range_loop)]

pub mod affine;
pub mod block_sparse;
pub mod dynamic;
pub mod harmonic;
pub mod mincut_bridge;
pub mod operator;
pub mod partition;
pub mod repair;
pub mod solvers;
pub mod syndrome;
pub mod transport;
pub mod witness;

use serde::{Deserialize, Serialize};

/// Vertex identifier. Deterministic layouts sort by this id.
pub type VertexId = u64;

/// Canonical undirected edge key. Orientation is always from the lower id
/// (`u`, tail) to the higher id (`v`, head): `(δx)_e = ρ_v x_v − ρ_u x_u`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EdgeKey {
    /// Tail vertex (lower id).
    pub u: VertexId,
    /// Head vertex (higher id).
    pub v: VertexId,
}

impl EdgeKey {
    /// Create a canonical edge key. Returns an error for self-loops.
    pub fn new(a: VertexId, b: VertexId) -> Result<Self, CohomologyError> {
        if a == b {
            return Err(CohomologyError::InvalidValue(format!(
                "self-loop on vertex {a} is not a valid 1-cell"
            )));
        }
        Ok(if a < b { Self { u: a, v: b } } else { Self { u: b, v: a } })
    }

    /// Endpoint of the edge (tail = lower id).
    pub fn endpoints(&self) -> (VertexId, VertexId) {
        (self.u, self.v)
    }
}

/// Which endpoint of an edge a restriction map belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Endpoint {
    /// The tail (lower vertex id).
    Tail,
    /// The head (higher vertex id).
    Head,
}

/// Gauge policy fixing the representative of the least-squares solution.
///
/// The syndrome `s = b − δx*` is gauge-invariant; the gauge only pins `x*`
/// inside `x* + ker δ` so that witnesses are reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GaugePolicy {
    /// Minimum-norm (Moore–Penrose) solution. CG started from the zero
    /// vector converges to this representative because every Krylov iterate
    /// stays in `range(L)`.
    #[default]
    MinimumNorm,
}

/// Validation limits enforced at every system boundary before allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationLimits {
    /// Maximum stalk dimension accepted for a vertex or edge.
    pub max_dim: usize,
    /// Maximum accepted operator norm bound for a restriction map.
    pub max_operator_norm: f64,
    /// Tolerance for orthogonality validation of frames and orthogonal maps.
    pub orthogonality_tol: f64,
    /// Maximum accepted absolute value for weights and observations.
    pub max_value: f64,
}

impl Default for ValidationLimits {
    fn default() -> Self {
        Self {
            max_dim: 4096,
            max_operator_norm: 1e6,
            orthogonality_tol: 1e-8,
            max_value: 1e12,
        }
    }
}

/// Errors produced by the cohomology engine.
#[derive(Debug, thiserror::Error)]
pub enum CohomologyError {
    /// A dimension did not match the declared stalk layout.
    #[error("dimension mismatch in {context}: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
        /// Human-readable location of the mismatch.
        context: String,
    },
    /// Referenced vertex does not exist.
    #[error("unknown vertex {0}")]
    UnknownVertex(VertexId),
    /// Referenced edge does not exist.
    #[error("unknown edge ({0}, {1})")]
    UnknownEdge(VertexId, VertexId),
    /// Vertex already exists.
    #[error("duplicate vertex {0}")]
    DuplicateVertex(VertexId),
    /// Edge already exists.
    #[error("duplicate edge ({0}, {1})")]
    DuplicateEdge(VertexId, VertexId),
    /// A value failed boundary validation.
    #[error("invalid value: {0}")]
    InvalidValue(String),
    /// A structural validation failed (norm, orthogonality, sparsity, size).
    #[error("validation failed: {0}")]
    ValidationFailed(String),
    /// A solver failed to reach its tolerance.
    #[error("solver failure: {0}")]
    SolverFailure(String),
    /// Modification of a protected edge without authorization.
    #[error("edge ({0}, {1}) is protected and the policy does not authorize modification")]
    ProtectedEdge(VertexId, VertexId),
    /// Operation requires a flushed (synchronized) state.
    #[error("operation requires flush(): {0}")]
    NotFlushed(String),
}

pub use affine::{AffineSheafSystem, EdgeField, EdgeWeights};
pub use block_sparse::{BlockCoboundary, CoboundaryBuilder};
pub use dynamic::{
    CohomologyEdit, DynamicCohomology, DynamicSheafEngine, EditReceipt, EngineConfig,
    SyndromeEstimate,
};
pub use harmonic::{HarmonicBasis, SpectralSummary};
pub use mincut_bridge::{
    ContainmentPolicy, InterventionReceipt, QuarantineBoundary, TensionGraph, TensionWeights,
};
pub use operator::{DenseMatrix, LinearRestriction, RestrictionMap};
pub use partition::CellPartition;
pub use repair::{EdgeCost, RepairPlan, RepairPolicy};
pub use solvers::{ResidualCertificate, SolverConfig};
pub use syndrome::{compute_syndrome, compute_syndrome_dense, EdgeContribution, SyndromeResult};
pub use witness::CohomologyWitness;
