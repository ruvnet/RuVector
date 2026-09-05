//! Block-sparse coboundary operator with deterministic indexing.
//!
//! Assembles the true weighted block sheaf operator from **both** endpoint
//! restriction maps (issue #669, implementation gaps 1–2): matrix-free
//! `apply`/`apply_transpose`, weighted Laplacian application, and dense
//! assembly for the reference oracle.

use crate::operator::{DenseMatrix, LinearRestriction, RestrictionMap};
use crate::{CohomologyError, EdgeKey, ValidationLimits, VertexId};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One edge block of the coboundary operator.
#[derive(Clone)]
pub struct EdgeBlock {
    /// Canonical edge key (`u < v`, oriented `u → v`).
    pub key: EdgeKey,
    /// Edge stalk dimension.
    pub dim: usize,
    /// Offset of this edge block inside `C¹`.
    pub offset: usize,
    /// Restriction from the tail stalk `F(u)` into `F(e)`.
    pub rho_u: Arc<RestrictionMap>,
    /// Restriction from the head stalk `F(v)` into `F(e)`.
    pub rho_v: Arc<RestrictionMap>,
}

/// Deterministic block-sparse coboundary `δ: C⁰ → C¹`.
///
/// Vertex blocks are laid out in ascending `VertexId` order; edge blocks in
/// ascending `EdgeKey` order. Orientation is always tail (lower id) → head:
/// `(δx)_e = ρ_v x_v − ρ_u x_u`.
#[derive(Clone)]
pub struct BlockCoboundary {
    vertices: Vec<(VertexId, usize, usize)>, // (id, dim, offset)
    vertex_pos: BTreeMap<VertexId, usize>,
    edges: Vec<EdgeBlock>,
    edge_pos: BTreeMap<EdgeKey, usize>,
    n0: usize,
    n1: usize,
}

impl BlockCoboundary {
    /// Total dimension of `C⁰` (sum of vertex stalk dimensions).
    pub fn dim_c0(&self) -> usize {
        self.n0
    }

    /// Total dimension of `C¹` (sum of edge stalk dimensions).
    pub fn dim_c1(&self) -> usize {
        self.n1
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Sorted vertex layout: `(id, stalk_dim, offset in C⁰)`.
    pub fn vertex_layout(&self) -> &[(VertexId, usize, usize)] {
        &self.vertices
    }

    /// Sorted edge blocks.
    pub fn edge_blocks(&self) -> &[EdgeBlock] {
        &self.edges
    }

    /// Position of a vertex in the sorted layout.
    pub fn vertex_position(&self, v: VertexId) -> Option<usize> {
        self.vertex_pos.get(&v).copied()
    }

    /// Position of an edge in the sorted layout.
    pub fn edge_position(&self, key: &EdgeKey) -> Option<usize> {
        self.edge_pos.get(key).copied()
    }

    /// Offset and dimension of a vertex block in `C⁰`.
    pub fn vertex_slice(&self, v: VertexId) -> Option<(usize, usize)> {
        self.vertex_pos.get(&v).map(|&p| {
            let (_, dim, off) = self.vertices[p];
            (off, dim)
        })
    }

    /// Offset and dimension of an edge block in `C¹`.
    pub fn edge_slice(&self, key: &EdgeKey) -> Option<(usize, usize)> {
        self.edge_pos.get(key).map(|&p| {
            let b = &self.edges[p];
            (b.offset, b.dim)
        })
    }

    /// `y = δ x`, matrix-free.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n0);
        debug_assert_eq!(y.len(), self.n1);
        let mut tmp = Vec::new();
        for e in &self.edges {
            let (uo, ud) = self.vertex_slice(e.key.u).expect("indexed vertex");
            let (vo, vd) = self.vertex_slice(e.key.v).expect("indexed vertex");
            let ye = &mut y[e.offset..e.offset + e.dim];
            e.rho_v.apply(&x[vo..vo + vd], ye);
            tmp.resize(e.dim, 0.0);
            e.rho_u.apply(&x[uo..uo + ud], &mut tmp);
            for (yi, ti) in ye.iter_mut().zip(tmp.iter()) {
                *yi -= ti;
            }
        }
    }

    /// `x = δᵀ y`, matrix-free.
    pub fn apply_transpose(&self, y: &[f64], x: &mut [f64]) {
        debug_assert_eq!(y.len(), self.n1);
        debug_assert_eq!(x.len(), self.n0);
        x.fill(0.0);
        let mut tmp = Vec::new();
        for e in &self.edges {
            let (uo, ud) = self.vertex_slice(e.key.u).expect("indexed vertex");
            let (vo, vd) = self.vertex_slice(e.key.v).expect("indexed vertex");
            let ye = &y[e.offset..e.offset + e.dim];
            tmp.resize(vd.max(ud), 0.0);
            e.rho_v.apply_transpose(ye, &mut tmp[..vd]);
            for (xi, ti) in x[vo..vo + vd].iter_mut().zip(tmp.iter()) {
                *xi += ti;
            }
            e.rho_u.apply_transpose(ye, &mut tmp[..ud]);
            for (xi, ti) in x[uo..uo + ud].iter_mut().zip(tmp.iter()) {
                *xi -= ti;
            }
        }
    }

    /// `out = L x = δᵀ W δ x` with per-edge scalar weights, matrix-free.
    pub fn apply_weighted_laplacian(&self, weights: &[f64], x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(weights.len(), self.edges.len());
        let mut mid = vec![0.0; self.n1];
        self.apply(x, &mut mid);
        for (i, e) in self.edges.iter().enumerate() {
            let w = weights[i];
            for m in &mut mid[e.offset..e.offset + e.dim] {
                *m *= w;
            }
        }
        self.apply_transpose(&mid, out);
    }

    /// Assemble `W^{1/2} δ` densely (reference oracle path only).
    pub fn assemble_scaled_dense(&self, weights: &[f64]) -> DenseMatrix {
        let mut a = DenseMatrix::zeros(self.n1, self.n0);
        for (i, e) in self.edges.iter().enumerate() {
            let sw = weights[i].sqrt();
            let (uo, _) = self.vertex_slice(e.key.u).expect("indexed vertex");
            let (vo, _) = self.vertex_slice(e.key.v).expect("indexed vertex");
            let du = e.rho_u.to_dense();
            let dv = e.rho_v.to_dense();
            for r in 0..e.dim {
                for c in 0..dv.cols {
                    a.set(e.offset + r, vo + c, sw * dv.get(r, c));
                }
                for c in 0..du.cols {
                    let cur = a.get(e.offset + r, uo + c);
                    a.set(e.offset + r, uo + c, cur - sw * du.get(r, c));
                }
            }
        }
        a
    }

    /// Deterministic SHA-256 hash of the full operator structure: sorted
    /// vertex layout, sorted edge keys, orientation convention, and the
    /// canonical hash of every restriction map.
    pub fn canonical_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"ruvector-cohomology/operator/v1");
        h.update((self.vertices.len() as u64).to_le_bytes());
        for (id, dim, _) in &self.vertices {
            h.update(id.to_le_bytes());
            h.update((*dim as u64).to_le_bytes());
        }
        h.update((self.edges.len() as u64).to_le_bytes());
        for e in &self.edges {
            h.update(e.key.u.to_le_bytes());
            h.update(e.key.v.to_le_bytes());
            h.update((e.dim as u64).to_le_bytes());
            h.update(e.rho_u.canonical_hash());
            h.update(e.rho_v.canonical_hash());
        }
        h.finalize().into()
    }
}

/// Builder enforcing boundary validation and deterministic layout.
pub struct CoboundaryBuilder {
    limits: ValidationLimits,
    vertex_dims: BTreeMap<VertexId, usize>,
    edges: BTreeMap<EdgeKey, (usize, Arc<RestrictionMap>, Arc<RestrictionMap>)>,
}

impl CoboundaryBuilder {
    /// New builder with default validation limits.
    pub fn new() -> Self {
        Self::with_limits(ValidationLimits::default())
    }

    /// New builder with explicit validation limits.
    pub fn with_limits(limits: ValidationLimits) -> Self {
        Self { limits, vertex_dims: BTreeMap::new(), edges: BTreeMap::new() }
    }

    /// Declare a vertex stalk.
    pub fn add_vertex(&mut self, id: VertexId, dim: usize) -> Result<&mut Self, CohomologyError> {
        if dim == 0 || dim > self.limits.max_dim {
            return Err(CohomologyError::ValidationFailed(format!(
                "vertex {id} stalk dim {dim} outside (0, {}]",
                self.limits.max_dim
            )));
        }
        if self.vertex_dims.insert(id, dim).is_some() {
            return Err(CohomologyError::DuplicateVertex(id));
        }
        Ok(self)
    }

    /// Declare an edge with both endpoint restriction maps. `rho_a` restricts
    /// the stalk of `a`, `rho_b` the stalk of `b`; they are stored against the
    /// canonical (tail, head) orientation internally.
    pub fn add_edge(
        &mut self,
        a: VertexId,
        b: VertexId,
        rho_a: RestrictionMap,
        rho_b: RestrictionMap,
    ) -> Result<&mut Self, CohomologyError> {
        let key = EdgeKey::new(a, b)?;
        let (rho_u, rho_v) = if a < b { (rho_a, rho_b) } else { (rho_b, rho_a) };
        rho_u.validate(&self.limits)?;
        rho_v.validate(&self.limits)?;
        let du = *self
            .vertex_dims
            .get(&key.u)
            .ok_or(CohomologyError::UnknownVertex(key.u))?;
        let dv = *self
            .vertex_dims
            .get(&key.v)
            .ok_or(CohomologyError::UnknownVertex(key.v))?;
        if rho_u.input_dim() != du {
            return Err(CohomologyError::DimensionMismatch {
                expected: du,
                actual: rho_u.input_dim(),
                context: format!("rho input dim at tail vertex {}", key.u),
            });
        }
        if rho_v.input_dim() != dv {
            return Err(CohomologyError::DimensionMismatch {
                expected: dv,
                actual: rho_v.input_dim(),
                context: format!("rho input dim at head vertex {}", key.v),
            });
        }
        if rho_u.output_dim() != rho_v.output_dim() {
            return Err(CohomologyError::DimensionMismatch {
                expected: rho_u.output_dim(),
                actual: rho_v.output_dim(),
                context: format!("edge stalk dim on ({}, {})", key.u, key.v),
            });
        }
        let dim = rho_u.output_dim();
        if self
            .edges
            .insert(key, (dim, Arc::new(rho_u), Arc::new(rho_v)))
            .is_some()
        {
            return Err(CohomologyError::DuplicateEdge(key.u, key.v));
        }
        Ok(self)
    }

    /// Finalize into a deterministic block coboundary.
    pub fn build(self) -> BlockCoboundary {
        let mut vertices = Vec::with_capacity(self.vertex_dims.len());
        let mut vertex_pos = BTreeMap::new();
        let mut off = 0usize;
        for (pos, (id, dim)) in self.vertex_dims.into_iter().enumerate() {
            vertices.push((id, dim, off));
            vertex_pos.insert(id, pos);
            off += dim;
        }
        let n0 = off;
        let mut edges = Vec::with_capacity(self.edges.len());
        let mut edge_pos = BTreeMap::new();
        let mut eoff = 0usize;
        for (pos, (key, (dim, rho_u, rho_v))) in self.edges.into_iter().enumerate() {
            edges.push(EdgeBlock { key, dim, offset: eoff, rho_u, rho_v });
            edge_pos.insert(key, pos);
            eoff += dim;
        }
        BlockCoboundary { vertices, vertex_pos, edges, edge_pos, n0, n1: eoff }
    }
}

impl Default for CoboundaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
