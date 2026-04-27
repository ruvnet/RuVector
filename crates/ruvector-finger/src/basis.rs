use crate::dist::{dot, l2_sq, saxpy};

const GRAM_SCHMIDT_EPS: f32 = 1e-7;

/// Precomputed residual basis for one graph node.
///
/// For node `u` with neighbors `{v_1, ..., v_M}`:
///   basis: K orthonormal vectors spanning the edge-residual subspace
///   edge_projs: [m*K .. (m+1)*K] = projections of (v_m - u) onto each basis vec
///   edge_norms_sq: ||v_m - u||^2 for each neighbor m
///
/// Memory per node: K*dim*4 + M*K*4 + M*4 bytes.
/// At K=4, M=16, D=128: 2048 + 256 + 64 = 2368 bytes ≈ 2.3 KB/node.
pub struct NodeBasis {
    /// K * dim, row-major. Row k = basis[k*dim .. (k+1)*dim].
    pub basis: Vec<f32>,
    /// M * K, row-major. edge_projs[m*k .. (m+1)*k] = projections of residual_m.
    pub edge_projs: Vec<f32>,
    /// edge_norms_sq[m] = ||neighbor_m - node||^2.
    pub edge_norms_sq: Vec<f32>,
    /// Actual number of basis vectors produced (min(k_max, rank(residuals))).
    pub k: usize,
    pub dim: usize,
}

impl NodeBasis {
    /// Precompute basis for a single node.
    ///
    /// `node_vec`: the node's embedding (dim elements).
    /// `neighbor_vecs`: slices for each neighbor's embedding.
    /// `k_max`: maximum number of basis vectors to produce.
    pub fn build(node_vec: &[f32], neighbor_vecs: &[&[f32]], k_max: usize) -> Self {
        let dim = node_vec.len();
        let m = neighbor_vecs.len();

        if m == 0 || k_max == 0 {
            return NodeBasis {
                basis: Vec::new(),
                edge_projs: Vec::new(),
                edge_norms_sq: Vec::new(),
                k: 0,
                dim,
            };
        }

        // Compute residual vectors: r_i = neighbor_i - node
        let mut residuals: Vec<Vec<f32>> = neighbor_vecs
            .iter()
            .map(|nv| {
                nv.iter().zip(node_vec.iter()).map(|(a, b)| a - b).collect()
            })
            .collect();

        // Modified Gram-Schmidt orthogonalization over the residuals.
        let mut basis_vecs: Vec<Vec<f32>> = Vec::with_capacity(k_max);
        for r in residuals.iter_mut() {
            // Orthogonalize r against all accepted basis vectors.
            for b in &basis_vecs {
                let proj = dot(r, b);
                saxpy(r, b, -proj);
            }
            let norm_sq: f32 = r.iter().map(|x| x * x).sum();
            if norm_sq > GRAM_SCHMIDT_EPS * GRAM_SCHMIDT_EPS {
                let inv = 1.0 / norm_sq.sqrt();
                let normalized: Vec<f32> = r.iter().map(|x| x * inv).collect();
                basis_vecs.push(normalized);
            }
            if basis_vecs.len() == k_max {
                break;
            }
        }

        let k = basis_vecs.len();

        // Flatten basis into row-major Vec<f32>.
        let mut basis = vec![0.0f32; k * dim];
        for (ki, bv) in basis_vecs.iter().enumerate() {
            basis[ki * dim..(ki + 1) * dim].copy_from_slice(bv);
        }

        // Precompute edge projections and norms.
        let mut edge_projs = vec![0.0f32; m * k];
        let mut edge_norms_sq = vec![0.0f32; m];

        for (mi, nv) in neighbor_vecs.iter().enumerate() {
            // Residual: already computed above, but let's use nv and node_vec directly.
            let r: Vec<f32> = nv.iter().zip(node_vec.iter()).map(|(a, b)| a - b).collect();
            edge_norms_sq[mi] = l2_sq(nv, node_vec);
            for ki in 0..k {
                let e = &basis[ki * dim..(ki + 1) * dim];
                edge_projs[mi * k + ki] = dot(&r, e);
            }
        }

        NodeBasis { basis, edge_projs, edge_norms_sq, k, dim }
    }

    /// Approximate distance from query to neighbor `m`.
    ///
    /// Inputs:
    ///   `query_proj`: precomputed projections of (query - node) onto each basis vector.
    ///                 Length = self.k.
    ///   `query_node_dist_sq`: exact squared distance from query to this node.
    ///   `m`: neighbor index.
    ///
    /// Returns an approximation to ||query - neighbor_m||.
    #[inline]
    pub fn approx_dist(&self, query_proj: &[f32], query_node_dist_sq: f32, m: usize) -> f32 {
        debug_assert_eq!(query_proj.len(), self.k);
        let edge_norm_sq = self.edge_norms_sq[m];
        let mut approx_dot = 0.0f32;
        let base = m * self.k;
        for ki in 0..self.k {
            approx_dot += query_proj[ki] * self.edge_projs[base + ki];
        }
        let dist_sq = (query_node_dist_sq + edge_norm_sq - 2.0 * approx_dot).max(0.0);
        dist_sq.sqrt()
    }

    /// Project `(query - node)` onto all basis vectors.
    ///
    /// `residual` = query - node_vec (precomputed by the caller).
    /// Returns a vec of length self.k.
    pub fn project(&self, residual: &[f32]) -> Vec<f32> {
        let mut proj = vec![0.0f32; self.k];
        for ki in 0..self.k {
            let e = &self.basis[ki * self.dim..(ki + 1) * self.dim];
            proj[ki] = dot(residual, e);
        }
        proj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3(x: f32, y: f32, z: f32) -> Vec<f32> {
        vec![x, y, z]
    }

    #[test]
    fn test_basis_orthonormal() {
        let node = vec3(0.0, 0.0, 0.0);
        let n1 = vec3(1.0, 0.0, 0.0);
        let n2 = vec3(0.0, 1.0, 0.0);
        let n3 = vec3(1.0, 1.0, 0.0); // linearly dependent on n1+n2
        let nb: Vec<&[f32]> = vec![&n1, &n2, &n3];
        let basis = NodeBasis::build(&node, &nb, 4);
        // n1, n2 are independent → 2 basis vecs; n3 = n1+n2 → degenerate → still 2
        assert_eq!(basis.k, 2, "expected 2 independent directions, got {}", basis.k);

        // Check orthonormality of each pair
        let e0 = &basis.basis[0..3];
        let e1 = &basis.basis[3..6];
        let cross = dot(e0, e1);
        assert!(cross.abs() < 1e-5, "basis not orthogonal: dot={cross}");
        let n0: f32 = e0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1_norm: f32 = e1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-5, "basis[0] not unit: norm={n0}");
        assert!((n1_norm - 1.0).abs() < 1e-5, "basis[1] not unit: norm={n1_norm}");
    }

    #[test]
    fn test_approx_dist_exact_for_span() {
        // When (query - node) lies exactly in the edge subspace,
        // the approximation should equal the exact distance.
        let node = vec3(0.0, 0.0, 0.0);
        let neighbor = vec3(1.0, 0.0, 0.0);
        let nb: Vec<&[f32]> = vec![&neighbor];
        let basis = NodeBasis::build(&node, &nb, 4);
        assert_eq!(basis.k, 1);

        // query = (0.5, 0, 0) is in the span of the single edge
        let query = vec3(0.5, 0.0, 0.0);
        let residual: Vec<f32> = query.iter().zip(node.iter()).map(|(a, b)| a - b).collect();
        let qn_dist_sq = l2_sq(&query, &node);
        let proj = basis.project(&residual);
        let approx = basis.approx_dist(&proj, qn_dist_sq, 0);
        let exact = l2_sq(&query, &neighbor).sqrt();
        assert!((approx - exact).abs() < 1e-5, "approx={approx}, exact={exact}");
    }

    #[test]
    fn test_no_neighbors_returns_empty_basis() {
        let node = vec3(1.0, 0.0, 0.0);
        let basis = NodeBasis::build(&node, &[], 4);
        assert_eq!(basis.k, 0);
    }
}
