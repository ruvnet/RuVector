//! Sparse weighted graph with dynamic edge operations.
//!
//! [`SparseGraph`] uses adjacency-list storage for O(1) edge insertion and
//! deletion, plus efficient CSR export for matrix operations. It supports
//! the Laplacian quadratic form `x^T L x` needed for spectral audits.

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SparsifierError};

// ---------------------------------------------------------------------------
// SparseGraph
// ---------------------------------------------------------------------------

/// A weighted undirected graph with dynamic edge support.
///
/// Internally stores adjacency lists keyed by vertex index. Vertices are
/// implicitly numbered `0..num_vertices`. When an edge `(u, v)` is
/// inserted with `max(u, v) >= num_vertices`, the vertex count grows
/// automatically.
///
/// # Thread safety
///
/// `SparseGraph` is *not* internally synchronised. Wrap it in
/// [`parking_lot::RwLock`] for concurrent access (the sparsifier does this).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseGraph {
    /// Adjacency lists: `adj[u]` maps neighbour `v` to edge weight.
    adj: Vec<HashMap<usize, f64>>,
    /// Total number of undirected edges (each stored in both directions).
    num_edges: usize,
    /// Sum of all edge weights (each edge counted once).
    total_weight: f64,
}

impl Default for SparseGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseGraph {
    // ----- construction ----------------------------------------------------

    /// Create an empty graph with no vertices.
    pub fn new() -> Self {
        Self {
            adj: Vec::new(),
            num_edges: 0,
            total_weight: 0.0,
        }
    }

    /// Create an empty graph pre-allocated for `n` vertices.
    pub fn with_capacity(n: usize) -> Self {
        // Pre-size each adjacency map for typical sparse-graph density (~4 neighbors avg).
        Self {
            adj: (0..n).map(|_| HashMap::with_capacity(4)).collect(),
            num_edges: 0,
            total_weight: 0.0,
        }
    }

    /// Build a graph from a list of weighted edges `(u, v, weight)`.
    ///
    /// Duplicate edges are silently overwritten with the last weight.
    pub fn from_edges(edges: &[(usize, usize, f64)]) -> Self {
        let n = edges
            .iter()
            .map(|&(u, v, _)| u.max(v) + 1)
            .max()
            .unwrap_or(0);
        let mut g = Self::with_capacity(n);
        for &(u, v, w) in edges {
            // Ignore errors from duplicate insertion; last write wins.
            let _ = g.insert_edge(u, v, w);
        }
        g
    }

    // ----- capacity --------------------------------------------------------

    /// Ensure the graph can represent at least `n` vertices.
    pub fn ensure_capacity(&mut self, n: usize) {
        if n > self.adj.len() {
            self.adj.resize_with(n, HashMap::new);
        }
    }

    // ----- queries ---------------------------------------------------------

    /// Number of vertices (equal to the length of the adjacency array).
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.adj.len()
    }

    /// Number of undirected edges.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Sum of all edge weights (each undirected edge counted once).
    #[inline]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Degree of vertex `u` (number of neighbours).
    ///
    /// Returns 0 if `u` is out of bounds.
    #[inline]
    pub fn degree(&self, u: usize) -> usize {
        self.adj.get(u).map_or(0, |m| m.len())
    }

    /// Weighted degree of vertex `u` (sum of incident edge weights).
    ///
    /// Note: This iterates incident edges each call. For hot-path usage,
    /// consider caching the result externally.
    #[inline]
    pub fn weighted_degree(&self, u: usize) -> f64 {
        self.adj.get(u).map_or(0.0, |m| m.values().copied().sum())
    }

    /// Iterator over neighbours of `u` yielding `(v, weight)`.
    ///
    /// # Panics
    ///
    /// Panics if `u >= num_vertices()`.
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.adj[u].iter().map(|(&v, &w)| (v, w))
    }

    /// Weight of the edge `(u, v)`, or `None` if it does not exist.
    #[inline]
    pub fn edge_weight(&self, u: usize, v: usize) -> Option<f64> {
        self.adj.get(u).and_then(|m| m.get(&v).copied())
    }

    /// Check whether edge `(u, v)` exists.
    #[inline]
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adj.get(u).is_some_and(|m| m.contains_key(&v))
    }

    /// Iterate over all edges yielding `(u, v, weight)` with `u < v`.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.adj.iter().enumerate().flat_map(|(u, nbrs)| {
            nbrs.iter()
                .filter(move |(&v, _)| v > u)
                .map(move |(&v, &w)| (u, v, w))
        })
    }

    // ----- mutations -------------------------------------------------------

    /// Insert an undirected edge `(u, v)` with the given weight.
    ///
    /// The vertex set is automatically expanded if necessary.
    ///
    /// # Errors
    ///
    /// - [`SparsifierError::InvalidWeight`] if `weight` is not positive/finite.
    /// - [`SparsifierError::EdgeAlreadyExists`] if the edge is present.
    pub fn insert_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<()> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(SparsifierError::InvalidWeight(weight));
        }
        let n = u.max(v) + 1;
        self.ensure_capacity(n);

        if self.adj[u].contains_key(&v) {
            return Err(SparsifierError::EdgeAlreadyExists(u, v));
        }

        self.adj[u].insert(v, weight);
        if u != v {
            self.adj[v].insert(u, weight);
        }
        self.num_edges += 1;
        self.total_weight += weight;
        Ok(())
    }

    /// Insert or overwrite an edge without returning an error on duplicates.
    ///
    /// Returns the old weight if the edge already existed.
    pub fn insert_or_update_edge(
        &mut self,
        u: usize,
        v: usize,
        weight: f64,
    ) -> Result<Option<f64>> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(SparsifierError::InvalidWeight(weight));
        }
        let n = u.max(v) + 1;
        self.ensure_capacity(n);

        let old = self.adj[u].insert(v, weight);
        if u != v {
            self.adj[v].insert(u, weight);
        }

        if let Some(old_w) = old {
            self.total_weight += weight - old_w;
            Ok(Some(old_w))
        } else {
            self.num_edges += 1;
            self.total_weight += weight;
            Ok(None)
        }
    }

    /// Delete the undirected edge `(u, v)`.
    ///
    /// # Errors
    ///
    /// Returns [`SparsifierError::EdgeNotFound`] if the edge does not exist.
    pub fn delete_edge(&mut self, u: usize, v: usize) -> Result<f64> {
        let w = self
            .adj
            .get_mut(u)
            .and_then(|m| m.remove(&v))
            .ok_or(SparsifierError::EdgeNotFound(u, v))?;
        if u != v {
            self.adj[v].remove(&u);
        }
        self.num_edges -= 1;
        self.total_weight -= w;
        Ok(w)
    }

    /// Update the weight of edge `(u, v)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge does not exist or the weight is invalid.
    pub fn update_weight(&mut self, u: usize, v: usize, new_weight: f64) -> Result<f64> {
        if !new_weight.is_finite() || new_weight <= 0.0 {
            return Err(SparsifierError::InvalidWeight(new_weight));
        }
        let old = self
            .adj
            .get_mut(u)
            .and_then(|m| m.get_mut(&v))
            .ok_or(SparsifierError::EdgeNotFound(u, v))?;
        let old_w = *old;
        *old = new_weight;
        if u != v {
            if let Some(entry) = self.adj[v].get_mut(&u) {
                *entry = new_weight;
            }
        }
        self.total_weight += new_weight - old_w;
        Ok(old_w)
    }

    /// Remove all edges and vertices.
    pub fn clear(&mut self) {
        self.adj.clear();
        self.num_edges = 0;
        self.total_weight = 0.0;
    }

    // ----- Laplacian -------------------------------------------------------

    /// Compute the Laplacian quadratic form `x^T L x`.
    ///
    /// For each edge `(u, v, w)`:  `sum += w * (x[u] - x[v])^2`
    ///
    /// # Panics
    ///
    /// Panics if `x.len() < num_vertices()`.
    pub fn laplacian_quadratic_form(&self, x: &[f64]) -> f64 {
        assert!(
            x.len() >= self.num_vertices(),
            "x.len()={} < num_vertices={}",
            x.len(),
            self.num_vertices()
        );
        let mut sum = 0.0;
        for (u, nbrs) in self.adj.iter().enumerate() {
            for (&v, &w) in nbrs {
                if v > u {
                    let diff = x[u] - x[v];
                    sum += w * diff * diff;
                }
            }
        }
        sum
    }

    // ----- CSR conversion --------------------------------------------------

    /// Export to compressed sparse row (CSR) format.
    ///
    /// Returns `(row_ptr, col_indices, values, n)` where `n` is the number
    /// of vertices. The CSR matrix represents the weighted adjacency matrix
    /// (symmetric, both directions stored).
    pub fn to_csr(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = self.num_vertices();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for u in 0..n {
            // Sort neighbours for deterministic output.
            let mut entries: Vec<(usize, f64)> =
                self.adj[u].iter().map(|(&v, &w)| (v, w)).collect();
            entries.sort_by_key(|&(v, w)| (v, OrderedFloat(w)));
            for (v, w) in entries {
                col_indices.push(v);
                values.push(w);
            }
            row_ptr.push(col_indices.len());
        }

        (row_ptr, col_indices, values, n)
    }

    /// Import from compressed sparse row (CSR) format.
    ///
    /// The CSR data is interpreted as a symmetric adjacency matrix.
    /// Only entries with `col >= row` are inserted to avoid double-counting.
    pub fn from_csr(row_ptr: &[usize], col_indices: &[usize], values: &[f64], n: usize) -> Self {
        let mut g = Self::with_capacity(n);
        for u in 0..n {
            let start = row_ptr[u];
            let end = row_ptr[u + 1];
            for idx in start..end {
                let v = col_indices[idx];
                let w = values[idx];
                if v >= u && !g.has_edge(u, v) {
                    let _ = g.insert_edge(u, v, w);
                }
            }
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query() {
        let mut g = SparseGraph::new();
        g.insert_edge(0, 1, 2.0).unwrap();
        g.insert_edge(1, 2, 3.0).unwrap();

        assert_eq!(g.num_vertices(), 3);
        assert_eq!(g.num_edges(), 2);
        assert!((g.total_weight() - 5.0).abs() < 1e-12);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.edge_weight(0, 1), Some(2.0));
        assert_eq!(g.edge_weight(1, 0), Some(2.0));
        assert_eq!(g.edge_weight(0, 2), None);
    }

    #[test]
    fn test_delete_edge() {
        let mut g = SparseGraph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0)]);
        assert_eq!(g.num_edges(), 2);

        let w = g.delete_edge(0, 1).unwrap();
        assert!((w - 1.0).abs() < 1e-12);
        assert_eq!(g.num_edges(), 1);
        assert!(!g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0));
    }

    #[test]
    fn test_update_weight() {
        let mut g = SparseGraph::from_edges(&[(0, 1, 1.0)]);
        let old = g.update_weight(0, 1, 5.0).unwrap();
        assert!((old - 1.0).abs() < 1e-12);
        assert_eq!(g.edge_weight(0, 1), Some(5.0));
        assert_eq!(g.edge_weight(1, 0), Some(5.0));
        assert!((g.total_weight() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_laplacian_quadratic_form() {
        // Triangle graph with unit weights.
        let g = SparseGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
        // x = [1, 0, 0] => L*x = [2, -1, -1], x^T L x = 2
        let x = vec![1.0, 0.0, 0.0];
        let val = g.laplacian_quadratic_form(&x);
        assert!((val - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_roundtrip() {
        let g = SparseGraph::from_edges(&[(0, 1, 1.5), (1, 2, 2.5), (0, 2, 3.5)]);
        let (rp, ci, vals, n) = g.to_csr();
        let g2 = SparseGraph::from_csr(&rp, &ci, &vals, n);

        assert_eq!(g2.num_vertices(), g.num_vertices());
        assert_eq!(g2.num_edges(), g.num_edges());
        assert!((g2.total_weight() - g.total_weight()).abs() < 1e-12);
    }

    #[test]
    fn test_edges_iterator() {
        let g = SparseGraph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]);
        let edges: Vec<_> = g.edges().collect();
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_invalid_weight() {
        let mut g = SparseGraph::new();
        assert!(g.insert_edge(0, 1, -1.0).is_err());
        assert!(g.insert_edge(0, 1, 0.0).is_err());
        assert!(g.insert_edge(0, 1, f64::NAN).is_err());
        assert!(g.insert_edge(0, 1, f64::INFINITY).is_err());
    }

    #[test]
    fn test_duplicate_edge() {
        let mut g = SparseGraph::new();
        g.insert_edge(0, 1, 1.0).unwrap();
        assert!(g.insert_edge(0, 1, 2.0).is_err());
    }
}
