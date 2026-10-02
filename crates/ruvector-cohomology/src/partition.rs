//! Deterministic bounded cell partition (issue #669, Phase 4).
//!
//! The dynamic engine partitions the complex into cells of bounded vertex
//! count. Edits mark only incident cells dirty; global synchronization is
//! deferred to `flush()`. The partition is a function of sorted vertex ids
//! and the insertion history, both deterministic.

use crate::{EdgeKey, VertexId};
use std::collections::BTreeMap;

/// Bounded cell partition over the vertex set.
#[derive(Debug, Clone)]
pub struct CellPartition {
    cells: Vec<Vec<VertexId>>,
    vertex_to_cell: BTreeMap<VertexId, usize>,
    capacity: usize,
}

impl CellPartition {
    /// Build from sorted vertex ids, chunking into cells of at most
    /// `capacity` vertices.
    pub fn build(sorted_vertices: impl IntoIterator<Item = VertexId>, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut cells: Vec<Vec<VertexId>> = Vec::new();
        let mut vertex_to_cell = BTreeMap::new();
        for v in sorted_vertices {
            if cells.last().map(|c| c.len() >= capacity).unwrap_or(true) {
                cells.push(Vec::with_capacity(capacity));
            }
            let idx = cells.len() - 1;
            cells.last_mut().expect("cell exists").push(v);
            vertex_to_cell.insert(v, idx);
        }
        Self { cells, vertex_to_cell, capacity }
    }

    /// Number of cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Configured capacity bound.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Cell of a vertex.
    pub fn cell_of(&self, v: VertexId) -> Option<usize> {
        self.vertex_to_cell.get(&v).copied()
    }

    /// Cells touched by an edge (deduplicated, sorted).
    pub fn edge_cells(&self, key: &EdgeKey) -> Vec<usize> {
        let mut cells: Vec<usize> = [self.cell_of(key.u), self.cell_of(key.v)]
            .into_iter()
            .flatten()
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    /// Whether an edge crosses a cell boundary (interface edge).
    pub fn is_interface(&self, key: &EdgeKey) -> bool {
        match (self.cell_of(key.u), self.cell_of(key.v)) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        }
    }

    /// Insert a vertex: appended to the last cell with room, otherwise a new
    /// cell. Returns the cell index. Deterministic in the edit sequence.
    pub fn insert_vertex(&mut self, v: VertexId) -> usize {
        if let Some(idx) = self.vertex_to_cell.get(&v) {
            return *idx;
        }
        let idx = match self.cells.last() {
            Some(c) if c.len() < self.capacity => self.cells.len() - 1,
            _ => {
                self.cells.push(Vec::with_capacity(self.capacity));
                self.cells.len() - 1
            }
        };
        self.cells[idx].push(v);
        self.vertex_to_cell.insert(v, idx);
        idx
    }

    /// Remove a vertex. Returns its former cell.
    pub fn remove_vertex(&mut self, v: VertexId) -> Option<usize> {
        let idx = self.vertex_to_cell.remove(&v)?;
        self.cells[idx].retain(|&x| x != v);
        Some(idx)
    }

    /// Vertices per cell (for inspection/testing).
    pub fn cells(&self) -> &[Vec<VertexId>] {
        &self.cells
    }
}
