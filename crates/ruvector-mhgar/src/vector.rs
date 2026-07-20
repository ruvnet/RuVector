//! In-memory flat vector index with exact cosine search.
//!
//! Intentionally simple — serves as the ground-truth oracle AND the retrieval
//! engine for all three MHGAR variants. For production, swap for HNSW or PQ.

use crate::{Hit, MhgarError};

/// Flat exact-search index over f32 vectors.
pub struct FlatIndex {
    dim: usize,
    ids: Vec<u32>,
    /// Row-major: ids[i] owns vectors[i*dim .. (i+1)*dim].
    vectors: Vec<f32>,
}

impl FlatIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, ids: Vec::new(), vectors: Vec::new() }
    }

    pub fn insert(&mut self, id: u32, vector: &[f32]) -> Result<(), MhgarError> {
        if vector.len() != self.dim {
            return Err(MhgarError::DimensionMismatch {
                index: self.dim,
                query: vector.len(),
            });
        }
        self.ids.push(id);
        self.vectors.extend_from_slice(vector);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return top-k by ascending cosine distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, MhgarError> {
        if self.ids.is_empty() {
            return Err(MhgarError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(MhgarError::DimensionMismatch {
                index: self.dim,
                query: query.len(),
            });
        }
        let q_norm = l2_norm(query);
        let mut scored: Vec<(u32, f32)> = self
            .ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| {
                let v = &self.vectors[i * self.dim..(i + 1) * self.dim];
                let dist = cosine_distance(query, v, q_norm);
                (id, dist)
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(id, dist)| Hit::new(id, dist, 0))
            .collect())
    }

    /// Fetch the vector for a given entity id.
    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        let pos = self.ids.iter().position(|&x| x == id)?;
        Some(&self.vectors[pos * self.dim..(pos + 1) * self.dim])
    }
}

/// Cosine distance in [0, 2]. 0 = identical direction, 2 = opposite.
pub fn cosine_distance(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let b_norm = l2_norm(b);
    if a_norm < 1e-8 || b_norm < 1e-8 {
        return 1.0;
    }
    1.0 - dot / (a_norm * b_norm)
}

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
