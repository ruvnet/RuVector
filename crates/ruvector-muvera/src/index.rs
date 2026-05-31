//! MUVERA flat-scan index over FDE-compressed multi-vector documents.
//!
//! In production, replace `flat_search` with an HNSW or IVF backend:
//! the FDE vector is a standard `Vec<f32>` compatible with any L2/dot index.

use crate::encoder::FdeEncoder;
use crate::error::MuveraError;

/// A stored document: its identifier and pre-computed FDE vector.
pub struct MuveraEntry {
    pub id: String,
    pub fde: Vec<f32>,
}

/// One result from a MUVERA search, ranked by FDE dot-product score.
#[derive(Debug, Clone)]
pub struct MuveraResult {
    pub id: String,
    /// FDE dot-product score (higher = more similar).
    pub score: f32,
}

/// Swappable backend trait — implement for HNSW, IVF, etc.
pub trait VectorBackend: Send + Sync {
    fn insert(&mut self, id: &str, vec: &[f32]);
    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)>;
    fn len(&self) -> usize;
}

/// Simple flat dot-product backend (O(n)).
pub struct FlatBackend {
    entries: Vec<(String, Vec<f32>)>,
}

impl FlatBackend {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }
}

impl Default for FlatBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorBackend for FlatBackend {
    fn insert(&mut self, id: &str, vec: &[f32]) {
        self.entries.push((id.to_owned(), vec.to_vec()));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(f32, &str)> = self
            .entries
            .iter()
            .map(|(id, v)| {
                let s: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (s, id.as_str())
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(k).map(|(s, id)| (id.to_string(), *s)).collect()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// MUVERA index: encodes multi-vector documents as FDE vectors, then delegates
/// nearest-neighbour search to a pluggable `VectorBackend`.
pub struct MuveraIndex<B: VectorBackend = FlatBackend> {
    pub encoder: FdeEncoder,
    backend: B,
}

impl MuveraIndex<FlatBackend> {
    /// Create a MUVERA index backed by a flat dot-product scan.
    pub fn new(encoder: FdeEncoder) -> Self {
        Self { encoder, backend: FlatBackend::new() }
    }
}

impl<B: VectorBackend> MuveraIndex<B> {
    /// Create a MUVERA index with a custom backend.
    pub fn with_backend(encoder: FdeEncoder, backend: B) -> Self {
        Self { encoder, backend }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.backend.len()
    }

    /// True if no documents have been inserted.
    pub fn is_empty(&self) -> bool {
        self.backend.len() == 0
    }

    /// Encode `tokens` as an FDE vector and insert it into the backend.
    pub fn insert(&mut self, id: String, tokens: &[Vec<f32>]) -> Result<(), MuveraError> {
        let fde = self.encoder.encode(tokens)?;
        self.backend.insert(&id, &fde);
        Ok(())
    }

    /// Encode `query_tokens` as an FDE vector and return top-k by dot-product score.
    pub fn search(
        &self,
        query_tokens: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<MuveraResult>, MuveraError> {
        let q_fde = self.encoder.encode(query_tokens)?;
        let hits = self.backend.search(&q_fde, k);
        Ok(hits.into_iter().map(|(id, score)| MuveraResult { id, score }).collect())
    }

    /// Estimated memory used by FDE vectors (bytes). Backend overhead not included.
    pub fn fde_memory_bytes(&self) -> usize {
        self.backend.len() * self.encoder.fde_dim() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::FdeConfig;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_index() -> MuveraIndex {
        let cfg = FdeConfig { dim: 8, buckets: 4, d_proj: 4, reps: 2 };
        let mut rng = StdRng::seed_from_u64(1);
        let enc = FdeEncoder::new(cfg, &mut rng).unwrap();
        MuveraIndex::new(enc)
    }

    fn tok(vals: &[f32]) -> Vec<Vec<f32>> {
        vec![vals.to_vec()]
    }

    #[test]
    fn insert_and_len() {
        let mut idx = make_index();
        assert!(idx.is_empty());
        idx.insert("d1".into(), &tok(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn search_returns_top_k() {
        let mut idx = make_index();
        for i in 0..5 {
            let v: Vec<f32> = (0..8).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
            idx.insert(format!("d{i}"), &tok(&v)).unwrap();
        }
        let results = idx.search(&tok(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3).unwrap();
        assert_eq!(results.len(), 3);
        // "d0" should be the top result (highest dot with query).
        assert_eq!(results[0].id, "d0");
    }

    #[test]
    fn search_empty_query_error() {
        let idx = make_index();
        assert!(idx.search(&[], 3).is_err());
    }

    #[test]
    fn memory_estimate() {
        let mut idx = make_index();
        idx.insert("d1".into(), &tok(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).unwrap();
        let expected = idx.encoder.fde_dim() * 4; // 1 entry × fde_dim × sizeof(f32)
        assert_eq!(idx.fde_memory_bytes(), expected);
    }
}
