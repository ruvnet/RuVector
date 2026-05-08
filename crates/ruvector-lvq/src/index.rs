//! Brute-force flat indexes over LVQ-quantized data, with a reranking API
//! suitable for plugging in front of any external ANN graph (HNSW, DiskANN,
//! IVF) where reranking is the dominant cost.
//!
//! The indexes here are *not* graph indexes. They demonstrate the encoder's
//! distance kernels, give us honest end-to-end recall+latency numbers, and
//! act as ground truth for higher-level integrations.

use std::cmp::Ordering;

use crate::distance::{lvq8_l2sq, lvq8x8_l2sq, lvq8x8_l2sq_primary};
use crate::error::LvqError;
use crate::quantize::Lvq8;
use crate::two_level::Lvq8x8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// fp32 baseline (no quantization).
    Flat,
    /// LVQ-8 single level.
    Lvq8,
    /// LVQ-8x8 with reranking from primary → full residual.
    Lvq8x8,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchHit {
    pub id: u32,
    pub score: f32,
}

impl SearchHit {
    fn cmp_score(a: &Self, b: &Self) -> Ordering {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then(a.id.cmp(&b.id))
    }
}

/// fp32 brute-force flat index. Used as ground truth.
pub struct FlatF32 {
    dim: usize,
    data: Vec<f32>,
    n: usize,
}

impl FlatF32 {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
            n: 0,
        }
    }
    pub fn push(&mut self, v: &[f32]) -> Result<(), LvqError> {
        if v.len() != self.dim {
            return Err(LvqError::DimMismatch {
                expected: self.dim,
                actual: v.len(),
            });
        }
        self.data.extend_from_slice(v);
        self.n += 1;
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn byte_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
    pub fn search_l2(&self, q: &[f32], k: usize) -> Result<Vec<SearchHit>, LvqError> {
        if q.len() != self.dim {
            return Err(LvqError::DimMismatch {
                expected: self.dim,
                actual: q.len(),
            });
        }
        if k > self.n {
            return Err(LvqError::KTooLarge(k, self.n));
        }
        let mut hits: Vec<SearchHit> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let off = i * self.dim;
            let row = &self.data[off..off + self.dim];
            let mut s = 0.0_f32;
            for j in 0..self.dim {
                let d = q[j] - row[j];
                s += d * d;
            }
            hits.push(SearchHit {
                id: i as u32,
                score: s,
            });
        }
        partial_sort(&mut hits, k);
        hits.truncate(k);
        Ok(hits)
    }
}

/// Flat index over either Lvq8 or Lvq8x8 storage. Search is a linear scan
/// against the asymmetric distance kernel.
pub struct FlatLvqIndex {
    pub kind: IndexKind,
    lvq8: Option<Lvq8>,
    lvq8x8: Option<Lvq8x8>,
    dim: usize,
}

impl FlatLvqIndex {
    pub fn new_lvq8(dim: usize) -> Self {
        Self {
            kind: IndexKind::Lvq8,
            lvq8: Some(Lvq8::new(dim)),
            lvq8x8: None,
            dim,
        }
    }

    pub fn new_lvq8x8(dim: usize) -> Self {
        Self {
            kind: IndexKind::Lvq8x8,
            lvq8: None,
            lvq8x8: Some(Lvq8x8::new(dim)),
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        match self.kind {
            IndexKind::Lvq8 => self.lvq8.as_ref().map_or(0, |q| q.len()),
            IndexKind::Lvq8x8 => self.lvq8x8.as_ref().map_or(0, |q| q.len()),
            IndexKind::Flat => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_size(&self) -> usize {
        match self.kind {
            IndexKind::Lvq8 => self.lvq8.as_ref().map_or(0, |q| q.byte_size()),
            IndexKind::Lvq8x8 => self.lvq8x8.as_ref().map_or(0, |q| q.byte_size()),
            IndexKind::Flat => 0,
        }
    }

    pub fn push(&mut self, v: &[f32]) -> Result<(), LvqError> {
        match self.kind {
            IndexKind::Lvq8 => self.lvq8.as_mut().unwrap().push(v),
            IndexKind::Lvq8x8 => self.lvq8x8.as_mut().unwrap().push(v),
            IndexKind::Flat => Err(LvqError::AlreadyBuilt),
        }
    }

    pub fn extend_from_flat(&mut self, flat: &[f32]) -> Result<(), LvqError> {
        match self.kind {
            IndexKind::Lvq8 => self.lvq8.as_mut().unwrap().extend_from_flat(flat),
            IndexKind::Lvq8x8 => self.lvq8x8.as_mut().unwrap().extend_from_flat(flat),
            IndexKind::Flat => Err(LvqError::AlreadyBuilt),
        }
    }

    /// Single-level search.
    pub fn search_l2(&self, q: &[f32], k: usize) -> Result<Vec<SearchHit>, LvqError> {
        if q.len() != self.dim {
            return Err(LvqError::DimMismatch {
                expected: self.dim,
                actual: q.len(),
            });
        }
        let n = self.len();
        if k > n {
            return Err(LvqError::KTooLarge(k, n));
        }
        let mut hits: Vec<SearchHit> = Vec::with_capacity(n);
        match self.kind {
            IndexKind::Lvq8 => {
                let q8 = self.lvq8.as_ref().unwrap();
                for i in 0..n {
                    let stats = q8.stats_at(i);
                    let row = q8.code_row(i);
                    hits.push(SearchHit {
                        id: i as u32,
                        score: lvq8_l2sq(q, row, stats),
                    });
                }
            }
            IndexKind::Lvq8x8 => {
                let q8x8 = self.lvq8x8.as_ref().unwrap();
                for i in 0..n {
                    hits.push(SearchHit {
                        id: i as u32,
                        score: lvq8x8_l2sq(q, i, q8x8),
                    });
                }
            }
            IndexKind::Flat => unreachable!(),
        }
        partial_sort(&mut hits, k);
        hits.truncate(k);
        Ok(hits)
    }

    /// Two-stage search for `Lvq8x8`: fetch a `rerank_k`-size candidate list
    /// using only the primary code (cheap), then rescore the candidates with
    /// the full primary+residual reconstruction. This is the recipe SVS
    /// reports: ~3x faster than full residual scan with no recall loss when
    /// `rerank_k = 10 * k` or so.
    pub fn search_l2_reranked(
        &self,
        q: &[f32],
        k: usize,
        rerank_k: usize,
    ) -> Result<Vec<SearchHit>, LvqError> {
        if !matches!(self.kind, IndexKind::Lvq8x8) {
            return self.search_l2(q, k);
        }
        if q.len() != self.dim {
            return Err(LvqError::DimMismatch {
                expected: self.dim,
                actual: q.len(),
            });
        }
        let n = self.len();
        let candidates = rerank_k.max(k).min(n);
        if k > n {
            return Err(LvqError::KTooLarge(k, n));
        }
        let q8x8 = self.lvq8x8.as_ref().unwrap();

        let mut prelim: Vec<SearchHit> = Vec::with_capacity(n);
        for i in 0..n {
            prelim.push(SearchHit {
                id: i as u32,
                score: lvq8x8_l2sq_primary(q, i, q8x8),
            });
        }
        partial_sort(&mut prelim, candidates);
        prelim.truncate(candidates);

        for h in &mut prelim {
            h.score = lvq8x8_l2sq(q, h.id as usize, q8x8);
        }
        partial_sort(&mut prelim, k);
        prelim.truncate(k);
        Ok(prelim)
    }
}

fn partial_sort(hits: &mut Vec<SearchHit>, k: usize) {
    if k == 0 || hits.is_empty() {
        return;
    }
    if k >= hits.len() {
        hits.sort_by(SearchHit::cmp_score);
        return;
    }
    hits.select_nth_unstable_by(k - 1, SearchHit::cmp_score);
    hits[..k].sort_by(SearchHit::cmp_score);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::{rngs::StdRng, Rng};

    fn make_dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn lvq8_recall_against_groundtruth() {
        let dim = 64;
        let n = 2_000;
        let nq = 64;
        let k = 10;
        let data = make_dataset(n, dim, 1);
        let queries = make_dataset(nq, dim, 2);

        let mut gt = FlatF32::new(dim);
        gt.extend(data.chunks_exact(dim)).unwrap();

        let mut lvq = FlatLvqIndex::new_lvq8(dim);
        lvq.extend_from_flat(&data).unwrap();

        let mut hits = 0usize;
        for q in queries.chunks_exact(dim) {
            let truth: Vec<u32> = gt
                .search_l2(q, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            let approx: Vec<u32> = lvq
                .search_l2(q, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            for id in &approx {
                if truth.contains(id) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / (k * nq) as f64;
        assert!(recall > 0.85, "recall@10 = {recall:.3}");
    }

    #[test]
    fn lvq8x8_reranking_meets_target() {
        let dim = 64;
        let n = 2_000;
        let nq = 64;
        let k = 10;
        let data = make_dataset(n, dim, 11);
        let queries = make_dataset(nq, dim, 12);

        let mut gt = FlatF32::new(dim);
        gt.extend(data.chunks_exact(dim)).unwrap();

        let mut lvq = FlatLvqIndex::new_lvq8x8(dim);
        lvq.extend_from_flat(&data).unwrap();

        let mut hits = 0usize;
        for q in queries.chunks_exact(dim) {
            let truth: Vec<u32> = gt
                .search_l2(q, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            let approx: Vec<u32> = lvq
                .search_l2_reranked(q, k, k * 10)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            for id in &approx {
                if truth.contains(id) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / (k * nq) as f64;
        assert!(recall > 0.97, "recall@10 = {recall:.3}");
    }
}

impl FlatF32 {
    pub fn extend<'a, I: IntoIterator<Item = &'a [f32]>>(
        &mut self,
        iter: I,
    ) -> Result<(), LvqError> {
        for v in iter {
            self.push(v)?;
        }
        Ok(())
    }
}
