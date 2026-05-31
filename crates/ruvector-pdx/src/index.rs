//! Three ANN backends behind one trait: RowMajorIndex, PdxFlatIndex, PdxPruneIndex.
//!
//! All share a simple flat (non-hierarchical) structure: one or more blocks of
//! vectors.  A real IVF integration would wrap these blocks as cluster shards,
//! but the flat layout is enough to benchmark the layout + pruning benefits.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{PdxError, Result};
use crate::layout::{PdxBlock, RowBlock};

// ── public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}

pub trait AnnIndex: Send + Sync {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn dim(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    fn label(&self) -> &str;
}

// ── bounded max-heap ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Entry { id: usize, score: f32 }

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool { self.score.to_bits() == o.score.to_bits() }
}
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering { self.score.total_cmp(&o.score) }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}

struct TopK { k: usize, heap: BinaryHeap<Entry> }

impl TopK {
    fn new(k: usize) -> Self { Self { k, heap: BinaryHeap::with_capacity(k + 1) } }

    #[inline]
    fn push(&mut self, id: usize, score: f32) {
        if self.heap.len() < self.k {
            self.heap.push(Entry { id, score });
        } else if let Some(top) = self.heap.peek() {
            if score < top.score {
                self.heap.pop();
                self.heap.push(Entry { id, score });
            }
        }
    }

    fn threshold(&self) -> f32 {
        self.heap.peek().map(|e| e.score).unwrap_or(f32::INFINITY)
    }

    fn into_sorted(self) -> Vec<SearchResult> {
        let mut v: Vec<_> = self.heap.into_iter()
            .map(|e| SearchResult { id: e.id, score: e.score })
            .collect();
        v.sort_by(|a, b| a.score.total_cmp(&b.score));
        v
    }
}

// ── RowMajorIndex ─────────────────────────────────────────────────────────────

/// Baseline: row-major storage, linear L2 scan.
pub struct RowMajorIndex {
    dim: usize,
    blocks: Vec<RowBlock>,
    block_size: usize,
    n: usize,
}

impl RowMajorIndex {
    pub fn new(dim: usize, block_size: usize) -> Self {
        Self { dim, blocks: Vec::new(), block_size, n: 0 }
    }

    fn scan_block(block: &RowBlock, query: &[f32], top: &mut TopK) {
        for i in 0..block.n {
            let row = block.row(i);
            let mut acc = 0.0f32;
            for d in 0..row.len() {
                let diff = query[d] - row[d];
                acc += diff * diff;
            }
            top.push(block.ids[i], acc);
        }
    }
}

impl AnnIndex for RowMajorIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(PdxError::DimMismatch { index_dim: self.dim, vec_dim: vector.len() });
        }
        if self.blocks.is_empty() || self.blocks.last().unwrap().n >= self.block_size {
            self.blocks.push(RowBlock::new(self.dim, self.block_size));
        }
        self.blocks.last_mut().unwrap().push(id, &vector);
        self.n += 1;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.n == 0 { return Err(PdxError::Empty); }
        if k > self.n { return Err(PdxError::KTooLarge { k, size: self.n }); }
        let mut top = TopK::new(k);
        for block in &self.blocks {
            Self::scan_block(block, query, &mut top);
        }
        Ok(top.into_sorted())
    }

    fn len(&self) -> usize { self.n }
    fn dim(&self) -> usize { self.dim }
    fn memory_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.memory_bytes()).sum::<usize>() + std::mem::size_of::<Self>()
    }
    fn label(&self) -> &str { "RowMajorIndex" }
}

// ── PdxFlatIndex ──────────────────────────────────────────────────────────────

/// PDX columnar storage, no pruning.  Shows layout-only gain over row-major.
pub struct PdxFlatIndex {
    dim: usize,
    blocks: Vec<PdxBlock>,
    block_size: usize,
    n: usize,
}

impl PdxFlatIndex {
    pub fn new(dim: usize, block_size: usize) -> Self {
        Self { dim, blocks: Vec::new(), block_size, n: 0 }
    }

    fn scan_block(block: &PdxBlock, query: &[f32], top: &mut TopK) {
        // Columnar scan: loop over dims in outer, vectors in inner.
        // Inner loop is stride-1 → LLVM auto-vectorises this with AVX2.
        let n = block.n;
        let mut partial = vec![0.0f32; n];
        for d in 0..block.dim {
            let qd = query[d];
            let col = block.col(d);
            for i in 0..n {
                let diff = qd - col[i];
                partial[i] += diff * diff;
            }
        }
        for i in 0..n {
            top.push(block.ids[i], partial[i]);
        }
    }
}

impl AnnIndex for PdxFlatIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(PdxError::DimMismatch { index_dim: self.dim, vec_dim: vector.len() });
        }
        if self.blocks.is_empty() || self.blocks.last().unwrap().n >= self.block_size {
            self.blocks.push(PdxBlock::new(self.dim, self.block_size));
        }
        self.blocks.last_mut().unwrap().push(id, &vector);
        self.n += 1;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.n == 0 { return Err(PdxError::Empty); }
        if k > self.n { return Err(PdxError::KTooLarge { k, size: self.n }); }
        let mut top = TopK::new(k);
        for block in &self.blocks {
            Self::scan_block(block, query, &mut top);
        }
        Ok(top.into_sorted())
    }

    fn len(&self) -> usize { self.n }
    fn dim(&self) -> usize { self.dim }
    fn memory_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.memory_bytes()).sum::<usize>() + std::mem::size_of::<Self>()
    }
    fn label(&self) -> &str { "PdxFlatIndex" }
}

// ── PdxPruneIndex ─────────────────────────────────────────────────────────────

/// PDX + exponential dimension schedule + lower-bound pruning (BOND variant).
///
/// Schedule: check dims at checkpoints {c₀, c₁, c₂, …, D} where
///     c₀ = first_check_dim (e.g. 16)
///     cᵢ₊₁ = min(cᵢ * 2, D)
///
/// At each checkpoint, prune vector i if `partial[i] > τ` (current k-th
/// distance).  Since partial L2 is a monotone lower bound on true L2, this
/// prune is **exact** — zero false negatives.
///
/// Pruned vectors are tracked in a u64 bitmask per block (max block_size=64).
pub struct PdxPruneIndex {
    dim: usize,
    blocks: Vec<PdxBlock>,
    block_size: usize,
    n: usize,
    first_check_dim: usize,
}

impl PdxPruneIndex {
    /// * `first_check_dim` — dimensions processed before the first pruning pass.
    ///   Good values: 8–32 depending on D. Smaller → prune earlier but less info.
    pub fn new(dim: usize, block_size: usize, first_check_dim: usize) -> Self {
        let first_check_dim = first_check_dim.max(1).min(dim);
        // Block size must fit in a u64 bitmask.
        let block_size = block_size.min(64);
        Self { dim, blocks: Vec::new(), block_size, n: 0, first_check_dim }
    }

    fn scan_block_pruning(block: &PdxBlock, query: &[f32], first_check: usize, top: &mut TopK) {
        let n = block.n;
        let mut partial = vec![0.0f32; n];
        let all_active: u64 = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let mut active: u64 = all_active;

        // Exponential dimension schedule: first_check, 2×, 4×, …
        let mut d = 0usize;
        let mut chunk_size = first_check.max(1);

        loop {
            let chunk_end = (d + chunk_size).min(block.dim);

            // Hybrid inner loop:
            //   • All vectors active → stride-1 columnar loop, LLVM auto-vectorises.
            //   • Some pruned       → bit-scan over survivors only.
            if active == all_active {
                // Fast path: all N vectors active — pure stride-1 inner loop.
                for dim_d in d..chunk_end {
                    let qd = query[dim_d];
                    let col = block.col(dim_d);
                    for i in 0..n {
                        let diff = qd - col[i];
                        partial[i] += diff * diff;
                    }
                }
            } else {
                // Slow path: sparse active set — iterate only live bits.
                for dim_d in d..chunk_end {
                    let qd = query[dim_d];
                    let col = block.col(dim_d);
                    let mut mask = active;
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        let diff = qd - col[i];
                        partial[i] += diff * diff;
                        mask &= mask - 1;
                    }
                }
            }
            d = chunk_end;

            // Pruning pass: partial L2 is a monotone lower bound on true L2².
            // Any vector with partial[i] > τ is certainly not in the top-k.
            let tau = top.threshold();
            if tau.is_finite() {
                let mut mask = active;
                while mask != 0 {
                    let i = mask.trailing_zeros() as usize;
                    if partial[i] > tau {
                        active &= !(1u64 << i);
                    }
                    mask &= mask - 1;
                }
            }

            if d >= block.dim || active == 0 {
                break;
            }
            chunk_size *= 2;
        }

        // Emit survivors.
        let mut mask = active;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            top.push(block.ids[i], partial[i]);
            mask &= mask - 1;
        }
    }
}

impl AnnIndex for PdxPruneIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(PdxError::DimMismatch { index_dim: self.dim, vec_dim: vector.len() });
        }
        if self.blocks.is_empty() || self.blocks.last().unwrap().n >= self.block_size {
            self.blocks.push(PdxBlock::new(self.dim, self.block_size));
        }
        self.blocks.last_mut().unwrap().push(id, &vector);
        self.n += 1;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.n == 0 { return Err(PdxError::Empty); }
        if k > self.n { return Err(PdxError::KTooLarge { k, size: self.n }); }
        let mut top = TopK::new(k);
        for block in &self.blocks {
            Self::scan_block_pruning(block, query, self.first_check_dim, &mut top);
        }
        Ok(top.into_sorted())
    }

    fn len(&self) -> usize { self.n }
    fn dim(&self) -> usize { self.dim }
    fn memory_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.memory_bytes()).sum::<usize>() + std::mem::size_of::<Self>()
    }
    fn label(&self) -> &str { "PdxPruneIndex" }
}
