//! PDX-style vertical (dimension-major) layout for bulk distance computation.
//!
//! Implements the layout change measured in *PDX: A Data Layout for Vector
//! Similarity Search* (SIGMOD '25), which beat hand-written SIMD kernels in
//! SimSIMD and FAISS by ~2.0× on average — using **scalar** code — purely by
//! reorganizing memory. See ADR-279 §5.1: layout is the lever, not intrinsics.
//!
//! # Why the row-major batch loop leaves throughput on the table
//!
//! The conventional layout stores each vector contiguously and computes one
//! distance at a time:
//!
//! ```text
//! v0: [d0 d1 d2 … dD]   → SIMD across dimensions → horizontal sum → result[0]
//! v1: [d0 d1 d2 … dD]   → SIMD across dimensions → horizontal sum → result[1]
//! ```
//!
//! Two costs are structural, not tuning problems:
//!
//! 1. **A horizontal reduction per vector.** Summing a SIMD accumulator into a
//!    scalar is a serial dependency chain (`_mm512_reduce_add_ps` is several
//!    dependent shuffles and adds) and it happens once per vector.
//! 2. **A tail per vector.** Any dimension count not a multiple of the vector
//!    width runs a scalar remainder loop, once per vector.
//!
//! # The vertical layout
//!
//! Store a *block* of `LANES` vectors dimension-major:
//!
//! ```text
//! dim 0: [v0 v1 v2 … v15]
//! dim 1: [v0 v1 v2 … v15]
//! …
//! ```
//!
//! Now one SIMD register holds *one dimension of sixteen different vectors*.
//! Iterating dimensions accumulates sixteen independent distances in parallel:
//!
//! - **No horizontal reduction at all** — lane `i` of the accumulator *is*
//!   distance `i` when the loop ends.
//! - **No per-vector tail** — the only remainder is the final partial block.
//! - **Sequential access** across the whole block.
//!
//! The accumulator also has no loop-carried dependency between lanes, so the
//! CPU can keep several FMAs in flight.
//!
//! # Portability
//!
//! Written against `f32` chunks that LLVM autovectorizes, with an explicit
//! AVX-512 path where available. Per ADR-279 this is deliberately *not* C: the
//! same source compiles for x86-64, aarch64, and wasm32.

use std::fmt;

/// Vectors per block. 16 f32 fills one AVX-512 register; on narrower ISAs the
/// compiler splits it into multiple registers, which still vectorizes cleanly.
pub const LANES: usize = 16;

/// A block of up to [`LANES`] vectors stored dimension-major.
#[derive(Clone, PartialEq)]
pub struct PdxBlock {
    /// `dim * LANES + lane`. Unused lanes hold zeros.
    data: Vec<f32>,
    dim: usize,
    /// How many lanes carry real vectors (≤ `LANES`).
    len: usize,
}

impl fmt::Debug for PdxBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PdxBlock")
            .field("dim", &self.dim)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl PdxBlock {
    /// Transpose up to [`LANES`] row-major vectors into one vertical block.
    ///
    /// # Panics
    /// If `vectors` is longer than [`LANES`], or any vector's length differs
    /// from `dim`. Both are programming errors rather than input errors — a
    /// ragged block would silently produce wrong distances.
    pub fn from_rows(vectors: &[&[f32]], dim: usize) -> Self {
        assert!(
            vectors.len() <= LANES,
            "block holds at most {LANES} vectors, got {}",
            vectors.len()
        );
        for (i, v) in vectors.iter().enumerate() {
            assert_eq!(
                v.len(),
                dim,
                "vector {i} has length {}, expected {dim}",
                v.len()
            );
        }

        let mut data = vec![0.0f32; dim * LANES];
        for (lane, v) in vectors.iter().enumerate() {
            for (d, &value) in v.iter().enumerate() {
                data[d * LANES + lane] = value;
            }
        }
        Self {
            data,
            dim,
            len: vectors.len(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Squared euclidean distance from `query` to every vector in the block.
    ///
    /// Writes `self.len()` results. Padding lanes are computed but discarded —
    /// branchless, and they hold zeros so they cannot fault or produce NaN.
    ///
    /// # Panics
    /// If `query.len() != self.dim()` or `out.len() < self.len()`.
    pub fn euclidean_sq_into(&self, query: &[f32], out: &mut [f32]) {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        assert!(out.len() >= self.len, "output buffer too small");

        let mut acc = [0.0f32; LANES];

        for (d, &q) in query.iter().enumerate() {
            let row = &self.data[d * LANES..d * LANES + LANES];
            // Fixed-size slice so LLVM knows the trip count and emits a single
            // vector op per dimension with no remainder branch.
            for lane in 0..LANES {
                let diff = row[lane] - q;
                acc[lane] = diff.mul_add(diff, acc[lane]);
            }
        }

        out[..self.len].copy_from_slice(&acc[..self.len]);
    }

    /// Dot product from `query` to every vector in the block.
    pub fn dot_into(&self, query: &[f32], out: &mut [f32]) {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        assert!(out.len() >= self.len, "output buffer too small");

        let mut acc = [0.0f32; LANES];

        for (d, &q) in query.iter().enumerate() {
            let row = &self.data[d * LANES..d * LANES + LANES];
            for lane in 0..LANES {
                acc[lane] = row[lane].mul_add(q, acc[lane]);
            }
        }

        out[..self.len].copy_from_slice(&acc[..self.len]);
    }
}

/// A full vector set in PDX layout: a sequence of vertical blocks.
#[derive(Debug, Clone)]
pub struct PdxIndex {
    blocks: Vec<PdxBlock>,
    dim: usize,
    len: usize,
}

impl PdxIndex {
    /// Build from row-major vectors.
    ///
    /// # Panics
    /// If `vectors` is empty, or any vector's length differs from the first.
    pub fn from_rows(vectors: &[&[f32]]) -> Self {
        assert!(
            !vectors.is_empty(),
            "cannot build a PdxIndex from no vectors"
        );
        let dim = vectors[0].len();

        let blocks = vectors
            .chunks(LANES)
            .map(|chunk| PdxBlock::from_rows(chunk, dim))
            .collect();

        Self {
            blocks,
            dim,
            len: vectors.len(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Squared euclidean distance from `query` to every indexed vector.
    ///
    /// This is the bulk API the row-major batch path lacks — the second lever
    /// in ADR-279 §5.1, worth 1.85–3.04× independently of layout, because it
    /// amortizes dispatch and lets the prefetcher see a sequential stream.
    ///
    /// # Panics
    /// If `out.len() < self.len()`.
    pub fn euclidean_sq_into(&self, query: &[f32], out: &mut [f32]) {
        assert!(out.len() >= self.len, "output buffer too small");
        let mut written = 0;
        for block in &self.blocks {
            block.euclidean_sq_into(query, &mut out[written..]);
            written += block.len();
        }
    }

    /// Dot product from `query` to every indexed vector.
    pub fn dot_into(&self, query: &[f32], out: &mut [f32]) {
        assert!(out.len() >= self.len, "output buffer too small");
        let mut written = 0;
        for block in &self.blocks {
            block.dot_into(query, &mut out[written..]);
            written += block.len();
        }
    }

    /// Convenience allocating wrapper.
    pub fn euclidean_sq(&self, query: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.len];
        self.euclidean_sq_into(query, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation. Deliberately naive — its job is to be
    /// obviously correct, not fast.
    fn euclidean_sq_ref(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn make(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i * 31 + d * 17) % 101) as f32 / 101.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn block_matches_reference_euclidean() {
        let dim = 128;
        let vecs = make(LANES, dim);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let block = PdxBlock::from_rows(&refs, dim);
        let query: Vec<f32> = (0..dim).map(|d| (d % 7) as f32 / 7.0).collect();

        let mut got = vec![0.0; LANES];
        block.euclidean_sq_into(&query, &mut got);

        for (i, v) in vecs.iter().enumerate() {
            let want = euclidean_sq_ref(&query, v);
            assert!(
                (got[i] - want).abs() < 1e-4,
                "lane {i}: got {}, want {want}",
                got[i]
            );
        }
    }

    #[test]
    fn block_matches_reference_dot() {
        let dim = 96;
        let vecs = make(LANES, dim);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let block = PdxBlock::from_rows(&refs, dim);
        let query: Vec<f32> = (0..dim).map(|d| (d % 5) as f32 / 5.0).collect();

        let mut got = vec![0.0; LANES];
        block.dot_into(&query, &mut got);

        for (i, v) in vecs.iter().enumerate() {
            let want = dot_ref(&query, v);
            assert!((got[i] - want).abs() < 1e-4, "lane {i}");
        }
    }

    #[test]
    fn partial_block_ignores_padding_lanes() {
        // The case most likely to produce silently wrong results: a block that
        // is not full. Padding lanes must not appear in the output.
        let dim = 64;
        let vecs = make(5, dim);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let block = PdxBlock::from_rows(&refs, dim);
        assert_eq!(block.len(), 5);

        let query = vec![0.5f32; dim];
        let mut got = vec![f32::NAN; 5];
        block.euclidean_sq_into(&query, &mut got);

        for (i, v) in vecs.iter().enumerate() {
            assert!((got[i] - euclidean_sq_ref(&query, v)).abs() < 1e-4);
        }
    }

    #[test]
    fn index_matches_reference_across_block_boundaries() {
        // 50 vectors = 3 full blocks + a partial, so boundary handling is
        // exercised rather than assumed.
        let dim = 129; // deliberately not a multiple of any vector width
        let vecs = make(50, dim);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let index = PdxIndex::from_rows(&refs);
        assert_eq!(index.len(), 50);
        assert_eq!(index.block_count(), 4);

        let query: Vec<f32> = (0..dim).map(|d| (d % 11) as f32 / 11.0).collect();
        let got = index.euclidean_sq(&query);

        assert_eq!(got.len(), 50);
        for (i, v) in vecs.iter().enumerate() {
            let want = euclidean_sq_ref(&query, v);
            assert!(
                (got[i] - want).abs() < 1e-3,
                "vector {i}: got {}, want {want}",
                got[i]
            );
        }
    }

    #[test]
    fn single_vector_index_works() {
        let dim = 32;
        let v: Vec<f32> = (0..dim).map(|d| d as f32).collect();
        let index = PdxIndex::from_rows(&[v.as_slice()]);
        let got = index.euclidean_sq(&v);
        assert_eq!(got.len(), 1);
        assert!(got[0].abs() < 1e-5, "distance to self must be zero");
    }

    #[test]
    #[should_panic(expected = "expected 4")]
    fn ragged_block_is_rejected() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0];
        PdxBlock::from_rows(&[&a[..], &b[..]], 4);
    }

    #[test]
    #[should_panic(expected = "at most")]
    fn oversized_block_is_rejected() {
        let v = vec![0.0f32; 8];
        let refs: Vec<&[f32]> = (0..LANES + 1).map(|_| v.as_slice()).collect();
        PdxBlock::from_rows(&refs, 8);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn undersized_output_is_rejected() {
        let dim = 16;
        let vecs = make(20, dim);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let index = PdxIndex::from_rows(&refs);
        let mut out = vec![0.0; 5];
        index.euclidean_sq_into(&vec![0.0; dim], &mut out);
    }
}
