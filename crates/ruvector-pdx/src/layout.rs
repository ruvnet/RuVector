//! PDX block: columnar vector storage within a fixed-size partition.
//!
//! Layout: data[dim * block_size + vec_idx] = corpus[vec_idx][dim]
//!
//! Accessing column d: &data[d * block_size .. (d+1) * block_size]
//! This is contiguous in memory — SIMD-friendly.
//!
//! Contrast with row-major: data[vec_idx * dim + d] — accessing a single
//! dimension across all vectors requires stride jumps of `dim` floats.

/// A columnar (PDX) block storing up to `block_size` vectors of `dim` floats.
///
/// Memory: dim × block_size × 4 bytes (no per-vector padding).
#[derive(Debug, Clone)]
pub struct PdxBlock {
    pub dim: usize,
    pub block_size: usize,
    /// Actual number of vectors stored (≤ block_size).
    pub n: usize,
    /// Column-major: data[d * block_size + i] = vector[i][d].
    pub data: Vec<f32>,
    /// External IDs of stored vectors.
    pub ids: Vec<usize>,
}

impl PdxBlock {
    /// Create an empty block.
    pub fn new(dim: usize, block_size: usize) -> Self {
        Self {
            dim,
            block_size,
            n: 0,
            data: vec![0.0f32; dim * block_size],
            ids: Vec::with_capacity(block_size),
        }
    }

    /// Add one vector. Returns `false` if block is full.
    pub fn push(&mut self, id: usize, vector: &[f32]) -> bool {
        debug_assert_eq!(vector.len(), self.dim);
        if self.n >= self.block_size {
            return false;
        }
        for (d, &v) in vector.iter().enumerate() {
            self.data[d * self.block_size + self.n] = v;
        }
        self.ids.push(id);
        self.n += 1;
        true
    }

    /// Column slice: the N values of dimension `d` across all stored vectors.
    #[inline]
    pub fn col(&self, d: usize) -> &[f32] {
        &self.data[d * self.block_size..d * self.block_size + self.n]
    }

    /// Memory used in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 4 + self.ids.len() * 8 + std::mem::size_of::<Self>()
    }

    /// Convert from a row-major slice of vectors.
    pub fn from_rows(dim: usize, block_size: usize, rows: &[(usize, Vec<f32>)]) -> Self {
        let mut block = Self::new(dim, block_size);
        for (id, vec) in rows {
            block.push(*id, vec);
        }
        block
    }
}

// ── row-major block (for fair baseline comparison) ────────────────────────────

/// Row-major block: data[vec_idx * dim + d] = vector[vec_idx][d].
#[derive(Debug, Clone)]
pub struct RowBlock {
    pub dim: usize,
    pub n: usize,
    /// Row-major: data[i * dim + d].
    pub data: Vec<f32>,
    pub ids: Vec<usize>,
}

impl RowBlock {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            dim,
            n: 0,
            data: Vec::with_capacity(dim * capacity),
            ids: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, id: usize, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        self.data.extend_from_slice(vector);
        self.ids.push(id);
        self.n += 1;
    }

    /// Row slice for vector `i`.
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 4 + self.ids.len() * 8 + std::mem::size_of::<Self>()
    }
}
