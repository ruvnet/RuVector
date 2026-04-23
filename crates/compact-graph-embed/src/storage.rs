use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::error::EmbedError;
use crate::traits::TokenStorage;

// ---------------------------------------------------------------------------
// EmbeddingTableF32
// ---------------------------------------------------------------------------

/// Float32 embedding table storing one embedding per (anchor, dist) pair.
///
/// Layout: `data[anchor_idx * max_dist_slots * dim + (dist-1) * dim .. +dim]`
pub struct EmbeddingTableF32 {
    data: Vec<f32>,
    num_anchors: usize,
    /// Number of distance slots (equals max_dist as usize; dist 1..=max_dist maps to index 0..max_dist).
    max_dist_slots: usize,
    dim: usize,
}

impl EmbeddingTableF32 {
    /// Create a table with random weights in `[-0.1, 0.1]` using a seeded SmallRng.
    pub fn new_random(num_anchors: usize, max_dist: u8, dim: usize, seed: u64) -> Self {
        let max_dist_slots = max_dist as usize;
        let total = num_anchors * max_dist_slots * dim;
        let mut rng = SmallRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..total).map(|_| rng.gen_range(-0.1_f32..=0.1_f32)).collect();
        EmbeddingTableF32 {
            data,
            num_anchors,
            max_dist_slots,
            dim,
        }
    }

    /// Byte size of the raw float data.
    pub fn byte_size(&self) -> usize {
        self.data.len() * 4
    }

    /// Number of anchor nodes this table was built for.
    pub fn num_anchors(&self) -> usize {
        self.num_anchors
    }

    /// Number of distance slots (max_dist).
    pub fn max_dist_slots(&self) -> usize {
        self.max_dist_slots
    }

    /// Compute the row start index (no bounds check — use after validation).
    #[inline]
    fn row_start_unchecked(&self, anchor_idx: u32, dist_idx: usize) -> usize {
        (anchor_idx as usize * self.max_dist_slots + dist_idx) * self.dim
    }

    /// Validate anchor and dist, return (dist_idx, row_start).
    #[inline]
    fn validate(&self, anchor_idx: u32, dist: u8) -> Result<usize, EmbedError> {
        if anchor_idx as usize >= self.num_anchors {
            return Err(EmbedError::AnchorOutOfRange(anchor_idx));
        }
        let dist_idx = dist
            .checked_sub(1)
            .ok_or(EmbedError::DistanceTooLarge(0, 0))? as usize;
        if dist_idx >= self.max_dist_slots {
            return Err(EmbedError::DistanceTooLarge(dist, self.max_dist_slots as u8));
        }
        Ok(self.row_start_unchecked(anchor_idx, dist_idx))
    }

    /// Raw access to a row slice (anchor_idx, dist).
    pub(crate) fn row_slice(&self, anchor_idx: u32, dist: u8) -> Result<&[f32], EmbedError> {
        let row_start = self.validate(anchor_idx, dist)?;
        Ok(&self.data[row_start..row_start + self.dim])
    }
}

impl TokenStorage for EmbeddingTableF32 {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_entries(&self) -> usize {
        self.num_anchors * self.max_dist_slots
    }

    fn embed_token(&self, anchor_idx: u32, dist: u8) -> Result<Vec<f32>, EmbedError> {
        Ok(self.row_slice(anchor_idx, dist)?.to_vec())
    }

    fn byte_size(&self) -> usize {
        self.byte_size()
    }

    #[inline]
    fn accumulate_token(
        &self,
        anchor_idx: u32,
        dist: u8,
        acc: &mut [f32],
    ) -> Result<(), EmbedError> {
        let row_start = self.validate(anchor_idx, dist)?;
        let row = &self.data[row_start..row_start + self.dim];
        for (a, &e) in acc.iter_mut().zip(row.iter()) {
            *a += e;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EmbeddingTableI8
// ---------------------------------------------------------------------------

/// Row-quantized int8 embedding table.
///
/// Each row (anchor, dist) is quantized independently.
/// Dequantization: `v[i] = data[row_start + i] as f32 * scale`.
pub struct EmbeddingTableI8 {
    data: Vec<i8>,
    /// One scale per (anchor, dist) row.
    scales: Vec<f32>,
    num_anchors: usize,
    max_dist_slots: usize,
    dim: usize,
}

impl EmbeddingTableI8 {
    /// Build an int8 table by quantizing each row of an f32 table.
    ///
    /// For each row: `scale = max(|v|) / 127.0` (or `1e-9` if all zeros).
    /// Quantized: `i8 = round(v / scale).clamp(-127, 127)`.
    pub fn from_f32(table: &EmbeddingTableF32) -> Self {
        let num_rows = table.num_anchors() * table.max_dist_slots();
        let dim = table.dim();
        let mut data: Vec<i8> = Vec::with_capacity(num_rows * dim);
        let mut scales: Vec<f32> = Vec::with_capacity(num_rows);

        for row_idx in 0..num_rows {
            let row_start = row_idx * dim;
            let row = &table.data[row_start..row_start + dim];

            // Compute max absolute value
            let max_abs = row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

            let scale = if max_abs < 1e-9 {
                1e-9_f32
            } else {
                max_abs / 127.0_f32
            };
            scales.push(scale);

            // Quantize
            for &v in row {
                let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                data.push(q);
            }
        }

        EmbeddingTableI8 {
            data,
            scales,
            num_anchors: table.num_anchors(),
            max_dist_slots: table.max_dist_slots(),
            dim,
        }
    }

    /// Byte size: i8 data + f32 scales.
    pub fn byte_size(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Number of anchor nodes this table was built for.
    pub fn num_anchors(&self) -> usize {
        self.num_anchors
    }

    /// Number of distance slots (max_dist).
    pub fn max_dist_slots(&self) -> usize {
        self.max_dist_slots
    }

    /// Validate and compute row index.
    #[inline]
    fn validate(&self, anchor_idx: u32, dist: u8) -> Result<usize, EmbedError> {
        if anchor_idx as usize >= self.num_anchors {
            return Err(EmbedError::AnchorOutOfRange(anchor_idx));
        }
        let dist_idx = dist
            .checked_sub(1)
            .ok_or(EmbedError::DistanceTooLarge(0, 0))? as usize;
        if dist_idx >= self.max_dist_slots {
            return Err(EmbedError::DistanceTooLarge(dist, self.max_dist_slots as u8));
        }
        Ok(anchor_idx as usize * self.max_dist_slots + dist_idx)
    }
}

impl TokenStorage for EmbeddingTableI8 {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_entries(&self) -> usize {
        self.num_anchors * self.max_dist_slots
    }

    fn embed_token(&self, anchor_idx: u32, dist: u8) -> Result<Vec<f32>, EmbedError> {
        let row_idx = self.validate(anchor_idx, dist)?;
        let scale = self.scales[row_idx];
        let row_start = row_idx * self.dim;
        let result: Vec<f32> = self.data[row_start..row_start + self.dim]
            .iter()
            .map(|&q| q as f32 * scale)
            .collect();
        Ok(result)
    }

    fn byte_size(&self) -> usize {
        self.byte_size()
    }

    #[inline]
    fn accumulate_token(
        &self,
        anchor_idx: u32,
        dist: u8,
        acc: &mut [f32],
    ) -> Result<(), EmbedError> {
        let row_idx = self.validate(anchor_idx, dist)?;
        let scale = self.scales[row_idx];
        let row_start = row_idx * self.dim;
        let row = &self.data[row_start..row_start + self.dim];
        for (a, &q) in acc.iter_mut().zip(row.iter()) {
            *a += q as f32 * scale;
        }
        Ok(())
    }
}
