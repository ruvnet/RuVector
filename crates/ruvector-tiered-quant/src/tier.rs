//! Tier types and vector record layout.

/// Which precision tier a vector currently occupies.
pub enum TierKind {
    /// Full f32 precision; highest quality, highest memory cost.
    Hot { f32s: Vec<f32> },
    /// Scalar-quantized u8 (one byte per dimension); ~4x compression.
    Warm {
        bytes: Vec<u8>,
        /// Per-dimension minimum used during quantization.
        min: Vec<f32>,
        /// Per-dimension step size: `scale[i] = (max[i] - min[i]) / 255.0`.
        scale: Vec<f32>,
    },
    /// 1-bit binary quantized; ~32x compression, highest approximation error.
    Cold {
        /// Packed bits: each u64 holds 64 dimensions.
        packed: Vec<u64>,
    },
}

impl TierKind {
    /// Approximate heap bytes used by this tier for a vector of `dims` dimensions.
    pub fn bytes_used(&self, dims: usize) -> usize {
        match self {
            TierKind::Hot { .. } => dims * 4,
            TierKind::Warm { .. } => dims + dims * 4 * 2, // bytes + min + scale
            TierKind::Cold { packed } => packed.len() * 8,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TierKind::Hot { .. } => "hot",
            TierKind::Warm { .. } => "warm",
            TierKind::Cold { .. } => "cold",
        }
    }
}

/// A stored vector along with its access metadata.
pub struct VectorRecord {
    pub id: usize,
    pub access_count: u32,
    pub tier: TierKind,
}

/// Aggregate statistics about tier distribution.
#[derive(Debug, Default, Clone)]
pub struct TierStats {
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
    /// Total estimated heap bytes across all tiers.
    pub total_bytes: usize,
}

impl TierStats {
    pub fn total_count(&self) -> usize {
        self.hot_count + self.warm_count + self.cold_count
    }

    pub fn hot_pct(&self) -> f64 {
        let n = self.total_count();
        if n == 0 {
            0.0
        } else {
            self.hot_count as f64 / n as f64 * 100.0
        }
    }

    pub fn warm_pct(&self) -> f64 {
        let n = self.total_count();
        if n == 0 {
            0.0
        } else {
            self.warm_count as f64 / n as f64 * 100.0
        }
    }

    pub fn cold_pct(&self) -> f64 {
        let n = self.total_count();
        if n == 0 {
            0.0
        } else {
            self.cold_count as f64 / n as f64 * 100.0
        }
    }

    /// Compression ratio relative to all-f32 storage.
    pub fn compression_ratio(&self, dims: usize) -> f64 {
        let n = self.total_count();
        if n == 0 || dims == 0 {
            return 1.0;
        }
        let baseline = n * dims * 4;
        if baseline == 0 {
            1.0
        } else {
            baseline as f64 / self.total_bytes.max(1) as f64
        }
    }
}
