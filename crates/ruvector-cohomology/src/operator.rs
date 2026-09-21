//! Serializable linear restriction operators.
//!
//! Restriction maps are explicit, hashable data — never closures — so they
//! can be serialized, transposed, norm-checked, and canonically hashed
//! (issue #669, implementation gap 4).

use crate::{CohomologyError, ValidationLimits};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Row-major dense matrix used for small blocks (restriction maps, frames,
/// reference-oracle assembly). Deterministic by construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major entries, `data[r * cols + c]`.
    pub data: Vec<f64>,
}

impl DenseMatrix {
    /// Zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Build from row-major data.
    pub fn from_rows(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, CohomologyError> {
        if data.len() != rows * cols {
            return Err(CohomologyError::DimensionMismatch {
                expected: rows * cols,
                actual: data.len(),
                context: "DenseMatrix::from_rows".into(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Entry accessor.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Entry mutator.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// `y = A x`.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        for r in 0..self.rows {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            let mut acc = 0.0;
            for (a, b) in row.iter().zip(x.iter()) {
                acc += a * b;
            }
            y[r] = acc;
        }
    }

    /// `x = Aᵀ y`.
    pub fn matvec_transpose(&self, y: &[f64], x: &mut [f64]) {
        debug_assert_eq!(y.len(), self.rows);
        debug_assert_eq!(x.len(), self.cols);
        x.fill(0.0);
        for r in 0..self.rows {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            let yr = y[r];
            for (xc, a) in x.iter_mut().zip(row.iter()) {
                *xc += a * yr;
            }
        }
    }

    /// `C = A · B`.
    pub fn matmul(&self, other: &DenseMatrix) -> Result<DenseMatrix, CohomologyError> {
        if self.cols != other.rows {
            return Err(CohomologyError::DimensionMismatch {
                expected: self.cols,
                actual: other.rows,
                context: "DenseMatrix::matmul".into(),
            });
        }
        let mut out = DenseMatrix::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(r, k);
                if a == 0.0 {
                    continue;
                }
                for c in 0..other.cols {
                    out.data[r * other.cols + c] += a * other.get(k, c);
                }
            }
        }
        Ok(out)
    }

    /// Transposed copy.
    pub fn transpose(&self) -> DenseMatrix {
        let mut out = DenseMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Frobenius norm (an upper bound on the spectral norm).
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Maximum deviation of `AᵀA` from the identity (orthonormal-column check).
    pub fn orthogonality_defect(&self) -> f64 {
        let mut defect: f64 = 0.0;
        for i in 0..self.cols {
            for j in i..self.cols {
                let mut dot = 0.0;
                for r in 0..self.rows {
                    dot += self.get(r, i) * self.get(r, j);
                }
                let target = if i == j { 1.0 } else { 0.0 };
                defect = defect.max((dot - target).abs());
            }
        }
        defect
    }

    /// Feed the exact bit pattern of the matrix into a hasher.
    pub fn hash_into(&self, hasher: &mut Sha256) {
        hasher.update((self.rows as u64).to_le_bytes());
        hasher.update((self.cols as u64).to_le_bytes());
        for v in &self.data {
            hasher.update(v.to_bits().to_le_bytes());
        }
    }

    /// All entries finite?
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }
}

/// A linear restriction map with explicit dimensions, transpose application,
/// an operator norm bound, and a canonical hash (issue #669 core interface).
pub trait LinearRestriction: Send + Sync {
    /// Dimension of the vertex stalk this map consumes.
    fn input_dim(&self) -> usize;
    /// Dimension of the edge stalk this map produces.
    fn output_dim(&self) -> usize;
    /// `y = ρ x`.
    fn apply(&self, x: &[f64], y: &mut [f64]);
    /// `x = ρᵀ y`.
    fn apply_transpose(&self, y: &[f64], x: &mut [f64]);
    /// A sound upper bound on the spectral norm ‖ρ‖₂.
    fn operator_norm_bound(&self) -> f64;
    /// Deterministic SHA-256 hash of the exact operator content.
    fn canonical_hash(&self) -> [u8; 32];
}

/// Constrained, serializable restriction map families (issue #669 §6: ship
/// identity, projection, orthogonal, and contractive operators first;
/// learned maps stay advisory).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RestrictionMap {
    /// Identity on a `dim`-dimensional stalk.
    Identity {
        /// Stalk dimension.
        dim: usize,
    },
    /// Scalar multiple of the identity (contractive when `|factor| ≤ 1`).
    Scale {
        /// Stalk dimension.
        dim: usize,
        /// Scale factor.
        factor: f64,
    },
    /// Coordinate projection selecting `coords` (strictly increasing) from an
    /// `input_dim`-dimensional stalk.
    Projection {
        /// Input stalk dimension.
        input_dim: usize,
        /// Selected coordinates, strictly increasing.
        coords: Vec<usize>,
    },
    /// Matrix with orthonormal rows or columns (validated).
    Orthogonal {
        /// The orthogonal block.
        matrix: DenseMatrix,
    },
    /// General dense map. Accepted only when its norm bound passes
    /// validation; intended for vetted contractive or learned-map-advisory
    /// use.
    Dense {
        /// The dense block.
        matrix: DenseMatrix,
    },
}

impl RestrictionMap {
    /// Validate the map against boundary limits before it is admitted.
    pub fn validate(&self, limits: &ValidationLimits) -> Result<(), CohomologyError> {
        let check_dims = |i: usize, o: usize| -> Result<(), CohomologyError> {
            if i == 0 || o == 0 || i > limits.max_dim || o > limits.max_dim {
                return Err(CohomologyError::ValidationFailed(format!(
                    "restriction dims ({o}×{i}) outside (0, {}]",
                    limits.max_dim
                )));
            }
            Ok(())
        };
        match self {
            Self::Identity { dim } => check_dims(*dim, *dim),
            Self::Scale { dim, factor } => {
                check_dims(*dim, *dim)?;
                if !factor.is_finite() || factor.abs() > limits.max_operator_norm {
                    return Err(CohomologyError::ValidationFailed(format!(
                        "scale factor {factor} not finite or exceeds norm limit"
                    )));
                }
                Ok(())
            }
            Self::Projection { input_dim, coords } => {
                check_dims(*input_dim, coords.len())?;
                let increasing = coords.windows(2).all(|w| w[0] < w[1]);
                let in_range = coords.iter().all(|&c| c < *input_dim);
                if !increasing || !in_range {
                    return Err(CohomologyError::ValidationFailed(
                        "projection coords must be strictly increasing and in range".into(),
                    ));
                }
                Ok(())
            }
            Self::Orthogonal { matrix } => {
                check_dims(matrix.cols, matrix.rows)?;
                if !matrix.is_finite() {
                    return Err(CohomologyError::ValidationFailed(
                        "orthogonal map has non-finite entries".into(),
                    ));
                }
                let defect = if matrix.rows >= matrix.cols {
                    matrix.orthogonality_defect()
                } else {
                    matrix.transpose().orthogonality_defect()
                };
                if defect > limits.orthogonality_tol {
                    return Err(CohomologyError::ValidationFailed(format!(
                        "orthogonality defect {defect:.3e} exceeds tolerance {:.3e}",
                        limits.orthogonality_tol
                    )));
                }
                Ok(())
            }
            Self::Dense { matrix } => {
                check_dims(matrix.cols, matrix.rows)?;
                if !matrix.is_finite() {
                    return Err(CohomologyError::ValidationFailed(
                        "dense map has non-finite entries".into(),
                    ));
                }
                if matrix.frobenius_norm() > limits.max_operator_norm {
                    return Err(CohomologyError::ValidationFailed(
                        "dense map norm bound exceeds limit".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Write the dense representation into `out` (used by the reference
    /// oracle). `out` must be `output_dim × input_dim`.
    pub fn to_dense(&self) -> DenseMatrix {
        match self {
            Self::Identity { dim } => DenseMatrix::identity(*dim),
            Self::Scale { dim, factor } => {
                let mut m = DenseMatrix::zeros(*dim, *dim);
                for i in 0..*dim {
                    m.set(i, i, *factor);
                }
                m
            }
            Self::Projection { input_dim, coords } => {
                let mut m = DenseMatrix::zeros(coords.len(), *input_dim);
                for (r, &c) in coords.iter().enumerate() {
                    m.set(r, c, 1.0);
                }
                m
            }
            Self::Orthogonal { matrix } | Self::Dense { matrix } => matrix.clone(),
        }
    }

    /// Right-compose with an orthogonal frame change: returns `ρ · Rᵀ`.
    ///
    /// Used by [`crate::transport`] when a vertex re-expresses its stalk in a
    /// new orthonormal frame `R = Q_new Q_oldᵀ`: stored states transform as
    /// `x_new = R x_old`, so restriction maps must become `ρ Rᵀ` to leave
    /// every coboundary — and therefore the syndrome — unchanged.
    pub fn compose_transpose_frame(
        &self,
        r_frame: &DenseMatrix,
    ) -> Result<RestrictionMap, CohomologyError> {
        if r_frame.rows != self.input_dim() || r_frame.cols != self.input_dim() {
            return Err(CohomologyError::DimensionMismatch {
                expected: self.input_dim(),
                actual: r_frame.rows,
                context: "compose_transpose_frame".into(),
            });
        }
        let composed = self.to_dense().matmul(&r_frame.transpose())?;
        // Composition with an orthogonal map preserves orthogonality of
        // orthogonal/identity families; everything else stays Dense.
        match self {
            Self::Identity { .. } | Self::Orthogonal { .. } => {
                Ok(RestrictionMap::Orthogonal { matrix: composed })
            }
            _ => Ok(RestrictionMap::Dense { matrix: composed }),
        }
    }
}

impl LinearRestriction for RestrictionMap {
    fn input_dim(&self) -> usize {
        match self {
            Self::Identity { dim } | Self::Scale { dim, .. } => *dim,
            Self::Projection { input_dim, .. } => *input_dim,
            Self::Orthogonal { matrix } | Self::Dense { matrix } => matrix.cols,
        }
    }

    fn output_dim(&self) -> usize {
        match self {
            Self::Identity { dim } | Self::Scale { dim, .. } => *dim,
            Self::Projection { coords, .. } => coords.len(),
            Self::Orthogonal { matrix } | Self::Dense { matrix } => matrix.rows,
        }
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.input_dim());
        debug_assert_eq!(y.len(), self.output_dim());
        match self {
            Self::Identity { .. } => y.copy_from_slice(x),
            Self::Scale { factor, .. } => {
                for (yi, xi) in y.iter_mut().zip(x.iter()) {
                    *yi = factor * xi;
                }
            }
            Self::Projection { coords, .. } => {
                for (yi, &c) in y.iter_mut().zip(coords.iter()) {
                    *yi = x[c];
                }
            }
            Self::Orthogonal { matrix } | Self::Dense { matrix } => matrix.matvec(x, y),
        }
    }

    fn apply_transpose(&self, y: &[f64], x: &mut [f64]) {
        debug_assert_eq!(y.len(), self.output_dim());
        debug_assert_eq!(x.len(), self.input_dim());
        match self {
            Self::Identity { .. } => x.copy_from_slice(y),
            Self::Scale { factor, .. } => {
                for (xi, yi) in x.iter_mut().zip(y.iter()) {
                    *xi = factor * yi;
                }
            }
            Self::Projection { coords, .. } => {
                x.fill(0.0);
                for (yi, &c) in y.iter().zip(coords.iter()) {
                    x[c] = *yi;
                }
            }
            Self::Orthogonal { matrix } | Self::Dense { matrix } => {
                matrix.matvec_transpose(y, x)
            }
        }
    }

    fn operator_norm_bound(&self) -> f64 {
        match self {
            Self::Identity { .. } | Self::Projection { .. } | Self::Orthogonal { .. } => 1.0,
            Self::Scale { factor, .. } => factor.abs(),
            Self::Dense { matrix } => matrix.frobenius_norm(),
        }
    }

    fn canonical_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        match self {
            Self::Identity { dim } => {
                h.update([0u8]);
                h.update((*dim as u64).to_le_bytes());
            }
            Self::Scale { dim, factor } => {
                h.update([1u8]);
                h.update((*dim as u64).to_le_bytes());
                h.update(factor.to_bits().to_le_bytes());
            }
            Self::Projection { input_dim, coords } => {
                h.update([2u8]);
                h.update((*input_dim as u64).to_le_bytes());
                h.update((coords.len() as u64).to_le_bytes());
                for c in coords {
                    h.update((*c as u64).to_le_bytes());
                }
            }
            Self::Orthogonal { matrix } => {
                h.update([3u8]);
                matrix.hash_into(&mut h);
            }
            Self::Dense { matrix } => {
                h.update([4u8]);
                matrix.hash_into(&mut h);
            }
        }
        h.finalize().into()
    }
}
