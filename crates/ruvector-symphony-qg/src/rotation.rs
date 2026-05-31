//! Random orthogonal rotation via Gram-Schmidt on a Gaussian matrix.
//!
//! We generate a D×D random normal matrix and orthogonalise it column-by-column
//! using the modified Gram-Schmidt process. The result is a true orthogonal
//! matrix (not merely random projections), matching the RaBitQ rotation
//! construction used in SymphonyQG.
//!
//! For PoC scale (D ≤ 256) this is fast. Production would cache the matrix.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Generates a D×D orthogonal rotation matrix with a fixed seed.
/// Stored in row-major order: entry (i,j) = matrix[i*dim + j].
pub fn random_orthogonal(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f64, 1.0).unwrap();

    // Sample D × D Gaussian matrix stored as columns for GSO
    let mut cols: Vec<Vec<f64>> = (0..dim)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // Modified Gram-Schmidt orthogonalisation
    for j in 0..dim {
        // Normalise column j
        let norm = cols[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            // Degenerate column — replace with a standard basis vector
            cols[j] = vec![0.0; dim];
            cols[j][j] = 1.0;
        } else {
            for x in cols[j].iter_mut() {
                *x /= norm;
            }
        }
        // Project out column j from all subsequent columns
        let cj = cols[j].clone();
        for k in (j + 1)..dim {
            let dot: f64 = cols[k].iter().zip(cj.iter()).map(|(a, b)| a * b).sum();
            for (ck, cj_val) in cols[k].iter_mut().zip(cj.iter()) {
                *ck -= dot * cj_val;
            }
        }
    }

    // Transpose: result[i][j] = cols[j][i], stored row-major so R[i,j] = result[i*dim+j]
    let mut matrix = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            matrix[i * dim + j] = cols[j][i] as f32;
        }
    }
    matrix
}

/// Apply rotation: y = R × x, result length = dim.
#[inline]
pub fn rotate(matrix: &[f32], x: &[f32], dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; dim];
    for i in 0..dim {
        let row = &matrix[i * dim..(i + 1) * dim];
        y[i] = row.iter().zip(x.iter()).map(|(r, v)| r * v).sum();
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonality() {
        let dim = 8;
        let r = random_orthogonal(dim, 42);
        // Check R × Rᵀ ≈ I
        for i in 0..dim {
            for j in 0..dim {
                let dot: f32 = (0..dim)
                    .map(|k| r[i * dim + k] * r[j * dim + k])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-5, "R×Rᵀ[{i},{j}] = {dot}");
            }
        }
    }
}
