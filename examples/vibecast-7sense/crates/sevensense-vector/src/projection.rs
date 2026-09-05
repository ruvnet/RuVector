//! Dimensionality reduction for the manifold view.
//!
//! ADR-012 selects PCA via randomized SVD as the default projection rather than
//! UMAP. Two properties decide it: the projection is deterministic, so saved
//! viewports stay valid and two people discussing "the cluster on the left" mean
//! the same cluster; and once fitted, projecting a new point is a single matrix
//! multiply, which is what the streaming path needs. UMAP has no cheap exact
//! out-of-sample extension.
//!
//! Randomized SVD is used because the embedding dimension is 1536 while only the
//! top two or three components are wanted. A full SVD would compute 1536
//! components to discard 1533 of them.
//!
//! # Example
//!
//! ```
//! use sevensense_vector::projection::PcaProjection;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Points spread along one axis, with small variation on the others.
//! let data: Vec<Vec<f32>> = (0..50)
//!     .map(|i| vec![i as f32, (i % 3) as f32 * 0.01, 0.0])
//!     .collect();
//!
//! let pca = PcaProjection::fit(&data, 2, 42)?;
//! let xy = pca.project(&data[0])?;
//!
//! assert_eq!(xy.len(), 2);
//! // Nearly all variance lies on the first component.
//! assert!(pca.explained_variance()[0] > 0.99);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Extra dimensions sampled beyond the target, improving subspace accuracy.
const OVERSAMPLES: usize = 6;

/// Power iterations applied to sharpen the sampled subspace.
///
/// Acoustic embeddings have a slowly decaying spectrum, where a single pass
/// leaves the leading components mixed. Two iterations is the usual recommendation
/// and is cheap at this size.
const POWER_ITERATIONS: usize = 2;

/// Jacobi sweeps used for the small symmetric eigenproblem.
const JACOBI_SWEEPS: usize = 60;

/// Largest radius a Poincare image may take.
///
/// The ball is open, so points must stay strictly inside it. This bound is far
/// enough from 1.0 to survive the `f32` rounding in a subsequent distance
/// computation.
const MAX_BALL_RADIUS: f32 = 1.0 - 1e-5;

/// Errors from fitting or applying a projection.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProjectionError {
    /// Fitting needs more points than target dimensions.
    #[error("need at least {needed} points to fit {dims} dimensions, got {got}")]
    TooFewPoints {
        /// Points required.
        needed: usize,
        /// Points supplied.
        got: usize,
        /// Target dimensionality.
        dims: usize,
    },
    /// Input vectors disagree on length.
    #[error("inconsistent vector lengths: expected {expected}, found {found}")]
    DimensionMismatch {
        /// Length established by the first vector.
        expected: usize,
        /// Length encountered.
        found: usize,
    },
    /// Target dimensionality is zero or exceeds the input dimensionality.
    #[error("cannot project {input_dim}-dimensional data to {dims} dimensions")]
    InvalidTarget {
        /// Input dimensionality.
        input_dim: usize,
        /// Requested output dimensionality.
        dims: usize,
    },
    /// The input carries no variance, so no meaningful axis exists.
    #[error("input has no variance; every point is identical")]
    DegenerateInput,
}

/// Which projection produced a set of coordinates.
///
/// Carried in the payload so a client can label the view honestly rather than
/// implying every mode has the same properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectionMethod {
    /// Deterministic linear projection. Extends to new points exactly.
    Pca,
    /// Poincare ball embedding, for hierarchical structure.
    Poincare,
}

/// A fitted PCA projection.
///
/// Holds the training mean and the leading principal components, which is all
/// that projecting a new point requires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcaProjection {
    /// Per-dimension mean of the training data.
    mean: Vec<f32>,
    /// Leading principal components, each of length `input_dim`.
    components: Vec<Vec<f32>>,
    /// Fraction of total variance captured by each component.
    explained_variance: Vec<f32>,
    /// Dimensionality of the input vectors.
    input_dim: usize,
}

impl PcaProjection {
    /// Fits a projection to `data`, reducing to `dims` dimensions.
    ///
    /// `seed` makes the randomized SVD reproducible; the same data and seed
    /// always produce the same components, which is the point of choosing PCA.
    ///
    /// # Errors
    /// Returns [`ProjectionError`] when there are too few points, when vector
    /// lengths disagree, when `dims` is invalid, or when the data has no variance.
    pub fn fit(data: &[Vec<f32>], dims: usize, seed: u64) -> Result<Self, ProjectionError> {
        let input_dim = validate(data, dims)?;
        let n = data.len();

        // Centering is what makes the components describe variance rather than
        // the offset of the cloud from the origin.
        let mut mean = vec![0.0f32; input_dim];
        for point in data {
            for (m, v) in mean.iter_mut().zip(point.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }

        let centered: Vec<Vec<f32>> = data
            .iter()
            .map(|p| p.iter().zip(mean.iter()).map(|(v, m)| v - m).collect())
            .collect();

        let total_variance: f32 = centered
            .iter()
            .map(|p| p.iter().map(|v| v * v).sum::<f32>())
            .sum::<f32>()
            / n as f32;

        if total_variance <= f32::EPSILON {
            return Err(ProjectionError::DegenerateInput);
        }

        let sketch_width = (dims + OVERSAMPLES).min(n).min(input_dim);
        let (components, eigenvalues) =
            randomized_subspace(&centered, sketch_width, dims, input_dim, seed);

        let explained_variance = eigenvalues
            .iter()
            .map(|lambda| (lambda / n as f32 / total_variance).clamp(0.0, 1.0))
            .collect();

        Ok(Self {
            mean,
            components,
            explained_variance,
            input_dim,
        })
    }

    /// Output dimensionality.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.components.len()
    }

    /// Input dimensionality this projection was fitted for.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Fraction of total variance captured by each component.
    ///
    /// On 1536-dimensional acoustic embeddings this typically sums to about 0.3
    /// across three components. The view is a real simplification, and surfacing
    /// this lets the UI say so rather than implying the picture is complete.
    #[must_use]
    pub fn explained_variance(&self) -> &[f32] {
        &self.explained_variance
    }

    /// Projects a single vector. This is the out-of-sample path: one dot product
    /// per component, so streaming points are cheap.
    ///
    /// # Errors
    /// Returns [`ProjectionError::DimensionMismatch`] when `point` has the wrong
    /// length.
    pub fn project(&self, point: &[f32]) -> Result<Vec<f32>, ProjectionError> {
        if point.len() != self.input_dim {
            return Err(ProjectionError::DimensionMismatch {
                expected: self.input_dim,
                found: point.len(),
            });
        }

        Ok(self
            .components
            .iter()
            .map(|component| {
                point
                    .iter()
                    .zip(self.mean.iter())
                    .zip(component.iter())
                    .map(|((v, m), c)| (v - m) * c)
                    .sum()
            })
            .collect())
    }

    /// Projects many vectors.
    ///
    /// # Errors
    /// Returns [`ProjectionError::DimensionMismatch`] on the first vector with
    /// the wrong length.
    pub fn project_batch(&self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, ProjectionError> {
        data.iter().map(|p| self.project(p)).collect()
    }
}

/// Rescales coordinates into `[-1, 1]` per axis, preserving relative position.
///
/// Normalizing server-side means a client never needs to know the data's scale to
/// frame it, and avoids every client reimplementing this slightly differently.
/// Axes with no extent map to 0.0 rather than dividing by zero.
#[must_use]
pub fn normalize_coordinates(points: &mut [Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let dims = points[0].len();
    let mut min = vec![f32::INFINITY; dims];
    let mut max = vec![f32::NEG_INFINITY; dims];

    for point in points.iter() {
        for (axis, value) in point.iter().enumerate().take(dims) {
            min[axis] = min[axis].min(*value);
            max[axis] = max[axis].max(*value);
        }
    }

    for point in points.iter_mut() {
        for (axis, value) in point.iter_mut().enumerate().take(dims) {
            let extent = max[axis] - min[axis];
            *value = if extent > f32::EPSILON {
                (*value - min[axis]) / extent * 2.0 - 1.0
            } else {
                0.0
            };
        }
    }

    (min, max)
}

/// Maps Euclidean coordinates into the open unit ball of the Poincare model.
///
/// Hierarchical structure shows up as radial depth: general calls sit near the
/// origin and specialized variants near the boundary, where hyperbolic space has
/// exponentially more room. Inputs are expected to be normalized already.
#[must_use]
pub fn to_poincare_ball(points: &[Vec<f32>], scale: f32) -> Vec<Vec<f32>> {
    points
        .iter()
        .map(|point| {
            let norm = point.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm <= f32::EPSILON {
                return point.clone();
            }
            // tanh compresses the radius, but saturates to exactly 1.0 in f32
            // once the argument passes about 9. The clamp keeps every image
            // strictly inside the ball, which the Poincare metric requires --
            // a point on the boundary is at infinite distance from everything.
            let radius = (norm * scale).tanh().min(MAX_BALL_RADIUS);
            let factor = radius / norm;
            point.iter().map(|v| v * factor).collect()
        })
        .collect()
}

/// Validates fit inputs, returning the established input dimensionality.
fn validate(data: &[Vec<f32>], dims: usize) -> Result<usize, ProjectionError> {
    if data.len() <= dims {
        return Err(ProjectionError::TooFewPoints {
            needed: dims + 1,
            got: data.len(),
            dims,
        });
    }

    let input_dim = data[0].len();
    if dims == 0 || dims > input_dim {
        return Err(ProjectionError::InvalidTarget { input_dim, dims });
    }

    for point in data {
        if point.len() != input_dim {
            return Err(ProjectionError::DimensionMismatch {
                expected: input_dim,
                found: point.len(),
            });
        }
    }

    Ok(input_dim)
}

/// Finds the leading `dims` principal directions by randomized subspace iteration.
///
/// Returns the components and their associated eigenvalues of the scatter matrix.
fn randomized_subspace(
    centered: &[Vec<f32>],
    sketch_width: usize,
    dims: usize,
    input_dim: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = Pcg32::new(seed);

    // Sketch the row space with a random Gaussian test matrix, then sharpen it
    // with power iterations.
    let mut basis: Vec<Vec<f32>> = (0..sketch_width)
        .map(|_| (0..input_dim).map(|_| rng.next_gaussian()).collect())
        .collect();
    orthonormalize(&mut basis);

    for _ in 0..POWER_ITERATIONS {
        basis = basis
            .iter()
            .map(|vector| scatter_multiply(centered, vector))
            .collect();
        orthonormalize(&mut basis);
    }

    // Project the scatter matrix into the small basis and solve there. The
    // eigenproblem is sketch_width squared, not input_dim squared.
    let width = basis.len();
    let mut small = vec![vec![0.0f32; width]; width];
    let projected: Vec<Vec<f32>> = basis
        .iter()
        .map(|vector| scatter_multiply(centered, vector))
        .collect();

    for (i, row) in small.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = dot(&basis[i], &projected[j]);
        }
    }
    // Force exact symmetry; accumulated rounding otherwise stalls Jacobi.
    for i in 0..width {
        for j in (i + 1)..width {
            let averaged = 0.5 * (small[i][j] + small[j][i]);
            small[i][j] = averaged;
            small[j][i] = averaged;
        }
    }

    let (eigenvalues, eigenvectors) = jacobi_eigen(&small);

    // Lift the small eigenvectors back into the original space.
    let mut order: Vec<usize> = (0..width).collect();
    order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut components = Vec::with_capacity(dims);
    let mut values = Vec::with_capacity(dims);

    for &index in order.iter().take(dims) {
        let mut component = vec![0.0f32; input_dim];
        for (b, basis_vector) in basis.iter().enumerate() {
            let weight = eigenvectors[b][index];
            for (c, value) in component.iter_mut().zip(basis_vector.iter()) {
                *c += weight * value;
            }
        }
        normalize_in_place(&mut component);
        components.push(component);
        values.push(eigenvalues[index].max(0.0));
    }

    (components, values)
}

/// Computes `X^T X v` without forming the scatter matrix.
///
/// The scatter matrix is 1536 by 1536; this keeps the cost at O(n·d) per call.
fn scatter_multiply(centered: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0f32; vector.len()];
    for row in centered {
        let projection = dot(row, vector);
        if projection != 0.0 {
            for (r, value) in result.iter_mut().zip(row.iter()) {
                *r += projection * value;
            }
        }
    }
    result
}

/// Modified Gram-Schmidt with reorthogonalization.
///
/// The second pass matters: a single pass loses orthogonality badly once the
/// vectors are nearly dependent, which is exactly what power iteration produces.
fn orthonormalize(vectors: &mut Vec<Vec<f32>>) {
    let mut kept: Vec<Vec<f32>> = Vec::with_capacity(vectors.len());

    for vector in vectors.iter() {
        let mut candidate = vector.clone();
        for _ in 0..2 {
            for basis in &kept {
                let projection = dot(&candidate, basis);
                for (c, b) in candidate.iter_mut().zip(basis.iter()) {
                    *c -= projection * b;
                }
            }
        }

        let norm = candidate.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for value in &mut candidate {
                *value /= norm;
            }
            kept.push(candidate);
        }
    }

    *vectors = kept;
}

/// Jacobi eigenvalue iteration for a small symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i][j]` is component
/// `i` of eigenvector `j`.
fn jacobi_eigen(matrix: &[Vec<f32>]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix
        .iter()
        .map(|row| row.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for _ in 0..JACOBI_SWEEPS {
        // Largest off-diagonal magnitude decides both the pivot and convergence.
        let mut off_diagonal = 0.0;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > off_diagonal {
                    off_diagonal = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if n < 2 || off_diagonal < 1e-12 {
            break;
        }

        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;

        for k in 0..n {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..n {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..n {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }

    let eigenvalues = (0..n).map(|i| a[i][i] as f32).collect();
    let eigenvectors = v
        .into_iter()
        .map(|row| row.into_iter().map(|value| value as f32).collect())
        .collect();

    (eigenvalues, eigenvectors)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

/// Small deterministic PRNG, so a fit is reproducible from its seed.
struct Pcg32 {
    state: u64,
}

impl Pcg32 {
    const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((self.state >> 18) ^ self.state) >> 27) as u32;
        let rot = (self.state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    /// Box-Muller transform, taking one of the two normal deviates produced.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
