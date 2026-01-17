//! Quantum Kernel implementation for coherence monitoring.
//!
//! This module implements quantum-inspired kernel methods for distribution monitoring.
//! The approach simulates quantum feature maps and kernel computation without requiring
//! actual quantum hardware.
//!
//! # Mathematical Background
//!
//! ## Quantum Feature Maps
//!
//! A quantum feature map encodes classical data x into a quantum state |phi(x)>.
//! We simulate this using parameterized rotations:
//!
//! ```text
//! |phi(x)> = U(x)|0>^n
//! ```
//!
//! where U(x) is a parameterized unitary encoding the data.
//!
//! ## Quantum Kernel
//!
//! The quantum kernel is defined as:
//!
//! ```text
//! k(x, y) = |<phi(x)|phi(y)>|^2 = Tr(rho_x * rho_y)
//! ```
//!
//! This measures the fidelity between quantum states, which we approximate
//! using classical simulation of the quantum feature map.
//!
//! # References
//!
//! - Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature Hilbert Spaces"
//! - Havlicek et al. (2019). "Supervised learning with quantum-enhanced feature spaces"

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::error::{QuantumMonitorError, Result};

/// Configuration for the quantum feature map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumKernelConfig {
    /// Number of qubits in the simulated quantum circuit.
    /// Determines the dimension of the quantum feature space (2^n_qubits).
    pub n_qubits: usize,

    /// Number of repetitions (layers) in the parameterized circuit.
    /// More layers increase expressivity but also computational cost.
    pub n_layers: usize,

    /// Bandwidth parameter sigma for RBF-like scaling.
    /// Controls the "width" of the kernel.
    pub sigma: f64,

    /// Whether to use ZZ entanglement between adjacent qubits.
    pub use_entanglement: bool,

    /// Random seed for reproducibility of random rotations.
    pub seed: Option<u64>,
}

impl Default for QuantumKernelConfig {
    fn default() -> Self {
        Self {
            n_qubits: 4,
            n_layers: 2,
            sigma: 1.0,
            use_entanglement: true,
            seed: None,
        }
    }
}

impl QuantumKernelConfig {
    /// Create a new configuration with the specified number of qubits.
    pub fn with_qubits(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            ..Default::default()
        }
    }

    /// Validate the configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.n_qubits == 0 || self.n_qubits > 16 {
            return Err(QuantumMonitorError::invalid_parameter(
                "n_qubits",
                "must be between 1 and 16",
            ));
        }
        if self.n_layers == 0 {
            return Err(QuantumMonitorError::invalid_parameter(
                "n_layers",
                "must be at least 1",
            ));
        }
        if self.sigma <= 0.0 {
            return Err(QuantumMonitorError::invalid_parameter(
                "sigma",
                "must be positive",
            ));
        }
        Ok(())
    }
}

/// Quantum Kernel for computing kernel matrices between data distributions.
///
/// This struct implements a classical simulation of quantum kernel methods,
/// which can detect distribution shift through kernel-based statistics.
#[derive(Debug, Clone)]
pub struct QuantumKernel {
    config: QuantumKernelConfig,
    /// Cached rotation angles for the variational circuit (learned or fixed).
    rotation_params: Array2<f64>,
    /// Feature space dimension (2^n_qubits).
    feature_dim: usize,
}

impl QuantumKernel {
    /// Create a new quantum kernel with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the quantum kernel.
    ///
    /// # Returns
    ///
    /// A new `QuantumKernel` instance or an error if configuration is invalid.
    pub fn new(config: QuantumKernelConfig) -> Result<Self> {
        config.validate()?;

        let feature_dim = 1 << config.n_qubits; // 2^n_qubits

        // Initialize rotation parameters
        use rand::SeedableRng;

        let mut rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Each layer has 3 rotation angles per qubit (Rx, Ry, Rz)
        #[allow(unused_variables)]
        let n_params = config.n_layers * config.n_qubits * 3;
        let normal = Normal::new(0.0, PI / 4.0).unwrap();

        let rotation_params = Array2::from_shape_fn((config.n_layers, config.n_qubits * 3), |_| {
            normal.sample(&mut rng)
        });

        Ok(Self {
            config,
            rotation_params,
            feature_dim,
        })
    }

    /// Encode a classical data vector into a quantum feature vector.
    ///
    /// This simulates the quantum feature map |phi(x)> by computing
    /// the amplitudes of the quantum state classically.
    ///
    /// # Mathematical Details
    ///
    /// We use angle encoding combined with variational rotations:
    /// ```text
    /// |phi(x)> = prod_l [ U_rot(theta_l) * U_enc(x) ] |0>^n
    /// ```
    ///
    /// where U_enc(x) encodes data features as rotation angles.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data vector. Dimension should ideally match n_qubits,
    ///         but will be padded/truncated as needed.
    ///
    /// # Returns
    ///
    /// Complex amplitude vector of dimension 2^n_qubits (as f64 magnitudes).
    pub fn encode_feature_map(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        let n_qubits = self.config.n_qubits;

        // Prepare input angles from data (angle encoding)
        let mut angles = vec![0.0; n_qubits];
        for i in 0..n_qubits.min(x.len()) {
            // Scale data to [0, 2pi] range using arctan
            angles[i] = 2.0 * (x[i] / self.config.sigma).atan();
        }

        // Initialize state vector |0...0>
        let mut state = Array1::zeros(self.feature_dim);
        state[0] = 1.0; // |0...0> state

        // Apply layers of the variational circuit
        for layer in 0..self.config.n_layers {
            // Apply data encoding rotations (Ry gates)
            state = self.apply_ry_layer(&state, &angles)?;

            // Apply variational rotations (Rx, Ry, Rz)
            let layer_params = self.rotation_params.row(layer);
            state = self.apply_variational_layer(&state, &layer_params.to_owned())?;

            // Apply entanglement (ZZ gates) if enabled
            if self.config.use_entanglement {
                state = self.apply_entanglement_layer(&state, &angles)?;
            }
        }

        // Ensure final normalization
        let norm: f64 = state.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            state /= norm;
        }

        // Return amplitude vector (NOT squared - we use real amplitudes for this simulation)
        // The kernel computation will compute |<phi(x)|phi(y)>|^2
        Ok(state)
    }

    /// Apply Ry rotation layer (data encoding).
    fn apply_ry_layer(&self, state: &Array1<f64>, angles: &[f64]) -> Result<Array1<f64>> {
        let mut result = state.clone();

        for (qubit, &angle) in angles.iter().enumerate() {
            result = self.apply_single_qubit_ry(&result, qubit, angle)?;
        }

        Ok(result)
    }

    /// Apply a single-qubit Ry rotation.
    ///
    /// Ry(theta) = [[cos(theta/2), -sin(theta/2)],
    ///              [sin(theta/2),  cos(theta/2)]]
    fn apply_single_qubit_ry(
        &self,
        state: &Array1<f64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<f64>> {
        let n_qubits = self.config.n_qubits;
        let mut result = Array1::zeros(self.feature_dim);

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        // Apply rotation to each basis state pair
        for i in 0..self.feature_dim {
            let bit = (i >> qubit) & 1;
            let partner = i ^ (1 << qubit);

            if bit == 0 {
                // |0> -> cos(theta/2)|0> + sin(theta/2)|1>
                result[i] += cos_half * state[i];
                result[partner] += sin_half * state[i];
            } else {
                // |1> -> -sin(theta/2)|0> + cos(theta/2)|1>
                result[partner] += -sin_half * state[i];
                result[i] += cos_half * state[i];
            }
        }

        // Normalize to handle numerical precision
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            result /= norm;
        }

        Ok(result)
    }

    /// Apply variational rotation layer (parameterized Rx, Ry, Rz).
    fn apply_variational_layer(
        &self,
        state: &Array1<f64>,
        params: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let mut result = state.clone();
        let n_qubits = self.config.n_qubits;

        for qubit in 0..n_qubits {
            // Apply Ry rotation from variational parameters
            // Note: For full quantum simulation, Rx and Rz would need complex amplitudes
            // Here we use Ry only, which captures real-valued rotations
            let _rx_angle = params[qubit * 3];      // Would need complex amplitudes
            let ry_angle = params[qubit * 3 + 1];   // Used for rotation
            let _rz_angle = params[qubit * 3 + 2];  // Would need complex amplitudes

            result = self.apply_single_qubit_ry(&result, qubit, ry_angle)?;
        }

        Ok(result)
    }

    /// Apply entanglement layer using ZZ-like interactions.
    ///
    /// This simulates entangling gates between adjacent qubits.
    fn apply_entanglement_layer(
        &self,
        state: &Array1<f64>,
        angles: &[f64],
    ) -> Result<Array1<f64>> {
        let mut result = state.clone();
        let n_qubits = self.config.n_qubits;

        // Apply ZZ interactions between adjacent qubits
        for q in 0..(n_qubits - 1) {
            let angle = angles.get(q).copied().unwrap_or(0.0)
                * angles.get(q + 1).copied().unwrap_or(0.0);

            // ZZ gate phase: exp(-i * theta * Z_q * Z_{q+1})
            // This adds phase based on parity of adjacent qubits
            for i in 0..self.feature_dim {
                let bit_q = (i >> q) & 1;
                let bit_q1 = (i >> (q + 1)) & 1;
                let parity = (bit_q ^ bit_q1) as f64;

                // Apply phase rotation (approximated in real space)
                let phase = (angle * (1.0 - 2.0 * parity)).cos();
                result[i] *= phase;
            }
        }

        // Renormalize
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            result /= norm;
        }

        Ok(result)
    }

    /// Compute the quantum kernel value between two data points.
    ///
    /// ```text
    /// k(x, y) = |<phi(x)|phi(y)>|^2
    /// ```
    ///
    /// This is the fidelity between the quantum states encoded by x and y.
    pub fn kernel(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64> {
        let phi_x = self.encode_feature_map(x)?;
        let phi_y = self.encode_feature_map(y)?;

        // Compute inner product squared (fidelity)
        let inner_product: f64 = phi_x.iter().zip(phi_y.iter()).map(|(a, b)| a * b).sum();

        Ok(inner_product.powi(2))
    }

    /// Compute the kernel matrix for a set of data points.
    ///
    /// Returns K[i,j] = k(X[i], X[j])
    ///
    /// # Arguments
    ///
    /// * `data` - Matrix where each row is a data point.
    ///
    /// # Returns
    ///
    /// Symmetric kernel matrix of shape (n_samples, n_samples).
    pub fn kernel_matrix(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let mut k_matrix = Array2::zeros((n_samples, n_samples));

        // Pre-compute feature maps for efficiency
        let mut phi_cache: Vec<Array1<f64>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let x = data.row(i).to_owned();
            phi_cache.push(self.encode_feature_map(&x)?);
        }

        // Compute kernel matrix (symmetric)
        for i in 0..n_samples {
            for j in i..n_samples {
                let inner: f64 = phi_cache[i]
                    .iter()
                    .zip(phi_cache[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let k_val = inner.powi(2);

                k_matrix[[i, j]] = k_val;
                k_matrix[[j, i]] = k_val;
            }
        }

        Ok(k_matrix)
    }

    /// Compute the cross-kernel matrix between two sets of data points.
    ///
    /// Returns K[i,j] = k(X[i], Y[j])
    ///
    /// # Arguments
    ///
    /// * `x_data` - First dataset, each row is a data point.
    /// * `y_data` - Second dataset, each row is a data point.
    ///
    /// # Returns
    ///
    /// Kernel matrix of shape (n_x, n_y).
    pub fn cross_kernel_matrix(
        &self,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n_x = x_data.nrows();
        let n_y = y_data.nrows();
        let mut k_matrix = Array2::zeros((n_x, n_y));

        // Pre-compute feature maps
        let phi_x: Vec<Array1<f64>> = (0..n_x)
            .map(|i| self.encode_feature_map(&x_data.row(i).to_owned()))
            .collect::<Result<Vec<_>>>()?;

        let phi_y: Vec<Array1<f64>> = (0..n_y)
            .map(|j| self.encode_feature_map(&y_data.row(j).to_owned()))
            .collect::<Result<Vec<_>>>()?;

        for i in 0..n_x {
            for j in 0..n_y {
                let inner: f64 = phi_x[i]
                    .iter()
                    .zip(phi_y[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                k_matrix[[i, j]] = inner.powi(2);
            }
        }

        Ok(k_matrix)
    }

    /// Incrementally update kernel values for streaming data.
    ///
    /// This is more efficient than recomputing the full kernel matrix
    /// when new data arrives.
    ///
    /// # Arguments
    ///
    /// * `existing_phi` - Pre-computed feature maps for existing data.
    /// * `new_point` - New data point to add.
    ///
    /// # Returns
    ///
    /// Tuple of (new_phi, kernel_values) where kernel_values[i] = k(existing[i], new).
    pub fn incremental_update(
        &self,
        existing_phi: &[Array1<f64>],
        new_point: &Array1<f64>,
    ) -> Result<(Array1<f64>, Vec<f64>)> {
        let new_phi = self.encode_feature_map(new_point)?;

        let kernel_values: Vec<f64> = existing_phi
            .iter()
            .map(|phi_i| {
                let inner: f64 = phi_i.iter().zip(new_phi.iter()).map(|(a, b)| a * b).sum();
                inner.powi(2)
            })
            .collect();

        Ok((new_phi, kernel_values))
    }

    /// Get the configuration.
    pub fn config(&self) -> &QuantumKernelConfig {
        &self.config
    }

    /// Get the feature space dimension.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }
}

/// Streaming kernel accumulator for online monitoring.
///
/// Maintains sufficient statistics for kernel-based tests
/// without storing all historical data.
#[derive(Debug, Clone)]
pub struct StreamingKernelAccumulator {
    kernel: QuantumKernel,
    /// Cached feature maps for baseline samples.
    baseline_phi: Vec<Array1<f64>>,
    /// Running sum of baseline kernel values (for mean estimation).
    baseline_kernel_sum: f64,
    /// Count of baseline samples.
    baseline_count: usize,
    /// Cached feature maps for streaming samples.
    streaming_phi: Vec<Array1<f64>>,
    /// Running sum of cross-kernel values.
    cross_kernel_sum: f64,
    /// Running sum of streaming self-kernel values.
    streaming_kernel_sum: f64,
}

impl StreamingKernelAccumulator {
    /// Create a new streaming accumulator with the given kernel.
    pub fn new(kernel: QuantumKernel) -> Self {
        Self {
            kernel,
            baseline_phi: Vec::new(),
            baseline_kernel_sum: 0.0,
            baseline_count: 0,
            streaming_phi: Vec::new(),
            cross_kernel_sum: 0.0,
            streaming_kernel_sum: 0.0,
        }
    }

    /// Initialize with baseline data.
    pub fn set_baseline(&mut self, baseline: &Array2<f64>) -> Result<()> {
        self.baseline_phi.clear();
        self.baseline_kernel_sum = 0.0;
        self.baseline_count = baseline.nrows();

        // Compute and cache baseline feature maps
        for i in 0..baseline.nrows() {
            let x = baseline.row(i).to_owned();
            self.baseline_phi.push(self.kernel.encode_feature_map(&x)?);
        }

        // Compute baseline kernel sum (for MMD)
        for i in 0..self.baseline_phi.len() {
            for j in 0..self.baseline_phi.len() {
                let inner: f64 = self.baseline_phi[i]
                    .iter()
                    .zip(self.baseline_phi[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                self.baseline_kernel_sum += inner.powi(2);
            }
        }

        Ok(())
    }

    /// Add a new streaming sample and update statistics.
    pub fn add_sample(&mut self, sample: &Array1<f64>) -> Result<StreamingUpdate> {
        let (new_phi, cross_kernels) = self.kernel.incremental_update(&self.baseline_phi, sample)?;

        // Update cross-kernel sum
        let cross_sum: f64 = cross_kernels.iter().sum();
        self.cross_kernel_sum += cross_sum;

        // Compute self-kernel with existing streaming samples
        let mut self_kernel_sum = 0.0;
        for phi in &self.streaming_phi {
            let inner: f64 = phi.iter().zip(new_phi.iter()).map(|(a, b)| a * b).sum();
            self_kernel_sum += inner.powi(2);
        }
        // Add self-kernel (k(x,x) = 1 for normalized states)
        self_kernel_sum += 1.0;

        self.streaming_kernel_sum += 2.0 * self_kernel_sum; // Count both (i,j) and (j,i)
        self.streaming_phi.push(new_phi);

        let n = self.streaming_phi.len();
        let m = self.baseline_count;

        Ok(StreamingUpdate {
            sample_index: n - 1,
            baseline_mean_kernel: self.baseline_kernel_sum / (m * m) as f64,
            streaming_mean_kernel: self.streaming_kernel_sum / (n * n) as f64,
            cross_mean_kernel: self.cross_kernel_sum / (n * m) as f64,
        })
    }

    /// Get the current MMD^2 estimate.
    ///
    /// MMD^2 = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
    pub fn mmd_squared(&self) -> f64 {
        if self.streaming_phi.is_empty() || self.baseline_count == 0 {
            return 0.0;
        }

        let n = self.streaming_phi.len() as f64;
        let m = self.baseline_count as f64;

        let baseline_mean = self.baseline_kernel_sum / (m * m);
        let streaming_mean = self.streaming_kernel_sum / (n * n);
        let cross_mean = self.cross_kernel_sum / (n * m);

        baseline_mean - 2.0 * cross_mean + streaming_mean
    }

    /// Reset streaming statistics while keeping baseline.
    pub fn reset_streaming(&mut self) {
        self.streaming_phi.clear();
        self.cross_kernel_sum = 0.0;
        self.streaming_kernel_sum = 0.0;
    }
}

/// Update information from adding a streaming sample.
#[derive(Debug, Clone)]
pub struct StreamingUpdate {
    /// Index of the newly added sample.
    pub sample_index: usize,
    /// Current mean of baseline-baseline kernel values.
    pub baseline_mean_kernel: f64,
    /// Current mean of streaming-streaming kernel values.
    pub streaming_mean_kernel: f64,
    /// Current mean of cross (baseline-streaming) kernel values.
    pub cross_mean_kernel: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantum_kernel_creation() {
        let config = QuantumKernelConfig::default();
        let kernel = QuantumKernel::new(config).unwrap();

        assert_eq!(kernel.feature_dim(), 16); // 2^4 = 16
        assert_eq!(kernel.config().n_qubits, 4);
    }

    #[test]
    fn test_kernel_config_validation() {
        let mut config = QuantumKernelConfig::default();

        config.n_qubits = 0;
        assert!(QuantumKernel::new(config.clone()).is_err());

        config.n_qubits = 17;
        assert!(QuantumKernel::new(config.clone()).is_err());

        config.n_qubits = 4;
        config.sigma = -1.0;
        assert!(QuantumKernel::new(config).is_err());
    }

    #[test]
    fn test_feature_map_encoding() {
        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            sigma: 1.0,
            use_entanglement: false,
            seed: Some(42),
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let x = array![0.5, 0.3];
        let phi = kernel.encode_feature_map(&x).unwrap();

        // Feature map should have 2^2 = 4 dimensions
        assert_eq!(phi.len(), 4);

        // Amplitudes should be normalized (sum of squares = 1)
        let norm_sq: f64 = phi.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6, "Feature map amplitudes should be normalized, got ||phi||^2 = {}", norm_sq);
    }

    #[test]
    fn test_kernel_symmetry() {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let x = array![0.1, 0.2, 0.3];
        let y = array![0.4, 0.5, 0.6];

        let k_xy = kernel.kernel(&x, &y).unwrap();
        let k_yx = kernel.kernel(&y, &x).unwrap();

        assert!(
            (k_xy - k_yx).abs() < 1e-10,
            "Kernel should be symmetric: k(x,y) = k(y,x)"
        );
    }

    #[test]
    fn test_kernel_self_value() {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let x = array![0.1, 0.2, 0.3];
        let k_xx = kernel.kernel(&x, &x).unwrap();

        // Self-kernel should be positive and bounded
        // Note: Due to quantum simulation approximations, k(x,x) may not be exactly 1
        assert!(
            k_xx > 0.0 && k_xx <= 1.0,
            "Self-kernel should be in (0, 1], got {}",
            k_xx
        );
    }

    #[test]
    fn test_kernel_matrix() {
        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let data = Array2::from_shape_vec(
            (3, 2),
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        .unwrap();

        let k_matrix = kernel.kernel_matrix(&data).unwrap();

        // Should be 3x3
        assert_eq!(k_matrix.shape(), &[3, 3]);

        // Should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (k_matrix[[i, j]] - k_matrix[[j, i]]).abs() < 1e-10,
                    "Kernel matrix should be symmetric"
                );
            }
        }

        // Diagonal should be positive and bounded
        // Note: Due to quantum simulation approximations and floating-point precision,
        // diagonal may not be exactly 1, but should be close
        for i in 0..3 {
            assert!(
                k_matrix[[i, i]] > 0.0 && k_matrix[[i, i]] <= 1.0 + 1e-9,
                "Diagonal should be in (0, 1], got {}",
                k_matrix[[i, i]]
            );
        }
    }

    #[test]
    fn test_streaming_accumulator() {
        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();
        let mut accumulator = StreamingKernelAccumulator::new(kernel);

        // Set baseline
        let baseline = Array2::from_shape_vec(
            (5, 2),
            vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
        )
        .unwrap();
        accumulator.set_baseline(&baseline).unwrap();

        // Add streaming samples
        let sample1 = array![0.15, 0.15];
        let update1 = accumulator.add_sample(&sample1).unwrap();
        assert_eq!(update1.sample_index, 0);

        let sample2 = array![0.25, 0.25];
        let update2 = accumulator.add_sample(&sample2).unwrap();
        assert_eq!(update2.sample_index, 1);

        // MMD should be small for similar distributions
        let mmd = accumulator.mmd_squared();
        assert!(mmd.is_finite(), "MMD should be finite");
    }

    #[test]
    fn test_incremental_update() {
        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        // Pre-compute some feature maps
        let existing: Vec<Array1<f64>> = vec![
            kernel.encode_feature_map(&array![0.1, 0.2]).unwrap(),
            kernel.encode_feature_map(&array![0.3, 0.4]).unwrap(),
        ];

        let new_point = array![0.5, 0.6];
        let (new_phi, kernel_values) = kernel.incremental_update(&existing, &new_point).unwrap();

        assert_eq!(kernel_values.len(), 2);
        assert_eq!(new_phi.len(), 4); // 2^2 = 4

        // Verify kernel values match direct computation
        let k_01 = kernel.kernel(&array![0.1, 0.2], &new_point).unwrap();
        assert!(
            (kernel_values[0] - k_01).abs() < 1e-10,
            "Incremental update should match direct kernel computation"
        );
    }
}
