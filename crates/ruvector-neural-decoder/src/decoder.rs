//! Mamba Decoder for Neural Quantum Error Decoding
//!
//! This module implements a selective state space model (Mamba) decoder:
//! - O(d^2) complexity for d x d syndrome grids
//! - Efficient hidden state management
//! - Selective gating for input-dependent state transitions
//! - Causal modeling of error propagation
//!
//! ## Architecture
//!
//! The Mamba decoder uses a state space model (SSM) with selective mechanisms:
//!
//! 1. **State Space Model**: Linear dynamical system x_t = A x_{t-1} + B u_t
//! 2. **Selective Mechanism**: Input-dependent gating of state transitions
//! 3. **Discretization**: Zero-order hold for continuous-to-discrete conversion
//! 4. **Parallel Scan**: Efficient O(n) parallel computation

use crate::error::{NeuralDecoderError, Result};
use crate::encoder::Linear;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Configuration for the Mamba Decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    /// Input dimension (from encoder)
    pub input_dim: usize,
    /// State dimension for SSM
    pub state_dim: usize,
    /// Expansion factor for inner dimension
    pub expand_factor: usize,
    /// Number of Mamba blocks
    pub num_layers: usize,
    /// Convolution kernel size
    pub conv_kernel_size: usize,
    /// Delta rank for low-rank approximation
    pub delta_rank: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Output dimension (for error predictions)
    pub output_dim: usize,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            state_dim: 16,
            expand_factor: 2,
            num_layers: 4,
            conv_kernel_size: 4,
            delta_rank: 8,
            dropout: 0.1,
            output_dim: 2, // (X error prob, Z error prob)
        }
    }
}

impl MambaConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.state_dim == 0 {
            return Err(NeuralDecoderError::ConfigError(
                "State dimension must be positive".to_string(),
            ));
        }
        if self.expand_factor == 0 {
            return Err(NeuralDecoderError::ConfigError(
                "Expand factor must be positive".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(NeuralDecoderError::ConfigError(format!(
                "Dropout must be in [0, 1], got {}",
                self.dropout
            )));
        }
        Ok(())
    }

    /// Get the inner (expanded) dimension
    pub fn inner_dim(&self) -> usize {
        self.input_dim * self.expand_factor
    }
}

/// 1D Depthwise Convolution layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthwiseConv1d {
    kernel: Array2<f32>, // (channels, kernel_size)
    kernel_size: usize,
    channels: usize,
}

impl DepthwiseConv1d {
    /// Create a new depthwise convolution
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / kernel_size as f32).sqrt();
        let normal = Normal::new(0.0, scale as f64).unwrap();

        let kernel = Array2::from_shape_fn((channels, kernel_size), |_| {
            normal.sample(&mut rng) as f32
        });

        Self {
            kernel,
            kernel_size,
            channels,
        }
    }

    /// Forward pass with causal padding
    ///
    /// # Arguments
    /// * `x` - Input (seq_len, channels)
    ///
    /// # Returns
    /// Output (seq_len, channels)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.shape()[0];
        let channels = x.shape()[1];
        let mut output = Array2::zeros((seq_len, channels));

        // Causal convolution: only look at past and current
        for t in 0..seq_len {
            for c in 0..channels.min(self.channels) {
                let mut sum = 0.0f32;
                for k in 0..self.kernel_size {
                    let idx = t as i64 - k as i64;
                    if idx >= 0 {
                        sum += x[[idx as usize, c]] * self.kernel[[c, k]];
                    }
                }
                output[[t, c]] = sum;
            }
        }

        output
    }
}

/// Selective State Space Model (S6) core
///
/// Implements the selective scan mechanism:
/// x_t = A_t * x_{t-1} + B_t * u_t
/// y_t = C_t * x_t + D * u_t
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveSSM {
    state_dim: usize,
    input_dim: usize,
    /// A parameter (state_dim,) - diagonal state matrix
    a_log: Array1<f32>,
    /// D parameter (input_dim,) - skip connection
    d: Array1<f32>,
    /// Delta projection for time step
    delta_proj: Linear,
    /// B projection from input
    b_proj: Linear,
    /// C projection from input
    c_proj: Linear,
    /// Delta rank
    delta_rank: usize,
}

impl SelectiveSSM {
    /// Create a new selective SSM
    pub fn new(input_dim: usize, state_dim: usize, delta_rank: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize A as log-spaced values for stability
        let a_log = Array1::from_shape_fn(state_dim, |i| {
            -(1.0 + i as f32).ln()
        });

        // D is initialized to 1 (identity skip connection)
        let d = Array1::ones(input_dim);

        Self {
            state_dim,
            input_dim,
            a_log,
            d,
            delta_proj: Linear::new(input_dim, delta_rank),
            b_proj: Linear::new(input_dim, state_dim),
            c_proj: Linear::new(input_dim, state_dim),
            delta_rank,
        }
    }

    /// Compute discretized state matrices
    ///
    /// # Arguments
    /// * `delta` - Time step (seq_len,)
    ///
    /// # Returns
    /// (A_bar, B_bar) where A_bar = exp(delta * A)
    fn discretize(&self, delta: &Array1<f32>, b: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let seq_len = delta.len();

        // A is diagonal, so exp(delta * A) is also diagonal
        let a = self.a_log.mapv(|x| x.exp());

        let mut a_bar = Array2::zeros((seq_len, self.state_dim));
        let mut b_bar = Array2::zeros((seq_len, self.state_dim));

        for t in 0..seq_len {
            let dt = delta[t].max(1e-4); // Clamp for stability
            for n in 0..self.state_dim {
                // Zero-order hold discretization
                let a_n = a[n];
                a_bar[[t, n]] = (dt * a_n.ln()).exp();
                b_bar[[t, n]] = if a_n.abs() > 1e-6 {
                    b[[t, n]] * (a_bar[[t, n]] - 1.0) / a_n.ln()
                } else {
                    b[[t, n]] * dt
                };
            }
        }

        (a_bar, b_bar)
    }

    /// Forward pass with selective scan
    ///
    /// # Arguments
    /// * `x` - Input (seq_len, input_dim)
    ///
    /// # Returns
    /// Output (seq_len, input_dim)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.shape()[0];
        let input_dim = x.shape()[1];

        // Compute delta (time step) from input
        let delta_raw = self.delta_proj.forward_batch(x);
        let delta: Array1<f32> = delta_raw.mean_axis(Axis(1)).unwrap().mapv(|v| softplus(v));

        // Compute B, C from input (selective mechanism)
        let b = self.b_proj.forward_batch(x);
        let c = self.c_proj.forward_batch(x);

        // Discretize
        let (a_bar, b_bar) = self.discretize(&delta, &b);

        // Selective scan
        let mut output = Array2::zeros((seq_len, input_dim));
        let mut h = Array1::zeros(self.state_dim);

        for t in 0..seq_len {
            // State update: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            let mut new_h = Array1::zeros(self.state_dim);
            for n in 0..self.state_dim {
                new_h[n] = a_bar[[t, n]] * h[n] + b_bar[[t, n]] * x[[t, n.min(input_dim - 1)]];
            }
            h = new_h;

            // Output: y_t = C_t * h_t + D * x_t
            for d in 0..input_dim {
                let mut y = 0.0;
                for n in 0..self.state_dim {
                    y += c[[t, n]] * h[n];
                }
                output[[t, d]] = y + self.d[d] * x[[t, d]];
            }
        }

        output
    }

    /// Get the hidden state dimension
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }
}

/// Mamba block combining convolution and selective SSM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaBlock {
    /// Input projection (expands dimension)
    in_proj: Linear,
    /// Depthwise convolution
    conv: DepthwiseConv1d,
    /// Selective SSM
    ssm: SelectiveSSM,
    /// Output projection
    out_proj: Linear,
    /// Layer normalization
    norm: Array1<f32>,
    norm_bias: Array1<f32>,
    /// Inner dimension
    inner_dim: usize,
    /// Input dimension
    input_dim: usize,
}

impl MambaBlock {
    /// Create a new Mamba block
    pub fn new(config: &MambaConfig) -> Self {
        let inner_dim = config.inner_dim();

        Self {
            in_proj: Linear::new(config.input_dim, inner_dim * 2),
            conv: DepthwiseConv1d::new(inner_dim, config.conv_kernel_size),
            ssm: SelectiveSSM::new(inner_dim, config.state_dim, config.delta_rank),
            out_proj: Linear::new(inner_dim, config.input_dim),
            norm: Array1::ones(config.input_dim),
            norm_bias: Array1::zeros(config.input_dim),
            inner_dim,
            input_dim: config.input_dim,
        }
    }

    /// Layer normalization
    fn layer_norm(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut output = Array2::zeros(x.raw_dim());
        let eps = 1e-5f32;

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mean = row.mean().unwrap_or(0.0);
            let variance: f32 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let std = (variance + eps).sqrt();

            for (j, &val) in row.iter().enumerate() {
                output[[i, j]] = (val - mean) / std * self.norm[j] + self.norm_bias[j];
            }
        }

        output
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input (seq_len, input_dim)
    ///
    /// # Returns
    /// Output (seq_len, input_dim)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.shape()[0];

        // Layer norm
        let x_norm = self.layer_norm(x);

        // Project and split into two branches
        let projected = self.in_proj.forward_batch(&x_norm);

        let mut x_branch = Array2::zeros((seq_len, self.inner_dim));
        let mut z_branch = Array2::zeros((seq_len, self.inner_dim));

        for t in 0..seq_len {
            for i in 0..self.inner_dim {
                x_branch[[t, i]] = projected[[t, i]];
                z_branch[[t, i]] = projected[[t, self.inner_dim + i]];
            }
        }

        // Convolution + SiLU activation on x branch
        let x_conv = self.conv.forward(&x_branch);
        let x_act = x_conv.mapv(|v| v * sigmoid(v)); // SiLU

        // SSM
        let x_ssm = self.ssm.forward(&x_act);

        // Gate with z branch (SiLU activation)
        let z_act = z_branch.mapv(|v| v * sigmoid(v));
        let gated = &x_ssm * &z_act;

        // Output projection
        let output = self.out_proj.forward_batch(&gated);

        // Residual connection
        x + &output
    }
}

/// Mamba Decoder for quantum error correction
///
/// Uses selective state space models to decode syndrome sequences
/// and predict error locations with O(d^2) complexity for d x d grids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaDecoder {
    config: MambaConfig,
    /// Mamba blocks
    blocks: Vec<MambaBlock>,
    /// Final projection for error prediction
    head: Linear,
    /// Hidden state for streaming
    hidden_state: Option<Array1<f32>>,
}

impl MambaDecoder {
    /// Create a new Mamba decoder
    pub fn new(config: MambaConfig) -> Result<Self> {
        config.validate()?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(MambaBlock::new(&config));
        }

        let head = Linear::new(config.input_dim, config.output_dim);

        Ok(Self {
            config,
            blocks,
            head,
            hidden_state: None,
        })
    }

    /// Decode a sequence of node embeddings
    ///
    /// # Arguments
    /// * `embeddings` - Node embeddings from encoder (seq_len, input_dim)
    ///
    /// # Returns
    /// Error predictions (seq_len, output_dim)
    pub fn decode(&mut self, embeddings: &Array2<f32>) -> Result<Array2<f32>> {
        let seq_len = embeddings.shape()[0];
        let input_dim = embeddings.shape()[1];

        if input_dim != self.config.input_dim {
            return Err(NeuralDecoderError::embed_dim(
                self.config.input_dim,
                input_dim,
            ));
        }

        if seq_len == 0 {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        // Forward through Mamba blocks
        let mut x = embeddings.clone();
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Project to output
        let logits = self.head.forward_batch(&x);

        // Apply sigmoid for probability output
        let probs = logits.mapv(|v| sigmoid(v));

        Ok(probs)
    }

    /// Decode a 2D syndrome grid (d x d)
    ///
    /// Flattens the grid to a sequence and applies the decoder.
    ///
    /// # Arguments
    /// * `grid_embeddings` - Grid embeddings (d, d, input_dim)
    /// * `scan_order` - Order to scan the grid: "row", "column", "snake", "hilbert"
    ///
    /// # Returns
    /// Error predictions reshaped to grid (d, d, output_dim)
    pub fn decode_grid(
        &mut self,
        grid_embeddings: &[Array2<f32>],
        scan_order: &str,
    ) -> Result<Vec<Array2<f32>>> {
        if grid_embeddings.is_empty() {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        let d = grid_embeddings.len();
        let input_dim = grid_embeddings[0].shape()[1];

        // Flatten grid to sequence based on scan order
        let sequence = self.flatten_grid(grid_embeddings, scan_order)?;

        // Decode
        let predictions = self.decode(&sequence)?;

        // Reshape back to grid
        let output_dim = predictions.shape()[1];
        self.unflatten_to_grid(&predictions, d, output_dim, scan_order)
    }

    /// Flatten a 2D grid to a 1D sequence
    fn flatten_grid(
        &self,
        grid: &[Array2<f32>],
        scan_order: &str,
    ) -> Result<Array2<f32>> {
        let d = grid.len();
        let row_len = grid[0].shape()[0];
        let input_dim = grid[0].shape()[1];

        let total_len = d * row_len;
        let mut sequence = Array2::zeros((total_len, input_dim));

        let indices = self.get_scan_indices(d, row_len, scan_order)?;

        for (seq_idx, (i, j)) in indices.iter().enumerate() {
            for k in 0..input_dim {
                sequence[[seq_idx, k]] = grid[*i][[*j, k]];
            }
        }

        Ok(sequence)
    }

    /// Unflatten a 1D sequence back to a 2D grid
    fn unflatten_to_grid(
        &self,
        sequence: &Array2<f32>,
        d: usize,
        output_dim: usize,
        scan_order: &str,
    ) -> Result<Vec<Array2<f32>>> {
        let row_len = sequence.shape()[0] / d;
        let indices = self.get_scan_indices(d, row_len, scan_order)?;

        let mut grid = vec![Array2::zeros((row_len, output_dim)); d];

        for (seq_idx, (i, j)) in indices.iter().enumerate() {
            for k in 0..output_dim {
                grid[*i][[*j, k]] = sequence[[seq_idx, k]];
            }
        }

        Ok(grid)
    }

    /// Get scan indices for different scan orders
    fn get_scan_indices(
        &self,
        rows: usize,
        cols: usize,
        scan_order: &str,
    ) -> Result<Vec<(usize, usize)>> {
        match scan_order {
            "row" => {
                // Row-major order
                Ok((0..rows)
                    .flat_map(|i| (0..cols).map(move |j| (i, j)))
                    .collect())
            }
            "column" => {
                // Column-major order
                Ok((0..cols)
                    .flat_map(|j| (0..rows).map(move |i| (i, j)))
                    .collect())
            }
            "snake" => {
                // Snake/boustrophedon order
                let mut indices = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    if i % 2 == 0 {
                        for j in 0..cols {
                            indices.push((i, j));
                        }
                    } else {
                        for j in (0..cols).rev() {
                            indices.push((i, j));
                        }
                    }
                }
                Ok(indices)
            }
            "hilbert" => {
                // Simplified Hilbert curve approximation for power-of-2 sizes
                let n = rows.max(cols);
                let order = (n as f32).log2().ceil() as usize;
                let size = 1 << order;

                let mut indices = Vec::new();
                self.hilbert_d2xy(order, size * size, &mut indices);

                // Filter to valid grid coordinates
                Ok(indices
                    .into_iter()
                    .filter(|(i, j)| *i < rows && *j < cols)
                    .collect())
            }
            _ => Err(NeuralDecoderError::ConfigError(format!(
                "Unknown scan order: {}",
                scan_order
            ))),
        }
    }

    /// Generate Hilbert curve coordinates
    fn hilbert_d2xy(&self, order: usize, n: usize, indices: &mut Vec<(usize, usize)>) {
        for d in 0..n {
            let (mut x, mut y) = (0, 0);
            let mut s = 1;
            let mut t = d;

            while s < (1 << order) {
                let rx = 1 & (t / 2);
                let ry = 1 & (t ^ rx);

                // Rotate
                if ry == 0 {
                    if rx == 1 {
                        x = s - 1 - x;
                        y = s - 1 - y;
                    }
                    std::mem::swap(&mut x, &mut y);
                }

                x += s * rx;
                y += s * ry;
                t /= 4;
                s *= 2;
            }

            indices.push((y, x));
        }
    }

    /// Reset the hidden state (for streaming mode)
    pub fn reset_state(&mut self) {
        self.hidden_state = None;
    }

    /// Get configuration
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }
}

/// Sigmoid activation function with numerical stability
fn sigmoid(x: f32) -> f32 {
    if x > 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Softplus activation function: log(1 + exp(x))
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depthwise_conv() {
        let conv = DepthwiseConv1d::new(8, 4);
        let x = Array2::from_shape_fn((10, 8), |(i, j)| {
            ((i + j) as f32) / 10.0
        });

        let output = conv.forward(&x);
        assert_eq!(output.shape(), &[10, 8]);
    }

    #[test]
    fn test_selective_ssm() {
        let ssm = SelectiveSSM::new(16, 8, 4);
        let x = Array2::from_shape_fn((20, 16), |(i, j)| {
            (i as f32 * 0.1).sin() + (j as f32 * 0.2).cos()
        });

        let output = ssm.forward(&x);
        assert_eq!(output.shape(), &[20, 16]);
    }

    #[test]
    fn test_mamba_block() {
        let config = MambaConfig {
            input_dim: 16,
            state_dim: 8,
            expand_factor: 2,
            num_layers: 1,
            conv_kernel_size: 4,
            delta_rank: 4,
            dropout: 0.1,
            output_dim: 2,
        };

        let block = MambaBlock::new(&config);
        let x = Array2::from_shape_fn((10, 16), |(i, j)| {
            ((i + j) as f32) / 100.0
        });

        let output = block.forward(&x);
        assert_eq!(output.shape(), &[10, 16]);
    }

    #[test]
    fn test_mamba_decoder() {
        let config = MambaConfig {
            input_dim: 16,
            state_dim: 8,
            expand_factor: 2,
            num_layers: 2,
            conv_kernel_size: 4,
            delta_rank: 4,
            dropout: 0.1,
            output_dim: 2,
        };

        let mut decoder = MambaDecoder::new(config).unwrap();
        let embeddings = Array2::from_shape_fn((25, 16), |(i, j)| {
            ((i * j) as f32) / 100.0
        });

        let predictions = decoder.decode(&embeddings).unwrap();
        assert_eq!(predictions.shape(), &[25, 2]);

        // Check predictions are in [0, 1] (sigmoid output)
        for &p in predictions.iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_grid_decoding() {
        let config = MambaConfig {
            input_dim: 8,
            state_dim: 4,
            expand_factor: 2,
            num_layers: 1,
            conv_kernel_size: 2,
            delta_rank: 2,
            dropout: 0.0,
            output_dim: 2,
        };

        let mut decoder = MambaDecoder::new(config).unwrap();

        // Create 5x5 grid
        let d = 5;
        let grid: Vec<Array2<f32>> = (0..d)
            .map(|i| Array2::from_shape_fn((d, 8), |(j, k)| {
                ((i + j + k) as f32) / 100.0
            }))
            .collect();

        for scan_order in &["row", "column", "snake"] {
            let predictions = decoder.decode_grid(&grid, scan_order).unwrap();
            assert_eq!(predictions.len(), d);
            assert_eq!(predictions[0].shape(), &[d, 2]);
        }
    }

    #[test]
    fn test_scan_orders() {
        let config = MambaConfig::default();
        let decoder = MambaDecoder::new(config).unwrap();

        // Test row order
        let row_indices = decoder.get_scan_indices(3, 3, "row").unwrap();
        assert_eq!(row_indices.len(), 9);
        assert_eq!(row_indices[0], (0, 0));
        assert_eq!(row_indices[3], (1, 0));

        // Test snake order
        let snake_indices = decoder.get_scan_indices(3, 3, "snake").unwrap();
        assert_eq!(snake_indices.len(), 9);
        assert_eq!(snake_indices[0], (0, 0));
        assert_eq!(snake_indices[3], (1, 2)); // Reversed direction
    }

    #[test]
    fn test_config_validation() {
        let mut config = MambaConfig::default();
        assert!(config.validate().is_ok());

        config.state_dim = 0;
        assert!(config.validate().is_err());

        config.state_dim = 16;
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_empty_input_error() {
        let config = MambaConfig::default();
        let mut decoder = MambaDecoder::new(config).unwrap();

        let empty = Array2::zeros((0, 64));
        let result = decoder.decode(&empty);
        assert!(matches!(result, Err(NeuralDecoderError::EmptyGraph)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = MambaConfig {
            input_dim: 64,
            ..Default::default()
        };
        let mut decoder = MambaDecoder::new(config).unwrap();

        let wrong_dim = Array2::zeros((10, 32)); // Wrong dimension
        let result = decoder.decode(&wrong_dim);
        assert!(matches!(result, Err(NeuralDecoderError::InvalidEmbeddingDimension { .. })));
    }

    #[test]
    fn test_activation_functions() {
        // Test sigmoid
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(-100.0) < 1e-6);
        assert!(sigmoid(100.0) > 1.0 - 1e-6);

        // Test softplus
        assert!(softplus(0.0) > 0.0);
        assert!((softplus(0.0) - 0.693).abs() < 0.01);
        assert!((softplus(-100.0)).abs() < 1e-6);
        assert!((softplus(100.0) - 100.0).abs() < 1e-6);
    }
}
