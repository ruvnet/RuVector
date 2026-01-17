//! Mamba State-Space Decoder
//!
//! Implements a Mamba-style state-space model for sequential decoding of
//! syndrome representations into error corrections.
//!
//! ## State Space Model
//!
//! The Mamba decoder uses selective state spaces with data-dependent parameters:
//! - Input-dependent state transition (A matrix selection)
//! - Input-dependent input projection (B matrix selection)
//! - Gated output with residual connection

use crate::error::{NeuralDecoderError, Result};
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Configuration for the Mamba decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    /// Input dimension (from GNN output)
    pub input_dim: usize,
    /// State dimension (internal recurrent state)
    pub state_dim: usize,
    /// Output dimension (correction probabilities)
    pub output_dim: usize,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            state_dim: 64,
            output_dim: 25, // 5x5 surface code
        }
    }
}

/// The recurrent state of the Mamba decoder
#[derive(Debug, Clone)]
pub struct MambaState {
    /// Hidden state vector
    pub hidden: Vec<f32>,
    /// State dimension
    pub dim: usize,
    /// Number of steps processed
    pub steps: usize,
}

impl MambaState {
    /// Create a new zero-initialized state
    pub fn new(dim: usize) -> Self {
        Self {
            hidden: vec![0.0; dim],
            dim,
            steps: 0,
        }
    }

    /// Reset the state to zeros
    pub fn reset(&mut self) {
        self.hidden.fill(0.0);
        self.steps = 0;
    }

    /// Get the current hidden state
    pub fn get(&self) -> &[f32] {
        &self.hidden
    }

    /// Update the hidden state
    pub fn update(&mut self, new_state: Vec<f32>) {
        assert_eq!(new_state.len(), self.dim);
        self.hidden = new_state;
        self.steps += 1;
    }
}

/// Linear layer with bias
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Linear {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl Linear {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let normal = Normal::new(0.0, scale as f64).unwrap();

        let weights = Array2::from_shape_fn(
            (output_dim, input_dim),
            |_| normal.sample(&mut rng) as f32
        );
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let x = ArrayView1::from(input);
        let output = self.weights.dot(&x) + &self.bias;
        output.to_vec()
    }
}

/// Selective scan block (core Mamba operation)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SelectiveScan {
    /// Projects input to delta (determines discretization)
    delta_proj: Linear,
    /// Projects input to B (input matrix)
    b_proj: Linear,
    /// Projects input to C (output matrix)
    c_proj: Linear,
    /// Discretization scale
    delta_scale: f32,
    /// State dimension
    state_dim: usize,
}

impl SelectiveScan {
    fn new(input_dim: usize, state_dim: usize) -> Self {
        Self {
            delta_proj: Linear::new(input_dim, state_dim),
            b_proj: Linear::new(input_dim, state_dim),
            c_proj: Linear::new(input_dim, state_dim),
            delta_scale: 0.1,
            state_dim,
        }
    }

    /// Perform one step of selective scan
    fn step(&self, input: &[f32], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Compute data-dependent parameters
        let delta_raw = self.delta_proj.forward(input);
        let b = self.b_proj.forward(input);
        let c = self.c_proj.forward(input);

        // Softplus for delta (ensures positive)
        let delta: Vec<f32> = delta_raw.iter()
            .map(|&x| (1.0 + (x * self.delta_scale).exp()).ln())
            .collect();

        // Discretized state transition: x = exp(-delta) * x + delta * B * u
        // Simplified: x = (1 - delta) * x + delta * B * input_proj
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);

        let mut new_state = vec![0.0; self.state_dim];
        for i in 0..self.state_dim {
            let decay = (-delta[i]).exp();
            let input_contrib = delta[i] * b[i] * (input_norm / (self.state_dim as f32).sqrt());
            new_state[i] = decay * state[i] + input_contrib;
        }

        // Output: y = C * x
        let output: f32 = c.iter().zip(new_state.iter())
            .map(|(ci, xi)| ci * xi)
            .sum();

        // Expand output to match input dimension for residual
        let output_vec = vec![output / (self.state_dim as f32).sqrt(); input.len()];

        (new_state, output_vec)
    }
}

/// Mamba block combining selective scan with gating
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MambaBlock {
    /// Input projection
    in_proj: Linear,
    /// Selective scan module
    ssm: SelectiveScan,
    /// Gate projection
    gate_proj: Linear,
    /// Output projection
    out_proj: Linear,
    /// Layer norm
    norm: Array1<f32>,
    /// State dimension
    state_dim: usize,
}

impl MambaBlock {
    fn new(input_dim: usize, state_dim: usize) -> Self {
        Self {
            in_proj: Linear::new(input_dim, state_dim),
            ssm: SelectiveScan::new(state_dim, state_dim),
            gate_proj: Linear::new(input_dim, state_dim),
            out_proj: Linear::new(state_dim, input_dim),
            norm: Array1::ones(state_dim),
            state_dim,
        }
    }

    fn forward(&self, input: &[f32], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Project input
        let x = self.in_proj.forward(input);

        // Selective scan
        let (new_state, ssm_out) = self.ssm.step(&x, state);

        // Gating
        let gate_raw = self.gate_proj.forward(input);
        let gate: Vec<f32> = gate_raw.iter()
            .map(|&g| 1.0 / (1.0 + (-g).exp()))
            .collect();

        // Apply gate
        let gated: Vec<f32> = ssm_out.iter().zip(gate.iter().cycle())
            .map(|(s, g)| s * g)
            .collect();

        // Output projection
        let output_raw = self.out_proj.forward(&gated[..self.state_dim.min(gated.len())]);

        // Residual connection
        let output: Vec<f32> = input.iter().zip(output_raw.iter().cycle())
            .map(|(i, o)| i + o)
            .collect();

        (new_state, output)
    }
}

/// Mamba decoder for syndrome-to-correction mapping
#[derive(Debug, Clone)]
pub struct MambaDecoder {
    config: MambaConfig,
    block: MambaBlock,
    output_proj: Linear,
    state: MambaState,
}

impl MambaDecoder {
    /// Create a new Mamba decoder
    pub fn new(config: MambaConfig) -> Self {
        let block = MambaBlock::new(config.input_dim, config.state_dim);
        let output_proj = Linear::new(config.input_dim, config.output_dim);
        let state = MambaState::new(config.state_dim);

        Self {
            config,
            block,
            output_proj,
            state,
        }
    }

    /// Decode node embeddings to correction probabilities
    pub fn decode(&mut self, embeddings: &Array2<f32>) -> Result<Array1<f32>> {
        if embeddings.shape()[0] == 0 {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        let expected_dim = self.config.input_dim;
        let actual_dim = embeddings.shape()[1];

        if actual_dim != expected_dim {
            return Err(NeuralDecoderError::embed_dim(expected_dim, actual_dim));
        }

        // Process each node embedding sequentially
        let mut aggregated = vec![0.0; self.config.input_dim];

        for row in embeddings.rows() {
            let input: Vec<f32> = row.to_vec();

            // Mamba block forward
            let (new_state, output) = self.block.forward(&input, self.state.get());
            self.state.update(new_state);

            // Aggregate outputs
            for (agg, out) in aggregated.iter_mut().zip(output.iter()) {
                *agg += out;
            }
        }

        // Normalize by number of nodes
        let num_nodes = embeddings.shape()[0] as f32;
        for val in &mut aggregated {
            *val /= num_nodes;
        }

        // Project to output dimension
        let logits = self.output_proj.forward(&aggregated);

        // Sigmoid activation for probabilities
        let probs: Vec<f32> = logits.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Ok(Array1::from_vec(probs))
    }

    /// Decode with explicit state management
    pub fn decode_step(&mut self, embedding: &[f32]) -> Result<Vec<f32>> {
        if embedding.len() != self.config.input_dim {
            return Err(NeuralDecoderError::embed_dim(
                self.config.input_dim,
                embedding.len()
            ));
        }

        let (new_state, output) = self.block.forward(embedding, self.state.get());
        self.state.update(new_state);

        Ok(output)
    }

    /// Get the current state
    pub fn state(&self) -> &MambaState {
        &self.state
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Get the configuration
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_config_default() {
        let config = MambaConfig::default();
        assert_eq!(config.input_dim, 128);
        assert_eq!(config.state_dim, 64);
        assert_eq!(config.output_dim, 25);
    }

    #[test]
    fn test_mamba_state_creation() {
        let state = MambaState::new(64);
        assert_eq!(state.dim, 64);
        assert_eq!(state.steps, 0);
        assert_eq!(state.get().len(), 64);

        // All zeros initially
        for &val in state.get() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_mamba_state_update() {
        let mut state = MambaState::new(4);
        let new_values = vec![1.0, 2.0, 3.0, 4.0];
        state.update(new_values.clone());

        assert_eq!(state.steps, 1);
        assert_eq!(state.get(), &new_values[..]);
    }

    #[test]
    fn test_mamba_state_reset() {
        let mut state = MambaState::new(4);
        state.update(vec![1.0, 2.0, 3.0, 4.0]);
        state.update(vec![5.0, 6.0, 7.0, 8.0]);

        assert_eq!(state.steps, 2);

        state.reset();

        assert_eq!(state.steps, 0);
        for &val in state.get() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_mamba_decoder_creation() {
        let config = MambaConfig::default();
        let decoder = MambaDecoder::new(config);

        assert_eq!(decoder.config().input_dim, 128);
        assert_eq!(decoder.state().dim, 64);
    }

    #[test]
    fn test_mamba_decode() {
        let config = MambaConfig {
            input_dim: 32,
            state_dim: 16,
            output_dim: 9,
        };
        let mut decoder = MambaDecoder::new(config);

        // Create embeddings for 9 nodes
        let embeddings = Array2::from_shape_fn((9, 32), |(_i, _j)| 0.5);

        let output = decoder.decode(&embeddings).unwrap();
        assert_eq!(output.len(), 9);

        // Output should be probabilities (0 to 1)
        for &prob in output.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_mamba_decode_updates_state() {
        let config = MambaConfig {
            input_dim: 32,
            state_dim: 16,
            output_dim: 9,
        };
        let mut decoder = MambaDecoder::new(config);

        let embeddings = Array2::from_shape_fn((9, 32), |(_i, _j)| 0.5);

        assert_eq!(decoder.state().steps, 0);

        decoder.decode(&embeddings).unwrap();

        // State should be updated (9 steps for 9 nodes)
        assert_eq!(decoder.state().steps, 9);
    }

    #[test]
    fn test_mamba_decode_step() {
        let config = MambaConfig {
            input_dim: 32,
            state_dim: 16,
            output_dim: 9,
        };
        let mut decoder = MambaDecoder::new(config);

        let embedding = vec![0.5; 32];
        let output = decoder.decode_step(&embedding).unwrap();

        assert_eq!(output.len(), 32); // Same as input_dim for residual
        assert_eq!(decoder.state().steps, 1);
    }

    #[test]
    fn test_mamba_decode_wrong_dimension() {
        let config = MambaConfig {
            input_dim: 32,
            state_dim: 16,
            output_dim: 9,
        };
        let mut decoder = MambaDecoder::new(config);

        // Wrong input dimension
        let embeddings = Array2::from_shape_fn((9, 64), |(_i, _j)| 0.5);
        let result = decoder.decode(&embeddings);

        assert!(result.is_err());
    }

    #[test]
    fn test_mamba_decode_empty() {
        let config = MambaConfig {
            input_dim: 32,
            state_dim: 16,
            output_dim: 9,
        };
        let mut decoder = MambaDecoder::new(config);

        let embeddings: Array2<f32> = Array2::zeros((0, 32));
        let result = decoder.decode(&embeddings);

        assert!(result.is_err());
    }

    #[test]
    fn test_mamba_reset() {
        let config = MambaConfig {
            input_dim: 32,
            state_dim: 16,
            output_dim: 9,
        };
        let mut decoder = MambaDecoder::new(config);

        let embeddings = Array2::from_shape_fn((9, 32), |(_i, _j)| 0.5);
        decoder.decode(&embeddings).unwrap();

        assert_eq!(decoder.state().steps, 9);

        decoder.reset();

        assert_eq!(decoder.state().steps, 0);
    }

    #[test]
    fn test_mamba_sequential_decode() {
        let config = MambaConfig {
            input_dim: 16,
            state_dim: 8,
            output_dim: 4,
        };
        let mut decoder = MambaDecoder::new(config);

        // Process nodes one by one
        let embeddings: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32 * 0.1; 16])
            .collect();

        let mut outputs = Vec::new();
        for emb in &embeddings {
            let out = decoder.decode_step(emb).unwrap();
            outputs.push(out);
        }

        assert_eq!(outputs.len(), 5);
        assert_eq!(decoder.state().steps, 5);
    }

    #[test]
    fn test_mamba_state_evolution() {
        let config = MambaConfig {
            input_dim: 8,
            state_dim: 4,
            output_dim: 2,
        };
        let mut decoder = MambaDecoder::new(config);

        let emb1 = vec![1.0; 8];
        let emb2 = vec![0.0; 8];

        decoder.decode_step(&emb1).unwrap();
        let state1 = decoder.state().get().to_vec();

        decoder.decode_step(&emb2).unwrap();
        let state2 = decoder.state().get().to_vec();

        // States should differ
        let diff: f32 = state1.iter().zip(state2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_selective_scan_step() {
        let ssm = SelectiveScan::new(8, 4);
        let input = vec![0.5; 8];
        let state = vec![0.0; 4];

        let (new_state, output) = ssm.step(&input, &state);

        assert_eq!(new_state.len(), 4);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_mamba_block_forward() {
        let block = MambaBlock::new(8, 4);
        let input = vec![0.5; 8];
        let state = vec![0.0; 4];

        let (new_state, output) = block.forward(&input, &state);

        assert_eq!(new_state.len(), 4);
        assert_eq!(output.len(), 8);
    }
}
