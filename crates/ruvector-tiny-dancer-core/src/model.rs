//! FastGRNN model implementation
//!
//! Lightweight Gated Recurrent Neural Network optimized for inference

use crate::error::{Result, TinyDancerError};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// FastGRNN model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastGRNNConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Gate non-linearity parameter
    pub nu: f32,
    /// Hidden non-linearity parameter
    pub zeta: f32,
    /// Rank constraint for low-rank factorization
    pub rank: Option<usize>,
}

impl Default for FastGRNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 5, // 5 features from feature engineering
            hidden_dim: 8,
            output_dim: 1,
            nu: 1.0,
            zeta: 1.0,
            rank: Some(4),
        }
    }
}

/// FastGRNN model for neural routing
pub struct FastGRNN {
    config: FastGRNNConfig,
    /// Weight matrix for reset gate (U_r)
    w_reset: Array2<f32>,
    /// Weight matrix for update gate (U_u)
    w_update: Array2<f32>,
    /// Weight matrix for candidate (U_c)
    w_candidate: Array2<f32>,
    /// Recurrent weight matrix (W)
    w_recurrent: Array2<f32>,
    /// Output weight matrix
    w_output: Array2<f32>,
    /// Bias for reset gate
    b_reset: Array1<f32>,
    /// Bias for update gate
    b_update: Array1<f32>,
    /// Bias for candidate
    b_candidate: Array1<f32>,
    /// Bias for output
    b_output: Array1<f32>,
    /// Whether the model is quantized
    quantized: bool,
}

impl FastGRNN {
    /// Create a new FastGRNN model with the given configuration
    pub fn new(config: FastGRNNConfig) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let w_reset = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let w_update = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let w_candidate = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let w_recurrent = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let w_output = Array2::from_shape_fn((config.output_dim, config.hidden_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });

        let b_reset = Array1::zeros(config.hidden_dim);
        let b_update = Array1::zeros(config.hidden_dim);
        let b_candidate = Array1::zeros(config.hidden_dim);
        let b_output = Array1::zeros(config.output_dim);

        Ok(Self {
            config,
            w_reset,
            w_update,
            w_candidate,
            w_recurrent,
            w_output,
            b_reset,
            b_update,
            b_candidate,
            b_output,
            quantized: false,
        })
    }

    /// Load model from a file (safetensors format).
    ///
    /// Reconstructs all weight/bias tensors and the [`FastGRNNConfig`] stored in
    /// the safetensors `__metadata__` map. Inference reproduces a saved model
    /// bit-for-bit.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        use safetensors::SafeTensors;

        let buffer = std::fs::read(path)?;
        let (_, metadata) = SafeTensors::read_metadata(&buffer)
            .map_err(|e| TinyDancerError::SerializationError(format!("safetensors header: {e}")))?;
        let config_json = metadata
            .metadata()
            .as_ref()
            .and_then(|m| m.get("config"))
            .ok_or_else(|| {
                TinyDancerError::SerializationError("missing model config in safetensors".into())
            })?;
        let config: FastGRNNConfig = serde_json::from_str(config_json)
            .map_err(|e| TinyDancerError::SerializationError(format!("config decode: {e}")))?;

        let st = SafeTensors::deserialize(&buffer)
            .map_err(|e| TinyDancerError::SerializationError(format!("safetensors body: {e}")))?;

        let read2 = |name: &str, rows: usize, cols: usize| -> Result<Array2<f32>> {
            let t = st
                .tensor(name)
                .map_err(|e| TinyDancerError::SerializationError(format!("tensor {name}: {e}")))?;
            let data: &[f32] = bytemuck::cast_slice(t.data());
            Array2::from_shape_vec((rows, cols), data.to_vec())
                .map_err(|e| TinyDancerError::SerializationError(format!("shape {name}: {e}")))
        };
        let read1 = |name: &str, len: usize| -> Result<Array1<f32>> {
            let t = st
                .tensor(name)
                .map_err(|e| TinyDancerError::SerializationError(format!("tensor {name}: {e}")))?;
            let data: &[f32] = bytemuck::cast_slice(t.data());
            if data.len() != len {
                return Err(TinyDancerError::SerializationError(format!(
                    "tensor {name}: expected {len} elems, got {}",
                    data.len()
                )));
            }
            Ok(Array1::from_vec(data.to_vec()))
        };

        let (h, i, o) = (config.hidden_dim, config.input_dim, config.output_dim);
        Ok(Self {
            w_reset: read2("w_reset", h, i)?,
            w_update: read2("w_update", h, i)?,
            w_candidate: read2("w_candidate", h, i)?,
            w_recurrent: read2("w_recurrent", h, h)?,
            w_output: read2("w_output", o, h)?,
            b_reset: read1("b_reset", h)?,
            b_update: read1("b_update", h)?,
            b_candidate: read1("b_candidate", h)?,
            b_output: read1("b_output", o)?,
            quantized: false,
            config,
        })
    }

    /// Save model to a file (safetensors format).
    ///
    /// Every weight/bias tensor is written as little-endian f32; the
    /// [`FastGRNNConfig`] is stored in the safetensors `__metadata__` map so the
    /// model is fully self-describing.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use safetensors::{tensor::TensorView, Dtype};

        // (name, shape, contiguous f32 data) for every parameter tensor.
        let params: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            (
                "w_reset",
                self.w_reset.shape().to_vec(),
                self.w_reset.iter().copied().collect(),
            ),
            (
                "w_update",
                self.w_update.shape().to_vec(),
                self.w_update.iter().copied().collect(),
            ),
            (
                "w_candidate",
                self.w_candidate.shape().to_vec(),
                self.w_candidate.iter().copied().collect(),
            ),
            (
                "w_recurrent",
                self.w_recurrent.shape().to_vec(),
                self.w_recurrent.iter().copied().collect(),
            ),
            (
                "w_output",
                self.w_output.shape().to_vec(),
                self.w_output.iter().copied().collect(),
            ),
            ("b_reset", vec![self.b_reset.len()], self.b_reset.to_vec()),
            (
                "b_update",
                vec![self.b_update.len()],
                self.b_update.to_vec(),
            ),
            (
                "b_candidate",
                vec![self.b_candidate.len()],
                self.b_candidate.to_vec(),
            ),
            (
                "b_output",
                vec![self.b_output.len()],
                self.b_output.to_vec(),
            ),
        ];

        // Own the byte buffers so the TensorViews can borrow them.
        let byte_bufs: Vec<Vec<u8>> = params
            .iter()
            .map(|(_, _, data)| bytemuck::cast_slice::<f32, u8>(data).to_vec())
            .collect();

        let mut views: HashMap<String, TensorView> = HashMap::new();
        for ((name, shape, _), bytes) in params.iter().zip(byte_bufs.iter()) {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map_err(|e| TinyDancerError::SerializationError(format!("view {name}: {e}")))?;
            views.insert((*name).to_string(), view);
        }

        let mut meta = HashMap::new();
        meta.insert(
            "config".to_string(),
            serde_json::to_string(&self.config)
                .map_err(|e| TinyDancerError::SerializationError(format!("config encode: {e}")))?,
        );
        meta.insert(
            "format".to_string(),
            "ruvector-tiny-dancer-fastgrnn-v1".to_string(),
        );

        let bytes = safetensors::serialize(&views, &Some(meta))
            .map_err(|e| TinyDancerError::SerializationError(format!("serialize: {e}")))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Forward pass through the FastGRNN model
    ///
    /// # Arguments
    /// * `input` - Input vector (sequence of features)
    /// * `initial_hidden` - Optional initial hidden state
    ///
    /// # Returns
    /// Output score (typically between 0.0 and 1.0 after sigmoid)
    pub fn forward(&self, input: &[f32], initial_hidden: Option<&[f32]>) -> Result<f32> {
        if input.len() != self.config.input_dim {
            return Err(TinyDancerError::InvalidInput(format!(
                "Expected input dimension {}, got {}",
                self.config.input_dim,
                input.len()
            )));
        }

        let x = Array1::from_vec(input.to_vec());
        let mut h = if let Some(hidden) = initial_hidden {
            Array1::from_vec(hidden.to_vec())
        } else {
            Array1::zeros(self.config.hidden_dim)
        };

        // FastGRNN cell computation
        // r_t = sigmoid(W_r * x_t + b_r)
        let r = sigmoid(&(self.w_reset.dot(&x) + &self.b_reset), self.config.nu);

        // u_t = sigmoid(W_u * x_t + b_u)
        let u = sigmoid(&(self.w_update.dot(&x) + &self.b_update), self.config.nu);

        // c_t = tanh(W_c * x_t + W * (r_t ⊙ h_{t-1}) + b_c)
        let c = tanh(
            &(self.w_candidate.dot(&x) + self.w_recurrent.dot(&(&r * &h)) + &self.b_candidate),
            self.config.zeta,
        );

        // h_t = u_t ⊙ h_{t-1} + (1 - u_t) ⊙ c_t
        h = &u * &h + &((Array1::<f32>::ones(u.len()) - &u) * &c);

        // Output: y = W_out * h_t + b_out
        let output = self.w_output.dot(&h) + &self.b_output;

        // Apply sigmoid to get probability
        Ok(sigmoid_scalar(output[0]))
    }

    /// Batch inference for multiple inputs
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        inputs
            .iter()
            .map(|input| self.forward(input, None))
            .collect()
    }

    /// Forward pass that retains intermediate activations for backprop.
    ///
    /// Mathematically identical to [`FastGRNN::forward`]; returns the prediction
    /// plus a [`ForwardCache`] consumed by [`FastGRNN::backward`].
    pub fn forward_cached(
        &self,
        input: &[f32],
        initial_hidden: Option<&[f32]>,
    ) -> Result<(f32, ForwardCache)> {
        if input.len() != self.config.input_dim {
            return Err(TinyDancerError::InvalidInput(format!(
                "Expected input dimension {}, got {}",
                self.config.input_dim,
                input.len()
            )));
        }

        let x = Array1::from_vec(input.to_vec());
        let h0 = match initial_hidden {
            Some(h) => Array1::from_vec(h.to_vec()),
            None => Array1::zeros(self.config.hidden_dim),
        };

        let r = sigmoid(&(self.w_reset.dot(&x) + &self.b_reset), self.config.nu);
        let u = sigmoid(&(self.w_update.dot(&x) + &self.b_update), self.config.nu);
        let rh = &r * &h0;
        let c = tanh(
            &(self.w_candidate.dot(&x) + self.w_recurrent.dot(&rh) + &self.b_candidate),
            self.config.zeta,
        );
        let ones = Array1::<f32>::ones(u.len());
        let h = &u * &h0 + &((&ones - &u) * &c);
        let output = self.w_output.dot(&h) + &self.b_output;
        let logit = output[0];
        let pred = sigmoid_scalar(logit);

        Ok((
            pred,
            ForwardCache {
                x,
                h0,
                r,
                u,
                rh,
                c,
                h,
                logit,
            },
        ))
    }

    /// Single-step backprop. `d_logit = dL/d(output_logit)`; for BCE-with-sigmoid
    /// this is simply `pred - target`. Returns gradients for every parameter.
    pub fn backward(&self, cache: &ForwardCache, d_logit: f32) -> FastGRNNGradients {
        let hidden = self.config.hidden_dim;
        let out = self.config.output_dim;

        let mut d_output = Array1::<f32>::zeros(out);
        d_output[0] = d_logit;

        // Output layer: o = W_output · h + b_output
        let g_w_output = outer(&d_output, &cache.h);
        let g_b_output = d_output.clone();
        let d_h = self.w_output.t().dot(&d_output);

        // h = u ⊙ h0 + (1 - u) ⊙ c
        let d_u = &d_h * &(&cache.h0 - &cache.c);
        let d_c = &d_h * &(Array1::<f32>::ones(hidden) - &cache.u);

        // c = tanh(zeta · a_c)  ⇒  dc/da_c = zeta · (1 - c²)
        let d_a_c = &d_c * &cache.c.mapv(|v| self.config.zeta * (1.0 - v * v));
        let g_w_candidate = outer(&d_a_c, &cache.x);
        let g_b_candidate = d_a_c.clone();
        let g_w_recurrent = outer(&d_a_c, &cache.rh);
        let d_rh = self.w_recurrent.t().dot(&d_a_c);

        // rh = r ⊙ h0
        let d_r = &d_rh * &cache.h0;

        // r = σ(nu · a_r)  ⇒  dr/da_r = nu · r · (1 - r)
        let d_a_r = &d_r * &cache.r.mapv(|v| self.config.nu * v * (1.0 - v));
        let g_w_reset = outer(&d_a_r, &cache.x);
        let g_b_reset = d_a_r.clone();

        // u = σ(nu · a_u)  ⇒  du/da_u = nu · u · (1 - u)
        let d_a_u = &d_u * &cache.u.mapv(|v| self.config.nu * v * (1.0 - v));
        let g_w_update = outer(&d_a_u, &cache.x);
        let g_b_update = d_a_u.clone();

        FastGRNNGradients {
            w: [
                g_w_reset,
                g_w_update,
                g_w_candidate,
                g_w_recurrent,
                g_w_output,
            ],
            b: [g_b_reset, g_b_update, g_b_candidate, g_b_output],
        }
    }

    /// Mutable references to the five weight matrices, in optimizer order:
    /// `[reset, update, candidate, recurrent, output]`.
    pub fn weights_mut(&mut self) -> [&mut Array2<f32>; 5] {
        [
            &mut self.w_reset,
            &mut self.w_update,
            &mut self.w_candidate,
            &mut self.w_recurrent,
            &mut self.w_output,
        ]
    }

    /// Mutable references to the four bias vectors, in optimizer order:
    /// `[reset, update, candidate, output]`.
    pub fn biases_mut(&mut self) -> [&mut Array1<f32>; 4] {
        [
            &mut self.b_reset,
            &mut self.b_update,
            &mut self.b_candidate,
            &mut self.b_output,
        ]
    }

    /// Quantize the model to INT8
    pub fn quantize(&mut self) -> Result<()> {
        // TODO: Implement INT8 quantization
        self.quantized = true;
        Ok(())
    }

    /// Apply magnitude-based pruning
    pub fn prune(&mut self, sparsity: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TinyDancerError::InvalidInput(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        // TODO: Implement magnitude-based pruning
        Ok(())
    }

    /// Get model size in bytes
    pub fn size_bytes(&self) -> usize {
        let params = self.w_reset.len()
            + self.w_update.len()
            + self.w_candidate.len()
            + self.w_recurrent.len()
            + self.w_output.len()
            + self.b_reset.len()
            + self.b_update.len()
            + self.b_candidate.len()
            + self.b_output.len();

        params * if self.quantized { 1 } else { 4 } // 1 byte for INT8, 4 bytes for f32
    }

    /// Get configuration
    pub fn config(&self) -> &FastGRNNConfig {
        &self.config
    }
}

/// Cached activations from [`FastGRNN::forward_cached`], consumed by backprop.
pub struct ForwardCache {
    x: Array1<f32>,
    h0: Array1<f32>,
    r: Array1<f32>,
    u: Array1<f32>,
    rh: Array1<f32>,
    c: Array1<f32>,
    h: Array1<f32>,
    /// Pre-sigmoid output logit (unused directly but retained for clarity/debug).
    #[allow(dead_code)]
    logit: f32,
}

/// Gradients for every FastGRNN parameter from a backward pass.
///
/// `w` order: `[reset, update, candidate, recurrent, output]`;
/// `b` order: `[reset, update, candidate, output]` — matching
/// [`FastGRNN::weights_mut`] / [`FastGRNN::biases_mut`] and the Adam state.
pub struct FastGRNNGradients {
    pub w: [Array2<f32>; 5],
    pub b: [Array1<f32>; 4],
}

impl FastGRNNGradients {
    /// Zero gradients shaped for `config`.
    pub fn zeros(config: &FastGRNNConfig) -> Self {
        let (h, i, o) = (config.hidden_dim, config.input_dim, config.output_dim);
        Self {
            w: [
                Array2::zeros((h, i)),
                Array2::zeros((h, i)),
                Array2::zeros((h, i)),
                Array2::zeros((h, h)),
                Array2::zeros((o, h)),
            ],
            b: [
                Array1::zeros(h),
                Array1::zeros(h),
                Array1::zeros(h),
                Array1::zeros(o),
            ],
        }
    }

    /// Accumulate `other * scale` into `self`.
    pub fn add_scaled(&mut self, other: &Self, scale: f32) {
        for k in 0..5 {
            self.w[k] = &self.w[k] + &(&other.w[k] * scale);
        }
        for k in 0..4 {
            self.b[k] = &self.b[k] + &(&other.b[k] * scale);
        }
    }

    /// Global L2 norm over all gradient elements (for gradient clipping).
    pub fn global_norm(&self) -> f32 {
        let mut s = 0.0f32;
        for k in 0..5 {
            s += self.w[k].iter().map(|v| v * v).sum::<f32>();
        }
        for k in 0..4 {
            s += self.b[k].iter().map(|v| v * v).sum::<f32>();
        }
        s.sqrt()
    }

    /// Scale every gradient element by `f`.
    pub fn scale(&mut self, f: f32) {
        for k in 0..5 {
            self.w[k].mapv_inplace(|v| v * f);
        }
        for k in 0..4 {
            self.b[k].mapv_inplace(|v| v * f);
        }
    }
}

/// Outer product `a ⊗ b` → matrix of shape `(a.len(), b.len())`.
fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let mut m = Array2::<f32>::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            m[[i, j]] = a[i] * b[j];
        }
    }
    m
}

/// Sigmoid activation with scaling parameter
fn sigmoid(x: &Array1<f32>, scale: f32) -> Array1<f32> {
    x.mapv(|v| sigmoid_scalar(v * scale))
}

/// Scalar sigmoid with numerical stability
fn sigmoid_scalar(x: f32) -> f32 {
    if x > 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Tanh activation with scaling parameter
fn tanh(x: &Array1<f32>, scale: f32) -> Array1<f32> {
    x.mapv(|v| (v * scale).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastgrnn_creation() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config).unwrap();
        assert!(model.size_bytes() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let config = FastGRNNConfig {
            input_dim: 10,
            hidden_dim: 8,
            output_dim: 1,
            ..Default::default()
        };
        let model = FastGRNN::new(config).unwrap();
        let input = vec![0.5; 10];
        let output = model.forward(&input, None).unwrap();
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_batch_inference() {
        let config = FastGRNNConfig {
            input_dim: 10,
            ..Default::default()
        };
        let model = FastGRNN::new(config).unwrap();
        let inputs = vec![vec![0.5; 10], vec![0.3; 10], vec![0.8; 10]];
        let outputs = model.forward_batch(&inputs).unwrap();
        assert_eq!(outputs.len(), 3);
    }

    /// Analytic gradients (backward) must match central finite differences.
    #[test]
    fn test_gradient_check() {
        use rand::Rng;
        let config = FastGRNNConfig {
            input_dim: 4,
            hidden_dim: 3,
            output_dim: 1,
            nu: 1.0,
            zeta: 1.0,
            rank: None,
        };
        let mut model = FastGRNN::new(config).unwrap();
        let mut rng = rand::thread_rng();
        let x: Vec<f32> = (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let target = 1.0f32;
        let bce = |p: f32| {
            let eps = 1e-7;
            let p = p.clamp(eps, 1.0 - eps);
            -target * p.ln() - (1.0 - target) * (1.0 - p).ln()
        };

        let (pred, cache) = model.forward_cached(&x, None).unwrap();
        let grads = model.backward(&cache, pred - target);

        let eps = 1e-3;
        // Check the parameters that carry gradient when h0 = 0:
        //   weight index 1 = update, 2 = candidate, 4 = output.
        for wk in [1usize, 2, 4] {
            let (rows, cols) = (grads.w[wk].shape()[0], grads.w[wk].shape()[1]);
            for i in 0..rows.min(2) {
                for j in 0..cols.min(2) {
                    let analytic = grads.w[wk][[i, j]];
                    let cur = {
                        let ws = model.weights_mut();
                        ws[wk][[i, j]]
                    };
                    {
                        let mut ws = model.weights_mut();
                        ws[wk][[i, j]] = cur + eps;
                    }
                    let lp = bce(model.forward(&x, None).unwrap());
                    {
                        let mut ws = model.weights_mut();
                        ws[wk][[i, j]] = cur - eps;
                    }
                    let lm = bce(model.forward(&x, None).unwrap());
                    {
                        let mut ws = model.weights_mut();
                        ws[wk][[i, j]] = cur;
                    }
                    let numeric = (lp - lm) / (2.0 * eps);
                    assert!(
                        (numeric - analytic).abs() < 1e-2,
                        "w[{wk}][{i},{j}] numeric={numeric} analytic={analytic}"
                    );
                }
            }
        }

        // With h0 = 0 the reset gate (0) and recurrent matrix (3) get zero gradient.
        assert!(grads.w[0].iter().all(|v| v.abs() < 1e-9));
        assert!(grads.w[3].iter().all(|v| v.abs() < 1e-9));
    }

    /// save → load reproduces inference exactly.
    #[test]
    fn test_save_load_roundtrip() {
        let config = FastGRNNConfig {
            input_dim: 6,
            hidden_dim: 5,
            output_dim: 1,
            ..Default::default()
        };
        let model = FastGRNN::new(config).unwrap();
        let inputs = vec![vec![0.2; 6], vec![-0.4; 6], vec![0.7; 6]];
        let before: Vec<f32> = inputs
            .iter()
            .map(|x| model.forward(x, None).unwrap())
            .collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        model.save(&path).unwrap();

        let loaded = FastGRNN::load(&path).unwrap();
        let after: Vec<f32> = inputs
            .iter()
            .map(|x| loaded.forward(x, None).unwrap())
            .collect();

        for (b, a) in before.iter().zip(after.iter()) {
            assert!((b - a).abs() < 1e-6, "before={b} after={a}");
        }
        assert_eq!(loaded.config().input_dim, 6);
        assert_eq!(loaded.config().hidden_dim, 5);
    }
}
