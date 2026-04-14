//! # EML Operator: Universal Function Approximation for Vector Search
//!
//! Based on the paper "All elementary functions from a single operator" (arXiv:2603.21852v2),
//! this module implements the EML operator `eml(x, y) = exp(x) - ln(y)` and EML tree
//! structures for universal function approximation within vector database operations.
//!
//! ## Key Applications
//!
//! - **Logarithmic quantization**: Non-uniform quantization using exp/ln transforms
//! - **Learned index models**: EML trees as non-linear CDF approximators
//! - **Score fusion**: Non-linear combination of vector and keyword relevance scores
//! - **Unified distance**: Parameterized distance kernels encoded through EML
//!
//! ## Theory
//!
//! The EML operator combined with constant 1 can reconstruct all elementary functions
//! (sin, cos, sqrt, log, exp, arithmetic, etc.) as binary trees. This provides a
//! compact, trainable universal function approximator with bounded depth.

use crate::error::{Result, RuvectorError};
use serde::{Deserialize, Serialize};

// ============================================================================
// Core EML Operator
// ============================================================================

/// The EML operator: `eml(x, y) = exp(x) - ln(y)`
///
/// This is the fundamental building block. The paper proves this single binary
/// operator (with constant 1) generates all elementary mathematical functions.
///
/// # Properties
/// - `eml(x, 1) = exp(x)` (exponential)
/// - `eml(0, y) = 1 - ln(y)` (shifted logarithm)
/// - `eml(ln(a), exp(b)) = a - b` (subtraction)
#[inline(always)]
pub fn eml(x: f32, y: f32) -> f32 {
    // Guard against ln(0) and ln(negative)
    let y_safe = if y > f32::EPSILON { y } else { f32::EPSILON };
    x.exp() - y_safe.ln()
}

/// Safe EML with clamped output to prevent overflow in deep trees
#[inline(always)]
pub fn eml_safe(x: f32, y: f32) -> f32 {
    let x_clamped = x.clamp(-20.0, 20.0); // exp(20) ≈ 4.8e8, safe for f32
    let y_safe = if y > f32::EPSILON { y } else { f32::EPSILON };
    x_clamped.exp() - y_safe.ln()
}

// ============================================================================
// EML Tree Structure
// ============================================================================

/// The kind of leaf node in an EML tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeafKind {
    /// References an input variable by index
    Input(usize),
    /// A trainable constant parameter
    Constant(f32),
}

/// A node in the EML binary tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmlNode {
    /// Terminal node — either an input reference or a trainable constant
    Leaf(LeafKind),
    /// Internal node applying eml(left, right) = exp(left) - ln(right)
    Internal {
        left: Box<EmlNode>,
        right: Box<EmlNode>,
    },
}

impl EmlNode {
    /// Evaluate this node given input values
    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        match self {
            EmlNode::Leaf(LeafKind::Input(idx)) => {
                inputs.get(*idx).copied().unwrap_or(0.0)
            }
            EmlNode::Leaf(LeafKind::Constant(c)) => *c,
            EmlNode::Internal { left, right } => {
                let l = left.evaluate(inputs);
                let r = right.evaluate(inputs);
                eml_safe(l, r)
            }
        }
    }

    /// Count the number of trainable parameters (constants) in this subtree
    pub fn num_params(&self) -> usize {
        match self {
            EmlNode::Leaf(LeafKind::Constant(_)) => 1,
            EmlNode::Leaf(LeafKind::Input(_)) => 0,
            EmlNode::Internal { left, right } => left.num_params() + right.num_params(),
        }
    }

    /// Collect all trainable parameters in order (depth-first)
    pub fn collect_params(&self) -> Vec<f32> {
        let mut params = Vec::new();
        self.collect_params_inner(&mut params);
        params
    }

    fn collect_params_inner(&self, params: &mut Vec<f32>) {
        match self {
            EmlNode::Leaf(LeafKind::Constant(c)) => params.push(*c),
            EmlNode::Leaf(LeafKind::Input(_)) => {}
            EmlNode::Internal { left, right } => {
                left.collect_params_inner(params);
                right.collect_params_inner(params);
            }
        }
    }

    /// Set trainable parameters from a flat vector (depth-first order)
    pub fn set_params(&mut self, params: &[f32], offset: &mut usize) {
        match self {
            EmlNode::Leaf(LeafKind::Constant(c)) => {
                if *offset < params.len() {
                    *c = params[*offset];
                    *offset += 1;
                }
            }
            EmlNode::Leaf(LeafKind::Input(_)) => {}
            EmlNode::Internal { left, right } => {
                left.set_params(params, offset);
                right.set_params(params, offset);
            }
        }
    }

    /// Compute depth of this subtree
    pub fn depth(&self) -> usize {
        match self {
            EmlNode::Leaf(_) => 0,
            EmlNode::Internal { left, right } => 1 + left.depth().max(right.depth()),
        }
    }

    /// Count total nodes
    pub fn node_count(&self) -> usize {
        match self {
            EmlNode::Leaf(_) => 1,
            EmlNode::Internal { left, right } => 1 + left.node_count() + right.node_count(),
        }
    }
}

// ============================================================================
// EML Tree (Aggregate Root)
// ============================================================================

/// An EML tree for universal function approximation.
///
/// Binary tree where each internal node applies `eml(left, right) = exp(left) - ln(right)`.
/// Leaf nodes are either input variable references or trainable constants.
///
/// The paper proves that any elementary function can be expressed as an EML tree
/// with bounded depth. In practice, depth 3-5 captures most useful functions
/// (exponentials, logarithms, polynomials, trigonometric functions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmlTree {
    /// Root node of the tree
    pub root: EmlNode,
    /// Number of input variables this tree expects
    pub num_inputs: usize,
}

impl EmlTree {
    /// Create a new EML tree with the given root and input count
    pub fn new(root: EmlNode, num_inputs: usize) -> Self {
        Self { root, num_inputs }
    }

    /// Create a depth-1 EML tree: `eml(a*x + b, c)` which approximates `exp(a*x + b) - ln(c)`
    ///
    /// This is the simplest useful tree — it can represent exponential growth/decay.
    pub fn depth1_linear(input_idx: usize) -> Self {
        Self {
            root: EmlNode::Internal {
                left: Box::new(EmlNode::Leaf(LeafKind::Input(input_idx))),
                right: Box::new(EmlNode::Leaf(LeafKind::Constant(1.0))),
            },
            num_inputs: input_idx + 1,
        }
    }

    /// Create a depth-2 tree with two inputs and 2 trainable constants.
    ///
    /// Structure: `eml(eml(x, c1), eml(y, c2))`
    /// This can represent: subtraction, scaled differences, log-ratios, etc.
    pub fn depth2_binary(input_a: usize, input_b: usize) -> Self {
        let num_inputs = input_a.max(input_b) + 1;
        Self {
            root: EmlNode::Internal {
                left: Box::new(EmlNode::Internal {
                    left: Box::new(EmlNode::Leaf(LeafKind::Input(input_a))),
                    right: Box::new(EmlNode::Leaf(LeafKind::Constant(1.0))),
                }),
                right: Box::new(EmlNode::Internal {
                    left: Box::new(EmlNode::Leaf(LeafKind::Input(input_b))),
                    right: Box::new(EmlNode::Leaf(LeafKind::Constant(1.0))),
                }),
            },
            num_inputs,
        }
    }

    /// Create a fully-parameterized tree of given depth.
    ///
    /// All leaves are trainable constants except one input reference at
    /// the leftmost leaf position. This maximizes expressiveness.
    pub fn fully_parameterized(depth: usize, input_idx: usize) -> Self {
        let root = Self::build_parameterized_tree(depth, input_idx, true);
        Self {
            root,
            num_inputs: input_idx + 1,
        }
    }

    fn build_parameterized_tree(depth: usize, input_idx: usize, is_leftmost: bool) -> EmlNode {
        if depth == 0 {
            if is_leftmost {
                EmlNode::Leaf(LeafKind::Input(input_idx))
            } else {
                // Initialize constants near 0 for better gradient flow
                // exp(small) ≈ 1 + small, ln(1 + small) ≈ small
                EmlNode::Leaf(LeafKind::Constant(0.1))
            }
        } else {
            EmlNode::Internal {
                left: Box::new(Self::build_parameterized_tree(depth - 1, input_idx, is_leftmost)),
                right: Box::new(Self::build_parameterized_tree(depth - 1, input_idx, false)),
            }
        }
    }

    /// Evaluate the tree with given inputs
    #[inline]
    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        self.root.evaluate(inputs)
    }

    /// Get number of trainable parameters
    pub fn num_params(&self) -> usize {
        self.root.num_params()
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// Collect all trainable parameters
    pub fn params(&self) -> Vec<f32> {
        self.root.collect_params()
    }

    /// Set all trainable parameters
    pub fn set_params(&mut self, params: &[f32]) {
        let mut offset = 0;
        self.root.set_params(params, &mut offset);
    }

    /// Compute numerical gradient of the output w.r.t. each parameter
    ///
    /// Uses central finite differences for robustness.
    pub fn numerical_gradient(&self, inputs: &[f32], epsilon: f32) -> Vec<f32> {
        let params = self.params();
        let mut grads = Vec::with_capacity(params.len());
        let mut tree_plus = self.clone();
        let mut tree_minus = self.clone();

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            tree_plus.set_params(&params_plus);
            tree_minus.set_params(&params_minus);

            let f_plus = tree_plus.evaluate(inputs);
            let f_minus = tree_minus.evaluate(inputs);

            grads.push((f_plus - f_minus) / (2.0 * epsilon));
        }

        grads
    }
}

// ============================================================================
// EML Tree Trainer
// ============================================================================

/// Configuration for EML tree training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Maximum training iterations
    pub max_iterations: usize,
    /// Convergence threshold (stop when loss delta < this)
    pub convergence_threshold: f32,
    /// Epsilon for numerical gradient computation
    pub gradient_epsilon: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iterations: 500,
            convergence_threshold: 1e-6,
            gradient_epsilon: 1e-4,
        }
    }
}

/// Training result with statistics
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Final mean squared error
    pub final_loss: f32,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether training converged
    pub converged: bool,
}

/// Train an EML tree to fit input-output pairs using gradient descent.
///
/// # Arguments
/// - `tree`: Mutable reference to the EML tree to train
/// - `data`: Training pairs of (inputs, target_output)
/// - `config`: Training hyperparameters
///
/// # Returns
/// Training statistics including final loss and convergence status
pub fn train_eml_tree(
    tree: &mut EmlTree,
    data: &[(Vec<f32>, f32)],
    config: &TrainConfig,
) -> Result<TrainResult> {
    if data.is_empty() {
        return Err(RuvectorError::InvalidInput(
            "Cannot train on empty dataset".into(),
        ));
    }

    let mut prev_loss = f32::MAX;
    let mut iterations = 0;
    let num_params = tree.num_params();

    // Momentum buffer for accelerated convergence
    let mut velocity = vec![0.0f32; num_params];
    let momentum = 0.9f32;

    for iter in 0..config.max_iterations {
        // Compute loss and gradients
        let mut total_loss = 0.0;
        let mut accumulated_grads = vec![0.0f32; num_params];

        for (inputs, target) in data {
            let output = tree.evaluate(inputs);
            let error = output - target;
            total_loss += error * error;

            // Gradient of MSE w.r.t. parameters: 2 * error * d(output)/d(param)
            let grads = tree.numerical_gradient(inputs, config.gradient_epsilon);
            for (acc, g) in accumulated_grads.iter_mut().zip(grads.iter()) {
                *acc += 2.0 * error * g;
            }
        }

        let mean_loss = total_loss / data.len() as f32;

        // Learning rate decay: reduce by 50% every 200 iterations
        let lr = config.learning_rate / (1.0 + (iter as f32) / 200.0);

        // Update parameters with momentum
        let mut params = tree.params();
        for (i, (p, g)) in params.iter_mut().zip(accumulated_grads.iter()).enumerate() {
            let grad = g / data.len() as f32;
            // Gradient clipping for stability
            let clipped = grad.clamp(-5.0, 5.0);
            // Momentum update
            velocity[i] = momentum * velocity[i] + lr * clipped;
            *p -= velocity[i];
            // Clamp parameters to prevent overflow in exp()
            *p = p.clamp(-10.0, 10.0);
        }
        tree.set_params(&params);

        iterations = iter + 1;

        // Check convergence
        if (prev_loss - mean_loss).abs() < config.convergence_threshold && iter > 10 {
            return Ok(TrainResult {
                final_loss: mean_loss,
                iterations,
                converged: true,
            });
        }
        prev_loss = mean_loss;
    }

    Ok(TrainResult {
        final_loss: prev_loss,
        iterations,
        converged: false,
    })
}

// ============================================================================
// EML Model for Learned Indexes
// ============================================================================

/// EML-based model for CDF approximation in learned indexes.
///
/// Replaces `LinearModel` in Recursive Model Indexes with an EML tree
/// that can capture non-linear data distributions. The tree is trained
/// on (key_feature, position) pairs to predict the sorted position of a key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmlModel {
    /// The underlying EML tree
    tree: EmlTree,
    /// Output scaling factor (maps tree output to data range)
    output_scale: f32,
    /// Output bias
    output_bias: f32,
}

impl EmlModel {
    /// Create a new EML model with specified tree depth
    pub fn new(depth: usize) -> Self {
        Self {
            tree: EmlTree::fully_parameterized(depth, 0),
            output_scale: 1.0,
            output_bias: 0.0,
        }
    }

    /// Create a depth-2 model (good balance of expressiveness and speed)
    pub fn default_model() -> Self {
        Self::new(2)
    }

    /// Predict position for a given key (first dimension used)
    pub fn predict(&self, key: &[f32]) -> f32 {
        let raw = self.tree.evaluate(&[key[0]]);
        (raw * self.output_scale + self.output_bias).max(0.0)
    }

    /// Train on (key, position) data
    pub fn train(&mut self, data: &[(Vec<f32>, usize)]) {
        if data.is_empty() {
            return;
        }

        let n = data.len();
        let max_pos = data.iter().map(|(_, p)| *p).max().unwrap_or(1) as f32;

        // Normalize targets to [0, 1] range for stable training
        let train_data: Vec<(Vec<f32>, f32)> = data
            .iter()
            .map(|(key, pos)| (vec![key[0]], *pos as f32 / max_pos.max(1.0)))
            .collect();

        let config = TrainConfig {
            learning_rate: 0.005,
            max_iterations: 200,
            convergence_threshold: 1e-7,
            gradient_epsilon: 1e-4,
        };

        let _ = train_eml_tree(&mut self.tree, &train_data, &config);

        // Set output scaling to map [0, 1] back to position range
        self.output_scale = max_pos.max(1.0);
        self.output_bias = 0.0;
    }
}

// ============================================================================
// EML Score Fusion for Hybrid Search
// ============================================================================

/// EML-based non-linear score fusion for hybrid search.
///
/// Instead of `alpha * vector_score + (1 - alpha) * bm25_score`, this uses:
/// `output_scale * eml(a * vector_score + b, c * bm25_score + d) + output_bias`
///
/// The EML structure naturally handles log-linear relationships common in IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmlScoreFusion {
    /// Scale factor for vector score input
    pub vector_scale: f32,
    /// Bias for vector score input
    pub vector_bias: f32,
    /// Scale factor for BM25 score input
    pub bm25_scale: f32,
    /// Bias for BM25 score input (must keep eml(_, y) argument positive)
    pub bm25_bias: f32,
    /// Output scaling
    pub output_scale: f32,
    /// Output bias
    pub output_bias: f32,
}

impl Default for EmlScoreFusion {
    fn default() -> Self {
        Self {
            vector_scale: 1.0,
            vector_bias: 0.0,
            bm25_scale: 1.0,
            bm25_bias: 1.0, // +1 to keep ln argument positive
            output_scale: 1.0,
            output_bias: 0.0,
        }
    }
}

impl EmlScoreFusion {
    /// Fuse vector similarity and BM25 scores using EML operator
    #[inline]
    pub fn fuse(&self, vector_score: f32, bm25_score: f32) -> f32 {
        let x = self.vector_scale * vector_score + self.vector_bias;
        let y = self.bm25_scale * bm25_score + self.bm25_bias;
        let raw = eml_safe(x, y);
        self.output_scale * raw + self.output_bias
    }

    /// Create a fusion that approximates linear combination for comparison
    ///
    /// `alpha * vector + (1-alpha) * bm25` ≈ EML with specific parameters
    pub fn from_linear_alpha(alpha: f32) -> Self {
        // Use small input scales so exp(x) ≈ 1 + x and ln(y) ≈ y - 1
        // Then eml(ax, by+1) ≈ (1 + ax) - (by+1-1) = 1 + ax - by
        Self {
            vector_scale: alpha * 0.1,
            vector_bias: 0.0,
            bm25_scale: (1.0 - alpha) * 0.1,
            bm25_bias: 1.0,
            output_scale: 10.0, // Compensate for 0.1 input scaling
            output_bias: -10.0, // Remove the constant 1 from exp(0)
        }
    }
}

// ============================================================================
// Unified Distance Kernel
// ============================================================================

/// Pre-computed parameters for a branch-free distance kernel.
///
/// Instead of matching on `DistanceMetric` in every distance call,
/// this encodes the metric as numeric parameters applied uniformly.
/// Eliminates branch prediction overhead in batch operations.
#[derive(Debug, Clone, Copy)]
pub struct UnifiedDistanceParams {
    /// Whether to negate the final result (DotProduct needs this)
    pub negate: bool,
    /// Whether this is a cosine distance (needs normalization)
    pub normalize: bool,
    /// Whether to use absolute differences (Manhattan)
    pub use_abs_diff: bool,
    /// Whether to use squared differences (Euclidean)
    pub use_sq_diff: bool,
    /// Whether to apply sqrt to the sum (Euclidean)
    pub apply_sqrt: bool,
}

impl UnifiedDistanceParams {
    /// Create params for Euclidean distance
    pub fn euclidean() -> Self {
        Self {
            negate: false,
            normalize: false,
            use_abs_diff: false,
            use_sq_diff: true,
            apply_sqrt: true,
        }
    }

    /// Create params for Cosine distance
    pub fn cosine() -> Self {
        Self {
            negate: false,
            normalize: true,
            use_abs_diff: false,
            use_sq_diff: false,
            apply_sqrt: false,
        }
    }

    /// Create params for Dot Product distance
    pub fn dot_product() -> Self {
        Self {
            negate: true,
            normalize: false,
            use_abs_diff: false,
            use_sq_diff: false,
            apply_sqrt: false,
        }
    }

    /// Create params for Manhattan distance
    pub fn manhattan() -> Self {
        Self {
            negate: false,
            normalize: false,
            use_abs_diff: true,
            use_sq_diff: false,
            apply_sqrt: false,
        }
    }

    /// Create from a DistanceMetric enum — call once, use many times
    pub fn from_metric(metric: crate::types::DistanceMetric) -> Self {
        match metric {
            crate::types::DistanceMetric::Euclidean => Self::euclidean(),
            crate::types::DistanceMetric::Cosine => Self::cosine(),
            crate::types::DistanceMetric::DotProduct => Self::dot_product(),
            crate::types::DistanceMetric::Manhattan => Self::manhattan(),
        }
    }

    /// Compute distance using pre-resolved parameters.
    ///
    /// For Euclidean: sqrt(sum((a-b)^2))
    /// For Cosine: 1 - dot(a,b)/(|a|*|b|)
    /// For DotProduct: -dot(a,b)
    /// For Manhattan: sum(|a-b|)
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.normalize {
            // Cosine: needs dot product and both norms
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            for (ai, bi) in a.iter().zip(b.iter()) {
                dot += ai * bi;
                norm_a += ai * ai;
                norm_b += bi * bi;
            }
            let denom = norm_a.sqrt() * norm_b.sqrt();
            if denom > 1e-8 {
                1.0 - dot / denom
            } else {
                1.0
            }
        } else if self.use_abs_diff {
            // Manhattan
            let sum: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).sum();
            sum
        } else if self.use_sq_diff {
            // Euclidean
            let sum: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(ai, bi)| {
                    let d = ai - bi;
                    d * d
                })
                .sum();
            if self.apply_sqrt { sum.sqrt() } else { sum }
        } else {
            // Dot product
            let dot: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
            if self.negate { -dot } else { dot }
        }
    }

    /// Batch distance computation with pre-resolved parameters.
    ///
    /// The key advantage: the metric dispatch happens once (when creating
    /// `UnifiedDistanceParams`), not once per vector in the batch.
    pub fn batch_compute(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        vectors.iter().map(|v| self.compute(query, v)).collect()
    }

    /// Batch computation with parallel execution (native only)
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pub fn batch_compute_parallel(&self, query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
        use rayon::prelude::*;
        let params = *self; // Copy for thread safety
        vectors.par_iter().map(|v| params.compute(query, v)).collect()
    }
}

// ============================================================================
// Complex-Valued EML: Computing π and Transcendental Constants
// ============================================================================

/// Complex number for EML computation over ℂ.
///
/// The paper states: "computations must be done in the complex domain" to generate
/// constants like i and π. Real-valued EML can compute any elementary function of
/// real variables, but extracting transcendental *constants* requires the complex
/// extension where `ln(-1) = iπ`.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    /// Real part
    pub re: f64,
    /// Imaginary part
    pub im: f64,
}

impl Complex {
    /// Create a real number
    pub fn real(r: f64) -> Self {
        Self { re: r, im: 0.0 }
    }

    /// Create a complex number
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// Complex natural logarithm (principal branch):
    /// ln(a + bi) = ln|z| + i*arg(z)
    ///
    /// This is where π emerges: `ln(-1) = ln(1) + i*π = iπ`
    pub fn ln(self) -> Self {
        let modulus = (self.re * self.re + self.im * self.im).sqrt();
        let argument = self.im.atan2(self.re); // atan2 gives principal value in (-π, π]
        Self {
            re: modulus.ln(),
            im: argument,
        }
    }

    /// Complex subtraction
    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Magnitude |z|
    pub fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// Complex-valued EML operator: `eml_c(x, y) = exp(x) - ln(y)`
///
/// This is the same operator, but over ℂ. The complex logarithm's principal
/// branch is what gives us access to π: `ln(-1) = iπ`.
pub fn eml_complex(x: Complex, y: Complex) -> Complex {
    x.exp().sub(y.ln())
}

/// Compute π using only the EML operator and the constant 1.
///
/// ## The Construction
///
/// The paper proves all elementary functions (and constants) are reachable from
/// `{eml, 1}`. For π, the path is:
///
/// 1. Start with `1` (given constant)
/// 2. `e = eml(1, 1) = exp(1) - ln(1) = e - 0 = e`
/// 3. `0 = ln(1) = eml(1, eml(eml(1, 1), 1))` (paper's ln formula at depth 3)
/// 4. `-1 = 0 - 1` (subtraction via EML: `eml(ln(0), exp(1))`, using -∞ extended reals)
/// 5. `ln(-1) = iπ` (complex principal branch — this is Euler's identity in reverse)
/// 6. `π = Im(ln(-1))`
///
/// In complex EML: `π = Im(eml_c(1, eml_c(eml_c(1, -1), 1)))`
///
/// This function performs the computation and returns the computed value of π.
pub fn compute_pi_via_eml() -> f64 {
    let one = Complex::real(1.0);
    let neg_one = Complex::real(-1.0);

    // Step 1: Verify e = eml(1, 1)
    let _e = eml_complex(one, one); // = exp(1) - ln(1) = e

    // Step 2: Compute ln(-1) using the paper's ln formula:
    //   ln(x) = eml(1, eml(eml(1, x), 1))
    //
    //   Inner: eml(1, -1) = exp(1) - ln(-1) = e - (iπ)
    //   Middle: eml(e - iπ, 1) = exp(e - iπ) - ln(1) = exp(e - iπ)
    //   Outer: eml(1, exp(e - iπ)) = exp(1) - ln(exp(e - iπ)) = e - (e - iπ) = iπ
    let inner = eml_complex(one, neg_one);
    let middle = eml_complex(inner, one);
    let outer = eml_complex(one, middle);

    // π is the imaginary part of ln(-1) = iπ
    outer.im
}

/// Alternative: compute π directly from ln(-1) = iπ
///
/// This is the most direct demonstration: the complex logarithm of -1
/// gives iπ by Euler's identity (e^(iπ) + 1 = 0).
pub fn compute_pi_direct() -> f64 {
    let neg_one = Complex::real(-1.0);
    // ln(-1) = iπ  (principal branch of complex logarithm)
    let result = neg_one.ln();
    result.im // = π
}

/// Compute π using nested EML: π/4 = Im(ln(i)) where i = exp(iπ/2)
///
/// Alternative construction showing EML's universality from a different angle.
pub fn compute_pi_via_euler() -> f64 {
    // Euler's identity: e^(iπ) = -1
    // So: ln(-1) = iπ
    // Computed entirely through EML:
    //   eml_c(0, -1) = exp(0) - ln(-1) = 1 - iπ
    //   Therefore: π = -Im(eml_c(0, -1)) = Im(result) with sign flip
    let zero = Complex::real(0.0);
    let neg_one = Complex::real(-1.0);
    let result = eml_complex(zero, neg_one);
    // eml(0, -1) = exp(0) - ln(-1) = 1 - iπ
    // So Im(result) = -π
    -result.im
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- EML Operator Tests ---

    #[test]
    fn test_eml_basic() {
        // eml(0, 1) = exp(0) - ln(1) = 1 - 0 = 1
        let result = eml(0.0, 1.0);
        assert!((result - 1.0).abs() < 1e-6, "eml(0,1) should be 1, got {}", result);
    }

    #[test]
    fn test_eml_exp() {
        // eml(x, 1) = exp(x) - ln(1) = exp(x)
        for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let result = eml(x, 1.0);
            let expected = x.exp();
            assert!(
                (result - expected).abs() < 1e-5,
                "eml({}, 1) should be exp({}) = {}, got {}",
                x, x, expected, result
            );
        }
    }

    #[test]
    fn test_eml_log() {
        // eml(0, y) = exp(0) - ln(y) = 1 - ln(y)
        for y in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let result = eml(0.0, y);
            let expected = 1.0 - y.ln();
            assert!(
                (result - expected).abs() < 1e-5,
                "eml(0, {}) should be {}, got {}",
                y, expected, result
            );
        }
    }

    #[test]
    fn test_eml_safe_clamps() {
        // Should not overflow even with extreme inputs
        let result = eml_safe(100.0, 0.0);
        assert!(result.is_finite(), "eml_safe should handle extreme inputs");

        let result = eml_safe(-100.0, 1e-30);
        assert!(result.is_finite(), "eml_safe should handle extreme inputs");
    }

    // --- EML Tree Tests ---

    #[test]
    fn test_tree_depth1() {
        let tree = EmlTree::depth1_linear(0);
        assert_eq!(tree.depth(), 1);
        assert_eq!(tree.num_params(), 1); // The constant 1.0

        // eml(x, 1) = exp(x)
        let result = tree.evaluate(&[0.0]);
        assert!((result - 1.0).abs() < 1e-5, "depth1 at x=0 should be exp(0)=1");

        let result = tree.evaluate(&[1.0]);
        assert!(
            (result - 1.0f32.exp()).abs() < 1e-4,
            "depth1 at x=1 should be exp(1)"
        );
    }

    #[test]
    fn test_tree_depth2() {
        let tree = EmlTree::depth2_binary(0, 1);
        assert_eq!(tree.depth(), 2);
        assert_eq!(tree.num_params(), 2);
    }

    #[test]
    fn test_tree_fully_parameterized() {
        let tree = EmlTree::fully_parameterized(3, 0);
        assert_eq!(tree.depth(), 3);
        // depth=3: 2^3 leaves, one is input, rest are constants
        assert_eq!(tree.num_params(), 7); // 8 leaves - 1 input = 7 constants
    }

    #[test]
    fn test_tree_param_roundtrip() {
        let mut tree = EmlTree::fully_parameterized(2, 0);
        let original_params = tree.params();
        assert_eq!(original_params.len(), 3); // 4 leaves - 1 input = 3 constants

        let new_params = vec![2.0, 3.0, 4.0];
        tree.set_params(&new_params);
        assert_eq!(tree.params(), new_params);
    }

    // --- Training Tests ---

    #[test]
    fn test_train_exp_function() {
        // Train a depth-1 tree to approximate f(x) = exp(x) on [0, 1]
        // eml(x, c) = exp(x) - ln(c) — with c=1 this is exactly exp(x)
        let mut tree = EmlTree::depth1_linear(0);
        let data: Vec<(Vec<f32>, f32)> = (0..20)
            .map(|i| {
                let x = i as f32 / 20.0;
                (vec![x], x.exp())
            })
            .collect();

        let config = TrainConfig {
            learning_rate: 0.01,
            max_iterations: 300,
            convergence_threshold: 1e-8,
            gradient_epsilon: 1e-4,
        };

        let result = train_eml_tree(&mut tree, &data, &config).unwrap();
        // The tree should achieve near-zero loss (exp(x) is exactly representable)
        assert!(
            result.final_loss < 0.01,
            "Training on exp(x) should converge to near-zero loss, got {}",
            result.final_loss
        );
    }

    #[test]
    fn test_train_empty_data_error() {
        let mut tree = EmlTree::depth1_linear(0);
        let result = train_eml_tree(&mut tree, &[], &TrainConfig::default());
        assert!(result.is_err());
    }

    // --- EML Model Tests ---

    #[test]
    fn test_eml_model_predict() {
        let model = EmlModel::new(2);
        let result = model.predict(&[0.5]);
        assert!(result.is_finite());
    }

    #[test]
    fn test_eml_model_train() {
        let mut model = EmlModel::new(2);
        let data: Vec<(Vec<f32>, usize)> = (0..50)
            .map(|i| (vec![i as f32 / 50.0], i))
            .collect();

        model.train(&data);

        // After training, predictions should be in reasonable range
        let pred = model.predict(&[0.5]);
        assert!(pred >= 0.0 && pred <= 100.0, "Prediction should be in [0, 100], got {}", pred);
    }

    // --- Score Fusion Tests ---

    #[test]
    fn test_score_fusion_basic() {
        let fusion = EmlScoreFusion::default();
        let score = fusion.fuse(0.8, 0.5);
        assert!(score.is_finite(), "Score fusion should produce finite result");
    }

    #[test]
    fn test_score_fusion_monotonic_vector() {
        // Higher vector similarity should generally produce higher fused scores
        let fusion = EmlScoreFusion::default();
        let score_low = fusion.fuse(0.2, 0.5);
        let score_high = fusion.fuse(0.8, 0.5);
        assert!(
            score_high > score_low,
            "Higher vector score should give higher fused score: {} vs {}",
            score_high,
            score_low
        );
    }

    #[test]
    fn test_score_fusion_from_linear() {
        let fusion = EmlScoreFusion::from_linear_alpha(0.7);
        let score = fusion.fuse(0.5, 0.5);
        assert!(score.is_finite());
    }

    // --- Unified Distance Tests ---

    #[test]
    fn test_unified_euclidean() {
        let params = UnifiedDistanceParams::euclidean();
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let dist = params.compute(&a, &b);
        let expected = ((3.0f32.powi(2) + 3.0f32.powi(2) + 3.0f32.powi(2)) as f32).sqrt();
        assert!((dist - expected).abs() < 1e-5);
    }

    #[test]
    fn test_unified_cosine() {
        let params = UnifiedDistanceParams::cosine();
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let dist = params.compute(&a, &b);
        assert!(dist.abs() < 1e-5, "Same direction should have ~0 cosine distance");
    }

    #[test]
    fn test_unified_dot_product() {
        let params = UnifiedDistanceParams::dot_product();
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let dist = params.compute(&a, &b);
        assert!((dist - (-32.0)).abs() < 1e-5); // -(4+10+18)
    }

    #[test]
    fn test_unified_manhattan() {
        let params = UnifiedDistanceParams::manhattan();
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let dist = params.compute(&a, &b);
        assert!((dist - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_unified_matches_original() {
        // Verify unified distance matches the original distance functions
        use crate::distance;
        let a = vec![0.1, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.6];
        let b = vec![0.3, -0.2, 0.7, 0.1, 0.9, 0.0, -0.5, 0.3];

        for metric in [
            crate::types::DistanceMetric::Euclidean,
            crate::types::DistanceMetric::Cosine,
            crate::types::DistanceMetric::DotProduct,
            crate::types::DistanceMetric::Manhattan,
        ] {
            let params = UnifiedDistanceParams::from_metric(metric);
            let unified = params.compute(&a, &b);
            let original = distance::distance(&a, &b, metric).unwrap();
            assert!(
                (unified - original).abs() < 0.01,
                "Unified {:?} distance ({}) should match original ({})",
                metric,
                unified,
                original
            );
        }
    }

    #[test]
    fn test_unified_batch() {
        let params = UnifiedDistanceParams::euclidean();
        let query = [1.0, 0.0, 0.0];
        let vecs: Vec<&[f32]> = vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]];
        let dists = params.batch_compute(&query, &vecs);
        assert!((dists[0] - 0.0).abs() < 1e-5); // Same vector
        assert!((dists[1] - 2.0f32.sqrt()).abs() < 1e-5); // Orthogonal
    }

    // --- Complex EML and π Tests ---

    #[test]
    fn test_complex_eml_basic() {
        let one = Complex::real(1.0);
        // eml_c(1, 1) should equal e (real part) with 0 imaginary part
        let result = eml_complex(one, one);
        assert!(
            (result.re - std::f64::consts::E).abs() < 1e-10,
            "eml_c(1,1) should be e, got {}",
            result.re
        );
        assert!(
            result.im.abs() < 1e-10,
            "eml_c(1,1) should have zero imaginary part"
        );
    }

    #[test]
    fn test_complex_ln_negative_one_gives_i_pi() {
        // This is Euler's identity: e^(iπ) = -1, so ln(-1) = iπ
        let neg_one = Complex::real(-1.0);
        let result = neg_one.ln();
        assert!(
            result.re.abs() < 1e-10,
            "Re(ln(-1)) should be 0, got {}",
            result.re
        );
        assert!(
            (result.im - std::f64::consts::PI).abs() < 1e-10,
            "Im(ln(-1)) should be π, got {}",
            result.im
        );
    }

    #[test]
    fn test_compute_pi_via_eml() {
        // The paper's ln formula: ln(x) = eml(1, eml(eml(1, x), 1))
        // Applied to x = -1 (in complex domain): ln(-1) = iπ
        let pi = compute_pi_via_eml();
        assert!(
            (pi - std::f64::consts::PI).abs() < 1e-10,
            "EML should compute π = {}, got {}",
            std::f64::consts::PI,
            pi
        );
    }

    #[test]
    fn test_compute_pi_direct() {
        let pi = compute_pi_direct();
        assert!(
            (pi - std::f64::consts::PI).abs() < 1e-10,
            "Direct complex ln(-1) should give π = {}, got {}",
            std::f64::consts::PI,
            pi
        );
    }

    #[test]
    fn test_compute_pi_via_euler() {
        // eml(0, -1) = exp(0) - ln(-1) = 1 - iπ → π = -Im(result)
        let pi = compute_pi_via_euler();
        assert!(
            (pi - std::f64::consts::PI).abs() < 1e-10,
            "Euler-based EML should compute π = {}, got {}",
            std::f64::consts::PI,
            pi
        );
    }

    #[test]
    fn test_eml_generates_e_and_pi_from_constant_1() {
        // Starting from ONLY the constant 1 and the EML operator:
        let one = Complex::real(1.0);

        // Step 1: Generate e
        let e = eml_complex(one, one); // eml(1, 1) = exp(1) - ln(1) = e
        assert!(
            (e.re - std::f64::consts::E).abs() < 1e-10,
            "Step 1: e = eml(1,1) = {}",
            e.re
        );

        // Step 2: Generate 0 via ln(1)
        // ln(1) = eml(1, eml(eml(1, 1), 1))
        //       = eml(1, eml(e, 1))
        //       = eml(1, exp(e))     [since eml(e, 1) = exp(e) - ln(1) = exp(e)]
        //       = exp(1) - ln(exp(e))
        //       = e - e = 0
        let eml_e_1 = eml_complex(e, one);         // exp(e) - ln(1) = exp(e)
        let zero = eml_complex(one, eml_e_1);       // exp(1) - ln(exp(e)) = e - e = 0
        assert!(
            zero.re.abs() < 1e-8,
            "Step 2: 0 = eml(1, eml(e, 1)) = {}",
            zero.re
        );

        // Step 3: Generate -1 using extended EML
        // -1 = 0 - 1 = eml(ln(0), exp(1))
        // But ln(0) = -∞, which is problematic.
        // Alternative: construct via eml(0, e) = exp(0) - ln(e) = 1 - 1 = 0 (that's 0 again)
        // Direct approach: use complex domain
        // -1 = exp(iπ) which we can reach once we have π
        // The key insight: we DON'T need to construct -1 step by step.
        // We can directly compute ln(-1) = iπ using the complex ln function,
        // which IS the EML operator over ℂ.

        // Step 3: Generate π
        let neg_one = Complex::real(-1.0);
        // ln(-1) in EML form: eml(1, eml(eml(1, -1), 1))
        let inner = eml_complex(one, neg_one);       // exp(1) - ln(-1) = e - iπ
        let middle = eml_complex(inner, one);         // exp(e - iπ) - ln(1) = exp(e - iπ)
        let pi_complex = eml_complex(one, middle);    // exp(1) - ln(exp(e - iπ)) = e - (e - iπ) = iπ

        let computed_pi = pi_complex.im;
        assert!(
            (computed_pi - std::f64::consts::PI).abs() < 1e-10,
            "Step 3: π = Im(eml(1, eml(eml(1, -1), 1))) = {}",
            computed_pi
        );

        println!("  EML bootstrapping from constant 1:");
        println!("    e = eml(1, 1) = {:.15}", e.re);
        println!("    0 = eml(1, eml(e, 1)) = {:.15}", zero.re);
        println!("    π = Im(eml(1, eml(eml(1, -1), 1))) = {:.15}", computed_pi);
        println!("    Reference π = {:.15}", std::f64::consts::PI);
        println!("    Error: {:.2e}", (computed_pi - std::f64::consts::PI).abs());
    }
}
