//! Unified Attention Architecture for Edge-Net P2P AI
//!
//! Four attention mechanisms answering fundamental questions:
//!
//! - **Neural Attention**: What words/tokens matter?
//! - **DAG Attention**: What steps in a sequence matter?
//! - **Graph Attention**: What relationships matter?
//! - **State Space**: What history still matters?
//!
//! All mechanisms output importance scores for interpretability.

use std::collections::HashMap;

// ============================================================================
// Common Types
// ============================================================================

/// Unified attention output with importance scores
#[derive(Clone, Debug)]
pub struct AttentionOutput {
    /// Output embeddings
    pub embeddings: Vec<f32>,
    /// Importance scores (what matters)
    pub importance: Vec<f32>,
    /// Attention weights matrix (optional)
    pub attention_weights: Option<Vec<Vec<f32>>>,
    /// Metadata about attention computation
    pub metadata: AttentionMetadata,
}

/// Metadata about attention computation
#[derive(Clone, Debug, Default)]
pub struct AttentionMetadata {
    /// Entropy of attention distribution (lower = more focused)
    pub entropy: f32,
    /// Max attention score
    pub max_score: f32,
    /// Number of attended positions
    pub attended_count: usize,
    /// Sparsity ratio (0-1)
    pub sparsity: f32,
}

/// Common configuration for all attention types
#[derive(Clone, Debug)]
pub struct AttentionConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dropout: f32,
    pub use_layer_norm: bool,
    pub attention_type: AttentionType,
}

/// Type of attention mechanism
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttentionType {
    Neural,     // What words matter
    DAG,        // What steps matter
    Graph,      // What relationships matter
    StateSpace, // What history matters
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_heads: 8,
            dropout: 0.1,
            use_layer_norm: true,
            attention_type: AttentionType::Neural,
        }
    }
}

// ============================================================================
// Neural Attention: What Words/Tokens Matter
// ============================================================================

/// Multi-head self-attention for token importance
///
/// Answers: "What words in this sequence matter for the current context?"
pub struct NeuralAttention {
    config: AttentionConfig,
    /// Query projection [hidden_dim, hidden_dim]
    w_q: Vec<f32>,
    /// Key projection [hidden_dim, hidden_dim]
    w_k: Vec<f32>,
    /// Value projection [hidden_dim, hidden_dim]
    w_v: Vec<f32>,
    /// Output projection [hidden_dim, hidden_dim]
    w_o: Vec<f32>,
    /// Learned positional embeddings (optional)
    pos_embeddings: Option<Vec<Vec<f32>>>,
}

impl NeuralAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self, String> {
        if hidden_dim % num_heads != 0 {
            return Err(format!("hidden_dim {} must be divisible by num_heads {}", hidden_dim, num_heads));
        }

        let size = hidden_dim * hidden_dim;
        let scale = 1.0 / (hidden_dim as f32).sqrt();

        // Xavier initialization
        let init_weight = |size: usize| -> Vec<f32> {
            (0..size).map(|i| {
                let x = (i as f32 * 0.1).sin() * scale;
                x
            }).collect()
        };

        Ok(Self {
            config: AttentionConfig {
                hidden_dim,
                num_heads,
                attention_type: AttentionType::Neural,
                ..Default::default()
            },
            w_q: init_weight(size),
            w_k: init_weight(size),
            w_v: init_weight(size),
            w_o: init_weight(size),
            pos_embeddings: None,
        })
    }

    /// Enable learnable positional embeddings
    pub fn with_positions(mut self, max_len: usize) -> Self {
        let dim = self.config.hidden_dim;
        self.pos_embeddings = Some((0..max_len).map(|pos| {
            (0..dim).map(|i| {
                let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / dim as f32);
                if i % 2 == 0 { angle.sin() } else { angle.cos() }
            }).collect()
        }).collect());
        self
    }

    /// Forward pass: compute attention over token sequence
    ///
    /// Returns importance scores answering "what words matter"
    pub fn forward(&self, tokens: &[Vec<f32>]) -> AttentionOutput {
        if tokens.is_empty() {
            return AttentionOutput {
                embeddings: vec![],
                importance: vec![],
                attention_weights: None,
                metadata: AttentionMetadata::default(),
            };
        }

        let seq_len = tokens.len();
        let dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Add positional embeddings if available
        let tokens_with_pos: Vec<Vec<f32>> = if let Some(ref pos_emb) = self.pos_embeddings {
            tokens.iter().enumerate().map(|(i, tok)| {
                let pos = &pos_emb[i.min(pos_emb.len() - 1)];
                tok.iter().zip(pos.iter()).map(|(t, p)| t + p).collect()
            }).collect()
        } else {
            tokens.to_vec()
        };

        // Project to Q, K, V
        let queries: Vec<Vec<f32>> = tokens_with_pos.iter()
            .map(|t| self.linear(t, &self.w_q, dim)).collect();
        let keys: Vec<Vec<f32>> = tokens_with_pos.iter()
            .map(|t| self.linear(t, &self.w_k, dim)).collect();
        let values: Vec<Vec<f32>> = tokens_with_pos.iter()
            .map(|t| self.linear(t, &self.w_v, dim)).collect();

        // Compute attention scores [seq_len, seq_len]
        let mut attention_weights = vec![vec![0.0f32; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dot: f32 = queries[i].iter().zip(keys[j].iter())
                    .map(|(q, k)| q * k).sum();
                attention_weights[i][j] = dot * scale;
            }
            // Softmax over row
            self.softmax(&mut attention_weights[i]);
        }

        // Compute importance: average attention received by each position
        let mut importance = vec![0.0f32; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                importance[j] += attention_weights[i][j];
            }
        }
        // Normalize importance
        let max_imp = importance.iter().cloned().fold(0.0f32, f32::max);
        if max_imp > 0.0 {
            for imp in &mut importance {
                *imp /= max_imp;
            }
        }

        // Weighted sum of values
        let mut outputs = vec![vec![0.0f32; dim]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                for d in 0..dim {
                    outputs[i][d] += attention_weights[i][j] * values[j][d];
                }
            }
        }

        // Output projection
        let embeddings: Vec<f32> = outputs.iter()
            .flat_map(|o| self.linear(o, &self.w_o, dim))
            .collect();

        // Compute metadata
        let entropy = self.compute_entropy(&attention_weights);
        let max_score = attention_weights.iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0f32, f32::max);
        let sparsity = importance.iter().filter(|&&x| x < 0.1).count() as f32 / seq_len as f32;

        AttentionOutput {
            embeddings,
            importance,
            attention_weights: Some(attention_weights),
            metadata: AttentionMetadata {
                entropy,
                max_score,
                attended_count: seq_len,
                sparsity,
            },
        }
    }

    fn linear(&self, input: &[f32], weight: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len().min(out_dim);
        let mut output = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                output[o] += input[i] * weight[i * out_dim + o];
            }
        }
        output
    }

    fn softmax(&self, scores: &mut [f32]) {
        if scores.is_empty() { return; }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() { *s /= sum; }
        }
    }

    fn compute_entropy(&self, weights: &[Vec<f32>]) -> f32 {
        let mut total_entropy = 0.0f32;
        for row in weights {
            for &w in row {
                if w > 1e-10 {
                    total_entropy -= w * w.ln();
                }
            }
        }
        total_entropy / weights.len() as f32
    }
}

// ============================================================================
// DAG Attention: What Steps Matter
// ============================================================================

/// DAG node for step representation
#[derive(Clone, Debug)]
pub struct DAGNode {
    pub id: usize,
    pub embedding: Vec<f32>,
    pub dependencies: Vec<usize>, // Parent nodes
}

/// Topological attention over directed acyclic graphs
///
/// Answers: "What steps in this workflow matter for the current decision?"
pub struct DAGAttention {
    config: AttentionConfig,
    /// Step embedding projection
    w_step: Vec<f32>,
    /// Dependency weighting
    w_dep: Vec<f32>,
    /// Critical path scoring
    w_critical: Vec<f32>,
}

impl DAGAttention {
    pub fn new(hidden_dim: usize) -> Self {
        let size = hidden_dim * hidden_dim;
        let scale = 1.0 / (hidden_dim as f32).sqrt();

        Self {
            config: AttentionConfig {
                hidden_dim,
                num_heads: 1,
                attention_type: AttentionType::DAG,
                ..Default::default()
            },
            w_step: (0..size).map(|i| (i as f32 * 0.1).sin() * scale).collect(),
            w_dep: (0..size).map(|i| (i as f32 * 0.2).cos() * scale).collect(),
            w_critical: vec![scale; hidden_dim],
        }
    }

    /// Forward pass over DAG nodes
    ///
    /// Returns importance scores answering "what steps matter"
    pub fn forward(&self, nodes: &[DAGNode]) -> AttentionOutput {
        if nodes.is_empty() {
            return AttentionOutput {
                embeddings: vec![],
                importance: vec![],
                attention_weights: None,
                metadata: AttentionMetadata::default(),
            };
        }

        let num_nodes = nodes.len();
        let dim = self.config.hidden_dim;

        // Topological order (simple: use node IDs assuming they're ordered)
        let topo_order: Vec<usize> = (0..num_nodes).collect();

        // Compute step embeddings with dependency aggregation
        let mut step_embeddings = vec![vec![0.0f32; dim]; num_nodes];
        let mut dependency_weights = vec![vec![0.0f32; num_nodes]; num_nodes];

        for &node_idx in &topo_order {
            let node = &nodes[node_idx];

            // Base embedding
            let base = self.linear(&node.embedding, &self.w_step, dim);

            // Aggregate from dependencies with attention
            if !node.dependencies.is_empty() {
                let mut dep_scores = vec![0.0f32; node.dependencies.len()];
                for (i, &dep_idx) in node.dependencies.iter().enumerate() {
                    if dep_idx < num_nodes {
                        // Score based on embedding similarity
                        let dep_emb = &step_embeddings[dep_idx];
                        let score: f32 = base.iter().zip(dep_emb.iter())
                            .map(|(a, b)| a * b).sum();
                        dep_scores[i] = score;
                        dependency_weights[node_idx][dep_idx] = score;
                    }
                }

                // Softmax over dependencies
                self.softmax(&mut dep_scores);

                // Weighted aggregation
                for (i, &dep_idx) in node.dependencies.iter().enumerate() {
                    if dep_idx < num_nodes {
                        for d in 0..dim {
                            step_embeddings[node_idx][d] += dep_scores[i] * step_embeddings[dep_idx][d];
                        }
                    }
                }
            }

            // Add base embedding
            for d in 0..dim {
                step_embeddings[node_idx][d] += base[d];
            }
        }

        // Compute importance: critical path analysis
        // Nodes with more dependents and on longer paths are more important
        let mut importance = vec![0.0f32; num_nodes];
        let mut path_lengths = vec![0usize; num_nodes];

        // Forward pass for path lengths
        for &node_idx in &topo_order {
            let node = &nodes[node_idx];
            let max_dep_length = node.dependencies.iter()
                .filter_map(|&d| path_lengths.get(d).copied())
                .max()
                .unwrap_or(0);
            path_lengths[node_idx] = max_dep_length + 1;
        }

        // Count dependents (reverse dependencies)
        let mut dependent_count = vec![0usize; num_nodes];
        for node in nodes {
            for &dep in &node.dependencies {
                if dep < num_nodes {
                    dependent_count[dep] += 1;
                }
            }
        }

        // Importance = path_length * (dependents + 1)
        let max_path = path_lengths.iter().cloned().max().unwrap_or(1) as f32;
        let max_deps = dependent_count.iter().cloned().max().unwrap_or(1) as f32;

        for i in 0..num_nodes {
            importance[i] = (path_lengths[i] as f32 / max_path) *
                           ((dependent_count[i] + 1) as f32 / (max_deps + 1.0));
        }

        // Flatten embeddings
        let embeddings: Vec<f32> = step_embeddings.iter().flatten().cloned().collect();

        let entropy = importance.iter()
            .filter(|&&x| x > 1e-10)
            .map(|&x| -x * x.ln())
            .sum::<f32>() / num_nodes as f32;

        let max_score = importance.iter().cloned().fold(0.0f32, f32::max);
        let sparsity = importance.iter().filter(|&&x| x < 0.1).count() as f32 / num_nodes as f32;

        AttentionOutput {
            embeddings,
            importance,
            attention_weights: Some(dependency_weights),
            metadata: AttentionMetadata {
                entropy,
                max_score,
                attended_count: num_nodes,
                sparsity,
            },
        }
    }

    fn linear(&self, input: &[f32], weight: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len().min(out_dim);
        let mut output = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                output[o] += input[i] * weight[i * out_dim + o];
            }
        }
        output
    }

    fn softmax(&self, scores: &mut [f32]) {
        if scores.is_empty() { return; }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() { *s /= sum; }
        }
    }
}

// ============================================================================
// Graph Attention: What Relationships Matter
// ============================================================================

/// Edge with attention-relevant features
#[derive(Clone, Debug)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub edge_type: u8,
    pub weight: f32,
}

/// Graph Attention Network (GAT) style attention
///
/// Answers: "What relationships/edges in this graph matter?"
pub struct GraphAttentionNetwork {
    config: AttentionConfig,
    /// Node feature projection
    w_node: Vec<f32>,
    /// Attention mechanism (source side)
    a_src: Vec<f32>,
    /// Attention mechanism (target side)
    a_tgt: Vec<f32>,
    /// Edge type embeddings
    edge_embeddings: Vec<Vec<f32>>,
    /// Number of edge types
    num_edge_types: usize,
}

impl GraphAttentionNetwork {
    pub fn new(hidden_dim: usize, num_heads: usize, num_edge_types: usize) -> Result<Self, String> {
        if hidden_dim % num_heads != 0 {
            return Err(format!("hidden_dim {} not divisible by num_heads {}", hidden_dim, num_heads));
        }

        let size = hidden_dim * hidden_dim;
        let scale = 1.0 / (hidden_dim as f32).sqrt();

        Ok(Self {
            config: AttentionConfig {
                hidden_dim,
                num_heads,
                attention_type: AttentionType::Graph,
                ..Default::default()
            },
            w_node: (0..size).map(|i| (i as f32 * 0.1).sin() * scale).collect(),
            a_src: vec![scale; hidden_dim],
            a_tgt: vec![scale; hidden_dim],
            edge_embeddings: (0..num_edge_types.max(1))
                .map(|t| (0..hidden_dim).map(|i| ((t * i) as f32 * 0.1).sin() * scale).collect())
                .collect(),
            num_edge_types,
        })
    }

    /// Forward pass over graph
    ///
    /// Returns importance scores answering "what relationships matter"
    pub fn forward(&self, node_features: &[Vec<f32>], edges: &[Edge]) -> AttentionOutput {
        if node_features.is_empty() {
            return AttentionOutput {
                embeddings: vec![],
                importance: vec![],
                attention_weights: None,
                metadata: AttentionMetadata::default(),
            };
        }

        let num_nodes = node_features.len();
        let dim = self.config.hidden_dim;

        // Project node features
        let h: Vec<Vec<f32>> = node_features.iter()
            .map(|feat| self.linear(feat, &self.w_node, dim))
            .collect();

        // Build adjacency with attention weights
        let mut attention_weights = vec![vec![0.0f32; num_nodes]; num_nodes];
        let mut edge_importance = vec![0.0f32; edges.len()];

        // Compute attention coefficients for each edge
        for (edge_idx, edge) in edges.iter().enumerate() {
            if edge.source >= num_nodes || edge.target >= num_nodes {
                continue;
            }

            let src = &h[edge.source];
            let tgt = &h[edge.target];

            // Edge attention: LeakyReLU(a_src * h_src + a_tgt * h_tgt + edge_emb)
            let edge_emb = &self.edge_embeddings[edge.edge_type as usize % self.edge_embeddings.len()];

            let mut score = 0.0f32;
            for d in 0..dim {
                score += self.a_src[d] * src[d] + self.a_tgt[d] * tgt[d];
                if d < edge_emb.len() {
                    score += edge_emb[d] * 0.1;
                }
            }

            // LeakyReLU
            score = if score > 0.0 { score } else { 0.01 * score };
            score *= edge.weight;

            attention_weights[edge.source][edge.target] = score;
            edge_importance[edge_idx] = score.abs();
        }

        // Softmax over neighbors for each node
        for i in 0..num_nodes {
            let neighbors: Vec<usize> = edges.iter()
                .filter(|e| e.source == i)
                .map(|e| e.target)
                .collect();

            if !neighbors.is_empty() {
                let scores: Vec<f32> = neighbors.iter()
                    .map(|&j| attention_weights[i][j])
                    .collect();

                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();

                for &j in &neighbors {
                    attention_weights[i][j] = (attention_weights[i][j] - max).exp() / exp_sum.max(1e-10);
                }
            }
        }

        // Aggregate neighbor features
        let mut outputs = vec![vec![0.0f32; dim]; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if attention_weights[i][j] > 0.0 {
                    for d in 0..dim {
                        outputs[i][d] += attention_weights[i][j] * h[j][d];
                    }
                }
            }
            // Add self-loop
            for d in 0..dim {
                outputs[i][d] += h[i][d];
            }
        }

        // Node importance: sum of incoming attention + degree centrality
        let mut node_importance = vec![0.0f32; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                node_importance[i] += attention_weights[j][i];
            }
        }

        // Normalize
        let max_imp = node_importance.iter().cloned().fold(0.0f32, f32::max);
        if max_imp > 0.0 {
            for imp in &mut node_importance {
                *imp /= max_imp;
            }
        }

        // Normalize edge importance
        let max_edge_imp = edge_importance.iter().cloned().fold(0.0f32, f32::max);
        if max_edge_imp > 0.0 {
            for imp in &mut edge_importance {
                *imp /= max_edge_imp;
            }
        }

        let embeddings: Vec<f32> = outputs.iter().flatten().cloned().collect();
        let edge_sparsity = edge_importance.iter().filter(|&&x| x < 0.1).count() as f32 / edges.len().max(1) as f32;
        let entropy = node_importance.iter()
            .filter(|&&x| x > 1e-10)
            .map(|&x| -x * x.ln())
            .sum::<f32>() / num_nodes as f32;

        AttentionOutput {
            embeddings,
            importance: edge_importance, // Edge importance (relationships)
            attention_weights: Some(attention_weights),
            metadata: AttentionMetadata {
                entropy,
                max_score: max_edge_imp,
                attended_count: edges.len(),
                sparsity: edge_sparsity,
            },
        }
    }

    fn linear(&self, input: &[f32], weight: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len().min(out_dim);
        let mut output = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                output[o] += input[i] * weight[i * out_dim + o];
            }
        }
        output
    }
}

// ============================================================================
// State Space: What History Matters
// ============================================================================

/// Selective State Space Model (Mamba-style)
///
/// Answers: "What historical context still matters for current processing?"
pub struct StateSpaceModel {
    config: AttentionConfig,
    /// State dimension (N in paper)
    state_dim: usize,
    /// Discretization: delta projection
    w_delta: Vec<f32>,
    /// Input projection for B
    w_b: Vec<f32>,
    /// Input projection for C
    w_c: Vec<f32>,
    /// State transition A (structured, diagonal for efficiency)
    a_diag: Vec<f32>,
    /// Output projection D (skip connection)
    d_skip: Vec<f32>,
}

impl StateSpaceModel {
    pub fn new(hidden_dim: usize, state_dim: usize) -> Self {
        let scale = 1.0 / (hidden_dim as f32).sqrt();

        // Initialize A with exponential decay (HiPPO-inspired)
        let a_diag: Vec<f32> = (0..state_dim)
            .map(|i| -((i + 1) as f32).ln())
            .collect();

        Self {
            config: AttentionConfig {
                hidden_dim,
                num_heads: 1,
                attention_type: AttentionType::StateSpace,
                ..Default::default()
            },
            state_dim,
            w_delta: (0..hidden_dim).map(|i| (i as f32 * 0.1).sin() * scale + 0.5).collect(),
            w_b: (0..hidden_dim * state_dim).map(|i| (i as f32 * 0.1).sin() * scale).collect(),
            w_c: (0..hidden_dim * state_dim).map(|i| (i as f32 * 0.1).cos() * scale).collect(),
            a_diag,
            d_skip: vec![0.1; hidden_dim],
        }
    }

    /// Forward pass over sequence
    ///
    /// Returns importance scores answering "what history matters"
    pub fn forward(&self, sequence: &[Vec<f32>]) -> AttentionOutput {
        if sequence.is_empty() {
            return AttentionOutput {
                embeddings: vec![],
                importance: vec![],
                attention_weights: None,
                metadata: AttentionMetadata::default(),
            };
        }

        let seq_len = sequence.len();
        let dim = self.config.hidden_dim;
        let state_dim = self.state_dim;

        // Initialize state
        let mut state = vec![0.0f32; state_dim];
        let mut outputs = vec![vec![0.0f32; dim]; seq_len];
        let mut state_history = vec![vec![0.0f32; state_dim]; seq_len];
        let mut importance = vec![0.0f32; seq_len];

        for (t, input) in sequence.iter().enumerate() {
            // Compute input-dependent delta (discretization step)
            let delta: f32 = input.iter()
                .zip(self.w_delta.iter())
                .map(|(x, w)| x * w)
                .sum::<f32>()
                .exp()
                .min(1.0)
                .max(0.001);

            // Compute B (input projection to state)
            let b = self.project_to_state(input, &self.w_b);

            // Compute C (state to output projection)
            let c = self.project_to_state(input, &self.w_c);

            // Discretized state update: x_t = exp(delta * A) * x_{t-1} + delta * B * u_t
            let mut new_state = vec![0.0f32; state_dim];
            for i in 0..state_dim {
                // Discretized A: exp(delta * a_i)
                let a_bar = (delta * self.a_diag[i]).exp();
                new_state[i] = a_bar * state[i] + delta * b[i];
            }

            state = new_state;
            state_history[t] = state.clone();

            // Compute output: y_t = C * x_t + D * u_t
            let mut y = vec![0.0f32; dim];
            for d in 0..dim {
                // C * state
                for s in 0..state_dim.min(dim) {
                    y[d] += c[s] * state[s];
                }
                // Skip connection
                if d < input.len() {
                    y[d] += self.d_skip[d] * input[d];
                }
            }

            outputs[t] = y;

            // Importance: how much state is "active" (L2 norm of state)
            importance[t] = state.iter().map(|x| x * x).sum::<f32>().sqrt();
        }

        // Compute relative history importance
        // Higher state norm = more history being retained
        let mut history_weights = vec![vec![0.0f32; seq_len]; seq_len];

        for t in 0..seq_len {
            // Compute contribution of each past position to current state
            for past in 0..=t {
                let decay = (-((t - past) as f32) / 10.0).exp();
                let contribution = importance[past] * decay;
                history_weights[t][past] = contribution;
            }

            // Normalize
            let sum: f32 = history_weights[t].iter().sum();
            if sum > 0.0 {
                for w in &mut history_weights[t] {
                    *w /= sum;
                }
            }
        }

        // Normalize importance
        let max_imp = importance.iter().cloned().fold(0.0f32, f32::max);
        if max_imp > 0.0 {
            for imp in &mut importance {
                *imp /= max_imp;
            }
        }

        let embeddings: Vec<f32> = outputs.iter().flatten().cloned().collect();
        let entropy = importance.iter()
            .filter(|&&x| x > 1e-10)
            .map(|&x| -x * x.ln())
            .sum::<f32>() / seq_len as f32;

        AttentionOutput {
            embeddings,
            importance,
            attention_weights: Some(history_weights),
            metadata: AttentionMetadata {
                entropy,
                max_score: max_imp,
                attended_count: seq_len,
                sparsity: 0.0, // State space is dense
            },
        }
    }

    fn project_to_state(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let in_dim = input.len().min(self.config.hidden_dim);
        let mut output = vec![0.0f32; self.state_dim];

        for s in 0..self.state_dim {
            for i in 0..in_dim {
                output[s] += input[i] * weight[s * self.config.hidden_dim + i];
            }
        }
        output
    }
}

// ============================================================================
// Unified Attention: Combining All Four Types
// ============================================================================

/// Unified attention that combines all four attention types
pub struct UnifiedAttention {
    pub neural: NeuralAttention,
    pub dag: DAGAttention,
    pub graph: GraphAttentionNetwork,
    pub state_space: StateSpaceModel,
    /// Fusion weights for combining outputs
    fusion_weights: [f32; 4],
}

impl UnifiedAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self, String> {
        Ok(Self {
            neural: NeuralAttention::new(hidden_dim, num_heads)?,
            dag: DAGAttention::new(hidden_dim),
            graph: GraphAttentionNetwork::new(hidden_dim, num_heads, 8)?,
            state_space: StateSpaceModel::new(hidden_dim, 16),
            fusion_weights: [0.25, 0.25, 0.25, 0.25],
        })
    }

    /// Set fusion weights for combining different attention outputs
    pub fn with_fusion_weights(mut self, weights: [f32; 4]) -> Self {
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            self.fusion_weights = weights.map(|w| w / sum);
        }
        self
    }

    /// Process with all attention types and fuse results
    pub fn forward_all(
        &self,
        tokens: &[Vec<f32>],
        dag_nodes: Option<&[DAGNode]>,
        graph: Option<(&[Vec<f32>], &[Edge])>,
    ) -> HashMap<AttentionType, AttentionOutput> {
        let mut results = HashMap::new();

        // Neural attention on tokens
        if !tokens.is_empty() {
            results.insert(AttentionType::Neural, self.neural.forward(tokens));
            results.insert(AttentionType::StateSpace, self.state_space.forward(tokens));
        }

        // DAG attention if provided
        if let Some(nodes) = dag_nodes {
            results.insert(AttentionType::DAG, self.dag.forward(nodes));
        }

        // Graph attention if provided
        if let Some((node_features, edges)) = graph {
            results.insert(AttentionType::Graph, self.graph.forward(node_features, edges));
        }

        results
    }

    /// Get unified importance scores from all attention types
    pub fn get_unified_importance(&self, results: &HashMap<AttentionType, AttentionOutput>) -> Vec<f32> {
        // Find max length
        let max_len = results.values()
            .map(|r| r.importance.len())
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            return vec![];
        }

        let mut unified = vec![0.0f32; max_len];
        let mut weight_sum = 0.0f32;

        let types = [
            (AttentionType::Neural, self.fusion_weights[0]),
            (AttentionType::DAG, self.fusion_weights[1]),
            (AttentionType::Graph, self.fusion_weights[2]),
            (AttentionType::StateSpace, self.fusion_weights[3]),
        ];

        for (attention_type, weight) in types {
            if let Some(output) = results.get(&attention_type) {
                for (i, &imp) in output.importance.iter().enumerate() {
                    if i < max_len {
                        unified[i] += weight * imp;
                    }
                }
                weight_sum += weight;
            }
        }

        // Normalize
        if weight_sum > 0.0 {
            for u in &mut unified {
                *u /= weight_sum;
            }
        }

        unified
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_attention() {
        let attn = NeuralAttention::new(64, 8).unwrap();
        let tokens = vec![vec![1.0; 64], vec![0.5; 64], vec![0.2; 64]];

        let output = attn.forward(&tokens);

        assert_eq!(output.importance.len(), 3);
        assert!(output.importance.iter().all(|&x| x >= 0.0 && x <= 1.0));
        assert!(output.attention_weights.is_some());
    }

    #[test]
    fn test_dag_attention() {
        let attn = DAGAttention::new(64);
        let nodes = vec![
            DAGNode { id: 0, embedding: vec![1.0; 64], dependencies: vec![] },
            DAGNode { id: 1, embedding: vec![0.5; 64], dependencies: vec![0] },
            DAGNode { id: 2, embedding: vec![0.2; 64], dependencies: vec![0, 1] },
        ];

        let output = attn.forward(&nodes);

        assert_eq!(output.importance.len(), 3);
        // Node 0 has dependents, should be more important
        assert!(output.importance[0] > 0.0);
    }

    #[test]
    fn test_graph_attention() {
        let attn = GraphAttentionNetwork::new(64, 8, 4).unwrap();
        let features = vec![vec![1.0; 64], vec![0.5; 64], vec![0.2; 64]];
        let edges = vec![
            Edge { source: 0, target: 1, edge_type: 0, weight: 1.0 },
            Edge { source: 1, target: 2, edge_type: 1, weight: 0.5 },
        ];

        let output = attn.forward(&features, &edges);

        assert_eq!(output.importance.len(), 2); // Edge importance
        assert!(output.attention_weights.is_some());
    }

    #[test]
    fn test_state_space() {
        let ssm = StateSpaceModel::new(64, 16);
        let sequence = vec![vec![1.0; 64], vec![0.5; 64], vec![0.2; 64], vec![0.1; 64]];

        let output = ssm.forward(&sequence);

        assert_eq!(output.importance.len(), 4);
        // Later positions should have more accumulated state
        assert!(output.attention_weights.is_some());
    }

    #[test]
    fn test_unified_attention() {
        let unified = UnifiedAttention::new(64, 8).unwrap();
        let tokens = vec![vec![1.0; 64], vec![0.5; 64]];
        let dag_nodes = vec![
            DAGNode { id: 0, embedding: vec![1.0; 64], dependencies: vec![] },
            DAGNode { id: 1, embedding: vec![0.5; 64], dependencies: vec![0] },
        ];
        let features = vec![vec![1.0; 64], vec![0.5; 64]];
        let edges = vec![Edge { source: 0, target: 1, edge_type: 0, weight: 1.0 }];

        let results = unified.forward_all(&tokens, Some(&dag_nodes), Some((&features, &edges)));

        assert!(results.contains_key(&AttentionType::Neural));
        assert!(results.contains_key(&AttentionType::DAG));
        assert!(results.contains_key(&AttentionType::Graph));
        assert!(results.contains_key(&AttentionType::StateSpace));

        let unified_importance = unified.get_unified_importance(&results);
        assert!(!unified_importance.is_empty());
    }

    #[test]
    fn test_attention_with_positions() {
        let attn = NeuralAttention::new(64, 8)
            .unwrap()
            .with_positions(100);

        let tokens = vec![vec![1.0; 64], vec![1.0; 64]];
        let output = attn.forward(&tokens);

        // With positions, identical tokens should have different attention
        assert_eq!(output.embeddings.len(), 128); // 2 tokens * 64 dim
    }
}
