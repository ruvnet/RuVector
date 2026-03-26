//! # GraphMAE: Masked Autoencoders for Graphs
//!
//! Self-supervised graph learning via masked feature reconstruction. Traditional
//! supervised graph learning requires expensive node/edge labels that are scarce in
//! real-world graphs. GraphMAE learns representations by masking and reconstructing
//! node features, requiring **zero labels**. The learned embeddings transfer well to
//! downstream tasks (classification, link prediction, clustering) because the model
//! must capture structural and semantic graph properties to reconstruct masked features
//! from their neighborhood context.
//!
//! Pipeline: Mask -> GAT Encode -> Re-mask latent -> Decode masked only -> SCE loss.
//!
//! Reference: Hou et al., "GraphMAE: Self-Supervised Masked Graph Autoencoders", KDD 2022.

use crate::error::GnnError;
use crate::layer::{LayerNorm, Linear};
use rand::seq::SliceRandom;
use rand::Rng;

/// Loss function variant for reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFn {
    /// Scaled Cosine Error: `(1 - cos_sim)^gamma`. Default for GraphMAE.
    Sce { /// Scaling exponent (default 2.0).
        gamma: f32 },
    /// Standard Mean Squared Error.
    Mse,
}

impl Default for LossFn {
    fn default() -> Self { Self::Sce { gamma: 2.0 } }
}

/// Configuration for a GraphMAE model.
#[derive(Debug, Clone)]
pub struct GraphMAEConfig {
    /// Fraction of nodes to mask (default 0.5).
    pub mask_ratio: f32,
    /// Number of GAT encoder layers.
    pub num_layers: usize,
    /// Hidden / latent dimension.
    pub hidden_dim: usize,
    /// Number of attention heads per encoder layer.
    pub num_heads: usize,
    /// Number of decoder layers.
    pub decoder_layers: usize,
    /// Secondary mask ratio applied to latent before decoding (default 0.0).
    pub re_mask_ratio: f32,
    /// Reconstruction loss function.
    pub loss_fn: LossFn,
    /// Input feature dimension.
    pub input_dim: usize,
}

impl Default for GraphMAEConfig {
    fn default() -> Self {
        Self {
            mask_ratio: 0.5, num_layers: 2, hidden_dim: 64, num_heads: 4,
            decoder_layers: 1, re_mask_ratio: 0.0, loss_fn: LossFn::default(), input_dim: 64,
        }
    }
}

/// Sparse graph representation.
#[derive(Debug, Clone)]
pub struct GraphData {
    /// Node feature matrix: `node_features[i]` is the feature vector for node `i`.
    pub node_features: Vec<Vec<f32>>,
    /// Adjacency list: `adjacency[i]` contains neighbor indices of node `i`.
    pub adjacency: Vec<Vec<usize>>,
    /// Number of nodes.
    pub num_nodes: usize,
}

/// Result of masking node features.
#[derive(Debug, Clone)]
pub struct MaskResult {
    /// Features after masking (mask token substituted).
    pub masked_features: Vec<Vec<f32>>,
    /// Indices of masked nodes.
    pub mask_indices: Vec<usize>,
}

/// Feature masking strategies for GraphMAE.
pub struct FeatureMasking {
    mask_token: Vec<f32>,
}

impl FeatureMasking {
    /// Create a masking module with a learnable `[MASK]` token of given dimension.
    pub fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self { mask_token: (0..dim).map(|_| rng.gen::<f32>() * 0.02 - 0.01).collect() }
    }

    /// Randomly mask `mask_ratio` of nodes, replacing features with `[MASK]` token.
    pub fn mask_nodes(&self, features: &[Vec<f32>], mask_ratio: f32) -> MaskResult {
        let n = features.len();
        let num_mask = ((n as f32) * mask_ratio.clamp(0.0, 1.0)).round() as usize;
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let mask_indices = indices[..num_mask.min(n)].to_vec();
        let mut masked = features.to_vec();
        for &i in &mask_indices { masked[i] = self.mask_token.clone(); }
        MaskResult { masked_features: masked, mask_indices }
    }

    /// Degree-centrality masking: higher-degree nodes are masked with higher probability.
    pub fn mask_by_degree(
        &self, features: &[Vec<f32>], adjacency: &[Vec<usize>], mask_ratio: f32,
    ) -> MaskResult {
        let n = features.len();
        let num_mask = ((n as f32) * mask_ratio.clamp(0.0, 1.0)).round() as usize;
        let degrees: Vec<f32> = adjacency.iter().map(|a| a.len() as f32 + 1.0).collect();
        let total: f32 = degrees.iter().sum();
        let probs: Vec<f32> = degrees.iter().map(|d| d / total).collect();
        let mut rng = rand::thread_rng();
        let mut avail: Vec<usize> = (0..n).collect();
        let mut mask_indices = Vec::with_capacity(num_mask);
        for _ in 0..num_mask.min(n) {
            if avail.is_empty() { break; }
            let rp: Vec<f32> = avail.iter().map(|&i| probs[i]).collect();
            let s: f32 = rp.iter().sum();
            if s <= 0.0 { break; }
            let thr = rng.gen::<f32>() * s;
            let mut cum = 0.0;
            let mut chosen = 0;
            for (pos, &p) in rp.iter().enumerate() {
                cum += p;
                if cum >= thr { chosen = pos; break; }
            }
            mask_indices.push(avail[chosen]);
            avail.swap_remove(chosen);
        }
        let mut masked = features.to_vec();
        for &i in &mask_indices { masked[i] = self.mask_token.clone(); }
        MaskResult { masked_features: masked, mask_indices }
    }
}

/// Single GAT layer with residual connection and layer normalization.
struct GATLayer {
    linear: Linear,
    attn_src: Vec<f32>,
    attn_dst: Vec<f32>,
    norm: LayerNorm,
    num_heads: usize,
}

impl GATLayer {
    fn new(input_dim: usize, output_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let hd = output_dim / num_heads.max(1);
        Self {
            linear: Linear::new(input_dim, output_dim),
            attn_src: (0..hd).map(|_| rng.gen::<f32>() * 0.1).collect(),
            attn_dst: (0..hd).map(|_| rng.gen::<f32>() * 0.1).collect(),
            norm: LayerNorm::new(output_dim, 1e-5),
            num_heads,
        }
    }

    fn forward(&self, features: &[Vec<f32>], adj: &[Vec<usize>]) -> Vec<Vec<f32>> {
        let proj: Vec<Vec<f32>> = features.iter().map(|f| self.linear.forward(f)).collect();
        let od = proj.first().map_or(0, |v| v.len());
        let hd = od / self.num_heads.max(1);
        let mut output = Vec::with_capacity(features.len());
        for i in 0..features.len() {
            if adj[i].is_empty() {
                output.push(elu_vec(&proj[i]));
                continue;
            }
            let mut agg = vec![0.0f32; od];
            for h in 0..self.num_heads {
                let (s, e) = (h * hd, (h + 1) * hd);
                let ss: f32 = proj[i][s..e].iter().zip(&self.attn_src).map(|(a, b)| a * b).sum();
                let mut scores: Vec<f32> = adj[i].iter().map(|&j| {
                    let ds: f32 = proj[j][s..e].iter().zip(&self.attn_dst).map(|(a, b)| a * b).sum();
                    let v = ss + ds;
                    if v >= 0.0 { v } else { 0.2 * v } // leaky relu
                }).collect();
                let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = scores.iter_mut().map(|v| (*v - mx).exp()).collect();
                let sm = exp.iter().sum::<f32>().max(1e-10);
                for (k, &j) in adj[i].iter().enumerate() {
                    let w = exp[k] / sm;
                    for d in s..e { agg[d] += w * proj[j][d]; }
                }
            }
            for v in &mut agg { *v /= self.num_heads as f32; }
            if features[i].len() == od {
                for (a, &f) in agg.iter_mut().zip(features[i].iter()) { *a += f; }
            }
            output.push(elu_vec(&self.norm.forward(&agg)));
        }
        output
    }
}

/// Multi-layer GAT encoder for GraphMAE.
pub struct GATEncoder { layers: Vec<GATLayer> }

impl GATEncoder {
    /// Build an encoder with `num_layers` GAT layers.
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize, num_heads: usize) -> Self {
        let layers = (0..num_layers).map(|i| {
            GATLayer::new(if i == 0 { input_dim } else { hidden_dim }, hidden_dim, num_heads)
        }).collect();
        Self { layers }
    }

    /// Encode node features through all GAT layers.
    pub fn encode(&self, features: &[Vec<f32>], adj: &[Vec<usize>]) -> Vec<Vec<f32>> {
        self.layers.iter().fold(features.to_vec(), |h, l| l.forward(&h, adj))
    }
}

/// Decoder that reconstructs only masked node features (key efficiency gain).
pub struct GraphMAEDecoder { layers: Vec<Linear>, norm: LayerNorm }

impl GraphMAEDecoder {
    /// Create a decoder mapping `hidden_dim` -> `output_dim`.
    pub fn new(hidden_dim: usize, output_dim: usize, num_layers: usize) -> Self {
        let n = num_layers.max(1);
        let layers = (0..n).map(|i| {
            let out = if i == n - 1 { output_dim } else { hidden_dim };
            Linear::new(if i == 0 { hidden_dim } else { hidden_dim }, out)
        }).collect();
        Self { layers, norm: LayerNorm::new(output_dim, 1e-5) }
    }

    /// Decode latent for masked nodes. Applies re-masking (zeroing dims) for regularization.
    pub fn decode(&self, latent: &[Vec<f32>], mask_idx: &[usize], re_mask: f32) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        mask_idx.iter().map(|&idx| {
            let mut h = latent[idx].clone();
            if re_mask > 0.0 {
                let nz = ((h.len() as f32) * re_mask).round() as usize;
                let mut dims: Vec<usize> = (0..h.len()).collect();
                dims.shuffle(&mut rng);
                for &d in dims.iter().take(nz) { h[d] = 0.0; }
            }
            for layer in &self.layers { h = elu_vec(&layer.forward(&h)); }
            self.norm.forward(&h)
        }).collect()
    }
}

/// Scaled Cosine Error: `mean((1 - cos_sim(pred, target))^gamma)` over masked nodes.
pub fn sce_loss(preds: &[Vec<f32>], targets: &[Vec<f32>], gamma: f32) -> f32 {
    if preds.is_empty() { return 0.0; }
    preds.iter().zip(targets).map(|(p, t)| {
        let dot: f32 = p.iter().zip(t).map(|(a, b)| a * b).sum();
        let np = p.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let nt = t.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        (1.0 - (dot / (np * nt)).clamp(-1.0, 1.0)).powf(gamma)
    }).sum::<f32>() / preds.len() as f32
}

/// Mean Squared Error across masked node reconstructions.
pub fn mse_loss(preds: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    if preds.is_empty() { return 0.0; }
    let n: usize = preds.iter().map(|v| v.len()).sum();
    if n == 0 { return 0.0; }
    preds.iter().zip(targets).flat_map(|(p, t)| {
        p.iter().zip(t).map(|(a, b)| (a - b).powi(2))
    }).sum::<f32>() / n as f32
}

/// GraphMAE self-supervised model.
pub struct GraphMAE {
    config: GraphMAEConfig,
    masking: FeatureMasking,
    encoder: GATEncoder,
    decoder: GraphMAEDecoder,
}

impl GraphMAE {
    /// Construct a new GraphMAE model from configuration.
    ///
    /// # Errors
    /// Returns `GnnError::LayerConfig` if dimensions are incompatible.
    pub fn new(config: GraphMAEConfig) -> Result<Self, GnnError> {
        if config.hidden_dim % config.num_heads != 0 {
            return Err(GnnError::layer_config(format!(
                "hidden_dim ({}) must be divisible by num_heads ({})",
                config.hidden_dim, config.num_heads
            )));
        }
        if !(0.0..=1.0).contains(&config.mask_ratio) {
            return Err(GnnError::layer_config("mask_ratio must be in [0.0, 1.0]"));
        }
        let masking = FeatureMasking::new(config.input_dim);
        let encoder = GATEncoder::new(config.input_dim, config.hidden_dim, config.num_layers, config.num_heads);
        let decoder = GraphMAEDecoder::new(config.hidden_dim, config.input_dim, config.decoder_layers);
        Ok(Self { config, masking, encoder, decoder })
    }

    /// Run one training step: mask -> encode -> re-mask -> decode -> loss.
    /// Returns the reconstruction loss computed only on masked nodes.
    pub fn train_step(&self, graph: &GraphData) -> f32 {
        let mr = self.masking.mask_nodes(&graph.node_features, self.config.mask_ratio);
        let latent = self.encoder.encode(&mr.masked_features, &graph.adjacency);
        let recon = self.decoder.decode(&latent, &mr.mask_indices, self.config.re_mask_ratio);
        let targets: Vec<Vec<f32>> = mr.mask_indices.iter().map(|&i| graph.node_features[i].clone()).collect();
        match self.config.loss_fn {
            LossFn::Sce { gamma } => sce_loss(&recon, &targets, gamma),
            LossFn::Mse => mse_loss(&recon, &targets),
        }
    }

    /// Encode without masking (inference mode). Returns latent embeddings for all nodes.
    pub fn encode(&self, graph: &GraphData) -> Vec<Vec<f32>> {
        self.encoder.encode(&graph.node_features, &graph.adjacency)
    }

    /// Returns node-level representations for downstream tasks.
    pub fn get_embeddings(&self, graph: &GraphData) -> Vec<Vec<f32>> { self.encode(graph) }
}

fn elu_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| if x >= 0.0 { x } else { x.exp() - 1.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(n: usize, d: usize) -> GraphData {
        let feats: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..d).map(|j| (i * d + j) as f32 * 0.1).collect()).collect();
        let adj: Vec<Vec<usize>> = (0..n).map(|i| {
            let mut nb = Vec::new();
            if i > 0 { nb.push(i - 1); }
            if i + 1 < n { nb.push(i + 1); }
            nb
        }).collect();
        GraphData { node_features: feats, adjacency: adj, num_nodes: n }
    }

    fn cfg(dim: usize) -> GraphMAEConfig {
        GraphMAEConfig {
            input_dim: dim, hidden_dim: 16, num_heads: 4, num_layers: 2,
            decoder_layers: 1, mask_ratio: 0.5, re_mask_ratio: 0.0, loss_fn: LossFn::default(),
        }
    }

    #[test]
    fn test_masking_ratio() {
        let feats: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 8]).collect();
        let m = FeatureMasking::new(8);
        let r = m.mask_nodes(&feats, 0.3);
        assert!((r.mask_indices.len() as i32 - 30).unsigned_abs() <= 1);
    }

    #[test]
    fn test_encoder_forward() {
        let g = graph(5, 16);
        let enc = GATEncoder::new(16, 16, 2, 4);
        let out = enc.encode(&g.node_features, &g.adjacency);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_decoder_reconstruction_shape() {
        let dec = GraphMAEDecoder::new(16, 8, 1);
        let lat: Vec<Vec<f32>> = (0..5).map(|_| vec![0.5; 16]).collect();
        let r = dec.decode(&lat, &[0, 2, 4], 0.0);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].len(), 8);
    }

    #[test]
    fn test_sce_loss_identical() {
        let loss = sce_loss(&[vec![1.0, 0.0, 0.0]], &[vec![1.0, 0.0, 0.0]], 2.0);
        assert!(loss < 1e-6, "SCE identical should be ~0, got {loss}");
    }

    #[test]
    fn test_sce_loss_orthogonal() {
        let loss = sce_loss(&[vec![1.0, 0.0]], &[vec![0.0, 1.0]], 2.0);
        assert!((loss - 1.0).abs() < 1e-5, "SCE orthogonal should be 1.0, got {loss}");
    }

    #[test]
    fn test_mse_loss() {
        assert!(mse_loss(&[vec![1.0, 2.0]], &[vec![1.0, 2.0]]) < 1e-8);
        assert!((mse_loss(&[vec![0.0, 0.0]], &[vec![1.0, 1.0]]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_step_returns_finite_loss() {
        let model = GraphMAE::new(cfg(16)).unwrap();
        let loss = model.train_step(&graph(10, 16));
        assert!(loss.is_finite() && loss >= 0.0, "bad loss: {loss}");
    }

    #[test]
    fn test_re_masking() {
        let dec = GraphMAEDecoder::new(16, 8, 1);
        let lat = vec![vec![1.0; 16]; 3];
        let a = dec.decode(&lat, &[0, 1, 2], 0.0);
        let b = dec.decode(&lat, &[0, 1, 2], 0.8);
        let diff: f32 = a[0].iter().zip(&b[0]).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-6, "re-masking should change output");
    }

    #[test]
    fn test_degree_based_masking() {
        let feats: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0; 8]).collect();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); 10];
        for i in 1..10 { adj[0].push(i); adj[i].push(0); }
        let r = FeatureMasking::new(8).mask_by_degree(&feats, &adj, 0.5);
        assert_eq!(r.mask_indices.len(), 5);
    }

    #[test]
    fn test_single_node_graph() {
        let g = GraphData { node_features: vec![vec![1.0; 16]], adjacency: vec![vec![]], num_nodes: 1 };
        assert!(GraphMAE::new(cfg(16)).unwrap().train_step(&g).is_finite());
    }

    #[test]
    fn test_encode_for_downstream() {
        let model = GraphMAE::new(cfg(16)).unwrap();
        let emb = model.get_embeddings(&graph(8, 16));
        assert_eq!(emb.len(), 8);
        assert_eq!(emb[0].len(), 16);
        for e in &emb { for &v in e { assert!(v.is_finite()); } }
    }

    #[test]
    fn test_invalid_config() {
        assert!(GraphMAE::new(GraphMAEConfig { hidden_dim: 15, num_heads: 4, ..cfg(16) }).is_err());
        assert!(GraphMAE::new(GraphMAEConfig { mask_ratio: 1.5, ..cfg(16) }).is_err());
    }
}
