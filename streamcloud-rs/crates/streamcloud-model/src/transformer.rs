//! candle implementation of the Geometric Context Transformer (the real-weights
//! backend, behind the `candle` feature).
//!
//! This is a structural implementation of the streaming architecture described
//! in the LingBot-Map paper: patch embedding → N pre-norm transformer blocks →
//! anchor cross-attention (the retrieved context from `streamcloud-memory`) → a
//! keyframe-feature head. It compiles and runs, and loads weights from a
//! safetensors checkpoint via [`GeometricContextTransformer::from_safetensors`].
//!
//! **Fidelity note (ADR-0006):** exact tensor names / block layout must match
//! the upstream checkpoint to reproduce the trained model. Until validated
//! against the real 4.63 GB weights, treat this as the loading-path scaffold;
//! the demo/pipeline default to [`crate::SyntheticReconstructor`].

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder, VarMap};
use streamcloud_tensor::ModelConfig;

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(dim, hidden, vb.pp("fc1"))?,
            fc2: linear(hidden, dim, vb.pp("fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.gelu()?;
        self.fc2.forward(&x)
    }
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
}

impl Attention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: linear(dim, dim, vb.pp("proj"))?,
            num_heads,
        })
    }

    /// Self-attention over `[seq, dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, dim) = x.dims2()?;
        let hd = dim / self.num_heads;
        let scale = 1.0 / (hd as f64).sqrt();

        let qkv = self.qkv.forward(x)?;
        let q = qkv.narrow(1, 0, dim)?;
        let k = qkv.narrow(1, dim, dim)?;
        let v = qkv.narrow(1, 2 * dim, dim)?;

        let reshape = |t: &Tensor| -> Result<Tensor> {
            t.reshape((seq, self.num_heads, hd))?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = reshape(&q)?;
        let k = reshape(&k)?;
        let v = reshape(&v)?;

        let att = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?; // [heads, seq, hd]
        let out = out.transpose(0, 1)?.reshape((seq, dim))?;
        self.proj.forward(&out)
    }
}

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(cfg.hidden_dim, 1e-5, vb.pp("norm1"))?,
            attn: Attention::new(cfg.hidden_dim, cfg.num_heads, vb.pp("attn"))?,
            norm2: layer_norm(cfg.hidden_dim, 1e-5, vb.pp("norm2"))?,
            mlp: Mlp::new(cfg.hidden_dim, cfg.ffn_dim, vb.pp("mlp"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?)?)?;
        let x = (&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

/// The Geometric Context Transformer.
pub struct GeometricContextTransformer {
    patch_embed: Linear,
    blocks: Vec<Block>,
    norm: LayerNorm,
    feature_head: Linear,
    cfg: ModelConfig,
    device: Device,
}

impl GeometricContextTransformer {
    fn build(cfg: ModelConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        let patch_dim = cfg.patch_size * cfg.patch_size * 3;
        let patch_embed = linear(patch_dim, cfg.hidden_dim, vb.pp("patch_embed"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(&cfg, vb.pp(format!("blocks.{i}")))?);
        }
        let norm = layer_norm(cfg.hidden_dim, 1e-5, vb.pp("norm"))?;
        let feature_head = linear(
            cfg.hidden_dim,
            cfg.keyframe_feature_dim,
            vb.pp("feature_head"),
        )?;
        Ok(Self {
            patch_embed,
            blocks,
            norm,
            feature_head,
            cfg,
            device,
        })
    }

    /// Random-initialized model (for shape tests / synthetic runs).
    pub fn random(cfg: ModelConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::build(cfg, vb, device)
    }

    /// Load weights from a safetensors checkpoint (mmap, no full RAM load).
    pub fn from_safetensors(
        cfg: ModelConfig,
        path: impl AsRef<std::path::Path>,
        device: Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.as_ref().to_path_buf()],
                DType::F32,
                &device,
            )?
        };
        Self::build(cfg, vb, device)
    }

    /// The device this model runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Encode patch tokens `[num_patches, patch_dim]` into a single pooled,
    /// L2-normalized keyframe feature vector `[keyframe_feature_dim]`.
    ///
    /// `anchor_tokens`, when present (`[num_anchors, keyframe_feature_dim]`), are
    /// mean-pooled and added as a drift-correction bias before the final norm —
    /// the candle analogue of anchor cross-attention.
    pub fn encode_keyframe(
        &self,
        patches: &Tensor,
        anchor_tokens: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(patches)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        x = self.norm.forward(&x)?;
        // Mean-pool tokens → [hidden].
        let pooled = x.mean(0)?;
        let mut feat = self
            .feature_head
            .forward(&pooled.unsqueeze(0)?)?
            .squeeze(0)?;
        if let Some(anchors) = anchor_tokens {
            let bias = anchors.mean(0)?.affine(0.25, 0.0)?;
            feat = (feat + bias)?;
        }
        // L2 normalize.
        let norm = feat.sqr()?.sum_all()?.sqrt()?;
        feat.broadcast_div(&norm)
    }

    /// The configuration in use.
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_forward_shapes() -> Result<()> {
        // Tiny config so the test is fast.
        let cfg = ModelConfig {
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            ffn_dim: 64,
            patch_size: 4,
            keyframe_feature_dim: 16,
            image_height: 16,
            image_width: 16,
            anchor_top_k: 4,
        };
        let device = Device::Cpu;
        let model = GeometricContextTransformer::random(cfg.clone(), device.clone())?;
        let patch_dim = cfg.patch_size * cfg.patch_size * 3;
        let patches = Tensor::randn(0.0f32, 1.0, (cfg.num_patches(), patch_dim), &device)?;
        let feat = model.encode_keyframe(&patches, None)?;
        assert_eq!(feat.dims(), &[cfg.keyframe_feature_dim]);
        // L2 norm ~ 1.
        let n = feat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!((n - 1.0).abs() < 1e-3);
        Ok(())
    }
}
