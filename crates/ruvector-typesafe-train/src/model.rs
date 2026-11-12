//! BERT encoder for training (bge-small-en-v1.5 layout, base tensor names).
//!
//! Not `candle_transformers::models::bert`: in candle-nn 0.9.2 the fused
//! `ops::layer_norm` is `apply_op3_no_bwd` and `LayerNorm::forward` takes that
//! path for contiguous input, and `softmax_last_dim` likewise has no backward.
//! Using it would silently cut the gradient at every LayerNorm. This file
//! composes both from differentiable primitives; the forward pass is otherwise
//! the HF BertModel (erf GELU, eps 1e-12, additive `(1-m)*f32::MIN` mask,
//! absolute positions 0..seq, CLS pooling, L2 normalization).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, Var, D};
use candle_nn::{Embedding, Linear, VarBuilder, VarMap};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BertConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub max_pos: usize,
    pub type_vocab: usize,
    pub eps: f64,
}

impl BertConfig {
    pub fn bge_small() -> Self {
        Self {
            vocab: 30522,
            hidden: 384,
            layers: 12,
            heads: 12,
            intermediate: 1536,
            max_pos: 512,
            type_vocab: 2,
            eps: 1e-12,
        }
    }

    /// Tiny config for tests.
    pub fn tiny(vocab: usize) -> Self {
        Self {
            vocab,
            hidden: 32,
            layers: 2,
            heads: 4,
            intermediate: 64,
            max_pos: 128,
            type_vocab: 2,
            eps: 1e-12,
        }
    }

    /// Every float parameter name with its shape, in base-safetensors naming.
    pub fn param_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden;
        let mut v = vec![
            (
                "embeddings.word_embeddings.weight".into(),
                vec![self.vocab, h],
            ),
            (
                "embeddings.position_embeddings.weight".into(),
                vec![self.max_pos, h],
            ),
            (
                "embeddings.token_type_embeddings.weight".into(),
                vec![self.type_vocab, h],
            ),
            ("embeddings.LayerNorm.weight".into(), vec![h]),
            ("embeddings.LayerNorm.bias".into(), vec![h]),
        ];
        for l in 0..self.layers {
            let p = format!("encoder.layer.{l}");
            for (m, o, i) in [
                ("attention.self.query", h, h),
                ("attention.self.key", h, h),
                ("attention.self.value", h, h),
                ("attention.output.dense", h, h),
                ("intermediate.dense", self.intermediate, h),
                ("output.dense", h, self.intermediate),
            ] {
                v.push((format!("{p}.{m}.weight"), vec![o, i]));
                v.push((format!("{p}.{m}.bias"), vec![o]));
            }
            for m in ["attention.output.LayerNorm", "output.LayerNorm"] {
                v.push((format!("{p}.{m}.weight"), vec![h]));
                v.push((format!("{p}.{m}.bias"), vec![h]));
            }
        }
        v
    }
}

/// Differentiable LayerNorm (biased variance inside the sqrt, like torch).
struct Ln {
    w: Tensor,
    b: Tensor,
    eps: f64,
}

impl Ln {
    fn load(vb: VarBuilder, n: usize, eps: f64) -> candle_core::Result<Self> {
        Ok(Self {
            w: vb.get(n, "weight")?,
            b: vb.get(n, "bias")?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        xn.broadcast_mul(&self.w)?.broadcast_add(&self.b)
    }
}

fn lin(vb: VarBuilder, i: usize, o: usize) -> candle_core::Result<Linear> {
    Ok(Linear::new(
        vb.get((o, i), "weight")?,
        Some(vb.get(o, "bias")?),
    ))
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln1: Ln,
    inter: Linear,
    out: Linear,
    ln2: Ln,
}

pub struct Bert {
    word: Embedding,
    pos: Embedding,
    tok: Embedding,
    ln: Ln,
    layers: Vec<Layer>,
    cfg: BertConfig,
    pub device: Device,
}

impl Bert {
    pub fn new(vb: VarBuilder, cfg: BertConfig) -> candle_core::Result<Self> {
        let h = cfg.hidden;
        let e = vb.pp("embeddings");
        let emb = |n: usize, name: &str| -> candle_core::Result<Embedding> {
            Ok(Embedding::new(e.pp(name).get((n, h), "weight")?, h))
        };
        let (word, pos, tok) = (
            emb(cfg.vocab, "word_embeddings")?,
            emb(cfg.max_pos, "position_embeddings")?,
            emb(cfg.type_vocab, "token_type_embeddings")?,
        );
        let ln = Ln::load(e.pp("LayerNorm"), h, cfg.eps)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = vb.pp(format!("encoder.layer.{l}"));
            let a = p.pp("attention");
            layers.push(Layer {
                q: lin(a.pp("self.query"), h, h)?,
                k: lin(a.pp("self.key"), h, h)?,
                v: lin(a.pp("self.value"), h, h)?,
                o: lin(a.pp("output.dense"), h, h)?,
                ln1: Ln::load(a.pp("output.LayerNorm"), h, cfg.eps)?,
                inter: lin(p.pp("intermediate.dense"), h, cfg.intermediate)?,
                out: lin(p.pp("output.dense"), cfg.intermediate, h)?,
                ln2: Ln::load(p.pp("output.LayerNorm"), h, cfg.eps)?,
            });
        }
        Ok(Self {
            word,
            pos,
            tok,
            ln,
            layers,
            cfg,
            device: vb.device().clone(),
        })
    }

    /// `last_hidden_state` [b, s, hidden]. `ids`/`types` are u32, `mask` f32 {0,1}.
    pub fn forward(
        &self,
        ids: &Tensor,
        types: &Tensor,
        mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, s) = ids.dims2()?;
        let pos_ids = Tensor::arange(0u32, s as u32, &self.device)?;
        let x = (self.word.forward(ids)? + self.tok.forward(types)?)?
            .broadcast_add(&self.pos.forward(&pos_ids)?)?;
        let mut x = self.ln.forward(&x)?;
        let ext = ((mask.ones_like()? - mask)? * f64::from(f32::MIN))?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let nh = self.cfg.heads;
        let dh = self.cfg.hidden / nh;
        let scale = 1.0 / (dh as f64).sqrt();
        let split = |t: Tensor| -> candle_core::Result<Tensor> {
            t.reshape((b, s, nh, dh))?.transpose(1, 2)?.contiguous()
        };
        for l in &self.layers {
            let q = split(l.q.forward(&x)?)?;
            let k = split(l.k.forward(&x)?)?;
            let v = split(l.v.forward(&x)?)?;
            let scores = (q.matmul(&k.t()?.contiguous()?)? * scale)?.broadcast_add(&ext)?;
            let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let ctx = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
                b,
                s,
                self.cfg.hidden,
            ))?;
            let a = l.ln1.forward(&(l.o.forward(&ctx)? + &x)?)?;
            let i = l.inter.forward(&a)?.gelu_erf()?;
            x = l.ln2.forward(&(l.out.forward(&i)? + &a)?)?;
        }
        Ok(x)
    }

    /// CLS-pooled, L2-normalized sentence embedding [b, hidden].
    pub fn embed(
        &self,
        ids: &Tensor,
        types: &Tensor,
        mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let cls = self.forward(ids, types, mask)?.i((.., 0, ..))?;
        l2_normalize(&cls)
    }
}

pub fn l2_normalize(x: &Tensor) -> candle_core::Result<Tensor> {
    let n = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f64::MAX)?;
    x.broadcast_div(&n)
}

/// A non-tracking view of the encoder (tensors share the Vars' storage but are
/// not variables), so evaluation builds no autograd graph and its activations
/// are freed as the forward pass proceeds. Reflects later optimizer updates,
/// because `Var::set` writes into the shared storage in place.
pub fn frozen_view(vm: &VarMap, cfg: BertConfig, device: &Device) -> Result<Bert> {
    let data = vm.data().lock().expect("varmap lock");
    let map: std::collections::HashMap<String, Tensor> = data
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
        .collect();
    drop(data);
    Ok(Bert::new(
        VarBuilder::from_tensors(map, DType::F32, device),
        cfg,
    )?)
}

/// Encoder variables populated from a safetensors file with base names.
/// Extra tensors in the file (pooler, `position_ids`) are ignored; a missing
/// or mis-shaped encoder tensor is an error.
pub fn load_encoder(path: &Path, cfg: BertConfig, device: &Device) -> Result<(VarMap, Bert)> {
    crate::pins::check_extension(path)?;
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("load {}", path.display()))?;
    let vm = VarMap::new();
    {
        let mut data = vm.data().lock().expect("varmap lock");
        for (name, shape) in cfg.param_shapes() {
            let t = tensors
                .get(&name)
                .with_context(|| format!("{} lacks encoder tensor {name}", path.display()))?;
            if t.dims() != shape.as_slice() {
                bail!("{name}: shape {:?}, expected {shape:?}", t.dims());
            }
            data.insert(name, Var::from_tensor(&t.to_dtype(DType::F32)?)?);
        }
    }
    let bert = Bert::new(VarBuilder::from_varmap(&vm, DType::F32, device), cfg)?;
    Ok((vm, bert))
}

/// Randomly initialised encoder (tests, and the synthetic transplant graph).
pub fn random_encoder(cfg: BertConfig, seed: u64, device: &Device) -> Result<(VarMap, Bert)> {
    let vm = VarMap::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    {
        let mut data = vm.data().lock().expect("varmap lock");
        for (name, shape) in cfg.param_shapes() {
            let n: usize = shape.iter().product();
            let vals: Vec<f32> = if name.contains("LayerNorm.weight") {
                vec![1.0; n]
            } else {
                gaussian(&mut rng, n, 0.05)
            };
            data.insert(
                name,
                Var::from_tensor(&Tensor::from_vec(vals, shape, device)?)?,
            );
        }
    }
    let bert = Bert::new(VarBuilder::from_varmap(&vm, DType::F32, device), cfg)?;
    Ok((vm, bert))
}

/// Box–Muller normal samples from a seeded ChaCha stream (deterministic init).
pub fn gaussian(rng: &mut ChaCha8Rng, n: usize, std: f32) -> Vec<f32> {
    use rand::Rng;
    (0..n)
        .map(|_| {
            let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
            let u2: f32 = rng.gen();
            std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect()
}

/// Cosine classifier heads: logits = scale · cos(e, w_c).
pub struct Heads {
    pub vm: VarMap,
    pub heads: BTreeMap<String, Tensor>,
    pub scale: f64,
}

impl Heads {
    pub fn new(
        spec: &[(String, usize)],
        hidden: usize,
        scale: f64,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        let vm = VarMap::new();
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6865_6164);
        let mut heads = BTreeMap::new();
        {
            let mut data = vm.data().lock().expect("varmap lock");
            for (name, n) in spec {
                let t =
                    Tensor::from_vec(gaussian(&mut rng, n * hidden, 0.02), (*n, hidden), device)?;
                let var = Var::from_tensor(&t)?;
                heads.insert(name.clone(), var.as_tensor().clone());
                data.insert(name.clone(), var);
            }
        }
        Ok(Self { vm, heads, scale })
    }

    /// Logits without autograd tracking (evaluation).
    pub fn logits_frozen(&self, name: &str, emb: &Tensor) -> Result<Tensor> {
        let w = self
            .heads
            .get(name)
            .with_context(|| format!("no head {name}"))?;
        Ok((emb.matmul(&l2_normalize(&w.detach())?.t()?)? * self.scale)?)
    }

    pub fn logits(&self, name: &str, emb: &Tensor) -> Result<Tensor> {
        let w = self
            .heads
            .get(name)
            .with_context(|| format!("no head {name}"))?;
        Ok((emb.matmul(&l2_normalize(w)?.t()?)? * self.scale)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes_and_unit_norm() {
        let dev = Device::Cpu;
        let (_vm, bert) = random_encoder(BertConfig::tiny(50), 1, &dev).unwrap();
        let ids = Tensor::new(&[[1u32, 5, 7, 0], [2, 3, 0, 0]], &dev).unwrap();
        let types = ids.zeros_like().unwrap();
        let mask = Tensor::new(&[[1f32, 1., 1., 0.], [1., 1., 0., 0.]], &dev).unwrap();
        let e = bert.embed(&ids, &types, &mask).unwrap();
        assert_eq!(e.dims(), &[2, 32]);
        let n: Vec<f32> = e.sqr().unwrap().sum(1).unwrap().to_vec1().unwrap();
        assert!(n.iter().all(|v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn padding_does_not_change_cls() {
        let dev = Device::Cpu;
        let (_vm, bert) = random_encoder(BertConfig::tiny(50), 2, &dev).unwrap();
        let a = Tensor::new(&[[1u32, 5, 7]], &dev).unwrap();
        let b = Tensor::new(&[[1u32, 5, 7, 0, 0]], &dev).unwrap();
        let ea = bert
            .embed(
                &a,
                &a.zeros_like().unwrap(),
                &Tensor::ones((1, 3), DType::F32, &dev).unwrap(),
            )
            .unwrap();
        let mb = Tensor::new(&[[1f32, 1., 1., 0., 0.]], &dev).unwrap();
        let eb = bert.embed(&b, &b.zeros_like().unwrap(), &mb).unwrap();
        let d: f32 = (ea - eb)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(d < 1e-5, "{d}");
    }

    #[test]
    fn frozen_view_tracks_nothing_but_sees_updates() {
        let dev = Device::Cpu;
        let (vm, bert) = random_encoder(BertConfig::tiny(50), 4, &dev).unwrap();
        let frozen = frozen_view(&vm, BertConfig::tiny(50), &dev).unwrap();
        let ids = Tensor::new(&[[1u32, 5, 7]], &dev).unwrap();
        let (t, m) = (
            ids.zeros_like().unwrap(),
            Tensor::ones((1, 3), DType::F32, &dev).unwrap(),
        );
        let a = frozen.embed(&ids, &t, &m).unwrap();
        assert!(!a.track_op(), "frozen forward must not build a graph");
        assert!(bert.embed(&ids, &t, &m).unwrap().track_op());
        let w = vm.data().lock().unwrap()["embeddings.LayerNorm.bias"].clone();
        w.set(&(w.as_tensor().detach() + 1.0).unwrap()).unwrap();
        let b = frozen.embed(&ids, &t, &m).unwrap();
        let d: f32 = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(d > 1e-4, "frozen view must reflect Var::set");
    }

    #[test]
    fn gradients_reach_every_layernorm() {
        // The whole reason this file exists: candle's fused LN has no backward.
        let dev = Device::Cpu;
        let (vm, bert) = random_encoder(BertConfig::tiny(50), 3, &dev).unwrap();
        let ids = Tensor::new(&[[1u32, 5, 7]], &dev).unwrap();
        let e = bert
            .embed(
                &ids,
                &ids.zeros_like().unwrap(),
                &Tensor::ones((1, 3), DType::F32, &dev).unwrap(),
            )
            .unwrap();
        let loss = e.sum_all().unwrap().sqr().unwrap();
        let grads = loss.backward().unwrap();
        let data = vm.data().lock().unwrap();
        for (name, var) in data.iter() {
            if name.contains("LayerNorm") || name.contains("layer.0.attention.self.query") {
                assert!(grads.get(var.as_tensor()).is_some(), "no grad for {name}");
            }
        }
    }
}
