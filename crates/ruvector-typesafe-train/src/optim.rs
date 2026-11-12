//! AdamW with parameter groups, global-norm gradient clipping and a linear
//! warmup / linear decay schedule (plan Step 3).
//!
//! `candle_nn::AdamW` applies one learning rate and one weight decay to every
//! variable and has no clipping, so the plan's "encoder lr 3e-5, heads lr 1e-3,
//! no decay on biases/LayerNorm, clip 1.0" needs this small implementation.

use anyhow::Result;
use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var};

pub struct Param {
    pub name: String,
    pub var: Var,
    pub lr: f64,
    pub decay: bool,
    m: Tensor,
    v: Tensor,
}

pub struct AdamW {
    pub params: Vec<Param>,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    t: usize,
}

/// HF convention: no weight decay on biases and LayerNorm parameters.
pub fn decays(name: &str) -> bool {
    !(name.ends_with(".bias") || name.contains("LayerNorm"))
}

impl AdamW {
    pub fn new(beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            params: Vec::new(),
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
        }
    }

    /// Register every var of a VarMap at a base learning rate.
    pub fn add_group(&mut self, vm: &candle_nn::VarMap, lr: f64) -> Result<()> {
        let data = vm.data().lock().expect("varmap lock");
        let mut names: Vec<&String> = data.keys().collect();
        names.sort(); // deterministic order
        for name in names {
            let var = data[name].clone();
            self.params.push(Param {
                name: name.clone(),
                m: var.as_tensor().zeros_like()?,
                v: var.as_tensor().zeros_like()?,
                decay: decays(name),
                lr,
                var,
            });
        }
        Ok(())
    }

    /// Global L2 norm of all gradients present in `grads`.
    pub fn grad_norm(&self, grads: &GradStore) -> Result<f64> {
        let mut acc: Option<Tensor> = None;
        for p in &self.params {
            if let Some(g) = grads.get(p.var.as_tensor()) {
                let s = g.sqr()?.sum_all()?;
                acc = Some(match acc {
                    None => s,
                    Some(a) => (a + s)?,
                });
            }
        }
        Ok(match acc {
            None => 0.0,
            Some(a) => f64::from(crate::loss::scalar(&a)?).sqrt(),
        })
    }

    /// One update. `lr_scale` comes from the schedule; gradients are scaled by
    /// `clip / max(norm, clip)` (global-norm clipping). Returns the pre-clip norm.
    pub fn step(&mut self, grads: &GradStore, lr_scale: f64, clip: f64) -> Result<f64> {
        let norm = self.grad_norm(grads)?;
        let gscale = if clip > 0.0 && norm > clip {
            clip / norm
        } else {
            1.0
        };
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for p in &mut self.params {
            let Some(g) = grads.get(p.var.as_tensor()) else {
                continue;
            };
            // Detach everything: gradients carry backprop history, and a moment
            // tensor built from them would chain every step's graph (a leak
            // that OOMed a 16 GB GPU after ~400 steps before this fix).
            let g = (g.detach() * gscale)?;
            let lr = p.lr * lr_scale;
            p.m = ((&p.m * self.beta1)? + (&g * (1.0 - self.beta1))?)?.detach();
            p.v = ((&p.v * self.beta2)? + (g.sqr()? * (1.0 - self.beta2))?)?.detach();
            let mhat = (&p.m / bc1)?;
            let vhat = (&p.v / bc2)?;
            let upd = (mhat / (vhat.sqrt()? + self.eps)?)?;
            let mut theta = p.var.as_tensor().detach();
            if p.decay && self.weight_decay > 0.0 {
                theta = (theta * (1.0 - lr * self.weight_decay))?;
            }
            p.var.set(&(theta - (upd * lr)?)?)?;
        }
        Ok(norm)
    }
}

/// Linear warmup to 1 over `warmup` steps, then linear decay to 0 at `total`.
pub fn lr_scale(step: usize, warmup: usize, total: usize) -> f64 {
    if warmup > 0 && step < warmup {
        return (step + 1) as f64 / warmup as f64;
    }
    let rest = total.saturating_sub(warmup).max(1);
    (1.0 - (step - warmup.min(step)) as f64 / rest as f64).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn schedule_shape() {
        assert!((lr_scale(0, 10, 100) - 0.1).abs() < 1e-12);
        assert!((lr_scale(9, 10, 100) - 1.0).abs() < 1e-12);
        assert!((lr_scale(10, 10, 100) - 1.0).abs() < 1e-12);
        assert!(lr_scale(55, 10, 100) < 0.51 && lr_scale(55, 10, 100) > 0.49);
        assert_eq!(lr_scale(100, 10, 100), 0.0);
    }

    #[test]
    fn decay_exemptions() {
        assert!(decays("encoder.layer.0.attention.self.query.weight"));
        assert!(!decays("encoder.layer.0.attention.self.query.bias"));
        assert!(!decays("embeddings.LayerNorm.weight"));
    }

    #[test]
    fn minimizes_a_quadratic_and_clips() {
        let vm = candle_nn::VarMap::new();
        let x = vm
            .get(
                (3,),
                "w.weight",
                candle_nn::Init::Const(5.0),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.0);
        opt.add_group(&vm, 0.1).unwrap();
        let mut first_norm = 0.0;
        for i in 0..300 {
            let loss = x.sqr().unwrap().sum_all().unwrap();
            let g = loss.backward().unwrap();
            let n = opt.step(&g, 1.0, 1.0).unwrap();
            if i == 0 {
                first_norm = n;
            }
        }
        let v: Vec<f32> = x.to_vec1().unwrap();
        assert!(first_norm > 1.0, "pre-clip norm reported");
        assert!(v.iter().all(|a| a.abs() < 0.2), "{v:?}");
    }

    #[test]
    fn moments_hold_no_autograd_history() {
        // Regression: undetached moments chained every step's graph and OOMed the GPU.
        let vm = candle_nn::VarMap::new();
        let x = vm
            .get(
                (4,),
                "w.weight",
                candle_nn::Init::Const(1.0),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.01);
        opt.add_group(&vm, 0.1).unwrap();
        for _ in 0..3 {
            let g = x.sqr().unwrap().sum_all().unwrap().backward().unwrap();
            opt.step(&g, 1.0, 1.0).unwrap();
        }
        for p in &opt.params {
            assert!(
                !p.m.track_op() && !p.v.track_op(),
                "{} keeps a graph",
                p.name
            );
        }
    }
}
