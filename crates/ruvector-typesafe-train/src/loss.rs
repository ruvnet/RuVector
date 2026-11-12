//! Losses (plan Step 3): supervised contrastive over the batch, soft-target
//! cross-entropy (covers label smoothing, ambiguous soft targets, class
//! weights and outlier exposure to the uniform distribution).

use anyhow::{bail, Result};
use candle_core::{Device, Tensor, D};

/// Numerically stable log-softmax over the last dim (max is detached; the
/// result is mathematically unchanged, and the graph stays differentiable).
pub fn log_softmax(x: &Tensor) -> candle_core::Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?.detach();
    let z = x.broadcast_sub(&m)?;
    let lse = z.exp()?.sum_keepdim(D::Minus1)?.log()?;
    z.broadcast_sub(&lse)
}

/// Supervised contrastive loss (Khosla et al. 2020) on L2-normalized `emb`.
/// Anchors without any positive are skipped. Returns None if no anchor has one.
pub fn supcon(emb: &Tensor, labels: &[u32], tau: f64) -> Result<Option<Tensor>> {
    let n = labels.len();
    if emb.dim(0)? != n {
        bail!("supcon: {} embeddings vs {n} labels", emb.dim(0)?);
    }
    let dev = emb.device();
    let mut pos = vec![0f32; n * n];
    let mut diag = vec![0f32; n * n];
    let mut anchor = vec![0f32; n];
    for i in 0..n {
        diag[i * n + i] = -1e9;
        let mut c = 0f32;
        for j in 0..n {
            if i != j && labels[i] == labels[j] {
                pos[i * n + j] = 1.0;
                c += 1.0;
            }
        }
        if c > 0.0 {
            for j in 0..n {
                pos[i * n + j] /= c;
            }
            anchor[i] = 1.0;
        }
    }
    let n_anchor: f32 = anchor.iter().sum();
    if n_anchor == 0.0 {
        return Ok(None);
    }
    let sim = ((emb.matmul(&emb.t()?)? / tau)? + Tensor::from_vec(diag, (n, n), dev)?)?;
    let logp = log_softmax(&sim)?;
    let pos = Tensor::from_vec(pos, (n, n), dev)?;
    let per_anchor = (logp * pos)?.sum(1)?; // mean log-prob over positives
    let anchor = Tensor::from_vec(anchor, n, dev)?;
    let loss = ((per_anchor * anchor)?.sum_all()? / f64::from(-n_anchor))?;
    Ok(Some(loss))
}

/// Weighted soft cross-entropy: Σ_i w_i · (−Σ_c t_ic log p_ic) / Σ_i w_i.
/// `targets` are probability rows; `weights` per row (class weights go here).
pub fn soft_ce(logits: &Tensor, targets: Vec<f32>, weights: Vec<f32>) -> Result<Tensor> {
    let (n, c) = logits.dims2()?;
    if targets.len() != n * c || weights.len() != n {
        bail!(
            "soft_ce: shape mismatch ({n}x{c}, {} targets, {} weights)",
            targets.len(),
            weights.len()
        );
    }
    let wsum: f32 = weights.iter().sum();
    if wsum <= 0.0 {
        bail!("soft_ce: zero total weight");
    }
    let dev = logits.device();
    let t = Tensor::from_vec(targets, (n, c), dev)?;
    let w = Tensor::from_vec(weights, n, dev)?;
    let nll = (log_softmax(logits)? * t)?.sum(1)?.neg()?;
    Ok(((nll * w)?.sum_all()? / f64::from(wsum))?)
}

/// One-hot with label smoothing ε: (1−ε)·onehot + ε/C.
pub fn smoothed(idx: usize, c: usize, eps: f64) -> Vec<f32> {
    let base = (eps / c as f64) as f32;
    let mut v = vec![base; c];
    v[idx] += (1.0 - eps) as f32;
    v
}

/// Inverse-frequency class weights normalized to mean 1 (plan: frustration).
pub fn inverse_frequency(counts: &[usize]) -> Vec<f32> {
    let inv: Vec<f64> = counts
        .iter()
        .map(|&n| if n == 0 { 0.0 } else { 1.0 / n as f64 })
        .collect();
    let nz = inv.iter().filter(|v| **v > 0.0).count().max(1) as f64;
    let mean = inv.iter().sum::<f64>() / nz;
    inv.iter().map(|v| (v / mean) as f32).collect()
}

pub fn scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_device(&Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?
        .to_scalar::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::l2_normalize;

    #[test]
    fn soft_ce_matches_hand_computation() {
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[2f32, 0.0], [0.0, 0.0]], &dev).unwrap();
        let l = soft_ce(&logits, vec![1., 0., 0.5, 0.5], vec![1., 1.]).unwrap();
        let p0 = (2f32.exp() / (2f32.exp() + 1.0)).ln();
        let want = (-p0 + 2f32.ln()) / 2.0;
        assert!((scalar(&l).unwrap() - want).abs() < 1e-5);
    }

    #[test]
    fn uniform_target_is_minimized_by_flat_logits() {
        let dev = Device::Cpu;
        let flat = Tensor::zeros((1, 4), candle_core::DType::F32, &dev).unwrap();
        let peaked = Tensor::new(&[[5f32, 0., 0., 0.]], &dev).unwrap();
        let u = vec![0.25f32; 4];
        let a = scalar(&soft_ce(&flat, u.clone(), vec![1.]).unwrap()).unwrap();
        let b = scalar(&soft_ce(&peaked, u, vec![1.]).unwrap()).unwrap();
        assert!(a < b && (a - 4f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn supcon_prefers_clustered_embeddings() {
        let dev = Device::Cpu;
        let clustered = l2_normalize(
            &Tensor::new(&[[1f32, 0.01], [1., -0.01], [0.01, 1.], [-0.01, 1.]], &dev).unwrap(),
        )
        .unwrap();
        let mixed = l2_normalize(
            &Tensor::new(&[[1f32, 0.01], [0.01, 1.], [1., -0.01], [-0.01, 1.]], &dev).unwrap(),
        )
        .unwrap();
        let labels = [0u32, 0, 1, 1];
        let a = scalar(&supcon(&clustered, &labels, 0.1).unwrap().unwrap()).unwrap();
        let b = scalar(&supcon(&mixed, &labels, 0.1).unwrap().unwrap()).unwrap();
        assert!(a < b, "{a} vs {b}");
        assert!(supcon(&clustered, &[0, 1, 2, 3], 0.1).unwrap().is_none());
    }

    #[test]
    fn smoothing_and_weights() {
        let s = smoothed(1, 4, 0.1);
        assert!((s.iter().sum::<f32>() - 1.0).abs() < 1e-6 && (s[1] - 0.925).abs() < 1e-6);
        let w = inverse_frequency(&[82, 39, 15]);
        assert!((w.iter().sum::<f32>() / 3.0 - 1.0).abs() < 1e-5 && w[2] > w[1] && w[1] > w[0]);
    }
}
