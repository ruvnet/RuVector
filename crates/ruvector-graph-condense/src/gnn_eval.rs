//! Minimal 2-layer GCN — **for evaluating condensation quality only**.
//!
//! The graph-condensation literature is benchmarked by one protocol: train a GNN
//! on the condensed graph, then test it on the *original* graph's held-out nodes,
//! and report `accuracy(condensed) / accuracy(full)` ("retention"). Structural
//! proxies (cut preservation, purity) do not substitute for it. This module is a
//! self-contained, dependency-free (plain `f64`) reference GCN so the crate can
//! report that number honestly.
//!
//! It is deliberately small: symmetric-normalised propagation `Â = D̃^{-1/2}
//! (A+I) D̃^{-1/2}`, two graph-conv layers with ReLU, softmax cross-entropy,
//! Adam, **analytic backprop** (gradient-checked in tests). Weights are
//! graph-agnostic, so a GCN trained on the small condensed graph can be applied
//! to the full graph at test time — exactly the condensation eval protocol.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Symmetric-normalised adjacency (with self-loops) in CSR form.
pub struct GcnGraph {
    n: usize,
    off: Vec<usize>,
    nbr: Vec<(usize, f64)>,
}

impl GcnGraph {
    /// Build `Â` from an undirected edge list `(i, j, w)` over `n` nodes.
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        // Degrees including the self-loop (A + I).
        let mut deg = vec![1f64; n];
        for &(i, j, w) in edges {
            deg[i] += w;
            deg[j] += w;
        }
        let inv_sqrt: Vec<f64> = deg.iter().map(|d| 1.0 / d.sqrt()).collect();

        // Count entries per row (neighbours both directions + self).
        let mut cnt = vec![1usize; n];
        for &(i, j, _) in edges {
            cnt[i] += 1;
            cnt[j] += 1;
        }
        let mut off = vec![0usize; n + 1];
        for i in 0..n {
            off[i + 1] = off[i] + cnt[i];
        }
        let mut cursor = off[..n].to_vec();
        let mut nbr = vec![(0usize, 0f64); off[n]];
        // Self-loops first.
        for i in 0..n {
            nbr[cursor[i]] = (i, inv_sqrt[i] * inv_sqrt[i]);
            cursor[i] += 1;
        }
        for &(i, j, w) in edges {
            let a = w * inv_sqrt[i] * inv_sqrt[j];
            nbr[cursor[i]] = (j, a);
            cursor[i] += 1;
            nbr[cursor[j]] = (i, a);
            cursor[j] += 1;
        }
        Self { n, off, nbr }
    }

    /// `Â · M` where `M` is row-major `n × d`.
    fn spmm(&self, m: &[f64], d: usize) -> Vec<f64> {
        let mut out = vec![0f64; self.n * d];
        for i in 0..self.n {
            let orow = &mut out[i * d..(i + 1) * d];
            for e in self.off[i]..self.off[i + 1] {
                let (j, w) = self.nbr[e];
                let mrow = &m[j * d..(j + 1) * d];
                for c in 0..d {
                    orow[c] += w * mrow[c];
                }
            }
        }
        out
    }
}

/// A trained 2-layer GCN classifier.
pub struct Gcn {
    w1: Vec<f64>, // f x h
    w2: Vec<f64>, // h x c
    f: usize,
    h: usize,
    c: usize,
}

/// Training hyper-parameters.
#[derive(Debug, Clone)]
pub struct GcnConfig {
    /// Hidden width.
    pub hidden: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Training epochs.
    pub epochs: usize,
    /// L2 weight decay.
    pub weight_decay: f64,
    /// Seed for weight init.
    pub seed: u64,
}

impl Default for GcnConfig {
    fn default() -> Self {
        Self {
            hidden: 16,
            learning_rate: 0.01,
            epochs: 200,
            weight_decay: 5e-4,
            seed: 0x6CD,
        }
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn mm(a: &[f64], b: &[f64], n: usize, p: usize, q: usize) -> Vec<f64> {
    // (n×p) · (p×q)
    let mut out = vec![0f64; n * q];
    for i in 0..n {
        for k in 0..p {
            let aik = a[i * p + k];
            if aik != 0.0 {
                for j in 0..q {
                    out[i * q + j] += aik * b[k * q + j];
                }
            }
        }
    }
    out
}

fn mm_at(a: &[f64], b: &[f64], n: usize, p: usize, q: usize) -> Vec<f64> {
    // (n×p)ᵀ · (n×q) = (p×q)
    let mut out = vec![0f64; p * q];
    for i in 0..n {
        for k in 0..p {
            let aik = a[i * p + k];
            if aik != 0.0 {
                for j in 0..q {
                    out[k * q + j] += aik * b[i * q + j];
                }
            }
        }
    }
    out
}

fn mm_bt(a: &[f64], b: &[f64], n: usize, q: usize, p: usize) -> Vec<f64> {
    // (n×q) · (p×q)ᵀ = (n×p)
    let mut out = vec![0f64; n * p];
    for i in 0..n {
        for k in 0..p {
            let mut acc = 0f64;
            for j in 0..q {
                acc += a[i * q + j] * b[k * q + j];
            }
            out[i * p + k] = acc;
        }
    }
    out
}

/// Forward intermediates kept for backprop.
struct Fwd {
    ax: Vec<f64>,
    h1: Vec<f64>,
    ar: Vec<f64>,
    probs: Vec<f64>,
}

impl Gcn {
    fn forward(&self, g: &GcnGraph, x: &[f64]) -> Fwd {
        let n = g.n;
        let ax = g.spmm(x, self.f);
        let h1 = mm(&ax, &self.w1, n, self.f, self.h);
        let r: Vec<f64> = h1.iter().map(|&v| relu(v)).collect();
        let ar = g.spmm(&r, self.h);
        let o = mm(&ar, &self.w2, n, self.h, self.c);
        let probs = softmax_rows(&o, n, self.c);
        Fwd { ax, h1, ar, probs }
    }

    /// Predicted class per node.
    pub fn predict(&self, g: &GcnGraph, x: &[f64]) -> Vec<usize> {
        let fwd = self.forward(g, x);
        (0..g.n)
            .map(|i| argmax(&fwd.probs[i * self.c..(i + 1) * self.c]))
            .collect()
    }

    /// Train on `(g, x, labels)` over the nodes in `mask`. Returns the trained
    /// classifier.
    pub fn train(
        cfg: &GcnConfig,
        g: &GcnGraph,
        x: &[f64],
        f: usize,
        labels: &[usize],
        c: usize,
        mask: &[usize],
    ) -> Gcn {
        let h = cfg.hidden;
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        // Xavier-ish init.
        let s1 = (6.0 / (f + h) as f64).sqrt();
        let s2 = (6.0 / (h + c) as f64).sqrt();
        let mut model = Gcn {
            w1: (0..f * h).map(|_| rng.gen_range(-s1..s1)).collect(),
            w2: (0..h * c).map(|_| rng.gen_range(-s2..s2)).collect(),
            f,
            h,
            c,
        };
        // Adam state.
        let (mut m1, mut v1) = (vec![0f64; f * h], vec![0f64; f * h]);
        let (mut m2, mut v2) = (vec![0f64; h * c], vec![0f64; h * c]);
        for t in 1..=cfg.epochs {
            let (dw1, dw2) = model.grads(g, x, labels, mask);
            adam_step(&mut model.w1, &dw1, &mut m1, &mut v1, cfg, t);
            adam_step(&mut model.w2, &dw2, &mut m2, &mut v2, cfg, t);
        }
        model
    }

    /// Analytic gradients of masked softmax-CE (+ L2) w.r.t. `w1`, `w2`.
    fn grads(
        &self,
        g: &GcnGraph,
        x: &[f64],
        labels: &[usize],
        mask: &[usize],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = g.n;
        let fwd = self.forward(g, x);
        let inv = 1.0 / mask.len().max(1) as f64;
        // dO = (P - onehot)/|mask| on masked rows, else 0.
        let mut d_o = vec![0f64; n * self.c];
        for &i in mask {
            for cc in 0..self.c {
                d_o[i * self.c + cc] = fwd.probs[i * self.c + cc] * inv;
            }
            d_o[i * self.c + labels[i]] -= inv;
        }
        let dw2 = mm_at(&fwd.ar, &d_o, n, self.h, self.c);
        let d_ar = mm_bt(&d_o, &self.w2, n, self.c, self.h);
        let d_r = g.spmm(&d_ar, self.h);
        let mut d_h1 = vec![0f64; n * self.h];
        for idx in 0..n * self.h {
            d_h1[idx] = if fwd.h1[idx] > 0.0 { d_r[idx] } else { 0.0 };
        }
        let dw1 = mm_at(&fwd.ax, &d_h1, n, self.f, self.h);
        // L2 weight decay is applied in `adam_step`, so gradients here are the
        // pure cross-entropy gradients (which the gradient check verifies).
        (dw1, dw2)
    }
}

fn adam_step(w: &mut [f64], grad: &[f64], m: &mut [f64], v: &mut [f64], cfg: &GcnConfig, t: usize) {
    let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);
    for i in 0..w.len() {
        let g = grad[i] + cfg.weight_decay * w[i];
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        w[i] -= cfg.learning_rate * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
    }
}

fn softmax_rows(o: &[f64], n: usize, c: usize) -> Vec<f64> {
    let mut p = vec![0f64; n * c];
    for i in 0..n {
        let row = &o[i * c..(i + 1) * c];
        let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0f64;
        for j in 0..c {
            let e = (row[j] - mx).exp();
            p[i * c + j] = e;
            s += e;
        }
        for j in 0..c {
            p[i * c + j] /= s;
        }
    }
    p
}

fn argmax(row: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// Accuracy over the nodes in `mask`.
pub fn accuracy(preds: &[usize], labels: &[usize], mask: &[usize]) -> f64 {
    if mask.is_empty() {
        return 0.0;
    }
    let correct = mask.iter().filter(|&&i| preds[i] == labels[i]).count();
    correct as f64 / mask.len() as f64
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index-heavy numeric test code
mod tests {
    use super::*;

    fn ring(n: usize) -> (GcnGraph, Vec<(usize, usize, f64)>) {
        let edges: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, (i + 1) % n, 1.0)).collect();
        (GcnGraph::from_edges(n, &edges), edges)
    }

    #[test]
    fn gradient_matches_finite_differences() {
        // The decisive correctness test for the GCN backprop.
        let n = 6;
        let (g, _e) = ring(n);
        let f = 3;
        let c = 2;
        let h = 4;
        let mut rng = StdRng::seed_from_u64(1);
        let x: Vec<f64> = (0..n * f).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let labels: Vec<usize> = (0..n).map(|i| i % c).collect();
        let mask: Vec<usize> = (0..n).collect();
        let model = Gcn {
            w1: (0..f * h).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            w2: (0..h * c).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            f,
            h,
            c,
        };
        let loss = |m: &Gcn| -> f64 {
            let fwd = m.forward(&g, &x);
            let inv = 1.0 / mask.len() as f64;
            let mut l = 0f64;
            for &i in &mask {
                l -= (fwd.probs[i * c + labels[i]].max(1e-12)).ln() * inv;
            }
            l
        };
        let (dw1, dw2) = model.grads(&g, &x, &labels, &mask);
        let hh = 1e-6;
        let mut max_err = 0f64;
        for idx in 0..f * h {
            let mut mp = Gcn {
                w1: model.w1.clone(),
                w2: model.w2.clone(),
                f,
                h,
                c,
            };
            mp.w1[idx] += hh;
            let mut mm_ = Gcn {
                w1: model.w1.clone(),
                w2: model.w2.clone(),
                f,
                h,
                c,
            };
            mm_.w1[idx] -= hh;
            let num = (loss(&mp) - loss(&mm_)) / (2.0 * hh);
            max_err = max_err.max((num - dw1[idx]).abs());
        }
        for idx in 0..h * c {
            let mut mp = Gcn {
                w1: model.w1.clone(),
                w2: model.w2.clone(),
                f,
                h,
                c,
            };
            mp.w2[idx] += hh;
            let mut mm_ = Gcn {
                w1: model.w1.clone(),
                w2: model.w2.clone(),
                f,
                h,
                c,
            };
            mm_.w2[idx] -= hh;
            let num = (loss(&mp) - loss(&mm_)) / (2.0 * hh);
            max_err = max_err.max((num - dw2[idx]).abs());
        }
        assert!(max_err < 1e-6, "GCN grad mismatch: {max_err}");
    }

    #[test]
    fn learns_a_separable_task() {
        // Two cliques, distinct features per class -> GCN should fit train set.
        let n = 20;
        let mut edges = Vec::new();
        for i in 0..10 {
            for j in (i + 1)..10 {
                edges.push((i, j, 1.0));
            }
        }
        for i in 10..20 {
            for j in (i + 1)..20 {
                edges.push((i, j, 1.0));
            }
        }
        let g = GcnGraph::from_edges(n, &edges);
        let f = 2;
        let c = 2;
        let mut x = vec![0f64; n * f];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let cls = i / 10;
            labels[i] = cls;
            x[i * f + cls] = 1.0;
        }
        let mask: Vec<usize> = (0..n).collect();
        let model = Gcn::train(&GcnConfig::default(), &g, &x, f, &labels, c, &mask);
        let preds = model.predict(&g, &x);
        assert!(accuracy(&preds, &labels, &mask) > 0.95);
    }
}
