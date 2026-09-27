//! HolE front-end of the 1-N kernel, through HolE's frequency view
//! (ADR-002 §2; HolE ≡ ComplEx, Hayashi & Shimbo 2017).
//!
//! HolE's exact inner-product factorisation `score = index(o) · query(r, s)`
//! (see `scorer/hole.rs`) makes 1-N a GEMM over the weighted half-spectrum
//! index table: `E[n] = HolE::index_vector(e_n)` (width `d + 2`, identical to
//! [`crate::BatchScorer`]'s rows) and `Q[b] = HolE::query_vector(r, anchor,
//! side)`. Parameters stay the real `d`-vectors; the kernel's gradients with
//! respect to `E` and `Q` are pulled back through the adjoints of those two
//! linear-in-each-argument maps.
//!
//! With half bins `m = 0..=d/2`, weights `w_m` (1 at DC/Nyquist, √2 inside)
//! and `x̂ = F(x)`:
//!
//! ```text
//! index:  E_n = [w·Re ê ; w·Im ê]            → H_m = w_m (g_re,m + i g_im,m)
//! query:  Q_b = (1/d)[w·Re p ; w·Im p]       → P_m = (w_m/d)(g_re,m + i g_im,m)
//!   tail  p = r̂ ⊙ ŝ :  ∂ŝ = conj(r̂)⊙P   ∂r̂ = conj(ŝ)⊙P
//!   head  p = conj(r̂) ⊙ ô :  ∂ô = r̂⊙P   ∂r̂ = ô⊙conj(P)
//! ```
//!
//! and a half-spectrum gradient `H` maps back to a real vector as
//! `∂x_n = Re Σ_{m≤d/2} H_m e^{+2πi mn/d}` — one unnormalised inverse FFT of
//! `H` zero-padded to `d` bins. Cost per batch: one FFT per entity row plus
//! four per query row, O((N + B)·d log d), negligible next to the GEMMs.

use super::{one_n_softmax_ce, OneNQuery, Reduction, Workspace};
use crate::scorer::fft::C;
use crate::{HolE, KgeError, Result, Scorer, Side, Tables};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Reusable HolE kernel state: FFT plans plus every batch-sized buffer.
pub struct HolEKernel {
    d: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    index: Vec<f32>,
    grad_index: Vec<f32>,
    q: Vec<f32>,
    gq: Vec<f32>,
    row_grads: Vec<f32>,
    targets: Vec<u32>,
    core: Workspace,
}

impl HolEKernel {
    /// Plans for dimension `d` (must match the `HolE` scorer; even).
    pub fn new(d: usize) -> Result<Self> {
        if d == 0 || !d.is_multiple_of(2) {
            return Err(KgeError::Invalid(format!(
                "hole kernel: d must be even and non-zero, got {d}"
            )));
        }
        let mut planner = FftPlanner::<f32>::new();
        Ok(Self {
            d,
            fwd: planner.plan_fft_forward(d),
            inv: planner.plan_fft_inverse(d),
            index: Vec::new(),
            grad_index: Vec::new(),
            q: Vec::new(),
            gq: Vec::new(),
            row_grads: Vec::new(),
            targets: Vec::new(),
            core: Workspace::default(),
        })
    }

    #[inline]
    fn weight(&self, m: usize) -> f32 {
        if m == 0 || m == self.d / 2 {
            1.0
        } else {
            std::f32::consts::SQRT_2
        }
    }

    fn spectrum(&self, x: &[f32]) -> Vec<C> {
        let mut buf: Vec<C> = x.iter().map(|&v| C::new(v, 0.0)).collect();
        self.fwd.process(&mut buf);
        buf
    }

    /// `out[n] = Re Σ_{m≤d/2} h[m] e^{+2πi mn/d}`, *added* into `out`.
    fn pull_back_add(&self, h: &mut Vec<C>, out: &mut [f32]) {
        h.resize(self.d, C::new(0.0, 0.0));
        self.inv.process(h);
        for (o, c) in out.iter_mut().zip(h.iter()) {
            *o += c.re;
        }
    }

    /// One batched 1-N softmax-CE step for HolE over `tables`.
    ///
    /// Writes dense `grad_entities` (`N×d`) and `grad_relations` (`R×d`),
    /// both **overwritten**, and returns the reduced loss. With both a
    /// `Side::Tail` row `(s, r, o)` and a `Side::Head` row `(o, r, s)` per
    /// positive, `Reduction::Sum` reproduces the per-triple
    /// `one_vs_all_step` loss and gradients.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        hole: &HolE,
        tables: &Tables,
        queries: &[OneNQuery],
        reduction: Reduction,
        grad_entities: &mut [f32],
        grad_relations: &mut [f32],
    ) -> Result<f32> {
        let d = self.d;
        if hole.dims() != d || tables.dims() != d {
            return Err(KgeError::Dims {
                expected: d,
                got: tables.dims(),
            });
        }
        let (n, nr) = (tables.num_entities(), tables.num_relations());
        let idim = d + 2;
        let h = d / 2 + 1;
        for (buf, want) in [(&*grad_entities, n * d), (&*grad_relations, nr * d)] {
            if buf.len() != want {
                return Err(KgeError::Dims {
                    expected: want,
                    got: buf.len(),
                });
            }
        }
        for qy in queries {
            tables.entity(qy.anchor)?;
            tables.relation(qy.relation)?;
        }
        let b = queries.len();

        // 1. Index table (the frequency view of E) — parallel over rows.
        self.index.resize(n * idim, 0.0);
        let ents = tables.entities_raw();
        for_each_row(&mut self.index, idim, |i, row| {
            row.copy_from_slice(&hole.index_vector(&ents[i * d..(i + 1) * d]));
        });

        // 2. Query rows.
        self.q.resize(b * idim, 0.0);
        self.gq.resize(b * idim, 0.0);
        self.targets.clear();
        for (row, qy) in self.q.chunks_mut(idim).zip(queries) {
            let r = tables.relation(qy.relation)?;
            let a = tables.entity(qy.anchor)?;
            row.copy_from_slice(&hole.query_vector(r, a, qy.side));
            self.targets.push(qy.target);
        }

        // 3. The GEMM kernel.
        self.grad_index.resize(n * idim, 0.0);
        let loss = one_n_softmax_ce(
            &self.q,
            &self.index,
            idim,
            &self.targets,
            reduction,
            &mut self.core,
            &mut self.gq,
            &mut self.grad_index,
        )?;

        // 4. Entity-table pull-back: grad_e = Re IFFT(w ⊙ G_idx), per row.
        let this = &*self;
        for_each_row(grad_entities, d, |i, out| {
            let g = &this.grad_index[i * idim..(i + 1) * idim];
            let mut hv: Vec<C> = (0..h)
                .map(|m| C::new(g[m], g[h + m]) * this.weight(m))
                .collect();
            out.fill(0.0);
            this.pull_back_add(&mut hv, out);
        });

        // 5. Query pull-back to (anchor, relation), per row in parallel into
        //    `row_grads` ([anchor d | relation d] per row) ...
        let mut row_grads = std::mem::take(&mut self.row_grads);
        row_grads.resize(b * 2 * d, 0.0);
        let inv_d = 1.0 / d as f32;
        let this = &*self;
        for_each_row(&mut row_grads, 2 * d, |i, out| {
            let qy = queries[i];
            let rhat = this.spectrum(tables.relation(qy.relation).expect("checked"));
            let ahat = this.spectrum(tables.entity(qy.anchor).expect("checked"));
            let g = &this.gq[i * idim..(i + 1) * idim];
            let p = |m: usize| C::new(g[m], g[h + m]) * (this.weight(m) * inv_d);
            let (mut ga, mut gr): (Vec<C>, Vec<C>) = (0..h)
                .map(|m| {
                    let pm = p(m);
                    match qy.side {
                        Side::Tail => (rhat[m].conj() * pm, ahat[m].conj() * pm),
                        Side::Head => (rhat[m] * pm, ahat[m] * pm.conj()),
                    }
                })
                .unzip();
            out.fill(0.0);
            let (oa, or) = out.split_at_mut(d);
            this.pull_back_add(&mut ga, oa);
            this.pull_back_add(&mut gr, or);
        });
        // ... then scattered sequentially in row order (deterministic).
        grad_relations.fill(0.0);
        for (rg, qy) in row_grads.chunks(2 * d).zip(queries) {
            let (a, r) = (qy.anchor as usize, qy.relation as usize);
            for (dst, &v) in grad_entities[a * d..(a + 1) * d].iter_mut().zip(&rg[..d]) {
                *dst += v;
            }
            for (dst, &v) in grad_relations[r * d..(r + 1) * d].iter_mut().zip(&rg[d..]) {
                *dst += v;
            }
        }
        self.row_grads = row_grads;
        Ok(loss)
    }
}

/// Apply `f(row_index, row)` to every `row_len`-wide row of `buf` — rows are
/// independent, so this is deterministic in or out of the `parallel` feature.
fn for_each_row<F>(buf: &mut [f32], row_len: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }
    #[cfg(not(feature = "parallel"))]
    for (i, row) in buf.chunks_mut(row_len).enumerate() {
        f(i, row);
    }
}
