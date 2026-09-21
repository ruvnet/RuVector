//! HolE — holographic embeddings via circular correlation (ADR-002 §1, §2).
//!
//! Score of a triple `(s, r, o)` is `r · (e_s ⋆ e_o)`, circular correlation.
//! With `ŝ = F(e_s)`, `ô = F(e_o)`, `r̂ = F(r)`, Parseval gives
//!
//! ```text
//! score = (1/d) · Re⟨ r̂ ⊙ ŝ , ô ⟩
//!       = (1/d) · ( Re(r̂⊙ŝ)·Re(ô) + Im(r̂⊙ŝ)·Im(ô) )
//! ```
//!
//! a plain real dot product — the seam that makes candidate retrieval a
//! `DotProduct`/MIPS problem (ADR-001 §3).
//!
//! ## Index representation — half-spectrum, `d + 2` reals
//!
//! `F(e)` of a real vector is Hermitian: bin `d−m` is the conjugate of bin `m`,
//! so only bins `0..=d/2` (that is `d/2 + 1` complex numbers) are independent.
//! We store those, real parts then imaginary parts, giving
//! `2·(d/2 + 1) = d + 2` reals — the "half the parameters" storage claim of
//! ADR-001, not the naïve `2d` full spectrum. `index_dims() = d + 2`.
//!
//! For a plain dot product over the *half* spectrum to equal the full-spectrum
//! score, the interior bins must be counted twice (bins `m` and `d−m`
//! contribute equal terms). We fold that into a per-bin weight applied to
//! **both** the index vector and the query vector:
//!
//! ```text
//! w(m) = 1           for m = 0 (DC) and m = d/2 (Nyquist)
//! w(m) = sqrt(2)     for 1 ≤ m ≤ d/2 − 1  (interior)
//! ```
//!
//! so the product carries `w(m)² = 2` on interior bins and `1` on DC/Nyquist,
//! reproducing `term(0) + term(d/2) + 2·Σ_interior term(m)` = the full sum.
//! `Im` of the DC and Nyquist bins is structurally zero, so two of the `d + 2`
//! slots are always `0.0`; keeping the layout regular is worth those two bytes.
//!
//! ## Query vectors — both sides derived
//!
//! `index_vector(open) · query_vector(r, anchor, side) = score`, exactly.
//!
//! - **Tail open** `(s, r, ?)`, `anchor = s`, open = `o`:
//!   `q = (1/d) · half([Re(r̂⊙ŝ); Im(r̂⊙ŝ)])`, dotted with `index_vector(o)`.
//! - **Head open** `(?, r, o)`, `anchor = o`, open = `s`:
//!   from `score = (1/d) Σ_m Re(ŝ[m] · r̂[m] · conj(ô[m]))`, grouping the
//!   variable `ŝ` gives `q = (1/d) · half([Re(w); Im(w)])` with
//!   `w = conj(r̂) ⊙ ô` — note the **conjugate on `r̂`** and that this equals
//!   `[Re(r̂ conj(ô)); −Im(r̂ conj(ô))]`, the sign-flipped imaginary half.
//!
//! Because HolE's index is an exact inner-product factorisation, `ann_exact()`
//! keeps its default `true`.

use crate::scorer::fft::{FftPlan, C};
use crate::scorer::Scorer;
use crate::{Differentiable, Result, Side};

/// Circular-correlation ("holographic") scorer. `dims` must be even.
pub struct HolE {
    dims: usize,
    plan: FftPlan,
    id: String,
}

impl HolE {
    /// Build a HolE scorer for embedding dimension `dims` (must be even).
    pub fn new(dims: usize) -> Result<Self> {
        let plan = FftPlan::new(dims)?;
        Ok(Self {
            dims,
            plan,
            id: format!("hole-fft@d{dims}"),
        })
    }

    /// Number of independent (half-spectrum) bins: `d/2 + 1`.
    #[inline]
    fn half_bins(&self) -> usize {
        self.dims / 2 + 1
    }

    /// Hermitian doubling weight for bin `m` (see module docs).
    #[inline]
    fn weight(&self, m: usize) -> f32 {
        if m == 0 || m == self.dims / 2 {
            1.0
        } else {
            std::f32::consts::SQRT_2
        }
    }

    /// Pack a full `d`-bin spectrum into the weighted half-spectrum layout
    /// `[w·Re(0..=d/2)] ++ [w·Im(0..=d/2)]`, optionally scaled by `scale`.
    fn pack_half(&self, spec: &[C], scale: f32) -> Vec<f32> {
        let h = self.half_bins();
        let mut out = Vec::with_capacity(2 * h);
        for (m, c) in spec.iter().take(h).enumerate() {
            out.push(self.weight(m) * c.re * scale);
        }
        for (m, c) in spec.iter().take(h).enumerate() {
            out.push(self.weight(m) * c.im * scale);
        }
        out
    }
}

impl Scorer for HolE {
    fn dims(&self) -> usize {
        self.dims
    }

    fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32 {
        assert_eq!(s.len(), self.dims, "score: s length must equal dims");
        assert_eq!(r.len(), self.dims, "score: r length must equal dims");
        assert_eq!(o.len(), self.dims, "score: o length must equal dims");
        // r · (e_s ⋆ e_o) — the HolE definition, via the tested FFT primitive.
        let corr = self.plan.circular_correlation(s, o);
        r.iter().zip(&corr).map(|(a, b)| a * b).sum()
    }

    fn index_vector(&self, e: &[f32]) -> Vec<f32> {
        assert_eq!(e.len(), self.dims, "index_vector: length must equal dims");
        let spec = self.plan.forward(e);
        self.pack_half(&spec, 1.0)
    }

    fn index_dims(&self) -> usize {
        self.dims + 2
    }

    fn query_vector(&self, r: &[f32], anchor: &[f32], side: Side) -> Vec<f32> {
        assert_eq!(r.len(), self.dims, "query_vector: r length must equal dims");
        assert_eq!(
            anchor.len(),
            self.dims,
            "query_vector: anchor length must equal dims"
        );
        let rhat = self.plan.forward(r);
        let ahat = self.plan.forward(anchor);
        let inv_d = 1.0 / self.dims as f32;
        let w: Vec<C> = match side {
            // Tail open: anchor = s, w = r̂ ⊙ ŝ.
            Side::Tail => rhat.iter().zip(&ahat).map(|(x, y)| x * y).collect(),
            // Head open: anchor = o, w = conj(r̂) ⊙ ô.
            Side::Head => rhat.iter().zip(&ahat).map(|(x, y)| x.conj() * y).collect(),
        };
        self.pack_half(&w, inv_d)
    }

    fn id(&self) -> &str {
        &self.id
    }
}

impl Differentiable for HolE {
    /// Analytic gradient of `score = r · (e_s ⋆ e_o)`. Differentiating the
    /// triple sum `Σ_k Σ_i r[k] s[i] o[(i+k) mod d]`:
    ///
    /// ```text
    /// ∂score/∂s = corr(r, o)     ∂score/∂r = corr(s, o)     ∂score/∂o = conv(r, s)
    /// ```
    ///
    /// each length `d`, all via the tested FFT primitives (`grad_r` reuses the
    /// same correlation `score` computes).
    fn grad(&self, s: &[f32], r: &[f32], o: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(s.len(), self.dims, "grad: s length must equal dims");
        assert_eq!(r.len(), self.dims, "grad: r length must equal dims");
        assert_eq!(o.len(), self.dims, "grad: o length must equal dims");
        let grad_s = self.plan.circular_correlation(r, o);
        let grad_r = self.plan.circular_correlation(s, o);
        let grad_o = self.plan.circular_convolution(r, s);
        (grad_s, grad_r, grad_o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(d: usize, mut state: u64) -> Vec<f32> {
        (0..d)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 11) as f32 / (1u64 << 53) as f32;
                2.0 * u - 1.0
            })
            .collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Central finite-difference check of `grad` against `score` for one input
    /// vector. `pick` selects which of (s, r, o) to perturb.
    fn assert_grad_matches_fd(
        h: &HolE,
        s: &[f32],
        r: &[f32],
        o: &[f32],
        analytic: &[f32],
        which: usize,
    ) {
        let eps = 1e-3f32;
        for j in 0..analytic.len() {
            let (mut sp, mut rp, mut op) = (s.to_vec(), r.to_vec(), o.to_vec());
            let (mut sm, mut rm, mut om) = (s.to_vec(), r.to_vec(), o.to_vec());
            match which {
                0 => {
                    sp[j] += eps;
                    sm[j] -= eps;
                }
                1 => {
                    rp[j] += eps;
                    rm[j] -= eps;
                }
                _ => {
                    op[j] += eps;
                    om[j] -= eps;
                }
            }
            let fd = (h.score(&sp, &rp, &op) - h.score(&sm, &rm, &om)) / (2.0 * eps);
            assert!(
                (fd - analytic[j]).abs() <= 1e-2 + 1e-2 * fd.abs(),
                "which={which} j={j}: fd {fd} vs analytic {}",
                analytic[j]
            );
        }
    }

    #[test]
    fn grad_matches_finite_difference() {
        for &d in &[8usize, 64] {
            let h = HolE::new(d).unwrap();
            let s = rand_vec(d, 0x9001 ^ d as u64);
            let r = rand_vec(d, 0x9002 ^ d as u64);
            let o = rand_vec(d, 0x9003 ^ d as u64);
            let (gs, gr, go) = h.grad(&s, &r, &o);
            assert_eq!(gs.len(), d);
            assert_eq!(gr.len(), d);
            assert_eq!(go.len(), d);
            assert_grad_matches_fd(&h, &s, &r, &o, &gs, 0);
            assert_grad_matches_fd(&h, &s, &r, &o, &gr, 1);
            assert_grad_matches_fd(&h, &s, &r, &o, &go, 2);
        }
    }

    #[test]
    fn odd_dims_rejected() {
        assert!(HolE::new(9).is_err());
        assert!(HolE::new(16).is_ok());
    }

    #[test]
    fn index_dims_is_d_plus_two() {
        let h = HolE::new(128).unwrap();
        assert_eq!(h.index_dims(), 130);
        assert_eq!(h.index_vector(&rand_vec(128, 1)).len(), 130);
    }

    /// The load-bearing test: the Fourier index dotted with the query vector
    /// reproduces the exact score, on BOTH sides, across the DC/Nyquist-weight
    /// edge cases at every dimension.
    #[test]
    fn index_dot_query_equals_score_both_sides() {
        for &d in &[8usize, 128, 256, 512] {
            let h = HolE::new(d).unwrap();
            for t in 0..8u64 {
                let s = rand_vec(d, 0x51 ^ (d as u64) ^ (t << 8));
                let r = rand_vec(d, 0x52 ^ (d as u64) ^ (t << 9));
                let o = rand_vec(d, 0x53 ^ (d as u64) ^ (t << 10));
                let score = h.score(&s, &r, &o);

                let q_tail = h.query_vector(&r, &s, Side::Tail);
                let idx_o = h.index_vector(&o);
                let via_tail = dot(&q_tail, &idx_o);

                let q_head = h.query_vector(&r, &o, Side::Head);
                let idx_s = h.index_vector(&s);
                let via_head = dot(&q_head, &idx_s);

                let tol = 1e-4 + 1e-3 * score.abs();
                assert!(
                    (via_tail - score).abs() <= tol,
                    "d={d} tail: {via_tail} vs {score}"
                );
                assert!(
                    (via_head - score).abs() <= tol,
                    "d={d} head: {via_head} vs {score}"
                );
            }
        }
    }
}
