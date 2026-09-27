//! ComplEx (Trouillon et al. 2016; Lacroix et al. 2018) — the product scorer
//! behind the ComplEx-N3-R recipe (ADR-007 §3, plan M1).
//!
//! Embeddings are complex `k`-vectors stored split as
//! `[re_0..re_{k-1}, im_0..im_{k-1}]` (real length `dims = 2k`; `k` is the
//! `complex_rank` recorded in receipts). The score is the trilinear form
//! `Re⟨r, s, conj(o)⟩ = Σ_i Re(r_i · s_i · conj(o_i))`.
//!
//! The score is *bilinear in the open-slot entity*, so the indexed
//! representation of an entity is the row itself (`index_vector = e`) and both
//! query sides are exact inner products (`ann_exact() == true`):
//!
//! ```text
//! tail (s, r, ?):  q = r·s                  → [Re(r·s); Im(r·s)]
//! head (?, r, o):  w = r·conj(o), q = conj(w) → [Re(w); −Im(w)]
//! ```
//!
//! The same module keeps the ADR-006 release gate "HolE ≡ ComplEx" (Hayashi &
//! Shimbo, arXiv:1702.05563): mapping HolE's real `d`-vectors through the full
//! `d`-point DFT gives ComplEx parameters with `k = d`, and
//! `ComplEx.score(F(s), F(r), F(o)) = d · HolE.score(s, r, o)`.

use crate::scorer::{Bilinear, Scorer};
use crate::{Differentiable, Result, Side};

/// ComplEx scorer. `dims` must be even (`dims = 2k`).
#[derive(Debug, Clone)]
pub struct ComplEx {
    dims: usize,
    id: String,
}

impl ComplEx {
    /// Build a ComplEx scorer over `dims` reals (`k = dims / 2` complex
    /// coordinates). `dims` must be even and non-zero.
    pub fn new(dims: usize) -> Result<Self> {
        if dims == 0 || !dims.is_multiple_of(2) {
            return Err(crate::KgeError::Invalid(format!(
                "complex dims must be even and non-zero, got {dims}"
            )));
        }
        Ok(Self {
            dims,
            id: format!("complex@k{}", dims / 2),
        })
    }

    /// Complex coordinates per embedding (`k`, "rank 1000" in kbc/ssl-RP).
    pub fn complex_rank(&self) -> usize {
        self.dims / 2
    }

    #[inline]
    fn k(&self) -> usize {
        self.dims / 2
    }

    fn check(&self, v: &[f32], what: &str) {
        assert_eq!(v.len(), self.dims, "complex: {what} length must equal dims");
    }
}

impl Scorer for ComplEx {
    fn dims(&self) -> usize {
        self.dims
    }

    /// `Re⟨r, s, conj(o)⟩ = Σ_i Re( r_i · s_i · conj(o_i) )`.
    fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32 {
        self.check(s, "s");
        self.check(r, "r");
        self.check(o, "o");
        let k = self.k();
        let (sr, si) = s.split_at(k);
        let (rr, ri) = r.split_at(k);
        let (or, oi) = o.split_at(k);
        let mut acc = 0.0f32;
        for i in 0..k {
            // r·s = (rr·sr − ri·si) + i(rr·si + ri·sr)
            let re_rs = rr[i] * sr[i] - ri[i] * si[i];
            let im_rs = rr[i] * si[i] + ri[i] * sr[i];
            // (r·s)·conj(o): Re = re_rs·or + im_rs·oi
            acc += re_rs * or[i] + im_rs * oi[i];
        }
        acc
    }

    fn query_vector(&self, r: &[f32], anchor: &[f32], side: Side) -> Vec<f32> {
        let mut out = vec![0.0; self.dims];
        self.query_into(r, anchor, side, &mut out);
        out
    }

    fn index_vector(&self, e: &[f32]) -> Vec<f32> {
        self.check(e, "e");
        e.to_vec()
    }

    fn index_dims(&self) -> usize {
        self.dims
    }

    fn id(&self) -> &str {
        &self.id
    }
}

impl Differentiable for ComplEx {
    /// Expanding `score = Σ_i (rr·sr − ri·si)·or + (rr·si + ri·sr)·oi`:
    ///
    /// ```text
    /// ∂/∂sr = rr·or + ri·oi     ∂/∂si = rr·oi − ri·or      (= head query)
    /// ∂/∂rr = sr·or + si·oi     ∂/∂ri = sr·oi − si·or
    /// ∂/∂or = rr·sr − ri·si     ∂/∂oi = rr·si + ri·sr      (= tail query)
    /// ```
    fn grad(&self, s: &[f32], r: &[f32], o: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.check(s, "s");
        self.check(r, "r");
        self.check(o, "o");
        let k = self.k();
        let (sr, si) = s.split_at(k);
        let (rr, ri) = r.split_at(k);
        let (or, oi) = o.split_at(k);
        let (mut gs, mut gr, mut go) = (
            vec![0.0; self.dims],
            vec![0.0; self.dims],
            vec![0.0; self.dims],
        );
        for i in 0..k {
            gs[i] = rr[i] * or[i] + ri[i] * oi[i];
            gs[k + i] = rr[i] * oi[i] - ri[i] * or[i];
            gr[i] = sr[i] * or[i] + si[i] * oi[i];
            gr[k + i] = sr[i] * oi[i] - si[i] * or[i];
            go[i] = rr[i] * sr[i] - ri[i] * si[i];
            go[k + i] = rr[i] * si[i] + ri[i] * sr[i];
        }
        (gs, gr, go)
    }

    fn as_bilinear(&self) -> Option<&dyn Bilinear> {
        Some(self)
    }
}

impl Bilinear for ComplEx {
    fn query_into(&self, r: &[f32], anchor: &[f32], side: Side, out: &mut [f32]) {
        self.check(r, "r");
        self.check(anchor, "anchor");
        self.check(out, "out");
        let k = self.k();
        let (rr, ri) = r.split_at(k);
        let (ar, ai) = anchor.split_at(k);
        let (qr, qi) = out.split_at_mut(k);
        for i in 0..k {
            match side {
                // anchor = s: q = r·s
                Side::Tail => {
                    qr[i] = rr[i] * ar[i] - ri[i] * ai[i];
                    qi[i] = rr[i] * ai[i] + ri[i] * ar[i];
                }
                // anchor = o: q = conj(r·conj(o)) = [rr·or + ri·oi ; rr·oi − ri·or]
                Side::Head => {
                    qr[i] = rr[i] * ar[i] + ri[i] * ai[i];
                    qi[i] = rr[i] * ai[i] - ri[i] * ar[i];
                }
            }
        }
    }

    fn query_backward(
        &self,
        r: &[f32],
        anchor: &[f32],
        side: Side,
        d_q: &[f32],
        d_r: &mut [f32],
        d_anchor: &mut [f32],
    ) {
        self.check(r, "r");
        self.check(anchor, "anchor");
        self.check(d_q, "d_q");
        self.check(d_r, "d_r");
        self.check(d_anchor, "d_anchor");
        let k = self.k();
        let (rr, ri) = r.split_at(k);
        let (ar, ai) = anchor.split_at(k);
        let (gr, gi) = d_q.split_at(k);
        let (drr, dri) = d_r.split_at_mut(k);
        let (dar, dai) = d_anchor.split_at_mut(k);
        for i in 0..k {
            match side {
                // qr = rr·ar − ri·ai ; qi = rr·ai + ri·ar
                Side::Tail => {
                    dar[i] += gr[i] * rr[i] + gi[i] * ri[i];
                    dai[i] += -gr[i] * ri[i] + gi[i] * rr[i];
                    drr[i] += gr[i] * ar[i] + gi[i] * ai[i];
                    dri[i] += -gr[i] * ai[i] + gi[i] * ar[i];
                }
                // qr = rr·ar + ri·ai ; qi = rr·ai − ri·ar
                Side::Head => {
                    dar[i] += gr[i] * rr[i] - gi[i] * ri[i];
                    dai[i] += gr[i] * ri[i] + gi[i] * rr[i];
                    drr[i] += gr[i] * ar[i] + gi[i] * ai[i];
                    dri[i] += gr[i] * ai[i] - gi[i] * ar[i];
                }
            }
        }
    }

    fn index_is_identity(&self) -> bool {
        true
    }
}

/// `conj(r)` in the split layout: the tied reciprocal row `r⁻¹` for ComplEx,
/// under which `score(o, conj(r), s) == score(s, r, o)` exactly.
pub fn conj(r: &[f32]) -> Vec<f32> {
    let k = r.len() / 2;
    r.iter()
        .enumerate()
        .map(|(i, &x)| if i < k { x } else { -x })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::hole::HolE;
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    /// Map a real `d`-vector to ComplEx parameters via the full `d`-point DFT,
    /// laid out as `[Re(F(v)); Im(F(v))]` (length `2d`, `k = d`).
    fn complexify(planner: &mut FftPlanner<f32>, v: &[f32]) -> Vec<f32> {
        let d = v.len();
        let fft = planner.plan_fft_forward(d);
        let mut buf: Vec<Complex<f32>> = v.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buf);
        let mut out = Vec::with_capacity(2 * d);
        out.extend(buf.iter().map(|c| c.re));
        out.extend(buf.iter().map(|c| c.im));
        out
    }

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

    /// ADR-006 release gate: HolE ≡ ComplEx on 100 random triples.
    #[test]
    fn hole_equivalent_to_complex() {
        let d = 128usize;
        let hole = HolE::new(d).unwrap();
        let complex = ComplEx::new(2 * d).unwrap();
        let mut planner = FftPlanner::<f32>::new();
        for t in 0..100u64 {
            let s = rand_vec(d, 0xA1 ^ (t << 3));
            let r = rand_vec(d, 0xB2 ^ (t << 5));
            let o = rand_vec(d, 0xC3 ^ (t << 7));

            let hole_score = hole.score(&s, &r, &o);
            let cx_score = complex.score(
                &complexify(&mut planner, &s),
                &complexify(&mut planner, &r),
                &complexify(&mut planner, &o),
            );
            // ComplEx.score(F·) = d · HolE.score. Mixed abs+rel tolerance so
            // near-zero scores don't blow a bare relative bound.
            let expected = d as f32 * hole_score;
            assert!(
                (cx_score - expected).abs() <= 1e-4 + 1e-3 * expected.abs(),
                "triple {t}: complex {cx_score} vs d·hole {expected}"
            );
        }
    }

    /// Sanity: ComplEx is genuinely trilinear.
    #[test]
    fn complex_uses_all_fields() {
        let cx = ComplEx::new(4).unwrap();
        // [re0, re1, im0, im1]
        let s = [1.0, 0.0, 0.0, 0.0];
        let r = [1.0, 0.0, 0.0, 0.0];
        let o = [1.0, 0.0, 0.0, 0.0];
        assert!((cx.score(&s, &r, &o) - 1.0).abs() < 1e-6);
        assert_eq!(cx.complex_rank(), 2);
        assert_eq!(cx.id(), "complex@k2");
    }

    #[test]
    fn odd_dims_rejected() {
        assert!(ComplEx::new(7).is_err());
        assert!(ComplEx::new(0).is_err());
        assert!(ComplEx::new(8).is_ok());
    }

    /// M1 acceptance: the analytic grad matches central finite differences
    /// within 1e-3 relative (the score is linear in each coordinate, so the
    /// central difference is exact up to rounding).
    #[test]
    fn grad_matches_finite_difference() {
        for &d in &[8usize, 64] {
            let cx = ComplEx::new(d).unwrap();
            let s = rand_vec(d, 0xB001 ^ d as u64);
            let r = rand_vec(d, 0xB002 ^ d as u64);
            let o = rand_vec(d, 0xB003 ^ d as u64);
            let (gs, gr, go) = cx.grad(&s, &r, &o);
            let eps = 1e-2f32;
            for (which, analytic) in [(0usize, &gs), (1, &gr), (2, &go)] {
                for j in 0..d {
                    let mut p = [s.clone(), r.clone(), o.clone()];
                    let mut m = [s.clone(), r.clone(), o.clone()];
                    p[which][j] += eps;
                    m[which][j] -= eps;
                    let fd = (cx.score(&p[0], &p[1], &p[2]) - cx.score(&m[0], &m[1], &m[2]))
                        / (2.0 * eps);
                    assert!(
                        (fd - analytic[j]).abs() <= 1e-4 + 1e-3 * analytic[j].abs(),
                        "d={d} which={which} j={j}: fd {fd} vs analytic {}",
                        analytic[j]
                    );
                }
            }
        }
    }

    /// Both query sides are exact inner products with the identity index.
    #[test]
    fn query_vectors_are_exact() {
        let d = 32;
        let cx = ComplEx::new(d).unwrap();
        for t in 0..20u64 {
            let s = rand_vec(d, 0x11 ^ (t << 4));
            let r = rand_vec(d, 0x22 ^ (t << 4));
            let o = rand_vec(d, 0x33 ^ (t << 4));
            let f = cx.score(&s, &r, &o);
            let qt = cx.query_vector(&r, &s, Side::Tail);
            let qh = cx.query_vector(&r, &o, Side::Head);
            assert!((dot(&qt, &cx.index_vector(&o)) - f).abs() <= 1e-5 + 1e-5 * f.abs());
            assert!((dot(&qh, &cx.index_vector(&s)) - f).abs() <= 1e-5 + 1e-5 * f.abs());
            // Gradient w.r.t. the open slot equals the query.
            let (gs, _, go) = cx.grad(&s, &r, &o);
            assert_eq!(go, qt);
            assert_eq!(gs, qh);
        }
        assert!(cx.ann_exact());
    }

    /// `query_backward` is the exact adjoint of `query_into`: for a random
    /// upstream `g`, `<g, dq>` equals the directional change in `q`.
    #[test]
    fn query_backward_matches_finite_difference() {
        let d = 16;
        let cx = ComplEx::new(d).unwrap();
        let r = rand_vec(d, 0x51);
        let a = rand_vec(d, 0x52);
        let g = rand_vec(d, 0x53);
        for side in [Side::Tail, Side::Head] {
            let mut d_r = vec![0.0; d];
            let mut d_a = vec![0.0; d];
            cx.query_backward(&r, &a, side, &g, &mut d_r, &mut d_a);
            let eps = 1e-2f32;
            let phi = |r: &[f32], a: &[f32]| dot(&g, &cx.query_vector(r, a, side));
            for j in 0..d {
                let (mut rp, mut rm) = (r.clone(), r.clone());
                rp[j] += eps;
                rm[j] -= eps;
                let fd = (phi(&rp, &a) - phi(&rm, &a)) / (2.0 * eps);
                assert!((fd - d_r[j]).abs() <= 1e-4 + 1e-3 * fd.abs(), "d_r {j}");
                let (mut ap, mut am) = (a.clone(), a.clone());
                ap[j] += eps;
                am[j] -= eps;
                let fd = (phi(&r, &ap) - phi(&r, &am)) / (2.0 * eps);
                assert!((fd - d_a[j]).abs() <= 1e-4 + 1e-3 * fd.abs(), "d_a {j}");
            }
        }
    }

    /// Tied reciprocal rows: with `r⁻¹ = conj(r)`, the tail query of the
    /// reciprocal triple `(o, r⁻¹, ?)` scores every candidate exactly as the
    /// head query `(?, r, o)` does (M1 acceptance, scorer level).
    #[test]
    fn conj_relation_is_exact_reciprocal() {
        let d = 32;
        let cx = ComplEx::new(d).unwrap();
        for t in 0..20u64 {
            let s = rand_vec(d, 0x61 ^ (t << 4));
            let r = rand_vec(d, 0x62 ^ (t << 4));
            let o = rand_vec(d, 0x63 ^ (t << 4));
            let rinv = conj(&r);
            let f = cx.score(&s, &r, &o);
            let g = cx.score(&o, &rinv, &s);
            assert!((f - g).abs() <= 1e-5 + 1e-5 * f.abs(), "{f} vs {g}");
            let qh = cx.query_vector(&r, &o, Side::Head);
            let qt_rec = cx.query_vector(&rinv, &o, Side::Tail);
            for (a, b) in qh.iter().zip(&qt_rec) {
                assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
            }
        }
    }
}
