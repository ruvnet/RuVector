//! ComplEx — internal, test-only (ADR-002 §1). Never a product scorer; it
//! exists solely to turn "HolE ≡ ComplEx" (Hayashi & Shimbo, arXiv:1702.05563)
//! into an executable release gate (ADR-006).
//!
//! ComplEx scores `Re⟨r, s, conj(o)⟩` over complex embeddings of dimension `k`,
//! stored split as `[re_0..re_{k-1}, im_0..im_{k-1}]` (length `2k`). Mapping
//! HolE's real `d`-vectors through the full `d`-point DFT yields ComplEx
//! parameters with `k = d`, and ADR-002 §2's Parseval identity gives
//! `ComplEx.score(F(s), F(r), F(o)) = d · HolE.score(s, r, o)` (the factor `d`
//! is HolE's `1/d` normalisation, absent from the raw complex trilinear form).
//! The whole module is `#[cfg(test)]`-gated by its declaration in `mod.rs`, so
//! it is compiled only for tests and never ships.

use crate::scorer::hole::HolE;
use crate::scorer::Scorer;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Test-only ComplEx over `k` complex dims stored as `[re; im]` (length `2k`).
struct ComplEx {
    k: usize,
}

impl ComplEx {
    /// `Re⟨r, s, conj(o)⟩ = Σ_i Re( r_i · s_i · conj(o_i) )`.
    fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32 {
        let k = self.k;
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
}

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

/// ADR-006 release gate: HolE ≡ ComplEx on 100 random triples.
#[test]
fn hole_equivalent_to_complex() {
    let d = 128usize;
    let hole = HolE::new(d).unwrap();
    let complex = ComplEx { k: d };
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

/// Sanity: ComplEx is genuinely trilinear (used only to keep the struct honest).
#[test]
fn complex_uses_all_fields() {
    let cx = ComplEx { k: 2 };
    // [re0, re1, im0, im1]
    let s = [1.0, 0.0, 0.0, 0.0];
    let r = [1.0, 0.0, 0.0, 0.0];
    let o = [1.0, 0.0, 0.0, 0.0];
    assert!((cx.score(&s, &r, &o) - 1.0).abs() < 1e-6);
}
