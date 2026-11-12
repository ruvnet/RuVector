//! RotatE — relation as rotation in complex space (ADR-002 §1, §4).
//!
//! The one family member that provably represents composition `r1 ∘ r2`
//! (arXiv:1902.10197). Entities are complex `k`-vectors stored split as
//! `[re_0..re_{k-1}, im_0..im_{k-1}]` (real length `d = 2k`). A relation is
//! projected to unit modulus per dimension (`r ↦ r / |r|`), so it acts as a
//! pure rotation. Score is `-‖e_s ∘ r − e_o‖` (γ-margin left to the loss).
//!
//! ## ANN — approximate MIPS, not exact (`ann_exact() == false`)
//!
//! `-‖a − o‖` is not an inner product. Using `‖a − o‖² = ‖a‖² + ‖o‖² − 2 a·o`
//! we retrieve with an appended-norm surrogate:
//!
//! ```text
//! index_vector(o) = [ 2·o ; −‖o‖² ; 1 ]          (length d + 2)
//! query_vector    = [ a   ;  1    ; −‖a‖² ]       (length d + 2)
//! dot             = 2 a·o − ‖o‖² − ‖a‖² = −‖a − o‖²
//! ```
//!
//! where `a = e_s ∘ r` (tail) or `a = e_o ∘ conj(r)` (head — valid because a
//! unit-modulus rotation preserves the norm, so
//! `‖e_s ∘ r − e_o‖ = ‖e_s − e_o ∘ conj(r)‖`). The dot equals `−‖a − o‖²`, a
//! strictly order-preserving transform of the true score `−‖a − o‖` over the
//! feasible region, so it ranks candidates correctly but is **not** the score
//! itself — hence `ann_exact()` is overridden to `false` and the exact rerank
//! is mandatory.

use crate::scorer::Scorer;
use crate::{Differentiable, Result, Side};

const UNIT_EPS: f32 = 1e-12;
/// Below this ‖s∘r − o‖ the score is at its cusp (max, 0) and non-smooth;
/// `grad` returns the zero subgradient there.
const GRAD_EPS: f32 = 1e-9;

/// Rotation ("RotatE") scorer. `dims` must be even (`d = 2k`).
pub struct RotatE {
    dims: usize,
    id: String,
}

impl RotatE {
    /// Build a RotatE scorer for embedding dimension `dims` (must be even).
    pub fn new(dims: usize) -> Result<Self> {
        if dims == 0 || !dims.is_multiple_of(2) {
            return Err(crate::KgeError::Invalid(format!(
                "rotate dims must be even and non-zero, got {dims}"
            )));
        }
        Ok(Self {
            dims,
            id: format!("rotate@d{dims}"),
        })
    }

    #[inline]
    fn k(&self) -> usize {
        self.dims / 2
    }

    /// Per-dimension unit-modulus projection of a relation: `r ↦ r / |r|`,
    /// returning `(cos, sin)` halves of length `k`. A near-zero-modulus
    /// dimension collapses to `1 + 0i` (identity rotation) to avoid NaN.
    fn unit(&self, r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let k = self.k();
        let (rr, ri) = r.split_at(k);
        let mut cos = vec![0.0f32; k];
        let mut sin = vec![0.0f32; k];
        for i in 0..k {
            let mag = (rr[i] * rr[i] + ri[i] * ri[i]).sqrt();
            if mag < UNIT_EPS {
                cos[i] = 1.0;
                sin[i] = 0.0;
            } else {
                cos[i] = rr[i] / mag;
                sin[i] = ri[i] / mag;
            }
        }
        (cos, sin)
    }

    /// Rotate entity `e` by relation `r` (complex mult by the unit relation),
    /// returning a `d`-length `[re; im]` vector. `conjugate` applies `conj(r)`
    /// (inverse rotation), used for the head side.
    fn apply(&self, e: &[f32], r: &[f32], conjugate: bool) -> Vec<f32> {
        let k = self.k();
        let (er, ei) = e.split_at(k);
        let (cos, sin_raw) = self.unit(r);
        let mut out = vec![0.0f32; self.dims];
        for i in 0..k {
            let c = cos[i];
            let s = if conjugate { -sin_raw[i] } else { sin_raw[i] };
            // (er + i·ei) · (c + i·s)
            out[i] = er[i] * c - ei[i] * s; // real
            out[k + i] = er[i] * s + ei[i] * c; // imag
        }
        out
    }

    /// Compose two relations: `compose(r1, r2) = r1 ∘ r2` as phase addition
    /// (complex mult of the unit relations), returned as a unit-modulus
    /// `[re; im]` relation vector of length `d`. The public helper behind the
    /// `COMPOSE(r1, r2)` query operator (ADR-001 §2).
    pub fn compose(&self, r1: &[f32], r2: &[f32]) -> Vec<f32> {
        let k = self.k();
        let (c1, s1) = self.unit(r1);
        let (c2, s2) = self.unit(r2);
        let mut out = vec![0.0f32; self.dims];
        for i in 0..k {
            out[i] = c1[i] * c2[i] - s1[i] * s2[i];
            out[k + i] = c1[i] * s2[i] + s1[i] * c2[i];
        }
        out
    }
}

/// `[2·v ; −‖v‖² ; 1]` — the entity side of the appended-norm MIPS surrogate.
fn index_augment(v: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let mut out = Vec::with_capacity(v.len() + 2);
    out.extend(v.iter().map(|x| 2.0 * x));
    out.push(-norm_sq);
    out.push(1.0);
    out
}

/// `[a ; 1 ; −‖a‖²]` — the query side of the appended-norm MIPS surrogate.
fn query_augment(a: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = a.iter().map(|x| x * x).sum();
    let mut out = Vec::with_capacity(a.len() + 2);
    out.extend_from_slice(a);
    out.push(1.0);
    out.push(-norm_sq);
    out
}

impl Scorer for RotatE {
    fn dims(&self) -> usize {
        self.dims
    }

    fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32 {
        assert_eq!(s.len(), self.dims, "score: s length must equal dims");
        assert_eq!(r.len(), self.dims, "score: r length must equal dims");
        assert_eq!(o.len(), self.dims, "score: o length must equal dims");
        let a = self.apply(s, r, false);
        let l2_sq: f32 = a.iter().zip(o).map(|(x, y)| (x - y) * (x - y)).sum();
        -l2_sq.sqrt()
    }

    fn index_vector(&self, e: &[f32]) -> Vec<f32> {
        assert_eq!(e.len(), self.dims, "index_vector: length must equal dims");
        index_augment(e)
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
        // Tail open: anchor = s, a = s ∘ r. Head open: anchor = o,
        // a = o ∘ conj(r) (unit modulus preserves the norm).
        let a = match side {
            Side::Tail => self.apply(anchor, r, false),
            Side::Head => self.apply(anchor, r, true),
        };
        query_augment(&a)
    }

    /// RotatE's ANN dot product is an order-preserving surrogate, not the exact
    /// score — the exact rerank is mandatory.
    fn ann_exact(&self) -> bool {
        false
    }

    fn id(&self) -> &str {
        &self.id
    }
}

impl Differentiable for RotatE {
    /// Analytic gradient of `score = −‖a − o‖` with `a = e_s ∘ u`, `u = r/|r|`.
    /// Let `g_a = ∂score/∂a = −(a − o)/‖a − o‖`. Then per complex dim `i`:
    ///
    /// - `∂score/∂o = (a − o)/‖a − o‖ = −g_a`
    /// - `∂score/∂s = conj(u) · g_a`  (complex mult)
    /// - `∂score/∂u = conj(s) · g_a`, projected onto the unit circle's tangent
    ///   by `(I − û ûᵀ)/|r|` to become `∂score/∂r` (the radial component is
    ///   killed by the normalisation `u = r/|r|`).
    ///
    /// At the score's cusp (`a = o`) the norm is non-smooth; the zero
    /// subgradient is returned.
    fn grad(&self, s: &[f32], r: &[f32], o: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(s.len(), self.dims, "grad: s length must equal dims");
        assert_eq!(r.len(), self.dims, "grad: r length must equal dims");
        assert_eq!(o.len(), self.dims, "grad: o length must equal dims");
        let d = self.dims;
        let k = self.k();
        let a = self.apply(s, r, false);
        let diff: Vec<f32> = a.iter().zip(o).map(|(x, y)| x - y).collect();
        let l2 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
        if l2 < GRAD_EPS {
            return (vec![0.0; d], vec![0.0; d], vec![0.0; d]);
        }

        let mut grad_o = vec![0.0f32; d];
        let mut grad_s = vec![0.0f32; d];
        let mut grad_r = vec![0.0f32; d];
        let (u_re, u_im) = self.unit(r);
        let (s_re, s_im) = s.split_at(k);
        let (r_re, r_im) = r.split_at(k);
        for i in 0..k {
            // g_a = -(a - o)/l2 ; grad_o = -g_a = diff/l2.
            grad_o[i] = diff[i] / l2;
            grad_o[k + i] = diff[k + i] / l2;
            let ga_re = -diff[i] / l2;
            let ga_im = -diff[k + i] / l2;

            // grad_s = conj(u) · g_a.
            grad_s[i] = ga_re * u_re[i] + ga_im * u_im[i];
            grad_s[k + i] = -ga_re * u_im[i] + ga_im * u_re[i];

            // grad wrt u = conj(s) · g_a, then tangent-project to grad wrt r.
            let gu_re = ga_re * s_re[i] + ga_im * s_im[i];
            let gu_im = -ga_re * s_im[i] + ga_im * s_re[i];
            let mag = (r_re[i] * r_re[i] + r_im[i] * r_im[i]).sqrt();
            if mag >= UNIT_EPS {
                // (I − û ûᵀ) g_u / mag, with 1 − û_re² = û_im².
                grad_r[i] = (gu_re * u_im[i] * u_im[i] - gu_im * u_re[i] * u_im[i]) / mag;
                grad_r[k + i] = (-gu_re * u_re[i] * u_im[i] + gu_im * u_re[i] * u_re[i]) / mag;
            }
        }
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

    fn assert_grad_matches_fd(
        rot: &RotatE,
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
            let fd = (rot.score(&sp, &rp, &op) - rot.score(&sm, &rm, &om)) / (2.0 * eps);
            assert!(
                (fd - analytic[j]).abs() <= 2e-2 + 2e-2 * fd.abs(),
                "which={which} j={j}: fd {fd} vs analytic {}",
                analytic[j]
            );
        }
    }

    #[test]
    fn grad_matches_finite_difference() {
        // Distinct s,r,o keep ‖s∘r − o‖ well away from the non-smooth cusp.
        for &d in &[8usize, 32] {
            let rot = RotatE::new(d).unwrap();
            let s = rand_vec(d, 0xA001 ^ d as u64);
            let r = rand_vec(d, 0xA002 ^ d as u64);
            let o = rand_vec(d, 0xA003 ^ d as u64);
            let (gs, gr, go) = rot.grad(&s, &r, &o);
            assert_eq!(gs.len(), d);
            assert_eq!(gr.len(), d);
            assert_eq!(go.len(), d);
            assert_grad_matches_fd(&rot, &s, &r, &o, &gs, 0);
            assert_grad_matches_fd(&rot, &s, &r, &o, &gr, 1);
            assert_grad_matches_fd(&rot, &s, &r, &o, &go, 2);
        }
    }

    #[test]
    fn odd_dims_rejected() {
        assert!(RotatE::new(7).is_err());
        assert!(RotatE::new(8).is_ok());
    }

    #[test]
    fn perfect_triple_scores_zero() {
        let d = 32;
        let rot = RotatE::new(d).unwrap();
        let s = rand_vec(d, 1);
        let r = rand_vec(d, 2);
        let o = rot.apply(&s, &r, false); // o = s ∘ r exactly
        assert!(rot.score(&s, &r, &o).abs() < 1e-5);
    }

    #[test]
    fn compose_is_unit_modulus() {
        let d = 32;
        let rot = RotatE::new(d).unwrap();
        let r1 = rand_vec(d, 3);
        let r2 = rand_vec(d, 4);
        let c = rot.compose(&r1, &r2);
        let k = d / 2;
        for i in 0..k {
            let mag = (c[i] * c[i] + c[k + i] * c[k + i]).sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "dim {i} modulus {mag}");
        }
    }

    /// Composition acts on entities: applying `compose(r1,r2)` equals applying
    /// r1 then r2 (the property RotatE exists for).
    #[test]
    fn composition_matches_sequential_rotation() {
        let d = 32;
        let rot = RotatE::new(d).unwrap();
        let e = rand_vec(d, 5);
        let r1 = rand_vec(d, 6);
        let r2 = rand_vec(d, 7);
        let seq = rot.apply(&rot.apply(&e, &r1, false), &r2, false);
        let composed = rot.apply(&e, &rot.compose(&r1, &r2), false);
        for (a, b) in seq.iter().zip(&composed) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// The MIPS surrogate ranks the true tail top of the pack: dot with the
    /// correct object exceeds dot with random distractors.
    #[test]
    fn surrogate_ranks_true_tail_first() {
        let d = 32;
        let rot = RotatE::new(d).unwrap();
        let s = rand_vec(d, 8);
        let r = rand_vec(d, 9);
        let o = rot.apply(&s, &r, false);
        let q = rot.query_vector(&r, &s, Side::Tail);
        let true_dot = dot(&q, &rot.index_vector(&o));
        for t in 20..40u64 {
            let distractor = rand_vec(d, t);
            assert!(true_dot > dot(&q, &rot.index_vector(&distractor)));
        }
    }
}
