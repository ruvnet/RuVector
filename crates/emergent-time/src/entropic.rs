//! Entropic time — a β-swept Gibbs-ensemble clock.
//!
//! The internal time of an observed sector can be defined from its *change in
//! entropy*:
//!
//! ```text
//!   τ_S = (S(λ) - S_0) / k
//! ```
//!
//! where `λ` is a lab control parameter. **What this module actually models** is
//! a *temperature sweep* of a Gibbs (thermal) ensemble: `λ` is interpreted as the
//! inverse temperature `β`, and `S(λ)` is the von Neumann entropy of `ρ = e^{−βH}/Z`.
//! This is an equilibrium one-parameter family, **not** closed-system irreversible
//! dynamics — there is no hidden sector exchanging entropy in real time here. It
//! is the simplest honest demonstrator of the entropic-time *reparametrization*:
//! it shows how `τ_S` tracks the entropy curve, and how the "speed of internal
//! time" `dτ_S/dλ` follows entropy production along the sweep. (A genuine
//! cold-atom mini-universe with irreversible entropy exchange — Barontini et al.,
//! *Phys. Rev. Research* 2026 — is the physical system this is an analogue *of*,
//! not what is simulated.)
//! Derivatives reparametrize as
//!
//! ```text
//!   dX/dτ_S = (k / (dS/dλ)) · dX/dλ.
//! ```
//!
//! The "speed of internal time" `dτ_S/dλ = (dS/dλ)/k` tracks entropy
//! production: when entropy exchange stalls, internal time freezes; when it
//! accelerates, internal time speeds up.

use crate::entropy::entropy_from_spectrum;
use crate::real_matrix::RealMatrix;

/// Maps observed-sector entropy onto an internal time coordinate.
#[derive(Clone, Copy, Debug)]
pub struct EntropicClock {
    /// Reference entropy `S_0` (internal time origin).
    pub s0: f64,
    /// Clock scale `k` (nats of entropy per unit internal time).
    pub k: f64,
}

impl EntropicClock {
    pub fn new(s0: f64, k: f64) -> Self {
        EntropicClock { s0, k }
    }

    /// Internal time `τ_S` for an observed entropy `s`.
    pub fn tau(&self, s: f64) -> f64 {
        (s - self.s0) / self.k
    }

    /// Speed of internal time `dτ_S/dλ = (dS/dλ)/k`.
    pub fn rate(&self, ds_dlambda: f64) -> f64 {
        ds_dlambda / self.k
    }

    /// Convert a `λ`-derivative into a `τ_S`-derivative,
    /// `dX/dτ_S = (k/(dS/dλ)) dX/dλ`.
    ///
    /// Returns `None` when entropy production vanishes (internal time is frozen,
    /// so the rate of change per unit internal time is undefined / unbounded).
    pub fn convert_derivative(&self, dx_dlambda: f64, ds_dlambda: f64) -> Option<f64> {
        if ds_dlambda.abs() < 1e-12 {
            None
        } else {
            Some((self.k / ds_dlambda) * dx_dlambda)
        }
    }

    /// Reparametrize a `λ`-sampled observable trajectory into internal time.
    /// Each input sample is `(λ, S(λ), X(λ))`; output is `(τ_S, X)`.
    pub fn reparametrize(&self, samples: &[(f64, f64, f64)]) -> Vec<(f64, f64)> {
        samples.iter().map(|&(_l, s, x)| (self.tau(s), x)).collect()
    }
}

/// Gibbs (thermal) density matrix `ρ = e^{-βH}/Z` for a real symmetric
/// Hamiltonian — the standard entropy source for an observed sector at inverse
/// temperature `β`.
pub fn gibbs_density(h: &RealMatrix, beta: f64) -> RealMatrix {
    let (energies, vecs) = h.symmetric_eigen();
    // Shift by the ground-state energy for numerical stability of exp.
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies
        .iter()
        .map(|&e| (-beta * (e - e_min)).exp())
        .collect();
    let z: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / z).collect();
    RealMatrix::from_spectrum(&probs, &vecs)
}

/// Von Neumann entropy of the Gibbs state at inverse temperature `β`.
pub fn gibbs_entropy(h: &RealMatrix, beta: f64) -> f64 {
    let (energies, _v) = h.symmetric_eigen();
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies
        .iter()
        .map(|&e| (-beta * (e - e_min)).exp())
        .collect();
    let z: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / z).collect();
    entropy_from_spectrum(&probs)
}

/// Sweep the inverse temperature `λ = β ∈ [lo, hi]` over the Gibbs ensemble,
/// returning `(λ, S(λ), τ_S(λ))` triples. This is a β-sweep of an equilibrium
/// state (not closed-system irreversible dynamics); it demonstrates how the
/// internal clock runs fast where the entropy curve changes quickly and stalls
/// where it saturates.
pub fn entropic_time_sweep(
    h: &RealMatrix,
    clock: &EntropicClock,
    lo: f64,
    hi: f64,
    steps: usize,
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let frac = if steps <= 1 {
            0.0
        } else {
            i as f64 / (steps - 1) as f64
        };
        let lam = lo + frac * (hi - lo);
        let s = gibbs_entropy(h, lam);
        out.push((lam, s, clock.tau(s)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_h() -> RealMatrix {
        RealMatrix::diag(&[0.0, 1.0, 2.0, 3.0])
    }

    #[test]
    fn gibbs_trace_one() {
        let rho = gibbs_density(&sample_h(), 0.8);
        let tr: f64 = (0..rho.n).map(|i| rho.get(i, i)).sum();
        assert!((tr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_monotone_in_temperature() {
        // Higher temperature (lower β) → higher entropy.
        let s_hot = gibbs_entropy(&sample_h(), 0.1);
        let s_cold = gibbs_entropy(&sample_h(), 5.0);
        assert!(s_hot > s_cold);
    }

    #[test]
    fn frozen_entropy_freezes_time() {
        let clock = EntropicClock::new(0.0, 1.0);
        assert!(clock.convert_derivative(1.0, 0.0).is_none());
        assert!(clock.convert_derivative(1.0, 2.0).unwrap().abs() > 0.0);
    }

    #[test]
    fn tau_reparametrization_formula_is_exact() {
        // Tests the DEFINITION τ_S = (S − S₀)/k as an arithmetic identity. This
        // is true by construction (it is just the formula evaluated) and cannot
        // discriminate a correct entropy curve from an incorrect one — it only
        // checks the reparametrization arithmetic. The discriminating test that
        // ties the clock to the *measured* entropy curve is below.
        let clock = EntropicClock::new(0.5, 2.0);
        assert!((clock.tau(2.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn internal_time_spacing_tracks_measured_entropy_production() {
        // DISCRIMINATING TEST: verify the reparametrization against the REAL
        // Gibbs entropy curve S(λ), not against the τ = (S − S₀)/k definition.
        //
        // The independence is in the *source* of dS/dλ. The clock's τ values are
        // one object; the entropy production is measured separately by finite-
        // differencing `gibbs_entropy(H, β)` — the physical entropy of the actual
        // thermal state ρ = e^{−βH}/Z — at λ points the clock never stored. We
        // then assert:
        //
        //   1. the clock rate (dτ/dλ from its τ samples) equals rate(measured
        //      dS/dλ) — i.e. it tracks the *measured* entropy curve;
        //   2. that entropy curve is physically non-trivial: monotone-rising with
        //      temperature (β decreasing ⇒ S increasing), with a varying slope,
        //      not a constant. A constant/flat/anti-correlated S(λ) would fail.
        //
        // A wrong entropy implementation (e.g. one that ignored the spectrum, or
        // returned a constant) would still satisfy the pure-arithmetic
        // `tau_reparametrization_formula_is_exact` test but would FAIL this one,
        // because here dS/dλ is recomputed from the real thermal state.
        let h = RealMatrix::diag(&[0.0, 1.0, 2.0, 3.0]);
        let k = 1.7;
        let clock = EntropicClock::new(0.0, k);
        let (lo, hi, steps) = (0.2_f64, 4.0_f64, 41);
        let sweep = entropic_time_sweep(&h, &clock, lo, hi, steps);
        let dlam = (hi - lo) / (steps - 1) as f64;
        let eps = dlam / 4.0; // independent probe step for measuring dS/dλ

        // The sweep runs lo→hi in β, i.e. hot→cold, so S should DECREASE along it.
        let s_first = sweep.first().unwrap().1;
        let s_last = sweep.last().unwrap().1;
        assert!(
            s_first - s_last > 0.1,
            "entropy must fall appreciably as β rises (hot→cold) for the test to bite"
        );

        let mut slopes = Vec::new();
        let mut checked = 0;
        for i in 1..sweep.len() - 1 {
            let lam = sweep[i].0;
            let tau_next = sweep[i + 1].2;
            let tau_prev = sweep[i - 1].2;

            // (1) Internal-time spacing per unit λ from the clock's τ samples.
            let dtau_dlam_from_clock = (tau_next - tau_prev) / (2.0 * dlam);

            // (2) Entropy production measured INDEPENDENTLY from the physical
            // thermal state at fresh λ points (not the stored sweep values).
            let s_plus = gibbs_entropy(&h, lam + eps);
            let s_minus = gibbs_entropy(&h, lam - eps);
            let ds_dlam_measured = (s_plus - s_minus) / (2.0 * eps);
            let rate_from_entropy = clock.rate(ds_dlam_measured);

            // The clock rate tracks the measured entropy production. Both are
            // centered differences of the same smooth curve at slightly different
            // step sizes, so they agree to finite-difference order.
            let tol = 5e-3 + 0.02 * rate_from_entropy.abs();
            assert!(
                (dtau_dlam_from_clock - rate_from_entropy).abs() < tol,
                "at λ={lam:.3}: dτ/dλ from clock = {dtau_dlam_from_clock:.5}, \
                 rate(independently-measured dS/dλ) = {rate_from_entropy:.5}"
            );
            slopes.push(ds_dlam_measured);
            checked += 1;
        }
        assert!(checked > 30, "should have checked the bulk of the sweep");

        // The entropy curve is genuinely non-trivial (varying slope), so the
        // clock's speed actually changes along the sweep — it is not a constant
        // reparametrization. Min and max measured slopes differ substantially.
        let smin = slopes.iter().cloned().fold(f64::INFINITY, f64::min);
        let smax = slopes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (smax - smin).abs() > 0.05,
            "entropy production must vary along the sweep (clock speeds up/slows down): \
             slope range [{smin:.4}, {smax:.4}]"
        );
        // And entropy production is negative throughout (S falls as β rises).
        assert!(smax < 0.0, "dS/dβ must be negative across the sweep");
    }
}
