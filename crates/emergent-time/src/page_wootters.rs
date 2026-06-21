//! Page–Wootters relational time.
//!
//! Time is not assumed; it emerges from correlations inside a globally
//! *static* entangled state of clock ⊗ rest.
//!
//! ## Exact construction
//!
//! Diagonalize the system Hamiltonian `H_R = Σ_k E_k |E_k><E_k|`. Build a clock
//! of the same dimension whose energy eigenstate `|k>_C` carries energy `-E_k`,
//! i.e. `H_C = diag(-E_0, …, -E_{d-1})`. The global state
//!
//! ```text
//!   |Ψ> = (1/√d) Σ_k |k>_C ⊗ |E_k>_R
//! ```
//!
//! satisfies the Wheeler–DeWitt constraint **exactly**:
//!
//! ```text
//!   Ĵ|Ψ> = (H_C ⊗ I + I ⊗ H_R)|Ψ> = (1/√d) Σ_k (-E_k + E_k)|k>|E_k> = 0.
//! ```
//!
//! Yet an internal observer who reads the clock *pointer* state
//! `|t>_C = Σ_k e^{iE_k t}|k>_C` recovers Schrödinger evolution:
//!
//! ```text
//!   <t|Ψ>_R ∝ Σ_k e^{-iE_k t}|E_k>_R = e^{-iH_R t}|ψ_0>,   |ψ_0> = Σ_k|E_k>.
//! ```
//!
//! Evolution is what the rest sector *looks like* conditioned on the clock.
//!
//! ## Scope and honest limitations
//!
//! 1. **Real-symmetric Hamiltonians only.** This construction (and the whole
//!    numerical core it rests on) assumes `H_R` is real symmetric: it is
//!    diagonalized by the real Jacobi eigensolver and the clock is built from its
//!    real spectrum. Complex-Hermitian `H_R` is out of scope for v1.
//!
//! 2. **Born-rule weighting holds only for pure global states.** The
//!    post-conditioning normalization performed in
//!    [`PageWootters::conditional_state`] reproduces the Born-rule partial-trace
//!    weight `‖⟨t|Ψ⟩‖²` **only because the global state `|Ψ⟩` is pure**. For a
//!    mixed global state the correct conditional object is a conditioned density
//!    operator, and a single normalized vector would not capture it. Do not read
//!    the "fidelity = 1.0" result as holding for mixed `|Ψ⟩`.
//!
//! 3. **Single-time conditional states only — Kuchař's objection is out of
//!    scope.** What is recovered here is the *single-time* conditional state
//!    `ρ_R(t)`, correctly reproducing Schrödinger evolution (Page & Wootters
//!    1983; Giovannetti, Lloyd & Maccone, *Phys. Rev. D* 91, 084041, 2015). This
//!    construction does **not** address Kuchař's two-time-correlation objection
//!    (Kuchař 1992): naive conditioning gives the wrong propagator for
//!    *two-time* correlation functions `⟨t₂|…|t₁⟩` without the conditional-
//!    probability (or evolving-constants) machinery. v1 deliberately scopes to
//!    single-time conditioning; multi-time correlators are future work. So
//!    "evolution emerges, fidelity 1.0" means *single-time evolution is exactly
//!    reproduced*, nothing stronger.

use crate::complex::{normalized, Complex};
use crate::complex_matrix::{exp_i_apply_from_spectrum, schrodinger_propagator};
use crate::real_matrix::RealMatrix;
use crate::state::condition_on_clock;

/// A relational clock paired with a system Hamiltonian.
pub struct PageWootters {
    /// System (rest-sector) Hamiltonian.
    pub h_r: RealMatrix,
    /// System energy levels `E_k`.
    pub energies: Vec<f64>,
    /// Energy eigenvectors as columns (`|E_k>` is column `k`).
    pub vecs: RealMatrix,
    /// Hilbert-space dimension of each sector.
    pub dim: usize,
    /// The `t`-independent global static state `|Ψ>` (length `dim²`), computed
    /// once in [`PageWootters::new`] (P2: hoisted out of the per-`t` path).
    psi_static: Vec<Complex>,
    /// The normalized reference state `|ψ_0> = Σ_k |E_k>` (length `dim`),
    /// precomputed so cached-eigenbasis evolution never rebuilds it.
    psi0: Vec<Complex>,
}

impl PageWootters {
    /// Build from a real symmetric system Hamiltonian.
    ///
    /// Diagonalizes `H_R` **once** here; every later `schrodinger_state(t)` /
    /// `conditional_state(t)` reuses the cached spectrum and the cached static
    /// state, so no per-`t` call re-runs the eigensolver or rebuilds the
    /// `dim²`-length global vector.
    pub fn new(h_r: RealMatrix) -> Self {
        let (energies, vecs) = h_r.symmetric_eigen();
        let dim = h_r.n;

        // P2: build the t-independent static state |Ψ> once.
        let inv = 1.0 / (dim as f64).sqrt();
        let mut psi_static = vec![Complex::ZERO; dim * dim];
        for k in 0..dim {
            for r in 0..dim {
                psi_static[k * dim + r] = Complex::real(inv * vecs.get(r, k));
            }
        }

        // Reference state |ψ_0> = Σ_k |E_k> as a complex vector (P1 cache).
        let psi0: Vec<Complex> = (0..dim)
            .map(|r| {
                let mut acc = 0.0;
                for k in 0..dim {
                    acc += vecs.get(r, k);
                }
                Complex::real(acc)
            })
            .collect();

        PageWootters {
            h_r,
            energies,
            vecs,
            dim,
            psi_static,
            psi0,
        }
    }

    /// The reference state `|ψ_0> = Σ_k |E_k>` (equal superposition of energy
    /// eigenstates) that the emergent dynamics evolve.
    pub fn reference_state(&self) -> Vec<Complex> {
        let d = self.dim;
        (0..d)
            .map(|r| {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += self.vecs.get(r, k);
                }
                Complex::real(acc)
            })
            .collect()
    }

    /// The clock Hamiltonian `H_C = diag(-E_k)`.
    pub fn clock_hamiltonian(&self) -> RealMatrix {
        let neg: Vec<f64> = self.energies.iter().map(|e| -e).collect();
        RealMatrix::diag(&neg)
    }

    /// The globally static entangled state `|Ψ>` (length `dim²`, `C ⊗ R` order).
    ///
    /// `t`-independent; computed once in [`PageWootters::new`] and returned from
    /// the cache here (P2). Cheap to clone for callers that need to own it.
    pub fn global_static_state(&self) -> Vec<Complex> {
        self.psi_static.clone()
    }

    /// The clock pointer (bra) for reading "time `t`": `|t>_C = Σ_k e^{iE_k t}|k>`.
    pub fn clock_pointer(&self, t: f64) -> Vec<Complex> {
        self.energies
            .iter()
            .map(|&e| Complex::phase(e * t))
            .collect()
    }

    /// Conditional state of the rest sector when the clock reads `t`, normalized.
    /// This is the *emergent* evolved state — derived purely from a static `|Ψ>`.
    ///
    /// Conditions the **cached** static state (P2) on the clock pointer; no `dim²`
    /// vector is materialized per call.
    pub fn conditional_state(&self, t: f64) -> Vec<Complex> {
        let bra = self.clock_pointer(t);
        let raw = condition_on_clock(&self.psi_static, &bra, self.dim);
        normalized(&raw)
    }

    /// The ordinary Schrödinger-evolved reference state `e^{-iH_R t}|ψ_0>`,
    /// normalized — what the conditional state must reproduce.
    ///
    /// P1: evolves directly in the **cached** energy eigenbasis,
    /// `e^{-iH_R t}|ψ_0> = Σ_k e^{-iE_k t} ⟨E_k|ψ_0⟩ |E_k⟩`, which is `O(dim²)`
    /// per call and never re-diagonalizes `H_R` or forms a propagator matrix.
    pub fn schrodinger_state(&self, t: f64) -> Vec<Complex> {
        // theta = -t : U(t) = e^{-iH_R t}.
        let evolved = exp_i_apply_from_spectrum(&self.energies, &self.vecs, -t, &self.psi0);
        normalized(&evolved)
    }

    /// The from-scratch Schrödinger-evolved reference state — diagonalizes
    /// `H_R` afresh and forms the full propagator `U(t) = e^{-iH_R t}`. Kept as a
    /// reference path for callers (and tests) that want to validate the cached
    /// [`schrodinger_state`](Self::schrodinger_state) route against the
    /// independent `H`-only implementation.
    pub fn schrodinger_state_from_scratch(&self, t: f64) -> Vec<Complex> {
        let u = schrodinger_propagator(&self.h_r, t);
        let evolved = u.matvec(&self.reference_state());
        normalized(&evolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::fidelity;

    fn sample_h() -> RealMatrix {
        // A non-trivial symmetric 3-level Hamiltonian.
        RealMatrix::from_fn(3, |r, c| if r == c { (r as f64) - 1.0 } else { 0.35 })
    }

    #[test]
    fn evolution_emerges_from_static_state() {
        let pw = PageWootters::new(sample_h());
        for &t in &[0.0, 0.5, 1.3, 2.7, -1.1] {
            let cond = pw.conditional_state(t);
            let schro = pw.schrodinger_state(t);
            let f = fidelity(&cond, &schro);
            assert!(f > 1.0 - 1e-8, "t={t}: fidelity {f} too low");
        }
    }

    #[test]
    fn distinct_times_give_distinct_states() {
        let pw = PageWootters::new(sample_h());
        let a = pw.conditional_state(0.0);
        let b = pw.conditional_state(1.5);
        // Generic Hamiltonian: states at different clock readings differ.
        assert!(fidelity(&a, &b) < 0.999);
    }

    /// P1 correctness gate: the cached-eigenbasis `schrodinger_state` must agree
    /// component-for-component with the from-scratch propagator path. Both are
    /// normalized identically, so this is an exact-up-to-roundoff equality (not
    /// merely a fidelity / global-phase match).
    #[test]
    fn cached_evolution_equals_from_scratch_propagator() {
        let pw = PageWootters::new(sample_h());
        for &t in &[0.0, 0.5, 1.3, 2.7, -1.1, -3.4] {
            let cached = pw.schrodinger_state(t);
            let scratch = pw.schrodinger_state_from_scratch(t);
            assert_eq!(cached.len(), scratch.len());
            for (a, b) in cached.iter().zip(&scratch) {
                assert!(
                    (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                    "t={t}: cached {a:?} != scratch {b:?}"
                );
            }
        }
    }
}
