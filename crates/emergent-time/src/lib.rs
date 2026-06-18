//! # emergent-time
//!
//! A small, dependency-free Rust implementation of the **calculus of emergent
//! time**: time treated not as a background substance but as an *ordered
//! internal variable* that a subsystem computes by tracking irreversible change
//! against the rest of a closed system.
//!
//! Four physics formalisms are implemented, each individually correct and
//! verified by tests, plus a new agentic primitive:
//!
//! 1. [`wheeler_dewitt`] — the timeless global constraint `Ĥ|Ψ> = 0`. The state
//!    of a closed universe carries no external clock; time must be found inside.
//! 2. [`page_wootters`] — relational time. A globally *static* entangled state
//!    looks dynamic to an internal observer correlated with a clock subsystem;
//!    Schrödinger evolution emerges from conditioning, `ρ_R(τ)`.
//! 3. [`entropic`] — entropic time `τ_S = (S(λ) − S₀)/k`. The cold-atom toy
//!    universe: the speed of internal time tracks entropy production.
//! 4. [`thermal`] — Connes–Rovelli thermal time. The modular Hamiltonian
//!    `K = −ln ρ` generates a time flow `A(s) = e^{isK} A e^{-isK}` from the
//!    statistical state itself.
//! 5. [`agentic`] + [`structural_clock`] — internal time for agents and quantum
//!    machines, culminating in **Structural Proper Time**: the arc length of a
//!    system's worldline through its own state manifold.
//!
//! ## The recipe
//!
//! Every construction follows the same six steps:
//!
//! 1. choose a closed total system;
//! 2. split it into *clock* ⊗ *rest*;
//! 3. pick a monotone internal variable (energy phase, entropy, modular flow,
//!    structural distance…);
//! 4. define states conditioned on that variable;
//! 5. replace `d/dt` with `d/dτ`;
//! 6. recover ordinary physics when `τ` behaves like clock time.
//!
//! ```text
//!   time  ≠  background
//!   time  =  ordered change measured from inside the system
//! ```

pub mod adaptive;
pub mod agentic;
pub mod agentic_time;
pub mod complex;
pub mod complex_matrix;
pub mod entropic;
pub mod entropy;
pub mod page_wootters;
pub mod real_matrix;
pub mod state;
pub mod structural_clock;
pub mod thermal;
pub mod weight_learning;
pub mod wheeler_dewitt;
pub mod witness;

// Convenience re-exports of the most-used types.
pub use complex::Complex;
pub use complex_matrix::CMatrix;
pub use real_matrix::RealMatrix;

pub use adaptive::{adaptive_alarm_step, adaptive_early_warning_lead, PageHinkley};
pub use agentic::CausalTimeline;
pub use agentic_time::{AgentHealth, AgentState, AgenticTime, AgenticWeights};
pub use entropic::EntropicClock;
pub use page_wootters::PageWootters;
pub use structural_clock::{
    Clock, EntropyClock, Scenario, StateSnapshot, StructuralMetric, StructuralProperTime, WallClock,
};

#[cfg(test)]
mod integration_tests {
    //! Cross-module checks tying the formalisms together.
    use super::*;

    #[test]
    fn timeless_state_yields_emergent_evolution() {
        // Wheeler–DeWitt kernel state == Page–Wootters static state, and
        // conditioning it on the clock recovers Schrödinger dynamics.
        let h = RealMatrix::from_fn(4, |r, c| {
            if r == c {
                (r as f64) - 1.5
            } else if (r as i64 - c as i64).abs() == 1 {
                0.25
            } else {
                0.0
            }
        });
        let pw = PageWootters::new(h);

        // The global state solves the timeless equation.
        let j = wheeler_dewitt::bipartite_constraint(&pw.clock_hamiltonian(), &pw.h_r);
        let psi = pw.global_static_state();
        assert!(wheeler_dewitt::constraint_residual(&j, &psi) < 1e-8);

        // Yet evolution emerges from it.
        for &t in &[0.3, 1.1, 2.4] {
            let f = complex::fidelity(&pw.conditional_state(t), &pw.schrodinger_state(t));
            assert!(f > 1.0 - 1e-8);
        }
    }

    #[test]
    fn structural_time_beats_wall_time_on_all_axes() {
        let sc = Scenario::default();
        let traj = structural_clock::generate_scenario(&sc);
        let spt = StructuralProperTime::new(StructuralMetric::default());

        let wall = structural_clock::evaluate(&WallClock, &traj, sc.fail_index, 30, 4.0, 10);
        let structural = structural_clock::evaluate(&spt, &traj, sc.fail_index, 30, 4.0, 10);

        assert!(structural.lead > wall.lead);
        assert!(structural.compression_error < wall.compression_error);
        assert!(structural.causal_order_ok && wall.causal_order_ok);
    }
}
