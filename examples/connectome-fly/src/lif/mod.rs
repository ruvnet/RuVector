//! Event-driven leaky integrate-and-fire kernel.
//!
//! Two interchangeable back-ends live side-by-side:
//!
//! - **Baseline**: `BinaryHeap<SpikeEvent>` priority queue + AoS
//!   neuron state. Simple, `O(log N)` per event.
//! - **Optimized**: bucketed timing-wheel queue + SoA neuron state +
//!   active-set tracking for sparse subthreshold updates + per-tick
//!   `exp()` hoisting. Amortized `O(1)` per event in the wheel
//!   horizon.
//!
//! Selected by `EngineConfig::use_optimized` at construction time.
//! See `docs/research/connectome-ruvector/03-neural-dynamics.md` §2
//! for the biophysical model and `../../BENCHMARK.md` for the
//! measured speed-ups.

pub mod delay_csr;
pub mod engine;
pub mod queue;
#[cfg(feature = "simd")]
pub mod simd;
pub mod types;

pub use delay_csr::DelaySortedCsr;
pub use engine::Engine;
pub use queue::{SpikeEvent, TimingWheel};
pub use types::{EngineConfig, LifError, NeuronParams, Spike};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{Connectome, ConnectomeConfig};
    use crate::observer::Observer;
    use crate::stimulus::Stimulus;

    #[test]
    fn engine_runs_without_panic() {
        let conn = Connectome::generate(&ConnectomeConfig {
            num_neurons: 128,
            avg_out_degree: 12.0,
            ..ConnectomeConfig::default()
        });
        let mut eng = Engine::new(&conn, EngineConfig::default());
        let stim = Stimulus::pulse_train(conn.sensory_neurons(), 20.0, 30.0, 60.0, 50.0);
        let mut obs = Observer::new(conn.num_neurons());
        eng.run_with(&stim, &mut obs, 80.0);
        let r = obs.finalize();
        assert!(r.total_spikes < 10_000_000);
    }
}
