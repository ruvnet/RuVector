//! Spike observer + Fiedler coherence-collapse detector + final
//! report type.
//!
//! Submodules:
//!
//! - `core`        — `Observer` and its public API.
//! - `report`      — serializable report + `CoherenceEvent`.
//! - `eigensolver` — Jacobi full-eigendecomposition for small windows
//!                   plus a shifted-power-iteration fallback for
//!                   larger ones.

pub mod core;
pub mod eigensolver;
pub mod report;

pub use core::Observer;
pub use report::{CoherenceEvent, Report};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::NeuronId;
    use crate::lif::Spike;

    #[test]
    fn empty_observer_report_is_safe() {
        let o = Observer::new(64);
        let r = o.finalize();
        assert_eq!(r.total_spikes, 0);
        assert!(r.coherence_events.is_empty());
    }

    #[test]
    fn coherence_detector_emits_on_constructed_collapse() {
        // Collapse here = the co-firing graph fragments from a
        // single well-connected cluster into two nearly-disjoint
        // halves. Fiedler value of the Laplacian drops sharply.
        let mut o = Observer::new(64).with_detector(50.0, 5.0, 3, 1.0);
        for k in 0..30 {
            let t = k as f32 * 10.0;
            for i in 0..16 {
                o.on_spike(Spike {
                    t_ms: t + i as f32 * 0.10,
                    neuron: NeuronId(i),
                });
            }
        }
        for k in 0..20 {
            let base = 300.0 + k as f32 * 10.0;
            for i in 0..8 {
                o.on_spike(Spike {
                    t_ms: base + i as f32 * 0.05,
                    neuron: NeuronId(i),
                });
            }
            for i in 8..16 {
                o.on_spike(Spike {
                    t_ms: base + 7.0 + (i - 8) as f32 * 0.05,
                    neuron: NeuronId(i),
                });
            }
        }
        let r = o.finalize();
        assert!(
            !r.coherence_events.is_empty(),
            "expected at least one coherence event after fragmentation"
        );
    }
}
