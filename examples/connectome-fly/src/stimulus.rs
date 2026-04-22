//! Deterministic stimulus stubs.
//!
//! ADR-154 §3(3): embodiment is deferred. This module injects
//! deterministic time-varying currents into designated sensory neurons
//! in place of a closed-loop body. A `Stimulus` is a *reproducible
//! schedule of current-injection events*, not a process; the engine
//! consumes it directly.

use crate::connectome::NeuronId;

/// One scheduled current-injection event. When the engine drains this
/// event from its queue it is converted into a direct `g_exc` kick on
/// `target` (sign-preserving; see `lif::Engine::run_with`).
#[derive(Copy, Clone, Debug)]
pub struct CurrentInjection {
    /// Simulation time (ms) at which the injection takes effect.
    pub t_ms: f32,
    /// Target neuron.
    pub target: NeuronId,
    /// Charge contribution (pA-equivalent). Positive drives the neuron
    /// toward spiking; negative hyperpolarizes.
    pub charge_pa: f32,
}

/// A deterministic schedule of current injections.
#[derive(Debug, Default, Clone)]
pub struct Stimulus {
    events: Vec<CurrentInjection>,
}

impl Stimulus {
    /// An empty schedule.
    pub fn empty() -> Self {
        Self { events: Vec::new() }
    }

    /// Iterate all events in insertion order.
    pub fn events(&self) -> &[CurrentInjection] {
        &self.events
    }

    /// Push one event.
    pub fn push(&mut self, ev: CurrentInjection) {
        self.events.push(ev);
    }

    /// Build a Poisson-like deterministic pulse train.
    ///
    /// Injects `amplitude_pa` into each neuron in `targets` at regular
    /// 1/rate_hz intervals within `[onset_ms, onset_ms + duration_ms]`.
    /// Deterministic — no RNG — so replay is exact.
    pub fn pulse_train(
        targets: &[NeuronId],
        onset_ms: f32,
        duration_ms: f32,
        amplitude_pa: f32,
        rate_hz: f32,
    ) -> Self {
        let mut s = Self::empty();
        if targets.is_empty() || rate_hz <= 0.0 || duration_ms <= 0.0 {
            return s;
        }
        let dt = 1000.0 / rate_hz;
        let mut t = onset_ms;
        let end = onset_ms + duration_ms;
        let n = targets.len() as f32;
        let mut k: usize = 0;
        while t <= end {
            // Rotate through targets to keep injection per-pulse small
            // and per-neuron smooth.
            let offset = (k as f32 / n).fract() * dt;
            for (i, id) in targets.iter().enumerate() {
                let t_i = t + (i as f32 / n) * dt * 0.5 + offset;
                s.events.push(CurrentInjection {
                    t_ms: t_i,
                    target: *id,
                    charge_pa: amplitude_pa,
                });
            }
            t += dt;
            k += 1;
        }
        s
    }

    /// Build a single-shot constant-current injection over a window.
    ///
    /// Useful for the constructed-collapse test: push a large
    /// synchronous pulse into a known subset to force a coherence-drop.
    pub fn step(
        targets: &[NeuronId],
        onset_ms: f32,
        duration_ms: f32,
        amplitude_pa: f32,
        steps: u32,
    ) -> Self {
        let mut s = Self::empty();
        if steps == 0 || duration_ms <= 0.0 {
            return s;
        }
        let dt = duration_ms / steps as f32;
        for k in 0..steps {
            let t = onset_ms + k as f32 * dt;
            for id in targets {
                s.events.push(CurrentInjection {
                    t_ms: t,
                    target: *id,
                    charge_pa: amplitude_pa,
                });
            }
        }
        s
    }

    /// Combine two schedules.
    pub fn combined(mut a: Stimulus, b: Stimulus) -> Stimulus {
        a.events.extend(b.events);
        a
    }

    /// Total number of injection events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// `true` iff the schedule contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_train_is_deterministic() {
        let targets = vec![NeuronId(0), NeuronId(1), NeuronId(2)];
        let a = Stimulus::pulse_train(&targets, 10.0, 20.0, 30.0, 100.0);
        let b = Stimulus::pulse_train(&targets, 10.0, 20.0, 30.0, 100.0);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.events().iter().zip(b.events()) {
            assert_eq!(x.t_ms.to_bits(), y.t_ms.to_bits());
            assert_eq!(x.target, y.target);
            assert_eq!(x.charge_pa.to_bits(), y.charge_pa.to_bits());
        }
    }
}
