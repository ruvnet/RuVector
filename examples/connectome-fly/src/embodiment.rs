//! Embodiment ABI — the slot where a physics body sits between the
//! connectome's motor outputs and the connectome's sensory inputs.
//!
//! Application #10 from [`Connectome-OS/README.md`](../../README.md#part-3--exotic-needs-phase-2-or-phase-3-scaffolding)
//! ("Embodied fly navigation in VR") needs a physics body at the
//! perimeter of the simulation — a MuJoCo 3 process or similar —
//! that (a) consumes motor-neuron spikes as actuator commands and
//! (b) emits sensor activations (proprioception, vision, contact)
//! back onto designated sensory neurons.
//!
//! This module defines the ABI (the [`BodySimulator`] trait) and ships
//! two implementations:
//!
//! - [`StubBody`] — a deterministic null body. Consumes motor spikes,
//!   drops them on the floor, emits a programmable clock-driven
//!   current into the sensory population via the existing
//!   [`Stimulus`] type. Fully usable *today*; preserves AC-1
//!   determinism. This is what the Tier-1 demo runs with.
//! - [`MujocoBody`] — a panic-stub. The MuJoCo + NeuroMechFly bridge
//!   is Phase-3 work in the implementation plan; the stub documents
//!   the FFI shape and panics with an actionable diagnostic when
//!   anything tries to step it. Opt-in behind the `mujoco` Cargo
//!   feature (not yet wired — see `Cargo.toml` comments for the
//!   unblock sequence).
//!
//! The ABI is deliberately minimal so a Phase-3 drop-in is a single
//! struct replacement: same three methods, same determinism contract,
//! same units.
//!
//! **Determinism contract:** `BodySimulator::step` is a pure function
//! of `(prev_internal_state, motor_spikes_this_tick, dt_ms)`. Two
//! repeat runs from the same initial state + the same motor-spike
//! sequence produce identical sensor activations. The MuJoCo bridge
//! preserves this by seeding MuJoCo's internal RNG and gating FP
//! mode to strict; see `BodyStepLog` for per-step capture.

use crate::connectome::NeuronId;
use crate::lif::Spike;
use crate::stimulus::{CurrentInjection, Stimulus};

/// Per-tick body interface. Consumes motor spikes produced by the
/// simulated brain and returns sensor-activation current injections
/// the engine should deliver on the next tick.
pub trait BodySimulator {
    /// Step the body forward by `dt_ms` milliseconds, consuming
    /// `motor_spikes` (spikes emitted by motor-class neurons during
    /// this tick) and returning sensor-input current injections to
    /// schedule on the engine's next tick.
    ///
    /// `t_now_ms` is the simulation clock at the END of this step —
    /// the time the returned injections should be scheduled for.
    fn step(&mut self, t_now_ms: f32, dt_ms: f32, motor_spikes: &[Spike]) -> Vec<CurrentInjection>;

    /// Reset the body to its initial state. Called between trials so
    /// AC-5-style paired-trial runs see a clean body each time.
    fn reset(&mut self);

    /// Backend name — used by benches and diagnostic reports to keep
    /// numbers from stub and MuJoCo arms clearly separated.
    fn name(&self) -> &'static str;
}

// -----------------------------------------------------------------
// StubBody — the deterministic null body used by the Tier-1 demo
// -----------------------------------------------------------------

/// A body that produces a pre-baked `Stimulus` schedule ignoring
/// motor spikes entirely. Matches what `connectome-fly` already does
/// — a clock-driven current injection into the sensory population.
/// Used by the shipped demo and as the reference-determinism baseline
/// against which the MuJoCo body must agree bit-for-bit on identical
/// motor-spike inputs. (Bit-for-bit agreement is impossible in the
/// MuJoCo direction; the constraint only binds on motor-input-empty
/// steps.)
pub struct StubBody {
    stim: Stimulus,
}

impl StubBody {
    /// Wrap a pre-built `Stimulus` as a `BodySimulator`. The same
    /// injections are emitted on each matching time window
    /// regardless of motor output — i.e., open-loop drive.
    pub fn new(stim: Stimulus) -> Self {
        Self { stim }
    }
}

impl BodySimulator for StubBody {
    fn step(&mut self, t_now_ms: f32, dt_ms: f32, _motor_spikes: &[Spike]) -> Vec<CurrentInjection> {
        // Return every injection whose t_ms falls in the window
        // (t_now_ms - dt_ms, t_now_ms], matching what the engine
        // does internally. Empty if no injection is scheduled here.
        let t_from = t_now_ms - dt_ms;
        self.stim
            .events()
            .iter()
            .copied()
            .filter(|inj| inj.t_ms > t_from && inj.t_ms <= t_now_ms)
            .collect()
    }

    fn reset(&mut self) {
        // Stubs are stateless — injection schedule is time-indexed.
    }

    fn name(&self) -> &'static str {
        "stub-body"
    }
}

// -----------------------------------------------------------------
// MujocoBody — Phase-3 panic-stub, same shape as the Tier-2 contract
// -----------------------------------------------------------------

/// Phase-3 MuJoCo + NeuroMechFly bridge. **Panics on any step** until
/// the `cxx` bindings + NeuroMechFly v2 MJCF ingest land; documented
/// in [ADR-154 §13](https://github.com/ruvnet/RuVector/blob/research/connectome-ruvector/docs/adr/ADR-154-connectome-embodied-brain-example.md)
/// and [`docs/research/connectome-ruvector/04-embodiment.md`](https://github.com/ruvnet/RuVector/blob/research/connectome-ruvector/docs/research/connectome-ruvector/04-embodiment.md).
///
/// The stub exists today so downstream code that types against
/// `BodySimulator` can be written, tested, and shipped against the
/// deterministic [`StubBody`] while the MuJoCo integration is still
/// unlanded. Swap the trait-object at wire time.
pub struct MujocoBody {
    /// Placeholder for the MuJoCo `cxx` handle. Phase-3 makes this
    /// `mujoco::Handle` behind the `mujoco` feature; today it's
    /// `()` to preserve size-of-type across feature flips.
    _handle: (),
    /// Map from motor NeuronId to MuJoCo actuator id. Phase-3 builds
    /// this from the NeuroMechFly MJCF at ctor time; today it is
    /// empty.
    _motor_map: Vec<(NeuronId, u32)>,
    /// Map from MuJoCo sensor id to the NeuronId whose injection
    /// current encodes the sensor reading. Same empty-today story.
    _sensor_map: Vec<(u32, NeuronId)>,
}

impl MujocoBody {
    /// Construct a panic-stub. Takes the motor- and sensor-maps the
    /// Phase-3 body will need; they're stored for shape completeness
    /// but ignored by the stub's `step`.
    ///
    /// The function does NOT panic on construction — only on `step`
    /// — so downstream code can `Box<dyn BodySimulator>` against the
    /// type without tripping a runtime failure before the trial loop
    /// actually asks for a body step.
    pub fn new(motor_map: Vec<(NeuronId, u32)>, sensor_map: Vec<(u32, NeuronId)>) -> Self {
        Self {
            _handle: (),
            _motor_map: motor_map,
            _sensor_map: sensor_map,
        }
    }
}

impl BodySimulator for MujocoBody {
    fn step(
        &mut self,
        _t_now_ms: f32,
        _dt_ms: f32,
        _motor_spikes: &[Spike],
    ) -> Vec<CurrentInjection> {
        panic!(
            "MujocoBody::step: Phase-3 MuJoCo + NeuroMechFly bridge not yet \
             linked. See ADR-154 §13 'NeuroMechFly / MuJoCo body' and \
             docs/research/connectome-ruvector/04-embodiment.md for the \
             unblock sequence. Until it lands, use StubBody for \
             deterministic open-loop drive."
        );
    }

    fn reset(&mut self) {
        panic!(
            "MujocoBody::reset: Phase-3 MuJoCo + NeuroMechFly bridge not \
             yet linked. See ADR-154 §13 and docs/research/connectome-ruvector/04-embodiment.md."
        );
    }

    fn name(&self) -> &'static str {
        "mujoco-body-phase-3-stub"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lif::Spike;
    use crate::stimulus::CurrentInjection;

    #[test]
    fn stub_body_is_open_loop_and_deterministic() {
        let mut stim = Stimulus::empty();
        stim.push(CurrentInjection {
            t_ms: 5.0,
            target: NeuronId(0),
            charge_pa: 100.0,
        });
        stim.push(CurrentInjection {
            t_ms: 15.0,
            target: NeuronId(1),
            charge_pa: 90.0,
        });
        let mut a = StubBody::new(stim.clone());
        let mut b = StubBody::new(stim);

        // Ignore motor spikes; open-loop behaviour.
        let dummy = [Spike {
            t_ms: 3.0,
            neuron: NeuronId(42),
        }];
        let oa = a.step(10.0, 10.0, &dummy);
        let ob = b.step(10.0, 10.0, &[]);
        // Same time window, same schedule — identical output regardless
        // of motor input.
        assert_eq!(oa.len(), ob.len());
        for (x, y) in oa.iter().zip(ob.iter()) {
            assert_eq!(x.t_ms.to_bits(), y.t_ms.to_bits());
            assert_eq!(x.target, y.target);
            assert_eq!(x.charge_pa.to_bits(), y.charge_pa.to_bits());
        }
    }

    #[test]
    fn stub_body_windows_injections_by_time() {
        let mut stim = Stimulus::empty();
        for i in 0..5 {
            stim.push(CurrentInjection {
                t_ms: i as f32 * 10.0 + 5.0,
                target: NeuronId(i as u32),
                charge_pa: 100.0,
            });
        }
        let mut b = StubBody::new(stim);
        // Step from 0..10 ms should catch the injection at t=5.
        let out = b.step(10.0, 10.0, &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].target, NeuronId(0));
        // Step from 10..20 ms should catch t=15.
        let out = b.step(20.0, 10.0, &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].target, NeuronId(1));
    }

    #[test]
    fn mujoco_body_construction_does_not_panic() {
        // Shape-level sanity: downstream code can hold a `MujocoBody`
        // without tripping the Phase-3 panic until it actually calls
        // step / reset.
        let _b = MujocoBody::new(vec![(NeuronId(10), 0)], vec![(0, NeuronId(20))]);
    }

    #[test]
    #[should_panic(expected = "Phase-3")]
    fn mujoco_body_step_panics_with_actionable_diagnostic() {
        let mut b = MujocoBody::new(vec![], vec![]);
        b.step(1.0, 1.0, &[]);
    }

    #[test]
    fn body_simulator_is_object_safe() {
        // If this compiles the trait is object-safe — downstream
        // engine wiring can `Box<dyn BodySimulator>` polymorphically
        // across stub and MuJoCo arms.
        let _b: Box<dyn BodySimulator> = Box::new(StubBody::new(Stimulus::empty()));
    }
}
