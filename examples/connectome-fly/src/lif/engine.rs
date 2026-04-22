//! Engine — the hot loop of the LIF kernel.
//!
//! Holds two parallel back-ends (AoS + BinaryHeap baseline, SoA +
//! TimingWheel optimized) and a small active-set tracker that skips
//! quiescent neurons in the subthreshold step. See `queue` for the
//! event-queue primitives and `types` for configuration.

use std::collections::BinaryHeap;

use crate::connectome::{Connectome, NeuronId, Sign};
use crate::observer::Observer;
use crate::stimulus::Stimulus;

use super::queue::{SpikeEvent, TimingWheel};
use super::types::{EngineConfig, NeuronParams, Spike};

/// Array-of-structs neuron state. Baseline layout.
#[derive(Copy, Clone, Debug)]
struct NeuronStateAoS {
    v: f32,
    g_e: f32,
    g_i: f32,
    last_update_ms: f32,
    refrac_until_ms: f32,
}

/// Structure-of-arrays state (optimized path).
#[derive(Default)]
struct NeuronStateSoA {
    v: Vec<f32>,
    g_e: Vec<f32>,
    g_i: Vec<f32>,
    last_update_ms: Vec<f32>,
    refrac_until_ms: Vec<f32>,
}

impl NeuronStateSoA {
    fn new(n: usize, v_rest: f32) -> Self {
        Self {
            v: vec![v_rest; n],
            g_e: vec![0.0; n],
            g_i: vec![0.0; n],
            last_update_ms: vec![0.0; n],
            refrac_until_ms: vec![0.0; n],
        }
    }
}

/// Event-driven LIF engine over a `Connectome`.
pub struct Engine<'c> {
    conn: &'c Connectome,
    cfg: EngineConfig,
    // AoS path
    aos: Vec<NeuronStateAoS>,
    heap: BinaryHeap<SpikeEvent>,
    // SoA + wheel path
    soa: NeuronStateSoA,
    wheel: TimingWheel,
    /// Active-set membership: `true` iff the neuron needs subthreshold
    /// processing this tick. Optimized-path only.
    active_mask: Vec<bool>,
    /// Dense index of active neurons for O(|active|) iteration.
    active_list: Vec<u32>,
    clock: f32,
    tmp_events: Vec<SpikeEvent>,
    total_spikes: u64,
}

impl<'c> Engine<'c> {
    /// Build a new engine bound to `conn`.
    pub fn new(conn: &'c Connectome, cfg: EngineConfig) -> Self {
        let n = conn.num_neurons();
        let aos = vec![
            NeuronStateAoS {
                v: cfg.params.v_rest,
                g_e: 0.0,
                g_i: 0.0,
                last_update_ms: 0.0,
                refrac_until_ms: 0.0,
            };
            n
        ];
        let meta = conn.all_meta();
        let mut active_mask = vec![false; n];
        let mut active_list: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            if meta[i].bias_pa.abs() > 1e-6 {
                active_mask[i] = true;
                active_list.push(i as u32);
            }
        }
        Self {
            conn,
            cfg,
            aos,
            heap: BinaryHeap::with_capacity(1 << 16),
            soa: NeuronStateSoA::new(n, cfg.params.v_rest),
            wheel: TimingWheel::new(0.1, 32.0),
            active_mask,
            active_list,
            clock: 0.0,
            tmp_events: Vec::with_capacity(1 << 12),
            total_spikes: 0,
        }
    }

    /// Total spikes observed so far.
    pub fn total_spikes(&self) -> u64 {
        self.total_spikes
    }

    /// Current simulation time (ms).
    pub fn clock_ms(&self) -> f32 {
        self.clock
    }

    /// Run until `t_end_ms`, applying `stim` and reporting to `obs`.
    pub fn run_with(&mut self, stim: &Stimulus, obs: &mut Observer, t_end_ms: f32) {
        for inj in stim.events() {
            let ev = SpikeEvent {
                t_ms: inj.t_ms,
                post: inj.target,
                pre: inj.target,
                w: inj.charge_pa,
            };
            self.push_event(ev);
        }
        if self.cfg.use_optimized {
            self.run_opt(obs, t_end_ms);
        } else {
            self.run_base(obs, t_end_ms);
        }
    }

    // --- Baseline path: BinaryHeap + AoS ----------------------------
    fn run_base(&mut self, obs: &mut Observer, t_end_ms: f32) {
        while self.clock < t_end_ms {
            loop {
                let due = matches!(self.heap.peek(), Some(top) if top.t_ms <= self.clock + 1e-9);
                if !due {
                    break;
                }
                let ev = self.heap.pop().expect("peek");
                self.dispatch_base(ev, obs);
            }
            self.clock += self.cfg.dt_ms;
            self.subthreshold_base(obs);
        }
    }

    fn dispatch_base(&mut self, ev: SpikeEvent, obs: &mut Observer) {
        let i = ev.post.idx();
        let p = self.cfg.params;
        Self::drift_state(&mut self.aos[i], ev.t_ms, &p);
        if self.aos[i].refrac_until_ms > ev.t_ms {
            return;
        }
        if ev.w >= 0.0 {
            self.aos[i].g_e += ev.w;
        } else {
            self.aos[i].g_i += -ev.w;
        }
        if self.aos[i].v >= p.v_thresh {
            self.emit_spike_base(ev.post, ev.t_ms, obs);
        }
    }

    fn subthreshold_base(&mut self, obs: &mut Observer) {
        let p = self.cfg.params;
        let now = self.clock;
        let n = self.aos.len();
        for i in 0..n {
            let refrac = self.aos[i].refrac_until_ms;
            if refrac > now {
                continue;
            }
            Self::drift_state(&mut self.aos[i], now, &p);
            let bias = self.conn.all_meta()[i].bias_pa;
            self.aos[i].v += self.cfg.dt_ms * p.r_m * bias / p.tau_m;
            if self.aos[i].v >= p.v_thresh {
                self.emit_spike_base(NeuronId(i as u32), now, obs);
            }
        }
    }

    fn emit_spike_base(&mut self, id: NeuronId, t_ms: f32, obs: &mut Observer) {
        let p = self.cfg.params;
        let st = &mut self.aos[id.idx()];
        st.v = p.v_reset;
        st.refrac_until_ms = t_ms + p.tau_refrac;
        st.last_update_ms = t_ms;
        self.total_spikes += 1;
        obs.on_spike(Spike { t_ms, neuron: id });
        let wg = self.cfg.weight_gain;
        for s in self.conn.outgoing(id) {
            let signed = wg
                * s.weight
                * match s.sign {
                    Sign::Excitatory => 1.0,
                    Sign::Inhibitory => -1.0,
                };
            self.heap.push(SpikeEvent {
                t_ms: t_ms + s.delay_ms,
                post: s.post,
                pre: id,
                w: signed,
            });
        }
    }

    // --- Optimized path: timing-wheel + SoA + active-set ------------
    fn run_opt(&mut self, obs: &mut Observer, t_end_ms: f32) {
        while self.clock < t_end_ms {
            self.tmp_events.clear();
            self.wheel.drain_due(self.clock, &mut self.tmp_events);
            let mut buf = std::mem::take(&mut self.tmp_events);
            for ev in buf.drain(..) {
                self.dispatch_opt(ev, obs);
            }
            self.tmp_events = buf;
            self.clock += self.cfg.dt_ms;
            self.subthreshold_opt(obs);
        }
    }

    fn dispatch_opt(&mut self, ev: SpikeEvent, obs: &mut Observer) {
        let i = ev.post.idx();
        if self.soa.refrac_until_ms[i] > ev.t_ms {
            return;
        }
        if ev.w >= 0.0 {
            self.soa.g_e[i] += ev.w;
        } else {
            self.soa.g_i[i] += -ev.w;
        }
        if !self.active_mask[i] {
            self.active_mask[i] = true;
            self.active_list.push(i as u32);
        }
        if self.soa.v[i] >= self.cfg.params.v_thresh {
            self.emit_spike_opt(ev.post, ev.t_ms, obs);
        }
    }

    fn subthreshold_opt(&mut self, obs: &mut Observer) {
        let p = self.cfg.params;
        let now = self.clock;
        let dt = self.cfg.dt_ms;
        let meta = self.conn.all_meta();
        // Pre-compute per-tick exponential decay factors. Replaces
        // ~3 exp() calls per neuron per tick with three muls.
        let alpha_m = (-dt / p.tau_m).exp();
        let alpha_e = (-dt / p.tau_syn_e).exp();
        let alpha_i = (-dt / p.tau_syn_i).exp();
        let v_bias_factor = dt * p.r_m / p.tau_m;
        let quiescent_tol = 1e-4_f32;
        let mut write = 0_usize;
        let len = self.active_list.len();
        for read in 0..len {
            let idx = self.active_list[read];
            let i = idx as usize;
            if self.soa.refrac_until_ms[i] > now {
                self.active_list[write] = idx;
                write += 1;
                continue;
            }
            let v = self.soa.v[i];
            let ge = self.soa.g_e[i];
            let gi = self.soa.g_i[i];
            let i_syn = ge * (p.e_exc - v) + gi * (p.e_inh - v);
            self.soa.v[i] = p.v_rest
                + (v - p.v_rest) * alpha_m
                + p.r_m * i_syn * (1.0 - alpha_m)
                + v_bias_factor * meta[i].bias_pa;
            self.soa.g_e[i] = ge * alpha_e;
            self.soa.g_i[i] = gi * alpha_i;
            self.soa.last_update_ms[i] = now;
            if self.soa.v[i] >= p.v_thresh {
                self.emit_spike_opt(NeuronId(idx), now, obs);
            }
            let bias = meta[i].bias_pa;
            let still_active = bias.abs() > 1e-6
                || self.soa.g_e[i].abs() > quiescent_tol
                || self.soa.g_i[i].abs() > quiescent_tol
                || (self.soa.v[i] - p.v_rest).abs() > quiescent_tol;
            if still_active {
                self.active_list[write] = idx;
                write += 1;
            } else {
                self.active_mask[i] = false;
            }
        }
        self.active_list.truncate(write);
    }

    fn emit_spike_opt(&mut self, id: NeuronId, t_ms: f32, obs: &mut Observer) {
        let p = self.cfg.params;
        let i = id.idx();
        self.soa.v[i] = p.v_reset;
        self.soa.refrac_until_ms[i] = t_ms + p.tau_refrac;
        self.soa.last_update_ms[i] = t_ms;
        self.total_spikes += 1;
        obs.on_spike(Spike { t_ms, neuron: id });
        if !self.active_mask[i] {
            self.active_mask[i] = true;
            self.active_list.push(i as u32);
        }
        let wg = self.cfg.weight_gain;
        for s in self.conn.outgoing(id) {
            let signed = wg
                * s.weight
                * match s.sign {
                    Sign::Excitatory => 1.0,
                    Sign::Inhibitory => -1.0,
                };
            self.push_event(SpikeEvent {
                t_ms: t_ms + s.delay_ms,
                post: s.post,
                pre: id,
                w: signed,
            });
        }
    }

    // --- shared helpers ---------------------------------------------
    fn push_event(&mut self, ev: SpikeEvent) {
        if self.cfg.use_optimized {
            self.wheel.push(ev);
        } else {
            self.heap.push(ev);
        }
    }

    fn drift_state(st: &mut NeuronStateAoS, to_ms: f32, p: &NeuronParams) {
        let dt = (to_ms - st.last_update_ms).max(0.0);
        if dt <= 0.0 {
            return;
        }
        let alpha_m = (-dt / p.tau_m).exp();
        let alpha_e = (-dt / p.tau_syn_e).exp();
        let alpha_i = (-dt / p.tau_syn_i).exp();
        let i_syn = st.g_e * (p.e_exc - st.v) + st.g_i * (p.e_inh - st.v);
        st.v = p.v_rest + (st.v - p.v_rest) * alpha_m + p.r_m * i_syn * (1.0 - alpha_m);
        st.g_e *= alpha_e;
        st.g_i *= alpha_i;
        st.last_update_ms = to_ms;
    }
}
