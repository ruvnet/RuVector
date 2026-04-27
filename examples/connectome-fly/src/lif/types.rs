//! Public value types for the LIF kernel: biophysical parameters,
//! engine configuration, emitted-spike observation struct, error
//! enum.

use crate::connectome::NeuronId;

/// Per-neuron biophysical parameters (defaults from the research).
#[derive(Copy, Clone, Debug)]
pub struct NeuronParams {
    /// Membrane time constant (ms).
    pub tau_m: f32,
    /// Resting potential (mV).
    pub v_rest: f32,
    /// Reset potential (mV).
    pub v_reset: f32,
    /// Threshold (mV).
    pub v_thresh: f32,
    /// Membrane resistance (MΩ).
    pub r_m: f32,
    /// Refractory period (ms).
    pub tau_refrac: f32,
    /// Excitatory reversal (mV).
    pub e_exc: f32,
    /// Inhibitory reversal (mV).
    pub e_inh: f32,
    /// Excitatory conductance decay (ms).
    pub tau_syn_e: f32,
    /// Inhibitory conductance decay (ms).
    pub tau_syn_i: f32,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            tau_m: 10.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_thresh: -50.0,
            r_m: 10.0,
            tau_refrac: 2.0,
            e_exc: 0.0,
            e_inh: -80.0,
            tau_syn_e: 5.0,
            tau_syn_i: 10.0,
        }
    }
}

/// Engine configuration.
#[derive(Copy, Clone, Debug)]
pub struct EngineConfig {
    /// Integration time-step (ms).
    pub dt_ms: f32,
    /// Global synaptic weight scale.
    pub weight_gain: f32,
    /// Max scheduled events before the queue is considered blown.
    pub max_queue: usize,
    /// Use the SoA + bucketed timing-wheel optimized path.
    ///
    /// `false` = baseline (BinaryHeap + AoS); `true` = optimized.
    pub use_optimized: bool,
    /// Use the delay-sorted SoA CSR for spike delivery (Opt D from
    /// ADR-154 §3.2 step 10). Only effective when `use_optimized` is
    /// `true`; ignored on the baseline path. Opt-in (default `false`)
    /// so AC-1 bit-exactness at N=1024 on the shipped scalar / SIMD
    /// paths is untouched — the delay-sorted CSR reorders intra-row
    /// pushes into the timing wheel and so can change which tie-broken
    /// event wins within a bucket, which stays within the ~10 % cross-
    /// path tolerance the demonstrator already documents (README
    /// §Determinism; ADR-154 §15.1) but is NOT bit-exact vs the
    /// insertion-order CSR.
    pub use_delay_sorted_csr: bool,
    /// Per-neuron default params.
    pub params: NeuronParams,
    /// Engine RNG seed (unused in the deterministic path but kept so
    /// future stochastic variants preserve the determinism contract).
    pub seed: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dt_ms: 0.1,
            weight_gain: 0.9,
            max_queue: 8_000_000,
            use_optimized: true,
            use_delay_sorted_csr: false,
            params: NeuronParams::default(),
            seed: 0xDECA_FBAD_F00D_CAFE,
        }
    }
}

/// A spike observation emitted by the engine (consumed by `Observer`).
#[derive(Copy, Clone, Debug)]
pub struct Spike {
    /// Simulation time in ms.
    pub t_ms: f32,
    /// Neuron that fired.
    pub neuron: NeuronId,
}

/// Errors surfaced by the LIF engine.
#[derive(Debug, thiserror::Error)]
pub enum LifError {
    /// Event queue grew past `EngineConfig::max_queue`.
    #[error("event queue blown: {0} entries")]
    QueueBlown(usize),
}
