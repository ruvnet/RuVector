//! Serializable types emitted by the observer at end-of-run.

use serde::Serialize;

/// One coherence-drop event surfaced by the detector.
#[derive(Clone, Debug, Serialize)]
pub struct CoherenceEvent {
    /// Simulation time at detection (ms).
    pub t_ms: f32,
    /// Fiedler value at detection.
    pub fiedler: f32,
    /// Baseline mean at detection.
    pub baseline_mean: f32,
    /// Baseline standard deviation.
    pub baseline_std: f32,
    /// Population rate (spikes per neuron per second) at detection.
    pub population_rate_hz: f32,
}

/// Final demo report serializable to JSON.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    /// Total spikes over the full run.
    pub total_spikes: u64,
    /// Population-rate trace, one sample per 5 ms bin.
    pub population_rate_hz: Vec<f32>,
    /// Bin centre times (ms) for `population_rate_hz`.
    pub population_rate_t_ms: Vec<f32>,
    /// Top coherence events (most-negative Δ against baseline first).
    pub coherence_events: Vec<CoherenceEvent>,
    /// Mean population rate (Hz / neuron).
    pub mean_population_rate_hz: f32,
    /// Number of neurons in the simulation.
    pub num_neurons: u32,
    /// Simulated window (ms).
    pub t_end_ms: f32,
}
