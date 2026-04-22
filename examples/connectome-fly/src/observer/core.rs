//! `Observer`: spike log, population-rate binning, and Fiedler
//! coherence-collapse detector.

use std::collections::VecDeque;

use crate::connectome::NeuronId;
use crate::lif::Spike;

use super::eigensolver::{approx_fiedler_power, jacobi_symmetric};
use super::incremental_fiedler::IncrementalCofireAccumulator;
use super::report::{CoherenceEvent, Report};
use super::sparse_fiedler::sparse_fiedler;

/// Active-neuron threshold above which the observer dispatches to the
/// sparse-Lanczos Fiedler path. Kept at 1024 per commit-9 measurement
/// (see ADR-154 §16 update). A speculative drop to 96 to route the
/// saturated N=1024 detector onto the sparse path measured a **3×
/// regression** (20.1 s vs 6.75 s on `lif_throughput_n_1024`): the
/// sparse path's HashMap accumulation and SparseGraph canonicalisation
/// add more overhead at n≈1024 than they save by skipping the dense
/// O(n²) Laplacian build. The sparse path is a scale win (memory +
/// time at n ≥ 10 000) not a demo-size speed win. The real saturated-
/// regime lever is adaptive detect cadence or an incremental Fiedler
/// accumulator, not threshold tuning.
const SPARSE_FIEDLER_N_THRESHOLD: usize = 1024;

/// Rolling observer: records spikes, maintains a co-firing window,
/// runs the Fiedler detector, and produces a final report.
pub struct Observer {
    num_neurons: u32,
    spikes: Vec<Spike>,
    // Fiedler detector state.
    window_ms: f32,
    cofire_window: VecDeque<Spike>,
    /// Incremental `BTreeMap<(NeuronId, NeuronId), u32>` of
    /// τ-coincident pair counts inside `cofire_window`. Replaces the
    /// O(S²) per-detect pair sweep — ADR-154 §16 lever 3.
    cofire_accum: IncrementalCofireAccumulator,
    last_detect_ms: f32,
    detect_every_ms: f32,
    baseline: RollingStats,
    warmup_samples: u32,
    threshold_factor: f32,
    events: Vec<CoherenceEvent>,
    // Population-rate binning.
    bin_ms: f32,
    t_end_hint_ms: f32,
}

/// Welford's running mean / std tracker.
#[derive(Default)]
struct RollingStats {
    n: u32,
    mean: f32,
    m2: f32,
}

impl RollingStats {
    fn push(&mut self, x: f32) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f32;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    fn std(&self) -> f32 {
        if self.n < 2 {
            0.0
        } else {
            (self.m2 / (self.n - 1) as f32).sqrt()
        }
    }
}

impl Observer {
    /// Default detector parameters: 50 ms co-firing window, detect
    /// every 5 ms, 20 samples warmup, threshold 2·std.
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons: num_neurons as u32,
            spikes: Vec::with_capacity(1 << 14),
            window_ms: 50.0,
            cofire_window: VecDeque::with_capacity(1 << 14),
            cofire_accum: IncrementalCofireAccumulator::new(),
            last_detect_ms: 0.0,
            detect_every_ms: 5.0,
            baseline: RollingStats::default(),
            warmup_samples: 20,
            threshold_factor: 2.0,
            events: Vec::new(),
            bin_ms: 5.0,
            t_end_hint_ms: 0.0,
        }
    }

    /// Override coherence-detector parameters.
    pub fn with_detector(
        mut self,
        window_ms: f32,
        detect_every_ms: f32,
        warmup_samples: u32,
        threshold_factor: f32,
    ) -> Self {
        self.window_ms = window_ms;
        self.detect_every_ms = detect_every_ms;
        self.warmup_samples = warmup_samples;
        self.threshold_factor = threshold_factor;
        self
    }

    /// Number of coherence events detected so far.
    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    /// Total spikes ingested.
    pub fn num_spikes(&self) -> usize {
        self.spikes.len()
    }

    /// Raw spike list.
    pub fn spikes(&self) -> &[Spike] {
        &self.spikes
    }

    /// Adaptive detect interval: under sustained saturated firing the
    /// Fiedler value barely changes between consecutive 5 ms detects,
    /// and each detect is O(n²) in window spikes + O(n²)–O(n³) in the
    /// Laplacian eigendecomposition. Backing off to 20 ms in saturation
    /// cuts the detector's share of wallclock 4× without losing any
    /// observable coherence event that AC-4's ≥ 50 ms strict-lead
    /// bound cares about (a 20 ms cadence still gives ≥ 2 detects
    /// inside any 50 ms lead window). See ADR-154 §16.
    ///
    /// Saturation signal: total spikes in the sliding co-firing window
    /// divided by window size exceeds 100 Hz average per neuron. At
    /// the default 50 ms window with N neurons, that threshold is
    /// `5 × N` spikes in the window.
    fn current_detect_interval_ms(&self) -> f32 {
        let saturation_spikes = (self.num_neurons as usize).saturating_mul(5);
        if self.cofire_window.len() > saturation_spikes {
            // 4× backoff under saturation. Matches AC-4 §8.3's
            // constructed-collapse test envelope (markers at t≥500 ms;
            // constructed collapses span > 60 ms, so a 20 ms cadence
            // still catches any ≥50 ms pre-marker event).
            (self.detect_every_ms * 4.0)
                .min(20.0)
                .max(self.detect_every_ms)
        } else {
            self.detect_every_ms
        }
    }

    /// Called by the engine on every spike emission.
    ///
    /// Order of operations matters for the incremental accumulator:
    ///
    /// 1. Append `s` to `cofire_window`.
    /// 2. Accumulator `push(s, …)` paired against the existing window
    ///    contents (excluding the just-pushed `s`). This adds edge
    ///    counts for every τ-coincident prior spike.
    /// 3. Expire-from-front: for each popped spike `q`, accumulator
    ///    `expire(q, remaining_window)` — decrements edge counts for
    ///    every τ-coincident remaining spike. `q` is always the
    ///    oldest spike in the window, so the τ-band it paired against
    ///    is near the front; walking forward from the new front lets
    ///    `expire` break out as soon as it leaves the band.
    /// 4. Detect. The accumulator is now exactly the pair-count state
    ///    implied by the current `cofire_window`, and
    ///    `compute_fiedler` reads it directly.
    pub fn on_spike(&mut self, s: Spike) {
        self.spikes.push(s);
        self.cofire_window.push_back(s);
        self.t_end_hint_ms = self.t_end_hint_ms.max(s.t_ms);

        // Incremental push: pair `s` against every prior window spike
        // within τ. `len() - 1` is safe because we just pushed `s`.
        let prior_len = self.cofire_window.len() - 1;
        self.cofire_accum
            .push(s, self.cofire_window.iter().take(prior_len));

        // Expire spikes that slid out of the window, and decrement
        // their pair counts against the remaining window contents.
        let cutoff = s.t_ms - self.window_ms;
        while let Some(front) = self.cofire_window.front() {
            if front.t_ms < cutoff {
                let q = self.cofire_window.pop_front().expect("front just checked");
                self.cofire_accum.expire(q, self.cofire_window.iter());
            } else {
                break;
            }
        }

        let interval = self.current_detect_interval_ms();
        if s.t_ms - self.last_detect_ms >= interval {
            self.last_detect_ms = s.t_ms;
            self.detect(s.t_ms);
        }
    }

    fn detect(&mut self, now_ms: f32) {
        let fiedler = self.compute_fiedler();
        if fiedler.is_nan() {
            return;
        }
        let (mean, std) = (self.baseline.mean, self.baseline.std());
        if self.baseline.n >= self.warmup_samples && std > 1e-6 {
            let drop = mean - fiedler;
            if drop > self.threshold_factor * std {
                let w = self.cofire_window.len() as f32;
                let pop_hz = (w / self.num_neurons as f32) / (self.window_ms / 1000.0);
                self.events.push(CoherenceEvent {
                    t_ms: now_ms,
                    fiedler,
                    baseline_mean: mean,
                    baseline_std: std,
                    population_rate_hz: pop_hz,
                });
            }
        }
        self.baseline.push(fiedler);
    }

    /// Fiedler value of the co-firing-window Laplacian.
    ///
    /// The adjacency is read from the incremental pair-count
    /// accumulator rather than re-derived by an `O(S²)` pair sweep on
    /// every call. See `IncrementalCofireAccumulator` for the update
    /// rules that keep the map consistent with `cofire_window`.
    fn compute_fiedler(&self) -> f32 {
        if self.cofire_window.len() < 2 {
            return f32::NAN;
        }
        let mut active: Vec<NeuronId> = self.cofire_window.iter().map(|s| s.neuron).collect();
        active.sort();
        active.dedup();
        let n = active.len();
        if n < 2 {
            return f32::NAN;
        }
        // Dispatch to the sparse shifted-power-iteration path above
        // the dense-matrix ceiling — avoids the O(n²) adjacency /
        // Laplacian allocation below. Threshold is 1024 so existing
        // demo-scale runs (N=1024 per ADR-154 §3) stay on the dense
        // path and AC-1 remains bit-exact vs head. The sparse path
        // still reconstructs edges from the live window rather than
        // the accumulator; that is a separate refactor (`snapshot_sparse`
        // is exposed on the accumulator for it) and is out of scope
        // for this lever.
        if n > SPARSE_FIEDLER_N_THRESHOLD {
            return sparse_fiedler(&active, &self.cofire_window, SPARSE_FIEDLER_N_THRESHOLD);
        }
        // Dense path: read the `n × n` adjacency directly from the
        // incremental accumulator instead of re-sweeping every pair
        // in the window. This replaces the O(S²) pair loop with an
        // O(|edges| · log n) map traversal.
        let a = self.cofire_accum.snapshot_adjacency(&active);
        let mut l = vec![0.0_f32; n * n];
        for i in 0..n {
            let mut d = 0.0_f32;
            for j in 0..n {
                d += a[i * n + j];
                if i != j {
                    l[i * n + j] = -a[i * n + j];
                }
            }
            l[i * n + i] = d;
        }
        if n <= 96 {
            let mut sorted = jacobi_symmetric(&l, n);
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for v in &sorted {
                if *v > 1e-6 {
                    return *v;
                }
            }
            return 0.0;
        }
        approx_fiedler_power(&a, n)
    }

    /// Finalize the run and produce a report. Keeps `&self` so the
    /// observer can be re-queried.
    pub fn finalize(&self) -> Report {
        let (pop_rate, pop_t) = self.population_rate_trace();
        let mean_rate = if pop_rate.is_empty() {
            0.0
        } else {
            pop_rate.iter().sum::<f32>() / pop_rate.len() as f32
        };
        let mut events = self.events.clone();
        events.sort_by(|a, b| {
            let da = a.baseline_mean - a.fiedler;
            let db = b.baseline_mean - b.fiedler;
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        Report {
            total_spikes: self.spikes.len() as u64,
            population_rate_hz: pop_rate,
            population_rate_t_ms: pop_t,
            coherence_events: events,
            mean_population_rate_hz: mean_rate,
            num_neurons: self.num_neurons,
            t_end_ms: self.t_end_hint_ms,
        }
    }

    fn population_rate_trace(&self) -> (Vec<f32>, Vec<f32>) {
        if self.spikes.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let t_max = self.t_end_hint_ms.max(self.spikes.last().unwrap().t_ms);
        let n_bins = (t_max / self.bin_ms).ceil() as usize + 1;
        let mut counts = vec![0_u32; n_bins];
        for s in &self.spikes {
            let i = (s.t_ms / self.bin_ms) as usize;
            if i < counts.len() {
                counts[i] += 1;
            }
        }
        let bin_s = self.bin_ms / 1000.0;
        let n = self.num_neurons as f32;
        let rate: Vec<f32> = counts.iter().map(|c| *c as f32 / (bin_s * n)).collect();
        let ts: Vec<f32> = (0..rate.len())
            .map(|i| (i as f32 + 0.5) * self.bin_ms)
            .collect();
        (rate, ts)
    }
}
