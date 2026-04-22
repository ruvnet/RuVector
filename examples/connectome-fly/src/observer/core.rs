//! `Observer`: spike log, population-rate binning, and Fiedler
//! coherence-collapse detector.

use std::collections::VecDeque;

use crate::connectome::NeuronId;
use crate::lif::Spike;

use super::eigensolver::{approx_fiedler_power, jacobi_symmetric};
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

    /// Called by the engine on every spike emission.
    pub fn on_spike(&mut self, s: Spike) {
        self.spikes.push(s);
        self.cofire_window.push_back(s);
        self.t_end_hint_ms = self.t_end_hint_ms.max(s.t_ms);
        let cutoff = s.t_ms - self.window_ms;
        while let Some(front) = self.cofire_window.front() {
            if front.t_ms < cutoff {
                self.cofire_window.pop_front();
            } else {
                break;
            }
        }
        if s.t_ms - self.last_detect_ms >= self.detect_every_ms {
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
        // path and AC-1 remains bit-exact vs head.
        if n > SPARSE_FIEDLER_N_THRESHOLD {
            return sparse_fiedler(&active, &self.cofire_window, SPARSE_FIEDLER_N_THRESHOLD);
        }
        let index_of = |id: NeuronId| -> Option<usize> { active.binary_search(&id).ok() };
        let tau = 5.0_f32;
        let mut a = vec![0.0_f32; n * n];
        let spikes: Vec<_> = self.cofire_window.iter().copied().collect();
        for (i, sa) in spikes.iter().enumerate() {
            let ai = match index_of(sa.neuron) {
                Some(x) => x,
                None => continue,
            };
            for sb in &spikes[i + 1..] {
                if (sb.t_ms - sa.t_ms).abs() > tau {
                    break;
                }
                if let Some(bi) = index_of(sb.neuron) {
                    if ai != bi {
                        a[ai * n + bi] += 1.0;
                        a[bi * n + ai] += 1.0;
                    }
                }
            }
        }
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
