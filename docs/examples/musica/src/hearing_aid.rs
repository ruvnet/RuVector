//! Binaural hearing aid streaming speech enhancer.
//!
//! Low-latency (<8ms) speech-in-noise enhancement using:
//! - Rolling graph over 4-6 frames at 8ms frame size, 4ms hop
//! - Binaural features: ILD, IPD, IC (interaural coherence)
//! - Graph Laplacian spectral clustering (Fiedler vector via power iteration)
//! - Dynamic mincut refinement for boundary stability
//! - Speech/noise seed priors (voicing, harmonicity, frontness, modulation)
//! - Soft mask generation with temporal smoothing
//! - Audiogram-based gain shaping post-separation

use ruvector_mincut::prelude::*;
use std::f64::consts::PI;

/// Audiogram: hearing thresholds per frequency.
#[derive(Debug, Clone)]
pub struct Audiogram {
    /// Frequencies in Hz.
    pub frequencies: Vec<f64>,
    /// Hearing loss in dB HL at each frequency.
    pub gains_db: Vec<f64>,
}

impl Default for Audiogram {
    fn default() -> Self {
        // Mild sloping high-frequency loss (typical presbycusis)
        Self {
            frequencies: vec![250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
            gains_db: vec![10.0, 15.0, 20.0, 30.0, 40.0, 50.0],
        }
    }
}

impl Audiogram {
    /// Interpolate gain at a given frequency.
    pub fn gain_at(&self, freq: f64) -> f64 {
        if self.frequencies.is_empty() {
            return 0.0;
        }
        if freq <= self.frequencies[0] {
            return self.gains_db[0];
        }
        if freq >= *self.frequencies.last().unwrap() {
            return *self.gains_db.last().unwrap();
        }

        for i in 0..self.frequencies.len() - 1 {
            if freq >= self.frequencies[i] && freq <= self.frequencies[i + 1] {
                let t = (freq - self.frequencies[i])
                    / (self.frequencies[i + 1] - self.frequencies[i]);
                return self.gains_db[i] + t * (self.gains_db[i + 1] - self.gains_db[i]);
            }
        }
        0.0
    }
}

/// Hearing aid configuration.
#[derive(Debug, Clone)]
pub struct HearingAidConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Frame size in milliseconds.
    pub frame_size_ms: f64,
    /// Hop size in milliseconds.
    pub hop_size_ms: f64,
    /// Number of critical bands.
    pub num_bands: usize,
    /// Rolling window size in frames.
    pub window_frames: usize,
    /// Weight for speech seed score.
    pub speech_weight: f64,
    /// Weight for noise seed score.
    pub noise_weight: f64,
    /// Temporal mask smoothing factor (0=no smoothing, 1=frozen).
    pub mask_smoothing: f64,
    /// User audiogram.
    pub audiogram: Audiogram,
    /// Minimum frequency (Hz).
    pub freq_min: f64,
    /// Maximum frequency (Hz).
    pub freq_max: f64,
}

impl Default for HearingAidConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000.0,
            frame_size_ms: 8.0,
            hop_size_ms: 4.0,
            num_bands: 32,
            window_frames: 5,
            speech_weight: 1.0,
            noise_weight: 0.5,
            mask_smoothing: 0.3,
            audiogram: Audiogram::default(),
            freq_min: 100.0,
            freq_max: 8000.0,
        }
    }
}

/// Binaural features for one critical band in one frame.
#[derive(Debug, Clone, Copy)]
pub struct BinauralFeatures {
    /// Interaural level difference (dB).
    pub ild: f64,
    /// Interaural phase difference (radians).
    pub ipd: f64,
    /// Interaural coherence (0-1).
    pub ic: f64,
    /// Left magnitude.
    pub magnitude_l: f64,
    /// Right magnitude.
    pub magnitude_r: f64,
    /// Voicing probability (0-1).
    pub voicing: f64,
    /// Harmonicity score (0-1).
    pub harmonicity: f64,
    /// Center frequency of this band (Hz).
    pub center_freq: f64,
    /// Band index.
    pub band: usize,
}

/// Result of processing one frame.
#[derive(Debug, Clone)]
pub struct SeparationFrame {
    /// Speech mask per band [0, 1].
    pub mask: Vec<f64>,
    /// Speech confidence score per band.
    pub speech_score: Vec<f64>,
    /// MinCut value (structural witness).
    pub cut_value: f64,
    /// Processing latency in microseconds.
    pub latency_us: u64,
}

/// Rolling state for streaming processing.
pub struct StreamingState {
    /// Dynamic graph for incremental mincut updates.
    graph: DynamicGraph,
    /// Previous frame's partition labels per band (for temporal coherence).
    prev_labels: Vec<usize>,
    /// Previous frame's mask (for smoothing).
    prev_mask: Vec<f64>,
    /// Rolling buffer of left-channel frames.
    frame_buffer_l: Vec<Vec<f64>>,
    /// Rolling buffer of right-channel frames.
    frame_buffer_r: Vec<Vec<f64>>,
    /// Frame counter.
    pub frame_count: u64,
    /// Rolling window of binaural features [frame][band].
    feature_buffer: Vec<Vec<BinauralFeatures>>,
    /// Band center frequencies.
    band_freqs: Vec<f64>,
    /// FFT frame size in samples.
    frame_samples: usize,
    /// Hop size in samples.
    hop_samples: usize,
}

impl StreamingState {
    /// Create new streaming state.
    pub fn new(config: &HearingAidConfig) -> Self {
        let frame_samples = (config.sample_rate * config.frame_size_ms / 1000.0) as usize;
        let hop_samples = (config.sample_rate * config.hop_size_ms / 1000.0) as usize;

        // Compute band center frequencies (ERB scale)
        let band_freqs = erb_frequencies(config.num_bands, config.freq_min, config.freq_max);

        Self {
            graph: DynamicGraph::new(),
            prev_labels: vec![0; config.num_bands],
            prev_mask: vec![0.5; config.num_bands],
            frame_buffer_l: Vec::new(),
            frame_buffer_r: Vec::new(),
            frame_count: 0,
            feature_buffer: Vec::new(),
            band_freqs,
            frame_samples,
            hop_samples,
        }
    }

    /// Process one frame of binaural audio.
    ///
    /// Returns a speech mask and diagnostic info.
    /// `left` and `right` should be `frame_samples` long.
    pub fn process_frame(
        &mut self,
        left: &[f64],
        right: &[f64],
        config: &HearingAidConfig,
    ) -> SeparationFrame {
        let start = std::time::Instant::now();
        let num_bands = config.num_bands;

        // 1. Extract binaural features
        let features = extract_binaural_features(left, right, &self.band_freqs, config);

        // 2. Update rolling buffers
        self.feature_buffer.push(features.clone());
        self.frame_buffer_l.push(left.to_vec());
        self.frame_buffer_r.push(right.to_vec());
        if self.feature_buffer.len() > config.window_frames {
            self.feature_buffer.remove(0);
            self.frame_buffer_l.remove(0);
            self.frame_buffer_r.remove(0);
        }

        // 3. Build graph over rolling window
        let (edges, num_nodes) = build_streaming_graph(&self.feature_buffer, config);

        // 4. Compute Fiedler vector for speech/noise partitioning
        let fiedler = if num_nodes > 2 && !edges.is_empty() {
            compute_fiedler_vector(num_nodes, &edges)
        } else {
            vec![0.0; num_nodes]
        };

        // 5. Compute speech/noise seed scores
        let speech_scores = compute_speech_scores(&features, &fiedler, num_bands, config);

        // 6. Dynamic mincut refinement for boundary stability
        let (cut_value, refined_labels) = if !edges.is_empty() {
            refine_with_mincut(&edges, &speech_scores, &self.prev_labels, num_bands)
        } else {
            (0.0, self.prev_labels.clone())
        };
        self.prev_labels = refined_labels;

        // 7. Rebuild dynamic graph for next frame's incremental update
        self.graph = DynamicGraph::new();
        for i in 0..num_nodes {
            self.graph.add_vertex(i as u64);
        }
        for &(u, v, w) in &edges {
            let _ = self.graph.insert_edge(u as u64, v as u64, w);
        }

        // 8. Generate soft mask from speech scores
        let mut mask = speech_scores.clone();
        for m in &mut mask {
            *m = sigmoid(*m * 3.0); // Sharpen with sigmoid
        }

        // 9. Temporal smoothing
        let alpha = config.mask_smoothing;
        for (i, m) in mask.iter_mut().enumerate() {
            *m = alpha * self.prev_mask[i] + (1.0 - alpha) * *m;
        }
        self.prev_mask = mask.clone();

        // 10. Audiogram gain shaping
        apply_audiogram_gain(&mut mask, &self.band_freqs, &config.audiogram);

        self.frame_count += 1;
        let latency_us = start.elapsed().as_micros() as u64;

        SeparationFrame {
            mask,
            speech_score: speech_scores,
            cut_value,
            latency_us,
        }
    }

    /// Apply mask to binaural audio and return enhanced left/right.
    pub fn apply_mask(
        &self,
        left: &[f64],
        right: &[f64],
        mask: &[f64],
        config: &HearingAidConfig,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = left.len().min(right.len());
        let mut out_l = vec![0.0; n];
        let mut out_r = vec![0.0; n];

        // Simple band-wise application via DFT-like filtering
        // In production, use filterbank; here we approximate
        let num_bands = config.num_bands;
        let band_width = n / num_bands;

        if band_width == 0 {
            return (left.to_vec(), right.to_vec());
        }

        for b in 0..num_bands {
            let start = b * band_width;
            let end = ((b + 1) * band_width).min(n);
            let gain = mask[b.min(mask.len() - 1)];

            for i in start..end {
                out_l[i] = left[i] * gain;
                out_r[i] = right[i] * gain;
            }
        }

        (out_l, out_r)
    }
}

// ── Feature extraction ──────────────────────────────────────────────────

/// Extract binaural features from left/right audio frames.
fn extract_binaural_features(
    left: &[f64],
    right: &[f64],
    band_freqs: &[f64],
    config: &HearingAidConfig,
) -> Vec<BinauralFeatures> {
    let num_bands = config.num_bands;
    let n = left.len().min(right.len());

    band_freqs
        .iter()
        .enumerate()
        .map(|(b, &cf)| {
            // Simple band energy estimate (in production: filterbank)
            let band_start = (b * n) / num_bands;
            let band_end = ((b + 1) * n) / num_bands;

            let mut energy_l = 0.0;
            let mut energy_r = 0.0;
            let mut cross_lr = 0.0;

            for i in band_start..band_end {
                energy_l += left[i] * left[i];
                energy_r += right[i] * right[i];
                cross_lr += left[i] * right[i];
            }

            let band_len = (band_end - band_start).max(1) as f64;
            energy_l /= band_len;
            energy_r /= band_len;
            cross_lr /= band_len;

            let mag_l = energy_l.sqrt();
            let mag_r = energy_r.sqrt();

            // ILD
            let ild = if mag_r > 1e-10 {
                20.0 * (mag_l / mag_r).log10()
            } else {
                0.0
            };

            // IPD (approximate from cross-correlation lag)
            let ipd = if mag_l > 1e-10 && mag_r > 1e-10 {
                (cross_lr / (mag_l * mag_r)).acos().min(PI)
            } else {
                0.0
            };

            // Interaural coherence
            let ic = if energy_l > 1e-10 && energy_r > 1e-10 {
                (cross_lr / (energy_l * energy_r).sqrt()).abs().min(1.0)
            } else {
                0.0
            };

            // Voicing: simple energy-based proxy
            let voicing = if cf >= 80.0 && cf <= 3000.0 {
                ((mag_l + mag_r) * 2.0).min(1.0)
            } else {
                ((mag_l + mag_r) * 0.5).min(1.0)
            };

            // Harmonicity: high IC + speech band -> likely harmonic
            let harmonicity = if cf >= 100.0 && cf <= 4000.0 {
                ic * voicing
            } else {
                ic * 0.3
            };

            BinauralFeatures {
                ild,
                ipd,
                ic,
                magnitude_l: mag_l,
                magnitude_r: mag_r,
                voicing,
                harmonicity,
                center_freq: cf,
                band: b,
            }
        })
        .collect()
}

/// Build streaming graph over rolling feature window.
fn build_streaming_graph(
    buffer: &[Vec<BinauralFeatures>],
    config: &HearingAidConfig,
) -> (Vec<(usize, usize, f64)>, usize) {
    let num_bands = config.num_bands;
    let num_frames = buffer.len();
    let num_nodes = num_frames * num_bands;

    if num_nodes == 0 {
        return (vec![], 0);
    }

    let mut edges = Vec::new();
    let node = |f: usize, b: usize| f * num_bands + b;

    for f in 0..num_frames {
        for b in 0..num_bands {
            let feat = &buffer[f][b];

            // Spectral neighbors (same frame, adjacent bands)
            if b + 1 < num_bands {
                let feat2 = &buffer[f][b + 1];
                let w = spectral_similarity(feat, feat2);
                if w > 0.01 {
                    edges.push((node(f, b), node(f, b + 1), w));
                }
            }

            // Temporal neighbors (same band, adjacent frames)
            if f + 1 < num_frames {
                let feat2 = &buffer[f + 1][b];
                let w = temporal_similarity(feat, feat2);
                if w > 0.01 {
                    edges.push((node(f, b), node(f + 1, b), w));
                }
            }

            // Harmonic neighbors (same frame, 2x/3x frequency)
            for h in [2, 3] {
                let target_band = b * h;
                if target_band < num_bands {
                    let feat2 = &buffer[f][target_band];
                    let w = harmonic_similarity(feat, feat2) * 0.5;
                    if w > 0.01 {
                        edges.push((node(f, b), node(f, target_band), w));
                    }
                }
            }
        }
    }

    (edges, num_nodes)
}

/// Spectral similarity between adjacent bands.
fn spectral_similarity(a: &BinauralFeatures, b: &BinauralFeatures) -> f64 {
    let mag_sim = 1.0 - (a.magnitude_l - b.magnitude_l).abs().min(1.0);
    let ic_sim = 1.0 - (a.ic - b.ic).abs();
    0.5 * mag_sim + 0.5 * ic_sim
}

/// Temporal similarity between same band across frames.
fn temporal_similarity(a: &BinauralFeatures, b: &BinauralFeatures) -> f64 {
    let mag_sim = 1.0 - ((a.magnitude_l - b.magnitude_l).abs() + (a.magnitude_r - b.magnitude_r).abs()).min(1.0);
    let phase_sim = 1.0 - (a.ipd - b.ipd).abs() / PI;
    let ic_sim = 1.0 - (a.ic - b.ic).abs();
    0.4 * mag_sim + 0.3 * phase_sim.max(0.0) + 0.3 * ic_sim
}

/// Harmonic similarity between bands at integer frequency ratios.
fn harmonic_similarity(a: &BinauralFeatures, b: &BinauralFeatures) -> f64 {
    let ic_sim = (a.ic * b.ic).sqrt();
    let voicing_sim = (a.voicing * b.voicing).sqrt();
    0.5 * ic_sim + 0.5 * voicing_sim
}

/// Compute speech scores from features and Fiedler vector.
fn compute_speech_scores(
    features: &[BinauralFeatures],
    fiedler: &[f64],
    num_bands: usize,
    config: &HearingAidConfig,
) -> Vec<f64> {
    features
        .iter()
        .enumerate()
        .map(|(b, feat)| {
            // Speech prior: voicing + harmonicity + IC + frontness (low ILD)
            let voicing_score = feat.voicing;
            let harmonic_score = feat.harmonicity;
            let ic_score = feat.ic;
            let frontness = 1.0 - (feat.ild.abs() / 20.0).min(1.0);

            let speech_prior = 0.3 * voicing_score
                + 0.25 * harmonic_score
                + 0.25 * ic_score
                + 0.2 * frontness;

            // Fiedler contribution (for the most recent frame's nodes)
            let fiedler_score = if b < fiedler.len() {
                // Use sign of Fiedler vector — speech partition gets positive score
                fiedler[fiedler.len().saturating_sub(num_bands) + b.min(fiedler.len() - 1)]
                    .signum()
                    * 0.2
            } else {
                0.0
            };

            (config.speech_weight * speech_prior + fiedler_score)
                / (config.speech_weight + config.noise_weight)
        })
        .collect()
}

/// Refine partition using dynamic mincut for boundary stability.
///
/// Uses the current speech scores and previous labels as seed priors,
/// then runs mincut to find stable boundaries between speech and noise.
fn refine_with_mincut(
    edges: &[(usize, usize, f64)],
    speech_scores: &[f64],
    prev_labels: &[usize],
    num_bands: usize,
) -> (f64, Vec<usize>) {
    let cut_value = compute_cut_value(edges);

    // Derive labels from mincut partition
    let edge_list: Vec<(u64, u64, f64)> = edges
        .iter()
        .map(|&(u, v, w)| (u as u64, v as u64, w))
        .collect();

    let builder = MinCutBuilder::new().exact().with_edges(edge_list);
    let labels = match builder.build() {
        Ok(mc) => {
            let result = mc.min_cut();
            if let Some((side_a, _side_b)) = result.partition {
                let mut lab = vec![1usize; num_bands];
                for &nid in &side_a {
                    let band = (nid as usize) % num_bands;
                    if band < num_bands {
                        lab[band] = 0;
                    }
                }
                // Temporal coherence: bias toward previous labels
                for (i, l) in lab.iter_mut().enumerate() {
                    if i < prev_labels.len() && *l != prev_labels[i] {
                        // Only flip if speech score strongly disagrees
                        let score = if i < speech_scores.len() {
                            speech_scores[i]
                        } else {
                            0.5
                        };
                        if (score - 0.5).abs() < 0.1 {
                            *l = prev_labels[i]; // Keep previous label for ambiguous bins
                        }
                    }
                }
                lab
            } else {
                prev_labels.to_vec()
            }
        }
        Err(_) => prev_labels.to_vec(),
    };

    (cut_value, labels)
}

/// Compute mincut value as structural witness.
fn compute_cut_value(edges: &[(usize, usize, f64)]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }

    let edge_list: Vec<(u64, u64, f64)> = edges
        .iter()
        .map(|&(u, v, w)| (u as u64, v as u64, w))
        .collect();

    let builder = MinCutBuilder::new().exact().with_edges(edge_list);
    match builder.build() {
        Ok(mc) => mc.min_cut_value(),
        Err(_) => 0.0,
    }
}

/// Apply audiogram gain shaping to mask.
fn apply_audiogram_gain(mask: &mut [f64], band_freqs: &[f64], audiogram: &Audiogram) {
    for (i, m) in mask.iter_mut().enumerate() {
        if i < band_freqs.len() {
            let loss_db = audiogram.gain_at(band_freqs[i]);
            // Apply gain boost proportional to hearing loss
            let gain_linear = 10.0f64.powf(loss_db / 40.0); // Half-gain rule
            *m = (*m * gain_linear).min(1.0);
        }
    }
}

/// ERB (Equivalent Rectangular Bandwidth) frequency scale.
fn erb_frequencies(num_bands: usize, freq_min: f64, freq_max: f64) -> Vec<f64> {
    let erb_min = 21.4 * (0.00437 * freq_min + 1.0).log10();
    let erb_max = 21.4 * (0.00437 * freq_max + 1.0).log10();

    (0..num_bands)
        .map(|i| {
            let erb = erb_min + (i as f64 + 0.5) * (erb_max - erb_min) / num_bands as f64;
            (10.0f64.powf(erb / 21.4) - 1.0) / 0.00437
        })
        .collect()
}

/// Compute the Fiedler vector (2nd smallest eigenvector of graph Laplacian)
/// via power iteration on D^{-1}A, then deflate the trivial eigenvector.
///
/// Edges are `(u, v, weight)` with 0-indexed node IDs.
fn compute_fiedler_vector(n: usize, edges: &[(usize, usize, f64)]) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0; n];
    }

    // Build degree vector and sparse adjacency
    let mut degree = vec![0.0f64; n];
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for &(u, v, w) in edges {
        if u < n && v < n {
            degree[u] += w;
            degree[v] += w;
            adj[u].push((v, w));
            adj[v].push((u, w));
        }
    }

    let d_inv: Vec<f64> = degree
        .iter()
        .map(|&d| if d > 1e-12 { 1.0 / d } else { 0.0 })
        .collect();

    // Initialize with non-uniform vector orthogonal to the constant vector
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) - 0.5).collect();

    // Power iteration on D^{-1}A to find the second eigenvector
    for _ in 0..30 {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for &(j, w) in &adj[i] {
                s += w * v[j];
            }
            new_v[i] = d_inv[i] * s;
        }

        // Orthogonalize against constant vector (first eigenvector)
        let mean: f64 = new_v.iter().sum::<f64>() / n as f64;
        for x in &mut new_v {
            *x -= mean;
        }

        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut new_v {
                *x /= norm;
            }
        }

        v = new_v;
    }

    v
}

/// Sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binaural_frame(config: &HearingAidConfig) -> (Vec<f64>, Vec<f64>) {
        let n = (config.sample_rate * config.frame_size_ms / 1000.0) as usize;
        let left: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / config.sample_rate;
                0.5 * (2.0 * PI * 300.0 * t).sin() + 0.1 * (2.0 * PI * 1000.0 * t).sin()
            })
            .collect();
        let right: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / config.sample_rate;
                0.4 * (2.0 * PI * 300.0 * t).sin() + 0.15 * (2.0 * PI * 1000.0 * t).sin()
            })
            .collect();
        (left, right)
    }

    #[test]
    fn test_streaming_latency() {
        let config = HearingAidConfig::default();
        let mut state = StreamingState::new(&config);
        let (left, right) = make_binaural_frame(&config);

        // Process multiple frames and check latency
        let mut max_latency_us = 0u64;
        for _ in 0..20 {
            let result = state.process_frame(&left, &right, &config);
            max_latency_us = max_latency_us.max(result.latency_us);
        }

        // Target: <8ms = 8000us algorithmic delay
        // The actual processing should be much faster
        println!("Max frame latency: {}us", max_latency_us);
        assert!(
            max_latency_us < 8_000,
            "Latency {}us exceeds 8ms budget",
            max_latency_us
        );
    }

    #[test]
    fn test_binaural_preservation() {
        let config = HearingAidConfig::default();
        let mut state = StreamingState::new(&config);
        let (left, right) = make_binaural_frame(&config);

        let result = state.process_frame(&left, &right, &config);
        let (out_l, out_r) = state.apply_mask(&left, &right, &result.mask, &config);

        // ILD should be approximately preserved
        let orig_ild: f64 = left.iter().map(|x| x * x).sum::<f64>()
            / right.iter().map(|x| x * x).sum::<f64>().max(1e-10);
        let enhanced_ild: f64 = out_l.iter().map(|x| x * x).sum::<f64>()
            / out_r.iter().map(|x| x * x).sum::<f64>().max(1e-10);

        let ild_diff = (orig_ild - enhanced_ild).abs() / orig_ild.max(1e-10);
        assert!(
            ild_diff < 0.5,
            "ILD not preserved: orig={orig_ild:.2}, enhanced={enhanced_ild:.2}"
        );
    }

    #[test]
    fn test_speech_enhancement() {
        let config = HearingAidConfig::default();
        let mut state = StreamingState::new(&config);

        // Generate speech-like signal (strong harmonics, coherent)
        let n = (config.sample_rate * config.frame_size_ms / 1000.0) as usize;
        let speech_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / config.sample_rate;
                0.8 * (2.0 * PI * 200.0 * t).sin()
                    + 0.3 * (2.0 * PI * 400.0 * t).sin()
                    + 0.1 * (2.0 * PI * 600.0 * t).sin()
            })
            .collect();
        let speech_r = speech_l.iter().map(|&x| x * 0.9).collect::<Vec<_>>();

        // Process enough frames for stable output
        for _ in 0..10 {
            state.process_frame(&speech_l, &speech_r, &config);
        }

        let result = state.process_frame(&speech_l, &speech_r, &config);

        // Speech bands should have higher mask values
        let speech_band_avg: f64 = result.mask[..config.num_bands / 2]
            .iter()
            .sum::<f64>()
            / (config.num_bands / 2) as f64;

        assert!(
            speech_band_avg > 0.1,
            "Speech band mask too low: {speech_band_avg:.3}"
        );
    }

    #[test]
    fn test_audiogram_gain() {
        let audiogram = Audiogram {
            frequencies: vec![250.0, 1000.0, 4000.0],
            gains_db: vec![0.0, 20.0, 40.0],
        };

        assert!((audiogram.gain_at(250.0) - 0.0).abs() < 0.1);
        assert!((audiogram.gain_at(1000.0) - 20.0).abs() < 0.1);
        assert!((audiogram.gain_at(4000.0) - 40.0).abs() < 0.1);

        // Interpolation
        let gain_625 = audiogram.gain_at(625.0);
        assert!(gain_625 > 0.0 && gain_625 < 20.0, "Interpolated gain: {gain_625}");
    }

    #[test]
    fn test_erb_frequencies() {
        let freqs = erb_frequencies(32, 100.0, 8000.0);
        assert_eq!(freqs.len(), 32);
        assert!(freqs[0] > 100.0, "First band should be above minimum");
        assert!(*freqs.last().unwrap() < 8000.0, "Last band should be below maximum");

        // Should be monotonically increasing
        for w in freqs.windows(2) {
            assert!(w[1] > w[0], "ERB frequencies should increase: {} vs {}", w[0], w[1]);
        }
    }
}
