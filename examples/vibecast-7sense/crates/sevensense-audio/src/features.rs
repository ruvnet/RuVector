//! Interpretable acoustic feature extraction.
//!
//! Perch embeddings are excellent for retrieval and opaque for explanation. This
//! module computes named spectral descriptors that a biologist would recognize --
//! centroid, spread, tonality, entropy, modulation -- from a linear-frequency power
//! spectrum, independent of the mel path used for model input.
//!
//! See ADR-011 for the rationale, including why these are not derived from
//! [`crate::MelSpectrogram`] (log-scaled, perceptually-spaced bins do not yield a
//! centroid in Hz).
//!
//! # Example
//!
//! ```
//! use sevensense_audio::features::{FeatureConfig, FeatureExtractor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = FeatureConfig::default();
//! let extractor = FeatureExtractor::new(config)?;
//!
//! // A 4 kHz tone at 32 kHz sample rate.
//! let samples: Vec<f32> = (0..32_000)
//!     .map(|i| (std::f32::consts::TAU * 4000.0 * i as f32 / 32_000.0).sin())
//!     .collect();
//!
//! let features = extractor.extract(&samples)?;
//! let centroid = features.summary.centroid_hz.mean;
//! assert!((centroid - 4000.0).abs() < 100.0, "centroid was {centroid}");
//! # Ok(())
//! # }
//! ```

use std::f32::consts::TAU;
use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};

use crate::AudioError;

/// Minimum number of frames required before modulation features are estimated.
///
/// Below roughly four modulation periods the FFT of the envelope is dominated by
/// windowing artifacts, so reporting a rate would be fabricating precision.
pub const MIN_FRAMES_FOR_MODULATION: usize = 32;

/// Lower bound of the modulation search band, in Hz.
const MODULATION_MIN_HZ: f32 = 2.0;

/// Upper bound of the modulation search band, in Hz, before Nyquist clamping.
const MODULATION_MAX_HZ: f32 = 200.0;

/// Floor applied to power values before taking logarithms.
const POWER_EPSILON: f32 = 1e-12;

/// Configuration for acoustic feature extraction.
///
/// Descriptor values depend on this configuration, so it is retained in
/// [`AcousticFeatures`] rather than left implicit -- comparing features computed
/// under different configurations is not meaningful.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// FFT window size in samples. Smaller than the mel path's window: these
    /// descriptors need time resolution more than frequency resolution.
    pub n_fft: usize,
    /// Hop between frames in samples. Matches the mel path so frame indices align.
    pub hop_length: usize,
    /// Sample rate of the input audio in Hz.
    pub sample_rate: u32,
    /// Lowest frequency included in analysis (Hz).
    pub f_min: f32,
    /// Highest frequency included in analysis (Hz).
    pub f_max: f32,
    /// Cumulative power fraction defining spectral rolloff (0.0 to 1.0).
    pub rolloff_percent: f32,
    /// Frames quieter than this are excluded from summary statistics (dBFS).
    pub energy_gate_db: f32,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 320,
            sample_rate: 32_000,
            f_min: 200.0,
            f_max: 16_000.0,
            rolloff_percent: 0.85,
            energy_gate_db: -60.0,
        }
    }
}

impl FeatureConfig {
    /// Frame rate in frames per second.
    #[must_use]
    pub fn frame_rate(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }

    /// Frequency resolution of one FFT bin in Hz.
    #[must_use]
    pub fn bin_width_hz(&self) -> f32 {
        self.sample_rate as f32 / self.n_fft as f32
    }

    fn validate(&self) -> Result<(), AudioError> {
        if self.n_fft < 2 {
            return Err(AudioError::invalid_data("n_fft must be at least 2"));
        }
        if self.hop_length == 0 {
            return Err(AudioError::invalid_data("hop_length must be non-zero"));
        }
        if self.sample_rate == 0 {
            return Err(AudioError::invalid_data("sample_rate must be non-zero"));
        }
        if !(self.f_min < self.f_max) {
            return Err(AudioError::invalid_data("f_min must be less than f_max"));
        }
        if !(self.rolloff_percent > 0.0 && self.rolloff_percent <= 1.0) {
            return Err(AudioError::invalid_data(
                "rolloff_percent must be in (0.0, 1.0]",
            ));
        }
        Ok(())
    }
}

/// Descriptors computed from a single analysis frame.
///
/// Normalized fields (`tonality`, `crest`, `entropy`) share a 0-1 range so a radar
/// plot can show them on common axes without per-axis rescaling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralFrame {
    /// Frame start time in milliseconds from the beginning of the input.
    pub time_ms: f64,
    /// Power-weighted mean frequency -- perceived "brightness" (Hz).
    pub centroid_hz: f32,
    /// Power-weighted standard deviation about the centroid -- bandwidth (Hz).
    pub spread_hz: f32,
    /// Third standardized moment of the power distribution. Positive means energy
    /// skews above the centroid.
    pub skewness: f32,
    /// Frequency below which `rolloff_percent` of the power lies (Hz).
    pub rolloff_hz: f32,
    /// Geometric mean divided by arithmetic mean of power. 1.0 is white noise.
    pub flatness: f32,
    /// `1.0 - flatness`. High for whistles and pure tones, low for noise.
    pub tonality: f32,
    /// Peakiness of the magnitude spectrum, normalized to 0-1.
    pub crest: f32,
    /// Shannon entropy of the normalized power spectrum, normalized to 0-1.
    pub entropy: f32,
    /// Least-squares slope of magnitude against frequency (per Hz). Negative means
    /// energy concentrated at low frequencies.
    pub slope: f32,
    /// Peak frequency, parabolically interpolated between bins (Hz).
    pub dominant_hz: f32,
    /// Frame energy in dBFS.
    pub energy_db: f32,
    /// Zero-crossing rate of the time-domain frame (0-1).
    pub zcr: f32,
    /// Whether this frame exceeded the energy gate.
    pub voiced: bool,
}

impl SpectralFrame {
    /// A frame with no measurable content, used for silence.
    fn silent(time_ms: f64, energy_db: f32, zcr: f32) -> Self {
        Self {
            time_ms,
            centroid_hz: 0.0,
            spread_hz: 0.0,
            skewness: 0.0,
            rolloff_hz: 0.0,
            flatness: 1.0,
            tonality: 0.0,
            crest: 0.0,
            entropy: 1.0,
            slope: 0.0,
            dominant_hz: 0.0,
            energy_db,
            zcr,
            voiced: false,
        }
    }
}

/// Mean and standard deviation of a descriptor over voiced frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Stat {
    /// Arithmetic mean.
    pub mean: f32,
    /// Population standard deviation.
    pub std: f32,
}

impl Stat {
    fn from_slice(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        Self {
            mean,
            std: variance.sqrt(),
        }
    }
}

/// Amplitude and frequency modulation measured across a sequence of frames.
///
/// Modulation is a property of a sequence, not of a frame, so these are absent for
/// inputs shorter than [`MIN_FRAMES_FOR_MODULATION`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModulationFeatures {
    /// Dominant modulation rate of the energy envelope (Hz).
    pub am_rate_hz: f32,
    /// Modulation depth of the energy envelope, 0-1.
    pub am_depth: f32,
    /// Dominant modulation rate of the centroid track (Hz).
    pub fm_rate_hz: f32,
    /// Standard deviation of the centroid track -- sweep width (Hz).
    pub fm_extent_hz: f32,
}

/// Per-segment summary of acoustic descriptors.
///
/// Statistics are computed over voiced frames only. Averaging across silence pulls
/// every descriptor toward whatever the noise floor happens to be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureSummary {
    /// Spectral centroid statistics (Hz).
    pub centroid_hz: Stat,
    /// Spectral spread statistics (Hz).
    pub spread_hz: Stat,
    /// Spectral rolloff statistics (Hz).
    pub rolloff_hz: Stat,
    /// Tonality statistics (0-1).
    pub tonality: Stat,
    /// Spectral crest statistics (0-1).
    pub crest: Stat,
    /// Spectral entropy statistics (0-1).
    pub entropy: Stat,
    /// Spectral slope statistics (per Hz).
    pub slope: Stat,
    /// Dominant frequency statistics (Hz).
    pub dominant_hz: Stat,
    /// Frame energy statistics (dBFS).
    pub energy_db: Stat,
    /// Zero-crossing rate statistics (0-1).
    pub zcr: Stat,
    /// Modulation features, absent for inputs that are too short.
    pub modulation: Option<ModulationFeatures>,
    /// Proportion of frames that exceeded the energy gate (0-1).
    pub voiced_fraction: f32,
    /// Total number of frames analyzed.
    pub n_frames: usize,
}

/// Complete feature extraction result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticFeatures {
    /// Per-frame descriptors, ordered by time.
    pub frames: Vec<SpectralFrame>,
    /// Aggregate statistics over voiced frames.
    pub summary: FeatureSummary,
    /// Configuration used, retained so values remain interpretable.
    pub config: FeatureConfig,
}

impl AcousticFeatures {
    /// Returns the centroid track, one value per frame.
    #[must_use]
    pub fn centroid_track(&self) -> Vec<f32> {
        self.frames.iter().map(|f| f.centroid_hz).collect()
    }

    /// Returns the energy track in dBFS, one value per frame.
    #[must_use]
    pub fn energy_track(&self) -> Vec<f32> {
        self.frames.iter().map(|f| f.energy_db).collect()
    }

    /// Returns true when the signal looks like broadband noise rather than a
    /// vocalization: low tonality with energy concentrated at low frequencies.
    ///
    /// Used to reject wind and rain before spending inference budget on them.
    #[must_use]
    pub fn is_likely_noise(&self) -> bool {
        self.summary.voiced_fraction < 0.05
            || (self.summary.tonality.mean < 0.15 && self.summary.centroid_hz.mean < 500.0)
    }

    /// Returns true when the waveform appears clipped.
    ///
    /// A clipped sine approaches a square wave, whose spectrum is far less peaked
    /// than the original, driving crest down while energy stays high.
    #[must_use]
    pub fn is_likely_clipped(&self) -> bool {
        self.summary.energy_db.mean > -1.0 && self.summary.crest.mean < 0.2
    }
}

/// Extracts acoustic descriptors from audio samples.
///
/// The FFT plan and window are built once and reused across frames and across
/// calls, so a single extractor should be held for the lifetime of a stream.
pub struct FeatureExtractor {
    config: FeatureConfig,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Frequency of each analyzed bin, in Hz.
    bin_freqs: Vec<f32>,
    /// First bin index at or above `f_min`.
    lo_bin: usize,
    /// One past the last bin at or below `f_max`.
    hi_bin: usize,
    /// Coherent gain of the analysis window, used to undo its amplitude scaling.
    window_gain: f32,
}

impl FeatureExtractor {
    /// Creates an extractor for the given configuration.
    ///
    /// # Errors
    /// Returns [`AudioError`] when the configuration is invalid or when the
    /// requested band contains no FFT bins.
    pub fn new(config: FeatureConfig) -> Result<Self, AudioError> {
        config.validate()?;

        let n_fft = config.n_fft;
        let n_bins = n_fft / 2 + 1;
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n_fft);

        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (TAU * i as f32 / n_fft as f32).cos()))
            .collect();
        let window_gain = window.iter().sum::<f32>() / n_fft as f32;

        let bin_width = config.bin_width_hz();
        let bin_freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * bin_width).collect();

        // Restrict to the configured band. Including DC drags the centroid toward
        // zero whenever there is low-frequency rumble, which there always is.
        let lo_bin = (config.f_min / bin_width).ceil() as usize;
        let hi_bin = ((config.f_max / bin_width).floor() as usize + 1).min(n_bins);

        // Two bins is the minimum for spread, slope, and entropy to mean anything; a
        // single-bin band would silently return zeros for all three.
        if hi_bin.saturating_sub(lo_bin) < 2 {
            return Err(AudioError::invalid_data(
                "frequency band spans fewer than 2 FFT bins; widen f_min..f_max or raise n_fft",
            ));
        }

        Ok(Self {
            config,
            fft,
            window,
            bin_freqs,
            lo_bin,
            hi_bin,
            window_gain,
        })
    }

    /// Returns the configuration this extractor was built with.
    #[must_use]
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }

    /// Number of frames that will be produced for an input of `n_samples`.
    #[must_use]
    pub fn frame_count(&self, n_samples: usize) -> usize {
        if n_samples < self.config.n_fft {
            0
        } else {
            (n_samples - self.config.n_fft) / self.config.hop_length + 1
        }
    }

    /// Extracts descriptors from mono audio samples.
    ///
    /// # Errors
    /// Returns [`AudioError`] when the input is shorter than one FFT window.
    pub fn extract(&self, samples: &[f32]) -> Result<AcousticFeatures, AudioError> {
        let n_frames = self.frame_count(samples.len());
        if n_frames == 0 {
            return Err(AudioError::invalid_data(
                "audio shorter than one FFT window",
            ));
        }

        let mut scratch = self.fft.make_scratch_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut input = vec![0.0f32; self.config.n_fft];
        let mut magnitude = vec![0.0f32; self.hi_bin - self.lo_bin];

        let frames: Vec<SpectralFrame> = (0..n_frames)
            .map(|idx| {
                let start = idx * self.config.hop_length;
                let block = &samples[start..start + self.config.n_fft];
                self.analyze_frame(
                    block,
                    start,
                    &mut input,
                    &mut spectrum,
                    &mut scratch,
                    &mut magnitude,
                )
            })
            .collect();

        let summary = self.summarize(&frames);
        Ok(AcousticFeatures {
            frames,
            summary,
            config: self.config.clone(),
        })
    }

    /// Computes descriptors for one frame, reusing caller-owned buffers.
    fn analyze_frame(
        &self,
        block: &[f32],
        start_sample: usize,
        input: &mut [f32],
        spectrum: &mut [realfft::num_complex::Complex<f32>],
        scratch: &mut [realfft::num_complex::Complex<f32>],
        magnitude: &mut [f32],
    ) -> SpectralFrame {
        let time_ms = start_sample as f64 * 1000.0 / f64::from(self.config.sample_rate);

        let rms = (block.iter().map(|s| s * s).sum::<f32>() / block.len() as f32).sqrt();
        let energy_db = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            f32::NEG_INFINITY
        };
        let zcr = zero_crossing_rate(block);

        if energy_db <= self.config.energy_gate_db {
            return SpectralFrame::silent(time_ms, energy_db, zcr);
        }

        for (dst, (&s, &w)) in input.iter_mut().zip(block.iter().zip(self.window.iter())) {
            *dst = s * w;
        }

        if self
            .fft
            .process_with_scratch(input, spectrum, scratch)
            .is_err()
        {
            return SpectralFrame::silent(time_ms, energy_db, zcr);
        }

        // Undo the window's amplitude scaling so magnitudes stay comparable to the
        // time-domain signal.
        let norm = 2.0 / (self.config.n_fft as f32 * self.window_gain);
        for (dst, c) in magnitude
            .iter_mut()
            .zip(spectrum[self.lo_bin..self.hi_bin].iter())
        {
            *dst = (c.re * c.re + c.im * c.im).sqrt() * norm;
        }

        let freqs = &self.bin_freqs[self.lo_bin..self.hi_bin];
        self.descriptors(magnitude, freqs, time_ms, energy_db, zcr)
    }

    /// Derives spectral descriptors from a magnitude spectrum and its bin frequencies.
    #[allow(clippy::too_many_lines)]
    fn descriptors(
        &self,
        magnitude: &[f32],
        freqs: &[f32],
        time_ms: f64,
        energy_db: f32,
        zcr: f32,
    ) -> SpectralFrame {
        let n = magnitude.len();
        let power: Vec<f32> = magnitude.iter().map(|m| m * m).collect();
        let total_power: f32 = power.iter().sum();

        if total_power <= POWER_EPSILON {
            return SpectralFrame::silent(time_ms, energy_db, zcr);
        }

        // Centroid and spread are the first two moments of the normalized power
        // distribution over frequency.
        let centroid_hz = freqs
            .iter()
            .zip(power.iter())
            .map(|(f, p)| f * p)
            .sum::<f32>()
            / total_power;

        let variance = freqs
            .iter()
            .zip(power.iter())
            .map(|(f, p)| (f - centroid_hz).powi(2) * p)
            .sum::<f32>()
            / total_power;
        let spread_hz = variance.sqrt();

        let skewness = if spread_hz > f32::EPSILON {
            freqs
                .iter()
                .zip(power.iter())
                .map(|(f, p)| ((f - centroid_hz) / spread_hz).powi(3) * p)
                .sum::<f32>()
                / total_power
        } else {
            0.0
        };

        // Rolloff: lowest frequency below which the configured fraction of power lies.
        let threshold = total_power * self.config.rolloff_percent;
        let mut cumulative = 0.0f32;
        let mut rolloff_hz = freqs[n - 1];
        for (f, p) in freqs.iter().zip(power.iter()) {
            cumulative += p;
            if cumulative >= threshold {
                rolloff_hz = *f;
                break;
            }
        }

        // Flatness via log-mean rather than a direct product: the product of ~1000
        // bin powers underflows f32 immediately.
        let log_sum: f32 = power.iter().map(|p| p.max(POWER_EPSILON).ln()).sum();
        let geometric_mean = (log_sum / n as f32).exp();
        let arithmetic_mean = total_power / n as f32;
        let flatness = if arithmetic_mean > POWER_EPSILON {
            (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Crest: peak-to-RMS of the magnitude spectrum, mapped to 0-1 so it shares
        // an axis with the other normalized descriptors.
        let peak = magnitude.iter().copied().fold(0.0f32, f32::max);
        let mag_rms = (power.iter().sum::<f32>() / n as f32).sqrt();
        let crest = if mag_rms > f32::EPSILON && peak > mag_rms {
            1.0 - (mag_rms / peak)
        } else {
            0.0
        };

        // Entropy of the normalized power distribution. Zero-power bins are skipped
        // rather than relying on 0*ln(0).
        let mut entropy = 0.0f32;
        for p in &power {
            let p_hat = p / total_power;
            if p_hat > 0.0 {
                entropy -= p_hat * p_hat.ln();
            }
        }
        let entropy = if n > 1 {
            (entropy / (n as f32).ln()).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let slope = least_squares_slope(freqs, magnitude);
        let dominant_hz = self.interpolated_peak(magnitude, freqs);

        SpectralFrame {
            time_ms,
            centroid_hz,
            spread_hz,
            skewness,
            rolloff_hz,
            flatness,
            tonality: 1.0 - flatness,
            crest,
            entropy,
            slope,
            dominant_hz,
            energy_db,
            zcr,
            voiced: true,
        }
    }

    /// Locates the spectral peak with parabolic interpolation.
    ///
    /// Without interpolation, resolution is quantized to one bin width (31.25 Hz at
    /// the default configuration), which shows up as visible banding in a frequency
    /// track.
    fn interpolated_peak(&self, magnitude: &[f32], freqs: &[f32]) -> f32 {
        let Some(peak_idx) = argmax(magnitude) else {
            return 0.0;
        };

        if peak_idx == 0 || peak_idx + 1 >= magnitude.len() {
            return freqs[peak_idx];
        }

        let y0 = magnitude[peak_idx - 1];
        let y1 = magnitude[peak_idx];
        let y2 = magnitude[peak_idx + 1];
        let denominator = y0 - 2.0 * y1 + y2;

        if denominator.abs() < f32::EPSILON {
            return freqs[peak_idx];
        }

        let delta = (0.5 * (y0 - y2) / denominator).clamp(-0.5, 0.5);
        freqs[peak_idx] + delta * self.config.bin_width_hz()
    }

    /// Aggregates per-frame descriptors over voiced frames.
    fn summarize(&self, frames: &[SpectralFrame]) -> FeatureSummary {
        let voiced: Vec<&SpectralFrame> = frames.iter().filter(|f| f.voiced).collect();
        let voiced_fraction = if frames.is_empty() {
            0.0
        } else {
            voiced.len() as f32 / frames.len() as f32
        };

        let collect = |extract: fn(&SpectralFrame) -> f32| -> Stat {
            let values: Vec<f32> = voiced.iter().map(|f| extract(f)).collect();
            Stat::from_slice(&values)
        };

        // Energy is summarized over all frames: the silent ones are the point of the
        // measurement, unlike for the spectral descriptors.
        let energy_values: Vec<f32> = frames
            .iter()
            .map(|f| f.energy_db)
            .filter(|v| v.is_finite())
            .collect();

        FeatureSummary {
            centroid_hz: collect(|f| f.centroid_hz),
            spread_hz: collect(|f| f.spread_hz),
            rolloff_hz: collect(|f| f.rolloff_hz),
            tonality: collect(|f| f.tonality),
            crest: collect(|f| f.crest),
            entropy: collect(|f| f.entropy),
            slope: collect(|f| f.slope),
            dominant_hz: collect(|f| f.dominant_hz),
            energy_db: Stat::from_slice(&energy_values),
            zcr: collect(|f| f.zcr),
            modulation: self.modulation(frames),
            voiced_fraction,
            n_frames: frames.len(),
        }
    }

    /// Estimates amplitude and frequency modulation across the frame sequence.
    fn modulation(&self, frames: &[SpectralFrame]) -> Option<ModulationFeatures> {
        if frames.len() < MIN_FRAMES_FOR_MODULATION {
            return None;
        }

        // Linear energy, not dB: modulation depth is a ratio of amplitudes, and a
        // logarithmic envelope would distort it.
        let envelope: Vec<f32> = frames
            .iter()
            .map(|f| {
                if f.energy_db.is_finite() {
                    10f32.powf(f.energy_db / 20.0)
                } else {
                    0.0
                }
            })
            .collect();
        let centroids: Vec<f32> = frames.iter().map(|f| f.centroid_hz).collect();

        let frame_rate = self.config.frame_rate();
        let (am_rate_hz, am_peak) = dominant_modulation(&envelope, frame_rate);
        let (fm_rate_hz, _) = dominant_modulation(&centroids, frame_rate);

        let mean_envelope = envelope.iter().sum::<f32>() / envelope.len() as f32;
        let am_depth = if mean_envelope > f32::EPSILON {
            (am_peak / mean_envelope).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Sweep width is better captured by dispersion of the centroid track than by
        // the modulation peak, which only reports rate.
        let voiced_centroids: Vec<f32> = frames
            .iter()
            .filter(|f| f.voiced)
            .map(|f| f.centroid_hz)
            .collect();
        let fm_extent_hz = Stat::from_slice(&voiced_centroids).std;

        Some(ModulationFeatures {
            am_rate_hz,
            am_depth,
            fm_rate_hz,
            fm_extent_hz,
        })
    }
}

/// Finds the dominant modulation frequency of a track and its normalized magnitude.
///
/// Returns `(rate_hz, peak_magnitude)`. The search band spans
/// [`MODULATION_MIN_HZ`] up to the lesser of [`MODULATION_MAX_HZ`] and the Nyquist
/// frequency of the frame rate -- at the default 100 fps that ceiling is 50 Hz, so
/// trills faster than 50 Hz require a smaller hop to measure.
///
/// Exposed for callers that want modulation analysis over their own track, such as
/// an amplitude envelope accumulated by the streaming pipeline.
#[must_use]
pub fn dominant_modulation(track: &[f32], frame_rate: f32) -> (f32, f32) {
    let n = track.len();
    if n < 4 || frame_rate <= 0.0 {
        return (0.0, 0.0);
    }

    let mean = track.iter().sum::<f32>() / n as f32;
    let fft_len = n.next_power_of_two();

    // Detrend and window before transforming: the DC term would otherwise dominate,
    // and the discontinuity at the ends would smear the peak.
    let mut input = vec![0.0f32; fft_len];
    for (i, (dst, value)) in input.iter_mut().zip(track.iter()).enumerate() {
        let w = 0.5 * (1.0 - (TAU * i as f32 / n as f32).cos());
        *dst = (value - mean) * w;
    }

    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(fft_len);
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    if fft
        .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
        .is_err()
    {
        return (0.0, 0.0);
    }

    let resolution = frame_rate / fft_len as f32;
    let nyquist = frame_rate / 2.0;
    let max_hz = MODULATION_MAX_HZ.min(nyquist);
    let lo = ((MODULATION_MIN_HZ / resolution).ceil() as usize).max(1);
    let hi = ((max_hz / resolution).floor() as usize + 1).min(spectrum.len());

    if lo >= hi {
        return (0.0, 0.0);
    }

    let scale = 2.0 / n as f32;
    let mut best_idx = lo;
    let mut best_mag = 0.0f32;
    for (offset, c) in spectrum[lo..hi].iter().enumerate() {
        let mag = (c.re * c.re + c.im * c.im).sqrt() * scale;
        if mag > best_mag {
            best_mag = mag;
            best_idx = lo + offset;
        }
    }

    (best_idx as f32 * resolution, best_mag)
}

/// Least-squares slope of `y` against `x`.
fn least_squares_slope(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        numerator += dx * (yi - mean_y);
        denominator += dx * dx;
    }

    if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Proportion of adjacent sample pairs whose sign differs.
fn zero_crossing_rate(block: &[f32]) -> f32 {
    if block.len() < 2 {
        return 0.0;
    }
    let crossings = block
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (block.len() - 1) as f32
}

/// Index of the largest value, or `None` when the slice is empty.
fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .fold(None, |best, (i, &v)| match best {
            Some((_, bv)) if bv >= v => best,
            _ => Some((i, v)),
        })
        .map(|(i, _)| i)
}
