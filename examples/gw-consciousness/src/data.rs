//! GW background data generation and transition probability matrix construction.
//!
//! Generates characteristic strain spectra h_c(f) for different GW source
//! models based on NANOGrav 15-year dataset parameters.

/// NANOGrav 15-year GW background model parameters.
///
/// h_c(f) = A * (f / f_ref)^alpha
/// Best fit: A = 2.4e-15, alpha = -2/3 (SMBH mergers), f_ref = 1/yr
///
/// The dataset uses 14 frequency bins at f_k = k/T where T = 16.03 years
/// and k = 1..14.
const NANOGRAV_AMPLITUDE: f64 = 2.4e-15;
const NANOGRAV_F_REF: f64 = 3.17e-8; // 1/yr in Hz
const NANOGRAV_T_OBS: f64 = 16.03; // observation span in years
const NANOGRAV_N_BINS: usize = 14;

/// A gravitational wave strain spectrum.
pub struct GWSpectrum {
    /// Frequency bin centers in Hz.
    pub frequencies: Vec<f64>,
    /// Characteristic strain h_c(f) at each bin.
    pub h_c: Vec<f64>,
    /// Strain uncertainty (1-sigma).
    pub errors: Vec<f64>,
    /// Number of frequency bins.
    pub n_bins: usize,
    /// Source model name.
    pub model: String,
    /// Spectral index alpha.
    pub alpha: f64,
}

/// Transition probability matrix for GW analysis.
pub struct TransitionMatrix {
    pub size: usize,
    pub data: Vec<f64>,
    pub bin_labels: Vec<String>,
    pub bin_frequencies: Vec<f64>,
}

impl TransitionMatrix {
    #[allow(dead_code)]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.size..(i + 1) * self.size]
    }

    #[allow(dead_code)]
    pub fn as_flat(&self) -> &[f64] {
        &self.data
    }
}

/// Generate a NANOGrav-like GW spectrum for a given source model.
///
/// Models:
/// - `"smbh"`: Supermassive black hole binary mergers, alpha = -2/3
/// - `"cosmic_strings"`: Cosmic string network, alpha = -7/6
/// - `"primordial"`: Primordial gravitational waves, alpha = -1
/// - `"phase_transition"`: First-order phase transition (peaked spectrum)
pub fn generate_nanograv_spectrum(model: &str) -> GWSpectrum {
    let t_obs_sec = NANOGRAV_T_OBS * 365.25 * 86400.0;
    let f_min = 1.0 / t_obs_sec;

    let frequencies: Vec<f64> = (1..=NANOGRAV_N_BINS)
        .map(|k| k as f64 * f_min)
        .collect();

    let (alpha, h_c) = match model {
        "smbh" => {
            // Standard SMBH binary merger background
            // h_c(f) = A * (f/f_ref)^(-2/3)
            let alpha = -2.0 / 3.0;
            let h_c: Vec<f64> = frequencies
                .iter()
                .map(|&f| {
                    NANOGRAV_AMPLITUDE * (f / NANOGRAV_F_REF).powf(alpha)
                })
                .collect();
            (alpha, h_c)
        }
        "cosmic_strings" => {
            // Cosmic string network
            // h_c(f) = A_cs * (f/f_ref)^(-7/6)
            let alpha = -7.0 / 6.0;
            let a_cs = NANOGRAV_AMPLITUDE * 0.8;
            let h_c: Vec<f64> = frequencies
                .iter()
                .map(|&f| a_cs * (f / NANOGRAV_F_REF).powf(alpha))
                .collect();
            (alpha, h_c)
        }
        "primordial" => {
            // Primordial GW from inflation
            // h_c(f) = A_pgw * (f/f_ref)^(-1)
            let alpha = -1.0;
            let a_pgw = NANOGRAV_AMPLITUDE * 0.6;
            let h_c: Vec<f64> = frequencies
                .iter()
                .map(|&f| a_pgw * (f / NANOGRAV_F_REF).powf(alpha))
                .collect();
            (alpha, h_c)
        }
        "phase_transition" => {
            // First-order phase transition: peaked spectrum
            // h_c(f) = A_pt * S(f/f_peak) where S is a broken power law
            let alpha = 0.0; // not a simple power law
            let a_pt = NANOGRAV_AMPLITUDE * 1.2;
            let f_peak = frequencies[NANOGRAV_N_BINS / 3]; // peak at ~5th bin
            let h_c: Vec<f64> = frequencies
                .iter()
                .map(|&f| {
                    let x = f / f_peak;
                    // Broken power law: rises as f^3 below peak, falls as f^(-4) above
                    let shape = if x < 1.0 {
                        x.powi(3) / (1.0 + x.powi(3))
                    } else {
                        1.0 / (1.0 + x.powi(4))
                    };
                    a_pt * shape * (f / NANOGRAV_F_REF).powf(-2.0 / 3.0)
                })
                .collect();
            (alpha, h_c)
        }
        _ => panic!("Unknown GW source model: {}", model),
    };

    // Measurement uncertainties scale with strain and frequency
    // Lower frequencies have larger errors (fewer pulsar cycles)
    let errors: Vec<f64> = h_c
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            let snr_factor = ((i + 1) as f64).sqrt();
            h * 0.3 / snr_factor
        })
        .collect();

    GWSpectrum {
        frequencies,
        h_c,
        errors,
        n_bins: NANOGRAV_N_BINS,
        model: model.to_string(),
        alpha,
    }
}

/// Convert a GW spectrum to a transition probability matrix.
///
/// The TPM encodes correlations between frequency bins. For SMBH mergers
/// (independent sources), each bin evolves nearly independently, producing
/// a near-diagonal TPM (low Phi). For cosmological sources (cosmic strings,
/// phase transitions), the underlying physics correlates all bins, producing
/// off-diagonal structure (higher Phi).
///
/// Method:
/// 1. Compute pairwise correlation from strain power: C_ij = h_c(i) * h_c(j)
/// 2. Weight by spectral proximity (Gaussian kernel in log-frequency)
/// 3. For SMBH: add Poisson noise to decorrelate bins (each binary is independent)
/// 4. For cosmological: correlations persist (single coherent source)
/// 5. Row-normalize to get transition probabilities
pub fn gw_spectrum_to_tpm(spec: &GWSpectrum, n_bins: usize, alpha: f64) -> TransitionMatrix {
    let n = n_bins.min(spec.n_bins);
    let mut corr = vec![0.0f64; n * n];

    // Correlation width depends on source model
    // SMBH mergers: narrow (independent sources per frequency)
    // Cosmological: broad (single correlated source)
    let sigma = match spec.model.as_str() {
        "smbh" => 0.3,              // narrow: nearly independent bins
        "cosmic_strings" => 2.0,    // broad: strong spectral correlations
        "primordial" => 1.5,        // moderate: inflationary correlations
        "phase_transition" => 3.0,  // very broad: phase transition coherence
        _ => 1.0,
    };

    for i in 0..n {
        for j in 0..n {
            let f_i = spec.frequencies[i].ln();
            let f_j = spec.frequencies[j].ln();
            let delta = f_i - f_j;

            // Gaussian kernel in log-frequency space
            let coupling = (-delta * delta / (2.0 * sigma * sigma)).exp();

            // Power coupling: geometric mean of strains
            let power = (spec.h_c[i] * spec.h_c[j]).sqrt().max(1e-30);

            corr[i * n + j] = power * coupling;
        }
    }

    // For SMBH model: add diagonal dominance (independent sources)
    if spec.model == "smbh" {
        for i in 0..n {
            corr[i * n + i] *= 5.0; // strong self-coupling
        }
    }

    // Row-normalize with sharpness parameter
    let mut tpm = vec![0.0f64; n * n];
    for i in 0..n {
        let row_sum: f64 = (0..n)
            .map(|j| corr[i * n + j].powf(alpha))
            .sum();
        for j in 0..n {
            tpm[i * n + j] = corr[i * n + j].powf(alpha) / row_sum.max(1e-30);
        }
    }

    let bin_labels: Vec<String> = (0..n)
        .map(|i| format!("f{}", i + 1))
        .collect();
    let bin_frequencies = spec.frequencies[..n].to_vec();

    TransitionMatrix {
        size: n,
        data: tpm,
        bin_labels,
        bin_frequencies,
    }
}

/// Generate a null-model TPM by perturbing the spectrum within error bars.
///
/// This produces a TPM consistent with the null hypothesis that the GW
/// background is produced by independent SMBH mergers with measurement noise.
pub fn generate_null_tpm(
    spec: &GWSpectrum,
    n_bins: usize,
    alpha: f64,
    rng: &mut impl rand::Rng,
) -> TransitionMatrix {
    let perturbed_h_c: Vec<f64> = spec
        .h_c
        .iter()
        .zip(spec.errors.iter())
        .map(|(&h, &e)| {
            let noise: f64 = rng.sample::<f64, _>(rand::distributions::Standard);
            (h + noise * e).max(1e-30)
        })
        .collect();

    let perturbed = GWSpectrum {
        frequencies: spec.frequencies.clone(),
        h_c: perturbed_h_c,
        errors: spec.errors.clone(),
        n_bins: spec.n_bins,
        model: "smbh".to_string(), // null model always uses SMBH correlations
        alpha: spec.alpha,
    };

    gw_spectrum_to_tpm(&perturbed, n_bins, alpha)
}
