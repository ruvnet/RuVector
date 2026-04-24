//! CMB data acquisition and transition probability matrix construction.

/// Planck 2018 CMB TT Power Spectrum
const PLANCK_URL: &str = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-full_R3.01.txt";

pub struct PowerSpectrum {
    pub ells: Vec<f64>,
    pub d_ells: Vec<f64>,
    pub errors: Vec<f64>,
}

pub struct TransitionMatrix {
    pub size: usize,
    pub data: Vec<f64>,
    pub bin_edges: Vec<f64>,
    pub bin_centers: Vec<f64>,
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

/// Download Planck power spectrum data using curl, with synthetic fallback.
pub fn download_power_spectrum() -> PowerSpectrum {
    // Try curl download
    if let Ok(output) = std::process::Command::new("curl")
        .args(["-sL", "--connect-timeout", "10", PLANCK_URL])
        .output()
    {
        if output.status.success() {
            if let Ok(text) = String::from_utf8(output.stdout) {
                if let Some(ps) = parse_planck_tt(&text) {
                    println!(
                        "Downloaded Planck 2018 TT power spectrum: {} multipoles",
                        ps.ells.len()
                    );
                    return ps;
                }
            }
        }
    }

    println!("Using synthetic LCDM power spectrum (download unavailable)");
    generate_synthetic_spectrum()
}

/// Parse Planck TT power spectrum ASCII format.
/// Columns: ell, D_ell, error_minus, error_plus
fn parse_planck_tt(text: &str) -> Option<PowerSpectrum> {
    let mut ells = Vec::new();
    let mut d_ells = Vec::new();
    let mut errors = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            if let (Ok(l), Ok(d), Ok(e1), Ok(e2)) = (
                parts[0].parse::<f64>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
                parts[3].parse::<f64>(),
            ) {
                if l >= 2.0 {
                    ells.push(l);
                    d_ells.push(d);
                    errors.push((e1.abs() + e2.abs()) / 2.0);
                }
            }
        }
    }

    if ells.len() > 100 {
        Some(PowerSpectrum {
            ells,
            d_ells,
            errors,
        })
    } else {
        None
    }
}

/// Generate synthetic LCDM power spectrum with acoustic peaks.
fn generate_synthetic_spectrum() -> PowerSpectrum {
    let mut ells = Vec::new();
    let mut d_ells = Vec::new();
    let mut errors = Vec::new();

    for l in 2..=2500 {
        let l_f = l as f64;
        // Sachs-Wolfe plateau + acoustic oscillations + damping
        let sw = 6000.0 / (1.0 + (l_f / 10.0).powi(2));
        let acoustic = 5800.0
            * (l_f * std::f64::consts::PI / 301.0).cos().powi(2)
            * (-l_f / 1500.0).exp()
            * (l_f / 200.0).min(1.0);
        let d_l = (sw + acoustic).max(0.0);

        ells.push(l_f);
        d_ells.push(d_l);
        errors.push(d_l * 0.02 + 10.0);
    }

    PowerSpectrum {
        ells,
        d_ells,
        errors,
    }
}

/// Convert power spectrum to transition probability matrix.
///
/// Method:
/// 1. Partition multipoles into N logarithmic bins
/// 2. Compute cross-band correlation from power coupling
/// 3. Row-normalize to get transition probabilities
pub fn power_spectrum_to_tpm(ps: &PowerSpectrum, n_bins: usize, alpha: f64) -> TransitionMatrix {
    // Log-space bin edges
    let l_min = ps.ells[0].ln();
    let l_max = ps.ells.last().unwrap().ln();
    let bin_edges: Vec<f64> = (0..=n_bins)
        .map(|i| (l_min + (l_max - l_min) * i as f64 / n_bins as f64).exp())
        .collect();

    let bin_centers: Vec<f64> = bin_edges
        .windows(2)
        .map(|w| (w[0] * w[1]).sqrt()) // geometric mean
        .collect();

    // Compute bin-averaged power
    let mut bin_power = vec![0.0f64; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    for (&l, &d) in ps.ells.iter().zip(ps.d_ells.iter()) {
        for b in 0..n_bins {
            if l >= bin_edges[b] && l < bin_edges[b + 1] {
                bin_power[b] += d;
                bin_counts[b] += 1;
                break;
            }
        }
    }

    for b in 0..n_bins {
        if bin_counts[b] > 0 {
            bin_power[b] /= bin_counts[b] as f64;
        }
    }

    // Cross-correlation matrix with Gaussian coupling kernel
    let mut corr = vec![0.0f64; n_bins * n_bins];
    for i in 0..n_bins {
        let w_i = bin_edges[i + 1] - bin_edges[i];
        for j in 0..n_bins {
            let w_j = bin_edges[j + 1] - bin_edges[j];
            let delta = (bin_centers[i] - bin_centers[j]).abs();
            let sigma2 = w_i * w_j;
            let coupling = (-delta * delta / (2.0 * sigma2.max(1.0))).exp();
            corr[i * n_bins + j] = (bin_power[i] * bin_power[j]).sqrt().max(1e-10) * coupling;
        }
    }

    // Row-normalize with sharpness alpha
    let mut tpm = vec![0.0f64; n_bins * n_bins];
    for i in 0..n_bins {
        let row_sum: f64 = (0..n_bins).map(|j| corr[i * n_bins + j].powf(alpha)).sum();
        for j in 0..n_bins {
            tpm[i * n_bins + j] = corr[i * n_bins + j].powf(alpha) / row_sum.max(1e-30);
        }
    }

    TransitionMatrix {
        size: n_bins,
        data: tpm,
        bin_edges,
        bin_centers,
    }
}

/// Generate a null-model TPM by perturbing the power spectrum within error bars.
pub fn generate_null_tpm(
    ps: &PowerSpectrum,
    n_bins: usize,
    alpha: f64,
    rng: &mut impl rand::Rng,
) -> TransitionMatrix {
    let perturbed_d_ells: Vec<f64> = ps
        .d_ells
        .iter()
        .zip(ps.errors.iter())
        .map(|(&d, &e)| {
            let noise: f64 = rng.sample::<f64, _>(rand::distributions::Standard);
            (d + noise * e).max(0.0)
        })
        .collect();

    let perturbed_ps = PowerSpectrum {
        ells: ps.ells.clone(),
        d_ells: perturbed_d_ells,
        errors: ps.errors.clone(),
    };

    power_spectrum_to_tpm(&perturbed_ps, n_bins, alpha)
}
