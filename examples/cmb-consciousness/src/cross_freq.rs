//! Cross-frequency IIT analysis for CMB foreground separation.
//!
//! Planck observed at 9 frequencies: 30, 44, 70, 100, 143, 217, 353, 545, 857 GHz.
//! Foregrounds (dust, synchrotron) create inter-frequency correlations.
//! Pure CMB should have Phi=0 across frequencies. Phi>0 = foreground contamination.

use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::types::{ComputeBudget, TransitionMatrix as ConsciousnessTPM};

/// Planck frequency bands in GHz.
const PLANCK_BANDS: [f64; 9] = [30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0, 857.0];

/// CMB blackbody temperature in Kelvin.
const T_CMB: f64 = 2.725;

/// Dust modified blackbody parameters.
const T_DUST: f64 = 19.6;
const BETA_DUST: f64 = 1.59;

/// Synchrotron spectral index.
const BETA_SYNC: f64 = -3.1;

/// Free-free spectral index.
const BETA_FF: f64 = -2.14;

/// Reference frequency for foreground normalizations (GHz).
const NU_REF: f64 = 100.0;

/// Results of cross-frequency foreground analysis.
pub struct CrossFreqResults {
    pub full_phi: f64,
    pub low_freq_phi: f64,
    pub clean_phi: f64,
    pub high_freq_phi: f64,
    pub frequencies: Vec<f64>,
    pub foreground_level: Vec<f64>,
}

/// Main entry point: build a 9x9 TPM from Planck frequency bands and compute Phi
/// to detect foreground contamination.
pub fn run_cross_frequency_analysis() -> CrossFreqResults {
    println!("  Planck frequency bands: {:?} GHz", PLANCK_BANDS);

    // Generate synthetic band-averaged temperature data
    let cmb_signal = generate_cmb_signal();
    let dust_signal = generate_dust_signal();
    let sync_signal = generate_synchrotron_signal();
    let ff_signal = generate_freefree_signal();

    // Total signal = CMB + foregrounds
    let total_signal: Vec<f64> = (0..9)
        .map(|i| cmb_signal[i] + dust_signal[i] + sync_signal[i] + ff_signal[i])
        .collect();

    // Compute foreground contamination fraction per band
    let foreground_level: Vec<f64> = (0..9)
        .map(|i| {
            let fg = dust_signal[i] + sync_signal[i] + ff_signal[i];
            fg / total_signal[i]
        })
        .collect();

    println!("\n  Per-band signals (arbitrary units):");
    println!("  {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "GHz", "CMB", "Dust", "Sync", "FF", "FG frac");
    for i in 0..9 {
        println!("  {:>8.0} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            PLANCK_BANDS[i], cmb_signal[i], dust_signal[i],
            sync_signal[i], ff_signal[i], foreground_level[i]);
    }

    // Build 9x9 TPM from cross-frequency correlations with foregrounds
    let tpm_with_fg = build_cross_frequency_tpm(&total_signal);

    // Build 9x9 TPM for pure CMB (no foregrounds)
    let tpm_pure_cmb = build_cross_frequency_tpm(&cmb_signal);

    let budget = ComputeBudget::default();

    // Compute Phi for pure CMB (should be ~0)
    println!("\n  --- Pure CMB (no foregrounds) ---");
    match auto_compute_phi(&tpm_pure_cmb, None, &budget) {
        Ok(phi) => println!("  Pure CMB Phi = {:.6} (expected ~0)", phi.phi),
        Err(e) => println!("  Pure CMB Phi computation failed: {}", e),
    }

    // Compute Phi for full 9-band system with foregrounds
    println!("\n  --- Full 9-band system (CMB + foregrounds) ---");
    let full_phi = match auto_compute_phi(&tpm_with_fg, None, &budget) {
        Ok(phi) => {
            println!("  Full system Phi = {:.6} (algorithm: {})", phi.phi, phi.algorithm);
            phi.phi
        }
        Err(e) => {
            println!("  Full system Phi computation failed: {}", e);
            0.0
        }
    };

    // Low-frequency subset (30, 44, 70 GHz) -- synchrotron dominated
    println!("\n  --- Low-frequency subset (30, 44, 70 GHz) -- synchrotron dominated ---");
    let low_signal: Vec<f64> = vec![total_signal[0], total_signal[1], total_signal[2]];
    let low_tpm = build_cross_frequency_tpm(&low_signal);
    let low_freq_phi = compute_subset_phi(&low_tpm, &budget, "Low-freq");

    // CMB-clean subset (100, 143, 217 GHz) -- cleanest bands
    println!("\n  --- Clean subset (100, 143, 217 GHz) -- CMB-dominated ---");
    let clean_signal: Vec<f64> = vec![total_signal[3], total_signal[4], total_signal[5]];
    let clean_tpm = build_cross_frequency_tpm(&clean_signal);
    let clean_phi = compute_subset_phi(&clean_tpm, &budget, "Clean");

    // High-frequency subset (353, 545, 857 GHz) -- dust dominated
    println!("\n  --- High-frequency subset (353, 545, 857 GHz) -- dust dominated ---");
    let high_signal: Vec<f64> = vec![total_signal[6], total_signal[7], total_signal[8]];
    let high_tpm = build_cross_frequency_tpm(&high_signal);
    let high_freq_phi = compute_subset_phi(&high_tpm, &budget, "High-freq");

    // Summary and interpretation
    println!("\n  === Cross-Frequency Foreground Analysis Summary ===");
    println!("  Full system Phi:     {:.6}", full_phi);
    println!("  Low-freq Phi:        {:.6}  (synchrotron bands)", low_freq_phi);
    println!("  Clean Phi:           {:.6}  (CMB-dominated bands)", clean_phi);
    println!("  High-freq Phi:       {:.6}  (dust bands)", high_freq_phi);

    println!("\n  Interpretation:");
    if full_phi < 1e-4 {
        println!("  -> Full system shows near-zero Phi: consistent with uncorrelated noise.");
    } else {
        println!("  -> Full system shows Phi > 0: foreground correlations detected.");
    }

    if high_freq_phi > clean_phi {
        println!("  -> High-frequency bands show more integration than clean bands:");
        println!("     thermal dust creates correlated emission across 353-857 GHz.");
    }
    if low_freq_phi > clean_phi {
        println!("  -> Low-frequency bands show more integration than clean bands:");
        println!("     synchrotron emission correlates the 30-70 GHz channels.");
    }
    if clean_phi < low_freq_phi.min(high_freq_phi) {
        println!("  -> The 100-217 GHz bands are cleanest (lowest Phi),");
        println!("     confirming these are the optimal CMB observation window.");
    }

    CrossFreqResults {
        full_phi,
        low_freq_phi,
        clean_phi,
        high_freq_phi,
        frequencies: PLANCK_BANDS.to_vec(),
        foreground_level,
    }
}

/// Compute Phi for a frequency subset TPM, handling errors gracefully.
fn compute_subset_phi(tpm: &ConsciousnessTPM, budget: &ComputeBudget, label: &str) -> f64 {
    match auto_compute_phi(tpm, None, budget) {
        Ok(phi) => {
            println!("  {} Phi = {:.6}", label, phi.phi);
            phi.phi
        }
        Err(e) => {
            println!("  {} Phi computation failed: {}", label, e);
            0.0
        }
    }
}

/// Generate CMB blackbody signal across Planck bands.
///
/// At 2.725K the Planck function in thermodynamic temperature units gives
/// the same brightness temperature in all bands -> uniform signal -> no correlations.
fn generate_cmb_signal() -> Vec<f64> {
    // CMB is the same temperature in all bands (by definition of thermodynamic temperature).
    // We use a normalized value of 1.0 for all bands.
    vec![1.0; 9]
}

/// Generate thermal dust emission using a modified blackbody.
///
/// S_dust(nu) ~ nu^(beta_dust+1) * B_nu(T_dust) / B_nu_ref
/// Dominates at high frequencies (>217 GHz).
fn generate_dust_signal() -> Vec<f64> {
    let ref_intensity = modified_blackbody(NU_REF, T_DUST, BETA_DUST);
    PLANCK_BANDS
        .iter()
        .map(|&nu| {
            let intensity = modified_blackbody(nu, T_DUST, BETA_DUST);
            // Normalize to a reasonable amplitude relative to CMB
            0.05 * intensity / ref_intensity.max(1e-30)
        })
        .collect()
}

/// Modified blackbody: nu^beta * B_nu(T)
/// B_nu(T) = 2h*nu^3/c^2 * 1/(exp(h*nu/kT) - 1)
fn modified_blackbody(nu_ghz: f64, temp: f64, beta: f64) -> f64 {
    // Constants in convenient units for GHz
    // h/k = 0.04799 K/GHz (Planck constant / Boltzmann constant)
    let h_over_k = 0.04799;
    let x = h_over_k * nu_ghz / temp;

    // nu^beta * B_nu (we drop the constant 2h/c^2 as it cancels in ratios)
    let nu_beta = nu_ghz.powf(beta);
    let planck_fn = if x > 500.0 {
        0.0 // Avoid overflow
    } else {
        nu_ghz.powi(3) / (x.exp() - 1.0).max(1e-30)
    };

    nu_beta * planck_fn
}

/// Generate synchrotron emission using a power law.
///
/// S_sync(nu) ~ (nu/nu_ref)^beta_s
/// Dominates at low frequencies (<70 GHz).
fn generate_synchrotron_signal() -> Vec<f64> {
    PLANCK_BANDS
        .iter()
        .map(|&nu| {
            // Synchrotron amplitude normalized to 0.1 at reference frequency
            0.1 * (nu / NU_REF).powf(BETA_SYNC)
        })
        .collect()
}

/// Generate free-free (bremsstrahlung) emission using a power law.
///
/// S_ff(nu) ~ (nu/nu_ref)^beta_ff
/// Subdominant foreground, roughly flat spectrum with slight decline.
fn generate_freefree_signal() -> Vec<f64> {
    PLANCK_BANDS
        .iter()
        .map(|&nu| {
            // Free-free amplitude normalized to 0.02 at reference frequency
            0.02 * (nu / NU_REF).powf(BETA_FF)
        })
        .collect()
}

/// Build a cross-frequency TPM from band-averaged signals.
///
/// T[i][j] = probability that signal at frequency i correlates with frequency j.
/// We use the normalized cross-correlation of spectral amplitudes:
///   corr[i][j] = s_i * s_j / sum_k(s_i * s_k)
///
/// For uniform signals (pure CMB), this produces a uniform TPM -> Phi = 0.
/// For structured signals (foregrounds), this produces non-trivial structure -> Phi > 0.
fn build_cross_frequency_tpm(signal: &[f64]) -> ConsciousnessTPM {
    let n = signal.len();
    let mut data = vec![0.0f64; n * n];

    for i in 0..n {
        let mut row_sum = 0.0f64;

        // Cross-correlation: coupling proportional to geometric mean of signals
        for j in 0..n {
            let coupling = (signal[i] * signal[j]).sqrt().max(1e-30);
            data[i * n + j] = coupling;
            row_sum += coupling;
        }

        // Row-normalize to get transition probabilities
        if row_sum > 1e-30 {
            for j in 0..n {
                data[i * n + j] /= row_sum;
            }
        }
    }

    ConsciousnessTPM::new(n, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_cmb_uniform() {
        let cmb = generate_cmb_signal();
        // All values should be equal
        for &v in &cmb {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dust_increases_with_frequency() {
        let dust = generate_dust_signal();
        // Dust should increase from low to high frequency (above ~100 GHz)
        assert!(dust[8] > dust[3], "Dust should dominate at 857 GHz vs 100 GHz");
    }

    #[test]
    fn test_synchrotron_decreases_with_frequency() {
        let sync = generate_synchrotron_signal();
        // Synchrotron should decrease with frequency
        assert!(sync[0] > sync[8], "Synchrotron should dominate at 30 GHz vs 857 GHz");
    }

    #[test]
    fn test_tpm_row_normalization() {
        let signal = vec![1.0, 2.0, 3.0];
        let tpm = build_cross_frequency_tpm(&signal);
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| tpm.get(i, j)).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_uniform_signal_gives_uniform_tpm() {
        let signal = vec![1.0; 4];
        let tpm = build_cross_frequency_tpm(&signal);
        let expected = 0.25; // 1/4
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (tpm.get(i, j) - expected).abs() < 1e-10,
                    "Uniform signal should give uniform TPM"
                );
            }
        }
    }
}
