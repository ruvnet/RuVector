//! Climate teleconnection data generation.
//!
//! Builds correlation matrices for 7 major climate indices based on known
//! teleconnection strengths. Generates neutral baseline and El Nino active
//! variants.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Climate index identifiers.
pub const INDEX_ENSO: usize = 0; // El Nino Southern Oscillation (Nino3.4)
pub const INDEX_NAO: usize = 1;  // North Atlantic Oscillation
pub const INDEX_PDO: usize = 2;  // Pacific Decadal Oscillation
pub const INDEX_AMO: usize = 3;  // Atlantic Multidecadal Oscillation
pub const INDEX_IOD: usize = 4;  // Indian Ocean Dipole
pub const INDEX_SAM: usize = 5;  // Southern Annular Mode
pub const INDEX_QBO: usize = 6;  // Quasi-Biennial Oscillation

pub const INDEX_NAMES: &[&str] = &["ENSO", "NAO", "PDO", "AMO", "IOD", "SAM", "QBO"];
pub const N_INDICES: usize = 7;

/// Regional subsets of climate indices.
pub const PACIFIC_INDICES: &[usize] = &[INDEX_ENSO, INDEX_PDO, INDEX_IOD];
pub const ATLANTIC_INDICES: &[usize] = &[INDEX_NAO, INDEX_AMO];
pub const POLAR_INDICES: &[usize] = &[INDEX_SAM, INDEX_QBO];

pub const REGION_NAMES: &[&str] = &["Pacific", "Atlantic", "Polar"];

pub fn all_regions() -> Vec<(&'static str, &'static [usize])> {
    vec![
        (REGION_NAMES[0], PACIFIC_INDICES),
        (REGION_NAMES[1], ATLANTIC_INDICES),
        (REGION_NAMES[2], POLAR_INDICES),
    ]
}

/// Climate mode correlation data.
pub struct ClimateCorrelations {
    pub n_indices: usize,
    /// Flat row-major correlation matrix (symmetric, diagonal = 1.0).
    pub correlations: Vec<f64>,
    /// Variant label.
    pub variant: String,
}

/// Transition probability matrix for consciousness analysis.
pub struct TransitionMatrix {
    pub size: usize,
    pub data: Vec<f64>,
}

/// Build the neutral baseline correlation matrix.
///
/// Known teleconnection strengths (approximate, from literature):
/// - ENSO <-> IOD: strong (0.5-0.7)
/// - ENSO <-> PDO: moderate (0.3-0.5)
/// - NAO <-> AMO: moderate (0.2-0.4)
/// - QBO <-> ENSO: weak but real (0.1-0.2)
/// - SAM: mostly independent
pub fn build_neutral_correlations() -> ClimateCorrelations {
    let n = N_INDICES;
    let mut corr = vec![0.0f64; n * n];
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Set diagonal to 1.0 (self-correlation)
    for i in 0..n {
        corr[i * n + i] = 1.0;
    }

    // Define known teleconnection strengths (symmetric)
    let connections: &[(usize, usize, f64, f64)] = &[
        // (index_a, index_b, min_corr, max_corr)
        (INDEX_ENSO, INDEX_IOD, 0.50, 0.70),  // Strong: ENSO-IOD coupling
        (INDEX_ENSO, INDEX_PDO, 0.30, 0.50),  // Moderate: Pacific basin coupling
        (INDEX_NAO, INDEX_AMO, 0.20, 0.40),   // Moderate: Atlantic coupling
        (INDEX_QBO, INDEX_ENSO, 0.10, 0.20),  // Weak: stratosphere-troposphere
        (INDEX_PDO, INDEX_IOD, 0.10, 0.25),   // Weak: Indo-Pacific coupling
        (INDEX_ENSO, INDEX_NAO, 0.05, 0.15),  // Weak: Pacific-Atlantic bridge
        (INDEX_ENSO, INDEX_SAM, 0.05, 0.15),  // Weak: tropical-polar link
        (INDEX_AMO, INDEX_PDO, 0.05, 0.10),   // Very weak: inter-basin
        (INDEX_SAM, INDEX_QBO, 0.03, 0.08),   // Very weak: polar-stratosphere
        (INDEX_NAO, INDEX_QBO, 0.05, 0.12),   // Weak: NAO-QBO link
        (INDEX_SAM, INDEX_NAO, 0.02, 0.06),   // Very weak: bipolar
        (INDEX_IOD, INDEX_AMO, 0.02, 0.06),   // Very weak: Indian-Atlantic
        (INDEX_AMO, INDEX_SAM, 0.01, 0.04),   // Negligible
        (INDEX_IOD, INDEX_NAO, 0.01, 0.04),   // Negligible
        (INDEX_IOD, INDEX_SAM, 0.01, 0.03),   // Negligible
        (INDEX_PDO, INDEX_NAO, 0.02, 0.06),   // Very weak
        (INDEX_PDO, INDEX_SAM, 0.01, 0.04),   // Negligible
        (INDEX_QBO, INDEX_AMO, 0.02, 0.05),   // Very weak
        (INDEX_QBO, INDEX_PDO, 0.03, 0.08),   // Very weak
        (INDEX_QBO, INDEX_IOD, 0.02, 0.06),   // Very weak
        (INDEX_QBO, INDEX_SAM, 0.03, 0.08),   // Very weak
    ];

    for &(a, b, min_c, max_c) in connections {
        let c = rng.gen_range(min_c..max_c);
        corr[a * n + b] = c;
        corr[b * n + a] = c;
    }

    ClimateCorrelations {
        n_indices: n,
        correlations: corr,
        variant: "Neutral".into(),
    }
}

/// Build the El Nino active variant.
///
/// During El Nino events, ENSO correlations strengthen by ~50%,
/// and the Pacific basin becomes more tightly coupled.
pub fn build_elnino_correlations() -> ClimateCorrelations {
    let mut data = build_neutral_correlations();
    data.variant = "El Nino Active".into();
    let n = data.n_indices;

    // Boost all ENSO correlations by 50%
    for j in 0..n {
        if j != INDEX_ENSO {
            let boosted = (data.correlations[INDEX_ENSO * n + j] * 1.5).min(0.95);
            data.correlations[INDEX_ENSO * n + j] = boosted;
            data.correlations[j * n + INDEX_ENSO] = boosted;
        }
    }

    // Also boost intra-Pacific correlations
    let pacific_boost: &[(usize, usize)] = &[
        (INDEX_PDO, INDEX_IOD),
    ];
    for &(a, b) in pacific_boost {
        let boosted = (data.correlations[a * n + b] * 1.3).min(0.90);
        data.correlations[a * n + b] = boosted;
        data.correlations[b * n + a] = boosted;
    }

    data
}

/// Convert a correlation matrix to a transition probability matrix.
///
/// Method (same approach as the CMB example):
/// 1. Use absolute correlations as coupling strengths.
/// 2. Apply a sharpness exponent (alpha = 2.0) to emphasize strong connections.
/// 3. Row-normalize to get transition probabilities.
pub fn correlation_to_tpm(data: &ClimateCorrelations) -> TransitionMatrix {
    let n = data.n_indices;
    let alpha = 2.0;
    let mut tpm = vec![0.0f64; n * n];

    for i in 0..n {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            let c = data.correlations[i * n + j].abs();
            let val = c.powf(alpha);
            tpm[i * n + j] = val;
            row_sum += val;
        }
        // Row-normalize
        if row_sum > 1e-30 {
            for j in 0..n {
                tpm[i * n + j] /= row_sum;
            }
        }
    }

    TransitionMatrix { size: n, data: tpm }
}

/// Extract a sub-TPM for a subset of climate indices.
pub fn extract_sub_tpm(tpm: &TransitionMatrix, indices: &[usize]) -> TransitionMatrix {
    let n = indices.len();
    let mut sub = vec![0.0f64; n * n];
    for (si, &ii) in indices.iter().enumerate() {
        let row_sum: f64 = indices.iter().map(|&ij| tpm.data[ii * tpm.size + ij]).sum();
        for (sj, &ij) in indices.iter().enumerate() {
            sub[si * n + sj] = tpm.data[ii * tpm.size + ij] / row_sum.max(1e-30);
        }
    }
    TransitionMatrix { size: n, data: sub }
}

/// Generate a null-model correlation matrix by shuffling off-diagonal entries.
pub fn generate_null_tpm(data: &ClimateCorrelations, rng: &mut impl rand::Rng) -> TransitionMatrix {
    let n = data.n_indices;
    let mut corr = data.correlations.clone();

    // Collect upper-triangular off-diagonal values
    let mut upper: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            upper.push(corr[i * n + j]);
        }
    }

    // Shuffle
    for k in (1..upper.len()).rev() {
        let swap_idx = rng.gen_range(0..=k);
        upper.swap(k, swap_idx);
    }

    // Put back (symmetric)
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            corr[i * n + j] = upper[idx];
            corr[j * n + i] = upper[idx];
            idx += 1;
        }
    }

    let shuffled = ClimateCorrelations {
        n_indices: n,
        correlations: corr,
        variant: "Null".into(),
    };

    correlation_to_tpm(&shuffled)
}

/// Generate monthly seasonal variation of correlations.
///
/// ENSO peaks in boreal winter (DJF), weakens in spring (MAM).
/// NAO is strongest in winter, weaker in summer.
pub fn generate_monthly_tpms(base: &ClimateCorrelations) -> Vec<(String, TransitionMatrix)> {
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let n = base.n_indices;

    // ENSO seasonal modulation: peaks in DJF (months 0,1,11)
    let enso_seasonal = [1.3, 1.2, 0.9, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3];
    // NAO seasonal modulation: peaks in DJF
    let nao_seasonal = [1.4, 1.3, 1.0, 0.7, 0.5, 0.4, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3];

    months
        .iter()
        .enumerate()
        .map(|(m, &name)| {
            let mut corr = base.correlations.clone();

            // Modulate ENSO connections
            for j in 0..n {
                if j != INDEX_ENSO {
                    corr[INDEX_ENSO * n + j] *= enso_seasonal[m];
                    corr[j * n + INDEX_ENSO] *= enso_seasonal[m];
                    // Clamp to valid correlation range
                    corr[INDEX_ENSO * n + j] = corr[INDEX_ENSO * n + j].min(0.95);
                    corr[j * n + INDEX_ENSO] = corr[j * n + INDEX_ENSO].min(0.95);
                }
            }

            // Modulate NAO connections
            for j in 0..n {
                if j != INDEX_NAO {
                    corr[INDEX_NAO * n + j] *= nao_seasonal[m];
                    corr[j * n + INDEX_NAO] *= nao_seasonal[m];
                    corr[INDEX_NAO * n + j] = corr[INDEX_NAO * n + j].min(0.95);
                    corr[j * n + INDEX_NAO] = corr[j * n + INDEX_NAO].min(0.95);
                }
            }

            let monthly = ClimateCorrelations {
                n_indices: n,
                correlations: corr,
                variant: format!("{} modulated", name),
            };

            (name.to_string(), correlation_to_tpm(&monthly))
        })
        .collect()
}
