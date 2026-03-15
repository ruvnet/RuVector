//! Microlensing Detection Pipeline (M0-M3) using RVF
//!
//! Demonstrates a four-stage microlensing detection pipeline for rogue planets
//! and exomoon candidates, extending ADR-040 to gravitational microlensing:
//!
//!   M0 Ingest:            Synthetic OGLE/MOA microlensing light curves
//!   M1 Single-Lens:       Paczynski PSPL curve fitting + SNR threshold
//!   M2 Anomaly Detection: Residual analysis for planetary/moon perturbations
//!   M3 Coherence Gating:  Multi-model comparison with mincut-style scoring
//!
//! Output: Ranked microlensing candidate list with anomaly classification
//!
//! Data approach: Synthetic simulations with known ground truth for validation.
//! In production, ingest real OGLE/MOA light curves from public archives.
//!
//! Run: cargo run --example microlensing_detection

use rvf_runtime::{
    FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore,
};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_types::DerivationType;
use rvf_crypto::{create_witness_chain, verify_witness_chain, shake256_256, WitnessEntry};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Metadata field IDs
// ---------------------------------------------------------------------------

const FIELD_SURVEY: u16 = 0;
const FIELD_EVENT_ID: u16 = 1;
const FIELD_EINSTEIN_TIME: u16 = 2;
const FIELD_ANOMALY_TYPE: u16 = 3;

// ---------------------------------------------------------------------------
// LCG helpers
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn lcg_f64(state: &mut u64) -> f64 {
    lcg_next(state);
    (*state >> 11) as f64 / ((1u64 << 53) as f64)
}

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut x = seed.wrapping_add(1);
    for _ in 0..dim {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    v
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Type of anomaly injected into a microlensing event
#[derive(Debug, Clone, Copy, PartialEq)]
enum AnomalyType {
    /// No anomaly — clean PSPL event
    None,
    /// Binary lens: bound planet causing caustic crossing
    BinaryPlanet,
    /// Free-floating (rogue) planet: short Einstein time event
    RoguePlanet,
    /// Exomoon perturbation: weak secondary bump near peak
    Exomoon,
}

impl AnomalyType {
    fn label(&self) -> &'static str {
        match self {
            AnomalyType::None => "clean",
            AnomalyType::BinaryPlanet => "binary",
            AnomalyType::RoguePlanet => "rogue",
            AnomalyType::Exomoon => "exomoon",
        }
    }
}

#[derive(Debug, Clone)]
struct MicrolensingEvent {
    event_id: u64,
    survey: &'static str,
    /// Einstein crossing time in days
    t_e: f64,
    /// Time of closest approach (days from start)
    t_0: f64,
    /// Minimum impact parameter (in Einstein radii)
    u_0: f64,
    /// Baseline magnitude
    baseline_mag: f64,
    /// Injected anomaly type
    anomaly_type: AnomalyType,
    /// Time samples (days)
    time: Vec<f64>,
    /// Magnification values
    magnification: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PSPLFit {
    event_id: u64,
    t_e_fit: f64,
    t_0_fit: f64,
    u_0_fit: f64,
    chi2: f64,
    snr: f64,
    residual_rms: f64,
}

#[derive(Debug, Clone)]
struct AnomalyCandidate {
    event_id: u64,
    pspl_fit: PSPLFit,
    anomaly_snr: f64,
    anomaly_duration: f64,
    anomaly_time: f64,
    classified_type: AnomalyType,
}

#[derive(Debug, Clone)]
struct ScoredCandidate {
    candidate: AnomalyCandidate,
    pspl_residual_score: f64,
    anomaly_strength: f64,
    temporal_consistency: f64,
    model_preference: f64,
    total_score: f64,
    passed: bool,
    true_type: AnomalyType,
}

// ---------------------------------------------------------------------------
// M0: Synthetic microlensing event generation
// ---------------------------------------------------------------------------

/// Paczynski magnification for a point-source point-lens (PSPL) event.
/// A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
fn pspl_magnification(u: f64) -> f64 {
    if u < 1e-10 {
        return 1e10; // avoid division by zero at perfect alignment
    }
    let u2 = u * u;
    (u2 + 2.0) / (u * (u2 + 4.0).sqrt())
}

/// Impact parameter as a function of time for PSPL.
/// u(t) = sqrt(u_0^2 + ((t - t_0) / t_E)^2)
fn impact_parameter(t: f64, t_0: f64, t_e: f64, u_0: f64) -> f64 {
    let tau = (t - t_0) / t_e;
    (u_0 * u_0 + tau * tau).sqrt()
}

fn generate_microlensing_event(event_id: u64, seed: u64) -> MicrolensingEvent {
    let mut rng = seed.wrapping_add(event_id * 7919);

    let surveys = ["ogle-iv", "moa-ii"];
    let survey = surveys[(lcg_next(&mut rng) >> 33) as usize % 2];

    // Decide anomaly type based on seed (roughly: 40% clean, 25% binary, 20% rogue, 15% exomoon)
    let anomaly_roll = lcg_f64(&mut rng);
    let anomaly_type = if anomaly_roll < 0.40 {
        AnomalyType::None
    } else if anomaly_roll < 0.65 {
        AnomalyType::BinaryPlanet
    } else if anomaly_roll < 0.85 {
        AnomalyType::RoguePlanet
    } else {
        AnomalyType::Exomoon
    };

    // Einstein crossing time: 1-80 days (rogue planets are short: 0.5-3 days)
    let t_e = if anomaly_type == AnomalyType::RoguePlanet {
        0.5 + lcg_f64(&mut rng) * 2.5
    } else {
        5.0 + lcg_f64(&mut rng) * 75.0
    };

    let t_0 = 50.0 + lcg_f64(&mut rng) * 20.0; // peak around day 50-70
    let u_0 = 0.01 + lcg_f64(&mut rng) * 0.8;   // impact parameter
    let baseline_mag = 18.0 + lcg_f64(&mut rng) * 4.0; // 18-22 mag
    let noise_level = 0.005 + lcg_f64(&mut rng) * 0.02;

    let num_points = 500;
    let total_duration = 120.0; // 120 days of observation
    let mut time = Vec::with_capacity(num_points);
    let mut magnification = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let t = (i as f64 / num_points as f64) * total_duration;
        time.push(t);

        let u = impact_parameter(t, t_0, t_e, u_0);
        let mut mag = pspl_magnification(u);

        // Inject anomalies
        match anomaly_type {
            AnomalyType::BinaryPlanet => {
                // Caustic crossing: sharp spike near t_0 offset by planet separation
                let planet_sep = 0.5 + lcg_f64(&mut rng) * 1.5; // in Einstein radii
                let t_anomaly = t_0 + planet_sep * t_e * 0.3;
                let anomaly_width = t_e * 0.02;
                let dt = (t - t_anomaly).abs();
                if dt < anomaly_width * 3.0 {
                    let caustic_bump = 0.3 * mag * (-dt * dt / (2.0 * anomaly_width * anomaly_width)).exp();
                    mag += caustic_bump;
                }
            }
            AnomalyType::RoguePlanet => {
                // Short-duration event already handled by small t_e
                // Add slight asymmetry from finite source effects
                let tau = (t - t_0) / t_e;
                if tau.abs() < 1.5 {
                    mag *= 1.0 + 0.02 * tau.signum() * (-tau.abs()).exp();
                }
            }
            AnomalyType::Exomoon => {
                // Weak secondary bump displaced from main peak
                let moon_offset = t_e * (0.05 + lcg_f64(&mut rng) * 0.15);
                let moon_width = t_e * 0.01;
                let dt = (t - (t_0 + moon_offset)).abs();
                if dt < moon_width * 5.0 {
                    let moon_bump = 0.05 * mag * (-dt * dt / (2.0 * moon_width * moon_width)).exp();
                    mag += moon_bump;
                }
            }
            AnomalyType::None => {}
        }

        // Add photometric noise
        let noise = (lcg_f64(&mut rng) - 0.5) * 2.0 * noise_level * mag;
        mag += noise;
        mag = mag.max(1.0); // magnification >= 1

        magnification.push(mag);
    }

    MicrolensingEvent {
        event_id,
        survey,
        t_e,
        t_0,
        u_0,
        baseline_mag,
        anomaly_type,
        time,
        magnification,
    }
}

// ---------------------------------------------------------------------------
// M1: PSPL fitting (simplified grid search)
// ---------------------------------------------------------------------------

fn fit_pspl(event: &MicrolensingEvent) -> PSPLFit {
    let mut best_chi2 = f64::MAX;
    let mut best_t_e = 0.0;
    let mut best_t_0 = 0.0;
    let mut best_u_0 = 0.0;

    // Grid search over t_E, t_0, u_0
    let t_e_trials: Vec<f64> = (1..=80).map(|x| x as f64).collect();
    let t_0_range_start = 40.0;
    let t_0_range_end = 80.0;
    let t_0_step = 1.0;

    for &t_e in &t_e_trials {
        let mut t_0_trial = t_0_range_start;
        while t_0_trial <= t_0_range_end {
            for u_0_idx in 1..=10 {
                let u_0 = u_0_idx as f64 * 0.1;

                let mut chi2 = 0.0;
                for (i, &t) in event.time.iter().enumerate() {
                    let u = impact_parameter(t, t_0_trial, t_e, u_0);
                    let model_mag = pspl_magnification(u);
                    let diff = event.magnification[i] - model_mag;
                    chi2 += diff * diff;
                }

                if chi2 < best_chi2 {
                    best_chi2 = chi2;
                    best_t_e = t_e;
                    best_t_0 = t_0_trial;
                    best_u_0 = u_0;
                }
            }
            t_0_trial += t_0_step;
        }
    }

    // Compute residual RMS and SNR
    let mut residual_sum_sq = 0.0;
    let mut peak_signal = 0.0f64;
    for (i, &t) in event.time.iter().enumerate() {
        let u = impact_parameter(t, best_t_0, best_t_e, best_u_0);
        let model_mag = pspl_magnification(u);
        let residual = event.magnification[i] - model_mag;
        residual_sum_sq += residual * residual;
        peak_signal = peak_signal.max(event.magnification[i] - 1.0);
    }
    let residual_rms = (residual_sum_sq / event.time.len() as f64).sqrt();
    let snr = if residual_rms > 0.0 { peak_signal / residual_rms } else { 0.0 };

    PSPLFit {
        event_id: event.event_id,
        t_e_fit: best_t_e,
        t_0_fit: best_t_0,
        u_0_fit: best_u_0,
        chi2: best_chi2,
        snr,
        residual_rms,
    }
}

// ---------------------------------------------------------------------------
// M2: Anomaly detection in residuals
// ---------------------------------------------------------------------------

fn detect_anomaly(event: &MicrolensingEvent, fit: &PSPLFit) -> Option<AnomalyCandidate> {
    // Compute residuals
    let mut residuals = Vec::with_capacity(event.time.len());
    for (i, &t) in event.time.iter().enumerate() {
        let u = impact_parameter(t, fit.t_0_fit, fit.t_e_fit, fit.u_0_fit);
        let model_mag = pspl_magnification(u);
        residuals.push(event.magnification[i] - model_mag);
    }

    // Compute baseline noise from wings (far from peak)
    let mut wing_residuals = Vec::new();
    for (i, &t) in event.time.iter().enumerate() {
        let tau = ((t - fit.t_0_fit) / fit.t_e_fit).abs();
        if tau > 3.0 {
            wing_residuals.push(residuals[i]);
        }
    }
    if wing_residuals.is_empty() {
        return None;
    }
    let wing_mean: f64 = wing_residuals.iter().sum::<f64>() / wing_residuals.len() as f64;
    let wing_var: f64 = wing_residuals.iter().map(|r| (r - wing_mean).powi(2)).sum::<f64>()
        / wing_residuals.len() as f64;
    let wing_std = wing_var.sqrt();

    if wing_std < 1e-10 {
        return None;
    }

    // Search for significant residual excursions near the peak
    let mut max_anomaly_snr = 0.0;
    let mut anomaly_time = 0.0;
    let mut anomaly_start = 0.0;
    let mut anomaly_end = 0.0;
    let mut in_anomaly = false;

    for (i, &t) in event.time.iter().enumerate() {
        let tau = ((t - fit.t_0_fit) / fit.t_e_fit).abs();
        if tau > 3.0 {
            if in_anomaly {
                in_anomaly = false;
            }
            continue;
        }

        let r_snr = residuals[i].abs() / wing_std;
        if r_snr > 3.0 {
            if !in_anomaly {
                in_anomaly = true;
                anomaly_start = t;
            }
            anomaly_end = t;
            if r_snr > max_anomaly_snr {
                max_anomaly_snr = r_snr;
                anomaly_time = t;
            }
        } else if in_anomaly {
            in_anomaly = false;
        }
    }

    if max_anomaly_snr < 3.0 {
        return None;
    }

    let anomaly_duration = anomaly_end - anomaly_start;

    // Classify anomaly type from residual characteristics
    let classified_type = if fit.t_e_fit < 4.0 {
        // Very short Einstein time → likely rogue planet
        AnomalyType::RoguePlanet
    } else if anomaly_duration < fit.t_e_fit * 0.05 && max_anomaly_snr > 10.0 {
        // Sharp, high-SNR spike → likely caustic crossing (binary planet)
        AnomalyType::BinaryPlanet
    } else if anomaly_duration < fit.t_e_fit * 0.08 && max_anomaly_snr < 8.0 {
        // Weak, short bump → likely exomoon
        AnomalyType::Exomoon
    } else {
        AnomalyType::BinaryPlanet // default to binary for strong anomalies
    };

    Some(AnomalyCandidate {
        event_id: event.event_id,
        pspl_fit: fit.clone(),
        anomaly_snr: max_anomaly_snr,
        anomaly_duration,
        anomaly_time,
        classified_type,
    })
}

// ---------------------------------------------------------------------------
// M3: Coherence gating (multi-model comparison)
// ---------------------------------------------------------------------------

fn coherence_gate(candidate: &AnomalyCandidate, event: &MicrolensingEvent) -> ScoredCandidate {
    // PSPL residual score: how much better is a non-PSPL model?
    let pspl_residual_score = 1.0 / (1.0 + (-0.3 * (candidate.pspl_fit.residual_rms * 100.0 - 2.0)).exp());

    // Anomaly strength: sigmoid on anomaly SNR
    let anomaly_strength = 1.0 / (1.0 + (-0.5 * (candidate.anomaly_snr - 5.0)).exp());

    // Temporal consistency: anomaly timing relative to peak
    let dt_peak = (candidate.anomaly_time - candidate.pspl_fit.t_0_fit).abs();
    let temporal_consistency = if dt_peak < candidate.pspl_fit.t_e_fit * 2.0 {
        1.0 / (1.0 + dt_peak / candidate.pspl_fit.t_e_fit)
    } else {
        0.1 // anomaly far from peak is suspicious
    };

    // Model preference: ratio of anomaly duration to Einstein time
    // Binary planets have short anomalies, rogue planets ARE the short event
    let duration_ratio = candidate.anomaly_duration / candidate.pspl_fit.t_e_fit.max(0.1);
    let model_preference = if candidate.classified_type == AnomalyType::RoguePlanet {
        // For rogue planets, short t_E is the signal itself
        1.0 / (1.0 + candidate.pspl_fit.t_e_fit / 3.0)
    } else {
        // For binary/exomoon, want short anomaly relative to event
        1.0 / (1.0 + duration_ratio * 5.0)
    };

    let total_score = pspl_residual_score * 0.2
        + anomaly_strength * 0.35
        + temporal_consistency * 0.25
        + model_preference * 0.2;

    let passed = total_score > 0.35 && candidate.anomaly_snr > 4.0;

    ScoredCandidate {
        candidate: candidate.clone(),
        pspl_residual_score,
        anomaly_strength,
        temporal_consistency,
        model_preference,
        total_score,
        passed,
        true_type: event.anomaly_type,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Microlensing Detection Pipeline (M0-M3) ===\n");

    let dim = 64;
    let num_events = 30;

    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let store_path = tmp_dir.path().join("microlensing_detection.rvf");

    let options = RvfOptions {
        dimension: dim as u16,
        metric: DistanceMetric::L2,
        ..Default::default()
    };

    let mut store = RvfStore::create(&store_path, options).expect("failed to create store");

    // ====================================================================
    // M0: Ingest — generate synthetic microlensing events
    // ====================================================================
    println!("--- M0. Ingest: Synthetic Microlensing Events ---");

    let events: Vec<MicrolensingEvent> = (0..num_events)
        .map(|i| generate_microlensing_event(i, 42))
        .collect();

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_metadata: Vec<MetadataEntry> = Vec::new();

    for event in &events {
        // NOTE: Synthetic embeddings — seed-based, not derived from light curve features.
        // In production, embeddings would be computed from residual structure.
        let vec = random_vector(dim, event.event_id * 31 + 7);
        all_vectors.push(vec);
        all_ids.push(event.event_id);

        all_metadata.push(MetadataEntry {
            field_id: FIELD_SURVEY,
            value: MetadataValue::String(event.survey.to_string()),
        });
        all_metadata.push(MetadataEntry {
            field_id: FIELD_EVENT_ID,
            value: MetadataValue::U64(event.event_id),
        });
        all_metadata.push(MetadataEntry {
            field_id: FIELD_EINSTEIN_TIME,
            value: MetadataValue::U64((event.t_e * 1000.0) as u64),
        });
        all_metadata.push(MetadataEntry {
            field_id: FIELD_ANOMALY_TYPE,
            value: MetadataValue::String(event.anomaly_type.label().to_string()),
        });
    }

    let vec_refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
    let ingest = store
        .ingest_batch(&vec_refs, &all_ids, Some(&all_metadata))
        .expect("ingest failed");

    let ogle_count = events.iter().filter(|e| e.survey == "ogle-iv").count();
    let moa_count = events.iter().filter(|e| e.survey == "moa-ii").count();
    let clean_count = events.iter().filter(|e| e.anomaly_type == AnomalyType::None).count();
    let binary_count = events.iter().filter(|e| e.anomaly_type == AnomalyType::BinaryPlanet).count();
    let rogue_count = events.iter().filter(|e| e.anomaly_type == AnomalyType::RoguePlanet).count();
    let moon_count = events.iter().filter(|e| e.anomaly_type == AnomalyType::Exomoon).count();

    println!("  Events:      {}", num_events);
    println!("  Ingested:    {}", ingest.accepted);
    println!("  Embedding:   {} dims", dim);
    println!("  Surveys:     {} OGLE-IV, {} MOA-II", ogle_count, moa_count);
    println!("  Injected:    {} clean, {} binary, {} rogue, {} exomoon",
        clean_count, binary_count, rogue_count, moon_count);

    println!("\n  Sample events:");
    for event in events.iter().take(5) {
        println!(
            "    id={:>2} survey={:<7} tE={:>6.2}d u0={:.3} type={}",
            event.event_id, event.survey, event.t_e, event.u_0, event.anomaly_type.label()
        );
    }

    // ====================================================================
    // M1: PSPL fitting
    // ====================================================================
    println!("\n--- M1. Single-Lens Detection (PSPL Fitting) ---");

    let fits: Vec<PSPLFit> = events.iter().map(|e| fit_pspl(e)).collect();

    println!("  PSPL fits computed: {}", fits.len());
    println!("\n  {:>4}  {:>8}  {:>8}  {:>6}  {:>8}  {:>6}", "ID", "tE_fit", "t0_fit", "u0", "chi2", "SNR");
    println!("  {:->4}  {:->8}  {:->8}  {:->6}  {:->8}  {:->6}", "", "", "", "", "", "");
    for fit in fits.iter().take(10) {
        println!(
            "  {:>4}  {:>8.2}  {:>8.2}  {:>6.3}  {:>8.2}  {:>6.1}",
            fit.event_id, fit.t_e_fit, fit.t_0_fit, fit.u_0_fit, fit.chi2, fit.snr
        );
    }

    let high_snr: Vec<&PSPLFit> = fits.iter().filter(|f| f.snr > 5.0).collect();
    println!("\n  High-SNR events (>5): {}/{}", high_snr.len(), fits.len());

    // ====================================================================
    // M2: Anomaly detection in residuals
    // ====================================================================
    println!("\n--- M2. Anomaly Detection (Residual Analysis) ---");

    let mut anomaly_candidates: Vec<AnomalyCandidate> = Vec::new();
    for (event, fit) in events.iter().zip(fits.iter()) {
        if let Some(candidate) = detect_anomaly(event, fit) {
            anomaly_candidates.push(candidate);
        }
    }

    println!("  Anomalies detected: {}/{}", anomaly_candidates.len(), num_events);

    println!("\n  {:>4}  {:>10}  {:>8}  {:>8}  {:>10}",
        "ID", "Anom SNR", "Duration", "Time", "Class");
    println!("  {:->4}  {:->10}  {:->8}  {:->8}  {:->10}", "", "", "", "", "");
    for c in &anomaly_candidates {
        println!(
            "  {:>4}  {:>10.2}  {:>8.3}  {:>8.2}  {:>10}",
            c.event_id, c.anomaly_snr, c.anomaly_duration, c.anomaly_time, c.classified_type.label()
        );
    }

    // ====================================================================
    // M3: Coherence gating — multi-model scoring
    // ====================================================================
    println!("\n--- M3. Coherence Gating (Multi-Model Scoring) ---");

    let mut scored: Vec<ScoredCandidate> = Vec::new();
    for c in &anomaly_candidates {
        let event = &events[c.event_id as usize];
        scored.push(coherence_gate(c, event));
    }

    scored.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap());

    println!("  Score components: PSPL_res(0.2), Anomaly(0.35), Temporal(0.25), Model(0.2)");
    println!("  Pass threshold:   score > 0.35 AND anomaly_SNR > 4.0\n");

    println!(
        "  {:>4}  {:>6}  {:>6}  {:>6}  {:>6}  {:>7}  {:>6}  {:>8}  {:>8}",
        "ID", "PSPLr", "Anom", "Temp", "Model", "Total", "Pass", "Class", "Truth"
    );
    println!(
        "  {:->4}  {:->6}  {:->6}  {:->6}  {:->6}  {:->7}  {:->6}  {:->8}  {:->8}",
        "", "", "", "", "", "", "", "", ""
    );
    for sc in &scored {
        let pass_str = if sc.passed { "YES" } else { "no" };
        let correct = if sc.candidate.classified_type == sc.true_type { "*" } else { "" };
        println!(
            "  {:>4}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>7.4}  {:>6}  {:>8}  {:>7}{}",
            sc.candidate.event_id,
            sc.pspl_residual_score,
            sc.anomaly_strength,
            sc.temporal_consistency,
            sc.model_preference,
            sc.total_score,
            pass_str,
            sc.candidate.classified_type.label(),
            sc.true_type.label(),
            correct,
        );
    }

    let passed_count = scored.iter().filter(|s| s.passed).count();
    let correct_class = scored.iter()
        .filter(|s| s.passed && s.candidate.classified_type == s.true_type)
        .count();
    println!("\n  Passed gating:       {}/{}", passed_count, scored.len());
    println!("  Correct class:       {}/{} passed", correct_class, passed_count);

    // ====================================================================
    // Filtered query: OGLE-only events
    // ====================================================================
    println!("\n--- Filtered Query: OGLE-Only Events ---");

    let query_vec = random_vector(dim, 99);
    let filter_ogle = FilterExpr::Eq(FIELD_SURVEY, FilterValue::String("ogle-iv".to_string()));
    let opts_ogle = QueryOptions {
        filter: Some(filter_ogle),
        ..Default::default()
    };
    let results_ogle = store
        .query(&query_vec, 10, &opts_ogle)
        .expect("filtered query failed");
    println!("  OGLE-IV events found: {}", results_ogle.len());

    // ====================================================================
    // Lineage: derive anomaly snapshot
    // ====================================================================
    println!("\n--- Lineage: Derive Anomaly Snapshot ---");

    let child_path = tmp_dir.path().join("microlensing_anomalies.rvf");
    let child_store = store
        .derive(&child_path, DerivationType::Filter, None)
        .expect("failed to derive child store");

    let parent_id = store.file_id();
    let child_parent_id = child_store.parent_id();
    assert_eq!(parent_id, child_parent_id, "lineage parent mismatch");
    assert_eq!(child_store.lineage_depth(), 1);

    println!("  Parent file_id:  {}", hex_string(parent_id));
    println!("  Child parent_id: {}", hex_string(child_parent_id));
    println!("  Lineage depth:   {}", child_store.lineage_depth());
    println!("  Lineage verified: parent_id matches");

    child_store.close().expect("failed to close child");

    // ====================================================================
    // Witness chain
    // ====================================================================
    println!("\n--- Witness Chain: Pipeline Provenance ---");

    let chain_steps = [
        ("genesis", 0x01u8),
        ("m0_ingest", 0x08),
        ("m0_normalize", 0x02),
        ("m0_segment", 0x02),
        ("m1_pspl_fit", 0x02),
        ("m1_snr_filter", 0x02),
        ("m2_residual_compute", 0x02),
        ("m2_anomaly_search", 0x02),
        ("m2_classify", 0x02),
        ("m3_pspl_residual_gate", 0x02),
        ("m3_anomaly_strength_gate", 0x02),
        ("m3_temporal_gate", 0x02),
        ("m3_model_preference", 0x02),
        ("m3_final_score", 0x02),
        ("lineage_derive", 0x01),
        ("pipeline_seal", 0x01),
    ];

    let entries: Vec<WitnessEntry> = chain_steps
        .iter()
        .enumerate()
        .map(|(i, (step, wtype))| {
            let action_data = format!("microlensing_detection:{}:step_{}", step, i);
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: shake256_256(action_data.as_bytes()),
                timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
                witness_type: *wtype,
            }
        })
        .collect();

    let chain_bytes = create_witness_chain(&entries);
    let verified = verify_witness_chain(&chain_bytes).expect("chain verification failed");

    println!("  Chain entries:  {}", verified.len());
    println!("  Chain size:     {} bytes", chain_bytes.len());
    println!("  Integrity:      VALID");

    println!("\n  Pipeline steps:");
    for (i, (step, _)) in chain_steps.iter().enumerate() {
        let wtype_name = match verified[i].witness_type {
            0x01 => "PROV",
            0x02 => "COMP",
            0x05 => "ATTS",
            0x08 => "DATA",
            _ => "????",
        };
        println!("    [{:>4}] {:>2} -> {}", wtype_name, i, step);
    }

    // ====================================================================
    // Summary
    // ====================================================================
    println!("\n=== Microlensing Detection Summary ===\n");
    println!("  Events analyzed:     {}", num_events);
    println!("  Events ingested:     {}", ingest.accepted);
    println!("  Anomalies detected:  {}", anomaly_candidates.len());
    println!("  Passed gating:       {}", passed_count);
    println!("  Correct class:       {}/{}", correct_class, passed_count);
    println!("  Witness entries:     {}", verified.len());
    println!("  Lineage:             parent -> anomaly snapshot");

    // Classification breakdown
    let passed_binary = scored.iter().filter(|s| s.passed && s.candidate.classified_type == AnomalyType::BinaryPlanet).count();
    let passed_rogue = scored.iter().filter(|s| s.passed && s.candidate.classified_type == AnomalyType::RoguePlanet).count();
    let passed_moon = scored.iter().filter(|s| s.passed && s.candidate.classified_type == AnomalyType::Exomoon).count();
    println!("\n  Passed by type:");
    println!("    Binary planet:   {}", passed_binary);
    println!("    Rogue planet:    {}", passed_rogue);
    println!("    Exomoon:         {}", passed_moon);

    if let Some(best) = scored.iter().find(|s| s.passed) {
        println!("\n  Top candidate:");
        println!("    Event ID:   {}", best.candidate.event_id);
        println!("    Class:      {}", best.candidate.classified_type.label());
        println!("    Truth:      {}", best.true_type.label());
        println!("    Anom SNR:   {:.2}", best.candidate.anomaly_snr);
        println!("    Score:      {:.4}", best.total_score);
        println!("    tE (fit):   {:.2} days", best.candidate.pspl_fit.t_e_fit);
    }

    store.close().expect("failed to close store");
    println!("\nDone.");
}

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
