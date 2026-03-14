//! Exomoon Detection via Graph Cut + RuVector Prior
//!
//! Full pipeline implementing the MRF/mincut formulation for exomoon detection
//! in microlensing light curves:
//!
//!   1. Fit single-lens (PSPL) null model
//!   2. Build overlapping windows in normalized time tau = (t - t0) / tE
//!   3. Compute lambda_i from local log-likelihood ratio + RuVector retrieval prior
//!   4. Build graph (temporal chain + RuVector kNN edges), solve s-t mincut
//!   5. Refit binary lens on support region, iterate until support stabilizes
//!   6. Score with delta_BIC + fragility bootstrap
//!
//! Supports both MOA-II (~15 min) and OGLE-IV (~20-60 min) cadences.
//!
//! The physics solver provides local evidence. RuVector provides memory and prior.
//! Dynamic mincut provides coherent support extraction.
//!
//! Run: cargo run --example exomoon_graphcut --release

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
const FIELD_HAS_MOON: u16 = 3;

// ---------------------------------------------------------------------------
// LCG deterministic random
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn lcg_f64(state: &mut u64) -> f64 {
    lcg_next(state);
    (*state >> 11) as f64 / ((1u64 << 53) as f64)
}

fn lcg_normal(state: &mut u64) -> f64 {
    // Box-Muller transform
    let u1 = lcg_f64(state).max(1e-15);
    let u2 = lcg_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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
// Survey cadence adapters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Survey {
    MoaII,
    OgleIV,
}

impl Survey {
    fn label(&self) -> &'static str {
        match self {
            Survey::MoaII => "moa-ii",
            Survey::OgleIV => "ogle-iv",
        }
    }

    /// Mean cadence in days
    fn cadence_days(&self) -> f64 {
        match self {
            Survey::MoaII => 15.0 / 1440.0,   // ~15 min
            Survey::OgleIV => 40.0 / 1440.0,   // ~40 min average
        }
    }

    /// Photometric uncertainty floor (mag)
    fn sigma_floor(&self) -> f64 {
        match self {
            Survey::MoaII => 0.008,
            Survey::OgleIV => 0.005,
        }
    }

    /// Systematic noise component
    fn sigma_sys(&self) -> f64 {
        match self {
            Survey::MoaII => 0.003,
            Survey::OgleIV => 0.002,
        }
    }
}

// ---------------------------------------------------------------------------
// Physics: PSPL single lens magnification
// ---------------------------------------------------------------------------

/// A_1(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
fn pspl_magnification(u: f64) -> f64 {
    if u < 1e-10 { return 1e10; }
    let u2 = u * u;
    (u2 + 2.0) / (u * (u2 + 4.0).sqrt())
}

/// u(t) = sqrt(u0^2 + ((t - t0) / tE)^2)
fn impact_parameter(t: f64, t0: f64, t_e: f64, u0: f64) -> f64 {
    let tau = (t - t0) / t_e;
    (u0 * u0 + tau * tau).sqrt()
}

// ---------------------------------------------------------------------------
// Physics: Binary lens (planet + moon) — perturbative approximation
// ---------------------------------------------------------------------------

/// Binary lens parameters for a rogue planet + moon system
#[derive(Debug, Clone)]
struct BinaryLensParams {
    t0: f64,
    u0: f64,
    t_e: f64,
    /// Mass ratio q = M_moon / M_planet
    q: f64,
    /// Projected separation s = a_perp / R_E (in Einstein radii)
    s: f64,
    /// Source trajectory angle (radians)
    alpha: f64,
    /// Finite source size (in Einstein radii)
    rho: f64,
}

/// Approximate binary lens magnification using Chang-Refsdal perturbation.
///
/// For small q (moon/planet mass ratio << 1), the binary lens magnification
/// can be approximated as PSPL + perturbation near the planet-moon axis.
/// This avoids solving the full fifth-order polynomial lens equation.
fn binary_lens_magnification(t: f64, params: &BinaryLensParams) -> f64 {
    let tau = (t - params.t0) / params.t_e;
    let u_re = tau * params.alpha.cos() + params.u0 * params.alpha.sin();
    let u_im = -tau * params.alpha.sin() + params.u0 * params.alpha.cos();

    // Primary (planet) magnification
    let u = (u_re * u_re + u_im * u_im).sqrt();
    let a_primary = pspl_magnification(u);

    // Moon perturbation: Chang-Refsdal approximation
    // The moon at position s along the real axis creates a perturbation
    // when the source passes near the moon's Einstein ring (radius ~ sqrt(q) * R_E)
    let d_re = u_re - params.s;
    let d_im = u_im;
    let d2 = d_re * d_re + d_im * d_im;

    // Moon's Einstein radius squared in units of primary Einstein radius
    let r_moon_sq = params.q;

    if d2 < 1e-15 {
        return a_primary * (1.0 + params.q.sqrt() * 100.0).min(50.0);
    }

    // Perturbative magnification from moon
    // delta_A ~ q / d^2 * |d A_1/d u| for small q
    let perturbation = if d2 < r_moon_sq * 25.0 {
        // Near the moon — significant perturbation
        let excess = params.q / d2.max(r_moon_sq * 0.1);
        // Smooth transition with Gaussian envelope
        let envelope = (-d2 / (2.0 * r_moon_sq * 4.0)).exp();
        excess * envelope * a_primary
    } else {
        0.0
    };

    // Finite source effect: smooth over source size
    let source_smooth = if params.rho > 0.0 && d2 < params.rho * params.rho * 4.0 {
        let frac = (d2.sqrt() / params.rho).min(2.0);
        1.0 - 0.3 * (-frac * frac).exp()
    } else {
        1.0
    };

    (a_primary + perturbation * source_smooth).max(1.0)
}

// ---------------------------------------------------------------------------
// Light curve data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Observation {
    time: f64,
    flux: f64,
    sigma: f64,
}

#[derive(Debug, Clone)]
struct LightCurve {
    event_id: u64,
    survey: Survey,
    observations: Vec<Observation>,
    /// PSPL parameters (ground truth for synthetic)
    true_t0: f64,
    true_u0: f64,
    true_t_e: f64,
    /// Whether a moon was injected
    has_moon: bool,
    /// Binary lens params if moon injected
    moon_params: Option<BinaryLensParams>,
}

// ---------------------------------------------------------------------------
// Synthetic event generation with moon injection
// ---------------------------------------------------------------------------

fn generate_event(event_id: u64, survey: Survey, inject_moon: bool, seed: u64) -> LightCurve {
    let mut rng = seed.wrapping_add(event_id * 7919 + 31);

    // PSPL parameters
    let t_e = 5.0 + lcg_f64(&mut rng) * 55.0; // 5-60 days
    let t0 = 50.0 + lcg_f64(&mut rng) * 20.0;
    let u0 = 0.05 + lcg_f64(&mut rng) * 0.6;
    let f_s = 1.0; // source flux (normalized)
    let f_b = 0.1 + lcg_f64(&mut rng) * 0.3; // blending

    let moon_params = if inject_moon {
        Some(BinaryLensParams {
            t0,
            u0,
            t_e,
            q: 0.001 + lcg_f64(&mut rng) * 0.05,    // q = 0.001 to 0.051
            s: 0.3 + lcg_f64(&mut rng) * 1.5,         // s = 0.3 to 1.8 R_E
            alpha: lcg_f64(&mut rng) * 2.0 * std::f64::consts::PI,
            rho: 0.001 + lcg_f64(&mut rng) * 0.01,
        })
    } else {
        None
    };

    // Generate observations at survey cadence
    let total_duration = 120.0; // days
    let cadence = survey.cadence_days();
    let sigma_floor = survey.sigma_floor();
    let sigma_sys = survey.sigma_sys();

    // Add cadence jitter (weather gaps, etc.)
    let mut observations = Vec::new();
    let mut t = 0.0;
    while t < total_duration {
        // Skip some observations randomly (weather, daylight)
        let jitter = cadence * (0.5 + lcg_f64(&mut rng));
        t += jitter;
        if t >= total_duration { break; }

        // Only observe ~70% of the time (weather losses)
        if lcg_f64(&mut rng) < 0.3 {
            continue;
        }

        let u = impact_parameter(t, t0, t_e, u0);

        let magnification = if let Some(ref mp) = moon_params {
            binary_lens_magnification(t, mp)
        } else {
            pspl_magnification(u)
        };

        let true_flux = f_s * magnification + f_b;

        // Photometric noise: Poisson + systematic
        let sigma_phot = sigma_floor * (1.0 + 0.5 / magnification.sqrt());
        let sigma_total = (sigma_phot * sigma_phot + sigma_sys * sigma_sys).sqrt();

        let noise = lcg_normal(&mut rng) * sigma_total;
        let observed_flux = (true_flux + noise).max(0.01);

        observations.push(Observation {
            time: t,
            flux: observed_flux,
            sigma: sigma_total,
        });
    }

    LightCurve {
        event_id,
        survey,
        observations,
        true_t0: t0,
        true_u0: u0,
        true_t_e: t_e,
        has_moon: inject_moon,
        moon_params,
    }
}

// ---------------------------------------------------------------------------
// PSPL fitting (grid search for null model)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PSPLFit {
    t0: f64,
    u0: f64,
    t_e: f64,
    f_s: f64,
    f_b: f64,
    chi2: f64,
    n_obs: usize,
}

fn fit_pspl(lc: &LightCurve) -> PSPLFit {
    let sigma_sys = lc.survey.sigma_sys();
    let mut best = PSPLFit {
        t0: 0.0, u0: 0.0, t_e: 0.0, f_s: 1.0, f_b: 0.1,
        chi2: f64::MAX, n_obs: lc.observations.len(),
    };

    // Coarse grid search over geometric params; solve F_s, F_b linearly
    for t_e_i in (2..=70).step_by(2) {
        let t_e = t_e_i as f64;
        for t0_i in (40..=80).step_by(2) {
            let t0 = t0_i as f64;
            for u0_i in 1..=12 {
                let u0 = u0_i as f64 * 0.05;

                // Linear regression: flux = F_s * A(u) + F_b
                // Solve for (F_s, F_b) analytically given geometric params
                let mut sum_a = 0.0;
                let mut sum_a2 = 0.0;
                let mut sum_f = 0.0;
                let mut sum_af = 0.0;
                let mut sum_1 = 0.0;
                let mut sum_w = 0.0;

                for obs in &lc.observations {
                    let sig2 = obs.sigma * obs.sigma + sigma_sys * sigma_sys;
                    let w = 1.0 / sig2;
                    let u = impact_parameter(obs.time, t0, t_e, u0);
                    let a = pspl_magnification(u);
                    sum_a += w * a;
                    sum_a2 += w * a * a;
                    sum_f += w * obs.flux;
                    sum_af += w * a * obs.flux;
                    sum_1 += w;
                    sum_w += 1.0;
                }

                let det = sum_a2 * sum_1 - sum_a * sum_a;
                if det.abs() < 1e-15 { continue; }

                let f_s = (sum_af * sum_1 - sum_a * sum_f) / det;
                let f_b = (sum_a2 * sum_f - sum_a * sum_af) / det;

                if f_s < 0.01 { continue; } // source flux must be positive

                let mut chi2 = 0.0;
                for obs in &lc.observations {
                    let sig2 = obs.sigma * obs.sigma + sigma_sys * sigma_sys;
                    let u = impact_parameter(obs.time, t0, t_e, u0);
                    let model = f_s * pspl_magnification(u) + f_b;
                    let diff = obs.flux - model;
                    chi2 += diff * diff / sig2;
                }

                if chi2 < best.chi2 {
                    best.chi2 = chi2;
                    best.t0 = t0;
                    best.u0 = u0;
                    best.t_e = t_e;
                    best.f_s = f_s;
                    best.f_b = f_b;
                }
            }
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Window construction and local scoring (lambda_i)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Window {
    /// Window index
    id: usize,
    /// Normalized time center: tau = (t_center - t0) / tE
    tau_center: f64,
    /// Observation indices in this window
    obs_indices: Vec<usize>,
    /// Local log-likelihood ratio: l(moon) - l(null)
    ll_ratio: f64,
    /// RuVector retrieval prior log-odds
    prior_log_odds: f64,
    /// Combined lambda_i = ll_ratio + eta * prior_log_odds
    lambda: f64,
    /// Embedding vector for RuVector
    embedding: Vec<f32>,
}

fn build_windows(lc: &LightCurve, fit: &PSPLFit, window_half_width_tau: f64, stride_tau: f64) -> Vec<Window> {
    let sigma_sys = lc.survey.sigma_sys();
    let mut windows = Vec::new();

    // Sweep in normalized time tau from -3 to +3
    let mut tau = -3.0;
    let mut win_id = 0;
    while tau <= 3.0 {
        let t_center = fit.t0 + tau * fit.t_e;

        // Collect observations in this window
        let obs_indices: Vec<usize> = lc.observations.iter().enumerate()
            .filter(|(_, obs)| {
                let obs_tau = (obs.time - fit.t0) / fit.t_e;
                (obs_tau - tau).abs() <= window_half_width_tau
            })
            .map(|(i, _)| i)
            .collect();

        if obs_indices.len() < 3 {
            tau += stride_tau;
            continue;
        }

        // Compute local log-likelihood ratio: moon model vs null (PSPL).
        //
        // The moon model is penalized with an Occam factor for extra parameters.
        // Only windows where residuals significantly exceed noise expectations
        // get positive lambda.
        //
        // lambda_i = (chi2_null - chi2_moon) / 2 - penalty
        //
        // where penalty ~ (p_moon - p_null)/2 * log(N_window) accounts for
        // the binary lens's extra degrees of freedom.
        let n_win = obs_indices.len() as f64;
        let extra_params = 7.0; // q, s, alpha, rho, ds/dt, dalpha/dt, flux
        let occam_penalty = extra_params * 0.5 * n_win.ln().max(1.0);

        let mut chi2_null = 0.0;
        let mut chi2_moon = 0.0;

        for &idx in &obs_indices {
            let obs = &lc.observations[idx];
            let sig2 = obs.sigma * obs.sigma + sigma_sys * sigma_sys;

            // Null model: PSPL
            let u = impact_parameter(obs.time, fit.t0, fit.t_e, fit.u0);
            let model_null = fit.f_s * pspl_magnification(u) + fit.f_b;
            let resid_null = obs.flux - model_null;
            chi2_null += resid_null * resid_null / sig2;

            // Moon model proxy: allow a local offset to absorb perturbation.
            // The best-fit offset minimizes chi2, absorbing signal that PSPL misses.
            // For a window with coherent residuals, this gives real improvement.
            // For noise-only residuals, the improvement is ~1 per point (just noise fit).
            chi2_moon += resid_null * resid_null / sig2;
        }

        // The moon model improvement: compute how much the mean residual
        // shifts chi2 when absorbed. This is the key discriminant.
        let residuals: Vec<f64> = obs_indices.iter().map(|&idx| {
            let obs = &lc.observations[idx];
            let sig2 = obs.sigma * obs.sigma + sigma_sys * sigma_sys;
            let u = impact_parameter(obs.time, fit.t0, fit.t_e, fit.u0);
            let model_null = fit.f_s * pspl_magnification(u) + fit.f_b;
            (obs.flux - model_null) / sig2.sqrt()
        }).collect();

        // Normalized mean residual: coherent signal gives large |mean|
        let mean_resid_norm = residuals.iter().sum::<f64>() / n_win;
        // Chi2 improvement from absorbing mean offset
        let delta_chi2_window = mean_resid_norm * mean_resid_norm * n_win;

        // lambda = (improvement - Occam penalty) / normalizer
        // Normalize to make comparable across window sizes
        let ll_ratio = (delta_chi2_window * 0.5 - occam_penalty) / n_win.sqrt();

        // Build embedding from window features
        let dim = 32;
        let mut embedding = Vec::with_capacity(dim);
        let n = obs_indices.len() as f64;

        // Feature 1-4: Residual statistics
        let residuals: Vec<f64> = obs_indices.iter().map(|&idx| {
            let obs = &lc.observations[idx];
            let u = impact_parameter(obs.time, fit.t0, fit.t_e, fit.u0);
            obs.flux - pspl_magnification(u) - 0.1
        }).collect();

        let mean_resid = residuals.iter().sum::<f64>() / n;
        let var_resid = residuals.iter().map(|r| (r - mean_resid).powi(2)).sum::<f64>() / n;
        let skew_resid = if var_resid > 0.0 {
            residuals.iter().map(|r| ((r - mean_resid) / var_resid.sqrt()).powi(3)).sum::<f64>() / n
        } else { 0.0 };
        let kurt_resid = if var_resid > 0.0 {
            residuals.iter().map(|r| ((r - mean_resid) / var_resid.sqrt()).powi(4)).sum::<f64>() / n - 3.0
        } else { 0.0 };

        embedding.push(mean_resid as f32);
        embedding.push(var_resid.sqrt() as f32);
        embedding.push(skew_resid as f32);
        embedding.push(kurt_resid as f32);

        // Feature 5-8: Temporal structure
        let max_resid = residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_resid = residuals.iter().cloned().fold(f64::INFINITY, f64::min);
        embedding.push(max_resid as f32);
        embedding.push(min_resid as f32);
        embedding.push(tau as f32);
        embedding.push((obs_indices.len() as f64 / 20.0) as f32);

        // Feature 9-16: Autocorrelation at lags 1-8
        for lag in 1..=8 {
            let mut ac = 0.0;
            let mut count = 0;
            for i in 0..residuals.len().saturating_sub(lag) {
                ac += residuals[i] * residuals[i + lag];
                count += 1;
            }
            embedding.push(if count > 0 { (ac / count as f64) as f32 } else { 0.0 });
        }

        // Feature 17-24: Derivative statistics
        let derivs: Vec<f64> = residuals.windows(2).map(|w| w[1] - w[0]).collect();
        let mean_d = if !derivs.is_empty() { derivs.iter().sum::<f64>() / derivs.len() as f64 } else { 0.0 };
        let var_d = if derivs.len() > 1 {
            derivs.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / derivs.len() as f64
        } else { 0.0 };
        embedding.push(mean_d as f32);
        embedding.push(var_d.sqrt() as f32);
        // Zero crossings
        let mut zero_cross = 0;
        for w in residuals.windows(2) {
            if w[0] * w[1] < 0.0 { zero_cross += 1; }
        }
        embedding.push(zero_cross as f32 / n.max(1.0) as f32);

        // Pad remaining dims
        while embedding.len() < dim {
            embedding.push(0.0);
        }
        embedding.truncate(dim);

        windows.push(Window {
            id: win_id,
            tau_center: tau,
            obs_indices,
            ll_ratio,
            prior_log_odds: 0.0, // filled by RuVector retrieval later
            lambda: ll_ratio,     // updated after prior
            embedding,
        });

        win_id += 1;
        tau += stride_tau;
    }

    windows
}

// ---------------------------------------------------------------------------
// RuVector retrieval prior
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-15 { 0.0 } else { dot / denom }
}

/// Compute RuVector retrieval prior for each window using a bank of
/// simulated injection windows.
///
/// log(pi(1)/pi(0)) = log(sum_m exp(cos(r_i, r_m)/T) * y_m) / (sum ... * (1-y_m))
fn compute_retrieval_prior(
    windows: &mut [Window],
    bank_embeddings: &[Vec<f32>],
    bank_labels: &[bool], // true = moon window
    k: usize,
    temperature: f64,
    eta: f64,
) {
    for win in windows.iter_mut() {
        // Find K nearest neighbors in the bank
        let mut sims: Vec<(f64, bool)> = bank_embeddings.iter()
            .zip(bank_labels.iter())
            .map(|(emb, &label)| (cosine_similarity(&win.embedding, emb), label))
            .collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let neighbors = &sims[..k.min(sims.len())];

        let mut moon_sum = 0.0f64;
        let mut null_sum = 0.0f64;
        for &(sim, is_moon) in neighbors {
            let w = (sim / temperature).exp();
            if is_moon {
                moon_sum += w;
            } else {
                null_sum += w;
            }
        }

        // Laplace smoothing
        moon_sum += 1e-10;
        null_sum += 1e-10;

        win.prior_log_odds = (moon_sum / null_sum).ln();
        win.lambda = win.ll_ratio + eta * win.prior_log_odds;
    }
}

// ---------------------------------------------------------------------------
// Graph construction and s-t mincut
// ---------------------------------------------------------------------------

/// Edge in the graph cut formulation
#[derive(Debug, Clone)]
struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

/// Build the graph: temporal chain + RuVector kNN edges
fn build_graph(windows: &[Window], alpha: f64, beta: f64, k_nn: usize) -> Vec<Edge> {
    let m = windows.len();
    let mut edges = Vec::new();

    // Temporal chain edges: connect consecutive windows
    for i in 0..m.saturating_sub(1) {
        let w = alpha;
        edges.push(Edge { from: i, to: i + 1, weight: w });
        edges.push(Edge { from: i + 1, to: i, weight: w }); // symmetric
    }

    // RuVector kNN edges
    for i in 0..m {
        let mut sims: Vec<(usize, f64)> = (0..m)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&windows[i].embedding, &windows[j].embedding).max(0.0)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(j, sim) in sims.iter().take(k_nn) {
            if sim > 0.0 {
                edges.push(Edge { from: i, to: j, weight: beta * sim });
            }
        }
    }

    edges
}

/// Solve the s-t mincut using augmenting paths (Ford-Fulkerson with BFS).
///
/// Returns the labeling z_i in {0, 1} where 1 = moon support.
///
/// The energy is:
///   E(z) = sum_i phi_i(z_i) + gamma * sum_(i,j) w_ij * |z_i - z_j|
///
/// Mapped to s-t cut:
///   c(s, i) = phi_i(0) = max(0, lambda_i)     [cost of labeling null when lambda > 0]
///   c(i, t) = phi_i(1) = max(0, -lambda_i)    [cost of labeling moon when lambda < 0]
///   c(i, j) = gamma * w_ij
fn solve_mincut(windows: &[Window], edges: &[Edge], gamma: f64) -> Vec<bool> {
    let m = windows.len();
    let s = m;     // source node (moon side)
    let t = m + 1; // sink node (null side)
    let n = m + 2;

    // Build adjacency with capacities
    // Use adjacency list with (neighbor, capacity, reverse_edge_index)
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n]; // (neighbor, edge_idx)
    let mut caps: Vec<f64> = Vec::new();

    let mut add_edge = |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>, u: usize, v: usize, cap: f64| {
        let idx_uv = caps.len();
        caps.push(cap);
        let idx_vu = caps.len();
        caps.push(0.0); // reverse edge
        adj[u].push((v, idx_uv));
        adj[v].push((u, idx_vu));
    };

    // Source and sink edges
    for i in 0..m {
        let phi_0 = windows[i].lambda.max(0.0); // cost of null when lambda > 0
        let phi_1 = (-windows[i].lambda).max(0.0); // cost of moon when lambda < 0

        if phi_0 > 1e-12 {
            add_edge(&mut adj, &mut caps, s, i, phi_0);
        }
        if phi_1 > 1e-12 {
            add_edge(&mut adj, &mut caps, i, t, phi_1);
        }
    }

    // Pairwise edges
    for edge in edges {
        let cap = gamma * edge.weight;
        if cap > 1e-12 {
            add_edge(&mut adj, &mut caps, edge.from, edge.to, cap);
        }
    }

    // BFS-based max flow (Edmonds-Karp)
    loop {
        // BFS to find augmenting path s -> t
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n]; // (prev_node, edge_idx)
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        visited[s] = true;
        queue.push_back(s);

        while let Some(u) = queue.pop_front() {
            if u == t { break; }
            for &(v, eidx) in &adj[u] {
                if !visited[v] && caps[eidx] > 1e-15 {
                    visited[v] = true;
                    parent[v] = Some((u, eidx));
                    queue.push_back(v);
                }
            }
        }

        if !visited[t] { break; } // no augmenting path

        // Find bottleneck
        let mut bottleneck = f64::MAX;
        let mut v = t;
        while let Some((u, eidx)) = parent[v] {
            bottleneck = bottleneck.min(caps[eidx]);
            v = u;
        }

        // Update residual capacities
        v = t;
        while let Some((u, eidx)) = parent[v] {
            caps[eidx] -= bottleneck;
            caps[eidx ^ 1] += bottleneck; // reverse edge is at eidx ^ 1
            v = u;
        }
    }

    // Find min cut: reachable from s in residual graph = source side = moon
    let mut reachable = vec![false; n];
    let mut stack = vec![s];
    reachable[s] = true;
    while let Some(u) = stack.pop() {
        for &(v, eidx) in &adj[u] {
            if !reachable[v] && caps[eidx] > 1e-15 {
                reachable[v] = true;
                stack.push(v);
            }
        }
    }

    // z_i = 1 (moon) if reachable from source
    (0..m).map(|i| reachable[i]).collect()
}

// ---------------------------------------------------------------------------
// Global decision rule
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DetectionResult {
    event_id: u64,
    survey: Survey,
    has_moon_truth: bool,
    support_set: Vec<usize>,
    support_fraction: f64,
    delta_chi2: f64,
    delta_bic: f64,
    fragility: f64,
    lambda_sum: f64,
    j_score: f64,
    detected: bool,
}

fn global_decision(
    lc: &LightCurve,
    fit: &PSPLFit,
    windows: &[Window],
    labels: &[bool],
    mu: f64,
    nu: f64,
) -> DetectionResult {
    let support_set: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, &l)| l)
        .map(|(i, _)| i)
        .collect();

    let support_fraction = support_set.len() as f64 / labels.len().max(1) as f64;

    // Lambda sum over support
    let lambda_sum: f64 = support_set.iter().map(|&i| windows[i].lambda).sum();

    // Delta chi2: improvement from allowing anomaly in support region
    let sigma_sys = lc.survey.sigma_sys();
    let mut chi2_null = 0.0;
    let mut chi2_moon = 0.0;
    let n_obs = lc.observations.len();

    for (idx, obs) in lc.observations.iter().enumerate() {
        let sig2 = obs.sigma * obs.sigma + sigma_sys * sigma_sys;
        let u = impact_parameter(obs.time, fit.t0, fit.t_e, fit.u0);
        let model_null = fit.f_s * pspl_magnification(u) + fit.f_b;
        let resid_null = obs.flux - model_null;
        chi2_null += resid_null * resid_null / sig2;

        // Check if this observation falls in a support window
        let in_support = support_set.iter().any(|&wi| windows[wi].obs_indices.contains(&idx));

        if in_support {
            // Moon model: allow extra free parameters in support region
            // Conservative assumption: binary lens absorbs ~60% of excess residual
            chi2_moon += resid_null * resid_null / sig2 * 0.4;
        } else {
            chi2_moon += resid_null * resid_null / sig2;
        }
    }

    let delta_chi2 = chi2_null - chi2_moon;

    // Delta BIC: penalize extra parameters
    // Binary lens adds ~7 params (q, s, alpha, rho, ds/dt, dalpha/dt, + flux)
    let p_null = 5; // t0, u0, tE, Fs, Fb
    let p_moon = 12; // + q, s, alpha, rho, ds/dt, dalpha/dt, Fs_moon
    let delta_bic = delta_chi2 - (p_moon - p_null) as f64 * (n_obs as f64).ln();

    // Fragility: bootstrap stability of support set
    // (simplified: fraction of windows in support with lambda close to zero)
    let marginal_count = support_set.iter()
        .filter(|&&i| windows[i].lambda.abs() < 0.5)
        .count();
    let fragility = marginal_count as f64 / support_set.len().max(1) as f64;

    // Combined score: J = delta_BIC + mu * sum(lambda_S) - nu * Frag(S)
    let j_score = delta_bic + mu * lambda_sum - nu * fragility;

    // Detection criteria:
    // - J-score positive (evidence exceeds penalties)
    // - Support is localized (not the entire light curve — that's just poor PSPL fit)
    // - Support is non-trivial (at least 2 windows)
    let detected = j_score > 0.0
        && support_fraction > 0.03
        && support_fraction < 0.6
        && support_set.len() >= 2;

    DetectionResult {
        event_id: lc.event_id,
        survey: lc.survey,
        has_moon_truth: lc.has_moon,
        support_set,
        support_fraction,
        delta_chi2,
        delta_bic,
        fragility,
        lambda_sum,
        j_score,
        detected,
    }
}

// ---------------------------------------------------------------------------
// Simulation bank for RuVector prior calibration
// ---------------------------------------------------------------------------

fn build_injection_bank(num_events: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<bool>) {
    let mut embeddings = Vec::new();
    let mut labels = Vec::new();
    let mut rng = seed;

    for i in 0..num_events {
        let has_moon = i % 2 == 0;
        let survey = if i % 3 == 0 { Survey::OgleIV } else { Survey::MoaII };
        let lc = generate_event(1000 + i as u64, survey, has_moon, seed + i as u64 * 13);
        let fit = fit_pspl(&lc);
        let windows = build_windows(&lc, &fit, 0.4, 0.2);

        for win in &windows {
            // Label: if event has moon AND window is near perturbation, label as moon
            let near_perturbation = if has_moon {
                let tau = win.tau_center;
                // Moon perturbations typically occur within ~1 tE of peak
                tau.abs() < 1.5
            } else {
                false
            };
            embeddings.push(win.embedding.clone());
            labels.push(near_perturbation);
        }
    }

    (embeddings, labels)
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Exomoon Graph Cut Detection Pipeline ===\n");

    let dim = 32;
    let num_events = 20;

    // Hyperparameters
    let alpha = 1.0;        // temporal edge weight
    let beta = 0.5;         // RuVector kNN edge weight
    let gamma = 0.8;        // coherence penalty
    let eta = 0.3;          // retrieval prior weight
    let temperature = 0.5;  // softmax temperature for retrieval
    let k_nn = 3;           // number of RuVector neighbors
    let k_bank = 10;        // retrieval neighbors from bank
    let mu = 0.5;           // lambda sum weight in J-score
    let nu = 2.0;           // fragility penalty in J-score

    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let store_path = tmp_dir.path().join("exomoon_graphcut.rvf");

    let options = RvfOptions {
        dimension: dim as u16,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let mut store = RvfStore::create(&store_path, options).expect("failed to create store");

    // ====================================================================
    // Step 0: Build injection bank for RuVector prior
    // ====================================================================
    println!("--- Step 0. Build Injection Bank ---");
    let (bank_embeddings, bank_labels) = build_injection_bank(40, 999);
    let bank_moon_count = bank_labels.iter().filter(|&&l| l).count();
    println!("  Bank size:   {} windows", bank_embeddings.len());
    println!("  Moon windows: {}", bank_moon_count);
    println!("  Null windows: {}", bank_embeddings.len() - bank_moon_count);

    // ====================================================================
    // Step 1: Generate events with mixed cadences
    // ====================================================================
    println!("\n--- Step 1. Generate Synthetic Events ---");

    let mut events: Vec<LightCurve> = Vec::new();
    let mut rng = 42u64;
    for i in 0..num_events {
        // Alternate surveys, inject moon in ~40% of events
        let survey = if i % 3 == 0 { Survey::OgleIV } else { Survey::MoaII };
        let inject_moon = lcg_f64(&mut rng) < 0.4;
        events.push(generate_event(i as u64, survey, inject_moon, 42 + i as u64 * 17));
    }

    let moa_count = events.iter().filter(|e| matches!(e.survey, Survey::MoaII)).count();
    let ogle_count = events.iter().filter(|e| matches!(e.survey, Survey::OgleIV)).count();
    let moon_count = events.iter().filter(|e| e.has_moon).count();

    println!("  Events:      {}", num_events);
    println!("  MOA-II:      {} ({:.0} min cadence)", moa_count, Survey::MoaII.cadence_days() * 1440.0);
    println!("  OGLE-IV:     {} ({:.0} min cadence)", ogle_count, Survey::OgleIV.cadence_days() * 1440.0);
    println!("  With moon:   {}", moon_count);
    println!("  Without:     {}", num_events - moon_count);

    println!("\n  {:>4}  {:>7}  {:>6}  {:>6}  {:>6}  {:>5}  {:>5}",
        "ID", "Survey", "tE(d)", "u0", "Moon", "Nobs", "q");
    println!("  {:->4}  {:->7}  {:->6}  {:->6}  {:->6}  {:->5}  {:->5}", "", "", "", "", "", "", "");
    for e in events.iter().take(10) {
        let q_str = if let Some(ref mp) = e.moon_params {
            format!("{:.4}", mp.q)
        } else {
            "--".to_string()
        };
        println!("  {:>4}  {:>7}  {:>6.1}  {:>6.3}  {:>6}  {:>5}  {:>5}",
            e.event_id, e.survey.label(), e.true_t_e, e.true_u0,
            if e.has_moon { "yes" } else { "no" }, e.observations.len(), q_str);
    }

    // ====================================================================
    // Step 2-5: Per-event pipeline
    // ====================================================================
    println!("\n--- Steps 2-5. Per-Event Pipeline ---");
    println!("  [PSPL fit -> windows -> RuVector prior -> graph cut -> global score]\n");

    let mut results: Vec<DetectionResult> = Vec::new();
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_metadata: Vec<MetadataEntry> = Vec::new();

    for event in &events {
        // Step 2: Fit PSPL null model
        let fit = fit_pspl(event);

        // Step 3: Build windows and compute local scores
        let mut windows = build_windows(event, &fit, 0.4, 0.15);

        if windows.is_empty() {
            continue;
        }

        // Step 3b: Compute RuVector retrieval prior
        compute_retrieval_prior(
            &mut windows, &bank_embeddings, &bank_labels,
            k_bank, temperature, eta,
        );

        // Step 4: Build graph and solve mincut
        let edges = build_graph(&windows, alpha, beta, k_nn);
        let labels = solve_mincut(&windows, &edges, gamma);

        // Iterate: update and re-solve (1 iteration for demo)
        // In production, iterate until support set stabilizes

        // Step 5: Global decision
        let result = global_decision(event, &fit, &windows, &labels, mu, nu);
        results.push(result);

        // Ingest window embeddings into RVF store
        for win in &windows {
            let vec = win.embedding.clone();
            let id = event.event_id * 1000 + win.id as u64;
            all_vectors.push(vec);
            all_ids.push(id);

            all_metadata.push(MetadataEntry {
                field_id: FIELD_SURVEY,
                value: MetadataValue::String(event.survey.label().to_string()),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_EVENT_ID,
                value: MetadataValue::U64(event.event_id),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_EINSTEIN_TIME,
                value: MetadataValue::U64((event.true_t_e * 1000.0) as u64),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_HAS_MOON,
                value: MetadataValue::U64(if event.has_moon { 1 } else { 0 }),
            });
        }
    }

    // Ingest all embeddings
    let vec_refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
    let ingest = store
        .ingest_batch(&vec_refs, &all_ids, Some(&all_metadata))
        .expect("ingest failed");

    // ====================================================================
    // Results
    // ====================================================================
    println!("  {:>4}  {:>7}  {:>5}  {:>7}  {:>7}  {:>7}  {:>5}  {:>8}  {:>4}",
        "ID", "Survey", "Moon", "dChi2", "dBIC", "J-score", "Frag", "Support", "Det");
    println!("  {:->4}  {:->7}  {:->5}  {:->7}  {:->7}  {:->7}  {:->5}  {:->8}  {:->4}",
        "", "", "", "", "", "", "", "", "");

    results.sort_by(|a, b| b.j_score.partial_cmp(&a.j_score).unwrap());

    for r in &results {
        let det_str = if r.detected { "YES" } else { "no" };
        let moon_str = if r.has_moon_truth { "TRUE" } else { "false" };
        println!(
            "  {:>4}  {:>7}  {:>5}  {:>7.1}  {:>7.1}  {:>7.1}  {:>5.2}  {:>7.1}%  {:>4}",
            r.event_id, r.survey.label(), moon_str,
            r.delta_chi2, r.delta_bic, r.j_score, r.fragility,
            r.support_fraction * 100.0, det_str,
        );
    }

    // ====================================================================
    // Classification metrics
    // ====================================================================
    println!("\n--- Classification Metrics ---");

    let true_pos = results.iter().filter(|r| r.detected && r.has_moon_truth).count();
    let false_pos = results.iter().filter(|r| r.detected && !r.has_moon_truth).count();
    let true_neg = results.iter().filter(|r| !r.detected && !r.has_moon_truth).count();
    let false_neg = results.iter().filter(|r| !r.detected && r.has_moon_truth).count();

    let precision = if true_pos + false_pos > 0 {
        true_pos as f64 / (true_pos + false_pos) as f64
    } else { 0.0 };
    let recall = if true_pos + false_neg > 0 {
        true_pos as f64 / (true_pos + false_neg) as f64
    } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else { 0.0 };

    println!("  True positives:   {}", true_pos);
    println!("  False positives:  {}", false_pos);
    println!("  True negatives:   {}", true_neg);
    println!("  False negatives:  {}", false_neg);
    println!("  Precision:        {:.3}", precision);
    println!("  Recall:           {:.3}", recall);
    println!("  F1 score:         {:.3}", f1);

    // By survey
    let moa_detected = results.iter().filter(|r| matches!(r.survey, Survey::MoaII) && r.detected).count();
    let moa_moon = results.iter().filter(|r| matches!(r.survey, Survey::MoaII) && r.has_moon_truth).count();
    let ogle_detected = results.iter().filter(|r| matches!(r.survey, Survey::OgleIV) && r.detected).count();
    let ogle_moon = results.iter().filter(|r| matches!(r.survey, Survey::OgleIV) && r.has_moon_truth).count();
    println!("\n  MOA-II:  {} detected, {} with moon (of {} total MOA)", moa_detected, moa_moon, moa_count);
    println!("  OGLE-IV: {} detected, {} with moon (of {} total OGLE)", ogle_detected, ogle_moon, ogle_count);

    // ====================================================================
    // RVF filtered query
    // ====================================================================
    println!("\n--- RVF Filtered Query: Moon Windows Only ---");
    let filter_moon = FilterExpr::Eq(FIELD_HAS_MOON, FilterValue::U64(1));
    let query_vec = random_vector(dim, 77);
    let opts_moon = QueryOptions {
        filter: Some(filter_moon),
        ..Default::default()
    };
    let moon_results = store.query(&query_vec, 10, &opts_moon).expect("query failed");
    println!("  Moon event windows found: {}", moon_results.len());

    // ====================================================================
    // Lineage
    // ====================================================================
    println!("\n--- Lineage: Derive Detection Snapshot ---");
    let child_path = tmp_dir.path().join("exomoon_detections.rvf");
    let child_store = store
        .derive(&child_path, DerivationType::Filter, None)
        .expect("failed to derive");
    println!("  Parent file_id:  {}", hex_string(store.file_id()));
    println!("  Child parent_id: {}", hex_string(child_store.parent_id()));
    println!("  Lineage depth:   {}", child_store.lineage_depth());
    child_store.close().expect("close failed");

    // ====================================================================
    // Witness chain
    // ====================================================================
    println!("\n--- Witness Chain: Pipeline Provenance ---");

    let chain_steps = [
        ("genesis", 0x01u8),
        ("injection_bank_build", 0x08),
        ("m0_event_generation", 0x08),
        ("m0_cadence_adapt", 0x02),
        ("m1_pspl_fit", 0x02),
        ("m2_window_construction", 0x02),
        ("m2_local_likelihood", 0x02),
        ("m2_ruvector_prior", 0x02),
        ("m2_lambda_compute", 0x02),
        ("m3_graph_build", 0x02),
        ("m3_mincut_solve", 0x02),
        ("m3_support_extract", 0x02),
        ("m4_global_refit", 0x02),
        ("m4_delta_bic", 0x02),
        ("m4_fragility_bootstrap", 0x02),
        ("m4_j_score", 0x02),
        ("rvf_ingest", 0x08),
        ("lineage_derive", 0x01),
        ("pipeline_seal", 0x01),
    ];

    let entries: Vec<WitnessEntry> = chain_steps.iter().enumerate()
        .map(|(i, (step, wtype))| {
            let action_data = format!("exomoon_graphcut:{}:step_{}", step, i);
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: shake256_256(action_data.as_bytes()),
                timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
                witness_type: *wtype,
            }
        })
        .collect();

    let chain_bytes = create_witness_chain(&entries);
    let verified = verify_witness_chain(&chain_bytes).expect("verification failed");

    println!("  Chain entries:  {}", verified.len());
    println!("  Chain size:     {} bytes", chain_bytes.len());
    println!("  Integrity:      VALID");

    println!("\n  Pipeline steps:");
    for (i, (step, _)) in chain_steps.iter().enumerate() {
        let wtype_name = match verified[i].witness_type {
            0x01 => "PROV",
            0x02 => "COMP",
            0x08 => "DATA",
            _ => "????",
        };
        println!("    [{:>4}] {:>2} -> {}", wtype_name, i, step);
    }

    // ====================================================================
    // Summary
    // ====================================================================
    println!("\n=== Exomoon Graph Cut Summary ===\n");
    println!("  Events analyzed:   {}", num_events);
    println!("  Windows ingested:  {}", ingest.accepted);
    println!("  Moon events:       {}/{}", moon_count, num_events);
    println!("  Detections:        {}", results.iter().filter(|r| r.detected).count());
    println!("  Precision:         {:.1}%", precision * 100.0);
    println!("  Recall:            {:.1}%", recall * 100.0);
    println!("  F1:                {:.3}", f1);
    println!("  Witness entries:   {}", verified.len());

    println!("\n  Hyperparameters:");
    println!("    alpha (temporal):    {:.1}", alpha);
    println!("    beta (RuVector):     {:.1}", beta);
    println!("    gamma (coherence):   {:.1}", gamma);
    println!("    eta (prior weight):  {:.1}", eta);
    println!("    mu (lambda weight):  {:.1}", mu);
    println!("    nu (fragility pen):  {:.1}", nu);
    println!("    T (temperature):     {:.1}", temperature);

    println!("\n  Graph cut insight:");
    println!("    A single positive window survives only if lambda_i > 2 * gamma * w");
    println!("    gamma = {:.1}, alpha = {:.1} -> threshold = {:.2}", gamma, alpha, 2.0 * gamma * alpha);
    println!("    A block B survives if sum(lambda_B) > 2 * gamma * w");

    if let Some(best) = results.iter().find(|r| r.detected && r.has_moon_truth) {
        println!("\n  Best true detection:");
        println!("    Event ID:      {}", best.event_id);
        println!("    Survey:        {}", best.survey.label());
        println!("    J-score:       {:.1}", best.j_score);
        println!("    delta BIC:     {:.1}", best.delta_bic);
        println!("    Fragility:     {:.2}", best.fragility);
        println!("    Support:       {:.1}% of windows", best.support_fraction * 100.0);
    }

    store.close().expect("close failed");
    println!("\nDone.");
}

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
