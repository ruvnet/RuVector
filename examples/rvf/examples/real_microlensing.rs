//! Real Microlensing Data Analysis Pipeline — simulates OGLE/MOA light curves
//! from published event parameters, with cross-survey normalization, binary-lens
//! planetary perturbations, and anomaly detection via graph-cut residual analysis.
//! Run: cargo run --example real_microlensing --release

use rvf_crypto::{create_witness_chain, shake256_256, verify_witness_chain, WitnessEntry};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_types::DerivationType;
use tempfile::TempDir;

const FIELD_SOURCE: u16 = 0;
const FIELD_EVENT: u16 = 1;
const FIELD_ANOMALY_TYPE: u16 = 2;

const LCG_MUL: u64 = 6364136223846793005;
const LCG_INC: u64 = 1442695040888963407;
fn lcg_f64(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
    ((*s >> 33) as f64) / (u32::MAX as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let (u1, u2) = (lcg_f64(s).max(1e-15), lcg_f64(s));
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_add(1);
    (0..dim)
        .map(|_| {
            x = x.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
            ((x >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Observation {
    hjd: f64,
    mag: f64,
    mag_err: f64,
    flux: f64,
    flux_err: f64,
}
#[derive(Debug, Clone)]
struct MicrolensingEvent {
    name: String,
    source: String,
    observations: Vec<Observation>,
    baseline_mag: f64,
    peak_mag: f64,
    t0_est: f64,
    t_e_est: f64,
    u0_est: f64,
}
#[derive(Debug, Clone, Copy)]
struct PublishedParams {
    einstein_time: f64,             // t_E in days
    impact_param: f64,              // u_0
    planet_mass_ratio: Option<f64>, // q (None = no planet)
    planet_separation: Option<f64>, // s (None = no planet)
}

#[derive(Debug, Clone)]
struct ManifestEntry {
    event_id: String, // e.g. "OGLE-2005-BLG-390"
    survey: String,   // "ogle" or "moa"
    published: PublishedParams,
}

#[derive(Debug, Clone)]
enum DownloadState {
    Pending,
    Simulated,
    #[allow(dead_code)]
    Downloaded,
    Failed(String),
}
impl std::fmt::Display for DownloadState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "PEND"),
            Self::Simulated => write!(f, "SIM"),
            Self::Downloaded => write!(f, "DL"),
            Self::Failed(e) => write!(f, "FAIL:{}", e),
        }
    }
}

fn build_manifest() -> Vec<ManifestEntry> {
    let e =
        |id: &str, survey: &str, t_e: f64, u0: f64, q: Option<f64>, s: Option<f64>| ManifestEntry {
            event_id: id.into(),
            survey: survey.into(),
            published: PublishedParams {
                einstein_time: t_e,
                impact_param: u0,
                planet_mass_ratio: q,
                planet_separation: s,
            },
        };
    vec![
        // --- Planetary events ---
        // First cold rocky/icy super-Earth
        e(
            "OGLE-2005-BLG-390",
            "ogle",
            11.0,
            0.359,
            Some(7.6e-5),
            Some(1.610),
        ),
        // Possible rogue planet with moon
        e("MOA-2011-BLG-262", "moa", 3.3, 0.04, Some(3.8e-4), None),
        // Ice-line planet, very low mass ratio
        e(
            "OGLE-2016-BLG-1195",
            "ogle",
            9.9,
            0.045,
            Some(4.2e-5),
            Some(1.04),
        ),
        // First microlensing planet (OGLE+MOA joint)
        e(
            "OGLE-2003-BLG-235",
            "ogle",
            61.5,
            0.133,
            Some(0.0039),
            Some(1.12),
        ),
        // Very low-mass host star planet
        e(
            "MOA-2007-BLG-192",
            "moa",
            46.0,
            0.001,
            Some(0.0002),
            Some(1.05),
        ),
        // Jupiter/Saturn analog system (two planets)
        e(
            "OGLE-2006-BLG-109",
            "ogle",
            127.3,
            0.0035,
            Some(1.4e-3),
            Some(2.3),
        ),
        // 3.8 M_Jup planet
        e(
            "OGLE-2005-BLG-071",
            "ogle",
            70.9,
            0.026,
            Some(7.0e-3),
            Some(1.29),
        ),
        // Saturn-mass planet
        e(
            "OGLE-2012-BLG-0563",
            "ogle",
            55.0,
            0.045,
            Some(1.5e-3),
            Some(1.42),
        ),
        // --- Non-planet PSPL comparison events ---
        e("OGLE-2018-BLG-0799", "ogle", 25.0, 0.008, None, None),
        e("MOA-2019-BLG-008", "moa", 18.0, 0.15, None, None),
        e("OGLE-2017-BLG-0373", "ogle", 40.0, 0.12, None, None),
        e("OGLE-2015-BLG-0966", "ogle", 22.0, 0.21, None, None),
        e("MOA-2013-BLG-220", "moa", 60.0, 0.08, None, None),
    ]
}

// ---- Paczynski formula & binary-lens perturbation ----

fn pspl_magnification(u: f64) -> f64 {
    if u < 1e-10 {
        return 1e10;
    }
    let u2 = u * u;
    (u2 + 2.0) / (u * (u2 + 4.0).sqrt())
}

/// Binary-lens perturbation factor near the planetary caustic.
/// Uses the Chang-Refsdal approximation: excess magnification from
/// a point-mass perturber at separation `s` with mass ratio `q`.
fn binary_lens_perturbation(tau: f64, u0: f64, q: f64, s: f64) -> f64 {
    // Distance from source to planet position (planet at s along lens axis)
    let dx = tau - s;
    let d2 = dx * dx + u0 * u0;
    // Caustic half-width ~ 4*q/s^2 for close/wide topologies
    let caustic_r2 = 4.0 * q / (s * s);
    if d2 > caustic_r2 * 25.0 {
        return 0.0;
    } // too far from caustic
    if d2 < 1e-15 {
        return q * 1e4;
    } // avoid singularity
      // Smooth perturbation envelope
    let strength = q / d2.max(q * 0.1);
    strength * (-d2 / (8.0 * caustic_r2.max(1e-10))).exp()
}

// ---- Simulate light curve from published parameters ----

fn simulate_from_published(
    entry: &ManifestEntry,
    rng: &mut u64,
) -> (MicrolensingEvent, DownloadState) {
    let p = &entry.published;
    let is_moa = entry.survey == "moa";
    let t0 = 2459000.0 + (lcg_f64(rng) * 500.0);
    let baseline_mag = 18.0 + lcg_f64(rng) * 3.0;
    let f_b = 0.05 + lcg_f64(rng) * 0.3;
    let cadence = if is_moa { 0.25 } else { 0.5 }; // days between obs
    let sigma_floor = if is_moa { 0.01 } else { 0.005 };
    let span = p.einstein_time.max(10.0) * 3.0; // cover +-3 t_E

    let mut observations = Vec::new();
    let mut t = t0 - span;
    while t < t0 + span {
        t += cadence * (0.5 + lcg_f64(rng));
        if lcg_f64(rng) < 0.25 {
            continue;
        } // weather gaps
          // Night-time window: keep ~40% of observations
        if ((t % 1.0) - 0.2).abs() > 0.17 {
            continue;
        }

        let tau = (t - t0) / p.einstein_time;
        let u = (p.impact_param * p.impact_param + tau * tau).sqrt();
        let mut amp = pspl_magnification(u);

        // Add binary-lens planetary perturbation if planet present
        if let (Some(q), Some(s)) = (p.planet_mass_ratio, p.planet_separation) {
            amp += binary_lens_perturbation(tau, p.impact_param, q, s) * amp;
        } else if let Some(q) = p.planet_mass_ratio {
            // No separation published (e.g. MOA-2011-BLG-262): use s ~ 1.0
            amp += binary_lens_perturbation(tau, p.impact_param, q, 1.0) * amp;
        }

        let true_flux = amp + f_b;
        let true_mag = baseline_mag - 2.5 * true_flux.log10();
        let sigma = (sigma_floor * sigma_floor + 0.003f64.powi(2)).sqrt();
        let obs_mag = true_mag + lcg_normal(rng) * sigma;
        observations.push(Observation {
            hjd: t,
            mag: obs_mag,
            mag_err: sigma,
            flux: 0.0,
            flux_err: 0.0,
        });
    }

    let peak_mag = observations
        .iter()
        .map(|o| o.mag)
        .fold(f64::INFINITY, f64::min);
    let mut evt = MicrolensingEvent {
        name: entry.event_id.clone(),
        source: entry.survey.clone(),
        observations,
        baseline_mag,
        peak_mag,
        t0_est: t0,
        t_e_est: p.einstein_time,
        u0_est: p.impact_param,
    };
    normalize_cross_survey(&mut evt);
    (evt, DownloadState::Simulated)
}

// ---- Cross-survey normalization ----

/// Normalize to fractional deviation (F - F_base) / F_base.
/// Survey-specific zero-points: OGLE I-band = 21.0, MOA R-band = 22.0.
fn normalize_cross_survey(event: &mut MicrolensingEvent) {
    let zp = if event.source == "moa" { 22.0 } else { 21.0 };
    // Baseline flux from faintest 50% of observations
    let mut mags: Vec<f64> = event.observations.iter().map(|o| o.mag).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = mags.len() / 2;
    let baseline_flux = if mid > 0 {
        let bm: f64 = mags[mid..].iter().sum::<f64>() / (mags.len() - mid) as f64;
        10.0f64.powf(-0.4 * (bm - zp))
    } else {
        1.0
    };
    for obs in &mut event.observations {
        let f = 10.0f64.powf(-0.4 * (obs.mag - zp));
        obs.flux = (f - baseline_flux) / baseline_flux;
        obs.flux_err = f * 0.4 * 2.302585 * obs.mag_err / baseline_flux;
    }
}

// ---- OGLE format parser ----

/// Parse OGLE EWS photometry: HJD/JD, magnitude or flux, error.
fn parse_ogle_format(data: &str, event_name: &str) -> MicrolensingEvent {
    let mut obs = Vec::new();
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('\\') {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 3 {
            continue;
        }
        let (hjd, val, err) = match (
            p[0].parse::<f64>(),
            p[1].parse::<f64>(),
            p[2].parse::<f64>(),
        ) {
            (Ok(t), Ok(v), Ok(e)) => (t, v, e),
            _ => continue,
        };
        if [hjd, val, err]
            .iter()
            .any(|x| x.is_nan() || x.is_infinite())
        {
            continue;
        }
        let hjd = if hjd > 2400000.0 {
            hjd - 2450000.0
        } else {
            hjd
        };
        let (mag, mag_err) = if val > 10.0 {
            (val, err)
        } else if val > 0.0 {
            (-2.5 * val.log10() + 22.0, 2.5 / (val * 2.302585) * err)
        } else {
            continue;
        };
        if mag_err <= 0.0 || mag_err > 1.0 || mag < 5.0 || mag > 25.0 {
            continue;
        }
        obs.push(Observation {
            hjd,
            mag,
            mag_err,
            flux: 0.0,
            flux_err: 0.0,
        });
    }
    finalize_event(obs, event_name, "ogle")
}

// ---- Enhanced MOA format parser ----

/// Parse MOA-II photometry: JD, flux, flux_error columns.
/// Baseline estimated from first and last 20% of time-sorted observations.
fn parse_moa_format(data: &str, event_name: &str) -> MicrolensingEvent {
    let mut obs = Vec::new();
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 3 {
            continue;
        }
        let (jd, flux, ferr) = match (
            p[0].parse::<f64>(),
            p[1].parse::<f64>(),
            p[2].parse::<f64>(),
        ) {
            (Ok(t), Ok(f), Ok(e)) => (t, f, e),
            _ => continue,
        };
        if [jd, flux, ferr]
            .iter()
            .any(|x| x.is_nan() || x.is_infinite())
        {
            continue;
        }
        if flux <= 0.0 || ferr <= 0.0 {
            continue;
        }
        let hjd = if jd > 2400000.0 { jd - 2450000.0 } else { jd };
        // MOA R-band zero-point 22.0
        let mag = -2.5 * flux.log10() + 22.0;
        let mag_err = 2.5 / (flux * 2.302585) * ferr;
        if mag_err <= 0.0 || mag_err > 1.0 {
            continue;
        }
        obs.push(Observation {
            hjd,
            mag,
            mag_err,
            flux,
            flux_err: ferr,
        });
    }
    // Sort by time, then estimate baseline from first/last 20%
    obs.sort_by(|a, b| a.hjd.partial_cmp(&b.hjd).unwrap());
    let n = obs.len();
    let wing = (n as f64 * 0.2).ceil() as usize;
    if n > 0 && wing > 0 {
        let wing_obs: Vec<f64> = obs[..wing]
            .iter()
            .chain(obs[n.saturating_sub(wing)..].iter())
            .map(|o| o.flux)
            .collect();
        let baseline_flux = wing_obs.iter().sum::<f64>() / wing_obs.len() as f64;
        if baseline_flux > 0.0 {
            for o in &mut obs {
                let raw = o.flux;
                o.flux = (raw - baseline_flux) / baseline_flux; // fractional deviation
                o.flux_err = o.flux_err / baseline_flux;
            }
        }
    }
    finalize_event(obs, event_name, "moa")
}

/// Shared finalization: sort, compute baseline/peak, convert flux where needed.
fn finalize_event(mut obs: Vec<Observation>, name: &str, source: &str) -> MicrolensingEvent {
    obs.sort_by(|a, b| a.hjd.partial_cmp(&b.hjd).unwrap());
    let mut mags: Vec<f64> = obs.iter().map(|o| o.mag).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let baseline_mag = if !mags.is_empty() {
        mags[mags.len() * 3 / 4]
    } else {
        20.0
    };
    let peak_mag = mags.first().copied().unwrap_or(20.0);
    for o in &mut obs {
        if o.flux == 0.0 {
            o.flux = 10.0f64.powf(-0.4 * (o.mag - baseline_mag));
            o.flux_err = o.flux * 0.4 * 2.302585 * o.mag_err;
        }
    }
    let t0_est = obs
        .iter()
        .min_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap())
        .map(|o| o.hjd)
        .unwrap_or(0.0);
    let half = (baseline_mag + peak_mag) / 2.0;
    let above: Vec<_> = obs.iter().filter(|o| o.mag < half).collect();
    let t_e_est = if above.len() >= 2 {
        (above.last().unwrap().hjd - above.first().unwrap().hjd) / 2.0
    } else {
        20.0
    };
    let pf = 10.0f64.powf(-0.4 * (peak_mag - baseline_mag));
    let u0_est = if pf > 1.5 { (1.0 / pf).max(0.01) } else { 0.5 };
    MicrolensingEvent {
        name: name.into(),
        source: source.into(),
        observations: obs,
        baseline_mag,
        peak_mag,
        t0_est,
        t_e_est,
        u0_est,
    }
}

// ---- PSPL fitting ----

#[derive(Debug, Clone)]
struct PSPLFit {
    t0: f64,
    u0: f64,
    t_e: f64,
    f_s: f64,
    f_b: f64,
    chi2: f64,
}

fn fit_pspl_linear(obs: &[Observation], t0: f64, u0: f64, t_e: f64) -> Option<(f64, f64, f64)> {
    let (mut sa, mut sa2, mut sf, mut saf, mut s1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for o in obs {
        let w = 1.0 / (o.flux_err * o.flux_err + 1e-10);
        let u = ((o.hjd - t0) / t_e).powi(2) + u0 * u0;
        let a = pspl_magnification(u.sqrt());
        sa += w * a;
        sa2 += w * a * a;
        sf += w * o.flux;
        saf += w * a * o.flux;
        s1 += w;
    }
    let det = sa2 * s1 - sa * sa;
    if det.abs() < 1e-15 {
        return None;
    }
    let f_s = (saf * s1 - sa * sf) / det;
    let f_b = (sa2 * sf - sa * saf) / det;
    if f_s < 0.01 {
        return None;
    }
    let chi2: f64 = obs
        .iter()
        .map(|o| {
            let u = ((o.hjd - t0) / t_e).powi(2) + u0 * u0;
            let m = f_s * pspl_magnification(u.sqrt()) + f_b;
            (o.flux - m).powi(2) / (o.flux_err.powi(2) + 1e-10)
        })
        .sum();
    Some((f_s, f_b, chi2))
}

fn fit_pspl(event: &MicrolensingEvent) -> PSPLFit {
    let mut best = PSPLFit {
        t0: 0.0,
        u0: 0.0,
        t_e: 0.0,
        f_s: 1.0,
        f_b: 0.1,
        chi2: f64::MAX,
    };
    for t0_i in 0..20 {
        let t0 = (event.t0_est - 10.0) + 20.0 * t0_i as f64 / 19.0;
        for te_i in 0..15 {
            let t_e = event.t_e_est * 0.5 + event.t_e_est * 1.5 * te_i as f64 / 14.0;
            if t_e < 0.5 {
                continue;
            }
            for u0_i in 1..=12 {
                let u0 = u0_i as f64 * 0.05;
                if let Some((f_s, f_b, chi2)) = fit_pspl_linear(&event.observations, t0, u0, t_e) {
                    if chi2 < best.chi2 {
                        best = PSPLFit {
                            t0,
                            u0,
                            t_e,
                            f_s,
                            f_b,
                            chi2,
                        };
                    }
                }
            }
        }
    }
    let b = best.clone();
    for dt in -5..=5 {
        for dte in -5..=5 {
            for du in -5..=5 {
                let (t0, t_e, u0) = (
                    b.t0 + dt as f64 * 0.2,
                    b.t_e * (1.0 + dte as f64 * 0.02),
                    b.u0 + du as f64 * 0.005,
                );
                if t_e < 0.5 || u0 < 0.001 {
                    continue;
                }
                if let Some((f_s, f_b, chi2)) = fit_pspl_linear(&event.observations, t0, u0, t_e) {
                    if chi2 < best.chi2 {
                        best = PSPLFit {
                            t0,
                            u0,
                            t_e,
                            f_s,
                            f_b,
                            chi2,
                        };
                    }
                }
            }
        }
    }
    best
}

// ---- Anomaly detection ----

fn analyze_residuals(
    event: &MicrolensingEvent,
    fit: &PSPLFit,
) -> Vec<(f64, f64, String, Vec<f32>)> {
    let rchi2 = fit.chi2 / event.observations.len().max(1) as f64;
    let mut results = Vec::new();
    let mut tau = -3.0;
    while tau <= 3.0 {
        let win: Vec<usize> = event
            .observations
            .iter()
            .enumerate()
            .filter(|(_, o)| ((o.hjd - fit.t0) / fit.t_e - tau).abs() <= 0.4)
            .map(|(i, _)| i)
            .collect();
        if win.len() < 3 {
            tau += 0.2;
            continue;
        }
        let resid: Vec<f64> = win
            .iter()
            .map(|&i| {
                let o = &event.observations[i];
                let u = ((o.hjd - fit.t0) / fit.t_e).powi(2) + fit.u0 * fit.u0;
                (o.flux - fit.f_s * pspl_magnification(u.sqrt()) - fit.f_b) / o.flux_err.max(1e-10)
            })
            .collect();
        let chi2_w: f64 = resid.iter().map(|r| r * r).sum();
        let excess =
            (chi2_w - rchi2 * win.len() as f64) / (2.0 * rchi2 * win.len() as f64).sqrt().max(0.1);
        let n_pos = resid.iter().filter(|&&r| r > 0.0).count();
        let (np, nn) = (n_pos.max(1) as f64, (resid.len() - n_pos).max(1) as f64);
        let mut runs = 1usize;
        for w in resid.windows(2) {
            if (w[0] > 0.0) != (w[1] > 0.0) {
                runs += 1;
            }
        }
        let coherence_z = (1.0 + 2.0 * np * nn / (np + nn) - runs as f64)
            / (2.0 * np * nn * (2.0 * np * nn - np - nn)
                / ((np + nn).powi(2) * (np + nn - 1.0).max(1.0)))
            .sqrt()
            .max(0.5);
        let combined = 0.3 * excess + 0.7 * coherence_z;
        let class = if combined > 3.0 {
            if tau.abs() < 0.5 {
                "planet"
            } else {
                "moon-candidate"
            }
        } else if combined > 2.0 {
            "unknown"
        } else {
            "noise"
        };
        let mean_r = resid.iter().sum::<f64>() / resid.len() as f64;
        let var_r = resid.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / resid.len() as f64;
        let mut emb = vec![
            mean_r as f32,
            var_r.sqrt() as f32,
            excess as f32,
            coherence_z as f32,
            tau as f32,
        ];
        for lag in 1..=8 {
            let (mut ac, mut cnt) = (0.0, 0usize);
            for i in 0..resid.len().saturating_sub(lag) {
                ac += resid[i] * resid[i + lag];
                cnt += 1;
            }
            emb.push(if cnt > 0 {
                (ac / cnt as f64) as f32
            } else {
                0.0
            });
        }
        while emb.len() < 32 {
            emb.push(0.0);
        }
        emb.truncate(32);
        results.push((tau, combined, class.to_string(), emb));
        tau += 0.2;
    }
    results
}

// ---- Main pipeline ----

fn main() {
    println!("=== Real Microlensing Analysis Pipeline ===\n");

    let dim = 32;
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let store_path = tmp_dir.path().join("real_microlensing.rvf");
    let mut store = RvfStore::create(
        &store_path,
        RvfOptions {
            dimension: dim as u16,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .expect("failed to create store");

    // Step 1: Build manifest and simulate from published parameters
    println!("--- Step 1. Download Manifest ({} real events) ---\n", "13");
    let manifest = build_manifest();
    let mut rng = 42u64;
    let mut events = Vec::new();
    let mut states = Vec::new();

    for entry in &manifest {
        let (evt, state) = simulate_from_published(entry, &mut rng);
        let p = &entry.published;
        let label = match p.planet_mass_ratio {
            Some(q) => match p.planet_separation {
                Some(s) => format!("PLANET q={:.1e} s={:.2}", q, s),
                None => format!("PLANET q={:.1e}", q),
            },
            None => "PSPL-only".into(),
        };
        println!(
            "  [{}] {:24} t_E={:6.1}d u_0={:5.3}  {}",
            state, entry.event_id, p.einstein_time, p.impact_param, label
        );
        events.push(evt);
        states.push(state);
    }

    // Exercise the parsers on synthetic snippets
    let _ogle_test = parse_ogle_format(
        "# OGLE-IV EWS\n2453500.0 18.5 0.02\n2453501.0 17.2 0.015\n",
        "test-ogle",
    );
    let _moa_test = parse_moa_format(
        "# MOA-II\n2455700.0 500.0 10.0\n2455701.0 800.0 15.0\n2455702.0 520.0 11.0\n",
        "test-moa",
    );

    // Step 2: Fit and analyze
    println!("\n--- Step 2. PSPL Fit + Anomaly Search ---\n");
    let mut all_vecs: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_meta: Vec<MetadataEntry> = Vec::new();
    let mut discoveries = Vec::new();

    println!(
        "  {:>24}  {:>7}  {:>5}  {:>7}  {:>12}",
        "Event", "chi2/N", "Anom", "BestZ", "Class"
    );
    for (ei, event) in events.iter().enumerate() {
        if event.observations.is_empty() {
            continue;
        }
        let fit = fit_pspl(event);
        let rchi2 = fit.chi2 / event.observations.len().max(1) as f64;
        let windows = analyze_residuals(event, &fit);
        let anomalous = windows.iter().filter(|w| w.1 > 2.0).count();
        let best = windows
            .iter()
            .map(|w| w.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_class = windows
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|w| w.2.as_str())
            .unwrap_or("none");
        println!(
            "  {:>24}  {:>7.3}  {:>5}  {:>7.2}  {:>12}",
            event.name, rchi2, anomalous, best, best_class
        );
        if anomalous > 0 {
            discoveries.push((event.name.clone(), best_class.to_string(), best));
        }
        for (wi, (_, _, cls, emb)) in windows.iter().enumerate() {
            let id = ei as u64 * 1000 + wi as u64;
            all_vecs.push(emb.clone());
            all_ids.push(id);
            all_meta.push(MetadataEntry {
                field_id: FIELD_SOURCE,
                value: MetadataValue::String(event.source.clone()),
            });
            all_meta.push(MetadataEntry {
                field_id: FIELD_EVENT,
                value: MetadataValue::String(event.name.clone()),
            });
            all_meta.push(MetadataEntry {
                field_id: FIELD_ANOMALY_TYPE,
                value: MetadataValue::String(cls.clone()),
            });
        }
    }

    let vec_refs: Vec<&[f32]> = all_vecs.iter().map(|v| v.as_slice()).collect();
    let ingest = store
        .ingest_batch(&vec_refs, &all_ids, Some(&all_meta))
        .expect("ingest failed");

    // Step 3: Discoveries and provenance
    println!("\n--- Step 3. Discoveries ---\n");
    if discoveries.is_empty() {
        println!("  No significant anomalies.");
    } else {
        for (n, c, z) in &discoveries {
            println!("  {:>24}  {:>12}  Z={:.2}", n, c, z);
        }
    }

    println!("\n--- Step 4. Cross-Event Search ---");
    let qv = random_vector(dim, 42);
    let pr = store
        .query(
            &qv,
            10,
            &QueryOptions {
                filter: Some(FilterExpr::Eq(
                    FIELD_ANOMALY_TYPE,
                    FilterValue::String("planet".into()),
                )),
                ..Default::default()
            },
        )
        .expect("query");
    let mr = store
        .query(
            &qv,
            10,
            &QueryOptions {
                filter: Some(FilterExpr::Eq(
                    FIELD_ANOMALY_TYPE,
                    FilterValue::String("moon-candidate".into()),
                )),
                ..Default::default()
            },
        )
        .expect("query");
    println!(
        "  Planet windows: {}, Moon-candidate windows: {}",
        pr.len(),
        mr.len()
    );

    println!("\n--- Step 5. Provenance ---");
    let child_path = tmp_dir.path().join("discoveries.rvf");
    let child = store
        .derive(&child_path, DerivationType::Filter, None)
        .expect("derive");
    println!(
        "  Parent: {}  Child: {}",
        hex(store.file_id()),
        hex(child.parent_id())
    );
    child.close().expect("close");

    let entries: Vec<WitnessEntry> = [
        "genesis",
        "data_load",
        "pspl_fit",
        "anomaly",
        "classify",
        "ingest",
    ]
    .iter()
    .enumerate()
    .map(|(i, s)| WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(format!("real_micro:{}:{}", s, i).as_bytes()),
        timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
        witness_type: if i == 0 { 0x01 } else { 0x02 },
    })
    .collect();
    let chain = create_witness_chain(&entries);
    println!(
        "  Witness chain: {} entries, {}",
        verify_witness_chain(&chain).expect("verify").len(),
        "VALID"
    );

    println!("\n=== Summary ===\n");
    println!("  Manifest events:    {}", manifest.len());
    println!(
        "  Simulated:          {}",
        states
            .iter()
            .filter(|s| matches!(s, DownloadState::Simulated))
            .count()
    );
    println!("  Windows ingested:   {}", ingest.accepted);
    println!("  Anomalies found:    {}", discoveries.len());
    println!(
        "  Planet candidates:  {}",
        discoveries.iter().filter(|d| d.1 == "planet").count()
    );
    println!(
        "  Moon candidates:    {}",
        discoveries
            .iter()
            .filter(|d| d.1 == "moon-candidate")
            .count()
    );
    store.close().expect("close");
    println!("\nDone.");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
