//! Gallery of Exotic Signals: what boundary-first detection finds that
//! amplitude-based SETI detectors miss.
//!
//! Six signal types are injected into a 128-channel x 100-timestep spectrogram
//! at amplitudes below the per-pixel detection threshold. Traditional
//! amplitude thresholding (flag pixels > N sigma) misses them. Boundary-
//! first detection builds a temporal coherence graph and finds structural
//! anomalies via min-cut analysis.
//!
//! Key insight: signals that are invisible per-pixel can create detectable
//! *correlations between channels*. The coherence graph captures this by
//! measuring how the inter-channel covariance matrix changes over time.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const CHANNELS: usize = 128;
const TIMESTEPS: usize = 100;
const SEED: u64 = 2025;
const NULL_PERMS: usize = 100;
const WIN_T: usize = 20;
const WIN_STEP: usize = 5;

fn n_wins() -> usize {
    (TIMESTEPS - WIN_T) / WIN_STEP + 1
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn noise_spec(rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..CHANNELS)
        .map(|_| (0..TIMESTEPS).map(|_| gauss(rng)).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Amplitude detector (traditional SETI approach)
// ---------------------------------------------------------------------------

/// Standard SETI: threshold individual pixels. Returns (n_above_3sigma,
/// n_above_2sigma, hit). HIT = significantly more exceedances than noise.
fn amplitude_detect(spec: &[Vec<f64>]) -> (usize, usize, bool) {
    let total = CHANNELS * TIMESTEPS;
    let n3 = spec
        .iter()
        .flat_map(|r| r.iter())
        .filter(|&&v| v.abs() > 3.0)
        .count();
    let n2 = spec
        .iter()
        .flat_map(|r| r.iter())
        .filter(|&&v| v.abs() > 2.0)
        .count();
    // Noise expectations (two-tailed)
    let exp_3 = (total as f64 * 0.0027) as usize; // ~35
    let exp_2 = (total as f64 * 0.0455) as usize; // ~582
                                                  // Very generous detection: 2x expected 3-sigma OR 30% excess 2-sigma
    let hit = n3 > exp_3 * 2 || n2 > (exp_2 as f64 * 1.3) as usize;
    (n3, n2, hit)
}

// ---------------------------------------------------------------------------
// Coherence features per time-window
// ---------------------------------------------------------------------------

/// Channel groups for covariance measurement: 32 groups of 4 channels.
/// Finer groups increase sensitivity to localized signals.
fn channel_groups() -> Vec<Vec<usize>> {
    (0..32).map(|g| (g * 4..(g + 1) * 4).collect()).collect()
}

/// Per-window feature: covariance matrix of group means.
/// Returns the upper triangle of the 16x16 covariance matrix.
fn window_cov_features(spec: &[Vec<f64>], t0: usize, groups: &[Vec<usize>]) -> Vec<f64> {
    let ng = groups.len();
    let n = WIN_T as f64;

    // Group means over time
    let gm: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            (0..WIN_T)
                .map(|dt| g.iter().map(|&ch| spec[ch][t0 + dt]).sum::<f64>() / g.len() as f64)
                .collect()
        })
        .collect();

    // Upper triangle of covariance matrix
    let mut feats = Vec::with_capacity(ng * (ng - 1) / 2);
    for i in 0..ng {
        let mi: f64 = gm[i].iter().sum::<f64>() / n;
        for j in (i + 1)..ng {
            let mj: f64 = gm[j].iter().sum::<f64>() / n;
            let cov: f64 = (0..WIN_T)
                .map(|k| (gm[i][k] - mi) * (gm[j][k] - mj))
                .sum::<f64>()
                / n;
            feats.push(cov);
        }
    }
    feats
}

/// L2 distance between feature vectors.
fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Coherence graph
// ---------------------------------------------------------------------------

fn coherence_graph(
    spec: &[Vec<f64>],
    groups: &[Vec<usize>],
) -> (Vec<(usize, usize, f64)>, Vec<(u64, u64, f64)>) {
    let nw = n_wins();
    let feats: Vec<Vec<f64>> = (0..nw)
        .map(|w| window_cov_features(spec, w * WIN_STEP, groups))
        .collect();

    let mut sp = Vec::new();
    let mut mc = Vec::new();
    for i in 0..nw {
        for j in (i + 1)..nw.min(i + 5) {
            let d = l2_dist(&feats[i], &feats[j]);
            let w = 1.0 / (1.0 + d * 5.0);
            sp.push((i, j, w));
            mc.push((i as u64, j as u64, w));
        }
    }
    (sp, mc)
}

fn cut_sweep(n: usize, edges: &[(usize, usize, f64)]) -> (usize, f64) {
    let mut cuts = vec![0.0_f64; n];
    for &(u, v, w) in edges {
        let (lo, hi) = (u.min(v), u.max(v));
        for k in (lo + 1)..=hi {
            cuts[k] += w;
        }
    }
    let m = 1;
    let mut best = (m, f64::INFINITY);
    for k in m..n.saturating_sub(m) {
        if cuts[k] < best.1 {
            best = (k, cuts[k]);
        }
    }
    best
}

fn fiedler_val(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
    if edges.is_empty() || n < 2 {
        return 0.0;
    }
    let lap = CsrMatrixView::build_laplacian(n, edges);
    estimate_fiedler(&lap, 200, 1e-10).0
}

fn global_mincut(mc: Vec<(u64, u64, f64)>) -> f64 {
    if mc.is_empty() {
        return 0.0;
    }
    MinCutBuilder::new()
        .exact()
        .with_edges(mc)
        .build()
        .map(|m| m.min_cut_value())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Null distributions
// ---------------------------------------------------------------------------

fn null_dists(rng: &mut StdRng, groups: &[Vec<usize>]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nw = n_wins();
    let (mut ss, mut gs, mut fs) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..NULL_PERMS {
        let spec = noise_spec(rng);
        let (sp, mc) = coherence_graph(&spec, groups);
        ss.push(cut_sweep(nw, &sp).1);
        fs.push(fiedler_val(nw, &sp));
        gs.push(global_mincut(mc));
    }
    (ss, gs, fs)
}

fn z(obs: f64, null: &[f64]) -> f64 {
    let n = null.len() as f64;
    let mu: f64 = null.iter().sum::<f64>() / n;
    let sd: f64 = (null.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n).sqrt();
    if sd < 1e-12 {
        0.0
    } else {
        (obs - mu) / sd
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

// ---------------------------------------------------------------------------
// Signal injectors -- each creates CORRELATED structure across many
// channels, not just per-pixel amplitude
// ---------------------------------------------------------------------------

/// Signal 1: "The Whisper" -- drifting narrowband affecting neighboring
/// channels coherently. The drift creates correlation between adjacent
/// channels in the time windows where it passes.
fn inject_whisper(spec: &mut [Vec<f64>]) {
    // Broadband drifting chirp: a correlated waveform spanning many channels
    // present only during timesteps 15-65. Each channel gets the same
    // temporal waveform phase-shifted by channel index (creating correlation).
    let amp = 0.6;
    for t in 15..65 {
        let phase_base = 2.0 * std::f64::consts::PI * (t - 15) as f64 / 20.0;
        for ch in 32..96 {
            // 64 channels
            let phase = phase_base + 0.05 * ch as f64;
            spec[ch][t] += amp * phase.sin();
        }
    }
}

/// Signal 2: "The Handshake" -- two widely separated frequency bands
/// pulse simultaneously with identical waveform. Creates cross-frequency
/// correlation that is impossible from noise.
fn inject_handshake(spec: &mut [Vec<f64>]) {
    let amp = 0.8;
    for t in 0..TIMESTEPS {
        if t % 20 < 5 {
            let env = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (t % 20) as f64 / 5.0).cos());
            for ch in 24..40 {
                spec[ch][t] += amp * env;
            }
            for ch in 88..104 {
                spec[ch][t] += amp * env;
            }
        }
    }
}

/// Signal 3: "The Shadow" -- absorption across a wide band during a
/// specific time interval. Reduces variance uniformly, creating a
/// correlated deficit region.
fn inject_shadow(spec: &mut [Vec<f64>]) {
    for ch in 32..96 {
        // 64 channels
        for t in 35..65 {
            spec[ch][t] *= 0.5;
        }
    }
}

/// Signal 4: "The Watermark" -- harmonic structure. Three related
/// frequency bands oscillate with the same phase, creating cross-band
/// correlation.
fn inject_watermark(spec: &mut [Vec<f64>]) {
    for t in 0..TIMESTEPS {
        for h in 1..=3u32 {
            let v = 0.7 * (2.0 * std::f64::consts::PI * h as f64 * t as f64 / 50.0).sin();
            let center = 16 * h as usize;
            for ch in center.saturating_sub(4)..=(center + 4).min(CHANNELS - 1) {
                spec[ch][t] += v;
            }
        }
    }
}

/// Signal 5: "The Phase Shift" -- a slowly rotating sinusoid added
/// identically to a band of channels. The phase coherence creates
/// inter-channel correlation without boosting per-channel power much.
fn inject_phase_shift(spec: &mut [Vec<f64>]) {
    for t in 0..TIMESTEPS {
        let v = 0.7 * (2.0 * std::f64::consts::PI * t as f64 / 25.0).sin();
        for ch in 40..80 {
            // 40 channels all get the same signal
            spec[ch][t] += v;
        }
    }
}

/// Signal 6: "The Conversation" -- two independent sources in different
/// spectral regions and time intervals. Each creates correlation within
/// its band during its active window.
fn inject_conversation(spec: &mut [Vec<f64>]) {
    let amp = 0.7;
    for t in 10..35 {
        let env = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (t - 10) as f64 / 25.0).cos());
        for ch in 16..48 {
            spec[ch][t] += amp * env;
        }
    }
    for t in 55..80 {
        let env = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (t - 55) as f64 / 25.0).cos());
        for ch in 80..112 {
            spec[ch][t] += amp * env;
        }
    }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

struct Res {
    name: &'static str,
    desc: &'static str,
    n3: usize,
    n2: usize,
    amp_hit: bool,
    zs: f64,
    zg: f64,
    zf: f64,
    bnd_hit: bool,
}

fn analyze(
    name: &'static str,
    desc: &'static str,
    rng: &mut StdRng,
    inject: fn(&mut [Vec<f64>]),
    groups: &[Vec<usize>],
    ns: &[f64],
    ng: &[f64],
    nf: &[f64],
) -> Res {
    let mut spec = noise_spec(rng);
    inject(&mut spec);
    let (n3, n2, amp_hit) = amplitude_detect(&spec);
    let nw = n_wins();
    let (sp, mc) = coherence_graph(&spec, groups);
    let sv = cut_sweep(nw, &sp).1;
    let fv = fiedler_val(nw, &sp);
    let gv = global_mincut(mc);
    let (zs, zg, zf) = (z(sv, ns), z(gv, ng), z(fv, nf));
    let bnd_hit = zs < -2.0 || zg < -2.0 || zf.abs() > 2.0;
    Res {
        name,
        desc,
        n3,
        n2,
        amp_hit,
        zs,
        zg,
        zf,
        bnd_hit,
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let groups = channel_groups();

    println!("================================================================");
    println!("  Gallery of Invisible Signals");
    println!("  What SETI Has Been Missing");
    println!("================================================================\n");
    println!("  Spectrogram:  {} ch x {} t", CHANNELS, TIMESTEPS);
    println!(
        "  Window:       {} t, stride {} ({} nodes)",
        WIN_T,
        WIN_STEP,
        n_wins()
    );
    println!(
        "  Features:     {} group-pair covariances per window",
        groups.len() * (groups.len() - 1) / 2
    );
    println!("  Null model:   {} pure-noise permutations\n", NULL_PERMS);

    println!("[NULL] Building null distributions...");
    let (ns, ng, nf) = null_dists(&mut rng, &groups);
    println!(
        "[NULL] sweep={:.4}  global={:.4}  fiedler={:.6}\n",
        mean(&ns),
        mean(&ng),
        mean(&nf)
    );

    let sigs: &[(&str, &str, fn(&mut [Vec<f64>]))] = &[
        (
            "The Whisper",
            "broadband chirp at 0.6sigma, t=15-65",
            inject_whisper,
        ),
        (
            "The Handshake",
            "correlated dual-band pulse at 0.8sigma",
            inject_handshake,
        ),
        (
            "The Shadow",
            "absorption dip to 0.5x, 64 ch, t=35-65",
            inject_shadow,
        ),
        (
            "The Watermark",
            "harmonic cross-band oscillation at 0.7sigma",
            inject_watermark,
        ),
        (
            "The Phase Shift",
            "coherent phase across 40 ch at 0.7sigma",
            inject_phase_shift,
        ),
        (
            "The Conversation",
            "two causal sources at 0.7sigma",
            inject_conversation,
        ),
    ];

    let res: Vec<Res> = sigs
        .iter()
        .map(|(n, d, f)| analyze(n, d, &mut rng, *f, &groups, &ns, &ng, &nf))
        .collect();

    println!("================================================================");
    println!("  RESULTS");
    println!("================================================================\n");

    let (mut at, mut bt) = (0usize, 0usize);
    for (i, r) in res.iter().enumerate() {
        let al = if r.amp_hit {
            at += 1;
            "HIT "
        } else {
            "MISS"
        };
        let bl = if r.bnd_hit {
            bt += 1;
            "HIT "
        } else {
            "MISS"
        };
        println!("Signal {}: \"{}\" ({})", i + 1, r.name, r.desc);
        println!(
            "  Amplitude detector: {}  ({} px>3s, {} px>2s)",
            al, r.n3, r.n2
        );
        println!(
            "  Boundary detector:  {}  (z_sweep={:.2}, z_global={:.2}, z_fiedler={:.2})",
            bl, r.zs, r.zg, r.zf
        );
        println!();
    }

    println!("================================================================");
    println!(
        "  SUMMARY: Traditional {}/{}  Boundary {}/{}",
        at,
        res.len(),
        bt,
        res.len()
    );
    println!("================================================================\n");

    if bt > at {
        println!(
            "  CONCLUSION: Boundary-first detection finds {} signal(s)",
            bt - at
        );
        println!("  that amplitude methods miss:");
        for r in &res {
            if r.bnd_hit && !r.amp_hit {
                let bz = if r.zs < -2.0 {
                    r.zs
                } else if r.zg < -2.0 {
                    r.zg
                } else {
                    -r.zf.abs()
                };
                println!("    - \"{}\" (z={:.2})", r.name, bz);
            }
        }
        println!("\n  Sub-threshold structure lives in the coherence graph,");
        println!("  not in pixel amplitudes.");
    } else {
        println!("  Both methods perform equally. The signals may need tuning");
        println!("  or a different coherence metric for this noise level.");
    }
    println!();
}
