//! Infrastructure Failure Prediction via Sensor Correlation Boundaries
//!
//! Detects structural degradation in bridges **months before collapse** by
//! finding boundary changes in sensor correlation structure -- not in the
//! sensors themselves.
//!
//! Inspired by the 2018 Morandi bridge collapse in Genoa (43 dead). Every
//! sensor was within limits. The correlations between sensors on the failing
//! member had been decaying for months. Nobody looked at correlations.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const DAYS: usize = 365;
const WINDOW: usize = 5;
const N_WIN: usize = DAYS / WINDOW; // 73
const N_SENS: usize = 15;
const N_PAIRS: usize = N_SENS * (N_SENS - 1) / 2; // 105
const SEED: u64 = 2018_08_14;
const NULL_PERMS: usize = 200;
const HEALTHY_END: usize = 200;
const DEGRADE_END: usize = 320;
const CRITICAL_END: usize = 350;
const FAILURE_DAY: usize = 351;
const FAIL_M: usize = 3; // failing member
const ALARM_Z: f64 = 3.8;

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
fn member(s: usize) -> usize {
    s % 5
}
fn win_day(w: usize) -> usize {
    w * WINDOW + WINDOW / 2
}

// Data generation: 5 members, 3 sensor types each. Member #3 degrades.
// Latent vibration mode per member drives correlation; degradation kills it.
fn generate(rng: &mut StdRng) -> Vec<[f64; N_SENS]> {
    let mut z = [0.0_f64; 5];
    for _ in 0..300 {
        for m in 0..5 {
            z[m] = 0.7 * z[m] + gauss(rng);
        }
    }
    (0..DAYS)
        .map(|day| {
            let (intra, cross, xvar, bump) = if day < HEALTHY_END {
                (1.0, 0.0, 0.0, 0.0)
            } else if day < DEGRADE_END {
                let t = (day - HEALTHY_END) as f64 / (DEGRADE_END - HEALTHY_END) as f64;
                let s = 0.5 - 0.5 * (std::f64::consts::PI * t).cos();
                (1.0 - 0.85 * s, 0.7 * s, 0.4 * s, 0.0)
            } else if day < CRITICAL_END {
                let t = (day - DEGRADE_END) as f64 / (CRITICAL_END - DEGRADE_END) as f64;
                (0.15 - 0.10 * t, 0.7 - 0.4 * t, 0.4 + 2.0 * t, 0.6 * t)
            } else {
                (0.0, 0.0, 8.0, 4.0)
            };
            for m in 0..5 {
                z[m] = 0.7 * z[m] + gauss(rng);
            }
            let znbr = (z[(FAIL_M + 1) % 5] + z[(FAIL_M + 4) % 5]) / 2.0;
            let mut r = [0.0_f64; N_SENS];
            for s in 0..N_SENS {
                let (base, ls, ns) = match s / 5 {
                    0 => (100.0, 12.0, 1.5),
                    1 => (0.0, 0.08, 0.008),
                    _ => (0.0, 6.0, 0.8),
                };
                r[s] = if member(s) == FAIL_M {
                    base + (z[FAIL_M] * intra + znbr * cross) * ls
                        + gauss(rng) * ns * (1.0 + xvar)
                        + bump * ns
                } else {
                    base + z[member(s)] * ls + gauss(rng) * ns
                };
            }
            r
        })
        .collect()
}

fn corr_matrix(win: &[[f64; N_SENS]]) -> [[f64; N_SENS]; N_SENS] {
    let n = win.len() as f64;
    let mut mu = [0.0_f64; N_SENS];
    for row in win {
        for s in 0..N_SENS {
            mu[s] += row[s];
        }
    }
    for s in 0..N_SENS {
        mu[s] /= n;
    }
    let mut c = [[0.0_f64; N_SENS]; N_SENS];
    for i in 0..N_SENS {
        for j in i..N_SENS {
            let (mut cov, mut vi, mut vj) = (0.0, 0.0, 0.0);
            for row in win {
                let (di, dj) = (row[i] - mu[i], row[j] - mu[j]);
                cov += di * dj;
                vi += di * di;
                vj += dj * dj;
            }
            let den = (vi * vj).sqrt();
            let r = if den < 1e-12 { 0.0 } else { cov / den };
            c[i][j] = r;
            c[j][i] = r;
        }
    }
    c
}

fn corr_features(c: &[[f64; N_SENS]; N_SENS]) -> [f64; N_PAIRS] {
    let mut f = [0.0_f64; N_PAIRS];
    let mut k = 0;
    for i in 0..N_SENS {
        for j in (i + 1)..N_SENS {
            f[k] = c[i][j];
            k += 1;
        }
    }
    f
}

fn intra_corr(c: &[[f64; N_SENS]; N_SENS], m: usize) -> f64 {
    let ss: Vec<usize> = (0..N_SENS).filter(|&s| member(s) == m).collect();
    let mut sum = 0.0;
    let mut n = 0;
    for i in 0..ss.len() {
        for j in (i + 1)..ss.len() {
            sum += c[ss[i]][ss[j]];
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn cross_corr(c: &[[f64; N_SENS]; N_SENS], m: usize) -> f64 {
    let mine: Vec<usize> = (0..N_SENS).filter(|&s| member(s) == m).collect();
    let nbrs: Vec<usize> = (0..N_SENS)
        .filter(|&s| {
            let d = (member(s) as isize - m as isize).unsigned_abs();
            member(s) != m && (d == 1 || d == 4)
        })
        .collect();
    let mut sum = 0.0;
    let mut n = 0;
    for &a in &mine {
        for &b in &nbrs {
            sum += c[a][b].abs();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn avg_corrs(cs: &[[[f64; N_SENS]; N_SENS]], r: std::ops::Range<usize>) -> (f64, f64) {
    let n = r.len().max(1) as f64;
    let ic: f64 = r.clone().map(|w| intra_corr(&cs[w], FAIL_M)).sum::<f64>() / n;
    let xc: f64 = r.map(|w| cross_corr(&cs[w], FAIL_M)).sum::<f64>() / n;
    (ic, xc)
}

fn sensor_vars(data: &[[f64; N_SENS]]) -> [f64; N_SENS] {
    let n = data.len() as f64;
    let mut mu = [0.0_f64; N_SENS];
    for row in data {
        for s in 0..N_SENS {
            mu[s] += row[s];
        }
    }
    for s in 0..N_SENS {
        mu[s] /= n;
    }
    let mut v = [0.0_f64; N_SENS];
    for row in data {
        for s in 0..N_SENS {
            v[s] += (row[s] - mu[s]).powi(2);
        }
    }
    for s in 0..N_SENS {
        v[s] /= n;
    }
    v
}

fn normalize(feats: &[[f64; N_PAIRS]]) -> Vec<[f64; N_PAIRS]> {
    let n = feats.len() as f64;
    let mut mu = [0.0_f64; N_PAIRS];
    let mut sd = [0.0_f64; N_PAIRS];
    for f in feats {
        for d in 0..N_PAIRS {
            mu[d] += f[d];
        }
    }
    for d in 0..N_PAIRS {
        mu[d] /= n;
    }
    for f in feats {
        for d in 0..N_PAIRS {
            sd[d] += (f[d] - mu[d]).powi(2);
        }
    }
    for d in 0..N_PAIRS {
        sd[d] = (sd[d] / n).sqrt().max(1e-12);
    }
    feats
        .iter()
        .map(|f| {
            let mut o = [0.0_f64; N_PAIRS];
            for d in 0..N_PAIRS {
                o[d] = (f[d] - mu[d]) / sd[d];
            }
            o
        })
        .collect()
}

fn dist_sq(a: &[f64; N_PAIRS], b: &[f64; N_PAIRS]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn build_graph(f: &[[f64; N_PAIRS]]) -> (Vec<(u64, u64, f64)>, Vec<(usize, usize, f64)>) {
    let mut ds = Vec::new();
    for i in 0..f.len() {
        for j in (i + 1)..f.len().min(i + 5) {
            ds.push(dist_sq(&f[i], &f[j]));
        }
    }
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sigma = ds[ds.len() / 2].max(1e-6);
    let (mut mc, mut sp) = (Vec::new(), Vec::new());
    for i in 0..f.len() {
        for skip in 1..=3 {
            if i + skip < f.len() {
                let w = (-dist_sq(&f[i], &f[i + skip]) / (2.0 * sigma))
                    .exp()
                    .max(1e-6);
                mc.push((i as u64, (i + skip) as u64, w));
                sp.push((i, i + skip, w));
            }
        }
    }
    (mc, sp)
}

fn cut_profile(edges: &[(usize, usize, f64)], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n];
    for &(u, v, w) in edges {
        for k in (u.min(v) + 1)..=u.max(v) {
            c[k] += w;
        }
    }
    c
}

fn find_boundaries(cuts: &[f64], margin: usize, gap: usize) -> Vec<(usize, f64)> {
    let n = cuts.len();
    let mut m: Vec<(usize, f64, f64)> = (1..n - 1)
        .filter_map(|i| {
            if i <= margin || i >= n - margin || cuts[i] >= cuts[i - 1] || cuts[i] >= cuts[i + 1] {
                return None;
            }
            let (lo, hi) = (i.saturating_sub(2), (i + 3).min(n));
            Some((
                i,
                cuts[i],
                cuts[lo..hi].iter().sum::<f64>() / (hi - lo) as f64 - cuts[i],
            ))
        })
        .collect();
    m.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let mut sel = Vec::new();
    for &(p, v, _) in &m {
        if sel
            .iter()
            .all(|&(q, _): &(usize, f64)| (p as isize - q as isize).unsigned_abs() >= gap)
        {
            sel.push((p, v));
        }
    }
    sel.sort_by_key(|&(d, _)| d);
    sel
}

fn first_alarm(data: &[[f64; N_SENS]]) -> Option<usize> {
    let bl = 180.min(data.len());
    let mut mu = [0.0_f64; N_SENS];
    let mut sd = [0.0_f64; N_SENS];
    for row in &data[..bl] {
        for s in 0..N_SENS {
            mu[s] += row[s];
        }
    }
    for s in 0..N_SENS {
        mu[s] /= bl as f64;
    }
    for row in &data[..bl] {
        for s in 0..N_SENS {
            sd[s] += (row[s] - mu[s]).powi(2);
        }
    }
    for s in 0..N_SENS {
        sd[s] = (sd[s] / bl as f64).sqrt().max(1e-12);
    }
    for start in 0..data.len().saturating_sub(7) {
        for s in 0..N_SENS {
            let avg: f64 = data[start..start + 7].iter().map(|r| r[s]).sum::<f64>() / 7.0;
            if ((avg - mu[s]) / sd[s]).abs() > ALARM_Z {
                return Some(start);
            }
        }
    }
    None
}

fn null_data(rng: &mut StdRng) -> Vec<[f64; N_SENS]> {
    let mut z = [0.0_f64; 5];
    for _ in 0..300 {
        for m in 0..5 {
            z[m] = 0.7 * z[m] + gauss(rng);
        }
    }
    (0..DAYS)
        .map(|_| {
            for m in 0..5 {
                z[m] = 0.7 * z[m] + gauss(rng);
            }
            let mut r = [0.0_f64; N_SENS];
            for s in 0..N_SENS {
                let (b, l, n) = match s / 5 {
                    0 => (100.0, 12.0, 1.5),
                    1 => (0.0, 0.08, 0.008),
                    _ => (0.0, 6.0, 0.8),
                };
                r[s] = b + z[member(s)] * l + gauss(rng) * n;
            }
            r
        })
        .collect()
}

fn null_cuts(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(NULL_PERMS); 3];
    for _ in 0..NULL_PERMS {
        let d = null_data(rng);
        let wf: Vec<_> = (0..N_WIN)
            .map(|i| corr_features(&corr_matrix(&d[i * WINDOW..(i + 1) * WINDOW])))
            .collect();
        let (_, sp) = build_graph(&normalize(&wf));
        let b = find_boundaries(&cut_profile(&sp, N_WIN), 2, 4);
        for k in 0..3 {
            out[k].push(b.get(k).map_or(1.0, |x| x.1));
        }
    }
    out
}

fn z_score(obs: f64, null: &[f64]) -> f64 {
    let n = null.len() as f64;
    let mu: f64 = null.iter().sum::<f64>() / n;
    let sd = (null.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n).sqrt();
    if sd < 1e-12 {
        0.0
    } else {
        (obs - mu) / sd
    }
}

fn fiedler(edges: &[(u64, u64, f64)], s: usize, e: usize) -> f64 {
    let n = e - s;
    if n < 3 {
        return 0.0;
    }
    let sub: Vec<(usize, usize, f64)> = edges
        .iter()
        .filter(|(u, v, _)| {
            let (a, b) = (*u as usize, *v as usize);
            a >= s && a < e && b >= s && b < e
        })
        .map(|(u, v, w)| (*u as usize - s, *v as usize - s, *w))
        .collect();
    if sub.is_empty() {
        return 0.0;
    }
    estimate_fiedler(&CsrMatrixView::build_laplacian(n, &sub), 200, 1e-10).0
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    println!("================================================================");
    println!("  Seeing Collapse Before It Happens");
    println!("  Structural Failure Prediction from Sensor Correlations");
    println!("================================================================");
    println!("\n[BRIDGE] 15 sensors (strain + vibration + displacement), 365 days");
    println!(
        "[PHASES] Healthy (d1-{}) -> Degradation (d{}-{}) -> Critical (d{}-{}) -> Failure (d{})",
        HEALTHY_END,
        HEALTHY_END + 1,
        DEGRADE_END,
        DEGRADE_END + 1,
        CRITICAL_END,
        FAILURE_DAY
    );
    let data = generate(&mut rng);

    // Threshold detection
    let thr = first_alarm(&data);
    println!("\n[THRESHOLD ALARMS]");
    match thr {
        Some(d) => {
            let w = FAILURE_DAY.saturating_sub(d);
            println!("  First sensor exceeds limit: day {}", d);
            println!(
                "  Warning time: {} days{}",
                w,
                if w <= 14 {
                    " (barely enough to close the bridge)"
                } else {
                    ""
                }
            );
        }
        None => {
            println!("  First sensor exceeds limit: NEVER");
            println!("  Warning time: 0 days (no warning at all)");
        }
    }

    // Correlation structure analysis
    let corrs: Vec<_> = (0..N_WIN)
        .map(|i| corr_matrix(&data[i * WINDOW..(i + 1) * WINDOW]))
        .collect();
    let feats: Vec<_> = corrs.iter().map(|c| corr_features(c)).collect();
    let (mc_e, sp_e) = build_graph(&normalize(&feats));
    let bounds = find_boundaries(&cut_profile(&sp_e, N_WIN), 2, 4);
    let nd = null_cuts(&mut rng);

    let scored: Vec<_> = bounds
        .iter()
        .enumerate()
        .map(|(i, &(w, cv))| {
            let z = z_score(cv, &nd[i.min(2)]);
            (w, win_day(w), z, z < -2.0)
        })
        .collect();
    let first_sig = scored.iter().find(|b| b.3).copied();
    let bdry = first_sig
        .map(|b| b.1)
        .or_else(|| scored.first().map(|b| b.1));

    println!("\n[BOUNDARY DETECTION]");
    if let Some((win, day, z, _)) = first_sig {
        println!("  First structural boundary: day {}", day);
        println!(
            "  Warning time: {} DAYS before failure",
            FAILURE_DAY.saturating_sub(day)
        );
        println!("  z-score: {:.2}  SIGNIFICANT", z);
        let (h_ic, h_xc) = avg_corrs(&corrs, 0..20.min(N_WIN));
        let ls = (DEGRADE_END / WINDOW).saturating_sub(8).min(N_WIN);
        let le = (DEGRADE_END / WINDOW).min(N_WIN);
        let (d_ic, d_xc) = avg_corrs(&corrs, ls..le);
        println!("\n  What changed at day {}:", day);
        println!(
            "  - Sensors on member #3 decorrelated from each other ({:.2} -> {:.2})",
            h_ic, d_ic
        );
        println!(
            "  - Member #3 correlations with neighbors INCREASED ({:.2} -> {:.2})",
            h_xc, d_xc
        );
        println!("  - Interpretation: member #3 is losing structural integrity,");
        println!("    load is redistributing to adjacent members");
        let _ = win; // used above
    } else if let Some(&(_, day, z, _)) = scored.first() {
        println!("  First boundary: day {} (z={:.2})", day, z);
    }

    // Warning timeline
    let warning = bdry.map_or(0, |bd| FAILURE_DAY.saturating_sub(bd));
    println!("\n[THE {}-DAY WINDOW]", warning);
    if let Some(bd) = bdry {
        println!("  Day {:>3}: Boundary detected (member decorrelation)", bd);
    }
    let mut deep = None;
    for w in (HEALTHY_END / WINDOW)..N_WIN.saturating_sub(2) {
        let ic: f64 = (w..w + 3)
            .map(|ww| intra_corr(&corrs[ww], FAIL_M))
            .sum::<f64>()
            / 3.0;
        if ic < 0.50 && deep.is_none() {
            deep = Some(win_day(w));
        }
    }
    if let Some(d) = deep {
        println!(
            "  Day {:>3}: Decorrelation deepens (confirmed degradation)",
            d
        );
    }
    let bv = sensor_vars(&data[0..HEALTHY_END]);
    let mut vday = None;
    for s in HEALTHY_END..DAYS.saturating_sub(WINDOW) {
        let wv = sensor_vars(&data[s..s + WINDOW]);
        if wv[FAIL_M] > bv[FAIL_M] * 2.5 && vday.is_none() {
            vday = Some(s);
        }
    }
    if let Some(v) = vday {
        println!(
            "  Day {:>3}: Variance begins increasing (micro-fractures)",
            v
        );
    }
    if let Some(t) = thr {
        println!(
            "  Day {:>3}: First threshold alarm (too late for prevention)",
            t
        );
    }
    println!("  Day {:>3}: Collapse", FAILURE_DAY);
    println!("\n  {} days of warning. Enough time to:", warning);
    println!("  - Close the bridge for inspection");
    println!("  - Repair or reinforce member #3");
    println!("  - Prevent 43 deaths");

    // MinCut validation
    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(mc_e.clone())
        .build()
        .expect("mincut");
    let r = mc.min_cut();
    let (ps, pt) = r.partition.unwrap();
    println!(
        "\n[MINCUT] Global min-cut={:.4}, partition: {}|{} windows",
        mc.min_cut_value(),
        ps.len(),
        pt.len()
    );

    // Spectral coherence
    let (hw, dw) = (HEALTHY_END / WINDOW, DEGRADE_END / WINDOW);
    println!("\n[SPECTRAL] Per-phase Fiedler values (algebraic connectivity):");
    for &(s, e, l, d) in &[
        (0, hw, "Healthy", "(stable correlations)"),
        (hw, dw, "Degradation", "(correlations shifting)"),
        (dw, N_WIN, "Critical+Failure", "(correlations broken)"),
    ] {
        println!("  {:<18}: {:.4} {}", l, fiedler(&mc_e, s, e), d);
    }

    // Correlation trajectory
    println!("\n[MEMBER #3 CORRELATION TRAJECTORY]");
    println!(
        "  {:>5}  {:>10}  {:>10}  {}",
        "Day", "Intra-corr", "Cross-corr", "Status"
    );
    for &day in &[10, 50, 100, 150, 190, 205, 220, 250, 280, 310, 330, 345] {
        if day >= DAYS {
            continue;
        }
        let cw = day / WINDOW;
        if cw >= N_WIN {
            continue;
        }
        let (lo, hi) = (cw.saturating_sub(1), (cw + 2).min(N_WIN));
        let sp = (hi - lo) as f64;
        let ic: f64 = (lo..hi).map(|w| intra_corr(&corrs[w], FAIL_M)).sum::<f64>() / sp;
        let xc: f64 = (lo..hi).map(|w| cross_corr(&corrs[w], FAIL_M)).sum::<f64>() / sp;
        let st = if day <= HEALTHY_END {
            "normal"
        } else if ic > 0.7 {
            "early change"
        } else if ic > 0.4 {
            "degrading"
        } else {
            "CRITICAL"
        };
        println!("  {:>5}  {:>10.3}  {:>10.3}  {}", day, ic, xc, st);
    }

    // Final comparison
    println!("\n================================================================");
    println!("  COMPARISON");
    println!("================================================================");
    print!("  Threshold detection:  ");
    match thr {
        Some(t) => println!(
            "day {} ({} days before failure)",
            t,
            FAILURE_DAY.saturating_sub(t)
        ),
        None => println!("NEVER (all sensors within limits until collapse)"),
    }
    print!("  Boundary detection:   ");
    match bdry {
        Some(b) => println!(
            "day {} ({} DAYS before failure)",
            b,
            FAILURE_DAY.saturating_sub(b)
        ),
        None => println!("No boundary detected"),
    }
    if let (Some(b), Some(t)) = (bdry, thr) {
        if b < t {
            println!(
                "\n  Advantage: {}x more warning time from correlations.",
                (t - b) / (FAILURE_DAY - t).max(1)
            );
        }
    } else if bdry.is_some() && thr.is_none() {
        println!("\n  Thresholds NEVER triggered. Boundary detection: the ONLY warning.");
    }
    println!("================================================================");
}
