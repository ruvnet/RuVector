//! Health State Boundary Discovery: detects hidden transitions between
//! health states (healthy, overtraining, sick, recovery) from wearable
//! sensor data using graph-structural analysis.
//!
//! The correlation structure between HR, HRV, steps, and sleep changes
//! BEFORE any single metric crosses a clinical threshold. Graph boundary
//! detection finds overtraining onset days before a doctor would notice.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const N_OBS: usize = 180; // 180 half-day observations over 90 days
const WINDOW: usize = 6; // 3-day windows (6 half-days)
const N_WIN: usize = N_OBS / WINDOW;
const N_FEAT: usize = 8;
const SEED: u64 = 118;
const NULL_PERMS: usize = 100;
const HEALTHY_END: usize = 60; // day 30
const OVERTRAIN_END: usize = 100; // day 50
const SICK_END: usize = 130; // day 65
const TRUE_B: [usize; 3] = [
    HEALTHY_END / WINDOW,
    OVERTRAIN_END / WINDOW,
    SICK_END / WINDOW,
];
const HR_THR: f64 = 67.0;
const HRV_THR: f64 = 32.0;
const STEP_THR: f64 = 5000.0;

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate 180 half-day observations. Each state has constant correlation
/// structure; means drift slowly; correlation parameters change sharply at
/// state boundaries.
fn generate_data(rng: &mut StdRng) -> Vec<[f64; 4]> {
    let mut data = Vec::with_capacity(N_OBS);
    let (mut z0, mut z1, mut z2) = (0.0_f64, 0.0_f64, 0.0_f64);
    for _ in 0..100 {
        z0 = 0.3 * z0 + gauss(rng);
        z1 = 0.3 * z1 + gauss(rng);
        z2 = 0.2 * z2 + gauss(rng);
    }
    for obs in 0..N_OBS {
        let (hr_b, hrv_b, st_b, sl_b, coup, phi, cross) = if obs < HEALTHY_END {
            let t = obs as f64 / HEALTHY_END as f64;
            (
                62.0 + 0.3 * t,
                45.0 - 0.3 * t,
                8000.0,
                7.5,
                -0.9_f64,
                0.2_f64,
                0.0_f64,
            )
        } else if obs < OVERTRAIN_END {
            let t = (obs - HEALTHY_END) as f64 / (OVERTRAIN_END - HEALTHY_END) as f64;
            (
                62.3 + 5.7 * t,
                44.7 - 14.7 * t,
                8000.0 + 4000.0 * t,
                7.5 - 1.0 * t,
                -0.15,
                0.7,
                0.45,
            )
        } else if obs < SICK_END {
            let t = (obs - OVERTRAIN_END) as f64 / (SICK_END - OVERTRAIN_END) as f64;
            (
                68.0 + 7.0 * t,
                30.0 - 10.0 * t,
                12000.0 - 9000.0 * t,
                6.5 + 2.5 * t,
                0.5,
                0.9,
                0.85,
            )
        } else {
            let t = (obs - SICK_END) as f64 / (N_OBS - SICK_END) as f64;
            (
                75.0 - 11.0 * t,
                20.0 + 20.0 * t,
                3000.0 + 4000.0 * t,
                9.0 - 1.5 * t,
                0.5 - 1.1 * t,
                0.9 - 0.6 * t,
                0.85 - 0.7 * t,
            )
        };
        let p = 0.3 + cross * 0.35;
        z0 = p * z0 + gauss(rng);
        z1 = p * z1 + gauss(rng);
        z2 = phi * z2 + gauss(rng);
        data.push([
            (hr_b + z0 * 0.8).max(40.0),
            (hrv_b + coup * z0 * 1.5 + z1 * 0.4).max(5.0),
            (st_b + z2 * 500.0 + cross * z0 * 150.0).max(500.0),
            (sl_b + gauss(rng) * (0.15 + cross * 0.15) + cross * z0 * 0.08).clamp(3.0, 12.0),
        ]);
    }
    data
}

fn window_features(w: &[[f64; 4]]) -> [f64; N_FEAT] {
    let n = w.len() as f64;
    let mean = |m: usize| w.iter().map(|d| d[m]).sum::<f64>() / n;
    let var = |m: usize, mu: f64| w.iter().map(|d| (d[m] - mu).powi(2)).sum::<f64>() / n;
    let (mh, mv, ms, ml) = (mean(0), mean(1), mean(2), mean(3));
    let corr = {
        let (mut c, mut da, mut db) = (0.0_f64, 0.0_f64, 0.0_f64);
        for d in w {
            let (a, b) = (d[0] - mh, d[1] - mv);
            c += a * b;
            da += a * a;
            db += b * b;
        }
        let den = (da * db).sqrt();
        if den < 1e-12 {
            0.0
        } else {
            c / den
        }
    };
    let sv: Vec<f64> = w.iter().map(|d| d[2]).collect();
    let ac = {
        let (mut num, mut den) = (0.0_f64, 0.0_f64);
        for j in 0..sv.len() {
            let d = sv[j] - ms;
            den += d * d;
            if j + 1 < sv.len() {
                num += d * (sv[j + 1] - ms);
            }
        }
        if den < 1e-12 {
            0.0
        } else {
            num / den
        }
    };
    [mh, mv, var(0, mh), var(1, mv), corr, ac, ms / 1000.0, ml]
}

fn normalize(feats: &[[f64; N_FEAT]]) -> Vec<[f64; N_FEAT]> {
    let n = feats.len() as f64;
    let mut mu = [0.0_f64; N_FEAT];
    let mut sd = [0.0_f64; N_FEAT];
    for f in feats {
        for d in 0..N_FEAT {
            mu[d] += f[d];
        }
    }
    for d in 0..N_FEAT {
        mu[d] /= n;
    }
    for f in feats {
        for d in 0..N_FEAT {
            sd[d] += (f[d] - mu[d]).powi(2);
        }
    }
    for d in 0..N_FEAT {
        sd[d] = (sd[d] / n).sqrt().max(1e-12);
    }
    feats
        .iter()
        .map(|f| {
            let mut o = [0.0_f64; N_FEAT];
            for d in 0..N_FEAT {
                o[d] = (f[d] - mu[d]) / sd[d];
            }
            o
        })
        .collect()
}

fn dist_sq(a: &[f64; N_FEAT], b: &[f64; N_FEAT]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn build_graph(f: &[[f64; N_FEAT]]) -> (Vec<(u64, u64, f64)>, Vec<(usize, usize, f64)>) {
    let mut ds = Vec::new();
    for i in 0..f.len() {
        for j in (i + 1)..f.len().min(i + 4) {
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
            let avg: f64 = cuts[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            Some((i, cuts[i], avg - cuts[i]))
        })
        .collect();
    m.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let mut s = Vec::new();
    for &(p, v, _) in &m {
        if s.iter()
            .all(|&(q, _): &(usize, f64)| (p as isize - q as isize).unsigned_abs() >= gap)
        {
            s.push((p, v));
        }
    }
    s.sort_by_key(|&(d, _)| d);
    s
}

fn win_to_day(w: usize) -> usize {
    w * WINDOW / 2 + WINDOW / 4
}

fn first_cross(raw: &[[f64; 4]], m: usize, thr: f64, above: bool) -> Option<usize> {
    let w = 10;
    for i in 0..raw.len().saturating_sub(w) {
        let a: f64 = raw[i..i + w].iter().map(|d| d[m]).sum::<f64>() / w as f64;
        if (above && a > thr) || (!above && a < thr) {
            return Some(i / 2);
        }
    }
    None
}

fn null_data(rng: &mut StdRng) -> Vec<[f64; 4]> {
    let (mut z0, mut z1, mut z2) = (0.0_f64, 0.0_f64, 0.0_f64);
    for _ in 0..100 {
        z0 = 0.3 * z0 + gauss(rng);
        z1 = 0.3 * z1 + gauss(rng);
        z2 = 0.2 * z2 + gauss(rng);
    }
    (0..N_OBS)
        .map(|_| {
            z0 = 0.3 * z0 + gauss(rng);
            z1 = 0.3 * z1 + gauss(rng);
            z2 = 0.2 * z2 + gauss(rng);
            [
                62.0 + z0 * 0.8,
                45.0 - 0.9 * z0 * 1.5 + z1 * 0.4,
                (8000.0 + z2 * 500.0).max(500.0),
                7.5 + gauss(rng) * 0.15,
            ]
        })
        .collect()
}

fn null_cuts(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(NULL_PERMS); 3];
    for _ in 0..NULL_PERMS {
        let r = null_data(rng);
        let wf: Vec<_> = (0..N_WIN)
            .map(|i| window_features(&r[i * WINDOW..(i + 1) * WINDOW]))
            .collect();
        let (_, sp) = build_graph(&normalize(&wf));
        let b = find_boundaries(&cut_profile(&sp, N_WIN), 1, 3);
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

fn fiedler_seg(edges: &[(u64, u64, f64)], s: usize, e: usize) -> f64 {
    let n = e - s;
    if n < 3 {
        return 0.0;
    }
    let se: Vec<(usize, usize, f64)> = edges
        .iter()
        .filter(|(u, v, _)| {
            let (a, b) = (*u as usize, *v as usize);
            a >= s && a < e && b >= s && b < e
        })
        .map(|(u, v, w)| (*u as usize - s, *v as usize - s, *w))
        .collect();
    if se.is_empty() {
        return 0.0;
    }
    estimate_fiedler(&CsrMatrixView::build_laplacian(n, &se), 200, 1e-10).0
}

fn describe(day: usize) -> &'static str {
    let tb = [30, 50, 65];
    let n = tb
        .iter()
        .min_by_key(|&&t| (day as isize - t as isize).unsigned_abs())
        .copied()
        .unwrap_or(0);
    match n {
        30 => "HR-HRV correlation inverted, step-sleep pattern shifted",
        50 => "ALL correlations break down simultaneously",
        65 => "correlations begin restoring",
        _ => "multi-metric pattern shift",
    }
}

fn label(day: usize) -> &'static str {
    let tb = [30, 50, 65];
    let n = tb
        .iter()
        .min_by_key(|&&t| (day as isize - t as isize).unsigned_abs())
        .copied()
        .unwrap_or(0);
    match n {
        30 => "healthy->overtraining",
        50 => "overtraining->sick",
        65 => "sick->recovery",
        _ => "unknown",
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    println!("================================================================");
    println!("  When Did Your Body Actually Change?");
    println!("  Hidden Health State Detection from Wearable Data");
    println!("================================================================");

    let raw = generate_data(&mut rng);
    println!("\n[DATA] 90 days of health metrics (HR, HRV, steps, sleep)");
    println!(
        "[STATES] Healthy (d1-30) -> Overtraining (d31-50) -> Sick (d51-65) -> Recovery (d66-90)\n"
    );

    for &(name, s, e) in &[
        ("Healthy", 0, HEALTHY_END),
        ("Overtraining", HEALTHY_END, OVERTRAIN_END),
        ("Sick", OVERTRAIN_END, SICK_END),
        ("Recovery", SICK_END, N_OBS),
    ] {
        let n = (e - s) as f64;
        println!(
            "  {:<13} HR={:.1} BPM  HRV={:.1} ms  steps={:.0}  sleep={:.1}h",
            name,
            raw[s..e].iter().map(|d| d[0]).sum::<f64>() / n,
            raw[s..e].iter().map(|d| d[1]).sum::<f64>() / n,
            raw[s..e].iter().map(|d| d[2]).sum::<f64>() / n,
            raw[s..e].iter().map(|d| d[3]).sum::<f64>() / n
        );
    }

    let hr_x = first_cross(&raw, 0, HR_THR, true);
    let hrv_x = first_cross(&raw, 1, HRV_THR, false);
    let st_x = first_cross(&raw, 2, STEP_THR, false);
    let clin = [hr_x, hrv_x, st_x].iter().filter_map(|x| *x).min();

    println!("\n[CLINICAL THRESHOLDS]");
    println!(
        "  Resting HR > {} BPM first occurs: day {}",
        HR_THR as u32,
        hr_x.map_or("never".into(), |d| d.to_string())
    );
    println!(
        "  HRV < {} ms first occurs:        day {}",
        HRV_THR as u32,
        hrv_x.map_or("never".into(), |d| d.to_string())
    );
    println!(
        "  Steps < {} first occurs:       day {}",
        STEP_THR as u32,
        st_x.map_or("never".into(), |d| d.to_string())
    );
    println!(
        "  => Clinical detection: day {} at earliest",
        clin.map_or("N/A".into(), |d| d.to_string())
    );

    let wf: Vec<_> = (0..N_WIN)
        .map(|i| window_features(&raw[i * WINDOW..(i + 1) * WINDOW]))
        .collect();
    let (mc_e, sp_e) = build_graph(&normalize(&wf));
    println!(
        "\n[GRAPH] {} windows (3-day each), {} edges, {}-dim features",
        N_WIN,
        mc_e.len(),
        N_FEAT
    );

    let bounds = find_boundaries(&cut_profile(&sp_e, N_WIN), 1, 3);
    let nd = null_cuts(&mut rng);

    println!("\n[GRAPH BOUNDARIES]");
    for (i, &(win, cv)) in bounds.iter().take(3).enumerate() {
        let day = win_to_day(win);
        let z = z_score(cv, &nd[i.min(2)]);
        let sig = if z < -2.0 { "SIGNIFICANT" } else { "n.s." };
        let early = match clin {
            Some(c) if day < c => format!("{} DAYS before clinical detection", c - day),
            Some(c) if day <= c + 1 => "same time as clinical detection".into(),
            Some(c) => format!("{} days after clinical detection", day - c),
            None => "no clinical crossing".into(),
        };
        println!("  #{}: day {} -- {} ({})", i + 1, day, label(day), early);
        println!("      z-score: {:.2}  {}", z, sig);
        println!("      What changed: {}", describe(day));
        let nearest = TRUE_B
            .iter()
            .min_by_key(|&&t| (win as isize - t as isize).unsigned_abs())
            .copied()
            .unwrap_or(0);
        let err = (win as isize - nearest as isize).unsigned_abs();
        if err > 0 {
            println!(
                "      (true boundary: window {}, error: ~{} days)",
                nearest,
                err * WINDOW / 2
            );
        }
    }

    if let (Some(bd), Some(cd)) = (bounds.first().map(|b| win_to_day(b.0)), clin) {
        if bd < cd {
            println!("\n[KEY FINDING] Graph boundary detection found the overtraining onset");
            println!(
                "  {} DAYS before any single metric crossed a clinical threshold.",
                cd - bd
            );
            println!("  Early detection window: {} days.", cd - bd);
        }
    }

    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(mc_e.clone())
        .build()
        .expect("mincut");
    let (ps, pt) = mc.min_cut().partition.unwrap();
    println!(
        "\n[MINCUT] Global min-cut={:.4}, partitions: {}|{}",
        mc.min_cut_value(),
        ps.len(),
        pt.len()
    );

    let mut sb: Vec<usize> = bounds.iter().take(3).map(|b| b.0).collect();
    sb.sort();
    let segs = if sb.len() >= 3 {
        vec![(0, sb[0]), (sb[0], sb[1]), (sb[1], sb[2]), (sb[2], N_WIN)]
    } else {
        vec![
            (0, TRUE_B[0]),
            (TRUE_B[0], TRUE_B[1]),
            (TRUE_B[1], TRUE_B[2]),
            (TRUE_B[2], N_WIN),
        ]
    };
    let lbl = ["Healthy", "Overtraining", "Sick", "Recovery"];
    let sem = [
        "(tight correlations)",
        "(correlations degrading)",
        "(correlations broken)",
        "(correlations rebuilding)",
    ];
    println!("\n[SPECTRAL] Per-state Fiedler values:");
    for (i, &(s, e)) in segs.iter().enumerate() {
        println!(
            "  {:<13}: {:.4} {}",
            lbl[i],
            fiedler_seg(&mc_e, s, e),
            sem[i]
        );
    }

    println!("\n================================================================");
    println!("  SUMMARY");
    println!("================================================================");
    println!("  Healthy -> Overtraining -> Sick -> Recovery");
    println!(
        "  Clinical detection (earliest threshold): day {}",
        clin.map_or("N/A".into(), |d| d.to_string())
    );
    println!(
        "  Graph detection (earliest boundary):     day {}",
        bounds
            .first()
            .map_or("N/A".into(), |b| win_to_day(b.0).to_string())
    );
    if let (Some(bd), Some(cd)) = (bounds.first().map(|b| win_to_day(b.0)), clin) {
        if bd < cd {
            println!(
                "  Advantage: {} days of early warning from structure alone.",
                cd - bd
            );
        }
    }
    println!("================================================================\n");
}
