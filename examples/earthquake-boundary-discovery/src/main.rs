//! **Earthquake Precursor Detection via Boundary-First Discovery**
//!
//! Amplitude-based detectors fire DURING the quake -- zero warning. But the
//! CORRELATION STRUCTURE of background seismic noise changes days before a
//! major event. Not the amplitude, but how monitoring stations relate to each
//! other. This experiment: 20 stations, 200 days, fault zone. Pre-seismic
//! correlations shift from isotropic (~0.3) to directional (~0.7 along
//! fault) while amplitudes stay normal. Boundary-first detection catches it.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const NS: usize = 20; // stations
const ND: usize = 200; // days
const HR: usize = 24;
const WD: usize = 5; // days per window
const NW: usize = ND / WD; // 40
const MAIN: usize = 161; // mainshock day
const PRE: usize = 121; // pre-seismic onset
const NULL_N: usize = 50;
const SEED: u64 = 2025;
const NC: usize = NS * (NS - 1) / 2; // 190 correlation pairs
const NE: usize = 5; // eigenvalues
const NF: usize = NS + NC + NE; // 215 features per window

// Station positions: 0..9 on-fault (near y=0), 10..19 off-fault
fn positions() -> [(f64, f64); NS] {
    let mut p = [(0.0, 0.0); NS];
    for i in 0..10 {
        p[i] = (i as f64 * 2.0, (i as f64 * 0.3).sin() * 0.5);
    }
    for i in 0..10 {
        p[10 + i] = (
            i as f64 * 2.0,
            if i % 2 == 0 { 1.0 } else { -1.0 } * (5.0 + i as f64 * 0.5),
        );
    }
    p
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Normal,
    Pre,
    Main,
    After,
}

fn phase(d: usize) -> Phase {
    if d < PRE {
        Phase::Normal
    } else if d < MAIN {
        Phase::Pre
    } else if d == MAIN {
        Phase::Main
    } else {
        Phase::After
    }
}

fn pname(p: Phase) -> &'static str {
    match p {
        Phase::Normal => "Normal",
        Phase::Pre => "Pre-seismic",
        Phase::Main => "Mainshock",
        Phase::After => "Aftershock",
    }
}

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate [day][hour][station] seismic amplitudes.
fn gen(rng: &mut StdRng, precursor: bool) -> Vec<Vec<[f64; NS]>> {
    let pos = positions();
    (0..ND)
        .map(|day| {
            let ph = if precursor { phase(day) } else { Phase::Normal };
            let base = 0.3_f64;
            let (rho_on, rho_off) = match ph {
                Phase::Normal => (base, base),
                Phase::Pre => {
                    let t = (day - PRE) as f64 / (MAIN - PRE) as f64;
                    (base + 0.30 + t * 0.20, base + t * 0.03) // on: 0.60->0.80, off: ~0.30
                }
                Phase::Main => (0.95, 0.95),
                Phase::After => {
                    let d = 0.95 / (1.0 + (day - MAIN) as f64 * 0.1);
                    (d.max(0.35), d.max(0.30))
                }
            };
            let amp = match ph {
                Phase::Normal | Phase::Pre => 1.0,
                Phase::Main => 50.0,
                Phase::After => 1.0 + 30.0 / ((day - MAIN) as f64 + 1.0),
            };
            let small = matches!(ph, Phase::Normal if rng.gen::<f64>() < 0.04)
                || matches!(ph, Phase::After if rng.gen::<f64>() < 0.25);
            let ea = if small {
                3.0 + rng.gen::<f64>() * 4.0
            } else {
                0.0
            };
            let eh: usize = rng.gen_range(0..HR);
            (0..HR)
                .map(|h| {
                    let zc = gauss(rng);
                    let mut v = [0.0_f64; NS];
                    for s in 0..NS {
                        let r = (if s < 10 { rho_on } else { rho_off }).clamp(0.0, 0.99);
                        let mut x = (r.sqrt() * zc + (1.0 - r).sqrt() * gauss(rng)) * amp;
                        if small && h == eh {
                            let d = ((pos[s].0 - pos[0].0).powi(2) + (pos[s].1 - pos[0].1).powi(2))
                                .sqrt();
                            x += ea * (-d / 10.0).exp();
                        }
                        v[s] = x;
                    }
                    v
                })
                .collect()
        })
        .collect()
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        c += da * db;
        va += da * da;
        vb += db * db;
    }
    let d = (va * vb).sqrt();
    if d < 1e-12 {
        0.0
    } else {
        c / d
    }
}

fn top_eigs(mat: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<f64> {
    let n = mat.len();
    let mut def: Vec<Vec<f64>> = mat.to_vec();
    (0..k)
        .map(|_| {
            let mut v: Vec<f64> = (0..n).map(|_| gauss(rng)).collect();
            let nm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter_mut().for_each(|x| *x /= nm);
            let mut ev = 0.0;
            for _ in 0..100 {
                let mv: Vec<f64> = (0..n)
                    .map(|i| (0..n).map(|j| def[i][j] * v[j]).sum())
                    .collect();
                ev = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
                if ev < 1e-12 {
                    break;
                }
                for i in 0..n {
                    v[i] = mv[i] / ev;
                }
            }
            for i in 0..n {
                for j in 0..n {
                    def[i][j] -= ev * v[i] * v[j];
                }
            }
            ev
        })
        .collect()
}

fn extract(data: &[Vec<[f64; NS]>], w: usize, rng: &mut StdRng) -> [f64; NF] {
    let (ds, de) = (w * WD, (w * WD + WD).min(data.len()));
    let mut f = [0.0_f64; NF];
    let mut tr: Vec<Vec<f64>> = vec![Vec::new(); NS];
    let mut mx = [0.0_f64; NS];
    for d in ds..de {
        for h in 0..data[d].len() {
            for s in 0..NS {
                let v = data[d][h][s];
                tr[s].push(v);
                if v.abs() > mx[s] {
                    mx[s] = v.abs();
                }
            }
        }
    }
    for s in 0..NS {
        f[s] = mx[s];
    }
    let mut cm = vec![vec![0.0_f64; NS]; NS];
    let mut idx = NS;
    for i in 0..NS {
        cm[i][i] = 1.0;
        for j in (i + 1)..NS {
            let r = pearson(&tr[i], &tr[j]);
            cm[i][j] = r;
            cm[j][i] = r;
            f[idx] = r;
            idx += 1;
        }
    }
    for (k, e) in top_eigs(&cm, NE, rng).into_iter().enumerate() {
        f[NS + NC + k] = e;
    }
    f
}

fn corr_subset(f: &[f64; NF], pred: fn(usize, usize) -> bool) -> f64 {
    let (mut s, mut c) = (0.0, 0);
    let mut idx = NS;
    for i in 0..NS {
        for j in (i + 1)..NS {
            if pred(i, j) {
                s += f[idx];
                c += 1;
            }
            idx += 1;
        }
    }
    if c > 0 {
        s / c as f64
    } else {
        0.0
    }
}
fn mean_corr(f: &[f64; NF]) -> f64 {
    f[NS..NS + NC].iter().sum::<f64>() / NC as f64
}
fn on_corr(f: &[f64; NF]) -> f64 {
    corr_subset(f, |i, j| i < 10 && j < 10)
}
fn off_corr(f: &[f64; NF]) -> f64 {
    corr_subset(f, |i, j| i >= 10 && j >= 10)
}

fn dsq(a: &[f64; NF], b: &[f64; NF]) -> f64 {
    let mut d = 0.0;
    for i in 0..NS {
        d += (a[i] - b[i]).powi(2);
    }
    for i in NS..NS + NC {
        d += 10.0 * (a[i] - b[i]).powi(2);
    } // correlations 10x weight
    for i in NS + NC..NF {
        d += 5.0 * (a[i] - b[i]).powi(2);
    }
    d
}

fn build_graph(feats: &[[f64; NF]]) -> (Vec<(u64, u64, f64)>, Vec<(usize, usize, f64)>) {
    let mut ds = Vec::new();
    for i in 0..feats.len() {
        for j in (i + 1)..feats.len().min(i + 5) {
            ds.push(dsq(&feats[i], &feats[j]));
        }
    }
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sigma = ds[ds.len() / 2].max(1e-6);
    let (mut mc, mut sp) = (Vec::new(), Vec::new());
    for i in 0..feats.len() {
        for sk in 1..=4usize {
            if i + sk < feats.len() {
                let w = (-dsq(&feats[i], &feats[i + sk]) / (2.0 * sigma))
                    .exp()
                    .max(1e-6);
                mc.push((i as u64, (i + sk) as u64, w));
                sp.push((i, i + sk, w));
            }
        }
    }
    (mc, sp)
}

fn cut_profile(e: &[(usize, usize, f64)], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n];
    for &(u, v, w) in e {
        for k in (u.min(v) + 1)..=u.max(v) {
            if k < n {
                c[k] += w;
            }
        }
    }
    c
}

fn find_bounds(cuts: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut found = Vec::new();
    let mut mask = vec![false; cuts.len()];
    for _ in 0..k {
        let mut best = (0usize, f64::INFINITY);
        for p in 2..cuts.len().saturating_sub(2) {
            if !mask[p] && cuts[p] < best.1 {
                best = (p, cuts[p]);
            }
        }
        if best.1 == f64::INFINITY {
            break;
        }
        found.push(best);
        for m in best.0.saturating_sub(5)..=(best.0 + 5).min(cuts.len() - 1) {
            mask[m] = true;
        }
    }
    found.sort_by_key(|&(w, _)| w);
    found
}

fn null_dist(rng: &mut StdRng) -> Vec<f64> {
    (0..NULL_N)
        .map(|_| {
            let d = gen(rng, false);
            let f: Vec<_> = (0..NW).map(|w| extract(&d, w, rng)).collect();
            let (_, sp) = build_graph(&f);
            let c = cut_profile(&sp, NW);
            (2..NW - 2).map(|k| c[k]).fold(f64::INFINITY, f64::min)
        })
        .collect()
}

fn zscore(obs: f64, null: &[f64]) -> f64 {
    let mu = null.iter().sum::<f64>() / null.len() as f64;
    let sd = (null.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / null.len() as f64).sqrt();
    if sd < 1e-12 {
        0.0
    } else {
        (obs - mu) / sd
    }
}

fn fiedler(edges: &[(usize, usize, f64)], w0: usize, w1: usize) -> f64 {
    if w1 - w0 < 3 {
        return 0.0;
    }
    let seg: Vec<_> = edges
        .iter()
        .filter(|&&(u, v, _)| u >= w0 && u < w1 && v >= w0 && v < w1)
        .map(|&(u, v, w)| (u - w0, v - w0, w))
        .collect();
    if seg.is_empty() {
        return 0.0;
    }
    estimate_fiedler(&CsrMatrixView::build_laplacian(w1 - w0, &seg), 200, 1e-10).0
}

fn amp_alert(feats: &[[f64; NF]]) -> Option<usize> {
    let bl: f64 = (0..20)
        .map(|w| (0..NS).map(|s| feats[w][s]).sum::<f64>() / NS as f64)
        .sum::<f64>()
        / 20.0;
    (0..feats.len())
        .find(|&w| (0..NS).map(|s| feats[w][s]).sum::<f64>() / NS as f64 > bl * 5.0)
        .map(|w| w * WD)
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    println!("================================================================");
    println!("  Can We See Earthquakes Coming?");
    println!("  Boundary-First Seismic Precursor Detection");
    println!("================================================================");
    println!(
        "[NETWORK] {} stations, {} days, fault zone monitoring",
        NS, ND
    );
    println!("[PHASES] Normal (1-{}) -> Pre-seismic ({}-{}) -> Mainshock (day {}) -> Aftershocks ({}-{})\n",
        PRE-1, PRE, MAIN-1, MAIN, MAIN+1, ND);

    let data = gen(&mut rng, true);
    let feats: Vec<_> = (0..NW).map(|w| extract(&data, w, &mut rng)).collect();

    // Amplitude detection
    let ad = amp_alert(&feats);
    println!("[AMPLITUDE DETECTION]");
    match ad {
        Some(d) if d >= MAIN => println!("  First alert: day {} (DURING the earthquake)\n  Warning time: 0 days\n  Usefulness: NONE (too late)\n", d),
        Some(d) => println!("  First alert: day {}\n  Warning time: {} days\n  Usefulness: {}\n", d, MAIN-d, if MAIN-d <= 1 {"minimal"} else {"limited"}),
        None => println!("  No amplitude alert before mainshock\n  Usefulness: NONE\n"),
    }

    // Build graph
    let (mc_e, sp_e) = build_graph(&feats);
    let bounds = find_bounds(&cut_profile(&sp_e, NW), 5);
    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(mc_e)
        .build()
        .expect("mincut");
    let (ps, pt) = mc.min_cut().partition.unwrap();
    println!(
        "[GRAPH] {} windows x {} features, partition {}|{}, global mincut={:.4}\n",
        NW,
        NF,
        ps.len(),
        pt.len(),
        mc.min_cut_value()
    );

    // Null test
    println!(
        "[NULL TEST] {} years of pure noise (no pre-seismic phase)...",
        NULL_N
    );
    let null = null_dist(&mut rng);

    // Score all boundaries
    let scored: Vec<(usize, f64, f64)> = bounds
        .iter()
        .map(|&(w, cv)| (w, cv, zscore(cv, &null)))
        .collect();
    let precursor = scored
        .iter()
        .filter(|(w, _, z)| *z < -2.0 && *w * WD < MAIN)
        .min_by_key(|(w, _, _)| *w);

    // Report
    println!("\n[BOUNDARY DETECTION]");
    if let Some(&(w, _, z)) = precursor {
        let det = w * WD;
        println!("  First structural boundary: day {}", det);
        println!("  Warning time: {} DAYS before mainshock", MAIN - det);
        println!("  z-score: {:.2}  SIGNIFICANT", z);
        println!("  What changed: inter-station correlation pattern shifted");
        println!(
            "  from isotropic ({:.2} everywhere) to directional ({:.2} along fault)",
            mean_corr(&feats[2]),
            on_corr(&feats[w])
        );
        println!(
            "  On-fault:  {:.2} -> {:.2}",
            on_corr(&feats[2]),
            on_corr(&feats[w])
        );
        println!(
            "  Off-fault: {:.2} -> {:.2}",
            off_corr(&feats[2]),
            off_corr(&feats[w])
        );
    } else if let Some(&(w, _, z)) = scored.iter().find(|(w, _, _)| *w * WD < MAIN) {
        println!(
            "  First boundary: day {} (z={:.2}), {} days before mainshock",
            w * WD,
            z,
            MAIN - w * WD
        );
    }

    let det_day = precursor.map(|&(w, _, _)| w * WD).or_else(|| {
        scored
            .iter()
            .find(|(w, _, _)| *w * WD < MAIN)
            .map(|&(w, _, _)| w * WD)
    });
    let bw = det_day.map(|d| MAIN - d).unwrap_or(0);

    println!("\n[THE {}-DAY WARNING WINDOW]", bw);
    if let Some(dd) = det_day {
        println!(
            "  Day {}: Correlation boundary detected (graph structure shifted)",
            dd
        );
        println!(
            "  Day {}-{}: Correlations continue building (confirmed trend)",
            dd,
            MAIN - 1
        );
        println!("  Day {}: Mainshock\n", MAIN);
        println!("  During the warning window:");
        println!("  - Seismograms look NORMAL (same amplitude)");
        println!("  - No individual station shows anything unusual");
        println!("  - ONLY the correlation structure reveals the stress buildup");
    }

    println!("\n[ALL BOUNDARIES]");
    for (i, &(w, _, z)) in scored.iter().enumerate() {
        let star = if precursor.map_or(false, |p| p.0 == w) {
            " <-- PRECURSOR"
        } else {
            ""
        };
        println!(
            "  #{}: day {:3} ({:12}) z={:6.2}  {}{}",
            i + 1,
            w * WD,
            pname(phase((w * WD).min(ND - 1))),
            z,
            if z < -2.0 { "SIGNIFICANT" } else { "n.s." },
            star
        );
    }

    // Correlation timeline
    println!("\n[CORRELATION TIMELINE]  (mean pairwise correlation per window)");
    print!("  ");
    for w in 0..NW {
        let c = mean_corr(&feats[w]);
        print!(
            "{}",
            if c > 0.6 {
                '#'
            } else if c > 0.4 {
                '='
            } else if c > 0.2 {
                '-'
            } else {
                '.'
            }
        );
    }
    println!();
    let (pw, mw) = (PRE / WD, MAIN / WD);
    print!("  ");
    for w in 0..NW {
        print!(
            "{}",
            if w == pw {
                'P'
            } else if w == mw {
                'M'
            } else {
                ' '
            }
        );
    }
    println!("  P=pre-seismic  M=mainshock");
    print!("  ");
    for w in 0..NW {
        print!(
            "{}",
            if bounds.iter().any(|&(b, _)| b == w) {
                '^'
            } else {
                ' '
            }
        );
    }
    println!("  ^=detected");

    // Directional analysis
    println!("\n[DIRECTIONAL ANALYSIS]  (on-fault vs off-fault correlation)");
    println!(
        "  {:>6}  {:>8}  {:>9}  {:>5}  {}",
        "Window", "On-fault", "Off-fault", "Ratio", "Phase"
    );
    for w in (0..NW).step_by(4) {
        let (on, off) = (on_corr(&feats[w]), off_corr(&feats[w]));
        println!(
            "  w{:2}({:3})    {:.3}      {:.3}    {:.1}x   {}",
            w,
            w * WD,
            on,
            off,
            on / off.abs().max(0.01),
            pname(phase((w * WD).min(ND - 1)))
        );
    }

    // Spectral
    println!("\n[SPECTRAL] Per-phase Fiedler values:");
    for (name, s, e) in [
        ("Normal", 0, pw),
        ("Pre-seismic", pw, mw),
        ("Aftershock", mw + 1, NW),
    ] {
        if e > s {
            println!(
                "  {:<14} (w{}-w{}): {:.4}",
                name,
                s,
                e,
                fiedler(&sp_e, s, e)
            );
        }
    }

    // Summary
    let aw = ad
        .map(|d| if d >= MAIN { 0 } else { MAIN - d })
        .unwrap_or(0);
    println!("\n================================================================");
    println!("  SUMMARY");
    println!("================================================================");
    println!("  Amplitude detection warning:  {} days", aw);
    println!("  Boundary detection warning:   {} days", bw);
    if bw > aw + 5 {
        println!(
            "\n  The correlation structure changed {} DAYS before the mainshock,",
            bw
        );
        println!("  while amplitude detection gave {} days warning.", aw);
        println!(
            "  Boundary-first detection found the precursor {} DAYS earlier.",
            bw - aw
        );
        println!("\n  The earthquake was invisible on seismograms during the warning window.");
        println!("  No single station amplitude changed. Only the WAY stations");
        println!("  correlated with each other revealed the approaching rupture.");
    }
    println!("================================================================");
}
