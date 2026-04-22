//! Detecting Hidden Weather Regime Changes via Boundary-First Discovery.
//!
//! Temperature follows a smooth sinusoid -- you cannot see regime shifts from
//! temp alone. But variance, pressure, humidity, and correlation structure
//! change sharply at boundaries. A temporal coherence graph detects WHEN the
//! regime changed, days before the thermometer crosses any threshold.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const DAYS: usize = 365;
const WS: usize = 5;
const NW: usize = DAYS / WS; // 73
const NR: usize = 5; // raw features: temp, pressure, humidity, wind, daily_range
const NS: usize = 5; // stats: mean, std, acf1, trend, range
const NF: usize = NR * NS; // 25 window features
const NULL_N: usize = 50;
const SEED: u64 = 2024;
const BOUNDS: [usize; 3] = [80, 170, 260]; // spring, summer, autumn onset days
const RNAMES: [&str; 4] = [
    "Winter (stable)",
    "Spring (volatile)",
    "Summer (stable)",
    "Autumn (transition)",
];
const TLABELS: [&str; 3] = ["Winter->Spring", "Spring->Summer", "Summer->Autumn"];

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// [temp_offset, temp_noise, pres_mean, pres_noise, hum_mean, hum_noise,
//  wind_mean, wind_noise, range_mean, range_noise]
fn regime(r: usize) -> [f64; 10] {
    match r {
        0 => [0.0, 3.0, 1028.0, 3.0, 30.0, 4.0, 6.0, 2.0, 7.0, 1.5], // Winter
        1 => [0.0, 14.0, 1008.0, 10.0, 60.0, 15.0, 16.0, 7.0, 24.0, 6.0], // Spring
        2 => [0.0, 3.0, 1016.0, 3.0, 80.0, 4.0, 5.0, 1.5, 8.0, 1.5], // Summer
        _ => [0.0, 8.0, 1010.0, 9.0, 45.0, 10.0, 18.0, 8.0, 18.0, 5.0], // Autumn
    }
}

fn regime_of(d: usize) -> usize {
    if d < 80 {
        0
    } else if d < 170 {
        1
    } else if d < 260 {
        2
    } else {
        3
    }
}

fn gen_year(rng: &mut StdRng, multi_regime: bool) -> Vec<[f64; NR]> {
    let uniform = [0.0, 6.0, 1018.0, 5.0, 55.0, 8.0, 10.0, 3.0, 14.0, 3.0];
    (0..DAYS)
        .map(|d| {
            let p = if multi_regime {
                regime(regime_of(d))
            } else {
                uniform
            };
            let base = 55.0 + 25.0 * (2.0 * std::f64::consts::PI * (d as f64 - 15.0) / 365.0).sin();
            [
                base + p[1] * gauss(rng),
                p[2] + p[3] * gauss(rng),
                (p[4] + p[5] * gauss(rng)).clamp(5.0, 100.0),
                (p[6] + p[7] * gauss(rng)).max(0.0),
                (p[8] + p[9] * gauss(rng)).max(1.0),
            ]
        })
        .collect()
}

// --- Statistics ---
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}
fn acf1(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    let (mut n, mut d) = (0.0_f64, 0.0_f64);
    for i in 0..v.len() {
        let x = v[i] - m;
        d += x * x;
        if i + 1 < v.len() {
            n += x * (v[i + 1] - m);
        }
    }
    if d < 1e-12 {
        0.0
    } else {
        n / d
    }
}
fn trend(v: &[f64]) -> f64 {
    let xm = (v.len() as f64 - 1.0) / 2.0;
    let ym = mean(v);
    let (mut n, mut d) = (0.0_f64, 0.0_f64);
    for (i, &x) in v.iter().enumerate() {
        let dx = i as f64 - xm;
        n += dx * (x - ym);
        d += dx * dx;
    }
    if d < 1e-12 {
        0.0
    } else {
        n / d
    }
}
fn vrange(v: &[f64]) -> f64 {
    let (lo, hi) = v
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| {
            (l.min(x), h.max(x))
        });
    hi - lo
}

fn extract(data: &[[f64; NR]]) -> Vec<[f64; NF]> {
    (0..NW)
        .map(|w| {
            let s = &data[w * WS..(w + 1) * WS];
            let mut f = [0.0_f64; NF];
            for v in 0..NR {
                let vals: Vec<f64> = s.iter().map(|d| d[v]).collect();
                let b = v * NS;
                f[b] = mean(&vals);
                f[b + 1] = std_dev(&vals);
                f[b + 2] = acf1(&vals);
                f[b + 3] = trend(&vals);
                f[b + 4] = vrange(&vals);
            }
            f
        })
        .collect()
}

// --- Graph construction ---
fn dsq(a: &[f64; NF], b: &[f64; NF]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn build_graph(feats: &[[f64; NF]]) -> Vec<(u64, u64, f64)> {
    let mut dists = Vec::new();
    for i in 0..feats.len() {
        for j in (i + 1)..feats.len().min(i + 4) {
            dists.push(dsq(&feats[i], &feats[j]));
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sigma = dists[dists.len() / 2].max(1e-6);
    let mut edges = Vec::new();
    for i in 0..feats.len() {
        for skip in 1..=3usize {
            if i + skip < feats.len() {
                edges.push((
                    i as u64,
                    (i + skip) as u64,
                    (-dsq(&feats[i], &feats[i + skip]) / (2.0 * sigma))
                        .exp()
                        .max(1e-6),
                ));
            }
        }
    }
    edges
}

// --- Cut sweep ---
fn cut_profile(edges: &[(u64, u64, f64)]) -> Vec<(usize, f64)> {
    (1..NW)
        .map(|s| {
            let v: f64 = edges
                .iter()
                .filter(|(u, v, _)| {
                    let (a, b) = (*u as usize, *v as usize);
                    (a < s && b >= s) || (b < s && a >= s)
                })
                .map(|(_, _, w)| w)
                .sum();
            (s, v)
        })
        .collect()
}

fn find_bounds(cuts: &[(usize, f64)], margin: usize, gap: usize) -> Vec<(usize, f64)> {
    let mut raw: Vec<(usize, f64)> = (1..cuts.len() - 1)
        .filter_map(|i| {
            if cuts[i].0 <= margin || cuts[i].0 >= NW - margin {
                return None;
            }
            if cuts[i].1 < cuts[i - 1].1 && cuts[i].1 < cuts[i + 1].1 {
                Some(cuts[i])
            } else {
                None
            }
        })
        .collect();
    raw.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut sel = Vec::new();
    for &(w, v) in &raw {
        if sel
            .iter()
            .all(|&(s, _): &(usize, f64)| (w as isize - s as isize).unsigned_abs() >= gap)
        {
            sel.push((w, v));
        }
    }
    sel
}

fn temp_crossings(data: &[[f64; NR]], thr: f64) -> Vec<usize> {
    let avgs: Vec<f64> = (0..NW)
        .map(|w| data[w * WS..(w + 1) * WS].iter().map(|d| d[0]).sum::<f64>() / WS as f64)
        .collect();
    let mut out = Vec::new();
    for i in 1..avgs.len() {
        if (avgs[i - 1] < thr) != (avgs[i] < thr) {
            let day = i * WS;
            if out.last().map_or(true, |&p: &usize| day - p > 15) {
                out.push(day);
            }
        }
    }
    out
}

fn null_dists(rng: &mut StdRng, k: usize) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(NULL_N); k];
    for _ in 0..NULL_N {
        let f = extract(&gen_year(rng, false));
        let e = build_graph(&f);
        let p = cut_profile(&e);
        let b = find_bounds(&p, 2, 12);
        for (i, bucket) in out.iter_mut().enumerate() {
            bucket.push(b.get(i).map_or(p[p.len() / 2].1, |v| v.1));
        }
    }
    out
}

fn zscore(obs: f64, null: &[f64]) -> f64 {
    let mu: f64 = null.iter().sum::<f64>() / null.len() as f64;
    let sd = (null.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / null.len() as f64).sqrt();
    if sd < 1e-12 {
        0.0
    } else {
        (obs - mu) / sd
    }
}

fn fiedler_seg(edges: &[(u64, u64, f64)], s: usize, e: usize) -> f64 {
    if e - s < 3 {
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
    estimate_fiedler(&CsrMatrixView::build_laplacian(e - s, &se), 100, 1e-8).0
}

fn describe(feats: &[[f64; NF]], win: usize) -> String {
    let bk = 3.min(win);
    let fwd = 3.min(NW - win);
    if bk == 0 || fwd == 0 {
        return "edge".into();
    }
    let avg = |start: usize, n: usize| -> Vec<f64> {
        (0..NF)
            .map(|f| (0..n).map(|i| feats[start + i][f]).sum::<f64>() / n as f64)
            .collect()
    };
    let (bef, aft) = (avg(win - bk, bk), avg(win, fwd));
    let vn = ["temp", "pressure", "humidity", "wind", "daily_range"];
    let mut ch: Vec<(String, f64)> = Vec::new();
    for v in 0..NR {
        // only report std (idx 1) change as variance ratio
        let (bi, ai) = (bef[v * NS + 1], aft[v * NS + 1]);
        let ratio = ai / bi.max(0.01);
        if ratio > 1.5 {
            ch.push((format!("{} variance jumps {:.1}x", vn[v], ratio), ratio));
        } else if ratio < 0.67 {
            ch.push((
                format!("{} variance drops {:.1}x", vn[v], 1.0 / ratio),
                1.0 / ratio,
            ));
        }
        // mean shift
        let dm = (aft[v * NS] - bef[v * NS]).abs();
        let denom = bef[v * NS].abs().max(1.0);
        if dm / denom > 0.1 {
            let dir = if aft[v * NS] > bef[v * NS] {
                "rises"
            } else {
                "drops"
            };
            ch.push((format!("{} {} {:.0}", vn[v], dir, dm), dm / denom));
        }
    }
    ch.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ch.truncate(3);
    if ch.is_empty() {
        "subtle multivariate shift".into()
    } else {
        ch.iter()
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn nearest(day: usize) -> (usize, usize) {
    BOUNDS
        .iter()
        .enumerate()
        .min_by_key(|(_, &t)| (day as isize - t as isize).unsigned_abs())
        .map(|(i, &t)| (i, (day as isize - t as isize).unsigned_abs()))
        .unwrap()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    println!("================================================================");
    println!("  When Does the Weather REALLY Change?");
    println!("  Detecting Hidden Regime Shifts");
    println!("================================================================\n");

    let data = gen_year(&mut rng, true);
    println!(
        "[YEAR] {} days, {} five-day windows, 4 weather regimes",
        DAYS, NW
    );
    println!(
        "[REGIMES] {} -> {} -> {} -> {}\n",
        RNAMES[0], RNAMES[1], RNAMES[2], RNAMES[3]
    );

    let thr = 60.0;
    let crossings = temp_crossings(&data, thr);
    print!("[THERMOMETER] Temperature crosses {:.0}F at:", thr);
    for c in &crossings {
        print!(" day {}", c);
    }
    println!("\n  => Suggests {} transition(s)\n", crossings.len());

    let feats = extract(&data);
    let edges = build_graph(&feats);
    println!(
        "[GRAPH] {} edges over {} windows, {} features per window\n",
        edges.len(),
        NW,
        NF
    );

    let profile = cut_profile(&edges);
    let detected = find_bounds(&profile, 2, 12);
    let top3: Vec<(usize, f64)> = detected.iter().take(3).copied().collect();

    println!("[NULL] {} shuffled years (no regime changes)...", NULL_N);
    let ndists = null_dists(&mut rng, top3.len().max(1));

    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(edges.clone())
        .build()
        .expect("mincut");
    let (ps, pt) = mc.min_cut().partition.unwrap();
    println!(
        "[MINCUT] Global={:.4}, partition: {}|{}\n",
        mc.min_cut_value(),
        ps.len(),
        pt.len()
    );

    println!("[GRAPH ANALYSIS] Found {} boundaries:", top3.len());
    let mut leads = Vec::new();
    for (i, &(win, cv)) in top3.iter().enumerate() {
        let day = win * WS;
        let z = zscore(cv, ndists.get(i).map_or(&[], |v| v.as_slice()));
        let (ti, err) = nearest(day);
        let tc = crossings
            .iter()
            .min_by_key(|&&c| (c as isize - BOUNDS[ti] as isize).unsigned_abs())
            .copied();
        let lead = tc.map(|c| c as isize - day as isize);
        if let Some(l) = lead {
            if l > 0 {
                leads.push(l);
            }
        }
        let ls = match lead {
            Some(l) if l > 0 => format!("{} days BEFORE thermometer", l),
            Some(l) if l < 0 => format!("{} days after thermometer", -l),
            _ => "no thermometer crossing nearby".into(),
        };
        println!("  #{}: day {:3} ({}) -- {}", i + 1, day, TLABELS[ti], ls);
        println!(
            "      error: {} days | z-score: {:.2}  {}",
            err,
            z,
            if z < -2.0 { "SIGNIFICANT" } else { "n.s." }
        );
    }

    if !leads.is_empty() {
        let ml = leads.iter().sum::<isize>() as f64 / leads.len() as f64;
        println!("\n[KEY FINDING] Graph boundaries PRECEDE temperature changes.");
        println!(
            "  Mean lead time: {:.0} days. The structure of weather changes",
            ml
        );
        println!("  before the temperature does.");
    } else {
        println!("\n[KEY FINDING] Graph detects boundaries invisible to thermometer.");
    }

    println!("\n[WHAT CHANGES AT EACH BOUNDARY]");
    for &(w, _) in &top3 {
        let (i, _) = nearest(w * WS);
        println!("  {}: {}", TLABELS[i], describe(&feats, w));
    }

    println!("\n[SPECTRAL] Per-regime connectivity (Fiedler value):");
    let mut sw: Vec<usize> = top3.iter().map(|d| d.0).collect();
    sw.sort();
    let ss: Vec<usize> = std::iter::once(0).chain(sw.iter().copied()).collect();
    let se: Vec<usize> = sw.iter().copied().chain(std::iter::once(NW)).collect();
    for (i, (&s, &e)) in ss.iter().zip(se.iter()).enumerate() {
        println!(
            "  {} (w{}-w{}): {:.4}",
            RNAMES.get(i).unwrap_or(&"???"),
            s,
            e,
            fiedler_seg(&edges, s, e)
        );
    }

    println!("\n================================================================");
    println!("  SUMMARY");
    println!("================================================================");
    println!(
        "  True boundaries:  day {} (spring), {} (summer), {} (autumn)",
        BOUNDS[0], BOUNDS[1], BOUNDS[2]
    );
    print!("  Graph detected:  ");
    for &(w, _) in &top3 {
        print!(" day {}", w * WS);
    }
    println!();
    print!("  Thermometer:     ");
    for c in &crossings {
        print!(" day {}", c);
    }
    println!();
    let all_sig = top3
        .iter()
        .enumerate()
        .all(|(i, &(_, cv))| zscore(cv, ndists.get(i).map_or(&[], |v| v.as_slice())) < -2.0);
    if all_sig && !top3.is_empty() {
        println!("  All {} boundaries significant (z < -2.0).", top3.len());
    }
    if !leads.is_empty() {
        println!(
            "  Mean lead time over thermometer: {:.0} days.",
            leads.iter().sum::<isize>() as f64 / leads.len() as f64
        );
    }
    println!("================================================================\n");
}
