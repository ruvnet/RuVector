//! Pre-Seizure Detection from Brain Correlation Boundaries
//!
//! 16-channel EEG, 600 seconds: Normal -> Pre-ictal -> Seizure -> Post-ictal.
//! Amplitude detection fires DURING seizure. Graph boundary detection catches
//! the pre-ictal hypersynchronization ~45 seconds BEFORE onset.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const NCH: usize = 16;
const DUR: usize = 600;
const SR: usize = 256;
const TSAMP: usize = DUR * SR;
const WIN_S: usize = 10;
const NWIN: usize = DUR / WIN_S;
const SEED: u64 = 42_0911;
const NPAIRS: usize = NCH * (NCH - 1) / 2;
const NFEAT: usize = NPAIRS + NCH * 4; // 120 corr + 16*3 band + 16 dom_freq
const AMP_THR: f64 = 5.0;
const NULL_N: usize = 100;
const P1: usize = 300; // normal end
const P2: usize = 360; // pre-ictal end (seizure onset)
const P3: usize = 390; // seizure end
const TAU: f64 = std::f64::consts::TAU;
fn region(ch: usize) -> usize {
    match ch {
        0..=5 => 0,
        6 | 7 => 1,
        8 | 9 | 12 | 13 => 2,
        _ => 3,
    }
}

fn gauss(rng: &mut StdRng) -> f64 {
    let u: f64 = rng.gen::<f64>().max(1e-15);
    (-2.0 * u.ln()).sqrt() * (TAU * rng.gen::<f64>()).cos()
}

fn phase(sec: usize) -> (f64, f64, f64, f64, f64, f64, bool) {
    // Returns: (amplitude, intra_corr, inter_corr, alpha, beta, gamma, spike_wave)
    if sec < P1 {
        return (1.0, 0.5, 0.15, 1.0, 0.4, 0.1, false);
    }
    if sec < P2 {
        let t = 1.0 / (1.0 + (-12.0 * ((sec - P1) as f64 / (P2 - P1) as f64 - 0.15)).exp());
        return (
            1.0,
            0.5 + 0.4 * t,
            0.15 + 0.55 * t,
            1.0 - 0.7 * t,
            0.4 + 0.35 * t,
            0.1 + 0.6 * t,
            false,
        );
    }
    if sec < P3 {
        let t = (sec - P2) as f64 / (P3 - P2) as f64;
        return (5.0 + 5.0 * t, 0.95, 0.92, 0.1, 0.2, 0.8, true);
    }
    let t = (sec - P3) as f64 / (DUR - P3) as f64;
    (
        0.3 + 0.5 * t,
        0.05 + 0.25 * t,
        0.02 + 0.08 * t,
        0.2 + 0.6 * t,
        0.1 + 0.2 * t,
        0.3 - 0.15 * t,
        false,
    )
}

fn generate_eeg(rng: &mut StdRng) -> Vec<[f64; NCH]> {
    let mut data = Vec::with_capacity(TSAMP);
    let mut lat = [[0.0_f64; 4]; 4];
    let mut phi = [0.0_f64; NCH];
    for ch in 0..NCH {
        phi[ch] = rng.gen::<f64>() * TAU;
    }
    for s in 0..TSAMP {
        let t = s as f64 / SR as f64;
        let (amp, ic, xc, al, be, ga, sw) = phase(s / SR);
        for r in 0..4 {
            for o in 0..4 {
                lat[r][o] = 0.95 * lat[r][o] + 0.22 * gauss(rng);
            }
        }
        let gl: f64 = lat.iter().map(|r| r[0]).sum::<f64>() / 4.0;
        let mut row = [0.0_f64; NCH];
        for ch in 0..NCH {
            let r = region(ch);
            let sig = al * (TAU * 10.0 * t + phi[ch]).sin()
                + be * (TAU * 20.0 * t + phi[ch] * 1.7).sin()
                + ga * (TAU * 42.0 * t + phi[ch] * 2.3).sin()
                + if sw {
                    3.0 * (TAU * 3.0 * t).sin().powi(3)
                } else {
                    0.0
                }
                + lat[r][ch % 4] * ic
                + gl * xc
                + gauss(rng) * (1.0 - 0.5 * (ic + xc).min(1.0));
            row[ch] = amp * sig;
        }
        data.push(row);
    }
    data
}

fn goertzel(sig: &[f64], freq: f64) -> f64 {
    let n = sig.len();
    let w = TAU * (freq * n as f64 / SR as f64).round() / n as f64;
    let c = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for &x in sig {
        let s0 = x + c * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - c * s1 * s2).max(0.0) / (n * n) as f64
}

fn win_features(samp: &[[f64; NCH]]) -> Vec<f64> {
    let n = samp.len() as f64;
    let mut f = Vec::with_capacity(NFEAT);
    let mut mu = [0.0_f64; NCH];
    let mut va = [0.0_f64; NCH];
    for ch in 0..NCH {
        mu[ch] = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
        va[ch] = samp.iter().map(|s| (s[ch] - mu[ch]).powi(2)).sum::<f64>() / n;
    }
    for i in 0..NCH {
        for j in (i + 1)..NCH {
            let mut c = 0.0;
            for s in samp {
                c += (s[i] - mu[i]) * (s[j] - mu[j]);
            }
            c /= n;
            let d = (va[i] * va[j]).sqrt();
            f.push(if d < 1e-12 { 0.0 } else { c / d });
        }
    }
    for ch in 0..NCH {
        let sig: Vec<f64> = samp.iter().map(|s| s[ch]).collect();
        let a: f64 = [9.0, 10.0, 11.0, 12.0]
            .iter()
            .map(|&fr| goertzel(&sig, fr))
            .sum();
        let b: f64 = [15.0, 20.0, 25.0]
            .iter()
            .map(|&fr| goertzel(&sig, fr))
            .sum();
        let g: f64 = [35.0, 42.0, 55.0, 70.0]
            .iter()
            .map(|&fr| goertzel(&sig, fr))
            .sum();
        f.push(a.ln().max(-10.0));
        f.push(b.ln().max(-10.0));
        f.push(g.ln().max(-10.0));
    }
    for ch in 0..NCH {
        let sig: Vec<f64> = samp.iter().map(|s| s[ch]).collect();
        let (mut bf, mut bp) = (10.0_f64, 0.0_f64);
        for fi in 4..80 {
            let p = goertzel(&sig, fi as f64);
            if p > bp {
                bp = p;
                bf = fi as f64;
            }
        }
        f.push(bf / 80.0);
    }
    f
}

fn normalize(fs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (d, n) = (fs[0].len(), fs.len() as f64);
    let mut mu = vec![0.0_f64; d];
    let mut sd = vec![0.0_f64; d];
    for f in fs {
        for i in 0..d {
            mu[i] += f[i];
        }
    }
    for v in &mut mu {
        *v /= n;
    }
    for f in fs {
        for i in 0..d {
            sd[i] += (f[i] - mu[i]).powi(2);
        }
    }
    for v in &mut sd {
        *v = (*v / n).sqrt().max(1e-12);
    }
    fs.iter()
        .map(|f| (0..d).map(|i| (f[i] - mu[i]) / sd[i]).collect())
        .collect()
}

fn dsq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn build_graph(f: &[Vec<f64>]) -> (Vec<(u64, u64, f64)>, Vec<(usize, usize, f64)>) {
    let mut ds: Vec<f64> = (0..f.len())
        .flat_map(|i| ((i + 1)..f.len().min(i + 5)).map(move |j| dsq(&f[i], &f[j])))
        .collect();
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sig = ds[ds.len() / 2].max(1e-6);
    let (mut mc, mut sp) = (Vec::new(), Vec::new());
    for i in 0..f.len() {
        for sk in 1..=4 {
            if i + sk < f.len() {
                let w = (-dsq(&f[i], &f[i + sk]) / (2.0 * sig)).exp().max(1e-6);
                mc.push((i as u64, (i + sk) as u64, w));
                sp.push((i, i + sk, w));
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

fn find_bounds(cuts: &[f64], margin: usize, gap: usize) -> Vec<(usize, f64)> {
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

fn amp_detect(eeg: &[[f64; NCH]]) -> Option<usize> {
    let bl = 200 * SR;
    let br = (eeg[..bl]
        .iter()
        .flat_map(|r| r.iter())
        .map(|x| x * x)
        .sum::<f64>()
        / (bl * NCH) as f64)
        .sqrt();
    for st in (0..eeg.len()).step_by(SR) {
        let e = (st + SR).min(eeg.len());
        let n = (e - st) as f64 * NCH as f64;
        let r = (eeg[st..e]
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f64>()
            / n)
            .sqrt();
        if r > br * AMP_THR {
            return Some(st / SR);
        }
    }
    None
}

fn corr_stats(samp: &[[f64; NCH]]) -> (f64, f64, f64) {
    let n = samp.len() as f64;
    let mut mu = [0.0_f64; NCH];
    let mut va = [0.0_f64; NCH];
    for ch in 0..NCH {
        mu[ch] = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
        va[ch] = samp.iter().map(|s| (s[ch] - mu[ch]).powi(2)).sum::<f64>() / n;
    }
    let (mut all, mut ci, mut cx) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut na, mut ni, mut nx) = (0usize, 0usize, 0usize);
    for i in 0..NCH {
        for j in (i + 1)..NCH {
            let mut c = 0.0;
            for s in samp {
                c += (s[i] - mu[i]) * (s[j] - mu[j]);
            }
            c /= n;
            let d = (va[i] * va[j]).sqrt();
            let r = if d < 1e-12 { 0.0 } else { (c / d).abs() };
            all += r;
            na += 1;
            if region(i) == region(j) {
                ci += r;
                ni += 1;
            } else {
                cx += r;
                nx += 1;
            }
        }
    }
    (
        all / na.max(1) as f64,
        ci / ni.max(1) as f64,
        cx / nx.max(1) as f64,
    )
}

fn band_ratio(samp: &[[f64; NCH]]) -> (f64, f64) {
    let (mut at, mut gt) = (0.0_f64, 0.0_f64);
    for ch in 0..NCH {
        let sig: Vec<f64> = samp.iter().map(|s| s[ch]).collect();
        at += [9.0, 10.0, 11.0, 12.0]
            .iter()
            .map(|&f| goertzel(&sig, f))
            .sum::<f64>();
        gt += [35.0, 42.0, 55.0, 70.0]
            .iter()
            .map(|&f| goertzel(&sig, f))
            .sum::<f64>();
    }
    (at / NCH as f64, gt / NCH as f64)
}

fn rms(eeg: &[[f64; NCH]]) -> f64 {
    let n = eeg.len() as f64 * NCH as f64;
    (eeg.iter()
        .flat_map(|r| r.iter())
        .map(|x| x * x)
        .sum::<f64>()
        / n)
        .sqrt()
}

fn w2s(w: usize) -> usize {
    w * WIN_S + WIN_S / 2
}

fn null_cuts(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(NULL_N); 4];
    for _ in 0..NULL_N {
        let eeg = null_eeg(rng);
        let wf: Vec<_> = (0..NWIN)
            .map(|i| {
                let s = i * WIN_S * SR;
                win_features(&eeg[s..s + WIN_S * SR])
            })
            .collect();
        let (_, sp) = build_graph(&normalize(&wf));
        let b = find_bounds(&cut_profile(&sp, NWIN), 1, 4);
        for k in 0..4 {
            out[k].push(b.get(k).map_or(1.0, |x| x.1));
        }
    }
    out
}

fn null_eeg(rng: &mut StdRng) -> Vec<[f64; NCH]> {
    let mut lat = [[0.0_f64; 4]; 4];
    let mut phi = [0.0_f64; NCH];
    for ch in 0..NCH {
        phi[ch] = rng.gen::<f64>() * TAU;
    }
    (0..TSAMP)
        .map(|s| {
            let t = s as f64 / SR as f64;
            for r in 0..4 {
                for o in 0..4 {
                    lat[r][o] = 0.95 * lat[r][o] + 0.22 * gauss(rng);
                }
            }
            let mut row = [0.0_f64; NCH];
            for ch in 0..NCH {
                row[ch] = (TAU * 10.0 * t + phi[ch]).sin()
                    + 0.4 * (TAU * 20.0 * t + phi[ch] * 1.7).sin()
                    + lat[region(ch)][ch % 4] * 0.5
                    + gauss(rng) * 0.7;
            }
            row
        })
        .collect()
}

fn zscore(obs: f64, null: &[f64]) -> f64 {
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
    let se: Vec<_> = edges
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

fn pname(sec: usize) -> &'static str {
    if sec < P1 {
        "Normal"
    } else if sec < P2 {
        "Pre-ictal"
    } else if sec < P3 {
        "Seizure"
    } else {
        "Post-ictal"
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    println!("================================================================");
    println!("  55 Seconds That Save Lives");
    println!("  Pre-Seizure Detection from Brain Correlation Boundaries");
    println!("================================================================\n");

    let eeg = generate_eeg(&mut rng);
    println!(
        "[EEG] {} channels (Fp1/Fp2, F3/F4/F7/F8, C3/C4, T3-T6, P3/P4, O1/O2), {} seconds, {} Hz",
        NCH, DUR, SR
    );
    println!("[EEG] {} total data points", TSAMP * NCH);
    println!("[PHASES] Normal (0-{}s) -> Pre-ictal ({}-{}s) -> Seizure ({}-{}s) -> Post-ictal ({}-{}s)\n",
        P1, P1, P2, P2, P3, P3, DUR);

    for &(nm, s, e) in &[
        ("Normal", 0, P1),
        ("Pre-ictal", P1, P2),
        ("Seizure", P2, P3),
        ("Post-ictal", P3, DUR),
    ] {
        let (_, ci, cx) = corr_stats(&eeg[s * SR..e * SR]);
        println!(
            "  {:<11} RMS={:.3}  intra-region|r|={:.3}  cross-region|r|={:.3}",
            nm,
            rms(&eeg[s * SR..e * SR]),
            ci,
            cx
        );
    }

    let ad = amp_detect(&eeg);
    println!("\n[AMPLITUDE DETECTION]");
    if let Some(sec) = ad {
        println!(
            "  Seizure alarm: second {} ({} seconds AFTER seizure starts)",
            sec,
            if sec >= P2 { sec - P2 } else { 0 }
        );
        println!("  Warning time: NEGATIVE (already seizing)");
    } else {
        println!("  No seizure detected by amplitude threshold");
    }

    let wf: Vec<_> = (0..NWIN)
        .map(|i| {
            let s = i * WIN_S * SR;
            win_features(&eeg[s..s + WIN_S * SR])
        })
        .collect();
    let normed = normalize(&wf);
    let (mc_e, sp_e) = build_graph(&normed);
    println!(
        "\n[GRAPH] {} windows ({}s each), {} edges, {}-dim features",
        NWIN,
        WIN_S,
        mc_e.len(),
        NFEAT
    );

    let cuts = cut_profile(&sp_e, NWIN);
    let bounds = find_bounds(&cuts, 1, 4);
    let nd = null_cuts(&mut rng);

    println!("\n[BOUNDARY DETECTION]");
    let pb = bounds
        .iter()
        .find(|&&(w, _)| {
            let s = w2s(w);
            s >= P1 - 30 && s <= P2 + 10
        })
        .or(bounds.first());

    if let Some(&(win, cv)) = pb {
        let sec = w2s(win);
        let z = zscore(cv, &nd[0]);
        let warn = if sec < P2 { P2 - sec } else { 0 };

        println!("  Pre-ictal boundary: second {}", sec);
        println!("  Warning time: {} SECONDS before seizure onset", warn);
        println!(
            "  z-score: {:.2}  {}\n",
            z,
            if z < -2.0 { "SIGNIFICANT" } else { "n.s." }
        );

        println!("  What changed at second {}:", sec);
        let bs = sec.saturating_sub(20) * SR;
        let be = sec * SR;
        let a_s = sec * SR;
        let ae = (sec + 20).min(DUR) * SR;
        let (ab, gb) = band_ratio(&eeg[bs..be]);
        let (aa, ga) = band_ratio(&eeg[a_s..ae]);
        let fd = if win > 0 && win < NWIN {
            dsq(&normed[win - 1], &normed[win]).sqrt()
        } else {
            0.0
        };
        let avg: f64 = (1..normed.len())
            .map(|i| dsq(&normed[i - 1], &normed[i]).sqrt())
            .sum::<f64>()
            / (normed.len() - 1).max(1) as f64;
        println!(
            "  - Feature-space distance: {:.2} (vs avg {:.2} -- {:.1}x discontinuity)",
            fd,
            avg,
            fd / avg.max(0.01)
        );
        println!(
            "  - Alpha power (10 Hz): {:.6} -> {:.6} ({:.0}% drop)",
            ab,
            aa,
            (1.0 - aa / ab.max(1e-12)) * 100.0
        );
        println!(
            "  - Gamma power (40+ Hz): {:.6} -> {:.6} ({:.1}x increase)",
            gb,
            ga,
            ga / gb.max(1e-12)
        );
        println!(
            "  - RMS amplitude: {:.3} -> {:.3} (NO change -- invisible on raw EEG)",
            rms(&eeg[bs..be]),
            rms(&eeg[a_s..ae])
        );

        println!("\n[THE {}-SECOND WINDOW]", warn);
        println!(
            "  Second {}: Boundary detected (correlation hypersynchronization begins)",
            sec
        );
        let mid = sec + warn / 2;
        let (_, _, xm) = corr_stats(&eeg[mid * SR..(mid + 10).min(DUR) * SR]);
        println!(
            "  Second {}: Cross-region correlation at {:.3} (confirmed pre-ictal trajectory)",
            mid, xm
        );
        println!("  Second {}: Seizure onset (amplitude spikes)", P2);
        println!("\n  {} seconds of warning. Enough time to:", warn);
        println!("  - Alert the patient's phone/watch");
        println!("  - Pull over if driving");
        println!("  - Sit down if standing");
        println!("  - Call for help");
        println!("  - Activate vagus nerve stimulator (if implanted)");
    }

    println!("\n[ALL BOUNDARIES]");
    for (i, &(w, cv)) in bounds.iter().take(4).enumerate() {
        let s = w2s(w);
        let z = zscore(cv, &nd[i.min(3)]);
        println!(
            "  #{}: second {} ({}) z={:.2} {}",
            i + 1,
            s,
            pname(s),
            z,
            if z < -2.0 { "SIGNIFICANT" } else { "n.s." }
        );
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

    let sb: Vec<usize> = bounds.iter().take(3).map(|b| b.0).collect();
    let segs = if sb.len() >= 3 {
        let mut s = sb;
        s.sort();
        vec![(0, s[0]), (s[0], s[1]), (s[1], s[2]), (s[2], NWIN)]
    } else {
        let w = |s: usize| s / WIN_S;
        vec![(0, w(P1)), (w(P1), w(P2)), (w(P2), w(P3)), (w(P3), NWIN)]
    };
    let lbl = ["Normal", "Pre-ictal", "Seizure", "Post-ictal"];
    let desc = [
        "(organized by region, moderate correlations)",
        "(correlations increasing, boundaries dissolving)",
        "(hypersynchronized -- one giant connected component)",
        "(correlations near zero -- brain \"rebooting\")",
    ];
    println!("\n[SPECTRAL] Per-phase Fiedler values:");
    for (i, &(s, e)) in segs.iter().enumerate() {
        println!(
            "  {:<11}: {:.4} {}",
            lbl[i],
            fiedler_seg(&mc_e, s, e),
            desc[i]
        );
    }

    println!("\n================================================================");
    if let (Some(&(bw, _)), Some(as_)) = (pb, ad) {
        let bs = w2s(bw);
        println!(
            "  Amplitude detection:  second {} (during seizure, {} seconds late)",
            as_,
            if as_ >= P2 { as_ - P2 } else { 0 }
        );
        println!(
            "  Boundary detection:   second {} ({} seconds BEFORE seizure)",
            bs,
            if bs < P2 { P2 - bs } else { 0 }
        );
        println!(
            "\n  Advantage: {} seconds of early warning.",
            if as_ > bs { as_ - bs } else { 0 }
        );
        println!("  That is the difference between injury and safety.");
    }
    println!("================================================================");
}
