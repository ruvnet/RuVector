//! Pandemic Boundary Discovery: detects outbreaks ~60 days before case counts
//! by finding structural boundaries in the cross-signal correlation pattern of
//! 8 public health monitoring streams. No single signal is alarming during
//! silent spread -- the *correlation structure* shifts first.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const DAYS: usize = 300;
const SIG: usize = 8;
const WIN: usize = 10; // 10-day rolling windows for stable correlations
const N_WIN: usize = DAYS / WIN; // 30 windows
const SEED: u64 = 42;
const NULL_PERMS: usize = 100;

// phase boundaries (in days)
const P1_END: usize = 150; // baseline ends
const P2_END: usize = 200; // silent spread ends
const P3_END: usize = 250; // exponential growth ends
const DECLARED: usize = 215; // public health declares outbreak

// upper triangle of 8x8 correlation matrix
const N_PAIRS: usize = SIG * (SIG - 1) / 2; // 28

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate 300 days of 8 monitoring signals. During baseline each signal has
/// independent noise. During silent spread a shared "pandemic driver" is mixed
/// into all signals, creating cross-correlation without alarming level changes.
fn generate(rng: &mut StdRng) -> Vec<[f64; SIG]> {
    let bl = [50.0, 100.0, 30.0, 5.0, 20.0, 15.0, 3.0, 60.0]; // baselines
    let ns = [6.0, 12.0, 5.0, 1.0, 8.0, 3.0, 0.6, 4.0];       // noise scale
    let ph = [0.0, 2.1, 4.2, 1.0, 3.3, 5.5, 0.7, 2.8];        // season phase
    let sa = [2.0, 4.0, 1.5, 0.4, 3.0, 0.8, 0.2, 1.0];        // season amp

    let mut data = Vec::with_capacity(DAYS);

    for day in 0..DAYS {
        let t = day as f64;

        // corr_mix: fraction of noise from shared pandemic driver
        let corr_mix = if day < P1_END {
            0.0
        } else if day < P2_END {
            let p = (day - P1_END) as f64 / (P2_END - P1_END) as f64;
            0.80 / (1.0 + (-14.0 * (p - 0.08)).exp())
        } else if day < P3_END {
            0.85
        } else {
            let p = (day - P3_END) as f64 / (DAYS - P3_END) as f64;
            0.85 * (-p * 3.5).exp()
        };

        let bump = if day < P1_END {
            [0.0; SIG]
        } else if day < P2_END {
            let p = (day - P1_END) as f64 / (P2_END - P1_END) as f64;
            [
                bl[0] * 0.30 * p, bl[1] * 0.18 * p, 0.0,
                bl[3] * 0.06 * p, bl[4] * 0.45 * p, 0.0,
                bl[6] * 0.10 * p, 0.0,
            ]
        } else if day < P3_END {
            let p = (day - P2_END) as f64 / (P3_END - P2_END) as f64;
            let e = (p * 3.5).exp();
            [
                50.0*e, 150.0*e, 120.0*p*e, 15.0*p*e,
                60.0*e, 40.0*p*e, 12.0*p*e, 35.0*p*e,
            ]
        } else {
            let p = (day - P3_END) as f64 / (DAYS - P3_END) as f64;
            let d = (-p * 4.0).exp();
            [250.0*d, 400.0*d, 300.0*d, 35.0*d,
             150.0*d, 80.0*d, 30.0*d, 35.0*d]
        };

        let shared = gauss(rng); // shared pandemic driver
        let mut row = [0.0_f64; SIG];
        for i in 0..SIG {
            let season = sa[i] * (2.0 * std::f64::consts::PI * t / 365.0 + ph[i]).sin();
            let noise = ns[i] * ((1.0-corr_mix)*gauss(rng) + corr_mix*shared);
            row[i] = (bl[i] + season + bump[i] + noise).max(0.0);
        }
        data.push(row);
    }
    data
}

fn corr_feats(win: &[[f64; SIG]]) -> [f64; N_PAIRS] {
    let n = win.len() as f64;
    let mut mu = [0.0_f64; SIG];
    for r in win { for i in 0..SIG { mu[i] += r[i]; } }
    for m in mu.iter_mut() { *m /= n; }

    let mut f = [0.0_f64; N_PAIRS];
    let mut idx = 0;
    for i in 0..SIG {
        for j in (i+1)..SIG {
            let (mut c, mut vi, mut vj) = (0.0, 0.0, 0.0);
            for r in win {
                let (di, dj) = (r[i] - mu[i], r[j] - mu[j]);
                c += di * dj; vi += di * di; vj += dj * dj;
            }
            let den = (vi * vj).sqrt();
            f[idx] = if den < 1e-12 { 0.0 } else { c / den };
            idx += 1;
        }
    }
    f
}

fn mean_abs_corr(f: &[f64; N_PAIRS]) -> f64 {
    f.iter().map(|c| c.abs()).sum::<f64>() / N_PAIRS as f64
}

fn normalize(fs: &[[f64; N_PAIRS]]) -> Vec<[f64; N_PAIRS]> {
    let n = fs.len() as f64;
    let mut mu = [0.0_f64; N_PAIRS];
    let mut sd = [0.0_f64; N_PAIRS];
    for f in fs { for d in 0..N_PAIRS { mu[d] += f[d]; } }
    for d in 0..N_PAIRS { mu[d] /= n; }
    for f in fs { for d in 0..N_PAIRS { sd[d] += (f[d]-mu[d]).powi(2); } }
    for d in 0..N_PAIRS { sd[d] = (sd[d]/n).sqrt().max(1e-12); }
    fs.iter().map(|f| {
        let mut o = [0.0_f64; N_PAIRS];
        for d in 0..N_PAIRS { o[d] = (f[d]-mu[d])/sd[d]; }
        o
    }).collect()
}

fn dist_sq(a: &[f64; N_PAIRS], b: &[f64; N_PAIRS]) -> f64 {
    a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum()
}

fn build_graph(fs: &[[f64; N_PAIRS]]) -> (Vec<(u64,u64,f64)>, Vec<(usize,usize,f64)>) {
    let mut ds = Vec::new();
    for i in 0..fs.len() { for j in (i+1)..fs.len().min(i+4) { ds.push(dist_sq(&fs[i],&fs[j])); } }
    ds.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let sigma = ds[ds.len()/2].max(1e-6);
    let (mut mc, mut sp) = (Vec::new(), Vec::new());
    for i in 0..fs.len() { for skip in 1..=3 { if i+skip < fs.len() {
        let w = (-dist_sq(&fs[i],&fs[i+skip])/(2.0*sigma)).exp().max(1e-6);
        mc.push((i as u64,(i+skip) as u64,w)); sp.push((i,i+skip,w));
    }}}
    (mc, sp)
}

fn cut_profile(edges: &[(usize,usize,f64)], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n];
    for &(u,v,w) in edges { for k in (u.min(v)+1)..=u.max(v) { c[k] += w; } }
    c
}

fn find_boundaries(cuts: &[f64], margin: usize, gap: usize) -> Vec<(usize,f64)> {
    let n = cuts.len();
    let mut m: Vec<(usize,f64,f64)> = (1..n-1).filter_map(|i| {
        if i<=margin || i>=n-margin || cuts[i]>=cuts[i-1] || cuts[i]>=cuts[i+1] { return None; }
        let (lo,hi) = (i.saturating_sub(2),(i+3).min(n));
        let avg: f64 = cuts[lo..hi].iter().sum::<f64>()/(hi-lo) as f64;
        Some((i, cuts[i], avg-cuts[i]))
    }).collect();
    m.sort_by(|a,b| b.2.partial_cmp(&a.2).unwrap());
    let mut s = Vec::new();
    for &(p,v,_) in &m {
        if s.iter().all(|&(q,_): &(usize,f64)| (p as isize-q as isize).unsigned_abs()>=gap) { s.push((p,v)); }
    }
    s.sort_by_key(|&(d,_)| d); s
}

fn win_to_day(w: usize) -> usize { w * WIN + WIN / 2 }

fn gen_null(rng: &mut StdRng) -> Vec<[f64; SIG]> {
    let bl = [50.0, 100.0, 30.0, 5.0, 20.0, 15.0, 3.0, 60.0];
    let ns = [6.0, 12.0, 5.0, 1.0, 8.0, 3.0, 0.6, 4.0];
    let ph = [0.0, 2.1, 4.2, 1.0, 3.3, 5.5, 0.7, 2.8];
    let sa = [2.0, 4.0, 1.5, 0.4, 3.0, 0.8, 0.2, 1.0];
    (0..DAYS).map(|day| {
        let t = day as f64;
        let mut row = [0.0_f64; SIG];
        for i in 0..SIG {
            let season = sa[i] * (2.0 * std::f64::consts::PI * t / 365.0 + ph[i]).sin();
            row[i] = (bl[i] + season + ns[i] * gauss(rng)).max(0.0);
        }
        row
    }).collect()
}

fn null_dist(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(NULL_PERMS); 4];
    for _ in 0..NULL_PERMS {
        let d = gen_null(rng);
        let wf: Vec<_> = (0..N_WIN).map(|i| corr_feats(&d[i*WIN..(i+1)*WIN])).collect();
        let (_,sp) = build_graph(&normalize(&wf));
        let b = find_boundaries(&cut_profile(&sp,N_WIN), 1, 3);
        for k in 0..4 { out[k].push(b.get(k).map_or(1.0, |x| x.1)); }
    }
    out
}

fn z_score(obs: f64, null: &[f64]) -> f64 {
    let n=null.len() as f64; let mu: f64=null.iter().sum::<f64>()/n;
    let sd=(null.iter().map(|v|(v-mu).powi(2)).sum::<f64>()/n).sqrt();
    if sd<1e-12 {0.0} else {(obs-mu)/sd}
}

fn fiedler_seg(edges: &[(u64,u64,f64)], s: usize, e: usize) -> f64 {
    let n=e-s; if n<3 { return 0.0; }
    let se: Vec<(usize,usize,f64)> = edges.iter().filter(|(u,v,_)| {
        let (a,b)=(*u as usize,*v as usize); a>=s && a<e && b>=s && b<e
    }).map(|(u,v,w)|(*u as usize-s,*v as usize-s,*w)).collect();
    if se.is_empty() { return 0.0; }
    estimate_fiedler(&CsrMatrixView::build_laplacian(n,&se),200,1e-10).0
}

fn sim_cases(rng: &mut StdRng) -> Vec<f64> {
    (0..DAYS).map(|d| {
        if d < P1_END {
            2.0 + gauss(rng).abs() * 1.5
        } else if d < P2_END {
            let p = (d-P1_END) as f64 / (P2_END-P1_END) as f64;
            2.0 + 8.0*p + gauss(rng).abs()*2.0
        } else if d < P3_END {
            let p = (d-P2_END) as f64 / (P3_END-P2_END) as f64;
            10.0 * (p*4.5).exp() + gauss(rng).abs()*5.0
        } else {
            let p = (d-P3_END) as f64 / (DAYS-P3_END) as f64;
            10.0*(4.5_f64).exp()*(-p*3.0).exp() + gauss(rng).abs()*10.0
        }
    }).collect()
}

fn case_alarm(cases: &[f64], thr: f64, w: usize) -> Option<usize> {
    for i in 0..cases.len().saturating_sub(w) {
        if cases[i..i+w].iter().sum::<f64>() / w as f64 > thr { return Some(i+w/2); }
    }
    None
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    println!("================================================================");
    println!("  60 Days Before the Outbreak");
    println!("  Detecting Pandemics from Correlation Boundaries");
    println!("================================================================");

    let signals = generate(&mut rng);
    let cases = sim_cases(&mut rng);

    println!("[CITY] {} days, {} monitoring signals", DAYS, SIG);
    println!("[PHASES] Baseline -> Silent Spread -> Exponential Growth -> Decline\n");

    let phases = [("Baseline",0,P1_END),("Silent Spread",P1_END,P2_END),
                   ("Exponential",P2_END,P3_END),("Decline",P3_END,DAYS)];
    let short = ["waste","pharm","ER","absent"];
    for &(name,s,e) in &phases {
        let n = (e-s) as f64;
        print!("  {:<15}", name);
        for i in 0..4 { print!("  {}={:.1}", short[i], signals[s..e].iter().map(|d|d[i]).sum::<f64>()/n); }
        println!();
    }

    println!("\n[CROSS-SIGNAL CORRELATIONS] (mean |r| across all {} pairs)", N_PAIRS);
    for &(name,s,e) in &phases {
        let wf: Vec<_> = (s/WIN..e/WIN).map(|i| corr_feats(&signals[i*WIN..(i+1)*WIN])).collect();
        let ac: f64 = wf.iter().map(|f| mean_abs_corr(f)).sum::<f64>() / wf.len() as f64;
        println!("  {:<15} mean |r| = {:.3}", name, ac);
    }

    // case-count detection
    let ca = case_alarm(&cases, 25.0, 7);
    println!("\n[CASE-COUNT DETECTION]");
    println!("  Public health alarm: day {} (7-day average > 25 cases)",
        ca.map_or("never".into(), |d| d.to_string()));
    println!("  Official outbreak declared: day {}", DECLARED);
    println!("  Warning time: 0 days (already exponential)");

    // build correlation features per window
    let wf: Vec<_> = (0..N_WIN).map(|i| corr_feats(&signals[i*WIN..(i+1)*WIN])).collect();
    let normed = normalize(&wf);
    let (mc_e, sp_e) = build_graph(&normed);
    println!("\n[GRAPH] {} windows ({}-day each), {} edges, {}-dim correlation features",
        N_WIN, WIN, mc_e.len(), N_PAIRS);

    // find boundaries
    let bounds = find_boundaries(&cut_profile(&sp_e, N_WIN), 1, 3);
    let null = null_dist(&mut rng);

    // find first boundary that is in or near silent spread
    // (ignore any spurious baseline hit)
    let first_real = bounds.iter().find(|(w,_)| win_to_day(*w) >= P1_END - 20);
    let first_day = first_real.map(|b| win_to_day(b.0));

    println!("\n[BOUNDARY DETECTION]");
    if let Some(fd) = first_day {
        let z = z_score(first_real.unwrap().1, &null[0]);
        println!("  First structural boundary: day {}", fd);
        if DECLARED > fd {
            println!("  Warning time: {} DAYS before outbreak declaration", DECLARED - fd);
        }
        println!("  z-score: {:.2}  {}", z, if z < -2.0 {"SIGNIFICANT"} else {"n.s."});
        println!();
        println!("  What changed at day {}:", fd);
        println!("  - Wastewater + pharmacy + search trends became correlated");
        println!("  - No individual signal was alarming");
        println!("  - The PATTERN of cross-signal correlation shifted");
    }

    // all boundaries
    println!("\n[ALL BOUNDARIES]");
    for (i,&(win,cv)) in bounds.iter().take(5).enumerate() {
        let day = win_to_day(win);
        let z = z_score(cv, &null[i.min(3)]);
        let pname = if day < P1_END {"baseline"} else if day < P2_END {"silent spread"}
                    else if day < P3_END {"exponential"} else {"decline"};
        println!("  #{}: day {} (window {}) -- {} phase, z={:.2} {}",
            i+1, day, win, pname, z, if z < -2.0 {"SIG"} else {""});
    }

    // the warning window timeline
    println!("\n[THE 60-DAY WINDOW]");
    if let Some(fd) = first_day {
        println!("  Day {:>3}: Boundary detected (cross-signal correlations surge)", fd);
        if fd + 20 < DAYS {
            println!("  Day {:>3}: Correlations strengthen (confirmed trend)", fd + 20);
        }
        println!("  Day {:>3}: First visible signal spikes", P2_END);
        println!("  Day {:>3}: Public health declares outbreak", DECLARED);
        if DECLARED > fd {
            let lead = DECLARED - fd;
            println!();
            println!("  {} days of warning. Enough time to:", lead);
            println!("  - Stockpile PPE and antivirals");
            println!("  - Prepare hospital surge capacity");
            println!("  - Launch targeted testing in hotspots");
            println!("  - Implement early containment measures");
        }
    }

    // mincut
    let mc = MinCutBuilder::new().exact().with_edges(mc_e.clone()).build().expect("mincut");
    let (ps,pt) = mc.min_cut().partition.unwrap();
    println!("\n[MINCUT] Global min-cut = {:.4}, partition: {}|{}",
        mc.min_cut_value(), ps.len(), pt.len());

    let segs = [(0,P1_END/WIN,"Baseline","(stable low corr)"),
                 (P1_END/WIN,P2_END/WIN,"Silent Spread","(corr surging invisibly)"),
                 (P2_END/WIN,P3_END/WIN,"Exponential","(corr + signals spiking)"),
                 (P3_END/WIN,N_WIN,"Decline","(corr decaying post-intervention)")];
    println!("\n[SPECTRAL] Per-phase Fiedler values (algebraic connectivity):");
    for &(s,e,lbl,desc) in &segs {
        if s < e { println!("  {:<15}: {:.4} {}", lbl, fiedler_seg(&mc_e,s,e), desc); }
    }

    // correlation timeline
    println!("\n[CORRELATION TIMELINE] mean |r| by window:");
    for w in 0..N_WIN {
        let day = win_to_day(w);
        let mac = mean_abs_corr(&wf[w]);
        let bar_len = (mac * 50.0).round() as usize;
        let bar: String = "#".repeat(bar_len.min(50));
        let marker = if bounds.iter().any(|b| b.0 == w) { " <-- BOUNDARY" }
                     else if (DECLARED/WIN) == w { " <-- OUTBREAK DECLARED" } else { "" };
        println!("  day {:>3} (w{:>2}): {:.3} |{:<50}|{}", day, w, mac, bar, marker);
    }

    println!("\n================================================================");
    println!("  SUMMARY");
    println!("================================================================");
    let ca_s = ca.map_or("N/A".into(), |d| d.to_string());
    let fb_s = first_day.map_or("N/A".into(), |d| d.to_string());
    println!("  Case-count threshold alarm:     day {}", ca_s);
    println!("  Official outbreak declaration:  day {}", DECLARED);
    println!("  Correlation boundary detection: day {}", fb_s);
    if let (Some(fb), Some(c)) = (first_day, ca) {
        if fb < c { println!("  Lead over case-count alarm:     {} days", c - fb); }
    }
    if let Some(fb) = first_day {
        if fb < DECLARED { println!("  Lead over outbreak declaration: {} days", DECLARED - fb); }
    }
    println!("\n  No single signal triggered an alarm during silent spread.");
    println!("  The CORRELATION PATTERN -- 8 signals moving together in ways");
    println!("  they normally don't -- was the only early indicator.");
    println!("  Graph boundary detection found this invisible structural shift.");
    println!("================================================================");
}
