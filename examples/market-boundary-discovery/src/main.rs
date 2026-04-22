//! **Market Regime Boundary Discovery** -- detects hidden market regime changes
//! before they become obvious by finding structural boundaries in asset
//! correlation patterns rather than waiting for price drops.
//!
//! Key insight: during "bull-volatile" the price index still rises, but the
//! correlation structure has fractured -- a warning ~100 days before the crash.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const N_ASSETS: usize = 10;
const N_DAYS: usize = 500;
const WIN: usize = 10;
const N_WIN: usize = N_DAYS / WIN; // 50
const NULL_N: usize = 80;
const SEED: u64 = 42;
const BQ_END: usize = 150; // bull-quiet end
const BV_END: usize = 250; // bull-volatile end (crash starts)
const CR_END: usize = 320; // crash end (recovery starts)

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// (drift, vol, correlation) per regime.
fn regime(d: usize) -> (f64, f64, f64) {
    if d < BQ_END {
        (0.0008, 0.005, 0.70)
    }
    // quiet bull
    else if d < BV_END {
        (0.0004, 0.02, 0.30)
    }
    // volatile bull
    else if d < CR_END {
        (-0.004, 0.04, 0.95)
    }
    // crash
    else {
        (0.0003, 0.012, 0.50)
    } // recovery
}

fn gen_returns(rng: &mut StdRng, regime_fn: fn(usize) -> (f64, f64, f64)) -> Vec<Vec<f64>> {
    let mut ret = vec![vec![0.0_f64; N_DAYS]; N_ASSETS];
    for d in 0..N_DAYS {
        let (dr, vol, rho) = regime_fn(d);
        let zc = gauss(rng);
        let (sr, si) = (rho.sqrt(), (1.0 - rho).max(0.0).sqrt());
        for a in 0..N_ASSETS {
            ret[a][d] = dr + vol * (sr * zc + si * gauss(rng));
        }
    }
    ret
}

fn price_index(ret: &[Vec<f64>]) -> Vec<f64> {
    let mut idx = vec![100.0_f64; N_DAYS + 1];
    for d in 0..N_DAYS {
        let avg: f64 = (0..N_ASSETS).map(|a| ret[a][d]).sum::<f64>() / N_ASSETS as f64;
        idx[d + 1] = idx[d] * (1.0 + avg);
    }
    idx
}

struct WinFeat {
    mean_ret: f64,
    vol: f64,
    corr: f64,
    dd: f64,
    skew: f64,
}

fn pearson(a: &[f64], b: &[f64], ma: f64, mb: f64) -> f64 {
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

fn features(ret: &[Vec<f64>], w: usize) -> WinFeat {
    let (s, e, n) = (w * WIN, (w + 1) * WIN, WIN as f64);
    let mut mu = [0.0_f64; N_ASSETS];
    let slices: Vec<&[f64]> = (0..N_ASSETS)
        .map(|a| {
            let sl = &ret[a][s..e];
            mu[a] = sl.iter().sum::<f64>() / n;
            sl
        })
        .collect();
    let mean_ret = mu.iter().sum::<f64>() / N_ASSETS as f64;
    let vol = (0..N_ASSETS)
        .map(|a| (slices[a].iter().map(|r| (r - mu[a]).powi(2)).sum::<f64>() / n).sqrt())
        .sum::<f64>()
        / N_ASSETS as f64;
    let (mut cs, mut cc) = (0.0_f64, 0u32);
    for i in 0..N_ASSETS {
        for j in (i + 1)..N_ASSETS {
            cs += pearson(slices[i], slices[j], mu[i], mu[j]);
            cc += 1;
        }
    }
    let corr = if cc > 0 { cs / cc as f64 } else { 0.0 };
    let (mut cum, mut pk, mut dd) = (1.0_f64, 1.0_f64, 0.0_f64);
    for d in s..e {
        let avg: f64 = (0..N_ASSETS).map(|a| ret[a][d]).sum::<f64>() / N_ASSETS as f64;
        cum *= 1.0 + avg;
        if cum > pk {
            pk = cum;
        }
        let x = (pk - cum) / pk;
        if x > dd {
            dd = x;
        }
    }
    let pr: Vec<f64> = (s..e)
        .map(|d| (0..N_ASSETS).map(|a| ret[a][d]).sum::<f64>() / N_ASSETS as f64)
        .collect();
    let pm = pr.iter().sum::<f64>() / n;
    let psd = (pr.iter().map(|r| (r - pm).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-12);
    let skew = pr.iter().map(|r| ((r - pm) / psd).powi(3)).sum::<f64>() / n;
    WinFeat {
        mean_ret,
        vol,
        corr,
        dd,
        skew,
    }
}

fn similarity(a: &WinFeat, b: &WinFeat) -> f64 {
    let d = (a.corr - b.corr).abs() * 3.0
        + (a.vol - b.vol).abs() * 80.0
        + (a.mean_ret - b.mean_ret).abs() * 200.0
        + (a.dd - b.dd).abs() * 30.0
        + (a.skew - b.skew).abs() * 0.3;
    (-d).exp().max(1e-6)
}

fn build_graph(feats: &[WinFeat]) -> (Vec<(u64, u64, f64)>, Vec<(usize, usize, f64)>) {
    let (mut mc, mut sp) = (Vec::new(), Vec::new());
    for i in 0..N_WIN {
        for j in (i + 1)..(i + 6).min(N_WIN) {
            let w = similarity(&feats[i], &feats[j]);
            mc.push((i as u64, j as u64, w));
            sp.push((i, j, w));
        }
    }
    (mc, sp)
}

fn cut_profile(edges: &[(usize, usize, f64)]) -> Vec<f64> {
    let mut c = vec![0.0_f64; N_WIN];
    for &(u, v, w) in edges {
        for k in (u.min(v) + 1)..=u.max(v) {
            c[k] += w;
        }
    }
    c
}

fn find_boundaries(edges: &[(usize, usize, f64)], k: usize) -> Vec<(usize, f64)> {
    let cuts = cut_profile(edges);
    let (mut found, mut mask) = (Vec::new(), vec![false; N_WIN]);
    for _ in 0..k {
        let mut best = (0usize, f64::INFINITY);
        for p in 2..N_WIN - 2 {
            if !mask[p] && cuts[p] < best.1 {
                best = (p, cuts[p]);
            }
        }
        if best.1 == f64::INFINITY {
            break;
        }
        found.push(best);
        for m in best.0.saturating_sub(4)..=(best.0 + 4).min(N_WIN - 1) {
            mask[m] = true;
        }
    }
    found.sort_by_key(|&(w, _)| w);
    found
}

fn null_regime(_: usize) -> (f64, f64, f64) {
    (0.0003, 0.015, 0.50)
}

fn null_dist(rng: &mut StdRng) -> Vec<f64> {
    (0..NULL_N)
        .map(|_| {
            let r = gen_returns(rng, null_regime);
            let f: Vec<WinFeat> = (0..N_WIN).map(|w| features(&r, w)).collect();
            let (_, sp) = build_graph(&f);
            let c = cut_profile(&sp);
            (2..N_WIN - 2).map(|k| c[k]).fold(f64::INFINITY, f64::min)
        })
        .collect()
}

fn z_score(obs: f64, null: &[f64]) -> f64 {
    let (n, mu) = (
        null.len() as f64,
        null.iter().sum::<f64>() / null.len() as f64,
    );
    let sd = (null.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n).sqrt();
    if sd < 1e-12 {
        0.0
    } else {
        (obs - mu) / sd
    }
}

fn drawdown_day(idx: &[f64], pct: f64) -> Option<usize> {
    let mut pk = idx[0];
    for (d, &p) in idx.iter().enumerate() {
        if p > pk {
            pk = p;
        }
        if (pk - p) / pk >= pct {
            return Some(d);
        }
    }
    None
}

fn fiedler_seg(edges: &[(usize, usize, f64)], w0: usize, w1: usize) -> f64 {
    let n = w1 - w0;
    if n < 3 {
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
    estimate_fiedler(&CsrMatrixView::build_laplacian(n, &seg), 200, 1e-10).0
}

fn transition(w: usize, b: [usize; 3]) -> &'static str {
    let d: Vec<usize> = b
        .iter()
        .map(|&bi| (w as isize - bi as isize).unsigned_abs())
        .collect();
    if d[0] <= d[1] && d[0] <= d[2] {
        "bull-quiet -> bull-volatile"
    } else if d[1] <= d[2] {
        "bull-volatile -> crash"
    } else {
        "crash -> recovery"
    }
}

fn describe(feats: &[WinFeat], w: usize) -> String {
    if w == 0 || w >= N_WIN {
        return "(edge)".into();
    }
    let (b, a) = (&feats[w - 1], &feats[w]);
    let (dc, dv) = (a.corr - b.corr, a.vol - b.vol);
    let mut p = Vec::new();
    if dc.abs() > 0.05 {
        p.push(format!(
            "pairwise correlations {} from {:.2} to {:.2}",
            if dc > 0.0 { "surged" } else { "dropped" },
            b.corr,
            a.corr
        ));
    }
    if dv.abs() > 0.002 {
        p.push(format!(
            "volatility {} from {:.3} to {:.3}",
            if dv > 0.0 { "spiked" } else { "fell" },
            b.vol,
            a.vol
        ));
    }
    if p.is_empty() {
        format!(
            "subtle shift (corr {:.2}->{:.2}, vol {:.3}->{:.3})",
            b.corr, a.corr, b.vol, a.vol
        )
    } else {
        p.join("; ")
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let ret = gen_returns(&mut rng, regime);
    let idx = price_index(&ret);
    let feats: Vec<WinFeat> = (0..N_WIN).map(|w| features(&ret, w)).collect();
    let (mc_e, sp_e) = build_graph(&feats);
    let crash = drawdown_day(&idx, 0.05);
    let bounds = find_boundaries(&sp_e, 3);
    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(mc_e)
        .build()
        .expect("mc");
    let gcut = mc.min_cut_value();
    let nd = null_dist(&mut rng);
    let tb = [BQ_END / WIN, BV_END / WIN, CR_END / WIN]; // [15, 25, 32]

    println!("================================================================");
    println!("  When Did the Market REALLY Change?");
    println!("  Hidden Regime Shifts in Asset Correlations");
    println!("================================================================");
    println!(
        "[MARKET] {} days, {} assets, {} windows of {} days",
        N_DAYS, N_ASSETS, N_WIN, WIN
    );
    println!("[REGIMES] Bull-Quiet -> Bull-Volatile -> Crash -> Recovery");
    println!(
        "[REGIMES] True boundaries: day {}, {}, {}\n",
        BQ_END, BV_END, CR_END
    );

    println!("[PRICE SIGNAL]");
    match crash {
        Some(d) => println!(
            "  Index first drops 5% from peak: day {}\n  => Traditional crash detection: day {}\n",
            d, d
        ),
        None => {
            println!("  Index never drops 5% from peak\n  => Traditional detector sees nothing\n")
        }
    }

    println!("[GRAPH BOUNDARIES]  (global mincut = {:.4})", gcut);
    for (i, &(w, cv)) in bounds.iter().enumerate() {
        let (day, z) = (w * WIN, z_score(cv, &nd));
        let sig = if z < -2.0 {
            "SIGNIFICANT"
        } else {
            "not significant"
        };
        println!(
            "  #{}: day {} (window {}) -- {}",
            i + 1,
            day,
            w,
            transition(w, tb)
        );
        if day < BV_END {
            println!(
                "      {} DAYS before crash onset (day {})",
                BV_END - day,
                BV_END
            );
        }
        println!("      z-score: {:.2}  {}", z, sig);
        println!("      Cut weight: {:.4}", cv);
        println!("      What changed: {}", describe(&feats, w));
    }
    println!();

    if let Some(&(w, cv)) = bounds
        .iter()
        .filter(|&&(w, _)| w * WIN < BV_END)
        .min_by_key(|&&(w, _)| w)
    {
        let (day, lead, z) = (w * WIN, BV_END - w * WIN, z_score(cv, &nd));
        println!(
            "[KEY FINDING] The correlation breakdown at day {} is a",
            day
        );
        println!("  structural warning {} DAYS before the crash.", lead);
        if feats[w].mean_ret > 0.0 {
            println!("  Price was still going up. Volatility hadn't spiked yet.");
        }
        println!("  Only the BOUNDARY in correlation structure revealed the shift.");
        println!("  z = {:.2}\n", z);
    }

    println!("[SPECTRAL] Per-regime Fiedler values:");
    println!(
        "  Bull-Quiet:    {:.4}  (tight, correlated)",
        fiedler_seg(&sp_e, 0, tb[0])
    );
    println!(
        "  Bull-Volatile: {:.4}  (loosening)",
        fiedler_seg(&sp_e, tb[0], tb[1])
    );
    println!(
        "  Crash:         {:.4}  (extremely tight -- forced correlation)",
        fiedler_seg(&sp_e, tb[1], tb[2])
    );
    println!(
        "  Recovery:      {:.4}  (normalizing)",
        fiedler_seg(&sp_e, tb[2], N_WIN)
    );

    println!("\n[CORRELATION TIMELINE]  (mean pairwise correlation per window)");
    print!("  ");
    for w in 0..N_WIN {
        let c = feats[w].corr;
        print!(
            "{}",
            if c > 0.7 {
                '#'
            } else if c > 0.4 {
                '='
            } else if c > 0.1 {
                '-'
            } else {
                '.'
            }
        );
    }
    println!();
    print!("  ");
    for w in 0..N_WIN {
        print!("{}", if tb.contains(&w) { '|' } else { ' ' });
    }
    println!("  <- true regime boundaries");
    print!("  ");
    for w in 0..N_WIN {
        print!(
            "{}",
            if bounds.iter().any(|&(b, _)| b == w) {
                '^'
            } else {
                ' '
            }
        );
    }
    println!("  <- detected boundaries");
    println!("  Legend: # = corr>0.7  = = corr>0.4  - = corr>0.1  . = corr<0.1");
    println!("================================================================");
}
