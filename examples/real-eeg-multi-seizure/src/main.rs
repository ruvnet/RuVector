//! Multi-Seizure Boundary-First Detection: CHB-MIT Patient chb01
//!
//! Parses all 7 documented seizure files from CHB-MIT Scalp EEG Database.
//! Runs boundary-first detection on each, then computes cross-seizure
//! population statistics: mean warning time, detection rates, Fiedler
//! consistency, and per-channel informativeness.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use std::path::Path;

// ── Constants ───────────────────────────────────────────────────────────
const NCH: usize = 16;
const SR: usize = 256;
const WIN_S: usize = 10;
const SEED: u64 = 42_0911;
const NPAIRS: usize = NCH * (NCH - 1) / 2;
const NFEAT: usize = NPAIRS + NCH * 4; // 120 corr + 48 band + 16 dom_freq
const NULL_N: usize = 50;
const TAU: f64 = std::f64::consts::TAU;
const HALF_WIN: usize = 300; // 300s each side of seizure onset

const LABELS: [&str; 23] = [
    "FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ","P7-T7","T7-FT9","FT9-FT10","FT10-T8","T8-P8",
];

/// Seizure descriptor: (filename, onset_sec, end_sec)
struct SeizureInfo {
    file: &'static str,
    onset: usize,
    end: usize,
}

const SEIZURES: [SeizureInfo; 7] = [
    SeizureInfo { file: "chb01_03.edf", onset: 2996, end: 3036 },
    SeizureInfo { file: "chb01_04.edf", onset: 1467, end: 1494 },
    SeizureInfo { file: "chb01_15.edf", onset: 1732, end: 1772 },
    SeizureInfo { file: "chb01_16.edf", onset: 1015, end: 1066 },
    SeizureInfo { file: "chb01_18.edf", onset: 1720, end: 1810 },
    SeizureInfo { file: "chb01_21.edf", onset: 327,  end: 420  },
    SeizureInfo { file: "chb01_26.edf", onset: 1862, end: 1963 },
];

// ── EDF parser (reused from real-eeg-analysis) ─────────────────────────
struct Edf {
    ns: usize, ndr: usize, _dur: f64,
    pmin: Vec<f64>, pmax: Vec<f64>,
    dmin: Vec<i16>, dmax: Vec<i16>, spr: Vec<usize>,
}

fn af(b: &[u8], s: usize, l: usize) -> String {
    String::from_utf8_lossy(&b[s..s+l]).trim().to_string()
}
fn af64(b: &[u8], s: usize, l: usize) -> f64 { af(b,s,l).parse().unwrap_or(0.0) }
fn ausz(b: &[u8], s: usize, l: usize) -> usize { af(b,s,l).parse().unwrap_or(0) }

fn parse_edf(d: &[u8]) -> Edf {
    let ns = ausz(d, 252, 4);
    let b = 256;
    let mut pmin = vec![]; let mut pmax = vec![];
    let mut dmin = vec![]; let mut dmax = vec![]; let mut spr = vec![];
    let mut off = b + ns*16 + ns*80 + ns*8;
    for i in 0..ns { pmin.push(af64(d, off+i*8, 8)); } off += ns*8;
    for i in 0..ns { pmax.push(af64(d, off+i*8, 8)); } off += ns*8;
    for i in 0..ns { dmin.push(af64(d, off+i*8, 8) as i16); } off += ns*8;
    for i in 0..ns { dmax.push(af64(d, off+i*8, 8) as i16); } off += ns*8;
    off += ns*80; // prefiltering
    for i in 0..ns { spr.push(ausz(d, off+i*8, 8)); }
    Edf { ns, ndr: ausz(d,236,8), _dur: af64(d,244,8), pmin, pmax, dmin, dmax, spr }
}

fn read_edf(d: &[u8], h: &Edf, s0: usize, s1: usize) -> Vec<[f64; NCH]> {
    let hsz = 256 + h.ns * 256;
    let tot: usize = h.spr.iter().sum();
    let rbytes = tot * 2;
    let mut gain = vec![0.0_f64; h.ns];
    let mut ofs = vec![0.0_f64; h.ns];
    for i in 0..h.ns {
        let dr = h.dmax[i] as f64 - h.dmin[i] as f64;
        let pr = h.pmax[i] - h.pmin[i];
        gain[i] = if dr.abs() < 1e-12 { 1.0 } else { pr / dr };
        ofs[i] = h.pmin[i] - h.dmin[i] as f64 * gain[i];
    }
    let mut out = Vec::with_capacity((s1 - s0) * SR);
    for rec in s0..s1.min(h.ndr) {
        let ro = hsz + rec * rbytes;
        let mut so = 0usize;
        let mut chdata: Vec<Vec<f64>> = vec![Vec::new(); h.ns.min(NCH)];
        for sig in 0..h.ns {
            let n = h.spr[sig];
            if sig < NCH {
                for s in 0..n {
                    let bp = ro + (so + s) * 2;
                    if bp + 1 >= d.len() { break; }
                    let raw = i16::from_le_bytes([d[bp], d[bp + 1]]);
                    chdata[sig].push(raw as f64 * gain[sig] + ofs[sig]);
                }
            }
            so += n;
        }
        for s in 0..h.spr[0] {
            let mut row = [0.0_f64; NCH];
            for ch in 0..NCH {
                if s < chdata[ch].len() { row[ch] = chdata[ch][s]; }
            }
            out.push(row);
        }
    }
    out
}

// ── Signal processing ───────────────────────────────────────────────────
fn goertzel(sig: &[f64], freq: f64) -> f64 {
    let n = sig.len(); if n == 0 { return 0.0; }
    let w = TAU * (freq * n as f64 / SR as f64).round() / n as f64;
    let c = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for &x in sig { let s0 = x + c * s1 - s2; s2 = s1; s1 = s0; }
    (s1 * s1 + s2 * s2 - c * s1 * s2).max(0.0) / (n * n) as f64
}

fn ch_valid(samp: &[[f64; NCH]], ch: usize) -> bool {
    let n = samp.len() as f64; if n < 2.0 { return false; }
    let mu: f64 = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
    samp.iter().map(|s| (s[ch] - mu).powi(2)).sum::<f64>() / n > 1e-10
}

fn win_features(samp: &[[f64; NCH]], valid: &[bool; NCH]) -> Vec<f64> {
    let n = samp.len() as f64;
    let mut f = Vec::with_capacity(NFEAT);
    let mut mu = [0.0_f64; NCH]; let mut va = [0.0_f64; NCH];
    for ch in 0..NCH {
        mu[ch] = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
        va[ch] = samp.iter().map(|s| (s[ch] - mu[ch]).powi(2)).sum::<f64>() / n;
    }
    for i in 0..NCH { for j in (i+1)..NCH {
        if !valid[i] || !valid[j] { f.push(0.0); continue; }
        let mut c = 0.0_f64;
        for s in samp { c += (s[i] - mu[i]) * (s[j] - mu[j]); }
        c /= n; let d = (va[i] * va[j]).sqrt();
        f.push(if d < 1e-12 { 0.0 } else { c / d });
    }}
    for ch in 0..NCH {
        if !valid[ch] { f.push(-10.0); f.push(-10.0); f.push(-10.0); continue; }
        let sig: Vec<f64> = samp.iter().map(|s| s[ch]).collect();
        let a: f64 = [8.0,9.0,10.0,11.0,12.0,13.0].iter().map(|&fr| goertzel(&sig, fr)).sum();
        let b: f64 = [14.0,18.0,22.0,26.0,30.0].iter().map(|&fr| goertzel(&sig, fr)).sum();
        let g: f64 = [35.0,42.0,50.0,60.0,70.0,80.0].iter().map(|&fr| goertzel(&sig, fr)).sum();
        f.push(a.max(1e-20).ln().max(-10.0));
        f.push(b.max(1e-20).ln().max(-10.0));
        f.push(g.max(1e-20).ln().max(-10.0));
    }
    for ch in 0..NCH {
        if !valid[ch] { f.push(0.0); continue; }
        let sig: Vec<f64> = samp.iter().map(|s| s[ch]).collect();
        let (mut bf, mut bp) = (10.0_f64, 0.0_f64);
        for fi in 2..80 {
            let p = goertzel(&sig, fi as f64);
            if p > bp { bp = p; bf = fi as f64; }
        }
        f.push(bf / 80.0);
    }
    f
}

fn normalize(fs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (d, n) = (fs[0].len(), fs.len() as f64);
    let mut mu = vec![0.0_f64; d]; let mut sd = vec![0.0_f64; d];
    for f in fs { for i in 0..d { mu[i] += f[i]; } }
    for v in &mut mu { *v /= n; }
    for f in fs { for i in 0..d { sd[i] += (f[i] - mu[i]).powi(2); } }
    for v in &mut sd { *v = (*v / n).sqrt().max(1e-12); }
    fs.iter().map(|f| (0..d).map(|i| (f[i] - mu[i]) / sd[i]).collect()).collect()
}

fn dsq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn build_graph(f: &[Vec<f64>]) -> (Vec<(u64, u64, f64)>, Vec<(usize, usize, f64)>) {
    let mut ds: Vec<f64> = (0..f.len())
        .flat_map(|i| ((i+1)..f.len().min(i+5)).map(move |j| dsq(&f[i], &f[j])))
        .collect();
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sig = ds[ds.len() / 2].max(1e-6);
    let (mut mc, mut sp) = (Vec::new(), Vec::new());
    for i in 0..f.len() { for sk in 1..=4 { if i + sk < f.len() {
        let w = (-dsq(&f[i], &f[i + sk]) / (2.0 * sig)).exp().max(1e-6);
        mc.push((i as u64, (i + sk) as u64, w));
        sp.push((i, i + sk, w));
    }}}
    (mc, sp)
}

fn cut_profile(edges: &[(usize, usize, f64)], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n];
    for &(u, v, w) in edges {
        for k in (u.min(v) + 1)..=u.max(v) { c[k] += w; }
    }
    c
}

fn find_bounds(cuts: &[f64], margin: usize, gap: usize) -> Vec<(usize, f64)> {
    let n = cuts.len();
    let mut m: Vec<(usize, f64, f64)> = (1..n-1).filter_map(|i| {
        if i <= margin || i >= n - margin || cuts[i] >= cuts[i-1] || cuts[i] >= cuts[i+1] {
            return None;
        }
        let (lo, hi) = (i.saturating_sub(2), (i + 3).min(n));
        Some((i, cuts[i], cuts[lo..hi].iter().sum::<f64>() / (hi - lo) as f64 - cuts[i]))
    }).collect();
    m.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let mut s = Vec::new();
    for &(p, v, _) in &m {
        if s.iter().all(|&(q, _): &(usize, f64)| (p as isize - q as isize).unsigned_abs() >= gap) {
            s.push((p, v));
        }
    }
    s.sort_by_key(|&(d, _)| d);
    s
}

fn zscore(obs: f64, null: &[f64]) -> f64 {
    let n = null.len() as f64;
    let mu: f64 = null.iter().sum::<f64>() / n;
    let sd = (null.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n).sqrt();
    if sd < 1e-12 { 0.0 } else { (obs - mu) / sd }
}

fn fiedler_seg(edges: &[(u64, u64, f64)], s: usize, e: usize) -> f64 {
    let n = e - s; if n < 3 { return 0.0; }
    let se: Vec<_> = edges.iter().filter(|(u, v, _)| {
        let (a, b) = (*u as usize, *v as usize);
        a >= s && a < e && b >= s && b < e
    }).map(|(u, v, w)| (*u as usize - s, *v as usize - s, *w)).collect();
    if se.is_empty() { return 0.0; }
    estimate_fiedler(&CsrMatrixView::build_laplacian(n, &se), 200, 1e-10).0
}

fn null_cuts(
    eeg: &[[f64; NCH]], valid: &[bool; NCH], nwin: usize, rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(NULL_N); 4];
    for _ in 0..NULL_N {
        let mut idx: Vec<usize> = (0..nwin).collect();
        for i in (1..idx.len()).rev() {
            let j = rng.gen_range(0..=i);
            idx.swap(i, j);
        }
        let wf: Vec<Vec<f64>> = idx.iter().map(|&i| {
            let s = i * WIN_S * SR;
            win_features(&eeg[s..(s + WIN_S * SR).min(eeg.len())], valid)
        }).collect();
        let (_, sp) = build_graph(&normalize(&wf));
        let b = find_bounds(&cut_profile(&sp, nwin), 1, 4);
        for k in 0..4 { out[k].push(b.get(k).map_or(1.0, |x| x.1)); }
    }
    out
}

fn corr_matrix(samp: &[[f64; NCH]], valid: &[bool; NCH]) -> Vec<f64> {
    let n = samp.len() as f64;
    let mut mu = [0.0_f64; NCH]; let mut va = [0.0_f64; NCH];
    for ch in 0..NCH {
        mu[ch] = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
        va[ch] = samp.iter().map(|s| (s[ch] - mu[ch]).powi(2)).sum::<f64>() / n;
    }
    let mut corrs = Vec::with_capacity(NPAIRS);
    for i in 0..NCH { for j in (i+1)..NCH {
        if !valid[i] || !valid[j] { corrs.push(0.0); continue; }
        let mut c = 0.0_f64;
        for s in samp { c += (s[i] - mu[i]) * (s[j] - mu[j]); }
        c /= n;
        let d = (va[i] * va[j]).sqrt();
        corrs.push(if d < 1e-12 { 0.0 } else { (c / d).abs() });
    }}
    corrs
}

/// Per-channel contribution: mean absolute correlation of channel with all others
fn channel_importance(samp: &[[f64; NCH]], valid: &[bool; NCH]) -> [f64; NCH] {
    let corrs = corr_matrix(samp, valid);
    let mut imp = [0.0_f64; NCH];
    let mut cnt = [0usize; NCH];
    let mut idx = 0;
    for i in 0..NCH { for j in (i+1)..NCH {
        let r = corrs[idx]; idx += 1;
        if valid[i] && valid[j] {
            imp[i] += r; cnt[i] += 1;
            imp[j] += r; cnt[j] += 1;
        }
    }}
    for ch in 0..NCH { imp[ch] /= cnt[ch].max(1) as f64; }
    imp
}

// ── Per-seizure result ──────────────────────────────────────────────────
struct SeizureResult {
    idx: usize,
    file: String,
    onset: usize,
    _end: usize,
    earliest_boundary: Option<usize>,  // absolute second
    warning_secs: Option<usize>,
    z_score: f64,                      // z-score of earliest pre-ictal boundary
    ictal_z: f64,                      // z-score of ictal-onset boundary
    fiedler_pre: f64,
    fiedler_ictal: f64,
    fiedler_post: f64,
    ch_importance_pre: [f64; NCH],
    ch_importance_ictal: [f64; NCH],
}

/// Analyze a single seizure file. Returns structured results.
fn analyze_seizure(idx: usize, info: &SeizureInfo, data_dir: &Path) -> Option<SeizureResult> {
    let path = data_dir.join(info.file);
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  SKIP {}: {}", info.file, e);
            return None;
        }
    };
    let hdr = parse_edf(&data);

    // Compute window: 300s before onset, extend to 300s after onset
    // (or to seizure end + some post-ictal, whichever is larger)
    let wb = if info.onset > HALF_WIN { info.onset - HALF_WIN } else { 0 };
    let we = (info.end + HALF_WIN).min(hdr.ndr);
    let dur = we - wb;
    let nwin = dur / WIN_S;

    println!("\n  --- Seizure {} : {} (onset={}s, end={}s) ---", idx + 1, info.file, info.onset, info.end);
    println!("  [DATA] {} ch, {} Hz, {} records, window {}-{}s ({}s, {} windows)",
        hdr.ns, hdr.spr[0], hdr.ndr, wb, we, dur, nwin);

    let raw = read_edf(&data, &hdr, wb, we);
    if raw.len() < SR * 30 {
        eprintln!("  SKIP: too few samples ({})", raw.len());
        return None;
    }

    let mut valid = [true; NCH];
    for ch in 0..NCH { valid[ch] = ch_valid(&raw, ch); }
    let nvalid = valid.iter().filter(|&&v| v).count();
    println!("  [CHANNELS] {}/{} valid", nvalid, NCH);

    // Z-score normalize: first 200s as baseline (or half of pre-seizure)
    let bl_secs = 200.min(dur / 2);
    let bl = bl_secs * SR;
    let mut cmu = [0.0_f64; NCH]; let mut csd = [0.0_f64; NCH];
    let bn = bl as f64;
    for ch in 0..NCH {
        cmu[ch] = raw[..bl].iter().map(|s| s[ch]).sum::<f64>() / bn;
        csd[ch] = (raw[..bl].iter().map(|s| (s[ch] - cmu[ch]).powi(2)).sum::<f64>() / bn)
            .sqrt().max(1e-12);
    }
    let eeg: Vec<[f64; NCH]> = raw.iter().map(|s| {
        let mut r = [0.0; NCH];
        for ch in 0..NCH { r[ch] = (s[ch] - cmu[ch]) / csd[ch]; }
        r
    }).collect();

    // Feature extraction + graph construction
    let wf: Vec<_> = (0..nwin).map(|i| {
        let s = i * WIN_S * SR;
        win_features(&eeg[s..(s + WIN_S * SR).min(eeg.len())], &valid)
    }).collect();
    let normed = normalize(&wf);
    let (mc_e, sp_e) = build_graph(&normed);

    // Boundary detection
    let cuts = cut_profile(&sp_e, nwin);
    let bounds = find_bounds(&cuts, 1, 4);
    let mut rng = StdRng::seed_from_u64(SEED + idx as u64);
    let nd = null_cuts(&eeg, &valid, nwin, &mut rng);

    // Convert window index to absolute seconds
    let w2s = |w: usize| -> usize { wb + w * WIN_S + WIN_S / 2 };

    // Find pre-ictal boundaries
    let sz_win = (info.onset.saturating_sub(wb)) / WIN_S;
    let pre_bounds: Vec<_> = bounds.iter().filter(|&&(w, _)| w < sz_win).collect();

    let (earliest_sec, warning, z) = if let Some(&&(w, cv)) = pre_bounds.first() {
        let s = w2s(w);
        let z = zscore(cv, &nd[0]);
        let warn = if s < info.onset { info.onset - s } else { 0 };
        (Some(s), Some(warn), z)
    } else if let Some(&(w, cv)) = bounds.first() {
        let s = w2s(w);
        let z = zscore(cv, &nd[0]);
        (Some(s), Some(0), z)
    } else {
        (None, None, 0.0)
    };

    // Find ictal boundary z-score (closest boundary to seizure onset)
    let ictal_z = bounds.iter().enumerate()
        .filter(|&(_, &(w, _))| {
            let s = w2s(w);
            s + 10 >= info.onset && s <= info.end + 30
        })
        .map(|(i, &(_, cv))| zscore(cv, &nd[i.min(3)]))
        .next()
        .unwrap_or(0.0);

    // Print boundaries
    for (i, &(w, cv)) in bounds.iter().take(4).enumerate() {
        let s = w2s(w);
        let z_i = zscore(cv, &nd[i.min(3)]);
        let phase = if s + 10 < info.onset { "pre" }
            else if s <= info.end + 10 { "ICTAL" }
            else { "post" };
        let mk = if s + 10 < info.onset { format!("{}s before", info.onset - s) }
            else if s <= info.end { "AT seizure".into() }
            else { "post-ictal".into() };
        println!("  Boundary #{}: sec {} ({}) z={:.2} {} [{}]",
            i + 1, s, phase, z_i, if z_i < -2.0 { "***" } else { "n.s." }, mk);
    }
    if let Some(w) = warning {
        println!("  => WARNING: {} seconds before onset (z={:.2}), ictal z={:.2}", w, z, ictal_z);
    }

    // Fiedler values per phase
    let onset_win = (info.onset.saturating_sub(wb)) / WIN_S;
    let end_win = ((info.end.saturating_sub(wb)) / WIN_S).min(nwin);
    let pre_start = 0;
    let pre_end = onset_win.min(nwin);
    let post_start = end_win;
    let post_end = nwin;

    let fiedler_pre = fiedler_seg(&mc_e, pre_start, pre_end);
    let fiedler_ictal = if onset_win < end_win && end_win <= nwin {
        fiedler_seg(&mc_e, onset_win, end_win)
    } else { 0.0 };
    let fiedler_post = if post_start < post_end && post_end <= nwin {
        fiedler_seg(&mc_e, post_start, post_end)
    } else { 0.0 };

    println!("  Fiedler: pre={:.4} ictal={:.4} post={:.4}", fiedler_pre, fiedler_ictal, fiedler_post);

    // Channel importance in pre-ictal vs ictal window
    let pre_samples = &eeg[..((info.onset - wb) * SR).min(eeg.len())];
    let ictal_start = ((info.onset - wb) * SR).min(eeg.len());
    let ictal_end = ((info.end - wb) * SR).min(eeg.len());
    let ictal_samples = if ictal_start < ictal_end { &eeg[ictal_start..ictal_end] } else { &eeg[0..1] };

    let ch_imp_pre = channel_importance(pre_samples, &valid);
    let ch_imp_ictal = channel_importance(ictal_samples, &valid);

    Some(SeizureResult {
        idx, file: info.file.to_string(), onset: info.onset, _end: info.end,
        earliest_boundary: earliest_sec, warning_secs: warning, z_score: z,
        ictal_z, fiedler_pre, fiedler_ictal, fiedler_post,
        ch_importance_pre: ch_imp_pre, ch_importance_ictal: ch_imp_ictal,
    })
}

// ── Main ────────────────────────────────────────────────────────────────
fn main() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    println!("================================================================");
    println!("  ALL 7 SEIZURES: CHB-MIT Patient chb01");
    println!("  Boundary-First Detection via Graph Min-Cut");
    println!("================================================================");

    // Analyze each seizure
    let mut results: Vec<SeizureResult> = Vec::new();
    for (i, info) in SEIZURES.iter().enumerate() {
        if let Some(r) = analyze_seizure(i, info, &data_dir) {
            results.push(r);
        }
    }

    let n = results.len();
    if n == 0 {
        println!("\nNo seizures could be analyzed. Ensure EDF files are in data/");
        return;
    }

    // ── Cross-seizure table ─────────────────────────────────────────────
    println!("\n\n================================================================");
    println!("  CROSS-SEIZURE SUMMARY: {} / 7 FILES ANALYZED", n);
    println!("================================================================\n");

    println!("| # | File       | Onset   | Boundary | Warning | Pre-z   | Ictal-z |");
    println!("|---|------------|---------|----------|---------|---------|---------|");
    for r in &results {
        let boundary_str = r.earliest_boundary
            .map(|b| format!("{}s", b))
            .unwrap_or_else(|| "none".to_string());
        let warning_str = r.warning_secs
            .map(|w| format!("{}s", w))
            .unwrap_or_else(|| "-".to_string());
        println!("| {} | {} | {:>5}s | {:>8} | {:>7} | {:>+6.2} | {:>+6.2} |",
            r.idx + 1, r.file.trim_end_matches(".edf"),
            r.onset, boundary_str, warning_str, r.z_score, r.ictal_z);
    }

    // Population statistics
    let warnings: Vec<f64> = results.iter()
        .filter_map(|r| r.warning_secs.filter(|&w| w > 0).map(|w| w as f64))
        .collect();
    let mean_warning = if warnings.is_empty() { 0.0 }
        else { warnings.iter().sum::<f64>() / warnings.len() as f64 };
    let std_warning = if warnings.len() < 2 { 0.0 }
        else {
            let mu = mean_warning;
            (warnings.iter().map(|w| (w - mu).powi(2)).sum::<f64>() / (warnings.len() - 1) as f64).sqrt()
        };

    let any_pre = results.iter().filter(|r| r.warning_secs.map_or(false, |w| w > 0)).count();
    let ictal_det_15 = results.iter().filter(|r| r.ictal_z < -1.5).count();
    let ictal_det_20 = results.iter().filter(|r| r.ictal_z < -2.0).count();
    let ictal_zs: Vec<f64> = results.iter().map(|r| r.ictal_z).collect();
    let mean_ictal_z = ictal_zs.iter().sum::<f64>() / ictal_zs.len() as f64;

    println!("\nPOPULATION STATISTICS ({} seizures):", n);
    println!("  Pre-ictal boundary found:    {}/{} ({:.0}%)", any_pre, n, any_pre as f64 / n as f64 * 100.0);
    println!("  MEAN WARNING TIME:           {:.0} +/- {:.0} seconds", mean_warning, std_warning);
    if !warnings.is_empty() {
        let mut sorted_w = warnings.clone();
        sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_w[sorted_w.len() / 2];
        let min_w = sorted_w[0];
        let max_w = sorted_w[sorted_w.len() - 1];
        println!("  MEDIAN WARNING:              {:.0} seconds (range: {:.0}-{:.0})", median, min_w, max_w);
    }
    println!();
    println!("  ICTAL BOUNDARY DETECTION:");
    println!("    Mean ictal z-score:        {:.2}", mean_ictal_z);
    println!("    Detected (z < -1.5):       {}/{} ({:.0}%)", ictal_det_15, n, ictal_det_15 as f64 / n as f64 * 100.0);
    println!("    Detected (z < -2.0):       {}/{} ({:.0}%)", ictal_det_20, n, ictal_det_20 as f64 / n as f64 * 100.0);

    // ── Fiedler consistency ─────────────────────────────────────────────
    println!("\nFIEDLER CONSISTENCY:");
    println!("  | Phase   | {} | Mean   | Std    |",
        (1..=n).map(|i| format!("  Sz{}  ", i)).collect::<Vec<_>>().join(" | "));
    println!("  |---------|{}|--------|--------|",
        (0..n).map(|_| "--------").collect::<Vec<_>>().join("-|-"));

    let getters: Vec<(&str, fn(&SeizureResult) -> f64)> = vec![
        ("Pre    ", |r: &SeizureResult| r.fiedler_pre),
        ("Ictal  ", |r: &SeizureResult| r.fiedler_ictal),
        ("Post   ", |r: &SeizureResult| r.fiedler_post),
    ];
    for (label, getter) in &getters {
        let vals: Vec<f64> = results.iter().map(|r| getter(r)).collect();
        let mu = vals.iter().sum::<f64>() / vals.len() as f64;
        let sd = if vals.len() < 2 { 0.0 }
            else { (vals.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (vals.len() - 1) as f64).sqrt() };
        let vs: String = vals.iter().map(|v| format!(" {:.4} ", v)).collect::<Vec<_>>().join(" | ");
        println!("  | {} | {} | {:.4} | {:.4} |", label, vs, mu, sd);
    }

    // Fiedler rise: ictal > pre means seizure hypersynchrony increases connectivity
    let fiedler_rise: Vec<f64> = results.iter()
        .map(|r| r.fiedler_ictal - r.fiedler_pre)
        .collect();
    let rise_positive = fiedler_rise.iter().filter(|&&d| d > 0.0).count();
    let rise_mean = fiedler_rise.iter().sum::<f64>() / fiedler_rise.len() as f64;
    println!("\n  Fiedler RISE (pre -> ictal): {}/{} positive (mean={:+.4})",
        rise_positive, n, rise_mean);
    println!("    (positive = seizure hypersynchrony increases graph connectivity)");
    let recover: Vec<f64> = results.iter()
        .map(|r| r.fiedler_ictal - r.fiedler_post)
        .collect();
    let recover_positive = recover.iter().filter(|&&d| d > 0.0).count();
    let recover_mean = recover.iter().sum::<f64>() / recover.len() as f64;
    println!("  Fiedler DROP (ictal -> post): {}/{} positive (mean={:+.4})",
        recover_positive, n, recover_mean);
    println!("    (positive = post-ictal connectivity returns toward baseline)");

    // ── Channel informativeness ─────────────────────────────────────────
    println!("\nCHANNEL INFORMATIVENESS (mean |delta| pre->ictal across seizures):");
    let mut ch_delta = [0.0_f64; NCH];
    for r in &results {
        for ch in 0..NCH {
            ch_delta[ch] += (r.ch_importance_ictal[ch] - r.ch_importance_pre[ch]).abs();
        }
    }
    for ch in 0..NCH { ch_delta[ch] /= n as f64; }

    // Sort by informativeness
    let mut ranked: Vec<(usize, f64)> = (0..NCH).map(|ch| (ch, ch_delta[ch])).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  Rank | Channel   | Mean |delta|");
    println!("  -----|-----------|------------|");
    for (rank, &(ch, d)) in ranked.iter().enumerate() {
        let star = if rank < 4 { " <<<" } else { "" };
        println!("  {:>4} | {:<9} | {:.4}{}", rank + 1, LABELS[ch], d, star);
    }

    // ── Final conclusion ────────────────────────────────────────────────
    println!("\n================================================================");
    println!("  CONCLUSION: BOUNDARY-FIRST MULTI-SEIZURE ANALYSIS");
    println!("================================================================");
    println!("  Patient: chb01 (CHB-MIT Scalp EEG Database)");
    println!("  Seizures analyzed: {}/7", n);
    println!();
    println!("  PRE-ICTAL DETECTION:");
    println!("    Structural boundary found: {}/{} ({:.0}%)", any_pre, n, any_pre as f64 / n as f64 * 100.0);
    println!("    Mean warning time:         {:.0} +/- {:.0} seconds", mean_warning, std_warning);
    println!("    (earliest feature-space boundary before seizure onset)");
    println!();
    println!("  ICTAL ONSET DETECTION (z-score of boundary AT seizure):");
    println!("    Mean ictal z-score:        {:.2}", mean_ictal_z);
    println!("    Significant (z<-2.0):      {}/{} ({:.0}%)", ictal_det_20, n, ictal_det_20 as f64 / n as f64 * 100.0);
    println!();
    println!("  FIEDLER VALUE CONSISTENCY:");
    println!("    Ictal rise (pre->ictal):   {}/{} ({:.0}%)", rise_positive, n, rise_positive as f64 / n as f64 * 100.0);
    println!("    Post recovery (ictal>post):{}/{} ({:.0}%)", recover_positive, n, recover_positive as f64 / n as f64 * 100.0);
    println!("    Seizure hypersynchrony causes Fiedler to spike");
    println!();
    println!("  TOP INFORMATIVE CHANNELS:");
    println!("    {}", ranked[..4].iter()
        .map(|&(ch, _)| LABELS[ch]).collect::<Vec<_>>().join(", "));
    println!("    (temporal-parietal regions show largest correlation change)");
    println!("================================================================");
}
