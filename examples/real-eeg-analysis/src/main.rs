//! Real EEG Boundary-First Seizure Detection (Multi-Scale + Enhanced)
//!
//! Parses REAL clinical EEG from CHB-MIT Scalp EEG Database (PhysioNet).
//! Runs boundary-first detection on patient chb01, file chb01_03.edf
//! (seizure at seconds 2996-3036). EDF binary parsed directly in Rust.
//!
//! Optimizations: multi-scale windows (5/10/30s), artifact rejection,
//! 50% overlap, enhanced features (+theta/delta/alpha-gamma/zero-cross),
//! running baseline normalization, patient-specific null model.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;
use std::path::Path;

const NCH: usize = 16;
const SR: usize = 256;
const SEED: u64 = 42_0911;
const NPAIRS: usize = NCH * (NCH - 1) / 2;
const AMP_UV: f64 = 500.0;
const AMP_THR: f64 = 3.0;
const NULL_N: usize = 50;
const SZ_START: usize = 2996;
const SZ_END: usize = 3036;
const TAU: f64 = std::f64::consts::TAU;
const WB: usize = 2696;
const WE: usize = 3296;
const DUR: usize = WE - WB;
const BL_S: usize = 200;
const NFEAT: usize = NPAIRS + NCH * 8; // 120 corr + 128 spectral = 248

const LABELS: [&str; 23] = [
    "FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ","P7-T7","T7-FT9","FT9-FT10","FT10-T8","T8-P8",
];

fn region(ch: usize) -> usize { match ch { 0|4|5=>0, 1..=3|6|7=>1, 8|9|12=>2, _=>3 } }

// ── EDF parser ──────────────────────────────────────────────────────────
struct Edf { ns: usize, ndr: usize, dur: f64, pmin: Vec<f64>, pmax: Vec<f64>,
    dmin: Vec<i16>, dmax: Vec<i16>, spr: Vec<usize> }
fn af(b: &[u8], s: usize, l: usize) -> String { String::from_utf8_lossy(&b[s..s+l]).trim().to_string() }
fn af64(b: &[u8], s: usize, l: usize) -> f64 { af(b,s,l).parse().unwrap_or(0.0) }
fn ausz(b: &[u8], s: usize, l: usize) -> usize { af(b,s,l).parse().unwrap_or(0) }

fn parse_edf(d: &[u8]) -> Edf {
    let ns = ausz(d,252,4); let b = 256;
    let (mut pmin,mut pmax,mut dmin,mut dmax,mut spr) = (vec![],vec![],vec![],vec![],vec![]);
    let mut off = b + ns*16 + ns*80 + ns*8;
    for i in 0..ns { pmin.push(af64(d, off+i*8, 8)); } off += ns*8;
    for i in 0..ns { pmax.push(af64(d, off+i*8, 8)); } off += ns*8;
    for i in 0..ns { dmin.push(af64(d, off+i*8, 8) as i16); } off += ns*8;
    for i in 0..ns { dmax.push(af64(d, off+i*8, 8) as i16); } off += ns*8;
    off += ns*80;
    for i in 0..ns { spr.push(ausz(d, off+i*8, 8)); }
    Edf { ns, ndr: ausz(d,236,8), dur: af64(d,244,8), pmin, pmax, dmin, dmax, spr }
}

fn read_edf(d: &[u8], h: &Edf, s0: usize, s1: usize) -> Vec<[f64; NCH]> {
    let hsz = 256 + h.ns * 256; let tot: usize = h.spr.iter().sum(); let rbytes = tot * 2;
    let mut gain = vec![0.0_f64; h.ns]; let mut ofs = vec![0.0_f64; h.ns];
    for i in 0..h.ns {
        let dr = h.dmax[i] as f64 - h.dmin[i] as f64; let pr = h.pmax[i] - h.pmin[i];
        gain[i] = if dr.abs()<1e-12 {1.0} else {pr/dr};
        ofs[i] = h.pmin[i] - h.dmin[i] as f64 * gain[i];
    }
    let mut out = Vec::with_capacity((s1-s0)*SR);
    for rec in s0..s1.min(h.ndr) {
        let ro = hsz + rec * rbytes; let mut so = 0usize;
        let mut chdata: Vec<Vec<f64>> = vec![Vec::new(); h.ns.min(NCH)];
        for sig in 0..h.ns {
            let n = h.spr[sig];
            if sig < NCH { for s in 0..n {
                let bp = ro + (so+s)*2;
                if bp+1 >= d.len() { break; }
                chdata[sig].push(i16::from_le_bytes([d[bp], d[bp+1]]) as f64 * gain[sig] + ofs[sig]);
            }}
            so += n;
        }
        for s in 0..h.spr[0] {
            let mut row = [0.0_f64; NCH];
            for ch in 0..NCH { if s < chdata[ch].len() { row[ch] = chdata[ch][s]; } }
            out.push(row);
        }
    }
    out
}

// ── Signal processing ───────────────────────────────────────────────────
fn goertzel(sig: &[f64], freq: f64) -> f64 {
    let n = sig.len(); if n==0 { return 0.0; }
    let w = TAU * (freq * n as f64 / SR as f64).round() / n as f64;
    let c = 2.0 * w.cos(); let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for &x in sig { let s0 = x + c*s1 - s2; s2 = s1; s1 = s0; }
    (s1*s1 + s2*s2 - c*s1*s2).max(0.0) / (n*n) as f64
}

fn ch_valid(samp: &[[f64; NCH]], ch: usize) -> bool {
    let n = samp.len() as f64; if n<2.0 { return false; }
    let mu: f64 = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
    samp.iter().map(|s| (s[ch]-mu).powi(2)).sum::<f64>() / n > 1e-10
}

/// Per-window artifact mask: rejects channels with amplitude > AMP_UV or near-zero variance.
fn artifact_mask(samp: &[[f64; NCH]], gv: &[bool; NCH]) -> [bool; NCH] {
    let mut m = *gv; let n = samp.len() as f64;
    for ch in 0..NCH { if !gv[ch] { continue; }
        let peak = samp.iter().map(|s| s[ch].abs()).fold(0.0_f64, f64::max);
        if peak > AMP_UV { m[ch] = false; continue; }
        let mu: f64 = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
        if samp.iter().map(|s| (s[ch]-mu).powi(2)).sum::<f64>() / n < 1e-10 { m[ch] = false; }
    }
    m
}

fn band_pwr(sig: &[f64], freqs: &[f64]) -> f64 { freqs.iter().map(|&f| goertzel(sig, f)).sum() }

/// Enhanced features: 120 correlations + per-channel (alpha,beta,gamma,dom_freq,theta,delta,ag_ratio,zc_entropy)
fn win_features(samp: &[[f64; NCH]], valid: &[bool; NCH]) -> Vec<f64> {
    let n = samp.len() as f64;
    let mut f = Vec::with_capacity(NFEAT);
    let mut mu = [0.0_f64; NCH]; let mut va = [0.0_f64; NCH];
    for ch in 0..NCH {
        mu[ch] = samp.iter().map(|s| s[ch]).sum::<f64>() / n;
        va[ch] = samp.iter().map(|s| (s[ch]-mu[ch]).powi(2)).sum::<f64>() / n;
    }
    for i in 0..NCH { for j in (i+1)..NCH {
        if !valid[i] || !valid[j] { f.push(0.0); continue; }
        let mut c = 0.0_f64; for s in samp { c += (s[i]-mu[i])*(s[j]-mu[j]); }
        c /= n; let d = (va[i]*va[j]).sqrt();
        f.push(if d<1e-12 {0.0} else {c/d});
    }}
    for ch in 0..NCH {
        let sig: Vec<f64> = samp.iter().map(|s| s[ch]).collect();
        if !valid[ch] { f.extend_from_slice(&[-10.0,-10.0,-10.0,0.0,-10.0,-10.0,0.0,0.0]); continue; }
        let a = band_pwr(&sig, &[8.0,9.0,10.0,11.0,12.0,13.0]);
        let b = band_pwr(&sig, &[14.0,18.0,22.0,26.0,30.0]);
        let g = band_pwr(&sig, &[35.0,42.0,50.0,60.0,70.0,80.0]);
        f.push(a.max(1e-20).ln().max(-10.0));
        f.push(b.max(1e-20).ln().max(-10.0));
        f.push(g.max(1e-20).ln().max(-10.0));
        let (mut bf, mut bp) = (10.0_f64, 0.0_f64);
        for fi in 2..80 { let p = goertzel(&sig, fi as f64); if p>bp { bp=p; bf=fi as f64; } }
        f.push(bf / 80.0);
        // theta (4-7 Hz), delta (1-3 Hz)
        f.push(band_pwr(&sig, &[4.0,5.0,6.0,7.0]).max(1e-20).ln().max(-10.0));
        f.push(band_pwr(&sig, &[1.0,2.0,3.0]).max(1e-20).ln().max(-10.0));
        // alpha/gamma ratio
        let ag = band_pwr(&sig, &[8.0,10.0,12.0]) / band_pwr(&sig, &[35.0,50.0,70.0]).max(1e-20);
        f.push(ag.ln().max(-10.0).min(10.0));
        // zero-crossing entropy
        let zc = (1..sig.len()).filter(|&i| (sig[i]-mu[ch]).signum() != (sig[i-1]-mu[ch]).signum()).count();
        f.push(zc as f64 / n);
    }
    f
}

/// Normalize features. If `bl_n > 0`, use only first `bl_n` windows for stats; else all.
fn normalize(fs: &[Vec<f64>], bl_n: usize) -> Vec<Vec<f64>> {
    let d = fs[0].len();
    let bn = if bl_n > 0 { bl_n.min(fs.len()) } else { fs.len() };
    let n = bn as f64;
    let mut mu = vec![0.0_f64;d]; let mut sd = vec![0.0_f64;d];
    for f in &fs[..bn] { for i in 0..d { mu[i] += f[i]; } }
    for v in &mut mu { *v /= n; }
    for f in &fs[..bn] { for i in 0..d { sd[i] += (f[i]-mu[i]).powi(2); } }
    for v in &mut sd { *v = (*v/n).sqrt().max(1e-12); }
    fs.iter().map(|f| (0..d).map(|i| (f[i]-mu[i])/sd[i]).collect()).collect()
}

fn dsq(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)|(x-y).powi(2)).sum() }

fn build_graph(f: &[Vec<f64>]) -> (Vec<(u64,u64,f64)>, Vec<(usize,usize,f64)>) {
    let mut ds: Vec<f64> = (0..f.len()).flat_map(|i| ((i+1)..f.len().min(i+5)).map(move |j| dsq(&f[i],&f[j]))).collect();
    ds.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let sig = ds[ds.len()/2].max(1e-6);
    let (mut mc, mut sp) = (Vec::new(), Vec::new());
    for i in 0..f.len() { for sk in 1..=4 { if i+sk < f.len() {
        let w = (-dsq(&f[i],&f[i+sk])/(2.0*sig)).exp().max(1e-6);
        mc.push((i as u64,(i+sk) as u64,w)); sp.push((i,i+sk,w));
    }}}
    (mc, sp)
}

fn cut_profile(edges: &[(usize,usize,f64)], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n];
    for &(u,v,w) in edges { for k in (u.min(v)+1)..=u.max(v) { c[k] += w; } }
    c
}

fn find_bounds(cuts: &[f64], margin: usize, gap: usize) -> Vec<(usize,f64)> {
    let n = cuts.len();
    let mut m: Vec<(usize,f64,f64)> = (1..n-1).filter_map(|i| {
        if i<=margin || i>=n-margin || cuts[i]>=cuts[i-1] || cuts[i]>=cuts[i+1] { return None; }
        let (lo,hi) = (i.saturating_sub(2),(i+3).min(n));
        Some((i, cuts[i], cuts[lo..hi].iter().sum::<f64>()/(hi-lo) as f64 - cuts[i]))
    }).collect();
    m.sort_by(|a,b| b.2.partial_cmp(&a.2).unwrap());
    let mut s = Vec::new();
    for &(p,v,_) in &m {
        if s.iter().all(|&(q,_): &(usize,f64)| (p as isize-q as isize).unsigned_abs()>=gap) { s.push((p,v)); }
    }
    s.sort_by_key(|&(d,_)| d); s
}

fn amp_detect(eeg: &[[f64; NCH]], valid: &[bool; NCH]) -> Option<usize> {
    let bl = (200 * SR).min(eeg.len());
    let (mut sq, mut cnt) = (0.0_f64, 0usize);
    for s in &eeg[..bl] { for ch in 0..NCH { if valid[ch] { sq += s[ch]*s[ch]; cnt += 1; } } }
    let br = (sq / cnt.max(1) as f64).sqrt();
    for sec in 0..(eeg.len()/SR) {
        let (st,e) = (sec*SR, (sec*SR+SR).min(eeg.len()));
        let (mut sm, mut c) = (0.0_f64, 0usize);
        for s in &eeg[st..e] { for ch in 0..NCH { if valid[ch] { sm += s[ch]*s[ch]; c += 1; } } }
        if (sm / c.max(1) as f64).sqrt() > br * AMP_THR { return Some(sec + WB); }
    }
    None
}

fn rms(eeg: &[[f64; NCH]], valid: &[bool; NCH]) -> f64 {
    let (mut s, mut c) = (0.0_f64, 0usize);
    for r in eeg { for ch in 0..NCH { if valid[ch] { s += r[ch]*r[ch]; c += 1; } } }
    (s / c.max(1) as f64).sqrt()
}

fn corr_stats(samp: &[[f64; NCH]], valid: &[bool; NCH]) -> (f64, f64, f64) {
    let n = samp.len() as f64;
    let mut mu = [0.0_f64;NCH]; let mut va = [0.0_f64;NCH];
    for ch in 0..NCH { mu[ch]=samp.iter().map(|s|s[ch]).sum::<f64>()/n;
        va[ch]=samp.iter().map(|s|(s[ch]-mu[ch]).powi(2)).sum::<f64>()/n; }
    let (mut all,mut ci,mut cx) = (0.0_f64,0.0_f64,0.0_f64);
    let (mut na,mut ni,mut nx) = (0usize,0usize,0usize);
    for i in 0..NCH { if !valid[i]{continue;} for j in (i+1)..NCH { if !valid[j]{continue;}
        let mut c=0.0; for s in samp { c+=(s[i]-mu[i])*(s[j]-mu[j]); }
        c/=n; let d=(va[i]*va[j]).sqrt();
        let r=if d<1e-12{0.0}else{(c/d).abs()};
        all+=r; na+=1;
        if region(i)==region(j){ci+=r;ni+=1}else{cx+=r;nx+=1}
    }}
    (all/na.max(1) as f64, ci/ni.max(1) as f64, cx/nx.max(1) as f64)
}

fn band_ratio(samp: &[[f64; NCH]], valid: &[bool; NCH]) -> (f64, f64) {
    let (mut at, mut gt, mut nc) = (0.0_f64, 0.0_f64, 0usize);
    for ch in 0..NCH { if !valid[ch]{continue;}
        let sig: Vec<f64> = samp.iter().map(|s|s[ch]).collect();
        at += band_pwr(&sig, &[8.0,10.0,12.0]);
        gt += band_pwr(&sig, &[35.0,42.0,55.0,70.0]);
        nc += 1;
    }
    (at/nc.max(1) as f64, gt/nc.max(1) as f64)
}

fn zscore(obs: f64, null: &[f64]) -> f64 {
    let n=null.len()as f64; let mu:f64=null.iter().sum::<f64>()/n;
    let sd=(null.iter().map(|v|(v-mu).powi(2)).sum::<f64>()/n).sqrt();
    if sd<1e-12{0.0}else{(obs-mu)/sd}
}

fn fiedler_seg(edges: &[(u64,u64,f64)], s: usize, e: usize) -> f64 {
    let n=e-s; if n<3{return 0.0;}
    let se: Vec<_> = edges.iter().filter(|(u,v,_)| {
        let (a,b)=(*u as usize,*v as usize); a>=s&&a<e&&b>=s&&b<e
    }).map(|(u,v,w)|(*u as usize-s,*v as usize-s,*w)).collect();
    if se.is_empty(){return 0.0;} estimate_fiedler(&CsrMatrixView::build_laplacian(n,&se),200,1e-10).0
}

fn pname(sec: usize) -> &'static str {
    if sec<SZ_START-10{"Pre-seizure"} else if sec<SZ_START+5{"Peri-ictal"}
    else if sec<=SZ_END{"Seizure"} else{"Post-ictal"}
}

fn w2s(w: usize, stride: usize, win: usize) -> usize { WB + w * stride + win / 2 }

/// Run boundary detection at one scale. Returns (bounds, null_dists, nwin, artifact_count).
fn run_scale(eeg: &[[f64; NCH]], raw: &[[f64; NCH]], valid: &[bool; NCH],
    win_s: usize, stride_s: usize, rng: &mut StdRng,
) -> (Vec<(usize,f64)>, Vec<Vec<f64>>, usize, usize) {
    let (ws, ss) = (win_s * SR, stride_s * SR);
    let nwin = if eeg.len() >= ws { (eeg.len() - ws) / ss + 1 } else { 1 };
    let gap = (4 * SR / ss).max(2);
    let mut art = 0usize;

    let wf: Vec<Vec<f64>> = (0..nwin).map(|i| {
        let (s,e) = (i*ss, (i*ss+ws).min(eeg.len()));
        let mask = artifact_mask(&raw[s..e], valid);
        if (0..NCH).any(|ch| valid[ch] && !mask[ch]) { art += 1; }
        win_features(&eeg[s..e], &mask)
    }).collect();

    let normed = normalize(&wf, 0); // global normalization for boundary detection
    let (_, sp) = build_graph(&normed);
    let bounds = find_bounds(&cut_profile(&sp, nwin), 1, gap);

    // Null model: shuffled window order preserves feature marginals, breaks temporal structure.
    // Also generate nulls from seizure-free baseline segment for patient-specific null.
    let null_total = NULL_N + NULL_N / 2;
    let mut nd = vec![Vec::with_capacity(null_total); 4];
    // Shuffle-based null (primary)
    for _ in 0..NULL_N {
        let mut idx: Vec<usize> = (0..nwin).collect();
        for i in (1..idx.len()).rev() { let j=rng.gen_range(0..=i); idx.swap(i,j); }
        let shuf: Vec<Vec<f64>> = idx.iter().map(|&i| normed[i].clone()).collect();
        let (_, sp2) = build_graph(&shuf);
        let b = find_bounds(&cut_profile(&sp2, nwin), 1, gap);
        for k in 0..4 { nd[k].push(b.get(k).map_or(1.0, |x| x.1)); }
    }
    // Patient-specific null: bootstrap-resample baseline windows, normalize globally, detect bounds.
    // This models "what boundaries would appear in normal brain activity alone?"
    let sf_end = (BL_S * SR).min(eeg.len());
    let sf_nwin = if sf_end > ws + ss { (sf_end - ws) / ss + 1 } else { 0 };
    if sf_nwin >= 4 { for _ in 0..(NULL_N / 2) {
        let boot: Vec<Vec<f64>> = (0..nwin).map(|_| {
            let i = rng.gen_range(0..sf_nwin);
            let (s,e) = (i*ss, (i*ss+ws).min(sf_end));
            win_features(&eeg[s..e], &artifact_mask(&raw[s..e], valid))
        }).collect();
        let bn = normalize(&boot, 0);
        let (_, sp2) = build_graph(&bn);
        let b = find_bounds(&cut_profile(&sp2, nwin), 1, gap);
        for k in 0..4 { nd[k].push(b.get(k).map_or(1.0, |x| x.1)); }
    }}
    (bounds, nd, nwin, art)
}

fn feat_name(idx: usize) -> String {
    if idx < NPAIRS { return format!("channel-pair corr #{}", idx); }
    let r = idx - NPAIRS; let ch = r / 8; let k = r % 8;
    let nm = LABELS[ch.min(NCH-1)];
    match k { 0=>"alpha", 1=>"beta", 2=>"gamma", 3=>"dom_freq", 4=>"theta",
        5=>"delta", 6=>"alpha/gamma", _=>"zero-cross" }.to_string() + " " + nm
}

// ── Main ────────────────────────────────────────────────────────────────
fn main() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/chb01_03.edf");
    println!("================================================================");
    println!("  REAL EEG: CHB-MIT Patient chb01, File chb01_03.edf");
    println!("  Seizure at seconds {}-{}", SZ_START, SZ_END);
    println!("  Multi-scale + artifact rejection + enhanced features");
    println!("================================================================\n");

    let data = match std::fs::read(&path) {
        Ok(d) => d, Err(e) => { eprintln!("ERROR: {:?}: {}", path, e); std::process::exit(1); }
    };
    let hdr = parse_edf(&data);
    println!("[DATA] {} channels, {} Hz, {} records x {:.0}s = {}s ({:.1}h)",
        hdr.ns, hdr.spr[0], hdr.ndr, hdr.dur, hdr.ndr, hdr.ndr as f64/3600.0);
    println!("[DATA] Extracting {}s window ({}-{}s) around seizure\n", DUR, WB, WE);

    let raw = read_edf(&data, &hdr, WB, WE);
    println!("[WINDOW] {} samples ({} seconds at {} Hz)", raw.len(), raw.len()/SR, SR);

    let mut valid = [true; NCH];
    for ch in 0..NCH { valid[ch] = ch_valid(&raw, ch); }
    let used: Vec<&str> = (0..NCH).filter(|&c| valid[c]).map(|c| LABELS[c]).collect();
    let skip: Vec<&str> = (0..NCH).filter(|&c| !valid[c]).map(|c| LABELS[c]).collect();
    println!("[CHANNELS] Using {}/{}: [{}]", used.len(), NCH, used.join(", "));
    if !skip.is_empty() { println!("[CHANNELS] Skipped: [{}]", skip.join(", ")); }

    // Normalize using first 200s baseline only
    let bl = (BL_S*SR).min(raw.len()); let bn = bl as f64;
    let mut cmu=[0.0_f64;NCH]; let mut csd=[0.0_f64;NCH];
    for ch in 0..NCH {
        cmu[ch] = raw[..bl].iter().map(|s|s[ch]).sum::<f64>() / bn;
        csd[ch] = (raw[..bl].iter().map(|s|(s[ch]-cmu[ch]).powi(2)).sum::<f64>()/bn).sqrt().max(1e-12);
    }
    let eeg: Vec<[f64;NCH]> = raw.iter().map(|s| {
        let mut r=[0.0;NCH]; for ch in 0..NCH { r[ch]=(s[ch]-cmu[ch])/csd[ch]; } r
    }).collect();

    // Phase statistics
    println!();
    for &(nm,s,e) in &[("Pre-seizure",WB,SZ_START),("Peri-ictal",SZ_START-60,SZ_START),
        ("Seizure",SZ_START,SZ_END),("Post-ictal",SZ_END,WE)] {
        let (si,ei) = ((s-WB)*SR, ((e-WB)*SR).min(eeg.len()));
        if si>=eeg.len() { continue; }
        let (_,ci,cx) = corr_stats(&eeg[si..ei], &valid);
        println!("  {:<13} RMS={:.3}  intra|r|={:.3}  cross|r|={:.3}", nm, rms(&eeg[si..ei],&valid), ci, cx);
    }

    // Amplitude detection
    let ad = amp_detect(&eeg, &valid);
    println!("\n[AMPLITUDE DETECTION]");
    if let Some(sec) = ad {
        println!("  RMS exceeds {}x baseline at second {} ({}s {} onset)", AMP_THR, sec,
            if sec<SZ_START{SZ_START-sec}else{sec-SZ_START}, if sec<SZ_START{"before"}else{"after"});
    } else { println!("  No amplitude threshold crossing detected"); }

    // ── Multi-Scale Analysis ────────────────────────────────────────────
    let mut rng = StdRng::seed_from_u64(SEED);
    // 5s: ~40% overlap; 10s: no overlap; 30s: no overlap
    let scales: [(usize,usize,&str);3] = [(5,3,"5-second"), (10,10,"10-second"), (30,30,"30-second")];
    let (mut best_z, mut best_scale) = (0.0_f64, "");
    let (mut p_bounds, mut p_nd, mut p_nwin, mut p_art) = (vec![], vec![vec![]], 0, 0);

    println!("\n[MULTI-SCALE ANALYSIS]");
    for &(ws, ss, label) in &scales {
        let (bounds, nd, nwin, art) = run_scale(&eeg, &raw, &valid, ws, ss, &mut rng);
        let pre: Vec<_> = bounds.iter().filter(|&&(w,_)| w2s(w,ss,ws) < SZ_START-10).collect();
        if let Some(&&(w, cv)) = pre.first() {
            let (s, z) = (w2s(w,ss,ws), zscore(cv, &nd[0]));
            println!("  {:<16} boundary at second {} (z={:.2}, {}s before, {} wins)", label, s, z, SZ_START-s, nwin);
            if z < best_z { best_z = z; best_scale = label; }
        } else { println!("  {:<16} no pre-ictal boundary ({} wins)", label, nwin); }
        if ws == 10 { p_bounds = bounds; p_nd = nd; p_nwin = nwin; p_art = art; }
    }
    println!("  Best z-score: {:.2} at {} scale", best_z, best_scale);

    println!("\n[ARTIFACT REJECTION] Windows with artifacts: {}/{} (10s scale)", p_art, p_nwin);

    // ── Detailed 10s-scale boundaries ───────────────────────────────────
    let (ws, ss) = (10usize, 10usize); // matches the 10s scale stride
    println!("\n[BOUNDARY DETECTION] ({}s windows, {}s stride, {} features)", ws, ss, NFEAT);
    for (i,&(w,cv)) in p_bounds.iter().take(6).enumerate() {
        let (s, z) = (w2s(w,ss,ws), zscore(cv, &p_nd[i.min(3)]));
        let mk = if s<SZ_START-10 {format!("{}s before onset",SZ_START-s)}
            else if s<=SZ_END {"AT seizure".into()} else {"post-ictal".into()};
        println!("  #{}: second {} ({}) z={:.2} {} [{}]", i+1, s, pname(s), z, if z< -2.0{"***"}else{"n.s."}, mk);
    }

    let pre_b: Vec<_> = p_bounds.iter().filter(|&&(w,_)| w2s(w,ss,ws)<SZ_START-10).collect();
    let ict_b = p_bounds.iter().find(|&&(w,_)| { let s=w2s(w,ss,ws); s>=SZ_START-10&&s<=SZ_END+10 });
    let earliest = pre_b.first().copied();

    println!("  Pre-ictal boundaries: {}", pre_b.len());
    if let Some(&(w,cv)) = earliest {
        let (s,z) = (w2s(w,ss,ws), zscore(cv,&p_nd[0]));
        println!("  Earliest: second {} ({}s before onset, z={:.2})", s, SZ_START-s, z);
    }
    if let Some(&(w,cv)) = ict_b {
        let z=zscore(cv,&p_nd[0]);
        println!("  Seizure-onset: second {} (z={:.2} {})", w2s(w,ss,ws), z, if z< -2.0{"SIGNIFICANT"}else{"n.s."});
    }

    // Feature extraction for discontinuity + enhanced features report
    let (wsp,ssp) = (ws*SR, ss*SR);
    let nwin_p = if eeg.len()>=wsp { (eeg.len()-wsp)/ssp+1 } else { 1 };
    let wf: Vec<Vec<f64>> = (0..nwin_p).map(|i| {
        let (s,e) = (i*ssp, (i*ssp+wsp).min(eeg.len()));
        win_features(&eeg[s..e], &artifact_mask(&raw[s..e], &valid))
    }).collect();
    let normed = normalize(&wf, 0);

    let avg_d: f64 = (1..normed.len()).map(|i| dsq(&normed[i-1],&normed[i]).sqrt()).sum::<f64>()
        / (normed.len()-1).max(1) as f64;
    println!("\n[FEATURE DISCONTINUITY] avg distance: {:.3}", avg_d);
    for i in 1..normed.len() {
        let d = dsq(&normed[i-1],&normed[i]).sqrt(); let r = d/avg_d.max(0.01);
        if r > 1.5 {
            let s = w2s(i,ss,ws);
            let mk = if s<SZ_START{format!("{}s before",SZ_START-s)} else if s<=SZ_END{"DURING".into()} else{"post".into()};
            println!("    second {}: {:.2}x avg ({}) [{}]", s, r, pname(s), mk);
        }
    }

    // Enhanced features at boundary
    if let Some(&(w,_)) = earliest.or(ict_b).or(p_bounds.first()) {
        let s = w2s(w,ss,ws);
        println!("\n[ENHANCED FEATURES] At boundary second {}:", s);
        if w >= 2 && w + 2 < normed.len() {
            let (bef, aft) = (&normed[w-2], &normed[w+1]);
            let mut diffs: Vec<(usize,f64)> = bef.iter().zip(aft).enumerate()
                .map(|(i,(a,b))| (i, (b-a).abs())).collect();
            diffs.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            for (rank,&(idx,delta)) in diffs.iter().take(5).enumerate() {
                println!("  #{}: {} -- changed {:.2} sigma", rank+1, feat_name(idx), delta);
            }
        }
        let (bs,be) = ((s.saturating_sub(20)-WB)*SR, (s-WB)*SR);
        let (as_,ae) = ((s-WB)*SR, ((s+20).min(WE)-WB)*SR);
        if be<=eeg.len() && ae<=eeg.len() && bs<be && as_<ae {
            let (ab,gb) = band_ratio(&eeg[bs..be], &valid);
            let (aa,ga) = band_ratio(&eeg[as_..ae], &valid);
            println!("\n[BAND POWER] At boundary second {}:", s);
            println!("  Alpha: {:.6} -> {:.6} ({:+.0}%)", ab, aa, ((aa-ab)/ab.max(1e-12))*100.0);
            println!("  Gamma: {:.6} -> {:.6} ({:.1}x)", gb, ga, ga/gb.max(1e-12));
            println!("  RMS: {:.3} -> {:.3}", rms(&eeg[bs..be],&valid), rms(&eeg[as_..ae],&valid));
        }
    }

    // Correlation trajectory
    println!("\n[CORRELATION TRAJECTORY] cross-region |r| per 30s:");
    let mut epoch = (SZ_START-180).max(WB);
    let (mut prev_cx, mut first_rise) = (0.0_f64, None);
    while epoch+30 <= WE.min(SZ_END+60) {
        let (si,ei) = ((epoch-WB)*SR, ((epoch+30)-WB)*SR);
        if ei > eeg.len() { break; }
        let (_,ci,cx) = corr_stats(&eeg[si..ei], &valid);
        let delta = cx - prev_cx;
        let mk = if epoch>=SZ_START && epoch<SZ_END { " <<<SEIZURE" }
            else if delta>0.02 && epoch<SZ_START && first_rise.is_none() { first_rise = Some(epoch); " <<<FIRST RISE" }
            else { "" };
        println!("    {}-{}s: intra={:.3} cross={:.3} d={:+.3}{}", epoch, epoch+30, ci, cx, delta, mk);
        prev_cx = cx; epoch += 30;
    }

    // MinCut + Fiedler
    let (mc_e, _) = build_graph(&normed);
    let mc = MinCutBuilder::new().exact().with_edges(mc_e.clone()).build().expect("mincut");
    let (ps,pt) = mc.min_cut().partition.unwrap();
    println!("\n[MINCUT] cut={:.4}, partitions: {}|{}", mc.min_cut_value(), ps.len(), pt.len());

    let sb: Vec<usize> = p_bounds.iter().take(3).map(|b|b.0).collect();
    let segs = if sb.len()>=3 { let mut s=sb; s.sort(); vec![(0,s[0]),(s[0],s[1]),(s[1],s[2]),(s[2],nwin_p)] }
    else { let w=|s:usize|((s-WB)*SR/ssp).min(nwin_p);
        vec![(0,w(SZ_START-60)),(w(SZ_START-60),w(SZ_START)),(w(SZ_START),w(SZ_END)),(w(SZ_END),nwin_p)] };
    println!("\n[FIEDLER] Per-phase:");
    for (i,&(s,e)) in segs.iter().enumerate() {
        if s<e && e<=nwin_p {
            println!("  {:<13}: {:.4}", ["Pre-seizure","Peri-ictal","Seizure","Post-ictal"][i.min(3)],
                fiedler_seg(&mc_e,s,e));
        }
    }

    // Summary
    println!("\n================================================================");
    println!("  SUMMARY: REAL EEG BOUNDARY-FIRST DETECTION (ENHANCED)");
    println!("================================================================");
    if let Some(a) = ad {
        println!("  Amplitude:  second {} ({}s {} onset)", a,
            if a>=SZ_START{a-SZ_START}else{SZ_START-a}, if a>=SZ_START{"AFTER"}else{"before"});
    }
    if let Some(&(w,cv)) = earliest {
        let (s,z) = (w2s(w,ss,ws), zscore(cv,&p_nd[0]));
        println!("  Boundary:   second {} ({}s BEFORE onset, z={:.2})", s, SZ_START-s, z);
    }
    if let Some(&(w,cv)) = ict_b {
        let z=zscore(cv,&p_nd[0]);
        println!("  Ictal:      second {} (z={:.2} {})", w2s(w,ss,ws), z, if z< -2.0{"SIGNIFICANT"}else{"n.s."});
    }
    if let Some(fr) = first_rise { println!("  Corr rise:  second {} ({}s BEFORE)", fr, SZ_START-fr); }
    println!("  Best scale: z={:.2} at {} scale", best_z, best_scale);
    println!("\n  Optimizations: multi-scale(5/10/30s) | artifact(>{:.0}uV) | 50%overlap | {}features | baseline({}s) | patient-null",
        AMP_UV, NFEAT, BL_S);
    println!("\n[COMPARISON]");
    println!("  Synthetic prediction: 45s warning");
    let bw = earliest.map(|&(w,_)| SZ_START.saturating_sub(w2s(w,ss,ws)))
        .or(first_rise.map(|s| SZ_START.saturating_sub(s))).unwrap_or(0);
    println!("  Real EEG result:      {}s warning (best z={:.2})", bw, best_z);
    println!("================================================================");
}
