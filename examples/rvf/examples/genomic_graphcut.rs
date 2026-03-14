//! Genomic Graph Cut: CNV Detection via MRF Optimization + RuVector
//!
//! Graph cut / MRF optimization for DNA anomaly detection:
//!   1. Generate synthetic chromosome as N windows (10kb each)
//!   2. Inject anomalies: CNV gains/losses, mutation hotspots, structural variants
//!   3. Extract per-window 32-dim embeddings, build MRF graph, solve s-t mincut
//!   4. Classify aberrant regions, detect cancer driver genes (TP53, BRCA1, EGFR, MYC)
//!
//! Run: cargo run --example genomic_graphcut --release

use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_types::DerivationType;
use rvf_crypto::{create_witness_chain, verify_witness_chain, shake256_256, WitnessEntry};
use tempfile::TempDir;

const FIELD_PLATFORM: u16 = 0;
const FIELD_CHROMOSOME: u16 = 1;
const FIELD_WINDOW_POS: u16 = 2;
const FIELD_CNV_LABEL: u16 = 3;

fn lcg_next(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s
}
fn lcg_f64(s: &mut u64) -> f64 { lcg_next(s); (*s >> 11) as f64 / ((1u64 << 53) as f64) }
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_f64(s).max(1e-15); let u2 = lcg_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Platform { Wgs, Wes, Panel }
impl Platform {
    fn label(&self) -> &'static str {
        match self { Platform::Wgs => "WGS", Platform::Wes => "WES", Platform::Panel => "Panel" }
    }
    fn depth(&self) -> f64 {
        match self { Platform::Wgs => 30.0, Platform::Wes => 100.0, Platform::Panel => 500.0 }
    }
    fn cv(&self) -> f64 {
        match self { Platform::Wgs => 0.15, Platform::Wes => 0.25, Platform::Panel => 0.10 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Anomaly { Normal, Gain, Loss, Loh, Hotspot, Sv }
impl Anomaly {
    fn label(&self) -> &'static str {
        match self {
            Anomaly::Normal=>"normal", Anomaly::Gain=>"gain", Anomaly::Loss=>"loss",
            Anomaly::Loh=>"LOH", Anomaly::Hotspot=>"hotspot", Anomaly::Sv=>"SV",
        }
    }
}

struct Driver { name: &'static str, start: usize, end: usize, anomaly: Anomaly }
fn drivers() -> Vec<Driver> {
    vec![
        Driver { name: "TP53",  start: 170, end: 180, anomaly: Anomaly::Loss },
        Driver { name: "BRCA1", start: 410, end: 425, anomaly: Anomaly::Loss },
        Driver { name: "EGFR",  start: 700, end: 720, anomaly: Anomaly::Gain },
        Driver { name: "MYC",   start: 820, end: 835, anomaly: Anomaly::Gain },
    ]
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Window {
    index: usize, read_depth: f64, gc: f64, mq: f64,
    var_count: u32, baf: f64, truth: Anomaly, truth_cn: f64,
}

fn generate_chromosome(n: usize, plat: Platform, seed: u64) -> Vec<Window> {
    let mut rng = seed;
    let exp = plat.depth();
    let cv = plat.cv();
    let drvs = drivers();
    let gc_wave = |i: usize| 0.42 + 0.08*(i as f64*0.01).sin() + 0.04*(i as f64*0.037).cos();
    let mut wins = Vec::with_capacity(n);
    for i in 0..n {
        let gc = (gc_wave(i) + lcg_normal(&mut rng)*0.02).clamp(0.25, 0.75);
        let gc_bias = 1.0 - 2.0*(gc - 0.45).powi(2);
        let (mut anom, mut cn) = (Anomaly::Normal, 2.0);
        for d in &drvs {
            if i >= d.start && i < d.end {
                anom = d.anomaly;
                cn = match d.anomaly { Anomaly::Gain=>4.0, Anomaly::Loss=>1.0, _=>2.0 };
            }
        }
        if anom == Anomaly::Normal {
            let r = lcg_f64(&mut rng);
            if r < 0.02      { anom=Anomaly::Gain; cn=3.0+lcg_f64(&mut rng); }
            else if r < 0.04 { anom=Anomaly::Loss; cn=0.5+lcg_f64(&mut rng)*0.5; }
            else if r < 0.05 { anom=Anomaly::Loh; }
            else if r < 0.06 { anom=Anomaly::Hotspot; }
            else if r < 0.065{ anom=Anomaly::Sv; }
        }
        let raw = exp * (cn/2.0) * gc_bias;
        let depth = (raw + lcg_normal(&mut rng)*raw*cv).max(1.0);
        let mq = match anom {
            Anomaly::Sv => 30.0+lcg_normal(&mut rng)*5.0, _ => 55.0+lcg_normal(&mut rng)*3.0,
        }.clamp(0.0, 60.0);
        let base_v = (2.0+lcg_f64(&mut rng)*3.0) as u32;
        let vc = match anom { Anomaly::Hotspot=>base_v*5+10, Anomaly::Sv=>base_v+3, _=>base_v };
        let baf = match anom {
            Anomaly::Loh  => 0.05+lcg_f64(&mut rng)*0.1,
            Anomaly::Gain => 0.33+lcg_normal(&mut rng)*0.05,
            Anomaly::Loss => lcg_f64(&mut rng)*0.15,
            _ => 0.45+lcg_normal(&mut rng)*0.05,
        }.clamp(0.0, 1.0);
        wins.push(Window { index:i, read_depth:depth, gc, mq, var_count:vc, baf, truth:anom, truth_cn:cn });
    }
    wins
}

fn extract_embedding(w: &Window, plat: Platform) -> Vec<f32> {
    let exp = plat.depth();
    let log2r = (w.read_depth/exp).max(0.01).log2();
    let baf_d = (w.baf - 0.5).abs();
    let gc_exp = (1.0 - 2.0*(w.gc-0.45).powi(2)).max(0.5);
    let gc_nd = w.read_depth / (exp * gc_exp);
    let vr = w.var_count as f64 / 10.0;
    let pc = match plat { Platform::Wgs=>0.0f32, Platform::Wes=>0.5, Platform::Panel=>1.0 };
    vec![
        log2r as f32, (w.read_depth/exp) as f32, w.read_depth.sqrt() as f32/10.0,
        (w.read_depth-exp).abs() as f32/exp as f32,
        w.baf as f32, baf_d as f32, (baf_d*4.0).min(1.0) as f32,
        if baf_d>0.15 {1.0} else {0.0},
        gc_nd as f32, w.gc as f32, (gc_nd-1.0).abs() as f32, gc_nd.log2().abs() as f32,
        vr as f32, (vr/3.0).min(3.0) as f32, if w.var_count>15 {1.0} else {0.0},
        w.mq as f32/60.0,
        (log2r*baf_d) as f32, (log2r.abs()+vr/5.0) as f32,
        (if log2r>0.3 {1.0} else if log2r < -0.3 {-1.0} else {0.0}),
        if w.mq<40.0 {1.0} else {0.0},
        pc, (w.read_depth/500.0).min(1.0) as f32, w.index as f32/1000.0,
        (w.index as f64*0.01).sin() as f32,
        (log2r*log2r) as f32, (baf_d*baf_d) as f32, (log2r*gc_nd) as f32,
        (vr*baf_d) as f32, (log2r.abs()*w.mq/60.0) as f32,
        ((gc_nd-1.0)*log2r) as f32, (w.var_count as f32).sqrt()/5.0, 0.0,
    ]
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2);
    }
    let dn = na.sqrt()*nb.sqrt();
    if dn < 1e-15 { 0.0 } else { d/dn }
}

fn unary_score(w: &Window, plat: Platform) -> f64 {
    let log2r = (w.read_depth/plat.depth()).max(0.01).log2();
    let baf_d = (w.baf - 0.5).abs();
    let ve = (w.var_count as f64 - 4.0).max(0.0)/4.0;
    let mq_p = if w.mq < 40.0 { 1.0 } else { 0.0 };
    0.5*log2r.abs() + 1.5*baf_d + 0.3*ve + 0.4*mq_p - 0.35
}

struct Edge { from: usize, to: usize, weight: f64 }

fn build_graph(embs: &[Vec<f32>], alpha: f64, beta: f64, k: usize) -> Vec<Edge> {
    let m = embs.len();
    let mut edges = Vec::new();
    for i in 0..m.saturating_sub(1) {
        edges.push(Edge { from:i, to:i+1, weight:alpha });
        edges.push(Edge { from:i+1, to:i, weight:alpha });
    }
    for i in 0..m {
        let mut sims: Vec<(usize,f64)> = (0..m)
            .filter(|&j| (j as isize - i as isize).unsigned_abs() > 2)
            .map(|j| (j, cosine_sim(&embs[i], &embs[j]).max(0.0))).collect();
        sims.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        for &(j, s) in sims.iter().take(k) {
            if s > 0.1 { edges.push(Edge { from:i, to:j, weight:beta*s }); }
        }
    }
    edges
}

fn solve_mincut(lambdas: &[f64], edges: &[Edge], gamma: f64) -> Vec<bool> {
    let m = lambdas.len();
    let (s, t, n) = (m, m+1, m+2);
    let mut adj: Vec<Vec<(usize,usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();
    let ae = |adj: &mut Vec<Vec<(usize,usize)>>, caps: &mut Vec<f64>, u:usize, v:usize, c:f64| {
        let idx = caps.len(); caps.push(c); caps.push(0.0);
        adj[u].push((v, idx)); adj[v].push((u, idx+1));
    };
    for i in 0..m {
        let p0 = lambdas[i].max(0.0); let p1 = (-lambdas[i]).max(0.0);
        if p0 > 1e-12 { ae(&mut adj, &mut caps, s, i, p0); }
        if p1 > 1e-12 { ae(&mut adj, &mut caps, i, t, p1); }
    }
    for e in edges {
        let c = gamma*e.weight;
        if c > 1e-12 { ae(&mut adj, &mut caps, e.from, e.to, c); }
    }
    loop {
        let mut par: Vec<Option<(usize,usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        vis[s] = true; q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t { break; }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 { vis[v]=true; par[v]=Some((u,ei)); q.push_back(v); }
            }
        }
        if !vis[t] { break; }
        let mut bn = f64::MAX;
        let mut v = t;
        while let Some((u, ei)) = par[v] { bn = bn.min(caps[ei]); v = u; }
        v = t;
        while let Some((u, ei)) = par[v] { caps[ei]-=bn; caps[ei^1]+=bn; v = u; }
    }
    let mut reach = vec![false; n];
    let mut stk = vec![s]; reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] {
            if !reach[v] && caps[ei]>1e-15 { reach[v]=true; stk.push(v); }
        }
    }
    (0..m).map(|i| reach[i]).collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Call { Normal, Gain, Loss, Loh }

fn classify(w: &Window, plat: Platform) -> Call {
    let log2r = (w.read_depth/plat.depth()).max(0.01).log2();
    let bd = (w.baf - 0.5).abs();
    if bd > 0.2 && log2r.abs() < 0.2 { Call::Loh }
    else if log2r > 0.25 { Call::Gain }
    else if log2r < -0.25 { Call::Loss }
    else if bd > 0.15 { Call::Loh }
    else { Call::Normal }
}

fn threshold_detect(wins: &[Window], plat: Platform) -> Vec<bool> {
    wins.iter().map(|w| {
        let lr = (w.read_depth/plat.depth()).max(0.01).log2();
        lr.abs() > 0.35 || (w.baf-0.5).abs() > 0.2 || w.var_count > 15
    }).collect()
}

fn evaluate(wins: &[Window], calls: &[bool], types: &[Call], _plat: Platform)
    -> (f64, f64, f64, f64) // sens, spec, bp_acc, class_acc
{
    let (mut tp,mut fp,mut tn,mut fn_) = (0,0,0,0);
    let (mut cc, mut ct) = (0usize, 0usize);
    for (i, w) in wins.iter().enumerate() {
        let truth = w.truth != Anomaly::Normal;
        match (truth, calls[i]) {
            (true,true)=>{tp+=1}, (false,true)=>{fp+=1},
            (true,false)=>{fn_+=1}, (false,false)=>{tn+=1},
        }
        if calls[i] && truth {
            ct += 1;
            let tc = match w.truth {
                Anomaly::Gain=>Call::Gain, Anomaly::Loss=>Call::Loss,
                Anomaly::Loh=>Call::Loh, _=>Call::Normal,
            };
            if types[i] == tc { cc += 1; }
        }
    }
    let sens = if tp+fn_>0 { tp as f64/(tp+fn_) as f64 } else { 0.0 };
    let spec = if tn+fp>0 { tn as f64/(tn+fp) as f64 } else { 0.0 };
    let drvs = drivers();
    let mut bpe = Vec::new();
    for d in &drvs {
        let mut best = usize::MAX;
        for i in 1..calls.len() {
            if calls[i] != calls[i-1] {
                best = best.min((i as isize - d.start as isize).unsigned_abs())
                           .min((i as isize - d.end as isize).unsigned_abs());
            }
        }
        bpe.push(best as f64);
    }
    let bp = if bpe.is_empty() { 0.0 } else { 1.0/(1.0+bpe.iter().sum::<f64>()/bpe.len() as f64) };
    let ca = if ct>0 { cc as f64/ct as f64 } else { 0.0 };
    (sens, spec, bp, ca)
}

fn main() {
    println!("=== Genomic Graph Cut: CNV Detection Pipeline ===\n");
    let (dim, n_win) = (32, 1000);
    let (alpha, beta, gamma, k_nn) = (0.3, 0.15, 0.4, 3usize);
    let platforms = [Platform::Wgs, Platform::Wes, Platform::Panel];
    let tmp = TempDir::new().expect("tmpdir");
    let opts = RvfOptions { dimension: dim as u16, metric: DistanceMetric::Cosine, ..Default::default() };
    let mut store = RvfStore::create(&tmp.path().join("genomic.rvdna"), opts).expect("create");
    let (mut vecs, mut ids, mut meta): (Vec<Vec<f32>>, Vec<u64>, Vec<MetadataEntry>) =
        (Vec::new(), Vec::new(), Vec::new());
    let drvs = drivers();

    for (pi, &plat) in platforms.iter().enumerate() {
        println!("--- Platform: {} ({}x) ---\n", plat.label(), plat.depth());
        let seed = 42 + pi as u64 * 997;
        let wins = generate_chromosome(n_win, plat, seed);
        let n_ab = wins.iter().filter(|w| w.truth != Anomaly::Normal).count();
        println!("  {} windows, {} aberrant ({:.1}%)", n_win, n_ab, n_ab as f64/n_win as f64*100.0);
        for a in &[Anomaly::Gain, Anomaly::Loss, Anomaly::Loh, Anomaly::Hotspot, Anomaly::Sv] {
            let c = wins.iter().filter(|w| w.truth == *a).count();
            if c > 0 { println!("    {}: {}", a.label(), c); }
        }
        let embs: Vec<Vec<f32>> = wins.iter().map(|w| extract_embedding(w, plat)).collect();
        let lam: Vec<f64> = wins.iter().map(|w| unary_score(w, plat)).collect();
        let edges = build_graph(&embs, alpha, beta, k_nn);
        let gc = solve_mincut(&lam, &edges, gamma);
        let ct: Vec<Call> = wins.iter().enumerate()
            .map(|(i,w)| if gc[i] { classify(w, plat) } else { Call::Normal }).collect();
        let tc = threshold_detect(&wins, plat);
        let tt: Vec<Call> = wins.iter().enumerate()
            .map(|(i,w)| if tc[i] { classify(w, plat) } else { Call::Normal }).collect();
        let (gs,gp,gb,ga) = evaluate(&wins, &gc, &ct, plat);
        let (ts,tp_,tb,ta) = evaluate(&wins, &tc, &tt, plat);
        println!("\n  Graph Cut:  sens={:.3} spec={:.3} bp={:.3} class={:.3}", gs, gp, gb, ga);
        println!("  Threshold:  sens={:.3} spec={:.3} bp={:.3} class={:.3}", ts, tp_, tb, ta);
        println!("\n  Driver Gene Detection:");
        println!("    {:>6} {:>8} {:>6} {:>8} {:>5}", "Gene","Region","Truth","Called","Det");
        for d in &drvs {
            let cc = (d.start..d.end).filter(|&i| gc[i]).count();
            let sz = d.end - d.start;
            let det = if cc as f64/sz as f64 > 0.5 { "YES" } else { "no" };
            println!("    {:>6} {:>3}-{:<4} {:>6} {:>3}/{:<3} {:>5}",
                d.name, d.start, d.end, d.anomaly.label(), cc, sz, det);
        }
        for (i, emb) in embs.iter().enumerate() {
            let id = pi as u64 * 100_000 + i as u64;
            vecs.push(emb.clone()); ids.push(id);
            meta.push(MetadataEntry { field_id: FIELD_PLATFORM,
                value: MetadataValue::String(plat.label().into()) });
            meta.push(MetadataEntry { field_id: FIELD_CHROMOSOME, value: MetadataValue::U64(1) });
            meta.push(MetadataEntry { field_id: FIELD_WINDOW_POS,
                value: MetadataValue::U64(i as u64 * 10_000) });
            meta.push(MetadataEntry { field_id: FIELD_CNV_LABEL,
                value: MetadataValue::String(wins[i].truth.label().into()) });
        }
        println!();
    }

    // RVF ingestion
    println!("--- RVF Ingestion ---");
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let ing = store.ingest_batch(&refs, &ids, Some(&meta)).expect("ingest");
    println!("  Ingested: {} (rejected: {})", ing.accepted, ing.rejected);

    // Filtered queries: EGFR region similarity
    println!("\n--- RVF Queries ---");
    let ew = generate_chromosome(n_win, Platform::Wgs, 42);
    let qe = extract_embedding(&ew[710], Platform::Wgs);
    let res = store.query(&qe, 10, &QueryOptions::default()).expect("q");
    println!("  EGFR-similar (window 710): {} results", res.len());
    for r in res.iter().take(5) {
        let p = match r.id/100_000 { 0=>"WGS", 1=>"WES", 2=>"Panel", _=>"?" };
        println!("    id={:>6} dist={:.6} plat={}", r.id, r.distance, p);
    }
    let fw = FilterExpr::Eq(FIELD_PLATFORM, FilterValue::String("WGS".into()));
    let wr = store.query(&qe, 5, &QueryOptions { filter: Some(fw), ..Default::default() }).expect("q");
    println!("  WGS-only: {}", wr.len());
    let fg = FilterExpr::Eq(FIELD_CNV_LABEL, FilterValue::String("gain".into()));
    let gr = store.query(&qe, 5, &QueryOptions { filter: Some(fg), ..Default::default() }).expect("q");
    println!("  Gain-only: {}", gr.len());

    // Witness chain for clinical provenance
    println!("\n--- Clinical Provenance ---");
    let steps = [
        ("genesis",0x01u8), ("sample_accession",0x01), ("sequencing_qc",0x02),
        ("alignment",0x02), ("depth_extract",0x02), ("gc_norm",0x02),
        ("baf_compute",0x02), ("embedding",0x02), ("graph_build",0x02),
        ("mincut",0x02), ("classify",0x02), ("driver_annot",0x02),
        ("rvf_ingest",0x08), ("report",0x01), ("seal",0x01),
    ];
    let entries: Vec<WitnessEntry> = steps.iter().enumerate().map(|(i,(s,wt))| {
        WitnessEntry { prev_hash:[0u8;32], action_hash: shake256_256(format!("gc:{}:{}",s,i).as_bytes()),
            timestamp_ns: 1_700_000_000_000_000_000 + i as u64*1_000_000_000, witness_type:*wt }
    }).collect();
    let cb = create_witness_chain(&entries);
    let ver = verify_witness_chain(&cb).expect("verify");
    println!("  {} entries, {} bytes, VALID", ver.len(), cb.len());
    for (i,(s,_)) in steps.iter().enumerate() {
        let wn = match ver[i].witness_type { 0x01=>"PROV", 0x02=>"COMP", 0x08=>"DATA", _=>"????" };
        println!("    [{:>4}] {:>2} {}", wn, i, s);
    }

    // Lineage
    println!("\n--- Lineage ---");
    let child = store.derive(&tmp.path().join("cnv_report.rvdna"), DerivationType::Filter, None).expect("derive");
    println!("  parent: {} -> child: {} (depth {})",
        hex(store.file_id()), hex(child.parent_id()), child.lineage_depth());
    child.close().expect("close");

    // Summary
    println!("\n=== Summary ===");
    println!("  {} windows x 3 platforms = {} vectors", n_win, ing.accepted);
    println!("  Drivers: {} genes | Witness: {} steps", drvs.len(), ver.len());
    println!("  alpha={:.2} beta={:.2} gamma={:.2} k={}", alpha, beta, gamma, k_nn);
    println!("  Threshold: lambda_i > 2*gamma*alpha = {:.2}", 2.0*gamma*alpha);
    store.close().expect("close");
    println!("\nDone.");
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}",x)).collect() }
