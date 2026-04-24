//! Climate Anomaly Detection via Graph Cut / MRF + RuVector
//!
//! MRF/mincut optimization for environmental monitoring:
//!   1. Generate 30x40 = 1200 station grid with realistic climate variables
//!   2. Inject anomalies: heat waves, pollution, drought, ocean warming, cold snaps, sensor faults
//!   3. Extract 32-dim embeddings, build spatial+similarity graph, solve s-t mincut
//!   4. Store embeddings in RVF with climate zone metadata, witness chain for audit
//!
//! Run: cargo run --example climate_graphcut --release

use rvf_crypto::{create_witness_chain, shake256_256, verify_witness_chain, WitnessEntry};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_types::DerivationType;
use tempfile::TempDir;

const ROWS: usize = 30;
const COLS: usize = 40;
const N: usize = ROWS * COLS;
const DIM: usize = 32;
const FIELD_REGION: u16 = 0;
const FIELD_ELEV: u16 = 1;
const FIELD_ZONE: u16 = 2;
const FIELD_ANOM: u16 = 3;

fn lcg_next(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}
fn lcg_f64(s: &mut u64) -> f64 {
    lcg_next(s);
    (*s >> 11) as f64 / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_f64(s).max(1e-15);
    let u2 = lcg_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Anom {
    Normal,
    HeatWave,
    Pollution,
    Drought,
    OceanWarm,
    ColdSnap,
    SensorFault,
}
impl Anom {
    fn label(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::HeatWave => "heat_wave",
            Self::Pollution => "pollution",
            Self::Drought => "drought",
            Self::OceanWarm => "ocean_warm",
            Self::ColdSnap => "cold_snap",
            Self::SensorFault => "sensor_fault",
        }
    }
}

#[derive(Debug, Clone)]
struct Station {
    row: usize,
    col: usize,
    lat: f64,
    lon: f64,
    elev: f64,
    temp: f64,
    hum: f64,
    precip: f64,
    wind: f64,
    aqi: f64,
    co2: f64,
    sst: f64,
    ndvi: f64,
    day: f64,
    coastal: bool,
    truth: Anom,
}

fn zone(lat: f64) -> &'static str {
    let a = lat.abs();
    if a > 60.0 {
        "polar"
    } else if a > 40.0 {
        "temperate"
    } else if a > 23.5 {
        "subtropical"
    } else {
        "tropical"
    }
}
fn elev_class(e: f64) -> &'static str {
    if e < 200.0 {
        "lowland"
    } else if e < 1000.0 {
        "highland"
    } else {
        "mountain"
    }
}
fn region(r: usize, c: usize) -> &'static str {
    match (r < ROWS / 2, c < COLS / 2) {
        (true, true) => "NW",
        (true, false) => "NE",
        (false, true) => "SW",
        _ => "SE",
    }
}

fn gen_clusters(rng: &mut u64, n: usize, rmin: f64, rmax: f64) -> Vec<(f64, f64, f64)> {
    (0..n)
        .map(|_| {
            (
                lcg_f64(rng) * ROWS as f64,
                lcg_f64(rng) * COLS as f64,
                rmin + lcg_f64(rng) * (rmax - rmin),
            )
        })
        .collect()
}
// Target ~4-6% anomaly rate: small spatial clusters + rare point anomalies

fn generate_stations(seed: u64, day: f64) -> Vec<Station> {
    let mut rng = seed;
    let pi2 = 2.0 * std::f64::consts::PI;
    let ss = (day / 365.0 * pi2).sin();
    let sc = (day / 365.0 * pi2).cos();
    let heat = gen_clusters(&mut rng, 1, 1.5, 3.0);
    let poll = gen_clusters(&mut rng, 1, 1.0, 2.5);
    let drought = gen_clusters(&mut rng, 1, 1.5, 3.0);
    let cold = gen_clusters(&mut rng, 1, 1.5, 2.5);
    let mut out = Vec::with_capacity(N);
    for r in 0..ROWS {
        for c in 0..COLS {
            let lat = 25.0 + (r as f64 / ROWS as f64) * 50.0;
            let lon = -125.0 + (c as f64 / COLS as f64) * 60.0;
            let elev = (200.0
                + lcg_normal(&mut rng) * 300.0
                + 800.0 * ((r as f64 * 0.2).sin() * (c as f64 * 0.15).cos()).abs())
            .max(0.0);
            let coastal = c < 3 || c >= COLS - 3 || r < 2 || r >= ROWS - 2;
            let tb = 30.0 - 0.6 * (lat - 25.0) + 12.0 * ss - elev * 0.006;
            let temp = tb + lcg_normal(&mut rng) * 3.0;
            let hum = (60.0 + 15.0 * sc + lcg_normal(&mut rng) * 10.0).clamp(10.0, 100.0);
            let precip = (20.0 + 30.0 * (0.5 + 0.5 * sc) + lcg_normal(&mut rng) * 15.0).max(0.0);
            let wind = (5.0 + lcg_normal(&mut rng) * 3.0).max(0.5);
            let aqi = (35.0 + lcg_normal(&mut rng) * 15.0).clamp(0.0, 500.0);
            let co2 = 420.0 + lcg_normal(&mut rng) * 5.0;
            let sst = if coastal {
                15.0 + 5.0 * ss - 0.2 * (lat - 40.0) + lcg_normal(&mut rng) * 1.5
            } else {
                0.0
            };
            let ndvi =
                (0.5 + 0.15 * ss + lcg_normal(&mut rng) * 0.1 - elev * 0.0001).clamp(0.0, 1.0);
            let in_cluster = |centers: &[(f64, f64, f64)]| {
                centers.iter().any(|&(cr, cc, rad)| {
                    ((r as f64 - cr).powi(2) + (c as f64 - cc).powi(2)).sqrt() <= rad
                })
            };
            let mut truth = Anom::Normal;
            if in_cluster(&heat) {
                truth = Anom::HeatWave;
            } else if in_cluster(&poll) {
                truth = Anom::Pollution;
            } else if in_cluster(&drought) {
                truth = Anom::Drought;
            } else if in_cluster(&cold) {
                truth = Anom::ColdSnap;
            } else if coastal && lat < 45.0 && lcg_f64(&mut rng) < 0.08 {
                truth = Anom::OceanWarm;
            }
            if truth == Anom::Normal && lcg_f64(&mut rng) < 0.005 {
                truth = Anom::SensorFault;
            }
            let (temp, hum, precip, aqi, co2, sst, ndvi) = match truth {
                Anom::HeatWave => (
                    temp + 8.0 + lcg_normal(&mut rng) * 2.0,
                    (hum - 20.0).max(10.0),
                    (precip * 0.2).max(0.0),
                    aqi + 30.0,
                    co2 + 10.0,
                    sst,
                    (ndvi - 0.15).max(0.0),
                ),
                Anom::Pollution => (
                    temp,
                    hum,
                    precip,
                    (180.0 + lcg_f64(&mut rng) * 200.0).min(500.0),
                    co2 + 25.0 + lcg_normal(&mut rng) * 10.0,
                    sst,
                    ndvi,
                ),
                Anom::Drought => (
                    temp + 4.0,
                    (hum - 30.0).max(10.0),
                    (precip * 0.05).max(0.0),
                    aqi + 15.0,
                    co2,
                    sst,
                    (ndvi - 0.25).max(0.0),
                ),
                Anom::OceanWarm => (
                    temp + 2.0,
                    hum,
                    precip,
                    aqi,
                    co2,
                    sst + 3.0 + lcg_normal(&mut rng) * 0.5,
                    ndvi,
                ),
                Anom::ColdSnap => (
                    temp - 15.0 - lcg_f64(&mut rng) * 5.0,
                    hum + 10.0,
                    precip + 5.0,
                    aqi,
                    co2,
                    sst,
                    (ndvi - 0.1).max(0.0),
                ),
                Anom::SensorFault => (
                    -80.0 + lcg_f64(&mut rng) * 160.0,
                    lcg_f64(&mut rng) * 200.0,
                    lcg_f64(&mut rng) * 500.0,
                    lcg_f64(&mut rng) * 500.0,
                    lcg_f64(&mut rng) * 1000.0,
                    if coastal {
                        lcg_f64(&mut rng) * 50.0
                    } else {
                        0.0
                    },
                    lcg_f64(&mut rng),
                ),
                Anom::Normal => (temp, hum, precip, aqi, co2, sst, ndvi),
            };
            out.push(Station {
                row: r,
                col: c,
                lat,
                lon,
                elev,
                temp,
                hum,
                precip,
                wind,
                aqi,
                co2,
                sst,
                ndvi,
                day,
                coastal,
                truth,
            });
        }
    }
    out
}

struct Stats {
    tm: f64,
    ts: f64,
    hm: f64,
    hs: f64,
    pm: f64,
    ps: f64,
    wm: f64,
    ws: f64,
    sm: f64,
    ss: f64,
    nm: f64,
    ns: f64,
}

fn stats(st: &[Station]) -> Stats {
    let n = st.len() as f64;
    let m = |f: &dyn Fn(&Station) -> f64| st.iter().map(f).sum::<f64>() / n;
    let s = |f: &dyn Fn(&Station) -> f64, v: f64| {
        (st.iter().map(|x| (f(x) - v).powi(2)).sum::<f64>() / n).sqrt()
    };
    let (tm, hm, pm, wm) = (
        m(&|x| x.temp),
        m(&|x| x.hum),
        m(&|x| x.precip),
        m(&|x| x.wind),
    );
    let cv: Vec<_> = st.iter().filter(|x| x.coastal).collect();
    let cn = cv.len().max(1) as f64;
    let sm = cv.iter().map(|x| x.sst).sum::<f64>() / cn;
    let ss = (cv.iter().map(|x| (x.sst - sm).powi(2)).sum::<f64>() / cn).sqrt();
    let nm = m(&|x| x.ndvi);
    Stats {
        tm,
        ts: s(&|x| x.temp, tm),
        hm,
        hs: s(&|x| x.hum, hm),
        pm,
        ps: s(&|x| x.precip, pm),
        wm,
        ws: s(&|x| x.wind, wm),
        sm,
        ss,
        nm,
        ns: s(&|x| x.ndvi, nm),
    }
}

fn embed(st: &Station, s: &Stats) -> Vec<f32> {
    let tz = (st.temp - s.tm) / s.ts.max(1e-6);
    let hz = (st.hum - s.hm) / s.hs.max(1e-6);
    let pz = (st.precip - s.pm) / s.ps.max(1e-6);
    let wz = (st.wind - s.wm) / s.ws.max(1e-6);
    let al = (st.aqi + 1.0).ln();
    let cd = (st.co2 - 420.0) / 20.0;
    let sa = if st.coastal {
        (st.sst - s.sm) / s.ss.max(1e-6)
    } else {
        0.0
    };
    let nd = (st.ndvi - s.nm) / s.ns.max(1e-6);
    let (pi, d2r) = (std::f64::consts::PI, std::f64::consts::PI / 180.0);
    vec![
        tz as f32,
        hz as f32,
        pz as f32,
        wz as f32,
        al as f32,
        cd as f32,
        sa as f32,
        nd as f32,
        (st.lat * d2r).sin() as f32,
        (st.lat * d2r).cos() as f32,
        (st.lon * d2r).sin() as f32,
        (st.lon * d2r).cos() as f32,
        ((st.elev + 1.0).ln() / 8.0) as f32,
        (st.day / 365.0 * 2.0 * pi).sin() as f32,
        (st.day / 365.0 * 2.0 * pi).cos() as f32,
        (tz.abs() + hz.abs()) as f32,
        (tz * pz) as f32,
        (al * cd) as f32,
        (tz * nd) as f32,
        if st.aqi > 150.0 { 1.0 } else { 0.0 },
        if tz > 2.0 {
            1.0
        } else if tz < -2.0 {
            -1.0
        } else {
            0.0
        },
        (tz * tz) as f32,
        (hz * hz) as f32,
        (pz * pz) as f32,
        (sa * sa) as f32,
        (tz.abs() * al) as f32,
        (pz * nd) as f32,
        (cd * tz) as f32,
        if st.coastal { 1.0 } else { 0.0 },
        (st.temp / 50.0).clamp(-1.0, 1.0) as f32,
        (st.aqi / 500.0) as f32,
        st.ndvi as f32,
    ]
}

fn unary(st: &Station, s: &Stats) -> f64 {
    let tz = ((st.temp - s.tm) / s.ts.max(1e-6)).abs();
    let hz = ((st.hum - s.hm) / s.hs.max(1e-6)).abs();
    let pz = ((st.precip - s.pm) / s.ps.max(1e-6)).abs();
    let af = if st.aqi > 100.0 {
        (st.aqi - 100.0) / 100.0
    } else {
        0.0
    };
    let cf = ((st.co2 - 420.0).abs() / 20.0).max(0.0);
    let sz = if st.coastal {
        ((st.sst - s.sm) / s.ss.max(1e-6)).abs()
    } else {
        0.0
    };
    let nz = ((st.ndvi - s.nm) / s.ns.max(1e-6)).abs();
    0.3 * tz + 0.15 * hz + 0.1 * pz + 0.2 * af + 0.1 * cf + 0.15 * sz + 0.1 * nz - 0.95
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let dn = na.sqrt() * nb.sqrt();
    if dn < 1e-15 {
        0.0
    } else {
        d / dn
    }
}

struct Edge {
    f: usize,
    t: usize,
    w: f64,
}

fn build_graph(st: &[Station], embs: &[Vec<f32>], alpha: f64, beta: f64, k: usize) -> Vec<Edge> {
    let mut edges = Vec::new();
    for r in 0..ROWS {
        for c in 0..COLS {
            let i = r * COLS + c;
            for &(dr, dc) in &[(0i32, 1i32), (1, 0)] {
                let (nr, nc) = (r as i32 + dr, c as i32 + dc);
                if nr >= ROWS as i32 || nc >= COLS as i32 {
                    continue;
                }
                let j = nr as usize * COLS + nc as usize;
                let dist =
                    ((st[i].lat - st[j].lat).powi(2) + (st[i].lon - st[j].lon).powi(2)).sqrt();
                let grad = (st[i].temp - st[j].temp).abs() + (st[i].aqi - st[j].aqi).abs() * 0.01;
                let w = alpha * (1.0 / (1.0 + dist)) * (-grad * 0.05).exp();
                edges.push(Edge { f: i, t: j, w });
                edges.push(Edge { f: j, t: i, w });
            }
        }
    }
    for i in 0..embs.len() {
        let mut sims: Vec<(usize, f64)> = (0..embs.len())
            .filter(|&j| {
                let (ri, ci) = (i / COLS, i % COLS);
                let (rj, cj) = (j / COLS, j % COLS);
                (ri as isize - rj as isize).unsigned_abs()
                    + (ci as isize - cj as isize).unsigned_abs()
                    > 2
            })
            .map(|j| (j, cosine(&embs[i], &embs[j]).max(0.0)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(j, s) in sims.iter().take(k) {
            if s > 0.1 {
                edges.push(Edge {
                    f: i,
                    t: j,
                    w: beta * s,
                });
            }
        }
    }
    edges
}

fn solve_mincut(lam: &[f64], edges: &[Edge], gamma: f64) -> Vec<bool> {
    let m = lam.len();
    let (s, t, n) = (m, m + 1, m + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();
    let ae =
        |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>, u: usize, v: usize, c: f64| {
            let i = caps.len();
            caps.push(c);
            caps.push(0.0);
            adj[u].push((v, i));
            adj[v].push((u, i + 1));
        };
    for i in 0..m {
        let (p0, p1) = (lam[i].max(0.0), (-lam[i]).max(0.0));
        if p0 > 1e-12 {
            ae(&mut adj, &mut caps, s, i, p0);
        }
        if p1 > 1e-12 {
            ae(&mut adj, &mut caps, i, t, p1);
        }
    }
    for e in edges {
        let c = gamma * e.w;
        if c > 1e-12 {
            ae(&mut adj, &mut caps, e.f, e.t, c);
        }
    }
    loop {
        let mut par: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        vis[s] = true;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t {
                break;
            }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 {
                    vis[v] = true;
                    par[v] = Some((u, ei));
                    q.push_back(v);
                }
            }
        }
        if !vis[t] {
            break;
        }
        let mut bn = f64::MAX;
        let mut v = t;
        while let Some((u, ei)) = par[v] {
            bn = bn.min(caps[ei]);
            v = u;
        }
        v = t;
        while let Some((u, ei)) = par[v] {
            caps[ei] -= bn;
            caps[ei ^ 1] += bn;
            v = u;
        }
    }
    let mut reach = vec![false; n];
    let mut stk = vec![s];
    reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] {
            if !reach[v] && caps[ei] > 1e-15 {
                reach[v] = true;
                stk.push(v);
            }
        }
    }
    (0..m).map(|i| reach[i]).collect()
}

fn threshold_detect(st: &[Station], s: &Stats) -> Vec<bool> {
    st.iter()
        .map(|x| {
            let tz = ((x.temp - s.tm) / s.ts.max(1e-6)).abs();
            tz > 3.0 || x.aqi > 150.0 || (x.co2 - 420.0).abs() > 40.0 || x.ndvi < s.nm - 2.5 * s.ns
        })
        .collect()
}

fn evaluate(st: &[Station], pred: &[bool]) -> (f64, f64, f64, f64) {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0, 0, 0);
    for (i, x) in st.iter().enumerate() {
        match (x.truth != Anom::Normal, pred[i]) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let p = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let r = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    };
    let fpr = if tn + fp > 0 {
        fp as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    (p, r, f1, fpr)
}

const ATYPES: [Anom; 6] = [
    Anom::HeatWave,
    Anom::Pollution,
    Anom::Drought,
    Anom::OceanWarm,
    Anom::ColdSnap,
    Anom::SensorFault,
];

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    println!("=== Climate Anomaly Detection via Graph Cut / MRF ===\n");
    let (alpha, beta, gamma, knn) = (0.25, 0.12, 0.5, 3usize);
    let days = [172.0, 355.0];
    let labels = ["Summer", "Winter"];
    let tmp = TempDir::new().expect("tmpdir");
    let opts = RvfOptions {
        dimension: DIM as u16,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut store = RvfStore::create(&tmp.path().join("climate.rvenv"), opts).expect("create");
    println!(
        "  Grid: {}x{}={} stations | Dim: {} | alpha={} beta={} gamma={} k={}\n",
        ROWS, COLS, N, DIM, alpha, beta, gamma, knn
    );

    let (mut av, mut ai, mut am): (Vec<Vec<f32>>, Vec<u64>, Vec<MetadataEntry>) =
        (Vec::new(), Vec::new(), Vec::new());

    for (si, &day) in days.iter().enumerate() {
        let st = generate_stations(42 + si as u64 * 7919, day);
        let na = st.iter().filter(|x| x.truth != Anom::Normal).count();
        println!("--- Season: {} (day {}) ---", labels[si], day as u32);
        println!(
            "  {} stations, {} anomalous ({:.1}%)",
            N,
            na,
            na as f64 / N as f64 * 100.0
        );
        for at in &ATYPES {
            let c = st.iter().filter(|x| x.truth == *at).count();
            if c > 0 {
                println!("    {}: {}", at.label(), c);
            }
        }
        let ss = stats(&st);
        let embs: Vec<Vec<f32>> = st.iter().map(|x| embed(x, &ss)).collect();
        let lam: Vec<f64> = st.iter().map(|x| unary(x, &ss)).collect();
        let edges = build_graph(&st, &embs, alpha, beta, knn);
        let gc = solve_mincut(&lam, &edges, gamma);
        let tc = threshold_detect(&st, &ss);
        let (gp, gr, gf, gfpr) = evaluate(&st, &gc);
        let (tp, tr, tf, tfpr) = evaluate(&st, &tc);
        println!(
            "\n  {:>12} {:>8} {:>8} {:>8} {:>8}",
            "Method", "Prec", "Recall", "F1", "FPR"
        );
        println!("  {:->12} {:->8} {:->8} {:->8} {:->8}", "", "", "", "", "");
        println!(
            "  {:>12} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            "Graph Cut", gp, gr, gf, gfpr
        );
        println!(
            "  {:>12} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            "Threshold", tp, tr, tf, tfpr
        );
        println!("\n  Per-event detection (Graph Cut):");
        println!(
            "  {:>14} {:>6} {:>6} {:>8}",
            "Event", "Det", "Total", "Rate"
        );
        println!("  {:->14} {:->6} {:->6} {:->8}", "", "", "", "");
        for at in &ATYPES {
            let total = st.iter().filter(|x| x.truth == *at).count();
            let det = st
                .iter()
                .enumerate()
                .filter(|(i, x)| x.truth == *at && gc[*i])
                .count();
            if total > 0 {
                println!(
                    "  {:>14} {:>6} {:>6} {:>7.1}%",
                    at.label(),
                    det,
                    total,
                    det as f64 / total as f64 * 100.0
                );
            }
        }
        // Spatial coherence: contiguous regions via flood fill
        let mut vis = vec![false; N];
        let mut nreg = 0usize;
        for i in 0..N {
            if gc[i] && !vis[i] {
                nreg += 1;
                let mut stk = vec![i];
                while let Some(u) = stk.pop() {
                    if vis[u] {
                        continue;
                    }
                    vis[u] = true;
                    let (ur, uc) = (u / COLS, u % COLS);
                    for &(dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let (nr, nc) = (ur as i32 + dr, uc as i32 + dc);
                        if nr >= 0 && nr < ROWS as i32 && nc >= 0 && nc < COLS as i32 {
                            let ni = nr as usize * COLS + nc as usize;
                            if gc[ni] && !vis[ni] {
                                stk.push(ni);
                            }
                        }
                    }
                }
            }
        }
        println!(
            "\n  Spatial coherence: {} regions ({} flagged)\n",
            nreg,
            gc.iter().filter(|&&x| x).count()
        );
        for (i, e) in embs.iter().enumerate() {
            let id = si as u64 * 100_000 + i as u64;
            av.push(e.clone());
            ai.push(id);
            am.push(MetadataEntry {
                field_id: FIELD_REGION,
                value: MetadataValue::String(region(st[i].row, st[i].col).into()),
            });
            am.push(MetadataEntry {
                field_id: FIELD_ELEV,
                value: MetadataValue::String(elev_class(st[i].elev).into()),
            });
            am.push(MetadataEntry {
                field_id: FIELD_ZONE,
                value: MetadataValue::String(zone(st[i].lat).into()),
            });
            am.push(MetadataEntry {
                field_id: FIELD_ANOM,
                value: MetadataValue::String(st[i].truth.label().into()),
            });
        }
    }

    println!("--- RVF Ingestion ---");
    let refs: Vec<&[f32]> = av.iter().map(|v| v.as_slice()).collect();
    let ing = store.ingest_batch(&refs, &ai, Some(&am)).expect("ingest");
    println!("  Ingested: {} (rejected: {})", ing.accepted, ing.rejected);

    println!("\n--- RVF Queries ---");
    let qv = av[0].clone();
    for z in &["polar", "temperate", "subtropical", "tropical"] {
        let f = FilterExpr::Eq(FIELD_ZONE, FilterValue::String(z.to_string()));
        let r = store
            .query(
                &qv,
                5,
                &QueryOptions {
                    filter: Some(f),
                    ..Default::default()
                },
            )
            .expect("q");
        println!(
            "  {:<12} -> {} results (top dist: {:.4})",
            z,
            r.len(),
            r.first().map(|x| x.distance).unwrap_or(0.0)
        );
    }
    let fa = FilterExpr::Eq(FIELD_ANOM, FilterValue::String("heat_wave".into()));
    let hr = store
        .query(
            &qv,
            5,
            &QueryOptions {
                filter: Some(fa),
                ..Default::default()
            },
        )
        .expect("q");
    println!("  heat_wave    -> {} results", hr.len());

    println!("\n--- Witness Chain (Environmental Audit) ---");
    let steps = [
        ("genesis", 0x01u8),
        ("observation", 0x01),
        ("qc_check", 0x02),
        ("normalize", 0x02),
        ("embed", 0x02),
        ("spatial_graph", 0x02),
        ("mincut", 0x02),
        ("classify", 0x02),
        ("per_event_eval", 0x02),
        ("alert", 0x01),
        ("rvf_ingest", 0x08),
        ("similarity", 0x02),
        ("lineage", 0x01),
        ("seal", 0x01),
    ];
    let entries: Vec<WitnessEntry> = steps
        .iter()
        .enumerate()
        .map(|(i, (s, wt))| WitnessEntry {
            prev_hash: [0u8; 32],
            action_hash: shake256_256(format!("climate_gc:{}:{}", s, i).as_bytes()),
            timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
            witness_type: *wt,
        })
        .collect();
    let chain = create_witness_chain(&entries);
    let ver = verify_witness_chain(&chain).expect("verify");
    println!("  {} entries, {} bytes, VALID", ver.len(), chain.len());
    for (i, (s, _)) in steps.iter().enumerate() {
        let wn = match ver[i].witness_type {
            0x01 => "PROV",
            0x02 => "COMP",
            0x08 => "DATA",
            _ => "????",
        };
        println!("    [{:>4}] {:>2} -> {}", wn, i, s);
    }

    println!("\n--- Lineage ---");
    let child = store
        .derive(
            &tmp.path().join("climate_report.rvenv"),
            DerivationType::Filter,
            None,
        )
        .expect("derive");
    println!(
        "  parent: {} -> child: {} (depth {})",
        hex(store.file_id()),
        hex(child.parent_id()),
        child.lineage_depth()
    );
    child.close().expect("close");

    println!("\n=== Summary ===");
    println!(
        "  {} stations x {} seasons = {} embeddings",
        N,
        days.len(),
        ing.accepted
    );
    println!(
        "  Anomaly types: 6 | Witness: {} steps | Climate zones: 4",
        ver.len()
    );
    println!(
        "  alpha={:.2} beta={:.2} gamma={:.2} k={}",
        alpha, beta, gamma, knn
    );
    store.close().expect("close");
    println!("\nDone.");
}
