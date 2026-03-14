//! Climate Anomaly Detection via Graph Cut / MRF + RuVector
//!
//! Applies MRF/mincut optimization to environmental monitoring data:
//!   1. Generate synthetic station grid (30x40 = 1200 monitoring stations)
//!   2. Inject anomalies: heat waves, pollution spikes, drought, ocean warming,
//!      cold snaps, sensor faults
//!   3. Extract 32-dim embeddings, build spatial+similarity graph, solve mincut
//!   4. Store station embeddings in RVF with climate zone metadata
//!   5. Evaluate: precision, recall, F1, per-event detection rates
//!
//! Run: cargo run --example climate_graphcut --release

use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_types::DerivationType;
use rvf_crypto::{create_witness_chain, verify_witness_chain, shake256_256, WitnessEntry};
use tempfile::TempDir;

const ROWS: usize = 30;
const COLS: usize = 40;
const N_STATIONS: usize = ROWS * COLS;
const DIM: usize = 32;

const FIELD_REGION: u16 = 0;
const FIELD_ELEV_CLASS: u16 = 1;
const FIELD_CLIMATE_ZONE: u16 = 2;
const FIELD_ANOMALY_TYPE: u16 = 3;

fn lcg_next(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s
}
fn lcg_f64(s: &mut u64) -> f64 { lcg_next(s); (*s >> 11) as f64 / ((1u64 << 53) as f64) }
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_f64(s).max(1e-15); let u2 = lcg_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AnomalyType { Normal, HeatWave, PollutionSpike, Drought, OceanWarming, ColdSnap, SensorFault }
impl AnomalyType {
    fn label(&self) -> &'static str {
        match self {
            Self::Normal => "normal", Self::HeatWave => "heat_wave",
            Self::PollutionSpike => "pollution", Self::Drought => "drought",
            Self::OceanWarming => "ocean_warm", Self::ColdSnap => "cold_snap",
            Self::SensorFault => "sensor_fault",
        }
    }
}

#[derive(Debug, Clone)]
struct Station {
    row: usize, col: usize,
    lat: f64, lon: f64, elevation: f64,
    temperature: f64, humidity: f64, precipitation: f64,
    wind_speed: f64, aqi: f64, co2: f64,
    sst: f64, ndvi: f64, day_of_year: f64,
    is_coastal: bool, truth: AnomalyType,
}

fn climate_zone(lat: f64) -> &'static str {
    let a = lat.abs();
    if a > 60.0 { "polar" } else if a > 40.0 { "temperate" } else if a > 23.5 { "subtropical" }
    else { "tropical" }
}

fn elev_class(e: f64) -> &'static str {
    if e < 200.0 { "lowland" } else if e < 1000.0 { "highland" } else { "mountain" }
}

fn region_label(row: usize, col: usize) -> &'static str {
    match (row < ROWS / 2, col < COLS / 2) {
        (true, true) => "NW", (true, false) => "NE",
        (false, true) => "SW", _ => "SE",
    }
}

fn generate_stations(seed: u64, day: f64) -> Vec<Station> {
    let mut rng = seed;
    let mut stations = Vec::with_capacity(N_STATIONS);
    let season = (day / 365.0 * 2.0 * std::f64::consts::PI).sin();
    let season_cos = (day / 365.0 * 2.0 * std::f64::consts::PI).cos();

    // Pre-generate anomaly clusters (spatial events)
    let n_heat = 2; let n_poll = 2; let n_drought = 1; let n_cold = 1;
    let mut heat_centers = Vec::new();
    for _ in 0..n_heat {
        heat_centers.push((lcg_f64(&mut rng) * ROWS as f64, lcg_f64(&mut rng) * COLS as f64,
            3.0 + lcg_f64(&mut rng) * 4.0));
    }
    let mut poll_centers = Vec::new();
    for _ in 0..n_poll {
        poll_centers.push((lcg_f64(&mut rng) * ROWS as f64, lcg_f64(&mut rng) * COLS as f64,
            2.0 + lcg_f64(&mut rng) * 3.0));
    }
    let mut drought_centers = Vec::new();
    for _ in 0..n_drought {
        drought_centers.push((lcg_f64(&mut rng) * ROWS as f64, lcg_f64(&mut rng) * COLS as f64,
            4.0 + lcg_f64(&mut rng) * 5.0));
    }
    let mut cold_centers = Vec::new();
    for _ in 0..n_cold {
        cold_centers.push((lcg_f64(&mut rng) * ROWS as f64, lcg_f64(&mut rng) * COLS as f64,
            3.0 + lcg_f64(&mut rng) * 3.0));
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            let lat = 25.0 + (r as f64 / ROWS as f64) * 50.0;  // 25N to 75N
            let lon = -125.0 + (c as f64 / COLS as f64) * 60.0; // 125W to 65W
            let elevation = (200.0 + lcg_normal(&mut rng) * 300.0
                + 800.0 * ((r as f64 * 0.2).sin() * (c as f64 * 0.15).cos()).abs()).max(0.0);
            let is_coastal = c < 3 || c >= COLS - 3 || r < 2 || r >= ROWS - 2;

            // Seasonal baseline temperature: warmer at lower latitudes, summer peak
            let t_base = 30.0 - 0.6 * (lat - 25.0) + 12.0 * season - elevation * 0.006;
            let temperature = t_base + lcg_normal(&mut rng) * 3.0;
            let humidity = (60.0 + 15.0 * season_cos + lcg_normal(&mut rng) * 10.0).clamp(10.0, 100.0);
            let precipitation = (20.0 + 30.0 * (0.5 + 0.5 * season_cos)
                + lcg_normal(&mut rng) * 15.0).max(0.0);
            let wind_speed = (5.0 + lcg_normal(&mut rng) * 3.0).max(0.5);
            let aqi = (35.0 + lcg_normal(&mut rng) * 15.0).clamp(0.0, 500.0);
            let co2 = 420.0 + lcg_normal(&mut rng) * 5.0;
            let sst = if is_coastal {
                15.0 + 5.0 * season - 0.2 * (lat - 40.0) + lcg_normal(&mut rng) * 1.5
            } else { 0.0 };
            let ndvi = (0.5 + 0.15 * season + lcg_normal(&mut rng) * 0.1
                - elevation * 0.0001).clamp(0.0, 1.0);

            let mut truth = AnomalyType::Normal;

            // Check spatial anomaly clusters
            for &(cr, cc, rad) in &heat_centers {
                let d = ((r as f64 - cr).powi(2) + (c as f64 - cc).powi(2)).sqrt();
                if d <= rad { truth = AnomalyType::HeatWave; }
            }
            for &(cr, cc, rad) in &poll_centers {
                let d = ((r as f64 - cr).powi(2) + (c as f64 - cc).powi(2)).sqrt();
                if d <= rad { truth = AnomalyType::PollutionSpike; }
            }
            for &(cr, cc, rad) in &drought_centers {
                let d = ((r as f64 - cr).powi(2) + (c as f64 - cc).powi(2)).sqrt();
                if d <= rad { truth = AnomalyType::Drought; }
            }
            for &(cr, cc, rad) in &cold_centers {
                let d = ((r as f64 - cr).powi(2) + (c as f64 - cc).powi(2)).sqrt();
                if d <= rad { truth = AnomalyType::ColdSnap; }
            }
            // Ocean warming: coastal belt anomaly
            if truth == AnomalyType::Normal && is_coastal && lat < 45.0 && lcg_f64(&mut rng) < 0.3 {
                truth = AnomalyType::OceanWarming;
            }
            // Sensor fault: rare random
            if truth == AnomalyType::Normal && lcg_f64(&mut rng) < 0.005 {
                truth = AnomalyType::SensorFault;
            }

            // Apply anomaly effects
            let (temperature, humidity, precipitation, aqi, co2, sst, ndvi) = match truth {
                AnomalyType::HeatWave => (
                    temperature + 8.0 + lcg_normal(&mut rng) * 2.0,
                    (humidity - 20.0).max(10.0), (precipitation * 0.2).max(0.0),
                    aqi + 30.0, co2 + 10.0, sst, (ndvi - 0.15).max(0.0)),
                AnomalyType::PollutionSpike => (
                    temperature, humidity,precipitation,
                    (180.0 + lcg_f64(&mut rng) * 200.0).min(500.0),
                    co2 + 25.0 + lcg_normal(&mut rng) * 10.0, sst, ndvi),
                AnomalyType::Drought => (
                    temperature + 4.0, (humidity - 30.0).max(10.0),
                    (precipitation * 0.05).max(0.0), aqi + 15.0, co2,
                    sst, (ndvi - 0.25).max(0.0)),
                AnomalyType::OceanWarming => (
                    temperature + 2.0, humidity, precipitation, aqi, co2,
                    sst + 3.0 + lcg_normal(&mut rng) * 0.5, ndvi),
                AnomalyType::ColdSnap => (
                    temperature - 15.0 - lcg_f64(&mut rng) * 5.0,
                    humidity + 10.0, precipitation + 5.0,
                    aqi, co2, sst, (ndvi - 0.1).max(0.0)),
                AnomalyType::SensorFault => (
                    -80.0 + lcg_f64(&mut rng) * 160.0,
                    lcg_f64(&mut rng) * 200.0,
                    lcg_f64(&mut rng) * 500.0,
                    lcg_f64(&mut rng) * 500.0,
                    lcg_f64(&mut rng) * 1000.0,
                    if is_coastal { lcg_f64(&mut rng) * 50.0 } else { 0.0 },
                    lcg_f64(&mut rng)),
                AnomalyType::Normal => (temperature, humidity, precipitation, aqi, co2, sst, ndvi),
            };

            stations.push(Station {
                row: r, col: c, lat, lon, elevation, temperature, humidity,
                precipitation, wind_speed, aqi, co2, sst, ndvi, day_of_year: day,
                is_coastal, truth,
            });
        }
    }
    stations
}

fn extract_embedding(st: &Station, stats: &StationStats) -> Vec<f32> {
    let t_anom = (st.temperature - stats.t_mean) / stats.t_std.max(1e-6);
    let h_z = (st.humidity - stats.h_mean) / stats.h_std.max(1e-6);
    let p_z = (st.precipitation - stats.p_mean) / stats.p_std.max(1e-6);
    let w_z = (st.wind_speed - stats.w_mean) / stats.w_std.max(1e-6);
    let aqi_log = (st.aqi + 1.0).ln();
    let co2_dev = (st.co2 - 420.0) / 20.0;
    let sst_anom = if st.is_coastal { (st.sst - stats.sst_mean) / stats.sst_std.max(1e-6) } else { 0.0 };
    let ndvi_dev = (st.ndvi - stats.ndvi_mean) / stats.ndvi_std.max(1e-6);
    let lat_sin = (st.lat * std::f64::consts::PI / 180.0).sin();
    let lat_cos = (st.lat * std::f64::consts::PI / 180.0).cos();
    let lon_sin = (st.lon * std::f64::consts::PI / 180.0).sin();
    let lon_cos = (st.lon * std::f64::consts::PI / 180.0).cos();
    let elev_log = (st.elevation + 1.0).ln() / 8.0;
    let season_sin = (st.day_of_year / 365.0 * 2.0 * std::f64::consts::PI).sin();
    let season_cos = (st.day_of_year / 365.0 * 2.0 * std::f64::consts::PI).cos();

    vec![
        t_anom as f32, h_z as f32, p_z as f32, w_z as f32,           // 0-3: z-scores
        aqi_log as f32, co2_dev as f32, sst_anom as f32, ndvi_dev as f32, // 4-7: derived
        lat_sin as f32, lat_cos as f32, lon_sin as f32, lon_cos as f32,   // 8-11: position
        elev_log as f32, season_sin as f32, season_cos as f32,            // 12-14: context
        (t_anom.abs() + h_z.abs()) as f32,                                // 15: temp+humidity
        (t_anom * p_z) as f32,                                            // 16: temp*precip
        (aqi_log * co2_dev) as f32,                                       // 17: aqi*co2
        (t_anom * ndvi_dev) as f32,                                       // 18: temp*ndvi
        if st.aqi > 150.0 { 1.0 } else { 0.0 },                         // 19: pollution flag
        if t_anom > 2.0 { 1.0 } else if t_anom < -2.0 { -1.0 } else { 0.0 }, // 20: temp flag
        (t_anom * t_anom) as f32,                                         // 21: t^2
        (h_z * h_z) as f32,                                              // 22: h^2
        (p_z * p_z) as f32,                                              // 23: p^2
        (sst_anom * sst_anom) as f32,                                    // 24: sst^2
        (t_anom.abs() * aqi_log) as f32,                                 // 25: |t|*aqi
        (p_z * ndvi_dev) as f32,                                         // 26: precip*ndvi
        (co2_dev * t_anom) as f32,                                       // 27: co2*t
        if st.is_coastal { 1.0 } else { 0.0 },                          // 28: coastal flag
        (st.temperature / 50.0).clamp(-1.0, 1.0) as f32,                // 29: raw temp norm
        (st.aqi / 500.0) as f32,                                        // 30: raw aqi norm
        (st.ndvi) as f32,                                                // 31: raw ndvi
    ]
}

struct StationStats {
    t_mean: f64, t_std: f64, h_mean: f64, h_std: f64,
    p_mean: f64, p_std: f64, w_mean: f64, w_std: f64,
    sst_mean: f64, sst_std: f64, ndvi_mean: f64, ndvi_std: f64,
}

fn compute_stats(stations: &[Station]) -> StationStats {
    let n = stations.len() as f64;
    let mean = |f: &dyn Fn(&Station) -> f64| stations.iter().map(f).sum::<f64>() / n;
    let std = |f: &dyn Fn(&Station) -> f64, m: f64|
        (stations.iter().map(|s| (f(s) - m).powi(2)).sum::<f64>() / n).sqrt();
    let tm = mean(&|s| s.temperature); let hm = mean(&|s| s.humidity);
    let pm = mean(&|s| s.precipitation); let wm = mean(&|s| s.wind_speed);
    let coastal: Vec<&Station> = stations.iter().filter(|s| s.is_coastal).collect();
    let cn = coastal.len().max(1) as f64;
    let sm = coastal.iter().map(|s| s.sst).sum::<f64>() / cn;
    let ss = (coastal.iter().map(|s| (s.sst - sm).powi(2)).sum::<f64>() / cn).sqrt();
    let nm = mean(&|s| s.ndvi);
    StationStats {
        t_mean: tm, t_std: std(&|s| s.temperature, tm),
        h_mean: hm, h_std: std(&|s| s.humidity, hm),
        p_mean: pm, p_std: std(&|s| s.precipitation, pm),
        w_mean: wm, w_std: std(&|s| s.wind_speed, wm),
        sst_mean: sm, sst_std: ss,
        ndvi_mean: nm, ndvi_std: std(&|s| s.ndvi, nm),
    }
}

fn unary_score(st: &Station, stats: &StationStats) -> f64 {
    let t_z = ((st.temperature - stats.t_mean) / stats.t_std.max(1e-6)).abs();
    let h_z = ((st.humidity - stats.h_mean) / stats.h_std.max(1e-6)).abs();
    let p_z = ((st.precipitation - stats.p_mean) / stats.p_std.max(1e-6)).abs();
    let aqi_f = if st.aqi > 100.0 { (st.aqi - 100.0) / 100.0 } else { 0.0 };
    let co2_f = ((st.co2 - 420.0).abs() / 20.0).max(0.0);
    let sst_z = if st.is_coastal {
        ((st.sst - stats.sst_mean) / stats.sst_std.max(1e-6)).abs()
    } else { 0.0 };
    let ndvi_z = ((st.ndvi - stats.ndvi_mean) / stats.ndvi_std.max(1e-6)).abs();
    0.3 * t_z + 0.15 * h_z + 0.1 * p_z + 0.2 * aqi_f + 0.1 * co2_f
        + 0.15 * sst_z + 0.1 * ndvi_z - 1.2
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2);
    }
    let dn = na.sqrt() * nb.sqrt();
    if dn < 1e-15 { 0.0 } else { d / dn }
}

struct Edge { from: usize, to: usize, weight: f64 }

fn build_graph(stations: &[Station], embs: &[Vec<f32>], alpha: f64, beta: f64, k: usize) -> Vec<Edge> {
    let mut edges = Vec::new();
    // 4-connected spatial adjacency
    for r in 0..ROWS { for c in 0..COLS {
        let idx = r * COLS + c;
        for &(dr, dc) in &[(0i32, 1i32), (1, 0)] {
            let (nr, nc) = (r as i32 + dr, c as i32 + dc);
            if nr >= ROWS as i32 || nc >= COLS as i32 { continue; }
            let nidx = nr as usize * COLS + nc as usize;
            let dist = ((stations[idx].lat - stations[nidx].lat).powi(2)
                + (stations[idx].lon - stations[nidx].lon).powi(2)).sqrt();
            let grad = (stations[idx].temperature - stations[nidx].temperature).abs()
                + (stations[idx].aqi - stations[nidx].aqi).abs() * 0.01;
            let w = alpha * (1.0 / (1.0 + dist)) * (-grad * 0.05).exp();
            edges.push(Edge { from: idx, to: nidx, weight: w });
            edges.push(Edge { from: nidx, to: idx, weight: w });
        }
    }}
    // kNN similarity edges from embeddings
    for i in 0..embs.len() {
        let mut sims: Vec<(usize, f64)> = (0..embs.len())
            .filter(|&j| {
                let (ri, ci) = (i / COLS, i % COLS);
                let (rj, cj) = (j / COLS, j % COLS);
                (ri as isize - rj as isize).unsigned_abs() + (ci as isize - cj as isize).unsigned_abs() > 2
            })
            .map(|j| (j, cosine_sim(&embs[i], &embs[j]).max(0.0))).collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(j, s) in sims.iter().take(k) {
            if s > 0.1 { edges.push(Edge { from: i, to: j, weight: beta * s }); }
        }
    }
    edges
}

fn solve_mincut(lambdas: &[f64], edges: &[Edge], gamma: f64) -> Vec<bool> {
    let m = lambdas.len();
    let (s, t, n) = (m, m + 1, m + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();
    let ae = |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>, u: usize, v: usize, c: f64| {
        let idx = caps.len(); caps.push(c); caps.push(0.0);
        adj[u].push((v, idx)); adj[v].push((u, idx + 1));
    };
    for i in 0..m {
        let p0 = lambdas[i].max(0.0); let p1 = (-lambdas[i]).max(0.0);
        if p0 > 1e-12 { ae(&mut adj, &mut caps, s, i, p0); }
        if p1 > 1e-12 { ae(&mut adj, &mut caps, i, t, p1); }
    }
    for e in edges {
        let c = gamma * e.weight;
        if c > 1e-12 { ae(&mut adj, &mut caps, e.from, e.to, c); }
    }
    loop {
        let mut par: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        vis[s] = true; q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t { break; }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 { vis[v] = true; par[v] = Some((u, ei)); q.push_back(v); }
            }
        }
        if !vis[t] { break; }
        let mut bn = f64::MAX;
        let mut v = t;
        while let Some((u, ei)) = par[v] { bn = bn.min(caps[ei]); v = u; }
        v = t;
        while let Some((u, ei)) = par[v] { caps[ei] -= bn; caps[ei ^ 1] += bn; v = u; }
    }
    let mut reach = vec![false; n];
    let mut stk = vec![s]; reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] {
            if !reach[v] && caps[ei] > 1e-15 { reach[v] = true; stk.push(v); }
        }
    }
    (0..m).map(|i| reach[i]).collect()
}

fn threshold_detect(stations: &[Station], stats: &StationStats) -> Vec<bool> {
    stations.iter().map(|st| {
        let t_z = ((st.temperature - stats.t_mean) / stats.t_std.max(1e-6)).abs();
        let aqi_hi = st.aqi > 150.0;
        let co2_hi = (st.co2 - 420.0).abs() > 40.0;
        let ndvi_lo = st.ndvi < stats.ndvi_mean - 2.5 * stats.ndvi_std;
        t_z > 3.0 || aqi_hi || co2_hi || ndvi_lo
    }).collect()
}

struct Metrics { precision: f64, recall: f64, f1: f64, fpr: f64 }

fn evaluate(stations: &[Station], pred: &[bool]) -> Metrics {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0, 0, 0);
    for (i, st) in stations.iter().enumerate() {
        let truth = st.truth != AnomalyType::Normal;
        match (truth, pred[i]) {
            (true, true) => tp += 1, (false, true) => fp += 1,
            (true, false) => fn_ += 1, (false, false) => tn += 1,
        }
    }
    let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let rec = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
    let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
    let fpr = if tn + fp > 0 { fp as f64 / (tn + fp) as f64 } else { 0.0 };
    Metrics { precision: prec, recall: rec, f1, fpr }
}

fn per_event_detection(stations: &[Station], pred: &[bool]) -> Vec<(&'static str, usize, usize)> {
    let types = [
        AnomalyType::HeatWave, AnomalyType::PollutionSpike, AnomalyType::Drought,
        AnomalyType::OceanWarming, AnomalyType::ColdSnap, AnomalyType::SensorFault,
    ];
    types.iter().map(|&at| {
        let total = stations.iter().filter(|s| s.truth == at).count();
        let detected = stations.iter().enumerate()
            .filter(|(i, s)| s.truth == at && pred[*i]).count();
        (at.label(), detected, total)
    }).collect()
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

fn main() {
    println!("=== Climate Anomaly Detection via Graph Cut / MRF ===\n");
    let (alpha, beta, gamma, k_nn) = (0.25, 0.12, 0.5, 3usize);
    let days = [172.0, 355.0]; // summer solstice, late December
    let day_labels = ["Summer", "Winter"];

    let tmp = TempDir::new().expect("tmpdir");
    let opts = RvfOptions { dimension: DIM as u16, metric: DistanceMetric::Cosine, ..Default::default() };
    let mut store = RvfStore::create(&tmp.path().join("climate.rvenv"), opts).expect("create");

    println!("  Grid: {}x{} = {} stations | Dim: {} | alpha={} beta={} gamma={} k={}\n",
        ROWS, COLS, N_STATIONS, DIM, alpha, beta, gamma, k_nn);

    let (mut all_vecs, mut all_ids, mut all_meta): (Vec<Vec<f32>>, Vec<u64>, Vec<MetadataEntry>) =
        (Vec::new(), Vec::new(), Vec::new());

    for (si, &day) in days.iter().enumerate() {
        let seed = 42 + si as u64 * 7919;
        let stations = generate_stations(seed, day);
        let n_anom = stations.iter().filter(|s| s.truth != AnomalyType::Normal).count();
        println!("--- Season: {} (day {}) ---", day_labels[si], day as u32);
        println!("  {} stations, {} anomalous ({:.1}%)", N_STATIONS, n_anom,
            n_anom as f64 / N_STATIONS as f64 * 100.0);
        for at in &[AnomalyType::HeatWave, AnomalyType::PollutionSpike, AnomalyType::Drought,
                    AnomalyType::OceanWarming, AnomalyType::ColdSnap, AnomalyType::SensorFault] {
            let c = stations.iter().filter(|s| s.truth == *at).count();
            if c > 0 { println!("    {}: {}", at.label(), c); }
        }

        let stats = compute_stats(&stations);
        let embs: Vec<Vec<f32>> = stations.iter().map(|s| extract_embedding(s, &stats)).collect();
        let lam: Vec<f64> = stations.iter().map(|s| unary_score(s, &stats)).collect();
        let edges = build_graph(&stations, &embs, alpha, beta, k_nn);
        let gc = solve_mincut(&lam, &edges, gamma);
        let tc = threshold_detect(&stations, &stats);

        let gc_m = evaluate(&stations, &gc);
        let tc_m = evaluate(&stations, &tc);
        println!("\n  {:>12} {:>8} {:>8} {:>8} {:>8}", "Method", "Prec", "Recall", "F1", "FPR");
        println!("  {:->12} {:->8} {:->8} {:->8} {:->8}", "", "", "", "", "");
        println!("  {:>12} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            "Graph Cut", gc_m.precision, gc_m.recall, gc_m.f1, gc_m.fpr);
        println!("  {:>12} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            "Threshold", tc_m.precision, tc_m.recall, tc_m.f1, tc_m.fpr);

        println!("\n  Per-event detection (Graph Cut):");
        println!("  {:>14} {:>6} {:>6} {:>8}", "Event", "Det", "Total", "Rate");
        println!("  {:->14} {:->6} {:->6} {:->8}", "", "", "", "");
        for (label, det, total) in per_event_detection(&stations, &gc) {
            if total > 0 {
                println!("  {:>14} {:>6} {:>6} {:>7.1}%", label, det, total,
                    det as f64 / total as f64 * 100.0);
            }
        }

        // Spatial coherence: count contiguous anomaly regions in graph cut
        let mut visited = vec![false; N_STATIONS];
        let mut n_regions = 0usize;
        for i in 0..N_STATIONS {
            if gc[i] && !visited[i] {
                n_regions += 1;
                let mut stk = vec![i];
                while let Some(u) = stk.pop() {
                    if visited[u] { continue; }
                    visited[u] = true;
                    let (ur, uc) = (u / COLS, u % COLS);
                    for &(dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let (nr, nc) = (ur as i32 + dr, uc as i32 + dc);
                        if nr >= 0 && nr < ROWS as i32 && nc >= 0 && nc < COLS as i32 {
                            let ni = nr as usize * COLS + nc as usize;
                            if gc[ni] && !visited[ni] { stk.push(ni); }
                        }
                    }
                }
            }
        }
        let gc_count = gc.iter().filter(|&&x| x).count();
        println!("\n  Spatial coherence: {} anomaly regions ({} flagged stations)",
            n_regions, gc_count);

        // Store embeddings with metadata
        for (i, emb) in embs.iter().enumerate() {
            let id = si as u64 * 100_000 + i as u64;
            all_vecs.push(emb.clone()); all_ids.push(id);
            all_meta.push(MetadataEntry { field_id: FIELD_REGION,
                value: MetadataValue::String(region_label(stations[i].row, stations[i].col).into()) });
            all_meta.push(MetadataEntry { field_id: FIELD_ELEV_CLASS,
                value: MetadataValue::String(elev_class(stations[i].elevation).into()) });
            all_meta.push(MetadataEntry { field_id: FIELD_CLIMATE_ZONE,
                value: MetadataValue::String(climate_zone(stations[i].lat).into()) });
            all_meta.push(MetadataEntry { field_id: FIELD_ANOMALY_TYPE,
                value: MetadataValue::String(stations[i].truth.label().into()) });
        }
        println!();
    }

    // RVF ingestion
    println!("--- RVF Ingestion ---");
    let refs: Vec<&[f32]> = all_vecs.iter().map(|v| v.as_slice()).collect();
    let ing = store.ingest_batch(&refs, &all_ids, Some(&all_meta)).expect("ingest");
    println!("  Ingested: {} embeddings (rejected: {})", ing.accepted, ing.rejected);

    // Filtered queries
    println!("\n--- RVF Queries ---");
    let qv = all_vecs[0].clone();
    for zone in &["polar", "temperate", "subtropical", "tropical"] {
        let f = FilterExpr::Eq(FIELD_CLIMATE_ZONE, FilterValue::String(zone.to_string()));
        let r = store.query(&qv, 5, &QueryOptions { filter: Some(f), ..Default::default() }).expect("q");
        println!("  {:<12} -> {} results (top dist: {:.4})", zone, r.len(),
            r.first().map(|x| x.distance).unwrap_or(0.0));
    }
    let fa = FilterExpr::Eq(FIELD_ANOMALY_TYPE, FilterValue::String("heat_wave".into()));
    let hr = store.query(&qv, 5, &QueryOptions { filter: Some(fa), ..Default::default() }).expect("q");
    println!("  heat_wave    -> {} results", hr.len());

    // Witness chain for environmental reporting audit
    println!("\n--- Witness Chain (Environmental Audit) ---");
    let steps = [
        ("genesis", 0x01u8), ("observation", 0x01), ("qc_check", 0x02),
        ("normalize", 0x02), ("embed", 0x02), ("spatial_graph", 0x02),
        ("mincut", 0x02), ("classify", 0x02), ("per_event_eval", 0x02),
        ("alert", 0x01), ("rvf_ingest", 0x08), ("similarity", 0x02),
        ("lineage", 0x01), ("seal", 0x01),
    ];
    let entries: Vec<WitnessEntry> = steps.iter().enumerate().map(|(i, (s, wt))| WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(format!("climate_gc:{}:{}", s, i).as_bytes()),
        timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
        witness_type: *wt,
    }).collect();
    let chain = create_witness_chain(&entries);
    let ver = verify_witness_chain(&chain).expect("verify");
    println!("  {} entries, {} bytes, VALID", ver.len(), chain.len());
    for (i, (s, _)) in steps.iter().enumerate() {
        let wn = match ver[i].witness_type { 0x01 => "PROV", 0x02 => "COMP", 0x08 => "DATA", _ => "????" };
        println!("    [{:>4}] {:>2} -> {}", wn, i, s);
    }

    // Lineage
    println!("\n--- Lineage ---");
    let child = store.derive(&tmp.path().join("climate_report.rvenv"),
        DerivationType::Filter, None).expect("derive");
    println!("  parent: {} -> child: {} (depth {})",
        hex(store.file_id()), hex(child.parent_id()), child.lineage_depth());
    child.close().expect("close");

    // Summary
    println!("\n=== Summary ===");
    println!("  {} stations x {} seasons = {} embeddings", N_STATIONS, days.len(), ing.accepted);
    println!("  Anomaly types: 6 | Witness: {} steps | Climate zones: 4", ver.len());
    println!("  alpha={:.2} beta={:.2} gamma={:.2} k={}", alpha, beta, gamma, k_nn);
    store.close().expect("close");
    println!("\nDone.");
}
