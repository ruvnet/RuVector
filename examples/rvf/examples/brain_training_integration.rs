//! Brain Training Integration — Feed Real-Data Discoveries into π.ruv.io
//!
//! Connects the graph-cut discovery pipeline to the π Brain MCP server for
//! SONA pattern learning, LoRA weight federation, and knowledge sharing.
//!
//! This example:
//!   1. Runs anomaly detection on all 3 real datasets (exoplanets, earthquakes, climate)
//!   2. Packages discoveries as training experiences (state → action → reward)
//!   3. Shares them with the π Brain server via its REST API
//!   4. Triggers a SONA training cycle and reports learning metrics
//!
//! Requires: BRAIN_URL (default: https://pi.ruv.io) and PI (API key) env vars
//!
//! Run: cargo run --example brain_training_integration --release

use std::collections::VecDeque;

// ── Graph-cut solver (Edmonds-Karp BFS) ─────────────────────────────────────

fn solve_mincut(lam: &[f64], edges: &[(usize, usize, f64)], gamma: f64) -> Vec<bool> {
    let m = lam.len();
    let (s, t, n) = (m, m + 1, m + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();
    let add = |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>, u: usize, v: usize, c: f64| {
        let i = caps.len();
        caps.push(c); caps.push(0.0);
        adj[u].push((v, i)); adj[v].push((u, i + 1));
    };
    for i in 0..m {
        let (p0, p1) = (lam[i].max(0.0), (-lam[i]).max(0.0));
        if p0 > 1e-12 { add(&mut adj, &mut caps, s, i, p0); }
        if p1 > 1e-12 { add(&mut adj, &mut caps, i, t, p1); }
    }
    for &(f, to, w) in edges {
        let c = gamma * w;
        if c > 1e-12 { add(&mut adj, &mut caps, f, to, c); }
    }
    loop {
        let mut par: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = VecDeque::new();
        vis[s] = true; q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t { break; }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 { vis[v] = true; par[v] = Some((u, ei)); q.push_back(v); }
            }
        }
        if !vis[t] { break; }
        let mut bn = f64::MAX; let mut v = t;
        while let Some((_, ei)) = par[v] { bn = bn.min(caps[ei]); v = par[v].unwrap().0; }
        v = t;
        while let Some((u, ei)) = par[v] { caps[ei] -= bn; caps[ei ^ 1] += bn; v = u; }
    }
    let mut reach = vec![false; n]; let mut stk = vec![s]; reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] { if !reach[v] && caps[ei] > 1e-15 { reach[v] = true; stk.push(v); } }
    }
    (0..m).map(|i| reach[i]).collect()
}

// ── CSV helpers ─────────────────────────────────────────────────────────────

fn parse_csv_field(s: &str) -> &str { s.trim().trim_matches('"') }

fn parse_f64(s: &str) -> Option<f64> {
    let v = parse_csv_field(s);
    if v.is_empty() { None } else { v.parse().ok() }
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    for ch in line.chars() {
        if ch == '"' { in_q = !in_q; }
        else if ch == ',' && !in_q { fields.push(cur.clone()); cur.clear(); }
        else { cur.push(ch); }
    }
    fields.push(cur);
    fields
}

// ── Training experience types ───────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TrainingExperience {
    domain: String,
    state: String,
    action: String,
    reward: f64,
    category: String,
    title: String,
    content: String,
    tags: Vec<String>,
}

// ── 1. Exoplanet discoveries → training data ────────────────────────────────

fn extract_exoplanet_experiences() -> Vec<TrainingExperience> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/confirmed_planets.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    struct Planet { name: String, log_period: f64, log_radius: f64, log_mass: f64, eq_temp: f64, ecc: f64, method: String }
    let mut planets = Vec::new();
    for line in data.lines().skip(1) {
        let f = split_csv_line(line);
        if f.len() < 15 { continue; }
        let period = match parse_f64(&f[3]) { Some(v) if v > 0.0 => v, _ => continue };
        let radius = match parse_f64(&f[4]) { Some(v) if v > 0.0 => v, _ => continue };
        let mass = match parse_f64(&f[5]) { Some(v) if v > 0.0 => v, _ => continue };
        let eq_temp = match parse_f64(&f[6]) { Some(v) => v, _ => continue };
        let ecc = parse_f64(&f[7]).unwrap_or(0.0);
        planets.push(Planet {
            name: parse_csv_field(&f[0]).to_string(),
            log_period: period.ln(), log_radius: radius.ln(), log_mass: mass.ln(),
            eq_temp, ecc, method: parse_csv_field(&f[13]).to_string(),
        });
    }
    if planets.is_empty() { return Vec::new(); }

    let n = planets.len() as f64;
    let mean = |f: &dyn Fn(&Planet) -> f64| planets.iter().map(f).sum::<f64>() / n;
    let std = |f: &dyn Fn(&Planet) -> f64, m: f64|
        (planets.iter().map(|p| (f(p) - m).powi(2)).sum::<f64>() / n).sqrt();
    let (mp, mr, mm, mt, me) = (mean(&|p| p.log_period), mean(&|p| p.log_radius),
        mean(&|p| p.log_mass), mean(&|p| p.eq_temp), mean(&|p| p.ecc));
    let (sp, sr, sm, st_s, se) = (std(&|p| p.log_period, mp), std(&|p| p.log_radius, mr),
        std(&|p| p.log_mass, mm), std(&|p| p.eq_temp, mt), std(&|p| p.ecc, me));

    let scores: Vec<f64> = planets.iter().map(|p| {
        let zp = ((p.log_period - mp) / sp.max(1e-6)).abs();
        let zr = ((p.log_radius - mr) / sr.max(1e-6)).abs();
        let zm = ((p.log_mass - mm) / sm.max(1e-6)).abs();
        let zt = ((p.eq_temp - mt) / st_s.max(1e-6)).abs();
        let ze = ((p.ecc - me) / se.max(1e-6)).abs();
        (zp + zr + zm + zt + ze) / 5.0
    }).collect();

    let threshold = 2.0;
    let lam: Vec<f64> = scores.iter().map(|s| s - threshold).collect();
    let features: Vec<[f64; 5]> = planets.iter().map(|p| [
        (p.log_period - mp) / sp.max(1e-6), (p.log_radius - mr) / sr.max(1e-6),
        (p.log_mass - mm) / sm.max(1e-6), (p.eq_temp - mt) / st_s.max(1e-6),
        (p.ecc - me) / se.max(1e-6),
    ]).collect();

    let k = 5;
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..planets.len() {
        let mut dists: Vec<(usize, f64)> = (0..planets.len()).filter(|&j| j != i).map(|j| {
            let d: f64 = (0..5).map(|d| (features[i][d] - features[j][d]).powi(2)).sum();
            (j, d.sqrt())
        }).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(j, d) in dists.iter().take(k) {
            edges.push((i, j, 1.0 / (1.0 + d)));
        }
    }

    let flagged = solve_mincut(&lam, &edges, 0.3);
    let mut experiences = Vec::new();

    // Package top anomalies as training experiences
    let mut ranked: Vec<(usize, f64)> = scores.iter().enumerate()
        .filter(|(i, _)| flagged[*i])
        .map(|(i, &s)| (i, s)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(i, score) in ranked.iter().take(10) {
        let p = &planets[i];
        let reward = (score / 5.0).min(1.0); // normalize to [0, 1]
        experiences.push(TrainingExperience {
            domain: "exoplanet".into(),
            state: format!("planet:logP={:.2},logR={:.2},logM={:.2},Teq={:.0},e={:.3}",
                p.log_period, p.log_radius, p.log_mass, p.eq_temp, p.ecc),
            action: format!("flagged_anomaly:method={},score={:.2}", p.method, score),
            reward,
            category: "pattern".into(),
            title: format!("Exoplanet anomaly: {}", p.name),
            content: format!(
                "Graph-cut flagged {} as anomalous (score={:.2}). Period={:.1}d, Radius={:.2}Re, Mass={:.0}Me, Teq={:.0}K, ecc={:.3}. Method: {}",
                p.name, score, p.log_period.exp(), p.log_radius.exp(), p.log_mass.exp(), p.eq_temp, p.ecc, p.method
            ),
            tags: vec!["exoplanet".into(), "anomaly".into(), "graph-cut".into(), p.method.to_lowercase()],
        });
    }

    println!("  Exoplanets: {} planets analyzed, {} flagged, {} experiences",
        planets.len(), flagged.iter().filter(|&&x| x).count(), experiences.len());
    experiences
}

// ── 2. Earthquake discoveries → training data ──────────────────────────────

fn extract_earthquake_experiences() -> Vec<TrainingExperience> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/earthquakes.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    struct Quake { lat: f64, lon: f64, depth: f64, mag: f64, place: String }
    let mut quakes = Vec::new();
    for line in data.lines().skip(1) {
        let f = split_csv_line(line);
        if f.len() < 15 { continue; }
        let lat = match parse_f64(&f[1]) { Some(v) => v, _ => continue };
        let lon = match parse_f64(&f[2]) { Some(v) => v, _ => continue };
        let depth = parse_f64(&f[3]).unwrap_or(10.0);
        let mag = match parse_f64(&f[4]) { Some(v) => v, _ => continue };
        let place = parse_csv_field(&f[13]).to_string();
        quakes.push(Quake { lat, lon, depth, mag, place });
    }

    // Build proximity graph
    let haversine = |lat1: f64, lon1: f64, lat2: f64, lon2: f64| -> f64 {
        let d2r = std::f64::consts::PI / 180.0;
        let (dlat, dlon) = ((lat2 - lat1) * d2r, (lon2 - lon1) * d2r);
        let a = (dlat / 2.0).sin().powi(2)
            + (lat1 * d2r).cos() * (lat2 * d2r).cos() * (dlon / 2.0).sin().powi(2);
        2.0 * 6371.0 * a.sqrt().asin()
    };

    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut nc = vec![0usize; quakes.len()];
    for i in 0..quakes.len() {
        for j in (i + 1)..quakes.len() {
            let d = haversine(quakes[i].lat, quakes[i].lon, quakes[j].lat, quakes[j].lon);
            if d < 200.0 {
                let w = 1.0 / (1.0 + d / 50.0);
                edges.push((i, j, w)); edges.push((j, i, w));
                nc[i] += 1; nc[j] += 1;
            }
        }
    }

    let mean_nc = nc.iter().sum::<usize>() as f64 / quakes.len() as f64;
    let std_nc = (nc.iter().map(|&c| (c as f64 - mean_nc).powi(2)).sum::<f64>() / quakes.len() as f64).sqrt();
    let mean_mag = quakes.iter().map(|q| q.mag).sum::<f64>() / quakes.len() as f64;

    let lam: Vec<f64> = quakes.iter().enumerate().map(|(i, q)| {
        let density_z = (nc[i] as f64 - mean_nc) / std_nc.max(1e-6);
        let deep = if q.depth > 300.0 { 1.5 } else { 0.0 };
        let mag_b = if q.mag > mean_mag + 2.0 { 1.0 } else { 0.0 };
        density_z * 0.5 + deep + mag_b - 1.2
    }).collect();

    let flagged = solve_mincut(&lam, &edges, 0.4);
    let mut experiences = Vec::new();

    // Deep quakes and strong events
    for (i, q) in quakes.iter().enumerate() {
        if !flagged[i] { continue; }
        let reward = if q.depth > 300.0 { 0.9 } else if q.mag > 6.0 { 0.8 } else { 0.5 };
        let anomaly_type = if q.depth > 300.0 { "deep_focus" }
            else if nc[i] as f64 > mean_nc + 2.0 * std_nc { "swarm_cluster" }
            else { "magnitude_outlier" };
        experiences.push(TrainingExperience {
            domain: "seismology".into(),
            state: format!("quake:lat={:.2},lon={:.2},depth={:.1},mag={:.1},neighbors={}",
                q.lat, q.lon, q.depth, q.mag, nc[i]),
            action: format!("flagged_{}:mag={:.1},depth={:.1}", anomaly_type, q.mag, q.depth),
            reward,
            category: "pattern".into(),
            title: format!("Seismic anomaly: M{:.1} {}", q.mag, &q.place[..q.place.len().min(40)]),
            content: format!(
                "Graph-cut flagged M{:.1} earthquake at ({:.2}, {:.2}), depth {:.1}km. Type: {}. Location: {}",
                q.mag, q.lat, q.lon, q.depth, anomaly_type, q.place
            ),
            tags: vec!["earthquake".into(), anomaly_type.into(), "graph-cut".into()],
        });
        if experiences.len() >= 15 { break; }
    }

    println!("  Earthquakes: {} events analyzed, {} flagged, {} experiences",
        quakes.len(), flagged.iter().filter(|&&x| x).count(), experiences.len());
    experiences
}

// ── 3. Climate discoveries → training data ──────────────────────────────────

fn extract_climate_experiences() -> Vec<TrainingExperience> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/global_temp_anomaly.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut years: Vec<(i32, f64)> = Vec::new();
    for line in data.lines() {
        if line.starts_with('#') || line.starts_with("Year") { continue; }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(y), Ok(a)) = (parts[0].trim().parse::<i32>(), parts[1].trim().parse::<f64>()) {
                years.push((y, a));
            }
        }
    }

    let n = years.len();
    let anomalies: Vec<f64> = years.iter().map(|y| y.1).collect();
    let global_mean = anomalies.iter().sum::<f64>() / n as f64;

    // CUSUM
    let mut cusum_pos = vec![0.0f64; n];
    let mut cusum_neg = vec![0.0f64; n];
    for i in 1..n {
        let diff = anomalies[i] - global_mean;
        cusum_pos[i] = (cusum_pos[i - 1] + diff - 0.02).max(0.0);
        cusum_neg[i] = (cusum_neg[i - 1] - diff - 0.02).max(0.0);
    }
    let cusum_max = cusum_pos.iter().chain(cusum_neg.iter()).cloned().fold(0.0f64, f64::max);
    let lam: Vec<f64> = (0..n).map(|i| {
        (cusum_pos[i] + cusum_neg[i]) / cusum_max.max(1e-6) - 0.15
    }).collect();

    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n.saturating_sub(1) {
        let diff = (anomalies[i] - anomalies[i + 1]).abs();
        edges.push((i, i + 1, 1.0 / (1.0 + diff * 5.0)));
        edges.push((i + 1, i, 1.0 / (1.0 + diff * 5.0)));
    }
    for i in 0..n.saturating_sub(5) {
        let diff = (anomalies[i] - anomalies[i + 5]).abs();
        edges.push((i, i + 5, 0.3 / (1.0 + diff * 3.0)));
        edges.push((i + 5, i, 0.3 / (1.0 + diff * 3.0)));
    }

    let regime = solve_mincut(&lam, &edges, 0.5);
    let mut experiences = Vec::new();

    // Regime transitions
    for i in 1..n {
        if regime[i] != regime[i - 1] {
            let before_start = if i > 10 { i - 10 } else { 0 };
            let after_end = (i + 10).min(n);
            let before_mean = anomalies[before_start..i].iter().sum::<f64>() / (i - before_start) as f64;
            let after_mean = anomalies[i..after_end].iter().sum::<f64>() / (after_end - i) as f64;
            let shift = after_mean - before_mean;
            experiences.push(TrainingExperience {
                domain: "climate".into(),
                state: format!("year:{},anomaly:{:.2},before_avg:{:.3}", years[i].0, anomalies[i], before_mean),
                action: format!("regime_transition:shift={:+.3}", shift),
                reward: shift.abs().min(1.0),
                category: "pattern".into(),
                title: format!("Climate regime shift at {}", years[i].0),
                content: format!(
                    "Graph-cut detected regime transition at {}. Anomaly: {:+.2}C. 10yr average shifted {:+.3}C ({:+.3} → {:+.3})",
                    years[i].0, anomalies[i], shift, before_mean, after_mean
                ),
                tags: vec!["climate".into(), "regime-shift".into(), "graph-cut".into()],
            });
        }
    }

    // Warming rate acceleration
    let decades = [(1970, 1990), (1990, 2010), (2010, 2026)];
    for &(y0, y1) in &decades {
        let vals: Vec<(f64, f64)> = years.iter()
            .filter(|y| y.0 >= y0 && y.0 < y1)
            .map(|y| (y.0 as f64, y.1)).collect();
        if vals.len() < 2 { continue; }
        let n_v = vals.len() as f64;
        let mx = vals.iter().map(|v| v.0).sum::<f64>() / n_v;
        let my = vals.iter().map(|v| v.1).sum::<f64>() / n_v;
        let slope = vals.iter().map(|v| (v.0 - mx) * (v.1 - my)).sum::<f64>()
            / vals.iter().map(|v| (v.0 - mx).powi(2)).sum::<f64>().max(1e-6);
        let rate = slope * 10.0;
        experiences.push(TrainingExperience {
            domain: "climate".into(),
            state: format!("period:{}-{},avg_anomaly:{:+.3}", y0, y1, my),
            action: format!("warming_rate:{:+.3}C/decade", rate),
            reward: rate.abs().min(1.0),
            category: "pattern".into(),
            title: format!("Warming rate {}-{}: {:+.3}C/decade", y0, y1, rate),
            content: format!(
                "Temperature anomaly trend {}-{}: {:+.3}C per decade (avg anomaly {:+.3}C). {}",
                y0, y1, rate, my,
                if rate > 0.3 { "ACCELERATING — exceeds 2x historical rate" }
                else if rate > 0.15 { "Sustained warming above pre-industrial trend" }
                else { "Moderate warming rate" }
            ),
            tags: vec!["climate".into(), "warming-rate".into(), "trend".into()],
        });
    }

    // Extreme years
    let mut sorted: Vec<(i32, f64)> = years.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(y, a) in sorted.iter().take(5) {
        experiences.push(TrainingExperience {
            domain: "climate".into(),
            state: format!("year:{},anomaly:{:+.2}", y, a),
            action: format!("extreme_warm:rank_top5,anomaly={:+.2}", a),
            reward: (a / 1.5).min(1.0),
            category: "pattern".into(),
            title: format!("Record warm year: {} ({:+.2}C)", y, a),
            content: format!("{} recorded {:+.2}C anomaly — among the 5 warmest years in 177-year record", y, a),
            tags: vec!["climate".into(), "extreme".into(), "record".into()],
        });
    }

    println!("  Climate: {} years analyzed, {} experiences", years.len(), experiences.len());
    experiences
}

// ── Brain API client ────────────────────────────────────────────────────────

struct BrainClient {
    base_url: String,
    api_key: String,
}

impl BrainClient {
    fn new() -> Self {
        Self {
            base_url: std::env::var("BRAIN_URL").unwrap_or_else(|_| "https://pi.ruv.io".into()),
            api_key: std::env::var("PI").unwrap_or_default(),
        }
    }

    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }

    /// Share a discovery as a memory on the brain
    fn share_memory(&self, exp: &TrainingExperience) -> Result<String, String> {
        let body = format!(
            r#"{{"title":"{}","content":"{}","category":"{}","tags":{}}}"#,
            exp.title.replace('"', r#"\""#),
            exp.content.replace('"', r#"\""#),
            exp.category,
            serde_json_tags(&exp.tags),
        );

        let output = std::process::Command::new("curl")
            .args(["-s", "-X", "POST",
                &format!("{}/v1/memories", self.base_url),
                "-H", "Content-Type: application/json",
                "-H", &format!("Authorization: Bearer {}", self.api_key),
                "-d", &body,
                "--max-time", "10"])
            .output()
            .map_err(|e| format!("curl error: {}", e))?;

        let resp = String::from_utf8_lossy(&output.stdout).to_string();
        if output.status.success() && !resp.contains("error") {
            Ok(resp)
        } else {
            Err(format!("API error: {}", resp))
        }
    }

    /// Trigger a SONA training cycle
    fn train(&self) -> Result<String, String> {
        let output = std::process::Command::new("curl")
            .args(["-s", "-X", "POST",
                &format!("{}/v1/train", self.base_url),
                "-H", "Content-Type: application/json",
                "-H", &format!("Authorization: Bearer {}", self.api_key),
                "-d", "{}",
                "--max-time", "15"])
            .output()
            .map_err(|e| format!("curl error: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get SONA learning stats
    fn sona_stats(&self) -> Result<String, String> {
        let output = std::process::Command::new("curl")
            .args(["-s",
                &format!("{}/v1/sona/stats", self.base_url),
                "-H", &format!("Authorization: Bearer {}", self.api_key),
                "--max-time", "10"])
            .output()
            .map_err(|e| format!("curl error: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get meta-learning exploration stats
    fn explore(&self) -> Result<String, String> {
        let output = std::process::Command::new("curl")
            .args(["-s",
                &format!("{}/v1/explore", self.base_url),
                "-H", &format!("Authorization: Bearer {}", self.api_key),
                "--max-time", "10"])
            .output()
            .map_err(|e| format!("curl error: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get temporal delta tracking
    fn temporal(&self) -> Result<String, String> {
        let output = std::process::Command::new("curl")
            .args(["-s",
                &format!("{}/v1/temporal", self.base_url),
                "-H", &format!("Authorization: Bearer {}", self.api_key),
                "--max-time", "10"])
            .output()
            .map_err(|e| format!("curl error: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

fn serde_json_tags(tags: &[String]) -> String {
    let inner: Vec<String> = tags.iter().map(|t| format!("\"{}\"", t)).collect();
    format!("[{}]", inner.join(","))
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("=========================================================================");
    println!("  BRAIN TRAINING INTEGRATION — Discovery → π.ruv.io Learning Pipeline");
    println!("=========================================================================\n");

    // Phase 1: Extract discoveries from all 3 datasets
    println!("Phase 1: Extracting discoveries via graph-cut anomaly detection\n");
    let mut all_experiences = Vec::new();
    all_experiences.extend(extract_exoplanet_experiences());
    all_experiences.extend(extract_earthquake_experiences());
    all_experiences.extend(extract_climate_experiences());

    println!("\n  Total training experiences: {}\n", all_experiences.len());

    // Print experience summary
    println!("  {:>3} {:<12} {:<50} {:>6}",
        "#", "Domain", "Title", "Reward");
    println!("  {:-<3} {:-<12} {:-<50} {:-<6}", "", "", "", "");
    for (i, exp) in all_experiences.iter().enumerate() {
        println!("  {:>3} {:<12} {:<50} {:>6.3}",
            i + 1, exp.domain, &exp.title[..exp.title.len().min(50)], exp.reward);
    }

    // Phase 2: Connect to π Brain
    println!("\n{}", "=".repeat(73));
    println!("  Phase 2: Connecting to π Brain for training");
    println!("{}\n", "=".repeat(73));

    let brain = BrainClient::new();

    if !brain.is_configured() {
        println!("  PI env var not set — running in dry-run mode");
        println!("  Set PI=<api-key> and optionally BRAIN_URL=<url> to enable\n");

        // Show what would be sent
        println!("  Dry-run: would share {} memories and trigger training\n", all_experiences.len());
        println!("  Example API calls that would be made:");
        println!("    POST {}/v1/memories  (x{})", brain.base_url, all_experiences.len());
        println!("    POST {}/v1/train     (trigger SONA cycle)", brain.base_url);
        println!("    GET  {}/v1/sona/stats", brain.base_url);
        println!("    GET  {}/v1/explore   (meta-learning)", brain.base_url);
        println!("    GET  {}/v1/temporal  (delta tracking)", brain.base_url);

        // Show experience structure
        if let Some(exp) = all_experiences.first() {
            println!("\n  Example experience payload:");
            println!("    domain:   {}", exp.domain);
            println!("    state:    {}", exp.state);
            println!("    action:   {}", exp.action);
            println!("    reward:   {:.3}", exp.reward);
            println!("    category: {}", exp.category);
            println!("    title:    {}", exp.title);
            println!("    tags:     {:?}", exp.tags);
        }
    } else {
        println!("  Brain URL: {}", brain.base_url);
        println!("  API key:   {}...{}\n",
            &brain.api_key[..4.min(brain.api_key.len())],
            &brain.api_key[brain.api_key.len().saturating_sub(4)..]);

        // Get baseline stats
        println!("  --- Pre-training stats ---");
        match brain.sona_stats() {
            Ok(s) => println!("  SONA: {}", s.trim()),
            Err(e) => println!("  SONA stats error: {}", e),
        }
        match brain.temporal() {
            Ok(s) => println!("  Temporal: {}", s.trim()),
            Err(e) => println!("  Temporal error: {}", e),
        }

        // Share discoveries
        println!("\n  --- Sharing {} discoveries ---", all_experiences.len());
        let mut shared = 0;
        let mut errors = 0;
        for (i, exp) in all_experiences.iter().enumerate() {
            match brain.share_memory(exp) {
                Ok(_) => {
                    shared += 1;
                    if i < 3 || i == all_experiences.len() - 1 {
                        println!("    [OK] {}", exp.title);
                    } else if i == 3 {
                        println!("    ... sharing remaining ...");
                    }
                }
                Err(e) => {
                    errors += 1;
                    if errors <= 3 {
                        println!("    [ERR] {}: {}", exp.title, e);
                    }
                }
            }
        }
        println!("  Shared: {}, Errors: {}\n", shared, errors);

        // Trigger training
        println!("  --- Triggering SONA training cycle ---");
        match brain.train() {
            Ok(s) => println!("  Training result: {}", s.trim()),
            Err(e) => println!("  Training error: {}", e),
        }

        // Post-training stats
        println!("\n  --- Post-training stats ---");
        match brain.sona_stats() {
            Ok(s) => println!("  SONA: {}", s.trim()),
            Err(e) => println!("  SONA stats error: {}", e),
        }
        match brain.explore() {
            Ok(s) => println!("  Meta-learning: {}", s.trim()),
            Err(e) => println!("  Explore error: {}", e),
        }
        match brain.temporal() {
            Ok(s) => println!("  Temporal: {}", s.trim()),
            Err(e) => println!("  Temporal error: {}", e),
        }
    }

    // Phase 3: Summary
    println!("\n{}", "=".repeat(73));
    println!("  Phase 3: Training Pipeline Summary");
    println!("{}\n", "=".repeat(73));

    let by_domain: Vec<(&str, usize)> = vec![
        ("exoplanet", all_experiences.iter().filter(|e| e.domain == "exoplanet").count()),
        ("seismology", all_experiences.iter().filter(|e| e.domain == "seismology").count()),
        ("climate", all_experiences.iter().filter(|e| e.domain == "climate").count()),
    ];
    for (domain, count) in &by_domain {
        println!("  {:<12} {} experiences", domain, count);
    }
    let avg_reward = all_experiences.iter().map(|e| e.reward).sum::<f64>() / all_experiences.len().max(1) as f64;
    println!("\n  Total: {} experiences, avg reward: {:.3}", all_experiences.len(), avg_reward);
    println!("  Pipeline: graph-cut anomaly detection → brain memory → SONA training");
    println!("  Endpoint: {} (MCP tools: brain_share, brain_train, brain_sona_stats)", brain.base_url);

    println!("\n=========================================================================");
    println!("  Training integration complete.");
    println!("=========================================================================");
}
