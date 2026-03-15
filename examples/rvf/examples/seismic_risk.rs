//! Seismic Risk Mapping with Gutenberg-Richter Anomaly Detection
//!
//! Analyses USGS earthquake data (~1942 events, M2.5+) using:
//!   1. Spatial grid binning (2-degree lat/lon cells)
//!   2. Per-cell Gutenberg-Richter b-value estimation (Aki 1965 MLE)
//!   3. Tectonic neighborhood graph with haversine proximity
//!   4. Edmonds-Karp min-cut segmentation for anomaly flagging
//!   5. Depth distribution analysis (subduction zone indicators)
//!
//! Run: cargo run --example seismic_risk --release

use std::collections::{HashMap, VecDeque};

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
                if !vis[v] && caps[ei] > 1e-15 {
                    vis[v] = true; par[v] = Some((u, ei)); q.push_back(v);
                }
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
        for &(v, ei) in &adj[u] {
            if !reach[v] && caps[ei] > 1e-15 { reach[v] = true; stk.push(v); }
        }
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

// ── Haversine distance ──────────────────────────────────────────────────────

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (r, d2r) = (6371.0, std::f64::consts::PI / 180.0);
    let (dlat, dlon) = ((lat2 - lat1) * d2r, (lon2 - lon1) * d2r);
    let a = (dlat / 2.0).sin().powi(2)
        + (lat1 * d2r).cos() * (lat2 * d2r).cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

// ── Data structures ─────────────────────────────────────────────────────────

struct Quake {
    lat: f64,
    lon: f64,
    depth: f64,
    mag: f64,
    place: String,
}

/// Grid cell key: (lat_bin, lon_bin) where bin = floor(coord / 2) * 2
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct CellKey {
    lat_bin: i32, // south edge of the 2-degree cell
    lon_bin: i32, // west edge of the 2-degree cell
}

struct CellStats {
    key: CellKey,
    event_count: usize,
    magnitudes: Vec<f64>,
    depths: Vec<f64>,
    a_value: f64,
    b_value: f64,
    energy_joules: f64,
    shallow: usize,   // < 70 km
    intermediate: usize, // 70-300 km
    deep: usize,       // > 300 km
    representative_place: String,
}

// ── Grid binning ────────────────────────────────────────────────────────────

fn lat_lon_to_cell(lat: f64, lon: f64) -> CellKey {
    CellKey {
        lat_bin: (lat / 2.0).floor() as i32 * 2,
        lon_bin: (lon / 2.0).floor() as i32 * 2,
    }
}

fn cell_center(key: &CellKey) -> (f64, f64) {
    (key.lat_bin as f64 + 1.0, key.lon_bin as f64 + 1.0)
}

// ── Main analysis ───────────────────────────────────────────────────────────

fn main() {
    println!("=========================================================================");
    println!("  SEISMIC RISK MAPPING — Gutenberg-Richter Anomaly Detection");
    println!("  USGS Earthquake Data, M2.5+, Edmonds-Karp Graph-Cut Segmentation");
    println!("=========================================================================");

    // ── Load data ───────────────────────────────────────────────────────────
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/earthquakes.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  [ERROR] Cannot read {}: {}", path, e);
            std::process::exit(1);
        }
    };

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
    println!("\n  Parsed {} earthquakes from USGS catalog\n", quakes.len());

    // ── 1. Spatial grid binning (2-degree cells) ────────────────────────────
    println!("{}", "=".repeat(70));
    println!("  1. SPATIAL GRID BINNING (2-degree lat/lon cells)");
    println!("{}\n", "=".repeat(70));

    let mut cell_quakes: HashMap<CellKey, Vec<usize>> = HashMap::new();
    for (i, q) in quakes.iter().enumerate() {
        let key = lat_lon_to_cell(q.lat, q.lon);
        cell_quakes.entry(key).or_default().push(i);
    }

    let total_cells = cell_quakes.len();
    let cells_ge5: usize = cell_quakes.values().filter(|v| v.len() >= 5).count();
    println!("  Total grid cells with events: {}", total_cells);
    println!("  Cells with >= 5 events (analysable): {}", cells_ge5);

    // ── 2. Per-cell Gutenberg-Richter estimation ────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("  2. GUTENBERG-RICHTER b-VALUE ESTIMATION (Aki 1965 MLE)");
    println!("{}\n", "=".repeat(70));

    let log10e = std::f64::consts::E.log10(); // log10(e) ~ 0.4343

    let mut cells: Vec<CellStats> = Vec::new();

    for (&key, indices) in &cell_quakes {
        if indices.len() < 5 { continue; }

        let magnitudes: Vec<f64> = indices.iter().map(|&i| quakes[i].mag).collect();
        let depths: Vec<f64> = indices.iter().map(|&i| quakes[i].depth).collect();

        let n = magnitudes.len();
        let m_min = magnitudes.iter().cloned().fold(f64::MAX, f64::min);
        let m_mean = magnitudes.iter().sum::<f64>() / n as f64;

        // Aki (1965) MLE b-value: b = log10(e) / (M_mean - M_min)
        let denom = m_mean - m_min;
        let b_value = if denom > 0.01 { log10e / denom } else { 1.0 };

        // a-value: log10(N) normalized — represents log10 of annual rate
        // We treat the catalog as a ~30-day window, so rate = N * 12
        let a_value = (n as f64 * 12.0).log10();

        // Seismic energy budget: sum of 10^(1.5*M + 4.8) joules
        let energy_joules: f64 = magnitudes.iter()
            .map(|&m| 10.0_f64.powf(1.5 * m + 4.8))
            .sum();

        // Depth distribution
        let shallow = depths.iter().filter(|&&d| d < 70.0).count();
        let intermediate = depths.iter().filter(|&&d| d >= 70.0 && d <= 300.0).count();
        let deep = depths.iter().filter(|&&d| d > 300.0).count();

        // Representative place name (from strongest event in cell)
        let strongest_idx = indices.iter()
            .max_by(|&&a, &&b| quakes[a].mag.partial_cmp(&quakes[b].mag).unwrap())
            .copied()
            .unwrap_or(indices[0]);
        let representative_place = quakes[strongest_idx].place.clone();

        cells.push(CellStats {
            key,
            event_count: n,
            magnitudes,
            depths,
            a_value,
            b_value,
            energy_joules,
            shallow,
            intermediate,
            deep,
            representative_place,
        });
    }

    // Sort cells by event count for display
    cells.sort_by(|a, b| b.event_count.cmp(&a.event_count));

    // Global b-value statistics
    let b_values: Vec<f64> = cells.iter().map(|c| c.b_value).collect();
    let b_mean = b_values.iter().sum::<f64>() / b_values.len() as f64;
    let b_std = (b_values.iter().map(|b| (b - b_mean).powi(2)).sum::<f64>()
        / b_values.len() as f64).sqrt();

    println!("  Global b-value statistics across {} cells:", cells.len());
    println!("    Mean b-value:  {:.3}", b_mean);
    println!("    Std dev:       {:.3}", b_std);
    println!("    Range:         {:.3} — {:.3}",
        b_values.iter().cloned().fold(f64::MAX, f64::min),
        b_values.iter().cloned().fold(f64::MIN, f64::max));
    println!("    Expected (tectonic): ~1.0 (Gutenberg & Richter 1944)");

    println!("\n  Top-15 cells by event count:");
    println!("  {:>6} {:>6} {:>5} {:>6} {:>6} {:>10} {:>4}/{:>4}/{:>4}  {:<35}",
        "Lat", "Lon", "N", "b-val", "a-val", "Energy(J)", "S", "I", "D", "Location");
    println!("  {:-<6} {:-<6} {:-<5} {:-<6} {:-<6} {:-<10} {:-<4} {:-<4} {:-<4}  {:-<35}",
        "", "", "", "", "", "", "", "", "", "");
    for c in cells.iter().take(15) {
        let (clat, clon) = cell_center(&c.key);
        println!("  {:>6.1} {:>6.1} {:>5} {:>6.3} {:>6.2} {:>10.2e} {:>4}/{:>4}/{:>4}  {:<35}",
            clat, clon, c.event_count, c.b_value, c.a_value, c.energy_joules,
            c.shallow, c.intermediate, c.deep,
            &c.representative_place[..c.representative_place.len().min(35)]);
    }

    // ── 3. Tectonic neighborhood graph ──────────────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("  3. TECTONIC NEIGHBORHOOD GRAPH (adjacency + 500km proximity)");
    println!("{}\n", "=".repeat(70));

    // Build cell index for adjacency lookups
    let _cell_index: HashMap<CellKey, usize> = cells.iter().enumerate()
        .map(|(i, c)| (c.key, i)).collect();

    let nc = cells.len();
    let mut graph_edges: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..nc {
        let (ci_lat, ci_lon) = cell_center(&cells[i].key);

        // Check grid neighbors (8-connected) and 500km proximity
        for j in (i + 1)..nc {
            let (cj_lat, cj_lon) = cell_center(&cells[j].key);

            let dlat = (cells[i].key.lat_bin - cells[j].key.lat_bin).abs();
            let dlon = (cells[i].key.lon_bin - cells[j].key.lon_bin).abs();
            let is_grid_neighbor = dlat <= 2 && dlon <= 2 && !(dlat == 0 && dlon == 0);

            let dist = haversine_km(ci_lat, ci_lon, cj_lat, cj_lon);
            let is_proximate = dist < 500.0;

            if is_grid_neighbor || is_proximate {
                let w = 1.0 / (1.0 + dist / 200.0);
                graph_edges.push((i, j, w));
                graph_edges.push((j, i, w));
            }
        }
    }

    println!("  Graph: {} cells, {} directed edges", nc, graph_edges.len());
    let avg_degree = if nc > 0 { graph_edges.len() as f64 / nc as f64 } else { 0.0 };
    println!("  Average degree: {:.1}", avg_degree);

    // ── Compute lambda: combined z-score of (a, b) relative to neighbors ──

    let a_values: Vec<f64> = cells.iter().map(|c| c.a_value).collect();
    let a_mean = a_values.iter().sum::<f64>() / a_values.len() as f64;
    let a_std = (a_values.iter().map(|a| (a - a_mean).powi(2)).sum::<f64>()
        / a_values.len() as f64).sqrt();

    // Build neighbor lists for local statistics
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); nc];
    for &(i, j, _) in &graph_edges {
        if !neighbors[i].contains(&j) { neighbors[i].push(j); }
    }

    let lam: Vec<f64> = (0..nc).map(|i| {
        let b = cells[i].b_value;
        let a = cells[i].a_value;

        // Global z-scores
        let b_z = (b - b_mean) / b_std.max(1e-6);
        let a_z = (a - a_mean) / a_std.max(1e-6);

        // Local z-scores relative to neighbors
        let (local_b_z, local_a_z) = if neighbors[i].len() >= 2 {
            let nb: Vec<f64> = neighbors[i].iter().map(|&j| cells[j].b_value).collect();
            let na: Vec<f64> = neighbors[i].iter().map(|&j| cells[j].a_value).collect();
            let nb_mean = nb.iter().sum::<f64>() / nb.len() as f64;
            let nb_std = (nb.iter().map(|v| (v - nb_mean).powi(2)).sum::<f64>()
                / nb.len() as f64).sqrt();
            let na_mean = na.iter().sum::<f64>() / na.len() as f64;
            let na_std = (na.iter().map(|v| (v - na_mean).powi(2)).sum::<f64>()
                / na.len() as f64).sqrt();
            (
                (b - nb_mean) / nb_std.max(1e-6),
                (a - na_mean) / na_std.max(1e-6),
            )
        } else {
            (b_z, a_z)
        };

        // Combined anomaly: weight local more than global
        let combined = (local_b_z.abs() + local_a_z.abs()) * 0.6
            + (b_z.abs() + a_z.abs()) * 0.4;

        // Boost for scientifically interesting b-value ranges
        let b_anomaly_boost = if b < 0.7 { 1.5 } // large-quake-prone
            else if b > 1.3 { 1.2 } // swarm/induced
            else { 0.0 };

        // Energy concentration bonus
        let log_energy = cells[i].energy_joules.log10();
        let energy_bonus = if log_energy > 14.0 { 0.5 } else { 0.0 };

        combined / 2.0 + b_anomaly_boost + energy_bonus - 1.5
    }).collect();

    // ── 4. Min-cut segmentation ─────────────────────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("  4. MIN-CUT SEGMENTATION (Edmonds-Karp anomaly flagging)");
    println!("{}\n", "=".repeat(70));

    let flagged = solve_mincut(&lam, &graph_edges, 0.4);
    let n_flagged = flagged.iter().filter(|&&x| x).count();
    println!("  Graph-cut flagged {} / {} cells as anomalous ({:.1}%)\n",
        n_flagged, nc, if nc > 0 { n_flagged as f64 / nc as f64 * 100.0 } else { 0.0 });

    // Categorize flagged cells
    let mut low_b: Vec<usize> = Vec::new();
    let mut high_b: Vec<usize> = Vec::new();
    let mut high_energy: Vec<usize> = Vec::new();

    for i in 0..nc {
        if !flagged[i] { continue; }
        if cells[i].b_value < 0.7 { low_b.push(i); }
        if cells[i].b_value > 1.3 { high_b.push(i); }
        if cells[i].energy_joules.log10() > 14.0 { high_energy.push(i); }
    }

    println!("  Anomaly categories:");
    println!("    Low b-value  (b < 0.7, large-quake-prone):   {} cells", low_b.len());
    println!("    High b-value (b > 1.3, swarm/induced):       {} cells", high_b.len());
    println!("    High energy concentration (>10^14 J):        {} cells", high_energy.len());

    if !low_b.is_empty() {
        println!("\n  Low b-value cells (higher probability of large events):");
        for &i in low_b.iter().take(5) {
            let (clat, clon) = cell_center(&cells[i].key);
            println!("    ({:>6.1}, {:>6.1}) b={:.3}, N={}, {}",
                clat, clon, cells[i].b_value, cells[i].event_count,
                &cells[i].representative_place[..cells[i].representative_place.len().min(40)]);
        }
    }

    if !high_b.is_empty() {
        println!("\n  High b-value cells (swarm/induced seismicity signature):");
        for &i in high_b.iter().take(5) {
            let (clat, clon) = cell_center(&cells[i].key);
            println!("    ({:>6.1}, {:>6.1}) b={:.3}, N={}, {}",
                clat, clon, cells[i].b_value, cells[i].event_count,
                &cells[i].representative_place[..cells[i].representative_place.len().min(40)]);
        }
    }

    // ── 5. Depth distribution analysis ──────────────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("  5. DEPTH DISTRIBUTION ANALYSIS (subduction zone indicators)");
    println!("{}\n", "=".repeat(70));

    println!("  Flagged cells with depth profiles:");
    println!("  {:>6} {:>6} {:>5} {:>6} {:>4} {:>4} {:>4} {:>8} {:<35}",
        "Lat", "Lon", "N", "b-val", "S", "I", "D", "Profile", "Location");
    println!("  {:-<6} {:-<6} {:-<5} {:-<6} {:-<4} {:-<4} {:-<4} {:-<8} {:-<35}",
        "", "", "", "", "", "", "", "", "");

    let mut subduction_candidates = Vec::new();

    for i in 0..nc {
        if !flagged[i] { continue; }
        let c = &cells[i];
        let (clat, clon) = cell_center(&c.key);

        // Classify depth profile
        let profile = if c.deep > 0 && c.shallow > 0 {
            "BIMODAL"  // subduction zone indicator
        } else if c.intermediate > 0 && c.shallow > 0 {
            "MIXED"
        } else if c.shallow == c.event_count {
            "SHALLOW"
        } else if c.deep > 0 {
            "DEEP"
        } else {
            "INTERM"
        };

        if c.deep > 0 && c.shallow > 0 {
            subduction_candidates.push(i);
        }

        println!("  {:>6.1} {:>6.1} {:>5} {:>6.3} {:>4} {:>4} {:>4} {:>8} {:<35}",
            clat, clon, c.event_count, c.b_value,
            c.shallow, c.intermediate, c.deep, profile,
            &c.representative_place[..c.representative_place.len().min(35)]);
    }

    if !subduction_candidates.is_empty() {
        println!("\n  Subduction zone candidates (bimodal depth distribution):");
        for &i in &subduction_candidates {
            let c = &cells[i];
            let (clat, clon) = cell_center(&c.key);
            let depth_range = c.depths.iter().cloned().fold(f64::MAX, f64::min);
            let depth_max = c.depths.iter().cloned().fold(f64::MIN, f64::max);
            println!("    ({:>6.1}, {:>6.1}) depth {:.0}-{:.0} km, S={} I={} D={}, {}",
                clat, clon, depth_range, depth_max,
                c.shallow, c.intermediate, c.deep,
                &c.representative_place[..c.representative_place.len().min(40)]);
        }
    }

    // ── 6. Risk report: top-10 highest-risk cells ───────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("  6. SEISMIC RISK RANKING — Top-10 Highest-Risk Cells");
    println!("{}\n", "=".repeat(70));

    // Composite risk score
    struct RiskEntry {
        cell_idx: usize,
        risk_score: f64,
    }

    let mut risk_entries: Vec<RiskEntry> = (0..nc).map(|i| {
        let c = &cells[i];

        // Low b-value = higher risk (more likely large events)
        let b_risk = if c.b_value < 0.7 { 3.0 }
            else if c.b_value < 0.9 { 2.0 }
            else if c.b_value > 1.3 { 1.5 } // swarm risk
            else { 0.5 };

        // Energy concentration
        let energy_risk = (c.energy_joules.log10() - 10.0).max(0.0) / 5.0;

        // Event rate
        let rate_risk = (c.a_value - 1.0).max(0.0);

        // Max magnitude in cell
        let max_mag = c.magnitudes.iter().cloned().fold(0.0_f64, f64::max);
        let mag_risk = if max_mag >= 6.0 { 3.0 }
            else if max_mag >= 5.0 { 2.0 }
            else if max_mag >= 4.0 { 1.0 }
            else { 0.0 };

        // Depth complexity (subduction indicator = higher risk)
        let depth_risk = if c.deep > 0 && c.shallow > 0 { 2.0 }
            else if c.intermediate > 0 { 1.0 }
            else { 0.0 };

        // Flagged bonus
        let flag_bonus = if flagged[i] { 2.0 } else { 0.0 };

        let risk_score = b_risk + energy_risk + rate_risk + mag_risk + depth_risk + flag_bonus;

        RiskEntry { cell_idx: i, risk_score }
    }).collect();

    risk_entries.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap());

    println!("  {:>3} {:>6} {:>6} {:>5} {:>6} {:>6} {:>10} {:>5} {:>4}/{:>4}/{:>4} {:>5} {:<30}",
        "#", "Lat", "Lon", "N", "b-val", "a-val", "Energy(J)", "MaxM",
        "S", "I", "D", "Flag", "Location");
    println!("  {:-<3} {:-<6} {:-<6} {:-<5} {:-<6} {:-<6} {:-<10} {:-<5} {:-<4} {:-<4} {:-<4} {:-<5} {:-<30}",
        "", "", "", "", "", "", "", "", "", "", "", "", "");

    for (rank, entry) in risk_entries.iter().take(10).enumerate() {
        let i = entry.cell_idx;
        let c = &cells[i];
        let (clat, clon) = cell_center(&c.key);
        let max_mag = c.magnitudes.iter().cloned().fold(0.0_f64, f64::max);
        let flag_str = if flagged[i] { "YES" } else { "no" };

        println!("  {:>3} {:>6.1} {:>6.1} {:>5} {:>6.3} {:>6.2} {:>10.2e} {:>5.1} {:>4}/{:>4}/{:>4} {:>5} {:<30}",
            rank + 1, clat, clon, c.event_count, c.b_value, c.a_value,
            c.energy_joules, max_mag,
            c.shallow, c.intermediate, c.deep,
            flag_str,
            &c.representative_place[..c.representative_place.len().min(30)]);
    }

    // ── Summary statistics ──────────────────────────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("  SUMMARY");
    println!("{}\n", "=".repeat(70));

    let total_energy: f64 = cells.iter().map(|c| c.energy_joules).sum();
    let total_events: usize = cells.iter().map(|c| c.event_count).sum();
    let all_mags: Vec<f64> = quakes.iter().map(|q| q.mag).collect();
    let global_m_min = all_mags.iter().cloned().fold(f64::MAX, f64::min);
    let global_m_mean = all_mags.iter().sum::<f64>() / all_mags.len() as f64;
    let global_b = log10e / (global_m_mean - global_m_min).max(0.01);

    println!("  Total earthquakes analysed:       {}", total_events);
    println!("  Analysable grid cells (N>=5):     {}", cells.len());
    println!("  Global b-value (whole catalog):   {:.3}", global_b);
    println!("  Total seismic energy:             {:.3e} J", total_energy);
    println!("  Anomalous cells (graph-cut):      {} / {} ({:.1}%)",
        n_flagged, nc, if nc > 0 { n_flagged as f64 / nc as f64 * 100.0 } else { 0.0 });
    println!("  Subduction zone candidates:       {}", subduction_candidates.len());

    // Magnitude distribution
    println!("\n  Magnitude distribution:");
    for &(lo, hi, label) in &[
        (2.5, 3.0, "M2.5-3.0"),
        (3.0, 4.0, "M3.0-4.0"),
        (4.0, 5.0, "M4.0-5.0"),
        (5.0, 6.0, "M5.0-6.0"),
        (6.0, 7.0, "M6.0-7.0"),
        (7.0, 10.0, "M7.0+  "),
    ] {
        let count = quakes.iter().filter(|q| q.mag >= lo && q.mag < hi).count();
        let bar: String = std::iter::repeat('#').take(count / 5 + if count > 0 { 1 } else { 0 }).collect();
        println!("    {} {:>5}  {}", label, count, bar);
    }

    println!("\n=========================================================================");
    println!("  Analysis complete. Risk cells flagged via Edmonds-Karp min-cut on");
    println!("  Gutenberg-Richter parameter space with tectonic neighborhood graph.");
    println!("=========================================================================");
}
