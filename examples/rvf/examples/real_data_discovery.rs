//! Real-Data Discovery Pipeline: Exoplanets + Earthquakes + Climate
//!
//! Analyses THREE real public datasets using graph-cut anomaly detection:
//!   1. NASA Exoplanet Archive — multi-dimensional outlier detection
//!   2. USGS Earthquakes (last 30 days) — spatial-temporal clustering
//!   3. NOAA Global Temperature Anomalies (1850–2026) — regime change detection
//!
//! Run: cargo run --example real_data_discovery --release

use std::collections::VecDeque;

// ── Graph-cut solver (Edmonds-Karp BFS) ─────────────────────────────────────

fn solve_mincut(lam: &[f64], edges: &[(usize, usize, f64)], gamma: f64) -> Vec<bool> {
    let m = lam.len();
    let (s, t, n) = (m, m + 1, m + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();
    let add =
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
            add(&mut adj, &mut caps, s, i, p0);
        }
        if p1 > 1e-12 {
            add(&mut adj, &mut caps, i, t, p1);
        }
    }
    for &(f, to, w) in edges {
        let c = gamma * w;
        if c > 1e-12 {
            add(&mut adj, &mut caps, f, to, c);
        }
    }
    loop {
        let mut par: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = VecDeque::new();
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
        while let Some((_, ei)) = par[v] {
            bn = bn.min(caps[ei]);
            v = par[v].unwrap().0;
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

// ── CSV helpers ─────────────────────────────────────────────────────────────

fn parse_csv_field(s: &str) -> &str {
    s.trim().trim_matches('"')
}

fn parse_f64(s: &str) -> Option<f64> {
    let v = parse_csv_field(s);
    if v.is_empty() {
        None
    } else {
        v.parse().ok()
    }
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    for ch in line.chars() {
        if ch == '"' {
            in_q = !in_q;
        } else if ch == ',' && !in_q {
            fields.push(cur.clone());
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    fields.push(cur);
    fields
}

// ── 1. EXOPLANET ANOMALY DETECTION ──────────────────────────────────────────

struct Planet {
    name: String,
    log_period: f64,
    log_radius: f64,
    log_mass: f64,
    eq_temp: f64,
    eccentricity: f64,
    method: String,
}

fn run_exoplanets() {
    println!("\n{}", "=".repeat(70));
    println!("  1. EXOPLANET ANOMALY DETECTION — NASA Exoplanet Archive");
    println!("{}\n", "=".repeat(70));

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/confirmed_planets.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            println!("  [SKIP] Cannot read {}: {}", path, e);
            return;
        }
    };

    let mut planets = Vec::new();
    for line in data.lines().skip(1) {
        let f = split_csv_line(line);
        if f.len() < 15 {
            continue;
        }
        let period = match parse_f64(&f[3]) {
            Some(v) if v > 0.0 => v,
            _ => continue,
        };
        let radius = match parse_f64(&f[4]) {
            Some(v) if v > 0.0 => v,
            _ => continue,
        };
        let mass = match parse_f64(&f[5]) {
            Some(v) if v > 0.0 => v,
            _ => continue,
        };
        let eq_temp = match parse_f64(&f[6]) {
            Some(v) => v,
            _ => continue,
        };
        let ecc = parse_f64(&f[7]).unwrap_or(0.0);
        planets.push(Planet {
            name: parse_csv_field(&f[0]).to_string(),
            log_period: period.ln(),
            log_radius: radius.ln(),
            log_mass: mass.ln(),
            eq_temp,
            eccentricity: ecc,
            method: parse_csv_field(&f[13]).to_string(),
        });
    }
    println!(
        "  Parsed {} planets with complete data (period, radius, mass, Teq)\n",
        planets.len()
    );

    // Compute z-scores
    let n = planets.len() as f64;
    let mean = |f: &dyn Fn(&Planet) -> f64| planets.iter().map(f).sum::<f64>() / n;
    let std = |f: &dyn Fn(&Planet) -> f64, m: f64| {
        (planets.iter().map(|p| (f(p) - m).powi(2)).sum::<f64>() / n).sqrt()
    };

    let (mp, mr, mm, mt, me) = (
        mean(&|p| p.log_period),
        mean(&|p| p.log_radius),
        mean(&|p| p.log_mass),
        mean(&|p| p.eq_temp),
        mean(&|p| p.eccentricity),
    );
    let (sp, sr, sm, st_s, se) = (
        std(&|p| p.log_period, mp),
        std(&|p| p.log_radius, mr),
        std(&|p| p.log_mass, mm),
        std(&|p| p.eq_temp, mt),
        std(&|p| p.eccentricity, me),
    );

    // Combined anomaly score per planet
    let scores: Vec<f64> = planets
        .iter()
        .map(|p| {
            let zp = ((p.log_period - mp) / sp.max(1e-6)).abs();
            let zr = ((p.log_radius - mr) / sr.max(1e-6)).abs();
            let zm = ((p.log_mass - mm) / sm.max(1e-6)).abs();
            let zt = ((p.eq_temp - mt) / st_s.max(1e-6)).abs();
            let ze = ((p.eccentricity - me) / se.max(1e-6)).abs();
            (zp + zr + zm + zt + ze) / 5.0
        })
        .collect();

    // Lambda: anomaly score minus threshold
    let threshold = 2.0;
    let lam: Vec<f64> = scores.iter().map(|s| s - threshold).collect();

    // Build kNN graph in normalized parameter space
    let features: Vec<[f64; 5]> = planets
        .iter()
        .map(|p| {
            [
                (p.log_period - mp) / sp.max(1e-6),
                (p.log_radius - mr) / sr.max(1e-6),
                (p.log_mass - mm) / sm.max(1e-6),
                (p.eq_temp - mt) / st_s.max(1e-6),
                (p.eccentricity - me) / se.max(1e-6),
            ]
        })
        .collect();

    let k = 5;
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..planets.len() {
        let mut dists: Vec<(usize, f64)> = (0..planets.len())
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = (0..5)
                    .map(|d| (features[i][d] - features[j][d]).powi(2))
                    .sum();
                (j, d.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(j, d) in dists.iter().take(k) {
            let w = 1.0 / (1.0 + d);
            edges.push((i, j, w));
        }
    }

    let flagged = solve_mincut(&lam, &edges, 0.3);
    let n_flagged = flagged.iter().filter(|&&x| x).count();
    println!(
        "  Graph-cut flagged {} / {} planets as anomalous ({:.1}%)\n",
        n_flagged,
        planets.len(),
        n_flagged as f64 / planets.len() as f64 * 100.0
    );

    // Sort by score, show top 20
    let mut ranked: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .filter(|(i, _)| flagged[*i])
        .map(|(i, &s)| (i, s))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!(
        "  {:>3} {:<30} {:>6} {:>6} {:>8} {:>6} {:>5} {:<14}",
        "#", "Planet", "Pscore", "ZPer", "ZRad", "ZMass", "ZTemp", "Method"
    );
    println!(
        "  {:-<3} {:-<30} {:-<6} {:-<6} {:-<8} {:-<6} {:-<5} {:-<14}",
        "", "", "", "", "", "", "", ""
    );
    for (rank, &(i, score)) in ranked.iter().take(20).enumerate() {
        let p = &planets[i];
        let zp = ((p.log_period - mp) / sp.max(1e-6)).abs();
        let zr = ((p.log_radius - mr) / sr.max(1e-6)).abs();
        let zm = ((p.log_mass - mm) / sm.max(1e-6)).abs();
        let zt = ((p.eq_temp - mt) / st_s.max(1e-6)).abs();
        println!(
            "  {:>3} {:<30} {:>6.2} {:>6.2} {:>8.2} {:>6.2} {:>5.1} {:<14}",
            rank + 1,
            &p.name[..p.name.len().min(30)],
            score,
            zp,
            zr,
            zm,
            zt,
            &p.method[..p.method.len().min(14)]
        );
    }

    // Interesting categories
    println!("\n  Discovery highlights:");
    let ultra_hot: Vec<_> = ranked
        .iter()
        .filter(|&&(i, _)| planets[i].eq_temp > 3000.0)
        .collect();
    if !ultra_hot.is_empty() {
        println!("   - Ultra-hot worlds (Teq > 3000K): {}", ultra_hot.len());
        for &&(i, _) in ultra_hot.iter().take(3) {
            println!(
                "     {} (Teq={:.0}K, P={:.2}d)",
                planets[i].name,
                planets[i].eq_temp,
                planets[i].log_period.exp()
            );
        }
    }
    let massive: Vec<_> = ranked
        .iter()
        .filter(|&&(i, _)| planets[i].log_mass.exp() > 5000.0)
        .collect();
    if !massive.is_empty() {
        println!("   - Super-massive (>5000 Mearth): {}", massive.len());
        for &&(i, _) in massive.iter().take(3) {
            println!(
                "     {} (M={:.0} Mearth, ecc={:.3})",
                planets[i].name,
                planets[i].log_mass.exp(),
                planets[i].eccentricity
            );
        }
    }
    let eccentric: Vec<_> = ranked
        .iter()
        .filter(|&&(i, _)| planets[i].eccentricity > 0.8)
        .collect();
    if !eccentric.is_empty() {
        println!(
            "   - Highly eccentric orbits (e > 0.8): {}",
            eccentric.len()
        );
        for &&(i, _) in eccentric.iter().take(3) {
            println!(
                "     {} (e={:.3}, P={:.1}d)",
                planets[i].name,
                planets[i].eccentricity,
                planets[i].log_period.exp()
            );
        }
    }
}

// ── 2. EARTHQUAKE CLUSTERING ────────────────────────────────────────────────

struct Quake {
    lat: f64,
    lon: f64,
    depth: f64,
    mag: f64,
    place: String,
    _day_offset: f64,
}

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (r, d2r) = (6371.0, std::f64::consts::PI / 180.0);
    let (dlat, dlon) = ((lat2 - lat1) * d2r, (lon2 - lon1) * d2r);
    let a = (dlat / 2.0).sin().powi(2)
        + (lat1 * d2r).cos() * (lat2 * d2r).cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

fn run_earthquakes() {
    println!("\n{}", "=".repeat(70));
    println!("  2. EARTHQUAKE CLUSTERING — USGS (Last 30 Days, M2.5+)");
    println!("{}\n", "=".repeat(70));

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/earthquakes.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            println!("  [SKIP] Cannot read {}: {}", path, e);
            return;
        }
    };

    let mut quakes = Vec::new();
    let lines: Vec<&str> = data.lines().collect();
    // Parse reference time from first data line
    let _ref_day: f64 = 0.0; // day offset relative to newest event
    for line in lines.iter().skip(1) {
        let f = split_csv_line(line);
        if f.len() < 15 {
            continue;
        }
        let lat = match parse_f64(&f[1]) {
            Some(v) => v,
            _ => continue,
        };
        let lon = match parse_f64(&f[2]) {
            Some(v) => v,
            _ => continue,
        };
        let depth = parse_f64(&f[3]).unwrap_or(10.0);
        let mag = match parse_f64(&f[4]) {
            Some(v) => v,
            _ => continue,
        };
        let place = parse_csv_field(&f[13]).to_string();
        let day_idx = quakes.len() as f64 * 0.5; // approximate ordering
        quakes.push(Quake {
            lat,
            lon,
            depth,
            mag,
            place,
            _day_offset: day_idx,
        });
    }
    println!("  Parsed {} earthquakes\n", quakes.len());

    // Build proximity graph: connect events within 200km
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut neighbor_count = vec![0usize; quakes.len()];
    for i in 0..quakes.len() {
        for j in (i + 1)..quakes.len() {
            let d = haversine_km(quakes[i].lat, quakes[i].lon, quakes[j].lat, quakes[j].lon);
            if d < 200.0 {
                let w = 1.0 / (1.0 + d / 50.0);
                edges.push((i, j, w));
                edges.push((j, i, w));
                neighbor_count[i] += 1;
                neighbor_count[j] += 1;
            }
        }
    }

    // Lambda: local density z-score + deep-quake bonus
    let mean_nc = neighbor_count.iter().sum::<usize>() as f64 / quakes.len() as f64;
    let std_nc = (neighbor_count
        .iter()
        .map(|&c| (c as f64 - mean_nc).powi(2))
        .sum::<f64>()
        / quakes.len() as f64)
        .sqrt();
    let mean_mag = quakes.iter().map(|q| q.mag).sum::<f64>() / quakes.len() as f64;

    let lam: Vec<f64> = quakes
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let density_z = (neighbor_count[i] as f64 - mean_nc) / std_nc.max(1e-6);
            let deep_bonus = if q.depth > 300.0 { 1.5 } else { 0.0 };
            let mag_bonus = if q.mag > mean_mag + 2.0 { 1.0 } else { 0.0 };
            density_z * 0.5 + deep_bonus + mag_bonus - 1.2
        })
        .collect();

    let flagged = solve_mincut(&lam, &edges, 0.4);
    let n_flagged = flagged.iter().filter(|&&x| x).count();
    println!(
        "  Graph-cut flagged {} / {} events as anomalous ({:.1}%)\n",
        n_flagged,
        quakes.len(),
        n_flagged as f64 / quakes.len() as f64 * 100.0
    );

    // Cluster flagged events by proximity
    let mut cluster_id = vec![0usize; quakes.len()];
    let mut next_cluster = 1usize;
    for i in 0..quakes.len() {
        if !flagged[i] || cluster_id[i] != 0 {
            continue;
        }
        let cid = next_cluster;
        next_cluster += 1;
        let mut stk = vec![i];
        while let Some(u) = stk.pop() {
            if cluster_id[u] != 0 {
                continue;
            }
            cluster_id[u] = cid;
            for j in 0..quakes.len() {
                if flagged[j] && cluster_id[j] == 0 {
                    let d =
                        haversine_km(quakes[u].lat, quakes[u].lon, quakes[j].lat, quakes[j].lon);
                    if d < 150.0 {
                        stk.push(j);
                    }
                }
            }
        }
    }

    // Report clusters
    let n_clusters = next_cluster - 1;
    println!("  Detected {} anomalous clusters/regions:\n", n_clusters);
    println!(
        "  {:>3} {:>5} {:>5} {:>7} {:>7} {:<40}",
        "Cl#", "Count", "MaxMg", "AvgDep", "AvgMag", "Representative Location"
    );
    println!(
        "  {:-<3} {:-<5} {:-<5} {:-<7} {:-<7} {:-<40}",
        "", "", "", "", "", ""
    );

    for cid in 1..next_cluster {
        let members: Vec<usize> = (0..quakes.len())
            .filter(|&i| cluster_id[i] == cid)
            .collect();
        if members.is_empty() {
            continue;
        }
        let count = members.len();
        let max_mag = members
            .iter()
            .map(|&i| quakes[i].mag)
            .fold(0.0f64, f64::max);
        let avg_depth = members.iter().map(|&i| quakes[i].depth).sum::<f64>() / count as f64;
        let avg_mag = members.iter().map(|&i| quakes[i].mag).sum::<f64>() / count as f64;
        let rep = &quakes[members[0]].place;
        if count >= 3 || max_mag >= 5.0 || avg_depth > 200.0 {
            println!(
                "  {:>3} {:>5} {:>5.1} {:>7.1} {:>7.2} {:<40}",
                cid,
                count,
                max_mag,
                avg_depth,
                avg_mag,
                &rep[..rep.len().min(40)]
            );
        }
    }

    // Deep earthquakes
    let deep: Vec<_> = quakes
        .iter()
        .enumerate()
        .filter(|(_, q)| q.depth > 300.0)
        .collect();
    if !deep.is_empty() {
        println!("\n  Deep earthquakes (> 300 km):");
        println!(
            "  {:>5} {:>7} {:>7} {:<40}",
            "Mag", "Depth", "Lat", "Location"
        );
        println!("  {:-<5} {:-<7} {:-<7} {:-<40}", "", "", "", "");
        for &(_, q) in deep.iter().take(10) {
            println!(
                "  {:>5.1} {:>7.1} {:>7.2} {:<40}",
                q.mag,
                q.depth,
                q.lat,
                &q.place[..q.place.len().min(40)]
            );
        }
    }

    // Strongest events
    let mut by_mag: Vec<usize> = (0..quakes.len()).collect();
    by_mag.sort_by(|&a, &b| quakes[b].mag.partial_cmp(&quakes[a].mag).unwrap());
    println!("\n  Top 10 strongest events:");
    println!(
        "  {:>5} {:>7} {:>7} {:>7} {:<40}",
        "Mag", "Depth", "Lat", "Lon", "Location"
    );
    println!("  {:-<5} {:-<7} {:-<7} {:-<7} {:-<40}", "", "", "", "", "");
    for &i in by_mag.iter().take(10) {
        let q = &quakes[i];
        let flag = if flagged[i] { " *" } else { "" };
        println!(
            "  {:>5.1} {:>7.1} {:>7.2} {:>7.2} {:<38}{}",
            q.mag,
            q.depth,
            q.lat,
            q.lon,
            &q.place[..q.place.len().min(38)],
            flag
        );
    }
}

// ── 3. CLIMATE REGIME DETECTION ─────────────────────────────────────────────

fn run_climate() {
    println!("\n{}", "=".repeat(70));
    println!("  3. CLIMATE REGIME DETECTION — NOAA Global Temp Anomalies 1850-2026");
    println!("{}\n", "=".repeat(70));

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/global_temp_anomaly.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            println!("  [SKIP] Cannot read {}: {}", path, e);
            return;
        }
    };

    let mut years: Vec<(i32, f64)> = Vec::new();
    for line in data.lines() {
        if line.starts_with('#') || line.starts_with("Year") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(y), Ok(a)) = (
                parts[0].trim().parse::<i32>(),
                parts[1].trim().parse::<f64>(),
            ) {
                years.push((y, a));
            }
        }
    }
    println!(
        "  Loaded {} years of temperature anomaly data ({}-{})\n",
        years.len(),
        years.first().map(|y| y.0).unwrap_or(0),
        years.last().map(|y| y.0).unwrap_or(0)
    );

    let n = years.len();
    let anomalies: Vec<f64> = years.iter().map(|y| y.1).collect();

    // CUSUM regime detection
    let global_mean = anomalies.iter().sum::<f64>() / n as f64;
    let mut cusum_pos = vec![0.0f64; n];
    let mut cusum_neg = vec![0.0f64; n];
    for i in 1..n {
        let diff = anomalies[i] - global_mean;
        cusum_pos[i] = (cusum_pos[i - 1] + diff - 0.02).max(0.0);
        cusum_neg[i] = (cusum_neg[i - 1] - diff - 0.02).max(0.0);
    }

    // Lambda: CUSUM magnitude indicates regime shift
    let cusum_max = cusum_pos
        .iter()
        .chain(cusum_neg.iter())
        .cloned()
        .fold(0.0f64, f64::max);
    let lam: Vec<f64> = (0..n)
        .map(|i| {
            let cusum_score = (cusum_pos[i] + cusum_neg[i]) / cusum_max.max(1e-6);
            cusum_score - 0.15
        })
        .collect();

    // Temporal chain graph
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n.saturating_sub(1) {
        let diff = (anomalies[i] - anomalies[i + 1]).abs();
        let w = 1.0 / (1.0 + diff * 5.0); // strong smoothing for similar temps
        edges.push((i, i + 1, w));
        edges.push((i + 1, i, w));
    }
    // Also connect with 5-year lag for longer trends
    for i in 0..n.saturating_sub(5) {
        let diff = (anomalies[i] - anomalies[i + 5]).abs();
        let w = 0.3 / (1.0 + diff * 3.0);
        edges.push((i, i + 5, w));
        edges.push((i + 5, i, w));
    }

    let regime = solve_mincut(&lam, &edges, 0.5);

    // Find transition points
    let mut transitions = Vec::new();
    for i in 1..n {
        if regime[i] != regime[i - 1] {
            transitions.push(i);
        }
    }

    println!(
        "  Graph-cut detected {} regime transition(s):\n",
        transitions.len()
    );
    println!(
        "  {:>6} {:>8} {:>12} {:>12} {:>10}",
        "Year", "Anomaly", "Before(avg)", "After(avg)", "Shift"
    );
    println!(
        "  {:-<6} {:-<8} {:-<12} {:-<12} {:-<10}",
        "", "", "", "", ""
    );

    for &ti in &transitions {
        let before_start = if ti > 10 { ti - 10 } else { 0 };
        let after_end = (ti + 10).min(n);
        let before_mean =
            anomalies[before_start..ti].iter().sum::<f64>() / (ti - before_start) as f64;
        let after_mean = anomalies[ti..after_end].iter().sum::<f64>() / (after_end - ti) as f64;
        println!(
            "  {:>6} {:>+8.2} {:>+12.3} {:>+12.3} {:>+10.3}",
            years[ti].0,
            anomalies[ti],
            before_mean,
            after_mean,
            after_mean - before_mean
        );
    }

    // Rate-of-change analysis: 10-year moving average slope
    println!("\n  Warming rate by decade (10-year moving average slope):\n");
    println!(
        "  {:>10} {:>12} {:>12}",
        "Period", "Avg Anomaly", "Rate C/dec"
    );
    println!("  {:-<10} {:-<12} {:-<12}", "", "", "");
    let decades: Vec<(i32, i32)> = vec![
        (1850, 1880),
        (1880, 1910),
        (1910, 1940),
        (1940, 1970),
        (1970, 1990),
        (1990, 2010),
        (2010, 2026),
    ];
    for &(y0, y1) in &decades {
        let vals: Vec<(f64, f64)> = years
            .iter()
            .filter(|y| y.0 >= y0 && y.0 < y1)
            .map(|y| (y.0 as f64, y.1))
            .collect();
        if vals.len() < 2 {
            continue;
        }
        let n_v = vals.len() as f64;
        let mx = vals.iter().map(|v| v.0).sum::<f64>() / n_v;
        let my = vals.iter().map(|v| v.1).sum::<f64>() / n_v;
        let slope = vals.iter().map(|v| (v.0 - mx) * (v.1 - my)).sum::<f64>()
            / vals
                .iter()
                .map(|v| (v.0 - mx).powi(2))
                .sum::<f64>()
                .max(1e-6);
        let rate_per_decade = slope * 10.0;
        println!(
            "  {:>4}-{:<4} {:>+12.3} {:>+12.3}",
            y0, y1, my, rate_per_decade
        );
    }

    // Extreme years
    let mut sorted: Vec<(i32, f64)> = years.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n  Top 10 warmest years:");
    for (rank, &(y, a)) in sorted.iter().take(10).enumerate() {
        let in_regime = if regime[years.iter().position(|yy| yy.0 == y).unwrap_or(0)] {
            "R1"
        } else {
            "R0"
        };
        println!("    {:>2}. {} {:>+.2}C [{}]", rank + 1, y, a, in_regime);
    }

    println!("\n  Top 5 coldest years:");
    for (rank, &(y, a)) in sorted.iter().rev().take(5).enumerate() {
        println!("    {:>2}. {} {:>+.2}C", rank + 1, y, a);
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("=========================================================================");
    println!("  REAL-DATA DISCOVERY PIPELINE — Graph-Cut Anomaly Detection");
    println!("  NASA Exoplanets | USGS Earthquakes | NOAA Climate");
    println!("=========================================================================");

    run_exoplanets();
    run_earthquakes();
    run_climate();

    println!("\n=========================================================================");
    println!("  All analyses complete. Anomalies flagged using Edmonds-Karp mincut.");
    println!("=========================================================================");
}
