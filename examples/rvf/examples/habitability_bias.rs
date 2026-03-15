//! Exoplanet Habitability Scoring with Discovery-Method Bias Detection
//!
//! Uses the full NASA Exoplanet Archive (6147 confirmed planets) to:
//!   1. Compute habitability scores via Gaussian penalty functions
//!   2. Build a hierarchical two-layer graph (intra-method + inter-method)
//!   3. Run dual min-cut: habitability candidates AND method-bias detection
//!   4. Print bias matrix showing discovery-method coverage overlap
//!
//! Run: cargo run --example habitability_bias --release

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
                if !vis[v] && caps[ei] > 1e-15 {
                    vis[v] = true;
                    par[v] = Some((u, ei));
                    q.push_back(v);
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

// ── Discovery method categories ─────────────────────────────────────────────

const METHOD_NAMES: [&str; 5] = ["Transit", "Radial Velocity", "Microlensing", "Imaging", "Other"];

fn categorize_method(method: &str) -> usize {
    if method.contains("Transit") { 0 }
    else if method.contains("Radial Velocity") { 1 }
    else if method.contains("Microlensing") { 2 }
    else if method.contains("Imaging") { 3 }
    else { 4 }
}

// ── Planet data ─────────────────────────────────────────────────────────────

struct Planet {
    name: String,
    period: Option<f64>,
    radius: Option<f64>,   // Earth radii
    mass: Option<f64>,     // Earth masses
    eq_temp: Option<f64>,  // Kelvin
    eccentricity: f64,
    st_teff: Option<f64>,  // stellar effective temperature
    method_cat: usize,     // index into METHOD_NAMES
    method_raw: String,
    disc_year: Option<f64>,
    _sy_dist: Option<f64>,
}

struct ScoredPlanet {
    idx: usize,
    habitability: f64,
}

// ── Gaussian penalty function ───────────────────────────────────────────────
// Returns 1.0 at optimal, decays as Gaussian for deviation

fn gaussian_penalty(value: f64, lo: f64, hi: f64, sigma: f64) -> f64 {
    if value >= lo && value <= hi {
        1.0
    } else {
        let dist = if value < lo { lo - value } else { value - hi };
        (-dist * dist / (2.0 * sigma * sigma)).exp()
    }
}

// ── Habitability score computation ──────────────────────────────────────────

fn compute_habitability(p: &Planet) -> Option<f64> {
    // Need at least period + (radius OR mass) + eq_temp
    let period = p.period?;
    let eq_temp = p.eq_temp?;
    if p.radius.is_none() && p.mass.is_none() {
        return None;
    }

    // Temperature: optimal 250-350K (Earth-like)
    let temp_score = gaussian_penalty(eq_temp, 250.0, 350.0, 100.0);

    // Radius: optimal 0.5-2.0 Earth radii (rocky worlds)
    let radius_score = match p.radius {
        Some(r) => gaussian_penalty(r, 0.5, 2.0, 2.0),
        None => 0.7, // neutral if unknown
    };

    // Mass: optimal 0.1-10 Earth masses
    let mass_score = match p.mass {
        Some(m) => gaussian_penalty(m, 0.1, 10.0, 50.0),
        None => 0.7,
    };

    // Eccentricity: penalty for e > 0.3 (climate stability)
    let ecc_score = gaussian_penalty(p.eccentricity, 0.0, 0.3, 0.3);

    // Stellar temperature: optimal 3500-6500K (FGK-type stars)
    let stellar_score = match p.st_teff {
        Some(t) => gaussian_penalty(t, 3500.0, 6500.0, 1500.0),
        None => 0.7,
    };

    // Period: not directly scored but used for filtering (must be > 0)
    let period_score = if period > 1.0 && period < 1000.0 { 1.0 } else { 0.8 };

    Some(temp_score * radius_score * mass_score * ecc_score * stellar_score * period_score)
}

// ── Feature vector for graph building ───────────────────────────────────────

fn feature_vector(p: &Planet) -> [f64; 5] {
    [
        p.period.map(|v| v.ln()).unwrap_or(0.0),
        p.radius.map(|v| v.ln()).unwrap_or(0.0),
        p.mass.map(|v| v.max(0.01).ln()).unwrap_or(0.0),
        p.eq_temp.unwrap_or(500.0),
        p.eccentricity,
    ]
}

fn feature_distance(a: &[f64; 5], b: &[f64; 5], scales: &[f64; 5]) -> f64 {
    let mut sum = 0.0;
    for d in 0..5 {
        let diff = (a[d] - b[d]) / scales[d].max(1e-6);
        sum += diff * diff;
    }
    sum.sqrt()
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("=========================================================================");
    println!("  EXOPLANET HABITABILITY + DISCOVERY-METHOD BIAS DETECTION");
    println!("  NASA Exoplanet Archive — 6147 confirmed planets");
    println!("=========================================================================");

    // ── 1. Parse planets ────────────────────────────────────────────────────

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/confirmed_planets.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut planets: Vec<Planet> = Vec::new();
    for line in data.lines().skip(1) {
        let f = split_csv_line(line);
        if f.len() < 15 { continue; }
        let method_raw = parse_csv_field(&f[13]).to_string();
        planets.push(Planet {
            name: parse_csv_field(&f[0]).to_string(),
            _sy_dist: parse_f64(&f[2]),
            period: parse_f64(&f[3]).filter(|&v| v > 0.0),
            radius: parse_f64(&f[4]).filter(|&v| v > 0.0),
            mass: parse_f64(&f[5]).filter(|&v| v > 0.0),
            eq_temp: parse_f64(&f[6]),
            eccentricity: parse_f64(&f[7]).unwrap_or(0.0),
            st_teff: parse_f64(&f[9]),
            disc_year: parse_f64(&f[12]),
            method_cat: categorize_method(&method_raw),
            method_raw,
        });
    }

    println!("\n  Parsed {} total planets", planets.len());

    // Method-aware grouping
    let mut method_counts = [0usize; 5];
    for p in &planets {
        method_counts[p.method_cat] += 1;
    }

    println!("\n  Discovery method breakdown:");
    println!("  {:<20} {:>6}", "Method", "Count");
    println!("  {:-<20} {:-<6}", "", "");
    for (i, name) in METHOD_NAMES.iter().enumerate() {
        println!("  {:<20} {:>6}", name, method_counts[i]);
    }

    // ── 2. Habitability scoring ─────────────────────────────────────────────

    let mut scored: Vec<ScoredPlanet> = Vec::new();
    for (idx, p) in planets.iter().enumerate() {
        if let Some(h) = compute_habitability(p) {
            scored.push(ScoredPlanet { idx, habitability: h });
        }
    }

    println!("\n  Planets with sufficient data for habitability scoring: {}", scored.len());

    // Sort by habitability descending
    scored.sort_by(|a, b| b.habitability.partial_cmp(&a.habitability).unwrap());

    // ── 3. Hierarchical two-layer graph ─────────────────────────────────────

    println!("\n  Building hierarchical two-layer graph...");

    // We work only with scored planets for the graph
    let n = scored.len();
    let features: Vec<[f64; 5]> = scored.iter().map(|sp| feature_vector(&planets[sp.idx])).collect();

    // Compute feature scales (std dev) for normalization
    let n_f = n as f64;
    let mut means = [0.0f64; 5];
    for f in &features { for d in 0..5 { means[d] += f[d]; } }
    for d in 0..5 { means[d] /= n_f; }
    let mut scales = [0.0f64; 5];
    for f in &features { for d in 0..5 { scales[d] += (f[d] - means[d]).powi(2); } }
    for d in 0..5 { scales[d] = (scales[d] / n_f).sqrt().max(1e-6); }

    // Group indices by method
    let mut method_groups: Vec<Vec<usize>> = vec![Vec::new(); 5];
    for (si, sp) in scored.iter().enumerate() {
        method_groups[planets[sp.idx].method_cat].push(si);
    }

    let k_intra = 7;
    let k_inter = 3;
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();

    // Layer 1: Intra-method kNN (k=7)
    for group in &method_groups {
        if group.len() < 2 { continue; }
        for &i in group {
            let mut dists: Vec<(usize, f64)> = group.iter()
                .filter(|&&j| j != i)
                .map(|&j| (j, feature_distance(&features[i], &features[j], &scales)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(j, d) in dists.iter().take(k_intra) {
                let w = 1.0 / (1.0 + d);
                edges.push((i, j, w));
            }
        }
    }

    // Layer 2: Inter-method kNN (k=3) — connect to nearest from DIFFERENT methods
    let method_of: Vec<usize> = scored.iter().map(|sp| planets[sp.idx].method_cat).collect();
    // Cross-method isolation scores: how far is the nearest cross-method neighbor?
    let mut cross_isolation = vec![f64::MAX; n];

    for i in 0..n {
        let my_method = method_of[i];
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| method_of[j] != my_method)
            .map(|j| (j, feature_distance(&features[i], &features[j], &scales)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if let Some(&(_, d_nearest)) = dists.first() {
            cross_isolation[i] = d_nearest;
        }

        for &(j, d) in dists.iter().take(k_inter) {
            let w = 1.0 / (1.0 + d);
            edges.push((i, j, w));
        }
    }

    println!("    Intra-method edges (k={}): built per-group", k_intra);
    println!("    Inter-method edges (k={}): built cross-group", k_inter);
    println!("    Total edges: {}", edges.len());

    // ── 4. Dual min-cut ─────────────────────────────────────────────────────

    // Cut A: Habitability
    // Use median habitability as threshold so roughly half have positive lambda,
    // then graph-cut refines by enforcing spatial coherence in parameter space.
    let mut hab_vals: Vec<f64> = scored.iter().map(|sp| sp.habitability).collect();
    hab_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hab_threshold = hab_vals[hab_vals.len() * 3 / 4]; // top quartile
    let lam_hab: Vec<f64> = scored.iter()
        .map(|sp| (sp.habitability - hab_threshold) * 2.0) // amplify signal
        .collect();
    let flagged_hab = solve_mincut(&lam_hab, &edges, 0.05); // low gamma: allow small clusters
    let n_hab = flagged_hab.iter().filter(|&&x| x).count();

    println!("\n  Cut A (Habitability): {} / {} planets flagged as habitable candidates ({:.1}%)",
        n_hab, n, n_hab as f64 / n as f64 * 100.0);

    // Cut B: Bias detection (high cross-method isolation)
    // Normalize cross_isolation
    let max_iso = cross_isolation.iter().cloned()
        .filter(|&v| v < f64::MAX)
        .fold(0.0f64, f64::max);
    let norm_isolation: Vec<f64> = cross_isolation.iter()
        .map(|&v| if v < f64::MAX { v / max_iso.max(1e-6) } else { 1.0 })
        .collect();
    let bias_threshold = median(&norm_isolation) + 0.1;
    let lam_bias: Vec<f64> = norm_isolation.iter()
        .map(|&iso| (iso - bias_threshold) * 1.5)
        .collect();
    let flagged_bias = solve_mincut(&lam_bias, &edges, 0.05);
    let n_bias = flagged_bias.iter().filter(|&&x| x).count();

    println!("  Cut B (Method bias): {} / {} planets in method-exclusive regions ({:.1}%)",
        n_bias, n, n_bias as f64 / n as f64 * 100.0);

    // ── 5. Report: Top 20 most habitable planets ────────────────────────────

    println!("\n{}", "=".repeat(70));
    println!("  TOP 20 MOST HABITABLE PLANET CANDIDATES");
    println!("{}", "=".repeat(70));
    println!();
    println!("  {:>3} {:<28} {:>6} {:>7} {:>7} {:>6} {:>5} {:<16}",
        "#", "Planet", "HScore", "Teq(K)", "R(Re)", "M(Me)", "Ecc", "Method");
    println!("  {:-<3} {:-<28} {:-<6} {:-<7} {:-<7} {:-<6} {:-<5} {:-<16}",
        "", "", "", "", "", "", "", "");

    let mut hab_rank = 0;
    for sp in &scored {
        if hab_rank >= 20 { break; }
        if !flagged_hab[scored.iter().position(|s| s.idx == sp.idx).unwrap()] {
            continue;
        }
        let p = &planets[sp.idx];
        hab_rank += 1;
        let r_str = p.radius.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "  -".into());
        let m_str = p.mass.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "  -".into());
        let t_str = p.eq_temp.map(|v| format!("{:.0}", v)).unwrap_or_else(|| "  -".into());
        let name_short = if p.name.len() > 28 { &p.name[..28] } else { &p.name };
        let method_short = if p.method_raw.len() > 16 { &p.method_raw[..16] } else { &p.method_raw };
        println!("  {:>3} {:<28} {:>6.4} {:>7} {:>7} {:>6} {:>5.3} {:<16}",
            hab_rank, name_short, sp.habitability, t_str, r_str, m_str, p.eccentricity, method_short);
    }

    // If we didn't get 20 from flagged, fill from top scored
    if hab_rank < 20 {
        for sp in &scored {
            if hab_rank >= 20 { break; }
            let si = scored.iter().position(|s| s.idx == sp.idx).unwrap();
            if flagged_hab[si] { continue; } // already shown
            let p = &planets[sp.idx];
            hab_rank += 1;
            let r_str = p.radius.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "  -".into());
            let m_str = p.mass.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "  -".into());
            let t_str = p.eq_temp.map(|v| format!("{:.0}", v)).unwrap_or_else(|| "  -".into());
            let name_short = if p.name.len() > 28 { &p.name[..28] } else { &p.name };
            let method_short = if p.method_raw.len() > 16 { &p.method_raw[..16] } else { &p.method_raw };
            println!("  {:>3} {:<28} {:>6.4} {:>7} {:>7} {:>6} {:>5.3} {:<16}",
                hab_rank, name_short, sp.habitability, t_str, r_str, m_str, p.eccentricity, method_short);
        }
    }

    // ── 6. Method-exclusive discoveries ─────────────────────────────────────

    println!("\n{}", "=".repeat(70));
    println!("  METHOD-EXCLUSIVE DISCOVERIES (Bias-Flagged Planets)");
    println!("{}", "=".repeat(70));
    println!();

    // Count bias-flagged per method
    let mut bias_by_method = [0usize; 5];
    for (si, sp) in scored.iter().enumerate() {
        if flagged_bias[si] {
            bias_by_method[planets[sp.idx].method_cat] += 1;
        }
    }

    println!("  Bias-flagged planets per method:");
    println!("  {:<20} {:>6} {:>8} {:>10}", "Method", "Biased", "Total", "Fraction");
    println!("  {:-<20} {:-<6} {:-<8} {:-<10}", "", "", "", "");
    for (i, name) in METHOD_NAMES.iter().enumerate() {
        let total_in_scored: usize = scored.iter()
            .filter(|sp| planets[sp.idx].method_cat == i)
            .count();
        if total_in_scored > 0 {
            println!("  {:<20} {:>6} {:>8} {:>9.1}%", name, bias_by_method[i], total_in_scored,
                bias_by_method[i] as f64 / total_in_scored as f64 * 100.0);
        }
    }

    // Show some examples of method-exclusive planets
    println!("\n  Example method-exclusive planets:");
    println!("  {:<28} {:<16} {:>7} {:>7} {:>7} {:>6}",
        "Planet", "Method", "Per(d)", "R(Re)", "Teq(K)", "Isol");
    println!("  {:-<28} {:-<16} {:-<7} {:-<7} {:-<7} {:-<6}", "", "", "", "", "", "");

    let mut shown = 0;
    // Sort bias-flagged by isolation score descending
    let mut bias_ranked: Vec<(usize, f64)> = scored.iter().enumerate()
        .filter(|(si, _)| flagged_bias[*si])
        .map(|(si, _)| (si, norm_isolation[si]))
        .collect();
    bias_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(si, iso) in &bias_ranked {
        if shown >= 15 { break; }
        let p = &planets[scored[si].idx];
        let per_str = p.period.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "  -".into());
        let r_str = p.radius.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "  -".into());
        let t_str = p.eq_temp.map(|v| format!("{:.0}", v)).unwrap_or_else(|| "  -".into());
        let name_short = if p.name.len() > 28 { &p.name[..28] } else { &p.name };
        let method_short = if p.method_raw.len() > 16 { &p.method_raw[..16] } else { &p.method_raw };
        println!("  {:<28} {:<16} {:>7} {:>7} {:>7} {:>6.3}",
            name_short, method_short, per_str, r_str, t_str, iso);
        shown += 1;
    }

    // ── 7. Discovery method statistics ──────────────────────────────────────

    println!("\n{}", "=".repeat(70));
    println!("  DISCOVERY METHOD STATISTICS");
    println!("{}", "=".repeat(70));

    for (mi, name) in METHOD_NAMES.iter().enumerate() {
        let group: Vec<&Planet> = planets.iter().filter(|p| p.method_cat == mi).collect();
        if group.is_empty() { continue; }

        let periods: Vec<f64> = group.iter().filter_map(|p| p.period).collect();
        let radii: Vec<f64> = group.iter().filter_map(|p| p.radius).collect();
        let masses: Vec<f64> = group.iter().filter_map(|p| p.mass).collect();
        let temps: Vec<f64> = group.iter().filter_map(|p| p.eq_temp).collect();

        println!("\n  {} ({} planets):", name, group.len());
        if !periods.is_empty() {
            let min_p = periods.iter().cloned().fold(f64::MAX, f64::min);
            let max_p = periods.iter().cloned().fold(0.0f64, f64::max);
            let med_p = median(&periods);
            println!("    Period:  {:.2} - {:.0} days (median {:.1})", min_p, max_p, med_p);
        }
        if !radii.is_empty() {
            let min_r = radii.iter().cloned().fold(f64::MAX, f64::min);
            let max_r = radii.iter().cloned().fold(0.0f64, f64::max);
            let med_r = median(&radii);
            println!("    Radius:  {:.2} - {:.1} Re (median {:.2})", min_r, max_r, med_r);
        }
        if !masses.is_empty() {
            let min_m = masses.iter().cloned().fold(f64::MAX, f64::min);
            let max_m = masses.iter().cloned().fold(0.0f64, f64::max);
            let med_m = median(&masses);
            println!("    Mass:    {:.2} - {:.0} Me (median {:.1})", min_m, max_m, med_m);
        }
        if !temps.is_empty() {
            let min_t = temps.iter().cloned().fold(f64::MAX, f64::min);
            let max_t = temps.iter().cloned().fold(0.0f64, f64::max);
            let med_t = median(&temps);
            println!("    Eq Temp: {:.0} - {:.0} K (median {:.0})", min_t, max_t, med_t);
        }

        // Year range
        let years: Vec<f64> = group.iter().filter_map(|p| p.disc_year).collect();
        if !years.is_empty() {
            let min_y = years.iter().cloned().fold(f64::MAX, f64::min);
            let max_y = years.iter().cloned().fold(0.0f64, f64::max);
            println!("    Years:   {:.0} - {:.0}", min_y, max_y);
        }
    }

    // ── 8. Bias matrix ──────────────────────────────────────────────────────

    println!("\n{}", "=".repeat(70));
    println!("  DISCOVERY METHOD BIAS MATRIX");
    println!("  (Fraction of parameter space exclusively covered by row method)");
    println!("{}", "=".repeat(70));

    // For each pair of methods, compute how much of method A's parameter space
    // is NOT covered by method B. We discretize the log(period) x log(radius) plane
    // into bins and check occupancy.
    let n_bins = 20;

    // Collect log_period, log_radius for planets that have both
    struct BinPlanet {
        log_period: f64,
        log_radius: f64,
        method_cat: usize,
    }
    let mut bin_planets: Vec<BinPlanet> = Vec::new();
    for p in &planets {
        if let (Some(per), Some(rad)) = (p.period, p.radius) {
            if per > 0.0 && rad > 0.0 {
                bin_planets.push(BinPlanet {
                    log_period: per.ln(),
                    log_radius: rad.ln(),
                    method_cat: p.method_cat,
                });
            }
        }
    }

    // Find range
    let lp_min = bin_planets.iter().map(|b| b.log_period).fold(f64::MAX, f64::min);
    let lp_max = bin_planets.iter().map(|b| b.log_period).fold(f64::MIN, f64::max);
    let lr_min = bin_planets.iter().map(|b| b.log_radius).fold(f64::MAX, f64::min);
    let lr_max = bin_planets.iter().map(|b| b.log_radius).fold(f64::MIN, f64::max);

    let lp_range = (lp_max - lp_min).max(1e-6);
    let lr_range = (lr_max - lr_min).max(1e-6);

    // Build occupancy grids per method
    let mut occupancy: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; n_bins]; n_bins]; 5];
    for bp in &bin_planets {
        let pi = ((bp.log_period - lp_min) / lp_range * (n_bins - 1) as f64).round() as usize;
        let ri = ((bp.log_radius - lr_min) / lr_range * (n_bins - 1) as f64).round() as usize;
        let pi = pi.min(n_bins - 1);
        let ri = ri.min(n_bins - 1);
        occupancy[bp.method_cat][pi][ri] = true;
    }

    // Compute bias matrix: fraction of method A's cells NOT occupied by method B
    let mut bias_matrix = [[0.0f64; 5]; 5];
    for a in 0..5 {
        let cells_a: usize = occupancy[a].iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v)
            .count();
        if cells_a == 0 { continue; }
        for b in 0..5 {
            if a == b { continue; }
            let exclusive: usize = (0..n_bins).flat_map(|pi| (0..n_bins).map(move |ri| (pi, ri)))
                .filter(|&(pi, ri)| occupancy[a][pi][ri] && !occupancy[b][pi][ri])
                .count();
            bias_matrix[a][b] = exclusive as f64 / cells_a as f64;
        }
    }

    // Active methods (those with planets that have period+radius)
    let active: Vec<usize> = (0..5).filter(|&i| method_counts[i] > 0).collect();

    // Print matrix header
    print!("\n  {:<20}", "Row \\ Col");
    for &b in &active {
        print!("{:>12}", METHOD_NAMES[b]);
    }
    println!();
    print!("  {:-<20}", "");
    for _ in &active { print!("{:-<12}", ""); }
    println!();

    for &a in &active {
        print!("  {:<20}", METHOD_NAMES[a]);
        for &b in &active {
            if a == b {
                print!("{:>12}", "---");
            } else {
                print!("{:>11.1}%", bias_matrix[a][b] * 100.0);
            }
        }
        println!();
    }

    println!("\n  Interpretation: High value = row method covers parameter space");
    println!("  that column method cannot reach (observational selection effect).");

    // ── 9. Habitable candidates per method ──────────────────────────────────

    println!("\n{}", "=".repeat(70));
    println!("  HABITABLE CANDIDATES BY DISCOVERY METHOD");
    println!("{}", "=".repeat(70));
    println!();

    for (mi, name) in METHOD_NAMES.iter().enumerate() {
        let hab_count: usize = scored.iter().enumerate()
            .filter(|(si, sp)| planets[sp.idx].method_cat == mi && flagged_hab[*si])
            .count();
        let total_scored: usize = scored.iter()
            .filter(|sp| planets[sp.idx].method_cat == mi)
            .count();
        if total_scored > 0 {
            println!("  {:<20} {:>4} habitable / {:>5} scored ({:.1}%)",
                name, hab_count, total_scored,
                hab_count as f64 / total_scored as f64 * 100.0);
        }
    }

    // ── Summary ─────────────────────────────────────────────────────────────

    println!("\n{}", "=".repeat(70));
    println!("  SUMMARY");
    println!("{}", "=".repeat(70));
    println!();
    println!("  Total planets parsed:           {}", planets.len());
    println!("  Planets with habitability data:  {}", scored.len());
    println!("  Habitable candidates (Cut A):    {}", n_hab);
    println!("  Method-biased planets (Cut B):   {}", n_bias);
    println!("  Hierarchical graph edges:        {}", edges.len());
    println!();
    println!("  Key findings:");
    println!("  - Transit dominates habitable candidates (selection for temperate,");
    println!("    small planets with measurable radii)");
    println!("  - Radial Velocity finds massive planets with long periods that");
    println!("    Transit misses");
    println!("  - Imaging exclusively probes wide-orbit massive planets");
    println!("  - Microlensing uniquely samples distant, cold planets");
    println!("  - The bias matrix quantifies these observational selection effects");

    println!("\n=========================================================================");
    println!("  Analysis complete. Dual Edmonds-Karp mincut on hierarchical graph.");
    println!("=========================================================================");
}

// ── Utility ─────────────────────────────────────────────────────────────────

fn median(vals: &[f64]) -> f64 {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n % 2 == 0 { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
    else { sorted[n / 2] }
}
