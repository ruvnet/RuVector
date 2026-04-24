//! Multi-Resolution Climate Tipping Point Detection
//!
//! Analyses 177 years of NOAA global temperature anomaly data (1850-2026) using
//! multi-scale graph-cut regime detection with cross-scale coherence analysis.
//!
//! Methodology:
//!   1. Multi-scale smoothing (raw, 5yr, 10yr, 20yr moving averages)
//!   2. Per-scale derivative computation (rate, acceleration, jerk)
//!   3. Per-scale temporal graph-cut (Edmonds-Karp min-cut)
//!   4. Cross-scale meta-graph coherence (final min-cut on unified graph)
//!   5. Confidence scoring (count of scales agreeing on each transition)
//!   6. Paris Agreement threshold projection via quadratic fit
//!
//! Run: cargo run --example climate_tipping --release

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

// ── Smoothing ───────────────────────────────────────────────────────────────

fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let half = window / 2;
    (0..n)
        .map(|i| {
            let lo = if i >= half { i - half } else { 0 };
            let hi = (i + half + 1).min(n);
            let count = hi - lo;
            data[lo..hi].iter().sum::<f64>() / count as f64
        })
        .collect()
}

// ── Derivative computation ──────────────────────────────────────────────────

fn first_derivative(series: &[f64]) -> Vec<f64> {
    let n = series.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        d[i] = (series[i + 1] - series[i - 1]) / 2.0;
    }
    d[0] = series[1] - series[0];
    d[n - 1] = series[n - 1] - series[n - 2];
    d
}

fn compute_derivative(series: &[f64]) -> Vec<f64> {
    first_derivative(series)
}

// ── Per-scale temporal graph-cut ────────────────────────────────────────────

/// Build a temporal graph-cut that partitions the series into regimes.
///
/// Uses a sliding-window change-point score: at each year, compare the mean
/// of a window before that year to the mean of a window after. Large jumps
/// indicate regime boundaries. The min-cut then enforces spatial (temporal)
/// coherence so only significant, sustained shifts are flagged.
fn per_scale_cut(
    series: &[f64],
    second_deriv: &[f64],
    half_window: usize,
    gamma: f64,
) -> Vec<bool> {
    let n = series.len();

    // Sliding-window change-point score
    let mut cp_score = vec![0.0f64; n];
    for i in half_window..n.saturating_sub(half_window) {
        let before: f64 = series[i - half_window..i].iter().sum::<f64>() / half_window as f64;
        let after: f64 = series[i..i + half_window].iter().sum::<f64>() / half_window as f64;
        cp_score[i] = (after - before).abs();
    }

    // Normalize change-point scores
    let cp_max = cp_score.iter().cloned().fold(0.0f64, f64::max).max(1e-8);
    let cp_mean = cp_score.iter().sum::<f64>() / n as f64;

    // Second-derivative boost
    let d2_std = (second_deriv.iter().map(|v| v * v).sum::<f64>() / n as f64)
        .sqrt()
        .max(1e-8);

    // Lambda: change-point score above mean = potential transition
    let lam: Vec<f64> = (0..n)
        .map(|i| {
            let cp_normalized = cp_score[i] / cp_max;
            let d2_boost = (second_deriv[i].abs() / d2_std - 1.0).max(0.0) * 0.05;
            cp_normalized + d2_boost - (cp_mean / cp_max) - 0.05
        })
        .collect();

    // Build temporal chain graph
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n.saturating_sub(1) {
        let diff = (series[i] - series[i + 1]).abs();
        let w = 1.0 / (1.0 + diff * 5.0);
        edges.push((i, i + 1, w));
        edges.push((i + 1, i, w));
    }
    // Also connect with 5-year lag for longer trends
    for i in 0..n.saturating_sub(5) {
        let diff = (series[i] - series[i + 5]).abs();
        let w = 0.3 / (1.0 + diff * 3.0);
        edges.push((i, i + 5, w));
        edges.push((i + 5, i, w));
    }

    solve_mincut(&lam, &edges, gamma)
}

// ── Transition finder ───────────────────────────────────────────────────────

fn find_transitions(regime: &[bool]) -> Vec<usize> {
    let mut trans = Vec::new();
    for i in 1..regime.len() {
        if regime[i] != regime[i - 1] {
            trans.push(i);
        }
    }
    trans
}

// ── Linear regression ───────────────────────────────────────────────────────

fn linreg(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let ss_xy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let ss_xx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let slope = ss_xy / ss_xx.max(1e-12);
    let intercept = my - slope * mx;
    (slope, intercept)
}

// ── Quadratic fit (least squares) ───────────────────────────────────────────

fn quadratic_fit(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    // Fit y = a*x^2 + b*x + c via normal equations
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sx2: f64 = xs.iter().map(|x| x * x).sum();
    let sx3: f64 = xs.iter().map(|x| x.powi(3)).sum();
    let sx4: f64 = xs.iter().map(|x| x.powi(4)).sum();
    let sy: f64 = ys.iter().sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let sx2y: f64 = xs.iter().zip(ys).map(|(x, y)| x * x * y).sum();

    // | sx4 sx3 sx2 | |a|   |sx2y|
    // | sx3 sx2 sx  | |b| = |sxy |
    // | sx2 sx  n   | |c|   |sy  |
    // Solve via Cramer's rule
    let det = |m: [[f64; 3]; 3]| -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };

    let d = det([[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, n]]);
    let da = det([[sx2y, sx3, sx2], [sxy, sx2, sx], [sy, sx, n]]);
    let db = det([[sx4, sx2y, sx2], [sx3, sxy, sx], [sx2, sy, n]]);
    let dc = det([[sx4, sx3, sx2y], [sx3, sx2, sxy], [sx2, sx, sy]]);

    (da / d, db / d, dc / d)
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("=========================================================================");
    println!("  MULTI-RESOLUTION CLIMATE TIPPING POINT DETECTION");
    println!("  NOAA Global Temperature Anomalies 1850-2026 (177 years)");
    println!("  Method: Multi-scale graph-cut with cross-scale coherence");
    println!("=========================================================================");

    // ── Load data ───────────────────────────────────────────────────────────
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/global_temp_anomaly.csv");
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  Cannot read {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut years: Vec<(i32, f64)> = Vec::new();
    for line in data.lines() {
        if line.starts_with('#') || line.starts_with("Year") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Some(y), Some(a)) = (parse_f64(parts[0]).map(|v| v as i32), parse_f64(parts[1]))
            {
                years.push((y, a));
            }
        }
    }

    let n = years.len();
    let year_labels: Vec<i32> = years.iter().map(|y| y.0).collect();
    let anomalies: Vec<f64> = years.iter().map(|y| y.1).collect();

    println!(
        "\n  Loaded {} years of data ({}-{})\n",
        n,
        year_labels.first().unwrap_or(&0),
        year_labels.last().unwrap_or(&0)
    );

    // ══════════════════════════════════════════════════════════════════════
    // 1. MULTI-SCALE SMOOTHING
    // ══════════════════════════════════════════════════════════════════════

    let scale_names = ["Raw (1yr)", "5yr MA", "10yr MA", "20yr MA"];
    let scales: Vec<Vec<f64>> = vec![
        anomalies.clone(),
        moving_average(&anomalies, 5),
        moving_average(&anomalies, 10),
        moving_average(&anomalies, 20),
    ];

    println!("{}", "=".repeat(70));
    println!("  1. MULTI-SCALE SMOOTHING");
    println!("{}", "=".repeat(70));
    println!("\n  Scale         | Min      | Max      | Range    | Std Dev");
    println!(
        "  {:-<14}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<8}",
        "", "", "", "", ""
    );
    for (i, s) in scales.iter().enumerate() {
        let mn = s.iter().cloned().fold(f64::MAX, f64::min);
        let mx = s.iter().cloned().fold(f64::MIN, f64::max);
        let mean = s.iter().sum::<f64>() / n as f64;
        let std = (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        println!(
            "  {:<14} | {:>+7.3}C | {:>+7.3}C | {:>7.3}C | {:>7.4}",
            scale_names[i],
            mn,
            mx,
            mx - mn,
            std
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // 2. DERIVATIVE COMPUTATION
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  2. DERIVATIVE ANALYSIS (per scale)");
    println!("{}", "=".repeat(70));

    struct ScaleDerivatives {
        d1: Vec<f64>, // first derivative (rate)
        d2: Vec<f64>, // second derivative (acceleration)
        d3: Vec<f64>, // third derivative (jerk)
    }

    let mut all_derivs: Vec<ScaleDerivatives> = Vec::new();
    for (i, s) in scales.iter().enumerate() {
        let d1 = compute_derivative(s);
        let d2 = compute_derivative(&d1);
        let d3 = compute_derivative(&d2);

        let max_rate = d1.iter().cloned().fold(f64::MIN, f64::max);
        let max_accel = d2.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let max_jerk = d3.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        // Find year of max warming rate
        let max_rate_idx = d1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        println!("\n  {} :", scale_names[i]);
        println!(
            "    Max warming rate:   {:>+.4} C/yr  (year {})",
            max_rate, year_labels[max_rate_idx]
        );
        println!("    Max |acceleration|: {:>.5} C/yr2", max_accel);
        println!("    Max |jerk|:         {:>.5} C/yr3", max_jerk);

        all_derivs.push(ScaleDerivatives { d1, d2, d3 });
    }

    // ══════════════════════════════════════════════════════════════════════
    // 3. PER-SCALE TEMPORAL GRAPH-CUT
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  3. PER-SCALE GRAPH-CUT REGIME DETECTION");
    println!("{}", "=".repeat(70));

    // Half-window sizes for change-point detection per scale
    let half_windows: [usize; 4] = [10, 15, 20, 25];
    let gammas = [0.3, 0.4, 0.5, 0.6];

    let mut scale_regimes: Vec<Vec<bool>> = Vec::new();
    let mut scale_transitions: Vec<Vec<usize>> = Vec::new();

    for (si, s) in scales.iter().enumerate() {
        let regime = per_scale_cut(s, &all_derivs[si].d2, half_windows[si], gammas[si]);
        let trans = find_transitions(&regime);

        println!(
            "\n  {} — {} transition(s) detected:",
            scale_names[si],
            trans.len()
        );
        if trans.is_empty() {
            println!("    (no transitions found at this scale)");
        } else {
            println!(
                "    {:>6} {:>8} {:>12} {:>12} {:>10} {:>10}",
                "Year", "Anomaly", "Before(avg)", "After(avg)", "Shift", "Rate(d1)"
            );
            println!(
                "    {:-<6} {:-<8} {:-<12} {:-<12} {:-<10} {:-<10}",
                "", "", "", "", "", ""
            );
            for &ti in &trans {
                let before_start = if ti > 10 { ti - 10 } else { 0 };
                let after_end = (ti + 10).min(n);
                let before_mean =
                    scales[si][before_start..ti].iter().sum::<f64>() / (ti - before_start) as f64;
                let after_mean =
                    scales[si][ti..after_end].iter().sum::<f64>() / (after_end - ti) as f64;
                println!(
                    "    {:>6} {:>+8.3} {:>+12.3} {:>+12.3} {:>+10.3} {:>+10.4}",
                    year_labels[ti],
                    scales[si][ti],
                    before_mean,
                    after_mean,
                    after_mean - before_mean,
                    all_derivs[si].d1[ti]
                );
            }
        }

        scale_transitions.push(trans);
        scale_regimes.push(regime);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 4. META-GRAPH CROSS-SCALE COHERENCE
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  4. CROSS-SCALE META-GRAPH COHERENCE ANALYSIS");
    println!("{}", "=".repeat(70));

    let n_scales = 4;
    let meta_n = n * n_scales; // total nodes in meta-graph

    // Node index: scale * n + year_idx
    let node_id = |scale: usize, year_idx: usize| -> usize { scale * n + year_idx };

    // Build meta-graph lambda: sliding-window change-point + regime prior
    let mut meta_lam = vec![0.0f64; meta_n];
    for si in 0..n_scales {
        let hw = half_windows[si];
        let mut cp = vec![0.0f64; n];
        for yi in hw..n.saturating_sub(hw) {
            let before: f64 = scales[si][yi - hw..yi].iter().sum::<f64>() / hw as f64;
            let after: f64 = scales[si][yi..yi + hw].iter().sum::<f64>() / hw as f64;
            cp[yi] = (after - before).abs();
        }
        let cp_max = cp.iter().cloned().fold(0.0f64, f64::max).max(1e-8);
        let cp_mean = cp.iter().sum::<f64>() / n as f64;
        for yi in 0..n {
            let cp_norm = cp[yi] / cp_max;
            let regime_prior = if scale_regimes[si][yi] { 0.15 } else { -0.05 };
            meta_lam[node_id(si, yi)] = cp_norm + regime_prior - (cp_mean / cp_max) - 0.02;
        }
    }

    // Build meta-graph edges
    let mut meta_edges: Vec<(usize, usize, f64)> = Vec::new();

    // Within each scale: temporal chain edges
    for si in 0..n_scales {
        for yi in 0..n.saturating_sub(1) {
            let diff = (scales[si][yi] - scales[si][yi + 1]).abs();
            let w = 1.0 / (1.0 + diff * 5.0);
            meta_edges.push((node_id(si, yi), node_id(si, yi + 1), w));
            meta_edges.push((node_id(si, yi + 1), node_id(si, yi), w));
        }
    }

    // Cross-scale: connect same year across adjacent scales
    for si in 0..n_scales - 1 {
        for yi in 0..n {
            let diff = (scales[si][yi] - scales[si + 1][yi]).abs();
            let w = 1.0 / (1.0 + diff * 3.0);
            meta_edges.push((node_id(si, yi), node_id(si + 1, yi), w));
            meta_edges.push((node_id(si + 1, yi), node_id(si, yi), w));
        }
    }
    // Also connect non-adjacent scales (raw <-> 10yr, 5yr <-> 20yr)
    for &(sa, sb) in &[(0usize, 2usize), (1usize, 3usize)] {
        for yi in 0..n {
            let diff = (scales[sa][yi] - scales[sb][yi]).abs();
            let w = 0.5 / (1.0 + diff * 3.0);
            meta_edges.push((node_id(sa, yi), node_id(sb, yi), w));
            meta_edges.push((node_id(sb, yi), node_id(sa, yi), w));
        }
    }

    let meta_regime = solve_mincut(&meta_lam, &meta_edges, 0.5);

    // Extract per-scale results from meta-graph
    let mut meta_scale_regimes: Vec<Vec<bool>> = Vec::new();
    for si in 0..n_scales {
        let regime: Vec<bool> = (0..n).map(|yi| meta_regime[node_id(si, yi)]).collect();
        meta_scale_regimes.push(regime);
    }

    // Find transitions in meta-graph per scale
    let mut meta_transitions: Vec<Vec<usize>> = Vec::new();
    for si in 0..n_scales {
        meta_transitions.push(find_transitions(&meta_scale_regimes[si]));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 5. TIPPING-POINT CONFIDENCE SCORING
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  5. TIPPING-POINT CONFIDENCE SCORING");
    println!("{}", "=".repeat(70));

    // Count how many scales flag each year (within +/-3 year tolerance)
    let tolerance = 3;
    let mut year_confidence: Vec<(usize, usize, Vec<usize>)> = Vec::new(); // (year_idx, count, which_scales)

    // Collect all unique transition years across all scales from meta-graph
    let mut all_trans_years: Vec<usize> = Vec::new();
    for si in 0..n_scales {
        for &ti in &meta_transitions[si] {
            all_trans_years.push(ti);
        }
    }
    all_trans_years.sort();
    all_trans_years.dedup();

    // Merge nearby transitions
    let mut merged_years: Vec<usize> = Vec::new();
    for &yi in &all_trans_years {
        if merged_years.is_empty() || yi > merged_years.last().unwrap() + tolerance {
            merged_years.push(yi);
        } else {
            // Keep the one closest to the center of the cluster
            let last = merged_years.last_mut().unwrap();
            *last = (*last + yi) / 2;
        }
    }

    for &center in &merged_years {
        let mut count = 0;
        let mut which_scales = Vec::new();
        for si in 0..n_scales {
            for &ti in &meta_transitions[si] {
                let diff = if ti > center {
                    ti - center
                } else {
                    center - ti
                };
                if diff <= tolerance {
                    count += 1;
                    which_scales.push(si);
                    break;
                }
            }
        }
        if count > 0 {
            year_confidence.push((center, count, which_scales));
        }
    }

    // Sort by confidence (descending), then by year
    year_confidence.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    println!("\n  Cross-scale coherent tipping points (meta-graph consensus):\n");
    println!(
        "  {:>6} {:>10} {:>8} {:>12} {:<30}",
        "Year", "Confidence", "Anomaly", "WarmRate", "Scales Agreeing"
    );
    println!(
        "  {:-<6} {:-<10} {:-<8} {:-<12} {:-<30}",
        "", "", "", "", ""
    );

    for &(yi, count, ref which) in &year_confidence {
        if yi >= n {
            continue;
        }
        let conf_str = match count {
            4 => "VERY HIGH",
            3 => "HIGH",
            2 => "MODERATE",
            1 => "LOW",
            _ => "?",
        };
        let scale_list: String = which
            .iter()
            .map(|&s| scale_names[s])
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  {:>6} {:>10} {:>+8.3} {:>+12.4} {}",
            year_labels[yi], conf_str, anomalies[yi], all_derivs[0].d1[yi], scale_list
        );
    }

    // High-confidence tipping points (3+ scales)
    let high_conf: Vec<_> = year_confidence
        .iter()
        .filter(|&&(_, count, _)| count >= 3)
        .collect();

    println!("\n  High-confidence tipping points (3+ scales agree):");
    if high_conf.is_empty() {
        println!("    None detected at this threshold. Showing top transitions instead.");
        for &(yi, count, ref _which) in year_confidence.iter().take(5) {
            if yi < n {
                println!("    {} ({}/4 scales)", year_labels[yi], count);
            }
        }
    } else {
        for &&(yi, count, ref _which) in &high_conf {
            if yi < n {
                println!(
                    "    {} ({}/4 scales, anomaly {:>+.3}C)",
                    year_labels[yi], count, anomalies[yi]
                );
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Per-scale transition summary table
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  PER-SCALE TRANSITION SUMMARY (independent cuts)");
    println!("{}", "=".repeat(70));

    for si in 0..n_scales {
        let trans = &scale_transitions[si];
        println!("\n  {} : {} transitions", scale_names[si], trans.len());
        for &ti in trans {
            if ti < n {
                println!("    {} (anomaly {:>+.3}C)", year_labels[ti], scales[si][ti]);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 6. PARIS AGREEMENT PROJECTION
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  6. PARIS AGREEMENT THRESHOLD PROJECTION");
    println!("{}", "=".repeat(70));

    // Fit quadratic to post-2000 data
    let post2000: Vec<(f64, f64)> = years
        .iter()
        .filter(|y| y.0 >= 2000)
        .map(|y| (y.0 as f64, y.1))
        .collect();

    if post2000.len() >= 3 {
        let xs: Vec<f64> = post2000.iter().map(|v| v.0).collect();
        let ys: Vec<f64> = post2000.iter().map(|v| v.1).collect();
        let (a, b, c) = quadratic_fit(&xs, &ys);

        println!(
            "\n  Quadratic fit (post-2000): anomaly = {:.6}*yr^2 + {:.4}*yr + {:.2}",
            a, b, c
        );
        println!(
            "  (based on {} data points: {}-{})",
            post2000.len(),
            post2000.first().unwrap().0 as i32,
            post2000.last().unwrap().0 as i32
        );

        // Find crossing years for 1.5C and 2.0C
        for &target in &[1.5, 2.0] {
            // Solve a*x^2 + b*x + (c - target) = 0
            let c_adj = c - target;
            let discriminant = b * b - 4.0 * a * c_adj;
            if discriminant >= 0.0 && a.abs() > 1e-12 {
                let x1 = (-b + discriminant.sqrt()) / (2.0 * a);
                let x2 = (-b - discriminant.sqrt()) / (2.0 * a);
                // Pick the root in a reasonable future range
                let crossing = if x1 > 2000.0 && x1 < 2200.0 {
                    if x2 > 2000.0 && x2 < 2200.0 {
                        x1.min(x2)
                    } else {
                        x1
                    }
                } else if x2 > 2000.0 && x2 < 2200.0 {
                    x2
                } else {
                    // If no root in range, extrapolate linearly
                    let (slope, intercept) = linreg(&xs, &ys);
                    (target - intercept) / slope
                };
                println!("\n  +{:.1}C threshold crossing: ~{:.0}", target, crossing);
                let predicted_at_crossing = a * crossing * crossing + b * crossing + c;
                println!(
                    "    Predicted anomaly at crossing: {:>+.3}C",
                    predicted_at_crossing
                );
            } else {
                // Fallback to linear fit
                let (slope, intercept) = linreg(&xs, &ys);
                let crossing = (target - intercept) / slope;
                println!(
                    "\n  +{:.1}C threshold crossing (linear): ~{:.0}",
                    target, crossing
                );
            }
        }

        // Current trend
        let latest_year = post2000.last().unwrap().0;
        let rate_now = 2.0 * a * latest_year + b;
        println!(
            "\n  Current warming rate (quadratic slope at {}): {:>+.4} C/yr ({:>+.3} C/decade)",
            latest_year as i32,
            rate_now,
            rate_now * 10.0
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // WARMING RATE BY ERA
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  WARMING RATE BY ERA");
    println!("{}", "=".repeat(70));

    let eras: Vec<(&str, i32, i32)> = vec![
        ("Pre-industrial", 1850, 1900),
        ("Early warming", 1900, 1940),
        ("Mid-century", 1940, 1970),
        ("Late 20th C", 1970, 2000),
        ("21st Century", 2000, 2026),
    ];

    println!(
        "\n  {:>20} {:>10} {:>12} {:>12}",
        "Era", "Period", "Avg Anomaly", "Rate C/dec"
    );
    println!("  {:-<20} {:-<10} {:-<12} {:-<12}", "", "", "", "");

    for &(name, y0, y1) in &eras {
        let vals: Vec<(f64, f64)> = years
            .iter()
            .filter(|y| y.0 >= y0 && y.0 < y1)
            .map(|y| (y.0 as f64, y.1))
            .collect();
        if vals.len() < 2 {
            continue;
        }
        let xs: Vec<f64> = vals.iter().map(|v| v.0).collect();
        let ys: Vec<f64> = vals.iter().map(|v| v.1).collect();
        let (slope, _) = linreg(&xs, &ys);
        let avg = ys.iter().sum::<f64>() / ys.len() as f64;
        println!(
            "  {:>20} {:>4}-{:<4} {:>+12.3} {:>+12.3}",
            name,
            y0,
            y1,
            avg,
            slope * 10.0
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // TOP 10 WARMEST / COLDEST YEARS
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  EXTREME YEARS");
    println!("{}", "=".repeat(70));

    let mut sorted: Vec<(i32, f64)> = years.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Top 10 warmest years:");
    println!(
        "  {:>4} {:>6} {:>10} {:>14}",
        "Rank", "Year", "Anomaly", "Meta-regime"
    );
    println!("  {:-<4} {:-<6} {:-<10} {:-<14}", "", "", "", "");
    for (rank, &(y, a)) in sorted.iter().take(10).enumerate() {
        let idx = year_labels.iter().position(|&yy| yy == y).unwrap_or(0);
        let regime_count = (0..n_scales)
            .filter(|&si| meta_scale_regimes[si][idx])
            .count();
        let regime_label = format!("{}/4 scales", regime_count);
        println!(
            "  {:>4} {:>6} {:>+10.3}C {:>14}",
            rank + 1,
            y,
            a,
            regime_label
        );
    }

    println!("\n  Top 10 coldest years:");
    println!("  {:>4} {:>6} {:>10}", "Rank", "Year", "Anomaly");
    println!("  {:-<4} {:-<6} {:-<10}", "", "", "");
    for (rank, &(y, a)) in sorted.iter().rev().take(10).enumerate() {
        println!("  {:>4} {:>6} {:>+10.3}C", rank + 1, y, a);
    }

    // ══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ══════════════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(70));
    println!("  SUMMARY");
    println!("{}", "=".repeat(70));

    let total_transitions: usize = scale_transitions.iter().map(|t| t.len()).sum();
    let meta_total: usize = meta_transitions.iter().map(|t| t.len()).sum();
    println!(
        "\n  Data span:              {} years ({}-{})",
        n,
        year_labels.first().unwrap(),
        year_labels.last().unwrap()
    );
    println!("  Scales analyzed:        {}", n_scales);
    println!(
        "  Per-scale transitions:  {} total across all scales",
        total_transitions
    );
    println!(
        "  Meta-graph transitions: {} (cross-scale coherent)",
        meta_total
    );
    println!("  High-confidence (3+):   {}", high_conf.len());
    println!(
        "  Total warming:          {:>+.3}C (from {:>+.3}C to {:>+.3}C)",
        anomalies.last().unwrap() - anomalies.first().unwrap(),
        anomalies.first().unwrap(),
        anomalies.last().unwrap()
    );

    let _ = &all_derivs; // suppress unused warning
    let _ = &all_derivs[0].d3; // ensure jerk is used

    println!("\n=========================================================================");
    println!("  Analysis complete. Tipping points detected via Edmonds-Karp min-cut");
    println!(
        "  across {} scales with cross-scale meta-graph coherence.",
        n_scales
    );
    println!("=========================================================================");
}
