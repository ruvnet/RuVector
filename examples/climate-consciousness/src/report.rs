//! Report generation: text summary and SVG visualization for climate modes.

use crate::analysis::AnalysisResults;
use crate::data::{self, ClimateCorrelations};

/// Print a text summary of the analysis results.
pub fn print_summary(results: &AnalysisResults) {
    println!("\n--- IIT Phi: Neutral vs El Nino ---");
    println!(
        "Neutral full Phi:    {:.6}  ({})",
        results.neutral_full_phi.phi, results.neutral_full_phi.algorithm
    );
    println!(
        "El Nino full Phi:    {:.6}  ({})",
        results.elnino_full_phi.phi, results.elnino_full_phi.algorithm
    );

    println!("\n--- Regional Phi (Neutral) ---");
    for (name, phi) in &results.neutral_regional_phis {
        println!("{:20} Phi = {:.6}", name, phi.phi);
    }

    println!("\n--- Regional Phi (El Nino) ---");
    for (name, phi) in &results.elnino_regional_phis {
        println!("{:20} Phi = {:.6}", name, phi.phi);
    }

    println!("\n--- Seasonal Phi Cycle ---");
    let max_monthly = results
        .monthly_phis
        .iter()
        .map(|(_, p)| *p)
        .fold(0.0f64, f64::max)
        .max(1e-10);
    for (month, phi) in &results.monthly_phis {
        let bar_len = (phi / max_monthly * 30.0) as usize;
        println!("  {:3}  Phi={:.4}  {}", month, phi, "|".repeat(bar_len));
    }

    println!("\n--- Causal Emergence (Neutral) ---");
    println!(
        "EI (micro):          {:.4} bits",
        results.neutral_emergence.ei_micro
    );
    println!(
        "Causal emergence:    {:.4}",
        results.neutral_emergence.causal_emergence
    );
    println!(
        "Determinism:         {:.4}",
        results.neutral_emergence.determinism
    );
    println!(
        "Degeneracy:          {:.4}",
        results.neutral_emergence.degeneracy
    );

    println!("\n--- SVD Emergence (Neutral) ---");
    println!(
        "Effective rank:      {}/7",
        results.neutral_svd_emergence.effective_rank
    );
    println!(
        "Spectral entropy:    {:.4}",
        results.neutral_svd_emergence.spectral_entropy
    );
    println!(
        "Emergence index:     {:.4}",
        results.neutral_svd_emergence.emergence_index
    );

    println!("\n--- Null Hypothesis Testing ---");
    let null_mean = if results.null_phis.is_empty() {
        0.0
    } else {
        results.null_phis.iter().sum::<f64>() / results.null_phis.len() as f64
    };
    println!("Phi (observed):      {:.6}", results.neutral_full_phi.phi);
    println!(
        "Phi (null mean):     {:.6}  ({} samples)",
        null_mean,
        results.null_phis.len()
    );
    println!("z-score:             {:.2}", results.z_score);
    println!("p-value:             {:.4}", results.p_value);

    println!("\n--- Key Findings ---");
    println!(
        "El Nino > Neutral:       {}",
        if results.elnino_increases_phi {
            "YES"
        } else {
            "NO"
        }
    );
    println!(
        "Pacific most integrated: {}",
        if results.pacific_most_integrated {
            "YES"
        } else {
            "NO"
        }
    );
}

/// Generate a self-contained SVG report with climate mode connection diagram.
pub fn generate_svg(results: &AnalysisResults, data: &ClimateCorrelations) -> String {
    let mut svg = String::with_capacity(20_000);

    svg.push_str(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1600" font-family="monospace" font-size="12">
<style>
  .title { font-size: 20px; font-weight: bold; fill: #333; }
  .subtitle { font-size: 14px; fill: #666; }
  .axis-label { font-size: 11px; fill: #444; }
  .bar { fill: #4a90d9; }
  .bar-elnino { fill: #e74c3c; }
  .bar-null { fill: #ccc; }
  .bar-month { fill: #2ecc71; }
  .node-pacific { fill: #3498db; }
  .node-atlantic { fill: #e67e22; }
  .node-polar { fill: #9b59b6; }
  .edge { stroke: #bbb; stroke-width: 0.5; fill: none; }
  .edge-strong { stroke: #2c3e50; stroke-width: 2; fill: none; }
</style>
<rect width="1200" height="1600" fill="white"/>
<text x="600" y="40" text-anchor="middle" class="title">Climate Teleconnection Consciousness Report</text>
<text x="600" y="65" text-anchor="middle" class="subtitle">IIT 4.0 Phi Analysis of Climate Mode Interactions</text>
"#,
    );

    // Panel 1: Climate mode connection diagram (y=100, h=350)
    svg.push_str(&render_connection_diagram(data, 50, 100, 500, 350));

    // Panel 2: Regional Phi comparison (y=100, x=600, h=350)
    svg.push_str(&render_phi_comparison(results, 620, 100, 530, 350));

    // Panel 3: Seasonal Phi cycle (y=500, h=250)
    svg.push_str(&render_seasonal_cycle(
        &results.monthly_phis,
        50,
        510,
        1100,
        250,
    ));

    // Panel 4: Null distribution (y=810, h=250)
    svg.push_str(&render_null_distribution(
        &results.null_phis,
        results.neutral_full_phi.phi,
        50,
        810,
        1100,
        250,
    ));

    // Panel 5: Summary stats (y=1110)
    svg.push_str(&render_summary_stats(results, 50, 1120));

    svg.push_str("</svg>\n");
    svg
}

/// Render the climate mode connection diagram.
fn render_connection_diagram(data: &ClimateCorrelations, x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Climate Mode Teleconnections (Neutral)</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    let n = data.n_indices;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let radius = (w.min(h) as f64 / 2.0) - 45.0;

    // Circular layout
    let mut positions = vec![(0.0f64, 0.0f64); n];
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        positions[i] = (cx + radius * angle.cos(), cy + radius * angle.sin());
    }

    // Draw edges (correlation strengths)
    for i in 0..n {
        for j in (i + 1)..n {
            let c = data.correlations[i * n + j];
            if c.abs() > 0.03 {
                let (x1, y1) = positions[i];
                let (x2, y2) = positions[j];
                let class = if c.abs() > 0.3 { "edge-strong" } else { "edge" };
                let width = (c.abs() * 5.0).max(0.5).min(3.0);
                s.push_str(&format!(
                    "<line x1=\"{:.0}\" y1=\"{:.0}\" x2=\"{:.0}\" y2=\"{:.0}\" class=\"{}\" stroke-width=\"{:.1}\"/>\n",
                    x1, y1, x2, y2, class, width
                ));
            }
        }
    }

    // Region colors for each index
    let node_classes = [
        "node-pacific",  // ENSO
        "node-atlantic", // NAO
        "node-pacific",  // PDO
        "node-atlantic", // AMO
        "node-pacific",  // IOD
        "node-polar",    // SAM
        "node-polar",    // QBO
    ];

    // Draw nodes
    for i in 0..n {
        let (px, py) = positions[i];
        s.push_str(&format!(
            "<circle cx=\"{:.0}\" cy=\"{:.0}\" r=\"18\" class=\"{}\" stroke=\"#333\" stroke-width=\"1\"/>\n",
            px, py, node_classes[i]
        ));
        s.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"9\" fill=\"white\" font-weight=\"bold\">{}</text>\n",
            px, py, data::INDEX_NAMES[i]
        ));
    }

    // Legend
    let legend_y = h - 30;
    let region_info = [
        ("Pacific", "node-pacific"),
        ("Atlantic", "node-atlantic"),
        ("Polar", "node-polar"),
    ];
    for (idx, (name, class)) in region_info.iter().enumerate() {
        let lx = 10 + idx as i32 * 140;
        s.push_str(&format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"6\" class=\"{}\"/>\n",
            lx, legend_y, class
        ));
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" class=\"axis-label\" dominant-baseline=\"middle\">{}</text>\n",
            lx + 10,
            legend_y,
            name
        ));
    }

    s.push_str("</g>\n");
    s
}

/// Render regional Phi comparison: neutral vs El Nino.
fn render_phi_comparison(results: &AnalysisResults, x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Regional Phi: Neutral vs El Nino</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    let mut all_phis: Vec<f64> = Vec::new();
    all_phis.push(results.neutral_full_phi.phi);
    all_phis.push(results.elnino_full_phi.phi);
    for (_, p) in &results.neutral_regional_phis {
        all_phis.push(p.phi);
    }
    for (_, p) in &results.elnino_regional_phis {
        all_phis.push(p.phi);
    }
    let max_phi = all_phis.iter().cloned().fold(0.0f64, f64::max).max(1e-10);

    let n_groups = results.neutral_regional_phis.len() + 1;
    let group_w = (w - 40) as f64 / n_groups as f64;
    let bar_w = group_w * 0.35;
    let chart_h = (h - 60) as f64;

    for (idx, (name, neutral_phi)) in results.neutral_regional_phis.iter().enumerate() {
        let gx = 20.0 + idx as f64 * group_w;

        let bh = (neutral_phi.phi / max_phi * chart_h) as i32;
        s.push_str(&format!(
            "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar\" rx=\"2\"/>\n",
            gx,
            h - 30 - bh,
            bar_w,
            bh
        ));

        if let Some((_, elnino_phi)) = results.elnino_regional_phis.iter().find(|(n, _)| n == name)
        {
            let ebh = (elnino_phi.phi / max_phi * chart_h) as i32;
            s.push_str(&format!(
                "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar-elnino\" rx=\"2\"/>\n",
                gx + bar_w + 2.0, h - 30 - ebh, bar_w, ebh
            ));
        }

        s.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\" font-size=\"9\">{}</text>\n",
            gx + bar_w, h - 15, name
        ));
    }

    // Full system group
    let gx = 20.0 + results.neutral_regional_phis.len() as f64 * group_w;
    let bh = (results.neutral_full_phi.phi / max_phi * chart_h) as i32;
    s.push_str(&format!(
        "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar\" rx=\"2\"/>\n",
        gx,
        h - 30 - bh,
        bar_w,
        bh
    ));
    let ebh = (results.elnino_full_phi.phi / max_phi * chart_h) as i32;
    s.push_str(&format!(
        "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar-elnino\" rx=\"2\"/>\n",
        gx + bar_w + 2.0, h - 30 - ebh, bar_w, ebh
    ));
    s.push_str(&format!(
        "<text x=\"{:.0}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\" font-size=\"9\">Full</text>\n",
        gx + bar_w, h - 15
    ));

    // Legend
    s.push_str(&format!(
        "<rect x=\"{}\" y=\"10\" width=\"12\" height=\"12\" class=\"bar\"/>\n",
        w - 150
    ));
    s.push_str(&format!(
        "<text x=\"{}\" y=\"20\" class=\"axis-label\">Neutral</text>\n",
        w - 135
    ));
    s.push_str(&format!(
        "<rect x=\"{}\" y=\"28\" width=\"12\" height=\"12\" class=\"bar-elnino\"/>\n",
        w - 150
    ));
    s.push_str(&format!(
        "<text x=\"{}\" y=\"38\" class=\"axis-label\">El Nino</text>\n",
        w - 135
    ));

    s.push_str("</g>\n");
    s
}

/// Render the seasonal Phi cycle as a bar chart.
fn render_seasonal_cycle(monthly: &[(String, f64)], x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Seasonal Phi Cycle (12 months)</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    if monthly.is_empty() {
        s.push_str("</g>\n");
        return s;
    }

    let max_phi = monthly
        .iter()
        .map(|(_, p)| *p)
        .fold(0.0f64, f64::max)
        .max(1e-10);
    let bar_w = (w - 40) as f64 / monthly.len() as f64;
    let chart_h = (h - 50) as f64;

    for (i, (month, phi)) in monthly.iter().enumerate() {
        let bx = 20.0 + i as f64 * bar_w;
        let bh = (phi / max_phi * chart_h) as i32;
        s.push_str(&format!(
            "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar-month\" rx=\"2\"/>\n",
            bx, h - bh - 25, (bar_w - 3.0).max(1.0), bh
        ));
        s.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\" font-size=\"9\">{}</text>\n",
            bx + bar_w / 2.0, h - 10, month
        ));
    }

    s.push_str("</g>\n");
    s
}

/// Render the null distribution histogram.
fn render_null_distribution(
    null_phis: &[f64],
    observed: f64,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Null Distribution (Shuffled Correlations) vs Observed Phi</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    if null_phis.is_empty() {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">No null samples</text>\n",
            w / 2, h / 2
        ));
        s.push_str("</g>\n");
        return s;
    }

    let n_hist_bins = 25usize;
    let phi_min = null_phis
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .min(observed)
        * 0.9;
    let phi_max = null_phis
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(observed)
        * 1.1;
    let range = (phi_max - phi_min).max(1e-10);
    let bin_width = range / n_hist_bins as f64;

    let mut hist = vec![0u32; n_hist_bins];
    for &p in null_phis {
        let bin = ((p - phi_min) / bin_width).floor() as usize;
        if bin < n_hist_bins {
            hist[bin] += 1;
        }
    }
    let max_count = *hist.iter().max().unwrap_or(&1);

    let bar_w = w as f64 / n_hist_bins as f64;
    for (i, &count) in hist.iter().enumerate() {
        let bar_h = (count as f64 / max_count as f64 * (h - 40) as f64) as i32;
        s.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{}\" width=\"{:.1}\" height=\"{}\" class=\"bar-null\" rx=\"1\"/>\n",
            i as f64 * bar_w, h - bar_h - 20, bar_w - 1.0, bar_h
        ));
    }

    let obs_x = ((observed - phi_min) / range * w as f64) as i32;
    s.push_str(&format!(
        "<line x1=\"{}\" y1=\"0\" x2=\"{}\" y2=\"{}\" stroke=\"#e74c3c\" stroke-width=\"2\"/>\n",
        obs_x,
        obs_x,
        h - 20
    ));
    s.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#e74c3c\" font-size=\"10\">Observed</text>\n",
        obs_x, h - 5
    ));

    s.push_str("</g>\n");
    s
}

/// Render summary statistics text.
fn render_summary_stats(results: &AnalysisResults, x: i32, y: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str("<text x=\"0\" y=\"0\" class=\"subtitle\">Summary Statistics</text>\n");

    let null_mean = if results.null_phis.is_empty() {
        0.0
    } else {
        results.null_phis.iter().sum::<f64>() / results.null_phis.len() as f64
    };

    let lines = vec![
        format!(
            "Neutral Full Phi:    {:.6}  (n=7)",
            results.neutral_full_phi.phi
        ),
        format!(
            "El Nino Full Phi:    {:.6}  (n=7)",
            results.elnino_full_phi.phi
        ),
        format!(
            "Null Mean Phi:       {:.6}  ({} samples)",
            null_mean,
            results.null_phis.len()
        ),
        format!("z-score:             {:.3}", results.z_score),
        format!("p-value:             {:.4}", results.p_value),
        format!(
            "EI (micro):          {:.4} bits",
            results.neutral_emergence.ei_micro
        ),
        format!(
            "Causal emergence:    {:.4}",
            results.neutral_emergence.causal_emergence
        ),
        format!(
            "SVD Eff. Rank:       {}/7",
            results.neutral_svd_emergence.effective_rank
        ),
        format!(
            "Emergence Index:     {:.4}",
            results.neutral_svd_emergence.emergence_index
        ),
        format!(
            "El Nino > Neutral:   {}",
            if results.elnino_increases_phi {
                "YES"
            } else {
                "NO"
            }
        ),
        format!(
            "Pacific top region:  {}",
            if results.pacific_most_integrated {
                "YES"
            } else {
                "NO"
            }
        ),
    ];

    for (i, line) in lines.iter().enumerate() {
        s.push_str(&format!(
            "<text x=\"0\" y=\"{}\" class=\"axis-label\">{}</text>\n",
            20 + i * 18,
            line
        ));
    }

    s.push_str("</g>\n");
    s
}
