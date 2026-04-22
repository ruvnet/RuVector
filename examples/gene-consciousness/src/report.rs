//! Report generation: text summary and SVG visualization for gene networks.

use crate::analysis::AnalysisResults;
use crate::data::{self, GeneNetwork};

/// Print a text summary of the analysis results.
pub fn print_summary(results: &AnalysisResults) {
    println!("\n--- IIT Phi: Normal vs Cancer ---");
    println!(
        "Normal full Phi:     {:.6}  ({})",
        results.normal_full_phi.phi, results.normal_full_phi.algorithm
    );
    println!(
        "Cancer full Phi:     {:.6}  ({})",
        results.cancer_full_phi.phi, results.cancer_full_phi.algorithm
    );

    println!("\n--- Module-Level Phi (Normal) ---");
    for (name, phi) in &results.normal_module_phis {
        println!("{:20} Phi = {:.6}", name, phi.phi);
    }

    println!("\n--- Module-Level Phi (Cancer) ---");
    for (name, phi) in &results.cancer_module_phis {
        println!("{:20} Phi = {:.6}", name, phi.phi);
    }

    println!("\n--- Causal Emergence (Normal) ---");
    println!(
        "EI (micro):          {:.4} bits",
        results.normal_emergence.ei_micro
    );
    println!(
        "Causal emergence:    {:.4}",
        results.normal_emergence.causal_emergence
    );
    println!(
        "Determinism:         {:.4}",
        results.normal_emergence.determinism
    );
    println!(
        "Degeneracy:          {:.4}",
        results.normal_emergence.degeneracy
    );

    println!("\n--- SVD Emergence (Normal) ---");
    println!(
        "Effective rank:      {}/16",
        results.normal_svd_emergence.effective_rank
    );
    println!(
        "Spectral entropy:    {:.4}",
        results.normal_svd_emergence.spectral_entropy
    );
    println!(
        "Emergence index:     {:.4}",
        results.normal_svd_emergence.emergence_index
    );

    println!("\n--- Null Hypothesis Testing ---");
    let null_mean = if results.null_phis.is_empty() {
        0.0
    } else {
        results.null_phis.iter().sum::<f64>() / results.null_phis.len() as f64
    };
    println!("Phi (observed):      {:.6}", results.normal_full_phi.phi);
    println!(
        "Phi (null mean):     {:.6}  ({} samples)",
        null_mean,
        results.null_phis.len()
    );
    println!("z-score:             {:.2}", results.z_score);
    println!("p-value:             {:.4}", results.p_value);

    println!("\n--- Key Findings ---");
    println!(
        "Modules > full network:  {}",
        if results.modules_more_integrated {
            "YES"
        } else {
            "NO"
        }
    );
    println!(
        "Cancer > normal Phi:     {}",
        if results.cancer_higher_cross_phi {
            "YES"
        } else {
            "NO"
        }
    );
}

/// Generate a self-contained SVG report with network graph visualization.
pub fn generate_svg(results: &AnalysisResults, net: &GeneNetwork) -> String {
    let mut svg = String::with_capacity(20_000);

    svg.push_str(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1600" font-family="monospace" font-size="12">
<style>
  .title { font-size: 20px; font-weight: bold; fill: #333; }
  .subtitle { font-size: 14px; fill: #666; }
  .axis-label { font-size: 11px; fill: #444; }
  .bar { fill: #4a90d9; }
  .bar-cancer { fill: #e74c3c; }
  .bar-null { fill: #ccc; }
  .node-cc { fill: #3498db; }
  .node-ap { fill: #e74c3c; }
  .node-gs { fill: #2ecc71; }
  .node-hk { fill: #95a5a6; }
  .edge { stroke: #bbb; stroke-width: 0.5; fill: none; }
  .edge-strong { stroke: #555; stroke-width: 1.5; fill: none; }
</style>
<rect width="1200" height="1600" fill="white"/>
<text x="600" y="40" text-anchor="middle" class="title">Gene Regulatory Network Consciousness Report</text>
<text x="600" y="65" text-anchor="middle" class="subtitle">IIT 4.0 Phi Analysis of Regulatory Modules</text>
"#,
    );

    // Panel 1: Network graph (y=100, h=400)
    svg.push_str(&render_network_graph(net, 50, 100, 500, 400));

    // Panel 2: Module Phi comparison (y=100, x=600, h=400)
    svg.push_str(&render_phi_comparison(results, 620, 100, 530, 400));

    // Panel 3: Null distribution (y=550, h=280)
    svg.push_str(&render_null_distribution(
        &results.null_phis,
        results.normal_full_phi.phi,
        50,
        560,
        1100,
        280,
    ));

    // Panel 4: Summary stats (y=900)
    svg.push_str(&render_summary_stats(results, 50, 900));

    svg.push_str("</svg>\n");
    svg
}

/// Render the gene regulatory network as a graph with nodes colored by module.
fn render_network_graph(net: &GeneNetwork, x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Gene Regulatory Network (Normal)</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    // Arrange genes in a circular layout, grouped by module
    let n = net.n_genes;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let radius = (w.min(h) as f64 / 2.0) - 40.0;

    let mut positions = vec![(0.0f64, 0.0f64); n];
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        positions[i] = (cx + radius * angle.cos(), cy + radius * angle.sin());
    }

    // Draw edges (only significant ones)
    for i in 0..n {
        for j in 0..n {
            let w_val = net.adjacency[i * n + j];
            if w_val.abs() > 0.05 && i != j {
                let (x1, y1) = positions[i];
                let (x2, y2) = positions[j];
                let class = if w_val.abs() > 0.2 {
                    "edge-strong"
                } else {
                    "edge"
                };
                let opacity = (w_val.abs() * 3.0).min(1.0);
                s.push_str(&format!(
                    "<line x1=\"{:.0}\" y1=\"{:.0}\" x2=\"{:.0}\" y2=\"{:.0}\" class=\"{}\" opacity=\"{:.2}\"/>\n",
                    x1, y1, x2, y2, class, opacity
                ));
            }
        }
    }

    // Draw nodes
    let node_classes = ["node-cc", "node-ap", "node-gs", "node-hk"];
    for i in 0..n {
        let (px, py) = positions[i];
        let class = node_classes[net.module_ids[i]];
        s.push_str(&format!(
            "<circle cx=\"{:.0}\" cy=\"{:.0}\" r=\"12\" class=\"{}\" stroke=\"#333\" stroke-width=\"1\"/>\n",
            px, py, class
        ));
        s.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"8\" fill=\"white\">{}</text>\n",
            px, py, &net.gene_labels[i]
        ));
    }

    // Legend
    let legend_y = h - 60;
    let modules = data::all_modules();
    for (idx, (name, _)) in modules.iter().enumerate() {
        let lx = 10 + idx as i32 * 120;
        let class = node_classes[idx];
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

/// Render module Phi comparison bar chart (normal vs cancer).
fn render_phi_comparison(results: &AnalysisResults, x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Module Phi: Normal vs Cancer</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    // Collect all phi values to determine scale
    let mut all_phis: Vec<f64> = Vec::new();
    all_phis.push(results.normal_full_phi.phi);
    all_phis.push(results.cancer_full_phi.phi);
    for (_, p) in &results.normal_module_phis {
        all_phis.push(p.phi);
    }
    for (_, p) in &results.cancer_module_phis {
        all_phis.push(p.phi);
    }
    let max_phi = all_phis.iter().cloned().fold(0.0f64, f64::max).max(1e-10);

    // Draw grouped bars: normal (blue) and cancer (red) for each module + full
    let n_groups = results.normal_module_phis.len() + 1; // +1 for "Full"
    let group_w = (w - 40) as f64 / n_groups as f64;
    let bar_w = group_w * 0.35;
    let chart_h = (h - 60) as f64;

    for (idx, (name, normal_phi)) in results.normal_module_phis.iter().enumerate() {
        let gx = 20.0 + idx as f64 * group_w;

        // Normal bar
        let bh = (normal_phi.phi / max_phi * chart_h) as i32;
        s.push_str(&format!(
            "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar\" rx=\"2\"/>\n",
            gx,
            h - 30 - bh,
            bar_w,
            bh
        ));

        // Cancer bar
        if let Some((_, cancer_phi)) = results.cancer_module_phis.iter().find(|(n, _)| n == name) {
            let cbh = (cancer_phi.phi / max_phi * chart_h) as i32;
            s.push_str(&format!(
                "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar-cancer\" rx=\"2\"/>\n",
                gx + bar_w + 2.0, h - 30 - cbh, bar_w, cbh
            ));
        }

        // Label
        s.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\" font-size=\"9\">{}</text>\n",
            gx + bar_w, h - 15, name.split_whitespace().next().unwrap_or(name)
        ));
    }

    // Full network group
    let gx = 20.0 + results.normal_module_phis.len() as f64 * group_w;
    let bh = (results.normal_full_phi.phi / max_phi * chart_h) as i32;
    s.push_str(&format!(
        "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar\" rx=\"2\"/>\n",
        gx,
        h - 30 - bh,
        bar_w,
        bh
    ));
    let cbh = (results.cancer_full_phi.phi / max_phi * chart_h) as i32;
    s.push_str(&format!(
        "<rect x=\"{:.0}\" y=\"{}\" width=\"{:.0}\" height=\"{}\" class=\"bar-cancer\" rx=\"2\"/>\n",
        gx + bar_w + 2.0, h - 30 - cbh, bar_w, cbh
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
        "<text x=\"{}\" y=\"20\" class=\"axis-label\">Normal</text>\n",
        w - 135
    ));
    s.push_str(&format!(
        "<rect x=\"{}\" y=\"28\" width=\"12\" height=\"12\" class=\"bar-cancer\"/>\n",
        w - 150
    ));
    s.push_str(&format!(
        "<text x=\"{}\" y=\"38\" class=\"axis-label\">Cancer</text>\n",
        w - 135
    ));

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
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Null Distribution (Shuffled Networks) vs Observed Phi</text>\n",
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

    // Mark observed value
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
            "Normal Full Phi:     {:.6}  (n=16)",
            results.normal_full_phi.phi
        ),
        format!(
            "Cancer Full Phi:     {:.6}  (n=16)",
            results.cancer_full_phi.phi
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
            results.normal_emergence.ei_micro
        ),
        format!(
            "Causal emergence:    {:.4}",
            results.normal_emergence.causal_emergence
        ),
        format!(
            "SVD Eff. Rank:       {}/16",
            results.normal_svd_emergence.effective_rank
        ),
        format!(
            "Emergence Index:     {:.4}",
            results.normal_svd_emergence.emergence_index
        ),
        format!(
            "Modules > Full:      {}",
            if results.modules_more_integrated {
                "YES"
            } else {
                "NO"
            }
        ),
        format!(
            "Cancer > Normal:     {}",
            if results.cancer_higher_cross_phi {
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
