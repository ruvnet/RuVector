//! Report generation: text summary and SVG circuit comparison.

use crate::analysis::CircuitResult;

/// Print a text summary of all circuit results.
pub fn print_summary(results: &[CircuitResult]) {
    println!("\n{:-<70}", "");
    println!(
        "{:<25} {:>6} {:>10} {:>10} {:>10}",
        "Circuit", "Qubits", "Phi", "Emergence", "SVD Rank"
    );
    println!("{:-<70}", "");

    for r in results {
        println!(
            "{:<25} {:>6} {:>10.6} {:>10.4} {:>5}/{:<4}",
            r.name,
            r.n_qubits,
            r.full_phi,
            r.emergence.causal_emergence,
            r.svd_emergence.effective_rank,
            r.tpm_size
        );
    }
    println!("{:-<70}", "");

    for r in results {
        println!("\n=== {} ===", r.name);
        println!("  Description:       {}", r.description);
        println!("  Qubits:            {}", r.n_qubits);
        println!("  TPM size:          {}x{}", r.tpm_size, r.tpm_size);
        println!("  Phi:               {:.6}  ({})", r.full_phi, r.algorithm);
        println!("  EI (micro):        {:.4} bits", r.emergence.ei_micro);
        println!("  EI (macro):        {:.4} bits", r.emergence.ei_macro);
        println!("  Causal emergence:  {:.4}", r.emergence.causal_emergence);
        println!("  Determinism:       {:.4}", r.emergence.determinism);
        println!("  Degeneracy:        {:.4}", r.emergence.degeneracy);
        println!(
            "  Effective rank:    {}/{}",
            r.svd_emergence.effective_rank, r.tpm_size
        );
        println!(
            "  Spectral entropy:  {:.4}",
            r.svd_emergence.spectral_entropy
        );
        println!(
            "  Emergence index:   {:.4}",
            r.svd_emergence.emergence_index
        );
        println!("  Reversibility:     {:.4}", r.svd_emergence.reversibility);
    }
}

/// Generate a self-contained SVG report with bar charts and circuit diagrams.
pub fn generate_svg(results: &[CircuitResult]) -> String {
    let width = 1200;
    let total_height = 900;

    let mut svg = String::with_capacity(20_000);
    svg.push_str(&format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" font-family="monospace" font-size="12">
<style>
  .title {{ font-size: 20px; font-weight: bold; fill: #333; }}
  .subtitle {{ font-size: 14px; fill: #666; font-weight: bold; }}
  .label {{ font-size: 10px; fill: #333; }}
  .stat {{ font-size: 11px; fill: #444; }}
  .bar-phi {{ fill: #3498db; }}
  .bar-emg {{ fill: #e67e22; }}
  .bar-svd {{ fill: #2ecc71; }}
</style>
<rect width="{}" height="{}" fill="white"/>
<text x="600" y="40" text-anchor="middle" class="title">Quantum Circuit Consciousness Analysis Report</text>
<text x="600" y="65" text-anchor="middle" class="stat">IIT Phi applied to quantum circuit measurement statistics</text>
"#,
        width, total_height, width, total_height
    ));

    // Panel 1: Phi comparison bar chart (top)
    svg.push_str(&render_phi_bars(results, 50, 100, 500, 300));

    // Panel 2: Emergence comparison (top right)
    svg.push_str(&render_emergence_bars(results, 600, 100, 550, 300));

    // Panel 3: Circuit descriptions and stats table (bottom)
    svg.push_str(&render_stats_table(results, 50, 450, 1100, 400));

    svg.push_str("</svg>\n");
    svg
}

/// Render Phi comparison bar chart.
fn render_phi_bars(results: &[CircuitResult], x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\" rx=\"5\"/>\n",
        w, h
    ));
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-8\" text-anchor=\"middle\" class=\"subtitle\">Phi Comparison</text>\n",
        w / 2
    ));

    let max_phi = results
        .iter()
        .map(|r| r.full_phi)
        .fold(0.0f64, f64::max)
        .max(1e-10);

    let bar_h = ((h - 60) as f64 / results.len() as f64).min(40.0);
    let margin = 10;

    for (i, r) in results.iter().enumerate() {
        let ry = margin + (i as f64 * (bar_h + 8.0)) as i32 + 20;
        let bw = ((r.full_phi / max_phi) * (w - 200) as f64) as i32;

        // Label
        s.push_str(&format!(
            "<text x=\"10\" y=\"{}\" class=\"label\">{}</text>\n",
            ry + bar_h as i32 / 2 + 4,
            r.name
        ));

        // Bar
        s.push_str(&format!(
            "<rect x=\"150\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"bar-phi\" rx=\"3\"/>\n",
            ry,
            bw.max(1),
            bar_h as i32 - 4
        ));

        // Value
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" class=\"label\">{:.6}</text>\n",
            155 + bw,
            ry + bar_h as i32 / 2 + 4,
            r.full_phi
        ));
    }

    s.push_str("</g>\n");
    s
}

/// Render emergence comparison bar chart.
fn render_emergence_bars(results: &[CircuitResult], x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\" rx=\"5\"/>\n",
        w, h
    ));
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-8\" text-anchor=\"middle\" class=\"subtitle\">Emergence Comparison</text>\n",
        w / 2
    ));

    let max_ei = results
        .iter()
        .map(|r| r.emergence.ei_micro)
        .fold(0.0f64, f64::max)
        .max(1e-10);

    let bar_h = ((h - 60) as f64 / results.len() as f64).min(40.0);
    let margin = 10;
    let half_w = (w - 200) / 2;

    for (i, r) in results.iter().enumerate() {
        let ry = margin + (i as f64 * (bar_h + 8.0)) as i32 + 20;

        // Label
        s.push_str(&format!(
            "<text x=\"10\" y=\"{}\" class=\"label\">{}</text>\n",
            ry + bar_h as i32 / 2 + 4,
            r.name
        ));

        // EI bar (blue)
        let ei_w = ((r.emergence.ei_micro / max_ei) * half_w as f64) as i32;
        s.push_str(&format!(
            "<rect x=\"150\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"bar-phi\" rx=\"2\" opacity=\"0.7\"/>\n",
            ry,
            ei_w.max(1),
            (bar_h as i32 - 4) / 2
        ));

        // Emergence bar (orange)
        let emg_w = ((r.emergence.causal_emergence.abs() / max_ei) * half_w as f64) as i32;
        s.push_str(&format!(
            "<rect x=\"150\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"bar-emg\" rx=\"2\" opacity=\"0.7\"/>\n",
            ry + (bar_h as i32 - 4) / 2,
            emg_w.max(1),
            (bar_h as i32 - 4) / 2
        ));

        // Values
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" class=\"label\">EI={:.3}</text>\n",
            155 + ei_w,
            ry + (bar_h as i32) / 4 + 3,
            r.emergence.ei_micro
        ));
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" class=\"label\">CE={:.3}</text>\n",
            155 + emg_w,
            ry + 3 * (bar_h as i32) / 4 + 3,
            r.emergence.causal_emergence
        ));
    }

    // Legend
    let ly = h - 25;
    s.push_str(&format!(
        "<rect x=\"150\" y=\"{}\" width=\"12\" height=\"12\" class=\"bar-phi\" opacity=\"0.7\"/>\n\
         <text x=\"167\" y=\"{}\" class=\"label\">EI (micro)</text>\n\
         <rect x=\"260\" y=\"{}\" width=\"12\" height=\"12\" class=\"bar-emg\" opacity=\"0.7\"/>\n\
         <text x=\"277\" y=\"{}\" class=\"label\">Causal Emergence</text>\n",
        ly,
        ly + 10,
        ly,
        ly + 10
    ));

    s.push_str("</g>\n");
    s
}

/// Render stats table at the bottom.
fn render_stats_table(results: &[CircuitResult], x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\" rx=\"5\"/>\n",
        w, h
    ));
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-8\" text-anchor=\"middle\" class=\"subtitle\">Detailed Results</text>\n",
        w / 2
    ));

    // Header
    let cols = [15, 180, 260, 380, 500, 620, 740, 870, 990];
    let headers = [
        "Circuit",
        "Qubits",
        "Phi",
        "Algorithm",
        "EI_micro",
        "Emergence",
        "SVD Rank",
        "Emg Index",
        "Reversibility",
    ];
    for (col, hdr) in cols.iter().zip(headers.iter()) {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"25\" class=\"stat\" font-weight=\"bold\">{}</text>\n",
            col, hdr
        ));
    }
    s.push_str(&format!(
        "<line x1=\"10\" y1=\"32\" x2=\"{}\" y2=\"32\" stroke=\"#ccc\"/>\n",
        w - 10
    ));

    for (i, r) in results.iter().enumerate() {
        let ry = 50 + i as i32 * 22;
        let vals = [
            r.name.clone(),
            format!("{}", r.n_qubits),
            format!("{:.6}", r.full_phi),
            r.algorithm.clone(),
            format!("{:.4}", r.emergence.ei_micro),
            format!("{:.4}", r.emergence.causal_emergence),
            format!("{}/{}", r.svd_emergence.effective_rank, r.tpm_size),
            format!("{:.4}", r.svd_emergence.emergence_index),
            format!("{:.4}", r.svd_emergence.reversibility),
        ];
        for (col, val) in cols.iter().zip(vals.iter()) {
            s.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" class=\"stat\">{}</text>\n",
                col, ry, val
            ));
        }
    }

    // Description section
    let desc_y = 50 + results.len() as i32 * 22 + 30;
    s.push_str(&format!(
        "<text x=\"15\" y=\"{}\" class=\"subtitle\">Circuit Descriptions</text>\n",
        desc_y
    ));
    for (i, r) in results.iter().enumerate() {
        s.push_str(&format!(
            "<text x=\"15\" y=\"{}\" class=\"stat\">{}: {}</text>\n",
            desc_y + 20 + i as i32 * 18,
            r.name,
            r.description
        ));
    }

    s.push_str("</g>\n");
    s
}
