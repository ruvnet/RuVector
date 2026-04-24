//! Report generation: text summary and SVG visualization for GW analysis.

use crate::analysis::AnalysisResults;
use crate::data::{GWSpectrum, TransitionMatrix};

/// Print a text summary of the analysis results.
pub fn print_summary(results: &AnalysisResults) {
    println!("\n--- IIT Phi by Source Model ---");
    for (name, phi) in &results.model_phis {
        println!("  {:20} Phi = {:.6}  ({})", name, phi.phi, phi.algorithm);
    }

    println!("\n--- Causal Emergence by Source Model ---");
    for (name, em) in &results.model_emergence {
        println!(
            "  {:20} EI = {:.4}, det = {:.4}, deg = {:.4}",
            name, em.ei_micro, em.determinism, em.degeneracy
        );
    }

    println!("\n--- SVD Emergence by Source Model ---");
    for (name, svd) in &results.model_svd {
        println!(
            "  {:20} rank = {}, entropy = {:.4}, emergence = {:.4}",
            name, svd.effective_rank, svd.spectral_entropy, svd.emergence_index
        );
    }

    if !results.smbh_phi_spectrum.is_empty() {
        println!("\n--- SMBH Phi Spectrum (sliding window) ---");
        let max_phi = results
            .smbh_phi_spectrum
            .iter()
            .map(|x| x.2)
            .fold(0.0f64, f64::max);
        for (start, end, phi) in &results.smbh_phi_spectrum {
            let bar_len = if max_phi > 0.0 {
                (phi / max_phi * 30.0) as usize
            } else {
                0
            };
            println!(
                "  f{}..f{}  Phi={:.4}  {}",
                start + 1,
                end,
                phi,
                "|".repeat(bar_len)
            );
        }
    }

    println!("\n--- Null Hypothesis Testing ---");
    let smbh_phi = results
        .model_phis
        .iter()
        .find(|(n, _)| n == "smbh")
        .map(|(_, p)| p.phi)
        .unwrap_or(0.0);
    let best_exotic = results
        .model_phis
        .iter()
        .filter(|(n, _)| n != "smbh")
        .max_by(|(_, a), (_, b)| a.phi.partial_cmp(&b.phi).unwrap())
        .map(|(n, p)| (n.as_str(), p.phi))
        .unwrap_or(("none", 0.0));
    let null_mean = if results.null_phis.is_empty() {
        0.0
    } else {
        results.null_phis.iter().sum::<f64>() / results.null_phis.len() as f64
    };

    println!("  Phi (SMBH):          {:.6}", smbh_phi);
    println!(
        "  Phi (best exotic):   {:.6}  ({})",
        best_exotic.1, best_exotic.0
    );
    println!(
        "  Phi (null mean):     {:.6}  ({} samples)",
        null_mean,
        results.null_phis.len()
    );
    println!("  z-score:             {:.2}", results.z_score);
    println!("  p-value:             {:.4}", results.p_value);
}

/// Generate a self-contained SVG report with charts.
pub fn generate_svg(
    results: &AnalysisResults,
    tpms: &[(&str, TransitionMatrix)],
    spectra: &[(&str, GWSpectrum)],
) -> String {
    let mut svg = String::with_capacity(25_000);

    svg.push_str(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1800" font-family="monospace" font-size="12">
<style>
  .title { font-size: 20px; font-weight: bold; fill: #333; }
  .subtitle { font-size: 14px; fill: #666; }
  .axis-label { font-size: 11px; fill: #444; }
  .bar-smbh { fill: #4a90d9; }
  .bar-cs { fill: #e67e22; }
  .bar-pgw { fill: #27ae60; }
  .bar-pt { fill: #8e44ad; }
  .bar-null { fill: #ccc; }
  .bar-obs { fill: #e74c3c; }
  .grid { stroke: #eee; stroke-width: 0.5; }
</style>
<rect width="1200" height="1800" fill="white"/>
<text x="600" y="40" text-anchor="middle" class="title">GW Background Consciousness Analysis Report</text>
<text x="600" y="65" text-anchor="middle" class="subtitle">IIT Phi applied to pulsar timing array GW background (NANOGrav 15yr model)</text>
"#,
    );

    // Panel 1: GW strain spectra (y=100, h=300)
    svg.push_str(&render_strain_spectra(spectra, 50, 100, 1100, 280));

    // Panel 2: TPM heatmaps (y=420, h=280) -- show 2 models side by side
    if let Some((_, smbh_tpm)) = tpms.iter().find(|(n, _)| *n == "smbh") {
        svg.push_str(&render_tpm_heatmap(smbh_tpm, "SMBH", 50, 420, 480, 260));
    }
    if let Some((_, cs_tpm)) = tpms.iter().find(|(n, _)| *n == "cosmic_strings") {
        svg.push_str(&render_tpm_heatmap(
            cs_tpm,
            "Cosmic Strings",
            600,
            420,
            480,
            260,
        ));
    }

    // Panel 3: Phi comparison bar chart (y=740, h=280)
    svg.push_str(&render_phi_comparison(
        &results.model_phis,
        50,
        740,
        1100,
        260,
    ));

    // Panel 4: Null distribution (y=1060, h=280)
    let best_exotic_phi = results
        .model_phis
        .iter()
        .filter(|(n, _)| n != "smbh")
        .map(|(_, p)| p.phi)
        .fold(0.0f64, f64::max);
    svg.push_str(&render_null_distribution(
        &results.null_phis,
        best_exotic_phi,
        50,
        1060,
        1100,
        260,
    ));

    // Panel 5: Summary stats (y=1400)
    svg.push_str(&render_summary_stats(results, 50, 1400));

    svg.push_str("</svg>\n");
    svg
}

/// Render GW strain spectra for all models on log-log axes.
fn render_strain_spectra(spectra: &[(&str, GWSpectrum)], x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">\
         GW Characteristic Strain Spectra h_c(f) by Source Model</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f8f8f8\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    // Determine global min/max for log-log axes
    let all_f: Vec<f64> = spectra
        .iter()
        .flat_map(|(_, sp)| sp.frequencies.iter().copied())
        .collect();
    let all_h: Vec<f64> = spectra
        .iter()
        .flat_map(|(_, sp)| sp.h_c.iter().copied())
        .collect();

    let f_min = all_f.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_max = all_f.iter().cloned().fold(0.0f64, f64::max);
    let h_min = all_h.iter().cloned().fold(f64::INFINITY, f64::min);
    let h_max = all_h.iter().cloned().fold(0.0f64, f64::max);

    let log_f_min = f_min.ln();
    let log_f_range = (f_max.ln() - log_f_min).max(1e-10);
    let log_h_min = h_min.ln();
    let log_h_range = (h_max.ln() - log_h_min).max(1e-10);

    let colors = ["#4a90d9", "#e67e22", "#27ae60", "#8e44ad"];
    let margin = 20;

    for (idx, (name, sp)) in spectra.iter().enumerate() {
        let color = colors[idx % colors.len()];
        s.push_str(&format!(
            "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"2\" points=\"",
            color
        ));
        for (&freq, &strain) in sp.frequencies.iter().zip(sp.h_c.iter()) {
            let px =
                margin + ((freq.ln() - log_f_min) / log_f_range * (w - 2 * margin) as f64) as i32;
            let py = h
                - margin
                - ((strain.ln() - log_h_min) / log_h_range * (h - 2 * margin) as f64) as i32;
            s.push_str(&format!("{},{} ", px, py));
        }
        s.push_str("\"/>\n");

        // Draw data points
        for (&freq, &strain) in sp.frequencies.iter().zip(sp.h_c.iter()) {
            let px =
                margin + ((freq.ln() - log_f_min) / log_f_range * (w - 2 * margin) as f64) as i32;
            let py = h
                - margin
                - ((strain.ln() - log_h_min) / log_h_range * (h - 2 * margin) as f64) as i32;
            s.push_str(&format!(
                "<circle cx=\"{}\" cy=\"{}\" r=\"3\" fill=\"{}\"/>\n",
                px, py, color
            ));
        }

        // Legend
        let ly = 15 + idx as i32 * 16;
        s.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"12\" height=\"12\" fill=\"{}\"/>\n",
            w - 180,
            ly,
            color
        ));
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" class=\"axis-label\">{}</text>\n",
            w - 164,
            ly + 10,
            name
        ));
    }

    // Axis labels
    s.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">Frequency (Hz)</text>\n",
        w / 2,
        h - 2
    ));

    s.push_str("</g>\n");
    s
}

/// Render a TPM heatmap for a single model.
fn render_tpm_heatmap(
    tpm: &TransitionMatrix,
    label: &str,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">\
         TPM: {}</text>\n",
        w / 2,
        label
    ));

    let cell_w = w as f64 / tpm.size as f64;
    let cell_h = h as f64 / tpm.size as f64;
    let max_val = tpm.data.iter().cloned().fold(0.0f64, f64::max).max(1e-10);

    for i in 0..tpm.size {
        for j in 0..tpm.size {
            let val = tpm.data[i * tpm.size + j] / max_val;
            let r = (255.0 * (1.0 - val)) as u8;
            let g = (255.0 * (1.0 - val * 0.5)) as u8;
            let b = 255u8;
            s.push_str(&format!(
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
                 fill=\"rgb({},{},{})\"/>\n",
                j as f64 * cell_w,
                i as f64 * cell_h,
                cell_w + 0.5,
                cell_h + 0.5,
                r,
                g,
                b
            ));
        }
    }

    s.push_str("</g>\n");
    s
}

/// Render a bar chart comparing Phi values across source models.
fn render_phi_comparison(
    model_phis: &[(String, ruvector_consciousness::types::PhiResult)],
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">\
         IIT Phi Comparison by GW Source Model</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f8f8f8\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    if model_phis.is_empty() {
        s.push_str("</g>\n");
        return s;
    }

    let max_phi = model_phis
        .iter()
        .map(|(_, p)| p.phi)
        .fold(0.0f64, f64::max)
        .max(1e-10);

    let colors = ["#4a90d9", "#e67e22", "#27ae60", "#8e44ad"];
    let n = model_phis.len();
    let bar_w = ((w - 100) as f64 / n as f64 * 0.7) as i32;
    let gap = ((w - 100) as f64 / n as f64) as i32;

    for (i, (name, phi)) in model_phis.iter().enumerate() {
        let bar_h = (phi.phi / max_phi * (h - 60) as f64) as i32;
        let bx = 50 + i as i32 * gap + (gap - bar_w) / 2;
        let color = colors[i % colors.len()];

        s.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" rx=\"2\"/>\n",
            bx,
            h - bar_h - 30,
            bar_w,
            bar_h,
            color
        ));
        // Label
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\" \
             transform=\"rotate(-30,{},{})\">{}</text>\n",
            bx + bar_w / 2,
            h - 10,
            bx + bar_w / 2,
            h - 10,
            name
        ));
        // Value
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">{:.4}</text>\n",
            bx + bar_w / 2,
            h - bar_h - 35,
            phi.phi
        ));
    }

    s.push_str("</g>\n");
    s
}

/// Render the null distribution histogram with observed value marker.
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
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">\
         Null Distribution (SMBH) vs Best Exotic Phi</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f8f8f8\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    if null_phis.is_empty() {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">\
             No null samples</text>\n",
            w / 2,
            h / 2
        ));
        s.push_str("</g>\n");
        return s;
    }

    let n_hist_bins = 30usize;
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
            "<rect x=\"{:.1}\" y=\"{}\" width=\"{:.1}\" height=\"{}\" \
             class=\"bar-null\" rx=\"1\"/>\n",
            i as f64 * bar_w,
            h - bar_h - 20,
            bar_w - 1.0,
            bar_h
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
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#e74c3c\" font-size=\"10\">\
         Best Exotic</text>\n",
        obs_x,
        h - 5
    ));

    s.push_str("</g>\n");
    s
}

/// Render summary statistics text block.
fn render_summary_stats(results: &AnalysisResults, x: i32, y: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str("<text x=\"0\" y=\"0\" class=\"subtitle\">Summary Statistics</text>\n");

    let smbh_phi = results
        .model_phis
        .iter()
        .find(|(n, _)| n == "smbh")
        .map(|(_, p)| p.phi)
        .unwrap_or(0.0);
    let best_exotic = results
        .model_phis
        .iter()
        .filter(|(n, _)| n != "smbh")
        .max_by(|(_, a), (_, b)| a.phi.partial_cmp(&b.phi).unwrap());
    let null_mean = if results.null_phis.is_empty() {
        0.0
    } else {
        results.null_phis.iter().sum::<f64>() / results.null_phis.len() as f64
    };

    let mut lines = vec![format!("Phi (SMBH baseline):    {:.6}", smbh_phi)];
    if let Some((name, phi)) = best_exotic {
        lines.push(format!(
            "Phi (best exotic):      {:.6}  ({})",
            phi.phi, name
        ));
        lines.push(format!(
            "Phi ratio (exotic/SMBH):{:.2}x",
            if smbh_phi > 0.0 {
                phi.phi / smbh_phi
            } else {
                0.0
            }
        ));
    }
    lines.push(format!(
        "Null mean Phi:          {:.6}  ({} samples)",
        null_mean,
        results.null_phis.len()
    ));
    lines.push(format!("z-score:                {:.3}", results.z_score));
    lines.push(format!("p-value:                {:.4}", results.p_value));

    // Emergence comparison
    if let Some((_, smbh_em)) = results.model_emergence.iter().find(|(n, _)| n == "smbh") {
        lines.push(format!(
            "EI_micro (SMBH):        {:.4} bits",
            smbh_em.ei_micro
        ));
    }
    if let Some((name, best_em)) = results
        .model_emergence
        .iter()
        .filter(|(n, _)| n != "smbh")
        .max_by(|(_, a), (_, b)| a.ei_micro.partial_cmp(&b.ei_micro).unwrap())
    {
        lines.push(format!(
            "EI_micro (best exotic): {:.4} bits  ({})",
            best_em.ei_micro, name
        ));
    }

    lines.push(String::new());
    if results.p_value < 0.05 {
        lines.push("Verdict: SIGNIFICANT -- GWB shows excess integration".to_string());
        lines.push("  Exotic source model produces higher Phi than SMBH null".to_string());
    } else {
        lines.push("Verdict: CONSISTENT with independent SMBH mergers".to_string());
        lines.push("  No evidence for correlated cosmological source".to_string());
    }

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
