//! Report generation: text summary and SVG visualization.

use crate::analysis::AnalysisResults;
use crate::data::{PowerSpectrum, TransitionMatrix};

pub fn print_summary(results: &AnalysisResults, tpm: &TransitionMatrix) {
    println!("\n--- IIT Phi Analysis ---");
    println!(
        "Full system Phi:     {:.6}  ({})",
        results.full_phi.phi, results.full_phi.algorithm
    );

    for (name, phi) in &results.regional_phis {
        println!("{:20} Phi = {:.6}", name, phi.phi);
    }

    println!("\n--- Phi Spectrum (sliding window) ---");
    let max_phi = results
        .phi_spectrum
        .iter()
        .map(|x| x.2)
        .fold(0.0f64, f64::max);
    for (start, end, phi) in &results.phi_spectrum {
        let bar_len = if max_phi > 0.0 {
            (phi / max_phi * 30.0) as usize
        } else {
            0
        };
        let l_start = tpm
            .bin_centers
            .get(*start)
            .map(|x| *x as u32)
            .unwrap_or(0);
        let l_end = tpm
            .bin_centers
            .get(end.saturating_sub(1))
            .map(|x| *x as u32)
            .unwrap_or(0);
        println!(
            "  l={:>4}..{:<4}  Phi={:.4}  {}",
            l_start,
            l_end,
            phi,
            "|".repeat(bar_len)
        );
    }

    println!("\n--- Causal Emergence ---");
    println!(
        "EI (micro):          {:.4} bits",
        results.emergence.ei_micro
    );
    println!("Determinism:         {:.4}", results.emergence.determinism);
    println!("Degeneracy:          {:.4}", results.emergence.degeneracy);

    println!("\n--- SVD Emergence ---");
    println!(
        "Effective rank:      {}/{}",
        results.svd_emergence.effective_rank, tpm.size
    );
    println!(
        "Spectral entropy:    {:.4}",
        results.svd_emergence.spectral_entropy
    );
    println!(
        "Emergence index:     {:.4}",
        results.svd_emergence.emergence_index
    );
    println!(
        "Reversibility:       {:.4}",
        results.svd_emergence.reversibility
    );

    println!("\n--- Null Hypothesis Testing ---");
    let null_mean = if results.null_phis.is_empty() {
        0.0
    } else {
        results.null_phis.iter().sum::<f64>() / results.null_phis.len() as f64
    };
    let null_std = if results.null_phis.len() > 1 {
        (results
            .null_phis
            .iter()
            .map(|p| (p - null_mean).powi(2))
            .sum::<f64>()
            / (results.null_phis.len() as f64 - 1.0))
            .sqrt()
    } else {
        0.0
    };
    println!("Phi (observed):      {:.6}", results.full_phi.phi);
    println!("Phi (null mean):     {:.6} +/- {:.6}", null_mean, null_std);
    println!("z-score:             {:.2}", results.z_score);
    println!("p-value:             {:.4}", results.p_value);
}

/// Generate a self-contained SVG report with charts.
pub fn generate_svg(
    results: &AnalysisResults,
    tpm: &TransitionMatrix,
    ps: &PowerSpectrum,
) -> String {
    let mut svg = String::with_capacity(20_000);

    svg.push_str(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1800" font-family="monospace" font-size="12">
<style>
  .title { font-size: 20px; font-weight: bold; fill: #333; }
  .subtitle { font-size: 14px; fill: #666; }
  .axis-label { font-size: 11px; fill: #444; }
  .bar { fill: #4a90d9; }
  .bar-null { fill: #ccc; }
  .bar-obs { fill: #e74c3c; }
  .grid { stroke: #eee; stroke-width: 0.5; }
</style>
<rect width="1200" height="1800" fill="white"/>
<text x="600" y="40" text-anchor="middle" class="title">CMB Consciousness Analysis Report</text>
<text x="600" y="65" text-anchor="middle" class="subtitle">IIT 4.0 Phi, Causal Emergence, Null Hypothesis Testing</text>
"#,
    );

    // Panel 1: Power spectrum (y=100, h=300)
    svg.push_str(&render_power_spectrum(ps, &tpm.bin_edges, 50, 100, 1100, 280));

    // Panel 2: TPM heatmap (y=420, h=300)
    svg.push_str(&render_tpm_heatmap(tpm, 50, 420, 500, 280));

    // Panel 3: Phi spectrum (y=420, h=300, x=600)
    svg.push_str(&render_phi_spectrum(
        &results.phi_spectrum,
        600,
        420,
        550,
        280,
    ));

    // Panel 4: Null distribution (y=750, h=300)
    svg.push_str(&render_null_distribution(
        &results.null_phis,
        results.full_phi.phi,
        50,
        750,
        1100,
        280,
    ));

    // Panel 5: Summary stats (y=1100)
    svg.push_str(&render_summary_stats(results, tpm, 50, 1100));

    svg.push_str("</svg>\n");
    svg
}

fn render_power_spectrum(
    ps: &PowerSpectrum,
    bin_edges: &[f64],
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">CMB TT Power Spectrum (D_l vs l)</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f8f8f8\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    let l_max = ps.ells.last().unwrap_or(&2500.0);
    let d_max = ps.d_ells.iter().cloned().fold(0.0f64, f64::max).max(1.0);

    // Plot power spectrum as polyline (subsample for SVG size)
    let step = (ps.ells.len() / 500).max(1);
    s.push_str("<polyline fill=\"none\" stroke=\"#4a90d9\" stroke-width=\"1.5\" points=\"");
    for i in (0..ps.ells.len()).step_by(step) {
        let px = (ps.ells[i].ln() / l_max.ln() * w as f64) as i32;
        let py = h - (ps.d_ells[i] / d_max * (h - 20) as f64) as i32;
        s.push_str(&format!("{},{} ", px, py));
    }
    s.push_str("\"/>\n");

    // Mark bin edges
    for &edge in bin_edges {
        if edge > 0.0 {
            let px = (edge.ln() / l_max.ln() * w as f64) as i32;
            s.push_str(&format!(
                "<line x1=\"{}\" y1=\"0\" x2=\"{}\" y2=\"{}\" stroke=\"#e74c3c\" stroke-width=\"0.5\" stroke-dasharray=\"3,3\"/>\n",
                px, px, h
            ));
        }
    }

    s.push_str("</g>\n");
    s
}

fn render_tpm_heatmap(tpm: &TransitionMatrix, x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Transition Probability Matrix</text>\n",
        w / 2
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
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({},{},{})\"/>\n",
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

fn render_phi_spectrum(
    spectrum: &[(usize, usize, f64)],
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str(&format!(
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Phi Spectrum (sliding window)</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f8f8f8\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    let max_phi = spectrum
        .iter()
        .map(|x| x.2)
        .fold(0.0f64, f64::max)
        .max(1e-10);
    let bar_w = w as f64 / spectrum.len().max(1) as f64;

    for (i, (_, _, phi)) in spectrum.iter().enumerate() {
        let bar_h = (phi / max_phi * (h - 30) as f64) as i32;
        let bx = (i as f64 * bar_w) as i32 + 1;
        s.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"bar\" rx=\"1\"/>\n",
            bx,
            h - bar_h - 15,
            (bar_w - 2.0).max(1.0) as i32,
            bar_h
        ));
    }

    s.push_str("</g>\n");
    s
}

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
        "<text x=\"{}\" y=\"-5\" text-anchor=\"middle\" class=\"subtitle\">Null Distribution (GRF) vs Observed Phi</text>\n",
        w / 2
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f8f8f8\" stroke=\"#ddd\"/>\n",
        w, h
    ));

    if null_phis.is_empty() {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">No null samples</text>\n",
            w / 2,
            h / 2
        ));
        s.push_str("</g>\n");
        return s;
    }

    // Histogram of null phis
    let n_hist_bins = 30usize;
    let phi_min =
        null_phis.iter().cloned().fold(f64::INFINITY, f64::min).min(observed) * 0.9;
    let phi_max =
        null_phis.iter().cloned().fold(0.0f64, f64::max).max(observed) * 1.1;
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
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#e74c3c\" font-size=\"10\">Observed</text>\n",
        obs_x,
        h - 5
    ));

    s.push_str("</g>\n");
    s
}

fn render_summary_stats(results: &AnalysisResults, tpm: &TransitionMatrix, x: i32, y: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);
    s.push_str("<text x=\"0\" y=\"0\" class=\"subtitle\">Summary Statistics</text>\n");

    let null_mean = if results.null_phis.is_empty() {
        0.0
    } else {
        results.null_phis.iter().sum::<f64>() / results.null_phis.len() as f64
    };
    let lines = vec![
        format!(
            "Full System Phi:  {:.6}  (n={})",
            results.full_phi.phi, tpm.size
        ),
        format!(
            "Null Mean Phi:    {:.6}  ({} samples)",
            null_mean,
            results.null_phis.len()
        ),
        format!("z-score:          {:.3}", results.z_score),
        format!("p-value:          {:.4}", results.p_value),
        format!(
            "EI (micro):       {:.4} bits",
            results.emergence.ei_micro
        ),
        format!("Determinism:      {:.4}", results.emergence.determinism),
        format!(
            "SVD Eff. Rank:    {}/{}",
            results.svd_emergence.effective_rank, tpm.size
        ),
        format!(
            "Emergence Index:  {:.4}",
            results.svd_emergence.emergence_index
        ),
        if results.p_value < 0.05 {
            "Verdict: ANOMALOUS -- warrants investigation".to_string()
        } else {
            "Verdict: CONSISTENT with Gaussian random field".to_string()
        },
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
