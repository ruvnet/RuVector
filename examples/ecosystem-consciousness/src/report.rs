//! Report generation: text summary and SVG food web visualization.

use crate::analysis::EcosystemResult;

/// Print a text summary of all ecosystem results.
pub fn print_summary(results: &[EcosystemResult]) {
    for r in results {
        println!("\n========== {} ==========", r.name);
        println!("Species count:       {}", r.n_species);
        println!("Full system Phi:     {:.6}  ({})", r.full_phi, r.algorithm);

        println!("\nSpecies Phi contributions (sorted by importance):");
        for (_, name, phi_without, contribution) in &r.species_contributions {
            let bar_len = ((contribution.abs() / r.full_phi.max(1e-10)) * 20.0) as usize;
            let bar_char = if *contribution > 0.0 { "+" } else { "-" };
            println!(
                "  {:20} {:+.6}  (Phi without: {:.6})  {}",
                name,
                contribution,
                phi_without,
                bar_char.repeat(bar_len.min(30))
            );
        }

        println!("\nCausal Emergence:");
        println!("  EI (micro):        {:.4} bits", r.emergence.ei_micro);
        println!("  EI (macro):        {:.4} bits", r.emergence.ei_macro);
        println!("  Causal emergence:  {:.4}", r.emergence.causal_emergence);
        println!("  Determinism:       {:.4}", r.emergence.determinism);
        println!("  Degeneracy:        {:.4}", r.emergence.degeneracy);

        println!("\nSVD Emergence:");
        println!(
            "  Effective rank:    {}/{}",
            r.svd_emergence.effective_rank, r.n_species
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

/// Generate a self-contained SVG report with food web diagrams.
pub fn generate_svg(results: &[EcosystemResult]) -> String {
    let panel_height = 500;
    let total_height = 100 + results.len() as i32 * (panel_height + 50);
    let width = 1200;

    let mut svg = String::with_capacity(30_000);
    svg.push_str(&format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" font-family="monospace" font-size="12">
<style>
  .title {{ font-size: 20px; font-weight: bold; fill: #333; }}
  .subtitle {{ font-size: 14px; fill: #666; font-weight: bold; }}
  .label {{ font-size: 10px; fill: #333; }}
  .bar {{ fill: #4a90d9; }}
  .bar-neg {{ fill: #e74c3c; }}
  .stat {{ font-size: 11px; fill: #444; }}
</style>
<rect width="{}" height="{}" fill="white"/>
<text x="600" y="40" text-anchor="middle" class="title">Ecosystem Consciousness Analysis Report</text>
<text x="600" y="65" text-anchor="middle" class="stat">IIT Phi measures integrated information in food web networks</text>
"#,
        width, total_height, width, total_height
    ));

    for (idx, r) in results.iter().enumerate() {
        let y_off = 100 + idx as i32 * (panel_height + 50);
        svg.push_str(&render_ecosystem_panel(
            r,
            30,
            y_off,
            width - 60,
            panel_height,
        ));
    }

    svg.push_str("</svg>\n");
    svg
}

/// Render a single ecosystem panel with food web and contribution bars.
fn render_ecosystem_panel(r: &EcosystemResult, x: i32, y: i32, w: i32, h: i32) -> String {
    let mut s = format!("<g transform=\"translate({},{})\">\n", x, y);

    // Panel background
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ddd\" rx=\"5\"/>\n",
        w, h
    ));

    // Title
    s.push_str(&format!(
        "<text x=\"15\" y=\"25\" class=\"subtitle\">{} (n={}, Phi={:.6})</text>\n",
        r.name, r.n_species, r.full_phi
    ));

    // Left panel: food web node diagram (circular layout)
    let cx = 200;
    let cy = h / 2 + 20;
    let radius = 140;
    let n = r.n_species;

    // Draw nodes in a circle, sized by Phi contribution
    let max_contrib = r
        .species_contributions
        .iter()
        .map(|(_, _, _, c)| c.abs())
        .fold(0.0f64, f64::max)
        .max(1e-10);

    // Build contribution lookup by species index
    let mut contrib_by_idx = vec![0.0f64; n];
    for (idx, _, _, c) in &r.species_contributions {
        contrib_by_idx[*idx] = *c;
    }

    // Node positions
    let positions: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let angle =
                2.0 * std::f64::consts::PI * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
            (
                cx as f64 + radius as f64 * angle.cos(),
                cy as f64 + radius as f64 * angle.sin(),
            )
        })
        .collect();

    // Draw nodes
    for i in 0..n {
        let (nx, ny) = positions[i];
        let node_r = 8.0 + (contrib_by_idx[i].abs() / max_contrib * 14.0);
        let color = &r.trophic_colors[i];
        s.push_str(&format!(
            "<circle cx=\"{:.0}\" cy=\"{:.0}\" r=\"{:.1}\" fill=\"{}\" stroke=\"#333\" stroke-width=\"1\"/>\n",
            nx, ny, node_r, color
        ));
        // Label
        let label = if r.species_names[i].len() > 8 {
            &r.species_names[i][..8]
        } else {
            &r.species_names[i]
        };
        s.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" class=\"label\">{}</text>\n",
            nx,
            ny + node_r + 12.0,
            label
        ));
    }

    // Right panel: contribution bar chart
    let bar_x = 420;
    let bar_w = w - bar_x - 30;
    let bar_h = h - 80;
    s.push_str(&format!(
        "<text x=\"{}\" y=\"50\" class=\"subtitle\">Species Phi Contributions</text>\n",
        bar_x
    ));

    let contributions = &r.species_contributions;
    if !contributions.is_empty() {
        let row_h = (bar_h as f64 / contributions.len() as f64).min(30.0);
        let max_abs = contributions
            .iter()
            .map(|(_, _, _, c)| c.abs())
            .fold(0.0f64, f64::max)
            .max(1e-10);

        for (i, (_, name, _, contrib)) in contributions.iter().enumerate() {
            let ry = 65 + (i as f64 * row_h) as i32;
            let bw = (contrib.abs() / max_abs * (bar_w as f64 * 0.5)) as i32;
            let bar_class = if *contrib >= 0.0 { "bar" } else { "bar-neg" };

            // Species name
            let display_name = if name.len() > 16 {
                &name[..16]
            } else {
                name.as_str()
            };
            s.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" class=\"label\">{}</text>\n",
                bar_x + 120,
                ry + (row_h as i32) / 2 + 4,
                display_name
            ));

            // Bar
            s.push_str(&format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"{}\" rx=\"2\"/>\n",
                bar_x + 125,
                ry,
                bw.max(1),
                (row_h - 4.0).max(4.0) as i32,
                bar_class
            ));

            // Value label
            s.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" class=\"label\">{:+.4}</text>\n",
                bar_x + 130 + bw,
                ry + (row_h as i32) / 2 + 4,
                contrib
            ));
        }
    }

    // Stats box at bottom
    let stats_y = h - 40;
    s.push_str(&format!(
        "<text x=\"15\" y=\"{}\" class=\"stat\">EI_micro={:.3}  Emergence={:.3}  SVD rank={}/{}  Emergence idx={:.3}</text>\n",
        stats_y,
        r.emergence.ei_micro,
        r.emergence.causal_emergence,
        r.svd_emergence.effective_rank,
        r.n_species,
        r.svd_emergence.emergence_index
    ));

    s.push_str("</g>\n");
    s
}
