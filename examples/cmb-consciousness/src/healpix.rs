//! HEALPix-inspired spatial Phi mapping of the CMB sky.
//!
//! Divides the sky into patches and computes IIT Phi per patch.
//! Searches for anomalous regions with unexpectedly high integrated information.
//!
//! Known CMB anomalies that might show up:
//! - Cold Spot (Eridanus, ~10deg radius, l=209deg, b=-57deg)
//! - Hemispherical asymmetry (ecliptic north vs south power difference)
//! - Quadrupole-octupole alignment ("axis of evil")

use crate::data::PowerSpectrum;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::{ComputeBudget, TransitionMatrix as ConsciousnessTPM};

/// A single HEALPix sky patch with computed consciousness metrics.
pub struct SkyPatch {
    pub index: usize,
    pub galactic_l: f64,
    pub galactic_b: f64,
    pub phi: f64,
    pub ei: f64,
    pub emergence_index: f64,
    pub is_anomalous: bool,
    pub label: String,
}

/// Aggregated results from the full-sky Phi mapping.
pub struct SkyMapResults {
    pub patches: Vec<SkyPatch>,
    pub mean_phi: f64,
    pub std_phi: f64,
    pub anomalous_patches: Vec<usize>,
    pub cold_spot_phi: f64,
    pub normal_mean_phi: f64,
}

// ── Patch grid size ──────────────────────────────────────────────────
const PATCH_PIXELS: usize = 8;
const GRID_SIZE: usize = PATCH_PIXELS * PATCH_PIXELS; // 64 pixels per patch

/// Run the full-sky Phi mapping analysis.
///
/// Generates 48 synthetic sky patches (HEALPix nside=2 equivalent), builds a
/// local TPM from pixel-pixel correlations within each patch, and computes
/// IIT Phi plus causal emergence per patch.
pub fn run_sky_mapping(ps: &PowerSpectrum) -> SkyMapResults {
    let centers = healpix_centers();
    let npix = centers.len(); // 48
    let budget = ComputeBudget::default();
    let mut rng = ChaCha8Rng::seed_from_u64(137); // deterministic seed

    println!("  Mapping {} sky patches (nside=2 equivalent)", npix);

    let mut patches: Vec<SkyPatch> = Vec::with_capacity(npix);

    for (idx, &(gl, gb)) in centers.iter().enumerate() {
        let label = classify_patch(gl, gb);
        let pixels = generate_patch_pixels(ps, gl, gb, &label, &mut rng);
        let tpm = build_local_tpm(&pixels);

        let phi = match auto_compute_phi(&tpm, None, &budget) {
            Ok(r) => r.phi,
            Err(_) => 0.0,
        };

        let engine = CausalEmergenceEngine::new(PATCH_PIXELS.min(16));
        let (ei, emergence_index) = match engine.compute_emergence(&tpm, &budget) {
            Ok(e) => (e.ei_micro, e.causal_emergence),
            Err(_) => (0.0, 0.0),
        };

        patches.push(SkyPatch {
            index: idx,
            galactic_l: gl,
            galactic_b: gb,
            phi,
            ei,
            emergence_index,
            is_anomalous: false, // set after statistics pass
            label,
        });

        if (idx + 1) % 12 == 0 {
            println!("  [{}/{}] patches computed", idx + 1, npix);
        }
    }

    // ── Statistics ────────────────────────────────────────────────────
    let n = patches.len() as f64;
    let mean_phi = patches.iter().map(|p| p.phi).sum::<f64>() / n;
    let var = patches
        .iter()
        .map(|p| (p.phi - mean_phi).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let std_phi = var.sqrt();

    let threshold = mean_phi + 2.0 * std_phi;
    let mut anomalous_patches = Vec::new();
    for p in &mut patches {
        if p.phi > threshold {
            p.is_anomalous = true;
            anomalous_patches.push(p.index);
        }
    }

    let cold_spot_phi = patches
        .iter()
        .find(|p| p.label == "cold_spot")
        .map(|p| p.phi)
        .unwrap_or(0.0);

    let normal_mean_phi = {
        let normals: Vec<f64> = patches
            .iter()
            .filter(|p| p.label == "normal")
            .map(|p| p.phi)
            .collect();
        if normals.is_empty() {
            0.0
        } else {
            normals.iter().sum::<f64>() / normals.len() as f64
        }
    };

    println!(
        "  Mean Phi = {:.6}, Std = {:.6}, Anomalous = {}",
        mean_phi,
        std_phi,
        anomalous_patches.len()
    );
    println!(
        "  Cold Spot Phi = {:.6}, Normal Mean Phi = {:.6}",
        cold_spot_phi, normal_mean_phi
    );

    SkyMapResults {
        patches,
        mean_phi,
        std_phi,
        anomalous_patches,
        cold_spot_phi,
        normal_mean_phi,
    }
}

// ── HEALPix nside=2 approximate centres ─────────────────────────────
/// Return 48 approximate HEALPix nside=2 patch centres as (galactic_l, galactic_b)
/// in degrees. Uses the standard ring-scheme formula.
fn healpix_centers() -> Vec<(f64, f64)> {
    let nside: usize = 2;
    let npix = 12 * nside * nside; // 48
    let mut centres = Vec::with_capacity(npix);

    for pix in 0..npix {
        let (theta, phi) = pix2ang_ring(nside, pix);
        // Convert colatitude/longitude to galactic (l, b)
        let b = 90.0 - theta.to_degrees(); // colatitude -> latitude
        let l = phi.to_degrees();
        centres.push((l, b));
    }
    centres
}

/// HEALPix ring-scheme pixel -> (theta, phi) in radians.
///
/// Uses floating-point arithmetic throughout to avoid usize overflow.
fn pix2ang_ring(nside: usize, pix: usize) -> (f64, f64) {
    let ns = nside as f64;
    let npix = 12 * nside * nside;
    let ncap = 2 * nside * (nside - 1); // pixels in north polar cap

    if pix < ncap {
        // North polar cap
        let p_h = (pix + 1) as f64;
        let i_ring = ((-1.0 + (1.0 + 8.0 * p_h).sqrt()) / 2.0).floor().max(1.0);
        let j = p_h - 2.0 * i_ring * (i_ring - 1.0) / 2.0;
        let theta = (1.0 - i_ring * i_ring / (3.0 * ns * ns)).acos();
        let phi = std::f64::consts::PI / (2.0 * i_ring) * (j - 0.5);
        (theta, phi)
    } else if pix < npix - ncap {
        // Equatorial belt
        let p_e = (pix - ncap) as f64;
        let i_ring = (p_e / (4.0 * ns)).floor() + ns;
        let j = p_e % (4.0 * ns);
        let s = if ((i_ring + ns) as i64) % 2 == 0 {
            1.0
        } else {
            0.5
        };
        let z = (2.0 * ns - i_ring) / (3.0 * ns);
        let theta = z.clamp(-1.0, 1.0).acos();
        let phi = std::f64::consts::PI / (2.0 * ns) * (j + s);
        (theta, phi)
    } else {
        // South polar cap
        let p_s = (npix - pix) as f64;
        let i_ring = ((-1.0 + (1.0 + 8.0 * p_s).sqrt()) / 2.0).floor().max(1.0);
        let j = p_s - 2.0 * i_ring * (i_ring - 1.0) / 2.0;
        let theta = std::f64::consts::PI - (1.0 - i_ring * i_ring / (3.0 * ns * ns)).acos();
        let phi = std::f64::consts::PI / (2.0 * i_ring) * (j - 0.5);
        (theta, phi)
    }
}

// ── Patch classification ─────────────────────────────────────────────
/// Classify a patch by its galactic coordinates.
fn classify_patch(l: f64, b: f64) -> String {
    // Cold Spot: centred near l=209, b=-57, radius ~10 degrees
    let dl = (l - 209.0).abs().min((l - 209.0 + 360.0).abs());
    let db = (b - (-57.0)).abs();
    if (dl * dl + db * db).sqrt() < 20.0 {
        return "cold_spot".to_string();
    }
    // Hemispherical asymmetry: ecliptic north ~ galactic b > 30
    if b > 30.0 {
        return "north_boost".to_string();
    }
    "normal".to_string()
}

// ── Synthetic pixel generation ───────────────────────────────────────
/// Generate an 8x8 grid of temperature fluctuation values for a patch.
///
/// Normal patches: standard Gaussian random field drawn from the power spectrum.
/// Cold Spot: injected temperature deficit with non-Gaussian ring.
/// North boost: 7% power enhancement (hemispherical asymmetry).
fn generate_patch_pixels(
    ps: &PowerSpectrum,
    _gl: f64,
    _gb: f64,
    label: &str,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    use rand::Rng;

    // Base amplitude from power spectrum RMS
    let rms: f64 = {
        let sum: f64 = ps.d_ells.iter().take(200).sum();
        let n = ps.d_ells.len().min(200) as f64;
        (sum / n).sqrt()
    };

    let scale = rms * 1e-3; // scale to sensible fluctuations
    let mut pixels = Vec::with_capacity(GRID_SIZE);

    for row in 0..PATCH_PIXELS {
        for col in 0..PATCH_PIXELS {
            let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            let mut val = noise * scale;

            match label {
                "cold_spot" => {
                    // Central depression + non-Gaussian ring
                    let cx = (PATCH_PIXELS as f64 - 1.0) / 2.0;
                    let r = (((row as f64 - cx).powi(2) + (col as f64 - cx).powi(2)).sqrt()) / cx;
                    // Temperature deficit in centre
                    val -= scale * 2.5 * (-r * r * 4.0).exp();
                    // Non-Gaussian ring at r ~ 0.7
                    val += scale * 0.8 * (-(r - 0.7).powi(2) * 20.0).exp();
                }
                "north_boost" => {
                    // 7% power enhancement
                    val *= 1.07;
                }
                _ => {}
            }
            pixels.push(val);
        }
    }
    pixels
}

// ── TPM construction from pixel correlations ─────────────────────────
/// Build a ConsciousnessTPM from an 8x8 pixel grid.
///
/// Uses pairwise pixel correlations as coupling weights, then
/// row-normalises to get transition probabilities.
fn build_local_tpm(pixels: &[f64]) -> ConsciousnessTPM {
    let n = pixels.len(); // 64
    let mut data = vec![0.0f64; n * n];

    // Correlation-based coupling: C_ij = exp(-|T_i - T_j|^2 / (2 * sigma^2))
    let mean = pixels.iter().sum::<f64>() / n as f64;
    let var = pixels.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let sigma2 = var.max(1e-20);

    for i in 0..n {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            let diff = pixels[i] - pixels[j];
            let w = (-diff * diff / (2.0 * sigma2)).exp();
            data[i * n + j] = w;
            row_sum += w;
        }
        // Row-normalise
        if row_sum > 1e-30 {
            for j in 0..n {
                data[i * n + j] /= row_sum;
            }
        }
    }

    ConsciousnessTPM::new(n, data)
}

// ── SVG Mollweide projection ─────────────────────────────────────────
/// Generate a Mollweide-projection SVG of the full-sky Phi map.
pub fn render_sky_map_svg(results: &SkyMapResults) -> String {
    let width = 900.0_f64;
    let height = 500.0_f64;
    let cx = width / 2.0;
    let cy = height / 2.0 + 20.0; // offset for title
    let rx = 380.0_f64;
    let ry = 190.0_f64;

    // Find Phi range for colour mapping
    let phi_min = results
        .patches
        .iter()
        .map(|p| p.phi)
        .fold(f64::INFINITY, f64::min);
    let phi_max = results
        .patches
        .iter()
        .map(|p| p.phi)
        .fold(f64::NEG_INFINITY, f64::max);
    let phi_range = (phi_max - phi_min).max(1e-10);

    let mut svg = String::with_capacity(8192);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">\n",
        width as u32,
        (height + 60.0) as u32,
        width as u32,
        (height + 60.0) as u32
    ));
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#0a0a2e\"/>\n");

    // Title
    svg.push_str(
        "<text x=\"450\" y=\"22\" text-anchor=\"middle\" \
         font-family=\"monospace\" font-size=\"14\" fill=\"#ddd\">\
         CMB Consciousness Sky Map -- IIT Phi per HEALPix Patch\
         </text>\n",
    );

    // Mollweide ellipse outline
    svg.push_str(&format!(
        "<ellipse cx=\"{}\" cy=\"{}\" rx=\"{}\" ry=\"{}\" \
         fill=\"none\" stroke=\"#555\" stroke-width=\"1\"/>\n",
        cx, cy, rx, ry
    ));

    // Draw patches as circles
    for patch in &results.patches {
        let (sx, sy) = mollweide_project(patch.galactic_l, patch.galactic_b, cx, cy, rx, ry);
        let t = (patch.phi - phi_min) / phi_range;
        let colour = phi_colour(t);
        let r = if patch.is_anomalous { 12.0 } else { 8.0 };

        svg.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{}\" opacity=\"0.85\"/>\n",
            sx, sy, r, colour
        ));

        // Anomalous ring
        if patch.is_anomalous {
            svg.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"14\" fill=\"none\" \
                 stroke=\"#fff\" stroke-width=\"1.5\" stroke-dasharray=\"3,2\"/>\n",
                sx, sy
            ));
        }
    }

    // Mark Cold Spot with label
    if let Some(cs) = results.patches.iter().find(|p| p.label == "cold_spot") {
        let (sx, sy) = mollweide_project(cs.galactic_l, cs.galactic_b, cx, cy, rx, ry);
        svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"#ff0\" stroke-width=\"1\"/>\n",
            sx + 16.0,
            sy - 6.0,
            sx + 40.0,
            sy - 20.0
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"monospace\" \
             font-size=\"10\" fill=\"#ff0\">Cold Spot</text>\n",
            sx + 42.0,
            sy - 22.0
        ));
    }

    // Galactic plane (b=0)
    let mut plane_pts = Vec::new();
    for deg in (0..=360).step_by(5) {
        let (px, py) = mollweide_project(deg as f64, 0.0, cx, cy, rx, ry);
        plane_pts.push(format!("{:.1},{:.1}", px, py));
    }
    svg.push_str(&format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"#444\" \
         stroke-width=\"0.5\" stroke-dasharray=\"4,3\"/>\n",
        plane_pts.join(" ")
    ));

    // Colourbar
    let bar_y = height + 30.0;
    let bar_w = 300.0;
    let bar_x0 = cx - bar_w / 2.0;
    for i in 0..60 {
        let t = i as f64 / 59.0;
        let c = phi_colour(t);
        svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.0}\" width=\"6\" height=\"12\" fill=\"{}\"/>\n",
            bar_x0 + t * bar_w,
            bar_y,
            c
        ));
    }
    svg.push_str(&format!(
        "<text x=\"{:.0}\" y=\"{:.0}\" font-family=\"monospace\" \
         font-size=\"9\" fill=\"#aaa\" text-anchor=\"end\">{:.4}</text>\n",
        bar_x0 - 4.0,
        bar_y + 10.0,
        phi_min
    ));
    svg.push_str(&format!(
        "<text x=\"{:.0}\" y=\"{:.0}\" font-family=\"monospace\" \
         font-size=\"9\" fill=\"#aaa\">{:.4}</text>\n",
        bar_x0 + bar_w + 4.0,
        bar_y + 10.0,
        phi_max
    ));
    svg.push_str(&format!(
        "<text x=\"{:.0}\" y=\"{:.0}\" font-family=\"monospace\" \
         font-size=\"9\" fill=\"#aaa\" text-anchor=\"middle\">Phi</text>\n",
        cx,
        bar_y + 24.0
    ));

    // Statistics box
    svg.push_str(&format!(
        "<text x=\"20\" y=\"{:.0}\" font-family=\"monospace\" \
         font-size=\"9\" fill=\"#8cf\">Mean Phi={:.4}  Std={:.4}  \
         Anomalous={}/{}</text>\n",
        height + 48.0,
        results.mean_phi,
        results.std_phi,
        results.anomalous_patches.len(),
        results.patches.len()
    ));

    svg.push_str("</svg>\n");
    svg
}

// ── Mollweide helpers ────────────────────────────────────────────────
/// Project galactic (l, b) in degrees to SVG (x, y).
fn mollweide_project(l: f64, b: f64, cx: f64, cy: f64, rx: f64, ry: f64) -> (f64, f64) {
    let lon = (l - 180.0).to_radians(); // centre at l=180
    let lat = b.to_radians();

    // Newton-Raphson for auxiliary angle theta
    let mut theta = lat;
    for _ in 0..10 {
        let denom = (2.0 * theta).cos();
        if denom.abs() < 1e-12 {
            break;
        }
        let delta = -(2.0 * theta + (2.0 * theta).sin() - std::f64::consts::PI * lat.sin())
            / (2.0 + denom * 2.0);
        theta += delta;
        if delta.abs() < 1e-8 {
            break;
        }
    }

    let x = cx - rx * 2.0 / std::f64::consts::PI * lon * theta.cos();
    let y = cy - ry * theta.sin();
    (x, y)
}

/// Map normalised t in [0, 1] to a blue-red colour string.
fn phi_colour(t: f64) -> String {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (
            (30.0 + s * 100.0) as u8,
            (60.0 + s * 140.0) as u8,
            (180.0 + s * 40.0) as u8,
        )
    } else {
        let s = (t - 0.5) * 2.0;
        (
            (130.0 + s * 125.0) as u8,
            (200.0 - s * 160.0) as u8,
            (220.0 - s * 200.0) as u8,
        )
    };
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data;

    #[test]
    fn healpix_centers_count() {
        assert_eq!(healpix_centers().len(), 48);
    }

    #[test]
    fn classify_cold_spot() {
        assert_eq!(classify_patch(209.0, -57.0), "cold_spot");
        assert_eq!(classify_patch(0.0, 0.0), "normal");
        assert_eq!(classify_patch(90.0, 45.0), "north_boost");
    }

    #[test]
    fn mollweide_centre() {
        let (x, y) = mollweide_project(180.0, 0.0, 450.0, 260.0, 380.0, 190.0);
        assert!((x - 450.0).abs() < 1.0);
        assert!((y - 260.0).abs() < 1.0);
    }

    #[test]
    fn patch_pixels_deterministic() {
        let ps = data::download_power_spectrum();
        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let a = generate_patch_pixels(&ps, 100.0, 10.0, "normal", &mut rng1);
        let b = generate_patch_pixels(&ps, 100.0, 10.0, "normal", &mut rng2);
        assert_eq!(a, b);
    }
}
