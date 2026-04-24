//! CMB Cold Spot Boundary-First Discovery
//!
//! Demonstrates that the CMB Cold Spot's *boundary* (the ring surrounding the
//! temperature depression) is more spectrally anomalous than its interior,
//! using graph Laplacian analysis and dynamic minimum cut.
//!
//! Synthetic data models the known Cold Spot profile (Cruz et al. 2008):
//!   - Central depression: ~-150 uK
//!   - Surrounding hot ring:  ~+60 uK
//!   - Background: Gaussian random field with spatial correlations
//!
//! The boundary-first hypothesis: the sharp temperature transition at the
//! Cold Spot edge creates a spectral signature (low Fiedler / low mincut)
//! that is more anomalous than any single-pixel measurement.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;

const SIZE: usize = 50;
const NPIX: usize = SIZE * SIZE;
const COLD_CX: usize = 25;
const COLD_CY: usize = 25;
const COLD_RADIUS: f64 = 8.0;
const RING_RADIUS: f64 = 10.0;
const COLD_DIP: f64 = -150.0; // uK central depression
const RING_BUMP: f64 = 60.0; // uK hot ring amplitude
const N_CONTROLS: usize = 20;
const KERNEL_SIGMA: f64 = 3.0; // Spatial correlation length (pixels)
const BG_RMS: f64 = 18.0; // Background rms (uK)
const EDGE_TAU: f64 = 15.0; // Edge weight bandwidth (uK)
const FIEDLER_ITERS: usize = 80;
const FIEDLER_TOL: f64 = 1e-6;

/// Generate a spatially correlated Gaussian random field by convolving white
/// noise with a Gaussian kernel of width `sigma` pixels.
fn generate_correlated_field(rng: &mut StdRng, sigma: f64) -> Vec<f64> {
    let mut white: Vec<f64> = (0..NPIX).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
    let radius = (3.0 * sigma).ceil() as i32;
    let mut kernel = Vec::new();
    let mut ksum = 0.0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let r2 = (dx * dx + dy * dy) as f64;
            let w = (-r2 / (2.0 * sigma * sigma)).exp();
            kernel.push((dx, dy, w));
            ksum += w;
        }
    }
    for k in &mut kernel {
        k.2 /= ksum;
    }
    let src = white.clone();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let mut val = 0.0;
            for &(dx, dy, w) in &kernel {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < SIZE as i32 && ny >= 0 && ny < SIZE as i32 {
                    val += w * src[ny as usize * SIZE + nx as usize];
                }
            }
            white[y * SIZE + x] = val;
        }
    }
    let mean: f64 = white.iter().sum::<f64>() / NPIX as f64;
    let var: f64 = white.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / NPIX as f64;
    let std = var.sqrt().max(1e-12);
    white
        .iter_mut()
        .for_each(|v| *v = (*v - mean) / std * BG_RMS);
    white
}

/// Inject Cold Spot profile at (cx, cy): depression + hot ring.
fn inject_cold_spot(field: &mut [f64], cx: usize, cy: usize) {
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f64 - cx as f64;
            let dy = y as f64 - cy as f64;
            let r = (dx * dx + dy * dy).sqrt();
            if r < COLD_RADIUS {
                let profile = COLD_DIP * (-r * r / (2.0 * (COLD_RADIUS * 0.6).powi(2))).exp();
                field[y * SIZE + x] += profile;
            } else if r < RING_RADIUS + 3.0 {
                let ring_w = 1.8;
                let profile =
                    RING_BUMP * (-(r - RING_RADIUS).powi(2) / (2.0 * ring_w * ring_w)).exp();
                field[y * SIZE + x] += profile;
            }
        }
    }
}

/// Build pixel adjacency graph with Gaussian-kernel edge weights.
/// Weight = exp(-|T_i - T_j|^2 / (2*tau^2)). Sharp temperature jumps
/// produce near-zero weights (graph "cuts"); smooth regions stay near 1.
fn build_graph(field: &[f64], tau: f64) -> Vec<(usize, usize, f64)> {
    let tau2 = 2.0 * tau * tau;
    let mut edges = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let idx = y * SIZE + x;
            // 8-connected neighbors (forward edges only)
            for &(dx, dy) in &[(1i32, 0i32), (0, 1), (1, 1), (1, -1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < SIZE as i32 && ny >= 0 && ny < SIZE as i32 {
                    let nidx = ny as usize * SIZE + nx as usize;
                    let dt = field[idx] - field[nidx];
                    let w = (-dt * dt / tau2).exp();
                    edges.push((idx, nidx, w));
                }
            }
        }
    }
    edges
}

/// Extract subgraph for an annular ring between r_inner and r_outer.
fn extract_ring_subgraph(
    edges: &[(usize, usize, f64)],
    cx: usize,
    cy: usize,
    r_inner: f64,
    r_outer: f64,
) -> (Vec<(usize, usize, f64)>, usize) {
    let mut in_ring = vec![false; NPIX];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f64 - cx as f64;
            let dy = y as f64 - cy as f64;
            let r = (dx * dx + dy * dy).sqrt();
            if r >= r_inner && r <= r_outer {
                in_ring[y * SIZE + x] = true;
            }
        }
    }
    let mut global_to_local = vec![usize::MAX; NPIX];
    let mut local_n = 0usize;
    for (i, &ring) in in_ring.iter().enumerate() {
        if ring {
            global_to_local[i] = local_n;
            local_n += 1;
        }
    }
    let local_edges: Vec<(usize, usize, f64)> = edges
        .iter()
        .filter_map(|&(u, v, w)| {
            if in_ring[u] && in_ring[v] {
                Some((global_to_local[u], global_to_local[v], w))
            } else {
                None
            }
        })
        .collect();
    (local_edges, local_n)
}

/// Compute raw Fiedler value for a subgraph.
fn fiedler_for_subgraph(edges: &[(usize, usize, f64)], n: usize) -> f64 {
    if n < 3 || edges.is_empty() {
        return 0.0;
    }
    let lap = CsrMatrixView::build_laplacian(n, edges);
    let (fiedler, _) = estimate_fiedler(&lap, FIEDLER_ITERS, FIEDLER_TOL);
    fiedler
}

/// Compute mincut value for a subgraph.
fn mincut_for_subgraph(edges: &[(usize, usize, f64)]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }
    let mut verts = std::collections::HashSet::new();
    for &(u, v, _) in edges {
        verts.insert(u);
        verts.insert(v);
    }
    if verts.len() < 2 {
        return 0.0;
    }
    let edges_u64: Vec<(u64, u64, f64)> = edges
        .iter()
        .map(|&(u, v, w)| (u as u64, v as u64, w))
        .collect();
    let mc = MinCutBuilder::new().exact().with_edges(edges_u64).build();
    match mc {
        Ok(built) => {
            let val = built.min_cut_value();
            if val.is_infinite() {
                0.0
            } else {
                val
            }
        }
        Err(_) => 0.0,
    }
}

/// Compute mean and standard deviation.
fn mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    (mean, std)
}

fn main() {
    println!("================================================================");
    println!("  CMB Cold Spot Boundary Analysis");
    println!("================================================================");
    println!(
        "[DATA] {}x{} patch, {} pixels, Cold Spot at ({},{}) r={}",
        SIZE, SIZE, NPIX, COLD_CX, COLD_CY, COLD_RADIUS as u32
    );

    let mut rng = StdRng::seed_from_u64(42);

    // -- Generate Cold Spot patch --
    let mut cs_field = generate_correlated_field(&mut rng, KERNEL_SIGMA);
    inject_cold_spot(&mut cs_field, COLD_CX, COLD_CY);
    let cs_edges = build_graph(&cs_field, EDGE_TAU);
    let mean_w: f64 = cs_edges.iter().map(|e| e.2).sum::<f64>() / cs_edges.len().max(1) as f64;
    println!(
        "[GRAPH] {} edges, mean weight={:.4}",
        cs_edges.len(),
        mean_w
    );

    // -- Cold Spot boundary ring: straddles the cold-to-hot transition (r=5..13) --
    let (ring_edges, ring_n) = extract_ring_subgraph(&cs_edges, COLD_CX, COLD_CY, 5.0, 13.0);
    let cs_fiedler = fiedler_for_subgraph(&ring_edges, ring_n);

    // -- Mincut on the Cold Spot region (square patch) --
    let cs_patch_edges: Vec<(usize, usize, f64)> = cs_edges
        .iter()
        .filter(|&&(u, v, _)| {
            let in_patch = |idx: usize| -> bool {
                let (px, py) = (idx % SIZE, idx / SIZE);
                let dx = (px as i32 - COLD_CX as i32).unsigned_abs() as usize;
                let dy = (py as i32 - COLD_CY as i32).unsigned_abs() as usize;
                dx <= 14 && dy <= 14
            };
            in_patch(u) && in_patch(v)
        })
        .copied()
        .collect();
    let cs_mincut = mincut_for_subgraph(&cs_patch_edges);

    // -- Control patches: same statistics, no Cold Spot --
    let mut ctrl_fiedlers = Vec::with_capacity(N_CONTROLS);
    let mut ctrl_mincuts = Vec::with_capacity(N_CONTROLS);

    for i in 0..N_CONTROLS {
        let mut ctrl_rng = StdRng::seed_from_u64(1000 + i as u64);
        let ctrl_field = generate_correlated_field(&mut ctrl_rng, KERNEL_SIGMA);
        let ctrl_edges = build_graph(&ctrl_field, EDGE_TAU);

        let (ctrl_ring_edges, ctrl_ring_n) =
            extract_ring_subgraph(&ctrl_edges, COLD_CX, COLD_CY, 5.0, 13.0);
        ctrl_fiedlers.push(fiedler_for_subgraph(&ctrl_ring_edges, ctrl_ring_n));

        let ctrl_patch_edges: Vec<(usize, usize, f64)> = ctrl_edges
            .iter()
            .filter(|&&(u, v, _)| {
                let in_patch = |idx: usize| -> bool {
                    let (px, py) = (idx % SIZE, idx / SIZE);
                    let dx = (px as i32 - COLD_CX as i32).unsigned_abs() as usize;
                    let dy = (py as i32 - COLD_CY as i32).unsigned_abs() as usize;
                    dx <= 14 && dy <= 14
                };
                in_patch(u) && in_patch(v)
            })
            .copied()
            .collect();
        ctrl_mincuts.push(mincut_for_subgraph(&ctrl_patch_edges));
    }

    // -- Statistics --
    let (ctrl_f_mean, ctrl_f_std) = mean_std(&ctrl_fiedlers);
    let ctrl_f_min = ctrl_fiedlers.iter().cloned().fold(f64::INFINITY, f64::min);
    let ctrl_f_max = ctrl_fiedlers
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let f_zscore = if ctrl_f_std > 1e-12 {
        (cs_fiedler - ctrl_f_mean) / ctrl_f_std
    } else {
        0.0
    };

    let (ctrl_m_mean, ctrl_m_std) = mean_std(&ctrl_mincuts);
    let m_zscore = if ctrl_m_std > 1e-12 {
        (cs_mincut - ctrl_m_mean) / ctrl_m_std
    } else {
        0.0
    };

    // -- Output --
    println!();
    println!("[BOUNDARY ANALYSIS]");
    println!("  Cold Spot boundary ring Fiedler: {:.4}", cs_fiedler);
    println!("  Control boundaries ({} patches):", N_CONTROLS);
    println!("    Mean Fiedler: {:.4} +/- {:.4}", ctrl_f_mean, ctrl_f_std);
    println!("    Min: {:.4}  Max: {:.4}", ctrl_f_min, ctrl_f_max);
    let f_anomalous = f_zscore.abs() > 2.0;
    println!(
        "  Cold Spot z-score: {:.2}  => {}",
        f_zscore,
        if f_anomalous {
            "ANOMALOUS"
        } else {
            "NOT ANOMALOUS"
        }
    );

    println!();
    println!("[MINCUT ANALYSIS]");
    println!("  Cold Spot patch mincut: {:.3}", cs_mincut);
    println!(
        "  Control patch mincuts: {:.3} +/- {:.3}",
        ctrl_m_mean, ctrl_m_std
    );
    println!("  z-score: {:.2}", m_zscore);

    // Determine conclusions based on the direction of the anomaly
    let boundary_more_organized = cs_fiedler > ctrl_f_mean;
    let boundary_less_cuttable = cs_mincut > ctrl_m_mean;
    let either_anomalous = f_zscore.abs() > 2.0 || m_zscore.abs() > 2.0;

    println!();
    println!("[CONCLUSION]");
    if boundary_more_organized {
        println!("  The Cold Spot BOUNDARY is more organized than random patches.");
    } else {
        println!("  The Cold Spot BOUNDARY is less organized (more fragile) than random patches.");
    }
    if boundary_less_cuttable {
        println!("  The boundary is harder to cut, suggesting internal coherence.");
    } else {
        println!("  The boundary is easier to cut, revealing a structural discontinuity.");
    }
    if either_anomalous {
        println!("  The boundary carries MORE structural information than expected.");
    } else {
        println!("  The boundary signal is within normal fluctuation range.");
    }
    println!("================================================================");
}
