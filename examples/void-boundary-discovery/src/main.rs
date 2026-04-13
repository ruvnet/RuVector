//! Cosmic Void Boundary Information Content
//!
//! Tests the "boundary-first" thesis: void boundaries (walls, filaments)
//! carry more structural information than void interiors or exteriors.
//!
//! Generates a synthetic 2D cosmic web with voids, builds a galaxy proximity
//! graph, and compares spectral metrics (Fiedler value, mincut) across
//! boundary, interior, and exterior regions of each void.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;
use std::collections::{HashMap, HashSet};

const N_GALAXIES: usize = 1000;
const N_VOIDS: usize = 7;
const BOX_SIZE: f64 = 100.0;
const LINKING_LENGTH: f64 = 5.0;
const VOID_RADIUS_MIN: f64 = 12.0;
const VOID_RADIUS_MAX: f64 = 22.0;
const SEED: u64 = 42;

struct Void { cx: f64, cy: f64, radius: f64 }

fn generate_void_centers(rng: &mut StdRng) -> Vec<Void> {
    (0..N_VOIDS).map(|_| Void {
        cx: rng.gen::<f64>() * BOX_SIZE,
        cy: rng.gen::<f64>() * BOX_SIZE,
        radius: VOID_RADIUS_MIN + rng.gen::<f64>() * (VOID_RADIUS_MAX - VOID_RADIUS_MIN),
    }).collect()
}

fn periodic_sep(a: f64, b: f64) -> f64 {
    (a - b).abs().min(BOX_SIZE - (a - b).abs())
}

fn periodic_dist(a: &(f64, f64), b: &(f64, f64)) -> f64 {
    let (dx, dy) = (periodic_sep(a.0, b.0), periodic_sep(a.1, b.1));
    (dx * dx + dy * dy).sqrt()
}

fn dist_to_nearest_void(x: f64, y: f64, voids: &[Void]) -> (f64, usize) {
    voids.iter().enumerate()
        .map(|(i, v)| {
            let d = (periodic_sep(x, v.cx).powi(2) + periodic_sep(y, v.cy).powi(2)).sqrt();
            (d, i)
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap_or((f64::MAX, 0))
}

/// Generate galaxy positions anti-correlated with void centers.
/// Acceptance probability ~ (d / R)^2 clamped to [0,1].
fn generate_cosmic_web(rng: &mut StdRng, voids: &[Void]) -> Vec<(f64, f64)> {
    let mut galaxies = Vec::with_capacity(N_GALAXIES);
    let mut attempts = 0;
    while galaxies.len() < N_GALAXIES && attempts < N_GALAXIES * 50 {
        let (x, y) = (rng.gen::<f64>() * BOX_SIZE, rng.gen::<f64>() * BOX_SIZE);
        let (d, vi) = dist_to_nearest_void(x, y, voids);
        let p = ((d / voids[vi].radius).powi(2)).min(1.0);
        if rng.gen::<f64>() < p { galaxies.push((x, y)); }
        attempts += 1;
    }
    galaxies
}

/// Build proximity graph using spatial hashing for O(n) edge construction.
fn build_proximity_graph(galaxies: &[(f64, f64)]) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    let cell = LINKING_LENGTH;
    let ncells = (BOX_SIZE / cell).ceil() as usize;
    let mut grid: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, &(x, y)) in galaxies.iter().enumerate() {
        grid.entry(((x / cell) as usize % ncells, (y / cell) as usize % ncells))
            .or_default().push(i);
    }
    for (i, &(x, y)) in galaxies.iter().enumerate() {
        let (cx, cy) = ((x / cell) as usize % ncells, (y / cell) as usize % ncells);
        for dx in [ncells - 1, 0, 1] {
            for dy in [ncells - 1, 0, 1] {
                if let Some(bucket) = grid.get(&((cx + dx) % ncells, (cy + dy) % ncells)) {
                    for &j in bucket {
                        if j > i {
                            let d = periodic_dist(&(x, y), &galaxies[j]);
                            if d < LINKING_LENGTH && d > 1e-10 {
                                edges.push((i, j, 1.0 / d));
                            }
                        }
                    }
                }
            }
        }
    }
    edges
}

// --- Region classification ---

struct VoidRegions { boundary: Vec<usize>, interior: Vec<usize>, exterior: Vec<usize> }

fn classify_galaxies(galaxies: &[(f64, f64)], v: &Void) -> VoidRegions {
    let (mut boundary, mut interior, mut exterior) = (Vec::new(), Vec::new(), Vec::new());
    for (i, g) in galaxies.iter().enumerate() {
        let d = (periodic_sep(g.0, v.cx).powi(2) + periodic_sep(g.1, v.cy).powi(2)).sqrt();
        let ratio = d / v.radius;
        if ratio < 0.5 { interior.push(i); }
        else if (0.8..=1.2).contains(&ratio) { boundary.push(i); }
        else if ratio > 1.5 { exterior.push(i); }
    }
    VoidRegions { boundary, interior, exterior }
}

// --- Subgraph extraction ---

fn extract_subgraph(nodes: &[usize], edges: &[(usize, usize, f64)]) -> (Vec<(usize, usize, f64)>, usize) {
    let set: HashSet<usize> = nodes.iter().copied().collect();
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut nxt = 0;
    for &n in nodes { map.entry(n).or_insert_with(|| { let id = nxt; nxt += 1; id }); }
    let sub: Vec<_> = edges.iter()
        .filter(|(u, v, _)| set.contains(u) && set.contains(v))
        .map(|(u, v, w)| (map[u], map[v], *w)).collect();
    (sub, nxt)
}

// --- Spectral and mincut metrics ---

fn compute_fiedler(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
    if n < 2 || edges.is_empty() { return 0.0; }
    estimate_fiedler(&CsrMatrixView::build_laplacian(n, edges), 200, 1e-10).0
}

fn compute_mincut(edges: &[(usize, usize, f64)]) -> f64 {
    if edges.is_empty() { return 0.0; }
    let mc_edges: Vec<_> = edges.iter().map(|&(u, v, w)| (u as u64, v as u64, w)).collect();
    MinCutBuilder::new().exact().with_edges(mc_edges).build().map_or(0.0, |mc| mc.min_cut_value())
}

#[derive(Debug, Clone)]
struct RegionMetrics { count: usize, fiedler: f64, mincut: f64, mean_deg: f64 }

fn analyze_region(nodes: &[usize], all_edges: &[(usize, usize, f64)]) -> RegionMetrics {
    if nodes.len() < 2 {
        return RegionMetrics { count: nodes.len(), fiedler: 0.0, mincut: 0.0, mean_deg: 0.0 };
    }
    let (sub, n) = extract_subgraph(nodes, all_edges);
    let deg = if n == 0 { 0.0 } else { 2.0 * sub.len() as f64 / n as f64 };
    RegionMetrics { count: nodes.len(), fiedler: compute_fiedler(n, &sub), mincut: compute_mincut(&sub), mean_deg: deg }
}

// --- Wilcoxon signed-rank test (two-sided, paired) ---

fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.len() < 3 { return 1.0; }
    let mut diffs: Vec<(f64, f64)> = a.iter().zip(b)
        .map(|(x, y)| { let d = x - y; (d.abs(), d.signum()) })
        .filter(|(abs_d, _)| *abs_d > 1e-15).collect();
    if diffs.len() < 3 { return 1.0; }
    diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let w_plus: f64 = diffs.iter().enumerate()
        .filter(|(_, (_, s))| *s > 0.0)
        .map(|(r, _)| (r + 1) as f64).sum();
    let nr = diffs.len() as f64;
    let mean = nr * (nr + 1.0) / 4.0;
    let var = nr * (nr + 1.0) * (2.0 * nr + 1.0) / 24.0;
    if var < 1e-15 { return 1.0; }
    2.0 * std_normal_cdf(-((w_plus - mean) / var.sqrt()).abs())
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn std_normal_cdf(x: f64) -> f64 {
    if x < -8.0 { return 0.0; }
    if x > 8.0 { return 1.0; }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let p = 0.3989422804014327 * (-x * x / 2.0).exp();
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
        + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 { 1.0 - p * poly } else { p * poly }
}

// --- Main ---

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    println!("================================================================");
    println!("  Cosmic Void Boundary Information Content");
    println!("================================================================");

    let voids = generate_void_centers(&mut rng);
    let galaxies = generate_cosmic_web(&mut rng, &voids);
    println!("[COSMIC WEB] {} galaxies, {} voids, box {}x{}", galaxies.len(), voids.len(), BOX_SIZE, BOX_SIZE);

    let edges = build_proximity_graph(&galaxies);
    println!("[GRAPH] {} edges, linking length = {:.1}\n", edges.len(), LINKING_LENGTH);

    let mut all_boundary = Vec::new();
    let mut all_interior = Vec::new();
    let mut all_exterior = Vec::new();
    let (mut bnd_fiedlers, mut int_fiedlers) = (Vec::new(), Vec::new());
    let (mut bnd_gt_int, mut bnd_gt_ext, mut valid) = (0usize, 0usize, 0usize);

    println!("[VOID-BY-VOID ANALYSIS]");
    for (vi, v) in voids.iter().enumerate() {
        let regions = classify_galaxies(&galaxies, v);
        let bm = analyze_region(&regions.boundary, &edges);
        let im = analyze_region(&regions.interior, &edges);
        let em = analyze_region(&regions.exterior, &edges);

        println!("  Void {} (center: {:.1},{:.1}, radius: {:.1}):", vi + 1, v.cx, v.cy, v.radius);
        println!("    Boundary: {} gal, Fiedler={:.4}, mincut={:.2}, deg={:.2}", bm.count, bm.fiedler, bm.mincut, bm.mean_deg);
        println!("    Interior: {} gal, Fiedler={:.4}, mincut={:.2}, deg={:.2}", im.count, im.fiedler, im.mincut, im.mean_deg);
        println!("    Exterior: {} gal, Fiedler={:.4}, mincut={:.2}, deg={:.2}", em.count, em.fiedler, em.mincut, em.mean_deg);

        if bm.count >= 3 && im.count >= 2 {
            valid += 1;
            bnd_fiedlers.push(bm.fiedler);
            int_fiedlers.push(im.fiedler);
            if bm.fiedler > im.fiedler { bnd_gt_int += 1; }
            if bm.fiedler > em.fiedler { bnd_gt_ext += 1; }
        }
        all_boundary.push(bm);
        all_interior.push(im);
        all_exterior.push(em);
    }

    // Aggregate
    println!("\n[AGGREGATE]");
    let mean_of = |ms: &[RegionMetrics], f: fn(&RegionMetrics) -> f64| {
        let v: Vec<f64> = ms.iter().filter(|m| m.count >= 2).map(f).collect();
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };
    let (bf, inf, ef) = (
        mean_of(&all_boundary, |m| m.fiedler),
        mean_of(&all_interior, |m| m.fiedler),
        mean_of(&all_exterior, |m| m.fiedler),
    );
    let (bmc, imc, emc) = (
        mean_of(&all_boundary, |m| m.mincut),
        mean_of(&all_interior, |m| m.mincut),
        mean_of(&all_exterior, |m| m.mincut),
    );
    println!("  Mean Fiedler:  Boundary={:.4}  Interior={:.4}  Exterior={:.4}", bf, inf, ef);
    println!("  Mean Mincut:   Boundary={:.4}  Interior={:.4}  Exterior={:.4}", bmc, imc, emc);

    if valid > 0 {
        println!("  Boundary > Interior in {}/{} voids ({:.0}%)", bnd_gt_int, valid, 100.0 * bnd_gt_int as f64 / valid as f64);
        println!("  Boundary > Exterior in {}/{} voids ({:.0}%)", bnd_gt_ext, valid, 100.0 * bnd_gt_ext as f64 / valid as f64);
        if bnd_fiedlers.len() >= 3 {
            println!("  Wilcoxon p-value (boundary vs interior): {:.4}", wilcoxon_signed_rank(&bnd_fiedlers, &int_fiedlers));
        } else {
            println!("  Wilcoxon p-value: insufficient paired samples");
        }
    } else {
        println!("  No voids with sufficient galaxies in both boundary and interior.");
    }

    // Conclusion
    println!("\n[CONCLUSION]");
    if valid > 0 && bnd_gt_int > valid / 2 {
        println!("  Void boundaries carry MORE structural information");
        println!("  than void interiors in {}/{} ({:.0}%) of analyzed voids.", bnd_gt_int, valid, 100.0 * bnd_gt_int as f64 / valid as f64);
        println!("  The boundary-first thesis is supported: walls and filaments");
        println!("  surrounding cosmic voids are spectrally richer than the");
        println!("  sparse interior, confirming that structural organization");
        println!("  concentrates at void boundaries.");
    } else {
        println!("  Void boundaries carry LESS structural information");
        println!("  than void interiors in the majority of analyzed voids.");
        println!("  The boundary-first thesis is NOT supported for this");
        println!("  configuration. Consider adjusting void radii or linking length.");
    }
    println!("================================================================");
}
