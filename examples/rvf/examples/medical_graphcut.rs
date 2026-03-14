//! Medical Imaging Lesion Detection via Graph Cut + RuVector
//!
//! Adapts the MRF/mincut formulation from the exomoon pipeline to medical
//! imaging lesion segmentation on synthetic 2D tissue grids:
//!
//!   1. Generate synthetic tissue with injected lesions (tumors)
//!   2. Extract per-voxel features: intensity, texture, multi-scale
//!   3. Build 4-connected grid graph, solve s-t mincut for segmentation
//!   4. Store voxel embeddings in RVF with modality metadata
//!   5. Evaluate: Dice, Jaccard, sensitivity, specificity
//!
//! Supports T1-MRI, T2-MRI, and CT modality noise profiles.
//!
//! Run: cargo run --example medical_graphcut --release

use rvf_runtime::{
    FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore,
};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_types::DerivationType;
use rvf_crypto::{create_witness_chain, verify_witness_chain, shake256_256, WitnessEntry};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRID_W: usize = 64;
const GRID_H: usize = 64;
const EMBED_DIM: usize = 32;

const FIELD_MODALITY: u16 = 0;
const FIELD_LESION_COUNT: u16 = 1;
const FIELD_GRID_X: u16 = 2;
const FIELD_GRID_Y: u16 = 3;

// ---------------------------------------------------------------------------
// LCG deterministic random (same pattern as exomoon_graphcut)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn lcg_f64(state: &mut u64) -> f64 {
    lcg_next(state);
    (*state >> 11) as f64 / ((1u64 << 53) as f64)
}

fn lcg_normal(state: &mut u64) -> f64 {
    let u1 = lcg_f64(state).max(1e-15);
    let u2 = lcg_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Imaging modality profiles
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Modality {
    T1Mri,
    T2Mri,
    Ct,
}

impl Modality {
    fn label(&self) -> &'static str {
        match self { Modality::T1Mri => "T1-MRI", Modality::T2Mri => "T2-MRI", Modality::Ct => "CT" }
    }
    /// Baseline healthy tissue intensity
    fn baseline(&self) -> f64 {
        match self { Modality::T1Mri => 120.0, Modality::T2Mri => 80.0, Modality::Ct => 40.0 }
    }
    /// Lesion intensity offset (positive = brighter than tissue)
    fn lesion_offset(&self) -> f64 {
        match self { Modality::T1Mri => -30.0, Modality::T2Mri => 60.0, Modality::Ct => 35.0 }
    }
    /// Gaussian noise sigma
    fn noise_sigma(&self) -> f64 {
        match self { Modality::T1Mri => 12.0, Modality::T2Mri => 15.0, Modality::Ct => 8.0 }
    }
    /// Lesion texture roughness (variance multiplier inside lesion)
    fn lesion_texture(&self) -> f64 {
        match self { Modality::T1Mri => 2.5, Modality::T2Mri => 3.0, Modality::Ct => 1.8 }
    }
}

// ---------------------------------------------------------------------------
// Synthetic lesion specification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Lesion {
    cx: f64,
    cy: f64,
    radius: f64,
}

// ---------------------------------------------------------------------------
// Synthetic tissue image generation
// ---------------------------------------------------------------------------

struct TissueImage {
    width: usize,
    height: usize,
    modality: Modality,
    /// Observed intensity per voxel (row-major)
    intensity: Vec<f64>,
    /// Ground truth: true if voxel is lesion
    ground_truth: Vec<bool>,
    lesions: Vec<Lesion>,
}

fn generate_tissue(modality: Modality, num_lesions: usize, seed: u64) -> TissueImage {
    let w = GRID_W;
    let h = GRID_H;
    let n = w * h;
    let mut rng = seed;
    let baseline = modality.baseline();
    let noise_sigma = modality.noise_sigma();

    let mut intensity = vec![0.0; n];
    let mut ground_truth = vec![false; n];

    // Generate lesion specs
    let mut lesions = Vec::with_capacity(num_lesions);
    for _ in 0..num_lesions {
        let cx = 10.0 + lcg_f64(&mut rng) * (w as f64 - 20.0);
        let cy = 10.0 + lcg_f64(&mut rng) * (h as f64 - 20.0);
        let radius = 3.0 + lcg_f64(&mut rng) * 6.0; // radius 3-9
        lesions.push(Lesion { cx, cy, radius });
    }

    // Fill voxels
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mut val = baseline;

            // Slow spatial variation (anatomical structure)
            val += 5.0 * ((x as f64 * 0.1).sin() + (y as f64 * 0.08).cos());

            // Check lesion membership
            let mut in_lesion = false;
            for les in &lesions {
                let dx = x as f64 - les.cx;
                let dy = y as f64 - les.cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= les.radius {
                    in_lesion = true;
                    // Smooth falloff near boundary
                    let falloff = 1.0 - (dist / les.radius).powi(2);
                    val += modality.lesion_offset() * falloff;
                    // Texture roughness inside lesion
                    val += lcg_normal(&mut rng) * noise_sigma * modality.lesion_texture() * falloff;
                }
            }

            ground_truth[idx] = in_lesion;

            // Imaging noise
            val += lcg_normal(&mut rng) * noise_sigma;
            intensity[idx] = val.max(0.0);
        }
    }

    TissueImage { width: w, height: h, modality, intensity, ground_truth, lesions }
}

// ---------------------------------------------------------------------------
// Per-voxel feature extraction (32-dim embedding)
// ---------------------------------------------------------------------------

fn extract_features(img: &TissueImage) -> Vec<Vec<f32>> {
    let w = img.width;
    let h = img.height;
    let n = w * h;
    let mut embeddings = Vec::with_capacity(n);

    for y in 0..h {
        for x in 0..w {
            let mut feat = Vec::with_capacity(EMBED_DIM);
            let center_val = img.intensity[y * w + x];

            // Multi-scale neighborhood statistics at windows 3x3, 5x5, 7x7
            for &half in &[1usize, 2, 3] {
                let (mut sum, mut sum2, mut gx_sum, mut gy_sum, mut count): (f64, f64, f64, f64, f64) =
                    (0.0, 0.0, 0.0, 0.0, 0.0);
                let (mut vmin, mut vmax) = (f64::MAX, f64::MIN);

                for dy in -(half as i32)..=(half as i32) {
                    for dx in -(half as i32)..=(half as i32) {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let v = img.intensity[ny as usize * w + nx as usize];
                        sum += v;
                        sum2 += v * v;
                        if v < vmin { vmin = v; }
                        if v > vmax { vmax = v; }
                        count += 1.0;

                        // Gradient magnitude approximation
                        if dx.abs() <= 1 && dy.abs() <= 1 && (dx != 0 || dy != 0) {
                            let diff = (v - center_val).abs();
                            gx_sum += if dx != 0 { diff } else { 0.0 };
                            gy_sum += if dy != 0 { diff } else { 0.0 };
                        }
                    }
                }

                let mean = sum / count.max(1.0);
                let variance = (sum2 / count.max(1.0) - mean * mean).max(0.0);
                let contrast = vmax - vmin;
                let grad_mag = (gx_sum * gx_sum + gy_sum * gy_sum).sqrt();

                // 4 features per scale: mean deviation, std, contrast, gradient
                feat.push((center_val - mean) as f32);
                feat.push(variance.sqrt() as f32);
                feat.push(contrast as f32);
                feat.push(grad_mag as f32);
            }

            // Absolute intensity (normalized)
            feat.push((center_val / 255.0) as f32);

            // Deviation from global baseline
            feat.push((center_val - img.modality.baseline()) as f32);

            // Local anisotropy: ratio of horizontal vs vertical gradient
            let gx = if x > 0 && x < w - 1 {
                (img.intensity[y * w + x + 1] - img.intensity[y * w + x - 1]).abs()
            } else { 0.0 };
            let gy = if y > 0 && y < h - 1 {
                (img.intensity[(y + 1) * w + x] - img.intensity[(y - 1) * w + x]).abs()
            } else { 0.0 };
            feat.push((gx / (gy + 1e-6)) as f32);

            // Local energy (sum of squared differences from mean in 3x3)
            let mut energy = 0.0;
            let mut local_mean = 0.0;
            let mut lc = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    local_mean += img.intensity[ny * w + nx];
                    lc += 1.0;
                }
            }
            local_mean /= lc;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let d = img.intensity[ny * w + nx] - local_mean;
                    energy += d * d;
                }
            }
            feat.push((energy / lc) as f32);

            // Pad / truncate to EMBED_DIM
            while feat.len() < EMBED_DIM { feat.push(0.0); }
            feat.truncate(EMBED_DIM);
            embeddings.push(feat);
        }
    }

    embeddings
}

// ---------------------------------------------------------------------------
// Healthy tissue model (estimate from border voxels)
// ---------------------------------------------------------------------------

struct HealthyModel {
    mean: f64,
    std: f64,
}

fn estimate_healthy_model(img: &TissueImage) -> HealthyModel {
    // Use border region (outer 8 pixels) as healthy reference
    let w = img.width;
    let h = img.height;
    let mut vals = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if x < 8 || x >= w - 8 || y < 8 || y >= h - 8 {
                vals.push(img.intensity[y * w + x]);
            }
        }
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    HealthyModel { mean, std: var.sqrt() }
}

// ---------------------------------------------------------------------------
// Graph cut segmentation (Edmonds-Karp max-flow, same pattern as exomoon)
// ---------------------------------------------------------------------------

fn build_unary_costs(img: &TissueImage, healthy: &HealthyModel) -> Vec<f64> {
    // Lambda_i: how anomalous is voxel i vs healthy tissue model
    // Positive = evidence for lesion, negative = evidence for healthy
    let n = img.width * img.height;
    let mut lambda = vec![0.0; n];
    for i in 0..n {
        let z = (img.intensity[i] - healthy.mean) / healthy.std.max(1e-6);
        // Anomaly score: deviation beyond 1 sigma is suspicious
        lambda[i] = z.abs() - 1.5;
    }
    lambda
}

fn solve_graphcut(
    width: usize,
    height: usize,
    lambda: &[f64],
    intensity: &[f64],
    gamma: f64,
) -> Vec<bool> {
    let m = width * height;
    let s = m;     // source (lesion side)
    let t = m + 1; // sink (healthy side)
    let n = m + 2;

    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();

    let add_edge = |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>,
                    u: usize, v: usize, cap: f64| {
        let idx_uv = caps.len();
        caps.push(cap);
        let idx_vu = caps.len();
        caps.push(0.0);
        adj[u].push((v, idx_uv));
        adj[v].push((u, idx_vu));
    };

    // Unary (source/sink) edges
    for i in 0..m {
        let phi_0 = lambda[i].max(0.0);       // cost of healthy when evidence says lesion
        let phi_1 = (-lambda[i]).max(0.0);     // cost of lesion when evidence says healthy
        if phi_0 > 1e-12 { add_edge(&mut adj, &mut caps, s, i, phi_0); }
        if phi_1 > 1e-12 { add_edge(&mut adj, &mut caps, i, t, phi_1); }
    }

    // Pairwise edges: 4-connected grid weighted by intensity gradient
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let neighbors: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            for &(dx, dy) in &neighbors {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let nidx = ny as usize * width + nx as usize;
                if nidx <= idx { continue; } // add each undirected edge once
                // Weight inversely proportional to gradient (strong edge = low cost)
                let grad = (intensity[idx] - intensity[nidx]).abs();
                let w = gamma * (-grad / 20.0).exp();
                if w > 1e-12 {
                    add_edge(&mut adj, &mut caps, idx, nidx, w);
                    add_edge(&mut adj, &mut caps, nidx, idx, w);
                }
            }
        }
    }

    // Edmonds-Karp BFS max-flow
    loop {
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        visited[s] = true;
        queue.push_back(s);

        while let Some(u) = queue.pop_front() {
            if u == t { break; }
            for &(v, eidx) in &adj[u] {
                if !visited[v] && caps[eidx] > 1e-15 {
                    visited[v] = true;
                    parent[v] = Some((u, eidx));
                    queue.push_back(v);
                }
            }
        }

        if !visited[t] { break; }

        let mut bottleneck = f64::MAX;
        let mut v = t;
        while let Some((u, eidx)) = parent[v] {
            bottleneck = bottleneck.min(caps[eidx]);
            v = u;
        }

        v = t;
        while let Some((u, eidx)) = parent[v] {
            caps[eidx] -= bottleneck;
            caps[eidx ^ 1] += bottleneck;
            v = u;
        }
    }

    // Min cut: reachable from source in residual = lesion
    let mut reachable = vec![false; n];
    let mut stack = vec![s];
    reachable[s] = true;
    while let Some(u) = stack.pop() {
        for &(v, eidx) in &adj[u] {
            if !reachable[v] && caps[eidx] > 1e-15 {
                reachable[v] = true;
                stack.push(v);
            }
        }
    }

    (0..m).map(|i| reachable[i]).collect()
}

// ---------------------------------------------------------------------------
// Simple thresholding baseline
// ---------------------------------------------------------------------------

fn threshold_segment(img: &TissueImage, healthy: &HealthyModel, threshold_sigma: f64) -> Vec<bool> {
    img.intensity.iter()
        .map(|&v| ((v - healthy.mean).abs() / healthy.std.max(1e-6)) > threshold_sigma)
        .collect()
}

// ---------------------------------------------------------------------------
// Evaluation metrics
// ---------------------------------------------------------------------------

struct Metrics {
    dice: f64,
    jaccard: f64,
    sensitivity: f64,
    specificity: f64,
}

fn compute_metrics(pred: &[bool], truth: &[bool]) -> Metrics {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    for (p, t) in pred.iter().zip(truth.iter()) {
        match (*p, *t) {
            (true, true)   => tp += 1,
            (true, false)  => fp += 1,
            (false, false) => tn += 1,
            (false, true)  => fn_ += 1,
        }
    }
    let dice = if tp + fp + fn_ > 0 {
        2.0 * tp as f64 / (2 * tp + fp + fn_) as f64
    } else { 1.0 };
    let jaccard = if tp + fp + fn_ > 0 {
        tp as f64 / (tp + fp + fn_) as f64
    } else { 1.0 };
    let sensitivity = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 1.0 };
    let specificity = if tn + fp > 0 { tn as f64 / (tn + fp) as f64 } else { 1.0 };
    Metrics { dice, jaccard, sensitivity, specificity }
}

/// Per-lesion detection rate: a lesion is "detected" if >50% of its voxels are labeled
fn per_lesion_detection(img: &TissueImage, pred: &[bool]) -> (usize, usize) {
    let w = img.width;
    let mut detected = 0;
    for les in &img.lesions {
        let mut total = 0;
        let mut hit = 0;
        for y in 0..img.height {
            for x in 0..w {
                let dx = x as f64 - les.cx;
                let dy = y as f64 - les.cy;
                if (dx * dx + dy * dy).sqrt() <= les.radius {
                    total += 1;
                    if pred[y * w + x] { hit += 1; }
                }
            }
        }
        if total > 0 && hit as f64 / total as f64 > 0.5 { detected += 1; }
    }
    (detected, img.lesions.len())
}

// ---------------------------------------------------------------------------
// Cosine similarity (for RuVector prior)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-15 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Medical Imaging Lesion Detection via Graph Cut ===\n");

    let modalities = [Modality::T1Mri, Modality::T2Mri, Modality::Ct];
    let gamma = 1.5;           // pairwise coherence weight
    let threshold_sigma = 2.0; // baseline thresholding

    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let store_path = tmp_dir.path().join("medical_graphcut.rvvis");

    let options = RvfOptions {
        dimension: EMBED_DIM as u16,
        metric: DistanceMetric::L2,
        ..Default::default()
    };
    let mut store = RvfStore::create(&store_path, options).expect("failed to create store");

    println!("  Grid size:      {}x{} ({} voxels)", GRID_W, GRID_H, GRID_W * GRID_H);
    println!("  Embedding dim:  {}", EMBED_DIM);
    println!("  Gamma:          {}", gamma);
    println!("  Modalities:     T1-MRI, T2-MRI, CT\n");

    // ====================================================================
    // Per-modality pipeline
    // ====================================================================
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_metadata: Vec<MetadataEntry> = Vec::new();
    let mut next_id: u64 = 0;

    struct ModalityResult {
        label: &'static str,
        num_lesions: usize,
        gc_metrics: Metrics,
        th_metrics: Metrics,
        gc_lesion_det: (usize, usize),
        th_lesion_det: (usize, usize),
    }

    let mut mod_results: Vec<ModalityResult> = Vec::new();

    for (mi, &modality) in modalities.iter().enumerate() {
        let num_lesions = 3 + mi; // 3, 4, 5 lesions per modality
        let seed = 42 + mi as u64 * 1000;

        println!("--- {} ({} lesions) ---", modality.label(), num_lesions);

        // Step 1: Generate synthetic tissue
        let img = generate_tissue(modality, num_lesions, seed);
        let gt_count = img.ground_truth.iter().filter(|&&v| v).count();
        println!("  Ground truth lesion voxels: {} ({:.1}%)",
            gt_count, gt_count as f64 / (GRID_W * GRID_H) as f64 * 100.0);

        // Step 2: Extract features
        let embeddings = extract_features(&img);

        // Step 3: Healthy tissue model
        let healthy = estimate_healthy_model(&img);
        println!("  Healthy model: mean={:.1}, std={:.1}", healthy.mean, healthy.std);

        // Step 4a: Graph cut segmentation
        let lambda = build_unary_costs(&img, &healthy);
        let gc_seg = solve_graphcut(img.width, img.height, &lambda, &img.intensity, gamma);
        let gc_m = compute_metrics(&gc_seg, &img.ground_truth);
        let gc_ld = per_lesion_detection(&img, &gc_seg);

        // Step 4b: Simple thresholding baseline
        let th_seg = threshold_segment(&img, &healthy, threshold_sigma);
        let th_m = compute_metrics(&th_seg, &img.ground_truth);
        let th_ld = per_lesion_detection(&img, &th_seg);

        println!("  Graph Cut  -> Dice={:.3}, Jaccard={:.3}, Sens={:.3}, Spec={:.3}, Lesions={}/{}",
            gc_m.dice, gc_m.jaccard, gc_m.sensitivity, gc_m.specificity, gc_ld.0, gc_ld.1);
        println!("  Threshold  -> Dice={:.3}, Jaccard={:.3}, Sens={:.3}, Spec={:.3}, Lesions={}/{}",
            th_m.dice, th_m.jaccard, th_m.sensitivity, th_m.specificity, th_ld.0, th_ld.1);

        // Step 5: Store embeddings in RVF
        for (i, emb) in embeddings.iter().enumerate() {
            let id = next_id;
            next_id += 1;
            let x = i % img.width;
            let y = i / img.width;

            all_vectors.push(emb.clone());
            all_ids.push(id);

            all_metadata.push(MetadataEntry {
                field_id: FIELD_MODALITY,
                value: MetadataValue::String(modality.label().to_string()),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_LESION_COUNT,
                value: MetadataValue::U64(num_lesions as u64),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_GRID_X,
                value: MetadataValue::U64(x as u64),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_GRID_Y,
                value: MetadataValue::U64(y as u64),
            });
        }

        mod_results.push(ModalityResult {
            label: modality.label(),
            num_lesions,
            gc_metrics: gc_m,
            th_metrics: th_m,
            gc_lesion_det: gc_ld,
            th_lesion_det: th_ld,
        });

        println!();
    }

    // ====================================================================
    // Ingest all voxel embeddings into RVF store
    // ====================================================================
    println!("--- RVF Ingest ---");
    let vec_refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
    let ingest = store
        .ingest_batch(&vec_refs, &all_ids, Some(&all_metadata))
        .expect("ingest failed");
    println!("  Ingested: {} voxel embeddings (rejected: {})", ingest.accepted, ingest.rejected);

    // ====================================================================
    // RVF filtered queries by modality
    // ====================================================================
    println!("\n--- RVF Filtered Queries ---");

    // Use a known lesion voxel embedding as query (center of first image's first lesion)
    let query_vec = all_vectors[0].clone(); // first voxel as proxy

    for mod_label in &["T1-MRI", "T2-MRI", "CT"] {
        let filter = FilterExpr::Eq(FIELD_MODALITY, FilterValue::String(mod_label.to_string()));
        let opts = QueryOptions { filter: Some(filter), ..Default::default() };
        let results = store.query(&query_vec, 5, &opts).expect("query failed");
        println!("  {} -> {} nearest voxels found", mod_label, results.len());
        for r in &results {
            println!("    id={:>5}  dist={:.4}", r.id, r.distance);
        }
    }

    // ====================================================================
    // Similarity search: find voxels similar to known lesion pattern
    // ====================================================================
    println!("\n--- Lesion Pattern Similarity Search ---");
    // Build a synthetic lesion prototype: average embedding of ground-truth lesion voxels
    // from the first image (T1-MRI)
    let first_n = GRID_W * GRID_H;
    let first_img = generate_tissue(Modality::T1Mri, 3, 42);
    let first_embs = extract_features(&first_img);
    let mut prototype = vec![0.0f32; EMBED_DIM];
    let mut proto_count = 0;
    for (i, gt) in first_img.ground_truth.iter().enumerate() {
        if *gt {
            for (j, &v) in first_embs[i].iter().enumerate() {
                prototype[j] += v;
            }
            proto_count += 1;
        }
    }
    if proto_count > 0 {
        for v in &mut prototype { *v /= proto_count as f32; }
    }

    let sim_results = store.query(&prototype, 10, &QueryOptions::default()).expect("query failed");
    println!("  Lesion prototype query -> top 10 similar voxels:");
    println!("    {:>6}  {:>10}  {:>8}", "ID", "Distance", "Modality");
    println!("    {:->6}  {:->10}  {:->8}", "", "", "");
    for r in &sim_results {
        // Determine modality from ID
        let mod_idx = (r.id as usize) / first_n;
        let mod_label = match mod_idx {
            0 => "T1-MRI", 1 => "T2-MRI", 2 => "CT", _ => "??",
        };
        println!("    {:>6}  {:>10.4}  {:>8}", r.id, r.distance, mod_label);
    }

    // ====================================================================
    // Lineage derivation
    // ====================================================================
    println!("\n--- Lineage: Derive Segmentation Snapshot ---");
    let child_path = tmp_dir.path().join("lesion_segmentation.rvvis");
    let child_store = store
        .derive(&child_path, DerivationType::Filter, None)
        .expect("failed to derive");
    println!("  Parent file_id:  {}", hex_string(store.file_id()));
    println!("  Child parent_id: {}", hex_string(child_store.parent_id()));
    println!("  Lineage depth:   {}", child_store.lineage_depth());
    child_store.close().expect("close failed");

    // ====================================================================
    // Witness chain for diagnostic provenance
    // ====================================================================
    println!("\n--- Witness Chain: Diagnostic Provenance ---");

    let chain_steps = [
        ("image_acquisition", 0x01u8),
        ("tissue_generation", 0x08),
        ("lesion_injection", 0x08),
        ("noise_simulation", 0x02),
        ("feature_extraction", 0x02),
        ("healthy_model_estimation", 0x02),
        ("unary_cost_computation", 0x02),
        ("graph_construction", 0x02),
        ("mincut_solve", 0x02),
        ("threshold_baseline", 0x02),
        ("metric_evaluation", 0x02),
        ("rvf_ingest", 0x08),
        ("similarity_search", 0x02),
        ("lineage_derive", 0x01),
        ("diagnostic_seal", 0x01),
    ];

    let entries: Vec<WitnessEntry> = chain_steps.iter().enumerate()
        .map(|(i, (step, wtype))| {
            let action_data = format!("medical_graphcut:{}:step_{}", step, i);
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: shake256_256(action_data.as_bytes()),
                timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
                witness_type: *wtype,
            }
        })
        .collect();

    let chain_bytes = create_witness_chain(&entries);
    let verified = verify_witness_chain(&chain_bytes).expect("verification failed");

    println!("  Chain entries:  {}", verified.len());
    println!("  Chain size:     {} bytes", chain_bytes.len());
    println!("  Integrity:      VALID");

    println!("\n  Pipeline steps:");
    for (i, (step, _)) in chain_steps.iter().enumerate() {
        let wtype_name = match verified[i].witness_type {
            0x01 => "PROV", 0x02 => "COMP", 0x08 => "DATA", _ => "????",
        };
        println!("    [{:>4}] {:>2} -> {}", wtype_name, i, step);
    }

    // ====================================================================
    // Summary table
    // ====================================================================
    println!("\n=== Segmentation Results Summary ===\n");
    println!("  {:>8}  {:>7}  {:>6}  {:>7}  {:>6}  {:>6}  {:>6}  {:>9}  {:>9}",
        "Modality", "Lesions", "Method", "Dice", "Jacc", "Sens", "Spec", "LesionDet", "Improve");
    println!("  {:->8}  {:->7}  {:->6}  {:->7}  {:->6}  {:->6}  {:->6}  {:->9}  {:->9}",
        "", "", "", "", "", "", "", "", "");

    for r in &mod_results {
        let gc_det_str = format!("{}/{}", r.gc_lesion_det.0, r.gc_lesion_det.1);
        let th_det_str = format!("{}/{}", r.th_lesion_det.0, r.th_lesion_det.1);
        let dice_improve = if r.th_metrics.dice > 0.0 {
            (r.gc_metrics.dice - r.th_metrics.dice) / r.th_metrics.dice * 100.0
        } else { 0.0 };

        println!("  {:>8}  {:>7}  {:>6}  {:>7.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>9}  {:>+8.1}%",
            r.label, r.num_lesions, "GC", r.gc_metrics.dice, r.gc_metrics.jaccard,
            r.gc_metrics.sensitivity, r.gc_metrics.specificity, gc_det_str, dice_improve);
        println!("  {:>8}  {:>7}  {:>6}  {:>7.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>9}  {:>9}",
            "", "", "Thresh", r.th_metrics.dice, r.th_metrics.jaccard,
            r.th_metrics.sensitivity, r.th_metrics.specificity, th_det_str, "baseline");
    }

    println!("\n  RVF store:       {} voxel embeddings", ingest.accepted);
    println!("  Witness chain:   {} provenance entries", verified.len());
    println!("  Modalities:      {}", modalities.len());

    store.close().expect("close failed");
    println!("\nDone.");
}

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
