//! Medical Imaging Lesion Detection via Graph Cut + RuVector
//!
//! Adapts the MRF/mincut formulation from the exomoon pipeline to medical
//! imaging lesion segmentation on synthetic 2D tissue grids:
//!
//!   1. Generate synthetic tissue with injected lesions (tumors)
//!   2. Extract per-voxel features: intensity, texture, multi-scale (32-dim)
//!   3. Build 4-connected grid graph, solve s-t mincut (Edmonds-Karp)
//!   4. Store voxel embeddings in RVF with modality metadata
//!   5. Evaluate: Dice, Jaccard, sensitivity, specificity vs thresholding
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

const GRID: usize = 64;
const DIM: usize = 32;
const FIELD_MODALITY: u16 = 0;
const FIELD_LESION_COUNT: u16 = 1;
const FIELD_GRID_SIZE: u16 = 2;

fn lcg_next(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}
fn lcg_f64(s: &mut u64) -> f64 { lcg_next(s); (*s >> 11) as f64 / ((1u64 << 53) as f64) }
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_f64(s).max(1e-15);
    let u2 = lcg_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Debug, Clone, Copy)]
enum Modality { T1Mri, T2Mri, Ct }

impl Modality {
    fn label(&self) -> &'static str {
        match self { Self::T1Mri => "T1-MRI", Self::T2Mri => "T2-MRI", Self::Ct => "CT" }
    }
    fn baseline(&self) -> f64 {
        match self { Self::T1Mri => 120.0, Self::T2Mri => 80.0, Self::Ct => 40.0 }
    }
    fn lesion_offset(&self) -> f64 {
        match self { Self::T1Mri => -30.0, Self::T2Mri => 60.0, Self::Ct => 35.0 }
    }
    fn noise_sigma(&self) -> f64 {
        match self { Self::T1Mri => 12.0, Self::T2Mri => 15.0, Self::Ct => 8.0 }
    }
    fn lesion_texture(&self) -> f64 {
        match self { Self::T1Mri => 2.5, Self::T2Mri => 3.0, Self::Ct => 1.8 }
    }
}

#[derive(Debug, Clone)]
struct Lesion { cx: f64, cy: f64, radius: f64 }

struct TissueImage {
    modality: Modality,
    intensity: Vec<f64>,
    ground_truth: Vec<bool>,
    lesions: Vec<Lesion>,
}

fn generate_tissue(modality: Modality, num_lesions: usize, seed: u64) -> TissueImage {
    let n = GRID * GRID;
    let mut rng = seed;
    let (baseline, sigma) = (modality.baseline(), modality.noise_sigma());
    let mut intensity = vec![0.0; n];
    let mut ground_truth = vec![false; n];

    let mut lesions = Vec::with_capacity(num_lesions);
    for _ in 0..num_lesions {
        lesions.push(Lesion {
            cx: 10.0 + lcg_f64(&mut rng) * (GRID as f64 - 20.0),
            cy: 10.0 + lcg_f64(&mut rng) * (GRID as f64 - 20.0),
            radius: 3.0 + lcg_f64(&mut rng) * 6.0,
        });
    }

    for y in 0..GRID {
        for x in 0..GRID {
            let idx = y * GRID + x;
            let mut val = baseline + 5.0 * ((x as f64 * 0.1).sin() + (y as f64 * 0.08).cos());
            let mut in_lesion = false;
            for les in &lesions {
                let dist = ((x as f64 - les.cx).powi(2) + (y as f64 - les.cy).powi(2)).sqrt();
                if dist <= les.radius {
                    in_lesion = true;
                    let falloff = 1.0 - (dist / les.radius).powi(2);
                    val += modality.lesion_offset() * falloff;
                    val += lcg_normal(&mut rng) * sigma * modality.lesion_texture() * falloff;
                }
            }
            ground_truth[idx] = in_lesion;
            intensity[idx] = (val + lcg_normal(&mut rng) * sigma).max(0.0);
        }
    }
    TissueImage { modality, intensity, ground_truth, lesions }
}

/// Extract 32-dim features: multi-scale (3x3,5x5,7x7) stats + intensity + texture
fn extract_features(img: &TissueImage) -> Vec<Vec<f32>> {
    let n = GRID * GRID;
    let mut embeddings = Vec::with_capacity(n);
    for y in 0..GRID {
        for x in 0..GRID {
            let mut f = Vec::with_capacity(DIM);
            let cv = img.intensity[y * GRID + x];

            for &half in &[1usize, 2, 3] {
                let (mut sum, mut sum2, mut gx, mut gy, mut cnt): (f64,f64,f64,f64,f64) =
                    (0.0, 0.0, 0.0, 0.0, 0.0);
                let (mut vmin, mut vmax) = (f64::MAX, f64::MIN);
                for dy in -(half as i32)..=(half as i32) {
                    for dx in -(half as i32)..=(half as i32) {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx < 0 || ny < 0 || nx >= GRID as i32 || ny >= GRID as i32 { continue; }
                        let v = img.intensity[ny as usize * GRID + nx as usize];
                        sum += v; sum2 += v * v; cnt += 1.0;
                        if v < vmin { vmin = v; }
                        if v > vmax { vmax = v; }
                        if dx.abs() <= 1 && dy.abs() <= 1 && (dx|dy) != 0 {
                            let d = (v - cv).abs();
                            if dx != 0 { gx += d; }
                            if dy != 0 { gy += d; }
                        }
                    }
                }
                let mean = sum / cnt.max(1.0);
                let var = (sum2 / cnt.max(1.0) - mean * mean).max(0.0);
                f.push((cv - mean) as f32);
                f.push(var.sqrt() as f32);
                f.push((vmax - vmin) as f32);
                f.push((gx * gx + gy * gy).sqrt() as f32);
            }
            // 12 features so far. Add 4 more scalar features.
            f.push((cv / 255.0) as f32);
            f.push((cv - img.modality.baseline()) as f32);
            let hg = if x > 0 && x < GRID-1 {
                (img.intensity[y*GRID+x+1] - img.intensity[y*GRID+x-1]).abs()
            } else { 0.0 };
            let vg = if y > 0 && y < GRID-1 {
                (img.intensity[(y+1)*GRID+x] - img.intensity[(y-1)*GRID+x]).abs()
            } else { 0.0 };
            f.push((hg / (vg + 1e-6)) as f32);
            // Local energy in 3x3
            let mut energy = 0.0;
            let mut lm = 0.0;
            let mut lc = 0.0f64;
            for dy in -1i32..=1 { for dx in -1i32..=1 {
                let v = img.intensity[
                    (y as i32+dy).clamp(0,GRID as i32-1) as usize * GRID +
                    (x as i32+dx).clamp(0,GRID as i32-1) as usize];
                lm += v; lc += 1.0;
            }}
            lm /= lc;
            for dy in -1i32..=1 { for dx in -1i32..=1 {
                let v = img.intensity[
                    (y as i32+dy).clamp(0,GRID as i32-1) as usize * GRID +
                    (x as i32+dx).clamp(0,GRID as i32-1) as usize];
                energy += (v - lm).powi(2);
            }}
            f.push((energy / lc) as f32);
            while f.len() < DIM { f.push(0.0); }
            f.truncate(DIM);
            embeddings.push(f);
        }
    }
    embeddings
}

struct HealthyModel { mean: f64, std: f64 }

fn estimate_healthy(img: &TissueImage) -> HealthyModel {
    let vals: Vec<f64> = (0..GRID*GRID).filter(|&i| {
        let (x, y) = (i % GRID, i / GRID);
        x < 8 || x >= GRID-8 || y < 8 || y >= GRID-8
    }).map(|i| img.intensity[i]).collect();
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    HealthyModel { mean, std: var.sqrt() }
}

/// Edmonds-Karp s-t mincut on 4-connected grid (same solver as exomoon_graphcut)
fn solve_graphcut(lambda: &[f64], intensity: &[f64], gamma: f64) -> Vec<bool> {
    let m = GRID * GRID;
    let (s, t, n) = (m, m + 1, m + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();

    let add_edge = |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>,
                    u: usize, v: usize, cap: f64| {
        let i = caps.len(); caps.push(cap); caps.push(0.0);
        adj[u].push((v, i)); adj[v].push((u, i + 1));
    };

    for i in 0..m {
        let (p0, p1) = (lambda[i].max(0.0), (-lambda[i]).max(0.0));
        if p0 > 1e-12 { add_edge(&mut adj, &mut caps, s, i, p0); }
        if p1 > 1e-12 { add_edge(&mut adj, &mut caps, i, t, p1); }
    }
    for y in 0..GRID { for x in 0..GRID {
        let idx = y * GRID + x;
        for &(dx, dy) in &[(1i32,0i32),(0,1)] {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if nx >= GRID as i32 || ny >= GRID as i32 { continue; }
            let nidx = ny as usize * GRID + nx as usize;
            let w = gamma * (-(intensity[idx] - intensity[nidx]).abs() / 20.0).exp();
            if w > 1e-12 {
                add_edge(&mut adj, &mut caps, idx, nidx, w);
                add_edge(&mut adj, &mut caps, nidx, idx, w);
            }
        }
    }}
    // Max-flow via BFS augmenting paths
    loop {
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        vis[s] = true; q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t { break; }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 { vis[v] = true; parent[v] = Some((u, ei)); q.push_back(v); }
            }
        }
        if !vis[t] { break; }
        let mut bn = f64::MAX;
        let mut v = t;
        while let Some((u, ei)) = parent[v] { bn = bn.min(caps[ei]); v = u; }
        v = t;
        while let Some((u, ei)) = parent[v] { caps[ei] -= bn; caps[ei ^ 1] += bn; v = u; }
    }
    let mut reach = vec![false; n];
    let mut stk = vec![s]; reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] { if !reach[v] && caps[ei] > 1e-15 { reach[v] = true; stk.push(v); } }
    }
    (0..m).map(|i| reach[i]).collect()
}

fn threshold_segment(img: &TissueImage, h: &HealthyModel, thr: f64) -> Vec<bool> {
    img.intensity.iter().map(|&v| ((v - h.mean).abs() / h.std.max(1e-6)) > thr).collect()
}

struct Metrics { dice: f64, jaccard: f64, sensitivity: f64, specificity: f64 }

fn compute_metrics(pred: &[bool], truth: &[bool]) -> Metrics {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0, 0, 0);
    for (&p, &t) in pred.iter().zip(truth) {
        match (p, t) { (true,true)=>tp+=1, (true,false)=>fp+=1, (false,false)=>tn+=1, (false,true)=>fn_+=1 }
    }
    Metrics {
        dice: if tp+fp+fn_>0 { 2.0*tp as f64/(2*tp+fp+fn_) as f64 } else { 1.0 },
        jaccard: if tp+fp+fn_>0 { tp as f64/(tp+fp+fn_) as f64 } else { 1.0 },
        sensitivity: if tp+fn_>0 { tp as f64/(tp+fn_) as f64 } else { 1.0 },
        specificity: if tn+fp>0 { tn as f64/(tn+fp) as f64 } else { 1.0 },
    }
}

fn per_lesion_detection(img: &TissueImage, pred: &[bool]) -> (usize, usize) {
    let mut detected = 0;
    for les in &img.lesions {
        let (mut total, mut hit) = (0, 0);
        for y in 0..GRID { for x in 0..GRID {
            if ((x as f64-les.cx).powi(2)+(y as f64-les.cy).powi(2)).sqrt() <= les.radius {
                total += 1;
                if pred[y*GRID+x] { hit += 1; }
            }
        }}
        if total > 0 && hit as f64 / total as f64 > 0.5 { detected += 1; }
    }
    (detected, img.lesions.len())
}

fn hex_string(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

fn main() {
    println!("=== Medical Imaging Lesion Detection via Graph Cut ===\n");
    let modalities = [Modality::T1Mri, Modality::T2Mri, Modality::Ct];
    let gamma = 1.5;
    let thr_sigma = 2.0;

    let tmp = TempDir::new().expect("tmpdir");
    let opts = RvfOptions { dimension: DIM as u16, metric: DistanceMetric::L2, ..Default::default() };
    let mut store = RvfStore::create(&tmp.path().join("med_gc.rvvis"), opts).expect("create");

    println!("  Grid: {}x{} | Dim: {} | Gamma: {} | Modalities: 3\n", GRID, GRID, DIM, gamma);

    let mut all_vecs: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_meta: Vec<MetadataEntry> = Vec::new();
    let mut next_id: u64 = 0;

    struct MR { l: &'static str, nl: usize, gc: Metrics, th: Metrics, gd: (usize,usize), td: (usize,usize) }
    let mut results: Vec<MR> = Vec::new();

    for (mi, &modality) in modalities.iter().enumerate() {
        let nl = 3 + mi;
        let img = generate_tissue(modality, nl, 42 + mi as u64 * 1000);
        let gt_n = img.ground_truth.iter().filter(|&&v| v).count();
        println!("--- {} ({} lesions, {} GT voxels = {:.1}%) ---",
            modality.label(), nl, gt_n, gt_n as f64 / (GRID*GRID) as f64 * 100.0);

        let embs = extract_features(&img);
        let healthy = estimate_healthy(&img);
        println!("  Healthy model: mean={:.1}, std={:.1}", healthy.mean, healthy.std);

        // Two-signal lambda: global z-score + local contrast
        let lambda: Vec<f64> = (0..GRID*GRID).map(|idx| {
            let (x, y) = (idx % GRID, idx / GRID);
            let v = img.intensity[idx];
            // Global z-score: how far from healthy tissue mean
            let global_z = (v - healthy.mean).abs() / healthy.std.max(1e-6);
            // Local contrast: gradient magnitude in 3x3 neighborhood
            let mut grad_sum = 0.0f64;
            let mut cnt = 0.0f64;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && nx < GRID as i32 && ny < GRID as i32 {
                        let nv = img.intensity[ny as usize * GRID + nx as usize];
                        grad_sum += (v - nv).abs();
                        cnt += 1.0;
                    }
                }
            }
            let local_contrast = grad_sum / cnt.max(1.0) / healthy.std.max(1e-6);
            // Global z-score dominates; local contrast boosts edges
            global_z + 0.3 * local_contrast - 1.5
        }).collect();
        let gc_seg = solve_graphcut(&lambda, &img.intensity, gamma);
        let gc_m = compute_metrics(&gc_seg, &img.ground_truth);
        let gc_ld = per_lesion_detection(&img, &gc_seg);

        let th_seg = threshold_segment(&img, &healthy, thr_sigma);
        let th_m = compute_metrics(&th_seg, &img.ground_truth);
        let th_ld = per_lesion_detection(&img, &th_seg);

        println!("  GC     Dice={:.3} Jacc={:.3} Sens={:.3} Spec={:.3} Lesions={}/{}",
            gc_m.dice, gc_m.jaccard, gc_m.sensitivity, gc_m.specificity, gc_ld.0, gc_ld.1);
        println!("  Thresh Dice={:.3} Jacc={:.3} Sens={:.3} Spec={:.3} Lesions={}/{}\n",
            th_m.dice, th_m.jaccard, th_m.sensitivity, th_m.specificity, th_ld.0, th_ld.1);

        for emb in embs.iter() {
            all_vecs.push(emb.clone());
            all_ids.push(next_id); next_id += 1;
            all_meta.push(MetadataEntry { field_id: FIELD_MODALITY,
                value: MetadataValue::String(modality.label().into()) });
            all_meta.push(MetadataEntry { field_id: FIELD_LESION_COUNT,
                value: MetadataValue::U64(nl as u64) });
        }
        results.push(MR { l: modality.label(), nl, gc: gc_m, th: th_m, gd: gc_ld, td: th_ld });
    }

    // Ingest into RVF
    println!("--- RVF Ingest ---");
    let vrefs: Vec<&[f32]> = all_vecs.iter().map(|v| v.as_slice()).collect();
    let ing = store.ingest_batch(&vrefs, &all_ids, Some(&all_meta)).expect("ingest");
    println!("  Ingested: {} embeddings (rejected: {})", ing.accepted, ing.rejected);

    // Filtered queries by modality
    println!("\n--- Filtered Queries ---");
    let qv = all_vecs[0].clone();
    for lab in &["T1-MRI", "T2-MRI", "CT"] {
        let f = FilterExpr::Eq(FIELD_MODALITY, FilterValue::String(lab.to_string()));
        let r = store.query(&qv, 5, &QueryOptions { filter: Some(f), ..Default::default() }).expect("q");
        println!("  {} -> {} results (top dist: {:.2})", lab, r.len(),
            r.first().map(|x| x.distance).unwrap_or(0.0));
    }

    // Lesion prototype similarity search
    println!("\n--- Lesion Prototype Similarity ---");
    let ref_img = generate_tissue(Modality::T1Mri, 3, 42);
    let ref_embs = extract_features(&ref_img);
    let mut proto = vec![0.0f32; DIM];
    let mut pc = 0usize;
    for (i, &gt) in ref_img.ground_truth.iter().enumerate() {
        if gt { for (j, &v) in ref_embs[i].iter().enumerate() { proto[j] += v; } pc += 1; }
    }
    if pc > 0 { for v in &mut proto { *v /= pc as f32; } }
    let sr = store.query(&proto, 10, &QueryOptions::default()).expect("q");
    println!("  {:>6}  {:>10}  {:>8}", "ID", "Distance", "Modality");
    println!("  {:->6}  {:->10}  {:->8}", "", "", "");
    for r in &sr {
        let ml = match (r.id as usize) / (GRID*GRID) { 0=>"T1-MRI", 1=>"T2-MRI", 2=>"CT", _=>"??" };
        println!("  {:>6}  {:>10.2}  {:>8}", r.id, r.distance, ml);
    }

    // Lineage
    println!("\n--- Lineage ---");
    let child = store.derive(&tmp.path().join("seg.rvvis"), DerivationType::Filter, None).expect("derive");
    println!("  Parent: {}  Child: {}  Depth: {}",
        hex_string(store.file_id()), hex_string(child.parent_id()), child.lineage_depth());
    child.close().expect("close");

    // Witness chain
    println!("\n--- Witness Chain ---");
    let steps = [
        ("acquisition",0x01u8), ("generation",0x08), ("lesion_inject",0x08),
        ("noise_sim",0x02), ("feature_extract",0x02), ("healthy_est",0x02),
        ("unary_cost",0x02), ("graph_build",0x02), ("mincut_solve",0x02),
        ("threshold",0x02), ("evaluation",0x02), ("rvf_ingest",0x08),
        ("similarity",0x02), ("lineage",0x01), ("seal",0x01),
    ];
    let entries: Vec<WitnessEntry> = steps.iter().enumerate().map(|(i, (step, wt))| WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(format!("med_gc:{}:{}", step, i).as_bytes()),
        timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
        witness_type: *wt,
    }).collect();
    let chain = create_witness_chain(&entries);
    let verified = verify_witness_chain(&chain).expect("verify");
    println!("  {} entries, {} bytes, VALID", verified.len(), chain.len());
    for (i, (step, _)) in steps.iter().enumerate() {
        let wn = match verified[i].witness_type { 0x01=>"PROV", 0x02=>"COMP", 0x08=>"DATA", _=>"????" };
        println!("    [{:>4}] {:>2} -> {}", wn, i, step);
    }

    // Summary
    println!("\n=== Summary ===\n");
    println!("  {:>8} {:>3} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>9}",
        "Modal", "#L", "Method", "Dice", "Jacc", "Sens", "Spec", "LDet", "vs Base");
    println!("  {:->8} {:->3} {:->6} {:->6} {:->6} {:->6} {:->6} {:->7} {:->9}",
        "","","","","","","","","");
    for r in &results {
        let imp = if r.th.dice > 0.0 { (r.gc.dice - r.th.dice)/r.th.dice*100.0 } else { 0.0 };
        println!("  {:>8} {:>3} {:>6} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>3}/{:<3} {:>+7.1}%",
            r.l, r.nl, "GC", r.gc.dice, r.gc.jaccard, r.gc.sensitivity, r.gc.specificity, r.gd.0, r.gd.1, imp);
        println!("  {:>8} {:>3} {:>6} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>3}/{:<3} {:>9}",
            "", "", "Thr", r.th.dice, r.th.jaccard, r.th.sensitivity, r.th.specificity, r.td.0, r.td.1, "baseline");
    }
    println!("\n  RVF: {} embeddings | Witness: {} entries | Modalities: {}", ing.accepted, verified.len(), 3);
    store.close().expect("close");
    println!("\nDone.");
}
