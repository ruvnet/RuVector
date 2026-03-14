//! Genomic Graph Cut: CNV Detection via MRF Optimization + RuVector
//!
//! Uses graph cut / MRF optimization for DNA/genomic sequence anomaly detection:
//!
//!   1. Generate synthetic chromosome as N windows (10kb each)
//!   2. Inject anomalies: CNV gains/losses, mutation hotspots, structural variants
//!   3. Extract per-window features into 32-dim embeddings
//!   4. Build MRF graph (linear chain + RuVector kNN edges), solve s-t mincut
//!   5. Classify aberrant regions as gain, loss, or LOH
//!   6. Detect cancer driver genes (TP53, BRCA1, EGFR, MYC)
//!
//! Sequencing platforms: WGS (30x), WES (100x), targeted panel (500x).
//!
//! Run: cargo run --example genomic_graphcut --release

use rvf_runtime::{
    FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore,
};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_types::DerivationType;
use rvf_crypto::{create_witness_chain, verify_witness_chain, shake256_256, WitnessEntry};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Metadata field IDs
// ---------------------------------------------------------------------------

const FIELD_PLATFORM: u16 = 0;
const FIELD_CHROMOSOME: u16 = 1;
const FIELD_WINDOW_POS: u16 = 2;
const FIELD_CNV_LABEL: u16 = 3;

// ---------------------------------------------------------------------------
// LCG deterministic random
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
// Sequencing platforms
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Platform {
    Wgs,     // Whole genome sequencing ~30x
    Wes,     // Whole exome sequencing ~100x
    Panel,   // Targeted panel ~500x
}

impl Platform {
    fn label(&self) -> &'static str {
        match self { Platform::Wgs => "WGS", Platform::Wes => "WES", Platform::Panel => "Panel" }
    }
    fn expected_depth(&self) -> f64 {
        match self { Platform::Wgs => 30.0, Platform::Wes => 100.0, Platform::Panel => 500.0 }
    }
    fn depth_noise_cv(&self) -> f64 {
        match self { Platform::Wgs => 0.15, Platform::Wes => 0.25, Platform::Panel => 0.10 }
    }
}

// ---------------------------------------------------------------------------
// Anomaly types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum AnomalyType {
    Normal,
    CnvGain,       // Copy number gain (e.g., 3-4 copies)
    CnvLoss,       // Copy number loss (e.g., 1 copy or homozygous deletion)
    Loh,           // Loss of heterozygosity (CN-neutral)
    MutationHotspot,
    StructuralVariant,
}

impl AnomalyType {
    fn label(&self) -> &'static str {
        match self {
            AnomalyType::Normal => "normal",
            AnomalyType::CnvGain => "gain",
            AnomalyType::CnvLoss => "loss",
            AnomalyType::Loh => "LOH",
            AnomalyType::MutationHotspot => "hotspot",
            AnomalyType::StructuralVariant => "SV",
        }
    }
}

// ---------------------------------------------------------------------------
// Cancer driver genes with characteristic CNV patterns
// ---------------------------------------------------------------------------

struct DriverGene {
    name: &'static str,
    window_start: usize,
    window_end: usize,
    anomaly: AnomalyType,
}

fn cancer_drivers() -> Vec<DriverGene> {
    vec![
        DriverGene { name: "TP53",  window_start: 170, window_end: 180, anomaly: AnomalyType::CnvLoss },
        DriverGene { name: "BRCA1", window_start: 410, window_end: 425, anomaly: AnomalyType::CnvLoss },
        DriverGene { name: "EGFR",  window_start: 700, window_end: 720, anomaly: AnomalyType::CnvGain },
        DriverGene { name: "MYC",   window_start: 820, window_end: 835, anomaly: AnomalyType::CnvGain },
    ]
}

// ---------------------------------------------------------------------------
// Genomic window data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GenomicWindow {
    index: usize,
    read_depth: f64,
    gc_content: f64,
    mapping_quality: f64,
    variant_count: u32,
    baf: f64,            // B-allele frequency (0.5 = heterozygous normal)
    truth_anomaly: AnomalyType,
    truth_copy_number: f64,
}

// ---------------------------------------------------------------------------
// Synthetic chromosome generator
// ---------------------------------------------------------------------------

fn generate_chromosome(
    n_windows: usize,
    platform: Platform,
    seed: u64,
) -> Vec<GenomicWindow> {
    let mut rng = seed;
    let expected_depth = platform.expected_depth();
    let cv = platform.depth_noise_cv();
    let drivers = cancer_drivers();

    // Background GC content wave (realistic GC variation along chromosome)
    let gc_wave = |i: usize| -> f64 {
        0.42 + 0.08 * (i as f64 * 0.01).sin() + 0.04 * (i as f64 * 0.037).cos()
    };

    let mut windows = Vec::with_capacity(n_windows);

    for i in 0..n_windows {
        let gc = (gc_wave(i) + lcg_normal(&mut rng) * 0.02).clamp(0.25, 0.75);

        // GC bias on depth: lower coverage at extreme GC
        let gc_bias = 1.0 - 2.0 * (gc - 0.45).powi(2);

        // Determine ground truth anomaly
        let mut anomaly = AnomalyType::Normal;
        let mut true_cn = 2.0;

        // Check driver gene regions
        for d in &drivers {
            if i >= d.window_start && i < d.window_end {
                anomaly = d.anomaly;
                true_cn = match d.anomaly {
                    AnomalyType::CnvGain => 4.0,
                    AnomalyType::CnvLoss => 1.0,
                    _ => 2.0,
                };
            }
        }

        // Additional random anomalies
        if anomaly == AnomalyType::Normal {
            let r = lcg_f64(&mut rng);
            if r < 0.02 {
                anomaly = AnomalyType::CnvGain;
                true_cn = 3.0 + lcg_f64(&mut rng);
            } else if r < 0.04 {
                anomaly = AnomalyType::CnvLoss;
                true_cn = 0.5 + lcg_f64(&mut rng) * 0.5;
            } else if r < 0.05 {
                anomaly = AnomalyType::Loh;
                true_cn = 2.0; // CN-neutral LOH
            } else if r < 0.06 {
                anomaly = AnomalyType::MutationHotspot;
                true_cn = 2.0;
            } else if r < 0.065 {
                anomaly = AnomalyType::StructuralVariant;
                true_cn = 2.0;
            }
        }

        // Read depth: proportional to copy number with noise
        let depth_ratio = true_cn / 2.0;
        let raw_depth = expected_depth * depth_ratio * gc_bias;
        let depth = (raw_depth + lcg_normal(&mut rng) * raw_depth * cv).max(1.0);

        // Mapping quality: lower near SVs and repetitive regions
        let mq = match anomaly {
            AnomalyType::StructuralVariant => 30.0 + lcg_normal(&mut rng) * 5.0,
            _ => 55.0 + lcg_normal(&mut rng) * 3.0,
        };

        // Variant count: elevated in hotspots
        let base_var = (2.0 + lcg_f64(&mut rng) * 3.0) as u32;
        let variant_count = match anomaly {
            AnomalyType::MutationHotspot => base_var * 5 + 10,
            AnomalyType::StructuralVariant => base_var + 3,
            _ => base_var,
        };

        // B-allele frequency
        let baf = match anomaly {
            AnomalyType::Loh => 0.05 + lcg_f64(&mut rng) * 0.1, // shifted toward 0 or 1
            AnomalyType::CnvGain => 0.33 + lcg_normal(&mut rng) * 0.05,
            AnomalyType::CnvLoss => 0.0 + lcg_f64(&mut rng) * 0.15,
            _ => 0.45 + lcg_normal(&mut rng) * 0.05,
        };

        windows.push(GenomicWindow {
            index: i,
            read_depth: depth,
            gc_content: gc,
            mapping_quality: mq.clamp(0.0, 60.0),
            variant_count,
            baf: baf.clamp(0.0, 1.0),
            truth_anomaly: anomaly,
            truth_copy_number: true_cn,
        });
    }
    windows
}

// ---------------------------------------------------------------------------
// Feature extraction -> 32-dim embedding
// ---------------------------------------------------------------------------

fn extract_embedding(w: &GenomicWindow, platform: Platform) -> Vec<f32> {
    let expected = platform.expected_depth();
    let mut emb = Vec::with_capacity(32);

    // 1-4: Depth features
    let log2_ratio = (w.read_depth / expected).max(0.01).log2();
    emb.push(log2_ratio as f32);
    emb.push((w.read_depth / expected) as f32);
    emb.push(w.read_depth.sqrt() as f32 / 10.0);
    emb.push((w.read_depth - expected).abs() as f32 / expected as f32);

    // 5-8: BAF features
    let baf_dev = (w.baf - 0.5).abs();
    emb.push(w.baf as f32);
    emb.push(baf_dev as f32);
    emb.push((baf_dev * 4.0).min(1.0) as f32); // normalized deviation
    emb.push(if baf_dev > 0.15 { 1.0 } else { 0.0 }); // LOH indicator

    // 9-12: GC-normalized depth
    let gc_expected = 1.0 - 2.0 * (w.gc_content - 0.45).powi(2);
    let gc_norm_depth = w.read_depth / (expected * gc_expected.max(0.5));
    emb.push(gc_norm_depth as f32);
    emb.push(w.gc_content as f32);
    emb.push((gc_norm_depth - 1.0).abs() as f32);
    emb.push(((gc_norm_depth).log2()).abs() as f32);

    // 13-16: Variant density
    let var_rate = w.variant_count as f64 / 10.0; // per 10kb window
    emb.push(var_rate as f32);
    emb.push((var_rate / 3.0).min(3.0) as f32); // normalized
    emb.push(if w.variant_count > 15 { 1.0 } else { 0.0 }); // hotspot flag
    emb.push(w.mapping_quality as f32 / 60.0);

    // 17-20: Combined features
    emb.push((log2_ratio * baf_dev) as f32);
    emb.push((log2_ratio.abs() + var_rate / 5.0) as f32);
    emb.push(if log2_ratio > 0.3 { 1.0 } else if log2_ratio < -0.3 { -1.0 } else { 0.0 });
    emb.push(if w.mapping_quality < 40.0 { 1.0 } else { 0.0 }); // SV indicator

    // 21-24: Platform-encoded features
    let plat_code = match platform {
        Platform::Wgs => 0.0f32, Platform::Wes => 0.5, Platform::Panel => 1.0,
    };
    emb.push(plat_code);
    emb.push((w.read_depth / 500.0).min(1.0) as f32);
    emb.push((w.index as f32) / 1000.0);
    emb.push(((w.index as f64 * 0.01).sin()) as f32);

    // 25-32: Polynomial features for nonlinear separation
    emb.push((log2_ratio * log2_ratio) as f32);
    emb.push((baf_dev * baf_dev) as f32);
    emb.push((log2_ratio * gc_norm_depth) as f32);
    emb.push((var_rate * baf_dev) as f32);
    emb.push((log2_ratio.abs() * w.mapping_quality / 60.0) as f32);
    emb.push(((gc_norm_depth - 1.0) * log2_ratio) as f32);
    emb.push((w.variant_count as f32).sqrt() / 5.0);

    while emb.len() < 32 { emb.push(0.0); }
    emb.truncate(32);
    emb
}

// ---------------------------------------------------------------------------
// Graph cut for CNV segmentation
// ---------------------------------------------------------------------------

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

/// Compute unary cost: how much does this window deviate from diploid?
fn unary_score(w: &GenomicWindow, platform: Platform) -> f64 {
    let expected = platform.expected_depth();
    let log2r = (w.read_depth / expected).max(0.01).log2();
    let baf_dev = (w.baf - 0.5).abs();
    let var_excess = (w.variant_count as f64 - 4.0).max(0.0) / 4.0;
    let mq_penalty = if w.mapping_quality < 40.0 { 1.0 } else { 0.0 };

    // Combined deviation score: positive = aberrant, threshold at ~0.35
    0.5 * log2r.abs() + 1.5 * baf_dev + 0.3 * var_excess + 0.4 * mq_penalty - 0.35
}

struct Edge { from: usize, to: usize, weight: f64 }

fn build_graph(embeddings: &[Vec<f32>], alpha: f64, beta: f64, k_nn: usize) -> Vec<Edge> {
    let m = embeddings.len();
    let mut edges = Vec::new();

    // Linear chain (genomic adjacency)
    for i in 0..m.saturating_sub(1) {
        edges.push(Edge { from: i, to: i + 1, weight: alpha });
        edges.push(Edge { from: i + 1, to: i, weight: alpha });
    }

    // kNN edges from RuVector similarity
    for i in 0..m {
        let mut sims: Vec<(usize, f64)> = (0..m)
            .filter(|&j| (j as isize - i as isize).unsigned_abs() > 2) // skip immediate neighbors
            .map(|j| (j, cosine_similarity(&embeddings[i], &embeddings[j]).max(0.0)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(j, sim) in sims.iter().take(k_nn) {
            if sim > 0.1 {
                edges.push(Edge { from: i, to: j, weight: beta * sim });
            }
        }
    }
    edges
}

/// Ford-Fulkerson s-t mincut (Edmonds-Karp BFS).
/// Returns labels: true = aberrant (source side), false = normal (sink side).
fn solve_mincut(lambdas: &[f64], edges: &[Edge], gamma: f64) -> Vec<bool> {
    let m = lambdas.len();
    let s = m;
    let t = m + 1;
    let n = m + 2;

    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();

    let add_edge = |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>,
                        u: usize, v: usize, cap: f64| {
        let idx = caps.len();
        caps.push(cap);
        caps.push(0.0);
        adj[u].push((v, idx));
        adj[v].push((u, idx + 1));
    };

    // Unary: s->i for positive lambda (cost of labeling normal), i->t for negative
    for i in 0..m {
        let phi_0 = lambdas[i].max(0.0);
        let phi_1 = (-lambdas[i]).max(0.0);
        if phi_0 > 1e-12 { add_edge(&mut adj, &mut caps, s, i, phi_0); }
        if phi_1 > 1e-12 { add_edge(&mut adj, &mut caps, i, t, phi_1); }
    }

    // Pairwise
    for e in edges {
        let cap = gamma * e.weight;
        if cap > 1e-12 { add_edge(&mut adj, &mut caps, e.from, e.to, cap); }
    }

    // Edmonds-Karp
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
        while let Some((_u, eidx)) = parent[v] {
            bottleneck = bottleneck.min(caps[eidx]);
            v = _u;
        }
        v = t;
        while let Some((_u, eidx)) = parent[v] {
            caps[eidx] -= bottleneck;
            caps[eidx ^ 1] += bottleneck;
            v = _u;
        }
    }

    // Reachability from source = aberrant
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
// Post-cut classification: gain vs loss vs LOH
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum CallType { Normal, Gain, Loss, Loh }

fn classify_aberrant(w: &GenomicWindow, platform: Platform) -> CallType {
    let expected = platform.expected_depth();
    let log2r = (w.read_depth / expected).max(0.01).log2();
    let baf_dev = (w.baf - 0.5).abs();

    if baf_dev > 0.2 && log2r.abs() < 0.2 {
        CallType::Loh
    } else if log2r > 0.25 {
        CallType::Gain
    } else if log2r < -0.25 {
        CallType::Loss
    } else if baf_dev > 0.15 {
        CallType::Loh
    } else {
        CallType::Normal
    }
}

// ---------------------------------------------------------------------------
// Simple thresholding baseline for comparison
// ---------------------------------------------------------------------------

fn threshold_detector(windows: &[GenomicWindow], platform: Platform) -> Vec<bool> {
    windows.iter().map(|w| {
        let expected = platform.expected_depth();
        let log2r = (w.read_depth / expected).max(0.01).log2();
        log2r.abs() > 0.35 || (w.baf - 0.5).abs() > 0.2 || w.variant_count > 15
    }).collect()
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

struct EvalMetrics {
    sensitivity: f64,
    specificity: f64,
    breakpoint_accuracy: f64,
    classification_accuracy: f64,
}

fn evaluate(
    windows: &[GenomicWindow],
    calls: &[bool],
    call_types: &[CallType],
    platform: Platform,
) -> EvalMetrics {
    let mut tp = 0usize; let mut fp = 0usize;
    let mut tn = 0usize; let mut fn_ = 0usize;
    let mut class_correct = 0usize; let mut class_total = 0usize;

    for (i, w) in windows.iter().enumerate() {
        let truth_abn = w.truth_anomaly != AnomalyType::Normal;
        let called = calls[i];
        match (truth_abn, called) {
            (true, true) => { tp += 1; }
            (false, true) => { fp += 1; }
            (true, false) => { fn_ += 1; }
            (false, false) => { tn += 1; }
        }
        if called && truth_abn {
            class_total += 1;
            let truth_class = match w.truth_anomaly {
                AnomalyType::CnvGain => CallType::Gain,
                AnomalyType::CnvLoss => CallType::Loss,
                AnomalyType::Loh => CallType::Loh,
                _ => CallType::Normal,
            };
            if call_types[i] == truth_class { class_correct += 1; }
        }
    }

    let sensitivity = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
    let specificity = if tn + fp > 0 { tn as f64 / (tn + fp) as f64 } else { 0.0 };

    // Breakpoint accuracy: for each true event boundary, find nearest called boundary
    let drivers = cancer_drivers();
    let mut bp_errors = Vec::new();
    for d in &drivers {
        // Find nearest called boundary to true start
        let mut best_dist = usize::MAX;
        for i in 1..calls.len() {
            if calls[i] != calls[i - 1] {
                let dist_s = (i as isize - d.window_start as isize).unsigned_abs();
                let dist_e = (i as isize - d.window_end as isize).unsigned_abs();
                best_dist = best_dist.min(dist_s).min(dist_e);
            }
        }
        bp_errors.push(best_dist as f64);
    }
    let _ = platform; // used above
    let breakpoint_accuracy = if bp_errors.is_empty() { 0.0 }
        else { 1.0 / (1.0 + bp_errors.iter().sum::<f64>() / bp_errors.len() as f64) };

    let classification_accuracy = if class_total > 0 {
        class_correct as f64 / class_total as f64
    } else { 0.0 };

    EvalMetrics { sensitivity, specificity, breakpoint_accuracy, classification_accuracy }
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Genomic Graph Cut: CNV Detection Pipeline ===\n");

    let dim = 32;
    let n_windows = 1000;

    // Hyperparameters
    let alpha = 0.3;   // genomic adjacency edge weight
    let beta = 0.15;   // RuVector kNN edge weight
    let gamma = 0.4;   // coherence penalty
    let k_nn = 3;      // similarity neighbors

    let platforms = [Platform::Wgs, Platform::Wes, Platform::Panel];
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let store_path = tmp_dir.path().join("genomic_graphcut.rvdna");

    let options = RvfOptions {
        dimension: dim as u16,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut store = RvfStore::create(&store_path, options).expect("failed to create store");

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_metadata: Vec<MetadataEntry> = Vec::new();

    let drivers = cancer_drivers();

    for (pi, &platform) in platforms.iter().enumerate() {
        println!("--- Platform: {} (expected depth: {}x) ---\n", platform.label(), platform.expected_depth());

        // Step 1: Generate synthetic chromosome
        let seed = 42 + pi as u64 * 997;
        let windows = generate_chromosome(n_windows, platform, seed);

        let n_aberrant = windows.iter().filter(|w| w.truth_anomaly != AnomalyType::Normal).count();
        println!("  Generated {} windows (10kb each = 10 Mb chromosome)", n_windows);
        println!("  Aberrant windows: {} ({:.1}%)", n_aberrant, n_aberrant as f64 / n_windows as f64 * 100.0);

        // Count by type
        for atype in &[AnomalyType::CnvGain, AnomalyType::CnvLoss, AnomalyType::Loh,
                       AnomalyType::MutationHotspot, AnomalyType::StructuralVariant] {
            let c = windows.iter().filter(|w| w.truth_anomaly == *atype).count();
            if c > 0 { println!("    {}: {} windows", atype.label(), c); }
        }

        // Step 2: Extract embeddings
        let embeddings: Vec<Vec<f32>> = windows.iter()
            .map(|w| extract_embedding(w, platform))
            .collect();

        // Step 3: Compute unary scores
        let lambdas: Vec<f64> = windows.iter()
            .map(|w| unary_score(w, platform))
            .collect();

        // Step 4: Build graph and solve mincut
        let edges = build_graph(&embeddings, alpha, beta, k_nn);
        let gc_calls = solve_mincut(&lambdas, &edges, gamma);

        // Step 5: Classify aberrant regions
        let call_types: Vec<CallType> = windows.iter().enumerate()
            .map(|(i, w)| if gc_calls[i] { classify_aberrant(w, platform) } else { CallType::Normal })
            .collect();

        // Baseline: simple thresholding
        let thresh_calls = threshold_detector(&windows, platform);
        let thresh_types: Vec<CallType> = windows.iter().enumerate()
            .map(|(i, w)| if thresh_calls[i] { classify_aberrant(w, platform) } else { CallType::Normal })
            .collect();

        // Step 6: Evaluate
        let gc_metrics = evaluate(&windows, &gc_calls, &call_types, platform);
        let th_metrics = evaluate(&windows, &thresh_calls, &thresh_types, platform);

        println!("\n  Graph Cut Results:");
        println!("    Sensitivity:      {:.3}", gc_metrics.sensitivity);
        println!("    Specificity:      {:.3}", gc_metrics.specificity);
        println!("    Breakpoint acc:   {:.3}", gc_metrics.breakpoint_accuracy);
        println!("    Classification:   {:.3}", gc_metrics.classification_accuracy);

        println!("  Simple Threshold Baseline:");
        println!("    Sensitivity:      {:.3}", th_metrics.sensitivity);
        println!("    Specificity:      {:.3}", th_metrics.specificity);
        println!("    Breakpoint acc:   {:.3}", th_metrics.breakpoint_accuracy);
        println!("    Classification:   {:.3}", th_metrics.classification_accuracy);

        // Step 7: Cancer driver gene detection
        println!("\n  Cancer Driver Gene Detection:");
        println!("    {:>6}  {:>8}  {:>6}  {:>8}  {:>6}", "Gene", "Region", "Truth", "Called", "Match");
        println!("    {:->6}  {:->8}  {:->6}  {:->8}  {:->6}", "", "", "", "", "");

        for d in &drivers {
            let truth = d.anomaly.label();
            let called_count = (d.window_start..d.window_end)
                .filter(|&i| gc_calls[i]).count();
            let region_size = d.window_end - d.window_start;
            let detected = called_count as f64 / region_size as f64 > 0.5;
            let match_str = if detected { "YES" } else { "no" };
            println!("    {:>6}  {:>3}-{:<4}  {:>6}  {:>4}/{:<3}  {:>6}",
                d.name, d.window_start, d.window_end, truth, called_count, region_size, match_str);
        }

        // Ingest embeddings into RVF store
        for (i, emb) in embeddings.iter().enumerate() {
            let id = pi as u64 * 100_000 + i as u64;
            all_vectors.push(emb.clone());
            all_ids.push(id);

            all_metadata.push(MetadataEntry {
                field_id: FIELD_PLATFORM,
                value: MetadataValue::String(platform.label().to_string()),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_CHROMOSOME,
                value: MetadataValue::U64(1),
            });
            all_metadata.push(MetadataEntry {
                field_id: FIELD_WINDOW_POS,
                value: MetadataValue::U64(i as u64 * 10_000), // genomic position
            });

            let cnv_label = match windows[i].truth_anomaly {
                AnomalyType::Normal => "normal",
                AnomalyType::CnvGain => "gain",
                AnomalyType::CnvLoss => "loss",
                AnomalyType::Loh => "LOH",
                AnomalyType::MutationHotspot => "hotspot",
                AnomalyType::StructuralVariant => "SV",
            };
            all_metadata.push(MetadataEntry {
                field_id: FIELD_CNV_LABEL,
                value: MetadataValue::String(cnv_label.to_string()),
            });
        }

        println!();
    }

    // ====================================================================
    // Ingest all embeddings into RVF store
    // ====================================================================
    println!("--- RVF Store Ingestion ---");
    let vec_refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
    let ingest = store
        .ingest_batch(&vec_refs, &all_ids, Some(&all_metadata))
        .expect("ingest failed");
    println!("  Ingested: {} vectors (rejected: {})", ingest.accepted, ingest.rejected);

    // ====================================================================
    // Filtered queries
    // ====================================================================
    println!("\n--- RVF Filtered Queries ---");

    // Query: find windows similar to known cancer driver region (EGFR gain)
    let egfr_seed = 42u64; // WGS platform
    let egfr_windows = generate_chromosome(n_windows, Platform::Wgs, egfr_seed);
    let query_emb = extract_embedding(&egfr_windows[710], Platform::Wgs);

    let results = store.query(&query_emb, 10, &QueryOptions::default()).expect("query failed");
    println!("  Similar to EGFR region (window 710): {} results", results.len());
    println!("    {:>8}  {:>10}  {:>10}", "ID", "Distance", "Platform");
    println!("    {:->8}  {:->10}  {:->10}", "", "", "");
    for r in results.iter().take(5) {
        let plat_idx = r.id / 100_000;
        let plat = match plat_idx { 0 => "WGS", 1 => "WES", 2 => "Panel", _ => "?" };
        println!("    {:>8}  {:>10.6}  {:>10}", r.id, r.distance, plat);
    }

    // Filter by platform
    let filter_wgs = FilterExpr::Eq(FIELD_PLATFORM, FilterValue::String("WGS".to_string()));
    let opts_wgs = QueryOptions { filter: Some(filter_wgs), ..Default::default() };
    let wgs_results = store.query(&query_emb, 5, &opts_wgs).expect("query failed");
    println!("\n  WGS-only results: {}", wgs_results.len());

    // Filter by CNV label
    let filter_gain = FilterExpr::Eq(FIELD_CNV_LABEL, FilterValue::String("gain".to_string()));
    let opts_gain = QueryOptions { filter: Some(filter_gain), ..Default::default() };
    let gain_results = store.query(&query_emb, 5, &opts_gain).expect("query failed");
    println!("  Gain-only results: {}", gain_results.len());

    // ====================================================================
    // Witness chain for clinical provenance
    // ====================================================================
    println!("\n--- Clinical Provenance (Witness Chain) ---");

    let chain_steps = [
        ("genesis",             0x01u8),
        ("sample_accessioning", 0x01),
        ("sequencing_qc",       0x02),
        ("alignment_bwa",       0x02),
        ("depth_extraction",    0x02),
        ("gc_normalization",    0x02),
        ("baf_computation",     0x02),
        ("embedding_build",     0x02),
        ("graph_construction",  0x02),
        ("mincut_solve",        0x02),
        ("cnv_classification",  0x02),
        ("driver_annotation",   0x02),
        ("rvf_ingest",          0x08),
        ("clinical_report",     0x01),
        ("pipeline_seal",       0x01),
    ];

    let entries: Vec<WitnessEntry> = chain_steps.iter().enumerate()
        .map(|(i, (step, wtype))| {
            let action_data = format!("genomic_graphcut:{}:step_{}", step, i);
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
    println!("  Chain entries: {} ({} bytes)", verified.len(), chain_bytes.len());
    println!("  Integrity:     VALID");
    println!("\n  Pipeline steps:");
    for (i, (step, _)) in chain_steps.iter().enumerate() {
        let wtype_name = match verified[i].witness_type {
            0x01 => "PROV", 0x02 => "COMP", 0x08 => "DATA", _ => "????",
        };
        println!("    [{:>4}] {:>2} -> {}", wtype_name, i, step);
    }

    // ====================================================================
    // Lineage
    // ====================================================================
    println!("\n--- Lineage: Derive CNV Report ---");
    let child_path = tmp_dir.path().join("cnv_report.rvdna");
    let child_store = store
        .derive(&child_path, DerivationType::Filter, None)
        .expect("failed to derive");
    println!("  Parent file_id:  {}", hex_string(store.file_id()));
    println!("  Child parent_id: {}", hex_string(child_store.parent_id()));
    println!("  Lineage depth:   {}", child_store.lineage_depth());
    child_store.close().expect("close failed");

    // ====================================================================
    // Summary
    // ====================================================================
    println!("\n=== Genomic Graph Cut Summary ===\n");
    println!("  Windows per chromosome: {}", n_windows);
    println!("  Window size:            10 kb");
    println!("  Platforms tested:       WGS (30x), WES (100x), Panel (500x)");
    println!("  Vectors ingested:       {}", ingest.accepted);
    println!("  Embedding dimension:    {}", dim);
    println!("  Driver genes tracked:   {}", drivers.len());
    println!("  Witness chain:          {} steps", verified.len());

    println!("\n  Hyperparameters:");
    println!("    alpha (adjacency):   {:.2}", alpha);
    println!("    beta (RuVector):     {:.2}", beta);
    println!("    gamma (coherence):   {:.2}", gamma);
    println!("    k_nn (neighbors):    {}", k_nn);

    println!("\n  Graph cut insight:");
    println!("    Single aberrant window survives if lambda_i > 2 * gamma * alpha = {:.2}",
        2.0 * gamma * alpha);
    println!("    Contiguous blocks amplify signal: sum(lambda_B) > 2 * gamma * alpha");
    println!("    kNN edges link distant windows with similar CNV profiles");

    store.close().expect("close failed");
    println!("\nDone.");
}

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
