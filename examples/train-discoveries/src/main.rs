//! Discovery ETL Pipeline using RuVector Sublinear Solver
//!
//! A three-stage pipeline:
//!   1. **Extract** — Load discovery JSON files from all data sources
//!   2. **Transform** — Embed text into 64-dim vectors, build similarity graph,
//!      run ForwardPush PPR (sublinear PageRank) for cross-domain correlation
//!   3. **Load** — Output ranked correlations and domain affinity matrix
//!
//! Uses the ruvector-solver ForwardPush algorithm (Andersen-Chung-Lang 2006)
//! which runs in O(1/epsilon) time, independent of graph size — true sublinear
//! discovery of hidden cross-domain connections.

use ruvector_core::distance::cosine_distance;
use ruvector_core::index::flat::FlatIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::DistanceMetric;
use ruvector_solver::forward_push::ForwardPushSolver;
use ruvector_solver::traits::SublinearPageRank;
use ruvector_solver::types::CsrMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// =========================================================================
// Data model — accepts both old-format and swarm-format discovery JSON
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Discovery {
    title: String,
    content: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    source: String,
    #[serde(default)]
    confidence: f64,
    // Old-format fields (optional)
    #[serde(default)]
    category: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    domain: String,
    #[serde(default = "default_source_api")]
    source_api: String,
    #[serde(default)]
    data_points: serde_json::Value,
}

fn default_source_api() -> String {
    "unknown".to_string()
}

// =========================================================================
// Stage 1: EXTRACT — Load all discovery JSON from disk
// =========================================================================

fn extract(dir: &Path) -> Vec<(String, Discovery)> {
    let mut all = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("  Cannot read {}: {}", dir.display(), err);
            return all;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let fname = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let raw = match fs::read_to_string(&path) {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Only process JSON arrays
        if !raw.trim_start().starts_with('[') {
            continue;
        }

        let discoveries: Vec<Discovery> = match serde_json::from_str(&raw) {
            Ok(d) => d,
            Err(err) => {
                eprintln!("  Skipping {} ({})", fname, err);
                continue;
            }
        };

        // Infer domain from filename
        let inferred_domain = infer_domain(&fname);

        for (i, mut d) in discoveries.into_iter().enumerate() {
            if d.domain.is_empty() {
                d.domain = inferred_domain.clone();
            }
            if d.source_api == "unknown" && !d.source.is_empty() {
                d.source_api = d.source.clone();
            }
            let id = format!("{}#{}", fname, i);
            all.push((id, d));
        }
    }
    all
}

fn infer_domain(filename: &str) -> String {
    let f = filename.to_lowercase();
    if f.contains("exoplanet")
        || f.contains("apod")
        || f.contains("mars")
        || f.contains("gw")
        || f.contains("neo")
        || f.contains("solar")
        || f.contains("cme")
        || f.contains("flare")
        || f.contains("geostorm")
        || f.contains("ips")
        || f.contains("sep")
        || f.contains("asteroid")
        || f.contains("spacex")
        || f.contains("iss")
    {
        "space".into()
    } else if f.contains("earthquake")
        || f.contains("climate")
        || f.contains("river")
        || f.contains("ocean")
        || f.contains("natural_event")
        || f.contains("fire")
        || f.contains("volcano")
        || f.contains("marine")
        || f.contains("epa")
    {
        "earth".into()
    } else if f.contains("genom")
        || f.contains("protein")
        || f.contains("medical")
        || f.contains("disease")
        || f.contains("genetic")
        || f.contains("endangered")
    {
        "life-science".into()
    } else if f.contains("economic") || f.contains("market") {
        "economics".into()
    } else if f.contains("arxiv")
        || f.contains("crossref")
        || f.contains("physics")
        || f.contains("material")
        || f.contains("academic")
        || f.contains("book")
        || f.contains("nobel")
    {
        "research".into()
    } else if f.contains("github")
        || f.contains("hacker")
        || f.contains("tech")
        || f.contains("airquality")
    {
        "technology".into()
    } else if f.contains("art")
        || f.contains("library")
        || f.contains("smithsonian")
        || f.contains("wiki")
        || f.contains("biodiversity")
    {
        "culture".into()
    } else {
        "misc".into()
    }
}

// =========================================================================
// Stage 2: TRANSFORM — Embed, build graph, run sublinear PageRank
// =========================================================================

const DIM: usize = 64;

/// Embed a discovery into a 64-dim feature vector
fn embed(d: &Discovery) -> Vec<f32> {
    let mut vec = vec![0.0f32; DIM];

    // Domain encoding (dims 0-7)
    let domain_idx = match d.domain.as_str() {
        "space" => 0,
        "earth" => 1,
        "life-science" => 2,
        "research" => 3,
        "economics" => 4,
        "technology" => 5,
        "culture" => 6,
        _ => 7,
    };
    vec[domain_idx] = 1.0;

    // Keyword activations (dims 8-31)
    let text = format!("{} {}", d.title, d.content).to_lowercase();
    let keywords: &[(&str, usize)] = &[
        ("solar", 8),
        ("flare", 9),
        ("cme", 10),
        ("earthquake", 11),
        ("gene", 12),
        ("protein", 13),
        ("cancer", 14),
        ("gdp", 15),
        ("asteroid", 16),
        ("hazardous", 17),
        ("volcano", 18),
        ("wildfire", 19),
        ("bitcoin", 20),
        ("ai", 21),
        ("neural", 22),
        ("climate", 23),
        ("disease", 24),
        ("endangered", 25),
        ("ocean", 26),
        ("mars", 27),
        ("gravitational", 28),
        ("exoplanet", 29),
        ("mutation", 30),
        ("inflation", 31),
    ];
    for &(kw, dim) in keywords {
        if text.contains(kw) {
            vec[dim] = 1.0;
        }
    }

    // Character trigram hashing (dims 32-55)
    let bytes = text.as_bytes();
    if bytes.len() >= 3 {
        for window in bytes.windows(3) {
            let hash = (window[0] as u32)
                .wrapping_mul(31)
                .wrapping_add(window[1] as u32)
                .wrapping_mul(31)
                .wrapping_add(window[2] as u32);
            let idx = 32 + (hash as usize % 24);
            vec[idx] += 1.0;
        }
    }

    // Numeric features (dims 56-63)
    vec[56] = d.confidence as f32;
    vec[57] = if d.confidence > 0.9 { 1.0 } else { 0.0 };
    // Timestamp recency (higher = more recent)
    vec[58] = if d.timestamp.contains("2026-03") {
        1.0
    } else {
        0.5
    };
    // Source diversity signal
    vec[59] = match d.source_api.as_str() {
        "nasa_donki" | "nasa_apod" | "nasa_neows" => 0.9,
        "usgs" | "noaa" => 0.85,
        "pubmed" | "ncbi" => 0.8,
        "worldbank" => 0.75,
        _ => 0.5,
    };

    // L2 normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    vec
}

/// Build a k-nearest-neighbor similarity graph as CSR matrix.
/// Each node connects only to its top-k most similar neighbors,
/// creating the sparse graph structure that ForwardPush needs.
fn build_knn_graph(vectors: &[(String, Vec<f32>)], k: usize) -> (CsrMatrix<f64>, Vec<String>) {
    let n = vectors.len();
    let ids: Vec<String> = vectors.iter().map(|(id, _)| id.clone()).collect();
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..n {
        // Compute similarity to all other nodes
        let mut sims: Vec<(usize, f64)> = Vec::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = cosine_distance(&vectors[i].1, &vectors[j].1);
            let sim = (1.0 - dist) as f64;
            sims.push((j, sim));
        }
        // Keep only top-k neighbors
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.truncate(k);

        // Normalize edge weights to sum to 1 (transition probability)
        let row_sum: f64 = sims.iter().map(|(_, s)| s).sum();
        if row_sum > 0.0 {
            for (j, sim) in &sims {
                entries.push((i, *j, sim / row_sum));
            }
        }
    }

    let matrix = CsrMatrix::<f64>::from_coo(n, n, entries);
    (matrix, ids)
}

/// Run ForwardPush PPR from individual "bridge" nodes to find cross-domain hubs.
/// Strategy: for each node, run single-source PPR. If the top-ranked hit is in
/// a DIFFERENT domain, that's a cross-domain bridge — a novel discovery.
fn run_sublinear_pagerank(
    graph: &CsrMatrix<f64>,
    ids: &[String],
    discoveries: &HashMap<String, Discovery>,
) -> Vec<RankedCorrelation> {
    let solver = ForwardPushSolver::new(0.85, 0.0001); // fine-grained epsilon
    let mut correlations = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (i, id) in ids.iter().enumerate() {
        let source_disc = match discoveries.get(id) {
            Some(d) => d,
            None => continue,
        };

        let ppr = match solver.ppr(graph, i, 0.85, 0.0001) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Find top cross-domain hits from this node
        for (node_idx, ppr_value) in &ppr {
            if *node_idx >= ids.len() || *node_idx == i {
                continue;
            }
            let target_id = &ids[*node_idx];
            if let Some(target_disc) = discoveries.get(target_id) {
                if target_disc.domain != source_disc.domain && *ppr_value > 0.005 {
                    let key = format!("{}→{}", source_disc.domain, target_id);
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);
                    correlations.push(RankedCorrelation {
                        source_domain: source_disc.domain.clone(),
                        target_domain: target_disc.domain.clone(),
                        target_title: target_disc.title.clone(),
                        target_id: target_id.clone(),
                        ppr_score: *ppr_value,
                        confidence: target_disc.confidence * ppr_value,
                    });
                }
            }
        }
    }

    correlations.sort_by(|a, b| b.ppr_score.partial_cmp(&a.ppr_score).unwrap());
    correlations.truncate(50); // top 50 cross-domain bridges
    correlations
}

#[derive(Debug, Clone, Serialize)]
struct RankedCorrelation {
    source_domain: String,
    target_domain: String,
    target_title: String,
    target_id: String,
    ppr_score: f64,
    confidence: f64,
}

// =========================================================================
// Stage 3: LOAD — Output results, write correlation JSON
// =========================================================================

fn load_results(
    _discoveries: &HashMap<String, Discovery>,
    correlations: &[RankedCorrelation],
    domain_matrix: &[(String, Vec<(String, f32)>)],
    output_dir: &Path,
) {
    // Print top correlations
    println!("\n  Top 20 Cross-Domain Correlations (by PPR score):");
    println!("  {:-<80}", "");
    for (i, c) in correlations.iter().take(20).enumerate() {
        println!(
            "  {:2}. [{} → {}] ppr={:.6} conf={:.4}",
            i + 1,
            c.source_domain,
            c.target_domain,
            c.ppr_score,
            c.confidence,
        );
        println!("      {}", truncate(&c.target_title, 72));
    }

    // Print domain affinity matrix
    println!("\n  Domain Affinity Matrix (centroid cosine similarity):");
    println!("  {:-<80}", "");
    print!("  {:>12}", "");
    let all_domains: Vec<&str> = domain_matrix.iter().map(|(d, _)| d.as_str()).collect();
    for d in &all_domains {
        print!(" {:>10}", abbrev(d));
    }
    println!();
    for (domain, sims) in domain_matrix {
        print!("  {:>12}", abbrev(domain));
        for (_, sim) in sims {
            print!(" {:>10.4}", sim);
        }
        println!();
    }

    // Write correlations to JSON
    let output_file = output_dir.join("pipeline_correlations.json");
    let json = serde_json::to_string_pretty(&correlations).unwrap_or_default();
    match fs::write(&output_file, &json) {
        Ok(_) => println!(
            "\n  Wrote {} correlations to {}",
            correlations.len(),
            output_file.display()
        ),
        Err(e) => eprintln!("\n  Could not write output: {}", e),
    }
}

// =========================================================================
// Main: orchestrate the ETL pipeline
// =========================================================================

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  RuVector Discovery ETL Pipeline                       ║");
    println!("║  Sublinear Solver × ForwardPush PPR × Cross-Domain     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let data_dir = resolve_data_dir();
    let t0 = Instant::now();

    // ── Stage 1: EXTRACT ──
    println!("━━ Stage 1: EXTRACT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let raw = extract(&data_dir);
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    for (_, d) in &raw {
        *domain_counts.entry(d.domain.clone()).or_insert(0) += 1;
    }
    println!(
        "  Loaded {} discoveries from {}",
        raw.len(),
        data_dir.display()
    );
    let mut sorted: Vec<_> = domain_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (domain, count) in &sorted {
        println!("    {:20} {:>4} items", domain, count);
    }
    let t_extract = t0.elapsed();
    println!("  Extract time: {:?}\n", t_extract);

    // ── Stage 2: TRANSFORM ──
    println!("━━ Stage 2: TRANSFORM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 2a. Embed all discoveries
    let t1 = Instant::now();
    let mut index = FlatIndex::new(DIM, DistanceMetric::Cosine);
    let mut discovery_map: HashMap<String, Discovery> = HashMap::new();
    let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();

    for (id, d) in &raw {
        let vec = embed(d);
        index
            .add(id.clone(), vec.clone())
            .expect("index add failed");
        vectors.push((id.clone(), vec));
        discovery_map.insert(id.clone(), d.clone());
    }
    println!("  Embedded {} vectors × {} dims", vectors.len(), DIM);

    // 2b. Build similarity graph (sublinear threshold)
    let knn_k = 12; // each node connects to top-12 most similar neighbors
    println!("  Building {}-NN similarity graph...", knn_k);
    let (graph, graph_ids) = build_knn_graph(&vectors, knn_k);
    println!(
        "  Graph: {} nodes, {} edges (density={:.4})",
        graph.rows,
        graph.nnz(),
        graph.nnz() as f64 / (graph.rows as f64 * graph.cols as f64),
    );

    // 2c. Run sublinear ForwardPush PPR
    println!("  Running ForwardPush PPR (alpha=0.85, eps=0.001)...");
    let correlations = run_sublinear_pagerank(&graph, &graph_ids, &discovery_map);
    println!("  Found {} cross-domain correlations", correlations.len());

    // 2d. Compute domain centroid similarity matrix
    let domain_names: Vec<String> = sorted.iter().map(|(d, _)| d.to_string()).collect();
    let mut domain_vecs: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for (id, d) in &raw {
        if let Some((_, vec)) = vectors.iter().find(|(vid, _)| vid == id) {
            domain_vecs
                .entry(d.domain.clone())
                .or_default()
                .push(vec.clone());
        }
    }

    let mut domain_matrix: Vec<(String, Vec<(String, f32)>)> = Vec::new();
    for d1 in &domain_names {
        let c1 = centroid(domain_vecs.get(d1).map(|v| v.as_slice()).unwrap_or(&[]));
        let mut row = Vec::new();
        for d2 in &domain_names {
            let c2 = centroid(domain_vecs.get(d2).map(|v| v.as_slice()).unwrap_or(&[]));
            let sim = 1.0 - cosine_distance(&c1, &c2);
            row.push((d2.clone(), sim));
        }
        domain_matrix.push((d1.clone(), row));
    }

    let t_transform = t1.elapsed();
    println!("  Transform time: {:?}\n", t_transform);

    // ── Stage 3: LOAD ──
    println!("━━ Stage 3: LOAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    load_results(&discovery_map, &correlations, &domain_matrix, &data_dir);

    let total = t0.elapsed();
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Pipeline complete                                     ║");
    println!(
        "║  {} discoveries → {} correlations            ",
        raw.len(),
        correlations.len()
    );
    println!("║  Total: {:?}                                  ", total);
    println!("║  Solver: ForwardPush PPR (sublinear O(1/ε))            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
}

// =========================================================================
// Helpers
// =========================================================================

fn resolve_data_dir() -> PathBuf {
    for c in &[
        PathBuf::from("examples/data/discoveries"),
        PathBuf::from("../../examples/data/discoveries"),
        PathBuf::from("../data/discoveries"),
    ] {
        if c.is_dir() {
            return c.clone();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/discoveries")
}

fn centroid(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return vec![0.0; DIM];
    }
    let n = vecs.len() as f32;
    let mut c = vec![0.0f32; DIM];
    for v in vecs {
        for (i, val) in v.iter().enumerate() {
            c[i] += val;
        }
    }
    for x in c.iter_mut() {
        *x /= n;
    }
    let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in c.iter_mut() {
            *x /= norm;
        }
    }
    c
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

fn abbrev(domain: &str) -> String {
    match domain {
        "space" => "space".into(),
        "earth" => "earth".into(),
        "life-science" => "life-sci".into(),
        "research" => "research".into(),
        "economics" => "econ".into(),
        "technology" => "tech".into(),
        "culture" => "culture".into(),
        other => {
            if other.len() > 8 {
                other[..8].to_string()
            } else {
                other.to_string()
            }
        }
    }
}
