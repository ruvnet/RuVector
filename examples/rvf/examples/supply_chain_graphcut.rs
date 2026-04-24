//! Supply Chain Anomaly Detection via Graph Cut / MRF + RuVector
//!
//! Detects disruptions in multi-tier supply networks using MRF optimization:
//!   1. Generate ~1500 supply chain events across 5 product lines, 5 tiers
//!   2. Inject realistic anomalies: quality defects, shortages, price spikes, etc.
//!   3. Extract 32-dim embeddings, build supply graph, solve s-t mincut
//!   4. Store embeddings in RVF with region/tier/product metadata
//!   5. Evaluate: precision, recall, F1, per-anomaly-type detection rates
//!
//! Run: cargo run --example supply_chain_graphcut --release

use rvf_crypto::{create_witness_chain, shake256_256, verify_witness_chain, WitnessEntry};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_types::DerivationType;
use tempfile::TempDir;

const DIM: usize = 32;
const FIELD_REGION: u16 = 0;
const FIELD_TIER: u16 = 1;
const FIELD_PRODUCT: u16 = 2;
const FIELD_ANOMALY: u16 = 3;

fn lcg_next(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}
fn lcg_f64(s: &mut u64) -> f64 {
    lcg_next(s);
    (*s >> 11) as f64 / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_f64(s).max(1e-15);
    let u2 = lcg_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

const REGIONS: [&str; 5] = ["NA", "EU", "APAC", "LATAM", "MEA"];
const PRODUCTS: [&str; 5] = ["Electronics", "Pharma", "Automotive", "Textiles", "Food"];

#[derive(Debug, Clone, Copy, PartialEq)]
enum AnomalyType {
    Normal,
    QualityDefect,
    SupplyShortage,
    PriceAnomaly,
    LogisticsDelay,
    CounterfeitRisk,
    DemandShock,
}
impl AnomalyType {
    fn label(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::QualityDefect => "quality_defect",
            Self::SupplyShortage => "supply_shortage",
            Self::PriceAnomaly => "price_anomaly",
            Self::LogisticsDelay => "logistics_delay",
            Self::CounterfeitRisk => "counterfeit_risk",
            Self::DemandShock => "demand_shock",
        }
    }
    fn all_anomalies() -> &'static [AnomalyType] {
        &[
            Self::QualityDefect,
            Self::SupplyShortage,
            Self::PriceAnomaly,
            Self::LogisticsDelay,
            Self::CounterfeitRisk,
            Self::DemandShock,
        ]
    }
}

#[derive(Debug, Clone)]
struct Event {
    index: usize,
    product: usize,
    tier: usize,
    region: usize,
    lead_time: f64,
    quantity: f64,
    unit_cost: f64,
    shipping_weight: f64,
    supplier_reliability: f64,
    defect_rate: f64,
    seasonal_factor: f64,
    transit_variance: f64,
    customs_delay: f64,
    truth: AnomalyType,
}

/// Category baselines: (mean_lead, mean_qty, mean_cost, mean_weight)
fn category_baseline(product: usize) -> (f64, f64, f64, f64) {
    match product {
        0 => (14.0, 500.0, 45.0, 2.5),   // Electronics
        1 => (21.0, 200.0, 120.0, 0.8),  // Pharma
        2 => (28.0, 100.0, 350.0, 25.0), // Automotive
        3 => (10.0, 2000.0, 8.0, 5.0),   // Textiles
        _ => (7.0, 5000.0, 3.0, 10.0),   // Food
    }
}

fn generate_events(n_per_product: usize, seed: u64) -> Vec<Event> {
    let mut rng = seed;
    let mut events = Vec::with_capacity(n_per_product * 5);
    for prod in 0..5 {
        let (bl, bq, bc, bw) = category_baseline(prod);
        for i in 0..n_per_product {
            let tier = (i * 5 / n_per_product).min(4);
            let region = (lcg_next(&mut rng) % 5) as usize;
            let t = i as f64 / n_per_product as f64;
            let seasonal = 1.0 + 0.3 * (2.0 * std::f64::consts::PI * t).sin();
            let tier_mult = 1.0 + tier as f64 * 0.15;
            let lead = (bl * tier_mult + lcg_normal(&mut rng) * bl * 0.15).max(1.0);
            let qty = (bq * seasonal + lcg_normal(&mut rng) * bq * 0.2).max(1.0);
            let cost = (bc + lcg_normal(&mut rng) * bc * 0.1).max(0.1);
            let weight = (bw + lcg_normal(&mut rng) * bw * 0.1).max(0.01);
            let reliability = (0.85 + lcg_normal(&mut rng) * 0.05).clamp(0.3, 1.0);
            let defect = (0.02 + lcg_normal(&mut rng) * 0.01).clamp(0.0, 0.5);
            let transit_var = (lcg_normal(&mut rng) * 0.5).abs();
            let customs = (1.0 + lcg_normal(&mut rng) * 0.5).max(0.0);

            // Inject anomalies (~6%)
            let r = lcg_f64(&mut rng);
            let mut ev = Event {
                index: events.len(),
                product: prod,
                tier,
                region,
                lead_time: lead,
                quantity: qty,
                unit_cost: cost,
                shipping_weight: weight,
                supplier_reliability: reliability,
                defect_rate: defect,
                seasonal_factor: seasonal,
                transit_variance: transit_var,
                customs_delay: customs,
                truth: AnomalyType::Normal,
            };
            if r < 0.015 {
                ev.truth = AnomalyType::QualityDefect;
                ev.defect_rate = 0.15 + lcg_f64(&mut rng) * 0.2;
                ev.supplier_reliability *= 0.6;
            } else if r < 0.025 {
                ev.truth = AnomalyType::SupplyShortage;
                ev.lead_time *= 2.5 + lcg_f64(&mut rng);
                ev.quantity *= 0.3;
            } else if r < 0.035 {
                ev.truth = AnomalyType::PriceAnomaly;
                ev.unit_cost *= 1.8 + lcg_f64(&mut rng) * 0.5;
            } else if r < 0.048 {
                ev.truth = AnomalyType::LogisticsDelay;
                ev.transit_variance = 3.0 + lcg_f64(&mut rng) * 4.0;
                ev.customs_delay = 5.0 + lcg_f64(&mut rng) * 10.0;
            } else if r < 0.055 {
                ev.truth = AnomalyType::CounterfeitRisk;
                ev.unit_cost *= 0.4;
                ev.defect_rate = 0.12 + lcg_f64(&mut rng) * 0.15;
                ev.supplier_reliability *= 0.5;
            } else if r < 0.065 {
                ev.truth = AnomalyType::DemandShock;
                ev.quantity *= 3.0 + lcg_f64(&mut rng) * 2.0;
            }
            events.push(ev);
        }
    }
    events
}

fn extract_embedding(ev: &Event) -> Vec<f32> {
    let (bl, bq, bc, _bw) = category_baseline(ev.product);
    let log_lead = (ev.lead_time / bl).max(0.01).ln();
    let log_qty = (ev.quantity / bq).max(0.01).ln();
    let log_cost = (ev.unit_cost / bc).max(0.01).ln();
    let rel = ev.supplier_reliability;
    let defect_z = (ev.defect_rate - 0.02) / 0.01_f64.max(0.001);
    let mut region_oh = [0.0f32; 5];
    region_oh[ev.region] = 1.0;
    let tier_norm = ev.tier as f64 / 4.0;
    let prod_norm = ev.product as f64 / 4.0;
    let t = ev.index as f64 / 1500.0;
    let sin_s = (2.0 * std::f64::consts::PI * t).sin();
    let cos_s = (2.0 * std::f64::consts::PI * t).cos();
    let transit_z = (ev.transit_variance - 0.5) / 0.5_f64.max(0.001);
    let cost_quality = ev.unit_cost / (ev.supplier_reliability.max(0.01) * 100.0);
    let lead_qty = log_lead / (log_qty.abs() + 1.0);
    // Rolling window deviation proxy: seasonal deviation
    let trend_dev = (ev.seasonal_factor - 1.0).abs();

    let mut f = Vec::with_capacity(DIM);
    f.push(log_lead as f32); // 0
    f.push(log_qty as f32); // 1
    f.push(log_cost as f32); // 2
    f.push(rel as f32); // 3
    f.push(ev.defect_rate as f32); // 4
    f.push(defect_z.clamp(-5.0, 5.0) as f32); // 5
    for v in &region_oh {
        f.push(*v);
    } // 6-10
    f.push(tier_norm as f32); // 11
    f.push(prod_norm as f32); // 12
    f.push(sin_s as f32); // 13
    f.push(cos_s as f32); // 14
    f.push(transit_z.clamp(-5.0, 5.0) as f32); // 15
    f.push(cost_quality as f32); // 16
    f.push(lead_qty as f32); // 17
    f.push(trend_dev as f32); // 18
    f.push(ev.customs_delay as f32 / 10.0); // 19
    f.push(ev.shipping_weight as f32 / 30.0); // 20
                                              // Interaction features
    f.push((log_lead * log_cost) as f32); // 21
    f.push((ev.defect_rate * (1.0 - rel)) as f32); // 22
    f.push((ev.transit_variance * ev.customs_delay / 10.0) as f32); // 23
    f.push((log_qty.abs() * trend_dev) as f32); // 24
    f.push((log_cost * log_cost) as f32); // 25
    f.push((log_lead * log_lead) as f32); // 26
    f.push(((1.0 - rel) * ev.defect_rate * 10.0) as f32); // 27
    f.push((ev.quantity / bq).min(5.0) as f32); // 28
    f.push((ev.lead_time / bl).min(5.0) as f32); // 29
    f.push((ev.unit_cost / bc).min(5.0) as f32); // 30
    f.push(0.0); // 31 padding
    f.truncate(DIM);
    f
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let dn = na.sqrt() * nb.sqrt();
    if dn < 1e-15 {
        0.0
    } else {
        d / dn
    }
}

fn unary_score(ev: &Event) -> f64 {
    let (bl, bq, bc, _) = category_baseline(ev.product);
    let lead_dev = ((ev.lead_time / bl).ln()).abs();
    let cost_dev = ((ev.unit_cost / bc).ln()).abs();
    let qty_dev = ((ev.quantity / bq).ln()).abs();
    let def_sig = (ev.defect_rate - 0.02).max(0.0) * 10.0;
    let rel_sig = (0.7 - ev.supplier_reliability).max(0.0) * 3.0;
    let transit_sig = (ev.transit_variance - 1.5).max(0.0);
    let customs_sig = (ev.customs_delay - 3.0).max(0.0) / 5.0;
    0.4 * lead_dev
        + 0.4 * cost_dev
        + 0.3 * qty_dev
        + 1.0 * def_sig
        + 0.8 * rel_sig
        + 0.5 * transit_sig
        + 0.3 * customs_sig
        - 0.40
}

struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

fn build_graph(events: &[Event], embs: &[Vec<f32>], alpha: f64, beta: f64, k: usize) -> Vec<Edge> {
    let m = events.len();
    let mut edges = Vec::new();
    // Tier chain: tier(n) -> tier(n+1) for same product line
    for i in 0..m {
        for j in (i + 1)..m.min(i + 60) {
            if events[i].product == events[j].product && events[j].tier == events[i].tier + 1 {
                edges.push(Edge {
                    from: i,
                    to: j,
                    weight: alpha,
                });
                edges.push(Edge {
                    from: j,
                    to: i,
                    weight: alpha,
                });
                break;
            }
        }
    }
    // Temporal chain: consecutive events in same product line
    let mut last_by_product = [usize::MAX; 5];
    for i in 0..m {
        let p = events[i].product;
        if last_by_product[p] != usize::MAX {
            let prev = last_by_product[p];
            edges.push(Edge {
                from: prev,
                to: i,
                weight: alpha * 0.5,
            });
            edges.push(Edge {
                from: i,
                to: prev,
                weight: alpha * 0.5,
            });
        }
        last_by_product[p] = i;
    }
    // Geographic proximity: same-region events share smoothing edges (sampled)
    for i in 0..m {
        let mut rc = 0;
        for j in (i + 1)..m.min(i + 100) {
            if events[i].region == events[j].region && rc < 3 {
                edges.push(Edge {
                    from: i,
                    to: j,
                    weight: alpha * 0.3,
                });
                edges.push(Edge {
                    from: j,
                    to: i,
                    weight: alpha * 0.3,
                });
                rc += 1;
            }
        }
    }
    // kNN similarity edges from embeddings
    for i in (0..m).step_by(3) {
        let mut sims: Vec<(usize, f64)> = (0..m)
            .filter(|&j| (j as isize - i as isize).unsigned_abs() > 5)
            .map(|j| (j, cosine_sim(&embs[i], &embs[j]).max(0.0)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(j, s) in sims.iter().take(k) {
            if s > 0.1 {
                edges.push(Edge {
                    from: i,
                    to: j,
                    weight: beta * s,
                });
            }
        }
    }
    edges
}

fn solve_mincut(lambdas: &[f64], edges: &[Edge], gamma: f64) -> Vec<bool> {
    let m = lambdas.len();
    let (s, t, n) = (m, m + 1, m + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();
    let ae =
        |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>, u: usize, v: usize, c: f64| {
            let idx = caps.len();
            caps.push(c);
            caps.push(0.0);
            adj[u].push((v, idx));
            adj[v].push((u, idx + 1));
        };
    for i in 0..m {
        let p0 = lambdas[i].max(0.0);
        let p1 = (-lambdas[i]).max(0.0);
        if p0 > 1e-12 {
            ae(&mut adj, &mut caps, s, i, p0);
        }
        if p1 > 1e-12 {
            ae(&mut adj, &mut caps, i, t, p1);
        }
    }
    for e in edges {
        let c = gamma * e.weight;
        if c > 1e-12 {
            ae(&mut adj, &mut caps, e.from, e.to, c);
        }
    }
    loop {
        let mut par: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        vis[s] = true;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t {
                break;
            }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 {
                    vis[v] = true;
                    par[v] = Some((u, ei));
                    q.push_back(v);
                }
            }
        }
        if !vis[t] {
            break;
        }
        let mut bn = f64::MAX;
        let mut v = t;
        while let Some((u, ei)) = par[v] {
            bn = bn.min(caps[ei]);
            v = u;
        }
        v = t;
        while let Some((u, ei)) = par[v] {
            caps[ei] -= bn;
            caps[ei ^ 1] += bn;
            v = u;
        }
    }
    let mut reach = vec![false; n];
    let mut stk = vec![s];
    reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] {
            if !reach[v] && caps[ei] > 1e-15 {
                reach[v] = true;
                stk.push(v);
            }
        }
    }
    (0..m).map(|i| reach[i]).collect()
}

fn threshold_detect(events: &[Event]) -> Vec<bool> {
    events
        .iter()
        .map(|ev| {
            let (bl, bq, bc, _) = category_baseline(ev.product);
            let lr = ((ev.lead_time / bl).ln()).abs();
            let cr = ((ev.unit_cost / bc).ln()).abs();
            let qr = ((ev.quantity / bq).ln()).abs();
            lr > 0.7 || cr > 0.5 || qr > 0.9 || ev.defect_rate > 0.08 || ev.transit_variance > 2.5
        })
        .collect()
}

fn evaluate(events: &[Event], calls: &[bool]) -> (f64, f64, f64, f64) {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0, 0, 0);
    for (i, ev) in events.iter().enumerate() {
        let truth = ev.truth != AnomalyType::Normal;
        match (truth, calls[i]) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let prec = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let rec = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if prec + rec > 0.0 {
        2.0 * prec * rec / (prec + rec)
    } else {
        0.0
    };
    let fpr = if tn + fp > 0 {
        fp as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    (prec, rec, f1, fpr)
}

fn per_type_detection(events: &[Event], calls: &[bool]) -> Vec<(&'static str, usize, usize)> {
    AnomalyType::all_anomalies()
        .iter()
        .map(|atype| {
            let total = events.iter().filter(|e| e.truth == *atype).count();
            let det = events
                .iter()
                .enumerate()
                .filter(|(i, e)| e.truth == *atype && calls[*i])
                .count();
            (atype.label(), det, total)
        })
        .collect()
}

fn resilience_score(events: &[Event], calls: &[bool]) -> f64 {
    let total_anom = events
        .iter()
        .filter(|e| e.truth != AnomalyType::Normal)
        .count();
    let detected = events
        .iter()
        .enumerate()
        .filter(|(i, e)| e.truth != AnomalyType::Normal && calls[*i])
        .count();
    let false_alarms = events
        .iter()
        .enumerate()
        .filter(|(i, e)| e.truth == AnomalyType::Normal && calls[*i])
        .count();
    if total_anom == 0 {
        return 1.0;
    }
    let det_rate = detected as f64 / total_anom as f64;
    let fa_penalty = (false_alarms as f64 / events.len() as f64).min(0.5);
    (det_rate * 0.7 + (1.0 - fa_penalty) * 0.3).clamp(0.0, 1.0)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    println!("=== Supply Chain Anomaly Detection via Graph Cut ===\n");
    let n_per_product = 300;
    let total = n_per_product * 5;
    let (alpha, beta, gamma, k_nn) = (0.25, 0.12, 0.35, 3usize);

    let events = generate_events(n_per_product, 42);
    let n_anom = events
        .iter()
        .filter(|e| e.truth != AnomalyType::Normal)
        .count();
    println!("  Events: {} across {} products x {} tiers", total, 5, 5);
    println!(
        "  Anomalies: {} ({:.1}%)\n",
        n_anom,
        n_anom as f64 / total as f64 * 100.0
    );

    println!("  {:>18} {:>5}", "Type", "Count");
    println!("  {:->18} {:->5}", "", "");
    for atype in AnomalyType::all_anomalies() {
        let c = events.iter().filter(|e| e.truth == *atype).count();
        if c > 0 {
            println!("  {:>18} {:>5}", atype.label(), c);
        }
    }

    // Extract embeddings and build graph
    let embs: Vec<Vec<f32>> = events.iter().map(|e| extract_embedding(e)).collect();
    let lam: Vec<f64> = events.iter().map(|e| unary_score(e)).collect();
    let edges = build_graph(&events, &embs, alpha, beta, k_nn);
    println!("\n  Graph: {} nodes, {} edges", total, edges.len());

    // Solve
    let gc = solve_mincut(&lam, &edges, gamma);
    let tc = threshold_detect(&events);
    let (gp, gr, gf, gfpr) = evaluate(&events, &gc);
    let (tp, tr, tf, tfpr) = evaluate(&events, &tc);
    let g_res = resilience_score(&events, &gc);
    let t_res = resilience_score(&events, &tc);

    println!(
        "\n  {:>10} {:>7} {:>7} {:>7} {:>7} {:>10}",
        "Method", "Prec", "Recall", "F1", "FPR", "Resilience"
    );
    println!(
        "  {:->10} {:->7} {:->7} {:->7} {:->7} {:->10}",
        "", "", "", "", "", ""
    );
    println!(
        "  {:>10} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>10.3}",
        "GraphCut", gp, gr, gf, gfpr, g_res
    );
    println!(
        "  {:>10} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>10.3}",
        "Threshold", tp, tr, tf, tfpr, t_res
    );

    // Per-anomaly-type detection
    let gc_types = per_type_detection(&events, &gc);
    let tc_types = per_type_detection(&events, &tc);
    println!("\n  Per-Anomaly Detection Rates:");
    println!("  {:>18} {:>8} {:>8}", "Type", "GC", "Thresh");
    println!("  {:->18} {:->8} {:->8}", "", "", "");
    for i in 0..gc_types.len() {
        let (name, gd, gt) = gc_types[i];
        let (_, td, _tt) = tc_types[i];
        if gt > 0 {
            println!("  {:>18} {:>3}/{:<3} {:>3}/{:<3}", name, gd, gt, td, gt);
        }
    }

    // RVF ingestion
    println!("\n--- RVF Ingestion ---");
    let tmp = TempDir::new().expect("tmpdir");
    let opts = RvfOptions {
        dimension: DIM as u16,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut store = RvfStore::create(&tmp.path().join("supply_chain.rvf"), opts).expect("create");
    let mut all_vecs: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_meta: Vec<MetadataEntry> = Vec::new();
    for (i, ev) in events.iter().enumerate() {
        all_vecs.push(embs[i].clone());
        all_ids.push(i as u64);
        all_meta.push(MetadataEntry {
            field_id: FIELD_REGION,
            value: MetadataValue::String(REGIONS[ev.region].into()),
        });
        all_meta.push(MetadataEntry {
            field_id: FIELD_TIER,
            value: MetadataValue::U64(ev.tier as u64),
        });
        all_meta.push(MetadataEntry {
            field_id: FIELD_PRODUCT,
            value: MetadataValue::String(PRODUCTS[ev.product].into()),
        });
        all_meta.push(MetadataEntry {
            field_id: FIELD_ANOMALY,
            value: MetadataValue::String(ev.truth.label().into()),
        });
    }
    let refs: Vec<&[f32]> = all_vecs.iter().map(|v| v.as_slice()).collect();
    let ing = store
        .ingest_batch(&refs, &all_ids, Some(&all_meta))
        .expect("ingest");
    println!("  Ingested: {} (rejected: {})", ing.accepted, ing.rejected);

    // Filtered queries: find similar disruption patterns by region
    println!("\n--- Filtered Queries ---");
    let anom_idx = events
        .iter()
        .position(|e| e.truth != AnomalyType::Normal)
        .unwrap_or(0);
    let qv = &embs[anom_idx];
    let res = store.query(qv, 10, &QueryOptions::default()).expect("q");
    println!(
        "  Similar to event {} ({}): {} results",
        anom_idx,
        events[anom_idx].truth.label(),
        res.len()
    );
    for r in res.iter().take(5) {
        let ev = &events[r.id as usize];
        println!(
            "    id={:>5} dist={:.4} region={:<5} tier={} prod={:<12} anom={}",
            r.id,
            r.distance,
            REGIONS[ev.region],
            ev.tier,
            PRODUCTS[ev.product],
            ev.truth.label()
        );
    }

    for reg in &["NA", "EU", "APAC"] {
        let f = FilterExpr::Eq(FIELD_REGION, FilterValue::String(reg.to_string()));
        let r = store
            .query(
                qv,
                5,
                &QueryOptions {
                    filter: Some(f),
                    ..Default::default()
                },
            )
            .expect("q");
        println!("  {}-only: {} results", reg, r.len());
    }

    // Witness chain
    println!("\n--- Witness Chain ---");
    let steps = [
        ("genesis", 0x01u8),
        ("event_ingest", 0x08),
        ("normalize", 0x02),
        ("embed", 0x02),
        ("graph_build", 0x02),
        ("mincut_solve", 0x02),
        ("classify", 0x02),
        ("alert_gen", 0x02),
        ("rvf_store", 0x08),
        ("query", 0x02),
        ("resilience", 0x02),
        ("report", 0x01),
        ("seal", 0x01),
    ];
    let entries: Vec<WitnessEntry> = steps
        .iter()
        .enumerate()
        .map(|(i, (step, wt))| WitnessEntry {
            prev_hash: [0u8; 32],
            action_hash: shake256_256(format!("sc_gc:{}:{}", step, i).as_bytes()),
            timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
            witness_type: *wt,
        })
        .collect();
    let chain = create_witness_chain(&entries);
    let verified = verify_witness_chain(&chain).expect("verify");
    println!("  {} entries, {} bytes, VALID", verified.len(), chain.len());
    for (i, (step, _)) in steps.iter().enumerate() {
        let wn = match verified[i].witness_type {
            0x01 => "PROV",
            0x02 => "COMP",
            0x08 => "DATA",
            _ => "????",
        };
        println!("    [{:>4}] {:>2} -> {}", wn, i, step);
    }

    // Lineage
    println!("\n--- Lineage ---");
    let child = store
        .derive(
            &tmp.path().join("sc_report.rvf"),
            DerivationType::Filter,
            None,
        )
        .expect("derive");
    println!(
        "  Parent: {}  Child: {}  Depth: {}",
        hex(store.file_id()),
        hex(child.parent_id()),
        child.lineage_depth()
    );
    child.close().expect("close");

    // Summary
    println!("\n=== Summary ===");
    println!(
        "  {} events | {} anomalies ({:.1}%) | {} edges",
        total,
        n_anom,
        n_anom as f64 / total as f64 * 100.0,
        edges.len()
    );
    println!("  GraphCut  F1={:.3} | Resilience={:.3}", gf, g_res);
    println!("  Threshold F1={:.3} | Resilience={:.3}", tf, t_res);
    println!(
        "  alpha={:.2} beta={:.2} gamma={:.2} k={}",
        alpha, beta, gamma, k_nn
    );
    println!(
        "  RVF: {} embeddings | Witness: {} steps",
        ing.accepted,
        verified.len()
    );
    store.close().expect("close");
    println!("\nDone.");
}
