//! Financial Fraud Detection via Graph Cut / MRF Optimization + RuVector
//!
//! Applies the same MRF/mincut formulation used in genomic and medical pipelines
//! to financial transaction fraud detection:
//!
//!   1. Generate ~2000 synthetic transactions with realistic distributions
//!   2. Inject fraud patterns: card-not-present, account takeover, card clone,
//!      synthetic identity, refund abuse (~0.5% fraud rate)
//!   3. Extract 32-dim embeddings, build transaction graph, solve s-t mincut
//!   4. Compare graph cut vs simple threshold baseline
//!   5. Store embeddings in RVF with merchant/amount metadata, witness chain
//!
//! Run: cargo run --example financial_fraud_graphcut --release

use rvf_crypto::{create_witness_chain, shake256_256, verify_witness_chain, WitnessEntry};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_types::DerivationType;
use tempfile::TempDir;

const DIM: usize = 32;
const N_TX: usize = 2000;
const FIELD_MERCHANT: u16 = 0;
const FIELD_AMOUNT_BUCKET: u16 = 1;
const FIELD_FRAUD_TYPE: u16 = 2;

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

#[derive(Debug, Clone, Copy, PartialEq)]
enum FraudType {
    Legit,
    CardNotPresent,
    AccountTakeover,
    CardClone,
    Synthetic,
    Refund,
}
impl FraudType {
    fn label(&self) -> &'static str {
        match self {
            Self::Legit => "legit",
            Self::CardNotPresent => "CNP",
            Self::AccountTakeover => "ATO",
            Self::CardClone => "clone",
            Self::Synthetic => "synth",
            Self::Refund => "refund",
        }
    }
}

const MERCHANTS: [&str; 10] = [
    "grocery",
    "gas",
    "restaurant",
    "online_retail",
    "travel",
    "electronics",
    "pharmacy",
    "subscription",
    "atm",
    "luxury",
];

#[derive(Debug, Clone)]
struct Transaction {
    id: usize,
    card_id: usize,
    amount: f64,
    hour: f64,
    merchant_cat: usize,
    card_present: bool,
    geo_distance_km: f64,
    velocity: f64,
    v_features: [f64; 10],
    fraud: FraudType,
}

fn generate_transactions(n: usize, seed: u64) -> Vec<Transaction> {
    let mut rng = seed;
    let n_cards = 200;
    let mut txs = Vec::with_capacity(n);
    let mut card_last_hour = vec![0.0f64; n_cards];
    let mut card_tx_count = vec![0usize; n_cards];
    let mut card_last_geo = vec![0.0f64; n_cards];

    for i in 0..n {
        let card_id = (lcg_next(&mut rng) as usize) % n_cards;
        // Log-normal amount: median ~$50, mean ~$88
        let log_amt = 3.9 + lcg_normal(&mut rng) * 0.8;
        let amount = log_amt.exp().min(5000.0);
        // Time of day: bimodal (lunch 12h, evening 19h)
        let hour = if lcg_f64(&mut rng) < 0.6 {
            12.0 + lcg_normal(&mut rng) * 3.0
        } else {
            19.0 + lcg_normal(&mut rng) * 2.5
        }
        .rem_euclid(24.0);
        let merchant_cat = (lcg_next(&mut rng) as usize) % 10;
        let card_present = lcg_f64(&mut rng) < 0.7;
        let home_geo = (card_id as f64 * 7.3) % 180.0;
        let geo_distance_km = (lcg_f64(&mut rng) * 15.0).max(0.1);
        let dt = hour - card_last_hour[card_id];
        let velocity = if dt.abs() > 0.1 {
            card_tx_count[card_id] as f64 / dt.abs().max(0.5)
        } else {
            card_tx_count[card_id] as f64 + 1.0
        };

        // PCA-like features V1-V10 from real distribution shapes
        let mut vf = [0.0f64; 10];
        for k in 0..10 {
            vf[k] = lcg_normal(&mut rng) * (1.0 / (k as f64 + 1.0).sqrt());
        }

        card_last_hour[card_id] = hour;
        card_last_geo[card_id] = home_geo;
        card_tx_count[card_id] += 1;

        txs.push(Transaction {
            id: i,
            card_id,
            amount,
            hour,
            merchant_cat,
            card_present,
            geo_distance_km,
            velocity,
            v_features: vf,
            fraud: FraudType::Legit,
        });
    }

    // Inject fraud (~0.5% = ~10 transactions across 5 types)
    let fraud_types = [
        FraudType::CardNotPresent,
        FraudType::AccountTakeover,
        FraudType::CardClone,
        FraudType::Synthetic,
        FraudType::Refund,
    ];
    for ft in &fraud_types {
        let count = 2;
        for _ in 0..count {
            let idx = (lcg_next(&mut rng) as usize) % n;
            let tx = &mut txs[idx];
            tx.fraud = *ft;
            match ft {
                FraudType::CardNotPresent => {
                    tx.card_present = false;
                    tx.amount = (tx.amount * 3.5 + 200.0).min(4500.0);
                    tx.hour = 3.0 + lcg_f64(&mut rng) * 3.0; // 3-6 AM
                    tx.merchant_cat = 3; // online_retail
                    tx.v_features[0] += 2.5;
                    tx.v_features[1] -= 1.8;
                }
                FraudType::AccountTakeover => {
                    tx.velocity = 8.0 + lcg_f64(&mut rng) * 5.0; // burst
                    tx.amount = 500.0 + lcg_f64(&mut rng) * 2000.0;
                    tx.v_features[2] += 3.0;
                    tx.v_features[3] -= 2.0;
                }
                FraudType::CardClone => {
                    tx.geo_distance_km = 800.0 + lcg_f64(&mut rng) * 5000.0; // impossible travel
                    tx.card_present = true;
                    tx.v_features[4] += 2.8;
                    tx.v_features[5] -= 1.5;
                }
                FraudType::Synthetic => {
                    tx.amount = 50.0 + lcg_f64(&mut rng) * 150.0; // gradual escalation
                    tx.merchant_cat = 9; // luxury
                    tx.v_features[6] += 2.0;
                    tx.v_features[7] += 1.5;
                }
                FraudType::Refund => {
                    tx.amount = -(20.0 + lcg_f64(&mut rng) * 80.0); // refund (negative)
                    tx.v_features[8] -= 2.5;
                    tx.v_features[9] += 1.8;
                }
                _ => {}
            }
        }
    }
    txs
}

fn extract_embedding(tx: &Transaction, mean_amt: f64, std_amt: f64) -> Vec<f32> {
    let log_amt = (tx.amount.abs() + 1.0).ln();
    let hour_sin = (tx.hour * std::f64::consts::PI / 12.0).sin();
    let hour_cos = (tx.hour * std::f64::consts::PI / 12.0).cos();
    let mut cat_oh = [0.0f32; 10];
    cat_oh[tx.merchant_cat] = 1.0;
    let cp_flag = if tx.card_present { 1.0f32 } else { 0.0 };
    let geo_log = (tx.geo_distance_km + 1.0).ln();
    let vel_log = (tx.velocity + 1.0).ln();
    let amt_zscore = if std_amt > 1e-6 {
        (tx.amount - mean_amt) / std_amt
    } else {
        0.0
    };
    let amt_vel = log_amt * vel_log;
    let geo_time = geo_log * (tx.hour / 24.0);

    let mut emb = Vec::with_capacity(DIM);
    emb.push(log_amt as f32); // 0
    emb.push(hour_sin as f32); // 1
    emb.push(hour_cos as f32); // 2
    for v in &cat_oh {
        emb.push(*v);
    } // 3-12
    emb.push(cp_flag); // 13
    emb.push(geo_log as f32); // 14
    emb.push(vel_log as f32); // 15
    emb.push(amt_zscore as f32); // 16
    for k in 0..10 {
        emb.push(tx.v_features[k] as f32);
    } // 17-26
    emb.push(amt_vel as f32); // 27
    emb.push(geo_time as f32); // 28
    emb.push(if tx.amount < 0.0 { 1.0 } else { 0.0 }); // 29 refund flag
    emb.push((tx.id as f64 / N_TX as f64) as f32); // 30 temporal position
    while emb.len() < DIM {
        emb.push(0.0);
    }
    emb.truncate(DIM);
    emb
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

fn unary_anomaly_score(tx: &Transaction, mean_amt: f64, std_amt: f64) -> f64 {
    let amt_dev = ((tx.amount - mean_amt) / std_amt.max(1e-6)).abs();
    let time_anom = if tx.hour < 5.0 || tx.hour > 23.0 {
        0.8
    } else {
        0.0
    };
    let vel_spike = if tx.velocity > 5.0 {
        (tx.velocity - 5.0) * 0.3
    } else {
        0.0
    };
    let geo_imp = if tx.geo_distance_km > 500.0 {
        (tx.geo_distance_km / 1000.0).min(2.0)
    } else {
        0.0
    };
    let refund_flag = if tx.amount < 0.0 { 0.6 } else { 0.0 };
    0.4 * amt_dev + 0.5 * time_anom + 0.6 * vel_spike + 0.8 * geo_imp + refund_flag - 0.7
}

struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

fn build_graph(
    txs: &[Transaction],
    embs: &[Vec<f32>],
    alpha: f64,
    beta: f64,
    k: usize,
) -> Vec<Edge> {
    let m = txs.len();
    let mut edges = Vec::new();

    // Temporal chain: consecutive transactions by same card
    let mut by_card: Vec<Vec<usize>> = vec![Vec::new(); 200];
    for (i, tx) in txs.iter().enumerate() {
        by_card[tx.card_id].push(i);
    }
    for card_txs in &by_card {
        for w in card_txs.windows(2) {
            edges.push(Edge {
                from: w[0],
                to: w[1],
                weight: alpha,
            });
            edges.push(Edge {
                from: w[1],
                to: w[0],
                weight: alpha,
            });
        }
    }

    // Merchant-category edges: connect same-merchant transactions (sampled)
    let mut by_merchant: Vec<Vec<usize>> = vec![Vec::new(); 10];
    for (i, tx) in txs.iter().enumerate() {
        by_merchant[tx.merchant_cat].push(i);
    }
    for mtxs in &by_merchant {
        for w in mtxs.windows(2) {
            edges.push(Edge {
                from: w[0],
                to: w[1],
                weight: alpha * 0.3,
            });
            edges.push(Edge {
                from: w[1],
                to: w[0],
                weight: alpha * 0.3,
            });
        }
    }

    // kNN similarity edges from embeddings
    for i in 0..m {
        let mut sims: Vec<(usize, f64)> = (0..m)
            .filter(|&j| j != i)
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

fn threshold_detect(txs: &[Transaction], mean_amt: f64, std_amt: f64) -> Vec<bool> {
    txs.iter()
        .map(|tx| {
            let amt_z = ((tx.amount - mean_amt) / std_amt.max(1e-6)).abs();
            amt_z > 3.0 || tx.velocity > 5.0 || tx.geo_distance_km > 500.0 || tx.amount < 0.0
        })
        .collect()
}

struct Metrics {
    precision: f64,
    recall: f64,
    f1: f64,
    fpr: f64,
}

fn evaluate(txs: &[Transaction], preds: &[bool]) -> Metrics {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0, 0, 0);
    for (tx, &p) in txs.iter().zip(preds) {
        let truth = tx.fraud != FraudType::Legit;
        match (p, truth) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let fpr = if tn + fp > 0 {
        fp as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    Metrics {
        precision,
        recall,
        f1,
        fpr,
    }
}

fn per_type_detection(txs: &[Transaction], preds: &[bool]) -> Vec<(&'static str, usize, usize)> {
    let types = [
        FraudType::CardNotPresent,
        FraudType::AccountTakeover,
        FraudType::CardClone,
        FraudType::Synthetic,
        FraudType::Refund,
    ];
    types
        .iter()
        .map(|ft| {
            let total = txs.iter().filter(|tx| tx.fraud == *ft).count();
            let detected = txs
                .iter()
                .zip(preds)
                .filter(|(tx, &p)| tx.fraud == *ft && p)
                .count();
            (ft.label(), detected, total)
        })
        .collect()
}

fn amount_bucket(amt: f64) -> &'static str {
    let a = amt.abs();
    if a < 25.0 {
        "micro"
    } else if a < 100.0 {
        "small"
    } else if a < 500.0 {
        "medium"
    } else if a < 2000.0 {
        "large"
    } else {
        "xlarge"
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    println!("=== Financial Fraud Detection via Graph Cut / MRF ===\n");

    let (alpha, beta, gamma, k_nn) = (0.25, 0.12, 0.35, 3usize);
    let seed = 42u64;

    // Generate transactions
    let txs = generate_transactions(N_TX, seed);
    let n_fraud = txs.iter().filter(|tx| tx.fraud != FraudType::Legit).count();
    let fraud_rate = n_fraud as f64 / N_TX as f64 * 100.0;
    println!(
        "  Transactions: {} | Fraud: {} ({:.2}%) | Cards: 200\n",
        N_TX, n_fraud, fraud_rate
    );

    // Distribution stats
    let amounts: Vec<f64> = txs.iter().map(|tx| tx.amount).collect();
    let mean_amt = amounts.iter().sum::<f64>() / amounts.len() as f64;
    let std_amt =
        (amounts.iter().map(|a| (a - mean_amt).powi(2)).sum::<f64>() / amounts.len() as f64).sqrt();
    let pos_amounts: Vec<f64> = amounts.iter().filter(|&&a| a > 0.0).cloned().collect();
    let mut sorted_pos = pos_amounts.clone();
    sorted_pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_pos[sorted_pos.len() / 2];
    println!(
        "  Amount stats: median=${:.0}, mean=${:.0}, std=${:.0}",
        median, mean_amt, std_amt
    );

    println!("  Fraud breakdown:");
    for ft in &[
        FraudType::CardNotPresent,
        FraudType::AccountTakeover,
        FraudType::CardClone,
        FraudType::Synthetic,
        FraudType::Refund,
    ] {
        let c = txs.iter().filter(|tx| tx.fraud == *ft).count();
        if c > 0 {
            println!("    {}: {}", ft.label(), c);
        }
    }

    // Extract embeddings
    let embs: Vec<Vec<f32>> = txs
        .iter()
        .map(|tx| extract_embedding(tx, mean_amt, std_amt))
        .collect();

    // Unary anomaly scores
    let lambdas: Vec<f64> = txs
        .iter()
        .map(|tx| unary_anomaly_score(tx, mean_amt, std_amt))
        .collect();

    // Build graph and solve mincut
    println!("\n  Building transaction graph...");
    let edges = build_graph(&txs, &embs, alpha, beta, k_nn);
    println!("  Graph: {} nodes, {} edges", N_TX, edges.len());

    let gc_preds = solve_mincut(&lambdas, &edges, gamma);
    let gc_flagged = gc_preds.iter().filter(|&&p| p).count();

    let th_preds = threshold_detect(&txs, mean_amt, std_amt);
    let th_flagged = th_preds.iter().filter(|&&p| p).count();

    // Evaluate
    let gc_m = evaluate(&txs, &gc_preds);
    let th_m = evaluate(&txs, &th_preds);

    println!(
        "\n  {:>12} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "Method", "Prec", "Recall", "F1", "FPR", "Flagged"
    );
    println!(
        "  {:->12} {:->9} {:->9} {:->9} {:->9} {:->7}",
        "", "", "", "", "", ""
    );
    println!(
        "  {:>12} {:>9.3} {:>9.3} {:>9.3} {:>9.4} {:>7}",
        "Graph Cut", gc_m.precision, gc_m.recall, gc_m.f1, gc_m.fpr, gc_flagged
    );
    println!(
        "  {:>12} {:>9.3} {:>9.3} {:>9.3} {:>9.4} {:>7}",
        "Threshold", th_m.precision, th_m.recall, th_m.f1, th_m.fpr, th_flagged
    );

    let f1_imp = if th_m.f1 > 0.0 {
        (gc_m.f1 - th_m.f1) / th_m.f1 * 100.0
    } else {
        0.0
    };
    let fpr_imp = if th_m.fpr > 0.0 {
        (th_m.fpr - gc_m.fpr) / th_m.fpr * 100.0
    } else {
        0.0
    };
    println!(
        "\n  Graph cut vs threshold: F1 {:+.1}%, FPR reduction {:.1}%",
        f1_imp, fpr_imp
    );

    // Per-type detection
    println!("\n  Per-fraud-type detection:");
    println!("  {:>8} {:>12} {:>12}", "Type", "GraphCut", "Threshold");
    println!("  {:->8} {:->12} {:->12}", "", "", "");
    let gc_types = per_type_detection(&txs, &gc_preds);
    let th_types = per_type_detection(&txs, &th_preds);
    for (gc_t, th_t) in gc_types.iter().zip(th_types.iter()) {
        println!(
            "  {:>8} {:>5}/{:<5} {:>5}/{:<5}",
            gc_t.0, gc_t.1, gc_t.2, th_t.1, th_t.2
        );
    }

    // RVF ingestion
    println!("\n--- RVF Ingestion ---");
    let tmp = TempDir::new().expect("tmpdir");
    let opts = RvfOptions {
        dimension: DIM as u16,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut store = RvfStore::create(&tmp.path().join("fraud.rvfin"), opts).expect("create");

    let mut vecs: Vec<Vec<f32>> = Vec::new();
    let mut ids: Vec<u64> = Vec::new();
    let mut meta: Vec<MetadataEntry> = Vec::new();

    for (i, tx) in txs.iter().enumerate() {
        vecs.push(embs[i].clone());
        ids.push(i as u64);
        meta.push(MetadataEntry {
            field_id: FIELD_MERCHANT,
            value: MetadataValue::String(MERCHANTS[tx.merchant_cat].into()),
        });
        meta.push(MetadataEntry {
            field_id: FIELD_AMOUNT_BUCKET,
            value: MetadataValue::String(amount_bucket(tx.amount).into()),
        });
        meta.push(MetadataEntry {
            field_id: FIELD_FRAUD_TYPE,
            value: MetadataValue::String(tx.fraud.label().into()),
        });
    }

    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let ing = store
        .ingest_batch(&refs, &ids, Some(&meta))
        .expect("ingest");
    println!("  Ingested: {} (rejected: {})", ing.accepted, ing.rejected);

    // Filtered queries: retrieve similar fraud patterns by merchant
    println!("\n--- Filtered Queries ---");
    let fraud_idx = txs
        .iter()
        .position(|tx| tx.fraud != FraudType::Legit)
        .unwrap_or(0);
    let qv = &embs[fraud_idx];
    let res = store.query(qv, 10, &QueryOptions::default()).expect("q");
    println!(
        "  Similar to fraud tx #{}: {} results",
        fraud_idx,
        res.len()
    );
    for r in res.iter().take(5) {
        let tx = &txs[r.id as usize];
        println!(
            "    id={:>5} dist={:.4} merchant={:<15} fraud={}",
            r.id,
            r.distance,
            MERCHANTS[tx.merchant_cat],
            tx.fraud.label()
        );
    }

    let fm = FilterExpr::Eq(FIELD_MERCHANT, FilterValue::String("online_retail".into()));
    let mr = store
        .query(
            qv,
            5,
            &QueryOptions {
                filter: Some(fm),
                ..Default::default()
            },
        )
        .expect("q");
    println!("  Online-retail only: {} results", mr.len());

    let ff = FilterExpr::Eq(FIELD_AMOUNT_BUCKET, FilterValue::String("large".into()));
    let lr = store
        .query(
            qv,
            5,
            &QueryOptions {
                filter: Some(ff),
                ..Default::default()
            },
        )
        .expect("q");
    println!("  Large-amount only: {} results", lr.len());

    // Witness chain: audit trail
    println!("\n--- Witness Chain (Audit Trail) ---");
    let steps = [
        ("tx_ingest", 0x08u8),
        ("feature_extract", 0x02),
        ("graph_build", 0x02),
        ("mincut", 0x02),
        ("classify", 0x02),
        ("alert", 0x01),
        ("seal", 0x01),
    ];
    let entries: Vec<WitnessEntry> = steps
        .iter()
        .enumerate()
        .map(|(i, (step, wt))| WitnessEntry {
            prev_hash: [0u8; 32],
            action_hash: shake256_256(format!("fraud_gc:{}:{}", step, i).as_bytes()),
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
            &tmp.path().join("fraud_report.rvfin"),
            DerivationType::Filter,
            None,
        )
        .expect("derive");
    println!(
        "  parent: {} -> child: {} (depth {})",
        hex(store.file_id()),
        hex(child.parent_id()),
        child.lineage_depth()
    );
    child.close().expect("close");

    // Summary
    println!("\n=== Summary ===");
    println!(
        "  {} transactions, {} fraud ({:.2}%)",
        N_TX, n_fraud, fraud_rate
    );
    println!(
        "  Graph cut:  prec={:.3} recall={:.3} F1={:.3} FPR={:.4}",
        gc_m.precision, gc_m.recall, gc_m.f1, gc_m.fpr
    );
    println!(
        "  Threshold:  prec={:.3} recall={:.3} F1={:.3} FPR={:.4}",
        th_m.precision, th_m.recall, th_m.f1, th_m.fpr
    );
    println!(
        "  RVF: {} embeddings | Witness: {} steps",
        ing.accepted,
        verified.len()
    );
    println!(
        "  alpha={:.2} beta={:.2} gamma={:.2} k={}",
        alpha, beta, gamma, k_nn
    );
    store.close().expect("close");
    println!("\nDone.");
}
