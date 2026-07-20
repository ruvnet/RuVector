//! Benchmark: association-native multi-lens retrieval vs single-embedding RAG.
//!
//! Within a topic every document shares the same *semantic* region ("fog"); the
//! discriminating signal lives in two minority lenses (a lexical fingerprint and
//! a structural class) that a flattened embedding drowns out but an
//! association-native store keeps as full-weight ranking lists. Unanswerable
//! queries (right topic, specifics matching no doc) test fail-closed abstention.
//! Acceptance targets (ADR-272): grounded accuracy ≥+15pp, unsupported claims
//! ≥−50%, recall@10 ≥+10pp, replayability 100%. Run with `cargo run --release`.

use ruvector_calyx::rng::Rng;
use ruvector_calyx::{
    anneal_weights, recall_at_k, signal_density, AnnealConfig, CalyxEngine, Constellation,
    ConstellationStore, GuardProfile, LensKind, LensManifest, Observation, Query,
};
use ruvector_calyx::assay::LensSignal;
use ruvector_calyx::fusion::{weighted_rrf, DEFAULT_RRF_K};
use ruvector_calyx::ledger::{grounded_path, Anchor, AnchorKind};
use ruvector_calyx::loom::min_agreement;
use ruvector_calyx::ward::{adjudicate, Candidate, GuardDecision};
use std::time::Instant;

// ── Dataset parameters ─────────────────────────────────────────────────────
const DIM_SEM: usize = 48;
const DIM_LEX: usize = 32;
const DIM_STR: usize = 16;
const N_TOPICS: usize = 12;
const N_PER_TOPIC: usize = 20;
const N_DOCS: usize = N_TOPICS * N_PER_TOPIC; // 240
const N_STRUCT_CLASSES: usize = 8;
const K: usize = 10;

// Query parameters
const N_ANSWERABLE: usize = 120;
const N_UNANSWERABLE: usize = 60;

// Lens economics (microseconds to measure one object). The dense semantic
// model is the expensive one; lexical / structural are cheap.
const COST_SEM: f32 = 12.0;
const COST_LEX: f32 = 2.0;
const COST_STR: f32 = 1.5;

const SEED: u64 = 42;

// ── Vector helpers ──────────────────────────────────────────────────────────
fn unit_gaussian(rng: &mut Rng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.signed()).collect();
    normalize(&v)
}
fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}
fn perturb(c: &[f32], noise: f32, rng: &mut Rng) -> Vec<f32> {
    let g = unit_gaussian(rng, c.len());
    let mixed: Vec<f32> = c.iter().zip(g.iter()).map(|(a, b)| a + noise * b).collect();
    normalize(&mixed)
}
fn blend(a: &[f32], b: &[f32], wa: f32, wb: f32) -> Vec<f32> {
    normalize(
        &a.iter()
            .zip(b.iter())
            .map(|(x, y)| wa * x + wb * y)
            .collect::<Vec<_>>(),
    )
}

// ── Dataset ─────────────────────────────────────────────────────────────────
struct Doc {
    id: u64,
    topic: usize,
    sem: Vec<f32>,
    lex: Vec<f32>,
    str_: Vec<f32>,
}

struct QuerySpec {
    sem: Vec<f32>,
    lex: Vec<f32>,
    str_: Vec<f32>,
    /// `Some(target_id)` for answerable queries; `None` if unanswerable.
    target: Option<u64>,
}

fn registry() -> Vec<LensManifest> {
    vec![
        LensManifest::new("semantic", "frozen-v1", LensKind::DenseSemantic, DIM_SEM, COST_SEM, COST_SEM),
        LensManifest::new("lexical", "frozen-v1", LensKind::Lexical, DIM_LEX, COST_LEX, COST_LEX),
        LensManifest::new("structural", "frozen-v1", LensKind::Structural, DIM_STR, COST_STR, COST_STR),
    ]
}

fn build_dataset(rng: &mut Rng) -> (Vec<Doc>, Vec<QuerySpec>) {
    let sem_centroids: Vec<Vec<f32>> = (0..N_TOPICS).map(|_| unit_gaussian(rng, DIM_SEM)).collect();
    let lex_centroids: Vec<Vec<f32>> = (0..N_TOPICS).map(|_| unit_gaussian(rng, DIM_LEX)).collect();
    let str_centroids: Vec<Vec<f32>> =
        (0..N_STRUCT_CLASSES).map(|_| unit_gaussian(rng, DIM_STR)).collect();

    let mut docs = Vec::with_capacity(N_DOCS);
    let mut id = 0u64;
    for t in 0..N_TOPICS {
        for j in 0..N_PER_TOPIC {
            let s = (t * N_PER_TOPIC + j) % N_STRUCT_CLASSES;
            // Semantic slot: topic fog — every doc in the topic is near-identical.
            let sem = perturb(&sem_centroids[t], 0.10, rng);
            // Lexical slot: topic flavour + a per-doc fingerprint (doc-level signal).
            let lex_fp = unit_gaussian(rng, DIM_LEX);
            let lex = blend(&lex_centroids[t], &lex_fp, 0.5, 0.85);
            // Structural slot: class centroid + per-doc jitter (doc-level signal).
            let str_ = perturb(&str_centroids[s], 0.30, rng);
            docs.push(Doc {
                id,
                topic: t,
                sem,
                lex,
                str_,
            });
            id += 1;
        }
    }

    // Answerable queries: pick a target doc; semantic comes from the *topic*
    // centroid (so semantic-only can't isolate the target), lexical & structural
    // are derived from the target (so the minority lenses pin it down).
    let mut queries = Vec::with_capacity(N_ANSWERABLE + N_UNANSWERABLE);
    for i in 0..N_ANSWERABLE {
        let d = &docs[(i * 7 + 3) % N_DOCS]; // spread targets across the corpus
        let sem = perturb(&sem_centroids[d.topic], 0.10, rng);
        let lex = perturb(&d.lex, 0.10, rng);
        let str_ = perturb(&d.str_, 0.06, rng);
        queries.push(QuerySpec {
            sem,
            lex,
            str_,
            target: Some(d.id),
        });
    }
    // Unanswerable queries: a real topic, but lexical & structural fingerprints
    // that match no document. The right answer is to abstain.
    for i in 0..N_UNANSWERABLE {
        let t = i % N_TOPICS;
        let sem = perturb(&sem_centroids[t], 0.10, rng);
        let lex = unit_gaussian(rng, DIM_LEX);
        let str_ = unit_gaussian(rng, DIM_STR);
        queries.push(QuerySpec {
            sem,
            lex,
            str_,
            target: None,
        });
    }
    (docs, queries)
}

fn build_store(docs: &[Doc]) -> ConstellationStore {
    let mut store = ConstellationStore::new(registry());
    for d in docs {
        // Every real document carries a grounding anchor (a passed test / label).
        let c = Constellation::new(d.id)
            .with_label(format!("topic{}", d.topic))
            .with_slot("semantic", d.sem.clone())
            .with_slot("lexical", d.lex.clone())
            .with_slot("structural", d.str_.clone())
            .with_anchor(Anchor::new(AnchorKind::PassedTest, format!("test-{}", d.id), 0.9));
        store.insert(c).expect("valid constellation");
    }
    store
}

fn to_query(q: &QuerySpec) -> Query {
    Query::new()
        .with_slot("semantic", q.sem.clone())
        .with_slot("lexical", q.lex.clone())
        .with_slot("structural", q.str_.clone())
}

// ── Metrics ─────────────────────────────────────────────────────────────────
#[derive(Default, Clone)]
struct Metrics {
    grounded_correct: usize, // answered AND correct (answerable only)
    answerable: usize,
    unsupported: usize, // answered but wrong, OR answered an unanswerable query
    recall_sum: f32,
    recall_n: usize,
    abstained: usize,
}

impl Metrics {
    fn grounded_accuracy(&self) -> f32 {
        if self.answerable == 0 {
            0.0
        } else {
            self.grounded_correct as f32 / self.answerable as f32
        }
    }
    fn recall_at_10(&self) -> f32 {
        if self.recall_n == 0 {
            0.0
        } else {
            self.recall_sum / self.recall_n as f32
        }
    }
}

/// Single-embedding RAG baseline: only the dense semantic lens, always answers.
fn run_baseline(store: &ConstellationStore, queries: &[QuerySpec]) -> Metrics {
    let mut m = Metrics::default();
    for q in queries {
        let ranked = store.search_lens("semantic", &q.sem, K);
        let top = ranked.first().map(|&(id, _)| id);
        match q.target {
            Some(target) => {
                m.answerable += 1;
                if top == Some(target) {
                    m.grounded_correct += 1;
                } else {
                    m.unsupported += 1;
                }
                let ids: Vec<u64> = ranked.iter().map(|&(id, _)| id).collect();
                m.recall_sum += recall_at_k(&[target], &ids);
                m.recall_n += 1;
            }
            None => {
                // Always answers → unsupported claim on an unanswerable query.
                if top.is_some() {
                    m.unsupported += 1;
                }
            }
        }
    }
    m
}

/// Pure association-native compose path for a single query (no ledger), used by
/// both evaluation and the anneal objective.
fn compose(
    store: &ConstellationStore,
    weights: &[f32],
    guard: &GuardProfile,
    q: &QuerySpec,
) -> (GuardDecision, Vec<u64>) {
    let qq = to_query(q);
    let lenses = store.lenses();
    let mut rankings = Vec::with_capacity(lenses.len());
    for m in lenses {
        let qv = qq.slots.get(&m.name).unwrap();
        rankings.push(store.search_lens(&m.name, qv, K));
    }
    let fused = weighted_rrf(&rankings, weights, DEFAULT_RRF_K);
    let ids: Vec<u64> = fused.iter().map(|&(id, _)| id).collect();
    let candidate = fused.first().map(|&(id, score)| {
        let rec = store.get(id).unwrap();
        Candidate {
            record_id: id,
            fusion_score: score,
            cross_lens_agreement: min_agreement(&qq.slots, rec),
            grounding: grounded_path(id, &rec.anchors, guard.min_grounding_weight),
        }
    });
    (adjudicate(candidate, guard), ids)
}

fn run_calyx(
    store: &ConstellationStore,
    weights: &[f32],
    guard: &GuardProfile,
    queries: &[QuerySpec],
) -> Metrics {
    let mut m = Metrics::default();
    for q in queries {
        let (decision, ids) = compose(store, weights, guard, q);
        let answered = decision.answered_id();
        match q.target {
            Some(target) => {
                m.answerable += 1;
                match answered {
                    Some(a) if a == target => m.grounded_correct += 1,
                    Some(_) => m.unsupported += 1, // answered, but wrong
                    None => m.abstained += 1,
                }
                m.recall_sum += recall_at_k(&[target], &ids);
                m.recall_n += 1;
            }
            None => match answered {
                Some(_) => m.unsupported += 1,
                None => m.abstained += 1,
            },
        }
    }
    m
}

// ── Reporting ───────────────────────────────────────────────────────────────
fn pct(x: f32) -> String {
    format!("{:.1}%", x * 100.0)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-calyx — Association-Native Memory Benchmark (ADR-272) ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Platform : {} / {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("Lenses   : semantic({DIM_SEM}d, {COST_SEM}µs) · lexical({DIM_LEX}d, {COST_LEX}µs) · structural({DIM_STR}d, {COST_STR}µs)");
    println!("Corpus   : {N_DOCS} docs across {N_TOPICS} topics × {N_PER_TOPIC} (semantic fog) · {N_STRUCT_CLASSES} structural classes");
    println!("Queries  : {N_ANSWERABLE} answerable + {N_UNANSWERABLE} unanswerable\n");

    let mut rng = Rng::new(SEED);
    let (docs, queries) = build_dataset(&mut rng);
    let store = build_store(&docs);

    // ── Signal density (Assay) ──────────────────────────────────────────────
    let signals = measure_signal_density(&store, &queries);
    println!("Signal density (Assay) — information each lens adds, per µs of cost");
    println!("  {:<12} {:>8} {:>10} {:>14}", "lens", "bits", "cost µs", "bits/µs");
    for s in &signals {
        println!(
            "  {:<12} {:>8.3} {:>10.1} {:>14.4}",
            s.lens, s.bits, s.cost_us, s.density
        );
    }
    println!();

    // ── Baseline vs Calyx (uniform weights) ─────────────────────────────────
    let guard = GuardProfile::default();
    let uniform = vec![1.0_f32; 3];

    let t0 = Instant::now();
    let base = run_baseline(&store, &queries);
    let base_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let calyx = run_calyx(&store, &uniform, &guard, &queries);
    let calyx_us = t1.elapsed().as_micros();

    // Replayability: run the *real* engine (with ledger) over all queries.
    let mut engine = CalyxEngine::new(store.clone())
        .with_weights(uniform.clone())
        .with_guard(guard.clone())
        .with_top_k(K);
    for q in &queries {
        let _ = engine.query(&to_query(q));
    }
    let replayable = engine.ledger().verify();
    let ledger_entries = engine.ledger().len();

    println!("Results (uniform fusion weights)");
    println!(
        "  {:<28} {:>16} {:>16}",
        "metric", "single-embedding", "calyx (multi-lens)"
    );
    println!("  {}", "-".repeat(62));
    println!(
        "  {:<28} {:>16} {:>16}",
        "grounded accuracy",
        pct(base.grounded_accuracy()),
        pct(calyx.grounded_accuracy())
    );
    println!(
        "  {:<28} {:>16} {:>16}",
        "recall@10 (answerable)",
        pct(base.recall_at_10()),
        pct(calyx.recall_at_10())
    );
    println!(
        "  {:<28} {:>16} {:>16}",
        "unsupported claims (count)", base.unsupported, calyx.unsupported
    );
    println!(
        "  {:<28} {:>16} {:>16}",
        "abstentions (count)", 0, calyx.abstained
    );
    println!(
        "  {:<28} {:>16} {:>16}",
        "latency (µs, all queries)", base_us, calyx_us
    );
    println!(
        "  {:<28} {:>16} {:>16}",
        "replayable provenance", "n/a", format!("{replayable} ({ledger_entries})")
    );
    println!();

    // ── Anneal: reversibly tune fusion weights for accepted-answers-per-cost ─
    let lens_cost = [COST_SEM, COST_LEX, COST_STR];
    let cfg = AnnealConfig::default();
    let store_for_anneal = store.clone();
    let guard_for_anneal = guard.clone();
    let queries_ref = &queries;
    let outcome = anneal_weights(uniform.clone(), &cfg, |w| {
        objective(&store_for_anneal, w, &guard_for_anneal, queries_ref, &lens_cost)
    });
    let tuned = run_calyx(&store, &outcome.best_weights, &guard, &queries);
    println!("Anneal — reversible fusion-weight optimization (accepted answers / cost)");
    println!(
        "  weights   {:?} → [{:.2}, {:.2}, {:.2}]",
        uniform, outcome.best_weights[0], outcome.best_weights[1], outcome.best_weights[2]
    );
    println!(
        "  utility   {:.4} → {:.4}   (accepted={}, reverted={})",
        outcome.initial_utility, outcome.best_utility, outcome.accepted_moves, outcome.reverted_moves
    );
    println!(
        "  tuned     grounded accuracy {}  ·  unsupported {}  ·  abstentions {}",
        pct(tuned.grounded_accuracy()),
        tuned.unsupported,
        tuned.abstained
    );
    println!();

    // ── Acceptance test ─────────────────────────────────────────────────────
    let acc_gain = calyx.grounded_accuracy() - base.grounded_accuracy();
    let recall_gain = calyx.recall_at_10() - base.recall_at_10();
    let claim_reduction = if base.unsupported == 0 {
        0.0
    } else {
        1.0 - (calyx.unsupported as f32 / base.unsupported as f32)
    };

    println!("Acceptance test (ADR-272 targets)");
    let c1 = acc_gain >= 0.15;
    let c2 = claim_reduction >= 0.50;
    let c3 = recall_gain >= 0.10;
    let c4 = replayable && ledger_entries == queries.len();
    let c5 = outcome.best_utility >= outcome.initial_utility && outcome.reverted_moves > 0;
    print_check("grounded accuracy +≥15pp", &format!("+{:.1}pp", acc_gain * 100.0), c1);
    print_check("unsupported claims −≥50%", &format!("−{:.1}%", claim_reduction * 100.0), c2);
    print_check("recall@10 +≥10pp", &format!("+{:.1}pp", recall_gain * 100.0), c3);
    print_check("replayable provenance 100%", &format!("{ledger_entries}/{}", queries.len()), c4);
    print_check("anneal improves & is reversible", &format!("Δu={:.4}", outcome.best_utility - outcome.initial_utility), c5);
    println!();

    if c1 && c2 && c3 && c4 && c5 {
        println!("→ BENCHMARK PASSED");
    } else {
        println!("→ BENCHMARK FAILED");
        std::process::exit(1);
    }
}

fn print_check(name: &str, value: &str, pass: bool) {
    println!(
        "  {:<34} {:>12}   {}",
        name,
        value,
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
}

/// Anneal objective: accepted-correct answers divided by normalized
/// measurement cost. A lens whose fusion weight drops to ~0 is "skipped",
/// saving its cost — so the optimizer trades accuracy against compute.
fn objective(
    store: &ConstellationStore,
    weights: &[f32],
    guard: &GuardProfile,
    queries: &[QuerySpec],
    lens_cost: &[f32],
) -> f32 {
    let m = run_calyx(store, weights, guard, queries);
    let used_cost: f32 = weights
        .iter()
        .zip(lens_cost.iter())
        .map(|(w, c)| if *w > 0.05 { *c } else { 0.0 })
        .sum();
    let cost_norm = 1.0 + used_cost / 16.0;
    // Reward correct accepted answers; the guard already suppresses wrong ones.
    (m.grounded_correct as f32) / cost_norm
}

/// Estimate per-lens signal density from the answerable queries: for each lens,
/// gather (similarity-to-target, relevant) observations against a few sampled
/// records and measure mutual information per µs.
fn measure_signal_density(store: &ConstellationStore, queries: &[QuerySpec]) -> Vec<LensSignal> {
    let lenses = store.lenses().to_vec();
    let mut out = Vec::new();
    for m in &lenses {
        let mut obs = Vec::new();
        for q in queries.iter().filter(|q| q.target.is_some()) {
            let target = q.target.unwrap();
            let qv = match m.name.as_str() {
                "semantic" => &q.sem,
                "lexical" => &q.lex,
                _ => &q.str_,
            };
            // Positive: the true target. Negatives: a handful of other docs.
            for (offset, rec) in store.records().iter().enumerate() {
                if offset % 9 != 0 && rec.id != target {
                    continue; // subsample negatives for speed
                }
                let sim = ruvector_calyx::cosine_sim(qv, rec.slot(&m.name).unwrap());
                obs.push(Observation {
                    similarity: sim,
                    relevant: rec.id == target,
                });
            }
        }
        out.push(signal_density(m.name.clone(), &obs, m.cost_us));
    }
    out
}
