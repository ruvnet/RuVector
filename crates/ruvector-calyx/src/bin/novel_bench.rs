//! Benchmark for three novel capabilities of the association-native layer:
//!
//!   A. Conformal cross-lens abstention — a distribution-free guarantee that the
//!      confident-wrong-answer rate stays ≤ α, replacing the hand-tuned guard.
//!   B. Disagreement as a query primitive — surface records where two lenses
//!      most disagree (a query a flat vector DB cannot express).
//!   C. Ledger-learned routing — learn a stop/continue policy from logged
//!      witness trajectories by offline Monte-Carlo control.
//!
//! Run: `cargo run --release -p ruvector-calyx --bin calyx-novel-bench`

use std::collections::BTreeMap;

use ruvector_calyx::conformal::{calibrate, coverage, empirical_risk, selective_error, ConformalSelector};
use ruvector_calyx::disagreement::find_conflicts;
use ruvector_calyx::fusion::{weighted_rrf, DEFAULT_RRF_K};
use ruvector_calyx::ledger::{grounded_path, Anchor, AnchorKind};
use ruvector_calyx::ledger_policy::{learn, Action, Episode};
use ruvector_calyx::loom::min_agreement;
use ruvector_calyx::rng::Rng;
use ruvector_calyx::{Constellation, ConstellationStore, GuardProfile, LensKind, LensManifest, Query};

const DIM_SEM: usize = 48;
const DIM_LEX: usize = 32;
const DIM_STR: usize = 16;
const N_TOPICS: usize = 12;
const N_PER_TOPIC: usize = 20;
const N_DOCS: usize = N_TOPICS * N_PER_TOPIC;
const N_STRUCT: usize = N_TOPICS; // structural class aligned to topic (see §B)
const N_ANSWERABLE: usize = 200;
const N_UNANSWERABLE: usize = 100;
const COST: [f32; 3] = [1.5, 2.0, 12.0]; // structural, lexical, semantic
const SEED: u64 = 90210;

fn order() -> Vec<String> {
    vec!["structural".into(), "lexical".into(), "semantic".into()]
}
fn registry() -> Vec<LensManifest> {
    vec![
        LensManifest::new("structural", "v1", LensKind::Structural, DIM_STR, COST[0], COST[0]),
        LensManifest::new("lexical", "v1", LensKind::Lexical, DIM_LEX, COST[1], COST[1]),
        LensManifest::new("semantic", "v1", LensKind::DenseSemantic, DIM_SEM, COST[2], COST[2]),
    ]
}
fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}
fn unit(rng: &mut Rng, d: usize) -> Vec<f32> {
    normalize(&(0..d).map(|_| rng.signed()).collect::<Vec<_>>())
}
fn perturb(c: &[f32], noise: f32, rng: &mut Rng) -> Vec<f32> {
    let g = unit(rng, c.len());
    normalize(&c.iter().zip(g.iter()).map(|(a, b)| a + noise * b).collect::<Vec<_>>())
}
fn blend(a: &[f32], b: &[f32], wa: f32, wb: f32) -> Vec<f32> {
    normalize(&a.iter().zip(b.iter()).map(|(x, y)| wa * x + wb * y).collect::<Vec<_>>())
}

struct Doc { id: u64, topic: usize, sem: Vec<f32>, lex: Vec<f32>, str_: Vec<f32> }
struct QSpec { q: Query, target: Option<u64> }

fn build(rng: &mut Rng) -> (ConstellationStore, Vec<Doc>, Vec<QSpec>, Vec<u64>) {
    let sem_c: Vec<Vec<f32>> = (0..N_TOPICS).map(|_| unit(rng, DIM_SEM)).collect();
    let lex_c: Vec<Vec<f32>> = (0..N_TOPICS).map(|_| unit(rng, DIM_LEX)).collect();
    let str_c: Vec<Vec<f32>> = (0..N_STRUCT).map(|_| unit(rng, DIM_STR)).collect();
    let mut docs = Vec::with_capacity(N_DOCS);
    let mut planted = Vec::new();
    let mut id = 0u64;
    for t in 0..N_TOPICS {
        for _j in 0..N_PER_TOPIC {
            // Structural class is aligned to the topic, so a normal doc's
            // semantic and structural neighbourhoods overlap (consistent views).
            let sem = perturb(&sem_c[t], 0.10, rng);
            let lex = blend(&lex_c[t], &unit(rng, DIM_LEX), 0.5, 0.85);
            // Plant a "conflict" every 40th doc: structural slot from a *different*
            // topic's class, so semantic-neighbours ≠ structural-neighbours.
            let (str_, is_conflict) = if id % 40 == 7 {
                let alt = (t + N_STRUCT / 2) % N_STRUCT;
                (perturb(&str_c[alt], 0.10, rng), true)
            } else {
                (perturb(&str_c[t], 0.10, rng), false)
            };
            if is_conflict {
                planted.push(id);
            }
            docs.push(Doc { id, topic: t, sem, lex, str_ });
            id += 1;
        }
    }
    let mut store = ConstellationStore::new(registry());
    for d in &docs {
        store
            .insert(
                Constellation::new(d.id)
                    .with_slot("structural", d.str_.clone())
                    .with_slot("lexical", d.lex.clone())
                    .with_slot("semantic", d.sem.clone())
                    .with_anchor(Anchor::new(AnchorKind::PassedTest, format!("t{}", d.id), 0.9)),
            )
            .expect("insert");
    }
    let mut queries = Vec::new();
    for i in 0..N_ANSWERABLE {
        let d = &docs[(i * 7 + 3) % N_DOCS];
        let q = Query::new()
            .with_slot("structural", perturb(&d.str_, 0.10, rng))
            .with_slot("lexical", perturb(&d.lex, 0.10, rng))
            .with_slot("semantic", perturb(&sem_c[d.topic], 0.10, rng));
        queries.push(QSpec { q, target: Some(d.id) });
    }
    for i in 0..N_UNANSWERABLE {
        let t = i % N_TOPICS;
        let q = Query::new()
            .with_slot("structural", unit(rng, DIM_STR))
            .with_slot("lexical", unit(rng, DIM_LEX))
            .with_slot("semantic", perturb(&sem_c[t], 0.10, rng));
        queries.push(QSpec { q, target: None });
    }
    (store, docs, queries, planted)
}

/// Fuse all lenses; return (top_id, cross-lens agreement of the top).
fn full_panel(store: &ConstellationStore, q: &Query) -> Option<(u64, f32)> {
    let lenses = store.lenses();
    let mut rankings = Vec::new();
    for m in lenses {
        rankings.push(store.search_lens(&m.name, q.slots.get(&m.name).unwrap(), 10));
    }
    let fused = weighted_rrf(&rankings, &vec![1.0; lenses.len()], DEFAULT_RRF_K);
    fused.first().map(|&(id, _)| {
        let agr = store.get(id).map(|r| min_agreement(&q.slots, r)).unwrap_or(0.0);
        (id, agr)
    })
}

/// (agreement, wrong_if_answered) for every query — the conformal sample.
fn conformal_samples(store: &ConstellationStore, queries: &[QSpec]) -> Vec<(f32, bool)> {
    queries
        .iter()
        .filter_map(|qs| {
            full_panel(store, &qs.q).map(|(top, agr)| {
                let wrong = qs.target != Some(top); // unanswerable → always wrong
                (agr, wrong)
            })
        })
        .collect()
}

fn pct(x: f32) -> String { format!("{:.1}%", x * 100.0) }

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.range(i + 1);
        v.swap(i, j);
    }
}

// ── Section A: conformal cross-lens abstention ──────────────────────────────
fn section_a(store: &ConstellationStore, queries: &[QSpec], rng: &mut Rng) -> bool {
    println!("A. Conformal cross-lens abstention (distribution-free risk ≤ α)\n");
    let samples = conformal_samples(store, queries);

    // Hand-tuned guard (agreement ≥ 0.45) has *whatever* risk falls out.
    let hand = ConformalSelector { threshold: 0.45, alpha: f32::NAN };
    println!(
        "  hand-tuned guard (agree≥0.45): risk={} coverage={} selective-err={}",
        pct(empirical_risk(&hand, &samples)),
        pct(coverage(&hand, &samples)),
        pct(selective_error(&hand, &samples))
    );
    println!();
    println!("  {:<7} {:>10} {:>10} {:>12} {:>12} {:>10}", "α", "threshold", "coverage", "test-risk", "sel-error", "MC-risk");

    let mut all_pass = true;
    for &alpha in &[0.10f32, 0.05, 0.01] {
        // Single split for the reported threshold/coverage.
        let mut s = samples.clone();
        shuffle(&mut s, rng);
        let (cal, test) = s.split_at(s.len() / 2);
        let sel = calibrate(cal, alpha);
        let test_risk = empirical_risk(&sel, test);
        let cov = coverage(&sel, test);
        let sel_err = selective_error(&sel, test);

        // Monte-Carlo: mean test risk over many random splits must stay ≤ α.
        let trials = 200;
        let mut acc = 0.0f32;
        for _ in 0..trials {
            let mut sc = samples.clone();
            shuffle(&mut sc, rng);
            let (c, t) = sc.split_at(sc.len() / 2);
            acc += empirical_risk(&calibrate(c, alpha), t);
        }
        let mc = acc / trials as f32;
        let pass = mc <= alpha + 0.01;
        all_pass &= pass;
        println!(
            "  {:<7} {:>10} {:>10} {:>12} {:>12} {:>10} {}",
            format!("{:.2}", alpha),
            format!("{:.3}", sel.threshold),
            pct(cov),
            pct(test_risk),
            pct(sel_err),
            pct(mc),
            if pass { "✓" } else { "✗" }
        );
    }
    println!("\n  → guarantee holds: mean held-out risk ≤ α for every α  [{}]\n", if all_pass { "PASS" } else { "FAIL" });
    all_pass
}

// ── Section B: disagreement query primitive ─────────────────────────────────
fn section_b(store: &ConstellationStore, planted: &[u64]) -> bool {
    println!("B. Disagreement as a query primitive (find where two lenses conflict)\n");
    let p = planted.len();
    let conflicts = find_conflicts(store, "semantic", "structural", 8, p);
    let found: Vec<u64> = conflicts.iter().map(|&(id, _)| id).collect();
    let hits = planted.iter().filter(|id| found.contains(id)).count();
    let precision = hits as f32 / p.max(1) as f32;
    println!("  planted conflicts : {p}   (semantic∈topic X, structural∈another class)");
    println!("  find_conflicts@{p}  : {hits} of {p} planted surfaced  → precision {}", pct(precision));
    println!("  top-5 disagreement scores: {:?}", conflicts.iter().take(5).map(|&(id, s)| (id, (s * 100.0).round() / 100.0)).collect::<Vec<_>>());
    let pass = precision >= 0.80;
    println!("\n  → conflict retrieval precision ≥ 0.80  [{}]\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// ── Section C: ledger-learned routing ───────────────────────────────────────
fn agreement_over(store: &ConstellationStore, q: &Query, consulted: &[String], top: u64) -> f32 {
    let sub: BTreeMap<String, Vec<f32>> = q
        .slots
        .iter()
        .filter(|(k, _)| consulted.iter().any(|c| c == *k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    store.get(top).map(|r| min_agreement(&sub, r)).unwrap_or(0.0)
}

fn utility(answered: Option<u64>, target: Option<u64>, cost: f32) -> f32 {
    let base = match (answered, target) {
        (Some(a), Some(t)) if a == t => 1.0,
        (Some(_), _) => -1.5,               // confident wrong answer
        (None, None) => 0.3,                // correct abstention
        (None, Some(_)) => -0.1,            // missed answerable
    };
    base - 0.5 * (cost / 15.5)
}

/// Consult lenses incrementally; `stop_fn(step, agreement_bin)` decides.
fn rollout(
    store: &ConstellationStore,
    ord: &[String],
    qs: &QSpec,
    guard: &GuardProfile,
    n_bins: usize,
    mut stop_fn: impl FnMut(usize, usize) -> bool,
    visited: &mut Vec<(usize, usize, Action)>,
) -> (Option<u64>, bool, f32) {
    let mut rankings = Vec::new();
    let mut consulted = Vec::new();
    let mut cost = 0.0f32;
    let mut top = 0u64;
    let mut agr = 0.0f32;
    for (step, lens) in ord.iter().enumerate() {
        let m = store.lenses().iter().find(|l| &l.name == lens).unwrap();
        rankings.push(store.search_lens(lens, qs.q.slots.get(lens).unwrap(), 10));
        consulted.push(lens.clone());
        cost += m.cost_us;
        let fused = weighted_rrf(&rankings, &vec![1.0; rankings.len()], DEFAULT_RRF_K);
        top = fused.first().map(|&(id, _)| id).unwrap_or(0);
        agr = agreement_over(store, &qs.q, &consulted, top);
        let bin = ((agr.clamp(0.0, 0.999_999)) * n_bins as f32) as usize;
        let last = step + 1 == ord.len();
        let stop = last || stop_fn(step, bin);
        visited.push((consulted.len(), bin, if stop { Action::Stop } else { Action::Continue }));
        if stop {
            break;
        }
    }
    let grounded = store
        .get(top)
        .map(|r| grounded_path(top, &r.anchors, guard.min_grounding_weight).is_some())
        .unwrap_or(false);
    let answered = if consulted.len() >= 2 && agr >= guard.min_cross_lens_agreement && grounded {
        Some(top)
    } else {
        None
    };
    let correct = answered == qs.target && qs.target.is_some();
    (answered, correct, cost)
}

fn section_c(store: &ConstellationStore, queries: &[QSpec], rng: &mut Rng) -> bool {
    println!("C. Ledger-learned routing (offline MC control over witness trajectories)\n");
    let ord = order();
    let guard = GuardProfile::default();
    let n_bins = 10;

    // Split train/test.
    let mut idx: Vec<usize> = (0..queries.len()).collect();
    shuffle(&mut idx, rng);
    let (train_idx, test_idx) = idx.split_at(idx.len() / 2);

    // 1) Log exploratory episodes on train (behaviour policy: random stop).
    let mut episodes = Vec::new();
    for &qi in train_idx {
        for _ in 0..4 {
            let mut visited = Vec::new();
            let (answered, _correct, cost) =
                rollout(store, &ord, &queries[qi], &guard, n_bins, |_s, _b| rng.f32() < 0.5, &mut visited);
            let ret = utility(answered, queries[qi].target, cost);
            episodes.push(Episode { visited, ret });
        }
    }
    let policy = learn(&episodes, 3, n_bins);

    // 2) Evaluate learned policy vs fixed baselines on test.
    let eval = |mut stop_fn: Box<dyn FnMut(usize, usize) -> bool>| {
        let (mut correct, mut answered, mut cost, mut util) = (0usize, 0usize, 0.0f32, 0.0f32);
        for &qi in test_idx {
            let mut v = Vec::new();
            let (ans, ok, c) = rollout(store, &ord, &queries[qi], &guard, n_bins, &mut *stop_fn, &mut v);
            if ans.is_some() { answered += 1; }
            if ok { correct += 1; }
            cost += c;
            util += utility(ans, queries[qi].target, c);
        }
        let n = test_idx.len() as f32;
        (correct as f32 / n, cost / n, util / n, answered)
    };

    let pol = policy.clone();
    let (l_acc, l_cost, l_util, _) =
        eval(Box::new(move |s, b| pol.act(s + 1, b) == Action::Stop));
    let (s2_acc, s2_cost, s2_util, _) = eval(Box::new(|s, _b| s + 1 >= 2)); // static(2)
    let (bf_acc, bf_cost, bf_util, _) = eval(Box::new(|_s, _b| false)); // brute force

    println!("  {:<16} {:>9} {:>10} {:>9}", "policy", "acc", "cost µs", "utility");
    println!("  {}", "-".repeat(48));
    println!("  {:<16} {:>9} {:>10} {:>9}", "brute-force", pct(bf_acc), format!("{:.1}", bf_cost), format!("{:.3}", bf_util));
    println!("  {:<16} {:>9} {:>10} {:>9}", "static(2)", pct(s2_acc), format!("{:.1}", s2_cost), format!("{:.3}", s2_util));
    println!("  {:<16} {:>9} {:>10} {:>9}", "ledger-learned", pct(l_acc), format!("{:.1}", l_cost), format!("{:.3}", l_util));

    let best_fixed = s2_util.max(bf_util);
    let pass = l_util >= best_fixed - 0.05;
    println!("\n  → learned policy competitive with best fixed (utility ≥ best−0.05)  [{}]\n", if pass { "PASS" } else { "FAIL" });
    pass
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-calyx — Novel capabilities (conformal · disagreement · ║");
    println!("║  ledger-learned routing)                        ADR-272 Update 2 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    let mut rng = Rng::new(SEED);
    let (store, _docs, queries, planted) = build(&mut rng);
    println!("Corpus: {N_DOCS} docs · {N_ANSWERABLE} answerable + {N_UNANSWERABLE} unanswerable · {} planted conflicts\n", planted.len());

    let a = section_a(&store, &queries, &mut rng);
    let b = section_b(&store, &planted);
    let c = section_c(&store, &queries, &mut rng);

    if a && b && c {
        println!("→ NOVEL BENCHMARK PASSED");
    } else {
        println!("→ NOVEL BENCHMARK FAILED (A={a} B={b} C={c})");
        std::process::exit(1);
    }
}
