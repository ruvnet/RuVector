//! Benchmark: adaptive calibrated lens routing vs brute-force and static order.
//!
//! Three systems retrieve over the same multi-lens corpus:
//! 1. brute-force — always consult every lens
//! 2. static — always consult the two cheap lenses (fixed policy)
//! 3. adaptive — consult cheapest-first, stop early on calibrated confidence,
//!    escalate only when unsure, abstain if the full panel is still not confident
//!
//! Acceptance (adaptive vs brute-force): accuracy loss ≤1pp, cost −≥30%,
//! latency −≥25%, ECE ≤0.08, abstention precision ≥0.80.
//! Run: `cargo run --release -p ruvector-calyx --bin calyx-routing-bench`

use ruvector_calyx::calibrate::{
    abstention_quality, expected_calibration_error, raw_confidence, Calibrator,
};
use ruvector_calyx::rng::Rng;
use ruvector_calyx::routing::{route, RoutingConfig, RoutingMode};
use ruvector_calyx::{Anchor, AnchorKind, Constellation, ConstellationStore, GuardProfile, LensKind, LensManifest, Query};

// ── Corpus parameters ───────────────────────────────────────────────────────
const DIM_SEM: usize = 48;
const DIM_LEX: usize = 32;
const DIM_STR: usize = 16;
const N_TOPICS: usize = 12;
const N_PER_TOPIC: usize = 20;
const N_DOCS: usize = N_TOPICS * N_PER_TOPIC;
const N_STRUCT: usize = 8;

const N_ANSWERABLE: usize = 160;
const N_UNANSWERABLE: usize = 80;

// Lens economics — semantic is expensive and (here) the low-signal "fog" lens;
// structural and lexical are cheap and discriminative. Cheapest-first order.
const COST_STR: f32 = 1.5;
const COST_LEX: f32 = 2.0;
const COST_SEM: f32 = 12.0;
const SEED: u64 = 2026;

fn order() -> Vec<String> {
    vec!["structural".into(), "lexical".into(), "semantic".into()]
}

fn registry() -> Vec<LensManifest> {
    vec![
        LensManifest::new("structural", "frozen-v1", LensKind::Structural, DIM_STR, COST_STR, COST_STR),
        LensManifest::new("lexical", "frozen-v1", LensKind::Lexical, DIM_LEX, COST_LEX, COST_LEX),
        LensManifest::new("semantic", "frozen-v1", LensKind::DenseSemantic, DIM_SEM, COST_SEM, COST_SEM),
    ]
}

// ── Vector helpers ──────────────────────────────────────────────────────────
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

struct Doc {
    id: u64,
    topic: usize,
    sem: Vec<f32>,
    lex: Vec<f32>,
    str_: Vec<f32>,
}
struct QSpec {
    q: Query,
    target: Option<u64>,
    train: bool,
}

fn build(rng: &mut Rng) -> (ConstellationStore, Vec<Doc>, Vec<QSpec>) {
    let sem_c: Vec<Vec<f32>> = (0..N_TOPICS).map(|_| unit(rng, DIM_SEM)).collect();
    let lex_c: Vec<Vec<f32>> = (0..N_TOPICS).map(|_| unit(rng, DIM_LEX)).collect();
    let str_c: Vec<Vec<f32>> = (0..N_STRUCT).map(|_| unit(rng, DIM_STR)).collect();

    let mut docs = Vec::with_capacity(N_DOCS);
    let mut id = 0u64;
    for t in 0..N_TOPICS {
        for j in 0..N_PER_TOPIC {
            let s = (t * N_PER_TOPIC + j) % N_STRUCT;
            let sem = perturb(&sem_c[t], 0.10, rng); // topic fog
            let lex = blend(&lex_c[t], &unit(rng, DIM_LEX), 0.5, 0.85); // + per-doc fingerprint
            let str_ = perturb(&str_c[s], 0.30, rng); // + per-doc jitter
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

    let mut queries = Vec::with_capacity(N_ANSWERABLE + N_UNANSWERABLE);
    for i in 0..N_ANSWERABLE {
        let d = &docs[(i * 7 + 3) % N_DOCS];
        // Half the answerable queries are "hard": noisier structural slot, so
        // the cheap structural lens alone is ambiguous and the router must
        // escalate to lexical.
        let hard = i % 2 == 1;
        let str_noise = if hard { 0.28 } else { 0.06 };
        let q = Query::new()
            .with_slot("structural", perturb(&d.str_, str_noise, rng))
            .with_slot("lexical", perturb(&d.lex, 0.10, rng))
            .with_slot("semantic", perturb(&sem_c[d.topic], 0.10, rng));
        queries.push(QSpec { q, target: Some(d.id), train: i % 2 == 0 });
    }
    for i in 0..N_UNANSWERABLE {
        let t = i % N_TOPICS;
        let q = Query::new()
            .with_slot("structural", unit(rng, DIM_STR)) // matches no doc
            .with_slot("lexical", unit(rng, DIM_LEX))
            .with_slot("semantic", perturb(&sem_c[t], 0.10, rng));
        queries.push(QSpec { q, target: None, train: i % 2 == 0 });
    }
    (store, docs, queries)
}

// ── Metrics ─────────────────────────────────────────────────────────────────
#[derive(Default)]
struct Agg {
    answerable: usize,
    grounded_correct: usize, // answered & correct (answerable)
    answered: usize,
    correct_answers: usize,
    total_cost: f32,
    total_latency: f32,
    n: usize,
    ece_samples: Vec<(f32, bool)>, // (conf, correct) over answered
    abstained: Vec<bool>,
    unanswerable: Vec<bool>,
}
impl Agg {
    fn grounded_acc(&self) -> f32 {
        if self.answerable == 0 { 0.0 } else { self.grounded_correct as f32 / self.answerable as f32 }
    }
    fn accepted_acc(&self) -> f32 {
        if self.answered == 0 { 1.0 } else { self.correct_answers as f32 / self.answered as f32 }
    }
    fn mean_cost(&self) -> f32 { self.total_cost / self.n.max(1) as f32 }
    fn mean_latency(&self) -> f32 { self.total_latency / self.n.max(1) as f32 }
    fn cost_per_accepted(&self) -> f32 { self.total_cost / self.answered.max(1) as f32 }
    fn ece(&self) -> f32 { expected_calibration_error(&self.ece_samples, 10) }
    fn abstention(&self) -> (f32, f32) {
        let q = abstention_quality(&self.abstained, &self.unanswerable);
        (q.precision, q.recall)
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    store: &ConstellationStore,
    order: &[String],
    queries: &[QSpec],
    cal: &Calibrator,
    cfg: &RoutingConfig,
    guard: &GuardProfile,
    mode: RoutingMode,
    train_split: bool,
) -> Agg {
    let mut a = Agg::default();
    for qs in queries.iter().filter(|q| q.train == train_split) {
        let out = route(store, order, &qs.q, cal, cfg, guard, mode);
        a.n += 1;
        a.total_cost += out.witness.cost_us;
        a.total_latency += out.witness.latency_us;
        let is_unans = qs.target.is_none();
        a.unanswerable.push(is_unans);
        a.abstained.push(out.answered.is_none());
        let top_correct = out.witness.top_candidate == qs.target && qs.target.is_some();
        if let Some(_t) = qs.target {
            a.answerable += 1;
        }
        if let Some(ans) = out.answered {
            a.answered += 1;
            let correct = qs.target == Some(ans);
            if correct {
                a.correct_answers += 1;
                if qs.target.is_some() {
                    a.grounded_correct += 1;
                }
            }
            // ECE over committed answers: does confidence track correctness?
            a.ece_samples.push((out.confidence, correct));
        }
        let _ = top_correct;
    }
    a
}

fn pct(x: f32) -> String { format!("{:.1}%", x * 100.0) }

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-calyx — Adaptive Routing + Calibration (ADR-272)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Lenses (cheapest-first): structural({COST_STR}µs) → lexical({COST_LEX}µs) → semantic({COST_SEM}µs)");
    println!("Corpus : {N_DOCS} docs · queries {N_ANSWERABLE} answerable + {N_UNANSWERABLE} unanswerable (50/50 train/test)\n");

    let mut rng = Rng::new(SEED);
    let (store, _docs, queries) = build(&mut rng);
    let ord = order();
    let guard = GuardProfile::default();
    let cfg = RoutingConfig { stop_threshold: 0.80, answer_threshold: 0.55, min_lenses_to_answer: 2, top_k: 10 };

    // ── Fit the calibrator on the TRAIN split (bootstrap: identity calibrator,
    //    adaptive routing, collect raw-confidence vs would-be-correctness) ─────
    let boot = Calibrator::identity(10);
    let mut samples = Vec::new();
    for qs in queries.iter().filter(|q| q.train) {
        let out = route(&store, &ord, &qs.q, &boot, &cfg, &guard, RoutingMode::Adaptive);
        let raw = raw_confidence(&out.witness.signals);
        let correct = out.witness.top_candidate == qs.target && qs.target.is_some();
        samples.push((raw, correct));
    }
    let cal = Calibrator::fit(&samples, 10);
    println!("Calibrator fit on {} train decisions (10-bin reliability map)\n", samples.len());

    // ── Evaluate the three systems on the TEST split ────────────────────────
    let brute = evaluate(&store, &ord, &queries, &cal, &cfg, &guard, RoutingMode::BruteForce, false);
    let stat = evaluate(&store, &ord, &queries, &cal, &cfg, &guard, RoutingMode::Static { max_lenses: 2 }, false);
    let adap = evaluate(&store, &ord, &queries, &cal, &cfg, &guard, RoutingMode::Adaptive, false);

    let row = |name: &str, a: &Agg| {
        let (ap, ar) = a.abstention();
        println!(
            "  {:<12} {:>9} {:>9} {:>9} {:>9} {:>7} {:>9} {:>7}",
            name,
            pct(a.grounded_acc()),
            pct(a.accepted_acc()),
            format!("{:.1}", a.mean_cost()),
            format!("{:.1}", a.mean_latency()),
            format!("{:.3}", a.ece()),
            pct(ap),
            pct(ar),
        );
    };
    println!("System comparison (test split)");
    println!(
        "  {:<12} {:>9} {:>9} {:>9} {:>9} {:>7} {:>9} {:>7}",
        "mode", "grnd-acc", "acc-acc", "cost µs", "lat µs", "ECE", "abst-P", "abst-R"
    );
    println!("  {}", "-".repeat(78));
    row("brute-force", &brute);
    row("static(2)", &stat);
    row("adaptive", &adap);
    println!();

    // Witness example (first adaptive answer on the test split).
    if let Some(qs) = queries.iter().find(|q| !q.train && q.target.is_some()) {
        let out = route(&store, &ord, &qs.q, &cal, &cfg, &guard, RoutingMode::Adaptive);
        println!("Example witness (adaptive)");
        println!("  lenses consulted : {:?}", out.witness.lenses_consulted);
        println!("  action / reason  : {} / {}", out.witness.action, out.witness.escalation_reason);
        println!(
            "  confidence       : {:.2}   agreement={:.2} margin={:.2} contradiction={:.2}",
            out.confidence,
            out.witness.signals.cross_lens_agreement,
            out.witness.signals.margin_ratio,
            out.witness.signals.contradiction
        );
        println!("  cost / latency   : {:.1}µs / {:.1}µs\n", out.witness.cost_us, out.witness.latency_us);
    }

    // ── Acceptance ──────────────────────────────────────────────────────────
    let acc_loss = brute.grounded_acc() - adap.grounded_acc();
    let cost_red = 1.0 - adap.mean_cost() / brute.mean_cost().max(1e-9);
    let lat_red = 1.0 - adap.mean_latency() / brute.mean_latency().max(1e-9);
    let ece = adap.ece();
    let (abst_p, _abst_r) = adap.abstention();

    println!("Acceptance test (adaptive vs brute-force)");
    let c1 = acc_loss <= 0.01;
    let c2 = cost_red >= 0.30;
    let c3 = lat_red >= 0.25;
    let c4 = ece <= 0.08;
    let c5 = abst_p >= 0.80;
    let chk = |name: &str, val: String, ok: bool| {
        println!("  {:<34} {:>12}   {}", name, val, if ok { "PASS ✓" } else { "FAIL ✗" });
    };
    chk("accuracy loss ≤ 1pp", format!("{:+.1}pp", acc_loss * 100.0), c1);
    chk("query cost reduction ≥ 30%", format!("−{:.1}%", cost_red * 100.0), c2);
    chk("latency reduction ≥ 25%", format!("−{:.1}%", lat_red * 100.0), c3);
    chk("ECE ≤ 0.08", format!("{:.3}", ece), c4);
    chk("abstention precision ≥ 0.80", pct(abst_p), c5);
    println!();
    println!("  cost per accepted answer: brute {:.1}µs → adaptive {:.1}µs", brute.cost_per_accepted(), adap.cost_per_accepted());
    println!();

    if c1 && c2 && c3 && c4 && c5 {
        println!("→ ROUTING BENCHMARK PASSED");
    } else {
        println!("→ ROUTING BENCHMARK FAILED");
        std::process::exit(1);
    }
}
