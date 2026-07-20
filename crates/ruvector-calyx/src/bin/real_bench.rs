//! Real-data benchmark harness for the association layer.
//!
//! Usage:
//!   cargo run --release -p ruvector-calyx --bin calyx-real-bench -- corpus.calyx
//!
//! With a `.calyx` path it loads a real precomputed corpus (see
//! `tools/build_codesearchnet.py`). With no argument it synthesizes a
//! **CodeSearchNet-shaped stand-in** (clearly labelled) and round-trips it
//! through the loader, so the full load → benchmark path is provable offline.
//!
//! Reports the standard IR metrics used by published baselines (MRR@10,
//! Recall@1/10, nDCG@10) for single-lens vs multi-lens fusion, plus the
//! association-native extras: conformal abstention risk and code↔doc
//! disagreement.

use ruvector_calyx::conformal::{calibrate, coverage, empirical_risk};
use ruvector_calyx::corpus::{load, save, RealQuery};
use ruvector_calyx::disagreement::find_conflicts;
use ruvector_calyx::fusion::{weighted_rrf, DEFAULT_RRF_K};
use ruvector_calyx::ledger::{Anchor, AnchorKind};
use ruvector_calyx::loom::min_agreement;
use ruvector_calyx::rng::Rng;
use ruvector_calyx::{Constellation, ConstellationStore, LensKind, LensManifest, Query};

const K: usize = 10;

// ── Retrieval over the lenses a query actually has ──────────────────────────
fn rank(store: &ConstellationStore, q: &Query, lenses: &[&str], top_k: usize) -> Vec<u64> {
    let mut rankings = Vec::new();
    for name in lenses {
        if let Some(v) = q.slots.get(*name) {
            rankings.push(store.search_lens(name, v, top_k));
        }
    }
    if rankings.is_empty() {
        return Vec::new();
    }
    let w = vec![1.0f32; rankings.len()];
    weighted_rrf(&rankings, &w, DEFAULT_RRF_K)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

// ── IR metrics (single relevant-or-more, binary relevance) ──────────────────
#[derive(Default, Clone)]
struct Ir {
    mrr: f32,
    r1: f32,
    r10: f32,
    ndcg: f32,
    n: usize,
}
impl Ir {
    fn finish(self) -> Ir {
        let n = self.n.max(1) as f32;
        Ir {
            mrr: self.mrr / n,
            r1: self.r1 / n,
            r10: self.r10 / n,
            ndcg: self.ndcg / n,
            n: self.n,
        }
    }
}

fn eval_ir(store: &ConstellationStore, queries: &[RealQuery], lenses: &[&str]) -> Ir {
    let mut m = Ir::default();
    for q in queries.iter().filter(|q| q.is_answerable()) {
        m.n += 1;
        let ranked = rank(store, &q.query, lenses, K);
        // first rank (1-indexed) of any relevant doc
        let pos = ranked
            .iter()
            .position(|id| q.relevant.contains(id))
            .map(|p| p + 1);
        if let Some(p) = pos {
            m.mrr += 1.0 / p as f32;
            if p == 1 {
                m.r1 += 1.0;
            }
            if p <= 10 {
                m.r10 += 1.0;
                m.ndcg += 1.0 / ((p as f32) + 1.0).log2(); // IDCG=1 for one relevant
            }
        }
    }
    m.finish()
}

// ── Conformal abstention samples: (cross-lens agreement, wrong-if-answered) ──
fn conformal_samples(store: &ConstellationStore, queries: &[RealQuery], lenses: &[&str]) -> Vec<(f32, bool)> {
    queries
        .iter()
        .filter_map(|q| {
            let ranked = rank(store, &q.query, lenses, K);
            let top = *ranked.first()?;
            let agr = store.get(top).map(|r| min_agreement(&q.query.slots, r)).unwrap_or(0.0);
            let wrong = !q.relevant.contains(&top); // unanswerable → always wrong
            Some((agr, wrong))
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

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-calyx — Real-data benchmark (CodeSearchNet)          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let path = std::env::args().nth(1);
    let (corpus, source) = match &path {
        Some(p) => {
            let mut f = std::fs::File::open(p).unwrap_or_else(|e| {
                eprintln!("cannot open {p}: {e}");
                std::process::exit(2);
            });
            (load(&mut f).expect("valid .calyx"), format!("REAL: {p}"))
        }
        None => {
            // Synthesize a CodeSearchNet-shaped corpus, then prove the loader by
            // saving + reloading it (identical byte path a real converter emits).
            let (store, queries, planted) = synth();
            let mut buf = Vec::new();
            save(&mut buf, &store, &queries).unwrap();
            let mut cur = std::io::Cursor::new(buf);
            let c = load(&mut cur).unwrap();
            PLANTED.with(|p| *p.borrow_mut() = planted);
            (c, "SYNTHETIC STAND-IN (no .calyx supplied) — round-tripped through loader".to_string())
        }
    };

    println!("Source : {source}");
    let answerable = corpus.queries.iter().filter(|q| q.is_answerable()).count();
    println!(
        "Corpus : {} functions · {} lenses [{}] · {} queries ({} answerable, {} unanswerable)\n",
        corpus.store.len(),
        corpus.store.lenses().len(),
        corpus.store.lenses().iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", "),
        corpus.queries.len(),
        answerable,
        corpus.queries.len() - answerable,
    );

    // ── Retrieval: single-lens baselines vs multi-lens fusion ───────────────
    let all: Vec<&str> = corpus.store.lenses().iter().map(|m| m.name.as_str()).collect();
    let has = |n: &str| all.contains(&n);
    println!("Retrieval (answerable queries): single-lens vs association-native fusion");
    println!("  {:<24} {:>8} {:>8} {:>8} {:>8}", "system", "MRR@10", "R@1", "R@10", "nDCG@10");
    println!("  {}", "-".repeat(60));
    let mut rows: Vec<(String, Ir)> = Vec::new();
    for &l in &all {
        rows.push((format!("single-lens [{l}]"), eval_ir(&corpus.store, &corpus.queries, &[l])));
    }
    let fused = eval_ir(&corpus.store, &corpus.queries, &all);
    for (name, m) in &rows {
        println!("  {:<24} {:>8} {:>8} {:>8} {:>8}", name, fmt(m.mrr), fmt(m.r1), fmt(m.r10), fmt(m.ndcg));
    }
    println!("  {:<24} {:>8} {:>8} {:>8} {:>8}", "fusion (all lenses)", fmt(fused.mrr), fmt(fused.r1), fmt(fused.r10), fmt(fused.ndcg));
    let best_single = rows.iter().map(|(_, m)| m.mrr).fold(0.0f32, f32::max);
    println!("  → fusion MRR {:.3} vs best single-lens {:.3}  ({:+.3})\n", fused.mrr, best_single, fused.mrr - best_single);

    // ── Conformal abstention on real labels ─────────────────────────────────
    let mut rng = Rng::new(12345);
    let samples = conformal_samples(&corpus.store, &corpus.queries, &all);
    println!("Conformal cross-lens abstention (distribution-free risk ≤ α)");
    println!("  {:<7} {:>10} {:>10} {:>10}", "α", "threshold", "coverage", "test-risk");
    for &alpha in &[0.10f32, 0.05] {
        let mut s = samples.clone();
        shuffle(&mut s, &mut rng);
        let (cal, test) = s.split_at(s.len() / 2);
        let sel = calibrate(cal, alpha);
        println!(
            "  {:<7} {:>10} {:>10} {:>10}",
            format!("{:.2}", alpha),
            format!("{:.3}", sel.threshold),
            pct(coverage(&sel, test)),
            pct(empirical_risk(&sel, test)),
        );
    }
    println!();

    // ── Disagreement: where the code lens and the doc lens conflict ─────────
    if has("code") && has("doc") {
        println!("Disagreement: code lens vs doc lens (stale/incorrect docstrings)");
        let conflicts = find_conflicts(&corpus.store, "code", "doc", 10, 8);
        let planted = PLANTED.with(|p| p.borrow().clone());
        if !planted.is_empty() {
            let found: Vec<u64> = conflicts.iter().map(|&(id, _)| id).collect();
            let hits = planted.iter().filter(|id| found.contains(id)).count();
            println!("  planted stale-doc functions: {} · surfaced {}/{} in top-{}  → precision {}",
                planted.len(), hits, planted.len(), planted.len().max(8), pct(hits as f32 / planted.len() as f32));
        }
        println!("  top code↔doc conflicts (id, score): {:?}",
            conflicts.iter().take(6).map(|&(id, s)| (id, (s * 100.0).round() / 100.0)).collect::<Vec<_>>());
        println!();
    }

    println!("Note: with no .calyx supplied this is a synthetic stand-in that exercises the");
    println!("full loader→metrics path. Run tools/build_codesearchnet.py to produce a real");
    println!("corpus.calyx and pass it as the first argument for real numbers.");
}

fn fmt(x: f32) -> String { format!("{:.3}", x) }

thread_local! {
    static PLANTED: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ── Synthetic CodeSearchNet-shaped corpus (offline stand-in) ────────────────
const DIM_CODE: usize = 48;
const DIM_DOC: usize = 32;
const DIM_LEX: usize = 24;
const N_MODULES: usize = 15;
const N_PER_MODULE: usize = 16;
const N_FUNCS: usize = N_MODULES * N_PER_MODULE;
const N_ANSWERABLE: usize = 180;
const N_UNANSWERABLE: usize = 60;

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

fn synth() -> (ConstellationStore, Vec<RealQuery>, Vec<u64>) {
    let mut rng = Rng::new(2024);
    // Each function has a per-function code identity; docstring lives in a
    // per-module doc region; lexical is a per-function token fingerprint.
    let code_c: Vec<Vec<f32>> = (0..N_MODULES).map(|_| unit(&mut rng, DIM_CODE)).collect();
    let doc_c: Vec<Vec<f32>> = (0..N_MODULES).map(|_| unit(&mut rng, DIM_DOC)).collect();
    // Lexical token clusters cut *across* modules: code/doc resolve to a module,
    // lexical to a cluster, and only their intersection is a unique function.
    let lex_cl: Vec<Vec<f32>> = (0..N_PER_MODULE).map(|_| unit(&mut rng, DIM_LEX)).collect();

    let lenses = vec![
        LensManifest::new("code", "unixcoder-synth", LensKind::DenseSemantic, DIM_CODE, 8.0, 8.0),
        LensManifest::new("doc", "bge-small-synth", LensKind::DenseSemantic, DIM_DOC, 3.0, 3.0),
        LensManifest::new("lexical", "bm25-synth", LensKind::Lexical, DIM_LEX, 1.0, 1.0),
    ];
    let mut store = ConstellationStore::new(lenses);

    struct Fn_ { id: u64, module: usize, lc: usize, code: Vec<f32>, doc: Vec<f32>, lex: Vec<f32> }
    let mut funcs = Vec::new();
    let mut planted = Vec::new();
    let mut id = 0u64;
    for mdl in 0..N_MODULES {
        for (j, lex_center) in lex_cl.iter().enumerate() {
            let code = perturb(&code_c[mdl], 0.20, &mut rng);
            // Plant a stale docstring every 32nd function: doc points to a
            // *different* module than the code (comment says one thing…).
            let (doc, stale) = if id % 32 == 5 {
                let other = (mdl + N_MODULES / 2) % N_MODULES;
                (perturb(&doc_c[other], 0.15, &mut rng), true)
            } else {
                (perturb(&doc_c[mdl], 0.15, &mut rng), false)
            };
            let lex = perturb(lex_center, 0.20, &mut rng); // cross-module token cluster
            if stale {
                planted.push(id);
            }
            funcs.push(Fn_ { id, module: mdl, lc: j, code, doc, lex });
            id += 1;
        }
    }
    for f in &funcs {
        store
            .insert(
                Constellation::new(f.id)
                    .with_slot("code", f.code.clone())
                    .with_slot("doc", f.doc.clone())
                    .with_slot("lexical", f.lex.clone())
                    .with_anchor(Anchor::new(AnchorKind::PassedTest, format!("test_{}", f.id), 0.9)),
            )
            .expect("insert");
    }

    let mut queries = Vec::new();
    for i in 0..N_ANSWERABLE {
        let f = &funcs[(i * 5 + 2) % N_FUNCS];
        // NL query. Each lens is individually ambiguous — code and doc resolve
        // only to the *module* (16 candidate functions), lexical is a noisy
        // per-function fingerprint — so only their fusion pins the exact one.
        let q = Query::new()
            .with_slot("code", perturb(&code_c[f.module], 0.18, &mut rng))
            .with_slot("doc", perturb(&doc_c[f.module], 0.15, &mut rng))
            .with_slot("lexical", perturb(&lex_cl[f.lc], 0.20, &mut rng));
        queries.push(RealQuery { query: q, relevant: vec![f.id] });
    }
    for _ in 0..N_UNANSWERABLE {
        // NL query about code not in the corpus: no relevant function.
        let q = Query::new()
            .with_slot("code", unit(&mut rng, DIM_CODE))
            .with_slot("doc", unit(&mut rng, DIM_DOC))
            .with_slot("lexical", unit(&mut rng, DIM_LEX));
        queries.push(RealQuery { query: q, relevant: vec![] });
    }
    (store, queries, planted)
}
