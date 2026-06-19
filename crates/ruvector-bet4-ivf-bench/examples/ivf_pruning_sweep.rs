//! BET 4 matched-recall sweep (M2/M3): LB-ordered branch-and-bound IVF probing vs the tuned plain
//! `IvfFlat` `nprobe` incumbent, on real 128-d arxiv embeddings AND a PCA-8 low-dim control.
//!
//! Three contenders share one index per `nclusters` (built once): plain `nprobe` (incumbent),
//! B&B in **LB-order** (the faithful BET-2 `RegionPruneIvf` kernel), and the **steelman** B&B —
//! centroid-distance order + LB-skip (the strongest version: if it can't beat `nprobe`, the bound
//! doesn't pay). Reports the exact-regime pruning fraction, matched-recall cost, and checks the
//! FROZEN gate (docs/plans/bet4-ivf-pruning/PRE-REGISTRATION.md) on the steelman ratio.
//!
//! Run: `cargo run --release -p ruvector-bet4-ivf-bench --example ivf_pruning_sweep -- [N]`

use ruvector_bet4_ivf_bench::data::load_feat_csv;
use ruvector_bet4_ivf_bench::kernel::BnBIvf;
use ruvector_bet4_ivf_bench::oracle::{brute_force_topk, recall_at_k};
use ruvector_bet4_ivf_bench::pca::project_topm;
use ruvector_rairs::SearchResult;
use std::time::Instant;

const K: usize = 10;
const R_TARGET: f64 = 0.95;
const NCLUSTERS: [usize; 3] = [64, 256, 1024];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_req: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let data =
        std::env::var("BET4_DATA").unwrap_or_else(|_| "target/m1-data/node-feat-100k.csv".into());

    let corpus = load_feat_csv(&data, n_req).unwrap_or_else(|e| {
        eprintln!("failed to load {data}: {e}");
        std::process::exit(1);
    });
    let n = corpus.len();
    let dim = corpus.first().map(|v| v.len()).unwrap_or(0);
    println!("# BET4 sweep  n={n} dim={dim} k={K} R_target={R_TARGET}  data={data}\n");

    run_regime("128-d (real arxiv features)", &corpus);

    println!("\n# Projecting to PCA-8 (low-dim control)…");
    let t = Instant::now();
    let corpus8 = project_topm(&corpus, 8, 60);
    println!("# PCA done in {:?}\n", t.elapsed());
    run_regime("PCA-8 (low-dim control — bound should be TIGHT, B&B should WIN)", &corpus8);
}

fn run_regime(label: &str, corpus: &[Vec<f32>]) {
    let n = corpus.len();
    let dim = corpus[0].len();
    let nq = 200.min(n);
    let queries: Vec<usize> = (0..nq).collect();
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|&q| brute_force_topk(corpus, &corpus[q], K))
        .collect();

    println!("════ REGIME: {label}   (dim={dim}) ════");
    let mut cells: Vec<Cell> = Vec::new();

    for &nc in &NCLUSTERS {
        let t_build = Instant::now();
        let idx = BnBIvf::build(corpus, nc, 15, 42);
        let nc_eff = idx.num_lists();
        let build = t_build.elapsed();

        // Exact-regime pruning fraction (LB-order full budget).
        let mut pruned = 0.0;
        for &q in &queries {
            let (_r, _e, probed) = idx.search(&corpus[q], K, None);
            pruned += (nc_eff - probed) as f64 / nc_eff as f64;
        }
        let prune_frac = pruned / nq as f64;

        let grid = knob_grid(nc_eff);
        let plain = matched(&queries, corpus, &truth, &grid, |q, knob| {
            let (r, ev, _) = idx.search_nprobe(q, K, knob);
            (ids(&r), ev)
        });
        let bnb_lb = matched(&queries, corpus, &truth, &grid, |q, knob| {
            let (r, ev, _) = idx.search(q, K, Some(knob));
            (ids(&r), ev)
        });
        let bnb_skip = matched(&queries, corpus, &truth, &grid, |q, knob| {
            let (r, ev, _) = idx.search_bnb_skip(q, K, Some(knob));
            (ids(&r), ev)
        });

        let eval_ratio = plain.evals / bnb_skip.evals.max(1.0);
        let wall_ratio = plain.wall_ns as f64 / bnb_skip.wall_ns.max(1) as f64;

        println!("\n## nclusters={nc_eff}  (build {build:?})  exact-regime prune={:.1}%", prune_frac * 100.0);
        print_row("plain nprobe   (incumbent)", &plain);
        print_row("B&B  LB-order  (BET-2 kernel)", &bnb_lb);
        print_row("B&B  steelman  (cdist+LB-skip)", &bnb_skip);
        println!(
            "   steelman vs incumbent: eval {eval_ratio:.2}x   wall {wall_ratio:.2}x"
        );

        cells.push(Cell { nc: nc_eff, eval_ratio, wall_ratio, prune_frac });
    }

    verdict(label, &cells);
}

struct Cell {
    nc: usize,
    eval_ratio: f64,
    wall_ratio: f64,
    prune_frac: f64,
}

struct Matched {
    knob: usize,
    recall: f64,
    evals: f64,
    wall_ns: u128,
}

fn print_row(name: &str, m: &Matched) {
    println!(
        "   {name:<32} knob={:<4} recall={:.4} evals/q={:>8.0} wall/q={:>6}µs",
        m.knob,
        m.recall,
        m.evals,
        m.wall_ns / 1000
    );
}

/// First knob (ascending) whose mean recall ≥ `R_TARGET`, with its mean member-evals and wall-time;
/// falls back to the largest knob if none reaches target.
fn matched<F>(
    queries: &[usize],
    corpus: &[Vec<f32>],
    truth: &[Vec<usize>],
    grid: &[usize],
    search: F,
) -> Matched
where
    F: Fn(&[f32], usize) -> (Vec<usize>, usize),
{
    let mut last = Matched { knob: 0, recall: 0.0, evals: 0.0, wall_ns: 0 };
    for &knob in grid {
        let t = Instant::now();
        let mut rec = 0.0;
        let mut ev = 0usize;
        for (qi, &q) in queries.iter().enumerate() {
            let (got, e) = search(&corpus[q], knob);
            ev += e;
            rec += recall_at_k(&truth[qi], &got, K);
        }
        let wall_ns = t.elapsed().as_nanos() / queries.len() as u128;
        last = Matched {
            knob,
            recall: rec / queries.len() as f64,
            evals: ev as f64 / queries.len() as f64,
            wall_ns,
        };
        if last.recall >= R_TARGET {
            return last;
        }
    }
    last
}

fn knob_grid(maxv: usize) -> Vec<usize> {
    let mut g = Vec::new();
    let mut x = 1usize;
    while x < maxv {
        g.push(x);
        x = ((x as f64) * 1.5).ceil() as usize;
    }
    g.push(maxv);
    g.dedup();
    g
}

fn ids(res: &[SearchResult]) -> Vec<usize> {
    res.iter().map(|r| r.id).collect()
}

fn verdict(label: &str, cells: &[Cell]) {
    let all_win = cells.iter().all(|c| c.eval_ratio >= 2.0 && c.wall_ratio > 1.0);
    let any_kill = cells.iter().any(|c| c.eval_ratio < 1.5 || c.wall_ratio < 1.0);
    let v = if all_win {
        "WIN (≥2× evals AND wall-clock win across all nclusters)"
    } else if any_kill {
        "KILL / NO-GO (<1.5× somewhere or wall reversed — bound too loose to pay)"
    } else {
        "QUALIFIED (1.5–2×, or mixed)"
    };
    println!("\n   ── verdict [{label}] ──");
    for c in cells {
        println!(
            "      nclusters={:<5} steelman eval={:.2}x wall={:.2}x  exact-prune={:.1}%",
            c.nc, c.eval_ratio, c.wall_ratio, c.prune_frac * 100.0
        );
    }
    println!("      => {v}");
}
