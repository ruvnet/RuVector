//! BET 5 matched-recall sweep (M1/M2/M3): **PQ/IVFADC within-list pruning** vs the strongest
//! PQ-free incumbent (plain full-L2 `nprobe` and the early-abandon exact-L2 steelman), on real
//! 128-d arxiv embeddings, at matched recall@10 = 0.95.
//!
//! All contenders share one k-means index per `nclusters` (deterministic seed → identical
//! centroids/lists; certified in `tests/pq_gate.rs`). Only the within-list scan differs:
//!   - **plain**   — full `D`-dim L2 on every member of the `nprobe` lists (ADR-205's incumbent).
//!   - **abandon** — exact L2, early-abandoned at `τ²` (the steelman; charged in dims-touched/D).
//!   - **PQ**      — cheap ADC scan of the same lists + exact L2 re-rank of the top-`R` (the bet).
//!
//! Matched-recall protocol (see PRE-REGISTRATION.md): tune the incumbent `nprobe` to the smallest
//! value reaching recall ≥ 0.95; PQ scans the *same* `nprobe` lists (it cannot re-rank a neighbour
//! it never scans) and we tune the smallest re-rank pool `R` that recovers ≥ 0.95. Everything is
//! charged in one unit — full-`D`-L2-equivalents — so the fixed 256-equiv ADC table build and the
//! `R` exact re-ranks are paid in full (no free lunch).
//!
//! Run: `cargo run --release -p ruvector-bet4-ivf-bench --example pq_pruning_sweep -- [N ...]`
//! (default N = 20000 50000 100000).

use ruvector_bet4_ivf_bench::data::load_feat_csv;
use ruvector_bet4_ivf_bench::kernel::{build_ivf, BnBIvf};
use ruvector_bet4_ivf_bench::oracle::{brute_force_topk, recall_at_k};
use ruvector_bet4_ivf_bench::pq::PqIvf;
use std::time::Instant;

const K: usize = 10;
const R_TARGET: f64 = 0.95;
const NCLUSTERS: [usize; 3] = [64, 256, 1024];
const M_VALUES: [usize; 2] = [16, 8];
const NQ: usize = 200;
const MAX_ITER: usize = 15;
const SEED: u64 = 42;

/// Per-nclusters verdict log: `(nclusters, [(N, full_win, best_ratio)])`.
type PerNcVerdicts = (usize, Vec<(usize, bool, f64)>);

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let scales = if args.is_empty() {
        vec![20_000usize, 50_000, 100_000]
    } else {
        args
    };
    let data =
        std::env::var("BET4_DATA").unwrap_or_else(|_| "target/m1-data/node-feat-100k.csv".into());

    println!("# BET5 PQ/IVFADC sweep  k={K} R_target={R_TARGET} nq={NQ}  data={data}");
    println!("# unit = full-D-L2-equivalent member-eval. PQ cost = 256(LUT) + adc_members*m/D + R(rerank).");
    println!("# crossover n* = smallest tested N where PQ beats the best PQ-free incumbent.\n");

    // Track, per nclusters, the verdict per scale to find the crossover and the gate.
    // (nclusters, [(N, full_win, best_ratio)]).
    let mut win_at: Vec<PerNcVerdicts> =
        NCLUSTERS.iter().map(|&nc| (nc, Vec::new())).collect();

    for &n_req in &scales {
        let corpus = match load_feat_csv(&data, n_req) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("failed to load {data}: {e}");
                std::process::exit(1);
            }
        };
        let n = corpus.len();
        let dim = corpus[0].len();
        let queries: Vec<usize> = (0..NQ.min(n)).collect();
        let t_truth = Instant::now();
        let truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|&q| brute_force_topk(&corpus, &corpus[q], K))
            .collect();
        println!("════════ N={n} dim={dim}  (truth in {:?}) ════════", t_truth.elapsed());

        for (nc_i, &nc) in NCLUSTERS.iter().enumerate() {
            let t_b = Instant::now();
            let parts = build_ivf(&corpus, nc, MAX_ITER, SEED); // shared k-means: once per cell
            let bnb = BnBIvf::from_parts(&parts);
            let nc_eff = bnb.num_lists();
            let build_ivf_t = t_b.elapsed();

            // ---- tune incumbent nprobe to the smallest reaching recall >= 0.95 ----
            let np_grid = nprobe_grid(nc_eff);
            let mut np_star = nc_eff;
            let mut inc_recall = 0.0;
            for &np in &np_grid {
                let r = mean_recall(&queries, &truth, |qi| {
                    bnb.search_nprobe(&corpus[qi], K, np).0
                });
                if r >= R_TARGET {
                    np_star = np;
                    inc_recall = r;
                    break;
                }
            }

            // plain full-L2 cost (members) and early-abandon cost (dims/D), both at np_star.
            let (plain_evals, abandon_dims, members, t_plain, t_abandon, abandon_recall) =
                incumbent_costs(&bnb, &corpus, &queries, &truth, np_star, dim);
            let plain_cost = plain_evals; // 1 per member
            let abandon_cost = abandon_dims / dim as f64;
            let best_inc = plain_cost.min(abandon_cost);
            let abandon_prune = 1.0 - abandon_dims / (members * dim as f64);
            // Routing: every contender computes q↔centroid for all nc_eff centroids to pick the
            // nprobe nearest lists. Charged EQUALLY to incumbent and PQ (the pre-reg's "no free
            // routing" adversarial check). It dilutes any ratio, decisively at high nclusters.
            let routing = nc_eff as f64;

            println!(
                "\n── nclusters={nc_eff} (build {build_ivf_t:?})  np*={np_star} inc_recall={inc_recall:.3}  routing={routing:.0} ev/q ──"
            );
            println!(
                "   incumbent  plain={plain_cost:8.0} | abandon={abandon_cost:8.0} ev (dim-prune {:.1}%, exact r={abandon_recall:.3})  members={members:.0}  | best+routing={:.0}",
                abandon_prune * 100.0,
                best_inc + routing
            );
            println!(
                "   wall/q     plain={:>8.1}µs | abandon={:>8.1}µs",
                t_plain, t_abandon
            );

            let mut cell_win = false;
            let mut cell_ratio = 0.0;
            for &m in &M_VALUES {
                let t_pq = Instant::now();
                let pq = PqIvf::from_parts(&parts, &corpus, m, MAX_ITER, SEED);
                let build_pq = t_pq.elapsed();

                // pure-ADC ceiling at np_star (no re-rank)
                let adc_ceiling = mean_recall(&queries, &truth, |qi| {
                    pq.search_adc_only(&corpus[qi], K, np_star)
                });

                // tune smallest R reaching recall >= 0.95 at np_star
                let r_grid = rerank_grid(members as usize);
                let mut r_star = None;
                for &rr in &r_grid {
                    let r = mean_recall(&queries, &truth, |qi| {
                        pq.search_adc_rerank(&corpus[qi], K, np_star, rr).0
                    });
                    if r >= R_TARGET {
                        r_star = Some(rr);
                        break;
                    }
                }

                match r_star {
                    None => {
                        println!(
                            "   PQ m={m:>2}  (build {build_pq:?}) ADC-ceiling={adc_ceiling:.3}  R*=NONE (cannot reach {R_TARGET} within working set) → KILL-path",
                        );
                    }
                    Some(rr) => {
                        // measure PQ cost + wall at (np_star, rr)
                        let t0 = Instant::now();
                        let mut cost_sum = 0.0;
                        let mut rec = 0.0;
                        for (j, &qi) in queries.iter().enumerate() {
                            let (res, c) = pq.search_adc_rerank(&corpus[qi], K, np_star, rr);
                            cost_sum += c.l2_equiv();
                            let got: Vec<usize> = res.iter().map(|r| r.id).collect();
                            rec += recall_at_k(&truth[j], &got, K);
                        }
                        let t_pq_q = t0.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
                        let pq_cost = cost_sum / queries.len() as f64;
                        let rec = rec / queries.len() as f64;
                        // Member-only ratio (transparency) and the gate-deciding TOTAL ratio with
                        // routing charged to both (the faithful full-L2-equivalent accounting).
                        let member_ratio = best_inc / pq_cost;
                        let total_ratio = (best_inc + routing) / (pq_cost + routing);
                        let wall_win = t_pq_q < t_plain.min(t_abandon);
                        let rr_full = rr >= members as usize; // re-rank == whole working set → bought nothing
                        let verdict = if rr_full {
                            "DEGENERATE(R≈WS)"
                        } else if total_ratio >= 2.0 && wall_win {
                            "WIN≥2×"
                        } else if total_ratio >= 1.5 {
                            "qualified"
                        } else {
                            "miss"
                        };
                        println!(
                            "   PQ m={m:>2}  ADC-ceil={adc_ceiling:.3}  R*={rr:>5}  cost={pq_cost:8.0}(+rt={:.0})  recall={rec:.3}  wall={t_pq_q:>7.1}µs  member={member_ratio:.2}× total={total_ratio:.2}×  [{verdict}{}]",
                            pq_cost + routing,
                            if wall_win { "" } else { ", WALL-REVERSES" }
                        );
                        if total_ratio > cell_ratio {
                            cell_ratio = total_ratio;
                        }
                        if total_ratio >= 2.0 && wall_win && !rr_full {
                            cell_win = true;
                        }
                    }
                }
            }
            win_at[nc_i].1.push((n, cell_win, cell_ratio));
        }
        println!();
    }

    // ---- gate summary: WIN needs >=2x + wall + all three nclusters at >= one N>=50k ----
    println!("\n════════ GATE (FROZEN: PRE-REGISTRATION.md) ════════");
    let scales_ge_50k: Vec<usize> = scales.iter().copied().filter(|&n| n >= 50_000).collect();
    let mut any_full_win = false;
    for &n in &scales_ge_50k {
        let all_nc_win = NCLUSTERS.iter().enumerate().all(|(i, _)| {
            win_at[i]
                .1
                .iter()
                .any(|&(nn, win, _)| nn == n && win)
        });
        if all_nc_win {
            any_full_win = true;
            println!("  N={n}: WIN at ALL nclusters → gate WIN condition met");
        }
    }
    if !any_full_win {
        println!("  No N≥50k wins at all three nclusters.");
        // best ratio seen per nclusters for the qualified/kill read
        for (nc, rows) in &win_at {
            let best = rows
                .iter()
                .map(|&(n, _, r)| format!("N{}:{:.2}×", n, r))
                .collect::<Vec<_>>()
                .join(" ");
            println!("    nclusters={nc}: best PQ ratio per scale → {best}");
        }
    }
}

/// Geometric-ish nprobe grid up to `nc`, dense at the low end where the tuned optimum lives.
fn nprobe_grid(nc: usize) -> Vec<usize> {
    let mut g = vec![1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768];
    g.push(nc);
    g.retain(|&x| x <= nc);
    g.sort_unstable();
    g.dedup();
    g
}

/// Re-rank pool grid up to the working set; dense at the low end (the win lives there).
fn rerank_grid(ws: usize) -> Vec<usize> {
    let mut g = vec![
        10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 8000,
        12000, 20000,
    ];
    g.push(ws);
    g.retain(|&x| x <= ws.max(1));
    g.sort_unstable();
    g.dedup();
    g
}

fn mean_recall<F>(queries: &[usize], truth: &[Vec<usize>], mut search: F) -> f64
where
    F: FnMut(usize) -> Vec<ruvector_rairs::SearchResult>,
{
    let mut acc = 0.0;
    for (j, &qi) in queries.iter().enumerate() {
        let got: Vec<usize> = search(qi).iter().map(|r| r.id).collect();
        acc += recall_at_k(&truth[j], &got, K);
    }
    acc / queries.len() as f64
}

/// Plain & early-abandon incumbent costs + wall-clock (µs/query) + abandon recall, all at `np`.
#[allow(clippy::too_many_arguments)]
fn incumbent_costs(
    bnb: &BnBIvf,
    corpus: &[Vec<f32>],
    queries: &[usize],
    truth: &[Vec<usize>],
    np: usize,
    _dim: usize,
) -> (f64, f64, f64, f64, f64, f64) {
    let mut members = 0usize;
    let mut dims = 0usize;
    let mut abandon_rec = 0.0;
    let t_plain0 = Instant::now();
    for &qi in queries {
        let (_r, e, _p) = bnb.search_nprobe(&corpus[qi], K, np);
        members += e;
    }
    let t_plain = t_plain0.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;

    let t_ab0 = Instant::now();
    for (j, &qi) in queries.iter().enumerate() {
        let (res, dt, _mem) = bnb.search_nprobe_abandon(&corpus[qi], K, np);
        dims += dt;
        let got: Vec<usize> = res.iter().map(|r| r.id).collect();
        abandon_rec += recall_at_k(&truth[j], &got, K);
    }
    let t_abandon = t_ab0.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;

    let nqf = queries.len() as f64;
    (
        members as f64 / nqf,
        dims as f64 / nqf,
        members as f64 / nqf,
        t_plain,
        t_abandon,
        abandon_rec / nqf,
    )
}
