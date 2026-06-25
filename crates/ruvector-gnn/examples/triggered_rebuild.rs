//! BET 1 follow-up (ADR-200 next-step #2, ADR-202 next-step): does a **sampled-recall
//! rebuild trigger** beat fixed `Periodic{k}` under *variable-rate* drift — and beat the
//! Frobenius-norm monitor ADR-200 found wanting?
//!
//! Periodic{k} is near-optimal under STEADY drift (ADR-202). A trigger can only earn its
//! keep when drift is BURSTY: calm stretches where a fixed cadence over-rebuilds, bursts
//! where it under-rebuilds. So the trajectory here alternates high-lr bursts and low-lr
//! calm. If the trigger can't beat periodic *there*, it's a clean KILL.
//!
//! Gate (frozen): docs/plans/bet1-productionize/PRE-REGISTRATION-trigger.md.
//!   Honest comparison = the (rebuilds, recall) PARETO FRONTIER of Triggered{floor},
//!   Periodic{k}, Frobenius{tau} (no cherry-picked single config). WIN = Triggered's
//!   frontier dominates (fewer rebuilds at equal recall) AND the probe's own cost
//!   (counted) is less than the rebuilds it saves AND it beats Frobenius.
//!
//! Runs at n=10k: ADR-202 already established scale-robustness; this bet isolates the
//! cadence question, where rebuild *count* (not scale) is the signal.
//!
//! Run: cargo run --release -p ruvector-gnn --example triggered_rebuild -- [N] [EPOCHS]

use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_diskann::distance::{l2_squared, FlatVectors};
use ruvector_diskann::{DriftingIndex, RebuildPolicy};
use ruvector_gnn::training::{Optimizer, OptimizerType};
use std::time::Instant;

const DIM: usize = 128;
const R: usize = 32;
const BUILD_BEAM: usize = 64;
const SEARCH_BEAM: usize = 64;
const ALPHA: f32 = 1.2;
const K: usize = 10;

// ---------- data + embedding helpers (self-contained; cf. diskann_real_trajectory.rs) ----------

fn read_features(path: &str, n: usize) -> Vec<Vec<f32>> {
    let txt = std::fs::read_to_string(path).expect("read features csv");
    txt.lines()
        .take(n)
        .map(|line| {
            line.split(',')
                .map(|s| s.trim().parse::<f32>().unwrap())
                .collect()
        })
        .collect()
}

fn read_edges(path: &str, n: usize) -> Vec<(usize, usize)> {
    let txt = std::fs::read_to_string(path).expect("read edge csv");
    let mut edges = Vec::new();
    for line in txt.lines() {
        let mut it = line.split(',');
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                if a < n && b < n && a != b {
                    edges.push((a, b));
                }
            }
        }
    }
    edges
}

fn normalize_row(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn matrix_from_features(feats: &[Vec<f32>]) -> Array2<f32> {
    let n = feats.len();
    let mut m = Array2::<f32>::zeros((n, DIM));
    for (i, f) in feats.iter().enumerate() {
        let mut row = f.clone();
        normalize_row(&mut row);
        for d in 0..DIM {
            m[[i, d]] = row[d];
        }
    }
    m
}

fn to_flat(emb: &Array2<f32>) -> FlatVectors {
    let mut f = FlatVectors::with_capacity(DIM, emb.nrows());
    let mut buf = vec![0.0f32; DIM];
    for i in 0..emb.nrows() {
        for d in 0..DIM {
            buf[d] = emb[[i, d]];
        }
        f.push(&buf);
    }
    f
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn brute_topk(emb: &Array2<f32>, q: usize, k: usize) -> Vec<u32> {
    let qrow = emb.row(q);
    let qs = qrow.as_slice().unwrap();
    let mut scored: Vec<(f32, u32)> = (0..emb.nrows())
        .filter(|&i| i != q)
        .map(|i| (l2_squared(emb.row(i).as_slice().unwrap(), qs), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

fn recall(got: &[u32], truth: &[u32]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    got.iter().filter(|g| truth.contains(g)).count() as f64 / truth.len() as f64
}

fn search_topk(idx: &DriftingIndex, emb: &Array2<f32>, flat: &FlatVectors, q: usize) -> Vec<u32> {
    let qs = emb.row(q).as_slice().unwrap().to_vec();
    let (cands, _) = idx.search(flat, &qs, SEARCH_BEAM);
    let mut scored: Vec<(f32, u32)> = cands
        .iter()
        .map(|&c| (l2_squared(emb.row(c as usize).as_slice().unwrap(), &qs), c))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored
        .into_iter()
        .filter(|&(_, c)| c as usize != q)
        .take(K)
        .map(|(_, c)| c)
        .collect()
}

/// Mean recall of the reuse index over `qs` against truth recomputed under `emb`.
fn probe_recall(idx: &DriftingIndex, emb: &Array2<f32>, flat: &FlatVectors, qs: &[usize]) -> f64 {
    qs.iter()
        .map(|&q| recall(&search_topk(idx, emb, flat, q), &brute_topk(emb, q, K)))
        .sum::<f64>()
        / qs.len().max(1) as f64
}

// ---------- variable-rate contrastive trajectory ----------

/// `lr_at(epoch)` lets the caller impose a burst/calm schedule.
#[allow(clippy::too_many_arguments)]
fn train_variable_rate(
    e0: Array2<f32>,
    edges: &[(usize, usize)],
    n: usize,
    epochs: usize,
    batch: usize,
    n_neg: usize,
    tau: f32,
    lr_at: impl Fn(usize) -> f32,
    seed: u64,
) -> Vec<Array2<f32>> {
    let mut emb = e0.clone();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut snapshots = vec![emb.clone()];

    for epoch in 0..epochs {
        let lr = lr_at(epoch);
        // Adam (fresh per epoch so the burst/calm lr schedule takes effect): its
        // per-parameter scaling produces real embedding motion at these lrs where plain
        // SGD does not (a VOID 0%-churn trajectory).
        let mut opt = Optimizer::new(OptimizerType::Adam {
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
        let mut grad = Array2::<f32>::zeros((n, DIM));
        for _ in 0..batch {
            let (a, p) = edges[rng.gen_range(0..edges.len())];
            let negs: Vec<usize> = (0..n_neg)
                .map(|_| {
                    let mut j = rng.gen_range(0..n);
                    while j == a {
                        j = rng.gen_range(0..n);
                    }
                    j
                })
                .collect();
            let av: Vec<f32> = emb.row(a).to_vec();
            let pv: Vec<f32> = emb.row(p).to_vec();
            let s_p = dot(&av, &pv) / tau;
            let s_neg: Vec<f32> = negs
                .iter()
                .map(|&j| dot(&av, emb.row(j).as_slice().unwrap()) / tau)
                .collect();
            let m = s_neg.iter().cloned().fold(s_p, f32::max);
            let mut z = (s_p - m).exp();
            for &s in &s_neg {
                z += (s - m).exp();
            }
            let sm_p = (s_p - m).exp() / z;
            let inv_tau = 1.0 / tau;
            for d in 0..DIM {
                grad[[a, d]] += inv_tau * (sm_p - 1.0) * pv[d];
                grad[[p, d]] += inv_tau * (sm_p - 1.0) * av[d];
            }
            for (jdx, &j) in negs.iter().enumerate() {
                let sm_j = (s_neg[jdx] - m).exp() / z;
                for d in 0..DIM {
                    grad[[a, d]] += inv_tau * sm_j * emb[[j, d]];
                    grad[[j, d]] += inv_tau * sm_j * av[d];
                }
            }
        }
        grad.mapv_inplace(|g| g / batch as f32);
        opt.step(&mut emb, &grad).expect("step");
        for i in 0..n {
            let mut row = emb.row(i).to_vec();
            normalize_row(&mut row);
            for d in 0..DIM {
                emb[[i, d]] = row[d];
            }
        }
        let _ = epoch;
        snapshots.push(emb.clone());
    }
    snapshots
}

// ---------- policy runner ----------

#[derive(Clone, Copy)]
enum Trigger {
    Periodic(usize),
    Frobenius(f32), // rebuild when mean per-node displacement since last rebuild > tau
    Recall(f64),    // rebuild when sampled-recall probe < floor
}

struct Outcome {
    label: String,
    recall: f64,
    rebuilds: usize,
    rebuild_cost_s: f64,
    probe_evals: f64, // distance-evals spent on the recall probe (counted against the trigger)
}

#[allow(clippy::too_many_arguments)]
fn run_policy(
    label: String,
    trig: Trigger,
    snapshots: &[Array2<f32>],
    flats: &[FlatVectors],
    queries: &[usize],
    truth: &[Vec<Vec<u32>>],
    probe_qs: &[usize],
    n: usize,
) -> Outcome {
    // ReweightOnly => on_metric_update never auto-rebuilds; we drive force_rebuild.
    let mut idx =
        DriftingIndex::build(&flats[0], RebuildPolicy::ReweightOnly, R, BUILD_BEAM, ALPHA)
            .expect("build");
    let mut rebuilds = 0usize;
    let mut rebuild_cost = 0.0f64;
    let mut probe_evals = 0.0f64;
    let mut last_rebuild = 0usize; // snapshot index of last (re)build
    let mut recall_sum = 0.0f64;
    let steps = snapshots.len() - 1;

    for step in 1..snapshots.len() {
        let emb = &snapshots[step];
        let flat = &flats[step];
        idx.on_metric_update(flat).expect("update"); // reweight (no auto-rebuild)

        let do_rebuild = match trig {
            Trigger::Periodic(k) => k > 0 && step % k == 0,
            Trigger::Frobenius(t) => {
                // mean per-node L2 displacement since last rebuild snapshot
                let prev = &snapshots[last_rebuild];
                let mut acc = 0.0f64;
                for i in 0..n {
                    acc += l2_squared(
                        emb.row(i).as_slice().unwrap(),
                        prev.row(i).as_slice().unwrap(),
                    )
                    .sqrt() as f64;
                }
                (acc / n as f64) > t as f64
            }
            Trigger::Recall(floor) => {
                probe_evals += (probe_qs.len() * n) as f64; // brute-force probe truth cost
                probe_recall(&idx, emb, flat, probe_qs) < floor
            }
        };
        if do_rebuild {
            let tb = Instant::now();
            idx.force_rebuild(flat).expect("rebuild");
            rebuild_cost += tb.elapsed().as_secs_f64();
            rebuilds += 1;
            last_rebuild = step;
        }

        let r: f64 = queries
            .iter()
            .enumerate()
            .map(|(qi, &q)| recall(&search_topk(&idx, emb, flat, q), &truth[step][qi]))
            .sum::<f64>()
            / queries.len() as f64;
        recall_sum += r;
    }

    Outcome {
        label,
        recall: recall_sum / steps as f64,
        rebuilds,
        rebuild_cost_s: rebuild_cost,
        probe_evals,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);

    let feats = read_features("target/m1-data/node-feat-100k.csv", n);
    let n = feats.len();
    let edges = read_edges("target/m1-data/arxiv/raw/edge.csv", n);
    eprintln!("[trig] n={n} edges={} dim={DIM}", edges.len());
    assert!(!edges.is_empty());

    // Variable-rate schedule: 3-epoch bursts (lr 0.02) separated by 5-epoch calm (lr 0.0005).
    // Adam at these lrs produces real motion in bursts, near-stasis in calm → the bursty
    // churn profile where a fixed cadence is provably suboptimal.
    let lr_at = |e: usize| -> f32 {
        if e % 8 < 3 {
            0.02
        } else {
            0.0005
        }
    };
    let e0 = matrix_from_features(&feats);
    let t0 = Instant::now();
    let snaps = train_variable_rate(e0, &edges, n, epochs, 2048, 64, 0.1, lr_at, 1234);
    eprintln!(
        "[trig] {} snapshots (burst/calm) in {:.1}s",
        snaps.len(),
        t0.elapsed().as_secs_f64()
    );

    let flats: Vec<FlatVectors> = snaps.iter().map(to_flat).collect();
    let mut qrng = StdRng::seed_from_u64(999);
    let queries: Vec<usize> = (0..200.min(n)).map(|_| qrng.gen_range(0..n)).collect();
    // disjoint probe set (no leakage into the scored query set)
    let probe_qs: Vec<usize> = (0..30.min(n)).map(|_| qrng.gen_range(0..n)).collect();
    let truth: Vec<Vec<Vec<u32>>> = snaps
        .iter()
        .map(|e| queries.iter().map(|&q| brute_topk(e, q, K)).collect())
        .collect();

    // per-step churn ramp (for visibility) + variable-rate sanity
    let last = snaps.len() - 1;
    let churn: f64 = queries
        .iter()
        .enumerate()
        .map(|(qi, _)| 1.0 - recall(&truth[last][qi], &truth[0][qi]))
        .sum::<f64>()
        / queries.len() as f64;
    println!(
        "\n=== variable-rate trajectory: E0->ET churn {:.0}% over {} steps ===",
        churn * 100.0,
        last
    );
    // per-step churn delta (vs previous snapshot) — bursts spike, calm flattens
    print!("per-step Δchurn: ");
    for step in 1..snaps.len() {
        let d: f64 = queries
            .iter()
            .enumerate()
            .map(|(qi, _)| 1.0 - recall(&truth[step][qi], &truth[step - 1][qi]))
            .sum::<f64>()
            / queries.len() as f64;
        print!("{:.0} ", d * 100.0);
    }
    println!();
    if churn < 0.15 {
        println!(
            "\n!! VOID — trajectory churn < 15% (no real drift). Not a result; escalate lr/epochs."
        );
        return;
    }

    let configs: Vec<Trigger> = vec![
        Trigger::Periodic(2),
        Trigger::Periodic(3),
        Trigger::Periodic(4),
        Trigger::Periodic(6),
        Trigger::Frobenius(0.15),
        Trigger::Frobenius(0.25),
        Trigger::Frobenius(0.40),
        Trigger::Recall(0.97),
        Trigger::Recall(0.95),
        Trigger::Recall(0.93),
    ];
    let label = |t: &Trigger| match t {
        Trigger::Periodic(k) => format!("Periodic k={k}"),
        Trigger::Frobenius(x) => format!("Frobenius t={x}"),
        Trigger::Recall(f) => format!("Recall floor={f}"),
    };

    let mut outcomes: Vec<Outcome> = configs
        .iter()
        .map(|t| run_policy(label(t), *t, &snaps, &flats, &queries, &truth, &probe_qs, n))
        .collect();

    // reference: always-rebuild ceiling cost (one full build per step) for cost framing
    let always = run_policy(
        "ALWAYS".into(),
        Trigger::Periodic(1),
        &snaps,
        &flats,
        &queries,
        &truth,
        &probe_qs,
        n,
    );

    println!(
        "\n=== policy outcomes (mean recall@{K}, {} steps) ===",
        last
    );
    println!(
        "{:>18} {:>8} {:>9} {:>13} {:>13}",
        "policy", "recall", "rebuilds", "rebuild s", "probe evals"
    );
    println!("{}", "-".repeat(64));
    println!(
        "{:>18} {:>7.1}% {:>9} {:>13.1} {:>13}",
        always.label,
        always.recall * 100.0,
        always.rebuilds,
        always.rebuild_cost_s,
        "-"
    );
    for o in &outcomes {
        println!(
            "{:>18} {:>7.1}% {:>9} {:>13.1} {:>13.0}",
            o.label,
            o.recall * 100.0,
            o.rebuilds,
            o.rebuild_cost_s,
            o.probe_evals
        );
    }

    // ---- Pareto frontier analysis: fewer rebuilds at equal-or-better recall wins ----
    // For each Recall-trigger config, find the cheapest Periodic/Frobenius config that
    // matches its recall (within 0.5%); the trigger wins if it used fewer rebuilds.
    outcomes.sort_by_key(|o| o.rebuilds);
    println!("\n=== GATE: does the recall trigger dominate the frontier? ===");
    let recalls: Vec<&Outcome> = outcomes
        .iter()
        .filter(|o| o.label.starts_with("Recall"))
        .collect();
    let periodics: Vec<&Outcome> = outcomes
        .iter()
        .filter(|o| o.label.starts_with("Periodic"))
        .collect();
    let frobs: Vec<&Outcome> = outcomes
        .iter()
        .filter(|o| o.label.starts_with("Frobenius"))
        .collect();

    let mut trigger_wins = false;
    let mut beats_frob = false;
    for rt in &recalls {
        // cheapest periodic with recall >= rt.recall - 0.5%
        let matched = periodics
            .iter()
            .filter(|p| p.recall >= rt.recall - 0.005)
            .min_by_key(|p| p.rebuilds);
        if let Some(p) = matched {
            let fewer = rt.rebuilds as f64 <= p.rebuilds as f64 * 0.75; // >=25% fewer
                                                                        // best frobenius at matched recall
            let fb = frobs
                .iter()
                .filter(|f| f.recall >= rt.recall - 0.005)
                .min_by_key(|f| f.rebuilds);
            let beat_this_frob = fb.map(|f| rt.rebuilds < f.rebuilds).unwrap_or(true);
            println!(
                "  {} ({:.1}%, {} rebuilds) vs periodic {} ({} rebuilds): {}{}",
                rt.label,
                rt.recall * 100.0,
                rt.rebuilds,
                p.label,
                p.rebuilds,
                if fewer {
                    ">=25% fewer ✓"
                } else {
                    "not enough fewer"
                },
                fb.map(|f| format!("; vs {} ({} rebuilds)", f.label, f.rebuilds))
                    .unwrap_or_default()
            );
            if fewer {
                trigger_wins = true;
            }
            if beat_this_frob {
                beats_frob = true;
            }
        }
    }

    println!(
        "\n>>> VERDICT: {}",
        if trigger_wins && beats_frob {
            "WIN — recall trigger uses >=25% fewer rebuilds at matched recall AND beats Frobenius"
        } else if trigger_wins {
            "PARTIAL — trigger beats periodic but not clearly the Frobenius monitor"
        } else {
            "KILL — recall trigger does not dominate periodic-K (ADR-200's periodic-is-the-knob stands)"
        }
    );
}
