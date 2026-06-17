//! BET 3 treewidth probe (ADR-203 / issue #534): is a *curated, bounded-degree*
//! knowledge graph low-treewidth enough that CCH contraction/build stays cheap —
//! the one backbone untested when CCH died on high-treewidth embedding/citation
//! graphs (ADR-199)?
//!
//! This is the frozen go/no-go gate of `docs/plans/bet3-kg-treewidth/
//! PRE-REGISTRATION.md`. It reuses the validated `ruvector-seprag` kernel
//! (`SepRag::build_with`, `blowup_ratio`, `elim_depth`, the Dijkstra oracle) and
//! adds only what the gate needs:
//!   * a generic KG/edge loader (WordNet / Freebase / Wikidata triple formats);
//!   * a scale sweep n ∈ {2k,5k,10k,20k} with an OLS log-log fit of the
//!     elimination-tree-height exponent p (elim_h ∝ n^p) — THE primary metric;
//!   * both separators (Balanced + BfsLayer) so the heuristic's contribution is
//!     visible;
//!   * a minor-min-width treewidth LOWER bound, so a NO-GO is structurally
//!     certain rather than a weak-heuristic artifact (the upgrade over ADR-199).
//!
//! Every KG result is printed bracketed between the two calibrated controls
//! (roadNet-PA = road-like p≈0.5; ogbn-arxiv = expander p≈1.0) IN THE SAME RUN.
//! If the road control does not reproduce ~√n / ~7.6×, the run is VOID.
//!
//! Run (controls live on disk; KG triples under target/m1-data/kg/):
//!   cargo run --release -p ruvector-seprag --example kg_treewidth_probe

use ruvector_seprag::graph::{Graph, NodeId};
use ruvector_seprag::query::{elim_depth, KnnIndex, QueryStats};
use ruvector_seprag::{gen, SepRag, SeparatorKind};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// One backbone source: a set of edge/triple files sharing a parse format.
struct Source {
    name: &'static str,
    role: &'static str, // "control-good" | "reference-bad" | "kg"
    paths: Vec<String>,
    sep: &'static [char],
    comment: Option<char>,
    skip_first: bool,    // OpenKE count-header line
    cols: (usize, usize), // 0-based endpoint columns (head, tail)
}

const DATA: &str = "target/m1-data";

fn sources() -> Vec<Source> {
    let kg = |f: &str| format!("{DATA}/kg/{f}");
    vec![
        // 🟢 calibration control — MUST reproduce ~√n / ~7.6× or the run is VOID.
        Source {
            name: "roadNet-PA (control)",
            role: "control-good",
            paths: vec![format!("{DATA}/roadNet-PA.txt")],
            sep: &['\t', ' '],
            comment: Some('#'),
            skip_first: false,
            cols: (0, 1),
        },
        // 🔴 known-NO-GO reference (ADR-199).
        Source {
            name: "ogbn-arxiv citation (ref)",
            role: "reference-bad",
            paths: vec![format!("{DATA}/arxiv/raw/edge.csv")],
            sep: &[',', '\t', ' '],
            comment: Some('#'),
            skip_first: false,
            cols: (0, 1),
        },
        // KGs under test — sparse hierarchical → hub-heavy → genuine Wikidata.
        Source {
            name: "WN18RR (WordNet)",
            role: "kg",
            paths: vec![kg("wn18rr_train.txt"), kg("wn18rr_valid.txt"), kg("wn18rr_test.txt")],
            sep: &['\t'],
            comment: None,
            skip_first: false,
            cols: (0, 2),
        },
        Source {
            name: "FB15k-237 (Freebase)",
            role: "kg",
            paths: vec![
                kg("fb15k237_train2id.txt"),
                kg("fb15k237_valid2id.txt"),
                kg("fb15k237_test2id.txt"),
            ],
            sep: &[' ', '\t'],
            comment: None,
            skip_first: true, // count header
            cols: (0, 1),     // OpenKE order: head tail relation
        },
        Source {
            name: "CoDEx-L (Wikidata)",
            role: "kg",
            paths: vec![kg("codex_l_train.txt"), kg("codex_l_valid.txt"), kg("codex_l_test.txt")],
            sep: &['\t'],
            comment: None,
            skip_first: false,
            cols: (0, 2),
        },
    ]
}

/// Scale-sweep points (capped at the backbone's giant-component size).
const SWEEP: [usize; 4] = [2_000, 5_000, 10_000, 20_000];
/// MMW lower bound is O(n²)-ish; cap its input for runtime (diagnostic only).
const MMW_CAP: usize = 5_000;
/// Stop growing n once a single Balanced build exceeds this — a slow contraction
/// at moderate n is the high-treewidth signal (fill-in ≈ tw²·n).
const BUILD_BUDGET_S: f64 = 10.0;
/// Only run the BfsLayer attribution control while subgraphs are this small.
const LAYER_CAP: usize = 10_000;

fn main() {
    let only = std::env::args().nth(1); // optional substring filter
    println!("=== BET 3 — curated-KG treewidth probe (frozen gate) ===\n");

    let mut summary: Vec<Row> = Vec::new();
    for src in sources() {
        if let Some(f) = &only {
            if !src.name.to_lowercase().contains(&f.to_lowercase()) {
                continue;
            }
        }
        match probe(&src) {
            Some(row) => summary.push(row),
            None => eprintln!("[skip] {} — data missing/empty", src.name),
        }
    }

    print_summary(&summary);
}

#[derive(Clone)]
struct Row {
    name: String,
    role: String,
    p_balanced: f64,
    p_layer: f64,
    blowup_max: f64,
    blowup_ref: f64, // blowup at the ADR-199-matched reference n (~2k), first sweep point
    n_ref: usize,
    elim_h_max: usize,
    n_max: usize,
    top_sep: usize,
    avg_deg: f64,
    max_deg: usize,
    recall_ok: usize,
    recall_q: usize,
    mmw_lb: usize,
    mmw_n: usize,
}

fn probe(src: &Source) -> Option<Row> {
    let t0 = Instant::now();
    let adj = match load_adjacency(src) {
        Some(a) if !a.is_empty() => a,
        _ => return None,
    };
    let nz: usize = adj.iter().filter(|r| !r.is_empty()).count();
    let edges: usize = adj.iter().map(Vec::len).sum::<usize>() / 2;
    println!("──────────────────────────────────────────────────────────────");
    println!(
        "▶ {}  [{}]\n  full graph: {} nodes ({} non-isolated), {} undirected edges, loaded {:.1}s",
        src.name,
        src.role,
        adj.len(),
        nz,
        edges,
        t0.elapsed().as_secs_f64()
    );

    // Seed BFS from the max-degree node → lands in the giant component.
    let seed = (0..adj.len())
        .max_by_key(|&i| adj[i].len())
        .unwrap_or(0) as u32;

    let mut ns = Vec::new();
    let mut hs_bal = Vec::new();
    let mut ns_lay = Vec::new();
    let mut hs_lay = Vec::new();
    let mut blow_bal_max = 0.0f64;
    let mut blow_ref = 0.0f64;
    let mut n_ref = 0usize;
    let mut elim_h_max = 0usize;
    let mut n_max = 0usize;
    let mut top_sep = 0usize;
    let mut avg_deg = 0.0;
    let mut max_deg = 0;
    let (mut rec_ok, mut rec_q) = (0usize, 0usize);
    let mut mmw_lb = 0usize;
    let mut mmw_n = 0usize;

    println!(
        "  {:>7} | {:>7} | {:>9} {:>8} | {:>9} {:>8} | {:>8} | {:>7}",
        "n", "edges", "elim_h↓B", "blowupB", "elim_h↓L", "blowupL", "build_s", "topsep"
    );

    for &target in &SWEEP {
        let g = bfs_ball(&adj, seed, target);
        if g.n < 500 {
            // backbone smaller than this sweep point — record what we have, stop.
            if g.n < 64 {
                break;
            }
        }
        let m = g.edges().count();
        let degs: Vec<usize> = g.adj.iter().map(Vec::len).collect();
        let amax = *degs.iter().max().unwrap_or(&0);
        let aavg = if g.n > 0 { 2.0 * m as f64 / g.n as f64 } else { 0.0 };

        // Balanced (the default / verdict separator).
        let tb = Instant::now();
        let sb = SepRag::build_with(g.clone(), SeparatorKind::Balanced);
        let build_s = tb.elapsed().as_secs_f64();
        let hb = (0..sb.graph.n as u32).map(|r| elim_depth(&sb.topo, r)).max().unwrap_or(0);
        let blow_b = sb.blowup_ratio();
        let tsep = sb.sep_tree.nodes[sb.sep_tree.root].separator.len();

        // Build-time budget: a high-treewidth backbone's contraction explodes
        // super-linearly (fill-in ≈ tw²·n), so a slow build at moderate n is
        // itself the NO-GO signal. Record this point, then stop growing n.
        // PREDICTIVE: even a *moderately* slow build predicts the next (larger,
        // ~2.5×) point will blow past the budget, so cap before attempting it —
        // a single high-treewidth build can otherwise run effectively forever.
        let stop_after = build_s > BUILD_BUDGET_S * 0.5;

        // BfsLayer (heuristic-attribution control) — only while cheap; 2 points
        // suffice for the attribution exponent.
        if g.n <= LAYER_CAP && !stop_after {
            let sl = SepRag::build_with(g.clone(), SeparatorKind::BfsLayer);
            let hl = (0..sl.graph.n as u32).map(|r| elim_depth(&sl.topo, r)).max().unwrap_or(0);
            let blow_l = sl.blowup_ratio();
            println!(
                "  {:>7} | {:>7} | {:>9} {:>7.1}x | {:>9} {:>7.1}x | {:>8.2} | {:>7}",
                g.n, m, hb, blow_b, hl, blow_l, build_s, tsep
            );
            ns_lay.push(g.n as f64);
            hs_lay.push(hl.max(1) as f64);
        } else {
            println!(
                "  {:>7} | {:>7} | {:>9} {:>7.1}x | {:>9} {:>8} | {:>8.2} | {:>7}",
                g.n, m, hb, blow_b, "—", "—", build_s, tsep
            );
        }

        // First sweep point ≈ ADR-199's N≈1.5–2k anchor → the matched-n blowup read.
        if n_ref == 0 {
            n_ref = g.n;
            blow_ref = blow_b;
        }

        ns.push(g.n as f64);
        hs_bal.push(hb.max(1) as f64);

        // Track stats at the LARGEST n reached (the verdict point).
        if g.n >= n_max {
            n_max = g.n;
            blow_bal_max = blow_b;
            elim_h_max = hb;
            top_sep = tsep;
            avg_deg = aavg;
            max_deg = amax;

            // Sampled Dijkstra-oracle recall@10 sanity (correctness, not gated).
            let pois = gen::sample_pois(sb.graph.n, sb.graph.n / 2, 7);
            let srcs = gen::sample_pois(sb.graph.n, 30, 13);
            let idx = KnnIndex::build(&sb.topo, &sb.metric, &pois);
            let (mut ok, mut q) = (0usize, 0usize);
            for &s in &srcs {
                let oracle = sb.graph.knn_oracle(s, &pois, 10);
                let mut st = QueryStats::default();
                let got = idx.knn(s, 10, true, &mut st);
                if dist_multiset_eq(&got, &oracle) {
                    ok += 1;
                }
                q += 1;
            }
            rec_ok = ok;
            rec_q = q;
        }

        // MMW lower bound at the largest sweep point ≤ cap.
        if g.n <= MMW_CAP && g.n > mmw_n {
            mmw_lb = minor_min_width(&g);
            mmw_n = g.n;
        }

        if stop_after {
            println!("  [build-budget reached at n={} ({:.1}s > {:.0}s) — high-treewidth signal; capping sweep]", g.n, build_s, BUILD_BUDGET_S);
            break;
        }
        if g.n < target {
            break; // backbone exhausted (giant component smaller than target)
        }
    }

    let p_bal = fit_exponent(&ns, &hs_bal);
    let p_lay = fit_exponent(&ns_lay, &hs_lay);

    println!(
        "  → fitted exponent p (elim_h ∝ n^p):  Balanced {:.3}   BfsLayer {:.3}",
        p_bal, p_lay
    );
    println!(
        "  → @n={}: blowup {:.1}x, elim_h {} ({:.2}·n), top-sep {}, avg_deg {:.1}, max_deg {}",
        n_max,
        blow_bal_max,
        elim_h_max,
        elim_h_max as f64 / n_max.max(1) as f64,
        top_sep,
        avg_deg,
        max_deg
    );
    println!(
        "  → treewidth LOWER bound (minor-min-width) @n={}: tw ≥ {}   ({:.2}·n)",
        mmw_n,
        mmw_lb,
        mmw_lb as f64 / mmw_n.max(1) as f64
    );
    println!("  → recall sanity: {rec_ok}/{rec_q} queries match Dijkstra oracle\n");

    Some(Row {
        name: src.name.to_string(),
        role: src.role.to_string(),
        p_balanced: p_bal,
        p_layer: p_lay,
        blowup_max: blow_bal_max,
        blowup_ref: blow_ref,
        n_ref,
        elim_h_max,
        n_max,
        top_sep,
        avg_deg,
        max_deg,
        recall_ok: rec_ok,
        recall_q: rec_q,
        mmw_lb,
        mmw_n,
    })
}

// ── verdict table + automated gate ──────────────────────────────────────────

fn print_summary(rows: &[Row]) {
    println!("\n══════════════════ SUMMARY (frozen gate) ══════════════════");
    println!(
        "{:<28} {:>6} {:>6} {:>10} {:>10} {:>9} {:>10}",
        "backbone", "p_bal", "p_lay", "blow@ref", "blow@max", "elim/n", "tw_lb/n"
    );
    for r in rows {
        println!(
            "{:<28} {:>6.3} {:>6.3} {:>7.1}x@{:<4} {:>7.1}x@{:<4} {:>9.2} {:>10.2}",
            r.name,
            r.p_balanced,
            r.p_layer,
            r.blowup_ref,
            kn(r.n_ref),
            r.blowup_max,
            kn(r.n_max),
            r.elim_h_max as f64 / r.n_max.max(1) as f64,
            r.mmw_lb as f64 / r.mmw_n.max(1) as f64,
        );
    }

    // VOID check: the road control is the in-run calibration anchor. It is VALID
    // iff it reproduces ADR-199's road-like signature — a sub-linear exponent
    // (p≈0.5) AND matched-n blowup ≈ ADR-199's 7.6×@~1.5k. Blowup grows with n
    // even for road networks, so the absolute read is taken at the reference n,
    // never at the largest n (that mis-anchoring would fail even a perfect road).
    let control = rows.iter().find(|r| r.role == "control-good");
    let void = match control {
        Some(c) => !(c.p_balanced <= 0.7 && c.blowup_ref <= 10.0),
        None => true,
    };
    println!("\n── gate application (PRIMARY metric = exponent p; blowup read at ADR-199-matched ref n) ──");
    if void {
        println!(
            "VOID — road control did not reproduce road-like signature (p={:.3}, blow@ref={:.1}x). \
             Probe miscalibrated; verdicts below are NOT valid.",
            control.map(|c| c.p_balanced).unwrap_or(f64::NAN),
            control.map(|c| c.blowup_ref).unwrap_or(f64::NAN),
        );
    } else {
        let c = control.unwrap();
        println!(
            "control OK (road p={:.3}, blow@ref={:.1}x ≈ √n at n={}) → verdicts valid.",
            c.p_balanced, c.blowup_ref, c.n_ref
        );
    }

    for r in rows.iter().filter(|r| r.role == "kg") {
        let verdict = if void {
            "VOID"
        } else if r.p_balanced <= 0.6 && r.blowup_ref <= 10.0 {
            "GO"
        } else if r.p_balanced >= 0.8 || r.blowup_ref >= 23.0 {
            "NO-GO (KILL)"
        } else {
            "INCONCLUSIVE"
        };
        // Lower-bound robustness note: is a NO-GO structurally certain?
        let lb_frac = r.mmw_lb as f64 / r.mmw_n.max(1) as f64;
        let lb_note = if r.mmw_lb >= 50 && lb_frac >= 0.05 {
            " [tw lower bound also large → structurally certain]"
        } else if verdict.starts_with("NO-GO") {
            " [lower bound modest → upper-bound-driven]"
        } else {
            ""
        };
        println!(
            "  {:<28} → {}{}  (p={:.3}, blow@ref={:.1}x, blow@max={:.1}x, tw≥{})",
            r.name, verdict, lb_note, r.p_balanced, r.blowup_ref, r.blowup_max, r.mmw_lb
        );
    }
}

/// Compact n label, e.g. 2000 → "2k", 20000 → "20k".
fn kn(n: usize) -> String {
    if n >= 1000 && n % 1000 == 0 {
        format!("{}k", n / 1000)
    } else if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

// ── loaders & graph extraction ──────────────────────────────────────────────

/// Load all of a source's files into one undirected adjacency (dense ids via
/// string interning). Relation labels are dropped — the bet is about structure.
fn load_adjacency(src: &Source) -> Option<Vec<Vec<u32>>> {
    let mut intern: HashMap<String, u32> = HashMap::new();
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut any = false;
    for path in &src.paths {
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        any = true;
        for (li, line) in data.lines().enumerate() {
            if src.skip_first && li == 0 {
                continue;
            }
            if line.is_empty() {
                continue;
            }
            if let Some(c) = src.comment {
                if line.starts_with(c) {
                    continue;
                }
            }
            let f: Vec<&str> = line.split(src.sep).filter(|s| !s.is_empty()).collect();
            let (i, j) = src.cols;
            if f.len() <= i.max(j) {
                continue;
            }
            let u = intern_id(&mut intern, f[i]);
            let v = intern_id(&mut intern, f[j]);
            if u != v {
                edges.push((u, v));
            }
        }
    }
    if !any {
        return None;
    }
    let n = intern.len();
    let mut adj = vec![Vec::new(); n];
    for (u, v) in edges {
        adj[u as usize].push(v);
        adj[v as usize].push(u);
    }
    Some(adj)
}

fn intern_id(map: &mut HashMap<String, u32>, s: &str) -> u32 {
    let next = map.len() as u32;
    *map.entry(s.to_string()).or_insert(next)
}

/// Induced connected subgraph: BFS from `seed` collecting up to `n_target`
/// nodes, unit edge weights (hop distance). Mirrors m1_arxiv's bfs_ball.
fn bfs_ball(adj: &[Vec<u32>], seed: u32, n_target: usize) -> Graph {
    let mut order = Vec::new();
    let mut seen = vec![false; adj.len()];
    let mut q = VecDeque::from([seed]);
    seen[seed as usize] = true;
    while let Some(u) = q.pop_front() {
        order.push(u);
        if order.len() >= n_target {
            break;
        }
        for &v in &adj[u as usize] {
            if !seen[v as usize] {
                seen[v as usize] = true;
                q.push_back(v);
            }
        }
    }
    let mut remap = vec![u32::MAX; adj.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old as usize] = new as u32;
    }
    let mut g = Graph::new(order.len());
    for &old in &order {
        let nu = remap[old as usize];
        for &v in &adj[old as usize] {
            let nv = remap[v as usize];
            if nv != u32::MAX && nu < nv {
                g.add_edge(nu, nv, 1.0);
            }
        }
    }
    g
}

// ── treewidth lower bound (minor-min-width / MMD) ────────────────────────────

/// Minor-min-width: a standard cheap treewidth LOWER bound. Repeatedly take a
/// minimum-degree vertex v, record its degree, then contract v into its
/// minimum-degree neighbour (forming a minor); the max degree-at-removal is a
/// valid lower bound on treewidth. A large value here means the high treewidth
/// is *structural*, not an artifact of our separator heuristic.
fn minor_min_width(g: &Graph) -> usize {
    use std::collections::HashSet;
    let mut nb: Vec<HashSet<u32>> = vec![HashSet::new(); g.n];
    for (u, row) in g.adj.iter().enumerate() {
        for &(v, _) in row {
            nb[u].insert(v);
        }
    }
    let mut alive: Vec<bool> = vec![true; g.n];
    let mut remaining = g.n;
    let mut lb = 0usize;

    while remaining > 0 {
        // Min-degree alive vertex.
        let mut v = usize::MAX;
        let mut dv = usize::MAX;
        for i in 0..g.n {
            if alive[i] && nb[i].len() < dv {
                dv = nb[i].len();
                v = i;
                if dv == 0 {
                    break;
                }
            }
        }
        if v == usize::MAX {
            break;
        }
        lb = lb.max(dv);

        if dv == 0 {
            alive[v] = false;
            remaining -= 1;
            continue;
        }
        // Min-degree neighbour u of v → contract v into u.
        let u = *nb[v]
            .iter()
            .min_by_key(|&&w| nb[w as usize].len())
            .unwrap() as usize;
        // Move v's other neighbours onto u.
        let vn: Vec<u32> = nb[v].iter().copied().collect();
        for w in vn {
            let w = w as usize;
            nb[w].remove(&(v as u32));
            if w != u {
                nb[u].insert(w as u32);
                nb[w].insert(u as u32);
            }
        }
        nb[u].remove(&(u as u32));
        alive[v] = false;
        nb[v].clear();
        remaining -= 1;
    }
    lb
}

// ── OLS log-log exponent fit + helpers ───────────────────────────────────────

/// Fit p in y ≈ a·n^p via least squares on (ln n, ln y). Needs ≥2 points.
fn fit_exponent(ns: &[f64], ys: &[f64]) -> f64 {
    let k = ns.len();
    if k < 2 {
        return f64::NAN;
    }
    let lx: Vec<f64> = ns.iter().map(|x| x.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|y| y.ln()).collect();
    let mx = lx.iter().sum::<f64>() / k as f64;
    let my = ly.iter().sum::<f64>() / k as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..k {
        num += (lx[i] - mx) * (ly[i] - my);
        den += (lx[i] - mx) * (lx[i] - mx);
    }
    if den.abs() < 1e-12 {
        f64::NAN
    } else {
        num / den
    }
}

fn dist_multiset_eq(a: &[(NodeId, f64)], b: &[(NodeId, f64)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut da: Vec<f64> = a.iter().map(|x| x.1).collect();
    let mut db: Vec<f64> = b.iter().map(|x| x.1).collect();
    da.sort_by(|x, y| x.partial_cmp(y).unwrap());
    db.sort_by(|x, y| x.partial_cmp(y).unwrap());
    da.iter().zip(&db).all(|(x, y)| (x - y).abs() < 1e-9)
}
