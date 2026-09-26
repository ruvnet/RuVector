//! Feasibility probe for the erasure audit.
//!
//! The first full run of `erasure-audit-bench` returned a paired accuracy of
//! *exactly* 0.5000 on every feature and every mode — i.e. every A/B pair tied
//! bit-for-bit. That is a setup diagnosis, not a finding, so this probe
//! measures the three things that could produce it:
//!
//! 1. Is the corpus so tightly clustered that recall@10 is meaningless
//!    (ground-truth neighbours are effectively ties)?
//! 2. Does inserting-then-erasing a vector change the graph at all — how many
//!    neighbour lists differ between the A and B indexes?
//! 3. Do those structural differences ever reach the adversary's observable
//!    query results?
//!
//! ```text
//! cargo run --release -p ruvector-erasure-audit --bin erasure-probe
//! ```

use ruvector_erasure_audit::audit::{observe, ProbeConfig};
use ruvector_erasure_audit::data::{ClusteredSource, DatasetConfig};
use ruvector_erasure_audit::erasure::{erase, ErasureMode};
use ruvector_hnsw_repair::{brute_force_knn_live, HnswConfig, HnswGraph};
use std::collections::HashSet;

const DIM: usize = 32;
const BASE_N: usize = 2_000;

fn hcfg() -> HnswConfig {
    HnswConfig {
        dim: DIM,
        m: 8,
        m0: 16,
        ef_construction: 40,
        ml: 1.0 / (8f64.ln()),
    }
}

fn build(src: &mut ClusteredSource, n: usize) -> HnswGraph {
    let mut g = HnswGraph::new(hcfg());
    for v in src.sample_many(n) {
        g.insert(v);
    }
    g
}

fn recall10(g: &HnswGraph, qs: &[Vec<f32>], ef: usize) -> f64 {
    let gt = brute_force_knn_live(g, qs, 10);
    let mut t = 0.0;
    for (i, q) in qs.iter().enumerate() {
        let s: HashSet<u32> = gt[i].iter().copied().collect();
        t += g.search(q, 10, ef).iter().filter(|x| s.contains(x)).count() as f64 / 10.0;
    }
    t / qs.len() as f64
}

/// Mean gap between the 1st and 10th true nearest-neighbour distance,
/// normalised by the 1st. Near zero means the top-10 are ties and recall@10
/// cannot discriminate.
fn gt_tie_ratio(g: &HnswGraph, qs: &[Vec<f32>]) -> f64 {
    let dim = g.config.dim;
    let mut acc = 0.0;
    for q in qs {
        let mut d: Vec<f32> = g
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !g.deleted[*i])
            .map(|(_, v)| ruvector_hnsw_repair::l2_sq(q, v, dim))
            .collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        acc += ((d[9] - d[0]) / d[0].max(1e-9)) as f64;
    }
    acc / qs.len() as f64
}

fn count_list_diffs(a: &HnswGraph, b: &HnswGraph) -> usize {
    let mut n = 0;
    for l in 0..a.layers.len().min(b.layers.len()) {
        for i in 0..a.layers[l].len().min(b.layers[l].len()) {
            let mut x = a.layers[l][i].clone();
            let mut y = b.layers[l][i].clone();
            x.sort_unstable();
            y.sort_unstable();
            if x != y {
                n += 1;
            }
        }
    }
    n
}

fn main() {
    println!("=== erasure-audit feasibility probe ===\n");

    // --- Probe 1: dataset geometry vs recall -------------------------------
    println!("Probe 1: corpus geometry (base_n={BASE_N}, dim={DIM})");
    println!(
        "{:>9} {:>9} {:>12} {:>12} {:>14}",
        "clusters", "sigma", "recall@10", "recall_ef256", "gt_tie_ratio"
    );
    for (clusters, sigma) in [
        (16usize, 0.18f32),
        (16, 0.35),
        (64, 0.35),
        (64, 0.60),
        (256, 0.60),
        (2000, 0.60),
    ] {
        let cfg = DatasetConfig {
            dim: DIM,
            clusters,
            sigma,
            seed: 0x5EED_0001,
        };
        let mut src = ClusteredSource::new(cfg.clone());
        let g = build(&mut src, BASE_N);
        let mut qsrc = ClusteredSource::new(DatasetConfig {
            seed: 0x9999_0001,
            ..cfg
        });
        let qs = qsrc.sample_many(100);
        println!(
            "{:>9} {:>9.2} {:>12.4} {:>12.4} {:>14.4}",
            clusters,
            sigma,
            recall10(&g, &qs, 64),
            recall10(&g, &qs, 256),
            gt_tie_ratio(&g, &qs)
        );
    }

    // --- Probe 2 & 3: structural imprint and observability ------------------
    println!("\nProbe 2/3: structural imprint of insert+erase, and whether it is observable");
    println!(
        "{:>9} {:>9} {:>14} {:>14} {:>16} {:>16}",
        "clusters", "sigma", "mode", "lists_diff", "obs_diff_rate", "mean_|dsum_a-b|"
    );
    let probe = ProbeConfig::default();
    for (clusters, sigma) in [(16usize, 0.18f32), (64, 0.60), (2000, 0.60)] {
        let cfg = DatasetConfig {
            dim: DIM,
            clusters,
            sigma,
            seed: 0x5EED_0001,
        };
        let mut src = ClusteredSource::new(cfg.clone());
        let base = build(&mut src, BASE_N);
        for mode in [
            ErasureMode::Tombstone,
            ErasureMode::EagerRepair,
            ErasureMode::LocalRebuild { ef_rebuild: 48 },
        ] {
            let mut tsrc = ClusteredSource::new(DatasetConfig {
                seed: 0x7A46_0001,
                ..cfg.clone()
            });
            let mut dsrc = ClusteredSource::new(DatasetConfig {
                seed: 0x7A46_0002,
                ..cfg.clone()
            });
            let trials = 40usize;
            let mut diffs = 0usize;
            let mut obs_diff = 0usize;
            let mut dsum_gap = 0.0f64;
            for _ in 0..trials {
                let target = tsrc.sample();
                let decoy = dsrc.sample();
                let mut ga = base.clone();
                let ida = ga.insert(target.clone()) as usize;
                erase(&mut ga, ida, mode);
                let mut gb = base.clone();
                let idb = gb.insert(decoy) as usize;
                erase(&mut gb, idb, mode);
                diffs += count_list_diffs(&ga, &gb);
                let fa = observe(&ga, &target, &probe);
                let fb = observe(&gb, &target, &probe);
                if fa.0 != fb.0 {
                    obs_diff += 1;
                }
                dsum_gap += (fa.0[0] - fb.0[0]).abs();
            }
            println!(
                "{:>9} {:>9.2} {:>14} {:>14.2} {:>16.3} {:>16.6}",
                clusters,
                sigma,
                mode.label(),
                diffs as f64 / trials as f64,
                obs_diff as f64 / trials as f64,
                dsum_gap / trials as f64
            );
        }
    }
}
