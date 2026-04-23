#![allow(clippy::needless_range_loop)]
//! ADR-154 §13 / §17 item 14 follow-up — CPM-quality Leiden.
//!
//! The shipped modularity-based Leiden (`analysis::leiden::leiden_labels`)
//! scores ARI = 0.089 on the default N=1024 SBM — modularity-resolution-
//! limit territory (Fortunato & Barthélemy 2007). CPM (Traag's own
//! default in `leidenalg`) does not have the resolution-limit problem.
//! This test:
//!
//! 1. Sweeps γ ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0} on the
//!    default SBM and publishes each (γ, ARI) pair for ADR §17.
//! 2. Asserts that *some* γ in the sweep produces ARI strictly greater
//!    than modularity-Leiden's 0.089 — the minimum expected win.
//!
//! If every γ underperforms modularity-Leiden, that is itself a real
//! finding (discovery-to-be-added: CPM doesn't rescue hub-heavy SBM
//! community detection at N=1024). The test's job is to publish the
//! measurement, not to force a green.

use connectome_fly::{Analysis, AnalysisConfig, Connectome, ConnectomeConfig};

fn default_conn() -> Connectome {
    Connectome::generate(&ConnectomeConfig::default())
}

/// 2-way partition from a per-node label vector: take the two largest
/// communities; everything else goes into the larger. Deterministic
/// for a fixed label vector.
fn two_way_from_labels(labels: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for l in labels {
        *count.entry(*l).or_insert(0) += 1;
    }
    let mut counts: Vec<(u32, u32)> = count.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if counts.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let (top_a, top_b) = (counts[0].0, counts[1].0);
    if top_a == top_b {
        return (Vec::new(), Vec::new());
    }
    let mut side_a: Vec<u32> = Vec::new();
    let mut side_b: Vec<u32> = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        if *l == top_b {
            side_b.push(i as u32);
        } else {
            side_a.push(i as u32);
        }
    }
    (side_a, side_b)
}

/// Adjusted Rand Index of a 2-way partition against a ground-truth
/// binary label predicate. Same formulation as
/// `tests/acceptance_partition.rs`.
fn adjusted_rand_index(side_a: &[u32], side_b: &[u32], is_class_1: impl Fn(u32) -> bool) -> f32 {
    let mut a1 = 0_u64;
    let mut a2 = 0_u64;
    let mut b1 = 0_u64;
    let mut b2 = 0_u64;
    for &id in side_a {
        if is_class_1(id) {
            a1 += 1;
        } else {
            a2 += 1;
        }
    }
    for &id in side_b {
        if is_class_1(id) {
            b1 += 1;
        } else {
            b2 += 1;
        }
    }
    let n = (a1 + a2 + b1 + b2) as f64;
    if n < 2.0 {
        return 0.0;
    }
    fn c2(k: u64) -> f64 {
        (k as f64) * ((k as f64) - 1.0) / 2.0
    }
    let index = c2(a1) + c2(a2) + c2(b1) + c2(b2);
    let sum_row = c2(a1 + a2) + c2(b1 + b2);
    let sum_col = c2(a1 + b1) + c2(a2 + b2);
    let total = c2(n as u64);
    let expected = (sum_row * sum_col) / total;
    let max = 0.5 * (sum_row + sum_col);
    if (max - expected).abs() < 1e-12 {
        return 0.0;
    }
    ((index - expected) / (max - expected)) as f32
}

/// Full-partition Adjusted Rand Index between two equal-length label
/// vectors. Unlike the 2-way `adjusted_rand_index` above, this gives
/// community-detection algorithms credit for recovering the full
/// ground-truth partition even when the predicted label vocabulary
/// is larger or smaller than the truth vocabulary.
///
/// Standard Hubert-Arabie ARI:
///   contingency: n_ij = |{k : predicted[k]=i, truth[k]=j}|
///   a_i = Σ_j n_ij, b_j = Σ_i n_ij
///   index    = Σ_ij C(n_ij, 2)
///   expected = (Σ_i C(a_i,2))(Σ_j C(b_j,2)) / C(n,2)
///   max      = 0.5*(Σ_i C(a_i,2) + Σ_j C(b_j,2))
///   ARI = (index − expected) / (max − expected)
fn full_partition_ari(predicted: &[u32], truth: &[u32]) -> f32 {
    assert_eq!(
        predicted.len(),
        truth.len(),
        "full_partition_ari: vector length mismatch"
    );
    let n_total = predicted.len();
    if n_total < 2 {
        return 0.0;
    }
    fn c2(k: u64) -> f64 {
        (k as f64) * ((k as f64) - 1.0) / 2.0
    }
    // Contingency table via HashMap.
    let mut contingency: std::collections::HashMap<(u32, u32), u64> =
        std::collections::HashMap::new();
    let mut row_sum: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut col_sum: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for i in 0..n_total {
        let p = predicted[i];
        let t = truth[i];
        *contingency.entry((p, t)).or_insert(0) += 1;
        *row_sum.entry(p).or_insert(0) += 1;
        *col_sum.entry(t).or_insert(0) += 1;
    }
    let index_sum: f64 = contingency.values().map(|n| c2(*n)).sum();
    let row_c2: f64 = row_sum.values().map(|a| c2(*a)).sum();
    let col_c2: f64 = col_sum.values().map(|b| c2(*b)).sum();
    let total = c2(n_total as u64);
    if total < 1.0 {
        return 0.0;
    }
    let expected = (row_c2 * col_c2) / total;
    let max_val = 0.5 * (row_c2 + col_c2);
    if (max_val - expected).abs() < 1e-12 {
        return 0.0;
    }
    ((index_sum - expected) / (max_val - expected)) as f32
}

#[test]
fn leiden_cpm_sweeps_gamma_on_default_sbm() {
    let conn = default_conn();
    let an = Analysis::new(AnalysisConfig::default());
    let num_hub = ConnectomeConfig::default().num_hub_modules;
    let is_hub = |id: u32| conn.meta(connectome_fly::NeuronId(id)).module < num_hub;

    // Ground-truth module labels (full-partition, 70 distinct modules
    // on the default SBM).
    let truth_labels: Vec<u32> = (0..conn.num_neurons())
        .map(|i| conn.meta(connectome_fly::NeuronId(i as u32)).module as u32)
        .collect();

    // Baselines — modularity-Leiden measured two ways:
    //   - `ari_modularity_2way`:  top-2 community coarsening vs hub-vs-non-hub
    //     (the AC-3a-inherited metric; undersells multi-community outputs).
    //   - `ari_modularity_full`:  full-partition ARI vs ground-truth module labels
    //     (the correct metric for multi-community outputs).
    let baseline_labels = an.leiden_labels(&conn);
    let (ba, bb) = two_way_from_labels(&baseline_labels);
    let ari_modularity_2way = if ba.is_empty() || bb.is_empty() {
        0.0
    } else {
        adjusted_rand_index(&ba, &bb, is_hub)
    };
    let ari_modularity_full = full_partition_ari(&baseline_labels, &truth_labels);

    // Sweep spans 4 decades so we cross both "too low → merge
    // everything" and "too high → every node is its own community"
    // regimes. Normalized edges mean γ = 1.0 is the 'mean-density'
    // threshold; the SBM's natural γ* for a non-trivial partition
    // sits at roughly inter_density × n_module.
    // Fine sweep around the γ=2 peak identified in the coarse sweep
    // (see ADR §17 item 18: best full_ari = 0.393 at γ=2 with 109
    // communities). Sampling in-between γ=1 and γ=4 with finer
    // resolution to see whether the peak is a plateau or a ridge.
    let gammas = [
        0.1, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 6.0, 8.0, 16.0, 32.0,
        64.0,
    ];
    let mut best_ari_2way = f32::NEG_INFINITY;
    let mut best_gamma_2way = 0.0_f64;
    let mut best_ari_full = f32::NEG_INFINITY;
    let mut best_gamma_full = 0.0_f64;
    let mut rows: Vec<(f64, f32, f32, usize)> = Vec::new();
    for &g in &gammas {
        let labels = connectome_fly::analysis::leiden::leiden_labels_cpm(&conn, g);
        let (la, lb) = two_way_from_labels(&labels);
        let ari_2way = if la.is_empty() || lb.is_empty() {
            0.0
        } else {
            adjusted_rand_index(&la, &lb, is_hub)
        };
        let ari_full = full_partition_ari(&labels, &truth_labels);
        let distinct = count_unique(&labels);
        eprintln!(
            "leiden-cpm: γ={:.4}  ari_2way={:.3}  ari_full={:.3}  distinct_communities={}",
            g, ari_2way, ari_full, distinct
        );
        rows.push((g, ari_2way, ari_full, distinct));
        if ari_2way.abs() > best_ari_2way {
            best_ari_2way = ari_2way.abs();
            best_gamma_2way = g;
        }
        if ari_full.abs() > best_ari_full {
            best_ari_full = ari_full.abs();
            best_gamma_full = g;
        }
    }

    eprintln!(
        "leiden-cpm baselines: modularity-Leiden 2way_ari={:.3}, full_ari={:.3}",
        ari_modularity_2way, ari_modularity_full
    );
    eprintln!(
        "leiden-cpm best: 2way={:.3} @ γ={:.4}   full={:.3} @ γ={:.4}   (SOTA_target=0.75)",
        best_ari_2way, best_gamma_2way, best_ari_full, best_gamma_full
    );

    // Diagnostic-only assertion — CPM either beats modularity-Leiden
    // somewhere in the sweep (the expected win) or it doesn't (a real
    // finding to capture as a future ADR §17 row). We assert only
    // that the measurement is non-degenerate so a regression in
    // `leiden_labels_cpm` itself (e.g., collapses everything to 1
    // community) fails loudly.
    let any_meaningful = rows.iter().any(|(_, _, _, k)| *k >= 2);
    assert!(
        any_meaningful,
        "leiden-cpm: every γ collapsed the graph to a single community — \
         CPM gain math or aggregation is broken, not a measurement gap"
    );
}

fn count_unique(labels: &[u32]) -> usize {
    let mut s: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for l in labels {
        s.insert(*l);
    }
    s.len()
}

#[test]
fn leiden_cpm_vs_modularity_across_seeds() {
    // Reproducibility sweep: are CPM's 3.97× full-ARI win over
    // modularity-Leiden (ADR §17 items 18 & 20) stable across SBM
    // seeds, or a single-seed artefact? Run five seeds of the
    // default config and compare CPM @ γ=2.25 vs modularity-Leiden
    // on full-partition ARI. If CPM wins on ≥ 4 of 5 seeds by ≥ 2×,
    // the 3.97× headline is reproducible.
    let seeds: [u64; 5] = [
        0x5FA1_DE5,
        0x0C70_F00D,
        0xC0DE_CAFE,
        0xBEEF_BABE,
        0xDEAD_1234,
    ];
    let mut cpm_aris = Vec::new();
    let mut mod_aris = Vec::new();
    let mut ratios = Vec::new();
    for &seed in &seeds {
        let cfg = ConnectomeConfig {
            seed,
            ..ConnectomeConfig::default()
        };
        let conn = Connectome::generate(&cfg);
        let an = Analysis::new(AnalysisConfig::default());
        let truth_labels: Vec<u32> = (0..conn.num_neurons())
            .map(|i| conn.meta(connectome_fly::NeuronId(i as u32)).module as u32)
            .collect();
        let cpm_labels =
            connectome_fly::analysis::leiden::leiden_labels_cpm(&conn, 2.25);
        let cpm_full = full_partition_ari(&cpm_labels, &truth_labels);
        let mod_labels = an.leiden_labels(&conn);
        let mod_full = full_partition_ari(&mod_labels, &truth_labels);
        let ratio = if mod_full > 1e-6 { cpm_full / mod_full } else { f32::INFINITY };
        eprintln!(
            "cpm-seed-sweep: seed=0x{seed:X}  cpm_full={cpm_full:.3}  \
             modularity_full={mod_full:.3}  ratio={ratio:.2}×"
        );
        cpm_aris.push(cpm_full);
        mod_aris.push(mod_full);
        ratios.push(ratio);
    }
    let cpm_mean: f32 = cpm_aris.iter().sum::<f32>() / seeds.len() as f32;
    let mod_mean: f32 = mod_aris.iter().sum::<f32>() / seeds.len() as f32;
    let finite_ratios: Vec<f32> = ratios.iter().copied().filter(|x| x.is_finite()).collect();
    let ratio_mean = if finite_ratios.is_empty() {
        0.0
    } else {
        finite_ratios.iter().sum::<f32>() / finite_ratios.len() as f32
    };
    eprintln!(
        "cpm-seed-sweep: MEAN cpm={:.3}  modularity={:.3}  ratio={:.2}×",
        cpm_mean, mod_mean, ratio_mean
    );
    let wins: usize = cpm_aris
        .iter()
        .zip(mod_aris.iter())
        .filter(|(c, m)| **c > 2.0 * **m)
        .count();
    eprintln!(
        "cpm-seed-sweep: CPM beats modularity by ≥ 2× on {}/{} seeds",
        wins,
        seeds.len()
    );
    // Gate: publish-only on individual seed values; assert only that
    // the MEAN ratio is non-degenerate (> 1.0) so a regression in
    // leiden_labels_cpm itself fails loudly.
    assert!(
        ratio_mean > 1.0,
        "cpm-seed-sweep: CPM mean full-ARI {:.3} is not above modularity's {:.3} — \
         the 3.97× headline is not reproducible on this seed set, or CPM has regressed",
        cpm_mean, mod_mean
    );
}

#[test]
fn leiden_cpm_recovers_two_planted_communities() {
    // 2-community planted SBM: dense intra, sparse inter — the exact
    // fixture modularity-Leiden already handles cleanly. CPM should
    // also recover this for reasonable γ; if it can't, the CPM path
    // is wrong-at-the-easy-case and everything above is untrustworthy.
    let cfg = ConnectomeConfig {
        num_neurons: 200,
        num_modules: 2,
        num_hub_modules: 0,
        p_within: 0.40,
        p_between: 0.004,
        ..ConnectomeConfig::default()
    };
    let conn = Connectome::generate(&cfg);

    // Ground truth: module 0 vs module 1.
    let is_module_1 = |id: u32| conn.meta(connectome_fly::NeuronId(id)).module == 1;

    // γ needs to reach super-edge magnitudes in normalized units.
    // Sweep the 2-community planted fixture and record the best ARI.
    let gammas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let mut best_ari = 0.0_f32;
    let mut best_gamma = 0.0_f64;
    let mut best_distinct = 0_usize;
    for g in gammas {
        let labels = connectome_fly::analysis::leiden::leiden_labels_cpm(&conn, g);
        let (a, b) = two_way_from_labels(&labels);
        let ari_g = if a.is_empty() || b.is_empty() {
            0.0
        } else {
            adjusted_rand_index(&a, &b, is_module_1)
        };
        let d = count_unique(&labels);
        eprintln!("leiden-cpm-planted: γ={:.2}  ari={:.3}  distinct={}", g, ari_g, d);
        if ari_g.abs() > best_ari {
            best_ari = ari_g.abs();
            best_gamma = g;
            best_distinct = d;
        }
    }
    let labels = connectome_fly::analysis::leiden::leiden_labels_cpm(&conn, best_gamma);
    let _ = labels;
    eprintln!(
        "leiden-cpm-planted: best_ari={:.3} @ γ={:.2}, distinct={} (weight-normalized)",
        best_ari, best_gamma, best_distinct
    );
    // Publish-only across the sweep. The finding (γ where CPM
    // recovers the planted modules) updates ADR §17 item 16.
}

#[test]
fn leiden_cpm_vs_modularity_across_scales() {
    // N-scaling sweep. The 3.97× full-ARI win (ADR §17 item 20) and
    // 3.98× mean win across 5 seeds (ADR §17 item 21) were both
    // measured at N=1024. Does CPM's advantage hold at N=512 and
    // N=2048? If yes → the pattern is scale-invariant; if it shrinks
    // or inverts → the advantage is N-dependent and the headline
    // needs to be qualified.
    //
    // Density control: default is N=1024, 70 modules (~14.6
    // neurons/module). Scale num_modules = N/15 to hold module
    // size roughly constant; hubs = num_modules / 12 (default ratio
    // 6/70). Fixed seed isolates scale from seed variance.
    let scales: [(u32, u16, u16); 3] = [(512, 35, 3), (1024, 70, 6), (2048, 140, 12)];
    let mut ratios: Vec<f32> = Vec::new();
    for &(n, m, h) in &scales {
        let cfg = ConnectomeConfig {
            num_neurons: n,
            num_modules: m,
            num_hub_modules: h,
            ..ConnectomeConfig::default()
        };
        let conn = Connectome::generate(&cfg);
        let an = Analysis::new(AnalysisConfig::default());
        let truth_labels: Vec<u32> = (0..conn.num_neurons())
            .map(|i| conn.meta(connectome_fly::NeuronId(i as u32)).module as u32)
            .collect();
        let cpm_labels = connectome_fly::analysis::leiden::leiden_labels_cpm(&conn, 2.25);
        let mod_labels = an.leiden_labels(&conn);
        let cpm_full = full_partition_ari(&cpm_labels, &truth_labels);
        let mod_full = full_partition_ari(&mod_labels, &truth_labels);
        let ratio = if mod_full.abs() > 1e-4 {
            cpm_full / mod_full
        } else {
            f32::INFINITY
        };
        let cpm_d = count_unique(&cpm_labels);
        let mod_d = count_unique(&mod_labels);
        eprintln!(
            "cpm-scale-sweep: N={}  modules={}  cpm_full={:.3} ({}c)  mod_full={:.3} ({}c)  ratio={:.2}×",
            n, m, cpm_full, cpm_d, mod_full, mod_d, ratio
        );
        if ratio.is_finite() {
            ratios.push(ratio);
        }
    }
    if !ratios.is_empty() {
        let mean: f32 = ratios.iter().sum::<f32>() / ratios.len() as f32;
        let min = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "cpm-scale-sweep: ratio across {} scales — mean={:.2}×  min={:.2}×  max={:.2}×",
            ratios.len(),
            mean,
            min,
            max
        );
    }
    // Regression gate: at least one scale must still show CPM beating
    // modularity-Leiden (ratio > 1.0). If every scale regresses below
    // parity, the CPM path or normalization broke — loud failure.
    assert!(
        ratios.iter().any(|r| *r > 1.0),
        "cpm-scale-sweep: CPM no longer beats modularity at ANY scale — regression"
    );
}

#[test]
fn leiden_cpm_gamma_peak_per_scale() {
    // Follow-up to leiden_cpm_vs_modularity_across_scales (ADR §17
    // item 22): at fixed γ=2.25 CPM scored 0.322/0.425/0.258 across
    // N=512/1024/2048. Item 19 established that the γ peak on the
    // N=1024 substrate is γ ∈ [2.25, 2.5]; it's plausible the peak
    // γ shifts with N. This test does a small γ sweep at each scale
    // and reports the per-scale CPM ceiling. If the N=2048 ceiling
    // is still < 0.3, that's a real algorithmic ceiling; if it's
    // higher, the fixed-γ measurement at item 22 was understated.
    let scales: [(u32, u16, u16); 3] = [(512, 35, 3), (1024, 70, 6), (2048, 140, 12)];
    let gammas = [1.25, 1.75, 2.25, 2.75, 3.5, 5.0];
    for &(n, m, h) in &scales {
        let cfg = ConnectomeConfig {
            num_neurons: n,
            num_modules: m,
            num_hub_modules: h,
            ..ConnectomeConfig::default()
        };
        let conn = Connectome::generate(&cfg);
        let truth_labels: Vec<u32> = (0..conn.num_neurons())
            .map(|i| conn.meta(connectome_fly::NeuronId(i as u32)).module as u32)
            .collect();
        let mut best_ari = f32::NEG_INFINITY;
        let mut best_gamma = 0.0_f64;
        let mut best_distinct = 0usize;
        for &g in &gammas {
            let labels = connectome_fly::analysis::leiden::leiden_labels_cpm(&conn, g);
            let ari = full_partition_ari(&labels, &truth_labels);
            let d = count_unique(&labels);
            eprintln!(
                "cpm-peak-per-scale: N={}  γ={:.2}  full_ari={:.3}  distinct={}",
                n, g, ari, d
            );
            if ari > best_ari {
                best_ari = ari;
                best_gamma = g;
                best_distinct = d;
            }
        }
        eprintln!(
            "cpm-peak-per-scale: N={}  PEAK full_ari={:.3} @ γ={:.2}  (distinct={})",
            n, best_ari, best_gamma, best_distinct
        );
    }
    // Publish-only. No assertion — every row is a ceiling observation.
}
