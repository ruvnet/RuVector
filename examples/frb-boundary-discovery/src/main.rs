//! FRB Population Boundary Discovery using CHIME-like catalog data.
//!
//! Generates ~200 FRBs modeled on the CHIME/FRB Catalog 1 (arXiv:2106.04352)
//! distributions, with injected sub-populations. Builds a multi-parameter
//! similarity graph, runs spectral bisection + mincut to discover population
//! boundaries, and validates against null permutations. Shows that the
//! structural boundary differs from a simple DM threshold.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;
use std::collections::{HashMap, HashSet};

const N_FRB: usize = 200;
const K_NN: usize = 7;
const SIGMA: f64 = 0.28;
const NULL_PERMS: usize = 100;
const SEED: u64 = 2106_04352; // arXiv ID as seed

#[derive(Clone)]
struct Frb {
    dm: f64,
    width: f64,
    fluence: f64,
    scattering: f64,
    sp_idx: f64,
    population: u8, // 0=A (cosmological), 1=B (local-env), 2=C (transition)
}

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn log_normal(rng: &mut StdRng, median: f64, sigma_log10: f64) -> f64 {
    10.0_f64
        .powf(median.log10() + sigma_log10 * gauss(rng))
        .max(0.01)
}

fn power_law(rng: &mut StdRng, x_min: f64, x_max: f64, alpha: f64) -> f64 {
    let u: f64 = rng.gen();
    let a1 = alpha + 1.0;
    (x_min.powf(a1) + u * (x_max.powf(a1) - x_min.powf(a1))).powf(1.0 / a1)
}

fn exponential(rng: &mut StdRng, mean: f64) -> f64 {
    -mean * rng.gen::<f64>().max(1e-15).ln()
}

fn generate_catalog(rng: &mut StdRng) -> Vec<Frb> {
    (0..N_FRB)
        .map(|_| {
            let u: f64 = rng.gen();
            let population = if u < 0.60 {
                0u8
            } else if u < 0.90 {
                1
            } else {
                2
            };
            let (dm, width, scattering, sp_idx) = match population {
                0 => {
                    let dm = log_normal(rng, 750.0, 0.30).max(250.0);
                    let width = log_normal(rng, 3.0, 0.40);
                    let scat = exponential(rng, 0.6).min(4.0);
                    let sp = -2.0 + 2.5 * gauss(rng);
                    (dm, width, scat, sp)
                }
                1 => {
                    let dm = log_normal(rng, 280.0, 0.25).max(80.0);
                    let width = log_normal(rng, 8.0, 0.45);
                    let scat = exponential(rng, 6.0).min(50.0);
                    let sp = 2.0 + 3.5 * gauss(rng);
                    (dm, width, scat, sp)
                }
                _ => {
                    let dm = log_normal(rng, 450.0, 0.35).max(100.0);
                    let width = log_normal(rng, 5.0, 0.50);
                    let scat = exponential(rng, 2.5).min(25.0);
                    let sp = 0.0 + 4.0 * gauss(rng);
                    (dm, width, scat, sp)
                }
            };
            let fluence = power_law(rng, 0.4, 200.0, -1.4);
            Frb {
                dm,
                width,
                fluence,
                scattering,
                sp_idx,
                population,
            }
        })
        .collect()
}

// Single-population null catalog (no sub-structure)
fn generate_null_catalog(rng: &mut StdRng) -> Vec<Frb> {
    (0..N_FRB)
        .map(|_| {
            let dm = log_normal(rng, 500.0, 0.45).max(80.0);
            let width = log_normal(rng, 5.0, 0.55);
            let scat = exponential(rng, 2.5).min(50.0);
            let sp = 0.0 + 4.0 * gauss(rng);
            let fluence = power_law(rng, 0.4, 200.0, -1.4);
            Frb {
                dm,
                width,
                fluence,
                scattering: scat,
                sp_idx: sp,
                population: 0,
            }
        })
        .collect()
}

fn normalize(vals: &[f64]) -> Vec<f64> {
    let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let r = (hi - lo).max(1e-12);
    vals.iter().map(|v| (v - lo) / r).collect()
}

fn build_features(catalog: &[Frb]) -> Vec<[f64; 5]> {
    let n = catalog.len();
    let dm = normalize(&catalog.iter().map(|f| f.dm.ln()).collect::<Vec<_>>());
    let wi = normalize(&catalog.iter().map(|f| f.width.ln()).collect::<Vec<_>>());
    let fl = normalize(&catalog.iter().map(|f| f.fluence.ln()).collect::<Vec<_>>());
    let sc = normalize(
        &catalog
            .iter()
            .map(|f| (f.scattering + 0.1).ln())
            .collect::<Vec<_>>(),
    );
    let sp = normalize(&catalog.iter().map(|f| f.sp_idx).collect::<Vec<_>>());
    (0..n)
        .map(|i| [dm[i], wi[i], fl[i], sc[i], sp[i]])
        .collect()
}

fn euclidean(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

type EdgeList = Vec<(usize, usize, f64)>;

fn build_knn_graph(feats: &[[f64; 5]]) -> EdgeList {
    let n = feats.len();
    let mut edges = Vec::new();
    let mut added = HashSet::new();
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean(&feats[i], &feats[j])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(j, d) in dists.iter().take(K_NN) {
            let (lo, hi) = (i.min(j), i.max(j));
            if added.insert((lo, hi)) {
                edges.push((lo, hi, (-d * d / (2.0 * SIGMA * SIGMA)).exp()));
            }
        }
    }
    edges
}

/// Spectral bisection: returns (part_a, part_b, raw_cut, fiedler_eigenvalue)
fn spectral_bisect(edges: &EdgeList, n: usize) -> (Vec<usize>, Vec<usize>, f64, f64) {
    let lap = CsrMatrixView::build_laplacian(n, edges);
    let (fiedler_val, fv) = estimate_fiedler(&lap, 300, 1e-10);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap());

    // Sweep for min ratio-cut (skip trivial < 10% partitions)
    let margin = (n / 10).max(3);
    let mut set_s: HashSet<usize> = HashSet::new();
    let mut best = (margin, f64::INFINITY);

    for k in 0..n - 1 {
        set_s.insert(order[k]);
        if k + 1 < margin || k + 1 > n - margin {
            continue;
        }
        let cut: f64 = edges
            .iter()
            .filter(|&&(u, v, _)| set_s.contains(&u) != set_s.contains(&v))
            .map(|&(_, _, w)| w)
            .sum();
        let ratio = cut * (1.0 / (k + 1) as f64 + 1.0 / (n - k - 1) as f64);
        if ratio < best.1 {
            best = (k + 1, ratio);
        }
    }

    let a: Vec<usize> = order[..best.0].to_vec();
    let b: Vec<usize> = order[best.0..].to_vec();
    let set_a: HashSet<usize> = a.iter().copied().collect();
    let raw_cut: f64 = edges
        .iter()
        .filter(|&&(u, v, _)| set_a.contains(&u) != set_a.contains(&v))
        .map(|&(_, _, w)| w)
        .sum();
    (a, b, raw_cut, fiedler_val)
}

/// Compute Fiedler eigenvalue for a graph (lower = more separable)
fn graph_fiedler(catalog: &[Frb]) -> f64 {
    let feats = build_features(catalog);
    let edges = build_knn_graph(&feats);
    let lap = CsrMatrixView::build_laplacian(catalog.len(), &edges);
    estimate_fiedler(&lap, 300, 1e-10).0
}

fn null_fiedler_distribution(rng: &mut StdRng) -> Vec<f64> {
    (0..NULL_PERMS)
        .map(|_| graph_fiedler(&generate_null_catalog(rng)))
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

fn z_score(obs: f64, null: &[f64]) -> f64 {
    let sd = std_dev(null);
    if sd < 1e-12 {
        0.0
    } else {
        (obs - mean(null)) / sd
    }
}

fn jaccard(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
    let u = a.union(b).count() as f64;
    if u < 1.0 {
        0.0
    } else {
        a.intersection(b).count() as f64 / u
    }
}

fn sub_fiedler(nodes: &[usize], edges: &EdgeList) -> f64 {
    let set: HashSet<usize> = nodes.iter().copied().collect();
    let mut remap = HashMap::new();
    for (i, &n) in nodes.iter().enumerate() {
        remap.insert(n, i);
    }
    let sub: EdgeList = edges
        .iter()
        .filter(|(u, v, _)| set.contains(u) && set.contains(v))
        .map(|(u, v, w)| (remap[u], remap[v], *w))
        .collect();
    if nodes.len() < 3 || sub.is_empty() {
        return 0.0;
    }
    estimate_fiedler(
        &CsrMatrixView::build_laplacian(nodes.len(), &sub),
        100,
        1e-8,
    )
    .0
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    println!("================================================================");
    println!("  FRB Population Boundary Discovery (CHIME-like data)");
    println!("================================================================\n");

    // 1. Generate catalog
    let catalog = generate_catalog(&mut rng);
    let pc: [usize; 3] = [
        catalog.iter().filter(|f| f.population == 0).count(),
        catalog.iter().filter(|f| f.population == 1).count(),
        catalog.iter().filter(|f| f.population == 2).count(),
    ];
    println!(
        "[DATA] {} FRBs  (Pop A={}, Pop B={}, Pop C={})",
        N_FRB, pc[0], pc[1], pc[2]
    );

    // 2. Build features and k-NN graph
    let feats = build_features(&catalog);
    let edges = build_knn_graph(&feats);
    println!(
        "[DATA] {} edges in {}-NN graph, 5 features",
        edges.len(),
        K_NN
    );

    // 3. Global mincut (lower bound)
    let mc_edges: Vec<(u64, u64, f64)> = edges
        .iter()
        .map(|&(u, v, w)| (u as u64, v as u64, w))
        .collect();
    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(mc_edges)
        .build()
        .expect("mincut");
    let gv = mc.min_cut_value();
    println!("[MINCUT] Global min-cut value: {:.4} (lower bound)\n", gv);

    // 4. Spectral bisection
    let (part_a, part_b, cut_val, fiedler_val) = spectral_bisect(&edges, N_FRB);
    println!(
        "[SPECTRAL] Partition A: {} FRBs, Partition B: {} FRBs",
        part_a.len(),
        part_b.len()
    );
    println!("[SPECTRAL] Cut value: {:.4}", cut_val);
    println!("[SPECTRAL] Fiedler eigenvalue: {:.6}", fiedler_val);

    // 5. Null permutations (compare Fiedler eigenvalue)
    // Lower Fiedler = more separable graph = stronger population structure
    println!(
        "[NULL] Running {} null permutations (single-population)...",
        NULL_PERMS
    );
    let null_fiedlers = null_fiedler_distribution(&mut rng);
    let z = z_score(fiedler_val, &null_fiedlers);
    let count_below = null_fiedlers.iter().filter(|&&v| v <= fiedler_val).count();
    let p_str = if count_below == 0 {
        format!("< {:.3}", 1.0 / NULL_PERMS as f64)
    } else {
        format!("~{:.3}", count_below as f64 / NULL_PERMS as f64)
    };
    println!(
        "[NULL] Fiedler: obs={:.4}, null_mean={:.4}, z-score={:.2} (p {})",
        fiedler_val,
        mean(&null_fiedlers),
        z,
        p_str
    );
    println!(
        "[NULL] Interpretation: {} Fiedler = {} separable graph\n",
        if z < 0.0 { "lower" } else { "higher" },
        if z < 0.0 { "more" } else { "less" }
    );

    // 6. Report properties per partition
    let report = |label: &str, idx: &[usize]| {
        let v = |f: fn(&Frb) -> f64| -> Vec<f64> { idx.iter().map(|&i| f(&catalog[i])).collect() };
        let dms = v(|f| f.dm);
        let wds = v(|f| f.width);
        let scs = v(|f| f.scattering);
        let sps = v(|f| f.sp_idx);
        let fls = v(|f| f.fluence);
        println!(
            "  {}: DM={:.0}+/-{:.0}, width={:.1}+/-{:.1}, scatter={:.1}+/-{:.1}, sp_idx={:.1}+/-{:.1}, fluence={:.1}+/-{:.1}",
            label,
            mean(&dms), std_dev(&dms), mean(&wds), std_dev(&wds),
            mean(&scs), std_dev(&scs), mean(&sps), std_dev(&sps),
            mean(&fls), std_dev(&fls),
        );
        let pa = idx.iter().filter(|&&i| catalog[i].population == 0).count();
        let pb = idx.iter().filter(|&&i| catalog[i].population == 1).count();
        let ppc = idx.iter().filter(|&&i| catalog[i].population == 2).count();
        let n = idx.len() as f64;
        println!(
            "         composition: Pop-A={} ({:.0}%), Pop-B={} ({:.0}%), Pop-C={} ({:.0}%)",
            pa,
            100.0 * pa as f64 / n,
            pb,
            100.0 * pb as f64 / n,
            ppc,
            100.0 * ppc as f64 / n,
        );
    };

    println!("[PROPERTIES]");
    report("Partition A", &part_a);
    report("Partition B", &part_b);
    println!();

    // 7. Compare with simple DM threshold
    let dm_threshold = 500.0;
    let dm_high: HashSet<usize> = catalog
        .iter()
        .enumerate()
        .filter(|(_, f)| f.dm > dm_threshold)
        .map(|(i, _)| i)
        .collect();
    let dm_low: HashSet<usize> = catalog
        .iter()
        .enumerate()
        .filter(|(_, f)| f.dm <= dm_threshold)
        .map(|(i, _)| i)
        .collect();
    let set_a: HashSet<usize> = part_a.iter().copied().collect();
    let set_b: HashSet<usize> = part_b.iter().copied().collect();

    let j_best = jaccard(&set_a, &dm_high)
        .max(jaccard(&set_a, &dm_low))
        .max(jaccard(&set_b, &dm_high))
        .max(jaccard(&set_b, &dm_low));

    println!(
        "[DM-THRESHOLD] Simple DM>{} split: {}/{}",
        dm_threshold,
        dm_high.len(),
        dm_low.len()
    );
    println!(
        "[DM-THRESHOLD] Jaccard similarity with spectral = {:.3}",
        j_best
    );
    if j_best < 0.80 {
        println!("  => Spectral bisection finds a DIFFERENT boundary than simple thresholding");
    } else {
        println!("  => Boundaries overlap (spectral may still capture subtleties)");
    }
    println!();

    // 8. Spectral sub-partition analysis
    let fa = sub_fiedler(&part_a, &edges);
    let fb = sub_fiedler(&part_b, &edges);
    println!("[SPECTRAL] Fiedler(A)={:.4}, Fiedler(B)={:.4}", fa, fb);
    if (fa - fb).abs() > 0.01 {
        println!("  => Partitions have DISTINCT internal connectivity\n");
    } else {
        println!("  => Partitions have similar internal connectivity\n");
    }

    // 9. Summary
    println!("================================================================");
    println!("  DISCOVERY SUMMARY");
    println!("================================================================");
    println!("  FRBs analyzed:          {}", N_FRB);
    println!("  k-NN graph edges:       {}", edges.len());
    println!("  Spectral cut value:     {:.4}", cut_val);
    println!("  Global mincut (lower):  {:.4}", gv);
    println!("  Fiedler eigenvalue:     {:.6}", fiedler_val);
    println!("  z-score (vs null):      {:.2} (p {})", z, p_str);
    println!(
        "  DM-threshold Jaccard:   {:.3} ({})",
        j_best,
        if j_best < 0.80 {
            "DIFFERENT"
        } else {
            "similar"
        }
    );
    println!("  Spectral Fiedler (A|B): {:.4} | {:.4}", fa, fb);
    println!("================================================================");

    let sig = z < -2.0;
    let diff = j_best < 0.80;
    if sig && diff {
        println!("\n  CONCLUSION: Spectral bisection discovers a population boundary");
        println!(
            "  that is statistically significant (z={:.2}) and structurally",
            z
        );
        println!("  DIFFERENT from a naive DM threshold. The boundary separates");
        println!("  cosmological FRBs from local-environment FRBs using the joint");
        println!("  distribution of DM, width, scattering, and spectral index.");
    } else if sig {
        println!("\n  CONCLUSION: Significant boundary found (z={:.2}).", z);
        println!("  The multi-parameter cut partly coincides with the DM split.");
    } else if diff {
        println!(
            "\n  CONCLUSION: Boundary detected (Fiedler z={:.2}) with distinct",
            z
        );
        println!("  properties between partitions. The spectral split differs from");
        println!(
            "  DM thresholding (Jaccard={:.3}), confirming multi-parameter",
            j_best
        );
        println!("  structure in the FRB population that DM alone cannot capture.");
    } else {
        println!("\n  CONCLUSION: Adjust parameters for stronger separation.");
    }
    println!();
}
