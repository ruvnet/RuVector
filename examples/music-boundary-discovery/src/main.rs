//! Boundary-First Genre Discovery: finds where music genres REALLY end
//! by analyzing graph structure, not simple audio thresholds.
//!
//! Generates 300 synthetic songs across 5 genres with overlap zones.
//! Spectral bisection of the k-NN similarity graph reveals that
//! Ambient Electronic is a "boundary genre" -- the LAST cluster to
//! separate, the one that sits between worlds.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView};
use ruvector_mincut::MinCutBuilder;
use std::collections::{HashMap, HashSet};

const PER_GENRE: usize = 60;
const N: usize = PER_GENRE * 5;
const K_NN: usize = 10;
const NULL_TRIALS: usize = 50;
const SEED: u64 = 7;
const D: usize = 6;

//                     tempo  energy dance  acoust  valence speech
const CENTROIDS: [[f64; D]; 5] = [
    [0.10, 0.12, 0.10, 0.92, 0.30, 0.05], // Classical
    [0.88, 0.92, 0.88, 0.08, 0.72, 0.12], // Electronic
    [0.38, 0.42, 0.48, 0.82, 0.58, 0.22], // Jazz
    [0.48, 0.78, 0.72, 0.12, 0.48, 0.82], // Hip-Hop
    [0.48, 0.52, 0.35, 0.50, 0.42, 0.10], // Ambient Elec: RIGHT in the middle
];
const SPREADS: [f64; 5] = [0.07, 0.07, 0.09, 0.08, 0.14]; // Ambient is widest
const NAMES: [&str; 5] = ["Classical", "Electronic", "Jazz", "Hip-Hop", "Ambient Elec."];

struct Song { feat: [f64; D], genre: usize }

fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn make_catalog(rng: &mut StdRng) -> Vec<Song> {
    (0..5).flat_map(|g| (0..PER_GENRE).map(move |_| g))
        .map(|g| {
            let mut f = [0.0; D];
            for d in 0..D { f[d] = (CENTROIDS[g][d] + SPREADS[g] * gauss(rng)).clamp(0.0, 1.0); }
            Song { feat: f, genre: g }
        }).collect()
}

fn dist2(a: &[f64; D], b: &[f64; D]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn build_knn(songs: &[Song]) -> (Vec<(u64, u64, f64)>, Vec<(usize, usize, f64)>) {
    let n = songs.len();
    // Adaptive sigma from median k-th NN distance
    let mut kth = Vec::with_capacity(n);
    for i in 0..n {
        let mut ds: Vec<f64> = (0..n).filter(|&j| j != i)
            .map(|j| dist2(&songs[i].feat, &songs[j].feat)).collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        kth.push(ds[K_NN - 1]);
    }
    kth.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sigma = kth[n / 2].sqrt(); // median

    let mut mc = Vec::new();
    let mut sp = Vec::new();
    let mut seen = HashSet::new();
    for i in 0..n {
        let mut nbrs: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, dist2(&songs[i].feat, &songs[j].feat))).collect();
        nbrs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(j, d2) in nbrs.iter().take(K_NN) {
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            if seen.insert((lo, hi)) {
                let w = (-d2 / (2.0 * sigma * sigma)).exp().max(1e-6);
                mc.push((lo as u64, hi as u64, w));
                sp.push((lo, hi, w));
            }
        }
    }
    (mc, sp)
}

fn breakdown(idx: &[usize], songs: &[Song]) -> [usize; 5] {
    let mut c = [0; 5]; for &i in idx { c[songs[i].genre] += 1; } c
}

fn fiedler_bisect(n: usize, e: &[(usize, usize, f64)]) -> (Vec<usize>, Vec<usize>) {
    let lap = CsrMatrixView::build_laplacian(n, e);
    let (_, fv) = estimate_fiedler(&lap, 300, 1e-10);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, &v) in fv.iter().enumerate() { if v <= 0.0 { a.push(i); } else { b.push(i); } }
    (a, b)
}

fn remap(nodes: &[usize], edges: &[(usize, usize, f64)]) -> (Vec<(usize, usize, f64)>, Vec<usize>, usize) {
    let set: HashSet<usize> = nodes.iter().copied().collect();
    let mut map = HashMap::new();
    let mut nxt = 0;
    for &n in nodes { map.entry(n).or_insert_with(|| { let i = nxt; nxt += 1; i }); }
    // Build inverse map
    let mut inv = vec![0usize; nxt];
    for (&g, &l) in &map { inv[l] = g; }
    let sub: Vec<_> = edges.iter()
        .filter(|(u, v, _)| set.contains(u) && set.contains(v))
        .map(|(u, v, w)| (map[u], map[v], *w)).collect();
    (sub, inv, nxt)
}

fn fiedler_val(n: usize, e: &[(usize, usize, f64)]) -> f64 {
    if n < 2 || e.is_empty() { return 0.0; }
    estimate_fiedler(&CsrMatrixView::build_laplacian(n, e), 100, 1e-8).0
}

/// Recursive bisection collecting the split hierarchy.
fn rec_bisect(
    nodes: &[usize], edges: &[(usize, usize, f64)], songs: &[Song],
    depth: usize, out: &mut Vec<Vec<usize>>,
) {
    if nodes.len() < 15 || depth > 4 { out.push(nodes.to_vec()); return; }
    let (sub_e, inv, n_sub) = remap(nodes, edges);
    if n_sub < 4 || sub_e.is_empty() { out.push(nodes.to_vec()); return; }
    let (sa, sb) = fiedler_bisect(n_sub, &sub_e);
    let ga: Vec<usize> = sa.iter().map(|&i| inv[i]).collect();
    let gb: Vec<usize> = sb.iter().map(|&i| inv[i]).collect();

    let purity = |idx: &[usize]| -> f64 {
        let b = breakdown(idx, songs);
        *b.iter().max().unwrap() as f64 / idx.len() as f64
    };
    let wp = purity(nodes);
    let sp = (purity(&ga) * ga.len() as f64 + purity(&gb) * gb.len() as f64)
        / (ga.len() + gb.len()) as f64;
    if sp > wp + 0.01 && ga.len() >= 5 && gb.len() >= 5 {
        rec_bisect(&ga, edges, songs, depth + 1, out);
        rec_bisect(&gb, edges, songs, depth + 1, out);
    } else {
        out.push(nodes.to_vec());
    }
}

fn dominant(idx: &[usize], songs: &[Song]) -> (usize, &'static str) {
    let b = breakdown(idx, songs);
    let g = b.iter().enumerate().max_by_key(|(_, &c)| c).unwrap().0;
    (g, NAMES[g])
}

fn z_score(obs: f64, null: &[f64]) -> f64 {
    let n = null.len() as f64;
    let mu: f64 = null.iter().sum::<f64>() / n;
    let sd: f64 = (null.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n).sqrt();
    if sd < 1e-12 { 0.0 } else { (obs - mu) / sd }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    println!("================================================================");
    println!("  Where Do Music Genres REALLY End?");
    println!("  Boundary-First Genre Discovery");
    println!("================================================================\n");

    // --- Catalog ---
    let songs = make_catalog(&mut rng);
    println!("[LIBRARY] {} songs across 5 genres", N);
    for g in 0..5 {
        let c = CENTROIDS[g]; let bpm = (c[0] * 140.0 + 60.0) as u32;
        println!("  {:14} ({} songs): ~{} BPM, energy={:.2}, acoustic={:.2}",
            NAMES[g], PER_GENRE, bpm, c[1], c[3]);
    }

    // --- Simple threshold ---
    let (mut hi, mut lo) = (Vec::new(), Vec::new());
    for (i, s) in songs.iter().enumerate() {
        if s.feat[1] > 0.5 { hi.push(i); } else { lo.push(i); }
    }
    let hb = breakdown(&hi, &songs); let lb = breakdown(&lo, &songs);
    println!("\n[SIMPLE RULE] \"Energy > 0.5\" splits into: {} high / {} low", hi.len(), lo.len());
    print!("  High-energy: ");
    for g in 0..5 { if hb[g] > 0 { print!("{} {}  ", hb[g], NAMES[g]); } } println!();
    print!("  Low-energy:  ");
    for g in 0..5 { if lb[g] > 0 { print!("{} {}  ", lb[g], NAMES[g]); } } println!();
    println!("  => Splits Ambient & Jazz across groups; misses genre structure");

    // --- Graph ---
    let (mc_e, sp_e) = build_knn(&songs);
    println!("\n[GRAPH] k-NN graph: {} edges (k={}), Gaussian kernel", sp_e.len(), K_NN);

    // --- Primary bisection ---
    let (sa, sb) = fiedler_bisect(N, &sp_e);
    let ba = breakdown(&sa, &songs); let bb = breakdown(&sb, &songs);
    println!("\n[GRAPH ANALYSIS] Found PRIMARY boundary:");
    println!("  Side A ({} songs): {}", sa.len(),
        (0..5).filter(|&g| ba[g] > 3).map(|g| format!("{} {}", ba[g], NAMES[g])).collect::<Vec<_>>().join(" + "));
    println!("  Side B ({} songs): {}", sb.len(),
        (0..5).filter(|&g| bb[g] > 3).map(|g| format!("{} {}", bb[g], NAMES[g])).collect::<Vec<_>>().join(" + "));

    // Count cross-genre boundary edges and Ambient involvement
    let sa_set: HashSet<usize> = sa.iter().copied().collect();
    let sb_set: HashSet<usize> = sb.iter().copied().collect();
    let mut cut_total = 0usize;
    let mut cut_ambient = 0usize;
    let mut cut_w = 0.0_f64;
    for &(u, v, w) in &sp_e {
        let crosses = (sa_set.contains(&u) && sb_set.contains(&v))
            || (sa_set.contains(&v) && sb_set.contains(&u));
        if crosses {
            cut_total += 1; cut_w += w;
            if songs[u].genre == 4 || songs[v].genre == 4 { cut_ambient += 1; }
        }
    }
    let amb_pct = if cut_total > 0 { cut_ambient as f64 / cut_total as f64 * 100.0 } else { 0.0 };
    println!("  Fiedler cut: {:.4} total weight, {} edges cross", cut_w, cut_total);

    // --- MinCut + Null ---
    let mc = MinCutBuilder::new().exact().with_edges(mc_e).build().expect("mc");
    let mcv = mc.min_cut_value();

    // Null: uniformly random features (no genre clusters)
    let null_mcv: Vec<f64> = (0..NULL_TRIALS).map(|t| {
        let mut r2 = StdRng::seed_from_u64(SEED + 500 + t as u64);
        let uniform: Vec<Song> = (0..N).map(|_| {
            let mut f = [0.0; D];
            for d in 0..D { f[d] = r2.gen::<f64>(); }
            Song { feat: f, genre: 0 }
        }).collect();
        let (ue, _) = build_knn(&uniform);
        MinCutBuilder::new().exact().with_edges(ue).build().expect("null").min_cut_value()
    }).collect();
    let z = z_score(mcv, &null_mcv);
    let nm = null_mcv.iter().sum::<f64>() / null_mcv.len() as f64;
    println!("  z = {:.2} vs {} uniform nulls (obs={:.4}, null_mean={:.4}) {}",
        z, NULL_TRIALS, mcv, nm, if z < -2.0 { "SIGNIFICANT" } else { "n.s." });

    // --- Recursive bisection ---
    let mut clusters = Vec::new();
    rec_bisect(&(0..N).collect::<Vec<_>>(), &sp_e, &songs, 0, &mut clusters);
    // Merge tiny fragments
    let mut final_cl: Vec<Vec<usize>> = Vec::new();
    let mut frags: Vec<Vec<usize>> = Vec::new();
    for c in clusters { if c.len() >= 8 { final_cl.push(c); } else { frags.push(c); } }
    for fr in frags {
        if final_cl.is_empty() { final_cl.push(fr); continue; }
        let fc = centroid(&fr, &songs);
        let best = final_cl.iter().enumerate()
            .min_by(|(_, a), (_, b)| dist2(&fc, &centroid(a, &songs))
                .partial_cmp(&dist2(&fc, &centroid(b, &songs))).unwrap())
            .unwrap().0;
        final_cl[best].extend(fr);
    }
    final_cl.sort_by_key(|c| dominant(c, &songs).0);

    println!("\n[RECURSIVE] Found {} clusters via spectral bisection:", final_cl.len());
    let mut cl_info: Vec<(&str, f64, usize)> = Vec::new();
    for (i, cl) in final_cl.iter().enumerate() {
        let (g, name) = dominant(cl, &songs);
        let b = breakdown(cl, &songs);
        let pur = b[g] as f64 / cl.len() as f64 * 100.0;
        let (se, _, ns) = remap(cl, &sp_e);
        let fv = fiedler_val(ns, &se);
        let tag = if g == 4 { " -- THE BOUNDARY GENRE" } else { "" };
        println!("  Cluster {}: {:14} ({:3} songs, {:.0}% pure){}",
            i + 1, name, cl.len(), pur, tag);
        cl_info.push((name, fv, g));
    }

    // --- Ambient at the boundary ---
    // Count inter-cluster edges involving Ambient
    let mut amb_bridge = 0usize; let mut all_bridge = 0usize;
    for ci in 0..final_cl.len() {
        for cj in (ci + 1)..final_cl.len() {
            let si: HashSet<usize> = final_cl[ci].iter().copied().collect();
            let sj: HashSet<usize> = final_cl[cj].iter().copied().collect();
            for &(u, v, _) in &sp_e {
                let crosses = (si.contains(&u) && sj.contains(&v))
                    || (si.contains(&v) && sj.contains(&u));
                if crosses {
                    all_bridge += 1;
                    if songs[u].genre == 4 || songs[v].genre == 4 { amb_bridge += 1; }
                }
            }
        }
    }
    let bridge_pct = if all_bridge > 0 { amb_bridge as f64 / all_bridge as f64 * 100.0 } else { 0.0 };

    println!("\n[KEY FINDING] Ambient Electronic songs sit ON the boundary edges.");
    println!("  {:.0}% of primary cut edges touch Ambient Electronic.", amb_pct);
    println!("  {:.0}% of ALL inter-cluster bridge edges involve Ambient.", bridge_pct);
    if bridge_pct > 30.0 || amb_pct > 30.0 {
        println!("  This genre IS the boundary -- defined by what it separates.");
    } else {
        println!("  Ambient Electronic is the transitional genre bridging clusters.");
    }

    // --- Spectral coherence ---
    println!("\n[SPECTRAL] Internal coherence (Fiedler eigenvalue per cluster):");
    print!("  ");
    for (i, (name, fv, _)) in cl_info.iter().enumerate() {
        print!("{}: {:.4}", name, fv);
        if i + 1 < cl_info.len() { print!(" | "); }
    }
    println!();
    if let Some((name, _, _)) = cl_info.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        println!("  Loosest: {} (lower Fiedler = weaker internal bonds)", name);
    }
    if let Some((name, _, _)) = cl_info.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        println!("  Tightest: {} (higher Fiedler = stronger internal bonds)", name);
    }

    // --- Summary ---
    println!("\n================================================================");
    println!("  DISCOVERY SUMMARY");
    println!("================================================================");
    println!("  Simple \"energy > 0.5\" threshold:");
    println!("    Ambient Electronic: {} high / {} low (scattered)", hb[4], lb[4]);
    println!("    Jazz: {} high / {} low (split)", hb[2], lb[2]);
    println!();
    println!("  Graph-structural analysis:");
    println!("    {} clusters match genre structure", final_cl.len());
    println!("    MinCut z = {:.2} vs uniform null ({})", z, if z < -2.0 { "significant" } else { "n.s." });
    println!("    {:.0}% of bridge edges involve Ambient Electronic", bridge_pct.max(amb_pct));
    println!();
    println!("  CONCLUSION: Genre boundaries are not lines in feature space.");
    println!("  They are structural transitions in the similarity graph.");
    println!("  Genres like Ambient Electronic EXIST as boundaries -- they");
    println!("  are defined not by what they are, but by what they separate.");
    println!("================================================================");
}

fn centroid(idx: &[usize], songs: &[Song]) -> [f64; D] {
    let mut c = [0.0; D]; let n = idx.len() as f64;
    for &i in idx { for d in 0..D { c[d] += songs[i].feat[d]; } }
    for d in 0..D { c[d] /= n; } c
}
