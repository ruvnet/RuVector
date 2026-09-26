/// Benchmark: page-coherent agent memory retrieval.
///
/// Measures latency, throughput, recall, and intra-page coherence for three
/// page store implementations across a deterministic random dataset.
use ruvector_coherence_pages::{
    brute_force, centroid::CentroidPageStore, flat::FlatStore, gen_unit_vecs,
    greedy::GreedyCoherenceStore, percentile, recall, PageStore,
};
use std::time::Instant;

// ── dataset parameters ─────────────────────────────────────────────────────────
const N: usize = 8_000; // number of vectors
const DIM: usize = 128; // embedding dimension
const Q: usize = 500; // number of queries
const K: usize = 10; // retrieve top-K
const NUM_PAGES: usize = 80; // target page count
const PAGE_SIZE: usize = N / NUM_PAGES; // ~100 vectors per page

// Probe budgets: fraction of total pages to probe per query.
const PROBE_CENTROID: usize = 8; // 10% of 80 pages
const PROBE_GREEDY: usize = 8;

fn print_env() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    RuVector • Page-Coherent Memory Benchmark                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    println!("OS:          {os} ({arch})");
    println!("Dataset:     {N} vectors × {DIM} dimensions");
    println!("Queries:     {Q}");
    println!("Retrieve:    top-{K}");
    println!("Pages:       {NUM_PAGES} target (≈{PAGE_SIZE} vecs/page)");
    println!("Probe:       centroid={PROBE_CENTROID}, greedy={PROBE_GREEDY} of {NUM_PAGES}");
    println!();
}

/// Run Q queries and return per-query latencies in microseconds.
fn bench_store(
    store: &dyn PageStore,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    probe: usize,
) -> (Vec<u128>, f32) {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;
    for (q, t) in queries.iter().zip(truth.iter()) {
        let t0 = Instant::now();
        let res = store.search(q, K, probe);
        latencies.push(t0.elapsed().as_micros());
        total_recall += recall(&res.ids, t);
    }
    let avg_recall = total_recall / queries.len() as f32;
    (latencies, avg_recall)
}

fn report(
    name: &str,
    build_ms: u128,
    pages: usize,
    coherence: f32,
    probe: usize,
    latencies: &mut Vec<u128>,
    avg_recall: f32,
    n: usize,
) {
    latencies.sort_unstable();
    let mean = latencies.iter().sum::<u128>() as f64 / latencies.len() as f64;
    let p50 = percentile(latencies, 50.0);
    let p95 = percentile(latencies, 95.0);
    let total_s = latencies.iter().sum::<u128>() as f64 / 1_000_000.0;
    let throughput = latencies.len() as f64 / total_s;
    // Approximate memory: centroid (dim f32) + per-vector (dim f32 + 8 bytes id)
    let mem_bytes = pages * DIM * 4 + n * (DIM * 4 + 8);
    let mem_mb = mem_bytes as f64 / 1_048_576.0;

    println!("┌─ {name} ─────────────────────────────────────────────────────────");
    println!("│  Build:      {build_ms} ms");
    println!("│  Pages:      {pages}  (probed: {probe}/{pages})");
    println!("│  Coherence:  {coherence:.4}  (avg intra-page cosine similarity)");
    println!("│  Recall@{K}:  {avg_recall:.4}");
    println!("│  Mean:       {mean:.1} µs");
    println!("│  p50:        {p50} µs");
    println!("│  p95:        {p95} µs");
    println!("│  Throughput: {throughput:.0} queries/s");
    println!("│  Mem est:    {mem_mb:.2} MB");
    println!("└──────────────────────────────────────────────────────────────────");
    println!();
}

fn main() {
    print_env();

    // ── Generate deterministic dataset ────────────────────────────────────────
    println!("Generating {N} random unit vectors (dim={DIM}, seed=0)…");
    let t0 = Instant::now();
    let vecs = gen_unit_vecs(N, DIM, 0);
    let data: Vec<(usize, Vec<f32>)> = vecs.into_iter().enumerate().collect();
    let queries = gen_unit_vecs(Q, DIM, 1);
    println!("  done in {} ms\n", t0.elapsed().as_millis());

    // ── Ground truth ──────────────────────────────────────────────────────────
    println!("Computing brute-force ground truth…");
    let t0 = Instant::now();
    let truth: Vec<Vec<usize>> = queries.iter().map(|q| brute_force(&data, q, K)).collect();
    println!("  done in {} ms\n", t0.elapsed().as_millis());

    // ── Variant 1: FlatStore (baseline) ───────────────────────────────────────
    println!("Building FlatStore…");
    let mut flat = FlatStore::default();
    let s_flat = flat.build_from(data.clone());
    let (mut lat_flat, rec_flat) = bench_store(&flat, &queries, &truth, 1);
    report(
        "flat (baseline)",
        s_flat.build_ms,
        s_flat.page_count,
        s_flat.avg_coherence,
        1,
        &mut lat_flat,
        rec_flat,
        N,
    );

    // ── Variant 2: CentroidPageStore ──────────────────────────────────────────
    println!("Building CentroidPageStore ({NUM_PAGES} pages, 10 k-means iters)…");
    let mut centroid = CentroidPageStore::new(NUM_PAGES, 10);
    let s_cen = centroid.build_from(data.clone());
    let (mut lat_cen, rec_cen) = bench_store(&centroid, &queries, &truth, PROBE_CENTROID);
    report(
        "centroid-pages",
        s_cen.build_ms,
        s_cen.page_count,
        s_cen.avg_coherence,
        PROBE_CENTROID,
        &mut lat_cen,
        rec_cen,
        N,
    );

    // ── Variant 3: GreedyCoherenceStore ───────────────────────────────────────
    println!("Building GreedyCoherenceStore (page_size={PAGE_SIZE})…");
    let mut greedy = GreedyCoherenceStore::new(PAGE_SIZE, PROBE_GREEDY);
    let s_gre = greedy.build_from(data.clone());
    let (mut lat_gre, rec_gre) = bench_store(&greedy, &queries, &truth, PROBE_GREEDY);
    report(
        "greedy-coherence",
        s_gre.build_ms,
        s_gre.page_count,
        s_gre.avg_coherence,
        PROBE_GREEDY,
        &mut lat_gre,
        rec_gre,
        N,
    );

    // ── Acceptance criteria ───────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════════");
    println!("ACCEPTANCE CRITERIA");
    println!("═══════════════════════════════════════════════════════════════════");

    let speedup_cen = lat_flat.iter().sum::<u128>() as f64 / lat_cen.iter().sum::<u128>() as f64;
    let speedup_gre = lat_flat.iter().sum::<u128>() as f64 / lat_gre.iter().sum::<u128>() as f64;
    let coherence_gain_cen = s_cen.avg_coherence - s_flat.avg_coherence;
    let coherence_gain_gre = s_gre.avg_coherence - s_flat.avg_coherence;

    let cen_faster = speedup_cen > 1.0;
    let gre_faster = speedup_gre > 1.0;
    let cen_coherent = coherence_gain_cen > 0.0;
    let gre_coherent = coherence_gain_gre > 0.0;
    let flat_perfect = (rec_flat - 1.0).abs() < 1e-4;
    // Thresholds calibrated for 10% probe rate on random unit vectors (D=128).
    // Random probe baseline at 10%: ~12.5% expected recall.
    // Centroid clustering (k-means) achieves ~2.8× above baseline (≥0.30).
    // Greedy coherence achieves ~1.8× above baseline (≥0.18) — it maximizes
    // local coherence rather than global centroid quality, trading recall for
    // higher intra-page cosine similarity (see research doc for analysis).
    let cen_recall_ok = rec_cen >= 0.30;
    let gre_recall_ok = rec_gre >= 0.18;

    let check = |pass: bool, label: &str| {
        let mark = if pass { "PASS" } else { "FAIL" };
        println!("  [{mark}] {label}");
    };

    check(flat_perfect, "FlatStore recall = 1.0 (exhaustive baseline)");
    check(
        cen_faster,
        &format!("CentroidPages faster than flat (×{speedup_cen:.2})"),
    );
    check(
        gre_faster,
        &format!("GreedyCoherence faster than flat (×{speedup_gre:.2})"),
    );
    check(
        cen_coherent,
        &format!("CentroidPages coherence > flat (+{coherence_gain_cen:.4})"),
    );
    check(
        gre_coherent,
        &format!("GreedyCoherence coherence > flat (+{coherence_gain_gre:.4})"),
    );
    check(
        cen_recall_ok,
        &format!("CentroidPages recall@{K} >= 0.30 ({rec_cen:.4}) [>2.4× random probe baseline]"),
    );
    check(
        gre_recall_ok,
        &format!("GreedyCoherence recall@{K} >= 0.18 ({rec_gre:.4}) [>1.4× random probe baseline]"),
    );
    check(
        s_gre.avg_coherence >= s_cen.avg_coherence,
        &format!(
            "GreedyCoherence coherence ({:.4}) >= CentroidPages ({:.4})",
            s_gre.avg_coherence, s_cen.avg_coherence
        ),
    );

    let all_pass = flat_perfect
        && cen_faster
        && gre_faster
        && cen_coherent
        && gre_coherent
        && cen_recall_ok
        && gre_recall_ok
        && s_gre.avg_coherence >= s_cen.avg_coherence;

    println!();
    if all_pass {
        println!("RESULT: ALL CHECKS PASSED ✓");
    } else {
        println!("RESULT: SOME CHECKS FAILED — see above");
        std::process::exit(1);
    }
}
