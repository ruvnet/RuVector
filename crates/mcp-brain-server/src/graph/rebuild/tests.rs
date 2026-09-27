//! Tests and the ignored timing benchmark for the off-lock rebuild
//! (ADR-349 item 6).

use super::*;
use crate::types::{BetaParams, BrainMemory};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};

const THRESHOLD: f64 = 0.55;

fn memory(id_seed: &mut StdRng, embedding: Vec<f32>, quality: BetaParams) -> BrainMemory {
    let mut bytes = [0u8; 16];
    id_seed.fill(&mut bytes);
    BrainMemory {
        id: Uuid::from_bytes(bytes),
        category: BrainCategory::Pattern,
        title: "t".into(),
        content: "c".into(),
        tags: vec![],
        code_snippet: None,
        embedding,
        contributor_id: "test".into(),
        quality_score: quality,
        partition_id: None,
        witness_hash: String::new(),
        rvf_gcs_path: None,
        redaction_log: None,
        dp_proof: None,
        witness_chain: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    }
}

fn gaussian(rng: &mut StdRng) -> f32 {
    // Box-Muller; good enough for synthetic embeddings.
    let u1: f64 = rng.gen_range(1e-12..1.0);
    let u2: f64 = rng.gen();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Clustered, L2-normalised embeddings: intra-cluster cosine ≈ 0.7 (above the
/// 0.55 threshold), inter-cluster ≈ 0, so each node has ≈ `cluster - 1` edges
/// — the live graph's shape (59,758 nodes, ~1.2M edges ⇒ degree ≈ 40).
fn synthetic(n: usize, dim: usize, cluster: usize, seed: u64) -> Vec<BrainMemory> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut center = vec![0f32; dim];
    let scale = 1.0 / (dim as f32).sqrt();
    for i in 0..n {
        if i % cluster == 0 {
            for x in center.iter_mut() {
                *x = gaussian(&mut rng);
            }
            crate::graph::normalize_embedding(&mut center);
        }
        let mut e: Vec<f32> = center
            .iter()
            .map(|c| c + 0.65 * scale * gaussian(&mut rng))
            .collect();
        crate::graph::normalize_embedding(&mut e);
        // Some nodes fall under the ADR-149 P2 quality floor.
        let q = if i % 97 == 13 {
            BetaParams {
                alpha: 0.001,
                beta: 1.0,
            }
        } else {
            BetaParams::new()
        };
        out.push(memory(&mut rng, e, q));
    }
    out
}

/// The pre-ADR-349 #6 edge pass, verbatim in behaviour: full
/// `cosine_similarity` per pair, row-major, quality floor on both ends.
fn reference_edges(memories: &[BrainMemory]) -> Vec<(Uuid, Uuid, f64)> {
    let q: Vec<f64> = memories.iter().map(|m| m.quality_score.mean()).collect();
    let mut out = Vec::new();
    for i in 0..memories.len() {
        if q[i] < EDGE_QUALITY_FLOOR {
            continue;
        }
        for j in (i + 1)..memories.len() {
            if q[j] < EDGE_QUALITY_FLOOR {
                continue;
            }
            let sim = cosine_similarity(&memories[i].embedding, &memories[j].embedding);
            if sim >= THRESHOLD {
                out.push((memories[i].id, memories[j].id, sim));
            }
        }
    }
    out
}

/// Bitwise edge equality, including weights.
fn assert_same_edges(got: &[(Uuid, Uuid, f64)], want: &[(Uuid, Uuid, f64)]) {
    assert_eq!(got.len(), want.len(), "edge count differs");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.0 == w.0 && g.1 == w.1 && g.2.to_bits() == w.2.to_bits(),
            "edge {k} differs: got {g:?}, want {w:?}"
        );
    }
}

/// Same edges as a set (bitwise weights). `add_memory` walks a `HashMap`, so
/// the order of one added node's edges is arbitrary; used wherever a test
/// replays an add.
fn assert_same_edge_set(got: &[(Uuid, Uuid, f64)], want: &[(Uuid, Uuid, f64)]) {
    let key = |e: &(Uuid, Uuid, f64)| (e.0, e.1, e.2.to_bits());
    let mut g: Vec<_> = got.iter().map(key).collect();
    let mut w: Vec<_> = want.iter().map(key).collect();
    g.sort();
    w.sort();
    assert_eq!(g.len(), w.len(), "edge count differs");
    assert!(g == w, "edge sets differ");
}

#[test]
fn prenormed_cosine_is_bitwise_identical() {
    let mut rng = StdRng::seed_from_u64(7);
    for dim in [0usize, 1, 3, 4, 7, 8, 128, 131] {
        for _ in 0..50 {
            let a: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();
            let b: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();
            let want = cosine_similarity(&a, &b);
            let got = cosine_prenormed(&a, split_norm(&a), &b, split_norm(&b));
            assert_eq!(got.to_bits(), want.to_bits(), "dim {dim}");
        }
    }
    // Zero vector and length mismatch take the same early exits.
    let z = vec![0f32; 8];
    let o = vec![1f32; 8];
    assert_eq!(cosine_prenormed(&z, 0.0, &o, split_norm(&o)), 0.0);
    assert_eq!(cosine_prenormed(&o, 1.0, &o[..4], 1.0), 0.0);
}

#[test]
fn build_batch_matches_reference_exactly_sequential_and_parallel() {
    let mut mems = synthetic(1200, 32, 20, 1);
    // Edge cases: a zero vector and a wrong-dimension vector.
    mems[5].embedding = vec![0.0; 32];
    mems[6].embedding = vec![0.5; 16];
    let want = reference_edges(&mems);
    assert!(
        want.len() > 5_000,
        "synthetic graph too sparse to be a test"
    );

    for threads in [1usize, 2, 3, 8] {
        let build = KnowledgeGraph::build_batch_inner(&mems, THRESHOLD, threads, 0);
        let mut g = KnowledgeGraph::new();
        let ticket = g.begin_rebuild().unwrap();
        g.install_batch(ticket, build).unwrap();
        assert_same_edges(&g.edges_snapshot(), &want);
        let ids: Vec<Uuid> = mems.iter().map(|m| m.id).collect();
        assert_eq!(
            g.node_ids_snapshot(),
            ids,
            "positions must follow input order"
        );
    }

    // The in-place wrapper goes through the same builder.
    let mut g = KnowledgeGraph::new();
    g.rebuild_from_batch(&mems);
    assert_same_edges(&g.edges_snapshot(), &want);
}

#[test]
fn balanced_ranges_cover_every_row_once() {
    for m in [0usize, 1, 2, 5, 100, 4097] {
        for parts in [1usize, 2, 3, 7] {
            let r = balanced_row_ranges(m, parts);
            assert_eq!(r.first().map(|x| x.start), Some(0));
            assert_eq!(r.last().map(|x| x.end), Some(m));
            for w in r.windows(2) {
                assert_eq!(w[0].end, w[1].start);
            }
        }
    }
}

#[test]
fn begin_rebuild_is_single_flight() {
    let mut g = KnowledgeGraph::new();
    let t = g.begin_rebuild().expect("first begin");
    assert!(g.begin_rebuild().is_none(), "second begin must be refused");
    assert!(g.rebuild_in_flight());
    assert!(g.abort_rebuild(t));
    assert!(!g.rebuild_in_flight());
    let t2 = g.begin_rebuild().expect("begin after abort");
    // A stale ticket can neither abort nor install the newer rebuild.
    assert!(!g.abort_rebuild(t));
    let b = KnowledgeGraph::build_batch(&[], THRESHOLD, 1);
    assert!(g.install_batch(t, b).is_none());
    assert!(g.rebuild_in_flight());
    assert!(g.abort_rebuild(t2));
}

#[test]
fn in_place_rebuild_supersedes_an_in_flight_one() {
    let mems = synthetic(200, 16, 10, 2);
    let mut g = KnowledgeGraph::new();
    let ticket = g.begin_rebuild().unwrap();
    let build = KnowledgeGraph::build_batch(&mems, ticket.threshold(), 1);
    g.rebuild_from_batch(&mems[..100]);
    assert!(g.install_batch(ticket, build).is_none());
    assert_eq!(g.node_count(), 100);
}

#[test]
fn mutations_during_build_are_replayed() {
    let mems = synthetic(300, 16, 10, 3);
    let mut g = KnowledgeGraph::new();
    g.rebuild_from_batch(&mems[..250]);

    let ticket = g.begin_rebuild().unwrap();
    // Snapshot taken after begin: 250..280 are "in the store".
    let snapshot: Vec<BrainMemory> = mems[..280].to_vec();
    // While building: one add already in the snapshot, one that is not, and
    // a removal of a snapshot node.
    g.add_memory(&mems[260]);
    g.add_memory(&mems[290]);
    let removed = mems[10].id;
    g.remove_memory(&removed);

    let build = KnowledgeGraph::build_batch(&snapshot, ticket.threshold(), 1);
    let (report, _retired) = g.install_batch(ticket, build).unwrap();
    assert_eq!(
        report.replayed_adds, 1,
        "snapshot member must not be re-added"
    );
    assert_eq!(report.replayed_removes, 1);

    // Expected: exact rebuild of the snapshot, then the same mutations.
    let mut want = KnowledgeGraph::new();
    want.rebuild_from_batch(&snapshot);
    want.add_memory(&mems[290]);
    want.remove_memory(&removed);
    assert_eq!(g.node_ids_snapshot(), want.node_ids_snapshot());
    assert_same_edge_set(&g.edges_snapshot(), &want.edges_snapshot());
    assert!(!g.rebuild_in_flight());
}

#[test]
fn install_batch_invalidates_an_in_flight_sparsifier_build() {
    let mems = synthetic(120, 16, 12, 4);
    let mut g = KnowledgeGraph::new();
    g.rebuild_from_batch(&mems);
    let Some((entries, nodes, edges, gen)) = g.sparsifier_snapshot() else {
        panic!("expected edges");
    };
    let spar = KnowledgeGraph::build_sparsifier_from(&entries, nodes);
    let ticket = g.begin_rebuild().unwrap();
    let build = KnowledgeGraph::build_batch(&mems, ticket.threshold(), 1);
    g.install_batch(ticket, build).unwrap();
    if let Some(s) = spar {
        assert!(!g.install_sparsifier(s, nodes, edges, gen));
    }
}

/// Cancelling the task that drives a rebuild (a client disconnect dropping
/// the handler future) must release the single-flight marker, or no rebuild
/// could ever run again on that instance.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_rebuild_releases_the_marker() {
    let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let task = tokio::spawn(rebuild_off_lock(
        graph.clone(),
        move || {
            let _ = rx.recv();
            Vec::new()
        },
        None,
    ));
    for _ in 0..200 {
        if graph.read().rebuild_in_flight() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(graph.read().rebuild_in_flight());
    task.abort();
    let _ = task.await;
    assert!(!graph.read().rebuild_in_flight(), "marker leaked");
    let _ = tx.send(());
    let again = rebuild_off_lock(graph.clone(), Vec::new, None).await;
    assert!(matches!(again, RebuildOutcome::Installed(_)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_rebuilds_are_single_flight() {
    let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let first = spawn_rebuild(
        graph.clone(),
        move || {
            let _ = rx.recv();
            Vec::new()
        },
        None,
    );
    while !graph.read().rebuild_in_flight() {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let second = rebuild_off_lock(graph.clone(), Vec::new, None).await;
    assert!(matches!(second, RebuildOutcome::AlreadyRunning));
    tx.send(()).unwrap();
    assert!(matches!(first.await.unwrap(), RebuildOutcome::Installed(_)));
}

/// Reader latency samples from `readers` OS threads until `stop` is set.
/// Each sample is one `graph.read()` + `ranked_search` — what `/v1/search`
/// does under the lock. Also counts samples taken while a rebuild was in
/// flight, to prove the reads really overlapped it.
fn spawn_readers(
    graph: &Arc<RwLock<KnowledgeGraph>>,
    query: Vec<f32>,
    readers: usize,
    stop: &Arc<AtomicBool>,
) -> Vec<std::thread::JoinHandle<(Vec<Duration>, usize)>> {
    (0..readers)
        .map(|_| {
            let (graph, query, stop) = (graph.clone(), query.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut samples = Vec::new();
                let mut overlapped = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let t = Instant::now();
                    let g = graph.read();
                    let in_flight = g.rebuild_in_flight();
                    let hits = g.ranked_search(&query, 10);
                    drop(g);
                    samples.push(t.elapsed());
                    std::hint::black_box(hits);
                    if in_flight {
                        overlapped += 1;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                (samples, overlapped)
            })
        })
        .collect()
}

fn collect(
    handles: Vec<std::thread::JoinHandle<(Vec<Duration>, usize)>>,
) -> (Vec<Duration>, usize) {
    let mut all = Vec::new();
    let mut overlapped = 0;
    for h in handles {
        let (s, o) = h.join().unwrap();
        all.extend(s);
        overlapped += o;
    }
    all.sort();
    (all, overlapped)
}

fn pct(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

/// The ADR-349 #6 acceptance test: a rebuild of a few thousand nodes runs on
/// one task while other threads search continuously. Every read must finish
/// well under the bound, and the installed graph must equal the exact
/// rebuild (plus the mutation made mid-build).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reads_stay_responsive_during_rebuild_and_result_is_exact() {
    let mems = synthetic(3000, 128, 30, 5);
    let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
    // Start from a populated graph, as in production (old graph serves reads
    // while the new one builds).
    graph.write().rebuild_from_batch(&mems[..2500]);

    let stop = Arc::new(AtomicBool::new(false));
    let readers = spawn_readers(&graph, mems[42].embedding.clone(), 2, &stop);

    let snapshot = mems[..2990].to_vec();
    let load_snapshot = snapshot.clone();
    let handle = spawn_rebuild(graph.clone(), move || load_snapshot, Some(usize::MAX));

    // A share landing mid-rebuild.
    while !graph.read().rebuild_in_flight() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    graph.write().add_memory(&mems[2995]);

    let outcome = handle.await.unwrap();
    stop.store(true, Ordering::Relaxed);
    let (samples, overlapped) = collect(readers);

    let report = match outcome {
        RebuildOutcome::Installed(r) => r,
        other => panic!("rebuild did not install: {other:?}"),
    };
    let max = *samples.last().unwrap();
    eprintln!(
        "rebuild build={:?} install={:?}; reads={} overlapped={} p50={:?} p99={:?} max={:?}",
        report.build.elapsed,
        report.install_elapsed,
        samples.len(),
        overlapped,
        pct(&samples, 0.5),
        pct(&samples, 0.99),
        max
    );
    assert!(
        overlapped >= 5,
        "reads must overlap the rebuild to test anything (overlapped={overlapped})"
    );
    assert!(
        max < Duration::from_millis(200),
        "a read took {max:?} during the rebuild"
    );

    let mut want = KnowledgeGraph::new();
    want.rebuild_from_batch(&snapshot);
    let exact = want.edges_snapshot();
    want.add_memory(&mems[2995]);
    let g = graph.read();
    // The rebuilt portion is order-exact; the replayed add is compared as a set.
    assert_same_edges(&g.edges_snapshot()[..exact.len()], &exact);
    assert_eq!(g.node_ids_snapshot(), want.node_ids_snapshot());
    assert_same_edge_set(&g.edges_snapshot(), &want.edges_snapshot());
    assert!(!g.rebuild_in_flight());
}

/// Old vs new timing. Not run by default:
///
/// `cargo test --release -p mcp-brain-server rebuild_benchmark -- --ignored --nocapture`
///
/// `REBUILD_BENCH_N` (comma list, default `10000`) and `REBUILD_BENCH_DIM`
/// (default 128, the live `EMBED_DIM`) control the size.
#[test]
#[ignore]
fn rebuild_benchmark() {
    let sizes: Vec<usize> = std::env::var("REBUILD_BENCH_N")
        .unwrap_or_else(|_| "10000".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let dim: usize = std::env::var("REBUILD_BENCH_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    for n in sizes {
        let mems = synthetic(n, dim, 40, 11);
        let query = mems[7].embedding.clone();

        // --- old: the previous code path, exact pass under graph.write() ---
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        graph.write().rebuild_from_batch(&mems);
        let _ = graph.read().ranked_search(&query, 10); // warm CSR
                                                        // Baseline: the same reads with no rebuild running.
        let stop = Arc::new(AtomicBool::new(false));
        let readers = spawn_readers(&graph, query.clone(), 2, &stop);
        std::thread::sleep(Duration::from_secs(3));
        stop.store(true, Ordering::Relaxed);
        let (idle, _) = collect(readers);
        eprintln!(
            "n={n} IDLE (no rebuild) read p50={:?} p99={:?} max={:?} (reads={})",
            pct(&idle, 0.5),
            pct(&idle, 0.99),
            idle.last().copied().unwrap_or_default(),
            idle.len()
        );
        let stop = Arc::new(AtomicBool::new(false));
        let readers = spawn_readers(&graph, query.clone(), 2, &stop);
        std::thread::sleep(Duration::from_millis(50));
        let t = Instant::now();
        let old_edges = {
            // The pre-fix body: per-pair full cosine with the write lock held.
            let _g = graph.write();
            reference_edges(&mems).len()
        };
        let old_wall = t.elapsed();
        std::thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        let (old_s, _) = collect(readers);

        for threads in [1usize, default_build_threads()] {
            std::env::set_var("GRAPH_REBUILD_THREADS", threads.to_string());
            let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
            graph.write().rebuild_from_batch(&mems);
            let _ = graph.read().ranked_search(&query, 10);
            let stop = Arc::new(AtomicBool::new(false));
            let readers = spawn_readers(&graph, query.clone(), 2, &stop);
            std::thread::sleep(Duration::from_millis(50));
            let snap = mems.clone();
            let t = Instant::now();
            let outcome = rt.block_on(rebuild_off_lock(graph.clone(), move || snap, None));
            let new_wall = t.elapsed();
            std::thread::sleep(Duration::from_millis(50));
            stop.store(true, Ordering::Relaxed);
            let (new_s, overlapped) = collect(readers);
            let RebuildOutcome::Installed(r) = outcome else {
                panic!("not installed");
            };
            assert_eq!(r.edges, old_edges);
            eprintln!(
                "n={n} dim={dim} edges={} | OLD wall={:?} read p99={:?} max={:?} (reads={}) \
                 | NEW threads={} wall={:?} (build={:?}, install lock={:?}) read p50={:?} \
                 p99={:?} max={:?} (reads={}, overlapped={})",
                r.edges,
                old_wall,
                pct(&old_s, 0.99),
                old_s.last().copied().unwrap_or_default(),
                old_s.len(),
                r.build.threads,
                new_wall,
                r.build.elapsed,
                r.install_elapsed,
                pct(&new_s, 0.5),
                pct(&new_s, 0.99),
                new_s.last().copied().unwrap_or_default(),
                new_s.len(),
                overlapped
            );
        }
        std::env::remove_var("GRAPH_REBUILD_THREADS");
    }
}
