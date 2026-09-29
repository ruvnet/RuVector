//! ADR §15 M2a acceptance dataset: recall@10 on 20k × 384 iid Gaussian
//! vectors (held-out queries, exact f32 ground truth with the same
//! `distance`). Unstructured data is the hard case for a graph index, so
//! this also fixes the beam width M2b needs there. `--nocapture` prints
//! the table.

mod common;

use common::*;
use ruvector_edge_index::{HnswIndex, HnswParams, Metric, QuantFlatIndex, QuantParams, SliceFetch};

const N20: usize = 20_000;
const Q: usize = 100;
/// Rerank candidates for M2a on unstructured data (`4·k` suffices).
const FLAT_CANDIDATES: usize = 4 * K;
/// Smallest swept `ef` reaching recall@10 ≥ 0.95 for M2b on this set, per
/// metric (the clustered set needs 64, see `recall.rs`). An f32-built
/// graph (hnsw_rs, same m/efc) needs the same: cosine 0.869 @ 512,
/// 0.974 @ 1024; l2 0.943 @ 512 — isotropic 384-d is intrinsically hard.
const HNSW_EF_RANDOM_L2: usize = 512;
const HNSW_EF_RANDOM_COSINE: usize = 1024;

fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut g = Gen::new(seed);
    (0..n * DIM).map(|_| g.gauss()).collect()
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    xs.sum::<f64>() / Q as f64
}

struct Case {
    name: &'static str,
    metric: Metric,
    quant: QuantParams,
    hnsw_ef: usize,
}

#[test]
fn random_20k_x_384_recall_at_10() {
    let base = gaussian(7, N20);
    let queries = gaussian(8, Q);
    let sample = &base[..1000 * DIM];
    let cases = [
        Case {
            name: "l2 trained",
            metric: Metric::L2,
            quant: QuantParams::train(Metric::L2, DIM, sample, 1).unwrap(),
            hnsw_ef: HNSW_EF_RANDOM_L2,
        },
        Case {
            name: "cosine trained",
            metric: Metric::Cosine,
            quant: QuantParams::train(Metric::Cosine, DIM, sample, 1).unwrap(),
            hnsw_ef: HNSW_EF_RANDOM_COSINE,
        },
        Case {
            name: "cosine fixed[-1,1]",
            metric: Metric::Cosine,
            quant: QuantParams::cosine_fixed(DIM, 1).unwrap(),
            hnsw_ef: HNSW_EF_RANDOM_COSINE,
        },
    ];
    let mut table = Vec::new();
    let mut failures = Vec::new();
    let mut fetch = SliceFetch::new(&base, DIM);
    for c in cases {
        let t = truth(c.metric, &base, &queries, DIM, K);
        let rows = || queries.chunks_exact(DIM).zip(&t);

        let mut flat = QuantFlatIndex::with_capacity(c.quant.clone(), N20 as u32, N20).unwrap();
        for (i, v) in base.chunks_exact(DIM).enumerate() {
            flat.upsert(i as u32, v).unwrap();
        }
        let raw = mean(rows().map(|(qv, t)| recall(t, &flat.search(qv, K).unwrap())));
        let rr = mean(rows().map(|(qv, t)| {
            recall(
                t,
                &flat
                    .search_rerank(qv, K, FLAT_CANDIDATES, &mut fetch)
                    .unwrap(),
            )
        }));
        table.push(format!(
            "{:18} flat          codes-only {raw:.3}  rerank({FLAT_CANDIDATES}) {rr:.3}",
            c.name
        ));
        if rr < 0.95 {
            failures.push(format!("{} flat {rr}", c.name));
        }

        let mut h = HnswIndex::with_capacity(HnswParams::default(), c.quant, N20).unwrap();
        for (i, v) in base.chunks_exact(DIM).enumerate() {
            h.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
        }
        for ef in [64, 128, 256, 512, 1024] {
            let raw = mean(rows().map(|(qv, t)| recall(t, &h.search(qv, K, ef).unwrap())));
            let rr = mean(
                rows().map(|(qv, t)| recall(t, &h.search_rerank(qv, K, ef, &mut fetch).unwrap())),
            );
            table.push(format!(
                "{:18} hnsw ef={ef:4} codes-only {raw:.3}  rerank {rr:.3}",
                c.name
            ));
            if ef == c.hnsw_ef && rr < 0.95 {
                failures.push(format!("{} hnsw ef {ef}: {rr}", c.name));
            }
        }
    }
    let table = table.join("\n");
    eprintln!("20k x 384 iid gaussian, recall@10 ({Q} held-out queries):\n{table}");
    assert!(failures.is_empty(), "{failures:?}\n{table}");
}
