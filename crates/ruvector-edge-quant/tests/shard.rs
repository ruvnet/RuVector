//! QuantShard behaviour: faithfulness to ruvector-rabitq, upsert/delete,
//! metrics, rerank, and budget refusals (413, never 500).

#![allow(clippy::field_reassign_with_default)]

mod common;

use common::*;
use ruvector_edge_quant::*;
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_rabitq::{AnnIndex, RabitqIndex};

fn shard(dim: usize, metric: Metric) -> QuantShard {
    QuantShard::new(QuantConfig::new(dim, metric, 42)).unwrap()
}

#[test]
fn codes_are_bit_identical_to_rabitq_index() {
    for (dim, kind) in [
        (37usize, RandomRotationKind::HaarDense),
        (64, RandomRotationKind::HaarDense),
        (384, RandomRotationKind::HaarDense),
        (100, RandomRotationKind::HadamardSigned),
        (384, RandomRotationKind::HadamardSigned),
    ] {
        let (data, _) = clustered(300, 0, dim, 5, 0.3, 1);
        let mut cfg = QuantConfig::new(dim, Metric::L2, 42);
        cfg.rotation = kind;
        let mut s = QuantShard::new(cfg).unwrap();
        s.upsert(&rows(&data)).unwrap();
        let mut idx = RabitqIndex::new_with_rotation(dim, 42, kind);
        for (i, v) in data.iter().enumerate() {
            idx.add(i, v.clone()).unwrap();
        }
        assert_eq!(s.packed(), idx.packed(), "dim {dim}");
        assert_eq!(s.norms(), idx.norms(), "dim {dim}");
        // L2 estimates rank like RabitqIndex's symmetric search.
        let q = &data[7];
        let ours: Vec<u64> = s
            .query(
                q,
                &QueryOptions {
                    top_k: 10,
                    rerank_factor: 1,
                },
                None,
            )
            .unwrap()
            .hits
            .iter()
            .map(|h| h.key)
            .collect();
        let theirs: Vec<u64> = idx
            .search(q, 10)
            .unwrap()
            .iter()
            .map(|r| r.id as u64)
            .collect();
        assert_eq!(ours[0], theirs[0], "dim {dim}");
    }
}

#[test]
fn full_rerank_is_exact_for_every_metric() {
    let (data, qs) = clustered(400, 20, 48, 8, 0.4, 2);
    for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
        let mut s = shard(48, metric);
        s.upsert(&rows(&data)).unwrap();
        let mut src = MemRows::from_data(&data);
        let opts = QueryOptions {
            top_k: 10,
            rerank_factor: 40,
        }; // 400 = all rows
        for q in &qs {
            let out = s.query(q, &opts, Some(&mut src)).unwrap();
            let got: Vec<u64> = out.hits.iter().map(|h| h.key).collect();
            assert_eq!(got, exact_topk(&data, q, 10, metric), "{metric:?}");
            assert!(out.hits.iter().all(|h| h.exact));
            assert_eq!(out.stats.reranked, 400);
        }
    }
}

#[test]
fn upsert_replace_and_delete_keep_index_consistent() {
    let (data, _) = clustered(50, 0, 16, 3, 0.3, 3);
    let mut s = shard(16, Metric::L2);
    let st = s.upsert(&rows(&data)).unwrap();
    assert_eq!((st.inserted, st.replaced), (50, 0));
    // Replace key 5 with row 9's vector: querying row 9 now finds 5 and 9 at 0.
    let st = s.upsert(&[(5, data[9].as_slice())]).unwrap();
    assert_eq!((st.inserted, st.replaced), (0, 1));
    let mut src = MemRows::from_data(&data);
    src.rows.insert(5, data[9].clone());
    let out = s
        .query(
            &data[9],
            &QueryOptions {
                top_k: 2,
                rerank_factor: 25,
            },
            Some(&mut src),
        )
        .unwrap();
    let mut keys: Vec<u64> = out.hits.iter().map(|h| h.key).collect();
    keys.sort();
    assert_eq!(keys, vec![5, 9]);
    // Delete first, middle, last and an absent key.
    assert_eq!(s.delete(&[0, 25, 49, 1000]), 3);
    assert_eq!(s.len(), 47);
    for k in [0u64, 25, 49] {
        assert!(!s.contains(k));
    }
    let all = s
        .query(
            &data[3],
            &QueryOptions {
                top_k: 47,
                rerank_factor: 1,
            },
            None,
        )
        .unwrap();
    let mut keys: Vec<u64> = all.hits.iter().map(|h| h.key).collect();
    keys.sort();
    let want: Vec<u64> = (0..50).filter(|k| ![0, 25, 49].contains(k)).collect();
    assert_eq!(keys, want);
    assert_eq!(s.keys().len() * s.n_words(), s.packed().len());
}

#[test]
fn rerank_drops_rows_the_store_no_longer_has() {
    let (data, _) = clustered(100, 0, 32, 4, 0.3, 4);
    let mut s = shard(32, Metric::Cosine);
    s.upsert(&rows(&data)).unwrap();
    let mut src = MemRows::from_data(&data);
    src.rows.remove(&10);
    let out = s
        .query(
            &data[10],
            &QueryOptions {
                top_k: 5,
                rerank_factor: 20,
            },
            Some(&mut src),
        )
        .unwrap();
    assert_eq!(out.stats.missing, 1);
    assert!(out.hits.iter().all(|h| h.key != 10));
    assert_eq!(out.hits.len(), 5);
}

#[test]
fn invalid_input_is_typed() {
    let mut s = shard(8, Metric::Cosine);
    let e = s.upsert(&[(1, &[0.0; 7][..])]).unwrap_err();
    assert!(matches!(
        e,
        QuantError::DimensionMismatch {
            expected: 8,
            actual: 7
        }
    ));
    assert_eq!(
        s.upsert(&[(1, &[f32::NAN; 8][..])]).unwrap_err().status(),
        400
    );
    assert_eq!(
        s.upsert(&[(1, &[0.0; 8][..])]).unwrap_err(),
        QuantError::NonFinite
    );
    // All-or-nothing: a bad row later in the batch writes nothing.
    let good = [1.0f32; 8];
    let bad = [f32::INFINITY; 8];
    assert!(s.upsert(&[(1, &good[..]), (2, &bad[..])]).is_err());
    assert!(s.is_empty());
    // Empty shard answers with no hits.
    let out = s.query(&good, &QueryOptions::default(), None).unwrap();
    assert!(out.hits.is_empty());
    assert!(QuantShard::new(QuantConfig::new(0, Metric::L2, 1)).is_err());
    assert!(QuantShard::new(QuantConfig::new(1537, Metric::L2, 1)).is_err());
}

fn assert_413(e: QuantError, resource: BudgetResource) {
    match &e {
        QuantError::BudgetExceeded { resource: r, .. } => assert_eq!(*r, resource),
        other => panic!("expected BudgetExceeded, got {other:?}"),
    }
    assert_eq!(e.status(), 413);
    let op: OpError = e.into();
    assert_eq!(op.code, ErrorCode::BudgetExceeded);
    assert_eq!(op.code.status(), 413);
    assert_eq!(op.code.as_str(), "budget_exceeded");
}

#[test]
fn budget_exhaustion_is_413_not_500() {
    let (data, _) = clustered(200, 0, 64, 4, 0.3, 5);
    let mut cfg = QuantConfig::new(64, Metric::L2, 7);

    // Vector cap: the whole batch is refused, nothing written.
    cfg.budget.max_vectors = 150;
    let mut s = QuantShard::new(cfg).unwrap();
    assert_413(s.upsert(&rows(&data)).unwrap_err(), BudgetResource::Vectors);
    assert!(s.is_empty());

    // Resident bytes: cap at exactly 100 rows.
    cfg.budget = Budget::default();
    cfg.budget.max_resident_bytes = budget::resident_bytes(100, 64, cfg.rotation);
    let mut s = QuantShard::new(cfg).unwrap();
    s.upsert(&rows(&data[..100])).unwrap();
    assert_eq!(s.resident_bytes(), cfg.budget.max_resident_bytes);
    // Replacing existing keys does not grow the shard: allowed.
    s.upsert(&[(3, data[150].as_slice())]).unwrap();
    assert_413(
        s.upsert(&[(100, data[100].as_slice())]).unwrap_err(),
        BudgetResource::ResidentBytes,
    );

    // Query budgets.
    let mut s = QuantShard::new(QuantConfig::new(64, Metric::L2, 7)).unwrap();
    s.upsert(&rows(&data)).unwrap();
    let q = &data[0];
    let e = s.query(
        q,
        &QueryOptions {
            top_k: 101,
            rerank_factor: 1,
        },
        None,
    );
    assert_413(e.unwrap_err(), BudgetResource::TopK);
    // top_k × factor above the rerank cap is clamped to the cap, not refused.
    let mut src = MemRows::from_data(&data);
    let mut b = Budget::default();
    b.max_rerank_candidates = 150;
    s.set_budget(b);
    let out = s
        .query(
            q,
            &QueryOptions {
                top_k: 100,
                rerank_factor: 11,
            },
            Some(&mut src),
        )
        .unwrap();
    assert_eq!((out.stats.candidates, src.fetched), (150, 150));
    // A cap below top_k cannot answer the query: 413 before the store is touched.
    b.max_rerank_candidates = 50;
    s.set_budget(b);
    let e = s.query(
        q,
        &QueryOptions {
            top_k: 100,
            rerank_factor: 1,
        },
        Some(&mut src),
    );
    assert_413(e.unwrap_err(), BudgetResource::RerankCandidates);
    assert_eq!(src.fetched, 150, "refused before touching the store");
    let mut b = Budget::default();
    b.max_query_units = budget::query_units(200, 64, cfg.rotation, 0) - 1;
    s.set_budget(b);
    let e = s.query(
        q,
        &QueryOptions {
            top_k: 10,
            rerank_factor: 1,
        },
        None,
    );
    assert_413(e.unwrap_err(), BudgetResource::QueryUnits);

    // Creating a shard whose rotation rebuild exceeds the load budget.
    let mut cfg = QuantConfig::new(1536, Metric::L2, 1);
    cfg.rotation = RandomRotationKind::HaarDense;
    cfg.budget.max_load_units = 1_000_000;
    assert_413(
        QuantShard::new(cfg).err().unwrap(),
        BudgetResource::LoadUnits,
    );
}

#[test]
fn corrupt_and_rerank_failures_are_503_not_500() {
    let e = QuantError::Corrupt(CorruptKind::BadMagic);
    assert_eq!(e.status(), 503);
    assert_eq!(QuantError::Rerank("x".into()).status(), 503);
    let op: OpError = e.into();
    assert_eq!(op.code, ErrorCode::ShardUnavailable);
}

#[test]
fn default_config_admits_every_common_dimension() {
    for dim in [1usize, 128, 384, 512, 768, 1024, 1536] {
        let cfg = QuantConfig::new(dim, Metric::Cosine, 1);
        assert_eq!(cfg.rotation, RandomRotationKind::HadamardSigned);
        assert!(QuantShard::new(cfg).is_ok(), "hadamard dim {dim}");
    }
    // Opt-in Haar fits the default load budget up to 512 dims; above that
    // the D³ rebuild is a clean 413, not a multi-second cold load.
    for (dim, ok) in [(384usize, true), (512, true), (768, false), (1536, false)] {
        let mut cfg = QuantConfig::new(dim, Metric::Cosine, 1);
        cfg.rotation = RandomRotationKind::HaarDense;
        match QuantShard::new(cfg) {
            Ok(_) => assert!(ok, "haar dim {dim} should be refused"),
            Err(e) => {
                assert!(!ok, "haar dim {dim}: {e:?}");
                assert_413(e, BudgetResource::LoadUnits);
            }
        }
    }
}

/// Regression: a finite row whose f32 sum of squares overflows used to get
/// norm = +inf (every later snapshot then failed cold load with 503) and an
/// all-ones code that scored 0 against every L2 query.
#[test]
fn large_norm_rows_round_trip_and_unrepresentable_norms_are_400() {
    use ruvector_edge_quant::persist::{load_frames, save_frames, MAX_FRAME_BYTES};
    let dim = 384;
    let (data, qs) = clustered(200, 5, dim, 4, 0.3, 11);
    let dir = &data[17];
    let big: Vec<f32> = dir.iter().map(|&x| x * 1e19).collect();
    let flat = vec![1e19f32; dim];
    let huge = vec![1e38f32; dim]; // f64 norm ≈ 2e39 > f32::MAX
    for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
        let mut s = shard(dim, metric);
        s.upsert(&rows(&data)).unwrap();
        s.upsert(&[(999, big.as_slice()), (1000, flat.as_slice())])
            .unwrap();
        let pos = |k: u64| s.keys().iter().position(|&x| x == k).unwrap();
        for k in [999, 1000] {
            let n = s.norms()[pos(k)];
            assert!(n.is_finite() && n > 1e20, "{metric:?} key {k}: norm {n}");
        }
        // Same direction as row 17 → (essentially) the same code.
        let nw = s.n_words();
        let code = |p: usize| s.packed()[p * nw..(p + 1) * nw].to_vec();
        let flips: u32 = code(pos(999))
            .iter()
            .zip(code(pos(17)))
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        assert!(flips <= 2, "{metric:?}: {flips} bits differ");

        // Unrepresentable norm: 400, nothing written.
        let before = s.len();
        let e = s.upsert(&[(2000, huge.as_slice())]).unwrap_err();
        assert_eq!(e, QuantError::NonFinite);
        assert_eq!(e.status(), 400);
        assert_eq!(s.len(), before);

        // Snapshots stay loadable.
        let frames = save_frames(&s, MAX_FRAME_BYTES).unwrap();
        let back = load_frames(frames.iter().map(|f| f.as_slice()), Budget::default()).unwrap();
        assert_eq!(back.norms(), s.norms());

        // Estimates stay finite and the huge rows do not score 0 for L2.
        let opts = QueryOptions {
            top_k: 3,
            rerank_factor: 1,
        };
        for q in &qs {
            let out = back.query(q, &opts, None).unwrap();
            assert!(out.hits.iter().all(|h| h.distance.is_finite()));
            if metric == Metric::L2 {
                assert!(out.hits.iter().all(|h| h.key < 999), "{:?}", out.hits);
            }
        }
        // A query whose own f32 norm overflows is still answered.
        let bigq: Vec<f32> = qs[0].iter().map(|&x| x * 1e30).collect();
        let out = back.query(&bigq, &opts, None).unwrap();
        assert!(out.hits.iter().all(|h| h.distance.is_finite()));
        if metric == Metric::Cosine {
            // Scale-free metric: the huge query ranks like the original
            // (for L2/Dot the big-norm rows legitimately win).
            let small = back.query(&qs[0], &opts, None).unwrap();
            assert_eq!(out.hits[0].key, small.hits[0].key, "{metric:?}");
        }
    }
}
