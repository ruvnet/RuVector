//! Regression tests for the M2 review findings on shard maintenance:
//! HNSW iid gaps after a rebuild (and their healing), sliced link repair,
//! compaction when dead slots block admission, the config-dependent flush
//! threshold / HNSW load cap, and the filtered-HNSW query step charge.

mod common;

use common::m2::*;
use common::T0;
use ruvector_edge_store::shard::maintain::{flush_ops, hnsw_node_cap};
use ruvector_edge_store::shard::{
    validate_query, Due, IndexConfig, IndexSource, FLUSH_OPS, SHARD_RESIDENT_CAP_BYTES,
};
use ruvector_edge_store::{schema, ErrorCode, MemSqlStore, Metric, SqlStore, VectorShard};

fn upsert(
    s: &mut VectorShard,
    st: &dyn SqlStore,
    d: &[(String, Vec<f32>)],
) -> Result<(), ErrorCode> {
    let (dm, cfg) = (dm(60), cfg(8, Metric::L2, hnsw()));
    let plan = s.plan_upsert(&dm, &cfg, rows(d)).map_err(|e| e.code)?;
    s.apply_upsert(st, plan, ACTOR, T0).map_err(|e| e.code)?;
    Ok(())
}

fn exact_ids(s: &mut VectorShard, st: &dyn SqlStore, q: &[f32]) -> Vec<String> {
    let (dm, cfg) = (dm(60), cfg(8, Metric::L2, hnsw()));
    let mut r = req(q, 10);
    r.rerank = Some(1000);
    r.ef = Some(1000);
    ranked(s, st, &dm, &cfg, &r)
        .into_iter()
        .map(|m| m.0)
        .collect()
}

#[test]
fn deleting_the_newest_rows_before_a_requantize_leaves_no_iid_gap() {
    // The review's repro: l2 HNSW, dim 8; 64 + 64 + 10 rows (iids
    // 129..=138); delete those 10; the requantize alarm rebuilds densely
    // (1..=128) and must also reset `next_iid` to 129, or the next insert
    // opens a gap the HNSW encoder refuses on every later flush.
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let d = data(143, 8, 31);
    upsert(&mut s, &st, &d[..64]).unwrap();
    upsert(&mut s, &st, &d[64..128]).unwrap();
    upsert(&mut s, &st, &d[128..138]).unwrap();
    let newest: Vec<String> = d[128..138].iter().map(|r| r.0.clone()).collect();
    s.delete(&st, &dm(60), &newest, ACTOR, false, T0).unwrap();
    assert_eq!(s.maintenance_due(), Some(Due::Now));
    let rep = s.maintain(&st).unwrap();
    assert!(rep.requantized, "{rep:?}");
    upsert(&mut s, &st, &d[138..143]).unwrap();
    // The flush encodes directly: no gap, so no healing rebuild needed.
    let rep = s.maintain(&st).unwrap();
    assert!(rep.flushed && !rep.rebuilt, "{rep:?}");
    for _ in 0..3 {
        s.maintain(&st).unwrap();
        assert!(!s.is_poisoned());
    }
    assert_eq!(s.pending_ops(), 0, "the tail was flushed");
    let mut kept = d[..128].to_vec();
    kept.extend_from_slice(&d[138..]);
    let mut cold = reopen(&st);
    assert_eq!(cold.index_source(), Some(IndexSource::Current));
    assert_eq!(cold.state_digest(), s.state_digest());
    for (_, q) in data(4, 8, 32) {
        assert_eq!(
            exact_ids(&mut cold, &st, &q),
            brute(Metric::L2, &kept, &q, 10)
        );
    }
}

#[test]
fn an_hnsw_iid_gap_heals_by_a_dense_rebuild_instead_of_poisoning() {
    // Force a gap the way a stale `meta.next_iid` would, then write.
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let d = data(80, 8, 33);
    upsert(&mut s, &st, &d[..64]).unwrap();
    s.maintain(&st).unwrap();
    st.exec(schema::META_PUT, &["next_iid".into(), "500".into()])
        .unwrap();
    let mut s = reopen(&st);
    upsert(&mut s, &st, &d[64..]).unwrap();
    let rep = s.maintain(&st).unwrap();
    assert!(rep.rebuilt && rep.flushed, "{rep:?}");
    assert!(!s.is_poisoned());
    assert_eq!(s.maintenance_due(), None);
    let mut cold = reopen(&st);
    assert_eq!(cold.index_source(), Some(IndexSource::Current));
    for (_, q) in data(4, 8, 34) {
        assert_eq!(exact_ids(&mut cold, &st, &q), brute(Metric::L2, &d, &q, 10));
    }
}

#[test]
fn hnsw_link_repair_runs_in_alarm_slices_with_writes_in_between() {
    // Heavy config (1536-d, m 48): a slice repairs a few hundred nodes.
    let heavy = IndexConfig::Hnsw {
        m: 48,
        ef_construction: 200,
    };
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(61), cfg(1536, Metric::Cosine, heavy));
    let mut d = data(1100, 1536, 35);
    ingest(&mut s, &st, &dm, &cfg, &d).unwrap();
    let gone: Vec<String> = (0..1100).step_by(7).map(|i| format!("v{i:05}")).collect();
    s.delete(&st, &dm, &gone, ACTOR, false, T0).unwrap();
    d.retain(|(id, _)| !gone.contains(id));
    let mut slices = 0;
    loop {
        assert_eq!(s.maintenance_due(), Some(Due::Now));
        let rep = s.maintain(&st).unwrap();
        if rep.compacted {
            break;
        }
        assert!(rep.repaired, "{rep:?}");
        slices += 1;
        if slices == 1 {
            // A write between slices is served and survives the purge.
            let extra = data(3, 1536, 36)
                .into_iter()
                .map(|(id, v)| (format!("x{id}"), v))
                .collect::<Vec<_>>();
            let plan = s.plan_upsert(&dm, &cfg, rows(&extra)).unwrap();
            s.apply_upsert(&st, plan, ACTOR, T0).unwrap();
            d.extend(extra);
        }
    }
    assert!(slices >= 2, "{slices} slices");
    let mut cold = reopen(&st);
    assert_eq!(cold.index_source(), Some(IndexSource::Current));
    assert_eq!(cold.len(), d.len());
    for (_, q) in data(3, 1536, 37) {
        let mut r = req(&q, 10);
        r.rerank = Some(1000);
        r.ef = Some(1000);
        let got: Vec<String> = ranked(&mut cold, &st, &dm, &cfg, &r)
            .into_iter()
            .map(|m| m.0)
            .collect();
        assert_eq!(got, brute(Metric::Cosine, &d, &q, 10));
    }
}

#[test]
fn dead_slots_blocking_admission_schedule_a_compaction() {
    // Flat 384-d: fill toward the cap, delete ~15% (below the 25% dead
    // threshold), re-insert new ids until the cap refuses; the refusal is
    // caused by dead slots, so compaction must become due and fix it.
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(62), cfg(384, Metric::Cosine, flat()));
    ingest(&mut s, &st, &dm, &cfg, &data(22_000, 384, 38)).unwrap();
    let gone: Vec<String> = (0..22_000).step_by(6).map(|i| format!("v{i:05}")).collect();
    for c in gone.chunks(500) {
        s.delete(&st, &dm, c, ACTOR, false, T0).unwrap();
    }
    while s.maintenance_due() == Some(Due::Now) {
        s.maintain(&st).unwrap();
    }
    let before = s.len();
    let fresh = data(8_000, 384, 39);
    let (mut refused, mut compactions) = (None, 0);
    for (b, chunk) in fresh.chunks(64).enumerate() {
        let batch: Vec<_> = chunk
            .iter()
            .map(|(id, v)| (format!("n{b:03}{id}"), v.clone()))
            .collect();
        match s.plan_upsert(&dm, &cfg, rows(&batch)) {
            Ok(p) => {
                s.apply_upsert(&st, p, ACTOR, T0).unwrap();
            }
            Err(e) => {
                refused = Some(e.code);
                break;
            }
        }
        // The DO alarm: runs whatever is due now.
        while s.maintenance_due() == Some(Due::Now) {
            compactions += usize::from(s.maintain(&st).unwrap().compacted);
        }
    }
    assert_eq!(refused, Some(ErrorCode::BudgetExceeded));
    assert!(s.resident_bytes() <= SHARD_RESIDENT_CAP_BYTES);
    // Before the fix the shard stuck at ≈ 15% dead slots (≈ 21.9k live,
    // no compaction ever due). Now the dead slots were reclaimed and the
    // re-inserts went past the original row count before the real cap.
    assert!(compactions >= 1);
    assert!(s.len() > 22_000, "{} live (was {before})", s.len());
    // At the real cap nothing is left for a compaction to reclaim.
    assert_ne!(s.maintenance_due(), Some(Due::Now));
}

#[test]
fn heavy_hnsw_configs_flush_sooner_and_cap_nodes_by_load_budget() {
    let default = cfg(384, Metric::Cosine, hnsw());
    assert_eq!(flush_ops(&default), FLUSH_OPS);
    assert_eq!(flush_ops(&cfg(384, Metric::Cosine, flat())), FLUSH_OPS);
    assert_eq!(hnsw_node_cap(&cfg(384, Metric::Cosine, flat())), u64::MAX);
    let heavy = cfg(
        1536,
        Metric::Cosine,
        IndexConfig::Hnsw {
            m: 48,
            ef_construction: 200,
        },
    );
    let n = flush_ops(&heavy);
    assert!((16..FLUSH_OPS).contains(&n), "{n}");
    // Current-path replay stays well inside the 1 s lazy-load budget at the
    // measured ≈ 4 ms per insert (wasm) for this config.
    assert!((n + 64) as f64 * 4.0 < 500.0);
    let cap = hnsw_node_cap(&heavy);
    // Below the 14 MB byte cap (≈ 6.6k nodes here): the rebuild budget binds.
    assert!(
        cap < hnsw_node_cap(&default) && (4_000..6_600).contains(&cap),
        "{cap}"
    );
    // Workers Paid review: the rebuild budget stays at 20 s so no shard
    // written under it regresses to refusing every write (`413`). Lowering
    // it to 15 s would drop the 384-d m 16 cap to ≈ 13.6k nodes.
    assert_eq!(
        ruvector_edge_store::shard::maintain::REBUILD_BUDGET_MS,
        20_000.0
    );
    assert!(
        (17_500..19_000).contains(&hnsw_node_cap(&default)),
        "{}",
        hnsw_node_cap(&default)
    );
    // The write path honours the lower threshold.
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    ingest(&mut s, &st, &dm(63), &heavy, &data(130, 1536, 40)).unwrap();
    assert!(s.pending_ops() < n + 64, "{}", s.pending_ops());
    assert!(s.index_state().cur.is_some());
}

#[test]
fn filtered_hnsw_queries_are_charged_for_the_exact_fallback() {
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(64), cfg(16, Metric::Cosine, hnsw()));
    ingest(&mut s, &st, &dm, &cfg, &data(3000, 16, 41)).unwrap();
    let q = data(1, 16, 42).remove(0).1;
    let mut r = req(&q, 10);
    let plain = s.query_steps(&validate_query(&r, &cfg).unwrap());
    r.filter = Some(serde_json::json!({"g": 1}));
    let filtered = s.query_steps(&validate_query(&r, &cfg).unwrap());
    // Beam + one filter pass over every row + up to 2000 reranked rows.
    assert!(filtered >= plain + 3000 + 2000 - 1000, "{plain} {filtered}");
}
