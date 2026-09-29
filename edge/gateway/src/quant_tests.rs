//! M4 rv-quant at the Durable Object core (ADR-351 §15 M4): a 50k × 384
//! `rabitq` shard over real SQLite — upsert in 500-row turns, snapshot
//! flush on the alarm, cold load in a fresh isolate within the CPU and
//! memory budgets, recall through the shard path — plus the chunked
//! rebuild (`503` until done) and `413` budget refusals.
//!
//! Run with `--nocapture` for the measured costs.

use crate::quant_shard::{self, QuantHost};
use crate::sqlite_mem::SqliteStore;
use crate::testkit::{tenant, Rng, T0};
use crate::wire::{ActorWire, CfgWire, ShardCall, ShardOut, ShardRequest};
use ruvector_edge_quant::budget;
use ruvector_edge_store::shard::SHARD_RESIDENT_CAP_BYTES;
use ruvector_edge_store::{
    ErrorCode, IndexConfig, Metric, OpError, QueryRequest, SqlStore, UpsertRow,
};
use std::time::Instant;

const UID: &str = "00112233445566778899aabbccddeeff";

pub fn cfg(dim: u32) -> CfgWire {
    CfgWire {
        dim,
        metric: Metric::L2,
        filterable_keys: vec![],
        index: IndexConfig::Rabitq,
    }
}

pub fn call(h: &mut QuantHost, st: &dyn SqlStore, c: ShardCall) -> Result<ShardOut, OpError> {
    let req = ShardRequest {
        tenant_key: tenant("org-q").as_str().to_string(),
        uid: UID.into(),
        shard: 0,
        call: c,
    };
    quant_shard::handle(h, "q0", None, st, req)
}

/// Standard normal samples (Box–Muller over the test RNG).
fn gauss(rng: &mut Rng, dim: usize) -> Vec<f32> {
    let mut u = || ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (0..dim)
        .map(|_| ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32)
        .collect()
}

/// `n` points around 100 N(0, 1) centres with N(0, 0.5²) noise (the quant
/// crate's recall fixture: clustered, like real embeddings).
pub fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    let centres: Vec<Vec<f32>> = (0..100).map(|_| gauss(&mut rng, dim)).collect();
    (0..n)
        .map(|_| {
            let c = &centres[(rng.next_u64() % 100) as usize];
            let noise = gauss(&mut rng, dim);
            c.iter().zip(noise).map(|(a, b)| a + 0.5 * b).collect()
        })
        .collect()
}

pub fn brute_top10(data: &[Vec<f32>], q: &[f32]) -> Vec<String> {
    let mut d: Vec<(f64, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let s: f64 = v.iter().zip(q).map(|(a, b)| f64::from(a - b).powi(2)).sum();
            (s, i)
        })
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.iter().take(10).map(|(_, i)| format!("v{i}")).collect()
}

pub fn upsert(
    h: &mut QuantHost,
    st: &dyn SqlStore,
    dim: u32,
    rows: Vec<UpsertRow>,
) -> Result<ShardOut, OpError> {
    let delta = match call(
        h,
        st,
        ShardCall::Plan {
            cfg: cfg(dim),
            rows: rows.clone(),
        },
    )? {
        ShardOut::Planned { delta } => delta,
        o => panic!("{o:?}"),
    };
    let actor = ActorWire {
        sub: "s".into(),
        jti: "j".into(),
        family_id: "f".into(),
        act_sub: None,
    };
    let apply = ShardCall::Apply {
        cfg: cfg(dim),
        rows,
        admitted: delta,
        actor,
        now: T0,
    };
    call(h, st, apply)
}

pub fn rows(data: &[Vec<f32>], from: usize) -> Vec<UpsertRow> {
    data.iter()
        .enumerate()
        .map(|(i, v)| UpsertRow {
            id: format!("v{}", from + i),
            values: v.clone(),
            metadata: None,
        })
        .collect()
}

pub fn query(
    h: &mut QuantHost,
    st: &dyn SqlStore,
    dim: u32,
    q: &[f32],
) -> Result<Vec<String>, OpError> {
    let req = QueryRequest {
        vector: q.to_vec(),
        top_k: 10,
        filter: None,
        include: vec![],
        ef: None,
        rerank: None,
    };
    match call(
        h,
        st,
        ShardCall::Query {
            cfg: cfg(dim),
            req,
            steps_before: 0,
        },
    )? {
        ShardOut::Matches { matches, .. } => Ok(matches.into_iter().map(|m| m.id).collect()),
        o => panic!("{o:?}"),
    }
}

pub fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn rabitq_50k_x_384_cold_load_fits_cpu_and_memory() {
    const N: usize = 50_000;
    const DIM: u32 = 384;
    let st = SqliteStore::default();
    let mut h = QuantHost::default();
    // Held-out queries from the same mixture.
    let mut data = clustered(N + 31, DIM as usize, 0x50_384);
    let qs = data.split_off(N);
    let mut upsert_ms = Vec::new();
    for (b, chunk) in data.chunks(500).enumerate() {
        let t = Instant::now();
        let out = upsert(&mut h, &st, DIM, rows(chunk, b * 500)).unwrap();
        upsert_ms.push(ms(t));
        assert!(
            matches!(out, ShardOut::Written { count: 500, .. }),
            "{out:?}"
        );
    }
    // Snapshot flush on the alarm (persist v2 frames), then nothing due.
    assert!(quant_shard::next_alarm(&h, "q0").is_some());
    let t = Instant::now();
    assert_eq!(quant_shard::alarm(&mut h, "q0", &st), None);
    let flush_ms = ms(t);
    let frames = st.query("SELECT idx FROM qframes", &[]).unwrap().len();
    // The decode alone (snapshot frames → resident codes), without SQL.
    let meta = crate::quant_store::QMeta::read(&st).unwrap();
    let t = Instant::now();
    let (rb, _) = crate::quant_load::open(&st, &meta, quant_shard::edge_budget()).unwrap();
    let open_ms = ms(t);
    drop(rb);

    // A fresh isolate: the first query cold-loads the snapshot (no row is
    // re-encoded) within the load budget and the resident cap. That load
    // is the whole turn (`503`, retry); the retry is served warm.
    let mut cold = QuantHost::default();
    let q = qs[30].clone();
    let t = Instant::now();
    let e = query(&mut cold, &st, DIM, &q).unwrap_err();
    let cold_ms = ms(t);
    assert_eq!(e.code, ErrorCode::ShardUnavailable);
    let t = Instant::now();
    let got = query(&mut cold, &st, DIM, &q).unwrap();
    let first_ms = ms(t);
    let load = cold.last_load.unwrap();
    let resident = cold.resident_bytes("q0");
    let b = quant_shard::edge_budget();
    assert_eq!(load.snapshot_rows, N as u64);
    assert!(!load.corrupt);
    assert!(load.load_units <= b.max_load_units, "{load:?}");
    assert!(resident <= SHARD_RESIDENT_CAP_BYTES, "{resident}");
    let truth = brute_top10(&data, &q);
    assert!(
        got.iter().filter(|id| truth.contains(id)).count() >= 9,
        "{got:?}"
    );

    // Warm queries: recall@10 through the shard path.
    let (mut hits, mut warm) = (0, 0.0);
    for q in &qs[..30] {
        let truth = brute_top10(&data, q);
        let t = Instant::now();
        let got = query(&mut cold, &st, DIM, q).unwrap();
        warm += ms(t);
        hits += got.iter().filter(|id| truth.contains(id)).count();
    }
    let warm_ms = warm / 30.0;
    let recall = hits as f64 / 300.0;
    let mean = upsert_ms.iter().sum::<f64>() / upsert_ms.len() as f64;
    let max = upsert_ms.iter().copied().fold(0.0, f64::max);
    let units = budget::query_units(N as u64, DIM as usize, crate::quant_load::ROTATION, 500);
    eprintln!(
        "rabitq 50k x 384 (native; cores opt 3 in test, opt z in --release): upsert-500 turn mean {mean:.1} ms / max {max:.1} ms \
         ({} units encode); flush {flush_ms:.1} ms ({frames} frames, {} B); cold load \
         turn {cold_ms:.1} ms (snapshot read + decode {open_ms:.1} ms, {} of {} units), then first query \
         {first_ms:.2} ms; warm query {warm_ms:.2} ms ({units} units); resident {resident} B; recall@10 {recall:.3}",
        budget::encode_units(500, DIM as usize, crate::quant_load::ROTATION),
        load.snapshot_bytes,
        load.load_units,
        b.max_load_units,
    );
    assert!(recall >= 0.95, "recall@10 {recall}");
}

#[test]
fn rebuild_without_snapshot_runs_in_turns_then_serves() {
    const DIM: u32 = 384;
    let st = SqliteStore::default();
    let mut h = QuantHost::default();
    let data = clustered(1_000, DIM as usize, 7);
    for (b, chunk) in data.chunks(500).enumerate() {
        upsert(&mut h, &st, DIM, rows(chunk, b * 500)).unwrap();
    }
    // The isolate dies before the flush: a fresh one must re-encode every
    // row, ≤ TURN_UNITS per turn, answering 503 until done.
    let mut cold = QuantHost::default();
    let per_turn = crate::quant_load::rows_per_turn(DIM as usize, crate::quant_load::TURN_UNITS);
    let mut turns = 0;
    loop {
        turns += 1;
        assert!(turns < 10);
        match query(&mut cold, &st, DIM, &data[3]) {
            Ok(ids) => {
                assert_eq!(ids[0], "v3");
                break;
            }
            Err(e) => {
                assert_eq!(e.code, ErrorCode::ShardUnavailable);
                // The alarm is armed urgently while the rebuild runs (and
                // for the flush once a load-only turn finished it).
                let urgent = Some(crate::shard_core::URGENT_ALARM_MS);
                if cold.is_ready("q0") {
                    assert!(quant_shard::next_alarm(&cold, "q0").is_some());
                } else {
                    assert_eq!(quant_shard::next_alarm(&cold, "q0"), urgent);
                }
            }
        }
    }
    // One more `503` when the finishing turn itself did a load turn's
    // worth of encoding (it then only loads; the retry is served).
    let n = 1_000usize.div_ceil(per_turn);
    let last = (1_000 - (n - 1) * per_turn) as u64;
    let last_units = budget::encode_units(last, DIM as usize, crate::quant_load::ROTATION);
    let extra = usize::from(last_units >= quant_shard::LOAD_TURN_UNITS);
    assert_eq!(turns, n + extra, "per turn {per_turn}");
    // Rebuilt rows are unflushed: the alarm writes the snapshot.
    assert!(quant_shard::next_alarm(&cold, "q0").is_some());
    quant_shard::alarm(&mut cold, "q0", &st);
    let mut again = QuantHost::default();
    assert_eq!(query(&mut again, &st, DIM, &data[3]).unwrap()[0], "v3");
    assert_eq!(again.last_load.unwrap().snapshot_rows, 1_000);
}

#[test]
fn budget_refusals_are_413_and_write_nothing() {
    let st = SqliteStore::default();
    let mut h = QuantHost::default();
    h.budget.max_vectors = 10;
    let data = clustered(11, 16, 9);
    let e = upsert(&mut h, &st, 16, rows(&data, 0)).unwrap_err();
    assert_eq!((e.code, e.code.status()), (ErrorCode::BudgetExceeded, 413));
    let n = st.query("SELECT rk FROM qrows", &[]).unwrap().len();
    assert_eq!(n, 0);
    // The fan-out step budget, before any scan.
    upsert(&mut h, &st, 16, rows(&data[..5], 0)).unwrap();
    let req = QueryRequest {
        vector: data[0].clone(),
        top_k: 1,
        filter: None,
        include: vec![],
        ef: None,
        rerank: None,
    };
    let big = ShardCall::Query {
        cfg: cfg(16),
        req: req.clone(),
        steps_before: ruvector_edge_store::shard::MAX_QUERY_STEPS,
    };
    assert_eq!(call(&mut h, &st, big).unwrap_err().code.status(), 413);
    // A filter has no index on the quant path: 400.
    let filtered = QueryRequest {
        filter: Some(serde_json::json!({ "k": "v" })),
        ..req
    };
    let q = ShardCall::Query {
        cfg: cfg(16),
        req: filtered,
        steps_before: 0,
    };
    assert_eq!(
        call(&mut h, &st, q).unwrap_err().code,
        ErrorCode::InvalidRequest
    );
}
