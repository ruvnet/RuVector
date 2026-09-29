//! `QuantShard` Durable Object storage (ADR-351 §3 rv-quant, M4): the
//! SQLite schema, the `qmeta` key/value state and row I/O.
//!
//! - `qrows`: one row per vector, keyed by a shard-local `u64` row key `rk`
//!   (the `QuantShard` code key), holding the f32 originals (for rerank,
//!   fetch and rebuild), the metadata JSON and `wseq`, the write sequence
//!   that last touched it (cold-load replay reads `wseq > snapshot_seq`).
//! - `qtomb`: row keys deleted since the last snapshot (`dseq`).
//! - `qframes`: the persist v2 (`rbqx0002`) snapshot, one ≤ 1 MiB frame
//!   per row (Durable Object SQLite rows are capped at 2 MB).
//! - `qmeta`: identity, configuration, counters and usage totals, updated
//!   in the same synchronous turn as the rows (the mock SQL grammar has no
//!   `COUNT`, so totals are maintained, not computed).
//!
//! Durable Object SQLite binds at most 100 parameters per statement, so
//! every `IN (…)` list is chunked at [`MAX_BIND`].

use ruvector_edge_store::{ErrorCode, Metric, OpError, SqlStore, Value};
use std::collections::BTreeMap;

/// Largest `IN (…)` list per statement (DO SQLite: 100 bound parameters).
pub const MAX_BIND: usize = 100;

/// Tables and indexes (idempotent).
pub const SCHEMA: [&str; 6] = [
    "CREATE TABLE IF NOT EXISTS qmeta (k TEXT PRIMARY KEY, v TEXT)",
    "CREATE TABLE IF NOT EXISTS qrows (rk INTEGER PRIMARY KEY, id TEXT, f32 BLOB, md TEXT, \
     wseq INTEGER)",
    "CREATE INDEX IF NOT EXISTS qrows_id ON qrows (id)",
    "CREATE INDEX IF NOT EXISTS qrows_wseq ON qrows (wseq)",
    "CREATE TABLE IF NOT EXISTS qtomb (rk INTEGER PRIMARY KEY, dseq INTEGER)",
    "CREATE TABLE IF NOT EXISTS qframes (idx INTEGER PRIMARY KEY, bytes BLOB)",
];

const META_ALL: &str = "SELECT k, v FROM qmeta";
const META_PUT: &str = "INSERT OR REPLACE INTO qmeta (k, v) VALUES (?, ?)";
const ROW_PUT: &str = "INSERT OR REPLACE INTO qrows (rk, id, f32, md, wseq) VALUES (?, ?, ?, ?, ?)";
const ROW_DELETE: &str = "DELETE FROM qrows WHERE rk = ?";
const TOMB_PUT: &str = "INSERT OR REPLACE INTO qtomb (rk, dseq) VALUES (?, ?)";
const TOMB_SINCE: &str = "SELECT rk FROM qtomb WHERE dseq > ?";
const TOMB_PRUNE: &str = "DELETE FROM qtomb WHERE dseq <= ?";
const PAGE_ALL: &str = "SELECT rk, f32 FROM qrows WHERE rk > ? ORDER BY rk LIMIT ?";
/// Replay keys: only the `wseq` condition, so the only usable access path
/// is the `qrows_wseq` index (covering: it holds the rowid). A single
/// `WHERE wseq > ? AND rk > ? ORDER BY rk` page lets the planner pick a
/// rowid scan that reads every row — ≈ 3–4 ms at 50k × 384 with nothing to
/// replay, on top of the snapshot decode in the same cold-load turn.
pub(crate) const REPLAY_KEYS: &str = "SELECT rk FROM qrows WHERE wseq > ?";
const FRAMES_ALL: &str = "SELECT idx, bytes FROM qframes ORDER BY idx";
const FRAME_PUT: &str = "INSERT INTO qframes (idx, bytes) VALUES (?, ?)";
/// Every table's rows (the collection-drop wipe).
pub const WIPE: [&str; 4] = [
    "DELETE FROM qrows",
    "DELETE FROM qtomb",
    "DELETE FROM qframes",
    "DELETE FROM qmeta",
];
const FRAMES_CLEAR: &str = "DELETE FROM qframes";

fn corrupt(what: &'static str) -> OpError {
    OpError::new(ErrorCode::ShardUnavailable, what)
}

/// Storage failure → `503 shard_unavailable` (retryable).
pub fn io<E>(_: E) -> OpError {
    corrupt("quant shard storage")
}

/// The `qmeta` state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QMeta {
    /// Hex DO name of the identity that initialised the shard.
    pub ident: Option<String>,
    /// Dimension.
    pub dim: u32,
    /// Metric.
    pub metric: Option<Metric>,
    /// Rotation seed.
    pub seed: u64,
    /// Next row key.
    pub next_key: u64,
    /// Last write sequence.
    pub write_seq: u64,
    /// Write sequence the stored snapshot covers.
    pub snap_seq: u64,
    /// Snapshot frames stored (0: none).
    pub frames: u64,
    /// Live rows.
    pub count: u64,
    /// Stored bytes (ids + f32 + metadata).
    pub bytes: u64,
    /// Dropped (every later request is 404).
    pub wiped: bool,
}

impl QMeta {
    /// Read the whole `qmeta` table.
    pub fn read(store: &dyn SqlStore) -> Result<QMeta, OpError> {
        let mut m = QMeta::default();
        for row in store.query(META_ALL, &[]).map_err(io)? {
            let (Some(Value::Text(k)), Some(Value::Text(v))) = (row.first(), row.get(1)) else {
                return Err(corrupt("qmeta row"));
            };
            let num = || v.parse::<u64>().map_err(|_| corrupt("qmeta number"));
            match k.as_str() {
                "ident" => m.ident = Some(v.clone()),
                "dim" => m.dim = num()? as u32,
                "metric" => m.metric = Some(Metric::parse(v).ok_or(corrupt("qmeta metric"))?),
                "seed" => m.seed = num()?,
                "next_key" => m.next_key = num()?,
                "write_seq" => m.write_seq = num()?,
                "snap_seq" => m.snap_seq = num()?,
                "frames" => m.frames = num()?,
                "count" => m.count = num()?,
                "bytes" => m.bytes = num()?,
                "wiped" => m.wiped = v == "1",
                _ => {}
            }
        }
        Ok(m)
    }

    /// Write the fields that differ from `before`.
    pub fn write(&self, store: &dyn SqlStore, before: &QMeta) -> Result<(), OpError> {
        let put = |k: &str, v: String| {
            store
                .exec(META_PUT, &[k.into(), v.into()])
                .map(|_| ())
                .map_err(io)
        };
        if self.ident != before.ident {
            put("ident", self.ident.clone().unwrap_or_default())?;
            put("dim", self.dim.to_string())?;
            put("metric", self.metric.map_or("", |m| m.as_str()).to_string())?;
            put("seed", self.seed.to_string())?;
        }
        let nums = [
            ("next_key", self.next_key, before.next_key),
            ("write_seq", self.write_seq, before.write_seq),
            ("snap_seq", self.snap_seq, before.snap_seq),
            ("frames", self.frames, before.frames),
            ("count", self.count, before.count),
            ("bytes", self.bytes, before.bytes),
        ];
        for (k, now, was) in nums {
            if now != was {
                put(k, now.to_string())?;
            }
        }
        Ok(())
    }
}

/// Create the tables.
pub fn ensure_schema(store: &dyn SqlStore) -> Result<(), OpError> {
    for sql in SCHEMA {
        store.exec(sql, &[]).map_err(io)?;
    }
    Ok(())
}

/// Little-endian f32 blob.
pub fn encode_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Inverse of [`encode_f32`].
pub fn decode_f32(b: &[u8]) -> Result<Vec<f32>, OpError> {
    if b.len() % 4 != 0 {
        return Err(corrupt("qrows f32"));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn int(v: Option<&Value>) -> Result<u64, OpError> {
    match v {
        Some(Value::Int(i)) if *i >= 0 => Ok(*i as u64),
        _ => Err(corrupt("qrows key")),
    }
}

fn blob(v: Option<&Value>) -> Result<&[u8], OpError> {
    match v {
        Some(Value::Blob(b)) => Ok(b),
        _ => Err(corrupt("qrows blob")),
    }
}

fn text(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::Text(s)) => Some(s),
        _ => None,
    }
}

fn in_list(n: usize) -> String {
    let mut s = "?, ".repeat(n);
    s.truncate(s.len().saturating_sub(2));
    s
}

/// A stored row's identity and size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowRef {
    /// Row key.
    pub rk: u64,
    /// Stored bytes (id + f32 + metadata).
    pub bytes: u64,
}

/// Stored bytes of a row.
pub fn row_bytes(id: &str, dim: u32, md: Option<&str>) -> u64 {
    id.len() as u64 + 4 * u64::from(dim) + md.map_or(0, |m| m.len() as u64)
}

/// Rows by vector id (absent ids are missing from the map).
pub fn refs_by_ids(
    store: &dyn SqlStore,
    ids: &[&str],
    dim: u32,
) -> Result<BTreeMap<String, RowRef>, OpError> {
    let mut out = BTreeMap::new();
    for part in ids.chunks(MAX_BIND) {
        let sql = format!(
            "SELECT rk, id, md FROM qrows WHERE id IN ({})",
            in_list(part.len())
        );
        let params: Vec<Value> = part.iter().map(|i| Value::from(*i)).collect();
        for r in store.query(&sql, &params).map_err(io)? {
            let id = text(r.get(1)).ok_or(corrupt("qrows id"))?.to_string();
            let bytes = row_bytes(&id, dim, text(r.get(2)));
            out.insert(
                id,
                RowRef {
                    rk: int(r.first())?,
                    bytes,
                },
            );
        }
    }
    Ok(out)
}

/// One full row.
#[derive(Debug, Clone, PartialEq)]
pub struct FullRow {
    /// Row key.
    pub rk: u64,
    /// Vector id.
    pub id: String,
    /// Values.
    pub values: Vec<f32>,
    /// Metadata JSON text.
    pub md: Option<String>,
}

/// Full rows by `col IN (…)` (`col` is `rk` or `id`).
fn full_rows(store: &dyn SqlStore, col: &str, keys: &[Value]) -> Result<Vec<FullRow>, OpError> {
    let mut out = Vec::with_capacity(keys.len());
    for part in keys.chunks(MAX_BIND) {
        let sql = format!(
            "SELECT rk, id, f32, md FROM qrows WHERE {col} IN ({})",
            in_list(part.len())
        );
        for r in store.query(&sql, part).map_err(io)? {
            out.push(FullRow {
                rk: int(r.first())?,
                id: text(r.get(1)).ok_or(corrupt("qrows id"))?.to_string(),
                values: decode_f32(blob(r.get(2))?)?,
                md: text(r.get(3)).map(str::to_string),
            });
        }
    }
    Ok(out)
}

/// Full rows by row key.
pub fn rows_by_rks(store: &dyn SqlStore, rks: &[u64]) -> Result<Vec<FullRow>, OpError> {
    let keys: Vec<Value> = rks.iter().map(|k| Value::Int(*k as i64)).collect();
    full_rows(store, "rk", &keys)
}

/// Full rows by vector id.
pub fn rows_by_ids(store: &dyn SqlStore, ids: &[String]) -> Result<Vec<FullRow>, OpError> {
    let keys: Vec<Value> = ids.iter().map(|i| Value::from(i.as_str())).collect();
    full_rows(store, "id", &keys)
}

/// Insert or replace one row.
pub fn put_row(
    store: &dyn SqlStore,
    rk: u64,
    id: &str,
    values: &[f32],
    md: Option<&str>,
    wseq: u64,
) -> Result<(), OpError> {
    let md = md.map_or(Value::Null, Value::from);
    let params = [
        Value::Int(rk as i64),
        id.into(),
        Value::Blob(encode_f32(values)),
        md,
        Value::Int(wseq as i64),
    ];
    store.exec(ROW_PUT, &params).map(|_| ()).map_err(io)
}

/// Delete one row and record its tombstone.
pub fn delete_row(store: &dyn SqlStore, rk: u64, dseq: u64) -> Result<(), OpError> {
    let k = Value::Int(rk as i64);
    store
        .exec(ROW_DELETE, std::slice::from_ref(&k))
        .map_err(io)?;
    store
        .exec(TOMB_PUT, &[k, Value::Int(dseq as i64)])
        .map(|_| ())
        .map_err(io)
}

/// Row keys deleted after `seq`.
pub fn tombs_since(store: &dyn SqlStore, seq: u64) -> Result<Vec<u64>, OpError> {
    store
        .query(TOMB_SINCE, &[Value::Int(seq as i64)])
        .map_err(io)?
        .iter()
        .map(|r| int(r.first()))
        .collect()
}

/// Up to `limit` `(rk, values)` with `rk > after`, ascending; only rows
/// written after `since` when it is set (snapshot replay).
pub fn page(
    store: &dyn SqlStore,
    since: Option<u64>,
    after: u64,
    limit: usize,
) -> Result<Vec<(u64, Vec<f32>)>, OpError> {
    if let Some(s) = since {
        // Replay rows are few (the alarm flushes urgently past FLUSH_ROWS):
        // select their keys from the index, page in memory, then read only
        // the page's f32 originals.
        let mut rks: Vec<u64> = store
            .query(REPLAY_KEYS, &[Value::Int(s as i64)])
            .map_err(io)?
            .iter()
            .map(|r| int(r.first()))
            .collect::<Result<_, _>>()?;
        rks.retain(|&k| k > after);
        rks.sort_unstable();
        rks.truncate(limit);
        let mut rows: Vec<(u64, Vec<f32>)> = rows_by_rks(store, &rks)?
            .into_iter()
            .map(|r| (r.rk, r.values))
            .collect();
        rows.sort_unstable_by_key(|r| r.0);
        return Ok(rows);
    }
    let (after, limit) = (Value::Int(after as i64), Value::Int(limit as i64));
    let rows = store.query(PAGE_ALL, &[after, limit]).map_err(io)?;
    rows.iter()
        .map(|r| Ok((int(r.first())?, decode_f32(blob(r.get(1))?)?)))
        .collect()
}

/// The stored snapshot frames, in order.
pub fn frames(store: &dyn SqlStore) -> Result<Vec<Vec<u8>>, OpError> {
    store
        .query(FRAMES_ALL, &[])
        .map_err(io)?
        .into_iter()
        .map(|mut r| match r.pop() {
            Some(Value::Blob(b)) => Ok(b),
            _ => Err(corrupt("qframes")),
        })
        .collect()
}

/// Replace the snapshot and prune tombstones it covers.
pub fn put_frames(store: &dyn SqlStore, frames: &[Vec<u8>], covers: u64) -> Result<(), OpError> {
    store.exec(FRAMES_CLEAR, &[]).map_err(io)?;
    for (i, f) in frames.iter().enumerate() {
        store
            .exec(FRAME_PUT, &[Value::Int(i as i64), Value::Blob(f.clone())])
            .map_err(io)?;
    }
    store
        .exec(TOMB_PRUNE, &[Value::Int(covers as i64)])
        .map(|_| ())
        .map_err(io)
}

/// Delete every row, leaving only `wiped`.
pub fn wipe(store: &dyn SqlStore) -> Result<(), OpError> {
    for sql in WIPE {
        store.exec(sql, &[]).map_err(io)?;
    }
    store
        .exec(META_PUT, &["wiped".into(), "1".into()])
        .map(|_| ())
        .map_err(io)
}
