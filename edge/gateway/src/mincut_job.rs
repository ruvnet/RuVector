//! `AnalyticsJob` Durable Object core (ADR-351 §3 rv-mincut, M4): one
//! async min-cut job per DO (`idFromName(do_name(tenant, mincut,
//! uid(job_id), 0))`).
//!
//! The request path only stores the input and arms the alarm (O(input)
//! copy; the gateway hands job-sized edge lists over as raw JSON text).
//! The work runs in alarm turns, each persisted before the next begins:
//!
//! 1. **admit**: parse and canonicalise the edge list, encode it in the
//!    analytics crate's persisted format (versioned, sha256-checksummed,
//!    ≤ 32k-edge chunks, one row each), decode it back, plan it against
//!    [`mc::EDGE_JOB`] and `JobDescriptor::submit` (the crate's
//!    `Profile::JOB`); a `413` here is a `failed` job with
//!    `budget_exceeded` / `payload_too_large`;
//! 2. **run**: `start` (the attempt), then `run` on the graph decoded from
//!    the pinned snapshot (digest-checked). The Worker shell awaits storage
//!    I/O between the two ([`prepare`] / [`solve`]), so the attempt is
//!    committed before the solve starts: an isolate killed for CPU mid-solve
//!    leaves `running`, and the next alarm restarts it (at most
//!    `MAX_ATTEMPTS`, then `failed` with `413 budget_exceeded` — on Workers
//!    Free, 10 ms per alarm, every job whose solve needs more ends this
//!    way, e.g. the 51k-edge K_320 acceptance job, ≈ 84 ms release native;
//!    it completes on Paid). Should even that write be lost, a job
//!    without progress for [`STALL_MS`] is failed on read (`413`
//!    `budget_exceeded`) instead of staying `queued` / `running` forever.
//!
//! The solve is one indivisible step (Stoer–Wagner is not resumable); its
//! cost is the analytics estimate the job was admitted with, against
//! [`mc::EDGE_JOB`] (the crate's job profile with the isolate-headroom
//! memory budget, `mincut_core`).
//!
//! **Retention**: a job deletes its storage [`JOB_TTL_MS`] after submit,
//! whatever its state (the alarm of every terminal turn is armed for that
//! instant; [`prepare`] answers [`Turn::Expire`]). The tenant catalog
//! counts the job as live until the same instant (`graph_catalog`).

use crate::graph_wire::{job_graph_uid, JobCall, JobRequest, JobView};
use crate::mincut_core::{self as mc, error_json};
use crate::shard_core::URGENT_ALARM_MS;
use crate::wire::WireErr;
use ruvector_edge_analytics::format::{decode_graph, encode_graph, Manifest, MAX_CHUNK_EDGES};
use ruvector_edge_analytics::job::MAX_ATTEMPTS;
use ruvector_edge_analytics::{
    plan, AnalyticsError, GraphLimits, JobDescriptor, JobState, QueryMode, TenantGraph,
};
use ruvector_edge_store::{ErrorCode, OpError, SqlStore, Value};
use ruvector_edge_tenancy::TenantKey;
use serde_json::{json, Value as Json};

const SCHEMA: [&str; 3] = [
    "CREATE TABLE IF NOT EXISTS jmeta (k TEXT PRIMARY KEY, v TEXT)",
    "CREATE TABLE IF NOT EXISTS jchunks (idx INTEGER PRIMARY KEY, bytes BLOB)",
    "CREATE TABLE IF NOT EXISTS jblob (k TEXT, idx INTEGER, bytes BLOB, PRIMARY KEY (k, idx))",
];
const BLOB_PUT: &str = "INSERT OR REPLACE INTO jblob (k, idx, bytes) VALUES (?, ?, ?)";
const BLOB_GET: &str = "SELECT idx, bytes FROM jblob WHERE k = ? ORDER BY idx";
/// Largest stored row (DO SQLite caps rows at 2 MB).
const BLOB_CHUNK: usize = 1 << 20;
const META_ALL: &str = "SELECT k, v FROM jmeta";
const META_PUT: &str = "INSERT OR REPLACE INTO jmeta (k, v) VALUES (?, ?)";
const CHUNKS_ALL: &str = "SELECT idx, bytes FROM jchunks ORDER BY idx";
const CHUNK_PUT: &str = "INSERT OR REPLACE INTO jchunks (idx, bytes) VALUES (?, ?)";
const WIPE: [&str; 3] = [
    "DELETE FROM jchunks",
    "DELETE FROM jblob",
    "DELETE FROM jmeta",
];

/// How long a job (input, snapshot, result) is kept after submit.
pub const JOB_TTL_MS: u64 = 24 * 60 * 60 * 1000;

fn io<E>(_: E) -> OpError {
    OpError::new(ErrorCode::ShardUnavailable, "job storage")
}

struct Meta(Vec<(String, String)>);

impl Meta {
    /// Read `jmeta`, creating the tables only when `create` (a submit):
    /// a status probe of an unknown job id never initialises storage, and a
    /// DO without tables is simply absent (`404`).
    fn read_as(store: &dyn SqlStore, create: bool) -> Result<Meta, OpError> {
        if create {
            for sql in SCHEMA {
                store.exec(sql, &[]).map_err(io)?;
            }
        }
        let rows = match store.query(META_ALL, &[]) {
            Ok(rows) => rows,
            Err(_) if !create => return Err(OpError::not_found()),
            Err(e) => return Err(io(e)),
        };
        Ok(Meta(
            rows.into_iter()
                .filter_map(|r| match (r.first(), r.get(1)) {
                    (Some(Value::Text(k)), Some(Value::Text(v))) => Some((k.clone(), v.clone())),
                    _ => None,
                })
                .collect(),
        ))
    }
    fn read(store: &dyn SqlStore) -> Result<Meta, OpError> {
        Self::read_as(store, false)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.as_str())
    }
    fn json<T: serde::de::DeserializeOwned>(&self, k: &str) -> Option<T> {
        self.get(k).and_then(|v| serde_json::from_str(v).ok())
    }
}

fn put(store: &dyn SqlStore, k: &str, v: &str) -> Result<(), OpError> {
    store
        .exec(META_PUT, &[k.into(), v.into()])
        .map(|_| ())
        .map_err(io)
}

/// Store a large value (the job input, the labels) in ≤ 1 MiB rows.
fn put_blob(store: &dyn SqlStore, k: &str, v: &[u8]) -> Result<(), OpError> {
    for (i, c) in v.chunks(BLOB_CHUNK).enumerate() {
        let params = [k.into(), Value::Int(i as i64), Value::Blob(c.to_vec())];
        store.exec(BLOB_PUT, &params).map_err(io)?;
    }
    Ok(())
}

fn get_blob(store: &dyn SqlStore, k: &str) -> Result<Vec<u8>, OpError> {
    let mut out = Vec::new();
    for r in store.query(BLOB_GET, &[k.into()]).map_err(io)? {
        if let Some(Value::Blob(b)) = r.get(1) {
            out.extend_from_slice(b);
        }
    }
    Ok(out)
}

fn put_json<T: serde::Serialize>(store: &dyn SqlStore, k: &str, v: &T) -> Result<(), OpError> {
    put(store, k, &serde_json::to_string(v).map_err(io)?)
}

/// Serve one encoded [`JobRequest`]; the reply and the alarm delay to arm.
pub fn serve(own_name: Option<&str>, store: &dyn SqlStore, body: &[u8]) -> (String, Option<u64>) {
    let (reply, alarm) = match serde_json::from_slice::<JobRequest>(body) {
        Ok(req) => match handle(own_name, store, req) {
            Ok((v, a)) => (Ok(v), a),
            Err(e) => (Err(WireErr::from_op(&e)), None),
        },
        Err(_) => (
            Err(WireErr::from_op(&OpError::invalid("malformed job call"))),
            None,
        ),
    };
    let text: Result<JobView, WireErr> = reply;
    let s = serde_json::to_string(&text)
        .unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#));
    (s, alarm)
}

fn handle(
    own_name: Option<&str>,
    store: &dyn SqlStore,
    req: JobRequest,
) -> Result<(JobView, Option<u64>), OpError> {
    let tenant = TenantKey::parse(&req.tenant_key).map_err(|_| OpError::invalid("tenant"))?;
    let expected = crate::graph_wire::job_do_name(&tenant, &req.job_id);
    if own_name.is_some_and(|n| n != expected.as_str()) {
        return Err(OpError::not_found());
    }
    let submit = matches!(req.call, JobCall::Submit { .. });
    let m = Meta::read_as(store, submit)?;
    if m.get("ident").is_some_and(|i| i != expected.as_str()) {
        return Err(OpError::not_found());
    }
    match req.call {
        JobCall::Get { .. } if m.get("ident").is_none() => Err(OpError::not_found()),
        JobCall::Get { now_ms } => {
            if stalled(&m, now_ms) {
                fail(store, &cpu_limit())?;
                let m = Meta::read(store)?;
                return Ok((
                    JobView {
                        view: view(store, &m),
                    },
                    retain(&m, now_ms),
                ));
            }
            Ok((
                JobView {
                    view: view(store, &m),
                },
                None,
            ))
        }
        JobCall::Submit { .. } if m.get("ident").is_some() => {
            Err(OpError::new(ErrorCode::Conflict, "job id exists"))
        }
        JobCall::Submit {
            mode,
            edges,
            labels,
            graph,
            revision,
            now_ms,
        } => {
            put(store, "ident", expected.as_str())?;
            put(store, "job_id", &req.job_id)?;
            put_blob(store, "input", edges.as_bytes())?;
            put_json(store, "mode", &mode)?;
            put_blob(store, "labels", &serde_json::to_vec(&labels).map_err(io)?)?;
            put_json(store, "graph", &graph)?;
            put(store, "revision", &revision.to_string())?;
            put(store, "submitted_ms", &now_ms.to_string())?;
            put(store, "expires_ms", &expires_at(now_ms).to_string())?;
            let m = Meta::read(store)?;
            Ok((
                JobView {
                    view: view(store, &m),
                },
                Some(1),
            ))
        }
    }
}

/// Expiry instant of a job submitted at `submitted_ms`.
pub fn expires_at(submitted_ms: u64) -> u64 {
    submitted_ms.saturating_add(JOB_TTL_MS)
}

fn expiry(m: &Meta) -> u64 {
    m.get("expires_ms")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            expires_at(
                m.get("submitted_ms")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
            )
        })
}

/// The alarm delay of a finished job: until its expiry (≥ 1 ms).
fn retain(m: &Meta, now_ms: u64) -> Option<u64> {
    Some(expiry(m).saturating_sub(now_ms).max(1))
}

/// Erase the job's rows (retention); the Worker shell then also calls
/// `deleteAll`. A later `Get` finds no identity: `404`.
pub fn expire(store: &dyn SqlStore) -> Result<(), OpError> {
    for sql in WIPE {
        store.exec(sql, &[]).map_err(io)?;
    }
    Ok(())
}

/// A non-terminal job whose last recorded progress is older than this is
/// failed on read. A solve killed for CPU discards the turn's writes
/// (including its `running` attempt), and the runtime's own alarm retries
/// (≤ 6, exponential backoff from 2 s) then stop, so without this such a
/// job would stay `queued` / `running` forever. 20 min > the 15 min alarm
/// wall limit plus the retry backoff.
pub const STALL_MS: u64 = 20 * 60 * 1000;

/// The failure of a job the isolate could not finish (`413`).
fn cpu_limit() -> OpError {
    OpError::new(
        ErrorCode::BudgetExceeded,
        "job made no progress: its solve exceeds the isolate CPU limit",
    )
}

fn stalled(m: &Meta, now_ms: u64) -> bool {
    if now_ms == 0 || m.get("error").is_some() {
        return false;
    }
    let submitted: u64 = m
        .get("submitted_ms")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let last = match m.json::<JobDescriptor>("desc") {
        Some(d) if d.is_terminal() => return false,
        Some(d) => d.updated_at_ms,
        None => submitted,
    };
    now_ms.saturating_sub(last) > STALL_MS
}

/// The public job view.
fn view(store: &dyn SqlStore, m: &Meta) -> Json {
    let submitted: u64 = m
        .get("submitted_ms")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut v = json!({
        "job_id": m.get("job_id"),
        "graph": m.json::<Option<String>>("graph").flatten(),
        "submitted_at_ms": submitted,
        "updated_at_ms": submitted,
        "state": "queued",
    });
    if let Some(err) = m.json::<Json>("error") {
        v["state"] = json!("failed");
        v["error"] = err;
        return v;
    }
    let Some(d) = m.json::<JobDescriptor>("desc") else {
        return v;
    };
    v["updated_at_ms"] = json!(d.updated_at_ms);
    v["estimate"] = serde_json::to_value(d.estimate).unwrap_or(Json::Null);
    v["snapshot_digest"] = json!(d.snapshot_digest);
    match &d.state {
        JobState::Queued => {}
        JobState::Running { attempt } => {
            v["state"] = json!("running");
            v["attempt"] = json!(attempt);
        }
        JobState::Done { report } => {
            // Only a finished job needs the (possibly multi-MB) labels.
            let labels: Option<Vec<String>> = get_blob(store, "labels")
                .ok()
                .and_then(|b| serde_json::from_slice::<Option<Vec<String>>>(&b).ok())
                .flatten();
            v["state"] = json!("done");
            v["result"] = mc::report_json(report, &d.mode, labels.as_deref());
        }
        JobState::Failed { code, detail } => {
            v["state"] = json!("failed");
            v["error"] = error_json(*code, detail);
        }
    }
    v
}

fn fail(store: &dyn SqlStore, e: &OpError) -> Result<(), OpError> {
    put_json(store, "error", &error_json(e.code, e.detail))
}

fn aerr(e: AnalyticsError) -> OpError {
    mc::aerr(e)
}

/// Load the pinned snapshot (row 0 = manifest, then the chunks).
fn load_graph(store: &dyn SqlStore) -> Result<TenantGraph, AnalyticsError> {
    let corrupt = || AnalyticsError::Corrupt(ruvector_edge_analytics::CorruptKind::Truncated);
    let rows = store.query(CHUNKS_ALL, &[]).map_err(|_| corrupt())?;
    let blobs: Vec<Vec<u8>> = rows
        .into_iter()
        .filter_map(|mut r| match r.pop() {
            Some(Value::Blob(b)) => Some(b),
            _ => None,
        })
        .collect();
    let (manifest, chunks) = blobs.split_first().ok_or_else(corrupt)?;
    let m = Manifest::decode(manifest)?;
    decode_graph(&m, chunks.iter().map(Vec::as_slice), &GraphLimits::JOB)
}

/// Admission turn: canonicalise, persist the snapshot, pin the descriptor.
fn admit(store: &dyn SqlStore, m: &Meta, now_ms: u64) -> Result<JobDescriptor, OpError> {
    let job_id = m.get("job_id").unwrap_or_default();
    let mode: QueryMode = m.json("mode").ok_or(io(()))?;
    let revision: u64 = m.get("revision").and_then(|v| v.parse().ok()).unwrap_or(0);
    let input = String::from_utf8(get_blob(store, "input")?)
        .map_err(|_| OpError::invalid("edges must be UTF-8 JSON"))?;
    // Checked before building anything (the request path checked it too).
    mc::job_memory_admissible(mc::raw_edge_count(&input))?;
    let edges = mc::parse_edges(&input)?;
    drop(input);
    let g = TenantGraph::from_edges(job_graph_uid(job_id), revision, &edges, &GraphLimits::JOB)
        .map_err(aerr)?;
    drop(edges);
    let enc = encode_graph(&g, MAX_CHUNK_EDGES).map_err(aerr)?;
    drop(g);
    store
        .exec(CHUNK_PUT, &[Value::Int(0), Value::Blob(enc.manifest)])
        .map_err(io)?;
    for (i, c) in enc.chunks.into_iter().enumerate() {
        store
            .exec(CHUNK_PUT, &[Value::Int(i as i64 + 1), Value::Blob(c)])
            .map_err(io)?;
    }
    let pinned = load_graph(store).map_err(aerr)?;
    // The isolate-headroom budget first; `submit` then re-plans against
    // the crate's (looser) job profile and pins the snapshot digest.
    plan(&pinned, &mode, &mc::EDGE_JOB).map_err(aerr)?;
    JobDescriptor::submit(job_id, &pinned, mode, now_ms).map_err(aerr)
}

/// What an alarm turn does after [`prepare`].
pub enum Turn {
    /// Nothing more this turn; the next alarm delay (`None`: finished).
    Done(Option<u64>),
    /// The attempt was recorded: solve it ([`solve`]) after the shell has
    /// awaited storage I/O (committing the attempt).
    Solve(Box<JobDescriptor>),
    /// The job reached its expiry: [`expire`] it (and `deleteAll`).
    Expire,
}

/// First half of an alarm turn: admission, or `start` of the next attempt
/// (persisted).
pub fn prepare(store: &dyn SqlStore, now_ms: u64) -> Turn {
    let Ok(m) = Meta::read(store) else {
        return Turn::Done(None);
    };
    if m.get("ident").is_none() {
        return Turn::Done(None);
    }
    // Before every other state: a failed or stuck job expires too.
    if now_ms >= expiry(&m) {
        return Turn::Expire;
    }
    let keep = retain(&m, now_ms);
    if m.get("error").is_some() {
        return Turn::Done(keep);
    }
    let Some(mut d) = m.json::<JobDescriptor>("desc") else {
        return Turn::Done(match admit(store, &m, now_ms) {
            Ok(d) => put_json(store, "desc", &d).ok().map(|_| URGENT_ALARM_MS),
            Err(e) => {
                let _ = fail(store, &e);
                keep
            }
        });
    };
    if d.is_terminal() {
        return Turn::Done(keep);
    }
    if matches!(d.state, JobState::Running { attempt } if attempt >= MAX_ATTEMPTS) {
        // Every attempt died mid-solve: the isolate was killed (CPU or
        // memory), i.e. the job exceeds what one alarm may spend — a
        // budget failure (`413`), not the crate's `JobState` conflict.
        let _ = fail(store, &cpu_limit());
        return Turn::Done(keep);
    }
    if d.start(now_ms).is_err() {
        let _ = put_json(store, "desc", &d);
        return Turn::Done(keep);
    }
    match put_json(store, "desc", &d) {
        Ok(()) => Turn::Solve(Box::new(d)),
        Err(_) => Turn::Done(Some(URGENT_ALARM_MS)),
    }
}

/// Second half: run the recorded attempt and persist its outcome.
pub fn solve(store: &dyn SqlStore, mut d: JobDescriptor, now_ms: u64) -> Option<u64> {
    match load_graph(store) {
        Ok(g) => {
            let _ = d.run(&g, now_ms);
        }
        Err(e) => d.fail(&e, now_ms),
    }
    put_json(store, "desc", &d).ok()?;
    retain(&Meta::read(store).ok()?, now_ms)
}

/// One whole alarm turn (native tests; the Worker shell awaits between
/// [`prepare`] and [`solve`]).
#[cfg(test)]
pub fn alarm(store: &dyn SqlStore, now_ms: u64) -> Option<u64> {
    match prepare(store, now_ms) {
        Turn::Done(next) => next,
        Turn::Solve(d) => solve(store, *d, now_ms),
        Turn::Expire => {
            let _ = expire(store);
            None
        }
    }
}
