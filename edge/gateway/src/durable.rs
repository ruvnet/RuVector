//! Workers glue for the two Durable Object classes (ADR-351 §6.1, §6.2),
//! the `SqlStore` adapter over `ctx.storage.sql`, and the stub-based
//! [`Backend`]. All logic lives in `ledger_core` / `shard_core`; the
//! classes only read the body, run the core synchronously and reply.
//!
//! Bindings (wrangler.toml): `TENANT_LEDGER` → class `TenantLedger`,
//! `VECTOR_SHARD` → class `VectorShard`, both SQLite-backed
//! (`new_sqlite_classes`). The DOs are reachable only through these
//! bindings from this Worker; they have no public route.

use crate::backend::Backend;
use crate::ledger_core::{self, FREE_PLAN};
use crate::shard_core::{self, ShardHost};
use crate::wire::unavailable;
use ruvector_edge_store::{OpError, Row, SqlStore, StoreError, TenantLedger as LedgerState, Value};
use ruvector_edge_tenancy::{DoName, EntropySource, TenancyError};
use std::cell::RefCell;

#[path = "shard_wipe.rs"]
pub mod wipe;
use worker::{
    durable_object, DurableObject, Env, Headers, Method, Request, RequestInit, Response, Result,
    SqlStorage, SqlStorageValue, State,
};

/// `TenantLedger` namespace binding.
pub const LEDGER_BINDING: &str = "TENANT_LEDGER";
/// `VectorShard` namespace binding.
pub const SHARD_BINDING: &str = "VECTOR_SHARD";

/// [`SqlStore`] over Durable Object SQLite. Every cursor is drained before
/// returning, so writes have happened when a call returns.
pub struct DoSql(pub SqlStorage);

fn store_err(e: worker::Error) -> StoreError {
    let text = e.to_string();
    if text.contains("constraint") || text.contains("UNIQUE") {
        StoreError::Constraint
    } else {
        StoreError::Backend("durable object sql".into())
    }
}

fn to_sql(v: &Value) -> SqlStorageValue {
    match v {
        Value::Null => SqlStorageValue::Null,
        Value::Int(i) => SqlStorageValue::Integer(*i),
        Value::Real(f) => SqlStorageValue::Float(*f),
        Value::Text(s) => SqlStorageValue::String(s.clone()),
        Value::Blob(b) => SqlStorageValue::Blob(b.clone()),
    }
}

fn from_sql(v: SqlStorageValue) -> Value {
    match v {
        SqlStorageValue::Null => Value::Null,
        SqlStorageValue::Boolean(b) => Value::Int(i64::from(b)),
        SqlStorageValue::Integer(i) => Value::Int(i),
        SqlStorageValue::Float(f) => Value::Real(f),
        SqlStorageValue::String(s) => Value::Text(s),
        SqlStorageValue::Blob(b) => Value::Blob(b),
    }
}

impl SqlStore for DoSql {
    fn exec(&self, sql: &str, params: &[Value]) -> std::result::Result<u64, StoreError> {
        let cursor = self
            .0
            .exec(sql, params.iter().map(to_sql).collect::<Vec<_>>())
            .map_err(store_err)?;
        for row in cursor.raw() {
            row.map_err(store_err)?;
        }
        Ok(cursor.rows_written() as u64)
    }

    fn query(&self, sql: &str, params: &[Value]) -> std::result::Result<Vec<Row>, StoreError> {
        let cursor = self
            .0
            .exec(sql, params.iter().map(to_sql).collect::<Vec<_>>())
            .map_err(store_err)?;
        cursor
            .raw()
            .map(|r| {
                r.map(|cells| cells.into_iter().map(from_sql).collect())
                    .map_err(store_err)
            })
            .collect()
    }
}

/// `EntropySource` over `crypto.getRandomValues`.
pub struct WorkerEntropy;

impl EntropySource for WorkerEntropy {
    fn fill(&self, out: &mut [u8]) -> std::result::Result<(), TenancyError> {
        #[cfg(target_arch = "wasm32")]
        {
            getrandom::getrandom(out).map_err(|_| TenancyError::MalformedIdentifier("entropy"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = out;
            Err(TenancyError::MalformedIdentifier("entropy"))
        }
    }
}

fn json_reply(body: String) -> Result<Response> {
    let headers = Headers::new();
    headers.set("Content-Type", "application/json")?;
    Ok(Response::ok(body)?.with_headers(headers))
}

async fn rpc_body(req: &mut Request) -> Result<Option<Vec<u8>>> {
    if req.method() != Method::Post {
        return Ok(None);
    }
    Ok(Some(req.bytes().await?))
}

/// One tenant's ledger (`idFromName(hex(sha256("v1|" + tenant_key + "|ledger")))`).
#[durable_object]
pub struct TenantLedger {
    sql: DoSql,
    slot: RefCell<Option<LedgerState>>,
}

impl DurableObject for TenantLedger {
    fn new(state: State, _env: Env) -> Self {
        TenantLedger {
            sql: DoSql(state.storage().sql()),
            slot: RefCell::new(None),
        }
    }

    async fn fetch(&self, mut req: Request) -> Result<Response> {
        let Some(body) = rpc_body(&mut req).await? else {
            return Response::error("not found", 404);
        };
        // No await from here on: one call, one coalesced commit.
        let out = ledger_core::serve(
            &mut self.slot.borrow_mut(),
            &self.sql,
            FREE_PLAN,
            &body,
            &WorkerEntropy,
        );
        json_reply(out)
    }
}

thread_local! {
    /// Resident shards of this isolate (shared by every `VectorShard`
    /// instance it hosts) with the LRU resident-set registry.
    static HOST: RefCell<ShardHost> = RefCell::new(ShardHost::default());
}

/// One shard of one collection (`idFromName(DoMeta::do_name())`).
#[durable_object]
pub struct VectorShard {
    sql: DoSql,
    state: State,
}

impl DurableObject for VectorShard {
    fn new(state: State, _env: Env) -> Self {
        VectorShard {
            sql: DoSql(state.storage().sql()),
            state,
        }
    }

    async fn fetch(&self, mut req: Request) -> Result<Response> {
        let Some(body) = rpc_body(&mut req).await? else {
            return Response::error("not found", 404);
        };
        let id = self.state.id();
        let key = id.to_string();
        let own_name = id.name();
        // Collection drop (ADR-351 §7.2): every row goes and `meta.wiped`
        // stays (no `deleteAll`: the marker is what refuses a write that
        // raced the drop). The reply carries the usage to release.
        if let Some((out, wiped)) = wipe::serve(own_name.as_deref(), &self.sql, &body) {
            if wiped {
                // Drop the resident index and any pending maintenance alarm,
                // so neither can flush stale state into the wiped storage (a
                // cold load sees the marker and schedules nothing).
                HOST.with(|h| h.borrow_mut().evict(&key));
                if let Err(e) = self.state.storage().delete_alarm().await {
                    worker::console_warn!("vector shard wipe: deleteAlarm failed: {e}");
                }
            }
            return json_reply(out);
        }
        let (out, next) = HOST.with(|h| {
            let mut host = h.borrow_mut();
            let out = shard_core::serve(&mut host, &key, own_name.as_deref(), &self.sql, &body);
            (out, shard_core::next_alarm(&host, &key))
        });
        // After the synchronous call (its SQL already issued): schedule
        // index maintenance (timer flush, compaction, requantize).
        if let Some(ms) = next {
            self.schedule(ms).await;
        }
        json_reply(out)
    }

    /// ADR-351 §6.1 alarm: one maintenance step, then re-arm if more is due.
    async fn alarm(&self) -> Result<Response> {
        let key = self.state.id().to_string();
        let next = HOST.with(|h| shard_core::alarm(&mut h.borrow_mut(), &key, &self.sql));
        if let Some(ms) = next {
            self.schedule(ms).await;
        }
        Response::ok("")
    }
}

impl VectorShard {
    /// Arm the alarm `delay_ms` from now: always for urgent work, else only
    /// when none is pending (so a stream of writes never postpones the
    /// timer flush indefinitely).
    async fn schedule(&self, delay_ms: u64) {
        let storage = self.state.storage();
        let urgent = delay_ms <= shard_core::URGENT_ALARM_MS;
        if urgent || matches!(storage.get_alarm().await, Ok(None)) {
            let ms = i64::try_from(delay_ms).unwrap_or(i64::MAX);
            if let Err(e) = storage.set_alarm(ms).await {
                worker::console_warn!("vector shard: set_alarm failed: {e}");
            }
        }
    }
}

/// [`Backend`] over DO stubs addressed with `idFromName`.
pub struct DoBackend<'a> {
    /// Worker environment (bindings).
    pub env: &'a Env,
}

impl DoBackend<'_> {
    async fn post(&self, binding: &str, name: &DoName, body: String) -> Result<String> {
        let stub = self
            .env
            .durable_object(binding)?
            .id_from_name(name.as_str())?
            .get_stub()?;
        let headers = Headers::new();
        headers.set("Content-Type", "application/json")?;
        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(body.into()));
        let req = Request::new_with_init("https://do.internal/rpc", &init)?;
        let mut resp = stub.fetch_with_request(req).await?;
        if resp.status_code() != 200 {
            return Err(worker::Error::RustError("durable object status".into()));
        }
        resp.text().await
    }
}

impl Backend for DoBackend<'_> {
    async fn call_ledger(
        &self,
        name: &DoName,
        body: String,
    ) -> std::result::Result<String, OpError> {
        self.post(LEDGER_BINDING, name, body)
            .await
            .map_err(|_| unavailable())
    }

    async fn call_shard(
        &self,
        name: &DoName,
        body: String,
    ) -> std::result::Result<String, OpError> {
        self.post(SHARD_BINDING, name, body)
            .await
            .map_err(|_| unavailable())
    }

    async fn charge_fanout(
        &self,
        ctx: &ruvector_edge_store::CallerContext,
        extra: u32,
    ) -> std::result::Result<(), OpError> {
        use crate::api::ratelimit::{admit_n, Class, EnvLimiter};
        admit_n(&EnvLimiter(self.env), Class::Read, ctx, extra)
            .await
            .map_err(|_| OpError::new(ruvector_edge_store::ErrorCode::RateLimited, "rate limited"))
    }
}
