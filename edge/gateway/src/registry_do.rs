//! Workers glue for the registry Durable Objects (ADR-351 §3 rv-registry,
//! M5) and the stub-based [`RegistryRpc`]. Logic lives in `registry_core`.
//!
//! - `RegistryRoot` (binding `REGISTRY_ROOT`, one object named
//!   [`ROOT_DO_NAME`]): the global scope directory.
//! - `RegistryScope` (binding `REGISTRY_SCOPE`, one object per scope named
//!   `keys::registry_do_name(scope)`): that scope's registry index. While
//!   sessions or released blobs are pending it keeps an alarm armed; the
//!   alarm runs the sweep and deletes (and aborts) in R2 everything the
//!   sweep hands out **before the object serves another request**: every
//!   request that arrives while the alarm awaits R2 is refused (503, the
//!   Worker maps it to `shard_unavailable`), so no pin can interleave
//!   between the index change and the delete. The refusal is a
//!   [`SweepGate`]: released by a drop guard, and void past a deadline, so a
//!   cancelled or abandoned alarm cannot wedge the scope. Work beyond one
//!   alarm's budget stays in the durable queue (`registry_pending`) and the
//!   alarm re-arms within seconds.
//!
//! Both are SQLite-backed (`new_sqlite_classes`) and reachable only
//! through their bindings from this Worker.

use crate::blob_r2::{R2, R2_BINDING};
use crate::durable::{DoSql, WorkerEntropy};
use crate::platform::WorkerClock;
use crate::registry_core::{gateway_config, serve_root, serve_scope};
use crate::registry_ports::{apply_sweep, RegistryRpc, ROOT_DO_NAME};
use crate::registry_sweep::{GateGuard, SweepGate, SweepWork};
use crate::registry_wire::RvfError;
use worker::{
    durable_object, DurableObject, Env, Headers, Method, Request, RequestInit, Response, Result,
    State,
};

/// `RegistryRoot` namespace binding.
pub const ROOT_BINDING: &str = "REGISTRY_ROOT";
/// `RegistryScope` namespace binding.
pub const SCOPE_BINDING: &str = "REGISTRY_SCOPE";
/// Sweep cadence while maintenance is pending.
const SWEEP_EVERY_MS: i64 = 15 * 60 * 1000;
/// Re-arm delay while queued R2 work is left.
const SWEEP_AGAIN_MS: i64 = 2000;

fn now_ms() -> u64 {
    worker::Date::now().as_millis()
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

/// The global scope directory.
#[durable_object]
pub struct RegistryRoot {
    sql: DoSql,
}

impl DurableObject for RegistryRoot {
    fn new(state: State, _env: Env) -> Self {
        RegistryRoot {
            sql: DoSql(state.storage().sql()),
        }
    }

    async fn fetch(&self, mut req: Request) -> Result<Response> {
        let Some(body) = rpc_body(&mut req).await? else {
            return Response::error("not found", 404);
        };
        json_reply(serve_root(&self.sql, &WorkerClock, &body))
    }
}

/// One scope's registry index.
#[durable_object]
pub struct RegistryScope {
    sql: DoSql,
    state: State,
    env: Env,
    gate: SweepGate,
}

impl RegistryScope {
    async fn arm(&self) -> Result<()> {
        let storage = self.state.storage();
        if storage.get_alarm().await?.is_none() {
            storage.set_alarm(SWEEP_EVERY_MS).await?;
        }
        Ok(())
    }

    async fn sweep(&self, guard: &GateGuard<'_>) -> Result<SweepWork> {
        let r2 = R2(self.env.bucket(R2_BINDING)?);
        apply_sweep(
            &self.sql,
            &WorkerClock,
            &WorkerEntropy,
            gateway_config(),
            &r2,
            &|| guard.live(now_ms()),
        )
        .await
        .map_err(|e| worker::Error::RustError(e.detail))
    }
}

impl DurableObject for RegistryScope {
    fn new(state: State, env: Env) -> Self {
        RegistryScope {
            sql: DoSql(state.storage().sql()),
            state,
            env,
            gate: SweepGate::default(),
        }
    }

    async fn fetch(&self, mut req: Request) -> Result<Response> {
        let Some(body) = rpc_body(&mut req).await? else {
            return Response::error("not found", 404);
        };
        // Checked after the last await before the core runs.
        if self.gate.busy(now_ms()) {
            return Response::error("sweeping", 503);
        }
        let served = serve_scope(
            &self.sql,
            &WorkerClock,
            &WorkerEntropy,
            gateway_config(),
            &body,
        );
        if served.maintenance {
            // Best effort: the next mutating request re-arms.
            let _ = self.arm().await;
        }
        json_reply(served.body)
    }

    async fn alarm(&self) -> Result<Response> {
        let res = {
            let guard = self.gate.enter(now_ms());
            self.sweep(&guard).await
        };
        let pending = crate::registry_kv::SqlKv::open(&self.sql)
            .map(|kv| crate::registry_core::pending(&kv))
            .unwrap_or(true);
        let again = match &res {
            // Soon only while this alarm made progress: a delete R2 keeps
            // refusing is retried on the slow cadence, not every 2 s.
            Ok(w) if w.remaining && !(w.delete.is_empty() && w.abort.is_empty()) => {
                Some(SWEEP_AGAIN_MS)
            }
            _ if pending => Some(SWEEP_EVERY_MS),
            _ => None,
        };
        if let Some(ms) = again {
            self.state.storage().set_alarm(ms).await?;
        }
        res?;
        Response::ok("swept")
    }
}

/// [`RegistryRpc`] over DO stubs addressed with `idFromName`.
pub struct DoRegistry<'a> {
    /// Worker environment (bindings).
    pub env: &'a Env,
}

impl DoRegistry<'_> {
    async fn post(&self, binding: &str, name: &str, body: String) -> Result<String> {
        let stub = self
            .env
            .durable_object(binding)?
            .id_from_name(name)?
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

impl RegistryRpc for DoRegistry<'_> {
    async fn call_root(&self, body: String) -> std::result::Result<String, RvfError> {
        self.post(ROOT_BINDING, ROOT_DO_NAME, body)
            .await
            .map_err(|_| RvfError::unavailable())
    }

    async fn call_scope(
        &self,
        do_name: &str,
        body: String,
    ) -> std::result::Result<String, RvfError> {
        self.post(SCOPE_BINDING, do_name, body)
            .await
            .map_err(|_| RvfError::unavailable())
    }
}
