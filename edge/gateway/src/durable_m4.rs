//! Workers glue for the M4 Durable Object classes (ADR-351 §3): the cores
//! live in `quant_shard`, `graph_store` and `mincut_job`; the classes only
//! read the body, run the core synchronously (no `await` between reading
//! resident state and issuing SQL) and arm alarms.
//!
//! Bindings (wrangler.toml, migration `v-m4-quant-graph`): `QUANT_SHARD` →
//! `QuantShard`, `GRAPH_STORE` → `GraphStore`, `ANALYTICS_JOB` →
//! `AnalyticsJob`, all SQLite-backed, reachable only from this Worker.

use crate::durable::DoSql;
use crate::graph_store::{self, GraphHost};
use crate::mincut_job;
use crate::quant_shard::{self, QuantHost};
use crate::shard_core::URGENT_ALARM_MS;
use std::cell::RefCell;
use worker::{
    durable_object, Date, DurableObject, Env, Headers, Method, Request, Response, Result, State,
};

/// `QuantShard` namespace binding.
pub const QUANT_BINDING: &str = "QUANT_SHARD";
/// `GraphStore` namespace binding.
pub const GRAPH_BINDING: &str = "GRAPH_STORE";
/// `AnalyticsJob` namespace binding.
pub const JOB_BINDING: &str = "ANALYTICS_JOB";

thread_local! {
    static QUANT: RefCell<QuantHost> = RefCell::new(QuantHost::default());
    static GRAPHS: RefCell<GraphHost> = RefCell::new(GraphHost::default());
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

/// Arm the alarm `delay_ms` from now: always for urgent work, else only
/// when none is pending (a stream of writes never postpones a flush).
async fn schedule(state: &State, delay_ms: u64) {
    let storage = state.storage();
    let urgent = delay_ms <= URGENT_ALARM_MS;
    if urgent || matches!(storage.get_alarm().await, Ok(None)) {
        let ms = i64::try_from(delay_ms).unwrap_or(i64::MAX);
        if let Err(e) = storage.set_alarm(ms).await {
            worker::console_warn!("m4 durable object: set_alarm failed: {e}");
        }
    }
}

/// One shard of an `index = rabitq` collection.
#[durable_object]
pub struct QuantShard {
    sql: DoSql,
    state: State,
}

impl DurableObject for QuantShard {
    fn new(state: State, _env: Env) -> Self {
        QuantShard {
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
        let own = id.name();
        let (out, wiped, next) = QUANT.with(|h| {
            let mut host = h.borrow_mut();
            let (out, wiped) =
                quant_shard::serve(&mut host, &key, own.as_deref(), &self.sql, &body);
            (out, wiped, quant_shard::next_alarm(&host, &key))
        });
        if wiped {
            if let Err(e) = self.state.storage().delete_alarm().await {
                worker::console_warn!("quant shard wipe: deleteAlarm failed: {e}");
            }
        } else if let Some(ms) = next {
            schedule(&self.state, ms).await;
        }
        json_reply(out)
    }

    async fn alarm(&self) -> Result<Response> {
        let key = self.state.id().to_string();
        let next = QUANT.with(|h| quant_shard::alarm(&mut h.borrow_mut(), &key, &self.sql));
        if let Some(ms) = next {
            schedule(&self.state, ms).await;
        }
        Response::ok("")
    }
}

/// One tenant graph (or the tenant's graph catalog).
#[durable_object]
pub struct GraphStore {
    sql: DoSql,
    state: State,
}

impl DurableObject for GraphStore {
    fn new(state: State, _env: Env) -> Self {
        GraphStore {
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
        let own = id.name();
        let out = GRAPHS.with(|h| {
            graph_store::serve(&mut h.borrow_mut(), &key, own.as_deref(), &self.sql, &body)
        });
        json_reply(out)
    }
}

/// One async min-cut job.
#[durable_object]
pub struct AnalyticsJob {
    sql: DoSql,
    state: State,
}

impl DurableObject for AnalyticsJob {
    fn new(state: State, _env: Env) -> Self {
        AnalyticsJob {
            sql: DoSql(state.storage().sql()),
            state,
        }
    }

    async fn fetch(&self, mut req: Request) -> Result<Response> {
        let Some(body) = rpc_body(&mut req).await? else {
            return Response::error("not found", 404);
        };
        let own = self.state.id().name();
        let (out, next) = mincut_job::serve(own.as_deref(), &self.sql, &body);
        if let Some(ms) = next {
            schedule(&self.state, ms.max(1)).await;
        }
        json_reply(out)
    }

    /// One job turn (admit, then run); re-armed while work remains.
    async fn alarm(&self) -> Result<Response> {
        let next = match mincut_job::prepare(&self.sql, Date::now().as_millis()) {
            mincut_job::Turn::Done(next) => next,
            mincut_job::Turn::Solve(d) => {
                // Yield to storage I/O so the recorded attempt commits
                // before the (possibly CPU-fatal) solve.
                let _ = self.state.storage().get_alarm().await;
                mincut_job::solve(&self.sql, *d, Date::now().as_millis())
            }
        };
        if let Some(ms) = next {
            schedule(&self.state, ms).await;
        }
        Response::ok("")
    }
}
