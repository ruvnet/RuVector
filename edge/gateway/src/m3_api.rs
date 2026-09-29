//! M3 routes (ADR-351 §7.2) and their Workers glue. `routes::handle` hands
//! an authenticated `/v1` request here when [`parse`] matches, before the
//! M1 table:
//!
//! | Route | Scope + role |
//! |---|---|
//! | `POST /v1/collections/{c}/snapshots`, `GET` same | write+editor / read+viewer |
//! | `POST /v1/collections/{c}/snapshots/{id}:restore` | admin+owner |
//! | `POST /v1/collections/{c}:export`, `GET /v1/exports/{id}` | read+viewer |
//! | `POST /v1/uploads`, `…/{id}/parts/{n}`, `…/{id}:complete` | write+editor |
//! | `POST /v1/collections/{c}:import`, `GET /v1/jobs/{id}` | write+editor / read+viewer |
//! | `POST /v1/collections` with `embedder` | M1 create rule |
//! | `…/vectors[:upsert]`, `…/query` with `text` | the M1 route's rule |
//!
//! Create / upsert / query bodies without `embedder` / `text` are passed to
//! the M1 REST handler untouched (same `Idempotency-Key` behaviour). A
//! text request's key is remembered here, against the **original** body
//! (`m3text:` namespace), because the embedded body is not bit-stable
//! across model calls. Every M3 request except a plain (no `text`) query emits
//! one audit event (`audited`), sent after the reply (`ctx.wait_until`).

use crate::api::{
    self,
    ratelimit::{Class, EnvLimiter},
};
use crate::audit::{self, AuditEvent};
use crate::audit_http::{flush, Deferred};
use crate::auth::Authenticated;
use crate::config::GatewayConfig;
use crate::durable::DoBackend;
use crate::idem::{self, Seen, Slot};
use crate::m3_ctx::M3;
use crate::m3_ports::{Blob, Queues, WorkersAi, DATA_BINDING, R2};
use crate::m3_wire::M3Backend;
use crate::rest::{self, ApiReply, ApiRoute, Caller};
use crate::{embed, export, jobs, respond, restore, service, snapshots, uploads};
use ruvector_edge_auth::{prm, Capability, Clock};
use ruvector_edge_snapshot::EmbeddingPort;
use ruvector_edge_store::Op;
use ruvector_edge_store::{ErrorCode, OpError};
use serde_json::Value as Json;
use worker::{Context, Env, Method, Request, Response, Result};

/// M3 routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum M3Route {
    /// `POST /v1/collections` (embedder-aware).
    Create,
    /// Upsert (text-aware).
    Upsert(String),
    /// Query (text-aware).
    Query(String),
    /// `POST …/snapshots`.
    SnapshotCreate(String),
    /// `GET …/snapshots`.
    SnapshotList(String),
    /// `POST …/snapshots/{id}:restore`.
    Restore(String, String),
    /// `POST …{c}:export`.
    Export(String),
    /// `GET /v1/exports/{id}`.
    ExportGet(String),
    /// `POST /v1/uploads`.
    UploadCreate,
    /// `POST /v1/uploads/{id}/parts/{n}`.
    UploadPart(String, u16),
    /// `POST /v1/uploads/{id}:complete`.
    UploadComplete(String),
    /// `POST …{c}:import`.
    Import(String),
    /// `GET /v1/jobs/{id}`.
    Job(String),
}

fn name_ok(c: &str) -> bool {
    !c.is_empty() && !c.contains(':') && c != "." && c != ".."
}

/// Exact route match; `None` → the M1 table.
pub fn parse(method: &Method, path: &str) -> Option<M3Route> {
    let rest = path.strip_prefix("/v1/")?;
    let seg: Vec<&str> = rest.split('/').collect();
    let (get, post) = (*method == Method::Get, *method == Method::Post);
    let s = |x: &str| x.to_string();
    let r = match seg.as_slice() {
        ["collections"] if post => M3Route::Create,
        ["collections", c, "vectors" | "vectors:upsert"] if post && name_ok(c) => {
            M3Route::Upsert(s(c))
        }
        ["collections", c, "query"] if post && name_ok(c) => M3Route::Query(s(c)),
        ["collections", c, "snapshots"] if name_ok(c) && post => M3Route::SnapshotCreate(s(c)),
        ["collections", c, "snapshots"] if name_ok(c) && get => M3Route::SnapshotList(s(c)),
        ["collections", c, "snapshots", id] if post && name_ok(c) => {
            let id = id.strip_suffix(":restore")?;
            M3Route::Restore(s(c), s(id))
        }
        ["collections", ce] if post => match ce.split_once(':') {
            Some((c, "export")) if name_ok(c) => M3Route::Export(s(c)),
            Some((c, "import")) if name_ok(c) => M3Route::Import(s(c)),
            _ => return None,
        },
        ["exports", id] if get => M3Route::ExportGet(s(id)),
        ["uploads"] if post => M3Route::UploadCreate,
        ["uploads", id, "parts", n] if post => M3Route::UploadPart(s(id), n.parse().ok()?),
        ["uploads", idc] if post => M3Route::UploadComplete(s(idc.strip_suffix(":complete")?)),
        ["jobs", id] if get => M3Route::Job(s(id)),
        _ => return None,
    };
    Some(r)
}

impl M3Route {
    /// Audit route name.
    pub fn name(&self) -> &'static str {
        match self {
            M3Route::Create => "collection.create",
            M3Route::Upsert(_) => "vector.upsert",
            M3Route::Query(_) => "vector.query",
            M3Route::SnapshotCreate(_) => "snapshot.create",
            M3Route::SnapshotList(_) => "snapshot.list",
            M3Route::Restore(..) => "snapshot.restore",
            M3Route::Export(_) => "collection.export",
            M3Route::ExportGet(_) => "export.download",
            M3Route::UploadCreate => "upload.create",
            M3Route::UploadPart(..) => "upload.part",
            M3Route::UploadComplete(_) => "upload.complete",
            M3Route::Import(_) => "collection.import",
            M3Route::Job(_) => "job.get",
        }
    }

    /// Routes whose body may be up to [`uploads::MAX_INLINE_BYTES`].
    pub fn large_body(&self) -> bool {
        matches!(self, M3Route::UploadPart(..) | M3Route::Import(_))
    }

    /// ADR-351 §10 layer 2 budget class: snapshot, restore, export,
    /// uploads, import and (text) create / upsert are writes; query,
    /// snapshot list, export download and job status are reads.
    pub fn class(&self) -> Class {
        match self {
            M3Route::Query(_)
            | M3Route::SnapshotList(_)
            | M3Route::ExportGet(_)
            | M3Route::Job(_) => Class::Read,
            M3Route::Create
            | M3Route::Upsert(_)
            | M3Route::SnapshotCreate(_)
            | M3Route::Restore(..)
            | M3Route::Export(_)
            | M3Route::UploadCreate
            | M3Route::UploadPart(..)
            | M3Route::UploadComplete(_)
            | M3Route::Import(_) => Class::Write,
        }
    }
}

/// Pre-auth size check: 8 MiB on upload parts and inline imports, the M5
/// registry check (part cap, else the M1 1 MiB) everywhere else.
pub fn declared_too_large(req: &Request) -> bool {
    let large = parse(&req.method(), &req.path()).is_some_and(|r| r.large_body());
    if !large {
        return crate::registry_http::declared_too_large(req);
    }
    req.headers()
        .get("Content-Length")
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .is_some_and(|n| n > uploads::MAX_INLINE_BYTES)
}

/// A decoded M3 request.
pub struct Input<'a> {
    /// Route.
    pub route: &'a M3Route,
    /// Body (bounded).
    pub body: Vec<u8>,
    /// `Content-Type: application/octet-stream`.
    pub octet_stream: bool,
    /// Query string (without `?`).
    pub query: Option<&'a str>,
    /// `Idempotency-Key`.
    pub key: Option<&'a str>,
}

/// Result of an M3 request.
#[derive(Debug)]
pub enum Out {
    /// JSON / problem reply.
    Reply(ApiReply),
    /// Stream this R2 object.
    Download(export::Download),
}

fn reply(r: std::result::Result<(u16, Json), OpError>, md: &str) -> ApiReply {
    match r {
        Ok((status, v)) => ApiReply::json(status, &v),
        Err(e) => ApiReply::problem(&e, md),
    }
}

/// `true` when a body may carry `text` (byte scan, no JSON parse): only
/// then does an upsert/query take the embedding path.
pub fn text_marker(body: &[u8]) -> bool {
    body.windows(6).any(|w| w == b"\"text\"")
}

/// Whether `dispatch` ships an event for this request: every write, but
/// not a plain (no `text`) query — the M1 read hot path — and not a
/// download (`serve` ships it once the stream's status is known).
pub fn audited(route: &M3Route, body: &[u8]) -> bool {
    match route {
        M3Route::Query(_) => text_marker(body),
        M3Route::ExportGet(_) => false,
        _ => true,
    }
}

/// Pure dispatch plus the audit event (native tests drive this with
/// in-memory ports).
pub async fn dispatch<B: M3Backend, R: Blob, Q: Queues, E: EmbeddingPort>(
    m: &M3<'_, B, R, Q>,
    ai: &E,
    caller: &Caller,
    md: &str,
    inp: Input<'_>,
) -> Out {
    let route = inp.route.name();
    let audit = audited(inp.route, &inp.body);
    let inline = matches!(inp.route, M3Route::Import(_)) && inp.octet_stream;
    if inline || matches!(inp.route, M3Route::UploadPart(..)) {
        m.note.bytes.set(inp.body.len() as u64);
    }
    let out = run(m, ai, caller, md, inp).await;
    if let (true, Out::Reply(r)) = (audit, &out) {
        if r.status < 300 {
            counts_from_reply(&m.note, &r.body);
        }
        let ev = AuditEvent::new(m.ctx, route, r.status, m.now_ms).with_note(&m.note);
        audit::emit(m.queues, &ev).await;
    }
    out
}

/// Rows and bytes a successful reply reports (`rows` / `upserted`,
/// `bytes`), unless the executor noted them already.
fn counts_from_reply(n: &audit::Note, body: &str) {
    let Ok(v) = serde_json::from_str::<Json>(body) else {
        return;
    };
    if n.rows.get() == 0 {
        let rows = v["rows"].as_u64().or(v["upserted"].as_u64());
        n.rows.set(rows.unwrap_or(0));
    }
    if n.bytes.get() == 0 {
        n.bytes.set(v["bytes"].as_u64().unwrap_or(0));
    }
}

async fn run<B: M3Backend, R: Blob, Q: Queues, E: EmbeddingPort>(
    m: &M3<'_, B, R, Q>,
    ai: &E,
    caller: &Caller,
    md: &str,
    inp: Input<'_>,
) -> Out {
    let now = m.now_ms / 1000;
    let r = match inp.route {
        M3Route::Create => match embed::create(m, &inp.body).await {
            Ok(Some(v)) => Ok(v),
            Ok(None) => {
                let route = ApiRoute::Create;
                let r = rest::handle(m.b, caller, &route, &inp.body, inp.key, now, md).await;
                return Out::Reply(r);
            }
            Err(e) => Err(e),
        },
        M3Route::Upsert(c) | M3Route::Query(c) => {
            return Out::Reply(text_route(m, ai, caller, md, &inp, c).await)
        }
        M3Route::SnapshotCreate(c) => snapshots::create(m, c).await,
        M3Route::SnapshotList(c) => snapshots::list(m, c).await,
        M3Route::Restore(c, id) => restore::restore(m, c, id).await,
        M3Route::Export(c) => export::export(m, c, inp.query).await,
        M3Route::ExportGet(id) => match export::download(m, id).await {
            Ok(d) => return Out::Download(d),
            Err(e) => Err(e),
        },
        M3Route::UploadCreate => uploads::create(m, &inp.body).await,
        M3Route::UploadPart(id, n) => uploads::part(m, id, *n, inp.body).await,
        M3Route::UploadComplete(id) => uploads::complete(m, id).await,
        M3Route::Import(c) => jobs::submit(m, c, inp.body, inp.octet_stream).await,
        M3Route::Job(id) => jobs::status(m, id).await,
    };
    Out::Reply(reply(r, md))
}

async fn text_route<B: M3Backend, R: Blob, Q: Queues, E: EmbeddingPort>(
    m: &M3<'_, B, R, Q>,
    ai: &E,
    caller: &Caller,
    md: &str,
    inp: &Input<'_>,
    c: &str,
) -> ApiReply {
    let now = m.now_ms / 1000;
    let upsert = matches!(inp.route, M3Route::Upsert(_));
    let api_route = if upsert {
        ApiRoute::Upsert(c.to_string())
    } else {
        ApiRoute::Query(c.to_string())
    };
    // Byte scan first: the M1 hot path never pays a second JSON parse.
    let has_text = text_marker(&inp.body)
        && serde_json::from_slice::<Json>(&inp.body).is_ok_and(|v| {
            v.get("text").is_some()
                || v.get("vectors")
                    .and_then(Json::as_array)
                    .is_some_and(|a| a.iter().any(|r| r.get("text").is_some()))
        });
    if !has_text {
        return rest::handle(m.b, caller, &api_route, &inp.body, inp.key, now, md).await;
    }
    // Scope and role first, so a stored replay is re-authorized (as in `rest`).
    let (op, cap) = if upsert {
        (Op::VectorUpsert, Capability::Write)
    } else {
        (Op::VectorQuery, Capability::Read)
    };
    m.note.scope.set(Some(cap.satisfying_scope()));
    match service::authorize_op(m.b, m.ctx, op).await {
        Ok(a) => m.note.role.set(a.role),
        Err(e) => return ApiReply::problem(&e, md),
    }
    let slot = match inp.key.filter(|_| upsert) {
        None => None,
        Some(k) if !idem::key_ok(k) => {
            let e = OpError::invalid("Idempotency-Key must be 1..=255 visible ASCII bytes");
            return ApiReply::problem(&e, md);
        }
        Some(k) => {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(format!("text-upsert/{c}\n").as_bytes());
            h.update(&inp.body);
            let slot = Slot {
                sub: m.ctx.sub().to_string(),
                key: format!("m3text:{k}"),
                sha256: h.finalize().into(),
            };
            let reused = "Idempotency-Key reused with a different request";
            match idem::check(m.b, m.ctx, &slot, true, reused, now).await {
                Err(e) => return ApiReply::problem(&e, md),
                Ok(Seen::Replay(stored)) => return replayed(&stored, md),
                Ok(Seen::Miss) => Some(slot),
            }
        }
    };
    let embedded = if upsert {
        embed::upsert_body(m, ai, c, &inp.body).await
    } else {
        embed::query_body(m, ai, c, &inp.body).await
    };
    let out = match embedded {
        Ok(Some((body, _wu))) => rest::handle(m.b, caller, &api_route, &body, None, now, md).await,
        Ok(None) => rest::handle(m.b, caller, &api_route, &inp.body, None, now, md).await,
        Err(e) => ApiReply::problem(&e, md),
    };
    if let Some(slot) = slot {
        let stored = (out.status < 300).then(|| format!("{} {}", out.status, out.body));
        idem::finish(m.b, m.ctx, slot, stored, now).await;
    }
    out
}

fn replayed(stored: &str, md: &str) -> ApiReply {
    match stored.split_once(' ').and_then(|(s, j)| {
        Some((
            s.parse::<u16>().ok()?,
            serde_json::from_str::<Json>(j).ok()?,
        ))
    }) {
        Some((s, j)) => ApiReply::json(s, &j),
        None => ApiReply::problem(&OpError::new(ErrorCode::ServerError, "stored response"), md),
    }
}

/// Workers glue: read the body, run [`dispatch`], render (or stream an
/// export); audit sends go out through `ctx.wait_until` after the reply.
pub async fn serve(
    mut req: Request,
    env: &Env,
    cfg: &GatewayConfig,
    route: M3Route,
    auth: Authenticated,
    wctx: &Context,
) -> Result<Response> {
    let caller = api::caller(auth);
    let md = prm::metadata_url(&cfg.rest_resource);
    let b = DoBackend { env };
    let now_ms = crate::platform::WorkerClock.now_unix() * 1000;
    // ADR-351 §10 layer 2 then §5.8, as on every `/v1` route: rate budget
    // and deny list before the body (up to 8 MiB) is read.
    let limiter = EnvLimiter(env);
    let guarded = api::guard(&b, &limiter, route.class(), &caller.ctx, now_ms / 1000, &md).await;
    if let Err(r) = guarded {
        if !matches!(route, M3Route::Query(_) | M3Route::ExportGet(_)) {
            let queues = Deferred::new(env);
            let ev = AuditEvent::new(&caller.ctx, route.name(), r.reply.status, now_ms);
            audit::emit(&queues, &ev).await;
            flush(&queues, env, wctx);
        }
        return api::refused(r);
    }
    let max = if route.large_body() {
        uploads::MAX_INLINE_BYTES
    } else {
        api::MAX_BODY_BYTES
    };
    let body = if matches!(req.method(), Method::Get) || req.inner().body().is_none() {
        Some(Vec::new())
    } else {
        api::read_capped(req.stream()?, max).await?
    };
    let Some(body) = body else {
        let e = OpError::new(ErrorCode::PayloadTooLarge, "body too large");
        let r = ApiReply::problem(&e, &md);
        return respond::raw(r.status, r.body, r.content_type, None);
    };
    let ct = req.headers().get("Content-Type").ok().flatten();
    let key = req.headers().get("Idempotency-Key").ok().flatten();
    let url = req.url()?;
    let blob = R2(env.bucket(DATA_BINDING)?);
    let queues = Deferred::new(env);
    let ai = WorkersAi(env);
    let m = M3 {
        b: &b,
        blob: &blob,
        queues: &queues,
        ctx: &caller.ctx,
        now_ms,
        note: audit::Note::default(),
    };
    let inp = Input {
        route: &route,
        body,
        octet_stream: ct.is_some_and(|c| c.starts_with("application/octet-stream")),
        query: url.query(),
        key: key.as_deref(),
    };
    let out = dispatch(&m, &ai, &caller, &md, inp).await;
    let resp = match out {
        // `api::render`: a 429 (e.g. query fan-out over budget) carries
        // `Retry-After`, as on the M1 / M2 routes.
        Out::Reply(r) => api::render(r),
        Out::Download(d) => {
            let size = d.size;
            let resp = stream(&blob, d).await;
            let status = resp.as_ref().map_or(500, Response::status_code);
            if status == 200 {
                m.note.bytes.set(size);
            }
            let ev = AuditEvent::new(m.ctx, route.name(), status, m.now_ms).with_note(&m.note);
            audit::emit(&queues, &ev).await;
            resp
        }
    };
    flush(&queues, env, wctx);
    resp
}

async fn stream(blob: &R2, d: export::Download) -> Result<Response> {
    let Some(obj) = blob.0.get(&d.key).execute().await? else {
        return respond::problem(ruvector_edge_tenancy::ProblemCode::NotFound, None);
    };
    let Some(body) = obj.body() else {
        return respond::problem(ruvector_edge_tenancy::ProblemCode::NotFound, None);
    };
    let mut resp = Response::from_body(body.response_body()?)?;
    let h = resp.headers_mut();
    h.set("Content-Type", "application/octet-stream")?;
    h.set("Content-Length", &d.size.to_string())?;
    h.set(
        "Content-Disposition",
        &format!("attachment; filename=\"{}\"", d.filename),
    )?;
    h.set("Cache-Control", "no-store")?;
    for (k, v) in respond::RESPONSE_CORS {
        h.set(k, v)?;
    }
    Ok(resp)
}
