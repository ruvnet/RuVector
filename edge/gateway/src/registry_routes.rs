//! rv-registry REST routes (ADR-351 §3, M5), served on the `/v1` resource
//! only (`/v1/mcp` has read tools and import, never publish):
//!
//! | Route | Needs |
//! |---|---|
//! | `POST /v1/rvf/scopes/{scope}` | `ruvector:admin` (claim + adopt) |
//! | `GET /v1/rvf/scopes` | `ruvector:read` |
//! | `POST /v1/rvf/{scope}/{name}/{version}/uploads` | `ruvector:write`, owned scope |
//! | `PUT …/uploads/{id}/parts/{n}` | `ruvector:write` (body ≤ `max_part_size`) |
//! | `POST …/uploads/{id}:finalize` | `ruvector:write` |
//! | `POST …/{version}:publish` | `ruvector:publish` ∩ owner |
//! | `POST …/{version}:yank` / `:unyank` | `ruvector:write` (uploader or admin) |
//! | `GET …/{version}` | `ruvector:read` (manifest view) |
//! | `GET …/{version}/blob` | `ruvector:read` (streamed bytes) |
//! | `GET /v1/rvf/{scope}/{name}` | `ruvector:read` (versions, paginated) |
//! | `POST /v1/collections/{c}:import-rvf` | `ruvector:read` on the package, `ruvector:write` on the collection |
//!
//! `{scope}` is `acme`, `@acme` or `%40acme`. Cross-tenant reads of a
//! non-public version are `404`.

use crate::backend::Backend;
use crate::registry_ports::{root, scope, BlobStore, RegistryRpc};
use crate::registry_wire::{CallerWire, Coords, RootCall, RootOut, RvfError, ScopeCall, ScopeOut};
use crate::rest::ApiReply;
use crate::rvf_upload::{self, body_json, Deps, Target};
use crate::service;
use ruvector_edge_registry::{Caller, PackageName, RegistryError, Scope, Version};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};
use serde::Deserialize;
use serde_json::{json, Value as Json};
use worker::Method;

/// A registry route (path components still unparsed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RvfRoute {
    /// `POST /v1/rvf/scopes/{scope}`.
    ClaimScope(String),
    /// `GET /v1/rvf/scopes`.
    ListScopes,
    /// `POST …/{version}/uploads`.
    Begin(String, String),
    /// `PUT …/uploads/{id}/parts/{n}`.
    Part(String, String, String, String),
    /// `POST …/uploads/{id}:finalize`.
    Finalize(String, String, String),
    /// `POST …/{version}:publish`.
    Publish(String, String),
    /// `POST …/{version}:yank`.
    Yank(String, String),
    /// `POST …/{version}:unyank`.
    Unyank(String, String),
    /// `GET …/{version}`.
    Get(String, String),
    /// `GET …/{version}/blob`.
    Blob(String, String),
    /// `GET /v1/rvf/{scope}/{name}`.
    Versions(String),
    /// `POST /v1/collections/{c}:import-rvf`.
    Import(String),
}

fn scope_seg(s: &str) -> &str {
    s.strip_prefix("%40")
        .or_else(|| s.strip_prefix('@'))
        .unwrap_or(s)
}

fn pkg(scope: &str, name: &str) -> String {
    format!("@{}/{name}", scope_seg(scope))
}

/// Exact route match for the registry surface (`None`: not a registry
/// route; the REST table decides).
pub fn parse(method: &Method, path: &str) -> Option<RvfRoute> {
    let (get, post, put) = (
        *method == Method::Get,
        *method == Method::Post,
        *method == Method::Put,
    );
    if let Some(c) = path
        .strip_prefix("/v1/collections/")
        .and_then(|r| r.strip_suffix(":import-rvf"))
    {
        let ok = !c.is_empty() && !c.contains(['/', ':']);
        return (post && ok).then(|| RvfRoute::Import(c.to_string()));
    }
    let rest = path.strip_prefix("/v1/rvf/")?;
    let seg: Vec<&str> = rest.split('/').collect();
    if seg.iter().any(|s| s.is_empty()) {
        return None;
    }
    let r = match seg.as_slice() {
        ["scopes"] if get => RvfRoute::ListScopes,
        ["scopes", s] if post => RvfRoute::ClaimScope(scope_seg(s).to_string()),
        [s, n] if get => RvfRoute::Versions(pkg(s, n)),
        [s, n, v] if get && !v.contains(':') => RvfRoute::Get(pkg(s, n), v.to_string()),
        [s, n, v] if post => {
            let (ver, action) = v.rsplit_once(':')?;
            let (p, ver) = (pkg(s, n), ver.to_string());
            match action {
                "publish" => RvfRoute::Publish(p, ver),
                "yank" => RvfRoute::Yank(p, ver),
                "unyank" => RvfRoute::Unyank(p, ver),
                _ => return None,
            }
        }
        [s, n, v, "blob"] if get => RvfRoute::Blob(pkg(s, n), v.to_string()),
        [s, n, v, "uploads"] if post => RvfRoute::Begin(pkg(s, n), v.to_string()),
        [s, n, v, "uploads", id] if post => {
            let id = id.strip_suffix(":finalize")?;
            RvfRoute::Finalize(pkg(s, n), v.to_string(), id.to_string())
        }
        [s, n, v, "uploads", id, "parts", k] if put => {
            RvfRoute::Part(pkg(s, n), v.to_string(), id.to_string(), k.to_string())
        }
        _ => return None,
    };
    Some(r)
}

/// `true` for the part upload route (its body cap is `max_part_size`, not
/// the 1 MiB data-route cap).
pub fn is_part_upload(method: &Method, path: &str) -> bool {
    matches!(parse(method, path), Some(RvfRoute::Part(..)))
}

/// Pagination from the query string.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Query {
    /// `cursor`.
    pub cursor: Option<String>,
    /// `limit`.
    pub limit: Option<usize>,
}

/// What a registry route answers.
#[derive(Debug, Clone, PartialEq)]
pub enum RvfReply {
    /// A JSON or problem reply.
    Api(ApiReply),
    /// Stream this R2 object (a pull).
    Blob {
        /// R2 key.
        key: String,
        /// Object size.
        size: u64,
        /// SHA-256 (hex), sent as the `ETag`.
        sha256: String,
        /// The version is yanked (pullable by exact version).
        yanked: bool,
    },
}

/// The registry caller for a verified request: tenant, subject and the
/// effective capabilities (token scope ∩ ledger role). Not a member:
/// `not_claimed` for an unclaimed tenant, else `role_required`.
pub async fn registry_caller<B: Backend>(b: &B, ctx: &CallerContext) -> Result<Caller, RvfError> {
    let a = service::access(b, ctx).await?;
    if a.role.is_none() {
        let (code, detail) = if a.claimed {
            (ErrorCode::RoleRequired, "not a member of this tenant")
        } else {
            (ErrorCode::NotClaimed, "tenant not claimed")
        };
        return Err(OpError::new(code, detail).into());
    }
    Ok(Caller {
        tenant: ctx.tenant_key().clone(),
        sub: ctx.sub().to_string(),
        caps: a.effective(ctx.scope_caps()),
    })
}

pub(crate) fn target(name: &str, version: &str) -> Result<Target, RvfError> {
    Ok(Target {
        name: PackageName::parse(name).map_err(|e| RvfError::from(RegistryError::from(e)))?,
        version: Version::parse(version).map_err(|e| RvfError::from(RegistryError::from(e)))?,
    })
}

fn at(name: &str, version: &str) -> Result<(Target, Coords), RvfError> {
    let t = target(name, version)?;
    let c = t.coords();
    Ok((t, c))
}

fn manifest_json(out: ScopeOut) -> Result<(u16, Json), RvfError> {
    match out {
        ScopeOut::Manifest { manifest } => Ok((
            200,
            serde_json::to_value(&manifest).map_err(|_| RvfError::unexpected())?,
        )),
        _ => Err(RvfError::unexpected()),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoBody {}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct YankBody {
    #[serde(default)]
    reason: String,
}

/// Claim `scope` in the directory, then adopt it in the scope's index.
/// Both steps are idempotent, so a retry heals a half-done claim.
pub async fn claim<R: RegistryRpc>(r: &R, c: &Caller, s: &str) -> Result<(u16, Json), RvfError> {
    let sc = Scope::parse(s).map_err(|e| RvfError::from(RegistryError::from(e)))?;
    let call = RootCall::Claim {
        caller: CallerWire::from(c),
        scope: sc.as_str().to_string(),
    };
    let RootOut::Claim { claim } = root(r, call).await? else {
        return Err(RvfError::unexpected());
    };
    let adopt = ScopeCall::Adopt {
        claim: claim.clone(),
    };
    scope(r, &sc, adopt).await?;
    Ok((
        200,
        json!({ "scope": claim.scope, "claimed_at": claim.claimed_at }),
    ))
}

/// Serve one registry route for a verified caller. `body` is the request
/// body (read under the route's cap).
pub async fn handle<B: Backend, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    ctx: &CallerContext,
    route: &RvfRoute,
    body: Vec<u8>,
    q: &Query,
    now: u64,
    metadata_url: &str,
) -> RvfReply {
    match route_inner(d, ctx, route, body, q, now).await {
        Ok(Ok((status, v))) => RvfReply::Api(ApiReply::json(status, &v)),
        Ok(Err(blob)) => blob,
        Err(e) => RvfReply::Api(e.reply(metadata_url)),
    }
}

type Inner = Result<Result<(u16, Json), RvfReply>, RvfError>;

async fn route_inner<B: Backend, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    ctx: &CallerContext,
    route: &RvfRoute,
    body: Vec<u8>,
    q: &Query,
    now: u64,
) -> Inner {
    let c = registry_caller(d.b, ctx).await?;
    let cw = CallerWire::from(&c);
    let r = d.r;
    let out = match route {
        RvfRoute::ClaimScope(s) => claim(r, &c, s).await?,
        RvfRoute::ListScopes => {
            if !c.caps.contains(ruvector_edge_auth::Capability::Read) {
                return Err(RegistryError::Forbidden(ruvector_edge_auth::Capability::Read).into());
            }
            let RootOut::Scopes { scopes } = root(r, RootCall::Scopes { caller: cw }).await? else {
                return Err(RvfError::unexpected());
            };
            (200, json!({ "scopes": scopes }))
        }
        RvfRoute::Begin(n, v) => rvf_upload::begin(d, &c, &target(n, v)?, &body).await?,
        RvfRoute::Part(n, v, id, k) => {
            rvf_upload::put_part(d, &c, &target(n, v)?, id, k, body).await?
        }
        RvfRoute::Finalize(n, v, id) => {
            crate::rvf_finalize::finalize(d, &c, &target(n, v)?, id, &body).await?
        }
        RvfRoute::Publish(n, v) => {
            let NoBody {} = body_json(&body)?;
            rvf_upload::publish(d, &c, &target(n, v)?).await?
        }
        RvfRoute::Yank(n, v) => {
            let (t, at) = at(n, v)?;
            let YankBody { reason } = body_json(&body)?;
            let call = ScopeCall::Yank {
                caller: cw,
                at,
                reason,
            };
            manifest_json(scope(r, t.name.scope(), call).await?)?
        }
        RvfRoute::Unyank(n, v) => {
            let (t, at) = at(n, v)?;
            let call = ScopeCall::Unyank { caller: cw, at };
            manifest_json(scope(r, t.name.scope(), call).await?)?
        }
        RvfRoute::Get(n, v) => {
            let (t, at) = at(n, v)?;
            manifest_json(scope(r, t.name.scope(), ScopeCall::Get { caller: cw, at }).await?)?
        }
        RvfRoute::Blob(n, v) => {
            let (t, at) = at(n, v)?;
            let call = ScopeCall::Pull { caller: cw, at };
            let ScopeOut::Pull { manifest, blob } = scope(r, t.name.scope(), call).await? else {
                return Err(RvfError::unexpected());
            };
            return Ok(Err(RvfReply::Blob {
                key: blob,
                size: manifest.total_size,
                sha256: hex::encode(manifest.sha256),
                yanked: manifest.yanked.is_some(),
            }));
        }
        RvfRoute::Versions(n) => {
            let name = PackageName::parse(n).map_err(|e| RvfError::from(RegistryError::from(e)))?;
            let call = ScopeCall::Versions {
                caller: cw,
                name: name.to_string(),
                cursor: q.cursor.clone(),
                limit: q.limit.unwrap_or(0),
            };
            let ScopeOut::Versions { page } = scope(r, name.scope(), call).await? else {
                return Err(RvfError::unexpected());
            };
            (
                200,
                serde_json::to_value(&page).map_err(|_| RvfError::unexpected())?,
            )
        }
        RvfRoute::Import(coll) => crate::rvf_import::import(d, ctx, &c, coll, &body, now).await?,
    };
    Ok(Ok(out))
}
