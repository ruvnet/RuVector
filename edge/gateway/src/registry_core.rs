//! Pure cores of the two registry Durable Objects (ADR-351 §3 rv-registry,
//! M5): request body in, reply text out, over a `SqlKv` index. The DO
//! classes (`registry_do`) only read the body, run a core synchronously
//! (one call, one coalesced commit) and reply; native tests run the same
//! cores over `MemSqlStore`.
//!
//! Besides the registry crate's index, `RegistryScope` keeps gateway
//! records under `gw/`: per session the R2 multipart upload of the staging
//! object (`registry_upload_core`), and the sweep's durable queue of R2
//! aborts and deletes (`registry_pending`).

use crate::registry_kv::{EntropyRef, SqlKv};
use crate::registry_upload_core::upload_call;
use crate::registry_wire::{
    decode, encode, CallerWire, Coords, PublishPlanWire, Reply, RootCall, RootOut, RvfError,
    ScopeCall, ScopeOut,
};
use ruvector_edge_auth::Clock;
use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_registry::registry::{PageRequest, RegistryConfig};
use ruvector_edge_registry::upload::{UploadId, UploadSession};
use ruvector_edge_registry::{
    PackageName, Registry, RegistryError, Scope, ScopeDirectory, Version,
};
use ruvector_edge_store::SqlStore;
use ruvector_edge_tenancy::{EntropySource, ProblemCode};

/// Scope cap per tenant (ADR-351 §3).
pub const MAX_SCOPES_PER_TENANT: usize = 10;

/// Registry configuration the gateway runs with: the crate defaults, but
/// parts of at most 8 MiB, because the Worker holds one part in memory
/// (JS plus wasm copy) while receiving or hashing it: the same 8 MiB
/// transfer size as `uploads::PART_BYTES`, which keeps the isolate budget
/// (`shard_core`) within 128 MB on Workers Paid (16 MiB before; R2
/// multipart minimum stays 5 MiB), and at most 1000
/// logical parts, so each can take `ATTEMPTS_PER_PART` distinct R2 part
/// numbers within R2's 10 000 (512 MiB needs at least 64 parts).
pub fn gateway_config() -> RegistryConfig {
    let mut c = RegistryConfig::default();
    c.upload.max_part_size = 8 << 20;
    c.upload.max_parts = 1000;
    c
}

pub(crate) type Reg<'a, S, C> = Registry<&'a SqlKv<S>, &'a C, EntropyRef<'a>>;

pub(crate) fn err(e: RegistryError) -> RvfError {
    e.into()
}

fn bad(detail: &'static str) -> RvfError {
    RvfError::new(ProblemCode::InvalidRequest, detail)
}

fn not_found() -> RvfError {
    err(RegistryError::NotFound)
}

/// Serve one `RegistryRoot` request (the scope directory).
pub fn serve_root<S: SqlStore, C: Clock>(sql: S, clock: &C, body: &[u8]) -> String {
    let run = || -> Reply<RootOut> {
        let kv = SqlKv::open(sql).map_err(|e| err(e.into()))?;
        let dir = ScopeDirectory::new(&kv, clock, MAX_SCOPES_PER_TENANT);
        let call: RootCall = serde_json::from_slice(body).map_err(|_| bad("malformed call"))?;
        match call {
            RootCall::Claim { caller, scope } => {
                let scope = Scope::parse(&scope).map_err(|e| err(e.into()))?;
                let claim = dir.claim(&caller.caller()?, &scope).map_err(err)?;
                Ok(RootOut::Claim { claim })
            }
            RootCall::Scopes { caller } => {
                let c = caller.caller()?;
                let scopes = dir.scopes_of(&c.tenant).map_err(err)?;
                Ok(RootOut::Scopes { scopes })
            }
        }
    };
    encode(&run())
}

pub(crate) fn coords(at: &Coords) -> Result<(PackageName, Version), RvfError> {
    let n = PackageName::parse(&at.name).map_err(|e| err(e.into()))?;
    let v = Version::parse(&at.version).map_err(|e| err(e.into()))?;
    Ok((n, v))
}

/// Load the caller's session and check it belongs to `at` (else 404).
pub(crate) fn session<S: SqlStore, C: Clock>(
    reg: &Reg<'_, S, C>,
    caller: &CallerWire,
    at: &Coords,
    id: &str,
) -> Result<(ruvector_edge_registry::Caller, UploadId, UploadSession), RvfError> {
    let c = caller.caller()?;
    let id = UploadId::parse(id).map_err(err)?;
    let (n, v) = coords(at)?;
    let s = reg.upload(&c, &id).map_err(err)?;
    if s.target.name != n || s.target.version != v {
        return Err(not_found());
    }
    Ok((c, id, s))
}

/// Outcome of one `RegistryScope` request.
pub struct Served {
    /// Reply text.
    pub body: String,
    /// Sessions or released blobs are pending: keep a sweep alarm armed.
    pub maintenance: bool,
}

/// Serve one `RegistryScope` request.
pub fn serve_scope<S: SqlStore, C: Clock>(
    sql: S,
    clock: &C,
    entropy: &dyn EntropySource,
    cfg: RegistryConfig,
    body: &[u8],
) -> Served {
    let kv = match SqlKv::open(sql) {
        Ok(kv) => kv,
        Err(e) => {
            return Served {
                body: encode::<ScopeOut>(&Err(err(e.into()))),
                maintenance: false,
            }
        }
    };
    let reg = Registry::new(&kv, clock, EntropyRef(entropy), cfg);
    let out = serde_json::from_slice::<ScopeCall>(body)
        .map_err(|_| bad("malformed call"))
        .and_then(|call| scope_call(&reg, &kv, clock.now_unix(), call));
    Served {
        body: encode(&out),
        maintenance: pending(&kv),
    }
}

/// `true` while sessions or released blobs remain for the sweep.
pub fn pending<K: KvStore>(kv: &K) -> bool {
    [
        "upload/",
        "gc/",
        crate::registry_pending::PD,
        crate::registry_pending::PA,
    ]
    .iter()
    .any(|p| kv.list(p, None, 1).is_ok_and(|r| !r.is_empty()))
}

pub(crate) fn view_out(
    m: ruvector_edge_registry::PackageManifest,
    c: &ruvector_edge_registry::Caller,
) -> ScopeOut {
    ScopeOut::Manifest {
        manifest: Box::new(ruvector_edge_registry::ManifestView::for_caller(&m, c)),
    }
}

fn scope_call<S: SqlStore, C: Clock>(
    reg: &Reg<'_, S, C>,
    kv: &SqlKv<S>,
    now: u64,
    call: ScopeCall,
) -> Reply<ScopeOut> {
    match call {
        ScopeCall::Adopt { claim } => {
            reg.adopt_scope(&claim).map_err(err)?;
            Ok(ScopeOut::Adopted)
        }
        call @ (ScopeCall::Begin { .. }
        | ScopeCall::Attach { .. }
        | ScopeCall::PartTarget { .. }
        | ScopeCall::RecordPart { .. }
        | ScopeCall::Plan { .. }
        | ScopeCall::Finalize { .. }) => {
            upload_call(reg, kv, now, call).unwrap_or_else(|| Err(RvfError::unexpected()))
        }
        ScopeCall::Get { caller, at } => {
            let c = caller.caller()?;
            let (n, v) = coords(&at)?;
            let manifest = Box::new(reg.get(&c, &n, &v).map_err(err)?);
            Ok(ScopeOut::Manifest { manifest })
        }
        ScopeCall::Pull { caller, at } => {
            let c = caller.caller()?;
            let (n, v) = coords(&at)?;
            let t = reg.pull(&c, &n, &v).map_err(err)?;
            Ok(ScopeOut::Pull {
                manifest: Box::new(t.manifest),
                blob: t.blob.to_string(),
            })
        }
        ScopeCall::Versions {
            caller,
            name,
            cursor,
            limit,
        } => {
            let c = caller.caller()?;
            let n = PackageName::parse(&name).map_err(|e| err(e.into()))?;
            let page = reg
                .list_versions(&c, &n, &PageRequest { cursor, limit })
                .map_err(err)?;
            Ok(ScopeOut::Versions { page })
        }
        ScopeCall::PublishPlan { caller, at } => {
            let c = caller.caller()?;
            let (n, v) = coords(&at)?;
            let p = reg.publish_plan(&c, &n, &v).map_err(err)?;
            Ok(ScopeOut::PublishPlan {
                plan: PublishPlanWire {
                    from: p.from.to_string(),
                    to: p.to.to_string(),
                    sha256: hex::encode(p.sha256),
                    size: p.size,
                    already_public: p.already_public,
                },
            })
        }
        ScopeCall::PublishCommit {
            caller,
            at,
            evidence,
        } => {
            let c = caller.caller()?;
            let (n, v) = coords(&at)?;
            // A released tenant blob is queued under `gc/` for the sweep;
            // it is never deleted here.
            let out = reg.publish_commit(&c, &n, &v, evidence).map_err(err)?;
            Ok(view_out(out.manifest, &c))
        }
        ScopeCall::Yank { caller, at, reason } => {
            let c = caller.caller()?;
            let (n, v) = coords(&at)?;
            let m = reg.yank(&c, &n, &v, &reason).map_err(err)?;
            Ok(view_out(m, &c))
        }
        ScopeCall::Unyank { caller, at } => {
            let c = caller.caller()?;
            let (n, v) = coords(&at)?;
            let m = reg.unyank(&c, &n, &v).map_err(err)?;
            Ok(view_out(m, &c))
        }
    }
}

/// Decode a root reply.
pub fn root_reply(text: &str) -> Reply<RootOut> {
    decode(text)
}

/// Decode a scope reply.
pub fn scope_reply(text: &str) -> Reply<ScopeOut> {
    decode(text)
}
