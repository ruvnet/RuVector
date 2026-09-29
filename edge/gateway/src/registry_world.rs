//! Native harness for the registry routes: the real ledger / shard cores
//! ([`MemBackend`]), the real registry DO cores ([`MemRegistry`]) and an
//! in-memory R2 ([`MemR2`]), driven through `registry_routes::handle`
//! exactly as the Worker glue drives it. RVF fixtures are written by the
//! rvf CLI's engine (`rvf-runtime`).

use crate::backend::mem::MemBackend;
use crate::registry_mem::{MemR2, MemRegistry};
use crate::registry_routes::{self, Query, RvfReply};
use crate::rvf_upload::Deps;
use crate::service;
use crate::testkit::*;
use ruvector_edge_auth::Clock;
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_store::CallerContext;
use ruvector_edge_tenancy::{Role, TenantKey};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};
use worker::Method;

/// Every capability (an owner's full token).
pub fn all_caps() -> CapabilitySet {
    caps(&[
        Capability::Read,
        Capability::Write,
        Capability::CreateCollection,
        Capability::Admin,
        Capability::PublishPublic,
    ])
}

/// The three ports.
pub struct World {
    /// Ledger + shards.
    pub b: MemBackend,
    /// Registry DOs.
    pub r: MemRegistry,
    /// R2.
    pub s: MemR2,
}

impl World {
    /// Empty world at `T0`.
    pub fn new() -> Self {
        World {
            b: MemBackend::new(),
            r: MemRegistry::at(T0),
            s: MemR2::default(),
        }
    }

    /// The flows' dependencies.
    pub fn deps(&self) -> Deps<'_, MemBackend, MemRegistry, MemR2> {
        Deps {
            b: &self.b,
            r: &self.r,
            s: &self.s,
            cfg: self.r.cfg(),
        }
    }

    /// Owner of tenant `org` (claims it).
    pub fn owner(&self, org: &str, user: &str) -> CallerContext {
        let c = ctx(&tenant(org), user, all_caps());
        block_on(service::claim(&call(&self.b, &c))).unwrap();
        c
    }

    /// Member `user` of `owner`'s tenant with `role` and token `scope`.
    pub fn member(
        &self,
        owner: &CallerContext,
        user: &str,
        role: Role,
        scope: CapabilitySet,
    ) -> CallerContext {
        let t: TenantKey = owner.tenant_key().clone();
        self.b
            .invite(&t, owner.sub(), &sub(user), role, T0)
            .unwrap();
        ctx(&t, user, scope)
    }

    /// Serve one registry request.
    pub fn call(&self, c: &CallerContext, m: Method, path: &str, body: &[u8]) -> RvfReply {
        let route = registry_routes::parse(&m, path).unwrap_or_else(|| panic!("route {path}"));
        let now = self.r.clock.now_unix();
        let d = self.deps();
        block_on(registry_routes::handle(
            &d,
            c,
            &route,
            body.to_vec(),
            &Query::default(),
            now,
            REST_MD,
        ))
    }

    /// Serve one request expecting JSON.
    pub fn json(&self, c: &CallerContext, m: Method, path: &str, body: Json) -> (u16, Json) {
        let bytes = if body.is_null() {
            Vec::new()
        } else {
            body.to_string().into_bytes()
        };
        match self.call(c, m, path, &bytes) {
            RvfReply::Api(r) => (
                r.status,
                serde_json::from_str(&r.body).unwrap_or(Json::Null),
            ),
            RvfReply::Blob { .. } => panic!("unexpected blob reply"),
        }
    }

    /// Claim `scope` for `c`'s tenant.
    pub fn claim(&self, c: &CallerContext, scope: &str) -> (u16, Json) {
        self.json(
            c,
            Method::Post,
            &format!("/v1/rvf/scopes/{scope}"),
            Json::Null,
        )
    }

    /// Begin an upload of `bytes` to `pkg` (`acme/x`).
    pub fn begin(
        &self,
        c: &CallerContext,
        pkg: &str,
        v: &str,
        vis: &str,
        bytes: &[u8],
    ) -> (u16, Json) {
        let body =
            json!({ "visibility": vis, "size": bytes.len(), "sha256": hex::encode(sha(bytes)) });
        self.json(c, Method::Post, &format!("/v1/rvf/{pkg}/{v}/uploads"), body)
    }

    /// Upload `bytes` as `parts` uniform parts.
    pub fn parts(
        &self,
        c: &CallerContext,
        pkg: &str,
        v: &str,
        id: &str,
        bytes: &[u8],
        parts: usize,
    ) {
        let chunk = bytes.len().div_ceil(parts.max(1)).max(1);
        for (i, part) in bytes.chunks(chunk).enumerate() {
            let path = format!("/v1/rvf/{pkg}/{v}/uploads/{id}/parts/{}", i + 1);
            match self.call(c, Method::Put, &path, part) {
                RvfReply::Api(r) => assert_eq!(r.status, 200, "{}", r.body),
                RvfReply::Blob { .. } => panic!("blob"),
            }
        }
    }

    /// Finalize.
    pub fn finalize(&self, c: &CallerContext, pkg: &str, v: &str, id: &str) -> (u16, Json) {
        self.json(
            c,
            Method::Post,
            &format!("/v1/rvf/{pkg}/{v}/uploads/{id}:finalize"),
            Json::Null,
        )
    }

    /// Begin, upload in `parts` parts, finalize.
    pub fn push(
        &self,
        c: &CallerContext,
        pkg: &str,
        v: &str,
        vis: &str,
        bytes: &[u8],
        parts: usize,
    ) -> (u16, Json) {
        let (s, b) = self.begin(c, pkg, v, vis, bytes);
        assert_eq!(s, 201, "{b}");
        let id = b["upload_id"].as_str().unwrap().to_string();
        self.parts(c, pkg, v, &id, bytes, parts);
        self.finalize(c, pkg, v, &id)
    }

    /// Pull: the blob bytes, or the problem status.
    pub fn pull(&self, c: &CallerContext, pkg: &str, v: &str) -> Result<(Vec<u8>, bool), u16> {
        match self.call(c, Method::Get, &format!("/v1/rvf/{pkg}/{v}/blob"), b"") {
            RvfReply::Blob {
                key,
                sha256,
                yanked,
                ..
            } => {
                let bytes = self.s.get(&key).expect("blob object");
                assert_eq!(hex::encode(sha(&bytes)), sha256);
                Ok((bytes, yanked))
            }
            RvfReply::Api(r) => Err(r.status),
        }
    }
}

/// SHA-256.
pub fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

/// Deterministic rows.
pub fn rows(n: u64, dim: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
    let mut rng = Rng(seed);
    (0..n).map(|i| (i, rng.vec(dim))).collect()
}

/// An `.rvf` file as the rvf CLI writes it: `n` vectors of `dim`, cosine,
/// ingested in batches of 64, then `deleted` removed. Returns the bytes and
/// the file (kept for the engine's own queries).
pub fn rvf_export(n: u64, dim: u16, seed: u64, deleted: &[u64]) -> (Vec<u8>, tempfile::TempDir) {
    use rvf_runtime::options::{DistanceMetric, RvfOptions};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("export.rvf");
    let mut store = rvf_runtime::RvfStore::create(
        &path,
        RvfOptions {
            dimension: dim,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .unwrap();
    let data = rows(n, dim as usize, seed);
    for chunk in data.chunks(64) {
        let vecs: Vec<&[f32]> = chunk.iter().map(|(_, v)| v.as_slice()).collect();
        let ids: Vec<u64> = chunk.iter().map(|(i, _)| *i).collect();
        store.ingest_batch(&vecs, &ids, None).unwrap();
    }
    if !deleted.is_empty() {
        store.delete(deleted).unwrap();
    }
    store.close().unwrap();
    (std::fs::read(&path).unwrap(), dir)
}
