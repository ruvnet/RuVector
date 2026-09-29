//! Worker side of push and public publish (ADR-351 §3 rv-registry, M5),
//! following the registry crate's contracts:
//!
//! - **begin**: the scope DO opens the session; the Worker starts an R2
//!   multipart upload at the session's staging key and attaches its id.
//! - **part**: the Worker measures the part (size, SHA-256) from the bytes
//!   it received — never from the client — uploads it to the R2 multipart
//!   part number the DO allotted to this attempt (unique per attempt, see
//!   `registry_upload_core`), then records the measurement, that number and
//!   R2's etag.
//! - **finalize**: `rvf_finalize`.
//! - **publish**: create-only write of the global public key (or read back
//!   the existing object), measure it, commit with that evidence.

use crate::registry_ports::{scope, BlobStore, RegistryRpc};
use crate::registry_wire::{CallerWire, Coords, RvfError, ScopeCall, ScopeOut};
use ruvector_edge_registry::registry::RegistryConfig;
use ruvector_edge_registry::upload::{ObjectEvidence, PartRecord};
use ruvector_edge_registry::{
    BeginUpload, Caller, PackageName, RegistryError, Version, Visibility,
};
use ruvector_edge_tenancy::ProblemCode;
use serde::Deserialize;
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

/// Everything a registry flow needs.
pub struct Deps<'a, B, R, S> {
    /// Ledger / shard transport (collections, access).
    pub b: &'a B,
    /// Registry DO transport.
    pub r: &'a R,
    /// R2.
    pub s: &'a S,
    /// Registry configuration (limits).
    pub cfg: RegistryConfig,
    /// Largest upload finalized in one request; larger ones are finalized
    /// in steps (`rvf_finalize_step::INLINE_FINALIZE_BYTES`).
    pub inline_finalize: u64,
}

/// A package version addressed by the request path.
pub struct Target {
    /// `@scope/name`.
    pub name: PackageName,
    /// Version.
    pub version: Version,
}

impl Target {
    /// Wire coordinates.
    pub fn coords(&self) -> Coords {
        Coords {
            name: self.name.to_string(),
            version: self.version.to_string(),
        }
    }
}

pub(crate) fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

pub(crate) fn hex32(s: &str) -> Result<[u8; 32], RvfError> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(s, &mut out)
        .map_err(|_| RvfError::new(ProblemCode::InvalidRequest, "sha256 must be 64 hex chars"))?;
    Ok(out)
}

pub(crate) fn body_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, RvfError> {
    let raw: &[u8] = if body.iter().all(u8::is_ascii_whitespace) {
        b"{}"
    } else {
        body
    };
    serde_json::from_slice(raw)
        .map_err(|_| RvfError::new(ProblemCode::InvalidRequest, "malformed body"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BeginBody {
    #[serde(default = "tenant_visibility")]
    visibility: Visibility,
    size: u64,
    sha256: String,
}

fn tenant_visibility() -> Visibility {
    Visibility::Tenant
}

/// `POST /v1/rvf/{scope}/{name}/{version}/uploads`.
pub async fn begin<B, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    c: &Caller,
    t: &Target,
    body: &[u8],
) -> Result<(u16, Json), RvfError> {
    let b: BeginBody = body_json(body)?;
    let req = BeginUpload {
        name: t.name.clone(),
        version: t.version.clone(),
        visibility: b.visibility,
        size: b.size,
        sha256: hex32(&b.sha256)?,
    };
    let cw = CallerWire::from(c);
    let call = ScopeCall::Begin {
        caller: cw.clone(),
        req,
    };
    let ScopeOut::Session {
        upload_id,
        staging,
        expires_at,
    } = scope(d.r, t.name.scope(), call).await?
    else {
        return Err(RvfError::unexpected());
    };
    let multipart = d.s.mp_create(&staging).await?;
    let attach = ScopeCall::Attach {
        caller: cw,
        upload_id: upload_id.clone(),
        multipart: multipart.clone(),
    };
    if let Err(e) = scope(d.r, t.name.scope(), attach).await {
        // The session has no multipart on record, so nothing else would
        // ever abort this one: abort it now (best effort).
        let _ = d.s.mp_abort(&staging, &multipart).await;
        return Err(e);
    }
    let lim = d.cfg.upload;
    Ok((
        201,
        json!({
            "upload_id": upload_id,
            "expires_at": expires_at,
            "max_parts": lim.max_parts,
            "min_part_size": lim.min_part_size,
            "max_part_size": lim.max_part_size,
        }),
    ))
}

/// `PUT …/uploads/{upload_id}/parts/{n}` with the part's bytes (already
/// read under the `max_part_size` cap).
pub async fn put_part<B, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    c: &Caller,
    t: &Target,
    upload_id: &str,
    n: &str,
    bytes: Vec<u8>,
) -> Result<(u16, Json), RvfError> {
    let number: u32 = n
        .parse()
        .ok()
        .filter(|v| (1..=u32::from(u16::MAX)).contains(v))
        .ok_or_else(|| RvfError::from(RegistryError::UploadLimit("part number out of range")))?;
    if bytes.is_empty() || bytes.len() as u64 > d.cfg.upload.max_part_size {
        return Err(RegistryError::UploadLimit("part size out of range").into());
    }
    let cw = CallerWire::from(c);
    let call = ScopeCall::PartTarget {
        caller: cw.clone(),
        at: t.coords(),
        upload_id: upload_id.to_string(),
        number,
    };
    let ScopeOut::PartTarget {
        staging,
        multipart,
        r2_part,
    } = scope(d.r, t.name.scope(), call).await?
    else {
        return Err(RvfError::unexpected());
    };
    let part = PartRecord {
        number,
        size: bytes.len() as u64,
        sha256: sha(&bytes),
    };
    let etag = d.s.mp_part(&staging, &multipart, r2_part, bytes).await?;
    let record = ScopeCall::RecordPart {
        caller: cw,
        at: t.coords(),
        upload_id: upload_id.to_string(),
        part: part.clone(),
        r2_part,
        etag,
    };
    scope(d.r, t.name.scope(), record).await?;
    Ok((
        200,
        json!({ "number": number, "size": part.size, "sha256": hex::encode(part.sha256) }),
    ))
}

/// `POST /v1/rvf/{scope}/{name}/{version}:publish` (`ruvector:publish`,
/// `/v1` only). The public key is global and write-once: it is written
/// create-only, and whatever object is there is measured and must match.
pub async fn publish<B, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    c: &Caller,
    t: &Target,
) -> Result<(u16, Json), RvfError> {
    let cw = CallerWire::from(c);
    let call = ScopeCall::PublishPlan {
        caller: cw.clone(),
        at: t.coords(),
    };
    let ScopeOut::PublishPlan { plan } = scope(d.r, t.name.scope(), call).await? else {
        return Err(RvfError::unexpected());
    };
    let sha256 = hex32(&plan.sha256)?;
    let evidence = if plan.already_public {
        // Idempotent re-publish: the commit does not look at evidence.
        ObjectEvidence {
            sha256,
            size: plan.size,
        }
    } else {
        if d.s.size(&plan.to).await?.is_none() {
            d.s.copy_create_only(&plan.from, &plan.to, sha256).await?;
        }
        // An absent object yields evidence that can never match.
        d.s.measure(&plan.to).await?.unwrap_or(ObjectEvidence {
            sha256: [0; 32],
            size: 0,
        })
    };
    let call = ScopeCall::PublishCommit {
        caller: cw,
        at: t.coords(),
        evidence,
    };
    match scope(d.r, t.name.scope(), call).await? {
        ScopeOut::Manifest { manifest } => Ok((
            200,
            serde_json::to_value(&manifest).map_err(|_| RvfError::unexpected())?,
        )),
        _ => Err(RvfError::unexpected()),
    }
}
