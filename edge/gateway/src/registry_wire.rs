//! Wire between the gateway Worker and the two registry Durable Object
//! classes (ADR-351 §3 rv-registry, M5): `RegistryRoot` (the global scope
//! directory, one object) and `RegistryScope` (one registry index per
//! `@scope`, named `keys::registry_do_name(scope)`).
//!
//! The DOs are reachable only through their bindings from this Worker, so
//! the caller identity in a request is the one the Worker derived from a
//! verified token and the caller's own ledger (as with `TenantLedger`).
//! Every reply is `{"Ok": …}` or `{"Err": RvfError}`.

use crate::rest::ApiReply;
use ruvector_edge_auth::{prm, Capability, CapabilitySet};
use ruvector_edge_registry::registry::{Page, VersionSummary};
use ruvector_edge_registry::upload::{ObjectEvidence, PartRecord};
use ruvector_edge_registry::{
    BeginUpload, Caller, ManifestView, Provenance, RegistryError, ScopeClaim, ValidatedRvf,
};
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::problem::PROBLEM_CONTENT_TYPE;
use ruvector_edge_tenancy::{Problem, ProblemCode, TenantKey};
use serde::{Deserialize, Serialize};

/// A registry failure as it crosses the DO boundary and reaches the client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RvfError {
    /// §7 problem code.
    pub code: String,
    /// HTTP status.
    pub status: u16,
    /// Safe detail (never payload bytes).
    pub detail: String,
    /// For `insufficient_scope`: the scope that satisfies it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_up: Option<String>,
    /// The DO refused because the Worker reported a validation failure
    /// (the Worker substitutes its own, precise validation error).
    #[serde(default)]
    pub validation: bool,
}

impl RvfError {
    /// From a problem code and detail.
    pub fn new(code: ProblemCode, detail: impl Into<String>) -> Self {
        let (status, c) = code.status_and_code();
        RvfError {
            code: c.to_string(),
            status,
            detail: detail.into(),
            step_up: None,
            validation: false,
        }
    }

    /// Transport failure or undecodable reply.
    pub fn unavailable() -> Self {
        RvfError::new(ProblemCode::ShardUnavailable, "registry unavailable")
    }

    /// Unexpected reply shape.
    pub fn unexpected() -> Self {
        RvfError::new(ProblemCode::ServerError, "unexpected registry reply")
    }

    /// Storage (R2) failure.
    pub fn storage() -> Self {
        RvfError::new(ProblemCode::ShardUnavailable, "object storage unavailable")
    }

    /// The RFC 9457 reply, with the §5.3 step-up challenge naming
    /// `metadata_url` for `insufficient_scope`.
    pub fn reply(&self, metadata_url: &str) -> ApiReply {
        let p = Problem {
            type_: "about:blank".into(),
            title: self.code.clone(),
            status: self.status,
            code: self.code.clone(),
            detail: Some(self.detail.clone()),
            request_id: None,
        };
        let www = self.step_up.as_deref().and_then(cap_of_scope).map(|c| {
            prm::www_authenticate_insufficient_scope(metadata_url, &prm::step_up_scope(c))
        });
        ApiReply {
            status: self.status,
            body: p.to_json(),
            content_type: PROBLEM_CONTENT_TYPE,
            www_authenticate: www,
        }
    }
}

/// Every capability, in a fixed order (the wire bit positions).
const CAPS: [Capability; 5] = [
    Capability::Read,
    Capability::Write,
    Capability::CreateCollection,
    Capability::Admin,
    Capability::PublishPublic,
];

/// The capability whose satisfying scope is `scope`.
pub fn cap_of_scope(scope: &str) -> Option<Capability> {
    CAPS.into_iter().find(|c| c.satisfying_scope() == scope)
}

impl From<RegistryError> for RvfError {
    fn from(e: RegistryError) -> Self {
        let mut r = RvfError::new(e.problem_code(), e.to_string());
        r.step_up = e.step_up_scope().map(str::to_string);
        r.validation = matches!(e, RegistryError::Validation(_));
        r
    }
}

impl From<OpError> for RvfError {
    fn from(e: OpError) -> Self {
        RvfError {
            code: e.code.as_str().to_string(),
            status: e.code.status(),
            detail: e.detail.to_string(),
            step_up: (e.code == ErrorCode::InsufficientScope)
                .then(|| e.scope.map(str::to_string))
                .flatten(),
            validation: false,
        }
    }
}

/// A registry [`Caller`] on the wire (capabilities as a bit set over
/// `Capability::ALL`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallerWire {
    /// Tenant key.
    pub tenant_key: String,
    /// Edge subject.
    pub sub: String,
    /// Effective capabilities (scope ∩ role).
    pub caps: u8,
}

impl From<&Caller> for CallerWire {
    fn from(c: &Caller) -> Self {
        let caps = CAPS
            .iter()
            .enumerate()
            .filter(|(_, cap)| c.caps.contains(**cap))
            .fold(0u8, |acc, (i, _)| acc | (1 << i));
        CallerWire {
            tenant_key: c.tenant.as_str().to_string(),
            sub: c.sub.clone(),
            caps,
        }
    }
}

impl CallerWire {
    /// Back to a [`Caller`] (the tenant key is strictly re-parsed).
    pub fn caller(&self) -> Result<Caller, RvfError> {
        let tenant = TenantKey::parse(&self.tenant_key)
            .map_err(|_| RvfError::new(ProblemCode::InvalidRequest, "tenant key"))?;
        let mut caps = CapabilitySet::EMPTY;
        for (i, c) in CAPS.iter().enumerate() {
            if self.caps & (1 << i) != 0 {
                caps.insert(*c);
            }
        }
        Ok(Caller {
            tenant,
            sub: self.sub.clone(),
            caps,
        })
    }
}

/// `RegistryRoot` calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum RootCall {
    /// Claim a scope (Admin).
    Claim {
        /// Caller.
        caller: CallerWire,
        /// Scope (without `@`).
        scope: String,
    },
    /// Scopes held by the caller's tenant.
    Scopes {
        /// Caller.
        caller: CallerWire,
    },
}

/// `RegistryRoot` results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum RootOut {
    /// An accepted (or existing, same-tenant) claim.
    Claim {
        /// The claim.
        claim: ScopeClaim,
    },
    /// The tenant's scopes.
    Scopes {
        /// Scope names.
        scopes: Vec<String>,
    },
}

/// Package coordinates from the request path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coords {
    /// `@scope/name`.
    pub name: String,
    /// Version.
    pub version: String,
}

/// What the Worker reports after streaming the frozen parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportWire {
    /// `Ok` summary, or `None` when validation failed.
    pub validated: Option<ValidatedRvf>,
    /// Parts as measured on this read.
    pub streamed: Vec<PartRecord>,
    /// Measured blob object, when written.
    pub copied: Option<ObjectEvidence>,
    /// Optional witness root.
    pub provenance: Option<Provenance>,
}

/// `RegistryScope` calls. Upload calls carry the path's coordinates and
/// refuse (404) a session opened for another name or version.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum ScopeCall {
    /// Record a directory claim in this scope's index.
    Adopt {
        /// The claim.
        claim: ScopeClaim,
    },
    /// Open an upload session.
    Begin {
        /// Caller.
        caller: CallerWire,
        /// Declaration.
        req: BeginUpload,
    },
    /// Remember the R2 multipart upload behind a session's staging object.
    Attach {
        /// Caller.
        caller: CallerWire,
        /// Session.
        upload_id: String,
        /// R2 multipart upload id.
        multipart: String,
    },
    /// Where part `number` goes (session open, part number in range).
    PartTarget {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
        /// Session.
        upload_id: String,
        /// Part number.
        number: u32,
    },
    /// Record a measured part and its R2 etag.
    RecordPart {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
        /// Session.
        upload_id: String,
        /// Measured part.
        part: PartRecord,
        /// R2 part number the attempt was uploaded to.
        r2_part: u16,
        /// R2 etag of the part.
        etag: String,
    },
    /// Freeze and plan the finalize.
    Plan {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
        /// Session.
        upload_id: String,
    },
    /// Commit the version.
    Finalize {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
        /// Session.
        upload_id: String,
        /// What the Worker measured.
        report: ReportWire,
    },
    /// Manifest view.
    Get {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
    },
    /// Manifest view plus the blob key.
    Pull {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
    },
    /// Versions of a package, paginated.
    Versions {
        /// Caller.
        caller: CallerWire,
        /// `@scope/name`.
        name: String,
        /// Cursor.
        cursor: Option<String>,
        /// Page size.
        limit: usize,
    },
    /// Plan a public publish.
    PublishPlan {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
    },
    /// Commit a public publish with the measured public object.
    PublishCommit {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
        /// Measured object at the public key.
        evidence: ObjectEvidence,
    },
    /// Yank.
    Yank {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
        /// Reason.
        reason: String,
    },
    /// Unyank.
    Unyank {
        /// Caller.
        caller: CallerWire,
        /// Coordinates.
        at: Coords,
    },
}

/// A frozen finalize plan plus the R2 multipart state of its staging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanWire {
    /// Staging object key.
    pub staging: String,
    /// R2 multipart upload of the staging object.
    pub multipart: Option<String>,
    /// Frozen parts, in order, each with its recorded (R2 part number,
    /// etag).
    pub parts: Vec<(PartRecord, Option<(u16, String)>)>,
    /// Blob key.
    pub blob: String,
    /// Write the blob.
    pub copy_required: bool,
    /// Declared size.
    pub size: u64,
    /// Declared SHA-256 (hex).
    pub sha256: String,
}

/// A public publish plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishPlanWire {
    /// Tenant blob.
    pub from: String,
    /// Public blob (create-only).
    pub to: String,
    /// Expected SHA-256 (hex).
    pub sha256: String,
    /// Expected size.
    pub size: u64,
    /// Nothing to copy.
    pub already_public: bool,
}

/// `RegistryScope` results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum ScopeOut {
    /// Scope adopted.
    Adopted,
    /// Session opened.
    Session {
        /// Upload id.
        upload_id: String,
        /// Staging object key.
        staging: String,
        /// Expiry (seconds).
        expires_at: u64,
    },
    /// Done, nothing to return.
    Done,
    /// Part target.
    PartTarget {
        /// Staging object key.
        staging: String,
        /// R2 multipart upload.
        multipart: String,
        /// R2 part number of this attempt.
        r2_part: u16,
    },
    /// Frozen plan.
    Plan {
        /// Plan.
        plan: PlanWire,
    },
    /// A manifest view.
    Manifest {
        /// View.
        manifest: Box<ManifestView>,
    },
    /// A manifest view and where its bytes are.
    Pull {
        /// View.
        manifest: Box<ManifestView>,
        /// R2 key.
        blob: String,
    },
    /// A page of versions.
    Versions {
        /// Page.
        page: Page<VersionSummary>,
    },
    /// Publish plan.
    PublishPlan {
        /// Plan.
        plan: PublishPlanWire,
    },
}

/// A decoded reply.
pub type Reply<T> = Result<T, RvfError>;

/// Encode a reply.
pub fn encode<T: Serialize>(r: &Reply<T>) -> String {
    serde_json::to_string(r).unwrap_or_else(|_| {
        serde_json::to_string(&Reply::<()>::Err(RvfError::unexpected())).unwrap_or_default()
    })
}

/// Decode a reply (`unavailable` if undecodable).
pub fn decode<T: for<'de> Deserialize<'de>>(text: &str) -> Reply<T> {
    serde_json::from_str::<Reply<T>>(text).unwrap_or_else(|_| Err(RvfError::unavailable()))
}
