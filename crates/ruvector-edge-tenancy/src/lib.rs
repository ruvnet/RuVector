//! # ruvector-edge-tenancy
//!
//! Tenancy model for ruvector edge (ADR-351 §4). The tenant is derived **only**
//! from signature-verified claims ([`TenantContext::from_verified`]) and is
//! always `(upstream_iss, org_id, workspace_id)`, with `upstream_iss` the
//! compiled [`UPSTREAM_ISSUER`]; no API accepts a raw tenant string.
//!
//! - [`names`]: `tenant_key` and Durable Object name derivation (§4.1, §4.3).
//! - [`uid`]: never-reused `collection_uid` allocation (§4.3).
//! - [`shard`]: shard counts and stable `hash(id) mod N` routing (§4.4).
//! - [`meta`]: in-object identity assertion for DOs (§4.3).
//! - [`validate`]: claim-component, edge-subject, collection-name and
//!   vector-id validators.
//! - [`context`]: the [`TenantContext`] type.
//! - [`membership`]: roles and capability = scope ∩ role (§4.2, §5.3).
//! - [`quota`]: quota limits, usage and admission (§10).
//! - [`lease`]: batch quota leases (§6.1).
//! - [`problem`]: RFC 9457 `application/problem+json` bodies (§7).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod context;
pub mod error;
pub mod lease;
pub mod membership;
pub mod meta;
pub mod names;
pub mod problem;
pub mod quota;
pub mod shard;
pub mod uid;
pub mod validate;

pub use context::{TenantContext, UPSTREAM_ISSUER};
pub use error::TenancyError;
pub use lease::{grant_lease, reconcile, LeaseAmount, LeaseError, QuotaLease};
pub use membership::{effective_capabilities, Role};
pub use meta::{DoMeta, IdentityCheck, LedgerMeta};
pub use names::{derive_tenant_key, do_name, ledger_do_name, DoName, Service, TenantKey};
pub use problem::{Problem, ProblemCode};
pub use quota::{admit, QuotaDelta, QuotaError, QuotaLimits, Usage};
pub use shard::{shard_for, ShardCount, ShardIndex, MAX_SHARDS};
pub use uid::{CollectionUid, EntropySource, UidAllocator};
pub use validate::{
    validate_claim_component, validate_edge_subject, validate_edge_token_id, validate_issuer,
    validate_subject, CollectionName, VectorId,
};
