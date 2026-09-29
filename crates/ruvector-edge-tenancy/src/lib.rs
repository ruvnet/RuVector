//! # ruvector-edge-tenancy
//!
//! Tenancy model for ruvector edge (ADR-351 §4). The tenant is derived **only**
//! from signature-verified claims ([`TenantContext::from_verified`]) and is
//! always `(iss, org_id, workspace_id)`; no API accepts a raw tenant string.
//!
//! - [`names`]: `tenant_key` and Durable Object name derivation (§4.1, §4.2).
//! - [`validate`]: claim-component, collection-name and vector-id validators.
//! - [`context`]: the [`TenantContext`] type.
//! - [`quota`]: quota limits, usage and admission (§10).
//! - [`problem`]: RFC 9457 `application/problem+json` bodies (§7).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod context;
pub mod error;
pub mod names;
pub mod problem;
pub mod quota;
pub mod validate;

pub use context::TenantContext;
pub use error::TenancyError;
pub use names::{derive_tenant_key, do_name, ledger_do_name, DoName, Service, TenantKey};
pub use problem::{Problem, ProblemCode};
pub use quota::{admit, QuotaDelta, QuotaError, QuotaLimits, Usage};
pub use validate::{validate_claim_component, CollectionName, VectorId};
