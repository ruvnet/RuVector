//! Endpoint logic of the edge AS, generic over the ports and returning
//! framework-neutral [`Reply`] values. The Durable Object calls these with
//! [`crate::sql_ports::SqlPorts`] and the Worker platform ports; native tests
//! call them with in-memory SQLite, a fixed clock and a sequential RNG.

pub mod authorize;
pub mod callback;
pub mod consent;
pub mod register;
pub mod token;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

use crate::config::AuthConfig;
use crate::http::Reply;
use crate::sql::SqlExec;
use crate::sql_ports::SqlPorts;
use ruvector_edge_authz::federation::UpstreamFlowState;
use ruvector_edge_authz::{
    AssertionReplayStore, AuthorizeError, ClientStore, Clock, CodeStore, FederationStore,
    RefreshStore, Rng, Signer, StoreError,
};

/// Housekeeping operations the endpoints need beyond the authz ports.
pub trait AdminStore {
    /// Registered-client count (DCR cap).
    fn client_count(&self) -> Result<u64, StoreError>;
    /// Delete expired one-time rows (best effort).
    fn purge_expired(&self, now: u64) -> Result<(), StoreError>;
    /// Count a registration attempt from `bucket`; the count in the window.
    fn dcr_rate_hit(&self, bucket: &str, now: u64, window: u64) -> Result<u64, StoreError>;
    /// The unexpired flow for `state`, **without** consuming it (consent).
    fn peek_flow(&self, state: &str, now: u64) -> Result<Option<UpstreamFlowState>, StoreError>;
    /// Record a completed token request by `client_id`.
    fn touch_client(&self, client_id: &str, now: u64) -> Result<(), StoreError>;
    /// Delete never-used and idle clients (see [`SqlPorts::purge_idle_clients`]).
    fn purge_idle_clients(
        &self,
        now: u64,
        unused_before: u64,
        idle_before: u64,
    ) -> Result<(), StoreError>;
}

impl<D: SqlExec> AdminStore for SqlPorts<D> {
    fn client_count(&self) -> Result<u64, StoreError> {
        SqlPorts::client_count(self)
    }
    fn purge_expired(&self, now: u64) -> Result<(), StoreError> {
        SqlPorts::purge_expired(self, now)
    }
    fn dcr_rate_hit(&self, bucket: &str, now: u64, window: u64) -> Result<u64, StoreError> {
        SqlPorts::dcr_rate_hit(self, bucket, now, window)
    }
    fn touch_client(&self, client_id: &str, now: u64) -> Result<(), StoreError> {
        SqlPorts::touch_client(self, client_id, now)
    }
    fn peek_flow(&self, state: &str, now: u64) -> Result<Option<UpstreamFlowState>, StoreError> {
        SqlPorts::peek_flow(self, state, now)
    }
    fn purge_idle_clients(
        &self,
        now: u64,
        unused_before: u64,
        idle_before: u64,
    ) -> Result<(), StoreError> {
        SqlPorts::purge_idle_clients(self, now, unused_before, idle_before)
    }
}

/// Every store the AS uses, as one handle (one Durable Object).
pub trait AuthStores:
    ClientStore + CodeStore + RefreshStore + FederationStore + AssertionReplayStore + AdminStore
{
}
impl<T> AuthStores for T where
    T: ClientStore + CodeStore + RefreshStore + FederationStore + AssertionReplayStore + AdminStore
{
}

/// Per-request context shared by the endpoints.
pub struct Ctx<'a, S: AuthStores, R: Rng, C: Clock, G: Signer> {
    /// Validated configuration.
    pub cfg: &'a AuthConfig,
    /// Storage.
    pub store: &'a S,
    /// Randomness.
    pub rng: &'a R,
    /// Time.
    pub clock: &'a C,
    /// Access-token signer.
    pub signer: &'a G,
}

/// Deliver an authorization error per RFC 6749 §4.1.2.1: redirect only to a
/// verified redirect URI, otherwise an error page.
pub fn deliver(e: &AuthorizeError, issuer: &str) -> Reply {
    match e.redirect_url(issuer) {
        Some(url) => Reply::redirect(&url),
        None => Reply::error_page(e.oauth()),
    }
}
