//! Ports (hexagonal boundaries). All sync: Durable Object SQLite `exec` is
//! synchronous in workers-rs, and sync traits automock cleanly with mockall.
//! The Worker implements these; unit tests mock them.

use crate::client::ClientRecord;
use crate::code::AuthorizationCodeRecord;
use crate::federation::UpstreamFlowState;
use crate::refresh::RefreshTokenRecord;

pub use ruvector_edge_auth::Clock;

/// Storage failure (the Worker maps it to `server_error`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreError(pub String);

/// Cryptographically secure randomness. `&self` because Worker handles are
/// shared (`crypto.getRandomValues` / getrandom `js`).
#[cfg_attr(test, mockall::automock)]
pub trait Rng {
    /// Fill `buf` with random bytes. An error must abort the operation (never
    /// fall back to weak or zeroed bytes).
    fn fill(&self, buf: &mut [u8]) -> Result<(), StoreError>;
}

/// ES256 signer for access tokens.
#[cfg_attr(test, mockall::automock)]
pub trait Signer {
    /// `kid` of the active key (its RFC 7638 thumbprint).
    fn kid(&self) -> String;
    /// Fixed-width `r || s` ES256 signature over `signing_input`.
    fn sign_es256(&self, signing_input: &[u8]) -> Result<[u8; 64], StoreError>;
}

/// Registered clients (DCR).
#[cfg_attr(test, mockall::automock)]
pub trait ClientStore {
    /// Insert a new client; `client_id` must be unused.
    fn insert_client(&self, record: &ClientRecord) -> Result<(), StoreError>;
    /// Look up by `client_id`.
    fn get_client(&self, client_id: &str) -> Result<Option<ClientRecord>, StoreError>;
}

/// Authorization codes, keyed by `secret_hash(code)`.
#[cfg_attr(test, mockall::automock)]
pub trait CodeStore {
    /// Insert a code record.
    fn insert_code(&self, record: &AuthorizationCodeRecord) -> Result<(), StoreError>;
    /// Atomically remove and return the record (one-time redemption: a second
    /// call for the same hash returns `None`).
    fn take_code(
        &self,
        code_hash: &[u8; 32],
    ) -> Result<Option<AuthorizationCodeRecord>, StoreError>;
    /// Record a tombstone for a successfully redeemed code: the grant
    /// (`family_id`) it started, kept until `expires_at` (unix seconds) so a
    /// replay can revoke that grant (RFC 6749 §4.1.2). Required, no default:
    /// a no-op would silently disable replay revocation.
    fn record_redeemed(
        &self,
        code_hash: &[u8; 32],
        family_id: &str,
        expires_at: u64,
    ) -> Result<(), StoreError>;
    /// The `family_id` of an unexpired tombstone for `code_hash` at `now`.
    fn redeemed_family(&self, code_hash: &[u8; 32], now: u64)
        -> Result<Option<String>, StoreError>;
}

/// Refresh tokens and families, keyed by `secret_hash(token)`.
#[cfg_attr(test, mockall::automock)]
pub trait RefreshStore {
    /// Insert a refresh token record.
    fn insert_refresh(&self, record: &RefreshTokenRecord) -> Result<(), StoreError>;
    /// Look up by hash (including already-rotated tokens, for reuse detection).
    fn get_refresh(&self, token_hash: &[u8; 32]) -> Result<Option<RefreshTokenRecord>, StoreError>;
    /// Mark a token rotated (used). Returns `false` if it was already used
    /// (compare-and-set, so concurrent rotation is detected).
    fn mark_rotated(&self, token_hash: &[u8; 32]) -> Result<bool, StoreError>;
    /// Revoke every token in a family.
    fn revoke_family(&self, family_id: &str) -> Result<(), StoreError>;
    /// Whether a family is revoked.
    fn is_family_revoked(&self, family_id: &str) -> Result<bool, StoreError>;
}

/// Pending upstream-federation flows, keyed by the upstream `state` value.
#[cfg_attr(test, mockall::automock)]
pub trait FederationStore {
    /// Insert pending flow state.
    fn insert_flow(&self, state: &UpstreamFlowState) -> Result<(), StoreError>;
    /// Atomically remove and return the flow (one-time).
    fn take_flow(&self, state_param: &str) -> Result<Option<UpstreamFlowState>, StoreError>;
}
