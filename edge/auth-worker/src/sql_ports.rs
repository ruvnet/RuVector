//! `ruvector-edge-authz` storage ports over Durable Object SQLite.
//!
//! Secrets are stored hashed (`secret_hash`), rows carry absolute expiries,
//! and one-time `take_*` operations are a single `DELETE ... RETURNING` so
//! redemption is atomic within the (single-threaded) Durable Object.

use ruvector_edge_authz::client::ClientRecord;
use ruvector_edge_authz::code::AuthorizationCodeRecord;
use ruvector_edge_authz::federation::UpstreamFlowState;
use ruvector_edge_authz::refresh::RefreshTokenRecord;
use ruvector_edge_authz::{ClientStore, CodeStore, FederationStore, RefreshStore, StoreError};
use worker::SqlStorage;

/// Schema version recorded in `meta`.
pub const SCHEMA_VERSION: u32 = 1;

/// Idempotent schema (applied on every DO construction).
pub const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clients (client_id TEXT PRIMARY KEY, record TEXT NOT NULL, issued_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS codes (code_hash BLOB PRIMARY KEY, record TEXT NOT NULL, expires_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS flows (state TEXT PRIMARY KEY, record TEXT NOT NULL, expires_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS refresh_tokens (token_hash BLOB PRIMARY KEY, family_id TEXT NOT NULL, record TEXT NOT NULL, rotated INTEGER NOT NULL DEFAULT 0, expires_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS refresh_by_family ON refresh_tokens(family_id);
CREATE TABLE IF NOT EXISTS refresh_families (family_id TEXT PRIMARY KEY, revoked INTEGER NOT NULL DEFAULT 0, revoked_at INTEGER);
";

/// All authz stores over one `SqlStorage`.
#[derive(Clone)]
pub struct SqlPorts {
    sql: SqlStorage,
}

impl SqlPorts {
    /// Wrap the DO's SQLite handle.
    pub fn new(sql: SqlStorage) -> Self {
        SqlPorts { sql }
    }

    /// Apply [`SCHEMA`] and record [`SCHEMA_VERSION`] in `meta`.
    pub fn migrate(&self) -> Result<(), StoreError> {
        let err = |e: worker::Error| StoreError(e.to_string());
        self.sql.exec(SCHEMA, None).map_err(err)?;
        self.sql
            .exec(
                "INSERT OR IGNORE INTO meta (k, v) VALUES ('schema_version', ?)",
                vec![SCHEMA_VERSION.to_string().into()],
            )
            .map(|_| ())
            .map_err(err)
    }
}

fn todo_store(what: &str) -> StoreError {
    StoreError(format!("not implemented: {what}"))
}

impl ClientStore for SqlPorts {
    fn insert_client(&self, record: &ClientRecord) -> Result<(), StoreError> {
        let _ = (record, &self.sql);
        Err(todo_store("insert_client"))
    }
    fn get_client(&self, client_id: &str) -> Result<Option<ClientRecord>, StoreError> {
        let _ = client_id;
        Err(todo_store("get_client"))
    }
}

impl CodeStore for SqlPorts {
    fn insert_code(&self, record: &AuthorizationCodeRecord) -> Result<(), StoreError> {
        let _ = record;
        Err(todo_store("insert_code"))
    }
    fn take_code(
        &self,
        code_hash: &[u8; 32],
    ) -> Result<Option<AuthorizationCodeRecord>, StoreError> {
        let _ = code_hash;
        Err(todo_store("take_code"))
    }
}

impl RefreshStore for SqlPorts {
    fn insert_refresh(&self, record: &RefreshTokenRecord) -> Result<(), StoreError> {
        let _ = record;
        Err(todo_store("insert_refresh"))
    }
    fn get_refresh(&self, token_hash: &[u8; 32]) -> Result<Option<RefreshTokenRecord>, StoreError> {
        let _ = token_hash;
        Err(todo_store("get_refresh"))
    }
    fn mark_rotated(&self, token_hash: &[u8; 32]) -> Result<bool, StoreError> {
        let _ = token_hash;
        Err(todo_store("mark_rotated"))
    }
    fn revoke_family(&self, family_id: &str) -> Result<(), StoreError> {
        let _ = family_id;
        Err(todo_store("revoke_family"))
    }
    fn is_family_revoked(&self, family_id: &str) -> Result<bool, StoreError> {
        let _ = family_id;
        Err(todo_store("is_family_revoked"))
    }
}

impl FederationStore for SqlPorts {
    fn insert_flow(&self, state: &UpstreamFlowState) -> Result<(), StoreError> {
        let _ = state;
        Err(todo_store("insert_flow"))
    }
    fn take_flow(&self, state_param: &str) -> Result<Option<UpstreamFlowState>, StoreError> {
        let _ = state_param;
        Err(todo_store("take_flow"))
    }
}
