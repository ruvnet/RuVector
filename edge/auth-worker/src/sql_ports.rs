//! `ruvector-edge-authz` storage ports over SQLite ([`SqlExec`]): Durable
//! Object SQLite in production, in-memory SQLite in native tests.
//!
//! Secrets are stored only as `secret_hash` (hex TEXT), rows carry absolute
//! expiries, and one-time `take_*` operations are a single
//! `DELETE ... RETURNING` so redemption is atomic within the single-threaded
//! Durable Object. Refresh rotation is a compare-and-set
//! `UPDATE ... WHERE rotated = 0 RETURNING`.

use crate::sql::{hex32, SqlArg, SqlExec, SqlVal};
use ruvector_edge_authz::client::ClientRecord;
use ruvector_edge_authz::code::AuthorizationCodeRecord;
use ruvector_edge_authz::federation::UpstreamFlowState;
use ruvector_edge_authz::refresh::RefreshTokenRecord;
use ruvector_edge_authz::{ClientStore, CodeStore, FederationStore, RefreshStore, StoreError};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Schema version recorded in `meta`.
pub const SCHEMA_VERSION: u32 = 1;

/// Idempotent schema (applied on every DO construction). Hash keys are
/// lowercase hex TEXT (not BLOB) to avoid ArrayBuffer/Uint8Array binding
/// ambiguity across the Workers boundary.
pub const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clients (client_id TEXT PRIMARY KEY, record TEXT NOT NULL, issued_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS codes (code_hash TEXT PRIMARY KEY, record TEXT NOT NULL, expires_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS flows (state TEXT PRIMARY KEY, record TEXT NOT NULL, expires_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS refresh_tokens (token_hash TEXT PRIMARY KEY, family_id TEXT NOT NULL, record TEXT NOT NULL, rotated INTEGER NOT NULL DEFAULT 0, expires_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS refresh_by_family ON refresh_tokens(family_id);
CREATE TABLE IF NOT EXISTS refresh_families (family_id TEXT PRIMARY KEY, revoked INTEGER NOT NULL DEFAULT 0, revoked_at INTEGER);
CREATE TABLE IF NOT EXISTS redeemed_codes (code_hash TEXT PRIMARY KEY, family_id TEXT NOT NULL, expires_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS client_activity (client_id TEXT PRIMARY KEY, last_used_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS dcr_rate (bucket TEXT PRIMARY KEY, window_start INTEGER NOT NULL, count INTEGER NOT NULL);
";

/// All authz stores over one SQL handle.
pub struct SqlPorts<D: SqlExec> {
    db: D,
}

fn to_json<T: Serialize>(v: &T) -> Result<String, StoreError> {
    serde_json::to_string(v).map_err(|e| StoreError(format!("encode: {e}")))
}

fn from_json<T: DeserializeOwned>(cell: Option<&SqlVal>) -> Result<T, StoreError> {
    let text = cell
        .and_then(SqlVal::as_text)
        .ok_or_else(|| StoreError("record column missing".into()))?;
    serde_json::from_str(text).map_err(|e| StoreError(format!("decode: {e}")))
}

impl<D: SqlExec> SqlPorts<D> {
    /// Wrap a SQL handle.
    pub fn new(db: D) -> Self {
        SqlPorts { db }
    }

    /// Apply [`SCHEMA`] and record [`SCHEMA_VERSION`] in `meta`.
    pub fn migrate(&self) -> Result<(), StoreError> {
        self.db.exec_script(SCHEMA)?;
        self.db
            .query(
                "INSERT OR IGNORE INTO meta (k, v) VALUES ('schema_version', ?)",
                vec![SCHEMA_VERSION.to_string().into()],
            )
            .map(|_| ())
    }

    /// Number of registered clients (DCR cap).
    pub fn client_count(&self) -> Result<u64, StoreError> {
        let rows = self.db.query("SELECT COUNT(*) FROM clients", vec![])?;
        rows.first()
            .and_then(|r| r.first())
            .and_then(SqlVal::as_int)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| StoreError("count failed".into()))
    }

    /// Delete expired codes, flows and refresh tokens (housekeeping; the
    /// cores also check expiry on every read).
    pub fn purge_expired(&self, now: u64) -> Result<(), StoreError> {
        for table in ["codes", "flows", "refresh_tokens", "redeemed_codes"] {
            self.db.query(
                &format!("DELETE FROM {table} WHERE expires_at <= ?"),
                vec![now.into()],
            )?;
        }
        self.db.query(
            "DELETE FROM dcr_rate WHERE window_start <= ?",
            vec![now
                .saturating_sub(crate::abuse::DCR_RATE_WINDOW_SECS)
                .into()],
        )?;
        Ok(())
    }

    /// Count one registration attempt for `bucket` (a salted IP hash) in a
    /// fixed window of `window` seconds; returns the count in the current
    /// window, this attempt included. SQLite evaluates every `SET`
    /// expression against the old row, so the window reset is atomic.
    pub fn dcr_rate_hit(&self, bucket: &str, now: u64, window: u64) -> Result<u64, StoreError> {
        let stale = now.saturating_sub(window);
        let rows = self.db.query(
            "INSERT INTO dcr_rate (bucket, window_start, count) VALUES (?, ?, 1) \
             ON CONFLICT(bucket) DO UPDATE SET \
             count = CASE WHEN window_start <= ? THEN 1 ELSE count + 1 END, \
             window_start = CASE WHEN window_start <= ? THEN excluded.window_start \
             ELSE window_start END RETURNING count",
            vec![bucket.into(), now.into(), stale.into(), stale.into()],
        )?;
        rows.first()
            .and_then(|r| r.first())
            .and_then(SqlVal::as_int)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| StoreError("rate count failed".into()))
    }

    /// The unexpired flow for `state`, read without consuming it (the
    /// consent POST; `/callback` still takes it exactly once).
    pub fn peek_flow(
        &self,
        state: &str,
        now: u64,
    ) -> Result<Option<UpstreamFlowState>, StoreError> {
        let rows = self.db.query(
            "SELECT record FROM flows WHERE state = ? AND expires_at > ?",
            vec![state.into(), now.into()],
        )?;
        rows.first().map(|r| from_json(r.first())).transpose()
    }

    /// Record that `client_id` completed a token request at `now`.
    pub fn touch_client(&self, client_id: &str, now: u64) -> Result<(), StoreError> {
        self.db
            .query(
                "INSERT INTO client_activity (client_id, last_used_at) VALUES (?, ?) \
                 ON CONFLICT(client_id) DO UPDATE SET last_used_at = excluded.last_used_at",
                vec![client_id.into(), now.into()],
            )
            .map(|_| ())
    }

    /// Delete abandoned clients: never used and registered at or before
    /// `unused_before`, or last used at or before `idle_before`; a client
    /// holding an unexpired refresh token is always kept.
    pub fn purge_idle_clients(
        &self,
        now: u64,
        unused_before: u64,
        idle_before: u64,
    ) -> Result<(), StoreError> {
        self.db.query(
            "DELETE FROM clients WHERE client_id IN (\
             SELECT c.client_id FROM clients c \
             LEFT JOIN client_activity a ON a.client_id = c.client_id \
             WHERE ((a.last_used_at IS NULL AND c.issued_at <= ?) \
                 OR (a.last_used_at IS NOT NULL AND a.last_used_at <= ?)) \
             AND NOT EXISTS (SELECT 1 FROM refresh_tokens r \
                 WHERE json_extract(r.record, '$.client_id') = c.client_id \
                 AND r.expires_at > ?))",
            vec![unused_before.into(), idle_before.into(), now.into()],
        )?;
        self.db
            .query(
                "DELETE FROM client_activity WHERE client_id NOT IN (SELECT client_id FROM clients)",
                vec![],
            )
            .map(|_| ())
    }

    fn take_record<T: DeserializeOwned>(
        &self,
        sql: &str,
        key: String,
    ) -> Result<Option<T>, StoreError> {
        let rows = self.db.query(sql, vec![key.into()])?;
        rows.first().map(|r| from_json(r.first())).transpose()
    }
}

impl<D: SqlExec> ClientStore for SqlPorts<D> {
    fn insert_client(&self, record: &ClientRecord) -> Result<(), StoreError> {
        self.db
            .query(
                "INSERT INTO clients (client_id, record, issued_at) VALUES (?, ?, ?)",
                vec![
                    record.client_id.as_str().into(),
                    to_json(record)?.into(),
                    record.client_id_issued_at.into(),
                ],
            )
            .map(|_| ())
    }

    fn get_client(&self, client_id: &str) -> Result<Option<ClientRecord>, StoreError> {
        self.take_record(
            "SELECT record FROM clients WHERE client_id = ?",
            client_id.to_string(),
        )
    }
}

impl<D: SqlExec> CodeStore for SqlPorts<D> {
    fn insert_code(&self, record: &AuthorizationCodeRecord) -> Result<(), StoreError> {
        self.db
            .query(
                "INSERT INTO codes (code_hash, record, expires_at) VALUES (?, ?, ?)",
                vec![
                    hex32(&record.code_hash).into(),
                    to_json(record)?.into(),
                    record.expires_at.into(),
                ],
            )
            .map(|_| ())
    }

    fn take_code(
        &self,
        code_hash: &[u8; 32],
    ) -> Result<Option<AuthorizationCodeRecord>, StoreError> {
        self.take_record(
            "DELETE FROM codes WHERE code_hash = ? RETURNING record",
            hex32(code_hash),
        )
    }

    fn record_redeemed(
        &self,
        code_hash: &[u8; 32],
        family_id: &str,
        expires_at: u64,
    ) -> Result<(), StoreError> {
        self.db
            .query(
                "INSERT INTO redeemed_codes (code_hash, family_id, expires_at) VALUES (?, ?, ?) \
                 ON CONFLICT(code_hash) DO UPDATE SET family_id = excluded.family_id, \
                 expires_at = excluded.expires_at",
                vec![hex32(code_hash).into(), family_id.into(), expires_at.into()],
            )
            .map(|_| ())
    }

    fn redeemed_family(
        &self,
        code_hash: &[u8; 32],
        now: u64,
    ) -> Result<Option<String>, StoreError> {
        let rows = self.db.query(
            "SELECT family_id FROM redeemed_codes WHERE code_hash = ? AND expires_at > ?",
            vec![hex32(code_hash).into(), now.into()],
        )?;
        Ok(rows
            .first()
            .and_then(|r| r.first())
            .and_then(SqlVal::as_text)
            .map(str::to_string))
    }
}

impl<D: SqlExec> RefreshStore for SqlPorts<D> {
    fn insert_refresh(&self, record: &RefreshTokenRecord) -> Result<(), StoreError> {
        self.db
            .query(
                "INSERT INTO refresh_tokens (token_hash, family_id, record, rotated, expires_at) \
                 VALUES (?, ?, ?, ?, ?)",
                vec![
                    hex32(&record.token_hash).into(),
                    record.family_id.as_str().into(),
                    to_json(record)?.into(),
                    SqlArg::Int(i64::from(record.rotated)),
                    record.expires_at.into(),
                ],
            )
            .map(|_| ())
    }

    /// The `rotated` column is authoritative over the JSON copy.
    fn get_refresh(&self, token_hash: &[u8; 32]) -> Result<Option<RefreshTokenRecord>, StoreError> {
        let rows = self.db.query(
            "SELECT record, rotated FROM refresh_tokens WHERE token_hash = ?",
            vec![hex32(token_hash).into()],
        )?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let mut record: RefreshTokenRecord = from_json(row.first())?;
        record.rotated = row.get(1).and_then(SqlVal::as_int).unwrap_or(1) != 0;
        Ok(Some(record))
    }

    fn mark_rotated(&self, token_hash: &[u8; 32]) -> Result<bool, StoreError> {
        let rows = self.db.query(
            "UPDATE refresh_tokens SET rotated = 1 WHERE token_hash = ? AND rotated = 0 \
             RETURNING token_hash",
            vec![hex32(token_hash).into()],
        )?;
        Ok(rows.len() == 1)
    }

    fn revoke_family(&self, family_id: &str) -> Result<(), StoreError> {
        self.db
            .query(
                "INSERT INTO refresh_families (family_id, revoked, revoked_at) \
                 VALUES (?, 1, CAST(strftime('%s','now') AS INTEGER)) \
                 ON CONFLICT(family_id) DO UPDATE SET revoked = 1, \
                 revoked_at = excluded.revoked_at",
                vec![family_id.into()],
            )
            .map(|_| ())
    }

    fn is_family_revoked(&self, family_id: &str) -> Result<bool, StoreError> {
        let rows = self.db.query(
            "SELECT revoked FROM refresh_families WHERE family_id = ?",
            vec![family_id.into()],
        )?;
        Ok(rows
            .first()
            .and_then(|r| r.first())
            .and_then(SqlVal::as_int)
            .is_some_and(|v| v != 0))
    }
}

impl<D: SqlExec> FederationStore for SqlPorts<D> {
    fn insert_flow(&self, state: &UpstreamFlowState) -> Result<(), StoreError> {
        self.db
            .query(
                "INSERT INTO flows (state, record, expires_at) VALUES (?, ?, ?)",
                vec![
                    state.state.as_str().into(),
                    to_json(state)?.into(),
                    state.expires_at.into(),
                ],
            )
            .map(|_| ())
    }

    fn take_flow(&self, state_param: &str) -> Result<Option<UpstreamFlowState>, StoreError> {
        self.take_record(
            "DELETE FROM flows WHERE state = ? RETURNING record",
            state_param.to_string(),
        )
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "sql_ports_tests.rs"]
mod tests;
