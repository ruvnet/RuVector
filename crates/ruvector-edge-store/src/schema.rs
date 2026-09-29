//! Every SQL statement the crate issues (ADR-351 §6.1 / §6.2).
//!
//! Keeping them as constants lets the in-memory mock interpret a fixed,
//! small grammar and lets `tests/sqlite_conformance.rs` run the very same
//! text against real SQLite. Deviations from the ADR tables, all additive
//! (§16.4):
//!
//! - `ops.body BLOB`: the upsert payload, so ops logged past
//!   `meta.write_seq` (a torn write) can be replayed write-through on open.
//!   The log is pruned to a bounded tail below `write_seq`
//!   (`meta.snapshot_seq` records the cut; `vectors` is the snapshot).
//! - `ops.act_sub TEXT`: `act.sub` of an exchanged token (the adapter that
//!   acted, §5.6/§16.3), `NULL` for direct writes. Part of the first shard
//!   schema: no `VectorShard` existed before M1, so no table needs an ALTER.
//! - `filter_idx` has a primary key `(key, value, id)`.
//! - M2: `ops.body` v2 carries the row's `iid` (upsert and delete), so an
//!   index epoch can be brought forward by replaying the op tail even for
//!   ids deleted since; `index_chunks` is created with the other tables.
//!   `vectors.q8` stays unused: the resident codes persist as chunks.
//! - `TenantLedger` tables: `ledger_meta`, `memberships`, `catalog` (with
//!   `filterable_keys`), `idempotency` (the §16.3 `op_id` store) with an
//!   `expires_at` index for bounded purges.

/// `VectorShard` DDL.
pub const SHARD_SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)",
    "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, iid INTEGER UNIQUE, f32 BLOB, \
     q8 BLOB, q8_epoch INTEGER, metadata TEXT, updated_at INTEGER, deleted INTEGER DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS ops (seq INTEGER PRIMARY KEY, op TEXT, id TEXT, ts INTEGER, \
     actor_sub TEXT, jti TEXT, family_id TEXT, act_sub TEXT, body BLOB)",
    "CREATE TABLE IF NOT EXISTS filter_idx (key TEXT, value TEXT, id TEXT, \
     PRIMARY KEY (key, value, id))",
    "CREATE TABLE IF NOT EXISTS index_chunks (epoch INTEGER, part INTEGER, bytes BLOB, \
     PRIMARY KEY (epoch, part))",
];

/// Ids per rerank / fetch-by-iid statement (the `IN` list width). Unused
/// slots are bound to iid `0`, which the store never allocates.
pub const IID_BATCH: usize = 32;

/// Paged slab load (no f32: M2 keeps only codes/links resident).
pub const VEC_PAGE_META: &str =
    "SELECT id, iid, metadata FROM vectors WHERE iid > ? ORDER BY iid LIMIT ?";
/// Paged f32 read for an index rebuild (no id, no metadata: neither is
/// used, and metadata can be 4 KiB a row).
pub const VEC_PAGE_F32: &str = "SELECT iid, f32 FROM vectors WHERE iid > ? ORDER BY iid LIMIT ?";
/// Quantizer training sample: up to [`IID_BATCH`] f32 rows by internal id.
pub const VEC_F32_BY_IIDS: &str = concat!(
    "SELECT iid, f32 FROM vectors WHERE iid IN (",
    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ",
    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
);
/// Rerank fetch: up to [`IID_BATCH`] rows by internal id.
pub const VEC_BY_IIDS: &str = concat!(
    "SELECT iid, id, f32, metadata FROM vectors WHERE iid IN (",
    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ",
    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
);
/// One row by external id (fetch).
pub const VEC_BY_ID: &str = "SELECT id, iid, f32, metadata FROM vectors WHERE id = ?";
/// Renumber one row (dense-iid compaction, ascending so no clash).
pub const VEC_SET_IID: &str = "UPDATE vectors SET iid = ? WHERE id = ?";

/// Write one index chunk row.
pub const CHUNK_PUT: &str =
    "INSERT OR REPLACE INTO index_chunks (epoch, part, bytes) VALUES (?, ?, ?)";
/// Stream one epoch's chunks, one row per call (`LIMIT 1`).
pub const CHUNK_GET: &str = "SELECT part, bytes FROM index_chunks WHERE epoch = ? AND part >= ? \
                             ORDER BY part LIMIT ?";
/// Drop an epoch and anything newer (clears a torn earlier attempt).
pub const CHUNK_DELETE_FROM: &str = "DELETE FROM index_chunks WHERE epoch >= ?";
/// Drop epochs older than the oldest retained one.
pub const CHUNK_DELETE_BELOW: &str = "DELETE FROM index_chunks WHERE epoch < ?";
/// Wipe `index_chunks`.
pub const CHUNK_DELETE_ALL: &str = "DELETE FROM index_chunks";

/// All `meta` rows.
pub const META_SELECT_ALL: &str = "SELECT k, v FROM meta";
/// Set one `meta` row.
pub const META_PUT: &str = "INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)";
/// Wipe `meta`.
pub const META_DELETE_ALL: &str = "DELETE FROM meta";

/// Insert or replace a vector row.
pub const VEC_PUT: &str = "INSERT OR REPLACE INTO vectors (id, iid, f32, metadata, updated_at, \
                           deleted) VALUES (?, ?, ?, ?, ?, ?)";
/// Delete one vector row.
pub const VEC_DELETE: &str = "DELETE FROM vectors WHERE id = ?";
/// Paged cold load by internal id (never a full `to_array`).
pub const VEC_PAGE: &str =
    "SELECT id, iid, f32, metadata FROM vectors WHERE iid > ? ORDER BY iid LIMIT ?";
/// Wipe `vectors`.
pub const VEC_DELETE_ALL: &str = "DELETE FROM vectors";

/// Append one op-log entry (`seq` is the shard `write_seq`).
pub const OPS_APPEND: &str = "INSERT INTO ops (seq, op, id, ts, actor_sub, jti, family_id, \
                              act_sub, body) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
/// Paged op-log read.
pub const OPS_PAGE: &str =
    "SELECT seq, op, id, body, ts FROM ops WHERE seq > ? ORDER BY seq LIMIT ?";
/// Who wrote each op-log entry (audit read: `actor_sub`, adapter `act_sub`).
pub const OPS_ACTORS: &str =
    "SELECT seq, op, actor_sub, act_sub FROM ops WHERE seq > ? ORDER BY seq LIMIT ?";
/// Drop op-log entries at or below a sequence (already reflected in
/// `vectors`; the shard keeps a bounded tail).
pub const OPS_PRUNE: &str = "DELETE FROM ops WHERE seq <= ?";
/// Wipe `ops`.
pub const OPS_DELETE_ALL: &str = "DELETE FROM ops";

/// Index one declared metadata key.
pub const FILTER_PUT: &str = "INSERT OR REPLACE INTO filter_idx (key, value, id) VALUES (?, ?, ?)";
/// Drop the filter rows of one vector.
pub const FILTER_DELETE_ID: &str = "DELETE FROM filter_idx WHERE id = ?";
/// Wipe `filter_idx`.
pub const FILTER_DELETE_ALL: &str = "DELETE FROM filter_idx";

/// `TenantLedger` DDL.
pub const LEDGER_SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS ledger_meta (k TEXT PRIMARY KEY, v TEXT)",
    "CREATE TABLE IF NOT EXISTS memberships (sub TEXT PRIMARY KEY, role TEXT, invited_by TEXT, \
     created_at INTEGER)",
    "CREATE TABLE IF NOT EXISTS catalog (collection_uid TEXT PRIMARY KEY, name TEXT, \
     service TEXT, kind TEXT, dim INTEGER, metric TEXT, embedder TEXT, shard_count INTEGER, \
     state TEXT, origin TEXT, created_by TEXT, created_at INTEGER, filterable_keys TEXT)",
    "CREATE TABLE IF NOT EXISTS idempotency (sub TEXT, key TEXT, body_sha256 BLOB, \
     response TEXT, bytes INTEGER, expires_at INTEGER, PRIMARY KEY (sub, key))",
    "CREATE INDEX IF NOT EXISTS idempotency_expires ON idempotency (expires_at)",
];

/// All `ledger_meta` rows.
pub const LMETA_SELECT_ALL: &str = "SELECT k, v FROM ledger_meta";
/// Set one `ledger_meta` row.
pub const LMETA_PUT: &str = "INSERT OR REPLACE INTO ledger_meta (k, v) VALUES (?, ?)";

/// All memberships.
pub const MEMBER_SELECT_ALL: &str = "SELECT sub, role, invited_by, created_at FROM memberships";
/// Insert a membership (fails on an existing `sub`).
pub const MEMBER_INSERT: &str =
    "INSERT INTO memberships (sub, role, invited_by, created_at) VALUES (?, ?, ?, ?)";
/// Insert or replace a membership (role change).
pub const MEMBER_PUT: &str =
    "INSERT OR REPLACE INTO memberships (sub, role, invited_by, created_at) VALUES (?, ?, ?, ?)";
/// Remove a membership.
pub const MEMBER_DELETE: &str = "DELETE FROM memberships WHERE sub = ?";

/// All catalog rows, live and tombstoned.
pub const CATALOG_SELECT_ALL: &str = "SELECT collection_uid, name, service, kind, dim, metric, \
     embedder, shard_count, state, origin, created_by, created_at, filterable_keys FROM catalog";
/// Insert a catalog row (the uid primary key rejects any reuse).
pub const CATALOG_INSERT: &str = "INSERT INTO catalog (collection_uid, name, service, kind, dim, \
     metric, embedder, shard_count, state, origin, created_by, created_at, filterable_keys) \
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
/// Tombstone a catalog row (rows are never deleted: uids stay reserved).
pub const CATALOG_SET_STATE: &str = "UPDATE catalog SET state = ? WHERE collection_uid = ?";

/// Look up an `op_id` for `(sub, key)`.
pub const IDEM_SELECT: &str =
    "SELECT body_sha256, response, expires_at, bytes FROM idempotency WHERE sub = ? AND key = ?";
/// Store an `op_id` response.
pub const IDEM_PUT: &str = "INSERT OR REPLACE INTO idempotency (sub, key, body_sha256, response, \
                            bytes, expires_at) VALUES (?, ?, ?, ?, ?, ?)";
/// A bounded batch of expired `op_id` rows (served by `idempotency_expires`).
pub const IDEM_EXPIRED: &str = "SELECT sub, key, bytes FROM idempotency WHERE expires_at <= ? \
                                ORDER BY expires_at LIMIT ?";
/// Drop one `op_id` row.
pub const IDEM_DELETE: &str = "DELETE FROM idempotency WHERE sub = ? AND key = ?";
