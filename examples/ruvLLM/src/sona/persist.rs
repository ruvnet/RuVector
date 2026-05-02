//! Persistent trajectory store (P1 sidecar)
//!
//! Replaces the lossy in-memory `ArrayQueue` trajectory buffer with a durable
//! SQLite-backed sidecar. Trajectories are submitted via a bounded mpsc channel
//! and drained on a background writer thread. The store is feature-gated behind
//! `persistence` so ESP32 / no_std targets continue using `TrajectoryBuffer`.
//!
//! ## Crash semantics
//!
//! - SQLite WAL mode + `synchronous = NORMAL`. This trades a small risk of
//!   losing the last few microseconds of in-flight transactions on power loss
//!   for a large throughput win. The DB is always consistent — WAL replays at
//!   open guarantee no torn writes.
//! - On `Drop` the writer is signaled and joined; any messages already in the
//!   channel are flushed first. Use `shutdown()` for an explicit error-checked
//!   flush.
//! - Channel-full = `record()` returns `false`, drop counter increments, and a
//!   rate-limited `tracing::warn!` is emitted. NEVER silently dropped.

use crate::sona::types::QueryTrajectory;
use rusqlite::{params, Connection, OpenFlags};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema version. Mismatch on open => error fast (no auto-migration in v1).
pub const SCHEMA_VERSION: i64 = 1;

/// Log a drop event at most once per this many drops (rate-limit log flood).
const DROP_LOG_EVERY: u64 = 1024;

/// Errors from the persistent trajectory store.
#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("bincode encode error: {0}")]
    BincodeEncode(#[from] bincode::error::EncodeError),

    #[error("bincode decode error: {0}")]
    BincodeDecode(#[from] bincode::error::DecodeError),

    #[error("schema version mismatch: db={db} expected={expected}")]
    SchemaMismatch { db: i64, expected: i64 },

    #[error("writer thread join failed")]
    JoinFailed,

    #[error("writer thread reported error: {0}")]
    Writer(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Internal control message for the writer thread.
enum WriterMsg {
    Trajectory(QueryTrajectory),
    Shutdown,
}

/// Persistent trajectory store: bounded channel + background SQLite writer.
pub struct PersistentTrajectoryStore {
    sender: SyncSender<WriterMsg>,
    writer: Option<JoinHandle<Result<(), PersistError>>>,
    persist_path: PathBuf,
    dropped: Arc<AtomicU64>,
    total_seen: Arc<AtomicU64>,
}

impl PersistentTrajectoryStore {
    /// Open (or create) a store at `persist_path` with `channel_capacity` slots
    /// in the bounded mpsc queue. Spawns the background writer thread.
    pub fn new(persist_path: PathBuf, channel_capacity: usize) -> Result<Self, PersistError> {
        if let Some(parent) = persist_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Open once on the main thread to verify schema before spawning writer.
        let conn = Connection::open_with_flags(
            &persist_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;
        Self::init_schema(&conn)?;
        Self::check_schema_version(&conn)?;
        drop(conn);

        let (sender, receiver) = sync_channel::<WriterMsg>(channel_capacity.max(1));
        let writer_path = persist_path.clone();

        let writer = std::thread::Builder::new()
            .name("ruvllm-trajectory-writer".into())
            .spawn(move || -> Result<(), PersistError> {
                let conn = Connection::open(&writer_path)?;
                conn.pragma_update(None, "journal_mode", "WAL")?;
                conn.pragma_update(None, "synchronous", "NORMAL")?;

                let mut stmt = conn.prepare(
                    "INSERT INTO trajectories \
                       (query_embedding, steps, final_quality, latency_us, \
                        model_route, context_ids, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )?;

                let cfg = bincode::config::standard();
                while let Ok(msg) = receiver.recv() {
                    match msg {
                        WriterMsg::Shutdown => break,
                        WriterMsg::Trajectory(t) => {
                            let qe = bincode::serde::encode_to_vec(&t.query_embedding, cfg)?;
                            let steps = bincode::serde::encode_to_vec(&t.steps, cfg)?;
                            let ctx = bincode::serde::encode_to_vec(&t.context_ids, cfg)?;
                            let now_us = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_micros() as i64)
                                .unwrap_or(0);
                            stmt.execute(params![
                                qe,
                                steps,
                                t.final_quality as f64,
                                t.latency_us as i64,
                                t.model_route,
                                ctx,
                                now_us,
                            ])?;
                        }
                    }
                }
                Ok(())
            })
            .map_err(PersistError::Io)?;

        Ok(Self {
            sender,
            writer: Some(writer),
            persist_path,
            dropped: Arc::new(AtomicU64::new(0)),
            total_seen: Arc::new(AtomicU64::new(0)),
        })
    }

    fn init_schema(conn: &Connection) -> Result<(), PersistError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS trajectories (
               id            INTEGER PRIMARY KEY,
               query_embedding BLOB NOT NULL,
               steps         BLOB NOT NULL,
               final_quality REAL NOT NULL,
               latency_us    INTEGER NOT NULL,
               model_route   TEXT,
               context_ids   BLOB,
               created_at    INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_trajectories_created_at
               ON trajectories(created_at DESC);",
        )?;
        // Insert version row if absent.
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM schema_meta", [], |r| r.get(0))?;
        if count == 0 {
            conn.execute(
                "INSERT INTO schema_meta (version) VALUES (?1)",
                params![SCHEMA_VERSION],
            )?;
        }
        Ok(())
    }

    fn check_schema_version(conn: &Connection) -> Result<(), PersistError> {
        let v: i64 = conn.query_row(
            "SELECT version FROM schema_meta ORDER BY version DESC LIMIT 1",
            [],
            |r| r.get(0),
        )?;
        if v != SCHEMA_VERSION {
            return Err(PersistError::SchemaMismatch {
                db: v,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(())
    }

    /// Record a trajectory non-blocking. Returns `false` if the channel is full
    /// (drop counter increments, rate-limited warn is logged).
    pub fn record(&self, t: QueryTrajectory) -> bool {
        self.total_seen.fetch_add(1, Ordering::Relaxed);
        match self.sender.try_send(WriterMsg::Trajectory(t)) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if dropped % DROP_LOG_EVERY == 1 {
                    tracing::warn!(
                        dropped,
                        path = %self.persist_path.display(),
                        "trajectory channel full or disconnected — drop event"
                    );
                }
                false
            }
        }
    }

    /// Number of trajectories dropped due to channel-full or disconnected.
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Total trajectories ever submitted via `record()`.
    pub fn total_seen(&self) -> u64 {
        self.total_seen.load(Ordering::Relaxed)
    }

    /// Load the most recent `n` trajectories (newest first by `created_at`).
    /// Used at restart to replay durable buffer into in-memory consumers.
    pub fn load_recent(&self, n: usize) -> Result<Vec<QueryTrajectory>, PersistError> {
        let conn = Connection::open_with_flags(
            &self.persist_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let mut stmt = conn.prepare(
            "SELECT id, query_embedding, steps, final_quality, latency_us, \
                    model_route, context_ids \
             FROM trajectories \
             ORDER BY created_at DESC LIMIT ?1",
        )?;

        let cfg = bincode::config::standard();
        let rows = stmt.query_map(params![n as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<Vec<u8>>>(6)?,
            ))
        })?;

        let mut out = Vec::with_capacity(n);
        for row in rows {
            let (id, qe_blob, steps_blob, fq, lat, route, ctx_blob) = row?;
            let (query_embedding, _) =
                bincode::serde::decode_from_slice(&qe_blob, cfg)?;
            let (steps, _) = bincode::serde::decode_from_slice(&steps_blob, cfg)?;
            let context_ids = match ctx_blob {
                Some(b) => bincode::serde::decode_from_slice(&b, cfg)?.0,
                None => Vec::new(),
            };
            out.push(QueryTrajectory {
                id: id as u64,
                query_embedding,
                steps,
                final_quality: fq as f32,
                latency_us: lat as u64,
                model_route: route,
                context_ids,
            });
        }
        Ok(out)
    }

    /// Flush + join the writer. Consumes the store.
    pub fn shutdown(mut self) -> Result<(), PersistError> {
        // Best-effort: if the channel is full at shutdown, fall back to a
        // blocking send — we want shutdown to complete, not lose final messages.
        let _ = self.sender.send(WriterMsg::Shutdown);
        if let Some(handle) = self.writer.take() {
            match handle.join() {
                Ok(res) => res?,
                Err(_) => return Err(PersistError::JoinFailed),
            }
        }
        Ok(())
    }
}

impl Drop for PersistentTrajectoryStore {
    fn drop(&mut self) {
        // Signal writer to flush remaining messages and exit. Errors are
        // swallowed in Drop — explicit shutdown() is the right path for
        // error-checked teardown.
        let _ = self.sender.send(WriterMsg::Shutdown);
        if let Some(handle) = self.writer.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sona::types::QueryTrajectory;

    #[test]
    fn test_open_and_schema_init() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traj.db");
        let store = PersistentTrajectoryStore::new(path.clone(), 16).unwrap();
        store.shutdown().unwrap();

        // Reopen succeeds with same schema version.
        let store2 = PersistentTrajectoryStore::new(path, 16).unwrap();
        store2.shutdown().unwrap();
    }

    #[test]
    fn test_record_and_load_recent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traj.db");
        let store = PersistentTrajectoryStore::new(path.clone(), 64).unwrap();
        for i in 0..10 {
            let t = QueryTrajectory::new(i as u64, vec![i as f32, (i + 1) as f32]);
            assert!(store.record(t));
        }
        store.shutdown().unwrap();

        let store2 = PersistentTrajectoryStore::new(path, 64).unwrap();
        let recent = store2.load_recent(10).unwrap();
        assert_eq!(recent.len(), 10);
        store2.shutdown().unwrap();
    }
}
