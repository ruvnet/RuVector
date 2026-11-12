//! Shard delete and wipe (ADR-351 §6.1 write path; deletes in one
//! coalesced batch, the op body carrying the row's iid so an index epoch
//! can replay it).

use super::codec::encode_delete_body;
use super::write::{ops_row, to_i64, UsageDelta};
use super::{Actor, VectorShard};
use crate::error::{ErrorCode, OpError};
use crate::ports::{SqlStore, StoreError, Value};
use crate::schema;
use ruvector_edge_tenancy::{DoMeta, IdentityCheck, VectorId};
use serde::Serialize;
use std::collections::BTreeSet;

/// Maximum ids per delete call (§7).
pub const MAX_DELETE_IDS: usize = 1000;

/// Result of a delete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeleteOutcome {
    /// Ids that existed and were (or would be) removed.
    pub deleted: u64,
    /// Shard `write_seq` after the write.
    pub write_seq: u64,
    /// Usage change (non-positive).
    pub delta: UsageDelta,
    /// `true` when nothing was written.
    pub dry_run: bool,
}

impl VectorShard {
    /// Delete ids (absent ids are ignored). `dry_run` writes nothing.
    pub fn delete(
        &mut self,
        store: &dyn SqlStore,
        expected: &DoMeta,
        ids: &[String],
        actor: Actor<'_>,
        dry_run: bool,
        now: u64,
    ) -> Result<DeleteOutcome, OpError> {
        self.ensure_live()?;
        if ids.len() > MAX_DELETE_IDS {
            return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
        }
        let mut present = Vec::new();
        let mut seen = BTreeSet::new();
        for id in ids {
            let id = VectorId::parse(id)?.as_str().to_string();
            if seen.insert(id.clone()) && self.slab.index.contains_key(&id) {
                present.push(id);
            }
        }
        let initialized = self.check(expected, true)? == IdentityCheck::Matched;
        let dim = self.config.as_ref().map_or(0, |c| c.dim as usize);
        let mut delta = UsageDelta::default();
        for id in &present {
            let slot = self.slab.index[id];
            delta.vectors -= 1;
            delta.floats -= dim as i64;
            delta.bytes -= to_i64(self.slab.row_bytes(slot, dim))?;
        }
        if dry_run || present.is_empty() || !initialized {
            return Ok(DeleteOutcome {
                deleted: present.len() as u64,
                write_seq: self.write_seq,
                delta,
                dry_run,
            });
        }
        self.index_for_request(store)?;
        let ts = to_i64(now)?;
        let res = (|| -> Result<u64, StoreError> {
            let mut seq = self.write_seq;
            for id in &present {
                seq += 1;
                let iid = self.slab.iids[self.slab.index[id]];
                let body = Value::Blob(encode_delete_body(iid));
                store.exec(
                    schema::OPS_APPEND,
                    &ops_row(seq, "delete", id, ts, actor, body)?,
                )?;
                store.exec(schema::VEC_DELETE, &[id.as_str().into()])?;
                store.exec(schema::FILTER_DELETE_ID, &[id.as_str().into()])?;
            }
            self.write_counters(store, seq, None)
        })();
        let snapshot = match res {
            Ok(s) => s,
            Err(e) => return self.poison(e),
        };
        for id in &present {
            if let Some(iid) = self.slab.remove(id, dim) {
                if let Some(ann) = self.ann.as_mut() {
                    ann.remove(iid);
                }
            }
        }
        self.write_seq += present.len() as u64;
        self.snapshot_seq = snapshot;
        self.after_write(store);
        Ok(DeleteOutcome {
            deleted: present.len() as u64,
            write_seq: self.write_seq,
            delta,
            dry_run: false,
        })
    }

    /// Erase all storage of this shard (collection drop: the DO is wiped
    /// before the catalog row is tombstoned). Returns the usage released.
    pub fn wipe(&mut self, store: &dyn SqlStore, expected: &DoMeta) -> Result<UsageDelta, OpError> {
        self.ensure_live()?;
        if self.check(expected, true)? != IdentityCheck::Matched {
            return Ok(UsageDelta::default());
        }
        let dim = self.config.as_ref().map_or(0, |c| c.dim as i64);
        let mut delta = UsageDelta::default();
        for slot in 0..self.slab.len() {
            delta.vectors -= 1;
            delta.floats -= dim;
            delta.bytes -= to_i64(self.slab.row_bytes(slot, dim as usize))?;
        }
        for sql in [
            schema::OPS_DELETE_ALL,
            schema::VEC_DELETE_ALL,
            schema::FILTER_DELETE_ALL,
            schema::CHUNK_DELETE_ALL,
            schema::META_DELETE_ALL,
        ] {
            if let Err(e) = store.exec(sql, &[]) {
                return self.poison(e);
            }
        }
        *self = VectorShard::default();
        Ok(delta)
    }
}
