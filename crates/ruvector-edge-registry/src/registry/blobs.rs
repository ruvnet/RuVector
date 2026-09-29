//! Reference counts of this scope's content-addressed blobs.
//!
//! The blob key carries the scope (`rvf/{tenant}/{scope}/blobs/…`), so this
//! DO's `blob/` records are the only counter for those objects. A record
//! counts every version pointing at the blob **and** every upload session
//! pinned between `finalize_plan` and its end, so nothing can release a blob
//! that a finalize is about to reference. `stored` says the bytes are known
//! to be in place (some finalize wrote them and succeeded); a pin alone does
//! not make a blob present, so two concurrent uploads of one content both
//! write it.
//!
//! Releasing never hands a key straight to a deleter. A blob whose count
//! reaches zero is queued under `gc/`, and a later pin revives it (dropping
//! the queue entry). [`Registry::sweep`] returns the queued keys whose count
//! is still zero. The caller deletes them **before the DO serves another
//! request** (run the sweep inside the DO under `blockConcurrencyWhile`), so
//! no pin can interleave between the check and the delete.
//!
//! Public blobs (`public/blobs/…`) are global and never counted here: a
//! public version is never deleted, so its bytes are never released.

use super::Registry;
use crate::error::Result;
use crate::keys::BlobKey;
use crate::ports::{Clock, EntropySource, KvStore};
use serde::{Deserialize, Serialize};

/// Reference count of one content-addressed blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    /// Versions pointing at it plus uploads pinning it.
    pub refs: u64,
    /// Object size.
    pub size: u64,
    /// The bytes are known to be stored under the key.
    pub stored: bool,
}

pub(crate) fn k_blob(b: &BlobKey) -> String {
    format!("blob/{}", b.as_str())
}

pub(crate) fn k_gc(b: &BlobKey) -> String {
    format!("gc/{}", b.as_str())
}

impl<S: KvStore, C: Clock, E: EntropySource> Registry<S, C, E> {
    /// Reference count of a blob, if indexed.
    pub fn blob_ref(&self, key: &BlobKey) -> Result<Option<BlobRef>> {
        self.load(&k_blob(key))
    }

    /// `true` if the bytes are known to be stored (dedupe: skip the copy).
    pub fn blob_present(&self, key: &BlobKey) -> Result<bool> {
        Ok(self.blob_ref(key)?.is_some_and(|r| r.stored))
    }

    /// Take a reference (a pin, or a version). Returns `true` if the bytes
    /// are not known to be stored, i.e. the caller must write them.
    pub(crate) fn pin_blob(&self, key: &BlobKey, size: u64) -> Result<bool> {
        let cur = self.blob_ref(key)?;
        let stored = cur.is_some_and(|r| r.stored);
        let refs = cur.map_or(0, |r| r.refs).saturating_add(1);
        self.store.delete(&k_gc(key))?;
        self.save(&k_blob(key), &BlobRef { refs, size, stored })?;
        Ok(!stored)
    }

    /// Mark the bytes stored (a finalize that wrote them succeeded).
    pub(crate) fn mark_stored(&self, key: &BlobKey) -> Result<()> {
        if let Some(mut r) = self.blob_ref(key)? {
            if !r.stored {
                r.stored = true;
                self.save(&k_blob(key), &r)?;
            }
        }
        Ok(())
    }

    /// Drop a reference. At zero the record goes and the key is queued for
    /// [`Registry::sweep`]; returns the queued key.
    pub(crate) fn release_blob(&self, key: &BlobKey) -> Result<Option<BlobKey>> {
        let Some(mut r) = self.blob_ref(key)? else {
            return Ok(None);
        };
        r.refs = r.refs.saturating_sub(1);
        if r.refs > 0 {
            self.save(&k_blob(key), &r)?;
            return Ok(None);
        }
        self.store.delete(&k_blob(key))?;
        self.store.put(&k_gc(key), b"{}")?;
        Ok(Some(key.clone()))
    }
}
