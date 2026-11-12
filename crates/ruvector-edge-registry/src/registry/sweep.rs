//! Maintenance, run inside the registry DO (an alarm) under
//! `blockConcurrencyWhile`: expire sessions, release the pins of sessions
//! whose Worker never came back, and hand out the objects that are safe to
//! delete. The DO deletes every returned key before serving another request
//! (see [`super::blobs`]).

use super::blobs::k_blob;
use super::Registry;
use crate::error::Result;
use crate::keys::{staging_key, BlobKey};
use crate::ports::{Clock, EntropySource, KvStore, StoreError};
use crate::upload::{UploadId, UploadSession};

/// What one sweep did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SweepReport {
    /// Sessions that expired while open or finalizing (now failed).
    pub expired: Vec<UploadId>,
    /// R2 objects to delete now: staging objects of ended sessions and
    /// released blobs nothing references.
    pub delete: Vec<BlobKey>,
    /// Pass as `after` to continue the session scan; `None` when done.
    pub next: Option<String>,
}

impl<S: KvStore, C: Clock, E: EntropySource> Registry<S, C, E> {
    /// Sweep up to `budget` sessions after `after` and up to `budget`
    /// released blobs.
    pub fn sweep(&self, after: Option<&str>, budget: usize) -> Result<SweepReport> {
        let now = self.now();
        let mut report = SweepReport::default();
        let rows = self.store.list("upload/", after, budget)?;
        if rows.len() == budget {
            report.next = rows.last().map(|(k, _)| k.clone());
        }
        for (key, bytes) in rows {
            let s: UploadSession = serde_json::from_slice(&bytes)
                .map_err(|_| StoreError::Corrupt("upload session"))?;
            if now < s.expires_at {
                continue;
            }
            report.delete.push(staging_key(&s.tenant, &s.id));
            if s.is_terminal() {
                self.store.delete(&key)?;
            } else {
                report.expired.push(s.id.clone());
                self.fail_session(s, "expired")?;
            }
        }
        for (key, _) in self.store.list("gc/", None, budget)? {
            self.store.delete(&key)?;
            let blob = BlobKey::from_index(&key["gc/".len()..]);
            // A pin since the release revived it: keep the bytes.
            if self.store.get(&k_blob(&blob))?.is_none() {
                report.delete.push(blob);
            }
        }
        Ok(report)
    }
}
