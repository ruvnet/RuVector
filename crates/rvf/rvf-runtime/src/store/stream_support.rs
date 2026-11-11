//! Producer-side hooks for stream-ready files (project Ofer, ADR-353).
//!
//! A stream-ready file carries progressive-layer INDEX segments next to the
//! store's own persisted index. These two methods let the producer
//! (`progressive_stream::build`) write them through the normal write path, so
//! the segments are hashed (SHAKE-256-128 in a signed store), aligned, and
//! committed by the signed manifest like any other segment.

use super::*;

impl RvfStore {
    /// Append `payload` as an INDEX segment and list it in the segment
    /// directory. Unlike the store's own index persistence, this does not
    /// supersede earlier INDEX entries; the store's next `close()` with a
    /// dirty index will (the bytes stay in the file as an unlisted segment,
    /// still covered by a stream descriptor).
    pub fn embed_index_segment(&mut self, payload: &[u8]) -> Result<u64, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }
        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (seg_id, offset) = {
            let mut w = Self::append_cursor(&self.file, &self.security)?;
            let written = writer
                .write_index_seg(&mut w, payload)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            w.flush().map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
        };
        self.segment_dir.push((
            seg_id,
            offset,
            payload.len() as u64,
            SegmentType::Index as u8,
        ));
        self.write_manifest()?;
        Ok(seg_id)
    }

    /// The in-memory HNSW index encoded as an INDEX payload, if one has been
    /// built (by a query over >= 1024 vectors). Producer support; not a
    /// stable API.
    #[doc(hidden)]
    pub fn index_payload(&self) -> Option<Vec<u8>> {
        let guard = self.index.read().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().map(|i| i.encode_payload())
    }
}
