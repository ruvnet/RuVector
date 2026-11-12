//! Typed application ("custom") segments (#1290), single-store scope.
//!
//! An application record is one segment of a type in
//! [`CUSTOM_SEGMENT_TYPE_MIN`]..=[`CUSTOM_SEGMENT_TYPE_MAX`], with payload
//! `key_len: u16 LE | key | value`. It is written through the same segment
//! writer as built-in segments, so it gets the store's content hash (SHAKE-256
//! in a signed store), a directory entry in the next manifest (and therefore
//! the signed root page's coverage), and an fsync before that manifest. Each
//! append also chains a witness entry whose action is
//! `"rvf.custom_segment\0" | type | seg_id (u64 LE) | payload`.
//!
//! Lookup is latest-wins per `(type, key)` in directory order. Compaction
//! carries every record verbatim, as it already does for unknown segment
//! types, so superseded versions are kept (open question 2 in #1290). Branch
//! inheritance and tombstones are not implemented: a COW child sees only its
//! own records.
//!
//! Format compatibility: readers that predate this API never look the type up
//! (the directory carries raw type bytes); they open the file and carry the
//! segments through compaction.

use super::*;

/// Lowest type byte reserved for application segments.
///
/// The range stops below 0xF0 because ADR-030 keeps 0xF0..=0xFF reserved
/// for the format (0xFE is already the stream root-page pseudo-entry).
pub const CUSTOM_SEGMENT_TYPE_MIN: u8 = 0xC0;
/// Highest type byte reserved for application segments.
pub const CUSTOM_SEGMENT_TYPE_MAX: u8 = 0xEF;
/// Longest application key, in bytes.
pub const MAX_CUSTOM_SEGMENT_KEY_LEN: usize = 255;
/// Largest application value, in bytes (16 MiB).
pub const MAX_CUSTOM_SEGMENT_VALUE_LEN: usize = 16 * 1024 * 1024;

const WITNESS_ACTION_TAG: &[u8] = b"rvf.custom_segment\0";

/// One application record: `(key, value)`.
pub type CustomRecord = (Vec<u8>, Vec<u8>);

/// A segment type byte checked to lie in the application range
/// (`0xC0..=0xEF`), which the format never assigns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CustomSegmentType(u8);

impl CustomSegmentType {
    /// Check `value` against the application range (`UnknownSegmentType`
    /// outside it).
    pub fn new(value: u8) -> Result<Self, RvfError> {
        if (CUSTOM_SEGMENT_TYPE_MIN..=CUSTOM_SEGMENT_TYPE_MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err(err(ErrorCode::UnknownSegmentType))
        }
    }

    /// The raw segment type byte.
    pub fn get(self) -> u8 {
        self.0
    }
}

/// Split a custom segment payload into `(key, value)`.
fn decode_record(payload: &[u8]) -> Result<(&[u8], &[u8]), RvfError> {
    let len_bytes: [u8; 2] = payload
        .get(..2)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| err(ErrorCode::TruncatedSegment))?;
    let key_len = usize::from(u16::from_le_bytes(len_bytes));
    if key_len > MAX_CUSTOM_SEGMENT_KEY_LEN {
        return Err(err(ErrorCode::TruncatedSegment));
    }
    let key = payload
        .get(2..2 + key_len)
        .ok_or_else(|| err(ErrorCode::TruncatedSegment))?;
    Ok((key, &payload[2 + key_len..]))
}

impl RvfStore {
    /// Append an application record `(key, value)` of type `ty`. A later
    /// append with the same `(ty, key)` supersedes it for
    /// [`RvfStore::read_segment`]. Returns the new segment's id.
    ///
    /// Durable on return: segment, witness entry, fsync, then the manifest
    /// that lists them (re-signed in a signed store). Keys longer than
    /// [`MAX_CUSTOM_SEGMENT_KEY_LEN`] or values longer than
    /// [`MAX_CUSTOM_SEGMENT_VALUE_LEN`] are rejected with `SegmentTooLarge`.
    pub fn append_segment(
        &mut self,
        ty: CustomSegmentType,
        key: &[u8],
        value: &[u8],
    ) -> Result<u64, RvfError> {
        if self.is_write_blocked() {
            return Err(err(ErrorCode::ReadOnly));
        }
        if key.len() > MAX_CUSTOM_SEGMENT_KEY_LEN || value.len() > MAX_CUSTOM_SEGMENT_VALUE_LEN {
            return Err(err(ErrorCode::SegmentTooLarge));
        }
        let mut payload = Vec::with_capacity(2 + key.len() + value.len());
        payload.extend_from_slice(&(key.len() as u16).to_le_bytes());
        payload.extend_from_slice(key);
        payload.extend_from_slice(value);

        let old_directory = self.segment_dir.clone();
        let old_epoch = self.epoch;
        let result = (|| {
            let writer = self
                .seg_writer
                .as_mut()
                .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
            let (seg_id, seg_offset) = {
                let mut buf_writer = Self::append_cursor(&self.file, &self.security)?;
                let written = writer
                    .write_typed_seg(&mut buf_writer, ty.get(), &payload)
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;
                buf_writer
                    .flush()
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;
                written
            };
            self.segment_dir
                .push((seg_id, seg_offset, payload.len() as u64, ty.get()));

            let mut action = Vec::with_capacity(WITNESS_ACTION_TAG.len() + 9 + payload.len());
            action.extend_from_slice(WITNESS_ACTION_TAG);
            action.push(ty.get());
            action.extend_from_slice(&seg_id.to_le_bytes());
            action.extend_from_slice(&payload);
            self.append_witness(witness_types::DATA_PROVENANCE, &action)?;

            // Data before pointer (ADR-280 §4).
            self.file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
            self.write_manifest()?;
            Ok(seg_id)
        })();
        if result.is_err() {
            // The bytes stay in the append-only file but are not published.
            self.segment_dir = old_directory;
            self.epoch = old_epoch;
        }
        result
    }

    /// The latest value stored under `(ty, key)` in this store, if any.
    pub fn read_segment(
        &self,
        ty: CustomSegmentType,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, RvfError> {
        for &(_, offset, _, seg_type) in self.segment_dir.iter().rev() {
            if seg_type != ty.get() {
                continue;
            }
            let payload = self.read_custom_payload(offset)?;
            let (record_key, value) = decode_record(&payload)?;
            if record_key == key {
                return Ok(Some(value.to_vec()));
            }
        }
        Ok(None)
    }

    /// Every live `(key, value)` of type `ty` in this store: the latest
    /// value per key, ordered by when that latest value was written.
    pub fn list_segments(&self, ty: CustomSegmentType) -> Result<Vec<CustomRecord>, RvfError> {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut newest_first = Vec::new();
        for &(_, offset, _, seg_type) in self.segment_dir.iter().rev() {
            if seg_type != ty.get() {
                continue;
            }
            let payload = self.read_custom_payload(offset)?;
            let (key, value) = decode_record(&payload)?;
            if seen.insert(key.to_vec()) {
                newest_first.push((key.to_vec(), value.to_vec()));
            }
        }
        newest_first.reverse();
        Ok(newest_first)
    }

    fn read_custom_payload(&self, offset: u64) -> Result<Vec<u8>, RvfError> {
        let mut reader = BufReader::new(&self.file);
        self.read_seg(&mut reader, offset)
            .map(|(_, payload)| payload)
            .map_err(|e| Self::map_read_error(&e, ErrorCode::InvalidChecksum))
    }
}
