//! Payload layouts the registry interprets: the rvf runtime's MANIFEST_SEG
//! (`rvf-runtime/src/write_path.rs::write_manifest_seg_with_identity`) and
//! its flat VEC_SEG (`dimension u16 | count u32 | (id u64, f32 × dim)*`).
//! Both are parsed strictly: every length is checked in `u64` before any
//! slice is taken (a `usize` product can wrap on wasm32), reserved bytes must
//! be zero, and the only trailer accepted after the deletion list is the
//! `FIDI` file-identity record.

use super::error::ValidationError;

/// Fixed manifest header: epoch, dim, total_vectors, seg_count, profile,
/// metric, 2 reserved bytes.
const MANIFEST_HEADER: u64 = 22;
/// One directory entry: seg_id, offset, payload_length, seg_type.
const DIR_ENTRY: u64 = 25;
/// `"FIDI"` marker and the 68-byte `FileIdentity` that follows it.
const FIDI_MAGIC: u32 = 0x4649_4449;
const FIDI_LEN: u64 = 68;
/// VEC_SEG payload prefix: dimension (u16) and vector count (u32).
pub(crate) const VEC_HEAD: usize = 6;

/// One manifest directory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirEntry {
    pub seg_id: u64,
    pub offset: u64,
    pub payload_length: u64,
    pub seg_type: u8,
}

/// A parsed runtime manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeManifest {
    pub epoch: u32,
    pub dim: u16,
    pub total_vectors: u64,
    pub profile_id: u8,
    pub metric_id: u8,
    pub directory: Vec<DirEntry>,
    pub deleted_ids: Vec<u64>,
    pub file_identity: bool,
}

fn u16_at(p: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([p[i], p[i + 1]])
}
fn u32_at(p: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]])
}
fn u64_at(p: &[u8], i: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&p[i..i + 8]);
    u64::from_le_bytes(b)
}

/// Parse a MANIFEST_SEG payload found at `offset`. `max_entries` bounds the
/// directory (the segments seen before this manifest, never more than
/// `max_segments`) and is checked before anything is allocated. Directory
/// offsets must be strictly increasing: every entry names a distinct
/// segment, so the live set can never amplify one segment into many.
pub(crate) fn parse_manifest(
    p: &[u8],
    offset: u64,
    max_entries: u64,
) -> Result<RuntimeManifest, ValidationError> {
    let bad = |reason| ValidationError::MalformedManifest { offset, reason };
    let len = p.len() as u64;
    if len < MANIFEST_HEADER + 4 {
        return Err(bad("shorter than the fixed header"));
    }
    if p[20] != 0 || p[21] != 0 {
        return Err(bad("reserved header bytes are non-zero"));
    }
    let seg_count = u64::from(u32_at(p, 14));
    if seg_count > max_entries {
        return Err(bad("directory has more entries than segments"));
    }
    let dir_end = seg_count
        .checked_mul(DIR_ENTRY)
        .and_then(|d| d.checked_add(MANIFEST_HEADER))
        .filter(|end| end + 4 <= len)
        .ok_or(bad("directory overruns the payload"))?;
    // In bounds by the check above; `dir_end` fits in usize because it is
    // at most `p.len()`.
    let dir_end = dir_end as usize;
    let directory: Vec<DirEntry> = p[MANIFEST_HEADER as usize..dir_end]
        .chunks_exact(DIR_ENTRY as usize)
        .map(|e| DirEntry {
            seg_id: u64_at(e, 0),
            offset: u64_at(e, 8),
            payload_length: u64_at(e, 16),
            seg_type: e[24],
        })
        .collect();
    if directory.windows(2).any(|w| w[0].offset >= w[1].offset) {
        return Err(bad("directory offsets are not strictly increasing"));
    }
    let del_count = u64::from(u32_at(p, dir_end));
    let del_start = dir_end as u64 + 4;
    let del_end = del_count
        .checked_mul(8)
        .and_then(|d| d.checked_add(del_start))
        .filter(|end| *end <= len)
        .ok_or(bad("deletion list overruns the payload"))?;
    let deleted_ids = p[del_start as usize..del_end as usize]
        .chunks_exact(8)
        .map(|c| u64_at(c, 0))
        .collect();
    let file_identity = match len - del_end {
        0 => false,
        n if n == 4 + FIDI_LEN && u32_at(p, del_end as usize) == FIDI_MAGIC => true,
        _ => return Err(bad("unrecognised trailer after the deletion list")),
    };
    Ok(RuntimeManifest {
        epoch: u32_at(p, 0),
        dim: u16_at(p, 4),
        total_vectors: u64_at(p, 6),
        profile_id: p[18],
        metric_id: p[19],
        directory,
        deleted_ids,
        file_identity,
    })
}

/// Check a VEC_SEG prefix against the header's payload length. Returns the
/// `(dimension, count)` it declares.
pub(crate) fn check_vec_head(
    head: &[u8; VEC_HEAD],
    payload_length: u64,
    offset: u64,
) -> Result<(u16, u32), ValidationError> {
    let dim = u16_at(head, 0);
    let count = u32_at(head, 2);
    if dim == 0 {
        return Err(ValidationError::VecSegMalformed {
            offset,
            reason: "zero dimension",
        });
    }
    let expected = (u64::from(dim) * 4 + 8)
        .checked_mul(u64::from(count))
        .and_then(|b| b.checked_add(VEC_HEAD as u64));
    if expected != Some(payload_length) {
        return Err(ValidationError::VecSegMalformed {
            offset,
            reason: "payload length does not match dimension × count",
        });
    }
    Ok((dim, count))
}

/// A decoded view of one VEC_SEG payload. Iterates `(id, vector)` records.
#[derive(Debug, Clone)]
pub struct VecSegView<'a> {
    dim: usize,
    records: core::slice::ChunksExact<'a, u8>,
}

impl<'a> VecSegView<'a> {
    /// Decode `payload` (the bytes after the 64-byte header) of a VEC_SEG
    /// that must hold `expected_dim`-dimensional vectors. Re-checks the
    /// layout, so it is safe on bytes that did not come from the validator.
    pub fn decode(payload: &'a [u8], expected_dim: u16) -> Result<Self, ValidationError> {
        let mut head = [0u8; VEC_HEAD];
        let prefix = payload
            .get(..VEC_HEAD)
            .ok_or(ValidationError::VecSegMalformed {
                offset: 0,
                reason: "shorter than the prefix",
            })?;
        head.copy_from_slice(prefix);
        let (dim, _) = check_vec_head(&head, payload.len() as u64, 0)?;
        if dim != expected_dim {
            return Err(ValidationError::DimensionMismatch {
                expected: expected_dim,
                found: dim,
            });
        }
        let dim = usize::from(dim);
        Ok(VecSegView {
            dim,
            records: payload[VEC_HEAD..].chunks_exact(8 + 4 * dim),
        })
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Iterator for VecSegView<'_> {
    type Item = (u64, Vec<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let rec = self.records.next()?;
        let id = u64_at(rec, 0);
        let values = rec[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some((id, values))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.records.size_hint()
    }
}

impl ExactSizeIterator for VecSegView<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(seg_count: u32, dir: &[u8], dels: &[u64], trailer: &[u8]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&7u32.to_le_bytes());
        p.extend_from_slice(&4u16.to_le_bytes());
        p.extend_from_slice(&3u64.to_le_bytes());
        p.extend_from_slice(&seg_count.to_le_bytes());
        p.extend_from_slice(&[0, 2, 0, 0]);
        p.extend_from_slice(dir);
        p.extend_from_slice(&(dels.len() as u32).to_le_bytes());
        dels.iter()
            .for_each(|d| p.extend_from_slice(&d.to_le_bytes()));
        p.extend_from_slice(trailer);
        p
    }

    #[test]
    fn parses_header_directory_deletions_and_fidi() {
        let mut dir = Vec::new();
        dir.extend_from_slice(&1u64.to_le_bytes());
        dir.extend_from_slice(&64u64.to_le_bytes());
        dir.extend_from_slice(&10u64.to_le_bytes());
        dir.push(1);
        let mut fidi = FIDI_MAGIC.to_le_bytes().to_vec();
        fidi.extend_from_slice(&[9u8; 68]);
        let m = parse_manifest(&manifest(1, &dir, &[5, 6], &fidi), 0, 1).unwrap();
        assert_eq!((m.epoch, m.dim, m.total_vectors, m.metric_id), (7, 4, 3, 2));
        assert_eq!(m.directory.len(), 1);
        assert_eq!(m.directory[0].offset, 64);
        assert_eq!(m.deleted_ids, vec![5, 6]);
        assert!(m.file_identity);
    }

    #[test]
    fn rejects_oversized_or_repeated_directories() {
        let entry = |off: u64| {
            let mut e = Vec::new();
            e.extend_from_slice(&1u64.to_le_bytes());
            e.extend_from_slice(&off.to_le_bytes());
            e.extend_from_slice(&10u64.to_le_bytes());
            e.push(1);
            e
        };
        let dup = [entry(64), entry(64)].concat();
        let why = |r: Result<RuntimeManifest, ValidationError>| match r {
            Err(ValidationError::MalformedManifest { reason, .. }) => reason,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            why(parse_manifest(&manifest(2, &dup, &[], &[]), 0, 4)),
            "directory offsets are not strictly increasing"
        );
        let down = [entry(128), entry(64)].concat();
        assert!(parse_manifest(&manifest(2, &down, &[], &[]), 0, 4).is_err());
        let up = [entry(64), entry(128)].concat();
        assert!(parse_manifest(&manifest(2, &up, &[], &[]), 0, 2).is_ok());
        assert_eq!(
            why(parse_manifest(&manifest(2, &up, &[], &[]), 0, 1)),
            "directory has more entries than segments"
        );
    }

    #[test]
    fn rejects_overruns_trailers_and_reserved_bytes() {
        assert!(parse_manifest(&manifest(u32::MAX, &[], &[], &[]), 0, u64::MAX).is_err());
        assert!(parse_manifest(&manifest(0, &[], &[], &[1, 2, 3]), 0, 0).is_err());
        let mut p = manifest(0, &[], &[], &[]);
        p[20] = 1;
        assert!(parse_manifest(&p, 0, 0).is_err());
        let mut p = manifest(0, &[], &[], &[]);
        let n = p.len();
        p[n - 4..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_manifest(&p, 0, 0).is_err());
        assert!(parse_manifest(&[0u8; 25], 0, 0).is_err());
    }

    #[test]
    fn vec_head_checks_exact_length_with_wide_arithmetic() {
        let head = |d: u16, c: u32| {
            let mut h = [0u8; VEC_HEAD];
            h[..2].copy_from_slice(&d.to_le_bytes());
            h[2..].copy_from_slice(&c.to_le_bytes());
            h
        };
        assert_eq!(check_vec_head(&head(2, 3), 6 + 3 * 16, 0).unwrap(), (2, 3));
        assert!(check_vec_head(&head(2, 3), 6 + 3 * 16 + 1, 0).is_err());
        assert!(check_vec_head(&head(0, 0), 6, 0).is_err());
        assert!(check_vec_head(&head(u16::MAX, u32::MAX), u64::MAX, 0).is_err());
    }

    #[test]
    fn view_decodes_records_and_checks_dimension() {
        let mut p = Vec::new();
        p.extend_from_slice(&2u16.to_le_bytes());
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&42u64.to_le_bytes());
        p.extend_from_slice(&1.5f32.to_le_bytes());
        p.extend_from_slice(&(-2.0f32).to_le_bytes());
        let got: Vec<_> = VecSegView::decode(&p, 2).unwrap().collect();
        assert_eq!(got, vec![(42, vec![1.5, -2.0])]);
        assert!(VecSegView::decode(&p, 3).is_err());
        assert!(VecSegView::decode(&p[..5], 2).is_err());
    }
}
