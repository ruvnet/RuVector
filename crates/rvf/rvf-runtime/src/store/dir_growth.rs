//! Keeps the segment directory, and so every manifest, small under many
//! small commits (#1382).
//!
//! Each commit appends a manifest that carries the full segment directory.
//! Before this module a single-vector ingest added three directory entries
//! (VEC_SEG, WITNESS_SEG and the previous MANIFEST_SEG), so the k-th ingest
//! wrote an O(k) manifest and N ingests grew the file O(N^2): 3,000 single
//! ingests of 128-dim vectors produced a 341 MB file for 1.5 MB of vectors.
//!
//! Three changes bound the directory without changing the file format:
//!
//! 1. Superseded manifests are dropped from the directory (only the newest
//!    previous one stays listed). Recovery never used them: it scans the file
//!    for the newest valid manifest.
//! 2. Only the newest [`WITNESS_DIR_WINDOW`] WITNESS_SEGs stay listed. The
//!    chain itself lives in the file and `rvf verify-witness` walks the file;
//!    the runtime only reads the newest listed witnesses to restore the chain
//!    head, and never looks further back than this window.
//!    Likewise only the newest [`META_DIR_WINDOW`] META_SEGs (two metadata
//!    snapshot cycles) stay listed; replay stops at the newest snapshot.
//! 3. Small VEC_SEGs are merged size-tiered: when the newest
//!    [`VEC_MERGE_FANOUT`] VEC_SEGs fall in the same size class they are
//!    rewritten as one segment. Each vector is rewritten about
//!    `log_F(n)` times, and merging stops once a merged segment would exceed
//!    [`VEC_MERGE_MAX_PAYLOAD`], so bulk ingests are never rewritten.
//!
//! The in-memory vectors, HNSW index and RaBitQ codes are untouched by a
//! merge: it only changes how the same rows are laid out on disk.

use super::*;

/// WITNESS_SEG entries kept in the segment directory. Matches the look-back
/// of `restore_witness_head`, so pruning never changes the restored head.
pub(super) const WITNESS_DIR_WINDOW: usize = 64;

/// META_SEG entries kept in the segment directory: the current snapshot
/// cycle plus the whole previous one (a cycle is a full snapshot and at most
/// `META_SNAPSHOT_INTERVAL` deltas), so the torn-snapshot fallback in
/// `restore_metadata` keeps an older snapshot to fall back to. Replay never
/// walks past the newest readable snapshot.
pub(super) const META_DIR_WINDOW: usize = 2 * (META_SNAPSHOT_INTERVAL as usize + 1);

/// Number of same-class VEC_SEGs merged into one.
pub(super) const VEC_MERGE_FANOUT: usize = 8;

/// Largest merged VEC_SEG payload. Segments at or above this size are left
/// alone, so a store ingested in large batches never pays for a rewrite.
pub(super) const VEC_MERGE_MAX_PAYLOAD: u64 = 4 << 20;

/// `floor(log_F(count))`: segments within a factor of `F` share a class.
fn size_class(count: u64) -> u32 {
    let mut class = 0;
    let mut x = count.max(1);
    while x >= VEC_MERGE_FANOUT as u64 {
        x /= VEC_MERGE_FANOUT as u64;
        class += 1;
    }
    class
}

/// Vector ids of a VEC_SEG payload written for `dim`, or `None` if the
/// payload does not have that shape.
fn vec_seg_ids(payload: &[u8], dim: usize) -> Option<Vec<u64>> {
    if payload.len() < 6 || u16::from_le_bytes([payload[0], payload[1]]) as usize != dim {
        return None;
    }
    let count = u32::from_le_bytes(payload[2..6].try_into().ok()?) as usize;
    let row = 8 + dim * 4;
    if payload.len() != count.checked_mul(row)?.checked_add(6)? {
        return None;
    }
    Some(
        (0..count)
            .map(|i| {
                let at = 6 + i * row;
                u64::from_le_bytes(payload[at..at + 8].try_into().expect("8-byte id"))
            })
            .collect(),
    )
}

impl RvfStore {
    /// Drop superseded MANIFEST entries, and all but the newest
    /// [`WITNESS_DIR_WINDOW`] WITNESS and [`META_DIR_WINDOW`] META entries,
    /// from the segment directory.
    /// Called right before a manifest is written.
    pub(super) fn prune_superseded_dir_entries(&mut self) {
        let manifest = SegmentType::Manifest as u8;
        if let Some(newest) = self.segment_dir.iter().rposition(|e| e.3 == manifest) {
            let mut position = 0usize;
            self.segment_dir.retain(|e| {
                let keep = e.3 != manifest || position == newest;
                position += 1;
                keep
            });
        }

        self.keep_newest(SegmentType::Witness as u8, WITNESS_DIR_WINDOW);
        self.keep_newest(SegmentType::Meta as u8, META_DIR_WINDOW);
    }

    /// Keep only the newest `window` directory entries of `seg_type`.
    fn keep_newest(&mut self, seg_type: u8, window: usize) {
        let listed = self.segment_dir.iter().filter(|e| e.3 == seg_type).count();
        let mut excess = listed.saturating_sub(window);
        if excess > 0 {
            self.segment_dir.retain(|e| {
                if e.3 == seg_type && excess > 0 {
                    excess -= 1;
                    false
                } else {
                    true
                }
            });
        }
    }

    /// Size-tiered merge of the newest VEC_SEGs (see the module docs).
    ///
    /// `newest` is the segment the current ingest just wrote and its ids; it
    /// is not committed yet, so its ids come from the caller rather than a
    /// read. Rows are taken from the in-memory vectors, which hold exactly
    /// the newest write of every id. A segment that cannot be read leaves the
    /// directory as it is (the merge is only an optimisation); a failed write
    /// is returned so the ingest rolls back like any other write failure.
    pub(super) fn merge_trailing_vec_segs(
        &mut self,
        mut newest: (u64, Vec<u64>),
    ) -> Result<(), RvfError> {
        if self.cow_engine.is_some() || self.membership_filter.is_some() {
            return Ok(());
        }
        let dim = self.options.dimension as usize;
        let row = 8 + dim as u64 * 4;
        let vec_type = SegmentType::Vec as u8;

        loop {
            let positions: Vec<usize> = self
                .segment_dir
                .iter()
                .enumerate()
                .filter(|(_, e)| e.3 == vec_type)
                .map(|(i, _)| i)
                .collect();
            if positions.len() < VEC_MERGE_FANOUT {
                return Ok(());
            }
            let tail = &positions[positions.len() - VEC_MERGE_FANOUT..];

            let mut total_payload = 0u64;
            let mut classes = Vec::with_capacity(tail.len());
            for &p in tail {
                let payload_len = self.segment_dir[p].2;
                if payload_len < 6 || !(payload_len - 6).is_multiple_of(row) {
                    return Ok(());
                }
                classes.push(size_class((payload_len - 6) / row));
                total_payload += payload_len;
            }
            let class = *classes.last().expect("fanout > 0");
            if classes.iter().any(|&c| c != class) || total_payload > VEC_MERGE_MAX_PAYLOAD {
                return Ok(());
            }

            // Ids in directory order; the newest occurrence of an id wins.
            let mut ids: Vec<u64> = Vec::new();
            for &p in tail {
                let (seg_id, offset, _, _) = self.segment_dir[p];
                if seg_id == newest.0 {
                    ids.extend_from_slice(&newest.1);
                    continue;
                }
                let payload = {
                    let mut reader = BufReader::new(&self.file);
                    match self.read_seg(&mut reader, offset) {
                        Ok((_, payload)) => payload,
                        Err(_) => return Ok(()),
                    }
                };
                match vec_seg_ids(&payload, dim) {
                    Some(seg_ids) => ids.extend(seg_ids),
                    None => return Ok(()),
                }
            }
            let mut seen = HashSet::with_capacity(ids.len());
            let mut merged_ids: Vec<u64> = ids
                .into_iter()
                .rev()
                .filter(|id| seen.insert(*id))
                .collect();
            merged_ids.reverse();
            merged_ids.retain(|&id| self.vectors.get(id).is_some());

            let rows: Vec<&[f32]> = merged_ids
                .iter()
                .map(|&id| self.vectors.get(id).expect("retained above"))
                .collect();
            let writer = self
                .seg_writer
                .as_mut()
                .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
            let (seg_id, offset) = {
                let mut buf_writer = Self::append_cursor(&self.file, &self.security)?;
                let written = writer
                    .write_vec_seg(&mut buf_writer, &rows, &merged_ids, self.options.dimension)
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;
                buf_writer
                    .flush()
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;
                written
            };
            let payload_len = 6 + merged_ids.len() as u64 * row;

            for &p in tail.iter().rev() {
                self.segment_dir.remove(p);
            }
            self.segment_dir
                .push((seg_id, offset, payload_len, vec_type));
            newest = (seg_id, merged_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_group_by_fanout() {
        assert_eq!(size_class(0), 0);
        assert_eq!(size_class(1), 0);
        assert_eq!(size_class(7), 0);
        assert_eq!(size_class(8), 1);
        assert_eq!(size_class(63), 1);
        assert_eq!(size_class(64), 2);
    }

    const DIM: usize = 8;

    fn v(id: u64, salt: u64) -> Vec<f32> {
        (0..DIM as u64)
            .map(|j| {
                let mut z = (id << 20 ^ salt << 8 ^ j).wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    fn count(store: &RvfStore, t: SegmentType) -> usize {
        store
            .segment_dir()
            .iter()
            .filter(|e| e.3 == t as u8)
            .count()
    }

    fn exact(store: &RvfStore, q: &[f32]) -> Vec<SearchResult> {
        let o = QueryOptions {
            force_exact: true,
            ..Default::default()
        };
        store.query(q, 1, &o).unwrap()
    }

    #[test]
    fn single_ingests_keep_the_directory_bounded_and_reopen_intact() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("g.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: DIM as u16,
                metric: DistanceMetric::L2,
                ..Default::default()
            },
        )
        .unwrap();
        let n = 700u64;
        for id in 0..n {
            let meta = [MetadataEntry {
                field_id: 0,
                value: MetadataValue::U64(id),
            }];
            store
                .ingest_batch(&[v(id, 0).as_slice()], &[id], Some(&meta[..]))
                .unwrap();
        }
        // Overwrite ids that now live in merged segments, and delete some.
        for id in [3u64, 150, 699] {
            store
                .ingest_batch(&[v(id, 1).as_slice()], &[id], None)
                .unwrap();
        }
        store.delete(&[10, 11]).unwrap();

        assert!(count(&store, SegmentType::Vec) <= 4 * VEC_MERGE_FANOUT);
        assert!(count(&store, SegmentType::Witness) <= WITNESS_DIR_WINDOW);
        assert!(count(&store, SegmentType::Meta) <= META_DIR_WINDOW);
        assert!(count(&store, SegmentType::Manifest) <= 2);
        let head = *store.last_witness_hash();
        let live = store.status().total_vectors;
        store.close().unwrap();

        let store = RvfStore::open(&path).unwrap();
        assert_eq!(
            store.last_witness_hash(),
            &head,
            "witness head survives pruning"
        );
        assert_eq!(store.status().total_vectors, live);
        for id in 0..n {
            if id == 10 || id == 11 {
                assert_ne!(
                    exact(&store, &v(id, 0))[0].id,
                    id,
                    "deleted id {id} is gone"
                );
                continue;
            }
            let salt = u64::from([3u64, 150, 699].contains(&id));
            let hit = &exact(&store, &v(id, salt))[0];
            assert_eq!(hit.id, id, "newest value of {id} reads back");
            assert!(hit.distance <= 1e-9);
            assert_eq!(
                store.get_metadata(id).unwrap()[0].value,
                MetadataValue::U64(id),
                "metadata of {id}"
            );
        }
    }

    #[test]
    fn vec_seg_ids_rejects_wrong_shape() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&42u64.to_le_bytes());
        payload.extend_from_slice(&[0u8; 8]);
        assert_eq!(vec_seg_ids(&payload, 2), Some(vec![42]));
        assert_eq!(vec_seg_ids(&payload, 3), None);
        assert_eq!(vec_seg_ids(&payload[..payload.len() - 1], 2), None);
    }
}
