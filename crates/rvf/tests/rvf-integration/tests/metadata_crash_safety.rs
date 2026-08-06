//! Crash-injection durability tests for ADR-280 self-contained metadata.
//!
//! Maps to ADR-280 acceptance criterion 6:
//!
//! > Crash injection after every commit-protocol write, fsync, rename,
//! > selector, and directory-fsync boundary exposes exactly the prior or new
//! > complete snapshot, never a mixed or torn snapshot.
//!
//! Mechanism. RVF is a single append-only file. Committing a metadata batch
//! appends a VEC segment, a META segment (a delta against the previous
//! generation, or a full snapshot when the delta chain is materialized), an
//! optional witness segment, and finally a MANIFEST segment, fsyncing before
//! and after the manifest write (see `RvfStore::ingest_batch_internal` /
//! `write_manifest`). The manifest is always written last, and
//! `read_path::find_latest_manifest` scans backward from EOF, skipping any
//! torn or unparseable manifest and falling back to the previous valid one.
//!
//! A crash during batch B therefore corresponds to the file being persisted
//! only up to some length `L` in `(len_after_A, len_after_B]`. This is exactly
//! the crash-injection mechanism the existing `e2e_crash_safety.rs` and
//! `cow_crash_recovery.rs` tests use (raw-byte truncation followed by reopen);
//! here it is driven across *every* segment/commit boundary of the metadata
//! commit protocol rather than a single hand-picked offset.
//!
//! For every such truncation `L`, reopening must succeed and expose the
//! metadata of exactly the prior (batch-A) snapshot or exactly the new
//! (batch-A + batch-B) snapshot -- never a partially applied mixture and
//! never an unrecoverable file.

use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, RvfOptions, VectorMetadata,
};
use rvf_runtime::RvfStore;
use rvf_types::{SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use tempfile::TempDir;

fn make_options(dim: u16) -> RvfOptions {
    RvfOptions {
        dimension: dim,
        metric: DistanceMetric::L2,
        ..Default::default()
    }
}

/// The observable metadata snapshot: per-vector records (`None` for an absent
/// record, `Some(fields)` sorted by `field_id` for a present one, including the
/// empty record) plus the probed file-level keys.
#[derive(Debug, Default, PartialEq)]
struct Snapshot {
    records: BTreeMap<u64, Option<Vec<MetadataEntry>>>,
    file: BTreeMap<String, Option<MetadataValue>>,
}

/// Open `bytes` as an RVF file and read back the metadata for `ids` and
/// `file_keys`.
fn snapshot_of(
    bytes: &[u8],
    ids: &[u64],
    file_keys: &[&str],
    probe_path: &std::path::Path,
) -> Snapshot {
    fs::write(probe_path, bytes).unwrap();
    let store = RvfStore::open_readonly(probe_path)
        .unwrap_or_else(|e| panic!("reopen after crash must recover a valid snapshot: {e:?}"));
    let mut snap = Snapshot::default();
    for &id in ids {
        let entry = store.get_metadata(id).map(|mut fields| {
            fields.sort_by_key(|f| f.field_id);
            fields
        });
        snap.records.insert(id, entry);
    }
    for &key in file_keys {
        snap.file
            .insert(key.to_string(), store.get_file_metadata(key).cloned());
    }
    snap
}

/// Walk the append-only file and return `(start, end)` byte ranges for every
/// complete, structurally-valid segment, in file order. Stops at the first
/// position that is not a segment header (e.g. a torn tail).
fn segment_ranges(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut off = 0usize;
    while off + SEGMENT_HEADER_SIZE <= bytes.len() {
        let magic = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if magic != SEGMENT_MAGIC {
            break;
        }
        let payload_len =
            u64::from_le_bytes(bytes[off + 0x10..off + 0x18].try_into().unwrap()) as usize;
        // Segments are written back-to-back: header, then payload, with no
        // inter-segment padding (`SegmentWriter::write_segment`).
        let end = match off.checked_add(SEGMENT_HEADER_SIZE + payload_len) {
            Some(end) if end <= bytes.len() => end,
            _ => break,
        };
        ranges.push((off, end));
        off = end;
    }
    ranges
}

/// Build the set of truncation lengths to test: every commit-protocol boundary
/// in the batch-B region, plus header-only / mid-payload / final-padding torn
/// points within each appended segment (so the torn-manifest fallback path is
/// exercised explicitly), plus a byte-granular sweep across the whole region.
fn crash_points(len_a: usize, len_b: usize, ranges: &[(usize, usize)]) -> Vec<usize> {
    let mut points: BTreeSet<usize> = BTreeSet::new();
    points.insert(len_a);
    points.insert(len_b);

    for &(start, end) in ranges {
        if end <= len_a {
            continue; // segment belongs to the already-committed batch-A prefix
        }
        // End of a complete appended segment (a clean crash boundary) and the
        // byte just before it (a segment missing its final byte).
        points.insert(end);
        points.insert(end - 1);
        // Header written but payload not yet complete.
        let header_only = start + SEGMENT_HEADER_SIZE;
        if header_only < end {
            points.insert(header_only);
        }
        // Torn mid-segment write.
        points.insert(start + (end - start) / 2);
    }

    // Byte-granular sweep, bounded to keep the probe count reasonable while
    // still covering torn points that do not fall on a segment boundary.
    let region = len_b.saturating_sub(len_a);
    let step = (region / 2048).max(1);
    let mut l = len_a;
    while l <= len_b {
        points.insert(l);
        l += step;
    }

    points
        .into_iter()
        .filter(|&l| l >= len_a && l <= len_b)
        .collect()
}

/// Drive the crash-injection sweep for a batch-B commit and assert every
/// truncation recovers to exactly the batch-A or the batch-A+B snapshot.
fn assert_atomic_metadata_commit(
    label: &str,
    all_ids: &[u64],
    file_keys: &[&str],
    bytes_a: &[u8],
    bytes_b: &[u8],
    probe_path: &std::path::Path,
) {
    let len_a = bytes_a.len();
    let len_b = bytes_b.len();

    // The format is append-only: batch B must extend batch A's bytes verbatim.
    assert!(len_b > len_a, "[{label}] batch B must append new bytes");
    assert_eq!(
        &bytes_b[..len_a],
        bytes_a,
        "[{label}] batch B must not rewrite the committed batch-A prefix"
    );

    let expected_a = snapshot_of(bytes_a, all_ids, file_keys, probe_path);
    let expected_b = snapshot_of(bytes_b, all_ids, file_keys, probe_path);
    assert_ne!(
        expected_a, expected_b,
        "[{label}] the two snapshots must be distinguishable for the test to mean anything"
    );

    let ranges = segment_ranges(bytes_b);
    let points = crash_points(len_a, len_b, &ranges);
    assert!(
        points.len() >= 4,
        "[{label}] expected several crash-injection points, got {}",
        points.len()
    );

    let mut saw_prior = false;
    let mut saw_new = false;
    for l in points {
        let observed = snapshot_of(&bytes_b[..l], all_ids, file_keys, probe_path);
        if observed == expected_a {
            saw_prior = true;
        } else if observed == expected_b {
            saw_new = true;
        } else {
            panic!(
                "[{label}] crash at byte {l}/{len_b} exposed a mixed/torn metadata snapshot: \
                 observed={observed:?}\n  prior (A)={expected_a:?}\n  new (A+B)={expected_b:?}"
            );
        }
    }

    // A crash exactly at len_a yields the prior snapshot; the full file yields
    // the new one. Both endpoints are in the sweep, so both must be observed.
    assert!(saw_prior, "[{label}] prior snapshot never observed");
    assert!(saw_new, "[{label}] new snapshot never observed");
}

/// Criterion 6: a metadata commit that upserts new records and overrides an
/// existing record is atomic under crash injection at every boundary.
#[test]
fn metadata_upsert_commit_is_atomic_under_crash_injection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("meta_crash_upsert.rvf");
    let probe = dir.path().join("probe.rvf");
    let dim: u16 = 2;
    let all_ids = [10u64, 20, 30, 40, 50];

    let mut store = RvfStore::create(&path, make_options(dim)).unwrap();

    // Batch A: three records with unequal field counts, including an empty one.
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]],
            &[10, 20, 30],
            &[
                VectorMetadata {
                    vector_id: 10,
                    fields: vec![
                        MetadataEntry {
                            field_id: 1,
                            value: MetadataValue::String("ten".into()),
                        },
                        MetadataEntry {
                            field_id: 2,
                            value: MetadataValue::U64(10),
                        },
                    ],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 20,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("twenty".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 30,
                    fields: vec![],
                    delete_record: false,
                },
            ],
        )
        .unwrap();
    let bytes_a = fs::read(&path).unwrap();

    // Batch B: override record 10 (whole-record replace) and add records 40, 50.
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 0.0], &[0.5, 0.5]],
            &[10, 40, 50],
            &[
                VectorMetadata {
                    vector_id: 10,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("ten-v2".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 40,
                    fields: vec![
                        MetadataEntry {
                            field_id: 1,
                            value: MetadataValue::String("forty".into()),
                        },
                        MetadataEntry {
                            field_id: 3,
                            value: MetadataValue::U64(40),
                        },
                    ],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 50,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("fifty".into()),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();
    let bytes_b = fs::read(&path).unwrap();
    store.close().unwrap();

    assert_atomic_metadata_commit("upsert", &all_ids, &[], &bytes_a, &bytes_b, &probe);
}

/// Criterion 6 (delete path): a metadata commit that tombstones an existing
/// record and adds a new one is atomic under crash injection at every boundary.
#[test]
fn metadata_tombstone_commit_is_atomic_under_crash_injection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("meta_crash_tombstone.rvf");
    let probe = dir.path().join("probe.rvf");
    let dim: u16 = 2;
    let all_ids = [10u64, 20, 30, 40];

    let mut store = RvfStore::create(&path, make_options(dim)).unwrap();

    // Batch A: same three-record base.
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]],
            &[10, 20, 30],
            &[
                VectorMetadata {
                    vector_id: 10,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("ten".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 20,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("twenty".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 30,
                    fields: vec![],
                    delete_record: false,
                },
            ],
        )
        .unwrap();
    let bytes_a = fs::read(&path).unwrap();

    // Batch B: tombstone record 20 (vector 20 stays live) and add record 40.
    store
        .ingest_batch_with_metadata(
            &[&[0.25, 0.75]],
            &[40],
            &[
                VectorMetadata {
                    vector_id: 20,
                    fields: vec![],
                    delete_record: true,
                },
                VectorMetadata {
                    vector_id: 40,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("forty".into()),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();
    let bytes_b = fs::read(&path).unwrap();
    store.close().unwrap();

    assert_atomic_metadata_commit("tombstone", &all_ids, &[], &bytes_a, &bytes_b, &probe);
}

/// Criterion 6 (delete path): `delete` drops the metadata records of the
/// vectors it tombstones and commits that as its own metadata generation, so it
/// must be atomic under the same crash injection as an ingest.
#[test]
fn delete_metadata_commit_is_atomic_under_crash_injection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("meta_crash_delete.rvf");
    let probe = dir.path().join("probe.rvf");
    let dim: u16 = 2;
    let all_ids = [10u64, 20, 30];

    let mut store = RvfStore::create(&path, make_options(dim)).unwrap();
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]],
            &[10, 20, 30],
            &[
                VectorMetadata {
                    vector_id: 10,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("ten".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 20,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("twenty".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 30,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("thirty".into()),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();
    let bytes_a = fs::read(&path).unwrap();

    store.delete(&[20]).unwrap();
    let bytes_b = fs::read(&path).unwrap();
    store.close().unwrap();

    assert_atomic_metadata_commit("delete", &all_ids, &[], &bytes_a, &bytes_b, &probe);
}

/// The runtime's segment content hash: an IEEE CRC32 of the payload rotated
/// into four little-endian lanes (`rvf-runtime::hashing`). Rewriting a payload
/// in place means repairing this, or the reader rejects the segment as damaged.
fn legacy_content_hash(data: &[u8]) -> [u8; 16] {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    let crc = !crc;
    let mut hash = [0u8; 16];
    for lane in 0..4 {
        hash[lane * 4..(lane + 1) * 4]
            .copy_from_slice(&crc.rotate_left(lane as u32 * 8).to_le_bytes());
    }
    hash
}

/// Renumber the newest `META_SEG` of `path` to generation `u64::MAX`, so the
/// next metadata commit has no generation number left to allocate.
///
/// The payload is decoded and re-encoded rather than patched byte-wise, so it
/// stays a segment the reader accepts on its own terms; the generation is a
/// fixed-width field, so the replacement is the same length and the
/// append-only offsets the manifest records still hold.
fn exhaust_metadata_generation(path: &std::path::Path) {
    let mut bytes = fs::read(path).unwrap();
    let mut meta: Option<(usize, usize)> = None;
    let mut offset = 0usize;
    while offset + SEGMENT_HEADER_SIZE <= bytes.len() {
        if u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) != SEGMENT_MAGIC {
            break;
        }
        let payload_len =
            u64::from_le_bytes(bytes[offset + 0x10..offset + 0x18].try_into().unwrap()) as usize;
        let end = match offset.checked_add(SEGMENT_HEADER_SIZE + payload_len) {
            Some(end) if end <= bytes.len() => end,
            _ => break,
        };
        if bytes[offset + 0x05] == rvf_types::SegmentType::Meta as u8 {
            meta = Some((offset + SEGMENT_HEADER_SIZE, payload_len));
        }
        offset = end;
    }
    let (payload_start, payload_len) = meta.expect("the store must have written a META_SEG");

    let mut segment = rvf_types::metadata::MetadataSegment::decode(
        &bytes[payload_start..payload_start + payload_len],
    )
    .expect("the newest META_SEG must decode");
    assert!(
        segment.full_snapshot,
        "the newest generation must be the child's first full snapshot"
    );
    segment.generation = u64::MAX;
    let payload = segment.encode().unwrap();
    assert_eq!(
        payload.len(),
        payload_len,
        "renumbering must not change the encoded length"
    );

    bytes[payload_start..payload_start + payload_len].copy_from_slice(&payload);
    let header = payload_start - SEGMENT_HEADER_SIZE;
    bytes[header + 0x28..header + 0x38].copy_from_slice(&legacy_content_hash(&payload));
    fs::write(path, &bytes).unwrap();
}

/// A `delete` that fails must leave no trace in memory. It tombstones vectors
/// and prunes the membership filter before writing its metadata generation, so
/// a rollback that restored only the metadata would leave the store claiming
/// deletions it never committed -- and the next successful commit would publish
/// a manifest whose deleted set contradicts the committed metadata, which open
/// rejects outright.
///
/// The failure is injected at the head of `write_metadata_generation`, which
/// numbers the next generation with a checked increment: a chain whose newest
/// generation is `u64::MAX` cannot be extended, so the commit fails after the
/// tombstone and the membership prune have already been applied. Rewriting the
/// generation number on disk is a self-contained way to reach that state --
/// unlike an encoder-level failure it does not require the store to be left in
/// a shape a write should have rejected in the first place (issue #772).
#[test]
fn failed_delete_leaves_no_uncommitted_tombstones() {
    let dir = TempDir::new().unwrap();
    let parent_path = dir.path().join("failed_delete_parent.rvf");
    let child_path = dir.path().join("failed_delete_child.rvf");
    let dim: u16 = 2;

    let mut parent = RvfStore::create(&parent_path, make_options(dim)).unwrap();
    for id in [1u64, 2, 3] {
        parent
            .ingest_batch_with_metadata(
                &[&[id as f32, 1.0]],
                &[id],
                &[VectorMetadata {
                    vector_id: id,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(id),
                    }],
                    delete_record: false,
                }],
            )
            .unwrap();
    }

    // The child commits the inherited records as one full snapshot on its
    // first metadata write; that is the generation whose number is rewritten.
    let mut child = parent.branch(&child_path).unwrap();
    child
        .set_file_metadata("app.owner".into(), MetadataValue::String("catalog".into()))
        .unwrap();
    child.close().unwrap();
    parent.close().unwrap();
    exhaust_metadata_generation(&child_path);

    let mut child = RvfStore::open(&child_path).unwrap();
    // Vector 3 is inherited from the parent, so deleting it prunes the
    // membership filter as well as tombstoning the record.
    let failure = child.delete(&[3]).expect_err("delete must fail to commit");

    // A later commit that writes no metadata generation still publishes a
    // manifest; it must not record the rolled-back deletion.
    child
        .ingest_batch(&[&[4.0, 1.0]], &[4], None)
        .unwrap_or_else(|e| panic!("[{failure:?}] a later ingest must still commit: {e:?}"));
    child.close().unwrap();

    let reopened = RvfStore::open_readonly(&child_path).unwrap_or_else(|e| {
        panic!("[{failure:?}] a rolled-back delete must not brick the artifact: {e:?}")
    });
    for id in [1u64, 2, 3] {
        assert!(
            reopened.get_metadata(id).is_some(),
            "record {id} must survive a delete that never committed"
        );
    }
}

/// Criterion 6 (file-metadata path): `set_file_metadata` commits a metadata
/// generation carrying only file-level state, and must expose exactly the prior
/// or the new value after a crash at any boundary.
#[test]
fn file_metadata_commit_is_atomic_under_crash_injection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("meta_crash_file_metadata.rvf");
    let probe = dir.path().join("probe.rvf");
    let dim: u16 = 2;
    let all_ids = [10u64];
    let file_keys = ["app.schema", "app.owner"];

    let mut store = RvfStore::create(&path, make_options(dim)).unwrap();
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0]],
            &[10],
            &[VectorMetadata {
                vector_id: 10,
                fields: vec![MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("ten".into()),
                }],
                delete_record: false,
            }],
        )
        .unwrap();
    store
        .set_file_metadata("app.schema".into(), MetadataValue::U64(1))
        .unwrap();
    store
        .set_file_metadata("app.owner".into(), MetadataValue::String("catalog".into()))
        .unwrap();
    let bytes_a = fs::read(&path).unwrap();

    // Batch B is a single commit that replaces one file-level value.
    store
        .set_file_metadata("app.schema".into(), MetadataValue::U64(2))
        .unwrap();
    let bytes_b = fs::read(&path).unwrap();
    store.close().unwrap();

    assert_atomic_metadata_commit(
        "file-metadata",
        &all_ids,
        &file_keys,
        &bytes_a,
        &bytes_b,
        &probe,
    );
}

/// A mutation that fails after its metadata generation is on disk must restore
/// the segment directory it started from, not merely drop the entries it
/// appended.
///
/// `write_metadata_generation` both prunes and appends when it re-anchors a
/// recovered chain: it drops the generations replay could not reach and adds
/// the snapshot that supersedes them. The directory therefore ends up *shorter*
/// than it started, so a rollback that only trimmed back to the original length
/// would keep every one of those edits -- publishing the uncommitted generation
/// to the next commit, which then serves it as authoritative and loses whatever
/// the failed mutation had removed (issue #771).
///
/// The failure is injected by removing the parent file of a COW child: the
/// child's manifest write has to re-derive the relative parent reference, which
/// is the first thing after the metadata generation that can fail on demand.
#[test]
fn failed_mutation_restores_the_pruned_segment_directory() {
    for case in ["delete", "set_file_metadata"] {
        let dir = TempDir::new().unwrap();
        let parent_path = dir.path().join("rollback_parent.rvf");
        let child_path = dir.path().join("rollback_child.rvf");
        let dim: u16 = 2;

        let mut parent = RvfStore::create(&parent_path, make_options(dim)).unwrap();
        parent
            .ingest_batch_with_metadata(
                &[&[1.0, 0.0]],
                &[1],
                &[VectorMetadata {
                    vector_id: 1,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("parent".into()),
                    }],
                    delete_record: false,
                }],
            )
            .unwrap();

        // Four child generations: a full snapshot of the inherited state plus
        // three deltas, so damaging the third strands two of them.
        let mut child = parent.branch(&child_path).unwrap();
        for id in [11u64, 12, 13, 14] {
            child
                .ingest_batch_with_metadata(
                    &[&[id as f32, 1.0]],
                    &[id],
                    &[VectorMetadata {
                        vector_id: id,
                        fields: vec![MetadataEntry {
                            field_id: 1,
                            value: MetadataValue::String(format!("child-{id}")),
                        }],
                        delete_record: false,
                    }],
                )
                .unwrap();
        }
        child.close().unwrap();
        parent.close().unwrap();
        let parent_bytes = fs::read(&parent_path).unwrap();

        // Damage the third generation so the child opens on a truncated chain
        // with orphans for the next metadata write to prune.
        let mut bytes = fs::read(&child_path).unwrap();
        let mut meta_offsets = Vec::new();
        let mut offset = 0usize;
        while offset + SEGMENT_HEADER_SIZE <= bytes.len() {
            if u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) != SEGMENT_MAGIC {
                break;
            }
            let payload_len =
                u64::from_le_bytes(bytes[offset + 0x10..offset + 0x18].try_into().unwrap())
                    as usize;
            let end = match offset.checked_add(SEGMENT_HEADER_SIZE + payload_len) {
                Some(end) if end <= bytes.len() => end,
                _ => break,
            };
            if bytes[offset + 0x05] == rvf_types::SegmentType::Meta as u8 {
                meta_offsets.push((offset + SEGMENT_HEADER_SIZE, payload_len));
            }
            offset = end;
        }
        assert_eq!(meta_offsets.len(), 4, "[{case}]");
        let (start, len) = meta_offsets[2];
        bytes[start + len / 2] ^= 0xFF;
        fs::write(&child_path, &bytes).unwrap();

        let mut child = RvfStore::open(&child_path).unwrap();
        assert_eq!(
            child.metadata_recovery().dropped_generations,
            2,
            "[{case}] generations 3 and 4 must be the orphans the next write prunes"
        );

        // Removing the parent makes the child's manifest write fail, while the
        // metadata generation before it still commits to the file.
        fs::remove_file(&parent_path).unwrap();
        let failure = match case {
            "delete" => child.delete(&[11]).err(),
            _ => child
                .set_file_metadata("app.owner".into(), MetadataValue::String("catalog".into()))
                .err(),
        };
        let failure = failure.unwrap_or_else(|| panic!("[{case}] the mutation must fail"));
        fs::write(&parent_path, &parent_bytes).unwrap();

        // A later commit that writes no metadata generation publishes whatever
        // segment directory the rollback left behind.
        child
            .ingest_batch(&[&[99.0, 1.0]], &[99], None)
            .unwrap_or_else(|e| panic!("[{case}] a later ingest must still commit: {e:?}"));
        child.close().unwrap();

        let reopened = RvfStore::open_readonly(&child_path).unwrap_or_else(|e| {
            panic!("[{case}] a rolled-back mutation must not brick the artifact: {e:?}")
        });
        assert_eq!(
            reopened.get_metadata(11).unwrap(),
            vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("child-11".into()),
            }],
            "[{case}] [{failure:?}] the record the failed delete removed must survive"
        );
        assert_eq!(
            reopened.get_metadata(12).unwrap(),
            vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("child-12".into()),
            }],
            "[{case}] [{failure:?}] the recovered chain must still be the served state"
        );
        assert!(
            reopened.get_file_metadata("app.owner").is_none(),
            "[{case}] [{failure:?}] the failed set_file_metadata must not become visible"
        );
    }
}
