//! Bounded metadata history for ADR-280 (§1 "META_SEG is the source of truth",
//! §5 replay ceilings, and the "Alternatives Considered" rejection of writing a
//! complete metadata snapshot on every mutation).
//!
//! Only the newest committed `META_SEG` generation is authoritative. Everything
//! older is superseded history, so neither the bytes a commit costs nor the work
//! an open performs may scale with the number of metadata commits, compaction
//! must reclaim the superseded generations, and corruption confined to a
//! superseded generation must not brick the artifact.

use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, RvfOptions, VectorMetadata,
};
use rvf_runtime::RvfStore;
use rvf_types::{SegmentType, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};
use std::time::Instant;
use tempfile::TempDir;

fn make_options(dim: u16) -> RvfOptions {
    RvfOptions {
        dimension: dim,
        metric: DistanceMetric::L2,
        ..Default::default()
    }
}

fn record(vector_id: u64, value: &str) -> VectorMetadata {
    VectorMetadata {
        vector_id,
        fields: vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String(value.into()),
        }],
        delete_record: false,
    }
}

/// Walk the append-only file and return `(payload_start, payload_len)` for every
/// structurally complete `META_SEG`, in file order.
fn meta_segments(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut offset = 0usize;
    while offset + SEGMENT_HEADER_SIZE <= bytes.len() {
        let magic = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if magic != SEGMENT_MAGIC {
            break;
        }
        let seg_type = bytes[offset + 0x05];
        let payload_len =
            u64::from_le_bytes(bytes[offset + 0x10..offset + 0x18].try_into().unwrap()) as usize;
        let end = match offset.checked_add(SEGMENT_HEADER_SIZE + payload_len) {
            Some(end) if end <= bytes.len() => end,
            _ => break,
        };
        if seg_type == SegmentType::Meta as u8 {
            found.push((offset + SEGMENT_HEADER_SIZE, payload_len));
        }
        offset = end;
    }
    found
}

/// The runtime's legacy segment content hash: an IEEE CRC32 of the payload
/// rotated into four little-endian lanes (`rvf-runtime::hashing`). Rewriting a
/// segment payload in place means repairing this, or the reader rejects the
/// segment as damaged and the test would exercise recovery instead of the
/// intact-chain path it means to pin.
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

/// Replace one `META_SEG` payload in place, repairing the segment header's
/// content hash so the segment still reads as intact. The replacement must be
/// the same length, since the file is append-only and offsets are fixed.
fn replace_meta_payload(bytes: &mut [u8], payload_start: usize, payload: &[u8]) {
    let old_len = {
        let header = payload_start - SEGMENT_HEADER_SIZE;
        u64::from_le_bytes(bytes[header + 0x10..header + 0x18].try_into().unwrap()) as usize
    };
    assert_eq!(
        payload.len(),
        old_len,
        "replacement must be the same length"
    );
    bytes[payload_start..payload_start + payload.len()].copy_from_slice(payload);
    let header = payload_start - SEGMENT_HEADER_SIZE;
    bytes[header + 0x28..header + 0x38].copy_from_slice(&legacy_content_hash(payload));
}

/// Several hundred single-vector metadata commits must not cost several hundred
/// full snapshots: the committed metadata bytes stay proportional to the live
/// snapshot, the artifact reopens with bounded work, and `compact()` reclaims
/// the superseded generations.
#[test]
fn metadata_history_stays_bounded_and_compact_reclaims_it() {
    const COMMITS: u64 = 600;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata_growth.rvf");
    let dim: u16 = 4;

    let mut store = RvfStore::create(&path, make_options(dim)).unwrap();
    for i in 0..COMMITS {
        let vector = [i as f32, 1.0, 2.0, 3.0];
        store
            .ingest_batch_with_metadata(&[&vector], &[i], &[record(i, &format!("record-{i}"))])
            .unwrap();
    }
    let size_before_compact = std::fs::metadata(&path).unwrap().len();
    store.close().unwrap();

    // Writing a full snapshot per commit is quadratic: the live snapshot after
    // `COMMITS` commits is roughly 26 KiB, so replaying it every time costs
    // several MiB. Delta generations with periodic materialization keep the
    // committed metadata within a small multiple of one live snapshot.
    let bytes = std::fs::read(&path).unwrap();
    let metadata_bytes: usize = meta_segments(&bytes).iter().map(|&(_, len)| len).sum();
    assert!(
        metadata_bytes < 1024 * 1024,
        "metadata history is unbounded: {COMMITS} commits wrote {metadata_bytes} META_SEG bytes"
    );

    // Reopening must replay the newest generation and its delta chain, not
    // every generation ever committed.
    let started = Instant::now();
    let reopened = RvfStore::open_readonly(&path)
        .unwrap_or_else(|e| panic!("{COMMITS} metadata commits must stay openable: {e:?}"));
    let open_time = started.elapsed();
    assert!(
        open_time.as_secs() < 10,
        "open replayed superseded metadata history: took {open_time:?}"
    );

    for i in 0..COMMITS {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String(format!("record-{i}")),
            }],
            "record {i} must survive reopen"
        );
    }
    drop(reopened);

    // Compaction is the repair path: it drops every superseded generation and
    // rewrites one full snapshot.
    let mut store = RvfStore::open(&path).unwrap();
    store.compact().unwrap();
    store.close().unwrap();

    let size_after_compact = std::fs::metadata(&path).unwrap().len();
    assert!(
        size_after_compact < size_before_compact / 2,
        "compact() must reclaim superseded history: {size_before_compact} -> {size_after_compact}"
    );
    let compacted = std::fs::read(&path).unwrap();
    assert_eq!(
        meta_segments(&compacted).len(),
        1,
        "compaction must leave exactly one authoritative metadata generation"
    );

    let reopened = RvfStore::open_readonly(&path).unwrap();
    for i in 0..COMMITS {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String(format!("record-{i}")),
            }],
            "record {i} must survive compaction"
        );
    }
}

/// Build a store with `commits` single-record metadata commits and return its
/// raw bytes alongside the file path.
fn commit_metadata_history(path: &std::path::Path, commits: u64) {
    let mut store = RvfStore::create(path, make_options(2)).unwrap();
    for i in 0..commits {
        store
            .ingest_batch_with_metadata(
                &[&[i as f32, 1.0]],
                &[i],
                &[record(i, &format!("value-{i}"))],
            )
            .unwrap();
    }
    store.close().unwrap();
}

fn expected_record(i: u64) -> Vec<MetadataEntry> {
    vec![MetadataEntry {
        field_id: 1,
        value: MetadataValue::String(format!("value-{i}")),
    }]
}

/// A generation older than the newest full snapshot contributes nothing to the
/// committed state, so corrupting one must not make the artifact unopenable
/// (ADR-280 §1: only the newest generation is authoritative).
#[test]
fn corrupt_superseded_generation_does_not_brick_the_artifact() {
    // Long enough that the runtime has materialized at least one later full
    // snapshot, which is what makes the earliest generations superseded.
    const COMMITS: u64 = 96;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("superseded_corruption.rvf");
    commit_metadata_history(&path, COMMITS);

    let mut bytes = std::fs::read(&path).unwrap();
    let segments = meta_segments(&bytes);
    assert_eq!(segments.len() as u64, COMMITS);

    // Flip a byte in the oldest generation's payload.
    let (start, len) = segments[0];
    assert!(len > 0);
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap_or_else(|e| {
        panic!("corruption in a superseded metadata generation must be tolerated: {e:?}")
    });
    for i in 0..COMMITS {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "the authoritative snapshot must still resolve record {i}"
        );
    }
}

/// A torn newest generation -- a crash between appending the payload and
/// publishing the manifest -- must fall back to the previous complete snapshot
/// rather than failing the open (ADR-280 §4: never a mixed or torn snapshot).
#[test]
fn torn_newest_generation_falls_back_to_the_previous_snapshot() {
    const COMMITS: u64 = 3;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("torn_newest.rvf");
    commit_metadata_history(&path, COMMITS);

    let mut bytes = std::fs::read(&path).unwrap();
    let segments = meta_segments(&bytes);
    assert_eq!(segments.len() as u64, COMMITS);

    let (start, len) = segments[segments.len() - 1];
    assert!(len > 0);
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let reopened = RvfStore::open_readonly(&path)
        .unwrap_or_else(|e| panic!("a torn newest generation must fall back: {e:?}"));
    for i in 0..COMMITS - 1 {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "the previous complete snapshot must resolve record {i}"
        );
    }
    assert!(
        reopened.get_metadata(COMMITS - 1).is_none(),
        "the torn generation must not be partially applied"
    );
}

/// One delta generation may both delete a field from one record and set that
/// same field on another. The deletion names the field without constraining its
/// type, so it must not collide with the real type declared by the other record.
#[test]
fn delta_mixing_field_deletion_and_reuse_round_trips() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("delta_field_reuse.rvf");

    let mut store = RvfStore::create(&path, make_options(2)).unwrap();
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[1, 2],
            &[
                VectorMetadata {
                    vector_id: 1,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(10),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 2,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(20),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();

    // Vector 1 (the lower identifier, so encoded first) drops field 1 while
    // vector 3 introduces it with a concrete type in the same generation.
    store
        .ingest_batch_with_metadata(
            &[&[0.5, 0.5]],
            &[3],
            &[
                VectorMetadata {
                    vector_id: 1,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::DeleteField,
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 3,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(30),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();
    store.close().unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(
        reopened.get_metadata(1).unwrap(),
        vec![],
        "the deleted field must not survive reopen"
    );
    assert_eq!(
        reopened.get_metadata(2).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::U64(20),
        }],
        "an untouched record must be inherited from the base generation"
    );
    assert_eq!(
        reopened.get_metadata(3).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::U64(30),
        }]
    );
}

/// Damage to one delta must not make the whole artifact unopenable. Replay
/// serves the longest complete prefix of the chain -- the newest snapshot plus
/// the consecutive valid deltas after it -- and reports what it dropped.
#[test]
fn corrupt_mid_chain_delta_recovers_the_longest_valid_prefix() {
    const COMMITS: u64 = 6;
    // Generation 1 is the full snapshot; 2..=6 are deltas. Corrupting
    // generation 4 must strip generations 4, 5 and 6.
    const CORRUPTED: usize = 3;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mid_chain_corruption.rvf");
    commit_metadata_history(&path, COMMITS);

    let mut bytes = std::fs::read(&path).unwrap();
    let segments = meta_segments(&bytes);
    assert_eq!(segments.len() as u64, COMMITS);
    let (start, len) = segments[CORRUPTED];
    assert!(len > 0);
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap_or_else(|e| {
        panic!("one damaged delta must not make the artifact unopenable: {e:?}")
    });

    let recovery = reopened.metadata_recovery();
    assert_eq!(
        recovery.generation, CORRUPTED as u64,
        "the served state must be the last generation before the damage"
    );
    assert!(
        recovery.dropped_generations > 0,
        "recovery must report that generations were dropped"
    );

    // Records committed before the damage survive; those committed by the
    // dropped generations are absent, but nothing is partially applied.
    for i in 0..CORRUPTED as u64 {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "record {i} predates the damage and must survive"
        );
    }
    for i in CORRUPTED as u64..COMMITS {
        assert!(
            reopened.get_metadata(i).is_none(),
            "record {i} came from a dropped generation"
        );
    }
    drop(reopened);

    // Recovery must converge: writes after it have to reach the committed
    // chain, or the artifact silently discards every later mutation.
    let mut store = RvfStore::open(&path).unwrap();
    store
        .ingest_batch_with_metadata(&[&[99.0, 1.0]], &[99], &[record(99, "after-recovery")])
        .unwrap();
    store.close().unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(
        reopened.get_metadata(99).unwrap(),
        vec![record_entry("after-recovery")],
        "a write after recovery must survive reopen"
    );
    for i in 0..CORRUPTED as u64 {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "the recovered state must survive alongside the new write"
        );
    }
    assert_eq!(
        reopened.metadata_recovery().dropped_generations,
        0,
        "the repaired chain must no longer report damage"
    );
    drop(reopened);

    // A second write must land too, so convergence is not a one-shot effect.
    let mut store = RvfStore::open(&path).unwrap();
    store
        .ingest_batch_with_metadata(&[&[98.0, 1.0]], &[98], &[record(98, "second-write")])
        .unwrap();
    store.close().unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(
        reopened.get_metadata(98).unwrap(),
        vec![record_entry("second-write")]
    );
    assert_eq!(
        reopened.get_metadata(99).unwrap(),
        vec![record_entry("after-recovery")]
    );
}

/// Recovery must also survive compaction: compact rewrites the file around the
/// truncated state, and writes after that must still land.
#[test]
fn recovery_then_compact_then_write_stays_coherent() {
    const COMMITS: u64 = 6;
    const CORRUPTED: usize = 3;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("recovery_compact.rvf");
    commit_metadata_history(&path, COMMITS);

    let mut bytes = std::fs::read(&path).unwrap();
    let (start, len) = meta_segments(&bytes)[CORRUPTED];
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let mut store = RvfStore::open(&path).unwrap();
    assert!(store.metadata_recovery().dropped_generations > 0);
    store.compact().unwrap();
    store
        .ingest_batch_with_metadata(&[&[42.0, 1.0]], &[42], &[record(42, "post-compact")])
        .unwrap();
    store.close().unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(
        reopened.get_metadata(42).unwrap(),
        vec![record_entry("post-compact")],
        "a write after recovery and compaction must survive reopen"
    );
    for i in 0..CORRUPTED as u64 {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "compaction must preserve the recovered state"
        );
    }
    assert_eq!(reopened.metadata_recovery().dropped_generations, 0);
}

/// A dropped generation may have carried a deletion, so replaying only the
/// prefix resurrects records whose vectors the newest manifest still lists as
/// deleted. That contradiction is a consequence of the damage, not evidence of
/// an inconsistent artifact: the resurrected records are dropped and counted,
/// not treated as a fatal error.
#[test]
fn truncated_recovery_drops_resurrected_records_instead_of_failing() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("recovery_with_delete.rvf");

    let mut store = RvfStore::create(&path, make_options(2)).unwrap();
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 1.0], &[2.0, 1.0], &[3.0, 1.0], &[4.0, 1.0]],
            &[1, 2, 3, 4],
            &[
                record(1, "value-1"),
                record(2, "value-2"),
                record(3, "value-3"),
                record(4, "value-4"),
            ],
        )
        .unwrap();
    // Generation 2 removes vector 3 and its record together.
    store.delete(&[3]).unwrap();
    store.close().unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    let segments = meta_segments(&bytes);
    assert_eq!(segments.len(), 2);
    let (start, len) = segments[1];
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap_or_else(|e| {
        panic!("a dropped generation carrying a deletion must not brick the artifact: {e:?}")
    });

    let recovery = reopened.metadata_recovery();
    assert!(recovery.dropped_generations > 0);
    assert_eq!(
        recovery.dropped_records, 1,
        "the record resurrected by the truncated replay must be reported"
    );

    // The surviving records are readable and the deleted vector stays deleted.
    for i in [1u64, 2, 4] {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "record {i} must remain readable"
        );
    }
    assert!(
        reopened.get_metadata(3).is_none(),
        "a record whose vector is deleted must not be resurrected"
    );
    assert_eq!(
        reopened
            .query(&[1.0, 1.0], 4, &Default::default())
            .unwrap()
            .len(),
        3,
        "the vectors themselves must stay readable"
    );
}

/// A record held in memory for an identifier the store does not yet hold is
/// excluded from the committed chain. Once that identifier becomes part of the
/// committed vector state, the next generation must persist the record: a delta
/// is encoded against what the chain contains, not against memory.
#[test]
fn record_becoming_committable_is_persisted_by_the_next_generation() {
    let dir = TempDir::new().unwrap();
    let parent_path = dir.path().join("late_commit_parent.rvf");
    let child_path = dir.path().join("late_commit_child.rvf");

    let mut parent = RvfStore::create(&parent_path, make_options(2)).unwrap();
    parent
        .ingest_batch_with_metadata(&[&[1.0, 0.0]], &[1], &[record(1, "parent-one")])
        .unwrap();

    // The child inherits the parent's records in memory but holds no vectors,
    // so record 1 is not part of its committed state yet.
    let mut child = parent
        .derive(&child_path, rvf_types::DerivationType::Clone, None)
        .unwrap();
    child
        .ingest_batch_with_metadata(&[&[0.0, 1.0]], &[9], &[record(9, "child-nine")])
        .unwrap();

    // Vector 1 now exists in the child, making the inherited record committable.
    child.ingest_batch(&[&[1.0, 0.0]], &[1], None).unwrap();

    // The next metadata generation is a delta, and must carry record 1.
    child
        .ingest_batch_with_metadata(&[&[0.5, 0.5]], &[8], &[record(8, "child-eight")])
        .unwrap();
    child.close().unwrap();
    parent.close().unwrap();

    let reopened = RvfStore::open_readonly(&child_path).unwrap();
    assert_eq!(
        reopened.get_metadata(1).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("parent-one".into()),
        }],
        "a record that became committable must reach the committed chain"
    );
    assert_eq!(
        reopened.get_metadata(9).unwrap(),
        vec![record_entry("child-nine")]
    );
    assert_eq!(
        reopened.get_metadata(8).unwrap(),
        vec![record_entry("child-eight")]
    );
}

fn record_entry(value: &str) -> MetadataEntry {
    MetadataEntry {
        field_id: 1,
        value: MetadataValue::String(value.into()),
    }
}

/// A `derive()` child starts with no vectors, so it must not persist metadata
/// records that reference identifiers only the parent holds (ADR-280 §4: records
/// are validated against the resulting committed vector snapshot).
#[test]
fn derived_child_metadata_write_stays_openable() {
    let dir = TempDir::new().unwrap();
    let parent_path = dir.path().join("derive_parent.rvf");
    let child_path = dir.path().join("derive_child.rvf");
    let dim: u16 = 2;

    let mut parent = RvfStore::create(&parent_path, make_options(dim)).unwrap();
    parent
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[1, 2],
            &[record(1, "parent-one"), record(2, "parent-two")],
        )
        .unwrap();

    let mut child = parent
        .derive(&child_path, rvf_types::DerivationType::Clone, None)
        .unwrap();

    // Any metadata-bearing write on the child commits a new META generation.
    child
        .ingest_batch_with_metadata(&[&[0.5, 0.5]], &[7], &[record(7, "child-seven")])
        .unwrap();
    child.close().unwrap();
    parent.close().unwrap();

    let reopened = RvfStore::open_readonly(&child_path)
        .unwrap_or_else(|e| panic!("derived child with metadata must reopen: {e:?}"));
    assert_eq!(
        reopened.get_metadata(7).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("child-seven".into()),
        }]
    );
    assert!(
        reopened.get_metadata(1).is_none(),
        "a derived child holds no parent vectors, so it carries no parent record"
    );
}

/// Build a store with vectors `1..=count` carrying metadata, then delete
/// `to_delete` in a second generation. Returns the file path.
fn store_with_metadata_then_delete(path: &std::path::Path, count: u64, to_delete: &[u64]) {
    let mut store = RvfStore::create(path, make_options(2)).unwrap();
    let vectors: Vec<[f32; 2]> = (1..=count).map(|i| [i as f32, 1.0]).collect();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let ids: Vec<u64> = (1..=count).collect();
    let records: Vec<VectorMetadata> = ids
        .iter()
        .map(|&i| record(i, &format!("value-{i}")))
        .collect();
    store
        .ingest_batch_with_metadata(&refs, &ids, &records)
        .unwrap();
    store.delete(to_delete).unwrap();
    store.close().unwrap();
}

/// CONTROL for the truncated-recovery tolerance: on an *intact* chain, metadata
/// that references a deleted vector is a genuine inconsistency and must still be
/// a hard error. Without this, relaxing the check unconditionally -- which makes
/// every other test in this file pass -- would silently convert a corruption
/// signal into data loss.
///
/// The inconsistency is built by retargeting generation 2's record tombstone
/// from vector 3 (which the same generation deleted) to vector 4, leaving an
/// intact, fully replayable chain whose metadata disagrees with the manifest's
/// deletion set.
#[test]
fn intact_chain_with_deleted_vector_metadata_is_still_fatal() {
    use rvf_types::metadata::{MetadataRecordOp, MetadataSegment};

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("intact_inconsistency.rvf");
    store_with_metadata_then_delete(&path, 4, &[3]);

    let mut bytes = std::fs::read(&path).unwrap();
    let segments = meta_segments(&bytes);
    assert_eq!(segments.len(), 2, "expected one snapshot and one delta");

    let (start, len) = segments[1];
    let mut delta = MetadataSegment::decode(&bytes[start..start + len]).unwrap();
    let tombstone = delta
        .records
        .iter_mut()
        .find(|r| matches!(r.operation, MetadataRecordOp::DeleteRecord))
        .expect("generation 2 must tombstone the deleted record");
    assert_eq!(tombstone.vector_id, 3);
    tombstone.vector_id = 4;
    let patched = delta.encode().unwrap();
    replace_meta_payload(&mut bytes, start, &patched);
    std::fs::write(&path, &bytes).unwrap();

    // The chain is intact, so the surviving record for the deleted vector 3 is
    // a real inconsistency rather than a consequence of damage.
    let error = RvfStore::open_readonly(&path)
        .err()
        .expect("metadata for a deleted vector on an intact chain must be fatal");
    assert!(
        format!("{error:?}").contains("InvalidMetadata"),
        "expected InvalidMetadata, got {error:?}"
    );
}

/// A mutation that publishes a manifest without emitting a metadata generation
/// must not destroy an artifact that opened through truncated recovery. The
/// recovery's bookkeeping only becomes durable together with the re-anchoring
/// snapshot, so a manifest-only write leaves the on-disk chain exactly as it
/// was and the next open recovers identically.
#[test]
fn manifest_only_write_after_truncated_recovery_preserves_the_artifact() {
    for (label, manifest_only) in [("ingest-without-metadata", 0u8), ("freeze", 1u8)] {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("manifest_only.rvf");
        store_with_metadata_then_delete(&path, 8, &[2, 4, 6, 8]);

        let mut bytes = std::fs::read(&path).unwrap();
        let (start, len) = meta_segments(&bytes)[1];
        bytes[start + len / 2] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        // Recovery drops the deletion-carrying generation, resurrecting the
        // records whose vectors the manifest still lists as deleted.
        let mut store = RvfStore::open(&path)
            .unwrap_or_else(|e| panic!("[{label}] recovery must open the artifact: {e:?}"));
        let recovery = store.metadata_recovery();
        assert!(recovery.dropped_generations > 0, "[{label}]");
        assert_eq!(recovery.dropped_records, 4, "[{label}]");

        // A mutation that writes a manifest but no metadata generation.
        if manifest_only == 0 {
            store.ingest_batch(&[&[99.0, 1.0]], &[99], None).unwrap();
            store.close().unwrap();
        } else {
            store.freeze().unwrap();
            drop(store);
        }

        let reopened = RvfStore::open_readonly(&path).unwrap_or_else(|e| {
            panic!("[{label}] a manifest-only write must not destroy the artifact: {e:?}")
        });
        for i in [1u64, 3, 5, 7] {
            assert_eq!(
                reopened.get_metadata(i).unwrap(),
                expected_record(i),
                "[{label}] the recovered state must still be served for record {i}"
            );
        }
        for i in [2u64, 4, 6, 8] {
            assert!(
                reopened.get_metadata(i).is_none(),
                "[{label}] record {i} belongs to a deleted vector"
            );
        }
    }
}
