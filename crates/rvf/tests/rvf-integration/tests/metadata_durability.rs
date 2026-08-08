//! Durable metadata integration tests for ADR-280.
//!
//! These exercise the self-contained-artifact promise through the public
//! `RvfStore` API at the integration boundary: metadata ingested in one
//! process image must survive close/reopen, filtered queries must be stable
//! across a reopen, `get_metadata` must return the committed values, and COW
//! children must inherit, override, and tombstone parent metadata without
//! mutating the parent.
//!
//! Maps to ADR-280 acceptance criteria 1, 4, 5, and 9.

use rvf_runtime::filter::{FilterExpr, FilterValue};
use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, VectorMetadata,
};
use rvf_runtime::RvfStore;
use tempfile::TempDir;

fn make_options(dim: u16) -> RvfOptions {
    RvfOptions {
        dimension: dim,
        metric: DistanceMetric::L2,
        ..Default::default()
    }
}

/// Criterion 1: metadata with unequal field counts per vector survives
/// close/reopen. Criterion 5: `get_metadata` returns the committed values.
#[test]
fn unequal_field_counts_survive_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("unequal_fields.rvf");
    let dim: u16 = 2;

    {
        let mut store = RvfStore::create(&path, make_options(dim)).unwrap();
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
                    // Zero fields: an existing but empty metadata record.
                    VectorMetadata {
                        vector_id: 30,
                        fields: vec![],
                        delete_record: false,
                    },
                ],
            )
            .unwrap();
        store.close().unwrap();
    }

    let store = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(
        store.get_metadata(10).unwrap(),
        vec![
            MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("ten".into()),
            },
            MetadataEntry {
                field_id: 2,
                value: MetadataValue::U64(10),
            },
        ],
        "two-field record must round-trip exactly"
    );
    assert_eq!(
        store.get_metadata(20).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("twenty".into()),
        }],
        "single-field record must round-trip exactly"
    );
    assert_eq!(
        store.get_metadata(30).unwrap(),
        vec![],
        "empty record must survive as an empty (present) record"
    );
}

/// Criterion 4: filter results are identical before and after reopen.
#[test]
fn filter_results_identical_before_and_after_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("filter_reopen.rvf");
    let dim: u16 = 4;

    let vectors: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32; dim as usize]).collect();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let ids: Vec<u64> = (1..=20).collect();
    let metadata: Vec<MetadataEntry> = ids
        .iter()
        .map(|&id| MetadataEntry {
            field_id: 0,
            value: MetadataValue::U64(id % 3),
        })
        .collect();

    let filter = FilterExpr::Eq(0, FilterValue::U64(1));
    let qopts = QueryOptions {
        filter: Some(filter),
        ..Default::default()
    };
    let query = vec![0.0f32; dim as usize];

    let before: Vec<u64> = {
        let mut store = RvfStore::create(&path, make_options(dim)).unwrap();
        store.ingest_batch(&refs, &ids, Some(&metadata)).unwrap();
        let mut hits: Vec<u64> = store
            .query(&query, 20, &qopts)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        hits.sort_unstable();
        store.close().unwrap();
        hits
    };

    // Reopen and run the identical filtered query.
    let store = RvfStore::open_readonly(&path).unwrap();
    let mut after: Vec<u64> = store
        .query(&query, 20, &qopts)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    after.sort_unstable();

    assert!(!before.is_empty(), "filter should match some records");
    assert_eq!(
        before, after,
        "filtered result set must be identical across reopen"
    );
    // Every survivor must genuinely satisfy the filter (category == 1).
    assert!(after.iter().all(|&id| id % 3 == 1));
}

/// Criterion 9: a COW child inherits parent metadata, overrides a field, and
/// tombstones a record, without changing the parent -- verified in memory and
/// after both files are reopened.
#[test]
fn cow_child_overlays_metadata_without_touching_parent() {
    let dir = TempDir::new().unwrap();
    let parent_path = dir.path().join("parent.rvf");
    let child_path = dir.path().join("child.rvf");
    let dim: u16 = 2;

    let mut parent = RvfStore::create(&parent_path, make_options(dim)).unwrap();
    parent
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[1, 2],
            &[
                VectorMetadata {
                    vector_id: 1,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("parent-one".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 2,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("parent-two".into()),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();

    let mut child = parent.branch(&child_path).unwrap();
    // Child sees inherited parent metadata.
    assert_eq!(
        child.get_metadata(1).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("parent-one".into()),
        }]
    );

    // Override field on vector 1, tombstone the record for vector 2.
    child
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0]],
            &[1],
            &[VectorMetadata {
                vector_id: 1,
                fields: vec![MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("child-one".into()),
                }],
                delete_record: false,
            }],
        )
        .unwrap();
    child
        .ingest_batch_with_metadata(
            &[&[0.0, 1.0]],
            &[2],
            &[VectorMetadata {
                vector_id: 2,
                fields: vec![],
                delete_record: true,
            }],
        )
        .unwrap();

    // Child view reflects the overlay.
    assert_eq!(
        child.get_metadata(1).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("child-one".into()),
        }],
        "child override must win over inherited value"
    );
    assert!(
        child.get_metadata(2).is_none(),
        "child record tombstone must hide the inherited record"
    );

    // Parent (still open) is untouched by the child's overlay.
    assert_eq!(
        parent.get_metadata(1).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("parent-one".into()),
        }],
        "parent metadata must not change when a child overrides it"
    );
    assert_eq!(
        parent.get_metadata(2).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("parent-two".into()),
        }],
        "parent record must not be tombstoned by a child tombstone"
    );

    child.close().unwrap();
    parent.close().unwrap();

    // Reopen both: the overlay and the untouched parent both persist.
    let reopened_child = RvfStore::open_readonly(&child_path).unwrap();
    assert_eq!(
        reopened_child.get_metadata(1).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("child-one".into()),
        }]
    );
    assert!(reopened_child.get_metadata(2).is_none());

    let reopened_parent = RvfStore::open_readonly(&parent_path).unwrap();
    assert_eq!(
        reopened_parent.get_metadata(1).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("parent-one".into()),
        }]
    );
    assert_eq!(
        reopened_parent.get_metadata(2).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("parent-two".into()),
        }]
    );
}

/// Criterion 5: metadata-bearing search hits return the committed values, and
/// the returned set is identical before and after a reopen. Without the
/// opt-in flag, hits carry no metadata.
#[test]
fn search_hits_include_metadata_before_and_after_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("search_hits_metadata.rvf");
    let dim: u16 = 2;

    {
        let mut store = RvfStore::create(&path, make_options(dim)).unwrap();
        store
            .ingest_batch_with_metadata(
                &[&[1.0, 0.0], &[0.0, 1.0]],
                &[10, 20],
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
                                value: MetadataValue::I64(-10),
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
                ],
            )
            .unwrap();
        store.close().unwrap();
    }

    let query = [1.0f32, 0.0];
    let with_meta = QueryOptions {
        include_metadata: true,
        ..Default::default()
    };

    let assert_hits = |store: &RvfStore| {
        // Default query: identifiers and distance only.
        let plain = store.query(&query, 2, &QueryOptions::default()).unwrap();
        assert!(plain.iter().all(|hit| hit.metadata.is_none()));

        let hits = store.query(&query, 2, &with_meta).unwrap();
        assert_eq!(hits.len(), 2);
        let ten = hits.iter().find(|h| h.id == 10).unwrap();
        assert_eq!(
            ten.metadata,
            Some(vec![
                MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("ten".into()),
                },
                MetadataEntry {
                    field_id: 2,
                    value: MetadataValue::I64(-10),
                },
            ]),
            "hit must carry the full committed record"
        );
        let twenty = hits.iter().find(|h| h.id == 20).unwrap();
        assert_eq!(
            twenty.metadata,
            Some(vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("twenty".into()),
            }])
        );
    };

    let store = RvfStore::open_readonly(&path).unwrap();
    assert_hits(&store);
    drop(store);
    let reopened = RvfStore::open_readonly(&path).unwrap();
    assert_hits(&reopened);
}

/// Criterion 5 + 9: search hits on a COW child reflect the child's field
/// overrides and record tombstones, not the inherited parent values, both in
/// memory and after the child file is reopened.
#[test]
fn cow_child_search_hits_reflect_overrides_and_tombstones() {
    let dir = TempDir::new().unwrap();
    let parent_path = dir.path().join("parent.rvf");
    let child_path = dir.path().join("child.rvf");
    let dim: u16 = 2;

    let mut parent = RvfStore::create(&parent_path, make_options(dim)).unwrap();
    parent
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[1, 2],
            &[
                VectorMetadata {
                    vector_id: 1,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("parent-one".into()),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 2,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::String("parent-two".into()),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();

    let mut child = parent.branch(&child_path).unwrap();
    // Override vector 1's field; tombstone vector 2's metadata record (the
    // vector itself remains a valid hit, just with no metadata).
    child
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0]],
            &[1],
            &[VectorMetadata {
                vector_id: 1,
                fields: vec![MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("child-one".into()),
                }],
                delete_record: false,
            }],
        )
        .unwrap();
    child
        .ingest_batch_with_metadata(
            &[&[0.0, 1.0]],
            &[2],
            &[VectorMetadata {
                vector_id: 2,
                fields: vec![],
                delete_record: true,
            }],
        )
        .unwrap();

    let with_meta = QueryOptions {
        include_metadata: true,
        ..Default::default()
    };

    let assert_child_hits = |store: &RvfStore| {
        let hits = store.query(&[1.0f32, 0.0], 2, &with_meta).unwrap();
        let one = hits.iter().find(|h| h.id == 1).unwrap();
        assert_eq!(
            one.metadata,
            Some(vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("child-one".into()),
            }]),
            "hit must carry the child override, not the inherited parent value"
        );
        let two = hits.iter().find(|h| h.id == 2).unwrap();
        assert_eq!(
            two.metadata, None,
            "a tombstoned record must surface as no metadata on the hit"
        );
    };
    assert_child_hits(&child);

    // Parent search hits are untouched by the child's overlay.
    let parent_hits = parent.query(&[1.0f32, 0.0], 2, &with_meta).unwrap();
    let parent_one = parent_hits.iter().find(|h| h.id == 1).unwrap();
    assert_eq!(
        parent_one.metadata,
        Some(vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("parent-one".into()),
        }])
    );

    child.close().unwrap();
    parent.close().unwrap();

    let reopened_child = RvfStore::open_readonly(&child_path).unwrap();
    assert_child_hits(&reopened_child);
}
