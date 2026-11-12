//! Round trip and the M3 kill-and-restore drill.

mod common;
use common::*;
use ruvector_edge_snapshot::*;

#[test]
fn snapshot_restore_round_trip_is_byte_identical() {
    let dim = 8;
    let data = rows(700, dim, 42);
    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let snap = write_snapshot(TENANT_A, &data, dim, 7, &chain, small_limits(dim), None);
    assert!(
        snap.chunks.len() >= 3,
        "expected several chunks, got {}",
        snap.chunks.len()
    );
    for (i, c) in snap.chunks.iter().enumerate() {
        assert_eq!(c.index as usize, i);
        assert!(c.bytes.len() <= small_limits(dim).max_chunk_bytes);
    }
    let entry = chain.append(&snap.sealed).unwrap();
    let chunk_bytes: Vec<Vec<u8>> = snap.chunks.iter().map(|c| c.bytes.clone()).collect();
    let back = restore(
        &snap.object,
        &chunk_bytes,
        TENANT_A,
        &[entry],
        chain.head(),
        SignaturePolicy::NotRequired,
    )
    .unwrap();
    assert_rows_identical(&data, &back);

    // Re-snapshotting the restored rows at the same epoch/clock reproduces
    // every chunk byte and the manifest root.
    let fresh = WitnessChain::new(TENANT_A).unwrap();
    let again = write_snapshot(TENANT_A, &back, dim, 7, &fresh, small_limits(dim), None);
    assert_eq!(again.sealed.root, snap.sealed.root);
    assert_eq!(again.object, snap.object);
    let again_bytes: Vec<Vec<u8>> = again.chunks.iter().map(|c| c.bytes.clone()).collect();
    assert_eq!(again_bytes, chunk_bytes);
}

#[test]
fn signed_bit_patterns_survive() {
    let dim = 4;
    let data = vec![
        Row {
            id: "a".into(),
            values: vec![-0.0, 0.0, f32::MIN_POSITIVE, f32::MAX],
            metadata: None,
        },
        Row {
            id: "b".into(),
            values: vec![1e-45, -1e-45, f32::MIN, 1.0],
            metadata: Some(String::new()),
        },
    ];
    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let snap = write_snapshot(
        TENANT_A,
        &data,
        dim,
        1,
        &chain,
        SnapshotLimits::default(),
        None,
    );
    let e = chain.append(&snap.sealed).unwrap();
    let bytes: Vec<Vec<u8>> = snap.chunks.iter().map(|c| c.bytes.clone()).collect();
    let back = restore(
        &snap.object,
        &bytes,
        TENANT_A,
        &[e],
        chain.head(),
        SignaturePolicy::NotRequired,
    )
    .unwrap();
    assert_rows_identical(&data, &back);
    assert_eq!(back[0].values[0].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn kill_and_restore_drill_returns_identical_results() {
    let dim = 16;
    let data = rows(1_000, dim, 7);
    let queries = rows(10, dim, 99);
    let before: Vec<_> = queries.iter().map(|q| topk(&data, &q.values, 10)).collect();

    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let signer = TestSigner::fixed(3);
    let snap = write_snapshot(
        TENANT_A,
        &data,
        dim,
        12,
        &chain,
        small_limits(dim),
        Some(&signer),
    );
    let entry = chain.append(&snap.sealed).unwrap();
    // "Kill": everything but the R2 objects and the ledger state is dropped.
    let objects: Vec<Vec<u8>> = snap.chunks.iter().map(|c| c.bytes.clone()).collect();
    let (manifest_obj, ledger_entries, ledger_head) =
        (snap.object.clone(), vec![entry], chain.head());
    drop(data);

    let mut v = Ed25519Verifier::new();
    v.add_key("snap-k1", signer.public()).unwrap();
    let restored = restore(
        &manifest_obj,
        &objects,
        TENANT_A,
        &ledger_entries,
        ledger_head,
        SignaturePolicy::Required(&v),
    )
    .unwrap();
    let after: Vec<_> = queries
        .iter()
        .map(|q| topk(&restored, &q.values, 10))
        .collect();
    assert_eq!(before, after);
}

#[test]
fn empty_shard_snapshots_and_restores() {
    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let snap = write_snapshot(TENANT_A, &[], 8, 1, &chain, SnapshotLimits::default(), None);
    assert!(snap.chunks.is_empty());
    let e = chain.append(&snap.sealed).unwrap();
    let back = restore(
        &snap.object,
        &[],
        TENANT_A,
        &[e],
        chain.head(),
        SignaturePolicy::NotRequired,
    )
    .unwrap();
    assert!(back.is_empty());
}

#[test]
fn keys_follow_adr_layout_and_are_known_before_finish() {
    let dim = 8;
    let spec = SnapshotSpec {
        shard: shard(TENANT_A),
        epoch: 42,
        dim,
        metric: Metric::Cosine,
        audit_head: [9; 32],
    };
    let mut w = SnapshotWriter::new(spec, small_limits(dim)).unwrap();
    // Each chunk is "PUT" to the key computed before it existed.
    let mut stored: Vec<(String, Vec<u8>)> = Vec::new();
    for r in rows(1_500, dim, 1) {
        let next = stored.len() as u32;
        let key = w.chunk_key(next);
        if let Some(c) = w.push(&r).unwrap() {
            assert_eq!(c.index, next);
            stored.push((key, c.bytes));
        }
    }
    assert!(stored.len() >= 2, "several chunks before finish");
    let tail_first = stored.len() as u32;
    let tail_keys: Vec<String> = (tail_first..tail_first + 2)
        .map(|n| w.chunk_key(n))
        .collect();
    let mkey = w.manifest_key();
    let (tail, sealed) = w.finish(&CLOCK, None).unwrap();
    for (c, key) in tail.into_iter().zip(tail_keys) {
        stored.push((key, c.bytes));
    }
    let m = &sealed.manifest;
    for (n, (key, _)) in stored.iter().enumerate() {
        assert_eq!(key, &snapshot_key(m, n as u32));
    }
    assert_eq!(mkey, manifest_key(m));
    let prefix = format!("snapshots/{TENANT_A}/vector/{UID}/0/00000000000000000042-");
    assert!(stored[0].0.starts_with(&prefix), "{}", stored[0].0);
    let id12 = &stored[0].0[prefix.len()..prefix.len() + 12];
    assert!(id12.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(stored[0].0, format!("{prefix}{id12}/seg-0.rvf"));
    // The key hashes the identity + epoch + audit head: another epoch or
    // audit head gives another prefix.
    let other = key_prefix(TENANT_A, "vector", UID, 0, 42, &[8; 32]);
    assert!(!stored[0].0.starts_with(&other));
    assert_eq!(
        key_prefix(TENANT_A, "vector", UID, 0, 42, &[9; 32]),
        m.key_prefix()
    );
}

#[test]
fn writer_refuses_bad_rows_and_limits() {
    let spec = |dim| SnapshotSpec {
        shard: shard(TENANT_A),
        epoch: 1,
        dim,
        metric: Metric::L2,
        audit_head: [0; 32],
    };
    let mut w = SnapshotWriter::new(spec(2), SnapshotLimits::default()).unwrap();
    let r = |id: &str, v: Vec<f32>| Row {
        id: id.into(),
        values: v,
        metadata: None,
    };
    assert!(matches!(
        w.push(&r("a", vec![1.0])),
        Err(SnapshotError::InvalidRow {
            reason: "dimension",
            ..
        })
    ));
    assert!(matches!(
        w.push(&r("a", vec![f32::NAN, 0.0])),
        Err(SnapshotError::InvalidRow { .. })
    ));
    assert!(matches!(
        w.push(&r("", vec![0.0, 0.0])),
        Err(SnapshotError::InvalidRow { .. })
    ));
    w.push(&r("b", vec![0.0, 0.0])).unwrap();
    assert!(matches!(
        w.push(&r("a", vec![0.0, 0.0])),
        Err(SnapshotError::InvalidRow {
            reason: "ids not strictly increasing",
            ..
        })
    ));
    let too_big = SnapshotLimits {
        max_chunk_bytes: MAX_CHUNK_BYTES + 1,
        ..SnapshotLimits::default()
    };
    assert!(matches!(
        SnapshotWriter::new(spec(2), too_big),
        Err(SnapshotError::QuotaExceeded { .. })
    ));
    let quota = SnapshotLimits {
        max_rows: 1,
        ..SnapshotLimits::default()
    };
    let mut w = SnapshotWriter::new(spec(2), quota).unwrap();
    w.push(&r("a", vec![0.0, 0.0])).unwrap();
    assert!(matches!(
        w.push(&r("b", vec![0.0, 0.0])),
        Err(SnapshotError::QuotaExceeded { .. })
    ));
    assert!(ShardRef::new("AAAA", "vector", UID, 0).is_err());
    assert!(ShardRef::new(TENANT_A, "../x", UID, 0).is_err());
    assert!(ShardRef::new(TENANT_A, "vector", "0123", 0).is_err());
}

#[test]
fn default_limits_keep_chunks_under_8_mib() {
    // 384-dim rows: ~2.7k rows per 4 MiB segment; 12k rows span several chunks.
    let dim = 384;
    let chain = WitnessChain::new(TENANT_A).unwrap();
    let snap = write_snapshot(
        TENANT_A,
        &rows(12_000, dim, 5),
        dim,
        1,
        &chain,
        SnapshotLimits::default(),
        None,
    );
    assert!(snap.chunks.len() >= 3);
    for c in &snap.chunks {
        assert!(c.bytes.len() <= MAX_CHUNK_BYTES);
    }
    assert_eq!(snap.sealed.manifest.row_count, 12_000);
}
