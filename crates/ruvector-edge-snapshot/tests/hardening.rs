//! Regression tests for the review findings on the snapshot side:
//! concurrent witness appends, tenant-scoped signing keys, the per-chunk
//! row cap (restore footprint) and bounded segment lengths in chunks.

mod common;
use common::*;
use ruvector_edge_snapshot::types::sha256;
use ruvector_edge_snapshot::*;

fn chunk_bytes(s: &Snap) -> Vec<Vec<u8>> {
    s.chunks.iter().map(|c| c.bytes.clone()).collect()
}

#[test]
fn concurrent_snapshots_of_one_tenant_append_in_either_order() {
    let dim = 4;
    for order in [[0usize, 1], [1, 0]] {
        let mut chain = WitnessChain::new(TENANT_A).unwrap();
        chain
            .append(
                &write_snapshot(
                    TENANT_A,
                    &rows(3, dim, 9),
                    dim,
                    1,
                    &chain,
                    SnapshotLimits::default(),
                    None,
                )
                .sealed,
            )
            .unwrap();
        // Two shards' hourly snapshots start from the same head…
        let data = [rows(40, dim, 1), rows(50, dim, 2)];
        let snaps: Vec<Snap> = data
            .iter()
            .enumerate()
            .map(|(i, d)| {
                write_snapshot(
                    TENANT_A,
                    d,
                    dim,
                    10 + i as u64,
                    &chain,
                    SnapshotLimits::default(),
                    None,
                )
            })
            .collect();
        // …and both are witnessed in whatever order they finish.
        let mut entries = Vec::new();
        for &i in &order {
            entries.push(chain.append(&snaps[i].sealed).unwrap());
        }
        let proof_from = entries[0].seq;
        let prev = entries[0].prev;
        for (i, s) in snaps.iter().enumerate() {
            let proof = ChainProof {
                entries: &entries,
                start_seq: proof_from,
                start_prev: prev,
                trusted_head: chain.head(),
            };
            let mut r = RestoreSession::begin(
                &s.object,
                &target(TENANT_A),
                QUOTA,
                proof,
                SignaturePolicy::NotRequired,
            )
            .unwrap();
            let mut out = Vec::new();
            for (n, c) in chunk_bytes(s).iter().enumerate() {
                r.accept_chunk(n as u32, c, &mut out).unwrap();
            }
            r.finish().unwrap();
            assert_rows_identical(&data[i], &out);
        }
    }
}

#[test]
fn chain_append_refuses_foreign_or_unrooted_manifests() {
    let dim = 4;
    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let sb = write_snapshot(
        TENANT_B,
        &rows(5, dim, 3),
        dim,
        1,
        &chain,
        SnapshotLimits::default(),
        None,
    );
    assert_eq!(
        chain.append(&sb.sealed).unwrap_err(),
        SnapshotError::TenantMismatch
    );
    let mut bad = write_snapshot(
        TENANT_A,
        &rows(5, dim, 4),
        dim,
        3,
        &chain,
        SnapshotLimits::default(),
        None,
    )
    .sealed;
    bad.manifest.row_count += 1;
    assert_eq!(
        chain.append(&bad).unwrap_err(),
        SnapshotError::ManifestChecksum
    );
    assert_eq!(chain.next_seq(), 0, "refusals do not advance the chain");
}

#[test]
fn signing_keys_are_scoped_to_their_tenant() {
    let dim = 4;
    let signer = TestSigner::fixed(11);
    let mut chain = WitnessChain::new(TENANT_B).unwrap();
    let s = write_snapshot(
        TENANT_B,
        &rows(20, dim, 5),
        dim,
        1,
        &chain,
        SnapshotLimits::default(),
        Some(&signer),
    );
    let e = chain.append(&s.sealed).unwrap();
    let run = |v: &Ed25519Verifier| {
        restore(
            &s.object,
            &chunk_bytes(&s),
            TENANT_B,
            std::slice::from_ref(&e),
            chain.head(),
            SignaturePolicy::Required(v),
        )
    };
    // Tenant A's key (same key id, attacker-chosen) must not vouch for B.
    let mut a_only = Ed25519Verifier::new();
    a_only
        .add_tenant_key(TENANT_A, "snap-k1", signer.public())
        .unwrap();
    assert_eq!(run(&a_only).unwrap_err(), SnapshotError::SignatureInvalid);
    let mut b_key = Ed25519Verifier::new();
    b_key
        .add_tenant_key(TENANT_B, "snap-k1", signer.public())
        .unwrap();
    assert!(run(&b_key).is_ok());
    let mut service = Ed25519Verifier::new();
    service.add_key("snap-k1", signer.public()).unwrap();
    assert!(run(&service).is_ok());
    assert!(a_only
        .add_tenant_key("NOT-A-TENANT", "k", signer.public())
        .is_err());
}

#[test]
fn chunks_are_capped_by_rows_and_restore_refuses_bigger_ones() {
    let dim = 2;
    let limits = SnapshotLimits {
        max_rows_per_chunk: 10,
        ..SnapshotLimits::default()
    };
    let data = rows(95, dim, 3);
    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let s = write_snapshot(TENANT_A, &data, dim, 1, &chain, limits, None);
    assert_eq!(s.chunks.len(), 10);
    assert!(s.chunks.iter().all(|c| c.rows <= 10));
    let e1 = chain.append(&s.sealed).unwrap();
    let back = restore(
        &s.object,
        &chunk_bytes(&s),
        TENANT_A,
        std::slice::from_ref(&e1),
        chain.head(),
        SignaturePolicy::NotRequired,
    )
    .unwrap();
    assert_rows_identical(&data, &back);

    let too_many = SnapshotLimits {
        max_rows_per_chunk: MAX_ROWS_PER_CHUNK + 1,
        ..limits
    };
    assert!(matches!(
        SnapshotWriter::new(
            SnapshotSpec {
                shard: shard(TENANT_A),
                epoch: 1,
                dim,
                metric: Metric::L2,
                audit_head: [0; 32]
            },
            too_many
        ),
        Err(SnapshotError::QuotaExceeded { .. })
    ));

    // A witnessed manifest that declares an over-cap chunk is refused at
    // `begin`, before any chunk is fetched or decoded.
    let mut forged = s.sealed.clone();
    forged.manifest.chunks.truncate(1);
    forged.manifest.chunks[0].rows = MAX_ROWS_PER_CHUNK + 1;
    forged.manifest.row_count = MAX_ROWS_PER_CHUNK + 1;
    forged.root = forged.manifest.compute_root();
    let entries = [e1, chain.append(&forged).unwrap()];
    let proof = ChainProof::from_genesis(TENANT_A, &entries, chain.head());
    let err = RestoreSession::begin(
        &forged.to_object(),
        &target(TENANT_A),
        QUOTA,
        proof,
        SignaturePolicy::NotRequired,
    );
    assert!(matches!(
        err.unwrap_err(),
        SnapshotError::QuotaExceeded {
            what: "rows per chunk",
            ..
        }
    ));
}

#[test]
fn chunk_with_huge_segment_length_is_corrupt_not_a_panic() {
    let dim = 4;
    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let s = write_snapshot(
        TENANT_A,
        &rows(3, dim, 1),
        dim,
        1,
        &chain,
        SnapshotLimits::default(),
        None,
    );
    // A chunk whose (hash-verified!) header declares a payload of ~2^64 bytes.
    let mut bytes = s.chunks[0].bytes.clone();
    bytes[16..24].copy_from_slice(&(u64::MAX - 10).to_le_bytes());
    let mut forged = s.sealed.clone();
    forged.manifest.chunks[0].sha256 = sha256(&[&bytes]);
    forged.root = forged.manifest.compute_root();
    let e = chain.append(&forged).unwrap();
    let err = restore(
        &forged.to_object(),
        &[bytes],
        TENANT_A,
        &[e],
        chain.head(),
        SignaturePolicy::NotRequired,
    );
    assert_eq!(
        err.unwrap_err(),
        SnapshotError::SegmentCorrupt {
            index: 0,
            reason: "segment length"
        }
    );
}

#[test]
fn restore_locates_the_manifest_from_the_ledger_entry() {
    // The restoring Worker holds only the ledger entry and the caller's
    // ShardRef; the key it derives must be the one the writer used.
    let dim = 4;
    let mut chain = WitnessChain::new(TENANT_A).unwrap();
    let s = write_snapshot(
        TENANT_A,
        &rows(30, dim, 8),
        dim,
        77,
        &chain,
        SnapshotLimits::default(),
        None,
    );
    let e = chain.append(&s.sealed).unwrap();
    let caller = shard(TENANT_A);
    assert_eq!(e.audit_head, s.sealed.manifest.audit_head);
    assert_eq!(
        e.manifest_key(&caller.tenant_key, &caller.service),
        manifest_key(&s.sealed.manifest)
    );
    assert_eq!(
        e.key_prefix(&caller.tenant_key, &caller.service),
        s.sealed.manifest.key_prefix()
    );
    // The audit head is part of the entry hash: editing it breaks the chain.
    let mut forged = e.clone();
    forged.audit_head[0] ^= 1;
    assert_eq!(
        verify_chain(TENANT_A, &[forged], &chain.head()).unwrap_err(),
        SnapshotError::ChainBreak { seq: 0 }
    );
}
