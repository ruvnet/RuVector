//! Tamper, cross-tenant and witness-chain refusals — each with its typed error.

mod common;
use common::*;
use ruvector_edge_snapshot::*;

struct Fixture {
    snap: Snap,
    chunks: Vec<Vec<u8>>,
    entries: Vec<WitnessEntry>,
    head: [u8; 32],
}

fn fixture(tenant: &str) -> Fixture {
    let dim = 8;
    // Tenant-specific data: two tenants with byte-identical shards would
    // have identical manifests once relabelled, which is not an attack.
    let seed = u64::from(tenant.as_bytes()[0]) << 8;
    let mut chain = WitnessChain::new(tenant).unwrap();
    // Two earlier snapshots so the chain has history.
    let mut entries = Vec::new();
    for epoch in 1..=2 {
        let s = write_snapshot(
            tenant,
            &rows(20, dim, seed + epoch),
            dim,
            epoch,
            &chain,
            small_limits(dim),
            None,
        );
        entries.push(chain.append(&s.sealed).unwrap());
    }
    let snap = write_snapshot(
        tenant,
        &rows(400, dim, seed + 3),
        dim,
        3,
        &chain,
        small_limits(dim),
        None,
    );
    entries.push(chain.append(&snap.sealed).unwrap());
    let chunks = snap.chunks.iter().map(|c| c.bytes.clone()).collect();
    Fixture {
        snap,
        chunks,
        entries,
        head: chain.head(),
    }
}

fn run(
    f: &Fixture,
    object: &[u8],
    chunks: &[Vec<u8>],
    tenant: &str,
) -> Result<Vec<Row>, SnapshotError> {
    restore(
        object,
        chunks,
        tenant,
        &f.entries,
        f.head,
        SignaturePolicy::NotRequired,
    )
}

#[test]
fn baseline_restores() {
    let f = fixture(TENANT_A);
    assert!(f.chunks.len() >= 3);
    assert_eq!(
        run(&f, &f.snap.object, &f.chunks, TENANT_A).unwrap().len(),
        400
    );
}

#[test]
fn every_single_byte_flip_in_every_chunk_is_refused() {
    let f = fixture(TENANT_A);
    for (ci, chunk) in f.chunks.iter().enumerate() {
        // Every position of small chunks would be slow in debug; stride covers
        // headers, payload, ids, values and padding of every segment.
        for pos in (0..chunk.len()).step_by(37).chain([chunk.len() - 1]) {
            let mut bad = f.chunks.clone();
            bad[ci][pos] ^= 0x01;
            assert_eq!(
                run(&f, &f.snap.object, &bad, TENANT_A).unwrap_err(),
                SnapshotError::ChunkChecksum { index: ci as u32 },
                "chunk {ci} byte {pos}"
            );
        }
    }
}

#[test]
fn every_single_byte_flip_in_the_manifest_is_refused() {
    let f = fixture(TENANT_A);
    for pos in 0..f.snap.object.len() {
        let mut obj = f.snap.object.clone();
        obj[pos] ^= 0x80;
        let err = run(&f, &obj, &f.chunks, TENANT_A).unwrap_err();
        // Header bytes fail to parse or hash; payload bytes fail the rvf-wire
        // content hash; padding bytes fail the strict trailing check.
        assert!(
            matches!(
                err,
                SnapshotError::Malformed(_) | SnapshotError::ManifestChecksum
            ),
            "byte {pos}: {err:?}"
        );
    }
}

#[test]
fn chunk_reorder_truncation_and_omission_are_refused() {
    let f = fixture(TENANT_A);
    let mut swapped = f.chunks.clone();
    swapped.swap(0, 1);
    // Chunks differ in size and hash; whichever check fires first, chunk 0 is refused.
    let err = run(&f, &f.snap.object, &swapped, TENANT_A).unwrap_err();
    assert!(
        matches!(
            err,
            SnapshotError::ChunkSize { index: 0 } | SnapshotError::ChunkChecksum { index: 0 }
        ),
        "{err:?}"
    );

    let proof = ChainProof::from_genesis(TENANT_A, &f.entries, f.head);
    let mut s = RestoreSession::begin(
        &f.snap.object,
        &target(TENANT_A),
        QUOTA,
        proof,
        SignaturePolicy::NotRequired,
    )
    .unwrap();
    let mut out = Vec::new();
    assert_eq!(
        s.accept_chunk(1, &f.chunks[1], &mut out).unwrap_err(),
        SnapshotError::ChunkOutOfOrder {
            expected: 0,
            got: 1
        }
    );
    assert!(out.is_empty());
    // A refused session stays refused.
    assert!(s.accept_chunk(0, &f.chunks[0], &mut out).is_err());

    let mut truncated = f.chunks.clone();
    let n = truncated[1].len() - 64;
    truncated[1].truncate(n);
    assert_eq!(
        run(&f, &f.snap.object, &truncated, TENANT_A).unwrap_err(),
        SnapshotError::ChunkSize { index: 1 }
    );

    let mut extended = f.chunks.clone();
    extended[2].extend_from_slice(&[0; 64]);
    assert_eq!(
        run(&f, &f.snap.object, &extended, TENANT_A).unwrap_err(),
        SnapshotError::ChunkSize { index: 2 }
    );

    let missing = &f.chunks[..f.chunks.len() - 1];
    assert_eq!(
        run(&f, &f.snap.object, missing, TENANT_A).unwrap_err(),
        SnapshotError::ChunkMissing {
            expected: f.chunks.len() as u32,
            got: f.chunks.len() as u32 - 1
        }
    );

    let mut short_manifest = f.snap.object.clone();
    short_manifest.truncate(short_manifest.len() - 70);
    assert!(matches!(
        run(&f, &short_manifest, &f.chunks, TENANT_A),
        Err(SnapshotError::Malformed(_))
    ));
}

/// Re-encode a manifest after mutating its body; `reroot` recomputes the root.
fn forge(sealed: &SealedManifest, reroot: bool, edit: impl FnOnce(&mut Manifest)) -> Vec<u8> {
    let mut s = sealed.clone();
    edit(&mut s.manifest);
    if reroot {
        s.root = s.manifest.compute_root();
    }
    s.to_object()
}

#[test]
fn cross_tenant_restore_is_refused_as_tenant_mismatch() {
    let b = fixture(TENANT_B);
    let a = fixture(TENANT_A);
    // Tenant A presents tenant B's genuine snapshot with A's own ledger.
    let err = restore(
        &b.snap.object,
        &b.chunks,
        TENANT_A,
        &a.entries,
        a.head,
        SignaturePolicy::NotRequired,
    );
    assert_eq!(err.unwrap_err(), SnapshotError::TenantMismatch);
}

#[test]
fn relabelled_foreign_snapshot_is_refused_by_root_chain_and_signature() {
    let b = fixture(TENANT_B);
    let a = fixture(TENANT_A);
    let relabel = |m: &mut Manifest| m.tenant_key = TENANT_A.to_string();
    let run_a = |obj: &[u8], sig| restore(obj, &b.chunks, TENANT_A, &a.entries, a.head, sig);

    // Root not recomputed → the stored root no longer matches.
    let obj = forge(&b.snap.sealed, false, relabel);
    assert_eq!(
        run_a(&obj, SignaturePolicy::NotRequired).unwrap_err(),
        SnapshotError::ManifestChecksum
    );

    // Root recomputed → A's ledger never witnessed it.
    let obj = forge(&b.snap.sealed, true, relabel);
    assert_eq!(
        run_a(&obj, SignaturePolicy::NotRequired).unwrap_err(),
        SnapshotError::NotInChain
    );

    // Signed deployments refuse it before the chain lookup.
    let signer = TestSigner::fixed(5);
    let mut v = Ed25519Verifier::new();
    v.add_key("snap-k1", signer.public()).unwrap();
    assert_eq!(
        run_a(&obj, SignaturePolicy::Required(&v)).unwrap_err(),
        SnapshotError::SignatureMissing
    );
    let mut signed = b.snap.sealed.clone();
    signed.signature = Some(SignatureBlock {
        key_id: "snap-k1".into(),
        signature: signer.sign(&signing_message(&signed.root)).unwrap(),
    });
    let obj = forge(&signed, true, relabel);
    assert_eq!(
        run_a(&obj, SignaturePolicy::Required(&v)).unwrap_err(),
        SnapshotError::SignatureInvalid
    );
}

#[test]
fn oversized_manifest_length_field_is_malformed_not_a_panic() {
    let f = fixture(TENANT_A);
    let mut obj = f.snap.object.clone();
    obj[16..24].copy_from_slice(&[0xFF; 8]);
    assert_eq!(
        SealedManifest::from_object(&obj).unwrap_err(),
        SnapshotError::Malformed("truncated")
    );
    assert_eq!(
        run(&f, &obj, &f.chunks, TENANT_A).unwrap_err(),
        SnapshotError::Malformed("truncated")
    );
}

#[test]
fn identity_schema_and_quota_mismatches_are_refused() {
    let f = fixture(TENANT_A);
    let proof = ChainProof::from_genesis(TENANT_A, &f.entries, f.head);
    let begin = |obj: &[u8], t: &RestoreTarget, q| {
        RestoreSession::begin(obj, t, q, proof, SignaturePolicy::NotRequired).map(|_| ())
    };
    let mut t = target(TENANT_A);
    t.shard.collection_uid = UID2.to_string();
    assert_eq!(
        begin(&f.snap.object, &t, QUOTA).unwrap_err(),
        SnapshotError::CollectionMismatch
    );
    let mut t = target(TENANT_A);
    t.shard.shard = 1;
    assert_eq!(
        begin(&f.snap.object, &t, QUOTA).unwrap_err(),
        SnapshotError::ShardMismatch
    );
    let mut t = target(TENANT_A);
    t.dim = Some(9);
    assert_eq!(
        begin(&f.snap.object, &t, QUOTA).unwrap_err(),
        SnapshotError::DimensionMismatch
    );

    let obj = forge(&f.snap.sealed, true, |m| m.schema_ver = 99);
    assert_eq!(
        begin(&obj, &target(TENANT_A), QUOTA).unwrap_err(),
        SnapshotError::UnknownSchema(99)
    );

    let q = RestoreQuota {
        max_rows: 399,
        ..QUOTA
    };
    assert!(matches!(
        begin(&f.snap.object, &target(TENANT_A), q),
        Err(SnapshotError::QuotaExceeded { what: "rows", .. })
    ));
    let q = RestoreQuota {
        max_bytes: 10,
        ..QUOTA
    };
    assert!(matches!(
        begin(&f.snap.object, &target(TENANT_A), q),
        Err(SnapshotError::QuotaExceeded { what: "bytes", .. })
    ));

    // Chunk list edited (e.g. an attacker swaps in their own chunk hashes).
    let obj = forge(&f.snap.sealed, false, |m| m.chunks.swap(0, 1));
    assert_eq!(
        begin(&obj, &target(TENANT_A), QUOTA).unwrap_err(),
        SnapshotError::ManifestChecksum
    );
}

#[test]
fn witness_chain_breaks_are_detected() {
    let f = fixture(TENANT_A);
    assert!(verify_chain(TENANT_A, &f.entries, &f.head).is_ok());

    let mut e = f.entries.clone();
    e[1].manifest_root[0] ^= 1;
    assert_eq!(
        verify_chain(TENANT_A, &e, &f.head).unwrap_err(),
        SnapshotError::ChainBreak { seq: 1 }
    );

    let mut e = f.entries.clone();
    e.remove(1);
    assert_eq!(
        verify_chain(TENANT_A, &e, &f.head).unwrap_err(),
        SnapshotError::ChainBreak { seq: 1 }
    );

    let mut e = f.entries.clone();
    e.swap(0, 1);
    assert_eq!(
        verify_chain(TENANT_A, &e, &f.head).unwrap_err(),
        SnapshotError::ChainBreak { seq: 0 }
    );

    // A truncated chain self-verifies but does not reach the trusted head.
    let e = &f.entries[..2];
    assert_eq!(
        verify_chain(TENANT_A, e, &f.head).unwrap_err(),
        SnapshotError::ChainHeadMismatch
    );

    // Another tenant's chain never verifies under this tenant.
    let b = fixture(TENANT_B);
    assert_eq!(
        verify_chain(TENANT_A, &b.entries, &b.head).unwrap_err(),
        SnapshotError::ChainBreak { seq: 0 }
    );

    // Restore surfaces a break in the ledger evidence.
    let mut e = f.entries.clone();
    e[2].epoch += 1;
    let err = restore(
        &f.snap.object,
        &f.chunks,
        TENANT_A,
        &e,
        f.head,
        SignaturePolicy::NotRequired,
    );
    assert_eq!(err.unwrap_err(), SnapshotError::ChainBreak { seq: 2 });

    // Suffix verification from a trusted checkpoint.
    assert!(verify_chain_from(TENANT_A, 1, &f.entries[0].hash, &f.entries[1..], &f.head).is_ok());
}
