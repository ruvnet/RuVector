//! Regression for #1275: the runtime must write witness segments that
//! `rvf_crypto::verify_witness_chain` accepts, as one chain that continues
//! across close/reopen, and that fails verification when altered.

use rvf_runtime::{RvfOptions, RvfStore};

fn witness_payloads(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 64 <= bytes.len() && bytes[pos..pos + 4] == 0x5256_4653u32.to_le_bytes() {
        let plen = u64::from_le_bytes(bytes[pos + 0x10..pos + 0x18].try_into().unwrap()) as usize;
        if bytes[pos + 5] == 0x0A {
            out.push(bytes[pos + 64..pos + 64 + plen].to_vec());
        }
        pos = (pos + 64 + plen + 63) & !63;
    }
    out
}

#[test]
fn runtime_witness_is_a_verifiable_chain_across_reopen() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("w.rvf");
    let o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    let mut s = RvfStore::create(&path, o).unwrap();
    s.ingest_batch(&[&[1.0f32, 0.0, 0.0, 0.0][..]], &[1], None)
        .unwrap();
    s.close().unwrap();
    for id in 2..=3u64 {
        let mut s = RvfStore::open(&path).unwrap(); // new session each time
        s.ingest_batch(&[&[id as f32, 0.0, 0.0, 0.0][..]], &[id], None)
            .unwrap();
        s.close().unwrap();
    }
    let payloads = witness_payloads(&std::fs::read(&path).unwrap());
    assert_eq!(payloads.len(), 3);
    assert!(
        payloads.iter().all(|p| p.len() == 73),
        "one 73-byte entry each"
    );
    let chain: Vec<u8> = payloads.concat();
    let entries = rvf_crypto::verify_witness_chain(&chain).expect("chain verifies");
    assert_eq!(entries.len(), 3);

    // Tamper with the first and middle entries: detected through the links.
    for at in [40usize, 73 + 40] {
        let mut bad = chain.clone();
        bad[at] ^= 1;
        assert!(rvf_crypto::verify_witness_chain(&bad).is_err(), "byte {at}");
    }
    // Dropping the middle entry breaks the link into the last one.
    let mut dropped = payloads[0].clone();
    dropped.extend_from_slice(&payloads[2]);
    assert!(rvf_crypto::verify_witness_chain(&dropped).is_err());
}

/// `query_audited` on an unsigned store appends a witness WITHOUT rewriting the
/// manifest, so it is not listed in the manifest's segment directory. The next
/// session must still continue the chain from the newest witness in FILE order.
#[test]
fn unlisted_audit_witness_does_not_break_the_chain_after_reopen() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("a.rvf");
    let mut o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    o.witness.audit_queries = true;
    let mut s = RvfStore::create(&path, o).unwrap();
    s.ingest_batch(&[&[1.0f32, 0.0, 0.0, 0.0][..]], &[1], None)
        .unwrap();
    s.query_audited(&[1.0, 0.0, 0.0, 0.0], 1, &Default::default())
        .unwrap();
    s.close().unwrap();
    let mut s = RvfStore::open(&path).unwrap();
    s.ingest_batch(&[&[2.0f32, 0.0, 0.0, 0.0][..]], &[2], None)
        .unwrap();
    s.close().unwrap();

    let payloads = witness_payloads(&std::fs::read(&path).unwrap());
    assert_eq!(payloads.len(), 3, "ingest, audited query, ingest");
    rvf_crypto::verify_witness_chain(&payloads.concat())
        .expect("file-order chain must verify after reopen");
}
