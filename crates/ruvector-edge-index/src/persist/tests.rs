use super::*;
use crate::error::DecodeError;

fn writer(payload: &[u8]) -> impl Fn(&mut dyn Sink) + '_ {
    move |s: &mut dyn Sink| {
        s.put(&payload[..payload.len() / 3]);
        s.put(&payload[payload.len() / 3..]);
    }
}

fn enc(payload: &[u8], max_chunk: usize) -> EncodedIndex {
    encode_all(9, IndexKind::Hnsw, max_chunk, &writer(payload)).unwrap()
}

fn dec(
    chunks: &[IndexChunk],
    kind: IndexKind,
    sha: Option<&[u8; 32]>,
) -> Result<Vec<u8>, DecodeError> {
    decode(&mut chunks.iter().cloned(), kind, sha, u64::MAX, |r| {
        r.bytes_vec(0, 1)
    })
}

fn raw(chunks: &[IndexChunk], sha: Option<&[u8; 32]>, len: usize) -> Result<Vec<u8>, DecodeError> {
    decode(
        &mut chunks.iter().cloned(),
        IndexKind::Hnsw,
        sha,
        u64::MAX,
        |r| r.bytes_vec(len, 1),
    )
}

#[test]
fn splits_streams_and_reassembles() {
    let payload: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
    let e = enc(&payload, 1000);
    let body = 1000 - CHUNK_HEADER_LEN;
    assert_eq!(e.chunks.len(), payload.len().div_ceil(body));
    assert!(e
        .chunks
        .iter()
        .all(|c| c.bytes.len() <= 1000 && c.epoch == 9));
    assert_eq!(e.payload_len, payload.len() as u64);
    assert_eq!(
        raw(&e.chunks, Some(&e.sha256), payload.len()).unwrap(),
        payload
    );
    assert_eq!(
        dec(&e.chunks, IndexKind::QuantFlat, None).unwrap_err(),
        DecodeError::WrongKind
    );
    assert_eq!(
        raw(&e.chunks, Some(&[0; 32]), payload.len()).unwrap_err(),
        DecodeError::DigestMismatch
    );
    assert_eq!(
        dec(&[], IndexKind::Hnsw, None).unwrap_err(),
        DecodeError::Empty
    );
    assert_eq!(e.sha256_hex().len(), 64);
    // Only the last part carries the digest.
    let last = e.chunks.len() - 1;
    assert!(e.chunks[..last]
        .iter()
        .all(|c| c.bytes[DIGEST_AT..CRC_AT] == [0; 32]));
    assert_eq!(e.chunks[last].bytes[DIGEST_AT..CRC_AT], e.sha256);
}

#[test]
fn chunks_are_emitted_before_the_encode_finishes() {
    let payload = vec![5u8; 50_000];
    let mut seen = Vec::new();
    let write = |s: &mut dyn Sink| {
        for c in payload.chunks(1000) {
            s.put(c);
        }
    };
    let d = encode::<()>(1, IndexKind::QuantFlat, 4096, &write, &mut |c| {
        seen.push(c.bytes.len());
        Ok(())
    })
    .unwrap();
    assert_eq!(seen.len() as u32, d.parts);
    assert!(seen.iter().all(|&n| n <= 4096));
    // A failing sink surfaces as `EmitError::Sink`, later parts are skipped.
    let mut calls = 0;
    let r = encode(1, IndexKind::QuantFlat, 4096, &write, &mut |_| {
        calls += 1;
        Err("sql")
    });
    assert_eq!(r.unwrap_err(), EmitError::Sink("sql"));
    assert_eq!(calls, 1);
}

#[test]
fn rejects_bad_sizes_and_hostile_lengths() {
    let w = writer(&[]);
    assert!(encode_all(0, IndexKind::Hnsw, CHUNK_HEADER_LEN, &w).is_err());
    assert!(encode_all(0, IndexKind::Hnsw, MAX_CHUNK_BYTES + 1, &w).is_err());
    let e = enc(&[], MAX_CHUNK_BYTES);
    assert_eq!(e.chunks.len(), 1);
    assert!(raw(&e.chunks, None, 0).unwrap().is_empty());

    let payload = vec![1u8; 3000];
    let e = enc(&payload, 1000);
    // Declared payload beyond the caller's limit: refused before parsing.
    let err = decode(
        &mut e.chunks.iter().cloned(),
        IndexKind::Hnsw,
        None,
        100,
        |r| r.u8(),
    )
    .unwrap_err();
    assert_eq!(
        err,
        DecodeError::TooLarge {
            declared: 3000,
            limit: 100
        }
    );
    // An extra trailing part, even a well-formed one, is rejected.
    let mut c = e.chunks.clone();
    c.push(e.chunks[0].clone());
    assert!(raw(&c, None, 3000).is_err());
    // Resealing after a body edit passes integrity (then parsing decides).
    let mut c = e.chunks.clone();
    c[1].bytes[CHUNK_HEADER_LEN] ^= 0xFF;
    assert_eq!(
        raw(&c, None, 3000).unwrap_err(),
        DecodeError::ChecksumMismatch { part: 1 }
    );
    reseal(&mut c);
    assert_eq!(raw(&c, None, 3000).unwrap()[928], 0xFE);
    // Parsing fewer bytes than declared is a typed error.
    assert!(raw(&e.chunks, None, 2999).is_err());
}
