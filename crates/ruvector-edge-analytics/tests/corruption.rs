//! Persisted edge-list corruption: every tampering is a typed error, never
//! a panic, and never silently yields a different graph.
mod common;
use ruvector_edge_analytics::*;
use sha2::{Digest, Sha256};

const UID: GraphUid = [9; 16];

fn fixture(weighted: bool) -> (TenantGraph, EncodedGraph) {
    let mut edges = common::sparse_graph(300, 1_000, 77);
    if !weighted {
        edges.iter_mut().for_each(|e| e.2 = 1.0);
    }
    let g = TenantGraph::from_edges(UID, 3, &edges, &GraphLimits::INLINE).unwrap();
    let enc = encode_graph(&g, 128).unwrap();
    (g, enc)
}

fn load(enc: &EncodedGraph) -> Result<TenantGraph> {
    let m = Manifest::decode(&enc.manifest)?;
    decode_graph(
        &m,
        enc.chunks.iter().map(Vec::as_slice),
        &GraphLimits::INLINE,
    )
}

/// Re-seal a record after editing its body, so the edit gets past the
/// trailer and reaches the structural checks behind it.
fn reseal(rec: &mut Vec<u8>) {
    rec.truncate(rec.len() - 32);
    let d = Sha256::digest(&rec);
    rec.extend_from_slice(&d);
}

fn corrupt(r: Result<TenantGraph>) -> CorruptKind {
    match r {
        Err(AnalyticsError::Corrupt(k)) => k,
        other => panic!("expected Corrupt, got {other:?}"),
    }
}

#[test]
fn roundtrip_both_weight_encodings() {
    for weighted in [true, false] {
        let (g, enc) = fixture(weighted);
        let back = load(&enc).unwrap();
        assert_eq!(back.edges(), g.edges());
        assert_eq!((back.uid(), back.revision()), (g.uid(), g.revision()));
        let m = Manifest::decode(&enc.manifest).unwrap();
        assert_eq!(back.snapshot_digest(), Some(&m.digest));
    }
}

#[test]
fn every_single_bit_flip_is_refused() {
    let (_, enc) = fixture(true);
    // Every byte of one chunk and of the manifest.
    for i in 0..enc.chunks[1].len() {
        let mut bad = enc.clone();
        bad.chunks[1][i] ^= 0x01;
        assert!(load(&bad).is_err(), "chunk byte {i}");
    }
    for i in 0..enc.manifest.len() {
        let mut bad = enc.clone();
        bad.manifest[i] ^= 0x80;
        assert!(load(&bad).is_err(), "manifest byte {i}");
    }
}

#[test]
fn checksum_truncation_and_trailing_bytes() {
    let (_, enc) = fixture(true);
    let mut bad = enc.clone();
    let last = bad.chunks[0].len() - 1;
    bad.chunks[0][last] ^= 0xff;
    assert_eq!(corrupt(load(&bad)), CorruptKind::Checksum);

    let mut bad = enc.clone();
    let keep = bad.chunks[2].len() - 40;
    bad.chunks[2].truncate(keep);
    assert!(load(&bad).is_err());
    let mut bad = enc.clone();
    bad.chunks[2].truncate(10);
    assert_eq!(corrupt(load(&bad)), CorruptKind::Truncated);

    // Extra payload byte, chunk re-sealed: its digest no longer matches the
    // manifest list (trailing bytes under a valid list: forged tests below).
    let mut bad = enc.clone();
    let at = bad.chunks[0].len() - 32;
    bad.chunks[0].insert(at, 0);
    reseal(&mut bad.chunks[0]);
    assert_eq!(corrupt(load(&bad)), CorruptKind::ChunkOrder);
    assert!(Manifest::decode(&[]).is_err());
    assert!(Manifest::decode(&enc.manifest[..40]).is_err());
}

#[test]
fn reordered_missing_extra_and_foreign_chunks() {
    let (g, enc) = fixture(true);
    let mut bad = enc.clone();
    bad.chunks.swap(0, 1);
    assert_eq!(corrupt(load(&bad)), CorruptKind::ChunkOrder);

    let mut bad = enc.clone();
    bad.chunks.pop();
    assert_eq!(corrupt(load(&bad)), CorruptKind::ChunkOrder);

    let mut bad = enc.clone();
    bad.chunks.push(enc.chunks[0].clone());
    assert_eq!(corrupt(load(&bad)), CorruptKind::ChunkOrder);

    // Same edges, other graph uid / revision: chunks are not interchangeable.
    let other =
        TenantGraph::from_edges([8; 16], 3, &g.solver_edges(), &GraphLimits::INLINE).unwrap();
    let foreign = encode_graph(&other, 128).unwrap();
    let mut bad = enc.clone();
    bad.chunks[1] = foreign.chunks[1].clone();
    assert_eq!(corrupt(load(&bad)), CorruptKind::ForeignChunk);
    let newer = TenantGraph::from_edges(UID, 4, &g.solver_edges(), &GraphLimits::INLINE).unwrap();
    let mut bad = enc.clone();
    bad.chunks[0] = encode_graph(&newer, 128).unwrap().chunks[0].clone();
    assert_eq!(corrupt(load(&bad)), CorruptKind::ForeignChunk);

    // A chunk from a different chunking of the same graph (valid on its own).
    let mut bad = enc.clone();
    bad.chunks[0] = encode_graph(&g, 64).unwrap().chunks[0].clone();
    assert!(load(&bad).is_err());
}

#[test]
fn manifest_counts_and_header() {
    let (_, enc) = fixture(true);
    // Vertex count off by one, re-sealed.
    let mut bad = enc.clone();
    bad.manifest[32] = bad.manifest[32].wrapping_add(1);
    reseal(&mut bad.manifest);
    assert_eq!(corrupt(load(&bad)), CorruptKind::CountMismatch);
    // Edge count inconsistent with chunk count.
    let mut bad = enc.clone();
    bad.manifest[40] = bad.manifest[40].wrapping_add(200);
    reseal(&mut bad.manifest);
    assert!(matches!(corrupt(load(&bad)), CorruptKind::CountMismatch));
    // Version bump and unknown flag bit.
    for (at, v) in [(4usize, 2u8), (7, 0x80)] {
        let mut bad = enc.clone();
        bad.manifest[at] = v;
        reseal(&mut bad.manifest);
        assert_eq!(corrupt(load(&bad)), CorruptKind::Header);
    }
    // A chunk presented as a manifest.
    assert_eq!(
        Manifest::decode(&enc.chunks[0]).unwrap_err(),
        AnalyticsError::Corrupt(CorruptKind::Header)
    );
}

/// Independent writer for the documented layout, used to forge records
/// whose checksums and digest list are all valid, so the payload checks
/// behind them are what gets exercised.
mod forge {
    use super::*;

    fn record(kind: u8, flags: u8, body: &[u8]) -> Vec<u8> {
        let mut r = b"RVMC".to_vec();
        r.extend_from_slice(&1u16.to_le_bytes());
        r.push(kind);
        r.push(flags);
        r.extend_from_slice(&UID);
        r.extend_from_slice(&3u64.to_le_bytes());
        r.extend_from_slice(body);
        let d = Sha256::digest(&r);
        r.extend_from_slice(&d);
        r
    }

    pub fn chunk(flags: u8, index: u32, count: u32, payload: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&index.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        b.extend_from_slice(payload);
        record(2, flags, &b)
    }

    pub fn graph(
        flags: u8,
        vertices: u64,
        chunk_edges: u32,
        chunks: Vec<(u32, Vec<u8>)>,
    ) -> EncodedGraph {
        let chunks: Vec<(u32, Vec<u8>)> = chunks;
        let edges: u64 = chunks.iter().map(|c| u64::from(c.0)).sum();
        let recs: Vec<Vec<u8>> = chunks
            .iter()
            .enumerate()
            .map(|(i, (n, p))| chunk(flags, i as u32, *n, p))
            .collect();
        let mut b = Vec::new();
        b.extend_from_slice(&vertices.to_le_bytes());
        b.extend_from_slice(&edges.to_le_bytes());
        b.extend_from_slice(&chunk_edges.to_le_bytes());
        b.extend_from_slice(&(recs.len() as u32).to_le_bytes());
        for r in &recs {
            b.extend_from_slice(&r[r.len() - 32..]);
        }
        EncodedGraph {
            manifest: record(1, flags, &b),
            chunks: recs,
        }
    }
}

fn weighted_edge(u: u8, dv: u8, w: f64) -> Vec<u8> {
    let mut p = vec![u, dv];
    p.extend_from_slice(&w.to_le_bytes());
    p
}

#[test]
fn forged_valid_graph_decodes() {
    // (1,2) w=2.5 and, after du=0, (1, 2+0+1 = 3) w=1.
    let mut p = weighted_edge(1, 0, 2.5);
    p.extend_from_slice(&[0, 0]);
    p.extend_from_slice(&1.0f64.to_le_bytes());
    let g = load(&forge::graph(0, 3, 8, vec![(2, p)])).unwrap();
    assert_eq!(g.solver_edges(), vec![(1, 2, 2.5), (1, 3, 1.0)]);
}

#[test]
fn forged_bad_weights_and_ids_are_refused() {
    for w in [f64::NAN, f64::INFINITY, -1.0] {
        let bad = forge::graph(0, 2, 8, vec![(1, weighted_edge(1, 0, w))]);
        assert_eq!(corrupt(load(&bad)), CorruptKind::Encoding, "weight {w}");
    }
    // u = u64::MAX, v = u + 0 + 1 overflows.
    let mut p = vec![0xff; 9];
    p.push(0x01);
    p.push(0x00);
    assert_eq!(
        corrupt(load(&forge::graph(1, 2, 8, vec![(1, p)]))),
        CorruptKind::Encoding
    );
    // Overlong varint for u.
    let p = vec![0x81, 0x00, 0x00];
    assert_eq!(
        corrupt(load(&forge::graph(1, 2, 8, vec![(1, p)]))),
        CorruptKind::Encoding
    );
    // Trailing byte after the last edge.
    assert_eq!(
        corrupt(load(&forge::graph(1, 2, 8, vec![(1, vec![1, 0, 0])]))),
        CorruptKind::Truncated
    );
    // Count says 2 edges, payload holds 1.
    assert_eq!(
        corrupt(load(&forge::graph(1, 2, 8, vec![(2, vec![1, 0])]))),
        CorruptKind::Truncated
    );
}

#[test]
fn forged_cross_chunk_duplicate_is_refused() {
    // chunk_edges = 1: (1,2) then (1,2) again in the next chunk.
    let bad = forge::graph(1, 2, 1, vec![(1, vec![1, 0]), (1, vec![1, 0])]);
    assert_eq!(corrupt(load(&bad)), CorruptKind::ChunkOrder);
    // Decreasing across chunks is refused too; increasing is fine.
    let bad = forge::graph(1, 3, 1, vec![(1, vec![2, 0]), (1, vec![1, 0])]);
    assert_eq!(corrupt(load(&bad)), CorruptKind::ChunkOrder);
    let ok = forge::graph(1, 3, 1, vec![(1, vec![1, 0]), (1, vec![2, 0])]);
    assert_eq!(load(&ok).unwrap().edge_count(), 2);
}

#[test]
fn forged_vertex_count_mismatch_is_refused() {
    let bad = forge::graph(1, 5, 8, vec![(1, vec![1, 0])]);
    assert_eq!(corrupt(load(&bad)), CorruptKind::CountMismatch);
}

#[test]
fn forged_huge_chunk_count_is_refused_without_overflow() {
    // Valid trailer, chunk_count = u32::MAX but no digest list behind it.
    let mut b = Vec::new();
    b.extend_from_slice(&2u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&u32::MAX.to_le_bytes());
    let mut r = b"RVMC".to_vec();
    r.extend_from_slice(&1u16.to_le_bytes());
    r.extend_from_slice(&[1, 1]);
    r.extend_from_slice(&UID);
    r.extend_from_slice(&3u64.to_le_bytes());
    r.extend_from_slice(&b);
    let d = Sha256::digest(&r);
    r.extend_from_slice(&d);
    assert_eq!(
        Manifest::decode(&r).unwrap_err(),
        AnalyticsError::Corrupt(CorruptKind::Truncated)
    );
}
