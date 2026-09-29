//! Persisted edge-list format `RVMC` v1: one manifest plus N chunks, each a
//! separate storage value (Durable Object SQLite BLOBs cap at 2 MB, ADR-351
//! §1; a chunk payload is at most [`MAX_CHUNK_PAYLOAD`]).
//!
//! Common header (32 bytes): `"RVMC" | version u16 | kind u8 | flags u8 |
//! graph_uid [16] | revision u64`, little-endian.
//!
//! * Manifest (kind 1): header `| vertex_count u64 | edge_count u64 |
//!   chunk_edges u32 | chunk_count u32 | chunk_digest [32] x chunk_count |
//!   sha256(all preceding) [32]`.
//! * Chunk (kind 2): header `| chunk_index u32 | edge_count u32 |
//!   payload_len u32 | payload | sha256(all preceding) [32]`.
//!
//! Chunk payload: edges sorted by `(u, v)`, `u < v`, delta-coded with
//! canonical LEB128. First edge of a chunk: `u, v-u-1`. Then `du`; if
//! `du == 0` it is followed by `v-prev_v-1`, else by `v-u-1`. Without the
//! unit-weight flag each edge is followed by its `f64` weight (LE).
//!
//! Binding: a chunk carries graph uid, revision, flags and its index; the
//! manifest lists every chunk's digest and is itself checksummed, so a
//! flipped bit, a swapped, missing, extra or foreign chunk and a
//! truncated record are all refused with a typed error.

use crate::codec::{put_varint, Reader};
use crate::error::{AnalyticsError, CorruptKind, Result};
use crate::graph::{EdgeRecord, GraphLimits, GraphUid, TenantGraph};
use sha2::{Digest, Sha256};

/// Format magic.
pub const MAGIC: [u8; 4] = *b"RVMC";
/// Format version written and accepted.
pub const FORMAT_VERSION: u16 = 1;
/// Weights are omitted: every edge weighs `1.0`.
pub const FLAG_UNIT_WEIGHTS: u8 = 0b0000_0001;
/// Hard cap on edges per chunk (32768 x <= 28 bytes < 1 MiB).
pub const MAX_CHUNK_EDGES: u32 = 32_768;
/// Hard cap on a chunk payload.
pub const MAX_CHUNK_PAYLOAD: u32 = 1 << 20;

const KIND_MANIFEST: u8 = 1;
const KIND_CHUNK: u8 = 2;
const HEADER_LEN: usize = 32;
const DIGEST_LEN: usize = 32;
const MANIFEST_FIXED: usize = HEADER_LEN + 8 + 8 + 4 + 4;

fn corrupt(kind: CorruptKind) -> AnalyticsError {
    AnalyticsError::Corrupt(kind)
}

/// Encoded snapshot: the manifest and chunk values, in chunk order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedGraph {
    /// Manifest record.
    pub manifest: Vec<u8>,
    /// Chunk records, index `i` at position `i`.
    pub chunks: Vec<Vec<u8>>,
}

/// A verified manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Graph identity.
    pub uid: GraphUid,
    /// Revision.
    pub revision: u64,
    /// Format flags.
    pub flags: u8,
    /// Distinct vertices.
    pub vertex_count: u64,
    /// Undirected edges.
    pub edge_count: u64,
    /// Edges per full chunk.
    pub chunk_edges: u32,
    /// Digest (trailer) of every chunk, in order.
    pub chunk_digests: Vec<[u8; DIGEST_LEN]>,
    /// This manifest's own trailer: a digest of the whole snapshot.
    pub digest: [u8; DIGEST_LEN],
}

fn put_header(out: &mut Vec<u8>, kind: u8, flags: u8, uid: &GraphUid, revision: u64) {
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.push(kind);
    out.push(flags);
    out.extend_from_slice(uid);
    out.extend_from_slice(&revision.to_le_bytes());
}

fn seal(mut rec: Vec<u8>) -> (Vec<u8>, [u8; DIGEST_LEN]) {
    let d: [u8; DIGEST_LEN] = Sha256::digest(&rec).into();
    rec.extend_from_slice(&d);
    (rec, d)
}

/// Verify the trailer; return the body (without trailer) and the digest.
fn unseal(rec: &[u8]) -> Result<(&[u8], [u8; DIGEST_LEN])> {
    if rec.len() < HEADER_LEN + DIGEST_LEN {
        return Err(corrupt(CorruptKind::Truncated));
    }
    let (body, trailer) = rec.split_at(rec.len() - DIGEST_LEN);
    let d: [u8; DIGEST_LEN] = Sha256::digest(body).into();
    if d[..] != trailer[..] {
        return Err(corrupt(CorruptKind::Checksum));
    }
    Ok((body, d))
}

struct Header {
    flags: u8,
    uid: GraphUid,
    revision: u64,
}

fn read_header(r: &mut Reader<'_>, kind: u8) -> Result<Header> {
    if r.bytes(4)? != MAGIC || r.u16()? != FORMAT_VERSION || r.u8()? != kind {
        return Err(corrupt(CorruptKind::Header));
    }
    let flags = r.u8()?;
    if flags & !FLAG_UNIT_WEIGHTS != 0 {
        return Err(corrupt(CorruptKind::Header));
    }
    let mut uid = [0u8; 16];
    uid.copy_from_slice(r.bytes(16)?);
    Ok(Header {
        flags,
        uid,
        revision: r.u64()?,
    })
}

/// Encode a graph into a manifest and `ceil(edges / chunk_edges)` chunks.
pub fn encode_graph(g: &TenantGraph, chunk_edges: u32) -> Result<EncodedGraph> {
    if chunk_edges == 0 || chunk_edges > MAX_CHUNK_EDGES {
        return Err(AnalyticsError::Invalid("chunk_edges must be 1..=32768"));
    }
    let flags = if g.unit_weights() {
        FLAG_UNIT_WEIGHTS
    } else {
        0
    };
    let mut chunks = Vec::new();
    let mut digests = Vec::new();
    for (index, part) in g.edges().chunks(chunk_edges as usize).enumerate() {
        let payload = encode_payload(part, flags);
        debug_assert!(payload.len() <= MAX_CHUNK_PAYLOAD as usize);
        let mut rec = Vec::with_capacity(HEADER_LEN + 12 + payload.len() + DIGEST_LEN);
        put_header(&mut rec, KIND_CHUNK, flags, g.uid(), g.revision());
        rec.extend_from_slice(&(index as u32).to_le_bytes());
        rec.extend_from_slice(&(part.len() as u32).to_le_bytes());
        rec.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        rec.extend_from_slice(&payload);
        let (rec, d) = seal(rec);
        chunks.push(rec);
        digests.push(d);
    }
    let mut m = Vec::with_capacity(MANIFEST_FIXED + DIGEST_LEN * (digests.len() + 1));
    put_header(&mut m, KIND_MANIFEST, flags, g.uid(), g.revision());
    m.extend_from_slice(&g.vertex_count().to_le_bytes());
    m.extend_from_slice(&g.edge_count().to_le_bytes());
    m.extend_from_slice(&chunk_edges.to_le_bytes());
    m.extend_from_slice(&(digests.len() as u32).to_le_bytes());
    for d in &digests {
        m.extend_from_slice(d);
    }
    let (manifest, _) = seal(m);
    Ok(EncodedGraph { manifest, chunks })
}

fn encode_payload(edges: &[EdgeRecord], flags: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(edges.len() * 6);
    let mut prev: Option<(u64, u64)> = None;
    for e in edges {
        match prev {
            None => {
                put_varint(&mut out, e.u);
                put_varint(&mut out, e.v - e.u - 1);
            }
            Some((pu, pv)) => {
                let du = e.u - pu;
                put_varint(&mut out, du);
                put_varint(&mut out, if du == 0 { e.v - pv - 1 } else { e.v - e.u - 1 });
            }
        }
        if flags & FLAG_UNIT_WEIGHTS == 0 {
            out.extend_from_slice(&e.w.to_le_bytes());
        }
        prev = Some((e.u, e.v));
    }
    out
}

impl Manifest {
    /// Verify and parse a manifest. Allocation is bounded by the input
    /// length: the digest list must exactly fill the record.
    pub fn decode(rec: &[u8]) -> Result<Manifest> {
        let (body, digest) = unseal(rec)?;
        let mut r = Reader::new(body);
        let h = read_header(&mut r, KIND_MANIFEST)?;
        let vertex_count = r.u64()?;
        let edge_count = r.u64()?;
        let chunk_edges = r.u32()?;
        let chunk_count = r.u32()?;
        if chunk_edges == 0 || chunk_edges > MAX_CHUNK_EDGES {
            return Err(corrupt(CorruptKind::Encoding));
        }
        if r.remaining() as u64 != u64::from(chunk_count) * DIGEST_LEN as u64 {
            return Err(corrupt(CorruptKind::Truncated));
        }
        if u64::from(chunk_count) != edge_count.div_ceil(u64::from(chunk_edges)) {
            return Err(corrupt(CorruptKind::CountMismatch));
        }
        let mut chunk_digests = Vec::with_capacity(chunk_count as usize);
        for _ in 0..chunk_count {
            let mut d = [0u8; DIGEST_LEN];
            d.copy_from_slice(r.bytes(DIGEST_LEN)?);
            chunk_digests.push(d);
        }
        Ok(Manifest {
            uid: h.uid,
            revision: h.revision,
            flags: h.flags,
            vertex_count,
            edge_count,
            chunk_edges,
            chunk_digests,
            digest,
        })
    }

    /// Hex of the snapshot digest (pins a job to exactly these bytes).
    pub fn digest_hex(&self) -> String {
        hex::encode(self.digest)
    }

    fn expected_edges(&self, index: usize) -> u64 {
        let full = u64::from(self.chunk_edges);
        let before = full * index as u64;
        (self.edge_count - before).min(full)
    }
}

/// Decode a graph from a verified manifest and its chunks in order. Counts
/// are checked against `limits` from the manifest before any chunk is
/// decoded, so an oversized graph is a 413 without loading it.
pub fn decode_graph<'a, I>(m: &Manifest, chunks: I, limits: &GraphLimits) -> Result<TenantGraph>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    limits.check(m.vertex_count, m.edge_count)?;
    let mut edges: Vec<EdgeRecord> = Vec::with_capacity(m.edge_count as usize);
    let mut last: Option<(u64, u64)> = None;
    let mut seen = 0usize;
    for (index, rec) in chunks.into_iter().enumerate() {
        let expected = m
            .chunk_digests
            .get(index)
            .ok_or(corrupt(CorruptKind::ChunkOrder))?;
        let (body, d) = unseal(rec)?;
        let mut r = Reader::new(body);
        let h = read_header(&mut r, KIND_CHUNK)?;
        if h.uid != m.uid || h.revision != m.revision || h.flags != m.flags {
            return Err(corrupt(CorruptKind::ForeignChunk));
        }
        if r.u32()? as usize != index || &d != expected {
            return Err(corrupt(CorruptKind::ChunkOrder));
        }
        let count = r.u32()?;
        if u64::from(count) != m.expected_edges(index) {
            return Err(corrupt(CorruptKind::CountMismatch));
        }
        let len = r.u32()?;
        if len > MAX_CHUNK_PAYLOAD || len as usize != r.remaining() {
            return Err(corrupt(CorruptKind::Truncated));
        }
        let payload = r.bytes(len as usize)?;
        decode_payload(payload, count, m.flags, &mut last, &mut edges)?;
        seen += 1;
    }
    if seen != m.chunk_digests.len() {
        return Err(corrupt(CorruptKind::ChunkOrder));
    }
    let g = TenantGraph::from_canonical(m.uid, m.revision, edges, limits)?;
    if g.vertex_count() != m.vertex_count {
        return Err(corrupt(CorruptKind::CountMismatch));
    }
    // Bind the graph to exactly these bytes (jobs pin to this digest).
    Ok(g.with_snapshot(m.digest))
}

fn decode_payload(
    payload: &[u8],
    count: u32,
    flags: u8,
    last: &mut Option<(u64, u64)>,
    out: &mut Vec<EdgeRecord>,
) -> Result<()> {
    let enc = || corrupt(CorruptKind::Encoding);
    let mut r = Reader::new(payload);
    let mut prev: Option<(u64, u64)> = None;
    for _ in 0..count {
        let (u, v) = match prev {
            None => {
                let u = r.varint()?;
                let v = u
                    .checked_add(r.varint()?)
                    .and_then(|x| x.checked_add(1))
                    .ok_or_else(enc)?;
                (u, v)
            }
            Some((pu, pv)) => {
                let du = r.varint()?;
                let dv = r.varint()?;
                if du == 0 {
                    (
                        pu,
                        pv.checked_add(dv)
                            .and_then(|x| x.checked_add(1))
                            .ok_or_else(enc)?,
                    )
                } else {
                    let u = pu.checked_add(du).ok_or_else(enc)?;
                    (
                        u,
                        u.checked_add(dv)
                            .and_then(|x| x.checked_add(1))
                            .ok_or_else(enc)?,
                    )
                }
            }
        };
        let w = if flags & FLAG_UNIT_WEIGHTS != 0 {
            1.0
        } else {
            f64::from_le_bytes(r.u64()?.to_le_bytes())
        };
        if !w.is_finite() || w < 0.0 {
            return Err(enc());
        }
        // Strict (u, v) order across chunk boundaries: no duplicates.
        if prev.is_none() && last.is_some_and(|l| (u, v) <= l) {
            return Err(corrupt(CorruptKind::ChunkOrder));
        }
        out.push(EdgeRecord { u, v, w });
        prev = Some((u, v));
        *last = prev;
    }
    if r.remaining() != 0 {
        return Err(corrupt(CorruptKind::Truncated));
    }
    Ok(())
}
