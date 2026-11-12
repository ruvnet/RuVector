//! Wire payloads of the progressive layers carried as INDEX segments.
//!
//! * Layer A (`RVPA`): entry points, top-layer adjacency, centroids, the
//!   centroid-to-id-range map, and **anchor vectors** (the vectors of the
//!   entry point and top-layer nodes, so the routing structure is usable the
//!   moment it is verified, without waiting for the segment that holds them).
//! * Layer B (`RVPB`): layer-0 adjacency of the hot nodes plus covered ranges.
//! * Layer C: the store's own persisted HNSW INDEX payload (`RVIX` trailer),
//!   so `RvfStore::open` loads the very same graph (index held constant).
//!
//! `rvf-index`'s INDEX_SEG codec carries adjacency only; centroids, entry
//! points, the partition map and B's covered ranges have no wire form there,
//! hence these envelopes. All counts are bounded by the remaining bytes
//! before any allocation; decoding never panics on hostile input.

use rvf_index::{HnswLayer, LayerA, LayerB, PartitionEntry};
use std::collections::BTreeMap;

/// Layer A payload magic.
pub const MAGIC_A: [u8; 4] = *b"RVPA";
/// Layer B payload magic.
pub const MAGIC_B: [u8; 4] = *b"RVPB";

/// Most partitions a Layer A may carry.
pub const MAX_PARTITIONS: usize = 65_536;

/// True if the partitions are well formed, sorted by `vector_id_start` and
/// pairwise disjoint. The producer (`build`) emits maps that satisfy this by
/// construction when ids are assigned cluster by cluster.
pub fn partitions_disjoint(parts: &[PartitionEntry]) -> bool {
    parts.iter().all(|p| p.vector_id_end >= p.vector_id_start)
        && parts
            .windows(2)
            .all(|w| w[0].vector_id_end <= w[1].vector_id_start)
}

/// A decoded Layer A with its anchor vectors.
pub struct DecodedA {
    /// The routing layer.
    pub layer: LayerA,
    /// Vector dimension.
    pub dim: usize,
    /// `(id, vector)` for the entry point and top-layer nodes.
    pub anchors: Vec<(u64, Vec<f32>)>,
}

/// Decoded in-memory size is limited to this multiple of the payload size
/// (plus a small constant). A signed payload therefore cannot make a client
/// allocate more than about twice its wire size; legitimate layers decode at
/// roughly 1.3x. Counts are charged *before* anything is allocated.
pub const MAX_DECODE_AMPLIFICATION: usize = 2;
const DECODE_SLACK: usize = 4096;
/// Approximate per-node overhead of a `BTreeMap<u64, Vec<u64>>` entry.
const NODE_OVERHEAD: usize = 80;

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
    budget: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self {
            b,
            p: 0,
            budget: b.len() * MAX_DECODE_AMPLIFICATION + DECODE_SLACK,
        }
    }
    /// Charge `n` bytes of decoded memory; `None` once the budget is spent.
    fn charge(&mut self, n: usize) -> Option<()> {
        self.budget = self.budget.checked_sub(n)?;
        Some(())
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn left(&self) -> usize {
        self.b.len() - self.p
    }
    /// A count whose `unit`-byte items must fit in the remaining bytes.
    fn count(&mut self, unit: usize) -> Option<usize> {
        let n = self.u32()? as usize;
        (n.checked_mul(unit)? <= self.left()).then_some(n)
    }
}

fn put_layer(out: &mut Vec<u8>, l: &HnswLayer) {
    out.extend_from_slice(&(l.adjacency.len() as u32).to_le_bytes());
    for (id, nb) in &l.adjacency {
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        for n in nb {
            out.extend_from_slice(&n.to_le_bytes());
        }
    }
}

fn get_layer(c: &mut Cur<'_>) -> Option<HnswLayer> {
    let n = c.count(12)?;
    let mut l = HnswLayer::default();
    for _ in 0..n {
        let id = c.u64()?;
        let k = c.count(8)?;
        c.charge(NODE_OVERHEAD + 8 * k)?;
        let mut nb = Vec::with_capacity(k);
        for _ in 0..k {
            nb.push(c.u64()?);
        }
        l.adjacency.insert(id, nb);
    }
    Some(l)
}

/// Encode Layer A.
pub fn encode_a(a: &LayerA, dim: usize, anchors: &[(u64, &[f32])]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(&MAGIC_A);
    o.extend_from_slice(&(dim as u32).to_le_bytes());
    o.extend_from_slice(&(a.top_layer_start as u32).to_le_bytes());
    o.extend_from_slice(&(a.entry_points.len() as u32).to_le_bytes());
    for (id, lvl) in &a.entry_points {
        o.extend_from_slice(&id.to_le_bytes());
        o.extend_from_slice(&lvl.to_le_bytes());
    }
    o.extend_from_slice(&(a.top_layers.len() as u32).to_le_bytes());
    for l in &a.top_layers {
        put_layer(&mut o, l);
    }
    o.extend_from_slice(&(a.centroids.len() as u32).to_le_bytes());
    for c in &a.centroids {
        for x in c {
            o.extend_from_slice(&x.to_le_bytes());
        }
    }
    o.extend_from_slice(&(a.partition_map.len() as u32).to_le_bytes());
    for p in &a.partition_map {
        o.extend_from_slice(&p.centroid_id.to_le_bytes());
        o.extend_from_slice(&p.vector_id_start.to_le_bytes());
        o.extend_from_slice(&p.vector_id_end.to_le_bytes());
    }
    o.extend_from_slice(&(anchors.len() as u32).to_le_bytes());
    for (id, v) in anchors {
        o.extend_from_slice(&id.to_le_bytes());
        for x in *v {
            o.extend_from_slice(&x.to_le_bytes());
        }
    }
    o
}

fn floats(c: &mut Cur<'_>, dim: usize) -> Option<Vec<f32>> {
    let raw = c.take(dim.checked_mul(4)?)?;
    Some(
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

/// Decode Layer A; `None` on any malformation.
pub fn decode_a(b: &[u8]) -> Option<DecodedA> {
    let mut c = Cur::new(b);
    if c.take(4)? != MAGIC_A {
        return None;
    }
    let dim = c.u32()? as usize;
    if dim == 0 || dim > 65_536 {
        return None;
    }
    let top_layer_start = c.u32()? as usize;
    let ne = c.count(12)?;
    c.charge(ne * std::mem::size_of::<(u64, u32)>())?;
    let mut entry_points = Vec::with_capacity(ne);
    for _ in 0..ne {
        entry_points.push((c.u64()?, c.u32()?));
    }
    let nl = c.count(4)?;
    c.charge(nl * (std::mem::size_of::<HnswLayer>() + 8))?;
    let mut top_layers = Vec::with_capacity(nl);
    for _ in 0..nl {
        top_layers.push(get_layer(&mut c)?);
    }
    let nc = c.count(dim * 4)?;
    c.charge(nc * (24 + 4 * dim))?;
    let mut centroids = Vec::with_capacity(nc);
    for _ in 0..nc {
        centroids.push(floats(&mut c, dim)?);
    }
    let np = c.count(20)?;
    // Partition maps are routing metadata, not data: cap them (F3-01).
    if np > MAX_PARTITIONS {
        return None;
    }
    c.charge(np * std::mem::size_of::<PartitionEntry>())?;
    let mut partition_map = Vec::with_capacity(np);
    for _ in 0..np {
        partition_map.push(PartitionEntry {
            centroid_id: c.u32()?,
            vector_id_start: c.u64()?,
            vector_id_end: c.u64()?,
            segment_ref: 0,
            block_ref: 0,
        });
    }
    // Partitions must be sorted by start and pairwise disjoint (zero-width is
    // fine), so one query visits any id at most once per matching partition
    // and the number of partitions bounds nothing but the map itself.
    if !partitions_disjoint(&partition_map) {
        return None;
    }
    let na = c.count(8 + dim * 4)?;
    c.charge(na * (8 + 24 + 4 * dim))?;
    let mut anchors = Vec::with_capacity(na);
    for _ in 0..na {
        let id = c.u64()?;
        anchors.push((id, floats(&mut c, dim)?));
    }
    Some(DecodedA {
        layer: LayerA {
            entry_points,
            top_layers,
            top_layer_start,
            centroids,
            partition_map,
        },
        dim,
        anchors,
    })
}

/// Encode Layer B.
pub fn encode_b(b: &LayerB) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(&MAGIC_B);
    o.extend_from_slice(&(b.partial_adjacency.len() as u32).to_le_bytes());
    for (id, nb) in &b.partial_adjacency {
        o.extend_from_slice(&id.to_le_bytes());
        o.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        for n in nb {
            o.extend_from_slice(&n.to_le_bytes());
        }
    }
    o.extend_from_slice(&(b.covered_ranges.len() as u32).to_le_bytes());
    for (s, e) in &b.covered_ranges {
        o.extend_from_slice(&s.to_le_bytes());
        o.extend_from_slice(&e.to_le_bytes());
    }
    o
}

/// Decode Layer B; `None` on any malformation.
pub fn decode_b(b: &[u8]) -> Option<LayerB> {
    let mut c = Cur::new(b);
    if c.take(4)? != MAGIC_B {
        return None;
    }
    let n = c.count(12)?;
    let mut partial_adjacency = BTreeMap::new();
    for _ in 0..n {
        let id = c.u64()?;
        let k = c.count(8)?;
        c.charge(NODE_OVERHEAD + 8 * k)?;
        let mut nb = Vec::with_capacity(k);
        for _ in 0..k {
            nb.push(c.u64()?);
        }
        partial_adjacency.insert(id, nb);
    }
    let r = c.count(16)?;
    c.charge(r * 16)?;
    let mut covered_ranges = Vec::with_capacity(r);
    for _ in 0..r {
        covered_ranges.push((c.u64()?, c.u64()?));
    }
    Some(LayerB {
        partial_adjacency,
        covered_ranges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_a() -> LayerA {
        let mut l = HnswLayer::default();
        l.adjacency.insert(7, vec![1, 2]);
        LayerA {
            entry_points: vec![(7, 3)],
            top_layers: vec![l],
            top_layer_start: 3,
            centroids: vec![vec![0.5, 1.5]],
            partition_map: vec![PartitionEntry {
                centroid_id: 0,
                vector_id_start: 0,
                vector_id_end: 9,
                segment_ref: 0,
                block_ref: 0,
            }],
        }
    }

    #[test]
    fn a_round_trips_and_truncation_never_panics() {
        let a = sample_a();
        let enc = encode_a(&a, 2, &[(7, &[1.0, 2.0])]);
        let d = decode_a(&enc).unwrap();
        assert_eq!(d.layer.entry_points, a.entry_points);
        assert_eq!(d.anchors, vec![(7, vec![1.0, 2.0])]);
        for cut in 0..enc.len() {
            assert!(decode_a(&enc[..cut]).is_none(), "prefix {cut}");
        }
        let mut hostile = enc.clone();
        hostile[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        let _ = decode_a(&hostile);
    }

    #[test]
    fn b_round_trips_and_rejects_bad_counts() {
        let mut m = BTreeMap::new();
        m.insert(4u64, vec![1u64, 2, 3]);
        let b = LayerB {
            partial_adjacency: m,
            covered_ranges: vec![(0, 10)],
        };
        let enc = encode_b(&b);
        let d = decode_b(&enc).unwrap();
        assert_eq!(d.partial_adjacency, b.partial_adjacency);
        assert_eq!(d.covered_ranges, b.covered_ranges);
        let mut bad = enc.clone();
        bad[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_b(&bad).is_none());
        for cut in 0..enc.len() {
            assert!(decode_b(&enc[..cut]).is_none());
        }
    }
}
