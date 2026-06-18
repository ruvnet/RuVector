//! Binary wire format for shard serialization.
//!
//! Layout:
//! ```text
//! Header   8 bytes  magic "RVSHARD\0"
//!          4 bytes  version (u32 LE) = 1
//!          4 bytes  variant (u32 LE): 0=Bfs, 1=Coherence, 2=Hub
//!          4 bytes  dim (u32 LE)
//!          8 bytes  node_count (u64 LE)
//!
//! Per node (repeated node_count times):
//!          4 bytes  node_id (u32 LE)
//!          dim × 4  vector (f32 LE each)
//!          4 bytes  n_local_neighbors (u32 LE)
//!          n_local_neighbors × 4  local neighbor IDs (u32 LE each)
//! ```

use crate::error::{ShardError, ShardResult};
use crate::shard::{Shard, ShardMeta, ShardVariant};

const MAGIC: &[u8; 8] = b"RVSHARD\0";
const VERSION: u32 = 1;
/// Sanity cap: refuse to deserialize shards with more than 1 M nodes.
const MAX_NODES: u64 = 1_000_000;

pub fn write_shard(shard: &Shard) -> Vec<u8> {
    let n = shard.node_ids.len();
    let dim = shard.dim;

    // Pre-calculate capacity for a single allocation.
    let nb_total: usize = shard.local_neighbors.iter().map(|nb| nb.len()).sum();
    let capacity = 8 + 4 + 4 + 4 + 8 + n * (4 + dim * 4 + 4) + nb_total * 4;
    let mut buf = Vec::with_capacity(capacity);

    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(variant_to_u32(shard.variant)).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&(n as u64).to_le_bytes());

    for (local, &global_id) in shard.node_ids.iter().enumerate() {
        buf.extend_from_slice(&global_id.to_le_bytes());
        for &f in shard.get_vector(local) {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        let nbs = &shard.local_neighbors[local];
        buf.extend_from_slice(&(nbs.len() as u32).to_le_bytes());
        for &nb in nbs {
            buf.extend_from_slice(&nb.to_le_bytes());
        }
    }

    buf
}

pub fn read_shard(bytes: &[u8]) -> ShardResult<Shard> {
    let mut pos = 0usize;

    // Header
    let magic = read_bytes::<8>(bytes, &mut pos)?;
    if &magic != MAGIC {
        return Err(ShardError::BadMagic);
    }
    let version = read_u32(bytes, &mut pos)?;
    if version != VERSION {
        return Err(ShardError::UnsupportedVersion(version));
    }
    let variant_raw = read_u32(bytes, &mut pos)?;
    let variant = u32_to_variant(variant_raw)?;
    let dim = read_u32(bytes, &mut pos)? as usize;
    let node_count = read_u64(bytes, &mut pos)?;
    if node_count > MAX_NODES {
        return Err(ShardError::NodeCountTooLarge(node_count));
    }
    let n = node_count as usize;

    let mut node_ids = Vec::with_capacity(n);
    let mut vectors = Vec::with_capacity(n * dim);
    let mut local_neighbors: Vec<Vec<u32>> = Vec::with_capacity(n);

    for _ in 0..n {
        node_ids.push(read_u32(bytes, &mut pos)?);
        for _ in 0..dim {
            let fb = read_bytes::<4>(bytes, &mut pos)?;
            vectors.push(f32::from_le_bytes(fb));
        }
        let nb_count = read_u32(bytes, &mut pos)? as usize;
        let mut nbs = Vec::with_capacity(nb_count);
        for _ in 0..nb_count {
            nbs.push(read_u32(bytes, &mut pos)?);
        }
        local_neighbors.push(nbs);
    }

    Ok(Shard {
        variant,
        dim,
        node_ids,
        vectors,
        local_neighbors,
        meta: ShardMeta {
            variant,
            extraction_us: 0,
        },
    })
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn read_bytes<const N: usize>(src: &[u8], pos: &mut usize) -> ShardResult<[u8; N]> {
    if *pos + N > src.len() {
        return Err(ShardError::Truncated(*pos));
    }
    let mut buf = [0u8; N];
    buf.copy_from_slice(&src[*pos..*pos + N]);
    *pos += N;
    Ok(buf)
}

fn read_u32(src: &[u8], pos: &mut usize) -> ShardResult<u32> {
    Ok(u32::from_le_bytes(read_bytes::<4>(src, pos)?))
}

fn read_u64(src: &[u8], pos: &mut usize) -> ShardResult<u64> {
    Ok(u64::from_le_bytes(read_bytes::<8>(src, pos)?))
}

fn variant_to_u32(v: ShardVariant) -> u32 {
    match v {
        ShardVariant::Bfs => 0,
        ShardVariant::Coherence => 1,
        ShardVariant::Hub => 2,
    }
}

fn u32_to_variant(v: u32) -> ShardResult<ShardVariant> {
    match v {
        0 => Ok(ShardVariant::Bfs),
        1 => Ok(ShardVariant::Coherence),
        2 => Ok(ShardVariant::Hub),
        other => Err(ShardError::UnsupportedVersion(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphConfig, KnnGraph};
    use crate::shard::{BfsShard, ShardExtractor};

    fn make_shard() -> Shard {
        let vecs: Vec<f32> = (0..10).flat_map(|i| vec![i as f32, 0.0]).collect();
        let g = KnnGraph::build(vecs, 2, &GraphConfig { k_neighbors: 3 }).unwrap();
        BfsShard.extract(&g, &[0], 5)
    }

    #[test]
    fn round_trip_preserves_node_ids() {
        let s = make_shard();
        let wire = write_shard(&s);
        let s2 = read_shard(&wire).unwrap();
        assert_eq!(s.node_ids, s2.node_ids);
    }

    #[test]
    fn round_trip_preserves_vectors() {
        let s = make_shard();
        let wire = write_shard(&s);
        let s2 = read_shard(&wire).unwrap();
        for (a, b) in s.vectors.iter().zip(s2.vectors.iter()) {
            assert!((a - b).abs() < 1e-6, "vector mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn round_trip_preserves_neighbors() {
        let s = make_shard();
        let wire = write_shard(&s);
        let s2 = read_shard(&wire).unwrap();
        assert_eq!(s.local_neighbors, s2.local_neighbors);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut wire = write_shard(&make_shard());
        wire[0] = 0xFF;
        assert!(matches!(read_shard(&wire), Err(ShardError::BadMagic)));
    }

    #[test]
    fn truncated_data_rejected() {
        let wire = write_shard(&make_shard());
        let short = &wire[..wire.len() / 2];
        assert!(matches!(read_shard(short), Err(ShardError::Truncated(_))));
    }
}
