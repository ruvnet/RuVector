//! VEC_SEG block codec.
//!
//! Parses and writes the block directory, columnar vector data, ID map with
//! delta-varint encoding, and per-block CRC32C.

use crate::delta::encode_delta;
use crate::hash::compute_crc32c;
use crate::varint::decode_varint;

/// A single entry in the block directory.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDirEntry {
    pub block_offset: u32,
    pub vector_count: u32,
    pub dim: u16,
    pub dtype: u8,
    pub tier: u8,
}

/// Size of the block directory header (block_count: u32).
const DIR_HEADER_SIZE: usize = 4;

/// Size of each directory entry: offset(4) + count(4) + dim(2) + dtype(1) + tier(1) = 12.
const DIR_ENTRY_SIZE: usize = 12;

/// Parsed VEC_SEG block directory.
#[derive(Clone, Debug)]
pub struct BlockDirectory {
    pub entries: Vec<BlockDirEntry>,
}

/// A decoded VEC_SEG block.
#[derive(Clone, Debug)]
pub struct VecBlock {
    /// Columnar vector data (all dims for dim 0, then dim 1, etc).
    pub vector_data: Vec<u8>,
    /// Decoded vector IDs.
    pub ids: Vec<u64>,
    /// Dimensions.
    pub dim: u16,
    /// Data type.
    pub dtype: u8,
    /// Temperature tier.
    pub tier: u8,
}

/// Parse the block directory from the start of a VEC_SEG payload.
///
/// Returns the directory and the number of bytes consumed.
pub fn read_block_directory(data: &[u8]) -> Result<(BlockDirectory, usize), &'static str> {
    if data.len() < DIR_HEADER_SIZE {
        return Err("block directory truncated");
    }
    let block_count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let dir_size = DIR_HEADER_SIZE + block_count * DIR_ENTRY_SIZE;
    if data.len() < dir_size {
        return Err("block directory entries truncated");
    }
    let mut entries = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let base = DIR_HEADER_SIZE + i * DIR_ENTRY_SIZE;
        entries.push(BlockDirEntry {
            block_offset: u32::from_le_bytes(data[base..base + 4].try_into().unwrap()),
            vector_count: u32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()),
            dim: u16::from_le_bytes([data[base + 8], data[base + 9]]),
            dtype: data[base + 10],
            tier: data[base + 11],
        });
    }
    // Align consumed size to 64 bytes
    let consumed = (dir_size + 63) & !63;
    Ok((BlockDirectory { entries }, consumed))
}

/// Write a block directory to a Vec<u8>. Pads to 64-byte alignment.
pub fn write_block_directory(entries: &[BlockDirEntry]) -> Vec<u8> {
    let dir_size = DIR_HEADER_SIZE + entries.len() * DIR_ENTRY_SIZE;
    let padded = (dir_size + 63) & !63;
    let mut buf = vec![0u8; padded];
    buf[0..4].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    for (i, e) in entries.iter().enumerate() {
        let base = DIR_HEADER_SIZE + i * DIR_ENTRY_SIZE;
        buf[base..base + 4].copy_from_slice(&e.block_offset.to_le_bytes());
        buf[base + 4..base + 8].copy_from_slice(&e.vector_count.to_le_bytes());
        buf[base + 8..base + 10].copy_from_slice(&e.dim.to_le_bytes());
        buf[base + 10] = e.dtype;
        buf[base + 11] = e.tier;
    }
    buf
}

/// Size of an element in bytes for a given dtype.
fn dtype_element_size(dtype: u8) -> usize {
    match dtype {
        0x00 => 4, // f32
        0x01 => 2, // f16
        0x02 => 2, // bf16
        0x03 => 1, // i8
        0x04 => 1, // u8
        _ => 1,    // fallback
    }
}

/// Default restart interval for delta-encoded ID maps.
const DEFAULT_RESTART_INTERVAL: u32 = 128;

/// Write a single VEC_SEG block (columnar vectors + ID map + CRC32C).
/// Returns the serialized block bytes, padded to 64-byte alignment.
pub fn write_vec_block(block: &VecBlock) -> Vec<u8> {
    let mut payload = Vec::new();

    // Columnar vector data
    payload.extend_from_slice(&block.vector_data);

    // ID map header
    let encoding: u8 = 1; // delta-varint
    let restart_interval: u16 = DEFAULT_RESTART_INTERVAL as u16;
    let id_count: u32 = block.ids.len() as u32;
    payload.push(encoding);
    payload.extend_from_slice(&restart_interval.to_le_bytes());
    payload.extend_from_slice(&id_count.to_le_bytes());

    // For delta encoding, we store restart offsets (simplified: omit for now)
    // and then the encoded IDs.
    let _restart_count = if block.ids.is_empty() {
        0u32
    } else {
        ((block.ids.len() as u32 - 1) / DEFAULT_RESTART_INTERVAL) + 1
    };
    let mut id_buf = Vec::new();
    encode_delta(&block.ids, DEFAULT_RESTART_INTERVAL, &mut id_buf);
    payload.extend_from_slice(&id_buf);

    // Block CRC32C
    let crc = compute_crc32c(&payload);
    payload.extend_from_slice(&crc.to_le_bytes());

    // Pad to 64-byte alignment
    let padded_len = (payload.len() + 63) & !63;
    payload.resize(padded_len, 0);
    payload
}

/// Read a single VEC_SEG block at the given offset in the payload.
///
/// `entry` provides the block metadata from the directory.
/// `payload` is the full VEC_SEG payload.
pub fn read_vec_block(payload: &[u8], entry: &BlockDirEntry) -> Result<VecBlock, &'static str> {
    let offset = entry.block_offset as usize;
    if offset >= payload.len() {
        return Err("block offset beyond payload");
    }
    let block_data = &payload[offset..];

    let elem_size = dtype_element_size(entry.dtype);
    let vector_data_len = (entry.vector_count as usize)
        .checked_mul(entry.dim as usize)
        .and_then(|n| n.checked_mul(elem_size))
        .ok_or("vector data size overflow")?;
    let header_end = vector_data_len
        .checked_add(7)
        .ok_or("block size overflow")?;
    if block_data.len() < header_end {
        return Err("block data truncated");
    }
    let vector_data = block_data[..vector_data_len].to_vec();

    let mut pos = vector_data_len;
    let encoding = block_data[pos];
    pos += 1;
    let restart_interval = u16::from_le_bytes([block_data[pos], block_data[pos + 1]]);
    pos += 2;
    let id_count = u32::from_le_bytes(block_data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if id_count != entry.vector_count as usize {
        return Err("ID count does not match vector count");
    }

    let ids = if encoding == 1 {
        // The low-level decode_delta API panics on malformed input. This
        // public Result-returning reader must validate before allocation and
        // propagate decoding/overflow failures instead.
        if restart_interval == 0 {
            return Err("restart interval must be positive");
        }
        if id_count > block_data.len() - pos {
            return Err("delta ID data truncated");
        }
        let mut ids = Vec::with_capacity(id_count);
        let mut previous = 0u64;
        for i in 0..id_count {
            let (value, consumed) =
                decode_varint(&block_data[pos..]).map_err(|_| "delta ID data truncated")?;
            pos += consumed;
            previous = if i % restart_interval as usize == 0 {
                value
            } else {
                previous.checked_add(value).ok_or("delta ID overflow")?
            };
            ids.push(previous);
        }
        ids
    } else if encoding == 0 {
        // Raw u64 IDs
        let raw_len = id_count.checked_mul(8).ok_or("raw ID size overflow")?;
        if raw_len > block_data.len() - pos {
            return Err("raw ID data truncated");
        }
        let mut ids = Vec::with_capacity(id_count);
        for i in 0..id_count {
            let id_offset = pos + i * 8;
            if id_offset + 8 > block_data.len() {
                return Err("raw ID data truncated");
            }
            ids.push(u64::from_le_bytes(
                block_data[id_offset..id_offset + 8].try_into().unwrap(),
            ));
        }
        ids
    } else {
        return Err("unsupported ID encoding");
    };

    Ok(VecBlock {
        vector_data,
        ids,
        dim: entry.dim,
        dtype: entry.dtype,
        tier: entry.tier,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_directory_round_trip() {
        let entries = vec![
            BlockDirEntry {
                block_offset: 64,
                vector_count: 100,
                dim: 128,
                dtype: 0,
                tier: 0,
            },
            BlockDirEntry {
                block_offset: 51264,
                vector_count: 200,
                dim: 128,
                dtype: 1,
                tier: 1,
            },
        ];
        let buf = write_block_directory(&entries);
        assert_eq!(buf.len() % 64, 0);
        let (dir, _consumed) = read_block_directory(&buf).unwrap();
        assert_eq!(dir.entries.len(), 2);
        assert_eq!(dir.entries[0].block_offset, 64);
        assert_eq!(dir.entries[0].vector_count, 100);
        assert_eq!(dir.entries[0].dim, 128);
        assert_eq!(dir.entries[1].dtype, 1);
        assert_eq!(dir.entries[1].tier, 1);
    }

    #[test]
    fn vec_block_write_read_round_trip() {
        // Create a block with 4 vectors of dimension 3 (f32)
        let dim = 3u16;
        let count = 4u32;
        // Columnar: all dim_0, then dim_1, then dim_2
        let mut vector_data = Vec::new();
        for d in 0..dim {
            for v in 0..count {
                let val = (d as f32) * 10.0 + (v as f32);
                vector_data.extend_from_slice(&val.to_le_bytes());
            }
        }

        let ids = vec![10, 20, 30, 40];
        let block = VecBlock {
            vector_data: vector_data.clone(),
            ids: ids.clone(),
            dim,
            dtype: 0, // f32
            tier: 0,
        };

        let block_bytes = write_vec_block(&block);
        assert_eq!(block_bytes.len() % 64, 0);

        // Build a payload with a directory entry pointing to offset 0
        let entry = BlockDirEntry {
            block_offset: 0,
            vector_count: count,
            dim,
            dtype: 0,
            tier: 0,
        };
        let decoded = read_vec_block(&block_bytes, &entry).unwrap();
        assert_eq!(decoded.vector_data, vector_data);
        assert_eq!(decoded.ids, ids);
    }

    #[test]
    fn empty_directory() {
        let buf = write_block_directory(&[]);
        let (dir, _) = read_block_directory(&buf).unwrap();
        assert!(dir.entries.is_empty());
    }
}
