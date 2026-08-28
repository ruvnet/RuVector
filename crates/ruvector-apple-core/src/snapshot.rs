use std::collections::HashSet;

use crate::{DistanceMetric, ExactVectorIndex, IndexConfig, IndexError};

const MAGIC: [u8; 8] = *b"RVAPIDX\0";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 48;
const CHECKSUM_START: usize = 40;
const CHECKSUM_END: usize = 44;

/// Maximum encoded snapshot size accepted by this crate.
pub const MAX_SNAPSHOT_BYTES: u64 = 96 * 1024 * 1024;

pub(crate) fn encode(index: &ExactVectorIndex) -> Result<Vec<u8>, IndexError> {
    let config = index.config();
    let count = u32::try_from(index.len())
        .map_err(|_| IndexError::CorruptSnapshot("entry count exceeds u32"))?;
    let entry_len = 8_u64 + u64::from(config.dimensions) * 4;
    let payload_len =
        u64::from(count)
            .checked_mul(entry_len)
            .ok_or(IndexError::SnapshotTooLarge {
                actual: u64::MAX,
                maximum: MAX_SNAPSHOT_BYTES,
            })?;
    let total_len =
        (HEADER_LEN as u64)
            .checked_add(payload_len)
            .ok_or(IndexError::SnapshotTooLarge {
                actual: u64::MAX,
                maximum: MAX_SNAPSHOT_BYTES,
            })?;
    if total_len > MAX_SNAPSHOT_BYTES {
        return Err(IndexError::SnapshotTooLarge {
            actual: total_len,
            maximum: MAX_SNAPSHOT_BYTES,
        });
    }

    let mut bytes = Vec::with_capacity(total_len as usize);
    bytes.extend_from_slice(&MAGIC);
    push_u32(&mut bytes, FORMAT_VERSION);
    push_u32(&mut bytes, crate::RVECTOR_APPLE_CORE_ABI_VERSION);
    push_u32(&mut bytes, config.metric as u32);
    push_u32(&mut bytes, config.dimensions);
    push_u32(&mut bytes, config.capacity);
    push_u32(&mut bytes, count);
    push_u64(&mut bytes, payload_len);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);

    let mut entries: Vec<_> = index.entries().iter().collect();
    entries.sort_unstable_by_key(|entry| entry.id);
    for entry in entries {
        push_u64(&mut bytes, entry.id);
        for value in &entry.vector {
            push_u32(&mut bytes, value.to_bits());
        }
    }
    let checksum = checksum(&bytes);
    bytes[CHECKSUM_START..CHECKSUM_END].copy_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

pub(crate) fn decode(bytes: &[u8]) -> Result<ExactVectorIndex, IndexError> {
    let byte_len = bytes.len() as u64;
    if byte_len > MAX_SNAPSHOT_BYTES {
        return Err(IndexError::SnapshotTooLarge {
            actual: byte_len,
            maximum: MAX_SNAPSHOT_BYTES,
        });
    }
    if bytes.len() < HEADER_LEN {
        return Err(IndexError::CorruptSnapshot("truncated header"));
    }
    if bytes[..MAGIC.len()] != MAGIC {
        return Err(IndexError::CorruptSnapshot("invalid magic"));
    }
    let stored_checksum = read_u32(bytes, CHECKSUM_START)?;
    if checksum(bytes) != stored_checksum {
        return Err(IndexError::CorruptSnapshot("checksum mismatch"));
    }
    if read_u32(bytes, 8)? != FORMAT_VERSION {
        return Err(IndexError::CorruptSnapshot("unsupported format version"));
    }
    if read_u32(bytes, 12)? != crate::RVECTOR_APPLE_CORE_ABI_VERSION {
        return Err(IndexError::CorruptSnapshot("unsupported ABI version"));
    }
    let metric_tag = read_u32(bytes, 16)?;
    let metric = DistanceMetric::try_from(metric_tag)
        .map_err(|_| IndexError::CorruptSnapshot("invalid metric"))?;
    let config = IndexConfig {
        metric,
        dimensions: read_u32(bytes, 20)?,
        capacity: read_u32(bytes, 24)?,
    }
    .validate()
    .map_err(|_| IndexError::CorruptSnapshot("invalid index configuration"))?;
    let count = read_u32(bytes, 28)?;
    if count > config.capacity {
        return Err(IndexError::CorruptSnapshot("entry count exceeds capacity"));
    }
    let payload_len = read_u64(bytes, 32)?;
    if read_u32(bytes, 44)? != 0 {
        return Err(IndexError::CorruptSnapshot(
            "reserved header field is non-zero",
        ));
    }

    let entry_len = 8_u64 + u64::from(config.dimensions) * 4;
    let expected_payload = u64::from(count)
        .checked_mul(entry_len)
        .ok_or(IndexError::CorruptSnapshot("payload length overflow"))?;
    if payload_len != expected_payload {
        return Err(IndexError::CorruptSnapshot("payload length mismatch"));
    }
    let expected_total = (HEADER_LEN as u64)
        .checked_add(payload_len)
        .ok_or(IndexError::CorruptSnapshot("total length overflow"))?;
    if expected_total != byte_len {
        return Err(IndexError::CorruptSnapshot("trailing or truncated bytes"));
    }

    let mut index = ExactVectorIndex::new(config)
        .map_err(|_| IndexError::CorruptSnapshot("invalid index configuration"))?;
    let mut ids = HashSet::with_capacity(count as usize);
    let mut offset = HEADER_LEN;
    for _ in 0..count {
        let id = read_u64(bytes, offset)?;
        offset += 8;
        if !ids.insert(id) {
            return Err(IndexError::CorruptSnapshot("duplicate vector ID"));
        }
        let mut vector = Vec::with_capacity(config.dimensions as usize);
        for _ in 0..config.dimensions {
            vector.push(f32::from_bits(read_u32(bytes, offset)?));
            offset += 4;
        }
        index
            .upsert(id, &vector)
            .map_err(|_| IndexError::CorruptSnapshot("invalid vector payload"))?;
    }
    Ok(index)
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, IndexError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(IndexError::CorruptSnapshot("truncated integer"))?;
    Ok(u32::from_le_bytes(
        value
            .try_into()
            .map_err(|_| IndexError::CorruptSnapshot("invalid u32"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, IndexError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(IndexError::CorruptSnapshot("truncated integer"))?;
    Ok(u64::from_le_bytes(
        value
            .try_into()
            .map_err(|_| IndexError::CorruptSnapshot("invalid u64"))?,
    ))
}

fn checksum(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for (index, byte) in bytes.iter().enumerate() {
        let byte = if (CHECKSUM_START..CHECKSUM_END).contains(&index) {
            0
        } else {
            *byte
        };
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite_checksum(bytes: &mut [u8]) {
        bytes[CHECKSUM_START..CHECKSUM_END].fill(0);
        let updated = checksum(bytes);
        bytes[CHECKSUM_START..CHECKSUM_END].copy_from_slice(&updated.to_le_bytes());
    }

    #[test]
    fn structural_validation_rejects_duplicate_ids_with_valid_checksum() {
        let mut index = ExactVectorIndex::new(IndexConfig {
            dimensions: 2,
            capacity: 2,
            metric: DistanceMetric::Dot,
        })
        .unwrap();
        index.upsert(1, &[1.0, 0.0]).unwrap();
        index.upsert(2, &[0.0, 1.0]).unwrap();
        let mut bytes = encode(&index).unwrap();
        let first_id = bytes[HEADER_LEN..HEADER_LEN + 8].to_vec();
        let second_id_offset = HEADER_LEN + 8 + 2 * 4;
        bytes[second_id_offset..second_id_offset + 8].copy_from_slice(&first_id);
        rewrite_checksum(&mut bytes);

        assert!(matches!(
            decode(&bytes),
            Err(IndexError::CorruptSnapshot("duplicate vector ID"))
        ));
    }

    #[test]
    fn structural_validation_rejects_nan_with_valid_checksum() {
        let mut index = ExactVectorIndex::new(IndexConfig {
            dimensions: 1,
            capacity: 1,
            metric: DistanceMetric::Dot,
        })
        .unwrap();
        index.upsert(7, &[1.0]).unwrap();
        let mut bytes = encode(&index).unwrap();
        bytes[HEADER_LEN + 8..HEADER_LEN + 12].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        rewrite_checksum(&mut bytes);

        assert!(matches!(
            decode(&bytes),
            Err(IndexError::CorruptSnapshot("invalid vector payload"))
        ));
    }
}
