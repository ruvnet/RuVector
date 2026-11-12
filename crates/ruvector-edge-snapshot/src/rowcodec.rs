//! Row-block payload carried inside snapshot `VEC_SEG`s.
//!
//! `vec_seg_codec` only holds `u64` ids, while shard rows have string ids and
//! metadata, so snapshot segments carry this private payload inside a
//! standard rvf-wire segment (64-byte header, XXH3-128 content hash, 64-byte
//! padding):
//!
//! ```text
//! "RVER" | ver u8 = 1 | reserved u8 | dim u16 | count u32 |
//!   count × ( id: u16 len + utf8 | has_meta u8 | [meta: u16 len + utf8] | f32 LE × dim )
//! ```

use crate::bytes::{put_f32s, put_str16, Reader};
use crate::types::{check_parts, Row, MAX_ID_BYTES, MAX_METADATA_BYTES};

const MAGIC: &[u8; 4] = b"RVER";
const VERSION: u8 = 1;
/// Row-block header length.
pub(crate) const BLOCK_HEADER: usize = 12;

/// Encoded size of one row.
pub(crate) fn row_size(row: &Row) -> usize {
    2 + row.id.len() + 1 + row.metadata.as_ref().map_or(0, |m| 2 + m.len()) + row.values.len() * 4
}

/// Worst-case encoded size of a row at `dim`.
pub(crate) fn max_row_size(dim: u16) -> usize {
    2 + MAX_ID_BYTES + 1 + 2 + MAX_METADATA_BYTES + usize::from(dim) * 4
}

/// Start a block (count patched by [`seal_block`]).
pub(crate) fn begin_block(out: &mut Vec<u8>, dim: u16) {
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(0);
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
}

/// Append one (already validated) row.
pub(crate) fn put_row(out: &mut Vec<u8>, row: &Row) {
    put_str16(out, &row.id);
    match &row.metadata {
        None => out.push(0),
        Some(m) => {
            out.push(1);
            put_str16(out, m);
        }
    }
    put_f32s(out, &row.values);
}

/// Patch the row count into a block started by [`begin_block`].
pub(crate) fn seal_block(block: &mut [u8], count: u32) {
    block[8..12].copy_from_slice(&count.to_le_bytes());
}

/// Strictly decode a block, appending rows to `out`. Every row is validated
/// against `dim`; more than `max_rows` rows and trailing bytes are refused.
pub(crate) fn decode_block(
    payload: &[u8],
    dim: u16,
    max_rows: u64,
    out: &mut Vec<Row>,
) -> Result<u32, &'static str> {
    let mut r = Reader::new(payload);
    if r.take(4) != Some(MAGIC.as_slice()) {
        return Err("row block magic");
    }
    if r.u8() != Some(VERSION) || r.u8() != Some(0) {
        return Err("row block version");
    }
    if r.u16() != Some(dim) {
        return Err("row block dim");
    }
    let count = r.u32().ok_or("row block count")?;
    if u64::from(count) > max_rows {
        return Err("row block count exceeds chunk");
    }
    // Each row needs at least 3 + 4·dim bytes: bound the count before looping.
    let min_row = 3 + usize::from(dim) * 4;
    if (count as usize).saturating_mul(min_row) > r.remaining() {
        return Err("row block count");
    }
    for _ in 0..count {
        let id = r.str16(MAX_ID_BYTES).ok_or("row id")?;
        let metadata = match r.u8().ok_or("row meta flag")? {
            0 => None,
            1 => Some(r.str16(MAX_METADATA_BYTES).ok_or("row metadata")?),
            _ => return Err("row meta flag"),
        };
        let mut values = Vec::with_capacity(usize::from(dim));
        r.f32s_into(usize::from(dim), &mut values)
            .ok_or("row values")?;
        check_parts(id, &values, metadata, dim)?;
        out.push(Row {
            id: id.to_string(),
            values,
            metadata: metadata.map(str::to_string),
        });
    }
    if r.remaining() != 0 {
        return Err("row block trailing bytes");
    }
    Ok(count)
}
