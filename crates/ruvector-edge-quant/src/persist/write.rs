//! Persist v2 writer: header frame, keys/norms/codes data frames, footer.

use super::format::*;
use crate::error::{QuantError, Result};
use crate::shard::QuantShard;
use std::io::Write;

/// Little-endian bytes of a section in one bulk pass (a per-byte iterator
/// chain costs several ns/byte in a size-optimised wasm build).
fn le_bytes<const W: usize, T: Copy>(xs: &[T], f: impl Fn(T) -> [u8; W]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * W);
    for &x in xs {
        out.extend_from_slice(&f(x));
    }
    out
}

/// Serialise one section's bytes into frames of `cap` payload bytes.
fn push_section(
    frames: &mut Vec<Vec<u8>>,
    crcs: &mut Crc32,
    section: u8,
    bytes: &[u8],
    total: usize,
    cap: usize,
) {
    debug_assert_eq!(bytes.len(), total);
    let mut left = total;
    let mut off = 0;
    while left > 0 {
        let len = left.min(cap);
        let idx = frames.len() as u32 - 1; // frames[0] is the header
        let mut f = Vec::with_capacity(FRAME_PREFIX_LEN + len);
        f.extend_from_slice(&idx.to_le_bytes());
        f.push(section);
        f.extend_from_slice(&[0, 0, 0]);
        f.extend_from_slice(&(len as u32).to_le_bytes());
        f.extend_from_slice(&[0; 4]); // crc placeholder
        f.extend_from_slice(&bytes[off..off + len]);
        off += len;
        let mut c = Crc32::new();
        c.update(&f[0..12]);
        c.update(&f[FRAME_PREFIX_LEN..]);
        let crc = c.finish();
        f[12..16].copy_from_slice(&crc.to_le_bytes());
        crcs.update(&crc.to_le_bytes());
        frames.push(f);
        left -= len;
    }
}

/// Encode `shard` as persist v2 frames: `[header, data…, footer]`, each at
/// most `max_frame_bytes` (`64..=1 MiB`). A DO host stores one frame per
/// storage row; [`save_to`] writes their concatenation.
pub fn save_frames(shard: &QuantShard, max_frame_bytes: usize) -> Result<Vec<Vec<u8>>> {
    if !(MIN_FRAME_BYTES..=MAX_FRAME_BYTES).contains(&max_frame_bytes) {
        return Err(QuantError::InvalidConfig(
            "max_frame_bytes must be 64..=1 MiB",
        ));
    }
    let cfg = shard.config();
    let n = shard.len() as u64;
    let cap = payload_cap(max_frame_bytes);
    let lens = section_lens(n, cfg.dim).ok_or(QuantError::InvalidConfig("row count"))?;
    let frames_n = frame_count(&lens, cap);
    let frames_u32 =
        u32::try_from(frames_n).map_err(|_| QuantError::InvalidConfig("too many frames"))?;
    let header = Header {
        dim: cfg.dim as u32,
        n,
        rotation: cfg.rotation,
        metric: cfg.metric,
        max_frame_bytes: max_frame_bytes as u32,
        seed: cfg.seed,
        fingerprint: rotation_fingerprint(shard.rotation()),
        frames: frames_u32,
        payload_bytes: lens.iter().sum(),
    };
    let mut frames = Vec::with_capacity(frames_n as usize + 2);
    frames.push(header.encode().to_vec());
    let mut crcs = Crc32::new();
    let keys = le_bytes(shard.keys(), u64::to_le_bytes);
    push_section(
        &mut frames,
        &mut crcs,
        SECTION_KEYS,
        &keys,
        lens[0] as usize,
        cap,
    );
    let norms = le_bytes(shard.norms(), f32::to_le_bytes);
    push_section(
        &mut frames,
        &mut crcs,
        SECTION_NORMS,
        &norms,
        lens[1] as usize,
        cap,
    );
    let codes = le_bytes(shard.packed(), u64::to_le_bytes);
    push_section(
        &mut frames,
        &mut crcs,
        SECTION_CODES,
        &codes,
        lens[2] as usize,
        cap,
    );
    debug_assert_eq!(frames.len() as u64, frames_n + 1);
    let mut footer = Vec::with_capacity(FOOTER_LEN);
    footer.extend_from_slice(FOOTER_MAGIC);
    footer.extend_from_slice(&frames_u32.to_le_bytes());
    footer.extend_from_slice(&crcs.finish().to_le_bytes());
    frames.push(footer);
    Ok(frames)
}

/// Write the concatenated frames to `w`.
pub fn save_to<W: Write>(shard: &QuantShard, w: &mut W, max_frame_bytes: usize) -> Result<()> {
    for f in save_frames(shard, max_frame_bytes)? {
        w.write_all(&f).map_err(|e| QuantError::Io(e.to_string()))?;
    }
    Ok(())
}
