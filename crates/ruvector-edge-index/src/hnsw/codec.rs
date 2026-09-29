//! HNSW payload codec.
//!
//! Payload: quant params · graph params (incl. `iid_base`) · `node_count`,
//! `max_slot`, entry, top level, live · levels · tombstone words · codes ·
//! layer-0 links · upper row count and rows in slot order. Offsets into
//! `upper` are not stored; decode recomputes them as a prefix sum of
//! levels, so the stored form is canonical whatever order nodes were
//! inserted in.
//!
//! Decode validates everything a traversal relies on, so a payload that
//! passes can never make search or insert index out of bounds: every
//! link is `< n`, not the node itself, unique within its row, `NONE` only
//! as tail padding, and — on layer `l ≥ 1` — points at a node whose level
//! is `≥ l`.

use super::{HnswIndex, HnswParams, ABSENT, NONE};
use crate::bytes::{Reader, Sink};
use crate::error::{DecodeError, EmitError, EncodeError};
use crate::persist::{
    decode, decode_slice, encode, encode_all, EncodedIndex, IndexChunk, IndexDigest, IndexKind,
};

use crate::quant::QuantParams;

const LINK: DecodeError = DecodeError::Malformed("link row");

/// Row check shared by encode and decode (see the module docs).
fn row_ok(row: &[u32], me: u32, layer: usize, levels: &[u8], seen: &mut Vec<u32>) -> bool {
    let len = row.iter().position(|&x| x == NONE).unwrap_or(row.len());
    if row[len..].iter().any(|&x| x != NONE) {
        return false;
    }
    let ok = row[..len].iter().all(|&x| {
        x != me
            && levels
                .get(x as usize)
                .is_some_and(|&l| l != ABSENT && l as usize >= layer)
    });
    if !ok {
        return false;
    }
    seen.clear();
    seen.extend_from_slice(&row[..len]);
    seen.sort_unstable();
    seen.windows(2).all(|w| w[0] != w[1])
}

impl HnswIndex {
    /// Every link row of every present node is canonical.
    fn rows_ok(&self) -> bool {
        let mut seen = Vec::new();
        (0..self.node_count()).all(|i| {
            let l = self.levels[i as usize];
            l == ABSENT
                || (0..=l as usize)
                    .all(|layer| row_ok(self.links(i, layer), i, layer, &self.levels, &mut seen))
        })
    }

    fn check_dense(&self) -> Result<(), EncodeError> {
        let n = self.node_count();
        if self.present != n {
            let first = self.levels.iter().position(|&l| l == ABSENT).unwrap_or(0) as u32;
            return Err(EncodeError::GappedIid {
                first_missing: self.iid(first),
                missing: n - self.present,
            });
        }
        if !self.rows_ok() {
            return Err(EncodeError::Invariant("link row"));
        }
        Ok(())
    }

    fn write_payload(&self, w: &mut dyn Sink) {
        let p = &self.params;
        self.quant.write(w);
        w.put_u16(p.m);
        w.put_u16(p.m0);
        w.put_u16(p.ef_construction);
        w.put_u8(p.max_level);
        w.put_u32(p.max_slots);
        w.put_u32(p.iid_base);
        let n = self.node_count();
        w.put_u32(n);
        w.put_u32(n.wrapping_sub(1)); // max slot; NONE when empty
        w.put_u32(self.entry);
        w.put_u8(self.top);
        w.put_u32(self.live);
        w.put(&self.levels);
        w.put_u64s(&self.deleted);
        w.put(&self.codes);
        w.put_u32s(&self.links0);
        let rows: u32 = self.levels.iter().map(|&l| u32::from(l)).sum();
        w.put_u32(rows);
        let m = p.m as usize;
        for (i, &l) in self.levels.iter().enumerate() {
            if l > 0 {
                let s = self.upper_off[i] as usize * m;
                w.put_u32s(&self.upper[s..s + l as usize * m]);
            }
        }
    }

    /// Stream the index into ≤ `max_chunk`-byte `index_chunks` rows
    /// stamped `epoch`: `emit` receives each row as soon as it fills (e.g.
    /// one `INSERT`), so peak extra memory is one chunk. Record the
    /// returned `sha256` as `meta.index_sha256`.
    ///
    /// Fails loudly with [`EncodeError::GappedIid`] unless every slot
    /// `0..node_count` holds a node (ADR §6.1 `node_count == max_iid + 1`).
    pub fn encode_into<E>(
        &self,
        epoch: u64,
        max_chunk: usize,
        emit: &mut dyn FnMut(IndexChunk) -> Result<(), E>,
    ) -> Result<IndexDigest, EmitError<E>> {
        self.check_dense().map_err(EmitError::Encode)?;
        encode(
            epoch,
            IndexKind::Hnsw,
            max_chunk,
            &|w| self.write_payload(w),
            emit,
        )
    }

    /// [`Self::encode_into`] collected in memory.
    pub fn to_chunks(&self, epoch: u64, max_chunk: usize) -> Result<EncodedIndex, EncodeError> {
        self.check_dense()?;
        encode_all(epoch, IndexKind::Hnsw, max_chunk, &|w| {
            self.write_payload(w)
        })
    }

    /// Decode and fully validate stored chunks. Any error means "do not use
    /// this epoch": load the previous one or replay the op log. Pass the
    /// recorded `meta.index_sha256` as `expected_sha256` to pin the epoch.
    pub fn from_chunks(
        chunks: &[IndexChunk],
        expected_sha256: Option<&[u8; 32]>,
    ) -> Result<Self, DecodeError> {
        decode_slice(chunks, IndexKind::Hnsw, expected_sha256, Self::parse)
    }

    /// Streaming decode: rows are pulled one at a time (e.g. from a
    /// `cursor.raw()` over `index_chunks ORDER BY part`), so only one row
    /// is held besides the index being built. `max_payload_bytes` bounds
    /// the declared payload before anything is allocated (the store passes
    /// its shard memory cap).
    pub fn from_chunk_iter<I: IntoIterator<Item = IndexChunk>>(
        chunks: I,
        expected_sha256: Option<&[u8; 32]>,
        max_payload_bytes: u64,
    ) -> Result<Self, DecodeError> {
        let mut it = chunks.into_iter();
        decode(
            &mut it,
            IndexKind::Hnsw,
            expected_sha256,
            max_payload_bytes,
            Self::parse,
        )
    }

    fn parse(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let quant = QuantParams::read(r)?;
        let params = HnswParams {
            m: r.u16()?,
            m0: r.u16()?,
            ef_construction: r.u16()?,
            max_level: r.u8()?,
            max_slots: r.u32()?,
            iid_base: r.u32()?,
        };
        params.check(quant.dim()).map_err(DecodeError::Malformed)?;
        let n = r.u32()?;
        let max_slot = r.u32()?;
        let (entry, top, live) = (r.u32()?, r.u8()?, r.u32()?);
        if n > params.max_slots || max_slot != n.wrapping_sub(1) {
            return Err(DecodeError::Malformed("node_count != max_iid + 1"));
        }
        let nu = n as usize;
        let levels = r.bytes_vec(nu, 1)?;
        if levels.iter().any(|&l| l > params.max_level) {
            return Err(DecodeError::Malformed("level (gapped iid or out of range)"));
        }
        let deleted = r.u64_vec(nu.div_ceil(64))?;
        let codes = r.bytes_vec(nu, quant.dim())?;
        let links0 = r.u32_vec(nu, params.m0 as usize)?;
        let rows = r.u32()? as usize;
        let sum = levels
            .iter()
            .try_fold(0usize, |a, &l| a.checked_add(l as usize));
        if sum != Some(rows) {
            return Err(DecodeError::Malformed("upper row count"));
        }
        let upper = r.u32_vec(rows, params.m as usize)?;
        let mut upper_off = Vec::with_capacity(nu);
        let mut off = 0u32;
        for &l in &levels {
            upper_off.push(if l > 0 { off } else { NONE });
            off = off.checked_add(u32::from(l)).ok_or(LINK)?;
        }
        let dead: u32 = deleted.iter().map(|w| w.count_ones()).sum();
        let tail_ok = n % 64 == 0 || deleted.last().map_or(true, |w| w >> (n % 64) == 0);
        if !tail_ok || dead > n || live != n - dead {
            return Err(DecodeError::Malformed("tombstone bitmap"));
        }
        let entry_ok = match levels.iter().copied().max() {
            None => entry == NONE && top == 0,
            Some(t) => entry < n && top == t && levels[entry as usize] == t,
        };
        if !entry_ok {
            return Err(DecodeError::Malformed("entry point"));
        }
        let idx = Self {
            params,
            quant,
            codes,
            levels,
            links0,
            upper_off,
            upper,
            deleted,
            entry,
            top,
            present: n,
            live,
        };
        if !idx.rows_ok() {
            return Err(LINK);
        }
        Ok(idx)
    }
}
