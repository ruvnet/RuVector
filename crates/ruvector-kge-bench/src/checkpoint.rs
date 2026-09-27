//! Resumable checkpoints (plan M3): tables + optimizer state + epoch + RNG
//! state + the example permutation + run history, in one file written
//! atomically (`canon::atomic_write`: temp + fsync + rename).
//!
//! Layout (all little-endian):
//! ```text
//! b"KGEBCK01" | u64 header_len | header JSON
//! | entities f32[E·D] | relations f32[R·D] | order u32[n]
//! | for each optimizer table: ids u32[k] | data f32[k·D]
//! | sha256(everything above) [32 bytes]
//! ```
//! Reading verifies the trailer before parsing anything else, then refuses a
//! checkpoint whose config hash, dataset or vocabulary differs from the run.

use crate::canon::{atomic_write, sha256_hex};
use crate::config::RunConfig;
use crate::metrics::RunProgress;
use anyhow::{bail, Context, Result};
use ruvector_kge::train::{OptimState, RowsState, TrainState};
use ruvector_kge::Tables;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

const MAGIC: &[u8; 8] = b"KGEBCK01";
pub const CHECKPOINT_FILE: &str = "checkpoint.bin";

/// The JSON header: identity checks, shapes and run history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub config_hash: String,
    pub run: RunConfig,
    pub dataset: String,
    pub splits_hash: String,
    pub entity_vocab_hash: String,
    pub relation_vocab_hash: String,
    pub threads: usize,
    pub num_entities: usize,
    pub num_relation_rows: usize,
    pub dims: usize,
    pub next_epoch: usize,
    pub sample_rng: u64,
    pub optim_t: u64,
    pub order_len: usize,
    /// Row count per optimizer table (`[ent, rel]` for Adagrad).
    pub optim_rows: Vec<usize>,
    pub progress: RunProgress,
}

/// A decoded checkpoint.
pub struct Checkpoint {
    pub header: Header,
    pub tables: Tables,
    pub state: TrainState,
}

fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}
fn put_u32s(out: &mut Vec<u8>, v: &[u32]) {
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Serialise and atomically write a checkpoint to `dir/checkpoint.bin`.
/// `header`'s shape fields are filled from `tables` / `state` here.
pub fn write(dir: &Path, mut header: Header, tables: &Tables, state: &TrainState) -> Result<()> {
    header.num_entities = tables.num_entities();
    header.num_relation_rows = tables.num_relations();
    header.dims = tables.dims();
    header.next_epoch = state.next_epoch;
    header.sample_rng = state.sample_rng;
    header.optim_t = state.optimizer.t;
    header.order_len = state.order.len();
    header.optim_rows = state.optimizer.rows.iter().map(|r| r.ids.len()).collect();
    let json = serde_json::to_vec(&header)?;
    let body_len = 16
        + json.len()
        + (tables.entities_raw().len() + tables.relations_raw().len() + state.order.len()) * 4
        + state
            .optimizer
            .rows
            .iter()
            .map(|r| (r.ids.len() + r.data.len()) * 4)
            .sum::<usize>();
    let mut out = Vec::with_capacity(body_len + 32);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    put_f32s(&mut out, tables.entities_raw());
    put_f32s(&mut out, tables.relations_raw());
    put_u32s(&mut out, &state.order);
    for r in &state.optimizer.rows {
        put_u32s(&mut out, &r.ids);
        put_f32s(&mut out, &r.data);
    }
    let digest = Sha256::digest(&out);
    out.extend_from_slice(&digest);
    atomic_write(&dir.join(CHECKPOINT_FILE), &out)
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .context("checkpoint truncated")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self
            .take(n.checked_mul(4).context("size overflow")?)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        Ok(self
            .take(n.checked_mul(4).context("size overflow")?)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }
}

/// Read and verify `dir/checkpoint.bin`.
pub fn read(dir: &Path) -> Result<Checkpoint> {
    let path = dir.join(CHECKPOINT_FILE);
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < MAGIC.len() + 8 + 32 || &bytes[..8] != MAGIC {
        bail!("{} is not a kge-bench checkpoint", path.display());
    }
    let (body, trailer) = bytes.split_at(bytes.len() - 32);
    if Sha256::digest(body).as_slice() != trailer {
        bail!(
            "{}: checksum mismatch (corrupt or torn checkpoint); refusing to resume",
            path.display()
        );
    }
    let mut c = Cursor { b: body, at: 8 };
    let hlen = u64::from_le_bytes(c.take(8)?.try_into()?) as usize;
    let header: Header = serde_json::from_slice(c.take(hlen)?).context("checkpoint header")?;
    let (ne, nr, d) = (header.num_entities, header.num_relation_rows, header.dims);
    let mut tables = Tables::try_new(ne, nr, d, 0, None)?;
    tables.entities_raw_mut().copy_from_slice(&c.f32s(ne * d)?);
    tables.relations_raw_mut().copy_from_slice(&c.f32s(nr * d)?);
    let order = c.u32s(header.order_len)?;
    let mut rows = Vec::with_capacity(header.optim_rows.len());
    for &k in &header.optim_rows {
        let ids = c.u32s(k)?;
        let data = c.f32s(k.checked_mul(d).context("size overflow")?)?;
        rows.push(RowsState { ids, data });
    }
    if c.at != body.len() {
        bail!("{}: trailing bytes after the last section", path.display());
    }
    let state = TrainState {
        next_epoch: header.next_epoch,
        order,
        sample_rng: header.sample_rng,
        optimizer: OptimState {
            t: header.optim_t,
            rows,
        },
    };
    Ok(Checkpoint {
        header,
        tables,
        state,
    })
}

/// sha256 of the checkpoint file on disk (recorded, e.g., for auditing).
pub fn file_sha256(dir: &Path) -> Result<String> {
    Ok(sha256_hex(std::fs::read(dir.join(CHECKPOINT_FILE))?))
}
