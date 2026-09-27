//! Tables-only weight export (ADR-007 §5 "Weights", plan M3 "Export").
//!
//! A directory holding `weights.bin` (entity rows then relation rows, f32
//! little-endian, row-major) and `manifest.json` (scorer, complex_rank,
//! reciprocal, seed, dataset pins, split hash, the entity/relation
//! vocabulary-order hashes, weights sha256, config hash). **No triples and no
//! labels** — rows are identified only by id, and ids are the sorted-label
//! order whose hash the manifest carries, so a consumer with the dataset can
//! rebuild the mapping and a consumer without it learns nothing else.

use crate::canon::{atomic_write, sha256_hex};
use crate::datasets::Dataset;
use anyhow::{bail, Context, Result};
use ruvector_kge::Tables;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const EXPORT_SCHEMA: &str = "ruvector-kge-bench/tables@1";

/// `manifest.json` of an export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub scorer: String,
    pub complex_rank: usize,
    pub dims: usize,
    pub reciprocal: bool,
    pub num_entities: usize,
    /// Relation **rows** (`2·R` under `reciprocal`).
    pub num_relation_rows: usize,
    pub seed: u64,
    pub dataset: String,
    pub dataset_file_hashes: BTreeMap<String, String>,
    pub splits_hash: String,
    /// Per-split hashes (train, valid, transfer, test).
    pub split_hashes: BTreeMap<String, String>,
    pub entity_vocab_hash: String,
    pub relation_vocab_hash: String,
    pub config_hash: String,
    /// The epoch whose tables these are (0-based index of the last epoch run).
    pub epoch: usize,
    pub weights_sha256: String,
}

/// What an export records besides the tables themselves.
pub struct ExportMeta<'a> {
    pub dataset: &'a Dataset,
    pub seed: u64,
    pub config_hash: &'a str,
    pub epoch: usize,
}

fn weight_bytes(t: &Tables) -> Vec<u8> {
    let mut out = Vec::with_capacity((t.entities_raw().len() + t.relations_raw().len()) * 4);
    for &x in t.entities_raw().iter().chain(t.relations_raw()) {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Write `tables` to `dir` (atomically, weights before manifest). Returns the
/// manifest.
pub fn export_tables(dir: &Path, tables: &Tables, meta: &ExportMeta) -> Result<Manifest> {
    let bytes = weight_bytes(tables);
    let m = Manifest {
        schema: EXPORT_SCHEMA.into(),
        scorer: "complex".into(),
        complex_rank: tables.dims() / 2,
        dims: tables.dims(),
        reciprocal: true,
        num_entities: tables.num_entities(),
        num_relation_rows: tables.num_relations(),
        seed: meta.seed,
        dataset: meta.dataset.name.clone(),
        dataset_file_hashes: meta.dataset.file_hashes.clone(),
        splits_hash: meta.dataset.splits_hash.clone(),
        split_hashes: meta.dataset.split_hashes.clone(),
        entity_vocab_hash: meta.dataset.entity_vocab_hash.clone(),
        relation_vocab_hash: meta.dataset.relation_vocab_hash.clone(),
        config_hash: meta.config_hash.into(),
        epoch: meta.epoch,
        weights_sha256: sha256_hex(&bytes),
    };
    atomic_write(&dir.join("weights.bin"), &bytes)?;
    atomic_write(
        &dir.join("manifest.json"),
        serde_json::to_string_pretty(&m)?.as_bytes(),
    )?;
    Ok(m)
}

/// Load an export, verifying the weights hash and shapes. When `dataset` is
/// given, the manifest must name it and match its pins and vocabulary hashes.
pub fn load_tables(dir: &Path, dataset: Option<&Dataset>) -> Result<(Tables, Manifest)> {
    let m: Manifest = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json"))
            .with_context(|| format!("read {}/manifest.json", dir.display()))?,
    )
    .context("parse manifest.json")?;
    if m.schema != EXPORT_SCHEMA {
        bail!("unsupported export schema '{}'", m.schema);
    }
    let bytes = std::fs::read(dir.join("weights.bin"))?;
    let got = sha256_hex(&bytes);
    if got != m.weights_sha256 {
        bail!("weights.bin sha256 {got} != manifest {}", m.weights_sha256);
    }
    if m.dims == 0 || m.dims != 2 * m.complex_rank {
        bail!("manifest dims/complex_rank inconsistent");
    }
    let (ne, nr, d) = (m.num_entities, m.num_relation_rows, m.dims);
    if bytes.len() != (ne + nr) * d * 4 {
        bail!(
            "weights.bin has {} bytes, manifest shape needs {}",
            bytes.len(),
            (ne + nr) * d * 4
        );
    }
    if let Some(ds) = dataset {
        if ds.name != m.dataset
            || ds.file_hashes != m.dataset_file_hashes
            || ds.splits_hash != m.splits_hash
            || ds.split_hashes != m.split_hashes
            || ds.entity_vocab_hash != m.entity_vocab_hash
            || ds.relation_vocab_hash != m.relation_vocab_hash
            || ds.num_entities != ne
        {
            bail!("export does not belong to dataset '{}' (pins, split hash or vocabulary order differ)", ds.name);
        }
    }
    let mut t = Tables::try_new(ne, nr, d, 0, None)?;
    let floats: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    t.entities_raw_mut().copy_from_slice(&floats[..ne * d]);
    t.relations_raw_mut().copy_from_slice(&floats[ne * d..]);
    Ok((t, m))
}
