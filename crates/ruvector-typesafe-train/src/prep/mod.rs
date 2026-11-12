//! `openjev prep`: build the train/val data directory (plan Step 1.2 layout)
//! from the frozen tickets fixture and the pinned public datasets.
//!
//! Also exposes [`heldout_hashes`], recomputed from the *sources* (not from an
//! exported file) so the trainer's Assertion A is independent of whichever
//! tool produced `train.jsonl`.

pub mod public;
pub mod tickets;

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::json;

use crate::data::{write_hashes, write_jsonl, Labels, Row, TICKETS};
use crate::leakage::{assert_no_leakage, colliding_descriptions, drop_heldout, LeakageReport};
use crate::norm::{sha256_hex, sha256_norm};
use crate::pins::{self, ensure};

/// Where the inputs live.
pub struct Sources {
    /// `npm/packages/typesafe/bench/fixtures/tickets-decisions.json`.
    pub tickets_fixture: PathBuf,
    /// Cache holding the pinned public dataset files.
    pub cache: PathBuf,
    pub allow_download: bool,
    /// Optional dataset subset (default: all four).
    pub datasets: Vec<String>,
}

pub struct Loaded {
    pub tickets: tickets::Tickets,
    pub public: Vec<public::Public>,
}

pub fn load_sources(src: &Sources) -> Result<Loaded> {
    let tickets = tickets::load(&src.tickets_fixture)?;
    let want = |n: &str| src.datasets.is_empty() || src.datasets.iter().any(|d| d == n);
    let mut public = Vec::new();
    let read = |p: &pins::Pin| -> Result<Vec<u8>> {
        let path = ensure(&src.cache, p, src.allow_download)?;
        Ok(fs::read(path)?)
    };
    if want(crate::data::BANKING77) {
        public.push(public::banking77(
            &read(&pins::BANKING77_TRAIN)?,
            &read(&pins::BANKING77_TEST)?,
            &read(&pins::BANKING77_CATEGORIES)?,
        )?);
    }
    if want(crate::data::CLINC150) {
        public.push(public::clinc150(&read(&pins::CLINC150)?)?);
    }
    if want(crate::data::HWU64) {
        public.push(public::hwu64(&read(&pins::HWU64)?)?);
    }
    Ok(Loaded { tickets, public })
}

/// Held-out hash set (union) + per-dataset held-out row counts.
/// Always includes all three public test splits when their files are cached,
/// even if a dataset is excluded from training.
pub fn heldout_hashes(src: &Sources) -> Result<(HashSet<String>, BTreeMap<String, usize>)> {
    let all = Sources {
        tickets_fixture: src.tickets_fixture.clone(),
        cache: src.cache.clone(),
        allow_download: src.allow_download,
        datasets: Vec::new(),
    };
    let l = load_sources(&all)?;
    let mut set = HashSet::new();
    let mut counts = BTreeMap::new();
    counts.insert(TICKETS.to_string(), l.tickets.heldout_texts.len());
    set.extend(l.tickets.heldout_texts.iter().map(|t| sha256_norm(t)));
    for p in &l.public {
        counts.insert(p.name.to_string(), p.heldout_texts.len());
        set.extend(p.heldout_texts.iter().map(|t| sha256_norm(t)));
    }
    Ok((set, counts))
}

pub struct PrepOutput {
    pub train: Vec<Row>,
    pub val: Vec<Row>,
    pub labels: Labels,
    pub report: LeakageReport,
}

/// Build rows, drop held-out duplicates, run Assertion A, write the data dir.
pub fn run(src: &Sources, out: &Path) -> Result<PrepOutput> {
    let (heldout, heldout_counts) = heldout_hashes(src)?;
    let l = load_sources(src)?;
    let mut labels: Labels = BTreeMap::new();
    labels.insert(TICKETS.into(), l.tickets.criteria.clone());
    let mut train = l.tickets.train.clone();
    let mut val = l.tickets.val.clone();
    for p in &l.public {
        labels.insert(p.name.into(), p.criteria.clone());
        train.extend(p.train.iter().cloned());
        val.extend(p.val.iter().cloned());
    }
    let (train, mut dropped) = drop_heldout(train, &heldout);
    let (val, dropped_val) = drop_heldout(val, &heldout);
    for (k, v) in dropped_val {
        *dropped.entry(k).or_default() += v;
    }
    let mut report = assert_no_leakage(&train, &val, &heldout, &heldout_counts, &dropped)
        .map_err(anyhow::Error::new)?;
    report.dropped_descriptions = colliding_descriptions(&labels, &heldout);

    fs::create_dir_all(out).with_context(|| format!("mkdir {}", out.display()))?;
    write_jsonl(&out.join("train.jsonl"), &train)?;
    write_jsonl(&out.join("val.jsonl"), &val)?;
    write_hashes(&out.join("heldout-hashes.txt"), heldout.iter().cloned())?;
    fs::write(out.join("labels.json"), serde_json::to_vec_pretty(&labels)?)?;
    fs::write(
        out.join("leakage-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let mut files = BTreeMap::new();
    for f in [
        "train.jsonl",
        "val.jsonl",
        "heldout-hashes.txt",
        "labels.json",
        "leakage-report.json",
    ] {
        files.insert(f, sha256_hex(&fs::read(out.join(f))?));
    }
    let mut counts = BTreeMap::new();
    for r in &train {
        *counts
            .entry(format!("{}/train", r.dataset))
            .or_insert(0usize) += 1;
    }
    for r in &val {
        *counts.entry(format!("{}/val", r.dataset)).or_insert(0usize) += 1;
    }
    let manifest = json!({
        "producer": "openjev prep (crates/ruvector-typesafe-train)",
        "norm": "NFKC -> lowercase -> [^\\p{L}\\p{N}]+ -> ' ' -> trim; sha256 hex",
        "val_carve": "banking77/hwu64: u32(first 8 hex of sha256(id)) % 10 == 0 -> val; clinc150: official val + oos_val; tickets: validation split",
        "tickets_excluded_same_text": l.tickets.excluded_same_text,
        "tickets_fixture_sha256": tickets::FROZEN_SHA256,
        "upstream_pins": pins::DATASET_PINS.iter().map(|p| json!({"name": p.name, "url": p.url, "sha256": p.sha256})).collect::<Vec<_>>(),
        "counts": counts,
        "files": files,
    });
    fs::write(
        out.join("data-manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(PrepOutput {
        train,
        val,
        labels,
        report,
    })
}
