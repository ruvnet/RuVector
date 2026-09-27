//! The append-only test-scoring ledger (ADR-007 §2.6, P0-5).
//!
//! Record of truth: `npm/packages/kge/bench/results/final/ledger.jsonl` on
//! origin's protected branch `kge-final-ledger` (plus `selection.json` next to
//! it). One JSON object per line, `intent` before scoring and `result` after,
//! keyed by `(dataset, seed)`. A dataset has exactly one `config_hash` (the
//! selected one) so `(dataset, seed)` subsumes `(dataset, config_hash, seed)`.
//!
//! [`check`] is the CI rule set (`kge-ledger-check`): it fails on malformed
//! lines, a seed outside {0..4}, a second intent or result for a key (duplicate
//! keys), a result without an intent or disagreeing with it, more than one
//! config hash per dataset or one that differs from `selection.json`, tag SHAs
//! that differ, and — given the previous ledger — any modified or deleted line.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Repo-relative location of the ledger and selection on the ledger branch.
pub const LEDGER_PATH: &str = "npm/packages/kge/bench/results/final/ledger.jsonl";
pub const SELECTION_PATH: &str = "npm/packages/kge/bench/results/final/selection.json";
pub const SIGNERS_PATH: &str = "npm/packages/kge/bench/results/final/SIGNERS";
pub const LEDGER_BRANCH: &str = "kge-final-ledger";
pub const PREREG_TAG: &str = "kge-prereg-v1";
/// The pre-registered final seeds.
pub const FINAL_SEEDS: std::ops::RangeInclusive<u64> = 0..=4;

/// One ledger line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Line {
    Intent {
        dataset: String,
        seed: u64,
        config_hash: String,
        tag_sha: String,
        checkpoint_sha256: String,
        utc: String,
    },
    Result {
        dataset: String,
        seed: u64,
        config_hash: String,
        tag_sha: String,
        checkpoint_sha256: String,
        receipt_sha256: String,
        bottom_ranks_sha256: String,
        random_ranks_sha256: String,
        utc: String,
    },
}

impl Line {
    pub fn key(&self) -> (&str, u64) {
        match self {
            Line::Intent { dataset, seed, .. } | Line::Result { dataset, seed, .. } => {
                (dataset, *seed)
            }
        }
    }
    fn common(&self) -> (&str, &str, &str) {
        match self {
            Line::Intent {
                config_hash,
                tag_sha,
                checkpoint_sha256,
                ..
            }
            | Line::Result {
                config_hash,
                tag_sha,
                checkpoint_sha256,
                ..
            } => (config_hash, tag_sha, checkpoint_sha256),
        }
    }
    /// The line as written to the file (one compact JSON object, no newline).
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).expect("ledger line serialises")
    }
}

/// `selection.json`: dataset → selected config hash (plus the HPO receipt
/// hashes it was selected from, not interpreted here).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    pub datasets: BTreeMap<String, SelectionEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionEntry {
    pub config_hash: String,
    #[serde(default)]
    pub hpo_receipts: Vec<String>,
}

impl Selection {
    pub fn parse(text: &str) -> Result<Self> {
        serde_json::from_str(text).context("parse selection.json")
    }
}

/// Parse every non-empty line (1-based line numbers in errors).
pub fn parse(text: &str) -> Result<Vec<Line>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).with_context(|| format!("ledger line {}: malformed", i + 1))
        })
        .collect()
}

/// State of one `(dataset, seed)` key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyState<'a> {
    Free,
    /// An intent with no result (a crashed or failed scoring; retry allowed
    /// only with the same checkpoint).
    IntentOnly(&'a Line),
    Done,
}

pub fn key_state<'a>(lines: &'a [Line], dataset: &str, seed: u64) -> KeyState<'a> {
    let mut intent = None;
    for l in lines.iter().filter(|l| l.key() == (dataset, seed)) {
        match l {
            Line::Result { .. } => return KeyState::Done,
            Line::Intent { .. } => intent = Some(l),
        }
    }
    intent.map_or(KeyState::Free, KeyState::IntentOnly)
}

/// The CI rule set (see the module docs). `previous` is the ledger as it was
/// before this change (the base commit), for the append-only rule.
pub fn check(
    text: &str,
    selection: Option<&Selection>,
    previous: Option<&str>,
    tag_sha: Option<&str>,
) -> Result<()> {
    if let Some(prev) = previous {
        let (old, new): (Vec<&str>, Vec<&str>) = (prev.lines().collect(), text.lines().collect());
        if new.len() < old.len() || old.iter().zip(&new).any(|(a, b)| a != b) {
            bail!("append-only violation: an existing ledger line was modified or deleted");
        }
    }
    let lines = parse(text)?;
    let mut intents: BTreeMap<(String, u64), &Line> = BTreeMap::new();
    let mut results: BTreeSet<(String, u64)> = BTreeSet::new();
    let mut hashes: BTreeMap<&str, &str> = BTreeMap::new();
    let mut tags: BTreeSet<&str> = BTreeSet::new();
    if selection.is_none() && !lines.is_empty() {
        bail!("ledger has lines but no selection.json");
    }
    for (i, l) in lines.iter().enumerate() {
        let n = i + 1;
        let (ds, seed) = l.key();
        let (ch, tag, ck) = l.common();
        if !FINAL_SEEDS.contains(&seed) {
            bail!("line {n}: seed {seed} is outside the pre-registered {{0..4}}");
        }
        if let Some(prev) = hashes.insert(ds, ch) {
            if prev != ch {
                bail!("line {n}: dataset '{ds}' has a second config_hash");
            }
        }
        if let Some(sel) = selection {
            let want = sel
                .datasets
                .get(ds)
                .with_context(|| format!("line {n}: dataset '{ds}' is not in selection.json"))?;
            if want.config_hash != ch {
                bail!("line {n}: config_hash differs from selection.json for '{ds}'");
            }
        }
        tags.insert(tag);
        let key = (ds.to_string(), seed);
        match l {
            Line::Intent { .. } => {
                if intents.insert(key, l).is_some() {
                    bail!("line {n}: duplicate intent for ({ds}, {seed})");
                }
            }
            Line::Result { .. } => {
                let it = intents.get(&key).with_context(|| {
                    format!("line {n}: result for ({ds}, {seed}) has no prior intent")
                })?;
                if it.common() != (ch, tag, ck) {
                    bail!("line {n}: result for ({ds}, {seed}) disagrees with its intent");
                }
                if !results.insert(key) {
                    bail!("line {n}: duplicate result for ({ds}, {seed})");
                }
            }
        }
    }
    if tags.len() > 1 {
        bail!("ledger lines carry more than one tag SHA");
    }
    if let (Some(want), Some(got)) = (tag_sha, tags.iter().next()) {
        if want != *got {
            bail!("ledger tag SHA {got} differs from {PREREG_TAG} ({want})");
        }
    }
    Ok(())
}
