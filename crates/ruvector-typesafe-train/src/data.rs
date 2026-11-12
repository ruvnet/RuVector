//! Exported training rows (plan Step 1.2 file layout) and JSONL IO.
//!
//! A data directory holds `train.jsonl`, `val.jsonl`, `labels.json`,
//! `heldout-hashes.txt` and `data-manifest.json`. It can be produced by
//! `openjev prep` (Rust) or by the bench's `export-openjev-data.mjs`; the
//! trainer only depends on this layout.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const TICKETS: &str = "tickets";
pub const BANKING77: &str = "banking77";
pub const CLINC150: &str = "clinc150";
pub const HWU64: &str = "hwu64";
pub const DATASETS: [&str; 4] = [TICKETS, BANKING77, CLINC150, HWU64];

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub dataset: String,
    pub id: String,
    pub text: String,
    /// Class label (tickets: department). `oos` rows carry `"oos"`.
    pub label: String,
    /// Soft department target for ambiguous tickets: {primary: .75, secondary: .25}.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft: Option<BTreeMap<String, f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub urgent: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frustration: Option<u8>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub oos: bool,
}

impl Row {
    pub fn new(dataset: &str, id: String, text: String, label: String) -> Self {
        Self {
            dataset: dataset.to_string(),
            id,
            text,
            label,
            soft: None,
            urgent: None,
            frustration: None,
            oos: false,
        }
    }
}

/// `dataset -> label -> description` (the criteria text of each label).
pub type Labels = BTreeMap<String, BTreeMap<String, String>>;

pub fn write_jsonl(path: &Path, rows: &[Row]) -> Result<()> {
    let mut f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    for r in rows {
        serde_json::to_writer(&mut f, r)?;
        f.write_all(b"\n")?;
    }
    Ok(())
}

pub fn read_jsonl(path: &Path) -> Result<Vec<Row>> {
    crate::pins::check_extension(path)?;
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(&line)
            .with_context(|| format!("{}:{} bad row", path.display(), i + 1))?;
        if !DATASETS.contains(&row.dataset.as_str()) {
            bail!(
                "{}:{} unknown dataset {:?}",
                path.display(),
                i + 1,
                row.dataset
            );
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Sorted, de-duplicated hex hashes, one per line.
pub fn write_hashes(path: &Path, hashes: impl IntoIterator<Item = String>) -> Result<usize> {
    let mut v: Vec<String> = hashes.into_iter().collect();
    v.sort();
    v.dedup();
    let mut s = v.join("\n");
    s.push('\n');
    fs::write(path, s).with_context(|| format!("write {}", path.display()))?;
    Ok(v.len())
}

pub fn read_hashes(path: &Path) -> Result<std::collections::HashSet<String>> {
    crate::pins::check_extension(path)?;
    let s = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = std::collections::HashSet::new();
    for (i, l) in s.lines().enumerate() {
        let l = l.trim();
        if l.is_empty() {
            continue;
        }
        if l.len() != 64 || !l.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("{}:{} not a sha256 hex line", path.display(), i + 1);
        }
        out.insert(l.to_ascii_lowercase());
    }
    Ok(out)
}

pub struct DataDir {
    pub train: Vec<Row>,
    pub val: Vec<Row>,
    pub labels: Labels,
}

impl DataDir {
    pub fn load(dir: &Path) -> Result<Self> {
        let train = read_jsonl(&dir.join("train.jsonl"))?;
        let val = read_jsonl(&dir.join("val.jsonl"))?;
        let lp = dir.join("labels.json");
        crate::pins::check_extension(&lp)?;
        let labels: Labels = serde_json::from_str(
            &fs::read_to_string(&lp).with_context(|| format!("read {}", lp.display()))?,
        )?;
        for r in train.iter().chain(val.iter()) {
            if r.oos {
                continue;
            }
            let known = labels
                .get(&r.dataset)
                .is_some_and(|m| m.contains_key(&r.label));
            if !known {
                bail!(
                    "row {} has label {:?} missing from labels.json",
                    r.id,
                    r.label
                );
            }
        }
        Ok(Self { train, val, labels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_round_trips_with_optional_fields_elided() {
        let mut r = Row::new(TICKETS, "t1".into(), "hi".into(), "billing".into());
        let s = serde_json::to_string(&r).unwrap();
        assert!(!s.contains("soft") && !s.contains("oos"));
        r.oos = true;
        r.urgent = Some(true);
        let back: Row = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
    }
}
