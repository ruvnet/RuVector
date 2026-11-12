//! Tickets fixture → rows, mirroring `bench/lib/fixture.mjs` exactly:
//! `assignSplit` (sha256(id) first 8 hex as u32 % 100: <15 calibration,
//! <30 transfer, else train) and `excludeHeldOutText` (raw-text equality with
//! any non-train split removes a train row).

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::data::{Row, TICKETS};
use crate::norm::sha256_hex;
use crate::pins::read_verified;

/// sha256 of the frozen `tickets-decisions.json` (bench/fixtures/HASHES.json).
pub const FROZEN_SHA256: &str = "2ac87d9b00fb6fcee8a3cf3118577fbb4056f03f8feaecdce892d358d2bc99cd";

#[derive(Deserialize)]
struct Fixture {
    departments: Vec<String>,
    split: BTreeMap<String, Vec<Item>>,
    gen0_questions: Questions,
}

#[derive(Deserialize)]
struct Questions {
    department: DeptQuestion,
}

#[derive(Deserialize)]
struct DeptQuestion {
    criteria: BTreeMap<String, String>,
}

#[derive(Deserialize, Clone)]
struct Item {
    id: String,
    text: String,
    label: ItemLabel,
    #[serde(default)]
    ambiguous: bool,
    #[serde(default)]
    secondary: Option<String>,
}

#[derive(Deserialize, Clone)]
struct ItemLabel {
    department: String,
    urgent: bool,
    frustration: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Calibration,
    Validation,
    Transfer,
    Test,
}

/// `assignSplit(id, nativeSplit)` from fixture.mjs.
pub fn assign_split(id: &str, native: &str) -> Split {
    match native {
        "test" => Split::Test,
        "val" => Split::Validation,
        _ => {
            let h = sha256_hex(id.as_bytes());
            let bucket = u32::from_str_radix(&h[..8], 16).expect("hex") % 100;
            if bucket < 15 {
                Split::Calibration
            } else if bucket < 30 {
                Split::Transfer
            } else {
                Split::Train
            }
        }
    }
}

pub struct Tickets {
    /// Gradient rows (136 at the 2026-09-26 freeze).
    pub train: Vec<Row>,
    /// Selection rows (validation, 150).
    pub val: Vec<Row>,
    /// Texts that must never be seen: test + transfer + calibration.
    pub heldout_texts: Vec<String>,
    /// department -> gen-0 criterion text.
    pub criteria: BTreeMap<String, String>,
    pub excluded_same_text: usize,
}

fn to_row(it: &Item) -> Row {
    let mut r = Row::new(
        TICKETS,
        it.id.clone(),
        it.text.clone(),
        it.label.department.clone(),
    );
    r.urgent = Some(it.label.urgent);
    r.frustration = Some(it.label.frustration);
    if it.ambiguous {
        if let Some(sec) = &it.secondary {
            if sec != &it.label.department {
                let mut soft = BTreeMap::new();
                soft.insert(it.label.department.clone(), 0.75);
                soft.insert(sec.clone(), 0.25);
                r.soft = Some(soft);
            }
        }
    }
    r
}

pub fn load(fixture: &Path) -> Result<Tickets> {
    let bytes = read_verified(fixture, FROZEN_SHA256)
        .context("tickets fixture is not the frozen one — refusing (ADR-006 §protocol)")?;
    let fx: Fixture = serde_json::from_slice(&bytes)?;
    let mut by: BTreeMap<&'static str, Vec<Item>> = BTreeMap::new();
    for native in ["train", "val", "test"] {
        let items = fx
            .split
            .get(native)
            .with_context(|| format!("fixture has no split {native}"))?;
        for it in items {
            let key = match assign_split(&it.id, native) {
                Split::Train => "train",
                Split::Calibration => "calibration",
                Split::Validation => "validation",
                Split::Transfer => "transfer",
                Split::Test => "test",
            };
            by.entry(key).or_default().push(it.clone());
        }
    }
    // assertDisjoint by id.
    let mut seen = HashSet::new();
    for items in by.values() {
        for it in items {
            if !seen.insert(it.id.as_str()) {
                bail!("tickets split overlap on id {}", it.id);
            }
        }
    }
    let heldout_texts: Vec<String> = ["calibration", "transfer", "test"]
        .iter()
        .flat_map(|s| by.get(s).into_iter().flatten().map(|i| i.text.clone()))
        .collect();
    let non_train: HashSet<&str> = ["calibration", "validation", "transfer", "test"]
        .iter()
        .flat_map(|s| by.get(s).into_iter().flatten().map(|i| i.text.as_str()))
        .collect();
    let all_train = by.get("train").cloned().unwrap_or_default();
    let train: Vec<Row> = all_train
        .iter()
        .filter(|i| !non_train.contains(i.text.as_str()))
        .map(to_row)
        .collect();
    let excluded_same_text = all_train.len() - train.len();
    let val: Vec<Row> = by
        .get("validation")
        .into_iter()
        .flatten()
        .map(to_row)
        .collect();
    let criteria = fx.gen0_questions.department.criteria;
    for d in &fx.departments {
        if !criteria.contains_key(d) {
            bail!("department {d} has no gen-0 criterion");
        }
    }
    Ok(Tickets {
        train,
        val,
        heldout_texts,
        criteria,
        excluded_same_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_splits_pass_through() {
        assert_eq!(assign_split("anything", "test"), Split::Test);
        assert_eq!(assign_split("anything", "val"), Split::Validation);
    }

    #[test]
    fn carve_is_a_stable_hash_bucket() {
        // Bucket boundaries are exercised across many ids; the proportion is ~70/15/15.
        let mut n = [0usize; 3];
        for i in 0..2000 {
            match assign_split(&format!("t{i}"), "train") {
                Split::Calibration => n[0] += 1,
                Split::Transfer => n[1] += 1,
                Split::Train => n[2] += 1,
                _ => unreachable!(),
            }
        }
        assert!(n[0] > 200 && n[1] > 200 && n[2] > 1200, "{n:?}");
        assert_eq!(assign_split("t440", "train"), assign_split("t440", "train"));
    }
}
