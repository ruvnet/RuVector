//! Assertion A (ADR-008 §2): before the first optimizer step, every normalized
//! train + validation text is hashed and intersected with the held-out hash set
//! (tickets test/transfer/calibration, Banking77 test, CLINC150 test +
//! oos_test, HWU64 test bucket). A non-empty intersection aborts with exit 3.

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use crate::data::Row;
use crate::norm::sha256_norm;

/// Process exit code for a leakage abort.
pub const LEAKAGE_EXIT: i32 = 3;

#[derive(Debug, Default, Clone, Serialize, PartialEq)]
pub struct DatasetLeak {
    pub train_rows: usize,
    pub val_rows: usize,
    pub heldout_rows: usize,
    pub intersection: usize,
    /// Rows removed at prep time because they equal a held-out text.
    pub dropped_duplicates: usize,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct LeakageReport {
    pub heldout_hashes: usize,
    pub per_dataset: BTreeMap<String, DatasetLeak>,
    /// Label-description texts (SupCon extra positives) that equal a held-out
    /// text; they are excluded from training, never used (dataset -> labels).
    #[serde(default)]
    pub dropped_descriptions: BTreeMap<String, Vec<String>>,
    /// Up to 10 offending (dataset, id) pairs, for the abort message.
    pub examples: Vec<(String, String)>,
}

impl LeakageReport {
    pub fn total_intersection(&self) -> usize {
        self.per_dataset.values().map(|d| d.intersection).sum()
    }
}

#[derive(Debug)]
pub struct LeakageError(pub LeakageReport);

impl std::fmt::Display for LeakageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LEAKAGE: {} train/val rows hash-match a held-out test text (e.g. {:?}); refusing to train",
            self.0.total_intersection(),
            self.0.examples
        )
    }
}

impl std::error::Error for LeakageError {}

/// Remove rows whose normalized hash is held out; returns (kept, dropped per dataset).
pub fn drop_heldout(
    rows: Vec<Row>,
    heldout: &HashSet<String>,
) -> (Vec<Row>, BTreeMap<String, usize>) {
    let mut dropped: BTreeMap<String, usize> = BTreeMap::new();
    let kept = rows
        .into_iter()
        .filter(|r| {
            let hit = heldout.contains(&sha256_norm(&r.text));
            if hit {
                *dropped.entry(r.dataset.clone()).or_default() += 1;
            }
            !hit
        })
        .collect();
    (kept, dropped)
}

/// Label descriptions whose normalized hash is held out. Found on the real data:
/// 12 humanised intent names (e.g. CLINC150 `goodbye`, `what is your name`)
/// are verbatim public *test* utterances, so they must never produce gradients.
pub fn colliding_descriptions(
    labels: &crate::data::Labels,
    heldout: &HashSet<String>,
) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (ds, m) in labels {
        for (label, desc) in m {
            if heldout.contains(&sha256_norm(desc)) {
                out.entry(ds.clone()).or_default().push(label.clone());
            }
        }
    }
    out
}

/// Assertion A. `heldout_by_dataset` is only used for per-dataset counts; the
/// intersection is against the union (a public train text equal to a *ticket*
/// test text is leakage too).
pub fn assert_no_leakage(
    train: &[Row],
    val: &[Row],
    heldout: &HashSet<String>,
    heldout_by_dataset: &BTreeMap<String, usize>,
    dropped: &BTreeMap<String, usize>,
) -> Result<LeakageReport, LeakageError> {
    let mut rep = LeakageReport {
        heldout_hashes: heldout.len(),
        ..Default::default()
    };
    for (ds, n) in heldout_by_dataset {
        rep.per_dataset.entry(ds.clone()).or_default().heldout_rows = *n;
    }
    for (ds, n) in dropped {
        rep.per_dataset
            .entry(ds.clone())
            .or_default()
            .dropped_duplicates = *n;
    }
    for (rows, is_train) in [(train, true), (val, false)] {
        for r in rows {
            let e = rep.per_dataset.entry(r.dataset.clone()).or_default();
            if is_train {
                e.train_rows += 1;
            } else {
                e.val_rows += 1;
            }
            if heldout.contains(&sha256_norm(&r.text)) {
                e.intersection += 1;
                if rep.examples.len() < 10 {
                    rep.examples.push((r.dataset.clone(), r.id.clone()));
                }
            }
        }
    }
    if rep.total_intersection() > 0 {
        Err(LeakageError(rep))
    } else {
        Ok(rep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TICKETS;

    fn row(id: &str, text: &str) -> Row {
        Row::new(TICKETS, id.into(), text.into(), "billing".into())
    }

    #[test]
    fn planted_collision_aborts_even_after_normalization() {
        let heldout: HashSet<String> = [sha256_norm("Refund my ORDER, please!")].into();
        let train = vec![
            row("a", "totally different"),
            row("b", "refund my order please"),
        ];
        let err = assert_no_leakage(&train, &[], &heldout, &BTreeMap::new(), &BTreeMap::new())
            .unwrap_err();
        assert_eq!(err.0.total_intersection(), 1);
        assert_eq!(err.0.examples, vec![(TICKETS.to_string(), "b".to_string())]);
        assert!(err.to_string().contains("LEAKAGE"));
    }

    #[test]
    fn validation_rows_are_checked_too() {
        let heldout: HashSet<String> = [sha256_norm("x y")].into();
        let err = assert_no_leakage(
            &[],
            &[row("v", "X-Y")],
            &heldout,
            &BTreeMap::new(),
            &BTreeMap::new(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn description_equal_to_a_test_text_is_reported() {
        let heldout: HashSet<String> = [sha256_norm("Goodbye!")].into();
        let mut labels = crate::data::Labels::new();
        labels.insert(
            "clinc150".into(),
            [
                ("goodbye".to_string(), "goodbye".to_string()),
                ("greeting".to_string(), "greeting".to_string()),
            ]
            .into(),
        );
        let hit = colliding_descriptions(&labels, &heldout);
        assert_eq!(hit["clinc150"], vec!["goodbye".to_string()]);
    }

    #[test]
    fn drop_then_assert_is_clean() {
        let heldout: HashSet<String> = [sha256_norm("dup")].into();
        let (kept, dropped) = drop_heldout(vec![row("a", "DUP"), row("b", "fine")], &heldout);
        assert_eq!(kept.len(), 1);
        assert_eq!(dropped[TICKETS], 1);
        let rep = assert_no_leakage(&kept, &[], &heldout, &BTreeMap::new(), &dropped).unwrap();
        assert_eq!(rep.per_dataset[TICKETS].dropped_duplicates, 1);
        assert_eq!(rep.total_intersection(), 0);
    }
}
