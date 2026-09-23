//! The example bank (ADR-004 loop 1): append-only, immutable, split-frozen.
//!
//! An example enters once, keyed by a content [`ExampleId`] over
//! `(question, label, text)`; a re-admission of the same content is a
//! [`Admission::Duplicate`], never a second row. Its [`Split`] is assigned
//! deterministically from the id at admission and is **frozen** — it never
//! changes, not on reload, not if the ratios change (ADR-006 "frozen splits").
//!
//! Only the metadata leaves the bank in a receipt: [`redacted_summary`] carries
//! counts and hashes, never text (ADR-005). The raw text is kept in memory so
//! prototypes can be re-embedded, but it is confined to this module's storage
//! and the full-fidelity [`Bank::to_json`] (the user's own data on their disk).
//!
//! Trust tiers follow ADR-004: tier C examples may be *stored* but never count
//! toward a promotion, so [`Bank::promotable`] excludes them.
//!
//! Note: hash-based assignment makes the splits disjoint and reproducible, but
//! it cannot by itself make Transfer a *different distribution* (ADR-006 gate 3
//! wants a different dataset/domain) — that is a bench/caller concern.

use crate::{Result, TypesafeError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Stable 64-bit content id of an example (no `RandomState`, no per-run salt).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExampleId(pub u64);

/// The five frozen splits (ADR-006). Instance-ID disjoint by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Calibration,
    Validation,
    Transfer,
    Test,
}

/// Trust tier of a label (ADR-004 §trust tiers, ADR-005 trust boundaries).
/// A: programmatic verifiers. B: LLM-judge labels (quarantined until confirmed).
/// C: user-supplied text — stored, but never authoritative for a promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TrustTier {
    A,
    B,
    C,
}

/// Immutable metadata for one bank example. The text lives beside it in the
/// bank (see [`Entry`]); only these fields are ever exposed in a receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Example {
    pub id: ExampleId,
    pub question: String,
    pub text_hash: u64,
    pub text_len: u32,
    pub label: String,
    pub split: Split,
    pub tier: TrustTier,
    pub added_seq: u64,
}

/// Result of an admission attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// Newly stored.
    Accepted(ExampleId),
    /// Same content already present; the bank is unchanged.
    Duplicate(ExampleId),
    /// Held out of the bank by the noisy-label filter, with a reason.
    Quarantined(String),
}

/// Split ratios as integer percentages summing to 100. Assignment is
/// `id % 100` against the cumulative boundaries, so it is deterministic and
/// stable across runs and targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitRatios {
    pub train: u8,
    pub calibration: u8,
    pub validation: u8,
    pub transfer: u8,
    pub test: u8,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            train: 60,
            calibration: 10,
            validation: 15,
            transfer: 10,
            test: 5,
        }
    }
}

impl SplitRatios {
    /// Reject ratios that do not sum to 100 (a silent mis-sum would skew every
    /// split assignment).
    pub fn new(train: u8, calibration: u8, validation: u8, transfer: u8, test: u8) -> Result<Self> {
        let sum =
            train as u16 + calibration as u16 + validation as u16 + transfer as u16 + test as u16;
        if sum != 100 {
            return Err(TypesafeError::Invalid(format!(
                "split ratios must sum to 100, got {sum}"
            )));
        }
        Ok(Self {
            train,
            calibration,
            validation,
            transfer,
            test,
        })
    }

    fn assign(&self, id: ExampleId) -> Split {
        let bucket = (id.0 % 100) as u16;
        let mut edge = self.train as u16;
        if bucket < edge {
            return Split::Train;
        }
        edge += self.calibration as u16;
        if bucket < edge {
            return Split::Calibration;
        }
        edge += self.validation as u16;
        if bucket < edge {
            return Split::Validation;
        }
        edge += self.transfer as u16;
        if bucket < edge {
            return Split::Transfer;
        }
        Split::Test
    }
}

/// A stored example plus its raw text. `text` is serialised in [`Bank::to_json`]
/// (the user's own data) but never in [`redacted_summary`] or any receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    #[serde(flatten)]
    example: Example,
    text: String,
}

/// A closure that vets a candidate example against its text — an agreement /
/// noisy-label check the engine can supply later. Returns `true` to admit,
/// `false` to quarantine. Aliased to keep clippy's `type_complexity` quiet.
pub type NoisyLabelFilter = Box<dyn Fn(&Example, &str) -> bool + Send + Sync>;

/// Redacted, receipt-safe view of the bank: counts and hashes, never text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactedSummary {
    pub total: usize,
    pub promotable: usize,
    /// `(question, split, tier, count)` rollup.
    pub by_bucket: Vec<BucketCount>,
    /// Content-id hashes only, in admission order.
    pub ids: Vec<ExampleId>,
    pub ratios: SplitRatios,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BucketCount {
    pub question: String,
    pub split: Split,
    pub tier: TrustTier,
    pub count: usize,
}

/// The append-only example bank.
#[derive(Serialize, Deserialize)]
pub struct Bank {
    ratios: SplitRatios,
    next_seq: u64,
    entries: Vec<Entry>,
    #[serde(skip)]
    ids: BTreeSet<u64>,
    #[serde(skip)]
    noisy_label_filter: Option<NoisyLabelFilter>,
}

impl std::fmt::Debug for Bank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bank")
            .field("ratios", &self.ratios)
            .field("next_seq", &self.next_seq)
            .field("entries", &self.entries.len())
            .field("has_filter", &self.noisy_label_filter.is_some())
            .finish()
    }
}

impl Default for Bank {
    fn default() -> Self {
        Self::new(SplitRatios::default())
    }
}

impl Bank {
    #[must_use]
    pub fn new(ratios: SplitRatios) -> Self {
        Self {
            ratios,
            next_seq: 0,
            entries: Vec::new(),
            ids: BTreeSet::new(),
            noisy_label_filter: None,
        }
    }

    /// Install the agreement / noisy-label check. Not serialised — reinstall it
    /// after [`from_json`](Self::from_json).
    #[must_use]
    pub fn with_noisy_label_filter(mut self, f: NoisyLabelFilter) -> Self {
        self.noisy_label_filter = Some(f);
        self
    }

    /// Admit one labeled example. Deduplicates by content id, assigns a frozen
    /// split, and runs the noisy-label filter (default: accept).
    pub fn admit(&mut self, question: &str, text: &str, label: &str, tier: TrustTier) -> Admission {
        let id = ExampleId(content_id(question, label, text));
        if self.ids.contains(&id.0) {
            return Admission::Duplicate(id);
        }
        let candidate = Example {
            id,
            question: question.to_string(),
            text_hash: text_hash(text),
            text_len: text.len() as u32,
            label: label.to_string(),
            split: self.ratios.assign(id),
            tier,
            added_seq: self.next_seq,
        };
        if let Some(filter) = &self.noisy_label_filter {
            if !filter(&candidate, text) {
                return Admission::Quarantined("noisy-label filter disagreed".into());
            }
        }
        self.ids.insert(id.0);
        self.next_seq += 1;
        self.entries.push(Entry {
            example: candidate,
            text: text.to_string(),
        });
        Admission::Accepted(id)
    }

    /// Check whether an example is already stored before paying for its
    /// embedding. Uses the same content identity as [`admit`](Self::admit).
    #[must_use]
    pub fn contains(&self, question: &str, text: &str, label: &str) -> bool {
        self.ids.contains(&content_id(question, label, text))
    }

    /// Admit one example into an explicit `split`, bypassing ratio assignment.
    /// Used by a campaign, whose validation/transfer/test membership is fixed by
    /// the caller's frozen fixture (so the champion's "test" is exactly the
    /// fixture's test ids), not re-derived from the content hash. Dedup and the
    /// noisy-label filter still apply.
    pub fn admit_into(
        &mut self,
        question: &str,
        text: &str,
        label: &str,
        tier: TrustTier,
        split: Split,
    ) -> Admission {
        let id = ExampleId(content_id(question, label, text));
        if self.ids.contains(&id.0) {
            return Admission::Duplicate(id);
        }
        let candidate = Example {
            id,
            question: question.to_string(),
            text_hash: text_hash(text),
            text_len: text.len() as u32,
            label: label.to_string(),
            split,
            tier,
            added_seq: self.next_seq,
        };
        if let Some(filter) = &self.noisy_label_filter {
            if !filter(&candidate, text) {
                return Admission::Quarantined("noisy-label filter disagreed".into());
            }
        }
        self.ids.insert(id.0);
        self.next_seq += 1;
        self.entries.push(Entry {
            example: candidate,
            text: text.to_string(),
        });
        Admission::Accepted(id)
    }

    /// Instance-ID disjointness of two splits (the idea from
    /// `sona::darwin_guard::assert_train_eval_disjoint`, reused not copied).
    /// `Ok(())` when disjoint; `Err(overlap)` names any id in both — which
    /// deterministic assignment should make impossible.
    pub fn assert_disjoint(&self, a: Split, b: Split) -> std::result::Result<(), Vec<ExampleId>> {
        if a == b {
            return Ok(());
        }
        let in_a: BTreeSet<ExampleId> = self
            .entries
            .iter()
            .filter(|e| e.example.split == a)
            .map(|e| e.example.id)
            .collect();
        let overlap: Vec<ExampleId> = self
            .entries
            .iter()
            .filter(|e| e.example.split == b && in_a.contains(&e.example.id))
            .map(|e| e.example.id)
            .collect();
        if overlap.is_empty() {
            Ok(())
        } else {
            Err(overlap)
        }
    }

    /// All examples for one `(question, split)`, in admission order.
    pub fn iter_split<'a>(
        &'a self,
        question: &'a str,
        split: Split,
    ) -> impl Iterator<Item = &'a Example> + 'a {
        self.entries
            .iter()
            .map(|e| &e.example)
            .filter(move |e| e.question == question && e.split == split)
    }

    /// Promotable examples for one `(question, split)`: tier A or B only — tier
    /// C is stored but never counts toward a promotion (ADR-004).
    pub fn iter_promotable<'a>(
        &'a self,
        question: &'a str,
        split: Split,
    ) -> impl Iterator<Item = &'a Example> + 'a {
        self.iter_split(question, split)
            .filter(|e| e.tier != TrustTier::C)
    }

    /// Every example, in admission order.
    pub fn iter(&self) -> impl Iterator<Item = &Example> {
        self.entries.iter().map(|e| &e.example)
    }

    /// The raw text for an id (needed to re-embed). Never put this in a receipt.
    #[must_use]
    pub fn text_of(&self, id: ExampleId) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.example.id == id)
            .map(|e| e.text.as_str())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count of promotable (tier A/B) examples across all questions and splits.
    #[must_use]
    pub fn promotable(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.example.tier != TrustTier::C)
            .count()
    }

    /// Full-fidelity serialisation, text included (the user's own data).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| TypesafeError::Invalid(format!("bank encode: {e}")))
    }

    /// Load a bank. The stored `split` on each example is authoritative and is
    /// never reassigned from `ratios` — that is what "frozen" means when the
    /// save-time and load-time ratios differ. The id set is rebuilt.
    pub fn from_json(s: &str) -> Result<Self> {
        let mut bank: Bank = serde_json::from_str(s)
            .map_err(|e| TypesafeError::Invalid(format!("bank decode: {e}")))?;
        bank.ids = bank.entries.iter().map(|e| e.example.id.0).collect();
        bank.noisy_label_filter = None;
        Ok(bank)
    }

    /// Receipt-safe rollup: counts and content-id hashes only, no text.
    #[must_use]
    pub fn redacted_summary(&self) -> RedactedSummary {
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<(String, Split, TrustTier), usize> = BTreeMap::new();
        for e in &self.entries {
            *buckets
                .entry((e.example.question.clone(), e.example.split, e.example.tier))
                .or_insert(0) += 1;
        }
        let by_bucket = buckets
            .into_iter()
            .map(|((question, split, tier), count)| BucketCount {
                question,
                split,
                tier,
                count,
            })
            .collect();
        RedactedSummary {
            total: self.entries.len(),
            promotable: self.promotable(),
            by_bucket,
            ids: self.entries.iter().map(|e| e.example.id).collect(),
            ratios: self.ratios,
        }
    }
}

/// Stable content id over `(question, label, text)` — FNV-1a 64 with `0x00`
/// separators so distinct field boundaries cannot alias.
fn content_id(question: &str, label: &str, text: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in [
        question.as_bytes(),
        b"\0",
        label.as_bytes(),
        b"\0",
        text.as_bytes(),
    ] {
        for b in part {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Stable hash of the text alone (recorded in [`Example::text_hash`]).
fn text_hash(text: &str) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15u64;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests;
