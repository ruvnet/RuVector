//! Append-only, tamper-evident receipts for every promotion-gate decision
//! (ADR-004 "Receipts", ADR-005 "no PII in logs").
//!
//! A [`Receipt`] records the *numbers* behind a decision — accuracies, the
//! sequential-test statistic, the model id, head and temperature — never any
//! `state`, criteria or example text. Receipts are content-hash chained: each
//! carries the previous receipt's hash, so [`ReceiptLog::verify_chain`] detects
//! any edit to any receipt after it was written.
//!
//! No timestamps live in the core (target-independent; the repo forbids
//! `SystemTime` in wasm crates). The caller may supply a `created` string and a
//! monotonic `created_seq`. Ed25519 signing is a v2 concern — [`Receipt`] keeps
//! a `signature: Option<String>` slot that is excluded from the content hash so
//! a later signature never invalidates the chain.

use crate::loop_gate::{GateDecision, Proposal, ProposalKind};
use crate::{Head, Result, TypesafeError};
use serde::{Deserialize, Serialize};

/// Genesis link for the first receipt in a log (32 hex zeros = 128-bit width).
pub const GENESIS_HASH: &str = "00000000000000000000000000000000";

/// A pair of accuracies on one split. Self-describing: both the baseline and
/// the champion number travel together so a receipt cannot misreport either.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub baseline_accuracy: f32,
    pub champion_accuracy: f32,
    pub n: u32,
    /// Expected calibration error (ADR-006), when measured. `None` in v1 loops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ece: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brier: Option<f32>,
}

impl Metrics {
    /// Accuracy of `hits` out of `n`, guarding `n == 0` so a receipt never
    /// stores `0/0 = NaN` (which serialises to JSON `null` and silently changes
    /// the content hash on reload).
    #[must_use]
    pub fn ratio(hits: u32, n: u32) -> f32 {
        if n == 0 {
            0.0
        } else {
            hits as f32 / n as f32
        }
    }
}

/// The anytime-valid paired test result recorded in a receipt (ADR-004 gate 2,
/// arXiv:2606.00878 / 2501.03982). `rejected` latches on the first crossing of
/// `threshold` — anytime-validity is a statement about `sup_t W_t`, so a run
/// that crosses and drifts back still counts as a rejection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestStatistic {
    pub alpha: f32,
    pub lambda: f32,
    /// Wealth at the end of the fed stream.
    pub wealth: f64,
    /// Largest wealth ever reached (the quantity Ville's inequality bounds).
    pub max_wealth: f64,
    /// `1 / alpha`.
    pub threshold: f64,
    /// Discordant pairs where the champion won (champion right, baseline wrong).
    pub n_champion_wins: u32,
    /// Discordant pairs where the baseline won (baseline right, champion wrong).
    pub n_baseline_wins: u32,
    pub rejected: bool,
    /// Number of discordant pairs seen when the rejection first latched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_discordant_at_rejection: Option<u32>,
}

/// One append-only receipt. `seq`, `prev_hash` and `hash` are assigned by
/// [`ReceiptLog::push`]; a receipt built by the gate leaves them at their
/// defaults until it is pushed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    /// Position in the log, assigned on push.
    #[serde(default)]
    pub seq: u64,
    pub proposal: Proposal,
    pub parent: Option<u64>,
    pub kind: ProposalKind,
    pub val: Metrics,
    pub transfer: Metrics,
    /// Test-split metrics — present only for a campaign's baseline and final
    /// champion, never for a mid-loop proposal (see `Gate::score_test`).
    pub test: Option<Metrics>,
    pub statistic: TestStatistic,
    pub decision: GateDecision,
    pub model_id: String,
    pub head: Head,
    pub temperature: f32,
    pub budget_consumed: u32,
    /// Caller-supplied logical clock (no `SystemTime` in core).
    pub created_seq: u64,
    /// Optional caller-supplied timestamp string; excluded from nothing —
    /// it is part of the hashed body once set at construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    /// Previous receipt's `hash`, assigned on push.
    #[serde(default)]
    pub prev_hash: String,
    /// This receipt's content hash, assigned on push. Excluded from the digest.
    #[serde(default)]
    pub hash: String,
    /// v2 Ed25519 signature; excluded from the digest so signing is additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl Receipt {
    /// Content hash over every field except `hash` and `signature`. `prev_hash`
    /// is included, which is what chains the log together.
    fn digest(&self) -> String {
        let mut v = serde_json::to_value(self).expect("receipt is serialisable");
        if let Some(obj) = v.as_object_mut() {
            obj.remove("hash");
            obj.remove("signature");
        }
        let bytes = serde_json::to_string(&v).expect("value is serialisable");
        content_hash(bytes.as_bytes())
    }
}

/// Append-only log of receipts with a verifiable content-hash chain.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReceiptLog {
    receipts: Vec<Receipt>,
}

impl ReceiptLog {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `receipt`, assigning its `seq`, `prev_hash` and `hash`. Returns
    /// the finalised receipt.
    pub fn push(&mut self, mut receipt: Receipt) -> &Receipt {
        receipt.seq = self.receipts.len() as u64;
        receipt.prev_hash = self
            .receipts
            .last()
            .map_or_else(|| GENESIS_HASH.to_string(), |r| r.hash.clone());
        receipt.hash = String::new();
        receipt.hash = receipt.digest();
        self.receipts.push(receipt);
        self.receipts.last().expect("just pushed")
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Receipt> {
        self.receipts.iter()
    }

    /// One JSON object per line (JSONL), in log order.
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        self.receipts
            .iter()
            .map(|r| serde_json::to_string(r).expect("receipt is serialisable"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Parse a JSONL log, keeping every stored `hash`/`prev_hash` verbatim so
    /// [`verify_chain`](Self::verify_chain) actually proves something.
    pub fn from_jsonl(s: &str) -> Result<Self> {
        let mut receipts = Vec::new();
        for line in s.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let r: Receipt = serde_json::from_str(line)
                .map_err(|e| TypesafeError::Invalid(format!("receipt parse: {e}")))?;
            receipts.push(r);
        }
        Ok(Self { receipts })
    }

    /// Recompute the chain. `Err(i)` names the first receipt whose stored hash,
    /// link to its predecessor, or `seq` does not check out — i.e. the first
    /// tampered or reordered entry.
    pub fn verify_chain(&self) -> std::result::Result<(), usize> {
        let mut expected_prev = GENESIS_HASH.to_string();
        for (i, r) in self.receipts.iter().enumerate() {
            if r.seq != i as u64 || r.prev_hash != expected_prev {
                return Err(i);
            }
            if r.digest() != r.hash {
                return Err(i);
            }
            expected_prev = r.hash.clone();
        }
        Ok(())
    }
}

/// 128-bit content hash (two decorrelated FNV-1a streams), hex-encoded. Stable
/// across runs and targets, no dependency. v2 replaces the chain link with an
/// Ed25519 signature; this stays as the fast integrity check.
#[must_use]
pub fn content_hash(bytes: &[u8]) -> String {
    let h1 = fnv1a(bytes, 0xcbf2_9ce4_8422_2325);
    // Second stream: different offset basis, length-mixed, to decorrelate.
    let mut h2 = fnv1a(bytes, 0x9e37_79b9_7f4a_7c15);
    h2 ^= (bytes.len() as u64).wrapping_mul(0x100_0000_01b3);
    h2 = fnv1a(&h2.to_le_bytes(), h2);
    format!("{h1:016x}{h2:016x}")
}

fn fnv1a(bytes: &[u8], offset: u64) -> u64 {
    let mut h = offset;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_gate::ProposalKind;

    fn sample_receipt(champ_val: f32) -> Receipt {
        Receipt {
            seq: 0,
            proposal: Proposal {
                id: 7,
                parent: None,
                kind: ProposalKind::BankGrowth,
                description_hash: 0xdead_beef,
            },
            parent: None,
            kind: ProposalKind::BankGrowth,
            val: Metrics {
                baseline_accuracy: 0.80,
                champion_accuracy: champ_val,
                n: 100,
                ece: None,
                brier: None,
            },
            transfer: Metrics {
                baseline_accuracy: 0.78,
                champion_accuracy: 0.79,
                n: 50,
                ece: None,
                brier: None,
            },
            test: None,
            statistic: TestStatistic {
                alpha: 0.05,
                lambda: 0.5,
                wealth: 25.0,
                max_wealth: 25.0,
                threshold: 20.0,
                n_champion_wins: 30,
                n_baseline_wins: 10,
                rejected: true,
                n_discordant_at_rejection: Some(38),
            },
            decision: GateDecision::Promote,
            model_id: "hash-bow-64@test-double".into(),
            head: Head::NearestPrototype,
            temperature: 1.0,
            budget_consumed: 1,
            created_seq: 42,
            created: None,
            prev_hash: String::new(),
            hash: String::new(),
            signature: None,
        }
    }

    #[test]
    fn chain_verifies_for_untouched_log() {
        let mut log = ReceiptLog::new();
        log.push(sample_receipt(0.90));
        log.push(sample_receipt(0.91));
        log.push(sample_receipt(0.92));
        assert!(log.verify_chain().is_ok());
        assert_eq!(log.len(), 3);
        // Chain links: each prev_hash equals the prior hash.
        let hashes: Vec<_> = log.iter().map(|r| r.hash.clone()).collect();
        assert_eq!(log.iter().nth(1).unwrap().prev_hash, hashes[0]);
    }

    #[test]
    fn mutated_receipt_text_fails_verification() {
        let mut log = ReceiptLog::new();
        log.push(sample_receipt(0.90));
        log.push(sample_receipt(0.91));
        let jsonl = log.to_jsonl();
        // A reloaded, untouched log still verifies.
        assert!(ReceiptLog::from_jsonl(&jsonl)
            .unwrap()
            .verify_chain()
            .is_ok());
        // Tamper with the on-disk text: flip a champion-accuracy digit.
        let tampered = jsonl.replacen("0.91", "0.99", 1);
        assert_ne!(tampered, jsonl);
        let reloaded = ReceiptLog::from_jsonl(&tampered).unwrap();
        assert_eq!(reloaded.verify_chain(), Err(1));
    }

    #[test]
    fn signature_is_excluded_from_the_hash() {
        let mut log = ReceiptLog::new();
        log.push(sample_receipt(0.90));
        let mut jsonl_receipt: Receipt =
            serde_json::from_str(log.to_jsonl().lines().next().unwrap()).unwrap();
        let before = jsonl_receipt.hash.clone();
        // Adding a v2 signature must not change the content digest.
        jsonl_receipt.signature = Some("ed25519:deadbeef".into());
        let mut relog = ReceiptLog {
            receipts: vec![jsonl_receipt],
        };
        // prev_hash for index 0 is genesis, seq 0 — still valid, hash unchanged.
        relog.receipts[0].prev_hash = GENESIS_HASH.to_string();
        relog.receipts[0].seq = 0;
        assert_eq!(relog.receipts[0].hash, before);
        assert!(relog.verify_chain().is_ok());
    }

    #[test]
    fn metrics_ratio_guards_zero_n() {
        assert_eq!(Metrics::ratio(0, 0), 0.0);
        assert_eq!(Metrics::ratio(3, 4), 0.75);
    }
}
