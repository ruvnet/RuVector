//! KGE receipt wrapper (ADR-004 "Receipts"). Typesafe's `Receipt` has no slot
//! for the KGE-specific fields ADR-004 requires, so instead of overloading its
//! `model_id`/`head`/`temperature`, each decision is wrapped:
//! `KgeReceipt { receipt: <typesafe Receipt>, kge: KgeReceiptExtra }`.
//!
//! Two chains run in parallel and are both verifiable:
//! - the **inner** typesafe hash chain over the wrapped `Receipt`, untouched, so
//!   `ruvector_typesafe_core`'s `ReceiptLog::verify_chain` still proves it;
//! - the **wrapper** hash chain, whose digest commits the inner receipt's hash
//!   *and* the `kge` extra *and* the previous wrapper hash — so tampering with
//!   any KGE field is caught even though the inner receipt is unchanged.

use crate::ScorerKind;
use ruvector_typesafe_core::receipt::{content_hash, Receipt, ReceiptLog, GENESIS_HASH};
use serde::{Deserialize, Serialize};

/// The KGE-specific fields ADR-004 wants on every receipt, kept beside the
/// typesafe receipt rather than crammed into it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KgeReceiptExtra {
    pub scorer: ScorerKind,
    pub dims: usize,
    pub knobs_hash: u64,
    /// EWC penalty weight — set only for continual-update receipts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ewc_lambda: Option<f32>,
    /// Fisher-snapshot id for a continual update (ADR-004).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fisher_id: Option<String>,
    /// Content-hash of the loaded model manifest (ADR-005), when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_hash: Option<String>,
    /// RVF lineage link (ADR-004 `lineage_depth`), when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<String>,
    /// Self-adversarial sampling temperature (distinct from typesafe's
    /// calibration temperature).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_temperature: Option<f32>,
}

/// One wrapped receipt. `kge_prev_hash`/`kge_hash` are assigned on push.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KgeReceipt {
    pub receipt: Receipt,
    pub kge: KgeReceiptExtra,
    #[serde(default)]
    pub kge_prev_hash: String,
    #[serde(default)]
    pub kge_hash: String,
}

impl KgeReceipt {
    /// Wrapper digest: commits the inner receipt's hash, the KGE extra and the
    /// previous wrapper hash. Excludes `kge_hash` itself.
    fn digest(&self) -> String {
        let body = (&self.receipt.hash, &self.kge, &self.kge_prev_hash);
        let bytes = serde_json::to_string(&body).expect("kge receipt body serialisable");
        content_hash(bytes.as_bytes())
    }
}

/// Append-only log of wrapped receipts. Holds the typesafe `ReceiptLog` (which
/// chains the inner receipts) plus the wrapper chain.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KgeReceiptLog {
    inner: ReceiptLog,
    receipts: Vec<KgeReceipt>,
}

impl KgeReceiptLog {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `receipt` (finalising its inner chain) wrapped with `kge`, and
    /// extend the wrapper chain. Returns the finalised wrapped receipt.
    pub fn push(&mut self, receipt: Receipt, kge: KgeReceiptExtra) -> &KgeReceipt {
        let finalized = self.inner.push(receipt).clone();
        let kge_prev_hash = self
            .receipts
            .last()
            .map_or_else(|| GENESIS_HASH.to_string(), |r| r.kge_hash.clone());
        let mut wrapped = KgeReceipt {
            receipt: finalized,
            kge,
            kge_prev_hash,
            kge_hash: String::new(),
        };
        wrapped.kge_hash = wrapped.digest();
        self.receipts.push(wrapped);
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

    pub fn iter(&self) -> impl Iterator<Item = &KgeReceipt> {
        self.receipts.iter()
    }

    /// One wrapped receipt per line (JSONL).
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        self.receipts
            .iter()
            .map(|r| serde_json::to_string(r).expect("kge receipt serialisable"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Parse a JSONL log, keeping every stored hash so `verify_chain` proves
    /// something. The inner `ReceiptLog` is rebuilt from the wrapped receipts on
    /// demand in `verify_chain`.
    pub fn from_jsonl(s: &str) -> Result<Self, String> {
        let mut receipts = Vec::new();
        for line in s.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let r: KgeReceipt =
                serde_json::from_str(line).map_err(|e| format!("kge receipt parse: {e}"))?;
            receipts.push(r);
        }
        Ok(Self {
            inner: ReceiptLog::new(),
            receipts,
        })
    }

    /// Verify both chains. `Err(i)` names the first wrapped receipt whose inner
    /// typesafe chain, wrapper link, or wrapper digest fails.
    pub fn verify_chain(&self) -> Result<(), usize> {
        // 1. Inner typesafe chain, rebuilt verbatim from the stored receipts.
        let inner_jsonl = self
            .receipts
            .iter()
            .map(|r| serde_json::to_string(&r.receipt).expect("inner serialisable"))
            .collect::<Vec<_>>()
            .join("\n");
        let inner = ReceiptLog::from_jsonl(&inner_jsonl).map_err(|_| 0usize)?;
        inner.verify_chain()?;

        // 2. Wrapper chain over (inner hash, kge extra, prev wrapper hash).
        let mut expected = GENESIS_HASH.to_string();
        for (i, r) in self.receipts.iter().enumerate() {
            if r.kge_prev_hash != expected || r.digest() != r.kge_hash {
                return Err(i);
            }
            expected = r.kge_hash.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_typesafe_core::loop_gate::{GateDecision, Proposal as TsProposal, ProposalKind};
    use ruvector_typesafe_core::receipt::{Metrics, TestStatistic};
    use ruvector_typesafe_core::Head;

    fn sample_receipt(champ_val: f32) -> Receipt {
        Receipt {
            seq: 0,
            proposal: TsProposal {
                id: 7,
                parent: None,
                kind: ProposalKind::ModelArm,
                description_hash: 0xdead_beef,
            },
            parent: None,
            kind: ProposalKind::ModelArm,
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
            model_id: "hole-d128".into(),
            head: Head::NearestPrototype,
            temperature: 1.0,
            budget_consumed: 1,
            created_seq: 0,
            created: None,
            prev_hash: String::new(),
            hash: String::new(),
            signature: None,
        }
    }

    fn extra(dims: usize) -> KgeReceiptExtra {
        KgeReceiptExtra {
            scorer: ScorerKind::Hole,
            dims,
            knobs_hash: 0xdead_beef,
            ewc_lambda: None,
            fisher_id: None,
            manifest_hash: None,
            lineage: None,
            sampling_temperature: Some(1.0),
        }
    }

    #[test]
    fn both_chains_verify_and_roundtrip() {
        let mut log = KgeReceiptLog::new();
        log.push(sample_receipt(0.90), extra(128));
        log.push(sample_receipt(0.91), extra(256));
        assert_eq!(log.len(), 2);
        assert!(log.verify_chain().is_ok());
        let jsonl = log.to_jsonl();
        let reloaded = KgeReceiptLog::from_jsonl(&jsonl).unwrap();
        assert!(reloaded.verify_chain().is_ok());
    }

    #[test]
    fn tampering_a_kge_field_is_caught() {
        let mut log = KgeReceiptLog::new();
        log.push(sample_receipt(0.90), extra(128));
        log.push(sample_receipt(0.91), extra(256));
        let jsonl = log.to_jsonl();
        // Flip a KGE-only field the inner receipt never sees.
        let tampered = jsonl.replacen("\"dims\":256", "\"dims\":512", 1);
        assert_ne!(tampered, jsonl);
        let reloaded = KgeReceiptLog::from_jsonl(&tampered).unwrap();
        assert_eq!(reloaded.verify_chain(), Err(1));
    }

    #[test]
    fn tampering_the_inner_receipt_is_caught() {
        let mut log = KgeReceiptLog::new();
        log.push(sample_receipt(0.90), extra(128));
        let jsonl = log.to_jsonl();
        // Flip an inner-receipt field (its content hash was taken over the old
        // value); the inner typesafe chain must reject it.
        let tampered = jsonl.replacen("hole-d128", "hole-d999", 1);
        assert_ne!(tampered, jsonl);
        let reloaded = KgeReceiptLog::from_jsonl(&tampered).unwrap();
        assert!(reloaded.verify_chain().is_err());
    }
}
