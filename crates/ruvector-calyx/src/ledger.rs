//! Grounding anchors and the hash-chained provenance ledger.
//!
//! Two of Calyx's three trust principles live here: *grounding is mandatory*
//! and every answer must be replayable. An [`Anchor`] ties a record to a
//! real-world outcome (the paper's *Lodestar* grounding kernels); the
//! [`Ledger`] (the paper's *Ledger*) records each query as a hash-chained entry
//! so the full answer path — input, lenses, retrieval, output — can be
//! reproduced and audited.

use crate::scoring::{fnv1a64, mix_hash};

/// The kind of real-world evidence an anchor represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorKind {
    /// A human-accepted answer.
    AcceptedAnswer,
    /// A test that passed against this record.
    PassedTest,
    /// A sensor reading / ground-truth label.
    SensorLabel,
    /// A citation to an authoritative source.
    Citation,
    /// A reward signal (e.g. RL / preference).
    Reward,
}

impl AnchorKind {
    fn as_str(self) -> &'static str {
        match self {
            AnchorKind::AcceptedAnswer => "accepted_answer",
            AnchorKind::PassedTest => "passed_test",
            AnchorKind::SensorLabel => "sensor_label",
            AnchorKind::Citation => "citation",
            AnchorKind::Reward => "reward",
        }
    }
}

/// A grounding anchor: evidence with a confidence weight in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Anchor {
    pub kind: AnchorKind,
    /// Free-form reference to the evidence (doc id, test name, sensor id, ...).
    pub reference: String,
    /// Confidence in `[0, 1]`.
    pub weight: f32,
}

impl Anchor {
    pub fn new(kind: AnchorKind, reference: impl Into<String>, weight: f32) -> Self {
        Self {
            kind,
            reference: reference.into(),
            weight: weight.clamp(0.0, 1.0),
        }
    }

    pub fn hash(&self) -> u64 {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.kind.as_str().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.reference.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&self.weight.to_bits().to_le_bytes());
        fnv1a64(&bytes)
    }
}

/// The strongest grounded path supporting a candidate, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundingPath {
    pub record_id: u64,
    pub best_kind: AnchorKind,
    pub best_weight: f32,
}

/// Return the best grounded path for `anchors` whose weight clears
/// `min_weight`, or `None` if the record has no sufficient grounding.
pub fn grounded_path(record_id: u64, anchors: &[Anchor], min_weight: f32) -> Option<GroundingPath> {
    anchors
        .iter()
        .filter(|a| a.weight >= min_weight)
        .max_by(|a, b| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|a| GroundingPath {
            record_id,
            best_kind: a.kind,
            best_weight: a.weight,
        })
}

/// One hash-chained record of an answer path.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerEntry {
    pub seq: u64,
    pub prev_hash: u64,
    /// Hash over the query's input slots.
    pub input_hash: u64,
    /// Folded hash over the fingerprints of every lens consulted.
    pub lens_hash: u64,
    /// Hash over the retrieval result (ranked ids).
    pub retrieval_hash: u64,
    /// Hash over the emitted decision.
    pub output_hash: u64,
    /// Chain hash: `mix(prev_hash, mix(... folded entry fields ...))`.
    pub hash: u64,
}

impl LedgerEntry {
    fn body_hash(&self) -> u64 {
        let mut h = self.input_hash;
        h = mix_hash(h, self.lens_hash);
        h = mix_hash(h, self.retrieval_hash);
        h = mix_hash(h, self.output_hash);
        h = mix_hash(h, self.seq);
        h
    }
}

/// Append-only hash-chained ledger of answer paths.
#[derive(Debug, Default)]
pub struct Ledger {
    entries: Vec<LedgerEntry>,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The chain head hash (`0` for an empty ledger).
    pub fn head(&self) -> u64 {
        self.entries.last().map(|e| e.hash).unwrap_or(0)
    }

    /// Append an entry, chaining it onto the current head.
    pub fn append(
        &mut self,
        input_hash: u64,
        lens_hash: u64,
        retrieval_hash: u64,
        output_hash: u64,
    ) -> u64 {
        let seq = self.entries.len() as u64;
        let prev_hash = self.head();
        let mut entry = LedgerEntry {
            seq,
            prev_hash,
            input_hash,
            lens_hash,
            retrieval_hash,
            output_hash,
            hash: 0,
        };
        entry.hash = mix_hash(prev_hash, entry.body_hash());
        self.entries.push(entry);
        self.head()
    }

    /// Replay the whole chain, recomputing every link. Returns `true` iff the
    /// ledger is internally consistent (100% replayable).
    pub fn verify(&self) -> bool {
        let mut prev = 0u64;
        for (i, e) in self.entries.iter().enumerate() {
            if e.seq != i as u64 || e.prev_hash != prev {
                return false;
            }
            if e.hash != mix_hash(e.prev_hash, e.body_hash()) {
                return false;
            }
            prev = e.hash;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grounding_picks_strongest_above_threshold() {
        let anchors = vec![
            Anchor::new(AnchorKind::Citation, "doc-1", 0.4),
            Anchor::new(AnchorKind::PassedTest, "test-7", 0.9),
        ];
        let p = grounded_path(3, &anchors, 0.5).unwrap();
        assert_eq!(p.best_kind, AnchorKind::PassedTest);
        assert!(grounded_path(3, &anchors, 0.95).is_none());
    }

    #[test]
    fn ledger_chains_and_verifies() {
        let mut l = Ledger::new();
        l.append(1, 2, 3, 4);
        l.append(5, 6, 7, 8);
        assert!(l.verify());
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn tamper_breaks_verification() {
        let mut l = Ledger::new();
        l.append(1, 2, 3, 4);
        l.append(5, 6, 7, 8);
        l.entries[0].output_hash ^= 0xdead; // tamper
        assert!(!l.verify());
    }
}
