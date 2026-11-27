//! `TileZeroGate`: a content-inspecting [`ProofGate`] built on
//! `cognitum-gate-tilezero`'s existing, tested `ThreeFilterDecision`
//! (structural min-cut + shift + evidence filters) — nightly research
//! 2026-11-10, `docs/research/nightly/2026-11-10-tilezero-gated-agent-memory-admission/`.
//!
//! ## Why this exists
//!
//! [`crate::ledger::AlwaysAdmitGate`] admits every acceptance unconditionally
//! and [`crate::ledger::WriteGateAdapter`] (feature `proof-gate`) only chains
//! receipts tamper-evidently — neither ever inspects *content*. This module
//! routes acceptance through `cognitum-gate-tilezero`'s three-filter
//! admission decision instead, so a candidate entry's content can actually
//! be judged against the ledger's recent admission history before it is
//! allowed to mutate the ledger.
//!
//! ## The single biggest ad hoc risk in this design — read this first
//!
//! [`ProofGate::admit`]'s signature is `fn admit(&mut self, entry_id: u64,
//! content: &[u8]) -> Result<AcceptanceReceipt, LedgerError>` — raw bytes,
//! no embedding channel. Changing that signature was out of scope (it is the
//! ledger's existing, unchanged admission contract). A real deployment would
//! almost certainly want to pass an upstream embedding it already computed
//! rather than deriving one from raw content bytes here. Lacking that
//! channel, [`embed_content`] derives a fixed-[`TILEZERO_EMBED_DIM`]
//! pseudo-embedding from the content bytes themselves using the
//! feature-hashing trick (Weinberger et al. 2009): every overlapping
//! 3-byte n-gram hashes (FNV-1a) to a dimension and a sign, so content
//! sharing many byte n-grams (e.g. two instances of the same template with
//! different fill-in numbers) hashes to similar vectors without a fitted
//! vocabulary or embedding model, while unrelated content hashes to
//! approximately orthogonal vectors (with the collision noise inherent to
//! any fixed-width hashing scheme). This is a legitimate, principled-if-crude
//! technique for turning bytes into a fixed-dimensional vector — but it is
//! **not a semantic embedding**: two paraphrases of the same idea in
//! different words will *not* score as similar under it. This experiment's
//! synthetic "coherent" vs "incoherent" streams are therefore constructed at
//! the byte/template level specifically to be separable by a byte-trigram
//! hashing vectorizer. That means this experiment substantially tests
//! *"can three threshold filters over a feature-hashed byte-similarity
//! signal separate byte-level-clustered vs. byte-level-unclustered synthetic
//! content"*, not *"can this gate detect semantically incoherent content in
//! a real embedding space."* We consider that a legitimate, honestly-scoped
//! first-order test of the wiring and decision logic — not a claim about
//! real semantic poisoning detection, which would need a real embedding
//! model upstream of this gate.
//!
//! ## How the per-admission graph is built
//!
//! `cognitum-gate-tilezero`'s [`ReducedGraph`] (see
//! `crates/cognitum-gate-tilezero/src/supergraph.rs`) is a scalar summary —
//! `global_cut`, `aggregate_shift_pressure`, `aggregate_evidence` — normally
//! populated by merging `WorkerReport`s from the 256-tile WASM fabric
//! (`merge.rs`/`ReportMerger`) or by `update_from_fragment`/
//! `update_coherence` from a live `WitnessFragment`/tile-coherence report.
//! None of that per-tile boundary-edge machinery exists for a single
//! ledger admission, so `TileZeroGate` instead sets the three scalars
//! directly (via `ReducedGraph`'s own public `set_global_cut` /
//! `set_shift_pressure` / `set_evidence` setters — the same setters the
//! crate's own unit tests use) from one underlying signal: the mean cosine
//! similarity between the candidate's embedding and a sliding window of the
//! ledger's `window_size` most-recently-*admitted* embeddings.
//!
//! Concretely, with `mean` = mean cosine similarity in `[-1, 1]` and
//! `mean01 = clamp01((mean + 1) / 2)` in `[0, 1]`:
//!
//! - `global_cut = mean01 * 10.0` (matches [`GateThresholds::default`]'s
//!   `min_cut = 5.0` sitting exactly at the "orthogonal to recent history"
//!   midpoint).
//! - `shift_pressure = 1.0 - mean01` — directly, "drift away from the
//!   recent baseline cluster," which is literally how this crate's own
//!   hypothesis names the quantity.
//! - `aggregate_evidence = mean01 * 150.0` (default `tau_permit = 100.0` is
//!   reachable only once mean01 exceeds ~0.67 — i.e. once a candidate is
//!   genuinely similar, not merely non-dissimilar, to recent history).
//!
//! **This is honestly a second, independent, ad hoc choice on top of the
//! embedding one above**: all three filters are deliberately monotonic
//! functions of the *same* single scalar (`mean01`), evaluated against
//! three different threshold scales. A real 256-tile deployment would have
//! genuinely independent structural (boundary-edge count), shift
//! (historical cut drift), and evidence (per-tile e-process) signals; this
//! single-admission integration collapses all three onto one signal because
//! there is no multi-tile fabric here to supply independent ones. The three
//! `GateThresholds` fields therefore do not test three independent
//! hypotheses about the candidate in this configuration — they test the same
//! similarity signal at three different cut points. This is the second
//! biggest external-validity gap in the experiment (after the embedding
//! itself) and is called out explicitly rather than presented as "three
//! independent filters" the way the native fabric's filters are.
//!
//! A cold-start admission (empty window) is permissive by construction: an
//! untouched [`ReducedGraph::new()`] already defaults to "high coherence,
//! high evidence, zero shift" (see `supergraph.rs`), so `TileZeroGate`
//! leaves the graph untouched in that case rather than special-casing it —
//! behaving like [`crate::ledger::AlwaysAdmitGate`] until the window has at
//! least one member.
//!
//! ## What happens to `GateDecision::Defer`
//!
//! [`ProofGate::admit`] is a binary admit-or-refuse boundary: there is no
//! "retry me later" return value. `TileZeroGate` therefore collapses
//! `GateDecision::Defer` into the same `Err(LedgerError::GateDenied(_))` as
//! `GateDecision::Deny` — the entry simply stays `Pending` and the caller
//! may call [`crate::ledger::TransactionalLedger::accept`] again later
//! (e.g. once the window has moved, or after human review). The finer-grained
//! decision (which filter rejected, and the three component scores) is not
//! lost — it is both embedded in the `GateDenied` message and retained for
//! introspection via [`TileZeroGate::last_outcome`] — but the ledger's own
//! state machine only ever sees "admitted" or "not admitted right now." This
//! is a real, named limitation of wiring a three-outcome gate behind a
//! two-outcome trait, not an oversight.

use std::collections::VecDeque;

use cognitum_gate_tilezero::{
    DecisionOutcome, GateDecision, GateThresholds, ReducedGraph, ThreeFilterDecision,
};

use crate::ledger::ProofGate;
use crate::ops::{fnv1a, AcceptanceReceipt, LedgerError};
use crate::scoring::cosine_sim;

/// Dimensionality of the feature-hashed pseudo-embedding [`embed_content`]
/// derives from raw content bytes. Chosen as a small power of two large
/// enough to keep hash-bucket collisions rare for the short (tens-to-low-
/// hundreds of bytes) statements this ledger's `content` strings typically
/// are, and small enough that a `window_size`-entry sliding window plus one
/// cosine-similarity scan per admission stays cheap (see the benchmark's
/// measured per-admission latency for the real cost at this dimension).
pub const TILEZERO_EMBED_DIM: usize = 64;

/// Default sliding-window size: how many of the most recently *admitted*
/// (not merely proposed) entries' embeddings `TileZeroGate` compares a new
/// candidate against. Large enough to average out single-entry noise in the
/// feature-hashed similarity signal; small enough that the O(window_size)
/// cosine-similarity scan per admission stays a small, bounded cost. Not
/// part of the Darwin-lite sweep (see the nightly research doc) — only
/// `GateThresholds` fields are mutated there, per the pre-registered search
/// space; this constant is a separate, documented, fixed design choice.
pub const TILEZERO_DEFAULT_WINDOW: usize = 64;

/// Derive a fixed-[`TILEZERO_EMBED_DIM`] pseudo-embedding from raw content
/// bytes via the feature-hashing trick: every overlapping 3-byte n-gram (or
/// the whole slice, for content shorter than 3 bytes) hashes with
/// [`crate::ops::fnv1a`] to a dimension (`hash % TILEZERO_EMBED_DIM`) and a
/// sign (the hash's top bit), and that signed unit is added into the
/// resulting vector, which is then L2-normalized. See the module doc's
/// "single biggest ad hoc risk" section for what this does and does not
/// capture. Returns the all-zero vector for empty content.
pub fn embed_content(content: &[u8]) -> Vec<f32> {
    let mut v = vec![0f32; TILEZERO_EMBED_DIM];
    if content.is_empty() {
        return v;
    }
    let n = content.len().min(3);
    for gram in content.windows(n) {
        let h = fnv1a(gram);
        let dim = (h as usize) % TILEZERO_EMBED_DIM;
        let sign = if (h >> 63) & 1 == 1 { 1.0f32 } else { -1.0f32 };
        v[dim] += sign;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// Scale from `mean01 in [0, 1]` onto the `ReducedGraph::global_cut` axis;
/// see the module doc for why `10.0` places `mean01 = 0.5` (an "orthogonal
/// to recent history" candidate) exactly at `GateThresholds::default()`'s
/// `min_cut = 5.0`.
const CUT_SCALE: f64 = 10.0;
/// Scale from `mean01 in [0, 1]` onto the `ReducedGraph::aggregate_evidence`
/// axis; see the module doc for why `150.0` requires `mean01 > 0.667` (true
/// similarity, not mere non-dissimilarity) to clear
/// `GateThresholds::default()`'s `tau_permit = 100.0`.
const EVIDENCE_SCALE: f64 = 150.0;

/// A content-inspecting [`ProofGate`] built on `cognitum-gate-tilezero`'s
/// [`ThreeFilterDecision`]. See the module doc for the full design, its
/// honestly-scoped limitations, and what `Defer` collapses to at this
/// trait's admit-or-refuse boundary.
pub struct TileZeroGate {
    decision: ThreeFilterDecision,
    window: VecDeque<Vec<f32>>,
    window_size: usize,
    seq: u64,
    last_outcome: Option<DecisionOutcome>,
}

impl TileZeroGate {
    /// A gate using [`TILEZERO_DEFAULT_WINDOW`] as its sliding-window size.
    pub fn new(thresholds: GateThresholds) -> Self {
        Self::with_window_size(thresholds, TILEZERO_DEFAULT_WINDOW)
    }

    /// A gate with an explicit sliding-window size (must be >= 1; `0` is
    /// treated as `1`).
    pub fn with_window_size(thresholds: GateThresholds, window_size: usize) -> Self {
        let window_size = window_size.max(1);
        Self {
            decision: ThreeFilterDecision::new(thresholds),
            window: VecDeque::with_capacity(window_size),
            window_size,
            seq: 0,
            last_outcome: None,
        }
    }

    /// Number of embeddings currently held in the sliding window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// The configured sliding-window capacity.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// The full [`DecisionOutcome`] (decision, confidence, per-filter
    /// scores, rejection reason) from the most recent [`ProofGate::admit`]
    /// call, for introspection and benchmarking — not retained by the
    /// ledger itself, since [`AcceptanceReceipt`] has no room for it.
    pub fn last_outcome(&self) -> Option<&DecisionOutcome> {
        self.last_outcome.as_ref()
    }

    /// The gate's configured thresholds.
    pub fn thresholds(&self) -> &GateThresholds {
        self.decision.thresholds()
    }

    /// Build the per-admission [`ReducedGraph`] for `candidate` against the
    /// current sliding window. See the module doc for the exact mapping.
    fn graph_for(&self, candidate: &[f32]) -> ReducedGraph {
        let mut graph = ReducedGraph::new();
        if self.window.is_empty() {
            // Cold start: nothing to compare against yet. ReducedGraph::new()
            // already defaults to high cut / high evidence / zero shift —
            // left untouched, matching AlwaysAdmitGate until the window has
            // at least one member.
            return graph;
        }
        let sims: Vec<f64> = self
            .window
            .iter()
            .map(|w| cosine_sim(w, candidate) as f64)
            .collect();
        let mean: f64 = sims.iter().sum::<f64>() / sims.len() as f64;
        let mean01 = ((mean + 1.0) / 2.0).clamp(0.0, 1.0);
        graph.set_global_cut(mean01 * CUT_SCALE);
        graph.set_shift_pressure(1.0 - mean01);
        graph.set_evidence(mean01 * EVIDENCE_SCALE);
        graph
    }
}

impl ProofGate for TileZeroGate {
    fn admit(&mut self, entry_id: u64, content: &[u8]) -> Result<AcceptanceReceipt, LedgerError> {
        let embedding = embed_content(content);
        let graph = self.graph_for(&embedding);
        let outcome = self.decision.evaluate(&graph);
        let decision = outcome.decision;
        self.last_outcome = Some(outcome.clone());

        match decision {
            GateDecision::Permit => {
                if self.window.len() >= self.window_size {
                    self.window.pop_front();
                }
                self.window.push_back(embedding);
                let sequence = self.seq;
                self.seq += 1;
                Ok(AcceptanceReceipt {
                    sequence,
                    commitment: commitment_from_outcome(entry_id, &outcome),
                })
            }
            GateDecision::Defer | GateDecision::Deny => Err(LedgerError::GateDenied(format!(
                "tilezero gate refused entry {entry_id}: {decision} (structural={:.4} \
                 shift={:.4} evidence={:.4} mincut={:.4}{})",
                outcome.structural_score,
                outcome.shift_score,
                outcome.evidence_score,
                outcome.mincut_value,
                outcome
                    .rejection_reason
                    .as_deref()
                    .map(|r| format!(" reason=\"{r}\""))
                    .unwrap_or_default(),
            ))),
        }
    }
}

/// A non-cryptographic, deterministic 32-byte commitment derived from the
/// admitted entry id and the decision's component scores — deliberately not
/// a security claim (same disclaimer as [`crate::ledger::AlwaysAdmitGate`]'s
/// zeroed commitment): it exists only so distinct admissions produce
/// distinguishable [`AcceptanceReceipt::commitment`] values for diagnostics,
/// not as a tamper-evidence or provenance primitive. Real tamper evidence
/// for acceptance still comes from the ledger's witness chain
/// ([`crate::ops::MemoryWitnessLog`]) and, if wrapped, [`crate::witness_signing::SignedWitnessSink`].
fn commitment_from_outcome(entry_id: u64, outcome: &DecisionOutcome) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(8 + 8 * 4);
    bytes.extend_from_slice(&entry_id.to_le_bytes());
    bytes.extend_from_slice(&outcome.structural_score.to_bits().to_le_bytes());
    bytes.extend_from_slice(&outcome.shift_score.to_bits().to_le_bytes());
    bytes.extend_from_slice(&outcome.evidence_score.to_bits().to_le_bytes());
    bytes.extend_from_slice(&outcome.mincut_value.to_bits().to_le_bytes());
    let h1 = fnv1a(&bytes);
    let h2 = fnv1a(&bytes[8..]);
    let mut commitment = [0u8; 32];
    commitment[0..8].copy_from_slice(&h1.to_le_bytes());
    commitment[8..16].copy_from_slice(&h2.to_le_bytes());
    commitment[16..24].copy_from_slice(&h1.rotate_left(17).to_le_bytes());
    commitment[24..32].copy_from_slice(&h2.rotate_right(13).to_le_bytes());
    commitment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_content_is_deterministic_and_normalized() {
        let a = embed_content(b"memory entry alpha variant 001 about topic coherence");
        let b = embed_content(b"memory entry alpha variant 001 about topic coherence");
        assert_eq!(a, b, "identical content must hash to an identical vector");
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
        assert!((cosine_sim(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn embed_content_empty_is_zero_vector() {
        let v = embed_content(b"");
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cold_start_permits_and_fills_window() {
        let mut gate = TileZeroGate::new(GateThresholds::default());
        assert_eq!(gate.window_len(), 0);
        let receipt = gate
            .admit(0, b"first ever entry")
            .expect("cold start permits");
        assert_eq!(receipt.sequence, 0);
        assert_eq!(gate.window_len(), 1);
        assert_eq!(gate.last_outcome().unwrap().decision, GateDecision::Permit);
    }

    #[test]
    fn window_slides_at_capacity() {
        let mut gate = TileZeroGate::with_window_size(GateThresholds::default(), 2);
        for i in 0..5u64 {
            // Re-admitting a fixed small set of near-identical templates
            // keeps every admission a Permit, so the window should always
            // be exactly at (not above) capacity once warmed.
            let _ = gate.admit(i, format!("steady template {}", i % 2).as_bytes());
        }
        assert!(gate.window_len() <= gate.window_size());
    }
}
