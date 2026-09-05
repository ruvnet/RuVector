//! TARL (Transaction-Aware Reliable Ledgers) executable memory operations,
//! ledger states, provenance records, and witness emission hooks.
//!
//! Implements the five-operation executable ledger from TARL
//! (Transaction-Aware Reliable Ledgers, arXiv:2608.03699) as scoped by
//! ADR-307 (PIR WP4). Every ledger state transition emits a witness record
//! whose fields follow ruvector's ADR-134 witness schema
//! (`docs/adr/ADR-134-witness-schema-log-format.md`); this slice defines the
//! serde-serializable record and a `WitnessSink` hook rather than depending
//! on the rvm crates directly (cross-repo anchoring is WP8).
//!
//! ## Tamper-evidence scope (read before relying on `verify_chain`)
//!
//! The witness chain uses keyless FNV-1a (ADR-134 §3 chooses it for speed,
//! explicitly not for cryptographic strength). The chain is therefore
//! tamper-EVIDENT against *accidental corruption and naive edits only*: an
//! adversary with write access to the log can recompute every `record_hash`
//! and relink every `prev_hash` in one O(n) pass, and FNV-1a's invertible
//! round function makes even an anchored head second-preimage-able in ~2^32
//! work. Real tamper evidence against a log-writing adversary requires
//! ADR-134's `WitnessSigner` escape hatch (§9, HMAC/Ed25519). Wiring a
//! `WitnessSigner` through `WitnessSink` is an explicit follow-up gate that
//! MUST land before WP8 cross-repo anchoring makes this log load-bearing
//! for acceptance decisions.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// The five TARL operations and ledger states
// ─────────────────────────────────────────────────────────────────────────────

/// The five executable TARL (Transaction-Aware Reliable Ledgers) memory
/// operations (arXiv:2608.03699). Every incoming statement maps to exactly
/// one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryOp {
    /// Admit a new statement into the ledger (enters `Pending`).
    Add,
    /// Record that a statement was seen and deliberately not retained.
    Ignore,
    /// A contradicting-but-credible statement supersedes an existing belief.
    ReviseOutdatedBelief,
    /// Mark an entry as unreliable; it moves to `Rejected`.
    RejectUnreliable,
    /// Park an entry for verification; it moves to (or stays in) `Pending`.
    DeferForVerification,
}

/// The three queryable ledger states maintained by TARL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LedgerState {
    /// Verified and usable by downstream inference.
    Accepted,
    /// Held for verification; not usable as accepted knowledge.
    Pending,
    /// Refused; retained only as history, never served as knowledge.
    Rejected,
}

impl LedgerState {
    /// Compact encoding used in the witness record `flags` field
    /// (0 = none, 1 = Accepted, 2 = Pending, 3 = Rejected).
    pub fn code(self) -> u8 {
        match self {
            LedgerState::Accepted => 1,
            LedgerState::Pending => 2,
            LedgerState::Rejected => 3,
        }
    }
}

/// What caused a transition record: one of the five TARL operations, the
/// proof-gated acceptance promotion, or a poisoning-containment demotion
/// cascaded from a revise/reject on an accepted dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionKind {
    /// One of the five TARL operations applied directly.
    Op(MemoryOp),
    /// `Pending -> Accepted`, admitted through the proof gate.
    Accept,
    /// Demotion of a dependent entry (`Accepted -> Pending`) triggered by a
    /// revise/reject on an entry it depends on.
    DependentDemotion,
}

// ─────────────────────────────────────────────────────────────────────────────
// Provenance
// ─────────────────────────────────────────────────────────────────────────────

/// Receipt returned by the proof gate for an acceptance transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceReceipt {
    /// Gate-local write sequence number.
    pub sequence: u64,
    /// The gate's chain commitment after admitting this acceptance.
    pub commitment: [u8; 32],
}

/// Full provenance for one ledger state transition. The complete history of
/// these records is retained and is sufficient to replay the ledger's final
/// state (see [`crate::ledger::replay_history`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRecord {
    /// Monotonic ledger-history sequence number.
    pub sequence: u64,
    /// Target entry, if any (`None` for `Ignore`, which retains no entry).
    pub entry_id: Option<u64>,
    /// The operation (or derived transition) that was applied.
    pub kind: TransitionKind,
    /// Entry state before this transition (`None` if the entry did not exist).
    pub prior_state: Option<LedgerState>,
    /// Entry state after this transition (`None` if no entry is retained).
    pub new_state: Option<LedgerState>,
    /// Wall-clock nanoseconds at commit time.
    pub timestamp_ns: u64,
    /// Identity of the actor that requested the transition.
    pub actor_id: String,
    /// Human-readable justification for the transition.
    pub reason: String,
    /// Proof-gate receipt (present only for `TransitionKind::Accept`).
    pub receipt: Option<AcceptanceReceipt>,
}

/// Errors returned by the transactional ledger.
#[derive(Debug)]
pub enum LedgerError {
    /// The referenced entry id does not exist.
    UnknownEntry(u64),
    /// The requested operation is not a legal transition from the entry's
    /// current state.
    InvalidTransition {
        entry_id: u64,
        from: LedgerState,
        op: &'static str,
    },
    /// An `add`/`revise` referenced a dependency id that does not exist.
    UnknownDependency { entry_id: u64, missing: u64 },
    /// Acceptance requires every dependency to already be `Accepted`.
    DependencyNotAccepted { entry_id: u64, dependency: u64 },
    /// The proof gate refused the acceptance.
    GateDenied(String),
    /// The witness sink refused the record; per ADR-134's "no witness, no
    /// mutation" invariant the transition was NOT applied.
    WitnessRejected(String),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::UnknownEntry(id) => write!(f, "unknown ledger entry {id}"),
            LedgerError::InvalidTransition { entry_id, from, op } => {
                write!(
                    f,
                    "invalid transition: {op} on entry {entry_id} in state {from:?}"
                )
            }
            LedgerError::UnknownDependency { entry_id, missing } => {
                write!(
                    f,
                    "entry {entry_id} references unknown dependency {missing}"
                )
            }
            LedgerError::DependencyNotAccepted {
                entry_id,
                dependency,
            } => {
                write!(
                    f,
                    "cannot accept entry {entry_id}: dependency {dependency} is not accepted"
                )
            }
            LedgerError::GateDenied(msg) => write!(f, "proof gate denied acceptance: {msg}"),
            LedgerError::WitnessRejected(msg) => {
                write!(f, "witness sink rejected record (mutation aborted): {msg}")
            }
        }
    }
}

impl std::error::Error for LedgerError {}

// ─────────────────────────────────────────────────────────────────────────────
// Witness records (ADR-134 schema, docs/adr/ADR-134-witness-schema-log-format.md)
// ─────────────────────────────────────────────────────────────────────────────

/// Action kinds for TARL ledger transitions.
///
/// ADR-134's `ActionKind` enum assigns 0x90-0x9F to vector/graph mutations
/// and leaves 0xA0+ unassigned; the schema is explicitly extensible without
/// breaking the record layout. This block claims 0xA0-0xA7 for the ledger.
pub mod action_kind {
    /// `MemoryOp::Add` — statement admitted as `Pending`.
    pub const LEDGER_ADD: u8 = 0xA0;
    /// `MemoryOp::Ignore` — statement seen, not retained.
    pub const LEDGER_IGNORE: u8 = 0xA1;
    /// `MemoryOp::ReviseOutdatedBelief` — old belief superseded.
    pub const LEDGER_REVISE: u8 = 0xA2;
    /// `MemoryOp::RejectUnreliable` — entry moved to `Rejected`.
    pub const LEDGER_REJECT: u8 = 0xA3;
    /// `MemoryOp::DeferForVerification` — entry parked in `Pending`.
    pub const LEDGER_DEFER: u8 = 0xA4;
    /// Proof-gated `Pending -> Accepted` promotion.
    pub const LEDGER_ACCEPT: u8 = 0xA5;
    /// Poisoning-containment demotion of a dependent entry.
    pub const LEDGER_DEPENDENT_DEMOTION: u8 = 0xA6;
    /// A compaction pass evicted an entry (ADR-341 mincut-gated forgetting;
    /// also applies to plain [`crate::compaction`] policies via
    /// [`crate::witnessed_compaction::compact_witnessed`]).
    pub const LEDGER_COMPACT_EVICT: u8 = 0xA7;
}

/// Evidence grade for a witness record, using the three-value vocabulary of
/// the program's canonical witness/receipt contract (ruflo ADR-322C, being
/// extracted into a formal spec in ruflo#3066):
///
/// - `Recomputed` — the recorded fact was re-derived from primary data
///   (e.g. an acceptance whose gate commitment is re-derivable from the
///   proof-gate chain).
/// - `SignatureVerified` — backed by a verified cryptographic signature
///   (not produced by this slice; reserved for WP8 RVM anchoring).
/// - `TrustedAssertion` — asserted by the acting ledger without independent
///   recomputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceGrade {
    Recomputed,
    SignatureVerified,
    TrustedAssertion,
}

impl EvidenceGrade {
    /// Compact code bound into the hashed `flags` bits
    /// (1 = recomputed, 2 = signature-verified, 3 = trusted-assertion).
    pub fn code(self) -> u8 {
        match self {
            EvidenceGrade::Recomputed => 1,
            EvidenceGrade::SignatureVerified => 2,
            EvidenceGrade::TrustedAssertion => 3,
        }
    }
}

/// FNV-1a hash, as specified for witness chaining by ADR-134 §3.
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// A ledger witness record. Field names, sizes, and byte layout follow the
/// 64-byte record of ADR-134 (`docs/adr/ADR-134-witness-schema-log-format.md`):
/// sequence, timestamp_ns, action_kind, proof_tier, flags, actor_partition_id,
/// target_object_id, capability_hash, payload, prev_hash, record_hash, aux.
///
/// Field usage for ledger transitions:
/// - `flags`: bits 0-7 new state code, bits 8-11 prior state code, bits
///   12-15 evidence grade code (extends ADR-134's RegionTierChange
///   "from in high byte, to in low byte" convention using the high byte's
///   spare nibble, so the evidence grade is part of the hash preimage).
/// - `actor_partition_id`: truncated FNV-1a of the actor id string.
/// - `target_object_id`: entry id (truncated to u32; `u32::MAX` when none).
/// - `capability_hash`: truncated FNV-1a of the gate commitment on `Accept`.
/// - `payload`: FNV-1a of the statement content when one is involved.
/// - `aux`: first 8 bytes (LE) of the gate commitment on `Accept`, else 0.
///
/// All fields are integers or unit-variant enums (no floats) and serde
/// serialization follows the stable declaration order below, keeping the
/// record canonical-JSON-friendly for the ruflo ADR-322C contract
/// (RFC 8785 JCS) — full shape conformance is a ruflo#3066 follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerWitnessRecord {
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub action_kind: u8,
    pub proof_tier: u8,
    pub flags: u16,
    pub actor_partition_id: u32,
    pub target_object_id: u32,
    pub capability_hash: u32,
    pub payload: u64,
    pub prev_hash: u64,
    pub record_hash: u64,
    pub aux: u64,
    /// ADR-322C evidence-grade annotation. Its code is also bound into the
    /// hashed `flags` bits 12-15, and [`MemoryWitnessLog::verify_chain`]
    /// cross-checks the two so a persisted record cannot advertise a
    /// stronger grade than its hash preimage records.
    pub evidence_grade: EvidenceGrade,
}

/// Pack prior/new state codes and the evidence grade into the witness
/// `flags` field (see [`LedgerWitnessRecord`] field docs for the layout).
pub fn pack_flags(
    prior: Option<LedgerState>,
    new: Option<LedgerState>,
    grade: EvidenceGrade,
) -> u16 {
    let prior = prior.map_or(0u16, |s| s.code() as u16);
    let new = new.map_or(0u16, |s| s.code() as u16);
    ((grade.code() as u16) << 12) | (prior << 8) | new
}

impl LedgerWitnessRecord {
    /// Canonical 64-byte little-endian encoding at the ADR-134 offsets.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut b = [0u8; 64];
        b[0..8].copy_from_slice(&self.sequence.to_le_bytes());
        b[8..16].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        b[16] = self.action_kind;
        b[17] = self.proof_tier;
        b[18..20].copy_from_slice(&self.flags.to_le_bytes());
        b[20..24].copy_from_slice(&self.actor_partition_id.to_le_bytes());
        b[24..28].copy_from_slice(&self.target_object_id.to_le_bytes());
        b[28..32].copy_from_slice(&self.capability_hash.to_le_bytes());
        b[32..40].copy_from_slice(&self.payload.to_le_bytes());
        b[40..48].copy_from_slice(&self.prev_hash.to_le_bytes());
        b[48..56].copy_from_slice(&self.record_hash.to_le_bytes());
        b[56..64].copy_from_slice(&self.aux.to_le_bytes());
        b
    }

    /// FNV-1a of bytes `[0..48]` — everything except `record_hash` and `aux`
    /// (plus the already-included `prev_hash`), per ADR-134 §3.
    pub fn compute_record_hash(&self) -> u64 {
        fnv1a(&self.to_bytes()[..48])
    }

    /// FNV-1a of the full 64 bytes; stored in the next record's `prev_hash`.
    pub fn chain_hash(&self) -> u64 {
        fnv1a(&self.to_bytes())
    }
}

/// Hook invoked for every ledger state transition, BEFORE the transition is
/// committed (ADR-134 "no witness, no mutation"): if the sink returns an
/// error, the ledger applies nothing.
///
/// # Atomicity contract
///
/// [`WitnessSink::emit_batch`] is all-or-nothing: on `Ok(())` every record
/// of the batch is durably retained; on `Err` the sink must retain NONE of
/// them. A sink whose medium can fail partway through (disk append, remote
/// log) must stage the batch internally (temp-file + atomic rename, single
/// transactional append, or compensating truncation of the partial prefix)
/// before reporting success. The ledger relies on this contract to keep its
/// chain cursors (`last_witness_hash` / `witness_seq`) and the persisted
/// log in lockstep — a sink that leaks a partial prefix on failure would
/// reintroduce orphan records, sequence reuse, and a permanent chain fork.
pub trait WitnessSink {
    /// Durably retain the whole batch atomically (see the trait-level
    /// atomicity contract). Batches are chained in order: `records[0]`'s
    /// `prev_hash` continues the sink's existing chain head.
    fn emit_batch(&mut self, records: &[LedgerWitnessRecord]) -> Result<(), LedgerError>;

    /// Convenience wrapper for a single record; delegates to
    /// [`WitnessSink::emit_batch`].
    fn emit(&mut self, record: &LedgerWitnessRecord) -> Result<(), LedgerError> {
        self.emit_batch(std::slice::from_ref(record))
    }
}

/// Default no-op sink: accepts every record and stores nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopWitnessSink;

impl WitnessSink for NoopWitnessSink {
    fn emit_batch(&mut self, _records: &[LedgerWitnessRecord]) -> Result<(), LedgerError> {
        Ok(())
    }
}

/// In-memory witness log that retains every emitted record and can verify
/// the hash chain independently of the writing ledger.
///
/// Alongside the records themselves the log maintains a *head commitment*
/// — the record count and the full-64-byte chain hash of the newest record
/// — updated atomically with every committed batch. [`Self::verify_chain`]
/// checks the walked chain terminates exactly at that commitment, which is
/// what makes tail truncation (rollback of the newest records) and
/// tampering with the newest record's `aux` field detectable. Export the
/// commitment via [`Self::head_commitment`] to anchor it out-of-band.
#[derive(Debug, Default, Clone)]
pub struct MemoryWitnessLog {
    pub records: Vec<LedgerWitnessRecord>,
    /// Number of records covered by the head commitment.
    pub committed_count: u64,
    /// Full-64-byte chain hash of the newest committed record (0 at genesis).
    /// Because [`LedgerWitnessRecord::chain_hash`] covers all 64 bytes, this
    /// commitment authenticates the last record's `aux` and `record_hash`
    /// fields, which the 48-byte ADR-134 `record_hash` preimage excludes.
    pub committed_head: u64,
}

impl MemoryWitnessLog {
    /// The head commitment as `(record_count, head_chain_hash)`. Anchor this
    /// pair out-of-band (or compare against a replica) to detect wholesale
    /// log replacement, which in-band verification cannot.
    pub fn head_commitment(&self) -> (u64, u64) {
        (self.committed_count, self.committed_head)
    }

    /// Walk the chain: each record's `record_hash` must re-derive from its
    /// own bytes, each `prev_hash` must equal the previous record's full
    /// chain hash (genesis `prev_hash` = 0), each record's serde
    /// `evidence_grade` must match the grade code bound into its hashed
    /// `flags` bits 12-15, and the walk must terminate exactly at the head
    /// commitment (count and chain hash).
    ///
    /// Detects *accidental corruption and naive edits*: bit flips,
    /// reordering, middle deletion, tail truncation, and edits to any field
    /// of any record (including the newest record's `aux`, covered by the
    /// head commitment). It is NOT proof against an adversary who can write
    /// the log: the chain is keyless FNV-1a, so such an adversary can
    /// relink it in one O(n) pass — see the module-level tamper-evidence
    /// note and the ADR-134 `WitnessSigner` follow-up.
    pub fn verify_chain(&self) -> bool {
        let mut prev = 0u64;
        for r in &self.records {
            let grade_in_flags = (r.flags >> 12) & 0xF;
            if r.prev_hash != prev
                || r.record_hash != r.compute_record_hash()
                || grade_in_flags != u16::from(r.evidence_grade.code())
            {
                return false;
            }
            prev = r.chain_hash();
        }
        self.records.len() as u64 == self.committed_count && prev == self.committed_head
    }
}

impl WitnessSink for MemoryWitnessLog {
    fn emit_batch(&mut self, records: &[LedgerWitnessRecord]) -> Result<(), LedgerError> {
        // Infallible in-memory append: the whole batch lands, then the head
        // commitment is advanced — trivially satisfying the atomicity
        // contract.
        self.records.extend_from_slice(records);
        self.committed_count = self.records.len() as u64;
        if let Some(last) = self.records.last() {
            self.committed_head = last.chain_hash();
        }
        Ok(())
    }
}
