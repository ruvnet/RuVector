//! The TARL (Transaction-Aware Reliable Ledgers) transactional memory ledger
//! (ADR-307, PIR WP4).
//!
//! [`TransactionalLedger`] applies the five TARL operations as explicit
//! state transitions over accepted / pending / rejected entries, retaining
//! full provenance history. Every transition follows a stage -> witness ->
//! commit discipline: ALL witness records for an operation (ADR-134 schema)
//! are emitted to the [`WitnessSink`] as one atomic batch BEFORE any state
//! is applied, and a sink failure aborts the whole operation with no
//! observable partial application on either side — no ledger mutation, no
//! history append, and (per the [`WitnessSink::emit_batch`] contract) no
//! partial batch left in the sink ("no witness, no mutation" — and for a
//! failed batch, "no mutation, no witness" too). Note the witness chain is
//! keyless FNV-1a: tamper-evident against accidental corruption only, not
//! against a log-writing adversary — see the tamper-evidence note in
//! [`crate::ops`] and the ADR-134 `WitnessSigner` follow-up gate that must
//! land before WP8 anchoring. Acceptance transitions (`Pending -> Accepted`) must
//! additionally be admitted by a [`ProofGate`] — the same proof-gated write
//! machinery ruvector uses elsewhere (ADR-194-proof-gated-writes / ADR-047),
//! adapted behind a trait because this crate had no prior gate wiring; the
//! real `ruvector-proof-gate` implementation is available behind the
//! `proof-gate` feature via [`WriteGateAdapter`].

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ops::{
    action_kind, fnv1a, pack_flags, AcceptanceReceipt, EvidenceGrade, LedgerError, LedgerState,
    LedgerWitnessRecord, MemoryOp, NoopWitnessSink, TransitionKind, TransitionRecord, WitnessSink,
};

// ─────────────────────────────────────────────────────────────────────────────
// Proof gate
// ─────────────────────────────────────────────────────────────────────────────

/// Admission gate for acceptance transitions (`Pending -> Accepted`).
///
/// `ruvector-agent-memory` had no existing proof-gate usage, so acceptance is
/// wired behind this trait; the production implementation adapts
/// `ruvector-proof-gate`'s `WriteGate` (ADR-194-proof-gated-writes.md,
/// ADR-047-proof-gated-mutation-protocol.md) behind the `proof-gate` feature.
pub trait ProofGate {
    /// Admit the acceptance of `entry_id` with the given statement content.
    /// Returns a receipt committing the gate's chain state, or denies.
    fn admit(&mut self, entry_id: u64, content: &[u8]) -> Result<AcceptanceReceipt, LedgerError>;
}

/// Permissive default gate for development and tests: admits everything and
/// returns zeroed commitments. Do NOT use where poisoning resistance matters.
#[derive(Debug, Default)]
pub struct AlwaysAdmitGate {
    seq: u64,
}

impl ProofGate for AlwaysAdmitGate {
    fn admit(&mut self, _entry_id: u64, _content: &[u8]) -> Result<AcceptanceReceipt, LedgerError> {
        let sequence = self.seq;
        self.seq += 1;
        Ok(AcceptanceReceipt {
            sequence,
            commitment: [0u8; 32],
        })
    }
}

/// Adapter routing acceptance through `ruvector-proof-gate`'s `WriteGate`
/// (`HashChainGate` / `MerkleGate`) — the existing ADR-194/047 proof-gated
/// write machinery.
#[cfg(feature = "proof-gate")]
pub struct WriteGateAdapter<G: ruvector_proof_gate::WriteGate> {
    inner: G,
}

#[cfg(feature = "proof-gate")]
impl<G: ruvector_proof_gate::WriteGate> WriteGateAdapter<G> {
    pub fn new(inner: G) -> Self {
        Self { inner }
    }

    /// Access the wrapped gate (e.g. for offline receipt verification).
    pub fn gate(&self) -> &G {
        &self.inner
    }
}

#[cfg(feature = "proof-gate")]
impl<G: ruvector_proof_gate::WriteGate> ProofGate for WriteGateAdapter<G> {
    fn admit(&mut self, entry_id: u64, content: &[u8]) -> Result<AcceptanceReceipt, LedgerError> {
        let payload = ruvector_proof_gate::WritePayload::new(entry_id, Vec::new())
            .with_metadata(content.to_vec());
        let receipt = self
            .inner
            .admit(&payload)
            .map_err(|e| LedgerError::GateDenied(e.to_string()))?;
        Ok(AcceptanceReceipt {
            sequence: receipt.sequence,
            commitment: receipt.chain_commitment,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entries and staging
// ─────────────────────────────────────────────────────────────────────────────

/// A statement retained by the ledger, with its lightweight dependency edges.
#[derive(Debug, Clone)]
pub struct LedgerEntry {
    pub id: u64,
    pub content: String,
    pub state: LedgerState,
    /// Entries this one was inferred from. A revise/reject on any of them
    /// demotes this entry back to `Pending` (poisoning containment).
    pub depends_on: Vec<u64>,
    pub created_at_ns: u64,
}

/// State mutation staged for atomic application after witnessing.
enum Effect {
    Insert(LedgerEntry),
    SetState { id: u64, state: LedgerState },
    None,
}

struct Staged {
    record: TransitionRecord,
    witness: LedgerWitnessRecord,
    effect: Effect,
}

// ─────────────────────────────────────────────────────────────────────────────
// TransactionalLedger
// ─────────────────────────────────────────────────────────────────────────────

/// TARL transactional ledger over accepted / pending / rejected entries.
pub struct TransactionalLedger<S: WitnessSink, G: ProofGate> {
    entries: BTreeMap<u64, LedgerEntry>,
    /// Reverse dependency edges: id -> ids that depend on it.
    dependents: BTreeMap<u64, Vec<u64>>,
    history: Vec<TransitionRecord>,
    sink: S,
    gate: G,
    next_id: u64,
    witness_seq: u64,
    last_witness_hash: u64,
}

impl TransactionalLedger<NoopWitnessSink, AlwaysAdmitGate> {
    /// Ledger with no witness persistence and a permissive gate.
    /// Development/testing convenience only.
    pub fn unguarded() -> Self {
        Self::new(NoopWitnessSink, AlwaysAdmitGate::default())
    }
}

impl<S: WitnessSink, G: ProofGate> TransactionalLedger<S, G> {
    pub fn new(sink: S, gate: G) -> Self {
        Self {
            entries: BTreeMap::new(),
            dependents: BTreeMap::new(),
            history: Vec::new(),
            sink,
            gate,
            next_id: 0,
            witness_seq: 0,
            last_witness_hash: 0,
        }
    }

    // ── The five TARL operations ─────────────────────────────────────────────

    /// `add`: admit a new statement. It enters `Pending`; only [`Self::accept`]
    /// (through the proof gate) can promote it to `Accepted`.
    pub fn add(
        &mut self,
        content: impl Into<String>,
        depends_on: &[u64],
        actor: &str,
        reason: &str,
    ) -> Result<u64, LedgerError> {
        let content = content.into();
        let id = self.next_id;
        for &dep in depends_on {
            if !self.entries.contains_key(&dep) {
                return Err(LedgerError::UnknownDependency {
                    entry_id: id,
                    missing: dep,
                });
            }
        }
        let now = now_ns();
        let entry = LedgerEntry {
            id,
            content: content.clone(),
            state: LedgerState::Pending,
            depends_on: depends_on.to_vec(),
            created_at_ns: now,
        };
        let staged = self.stage(
            TransitionKind::Op(MemoryOp::Add),
            Some(id),
            None,
            Some(LedgerState::Pending),
            actor,
            reason,
            None,
            action_kind::LEDGER_ADD,
            1,
            EvidenceGrade::TrustedAssertion,
            fnv1a(content.as_bytes()),
            0,
            Effect::Insert(entry),
            now,
            0,
        );
        self.emit_and_apply(vec![staged])?;
        self.next_id += 1;
        Ok(id)
    }

    /// `ignore`: record that a statement was seen and deliberately not
    /// retained. No entry is created; the decision itself is witnessed.
    pub fn ignore(&mut self, content: &str, actor: &str, reason: &str) -> Result<(), LedgerError> {
        let staged = self.stage(
            TransitionKind::Op(MemoryOp::Ignore),
            None,
            None,
            None,
            actor,
            reason,
            None,
            action_kind::LEDGER_IGNORE,
            1,
            EvidenceGrade::TrustedAssertion,
            fnv1a(content.as_bytes()),
            0,
            Effect::None,
            now_ns(),
            0,
        );
        self.emit_and_apply(vec![staged])
    }

    /// `revise-outdated-belief`: a contradicting-but-credible statement
    /// supersedes `entry_id`. The old belief moves to `Rejected`; if it was
    /// `Accepted`, every (transitive) accepted dependent is demoted to
    /// `Pending`. The replacement enters `Pending` with the old entry's
    /// dependency edges and must pass the gate to be accepted.
    ///
    /// Returns the replacement entry's id.
    pub fn revise_outdated_belief(
        &mut self,
        entry_id: u64,
        new_content: impl Into<String>,
        actor: &str,
        reason: &str,
    ) -> Result<u64, LedgerError> {
        let old = self
            .entries
            .get(&entry_id)
            .ok_or(LedgerError::UnknownEntry(entry_id))?;
        if old.state == LedgerState::Rejected {
            return Err(LedgerError::InvalidTransition {
                entry_id,
                from: old.state,
                op: "revise-outdated-belief",
            });
        }
        let was_accepted = old.state == LedgerState::Accepted;
        let depends_on = old.depends_on.clone();
        let prior = old.state;
        let new_content = new_content.into();
        let new_id = self.next_id;
        let now = now_ns();

        let mut staged = vec![self.stage(
            TransitionKind::Op(MemoryOp::ReviseOutdatedBelief),
            Some(entry_id),
            Some(prior),
            Some(LedgerState::Rejected),
            actor,
            reason,
            None,
            action_kind::LEDGER_REVISE,
            1,
            EvidenceGrade::TrustedAssertion,
            fnv1a(new_content.as_bytes()),
            new_id,
            Effect::SetState {
                id: entry_id,
                state: LedgerState::Rejected,
            },
            now,
            0,
        )];
        if was_accepted {
            self.stage_demotions(entry_id, actor, &mut staged, now);
        }
        let idx = staged.len();
        staged.push(self.stage(
            TransitionKind::Op(MemoryOp::Add),
            Some(new_id),
            None,
            Some(LedgerState::Pending),
            actor,
            &format!("revision of entry {entry_id}"),
            None,
            action_kind::LEDGER_ADD,
            1,
            EvidenceGrade::TrustedAssertion,
            fnv1a(new_content.as_bytes()),
            entry_id,
            Effect::Insert(LedgerEntry {
                id: new_id,
                content: new_content,
                state: LedgerState::Pending,
                depends_on,
                created_at_ns: now,
            }),
            now,
            idx as u64,
        ));
        self.emit_and_apply(staged)?;
        self.next_id += 1;
        Ok(new_id)
    }

    /// `reject-unreliable`: move `entry_id` to `Rejected`. If it was
    /// `Accepted`, every (transitive) accepted dependent is demoted to
    /// `Pending` so no downstream inference silently stays accepted on top
    /// of a rejected premise.
    pub fn reject_unreliable(
        &mut self,
        entry_id: u64,
        actor: &str,
        reason: &str,
    ) -> Result<(), LedgerError> {
        let entry = self
            .entries
            .get(&entry_id)
            .ok_or(LedgerError::UnknownEntry(entry_id))?;
        if entry.state == LedgerState::Rejected {
            return Err(LedgerError::InvalidTransition {
                entry_id,
                from: entry.state,
                op: "reject-unreliable",
            });
        }
        let prior = entry.state;
        let now = now_ns();
        let mut staged = vec![self.stage(
            TransitionKind::Op(MemoryOp::RejectUnreliable),
            Some(entry_id),
            Some(prior),
            Some(LedgerState::Rejected),
            actor,
            reason,
            None,
            action_kind::LEDGER_REJECT,
            1,
            EvidenceGrade::TrustedAssertion,
            0,
            0,
            Effect::SetState {
                id: entry_id,
                state: LedgerState::Rejected,
            },
            now,
            0,
        )];
        if prior == LedgerState::Accepted {
            self.stage_demotions(entry_id, actor, &mut staged, now);
        }
        self.emit_and_apply(staged)
    }

    /// `defer-for-verification`: park `entry_id` in `Pending`. A `Pending`
    /// entry stays pending (the deferral itself is witnessed); an `Accepted`
    /// entry is demoted to `Pending` and its accepted dependents are demoted
    /// with it. Round-trip back to `Accepted` goes through [`Self::accept`].
    pub fn defer_for_verification(
        &mut self,
        entry_id: u64,
        actor: &str,
        reason: &str,
    ) -> Result<(), LedgerError> {
        let entry = self
            .entries
            .get(&entry_id)
            .ok_or(LedgerError::UnknownEntry(entry_id))?;
        if entry.state == LedgerState::Rejected {
            return Err(LedgerError::InvalidTransition {
                entry_id,
                from: entry.state,
                op: "defer-for-verification",
            });
        }
        let prior = entry.state;
        let now = now_ns();
        let mut staged = vec![self.stage(
            TransitionKind::Op(MemoryOp::DeferForVerification),
            Some(entry_id),
            Some(prior),
            Some(LedgerState::Pending),
            actor,
            reason,
            None,
            action_kind::LEDGER_DEFER,
            1,
            EvidenceGrade::TrustedAssertion,
            0,
            0,
            Effect::SetState {
                id: entry_id,
                state: LedgerState::Pending,
            },
            now,
            0,
        )];
        if prior == LedgerState::Accepted {
            self.stage_demotions(entry_id, actor, &mut staged, now);
        }
        self.emit_and_apply(staged)
    }

    // ── Proof-gated acceptance ───────────────────────────────────────────────

    /// Promote a `Pending` entry to `Accepted`. The transition must be
    /// admitted by the proof gate (ADR-194/047 machinery) and requires every
    /// dependency to already be `Accepted`. Returns the gate receipt, which
    /// is also retained in the transition's provenance record.
    pub fn accept(
        &mut self,
        entry_id: u64,
        actor: &str,
        reason: &str,
    ) -> Result<AcceptanceReceipt, LedgerError> {
        let entry = self
            .entries
            .get(&entry_id)
            .ok_or(LedgerError::UnknownEntry(entry_id))?;
        if entry.state != LedgerState::Pending {
            return Err(LedgerError::InvalidTransition {
                entry_id,
                from: entry.state,
                op: "accept",
            });
        }
        for &dep in &entry.depends_on {
            if self.entries[&dep].state != LedgerState::Accepted {
                return Err(LedgerError::DependencyNotAccepted {
                    entry_id,
                    dependency: dep,
                });
            }
        }
        let content = entry.content.clone();
        // Gate first: a denied acceptance mutates nothing and witnesses
        // nothing (the denial surfaces as an error to the caller).
        //
        // Known asymmetry (granted-then-unwitnessed): the gate must be
        // consulted BEFORE staging because the witness record binds the
        // gate's receipt (`aux` carries the commitment prefix,
        // `capability_hash` its digest). If the witness sink then refuses
        // the batch, the gate has already advanced its own chain/sequence
        // (`WriteGate::admit` mutates on admission and offers no rollback),
        // while nothing is witnessed or applied. Gate sequence numbers can
        // therefore run AHEAD of the ledger's committed `Accept` records;
        // during reconciliation, gate admissions with no corresponding
        // `Accept` witness/history record are expected benign gaps, not
        // evidence of a bypass (a bypass would be the opposite: an `Accept`
        // record with no gate admission).
        let receipt = self.gate.admit(entry_id, content.as_bytes())?;
        let aux = u64::from_le_bytes(receipt.commitment[..8].try_into().unwrap());
        let staged = self.stage(
            TransitionKind::Accept,
            Some(entry_id),
            Some(LedgerState::Pending),
            Some(LedgerState::Accepted),
            actor,
            reason,
            Some(receipt.clone()),
            action_kind::LEDGER_ACCEPT,
            2,
            EvidenceGrade::Recomputed,
            fnv1a(content.as_bytes()),
            aux,
            Effect::SetState {
                id: entry_id,
                state: LedgerState::Accepted,
            },
            now_ns(),
            0,
        );
        self.emit_and_apply(vec![staged])?;
        Ok(receipt)
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    pub fn entry(&self, id: u64) -> Option<&LedgerEntry> {
        self.entries.get(&id)
    }

    pub fn state_of(&self, id: u64) -> Option<LedgerState> {
        self.entries.get(&id).map(|e| e.state)
    }

    /// All entries currently in `state` (each ledger is distinctly queryable).
    pub fn entries_in(&self, state: LedgerState) -> Vec<&LedgerEntry> {
        self.entries.values().filter(|e| e.state == state).collect()
    }

    /// Current state of every retained entry.
    pub fn states(&self) -> BTreeMap<u64, LedgerState> {
        self.entries.iter().map(|(&id, e)| (id, e.state)).collect()
    }

    /// Full provenance history, in commit order.
    pub fn history(&self) -> &[TransitionRecord] {
        &self.history
    }

    pub fn witness_sink(&self) -> &S {
        &self.sink
    }

    /// Consume the ledger and recover ownership of its witness sink, e.g.
    /// to call `SignedWitnessSink::flush` and inspect signed spans once a
    /// run is done. Ledger state (`entries`, `history`) is discarded; the
    /// sink already holds everything durable.
    pub fn into_witness_sink(self) -> S {
        self.sink
    }

    /// Read access to the acceptance gate (e.g. for offline receipt or
    /// chain-integrity verification).
    pub fn proof_gate(&self) -> &G {
        &self.gate
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ── Internals ────────────────────────────────────────────────────────────

    /// Stage the poisoning-containment cascade: BFS over reverse dependency
    /// edges from `root`, demoting every currently-`Accepted` dependent to
    /// `Pending` (transitively — a demoted dependent's own accepted
    /// dependents are demoted too).
    fn stage_demotions(&self, root: u64, actor: &str, staged: &mut Vec<Staged>, now: u64) {
        let mut demoted: Vec<u64> = Vec::new();
        let mut queue = vec![root];
        while let Some(id) = queue.pop() {
            for &dep_id in self.dependents.get(&id).map_or(&[][..], |v| &v[..]) {
                if demoted.contains(&dep_id) {
                    continue;
                }
                if self.entries[&dep_id].state == LedgerState::Accepted {
                    demoted.push(dep_id);
                    queue.push(dep_id);
                }
            }
        }
        demoted.sort_unstable();
        for (i, dep_id) in demoted.into_iter().enumerate() {
            staged.push(self.stage(
                TransitionKind::DependentDemotion,
                Some(dep_id),
                Some(LedgerState::Accepted),
                Some(LedgerState::Pending),
                actor,
                &format!("dependency {root} left the accepted ledger"),
                None,
                action_kind::LEDGER_DEPENDENT_DEMOTION,
                1,
                EvidenceGrade::TrustedAssertion,
                root,
                0,
                Effect::SetState {
                    id: dep_id,
                    state: LedgerState::Pending,
                },
                now,
                (i + 1) as u64,
            ));
        }
    }

    /// Build one staged transition. `seq_offset` positions the record within
    /// a multi-transition operation (history/witness sequences are assigned
    /// relative to the current cursors; nothing is committed here).
    #[allow(clippy::too_many_arguments)]
    fn stage(
        &self,
        kind: TransitionKind,
        entry_id: Option<u64>,
        prior_state: Option<LedgerState>,
        new_state: Option<LedgerState>,
        actor: &str,
        reason: &str,
        receipt: Option<AcceptanceReceipt>,
        action: u8,
        proof_tier: u8,
        grade: EvidenceGrade,
        payload: u64,
        aux: u64,
        effect: Effect,
        now: u64,
        seq_offset: u64,
    ) -> Staged {
        let capability_hash = receipt
            .as_ref()
            .map_or(0u32, |r| (fnv1a(&r.commitment) & 0xFFFF_FFFF) as u32);
        let mut witness = LedgerWitnessRecord {
            sequence: self.witness_seq + seq_offset,
            timestamp_ns: now,
            action_kind: action,
            proof_tier,
            flags: pack_flags(prior_state, new_state, grade),
            actor_partition_id: (fnv1a(actor.as_bytes()) & 0xFFFF_FFFF) as u32,
            target_object_id: entry_id.map_or(u32::MAX, |id| id as u32),
            capability_hash,
            payload,
            prev_hash: 0, // fixed up during emit_and_apply chaining
            record_hash: 0,
            aux,
            evidence_grade: grade,
        };
        witness.record_hash = 0; // recomputed after prev_hash is chained
        Staged {
            record: TransitionRecord {
                sequence: self.history.len() as u64 + seq_offset,
                entry_id,
                kind,
                prior_state,
                new_state,
                timestamp_ns: now,
                actor_id: actor.to_string(),
                reason: reason.to_string(),
                receipt,
            },
            witness,
            effect,
        }
    }

    /// Chain + emit every staged witness record as ONE atomic batch, then
    /// apply all effects. If the batch is refused, NO record is durably
    /// retained (the [`WitnessSink::emit_batch`] atomicity contract) and NO
    /// state is applied — sink and ledger observe either the whole
    /// operation or none of it, so a failed batch can neither orphan
    /// records, reuse sequence numbers, nor fork the persisted chain.
    fn emit_and_apply(&mut self, mut staged: Vec<Staged>) -> Result<(), LedgerError> {
        // Chain the witness records off the current head.
        let mut prev = self.last_witness_hash;
        for s in &mut staged {
            s.witness.prev_hash = prev;
            s.witness.record_hash = s.witness.compute_record_hash();
            prev = s.witness.chain_hash();
        }
        // Witness first: no witness, no mutation. The whole batch commits
        // atomically or not at all; chain cursors advance only after that.
        let witnesses: Vec<LedgerWitnessRecord> = staged.iter().map(|s| s.witness).collect();
        self.sink.emit_batch(&witnesses)?;
        // Commit.
        self.last_witness_hash = prev;
        self.witness_seq += staged.len() as u64;
        for s in staged {
            match s.effect {
                Effect::Insert(entry) => {
                    for &dep in &entry.depends_on {
                        self.dependents.entry(dep).or_default().push(entry.id);
                    }
                    self.entries.insert(entry.id, entry);
                }
                Effect::SetState { id, state } => {
                    self.entries
                        .get_mut(&id)
                        .expect("staged entry must exist")
                        .state = state;
                }
                Effect::None => {}
            }
            self.history.push(s.record);
        }
        Ok(())
    }
}

/// Reconstruct final per-entry states purely from a provenance history.
/// Replaying [`TransactionalLedger::history`] must reproduce
/// [`TransactionalLedger::states`] exactly.
pub fn replay_history(history: &[TransitionRecord]) -> BTreeMap<u64, LedgerState> {
    let mut states = BTreeMap::new();
    for record in history {
        if let (Some(id), Some(state)) = (record.entry_id, record.new_state) {
            states.insert(id, state);
        }
    }
    states
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
