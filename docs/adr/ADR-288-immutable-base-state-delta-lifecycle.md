# ADR-288: Immutable Base RVF and Encrypted State Delta Lifecycle

- **Status**: Accepted
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — partial implementation: Reader state capsules are encrypted (ChaCha20-Poly1305), per-install keyed, base-RVF lineage-bound with mismatch rejection; base immutability by construction. CompressedCheckpoint/WitnessDelta reconstruction and branch/merge/migrate are cross-repo rvm-state scope.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-285, ADR-286, ADR-287, ADR-289
- **Tags**: rvf, state, deltas, checkpoints, lineage, encryption, updates, witness

## Context

A canonical `.rvf` artifact is signed once and then distributed inside Windows,
macOS, Linux, and RVM packages. The core invariant is that the embedded RVF
hash is identical across every platform package and that mutable state never
modifies the signed base identity.

Agents are not stateless. They accumulate memory, tool results, conversation
history, and execution checkpoints. If that state were written back into the
artifact, three things break at once: the cross-platform hash equality that
Forge's acceptance test depends on, the signature that the Reader verifies
before loading executable segments, and the provenance chain that ties an
installed package back to a specific input RVF and builder identity.

The runtime also has to survive restart, suspend on one backend and resume on
another, and reconstruct state from a checkpoint plus witness deltas — while
still rejecting state that came from a different RVF lineage. Meanwhile the
enterprise posture requires that a customer can hold their own state
encryption keys, and that deleting an agent's accumulated state does not
require deleting or rebuilding the base artifact.

Separately, updates must be signed, version constrained, reversible, and linked
to the prior RVF identity. An update is therefore not an in-place mutation of a
base artifact; it is a new base artifact with a recorded relationship to its
predecessor, and existing state must have a defined disposition across that
transition.

## Decision

The base RVF is immutable. All mutable runtime state lives in encrypted RVF
delta segments that are cryptographically bound to the base RVF identity, are
reconstructible from a `CompressedCheckpoint` plus `WitnessDelta` records, and
are deletable independently of the base.

### 1. The base artifact is read-only for the lifetime of its identity

The base RVF is never written to after signing. The Reader opens it read-only,
verifies signatures before loading executable segments, and treats any
in-place modification as tampering that prevents execution.

This makes the base hash a stable identity: the same bytes are embedded in the
`.exe`, `.msi`, `.dmg`, `.deb`, `.rpm`, and `.AppImage` outputs, and the
installed Reader on any platform computes the same value.

### 2. State is a chain of encrypted delta segments

Runtime state is stored as RVF delta segments, separate from the base artifact
and separate from the installer payload:

```text
base.rvf  (immutable, signed)
   └─ state/
        checkpoint-000  CompressedCheckpoint
        delta-001       WitnessDelta (encrypted)
        delta-002       WitnessDelta (encrypted)
        ...
```

Delta segments use the RVF segment format, so the same verification, encryption,
and progressive-loading machinery applies to state as to model and runtime
segments. Deltas are encrypted at rest by default.

### 3. `CompressedCheckpoint` for snapshots, `WitnessDelta` for reconstruction

Two distinct record types serve two distinct purposes:

- **`CompressedCheckpoint`** — a compacted execution snapshot. It captures the
  runtime instance state needed to resume: component linear memory, quota
  consumption, and the deterministic clock and randomness cursors defined in
  ADR-287. A checkpoint is a materialization, not a log.
- **`WitnessDelta`** — the incremental, witness-linked record of state changes
  since the last checkpoint. Deltas are what make reconstruction verifiable:
  replaying them onto a checkpoint yields a state whose derivation is auditable
  rather than asserted.

Reconstruction is `checkpoint + ordered witness deltas`. Compaction folds a
delta run into a new checkpoint; the checkpoint records which delta range it
subsumes, so a verifier can confirm that compaction did not drop or invent
state. Both record types are versioned by `stateSchemaVersion` in the Forge
integration contract, and both are covered by `witnessSchemaVersion`.

### 4. Lineage binding and rejection

Every delta and every checkpoint carries the base RVF identity it belongs to.

On open, the runtime compares the recorded base identity against the base RVF
it has actually loaded. A mismatch is a **lineage rejection**: the state is
refused, execution does not begin with partial state, and the rejection is
recorded in the witness chain.

Lineage rejection covers the cases that matter operationally: state copied
between two different agents, state carried across an unrelated RVF that
happens to have the same application name, and state that survives a base
artifact substitution attack. Modified runtime, policy, model, and state
segments are all rejected on this same principle — the acceptance test requires
each of them independently.

Lineage is a chain, not a single value. When an update establishes a new base
identity (§7), the new base records its predecessor, so state whose recorded
base identity is an accepted ancestor of the loaded base can be migrated
forward under §5 rather than rejected.

### 5. State operations

The runtime exposes five state operations, each producing witness records:

| Operation | Meaning |
|---|---|
| **branch** | fork the delta chain at a checkpoint, producing an independent line of state that shares history up to the fork point |
| **rollback** | discard deltas after a chosen checkpoint or delta index, returning to an earlier reconstructible state |
| **merge** | combine two branches into one chain under a declared conflict-resolution rule |
| **migrate** | carry state forward across a base RVF update or across a backend, re-encoding to the target `stateSchemaVersion` |
| **reset** | discard all deltas and checkpoints, returning the agent to the base artifact's initial state |

None of these operations touches the base artifact. `reset` and `rollback` are
implemented by discarding state records, not by rewriting the base.

Branching and merging operate on the delta chain, so a branch is cheap: it is a
new chain head referencing a shared checkpoint, not a copy of accumulated state.

### 6. Deletion and customer-controlled keys

**Deletion.** State can be deleted without deleting the base RVF. Uninstallation
removes the runtime and state according to policy, and the base artifact's
disposition is a separate policy decision from the state's. An operator can
purge an agent's accumulated memory and leave the installed, signed artifact in
place, ready to run from initial state.

**Keys.** State encryption keys may be customer controlled. The runtime accepts
a key reference — including a KMS or HSM reference — rather than requiring the
key material itself, consistent with the requirement that private signing keys
are never exported. When a customer holds the key, the delta chain is opaque to
anyone without it, including any hosted infrastructure that stores it. Loss of a
customer-held key makes state unrecoverable; that is the intended property, and
`reset` remains available because it does not require reading the deltas.

### 7. Updates (FR005 / FR006)

An update produces a **new base RVF**, not a mutation of the old one. Each
update is:

- **signed** — verified against the publisher identity before installation or
  execution, with revoked publisher identities rejected at both points;
- **version constrained** — the update declares the base versions it may be
  applied from, and an out-of-range application is refused;
- **reversible** — the prior base and its state chain remain restorable, so a
  bad update can be rolled back rather than only rolled forward;
- **linked to the prior RVF identity** — the new base records its predecessor's
  identity, forming the lineage chain that §4 walks.

State carried across an update goes through **migrate**, not through implicit
acceptance. The migration re-binds the delta chain to the new base identity and
records both identities in the witness chain, so an auditor can see exactly
which state crossed which update boundary. When an update declares a state
schema change, migration re-encodes to the new `stateSchemaVersion`; when it
cannot, the update declares that state is not migratable and the operator
chooses between `reset` and staying on the prior base.

### 8. Deterministic reconstruction across platforms

Reconstructing the same checkpoint and delta chain on hosted Linux, Windows,
macOS, QEMU, and bare metal RVM must yield **identical** state. This requires:

- a canonical, backend-independent serialization for checkpoints and deltas —
  no host-endianness, path-format, or allocator-layout dependence;
- deterministic ordering of deltas by their recorded sequence, not by storage
  enumeration order;
- inclusion of the ADR-287 virtual-clock and seeded-randomness cursors, so a
  resumed instance continues the same deterministic streams;
- exclusion of ambient host values from state, so nothing platform-specific is
  captured in the first place.

State that survives restart must not modify the base RVF, and identical
reconstruction is what makes suspend-on-one-backend / resume-on-another a
verifiable claim rather than a best effort.

## Acceptance criteria

1. After arbitrary agent execution, the base RVF bytes and SHA256 hash are
   unchanged, and match the hash embedded in every platform package.
2. State survives process restart and is reconstructed from checkpoint plus
   witness deltas without modifying the base RVF.
3. A delta or checkpoint whose recorded base identity does not match the loaded
   base RVF — and is not an accepted ancestor — is rejected, execution does not
   begin with partial state, and the rejection is witnessed.
4. Modified runtime, policy, model, or state segments are each independently
   rejected.
5. `branch`, `rollback`, `merge`, `migrate`, and `reset` each complete, produce
   witness records, and leave the base artifact byte-identical.
6. Compaction into a new checkpoint records the delta range it subsumes, and a
   verifier confirms the compacted state equals the replayed state.
7. State deletion succeeds while the base RVF remains installed and runnable
   from initial state.
8. Uninstallation removes runtime and state according to policy.
9. With a customer-controlled key reference, delta contents are unreadable
   without the key, and `reset` still succeeds without it.
10. The same checkpoint and delta chain reconstructed on hosted Linux, Windows,
    macOS, QEMU, and bare metal RVM produce identical state and identical
    deterministic evaluation hashes.
11. An update from an out-of-range prior version is refused; an in-range update
    is applied, records the prior RVF identity, and is reversible to the prior
    base with its state chain intact.
12. An update signed by a revoked publisher identity is rejected at installation
    and at execution.
13. The witness chain covering checkpoints, deltas, migrations, and rejections
    verifies cryptographically end to end.

## Consequences

### Positive

- Cross-platform hash equality and signature validity hold for the artifact's
  entire operational life, not just at install time.
- Rollback, branching, and update reversal become ordinary operations rather
  than recovery procedures.
- Lineage binding closes state-substitution attacks without requiring the
  runtime to trust the storage layer.
- Customers can hold state keys without the hosted service losing the ability
  to reset or reinstall.
- Deterministic reconstruction makes cross-backend suspend/resume testable.

### Negative

- Reconstruction cost grows with delta-chain length until compaction runs, so
  compaction scheduling becomes an operational concern.
- Canonical serialization constrains internal state representations and makes
  runtime refactors state-schema-visible.
- Customer-held keys make state unrecoverable on key loss, which will surface
  as support incidents.
- Merge requires a declared conflict-resolution rule; there is no universally
  correct default, so some merges will need application-specific policy.
- Every state operation emits witness records, which adds storage and I/O cost
  proportional to state churn.

## Alternatives Considered

- **Write state back into the RVF**: rejected because it destroys cross-platform
  hash equality, invalidates the signature, and breaks build provenance.
- **Store state in an opaque host-native database**: rejected because it would
  not be lineage-bound, not reconstructible from a witness chain, and not
  portable across backends for suspend/resume.
- **Deltas only, no checkpoints**: rejected because unbounded replay makes
  startup cost grow without limit.
- **Checkpoints only, no witness deltas**: rejected because a checkpoint alone
  asserts state without an auditable derivation.
- **Trust the base identity recorded in the state**: rejected because it lets
  attacker-supplied state name whatever base it wants; the runtime compares
  against the base it actually loaded.
- **In-place updates that mutate the installed artifact**: rejected because
  updates must be reversible and linked to the prior identity, which requires
  the prior artifact to still exist.
- **Service-held state keys only**: rejected because the enterprise posture
  requires that models, prompts, and data need never be readable by hosted
  infrastructure.

## Implementation Surfaces

- `rvm-state` — `CompressedCheckpoint`, `WitnessDelta`, compaction, branch,
  rollback, merge, migrate, reset
- `rvm-rvf` — delta segment format, lineage binding, base identity preservation
- `rvm-witness` — state operation records, lineage rejections, migration records
- `rvm-policy` — signed state size limits, retention and deletion policy, update
  version constraints
- `rvm-launch` — `checkpoint` and lifecycle commands over the state chain
- Canonical serialization schema for checkpoints and deltas, versioned by
  `stateSchemaVersion`
- Key reference resolution for customer-controlled KMS/HSM state keys
- Publisher revocation checks at installation and execution
