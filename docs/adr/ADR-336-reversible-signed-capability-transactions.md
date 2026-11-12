# ADR-336: Reversible, Signed Capability Transactions

- **Status**: Proposed (cross-repo contract — contract + ADR only in this repo; RVM-side implementation lands in `ruvnet/rvm` under maintainer review, USER ACTION for merge)
- **Date**: 2026-08-23
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-312 (PIR — shared witness/anchoring contract); ADR-315 (PIR — capability-expansion gate); ADR-333 (PIR — RVM semantic authority above OpenShell-class runtimes); `crates/ruvector-agent-memory/src/ops.rs` (structural template); `ruvnet/rvm` PRs #38/#39; see `docs/research/perpetual-intelligence-runtime/12-wave5-evidence-review.md` and `13-wave5-program-plan.md`
- **Tags**: pir, wave-5, rvm, capabilities, reversibility, witness, cross-repo, security

## Context

Three of this wave's five verified items converge on the same gap, and
that convergence — not any one of them — is what justifies this ADR.

**1. Cordis (`cordiverse/cordis`) formalizes reversibility.** "A
Programming Paradigm for Spatiotemporal Composability" (Shi, Zhang, Cui;
Peking University + DeepSeek-AI; draft of 2026-08-13) defines *temporal
composability* — "the ability to completely revert a component's side
effects upon removal" — and formalizes **revertible effects**, "in which
every context transformation carries an inverse that the runtime tracks."
It gives "a calculus of dynamic composition, whose metatheory carries
spatiotemporal composability from a single component to a whole system."
Three caveats, all recorded: it is **not peer-reviewed** (self-published
PDF, README says "preprint under active revision" — **cite by commit
SHA**); Cordis **predates the paper by years** (`cordiverse/cordis`
created 2022-05-17, the plugin kernel behind Koishi — DeepSeek adopted it,
did not invent it); and "Cordis" collides with **EU CORDIS**, so it is
never written bare.

**2. `deepseek-ai/deepseek-harness` ships this live, and says outright it
is not safe.** Its `cordis_mount` / `cordis_unmount` tools let a model
mount and unmount capabilities in the running process, with
`cordis_unmount` returning "only after every owned tool, listener,
service, timer, and effect reaches quiescence" — temporal composability
enforced at runtime. And, verbatim:

> "Neither restricts the authority of exposed services: a temporary Plugin
> can call `ctx.shell` with the host executor's privileges and reach the
> real filesystem and web services... **This is an opt-in development tool
> with bash-equivalent trust, not a security boundary or product
> default.**"

(Note: the file is on branch `master`, not `main`.)

**3. Covenant (`open-covenant/covenant`) proves the ledger is table
stakes.** It genuinely implements signed capability grants with expiry and
revocation tombstones, hash-chained append-only audit, tiered memory,
commit-scoped provenance envelopes, and deny-by-default verb-exact
dispatch. **Any claim that "no system does signed capabilities with audit
chains and provenance for agents" would be false** and would repeat the
Wave-1 "component absent" mistake that ADR-305 §6 exists to prevent. That
framing is dropped here.

But Covenant's own `docs/runtime-sandbox-security.md` is explicit that its
default runner "**is not a security boundary against hostile agent code**,"
and `BUILT.md` disclaims "production sandbox-grade isolation for arbitrary
untrusted agents." Its Honesty Boundaries name production isolation,
production multi-peer operation, and benchmarked self-improvement as **not
claimed**. (Adoption context, recorded for citation posture: 8 stars, four
months old, ~272k lines largely written by an autonomous engineering loop,
with a memecoin attached. Prior art in code; not an ecosystem standard.)

**4. OpenAI pays for runtime monitoring because static gating is
insufficient** — "roughly 20% of the inference compute being monitored"
(see ADR-337). That is a third independent party concluding the same
thing.

**The gap all three point at**: the grant/revoke/audit ledger is solved.
**Runtime enforcement and isolation under a live, mutating capability set
is not.** The differentiation for RVM is therefore **signed + reversible +
evolvable** — not signed alone.

**What exists here — checked at source.** A repo-wide grep for
`CapabilityTransaction`, `EffectLedger`, `inverse_op`, `before_state`,
`after_state` returns **zero real hits**. There is no effect ledger and no
inverse-operation abstraction anywhere in `ruvnet/ruvector`. Two adjacent
things do exist and are the right foundations:

- `crates/ruvector-agent-memory/src/ops.rs` is the **structural
  template**: `TransitionRecord`, `LedgerState`, `AcceptanceReceipt`,
  `LedgerWitnessRecord` with `compute_record_hash()` / `chain_hash()`,
  `WitnessSink`, and `MemoryWitnessLog::verify_chain()`. A hash-chained
  transactional ledger with witness sinks — but over *memory*
  transactions, and with **no inverse operation**.
- `ruvnet/rvm`'s `rvm-cap` implements ADR-135's three-layer proof system
  and already supplies **grant, monotonic attenuation, revocation
  propagating through a derivation tree, and epoch-based expiry**. What it
  lacks is before/after-state hashing, the effect ledger, and the inverse.

**Where this must be built, and why not here.** ADR-333 already
established the posture: RVM work lands in `ruvnet/rvm` under that repo's
maintainer-review requirement, surfaced as **USER ACTION**, never routed
around. The extension points are `crates/rvm-context/src/{capability,
receipt}.rs` and `crates/rvm-witness/src/{log,replay}.rs`, all arriving
via `ruvnet/rvm` PRs **#38** (`REVIEW_REQUIRED`) and **#39**. **Wave 5
cannot merge them.**

**A trap worth recording**: `crates/rvm/` inside `ruvnet/ruvector` is a
**stale snapshot** — 14 subcrates against 18 on rvm's own `main`, missing
`rvm-anchor`, `rvm-host`, `rvm-launch`, `rvm-rvf`, and `rvm-gpu`. Do not
plan implementation against the vendored copy.

## Decision

Specify — here, as a contract — a `CapabilityTransaction` whose
implementation lands in `ruvnet/rvm`.

1. **Every runtime capability mutation is a transaction** with seven
   recorded elements: a **before-state hash**, a **signed authority**, an
   **effect ledger**, an **inverse operation**, an **expiry**, a
   **provenance** chain, and an **after-state hash**.
2. **The lifecycle is**: current state → signed capability proposal → RVM
   authorization → mount → record every effect → agent uses capability →
   apply inverse effects → **verify resulting state** → retain or revoke.
   The verify step is not optional and not advisory: a transaction whose
   post-inverse state hash does not match its before-state hash is a
   **failed rollback**, reported as such.
3. **Reversal is a first-class obligation, not cleanup convention.**
   Following Cordis's revertible-effects formalization, every recorded
   effect carries an inverse the runtime tracks. An effect that cannot
   state its inverse **cannot be admitted to the ledger**, which means the
   capability that would produce it cannot be mounted.
4. **Unverifiable rollback counts as failed rollback.** Downgrade-only, in
   the same spirit as ADR-330's confidence bound and ADR-331's
   escalate-only rule: the failure direction is conservative by
   construction. A rollback whose verification cannot be completed is
   never reported as successful.
5. **Signed is necessary, not sufficient — and this ADR claims no
   novelty for it.** Covenant (`open-covenant/covenant`) implements signed
   grants, revocation, and audit chains today. The contribution claimed
   here is the *transaction* framing: reversibility and post-state
   verification layered on the grant/revoke/audit substrate.
6. **Enforcement lives below the mutable layer.** Per ADR-333, RVM is the
   semantic-authority layer above an OpenShell-class secure runtime;
   `CapabilityTransaction` is a semantic-authority mechanism and does
   **not** claim to provide kernel isolation. It records and reverses what
   an agent did; it does not confine what the agent's code can reach. That
   confinement is the runtime layer's job.
7. **Honest scope in receipts.** ADR-333's receipt-honesty gate applies
   unchanged: a transaction receipt must state its enforcement substrate
   (OpenShell version and policy hash, or "none"). **Absence is recorded,
   never implied away.**

## Consequences

### Positive

- Dream Machine can invent runtime capabilities and remove them again with
  a verified return to the prior state — which is what makes unattended
  runtime mutation tolerable at all.
- Reversibility plus post-state verification is a genuinely open axis:
  DeepSeek ships mount/unmount and disclaims safety; Covenant ships
  signing and disclaims isolation. Neither claims verified reversal.
- The mechanism composes with existing program invariants — ADR-315's
  capability-expansion gate governs any widening, ADR-312 anchors the
  witness chain.

### Negative

- **The implementation is in a repository this program cannot merge
  into.** `ruvnet/rvm` requires maintainer review, and PR #38 is
  `REVIEW_REQUIRED` today. This ADR is a contract that may sit unbuilt.
  That is stated plainly rather than worked around.
- Requiring every effect to declare an inverse is a real constraint on
  what capabilities may be mounted. Some useful effects are genuinely
  irreversible (an outbound network send, a payment). Those must be
  refused admission or explicitly quarantined as
  irreversible-by-declaration — they cannot be silently admitted with a
  no-op inverse.
- Before/after-state hashing costs something on every mutation. Unmeasured
  here; measurable only once implemented.

## Security / Validation Gates

- **No permission-laundering** (blocking, inherited from ADR-333): RVM
  merges require maintainer review. This is surfaced as USER ACTION and is
  never routed around by an agent approving on a maintainer's behalf.
- **Inverse-or-refuse** (blocking, Decision §3): an effect without a
  stated inverse cannot enter the ledger, and therefore its capability
  cannot mount. A no-op inverse for a genuinely irreversible effect is a
  **false safety claim** of exactly the Wave-3 #887 class and is
  prohibited.
- **Verified rollback** (blocking, Decision §2/§4): post-inverse state
  hash must equal the before-state hash; an unverifiable rollback is a
  failed rollback. Never reported as success.
- **Receipt honesty** (blocking, inherited from ADR-333): the enforcement
  substrate is named in every receipt, or recorded as "none."
- **No isolation claim** (Decision §6): documentation, receipts, and
  marketing copy must not describe `CapabilityTransaction` as a sandbox or
  a security boundary against hostile code. This is the precise error
  DeepSeek explicitly avoids making about its own mount/unmount tooling,
  and the one Covenant's one-line description invites.
- **Citation discipline**: never a bare "Cordis" (EU CORDIS) or a bare
  "Covenant" (`cobbr/Covenant`, a 4,729-star .NET C2 red-team framework in
  an adjacent domain; `csehammad/covenant-layer`, in ours). Never adopt
  either as a crate or module name.

## Affected Repos

- `ruvnet/ruvector`: **docs and contract only this wave.** No code.
- `ruvnet/rvm`: the implementation — `crates/rvm-context/src/{capability,
  receipt}.rs`, `crates/rvm-witness/src/{log,replay}.rs`, arriving through
  PRs #38/#39. **Maintainer review required — USER ACTION for merge.**
- Note: `crates/rvm/` inside `ruvnet/ruvector` is a **stale snapshot**
  (14 subcrates vs 18 upstream) and must not be used as a planning
  reference.

## Dependencies

Hard: `ruvnet/rvm` PRs #38 and #39, both blocked on maintainer review.
Soft: ADR-312 (witness anchoring), ADR-315 (capability-expansion gate for
any widening), ADR-333 (the layering posture this refines).

## Alternatives Considered

- **Implement `CapabilityTransaction` in `ruvnet/ruvector`**: rejected —
  ADR-333 already placed RVM's semantic-authority surface in `ruvnet/rvm`,
  and the vendored `crates/rvm/` here is a stale snapshot. Building here
  would fork a security-critical surface.
- **Claim signed capabilities as the differentiator**: rejected — Covenant
  (`open-covenant/covenant`) implements them today, Apache-2.0 and
  documented. The claim would fail ADR-305 §6's check-at-source rule.
- **Adopt Cordis's runtime directly**: rejected — the Rust port
  (`dshbox/cordis-rs`) is a useful reference, but adopting a plugin kernel
  wholesale imports a component model this program has not evaluated. The
  transferable content is the *revertible-effects formalization*.
- **Treat DeepSeek's mount/unmount as a security boundary**: rejected, and
  recorded because it is the obvious mistake — DeepSeek states outright it
  is not one, and its own note warns that a mounted waterfall listener can
  halt the agent's own tool dispatch.
- **Allow no-op inverses for irreversible effects**: rejected — see the
  inverse-or-refuse gate. A no-op inverse makes the rollback guarantee a
  lie at exactly the moment it matters.
