# ADR-315: Governance Constitution for Capability Expansion

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-312 (PIR, depends on); ADR-306 (PIR, distinct from ordinary promotion gating); ADR-305 (PIR, invariant 7 — proposer/promotion separation, binding here); autogenous ADR-393 "Product Thesis — Adaptive Agent Firewall" (autogenous repo); autogenous ADR-401 capability 5 and Update 1 §3 (governed self-improvement, `promoteAuthorized` — Done)
- **Tags**: pir, governance, constitution, capability-expansion, security

## Context

The acceptance test requires "zero unapproved capability expansion" — no
mutation may silently grant an agent a new tool, a new physical action
class, or a new communication peer without an explicit, higher-bar approval
distinct from ordinary behavioral-mutation promotion (ADR-306).
`ruvnet/autogenous` already implements a constitution/admission-gate pattern
for exactly this purpose (`constitution` crate, part of its
observe → explain → mutate → prove → canary → promote/rollback loop), and its
own ADR-401 (see ADR-305's reconciliation) frames governed self-improvement
as capability 5 of its 10-capability map: `Promote = Better ∧ Safe ∧
Authorized ∧ Reversible`.

**Correction from PR review**: an earlier draft of this ADR described that
predicate as "not yet fully closed as one checked gate," citing ADR-401's
capability-table row 5. **That row is stale relative to ADR-401's own
Decision section.** ADR-401's **Update 1 §3 is marked DONE**: `mesh-evolve.ts`
exports `promoteAuthorized(candidate, champion, { authorized, reversible })
→ PromotionDecision`, the single gate is implemented with each conjunct
independently blocking, `evolveMesh` routes every promotion through it, and
`test/promote-authorized.test.ts` proves no promotion path can skip a
conjunct. The predicate is closed upstream — this ADR does not scope work
to close it.

**Maturity caveat, corrected in this program's verification pass** (see
`04-verification-addendum.md` §2): `ruvnet/autogenous`'s README self-labels
the whole repo **"research prototype"** status, and states plainly that
"performance and economics claims in the ADRs are hypotheses until
benchmarked" — even though several of its individual ADRs (400, 401, 402,
403) carry **Accepted** status. This program treats the repo-level badge as
the governing signal for API stability: autogenous's constitution/
admission-gate API should be treated as unstable, and this program budgets
time for API churn rather than assuming production-grade stability. The
governance loop's actual implementation lives in `packages/radio-moe/src/*.ts`
(action-gate.ts, mesh-evolve.ts, disclosure.ts, reputation.ts — 6,171 LOC
total), not the Rust `constitution` crate alone.

## Decision

Adopt `autogenous`'s constitution/admission-gate pattern, **anchored on the
already-implemented `promoteAuthorized` predicate**, as the enforcement point
for zero-unapproved-capability-expansion:

1. Every mutation that would expand an agent's capability set — new tool
   access, new physical action class, new communication peer — requires
   explicit constitutional approval logged to the witness chain, **distinct
   from** ordinary behavioral-mutation promotion (ADR-306's evaluation
   pipeline). A mutation passing ADR-306's promotion gate does not
   automatically pass this gate; capability expansion needs its own,
   separately-authorized approval record. This gate is itself bound by
   governing invariant 7 (adopted in ADR-305, from ruflo ADR-322B): whatever
   proposed the capability-expanding mutation cannot also issue this gate's
   approval — approval authority and proposal authority must be held by
   distinct actors in the witness chain, the same separation-of-powers rule
   ADR-313 enforces on Darwin's mutation surfaces.
2. This program **adopts** `mesh-evolve.ts`'s `promoteAuthorized` predicate
   (`Promote = Better ∧ Safe ∧ Authorized ∧ Reversible`, all four conjuncts
   already independently blocking upstream) as the mechanism whose
   "Authorized" conjunct this gate's capability-expansion approval satisfies.
   The residual work this program actually scopes is narrower than the
   original framing: (a) the `ruvector`-side integration wiring PIR's
   capability/tool/action tables into `promoteAuthorized`'s `authorized`
   parameter, and (b) the distinctness rule in §1 above — ensuring a
   capability-expanding mutation's "Authorized" check is evaluated against
   *this program's* capability-expansion approval record specifically, not
   merely against `promoteAuthorized`'s general authorization state. This
   program does not build or close the predicate itself; that work is
   already done upstream.
3. Approval records for capability expansion are anchored via this
   program's shared witness contract (ADR-312), so the 30-day acceptance
   harness (per `03-program-plan.md`) can query "zero unapproved capability
   expansion" directly by diffing the capability/tool/action tables at day 0
   vs. day 30 against this gate's approval log.
4. Given autogenous's self-labeled research-prototype status, this program's
   integration against its constitution/admission-gate API is built with an
   explicit API-stability buffer (versioned interface, integration tests
   that fail loudly on upstream API changes) rather than a tight, assumed-
   stable coupling.

## Consequences

### Positive

- Reuses `autogenous`'s already-designed governance concept (constitution,
  admission gate) instead of inventing a parallel capability-expansion
  approval mechanism.
- Directly operationalizes the acceptance test's "zero unapproved capability
  expansion" requirement as a queryable log, not just a stated goal.
- Explicitly separating capability-expansion approval from ordinary mutation
  promotion (ADR-306) prevents the specific failure mode where a
  statistically-significant, well-tested mutation quietly also expands an
  agent's capability set without anyone noticing.
- `promoteAuthorized` being DONE upstream (ADR-401 Update 1 §3) meaningfully
  reduces this ADR's scope versus the original framing — this program
  integrates against a finished, tested predicate instead of building or
  completing one.

### Negative

- Coupling to a self-labeled research-prototype API means this integration
  carries real churn risk; the API-stability buffer in Decision §4 adds
  engineering overhead specifically to manage that risk.
- This program's residual scope depends on `promoteAuthorized`'s upstream
  API remaining stable (see the research-prototype caveat above) — a
  breaking change to its `authorized`/`reversible` parameters or return
  shape would require rework on the `ruvector`-side integration, even though
  the predicate's core logic is not this program's responsibility to build.
- Two separate approval paths (behavioral promotion via ADR-306, capability
  expansion via this ADR) add process overhead versus a single unified
  promotion gate — deliberate, per the acceptance test's own requirement
  that these be distinct, but a real cost.

## Security / Validation Gates

- **Constitutional admission gate** (this ADR's core mechanism): every
  capability-expanding mutation requires explicit, separately-logged
  approval before it can take effect — blocking, not advisory.
- **Witness-chain requirement**: every constitutional approval or denial is
  anchored via ADR-312's shared witness contract, queryable end-to-end for
  the 30-day acceptance harness's capability-table diff.
- **Distinctness from behavioral promotion**: this gate is never satisfied
  merely by a mutation passing ADR-306's evaluation pipeline — the two are
  independently blocking.
- **Separation-of-powers invariant** (governing invariant 7, ADR-305): the
  actor that proposed a capability-expanding mutation cannot also approve
  it through this gate.
- **Hosted-RVM honesty discipline** (ruvector ADR-285): if any part of this
  gate's enforcement runs in a hosted (non-bare-metal) mode, isolation
  claims must match what has actually been tested.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change touching capability tables or the admission-gate integration.

## Affected Repos

- `ruvnet/autogenous` (`constitution` crate, `packages/radio-moe/src/action-gate.ts` and `mesh-evolve.ts`)
- `ruvnet/ruvector` (integration point, capability/tool tables)
- `ruvnet/rvm` (witness anchoring via ADR-312)

## Dependencies

Depends on ADR-312 (the shared witness/anchoring contract this gate's
approval records use). Distinct from, but coordinates with, ADR-306's
promotion pipeline — a mutation may need to pass both gates independently.

## Alternatives Considered

- **Fold capability-expansion approval into ADR-306's ordinary promotion
  gate as one more check**: rejected — the acceptance test explicitly
  requires querying capability expansion independently of general
  performance-based promotion; conflating the two would make that query
  impossible to answer cleanly.
- **Build a `ruvector`-native constitutional gate instead of integrating
  with autogenous's**: rejected — autogenous's constitution/admission-gate
  concept is already designed for exactly this purpose, and building a
  parallel mechanism would repeat the same anti-pattern ADR-305 already
  rejected at the program level (parallel architecture instead of adopting
  existing design work).
- **Treat autogenous's Accepted-status ADRs (400/401) as sufficient
  evidence of API stability, ignoring the repo-level prototype badge**:
  rejected — per this program's verification pass, the badge and individual
  ADR statuses are not always in agreement, and the badge is the more
  conservative signal for integration planning.
