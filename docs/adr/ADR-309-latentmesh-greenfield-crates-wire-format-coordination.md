# ADR-309: Build LatentMesh Integration Inside ruvector as New Crates, Coordinated on Wire Format

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-305 (PIR, depends on); ADR-307 (PIR, depends on); ADR-310, ADR-311 (PIR, depend on this); LatentMesh ADR-003, ADR-005, ADR-006, ADR-008 (LatentMesh repo); ruvector ADR-159 (rvagent-a2a)
- **Tags**: pir, latentmesh, greenfield, cross-repo, transport

## Context

Invariant 3 of the program requires: *"every agent communication is
attributable."* The original framing of this work package described it as
"wiring LatentMesh live" — implying an existing integration that merely
needed its remaining pieces connected. Direct verification (see
`04-verification-addendum.md` §1) corrects that framing on two points:

1. **Inside `ruvector`, there is zero existing wiring to `ruvnet/LatentMesh`
   today.** The only trace is an unmerged branch, `origin/docs/link-latentmesh`.
   It is not cloned locally and not integrated with RuVector in any shipped
   form.
2. **`ruvnet/LatentMesh` itself is a small research prototype**, not a
   mature system to integrate against: 1,407 LOC across four crates
   (`latentmesh-core` 272, `latentmesh-align` 454, `latentmesh-gate` 534,
   `latentmesh-bench` 147; 23 tests total), and **there is no network
   transport crate anywhere in its workspace**. LatentMesh's own ADR-009
   confirms this: the statistical primitive (ADR-003) and admission gate
   (ADR-008) are implemented, but network transport, RVF packaging, and RVM
   enforcement wiring are all explicitly marked "not implemented" —
   externally, not just inside `ruvector`.

This changes the work's nature: this is greenfield engineering inside
`ruvector`, coordinated against an external, evolving, small-scale design
contract — not an integration task against existing code on either side.

## Decision

Build the `ruvector`-side LatentMesh integration as **new crates**, not as a
completion of an existing integration:

1. House the new crates under `crates/rvAgent/`, alongside the existing
   `rvagent-a2a` (ruvector ADR-159), or as a new `latentmesh` crate family —
   the exact placement is an implementation decision for the owning work
   package, not fixed by this ADR.
2. Implement, inside `ruvector`, the three pieces LatentMesh's own ADRs mark
   as not implemented anywhere: network transport, RVF packaging (model
   identity, transform, permitted recipients, provenance, witness history as
   artifact metadata, per LatentMesh ADR-008), and RVM `rvm-cap` admission
   enforcement.
3. Open an explicit, standing coordination channel with the
   `ruvnet/LatentMesh` maintainers so the two sides converge on one wire
   format rather than diverging — this program does not unilaterally define
   a wire format LatentMesh's own crates (`latentmesh-core`,
   `latentmesh-align`, `latentmesh-gate`) must then be retrofitted to match.
4. The acceptance bar for this work package is a live multi-agent task
   running LatentMesh's causal-edge verification (LatentMesh ADR-003)
   end-to-end for the first time from `ruvector` — per LatentMesh ADR-009,
   this has never been run against a live multi-agent task anywhere, inside
   or outside `ruvector`.

## Consequences

### Positive

- Scoping this as greenfield construction (not integration) sets accurate
  expectations for the owning work package's effort and timeline — a
  transport layer does not exist on either side of this coordination and
  must be built, not merely wired.
- A live first run of LatentMesh's causal-edge verification would be a
  genuine first for the mechanism anywhere in the `ruvnet` org, not just for
  this program.
- Coordinating on wire format from the start avoids the two sides
  independently building incompatible transports and discovering the
  mismatch late.

### Negative

- This is real, unstarted engineering effort on a mechanism (causal-edge
  latent communication) with no live precedent anywhere — schedule risk is
  the program's own top-ranked risk as of this research pass (see
  `03-program-plan.md`, Top Risks §1).
- LatentMesh's maintainers are still actively revising scope (its own
  ADR-009 was revised twice in one day); a wire-format coordination channel
  reduces but does not eliminate exposure to that churn.
- LatentMesh's small scale (1,407 LOC) means this program may end up writing
  substantially more `ruvector`-side code than exists in the entire upstream
  repo it is coordinating against — a proportion worth surfacing to
  reviewers who assume "integrating with an existing project" implies a
  smaller `ruvector`-side footprint.

## Security / Validation Gates

- **Causal-audit CI gate** (shared with ADR-310): any PR touching the new
  `latentmesh`-integration crates or downstream consumers must pass a
  controlled-replacement causal audit before merge.
- **Witness-chain requirement**: RVF packaging for every latent frame
  includes provenance and witness history as artifact metadata (LatentMesh
  ADR-008's own requirement), anchored via this program's shared witness
  contract (ADR-312).
- **RVM admission enforcement**: no latent frame is accepted without passing
  `rvm-cap` admission — this is the concrete implementation of invariant 3
  ("every agent communication is attributable") for this bounded context.
- **Hosted-RVM honesty discipline** (ruvector ADR-285): if any part of this
  transport layer runs in a hosted (non-bare-metal) mode, it must not claim
  bare-metal isolation strength.

## Affected Repos

- `ruvnet/ruvector` (new crates, primary build location)
- `ruvnet/LatentMesh` (wire-format coordination, primary as design contract)
- `ruvnet/rvm` (`rvm-cap` admission enforcement)

## Dependencies

Depends on ADR-305 (adopts LatentMesh ADR-009 as the design contract this
work implements) and ADR-307 (RVF/RVM provenance wiring the memory-ledger
side of this transport depends on). ADR-310 (causal-attribution gate) and
ADR-311 (anomaly quarantine) both depend on this ADR — neither has a
transport layer to gate or quarantine until this ADR's crates exist.

## Alternatives Considered

- **Wait for `ruvnet/LatentMesh` to build its own transport layer before
  starting `ruvector`-side work**: rejected — LatentMesh's own maintainers
  have not committed to a timeline, and this program's acceptance test
  depends on a live run of the causal-edge verification mechanism; waiting
  indefinitely is not compatible with the program's schedule.
- **Build a `ruvector`-only latent-communication mechanism instead of
  coordinating with LatentMesh's wire format**: rejected — this is exactly
  the "parallel architecture" ADR-305 already rejected at the program level;
  a `ruvector`-only mechanism would not benefit from LatentMesh's existing
  causal-edge verification math (`latentmesh-align`, `latentmesh-gate`) and
  would fragment the `ruvnet` org's latent-communication story further.
