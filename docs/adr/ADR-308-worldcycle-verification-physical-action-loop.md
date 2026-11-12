# ADR-308: WorldCycle-Style Verification for the Physical Action Loop

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-313 (PIR, depends on); ADR-306 (PIR, downstream consumer)
- **Tags**: pir, worldcycle, verification, physical-action, evaluation

## Context

Invariant 6 of the program requires: *"every physical action produces new
evidence."* Per
`docs/research/perpetual-intelligence-runtime/01-evidence-review.md`, this is
grounded in a grade-A source:

- **WorldCycle** — [arXiv:2608.04964](https://arxiv.org/abs/2608.04964),
  "WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video
  World Models," submitted 2026-08-05. The abstract states verbatim:
  "WorldCycle reduces state returning drift by up to 44% and lifts
  composite-action accuracy nearly 4x over the base model." A diagnostic
  benchmark, CycleBench, ships alongside it. The mechanism: verification
  signals derived from reversible/closed action cycles — an action sequence
  plus its inverse should return to the initial state — used for
  annotation-free supervision. Grade A, exact figure match against the
  program brief; no name collision found in the evidence review's search.

This paper's reversible-action-sequence verification is exactly the kind of
"new evidence" invariant 6 asks for: an executed action either returns the
world to a predictable state (when composed with its inverse) or it does not,
and that discrepancy is itself evidence usable for supervision, independent
of any external label.

## Decision

Adopt WorldCycle's reversible-action-sequence verification (arXiv:2608.04964)
as an explicit stage in the physical action loop, positioned between
"observe consequences" and "Dream Machine evaluation" (ADR-306). Concretely:

1. For every executed physical action sequence where a reasonable inverse
   exists, construct the inverse sequence and measure state-return drift
   against the pre-action state.
2. Use the drift measurement as an annotation-free verification signal fed
   into the mutation's evaluation record, alongside whatever supervised or
   task-specific signals already exist.
3. Adopt the paper's reported figures — **up to 44% long-horizon drift
   reduction and ~4x composite-action accuracy** over an unverified baseline —
   as the acceptance bar for this program's Phase 3 world-model component.
   This program has not yet run its own CycleBench-equivalent benchmark; the
   44%/4x figures are the paper's reported results on its own benchmark, not
   yet confirmed against this program's task set. This ADR commits to
   re-measuring against CycleBench or an internal equivalent before treating
   the figures as validated in this program's context.

## Consequences

### Positive

- Gives invariant 6 a concrete, annotation-free verification mechanism
  instead of relying solely on external task-success labels, which are
  expensive and may not exist for every physical action.
- The reversible-action-cycle technique composes cleanly with Dream Machine
  (ADR-306): drift measurements are exactly the kind of pre-filter signal
  Dream Machine's dream-replay stage is meant to cheaply evaluate.
- Grade-A, exact-figure-match evidence — this ADR can cite a concrete
  acceptance bar rather than an aspirational one.

### Negative

- Not every physical action has a well-defined, safe-to-execute inverse
  (e.g. destructive or irreversible actions); this ADR does not specify a
  fallback verification path for that case, which is an open implementation
  question for the owning work package (WP10).
- The 44%/4x figures come from CycleBench, a benchmark this program has not
  reproduced; treating them as this program's acceptance bar before
  re-measurement risks anchoring on numbers that may not transfer to PIR's
  actual task distribution.
- Adds execution cost: every verified action sequence now also requires
  executing (or simulating) its inverse, which is not free in a physical or
  physically-simulated environment.

## Security / Validation Gates

- **Proof-gated promotion**: drift-verification results feed into the same
  `ruvector-proof-gate`/`rvm-proof` promotion path as Dream Machine's verdict
  (ADR-306) — this ADR does not introduce a separate, unaudited promotion
  path.
- **Structural frozen-weights enforcement** (shared with ADR-313): the
  verification stage must not introduce a code path that could be mistaken
  for a fine-tuning signal on the frozen foundation model — drift
  measurements feed the mutation-evaluation record, never a gradient update.
- **Witness-chain requirement**: each drift-verification result is logged to
  the witness chain (ruvector ADR-134 schema) as part of the mutation's
  evaluation record, so it is auditable alongside the eventual promotion
  decision.

## Affected Repos

- `ruvnet/ruvector` (`crates/ruvector-nervous-system`; new crate
  `ruvector-worldcycle-verify`, per `03-program-plan.md`'s Crates section)

## Dependencies

Depends on ADR-313 (SHAPER-pattern skill/harness evolution loop) — this
verification stage sits inside the physical action loop that ADR-313's
mutation surfaces produce candidates for. ADR-306 (Dream Machine) is a
downstream consumer of this stage's drift-verification signal.

## Alternatives Considered

- **Rely only on external task-success labels for invariant 6**: rejected —
  labels are expensive, not always available for physical actions, and do
  not by themselves establish that "new evidence" was produced by the action
  as invariant 6 requires.
- **Adopt WorldCycle's full video-world-model architecture, not just its
  verification signal**: rejected as out of scope for this ADR — the program
  brief and evidence review both frame WorldCycle's contribution to this
  program narrowly as its annotation-free verification mechanism, not a
  replacement for PIR's own physical-action execution stack.
