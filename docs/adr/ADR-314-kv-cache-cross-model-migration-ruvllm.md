# ADR-314: KV-Cache Cross-Model Migration in ruvLLM (Fast-Follow)

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: none within this program's critical path (independent parallel track)
- **Tags**: pir, kv-cache, ruvllm, fast-follow, cross-model

## Context

Fast-follow scope item 4 of the program (per `03-program-plan.md`'s scope
decision) is KV-cache cross-model migration, selected because it has the
program's strongest evidence and is the cheapest item to ship. Per
`docs/research/perpetual-intelligence-runtime/01-evidence-review.md`:

- **Cross-model KV-cache mapping** —
  [arXiv:2608.03893](https://arxiv.org/abs/2608.03893), "Cross-Model KV
  Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill
  Reuse," submitted 2026-08-04 16:26 UTC (matches the brief's date exactly).
  **Grade A — the strongest-evidence item in the evidence review.** Exact
  figure match on every reported number: the closed-form linear mapper runs
  "2.7-25x faster than re-prefill." Tested across six model pairs spanning
  three model families; the linear mapper retains 73-98% of the receiver's
  standalone-prefill accuracy on four pairs, while **two pairs degrade
  sharply**. A nonlinear MLP variant recovers up to +37pp HellaSwag accuracy
  on the failing pairs.

This is the concrete implementation target for a `ruvllm` capability:
`crates/ruvllm` already provides paged attention, KV cache management, and
SONA learning (built on `ruvector-core`/`ruvector-sona`), but cross-*provider*
or cross-*model-family* KV-cache migration is not centered here today — that
role sits closer to claude-flow's ADR-026 3-tier routing, which is a
different concern (routing between providers, not migrating cache state
between models).

## Decision

Implement arXiv:2608.03893's closed-form linear KV-cache mapper as a
`ruvllm` capability, ships independently of the Phase 1–3 PIR branch since it
depends only on `ruvllm`:

1. Implement the closed-form linear mapper for same-family model migration
   in `crates/ruvllm`, targeting `kv_cache.rs`, `paged_attention.rs`, and
   `serving/kv_cache_manager.rs`.
2. Implement the paper's nonlinear MLP fallback specifically for the pairs
   the paper itself identifies as degrading under the linear mapper — this
   is not optional; the evidence review explicitly notes this fallback is
   what "directly supports the brief's implication that migration quality
   must be predicted/handled before blind use."
3. Add a routing gate that predicts transfer quality **before** migrating —
   never migrate blind. The gate's prediction determines whether to use the
   linear mapper, the MLP fallback, or refuse migration and fall back to a
   full re-prefill for a pair judged too degraded for either.
4. This work package has no dependency on the rest of this program's ADR set
   (ADR-305 through ADR-313, ADR-315) and can start immediately in parallel.

## Consequences

### Positive

- Grade-A, exact-figure-match evidence gives this ADR a concrete,
  independently reproducible acceptance bar (2.7-25x speedup; 73-98%
  accuracy retention on four of six tested pairs).
- Fully independent of the rest of the program's critical path — delivers
  value on its own schedule without blocking or being blocked by ADR-305
  through ADR-313 or ADR-315.
- The "predict transfer quality before migrating" routing gate directly
  addresses the paper's own finding that two of six pairs degrade sharply —
  this ADR does not treat the mapper as universally safe.

### Negative

- The paper's reported figures come from its own six-pair benchmark; this
  program has not yet reproduced them against its own model inventory —
  the acceptance bar should be re-confirmed against `ruvllm`'s actual served
  models, not assumed to transfer directly.
- The routing gate's prediction accuracy is itself an open engineering
  question — a gate that mispredicts transfer quality could either block
  safe migrations (losing the speedup) or allow unsafe ones (the exact
  failure mode this ADR is designed to prevent).
- Two additional code paths (linear mapper, MLP fallback) plus a routing
  gate add real maintenance surface to `ruvllm`'s KV-cache management code.

## Security / Validation Gates

- **Proof-gated promotion**: this capability's rollout follows the same
  `ruvector-proof-gate`/`rvm-proof` promotion path as any other mutation to
  `ruvllm`'s serving path — it is not exempt from standard promotion gating
  merely because it ships on an independent schedule.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to `kv_cache.rs`, `paged_attention.rs`, or
  `serving/kv_cache_manager.rs`, since KV-cache handling is a
  security-sensitive serving-path component.

## Affected Repos

- `ruvnet/ruvector` (`crates/ruvllm`; new npm package `@ruvector/kv-migrate`)

## Dependencies

None — independent parallel track, can start immediately regardless of the
status of ADR-305 through ADR-313 or ADR-315.

## Alternatives Considered

- **Implement only the linear mapper, skip the MLP fallback**: rejected —
  the evidence review is explicit that the two-of-six-pairs-degrade finding
  is exactly why a fallback (and a routing gate) is needed; shipping the
  linear mapper alone would silently produce degraded migrations on known
  bad pairs.
- **Migrate blind (no routing gate) and rely on downstream task performance
  to catch bad migrations**: rejected — this is the same
  confounded-correlation failure mode ADR-310's causal-audit gate is
  designed to prevent elsewhere in this program; predicting transfer
  quality before migrating is cheaper and safer than detecting a bad
  migration after the fact.
