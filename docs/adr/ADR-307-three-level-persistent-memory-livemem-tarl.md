# ADR-307: Three-Level Persistent Memory Architecture (LiveMem + TARL Pattern) on RuVector

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-305 (PIR, depends on); ADR-309 (PIR, depends on this)
- **Tags**: pir, memory, livemem, tarl, rvf, rvm, witness

## Context

Invariants 1 and 2 of the program require: *"every observation may change
memory"* and *"every memory change is transactional."* Per
`docs/research/perpetual-intelligence-runtime/01-evidence-review.md`, two
grade-A primary sources ground the design of a persistent memory tier:

- **LiveMem** — [arXiv:2608.02515](https://arxiv.org/abs/2608.02515),
  "LiveMem: Maintaining Memory State Continuity in Long-Running LLM
  Inference," submitted 2026-08-03 (v1), revised 2026-08-07 (v2). Confirms a
  fixed-capacity memory state whose lifetime is independent of the active
  context, maintained via a parallel memory branch (Gated DeltaNet-2
  backbone) alongside a bounded KV attention window, reframing long-running
  inference as "computational state preservation." The system can answer
  questions from memory state after supporting evidence has left the context
  window. Grade A — evidence review flags a name collision with an unrelated
  consumer iOS app ("LiveMem – Live Photo Editor"); this ADR always means the
  arXiv paper.
- **TARL (Transaction-Aware Reliable Ledgers)** —
  [arXiv:2608.03699](https://arxiv.org/abs/2608.03699), "TARL:
  Transaction-Aware Reliable Ledgers for Executable Memory Management in
  Long-Term Agents," submitted 2026-08-04 (v1), revised 2026-08-11 (v2).
  Confirms a five-operation executable ledger — add / ignore /
  revise-outdated-belief / reject-unreliable / defer-for-verification — with
  accepted, pending, and rejected ledgers, explicitly framed as reducing
  "memory pollution" and limiting cumulative corruption. Grade A. **This name
  is heavily overloaded in prior literature** (Taint Analysis and
  Reinforcement Learning, arXiv:2005.03813; Target-Aligned Reinforcement
  Learning, arXiv:2603.29501; an unrelated `xushoukai/TARL` GitHub repo) —
  none of which relate to agent memory ledgers. "TARL (Transaction-Aware
  Reliable Ledgers)" is spelled out here on first use, and every subsequent
  reference to TARL in this program's documents should do the same.

RuVector already provides the durable substrate these two patterns need:
RVF (`crates/rvf` — append-only crash-safe segments, progressive indexing,
post-quantum signatures, canonical format per ruvector ADR-029) and RVM
(`crates/rvm` — mandatory hash-chained witness records for every privileged
action, per ruvector ADR-134).

**Revised scope (per the deep-researcher's asset-map revision)**: the
three-tier architecture below is not a from-scratch build. Each tier already
has a concrete home in `ruvector`'s existing memory stack, and proof-gated
writes for that stack **already exist** (ruvector ADR-194, ADR-047) — meaning
"sign every memory transition via RVM witness records" is largely already
built for the underlying gate, not net-new. The net-new work this ADR
actually scopes is the TARL five-operation ledger *states and semantics*
layered on top of that existing proof-gate, not the gate itself.

## Decision

Implement three explicit memory tiers, formalizing components that mostly
already exist in `ruvector` rather than building three new stores:

1. **Working context** — maps to `ruvllm`'s `working_memory` module,
   unchanged by this ADR.
2. **Continuous latent state** — maps to `ruvllm`'s
   `episodic_memory`/`semantic_cache`/`agentic_memory` modules, formalized to
   follow LiveMem's fixed-capacity recurrent-state design (arXiv:2608.02515):
   capacity is bounded and independent of context length, so state can
   outlive whatever evidence originally produced it.
3. **Transactional RuVector memory** — maps to `ruvector-agent-memory`
   (ruvector ADR-252) plus `reasoning_bank` and `ruvector-temporal-coherence`
   (ruvector ADR-211), extended with TARL's five-operation ledger
   (arXiv:2608.03699): every incoming statement maps to exactly one of add /
   ignore / revise-outdated-belief / reject-unreliable / defer-for-verification,
   with accepted, pending, and rejected ledger states persisted via RVF and
   signed via the RVM witness records this tier's writes already produce
   (ruvector ADR-194, ADR-047, ADR-134 schema).

No incoming observation writes directly to the transactional tier without
passing through the TARL-style operation classification — this is what makes
invariant 2 ("every memory change is transactional") enforceable rather than
aspirational, on top of the write-path proof-gating that already exists.

## Consequences

### Positive

- Both source patterns are grade-A, independently verifiable primary sources
  — this ADR can cite concrete published mechanisms rather than inventing an
  architecture from the brief's prose alone.
- The TARL ledger's reject/defer states give the program a structural answer
  to memory poisoning (invariant 2), not just an append-only log that trusts
  every write.
- Reuses RVF/RVM's existing durability and witness primitives rather than
  building a parallel persistence layer.

### Negative

- Neither paper's reference implementation is open-sourced as of the
  evidence review; this program is implementing the *pattern* described in
  each paper's abstract and methodology, not integrating existing code — the
  behavioral fidelity to each paper's reported properties is unverified until
  this program's own implementation is tested against comparable benchmarks.
- Three explicit tiers add operational complexity (three storage/lifecycle
  policies to reason about) versus a single undifferentiated memory store.
- The middle (continuous latent state) tier's fixed capacity means it can
  legitimately forget — this is a design tradeoff, not a bug, but it must be
  documented clearly so downstream consumers (e.g. ADR-309's latent
  communication fabric) do not assume unbounded retention.

## Security / Validation Gates

- **Witness-chain requirement**: every state transition in this tier —
  every TARL ledger operation (add/ignore/revise/reject/defer) and every
  continuous-latent-state checkpoint — must emit an RVM witness record
  (ruvector ADR-134 schema) before it is considered committed. No RVF write
  without a corresponding witness entry.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to the ledger's write path, since it is a direct target for the
  memory-poisoning attacks this design is meant to resist.

## Affected Repos

- `ruvnet/ruvector` (`crates/rvf`, `crates/rvm`, `crates/ruvector-core`)
- `agentdb` (memory subsystem consumer of this tiering)

## Dependencies

Depends on ADR-305 (adopts the control-loop this tier is embedded in). ADR-309
(wiring LatentMesh live) depends on this ADR for RVF/RVM provenance wiring on
the memory-ledger side of latent communication.

## Alternatives Considered

- **A single undifferentiated memory store instead of three tiers**:
  rejected — collapses the working-context/continuous-state/transactional
  distinction that both source papers treat as load-bearing, and removes the
  structural basis for enforcing invariant 2.
- **Build the transactional ledger without the TARL five-operation
  classification (plain accept-or-reject)**: rejected — TARL's
  revise-outdated-belief and defer-for-verification states are what let the
  ledger correct itself over time rather than only ever accumulating or
  discarding; a binary scheme loses that self-correction path.
