# ADR-325: D²ACCI-Pattern Stage-Level Memory Diagnostic Gate

- **Status**: Proposed
- **Date**: 2026-08-21
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-307 (PIR, extends — three-level persistent memory, WP3/WP4); ADR-320 (PIR, extends — MemFuse-pattern atomic observation and causal graph, WP18); ruvector ADR-282 (research-gate, complementary outcome-level gate — this ADR is stage-level, not a duplicate); see `docs/research/perpetual-intelligence-runtime/08-wave3-evidence-review.md` and `09-wave3-program-plan.md`
- **Tags**: pir, wave-3, d2acci, memory, diagnostic-gate, provenance, darwin

## Context

Wave-3 evidence review grades this paper **A** —
[arXiv:2608.17756](https://arxiv.org/abs/2608.17756), "D²ACCI: A Dual-Loop
Diagnostic Protocol for Evidence-Preserving Agent Memory," submitted
2026-08-18 (v1), revised 2026-08-19 (v2). **The title above is the paper's
descriptive subtitle, not its acronym expansion — the abstract spells the
acronym out separately and verbatim: "D²ACCI (Diagnostic-Driven
Artifact-based Closed-loop Controlled Iteration)."** That expansion —
**"D²ACCI (Diagnostic-Driven Artifact-based Closed-loop Controlled
Iteration)"** — is spelled out here on first use; every subsequent
reference to D²ACCI in this program's documents should use this same
acronym expansion (not the paper's title subtitle, which may still be
cited separately as the descriptive title), following the same discipline
ADR-307 applies to TARL (Transaction-Aware Reliable Ledgers).

Confirmed mechanism, verbatim against the abstract: a dual-loop design — an
**inner loop**, the actual memory-augmented-agent execution, and an **outer
loop**, a "diagnostic gate [that] promotes, feature-flags, or rejects memory
interventions based on paired evidence, protected-slice monitoring, and
trace-level localizability." The paper instantiates this in a reference
system, "MemStack," and introduces two supporting artifacts: DCR, a graded
observability metric, and D²ACCI-Eval, a reusable gate-replay artifact.

Numbers (verbatim, exact match to the program brief, plus one bonus figure
not in the original brief that this ADR does not drop): **93.59% on LoCoMo,
90.93% on LongMemEval, and 57.20% on PersonaMem-V2.** Ablations, verbatim:
"Five paired ablations show that supplement extraction, session-memory
retrieval, and Forget Guard yield statistically significant gains (+1.9 to
+3.7pp, all p ≤ .003)." Notably, the paper's own fifth ablation is a negative
result worth citing precisely *because* it demonstrates the gate working as
designed: **"BM25/RRF is retained as a monitored feature flag"** — i.e. this
intervention did not clear the bar for full promotion and was correctly held
back rather than promoted, the diagnostic gate's own worked example of
rejecting a weak intervention rather than promoting everything it evaluates.
DCR figure, verbatim: "Diagnostic artifacts reach 98–100% DCR@3 versus 0% for
results-only logs."

**Artifact availability — checked, not assumed: no code found.** The arXiv
page has no Comments field and no code/data URL. A targeted GitHub search for
"MemStack" and "D2ACCI"/"D2ACCI-Eval" surfaced only unrelated projects (a
Claude Code skill framework, a C memory-allocation library, a flash-card app,
student course exercises) and two research-digest blog posts summarizing the
abstract — neither an authors' repository. This ADR is therefore a
**first-party build from the paper's description**, the same posture ADR-319
(TRUSS) and ADR-318 (StagedWorkspace) already take for their own
no-code sources.

**Name collision — low risk.** "D²ACCI" itself is distinctive; no
conflicting project was found. "MemStack" collides with generic unrelated
repos in a targeted search — low risk, but this program does not adopt
"MemStack" as its own package or module name regardless, to avoid
reintroducing the exact ambiguity ADR-320's MemFuse naming discipline exists
to prevent for an adjacent memory-system name.

ADR-307 already formalizes `ruvector`'s three memory tiers (working context,
continuous latent state, transactional RuVector memory via TARL's
five-operation ledger). ADR-320 already adds `AtomicObservation` records and
a causal episodic graph on top of the continuous-latent-state tier,
consumed as evidence by ADR-310's causal-attribution gate. **D²ACCI's
genuinely new contribution sits underneath both**: `ruvector` ADR-282's
`research-gate` (adopted by this program via ADR-306) already performs
**outcome-level** gating — did the mutation's measured end-result improve?
D²ACCI's dual-loop diagnostic protocol is **stage-level** — when a memory
intervention regresses or fails to improve, which of the memory pipeline's
**four** distinct stages — the abstract's own enumeration, verbatim, "its
multi-stage pipeline (ingestion, retrieval, filtering, generation)" — is
responsible? ("Supplement extraction" is one of the paper's five ablated
*interventions*, not a named pipeline stage; conflating the two would
overstate the abstract's own stage count.) This is a different axis of
diagnosis, not a duplicate of research-gate's existing outcome-level
promotion decision, and this ADR does not replace `research-gate` — it
adds a complementary, finer-grained fault-localization layer underneath it.

**Preprint-reproduction rule** (applies uniformly across this program, per
`09-wave3-program-plan.md`): D²ACCI's reported figures (93.59% / 90.93% /
57.20%, and the +1.9–3.7pp ablation deltas) are the source paper's own
measurements on its own benchmark suite (LoCoMo, LongMemEval, PersonaMem-V2)
against its own reference implementation (MemStack) — not a promotion bar
this program is entitled to claim without its own `research-gate`-recomputed
paired-bootstrap delta. Every reported gain is a hypothesis until this
program's own benchmark confirms it, exactly as ADR-317 already states for
HarnessRisk and ADR-324 now states for SPADE.

## Decision

Make every RuVector memory strategy an **immutable versioned policy
artifact**, gated by a stage-level diagnostic protocol informed by D²ACCI's
mechanism (arXiv:2608.17756):

1. **Immutable versioned memory-strategy artifacts.** Each ingestion
   pipeline, retrieval algorithm, filtering policy, and generation strategy
   in ADR-307's memory tiers — the abstract's own four pipeline stages —
   becomes an immutable, versioned artifact — a candidate mutation, not a
   strategy mutated in place. This gives every memory change a stable
   identity to diagnose against and promote/reject as a unit, mirroring how
   ADR-313 already treats harness/skill mutations as versioned candidates
   rather than in-place edits.
2. **A memory mutation promotes only when three conditions hold jointly**,
   following D²ACCI's confirmed dual-loop pattern: (a) paired evaluation
   (this program's existing `research-gate` paired-bootstrap statistics,
   `crates/ruvector-sota-bench/harness/src/statistics.ts`) shows genuine
   improvement; (b) protected workloads — a held-out slice of existing
   memory-dependent tasks the mutation is not tuned against — do not
   regress, mirroring D²ACCI's confirmed "protected-slice monitoring"; and
   (c) failure provenance, when the mutation does regress, stays
   **localizable to exactly one of the four memory-pipeline stages**
   (ingestion / retrieval / filtering / generation) rather than surfacing
   only as an undifferentiated end-to-end score drop. All three conditions
   are required; any one failing blocks promotion.
3. **Retrieval algorithms become Darwin candidates.** HNSW, graph-based
   retrieval, BM25, temporal retrieval, attractor-based retrieval, and
   learned retrieval strategies are each registered as Darwin mutation
   candidates under ADR-313's existing harness-evolution loop, evaluated
   through this ADR's stage-level gate before promotion — exactly the
   correctly-cautious posture D²ACCI's own BM25/RRF ablation models: a
   plausible-sounding retrieval strategy is a monitored feature flag, not an
   automatic promotion, until it clears the joint bar in Decision §2.
4. **Stage-level diagnosis is complementary to, not a replacement for,
   `research-gate`'s existing outcome-level gate (ADR-282, adopted via
   ADR-306).** A memory-strategy candidate must still clear
   `research-gate`'s ordinary outcome-level promotion path; this ADR adds a
   finer-grained, stage-localized diagnostic requirement on top, specific to
   the memory pipeline's four-stage structure, which `research-gate`'s
   general-purpose statistics do not model.
5. **First-party build, no code to adapt.** Because no reference
   implementation of D²ACCI or MemStack was found, this program implements
   the dual-loop diagnostic protocol, the DCR observability metric, and a
   D²ACCI-Eval-style gate-replay artifact from the paper's described
   mechanism, the same from-scratch posture ADR-318 and ADR-319 already take
   for their own no-code sources — not a port of existing code.

## Consequences

### Positive

- Gives the memory subsystem a genuine stage-level fault-localization
  capability that `research-gate`'s existing outcome-level statistics do not
  provide today — a regression can be attributed to ingestion, retrieval,
  filtering, or generation specifically, rather than only "the end-to-end
  score got worse."
- The three-condition joint-promotion bar (paired improvement, protected-slice
  non-regression, single-stage localizability) gives ADR-307's and ADR-320's
  memory tiers a structural defense against a mutation that improves one
  metric while silently regressing another workload — the exact failure
  mode D²ACCI's protected-slice monitoring is designed to catch.
- Treating retrieval algorithms as first-class Darwin candidates, evaluated
  against this gate, directly extends ADR-313's harness-evolution loop and
  ADR-321's/ADR-324's candidate-source pattern into the memory subsystem
  specifically, without inventing a parallel mutation-governance mechanism.
- D²ACCI's own BM25/RRF ablation gives this program a concrete, cite-able
  worked example of the gate correctly declining to promote a plausible but
  unproven intervention — a template this program's own gate implementation
  can be validated against.

### Negative

- No reference implementation exists for either D²ACCI or MemStack; this
  program's stage-level diagnostic gate, DCR metric, and gate-replay artifact
  are all first-party builds whose behavioral fidelity to the paper's
  reported properties (98–100% DCR@3, the five-ablation deltas) is unverified
  until independently tested against this program's own benchmarks — the
  same open-ended validation burden ADR-307 already carries for LiveMem and
  TARL, and ADR-318/ADR-319 carry for their own no-code sources.
- A joint three-condition promotion bar is stricter, and more expensive to
  evaluate, than a single outcome-level statistical test — every memory
  mutation now requires protected-slice replay and stage-localization
  analysis in addition to `research-gate`'s existing paired-bootstrap
  evaluation, adding CI/evaluation latency to the memory-mutation path
  specifically.
- Stage-level localizability is itself a design and validation challenge:
  a memory pipeline's four stages are not always cleanly separable in a real
  implementation (e.g. a retrieval-stage change can shift what the filtering
  stage sees), and this ADR does not itself resolve how boundary cases are
  attributed — that is left to the owning work package's implementation.
- Depends on ADR-307's memory tiers and ADR-320's `AtomicObservation`/causal
  graph existing first, since stage-level provenance needs both a tiered
  memory substrate and per-observation causal parentage to localize a
  failure to a specific stage.

## Security / Validation Gates

- **Joint three-condition promotion gate (this ADR's core mechanism)**:
  paired-evaluation improvement, protected-slice non-regression, and
  single-stage failure localizability are all independently required; any
  one failing blocks promotion of a memory-strategy artifact, blocking not
  advisory.
- **Immutability and versioning**: a memory-strategy artifact, once
  promoted, is never mutated in place; a change is always a new versioned
  candidate evaluated by this gate, giving every promoted (or rejected)
  memory strategy an auditable version history.
- **Protected-slice isolation**: the protected workload slice used for
  non-regression testing is held separate from the slice a candidate
  mutation is tuned or evaluated against, mirroring the held-out task
  isolation ADR-324 requires for its own acceptance criterion.
- **Witness-chain requirement**: every promotion, rejection, and
  feature-flag decision made by this gate, plus its stage-localization
  verdict, emits an RVM witness record (ruvector ADR-134 schema), anchored
  via this program's shared witness/anchoring contract (ADR-312).
- **Complementary to, not a substitute for, `research-gate` (ADR-282, via
  ADR-306)**: this gate's stage-level pass does not itself constitute a
  passed outcome-level promotion; a candidate must clear both.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to the diagnostic gate's write path or the memory-strategy
  version registry.

## Affected Repos

- `ruvnet/ruvector` only — `crates/ruvector-agent-memory/src/{ledger.rs,
  observation.rs, fusion.rs, scoring.rs}` (ADR-307's TARL ledger, ADR-320's
  causal-graph fusion, and the scoring surface this gate's joint
  three-condition check extends) and `scripts/research-gate/research_gate.py`
  (new diagnostic-trace consumer, layered underneath the existing
  paired-bootstrap statistics), plus registered retrieval-algorithm
  candidates in `crates/ruvector-sota-bench/harness`. Single-repo scope, per
  `09-wave3-program-plan.md`.

## Dependencies

Depends on ADR-307 (the three-level memory tiers this gate governs
mutations against) and ADR-320 (the `AtomicObservation`/causal-graph
provenance this gate's stage-localization requirement relies on). Sits
underneath and complements, but does not depend on, `research-gate`
(ruvector ADR-282, adopted via ADR-306) — the outcome-level gate continues
to operate unchanged; this ADR adds a stage-level layer alongside it.

## Alternatives Considered

- **Rely on `research-gate`'s existing outcome-level statistics alone,
  without a stage-level diagnostic layer**: rejected — outcome-level
  statistics can confirm *that* a memory mutation regressed but not *which
  stage* caused it, leaving every regression a from-scratch investigation;
  D²ACCI's confirmed dual-loop mechanism is adopted specifically to close
  that gap.
- **Promote a memory-strategy candidate on paired-evaluation improvement
  alone, without protected-slice monitoring**: rejected — this is exactly
  the failure mode D²ACCI's protected-slice monitoring exists to catch: an
  intervention that improves the metric it is tuned against while silently
  regressing an unrelated protected workload.
- **Treat D²ACCI's reported figures (93.59%/90.93%/57.20%, +1.9–3.7pp) as
  this program's own expected result**: rejected, per the preprint-
  reproduction rule — those are the source paper's own numbers on its own
  benchmark suite and reference implementation; only this program's
  `research-gate`-measured delta may be cited as this program's result.
- **Adopt "MemStack" as this program's own module or package name**:
  rejected — while lower-risk than ADR-320's MemFuse collision, reusing an
  adjacent memory-system name invites exactly the kind of casual-reference
  ambiguity this program's naming discipline exists to prevent.
