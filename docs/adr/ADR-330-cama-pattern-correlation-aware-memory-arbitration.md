# ADR-330: CAMA-Pattern Correlation-Aware Memory Arbitration

- **Status**: Proposed
- **Date**: 2026-08-22
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-307 (PIR, extends — three-level persistent memory, TARL ledger); ADR-320 (PIR, extends — AtomicObservation + causal episodic graph, the provenance substrate this ADR arbitrates over); ADR-325 (PIR — stage-level diagnostic gate; arbitration registers as a retrieval/filtering-stage policy under its taxonomy); see `docs/research/perpetual-intelligence-runtime/10-wave4-evidence-review.md` and `11-wave4-program-plan.md`
- **Tags**: pir, wave-4, cama, memory-arbitration, provenance, evidence, security

## Context

Wave-4 evidence review grades this paper **A on mechanism** —
[arXiv:2608.19701](https://arxiv.org/abs/2608.19701), "Beyond Memory
Majority: Latent-Source Reasoning for Multi-Agent Memory Arbitration"
(CAMA — Correlation-Aware Memory Arbitration, the abstract's own bolded
expansion), submitted 2026-08-20 — with the explicit caveat that **the
abstract contains no quantitative results whatsoever** ("Experiments on
multiple benchmarks demonstrate the superiority of our method" —
qualitative only). This is the weakest evidence posture of the wave, the
same class as Wave 3's DeAR; the mechanism, not any number, is what this
ADR adopts.

Confirmed mechanism, verbatim against the abstract: "memories written by
different agents may inherit the same upstream source or shared bias,
causing correlated evidence to be repeatedly counted and creating a false
majority. We term this failure mode **Memory Correlation Bias**." CAMA
"combine[s] neural dependency inference with provenance-based symbolic
priors to estimate the effective number of independent evidence sources,"
and additionally "learns a sequential recovery policy that actively
retrieves alternative evidence or traces upstream sources before making
the final decision."

**Artifact availability — checked, not assumed: no code anywhere.**
From-scratch reproduction posture.

**What exists in-repo today — checked at source**:
`crates/ruvector-agent-memory` already provides everything the symbolic
half needs. `observation.rs` (ADR-320) gives signed, content-addressed
`AtomicObservation`s carrying `source`, `confidence`, `tenant`, and
`causal_parents` — "provenance is a cryptographic fact, not a bookkeeping
convention." `fusion.rs` gives the `CausalEpisodicGraph` whose
`resolve_provenance()` guarantees every derived node resolves back to its
atomic sources, with hard rejection of unresolvable or self-referential
parents. What does **not** exist is any arbitration layer: nothing today
clusters retrieved memories by causal ancestry before they are counted as
evidence, so ten memories derived from one upstream source count as ten.

This is the third clause of ruv's Wave-4 acceptance test, verbatim:
"RuVector should correctly treat ten agent memories derived from one
source as one effective evidence lineage rather than ten independent
votes."

**Two Wave-3 lessons are binding design constraints here**:

1. **Non-finite inputs defeat comparison-based gates silently** (Wave-3
   finding #888): a NaN confidence/reliability/freshness value would flow
   through every `<`-comparison as false and poison any mean. All numeric
   inputs are rejected at the choke point before arbitration, and
   aggregation functions skip or zero non-finite values as
   defense-in-depth.
2. **Metric-integrity mechanisms must be downgrade-only + fail-loud**:
   an arbitration layer that could *increase* apparent evidence would be
   a new attack surface (manufacture "independent" lineages to inflate a
   claim). Arbitrated effective evidence is therefore bounded above by
   the naive count by construction.

**Preprint-reproduction rule** (standing): CAMA reports no numbers to
reproduce; the acceptance bar is this program's own — the ten-to-one
lineage property, demonstrated in this repo's own tests and benchmark,
plus research-gate evaluation for any downstream promotion claim.

## Decision

Implement correlation-aware arbitration as a new `arbitration.rs` module
in `crates/ruvector-agent-memory`:

1. **Provenance-rooted clustering before counting.** At arbitration time,
   retrieved memories are resolved to their atomic sources via
   `fusion.rs::resolve_provenance()` and clustered by causal ancestry:
   memories sharing an upstream provenance root belong to one **evidence
   lineage**. Ten memories derived from one source form one lineage — one
   effective vote.
2. **Effective-evidence scoring.** Confidence is computed per lineage,
   not per memory:
   `effective_evidence = Σ over independent lineages (source_reliability × freshness)`
   replacing the naive `Σ over supporting memories`. Source reliability
   derives from the observation's `SourceKind` and signature status;
   freshness from observation timestamps under a monotone decay. All
   three factors are clamped to [0, 1] after the non-finite rejection
   gate.
3. **Downgrade-only, by construction.** For any memory set,
   `effective_evidence ≤ naive_count` always; the arbitration layer can
   merge lineages (reducing evidence) but has no operation that splits a
   lineage into more votes than its member count or scales any factor
   above 1. A violation of the bound is a panic-class invariant failure,
   not a score.
4. **Symbolic first, neural later.** The symbolic half (known dependency
   via signed provenance) ships now on the substrate that already exists.
   The paper's neural dependency inference (latent dependency between
   memories with *no* recorded provenance link) and its learned
   sequential evidence-recovery policy are explicitly follow-up, each
   gated on its own research-gate delta — and per Invariant 7, any such
   learned component proposes dependency hypotheses only; it cannot
   overrule a recorded provenance fact to *increase* independence
   (recorded same-root dependency is a floor, only further merging is
   learnable).
5. **Registered under the stage taxonomy.** Arbitration registers as a
   retrieval/filtering-stage policy under ADR-325's diagnostic gate, so
   an arbitration-policy change is a versioned candidate mutation subject
   to the joint three-condition promotion bar like any other memory
   strategy.
6. **Tenant boundary preserved.** Arbitration never clusters across
   tenants; cross-tenant observations are already rejected at ingest and
   remain invisible to each other here.

## Consequences

### Positive

- Closes a real, named failure mode — Memory Correlation Bias — in the
  exact subsystem (multi-agent shared memory) where this program's swarm
  agents write correlated observations today; twenty agents repeating one
  rumor count as one source.
- Nearly all substrate cost was already paid: ADR-320's signed
  content-addressed observations and ADR-321's-era causal graph make the
  symbolic implementation a traversal + scoring layer, not a new store —
  the natural neuro-symbolic extension the briefing identified.
- Downgrade-only construction means the new layer cannot be weaponized to
  inflate confidence — the failure direction is conservative by design.
- First-mover position on provenance-aware distributed memory consensus,
  built on cryptographic provenance rather than heuristics.

### Negative

- The symbolic half only sees *recorded* dependency; two memories from
  one source with no recorded provenance link still count as independent
  until the neural follow-up lands — the ADR is explicit that shipped
  scope under-detects correlation rather than pretending otherwise.
  **Security corollary: shipped scope detects DECLARED derivation only,
  so no sufficiency threshold over effective evidence may be relied on
  as a trust gate against repeated-claim attacks.** An agent that copies
  content without recording a causal parent mints a fresh independent
  lineage; the WP28-era audit demonstrated 20 rumor-copying agents
  yielding 21 lineages and zero downgrade. Arbitration is a correctness
  mechanism for honest-but-correlated writers, not an adversarial
  defense — that requires the neural dependency inference (content-level
  similarity irrespective of declared parentage) plus write-path
  attestation, both follow-up.
- Arbitration adds a graph-traversal cost to every evidence-weighted
  retrieval; bounded by episode size in practice but must be benchmarked
  (the agent-memory bench binary gains an arbitration suite).
- Source-reliability and freshness parameters are policy choices with no
  paper-provided values (the abstract has none); initial values are
  conservative defaults, versioned and mutable only through the ADR-325
  gate.

## Security / Validation Gates

- **Non-finite rejection at the choke point (blocking)**: every numeric
  input (confidence, reliability, freshness, similarity) is checked
  `is_finite()` before arbitration; non-finite ⇒ observation excluded
  and the exclusion logged. Aggregations additionally skip non-finite
  values as defense-in-depth.
- **Downgrade-only bound (blocking, invariant-checked)**:
  `effective_evidence ≤ naive_count`, enforced and tested, including
  property tests over adversarial lineage shapes.
- **Provenance integrity**: arbitration consumes only
  signature-verified observations; `resolve_provenance()`'s
  hard-rejection of unresolvable/self-referential parents is relied on
  and re-tested from this consumer's side.
- **Ten-to-one acceptance property**: a dedicated test constructs ten
  memories derived from one signed source and asserts exactly one
  effective lineage — the acceptance-test clause, executable.
- **Standard repo gate**: `cargo nextest run -p ruvector-agent-memory`,
  plus `npx @claude-flow/cli@latest security scan` after the arbitration
  write path lands.

## Affected Repos

- `ruvnet/ruvector` only — `crates/ruvector-agent-memory/src/`
  (new `arbitration.rs`, exports via `lib.rs`, bench suite in
  `main.rs`'s bench binary). Single-repo scope.

## Dependencies

Depends on ADR-320 (`observation.rs`/`fusion.rs` substrate — hard) and
ADR-325 (the gate arbitration policies register under — hard for
promotion, soft for initial landing behind a default-conservative
policy). Independent of the other Wave-4 ADRs.

## Alternatives Considered

- **Naive majority with a correlation penalty heuristic (no provenance)**:
  rejected — heuristic penalties are exactly the unverifiable
  dependency-guessing CAMA's symbolic half exists to replace, and this
  repo already owns cryptographic provenance; ignoring it would discard
  the strongest available signal.
- **Implement the neural dependency inference now**: rejected for this
  wave — no reference code, no reported numbers to calibrate against,
  and the symbolic half alone satisfies the acceptance-test clause;
  learned components arrive later under Invariant 7's proposer-only
  constraint.
- **Home in `ruvector-retrieval-receipt`**: rejected — its receipts are
  unsigned engine-produced commitments that detect post-issuance mutation
  only and do not protect against a dishonest engine; strictly weaker
  provenance than `observation.rs`'s signed content-addressing (checked
  at source, per its own threat model).
- **Adopt "CAMA" as the module/crate name**: rejected — acronym-collision
  discipline; the module is `arbitration`, cited to arXiv:2608.19701.
