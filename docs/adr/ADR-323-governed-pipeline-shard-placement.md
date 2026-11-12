# ADR-323: Governed Pipeline-Shard Placement for Multi-Node ruvLLM Serving

- **Status**: Proposed
- **Date**: 2026-08-20
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-314 (PIR, extends — KV-cache cross-model migration in ruvLLM, WP13); ADR-305 (PIR, cross-repo ADR-numbering discipline, §4); ADR-316 (PIR/repo, ADR numbering hygiene — canonical counter, disambiguation-by-slug); see `docs/research/perpetual-intelligence-runtime/06-wave2-evidence-review.md` §6
- **Tags**: pir, wave-2, pipeline-shards, ruvllm, serving, governance, adr-numbering

## Numbering note — why this ADR is 323, not 322

**This ADR is deliberately numbered ADR-323. `ruvector`'s own ADR-322 number
is skipped, not merely left unused by coincidence.** The reason is a
cross-repo citation-ambiguity risk this program has already hit four
separate times (per ADR-305 §6 and ADR-313's Context): the ADR-103
cross-repo ambiguity (ADR-305 §4); the "metaharness ADR-322" misattribution,
resolved to `ruflo` ADR-322 (`04-verification-addendum.md` §5); the phantom
"metaharness ADR-251" citation, which does not exist at all
(`04-verification-addendum.md` §8b); and the "ruvector ADR-150"
misattribution, resolved to `ruflo` ADR-150 (`04-verification-addendum.md`
§8c) — each one assuming the wrong repo owns an ADR number before checking.

**`ruflo` ADR-322** (with children 322A/322B/322C) is the single
most-cited external ADR in this program: ADR-305, ADR-306, ADR-310,
ADR-312, and ADR-313 all cite it by number, repeatedly, as the
flywheel-receipt/promotion-separation authority this program has adopted
(the `Promote = Better ∧ Safe ∧ Authorized ∧ Reversible` separation-of-powers
invariant, and the canonical-encoding/domain-separated-Ed25519 witness
format ADR-312 adopted). A `ruvector`-local ADR-322 for pipeline-sharded
serving would **not** be a literal filename collision — `ruflo`'s ADR-322
lives in a different repo's `v3/docs/adr/` tree — but it would put two
heavily-cited, differently-scoped "ADR-322"s in **concurrent use across
sibling repos of the same program**, which is exactly the ambiguity ADR-305
§4 requires every cross-repo reference to avoid, and exactly the failure
mode ADR-316's disambiguation-by-slug citation rule (`ADR-NNN (slug)` for
duplicated numbers) exists to catch when it happens *within* one repo.
Skipping the number in `ruvector` avoids manufacturing the ambiguity in the
first place, rather than relying on every future author to apply ADR-316's
disambiguation rule correctly every single time this ADR is cited.

Per **`docs/adr/INDEX.md`** (as of the numbering check performed for this
ADR set, PR #857's regenerated index at 348 entries) and ADR-316's policy
that "allocated" includes numbers claimed on open ADR-authoring branches,
`ruvector` ADR-322 is treated as **reserved-but-unused** for this reason,
and **ADR-323** is the number this program allocates for pipeline-shard
placement. Every citation of the `ruflo` ADR elsewhere in this document, and
in every other PIR document, uses the repo-qualified form **"ruflo
ADR-322"**, per ADR-316's citation rule. ADR-316's collision-check tooling
(`scripts/adr-index.mjs --check`) gates on duplicate numbers, not on gaps in
the sequence, so this deliberate skip does not trip that gate — confirmed
against ADR-316's own Decision §5 (the check "exits non-zero if `docs/adr/`
contains a duplicate plain number that is not in the frozen historical list
above (or if a frozen number's file count grows)," not an unused one).

## Context

Wave-2 evidence review grades this paper **A** — the strongest-evidence item
in the wave — [arXiv:2608.19147](https://arxiv.org/abs/2608.19147),
"Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC
Fleets," submitted 2026-08-19. Every claim the Wave-2 brief cited checks out
verbatim against the abstract, with one important precision correction:

- **OpenVINO precompiled per-stage shards**, with a specific optimization
  detail beyond the brief's summary: a naive per-stage export "misses an
  OpenVINO GPU optimization," and injecting a `beam_idx` Gather into each
  shard triggers the `IndirectKVCache` fusion, bringing shards to parity
  with the unsplit model.
- **Speculative decoding on stateful OpenVINO models** — confirmed.
- **1.79× concurrent throughput** — abstract, verbatim: "a two-node Llama
  3.1 8B INT4 pipeline serves two concurrent users at 1.79x the single-user
  throughput of the unsplit model on the same hardware." **This figure
  applies only to the two-node, Llama-3.1-8B-INT4 configuration.**
- **Four-node Lunar Lake, 70B model** — abstract, verbatim: "a four-node
  deployment of Lunar Lake AI PCs on Intel Tiber Cloud serves a single user
  at interactive speed, with output token-for-token identical to the same
  four-node pipeline decoding without speculation." This is a **separate
  result**, reported in the paper's full text as 5.72 tok/s single-stream at
  72.2% accept rate and 6.43 tok/s aggregate two-stream — **no 1.79× figure
  is attached to this configuration.**

**Precision requirement (binding on this ADR and every derived work
package or status report)**: the Wave-2 brief's phrasing conflated these
into what reads as one combined result. They are two separate results in
the source paper. This ADR, WP20, and any downstream citation must state the
1.79× figure as specific to the **2-node/8B** configuration, and the
4-node/70B feasibility result as a **separate**, non-1.79×-labeled target —
never combined into one headline number.

**Artifact availability — checked, not assumed, and the strongest of the
six papers in this wave**: the abstract states, "Code, raw benchmark logs,
and reproduction scripts ship as a self-contained package at
`github.com/labscommunity/pipeline-sharded-inference-paper` (in the
top-level `reproduction/` directory)." Verified directly via the GitHub API:
public, 653 KB, last pushed 2026-08-19T16:59:13Z. The `reproduction/`
directory was checked directly and contains `CLAIMS.md`, `HARDWARE.md`,
`MODELS.md`, `RESULTS.md`, `configs/`, and `scripts/` — a genuine,
structured reproduction package, not a code dump. This ADR adapts this
released reproduction package rather than rebuilding from the paper alone.

**Name collision**: no significant collision found for "pipeline-sharded
inference" or the paper's likely short names in a targeted search.

**Preprint-reproduction rule** (applies uniformly across this program, per
`07-wave2-program-plan.md`): the paper's own 1.79× concurrent-throughput
figure and its 4-node/70B interactive-serving figures (5.72/6.43 tok/s)
describe the source paper's own hardware and benchmark run, not this
program's fleet. This ADR treats governed pipeline-shard placement as a
**candidate mutation**, not adopted prior art — promotion follows the same
`research-gate`-recomputed-delta discipline as every other Wave-2 item
(ADR-306): the paper's figures are the target shape this program aims to
reproduce internally, never numbers this program is entitled to cite as its
own acceptance bar without independently measuring them on its own
2-node/8B and 4-node/70B setups.

ADR-314 already implements `ruvllm`'s closed-form linear KV-cache mapper for
**cross-model** migration within a single node (`kv_cache.rs`,
`paged_attention.rs`, `serving/kv_cache_manager.rs`). This ADR is a distinct
serving-layer improvement — pipeline parallelism **across nodes** with
pre-compiled per-stage graphs and speculative decoding — addressing
multi-node topology, not cross-model cache reuse. It ships independently of
ADR-314 but touches the same `crates/ruvllm` serving surface and the same
files, requiring coordination to avoid merge conflicts (per
`07-wave2-program-plan.md`'s WP20 repo assignment).

## Decision

Implement governed, pre-compiled pipeline-shard placement in `crates/ruvllm`
for heterogeneous multi-node ruvLLM serving fleets, adapting the confirmed-
live `labscommunity/pipeline-sharded-inference-paper` reproduction package
(arXiv:2608.19147):

1. **Pre-compile per-stage shards** using the paper's confirmed
   OpenVINO-equivalent approach: export each pipeline stage as a precompiled
   graph, and apply the `beam_idx` Gather injection needed to trigger
   `IndirectKVCache`-equivalent fusion so sharded stages reach parity with
   the unsplit model rather than silently regressing on a naive per-stage
   export.
2. **Shard placement across a multi-node fleet is governed by an explicit
   placement policy**, weighing:
   - **Latency** — inter-node communication cost for a given shard
     assignment.
   - **Cost** — resource/compute cost of the candidate placement.
   - **Trust** — node attestation and witness status (per this program's
     existing RVM witness-chain discipline, ADR-134 (witness-schema-log-format)/ADR-312) — an
     unattested or under-trusted node is not eligible to host a shard
     carrying sensitive state.
   - **Data residency** — placement constraints where shard state must not
     cross a residency boundary.
   - **Hardware capability** — a shard is only placed on hardware confirmed
     capable of running it (matching model/quantization requirements, per
     the paper's own INT4/Lunar-Lake hardware specificity).
   Placement is not a single greedy optimization over one of these axes; a
   candidate placement is rejected if it fails any hard constraint (trust,
   residency, hardware capability) regardless of its latency/cost score.
3. **Implement speculative decoding on stateful sharded models**, per the
   paper's confirmed mechanism, as the path to the 4-node/70B interactive-
   serving target.
4. **Target the two benchmark results separately, per the precision
   requirement above**: the internal 2-node/8B-class same-family setup
   targets the verified 1.79× concurrent-throughput figure; the 4-node/70B
   configuration targets single-user interactive-speed serving via
   speculative decoding (the paper's 5.72/6.43 tok/s figures as reference
   points), reported and evaluated as a wholly separate benchmark, never
   combined with the 1.79× figure into one number.
5. Coordinates with ADR-314/WP13 on the shared `crates/ruvllm` surface
   (`kv_cache.rs`, `paged_attention.rs`, `serving/kv_cache_manager.rs`) to
   avoid merge conflicts — the two ADRs are independent in scope (multi-node
   topology vs. cross-model cache reuse) but touch the same files.

## Consequences

### Positive

- The confirmed-live reproduction package (`CLAIMS.md`, `HARDWARE.md`,
  `MODELS.md`, `RESULTS.md`, `configs/`, `scripts/`) makes this the
  best-evidenced port-and-adapt effort across both Wave 1 and Wave 2 — WP20
  has a genuine, structured reference to adapt from, not a from-scratch
  build.
- Governing shard placement on trust/residency/hardware, not just
  latency/cost, extends this program's existing witness-chain and isolation
  disciplines (ADR-134 (witness-schema-log-format), ADR-312, the hosted-RVM honesty discipline from
  ruvector ADR-285) into the serving-placement layer, rather than treating
  placement as a pure performance-optimization problem.
- Explicitly separating the 1.79× and 4-node/70B results prevents this
  program from over-claiming a combined figure that does not exist in the
  source paper — directly closing the conflation flag the evidence review
  raised.

### Negative

- Deliberately skipping ADR-322 in `ruvector`'s numbering sequence is a
  one-time deviation from the "always allocate from true max + 1" default
  ADR-316 otherwise establishes — every future numbering-hygiene pass over
  this repo's ADR sequence must be aware this is an intentional skip with a
  documented rationale, not an accidental gap or an error to "fix" by
  renumbering ADR-323 down to 322.
- Governed placement (trust/residency/hardware hard constraints) is
  additional engineering surface beyond the paper's own reproduction
  package, which does not itself model a multi-tenant, multi-trust-level
  fleet — this program's placement-policy layer is net-new work on top of
  the ported shard-compilation mechanism.
- Shares `kv_cache.rs`, `paged_attention.rs`, and
  `serving/kv_cache_manager.rs` with ADR-314/WP13 — real merge-conflict risk
  if the two work packages are not coordinated, even though neither
  blocks the other functionally.

## Security / Validation Gates

- **Trust/residency/hardware hard constraints**: a candidate shard placement
  failing any of these three is rejected outright, regardless of its
  latency/cost score.
- **Witness-chain requirement**: node trust/attestation status consumed by
  the placement policy is verified via this program's existing RVM witness
  chain (ADR-134 (witness-schema-log-format), anchored per ADR-312), not a separately-trusted signal.
- **Precision citation discipline**: the 1.79× figure is always cited as
  specific to the 2-node/8B-class configuration; the 4-node/70B result is
  always cited separately, never combined into one headline number.
- **Cross-repo numbering discipline**: every reference to `ruflo`'s ADR-322
  (or its children 322A/322B/322C) in this ADR or any derived document uses
  the repo-qualified form "ruflo ADR-322," per ADR-316's citation rule;
  `ruvector` ADR-322 remains permanently unused and unreserved for any other
  purpose.
- **Proof-gated promotion**: this capability's rollout follows the same
  `ruvector-proof-gate`/`rvm-proof` promotion path as ADR-314's KV-cache
  migration work — not exempt from standard promotion gating merely because
  it ships on an independent schedule.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to `kv_cache.rs`, `paged_attention.rs`, or
  `serving/kv_cache_manager.rs`, since both this ADR and ADR-314 touch a
  security-sensitive serving-path component.

## Affected Repos

- `ruvnet/ruvector` only — `crates/ruvllm` (`kv_cache.rs`,
  `paged_attention.rs`, `serving/kv_cache_manager.rs` — same files ADR-314/
  WP13 touches; coordinate to avoid merge conflicts). Single-repo scope per
  `07-wave2-program-plan.md`.

## Dependencies

Depends on ADR-314 sharing the same `crates/ruvllm` serving surface — no
hard blocking dependency; can start immediately in parallel, the same
independent-parallel-track pattern ADR-314 itself used relative to the rest
of Wave 1's critical path. No dependency on ADR-317 through ADR-321 — this
ADR is not part of the Wave-2 combined acceptance test.

## Alternatives Considered

- **Allocate ADR-322 in `ruvector` for this decision, relying on ADR-316's
  disambiguation-by-slug rule to prevent confusion with `ruflo` ADR-322**:
  rejected — see the Numbering note above; ADR-322 is this program's
  single most-cited external ADR number, and manufacturing a second,
  differently-scoped "ADR-322" in concurrent use is a self-inflicted version
  of exactly the ambiguity ADR-305 §4 and ADR-316 exist to prevent, even
  though the disambiguation rule would technically resolve it after the
  fact.
- **Place shards by latency/cost optimization alone, without trust/
  residency/hardware hard constraints**: rejected — this would treat
  placement as a pure performance problem and silently allow a shard
  carrying sensitive state onto an unattested or residency-inappropriate
  node whenever doing so improved latency or cost, contradicting this
  program's existing witness-chain and isolation disciplines.
- **Report the 1.79× figure as applying to both the 2-node/8B and
  4-node/70B configurations, matching the brief's original combined
  phrasing**: rejected outright — the evidence review's precision flag
  confirms these are two separate results in the source paper; reporting
  them combined would be an inaccurate citation this program's evidence
  discipline does not permit.
