# Wave 2 Program Plan — Six Aug 18–19, 2026 Papers

Status: draft for ADR seeding, swarm work-package assignment, and GitHub
issue filing. Compiled 2026-08-20. Depends on `06-wave2-evidence-review.md`
(evidence grades and artifact availability) and, transitively, on
`01-04` and the merged ADR-305–315 (Phase 2 complete, `main` `0049892cb`).
**No Wave-2 issues are filed by this document** — per the master tracking
issue's stated convention, filing happens only after this plan is approved
by the coordinator, the same gate Wave 1's `03-program-plan.md` used.

## How this wave differs from Wave 1

Wave 1 added six new bounded contexts and eleven ADRs because the founding
claims described genuinely new mechanisms with no PIR home yet. **Every
paper in Wave 2 extends an ADR that already exists on `main`.** None of the
six needs a new bounded context. The work here is narrower and more
verification-heavy: six candidate mutations to an already-defined system,
each requiring its own benchmark delta before it earns a place in the
runtime — not six new subsystems.

**Ruv's handling of the preprint-reproduction risk, applied uniformly below**:
every one of these six papers is treated as a **candidate mutation**, run
through the *existing* Darwin → Dream Machine/`research-gate` → proof-gate
pipeline (ADR-306, already merged) exactly like any other proposed change —
not adopted as trusted prior art. This resolves the artifact-availability
gap `06-wave2-evidence-review.md` found for TRUSS and StagedWorkspace (both
confirmed to have no available code today — MemFuse was initially suspected
the same but a second verification pass found and confirmed its code and
benchmark at `github.com/Darwin-Agent/Mi-Memory/tree/master/MemFuse`): since
nothing is adopted wholesale even where code exists, the missing upstream
code for TRUSS/StagedWorkspace is not a blocker, only a reason the
reproduction work is larger for those two than for HarnessRisk, MemFuse,
SkillForge, and pipeline-shards (which all have confirmed code to start
from). **Promotion requires the internal
benchmark to show a delta over the current baseline, independently measured
by `research-gate`'s paired-bootstrap statistics — not a citation of the
paper's own reported numbers.**

**Acceptance test — ruv's wording, verbatim (resolves the earlier "not
located" gap).** A prior pass of this document flagged that the brief's
requested "ruv's acceptance test verbatim" text could not be found in the
master tracking issue (`ruvnet/ruvector#837`) and declined to fabricate one.
That gap is now closed: ruv supplied the exact wording directly (2026-08-20),
reproduced here verbatim and attributed:

> Take MetaHarness today, run HarnessRisk as the baseline, add RVF bound
> workspace states and TRUSS style shadow execution, then rerun the
> identical cases. The implementation is successful only if utility stays
> above 90% while attack success and persistent compromise both fall by at
> least 75% relative to baseline.
>
> — ruv, 2026-08-20

This is a Wave-2-specific acceptance test, **distinct from** the existing
30-day acceptance harness (WP12, ADR-305 §5's acceptance-test section in
`03-program-plan.md`) — it governs the combined ADR-317/318/319 slice
specifically (HarnessRisk lifecycle benchmark + StagedWorkspace-pattern RVF
state binding + TRUSS-pattern shadow execution), not the whole program. It
is carried verbatim as ADR-317's acceptance criterion (see that ADR) because
HarnessRisk is the baseline instrument the test names directly, and is
referenced from ADR-318 and ADR-319 as the combined-effect test their own
work packages must ultimately satisfy once WP15–17 all land. Per the
"candidate mutation" framing above, satisfying this test requires an
internal, `research-gate`-recomputed measurement against a rerun of
HarnessRisk's own 128 cases — not a citation of HarnessRisk's own reported
baseline numbers as if they already applied to a MetaHarness-plus-RVF-plus-
TRUSS-shadow-execution configuration nobody has benchmarked yet.

**ADR numbering**: per the master index (PR #857, `docs/adr/INDEX.md`
regenerated at 348 entries), **ADR-316 is reserved for the WP0a numbering-hygiene
policy ADR** (not yet merged) and **the next available number is
ADR-317**. This plan assigns ADR-317 through ADR-321, then **skips 322 and
uses ADR-323**, for the six papers, in ruv's stated priority order, and
confirms each against the live filename list at kickoff exactly as ADR-305
§4 requires — the numbers below are placeholders subject to that check,
same as Wave 1's plan stated for ADR-305–315.

**Why 322 is skipped deliberately, not just left free**: a `ruvector`-local
ADR-322 would not be a filename collision (`ruflo`'s ADR-322 lives in a
different repo's `v3/docs/adr/`), but this exact program has already hit the
"wrong repo owns this ADR number" failure mode four times: the ADR-103
cross-repo ambiguity (ADR-305 §4); "metaharness ADR-322" → actually
`ruflo` ADR-322 (`04-verification-addendum.md` §5); the phantom "metaharness
ADR-251" citation, which does not exist at all
(`04-verification-addendum.md` §8b); and "ruvector ADR-150" → actually
`ruflo` ADR-150 (`04-verification-addendum.md` §8c). Separately,
ADR-305/306/310/312/313 already cite `ruflo` ADR-322/322A/322B/322C by
number repeatedly as the flywheel-receipt/promotion authority this very
program adopted. Minting an unrelated `ruvector` ADR-322 for pipeline-sharded
serving would put two heavily-cited, differently-scoped "ADR-322"s in
concurrent use across sibling repos of the same program — exactly the
ambiguity ADR-305 §4 requires every cross-repo reference to avoid. ADR-323 is
free (per the current `docs/adr/INDEX.md`) and carries no such collision;
confirm this is still true against the live filename list at kickoff, same
as every other placeholder number in this document.

**WP numbering**: continues from the highest existing number. WP0a/WP0b and
WP1–WP14 already exist (WP14 = RuView/CSI, still the highest-numbered,
unrelated stretch item). This wave adds **WP15 through WP20**.

---

## Priority order (ruv's stated sequence, carried verbatim)

> HarnessRisk + StagedWorkspace + TRUSS first; then MemFuse; then
> SkillForge; shards later.

Mapped to phases below: **Phase W2-1** = WP15, WP16, WP17 (parallel, no
cross-dependencies among the three). **Phase W2-2** = WP18. **Phase W2-3** =
WP19. **Phase W2-4** = WP20 (no dependency on the others, can start anytime
resourcing allows — same "independent parallel track" pattern Wave 1 used
for WP13).

---

## ADR list and mapping

| # | Paper | Proposed ADR | Extends (merged) | Genuinely new |
|---|---|---|---|---|
| 1 | HarnessRisk | **ADR-317** — Lifecycle harness-safety benchmark gate on Darwin mutations | ADR-313 (SHAPER frozen-weight harness evolution, WP9) and ADR-306 (Dream Machine gate, WP2) | A phase-by-phase (not just outcome-level) security benchmark on every harness mutation Darwin proposes, with "Configuration" flagged as the highest-risk phase to weight accordingly in the proof-gate's veto rules |
| 2 | StagedWorkspace | **ADR-318** — Content-hash-bound versioned views for workspace/knowledge-work artifacts | ADR-307 (three-level persistent memory, WP3/WP4) and ADR-312 (shared witness schema + anchoring contract, WP8) | Generalizes ADR-312's anchoring contract — already designed for mutation/promotion records — to arbitrary knowledge-work artifacts (documents, spreadsheets, notebooks) the runtime's agents read/write, closing a gap ADR-312 didn't originally scope |
| 3 | TRUSS | **ADR-319** — Shadow-execution admission gate for generated skills and harness candidates | ADR-311 (anomaly quarantine, WP7), ADR-315 (capability-expansion gate, WP11), and ADR-306 (Dream Machine, WP2) | A brokered-tool-call sandbox stage between "Darwin proposes" and "Dream Machine evaluates" — every candidate's tool calls are inspected before reaching the execution backend, not just its final behavior scored after the fact |
| 4 | MemFuse | **ADR-320** — Cross-source causal-graph fusion layer on the continuous-latent-state memory tier | ADR-307 (three-level memory, specifically the continuous-latent-state tier) and ADR-310 (causal-attribution gate, WP6) | Fuses atomic, source-tagged observations from multiple agents/sensors into one causal graph with provenance back to the originating event — extending ADR-307's single-agent memory design to the Latent Communication Fabric's multi-agent setting |
| 5 | SkillForge | **ADR-321** — Self-distilled, entity-grounded skill synthesis inside the Darwin mutation loop | ADR-313 (SHAPER, WP9), directly | A specific mutation-generation strategy for Darwin: synthesize test-covered-functionality issues from the target repo itself (rather than waiting for real incidents), distill entity-grounded skills, feed them into the existing frozen-weight harness-evolution loop as one more candidate source |
| 6 | Pipeline shards | **ADR-323** *(322 deliberately skipped — see numbering note above)* — Pre-compiled pipeline-sharded serving for ruvLLM on heterogeneous multi-node fleets | ADR-314 (KV-cache cross-model migration, WP13) | A distinct serving-layer improvement (pipeline parallelism across nodes with pre-compiled per-stage OpenVINO-equivalent graphs and speculative decoding) — ships independently of ADR-314 since it addresses multi-node topology, not cross-model cache reuse, but lives in the same Cross-Model Cognition Migration context and the same `crates/ruvllm` surface |

None of the six requires a new bounded context (see table in `03-program-plan.md`) — all six map onto contexts already defined: **World Verification** (317, 319), **Persistent Memory Governance** (318, 320), **Physical Skill Evolution** (321), **Cross-Model Cognition Migration** (323).

---

## Work packages (sized for 6–8 agent swarm teams, per project anti-drift config)

| # | Package | Extends | Team composition | Depends on |
|---|---|---|---|---|
| WP15 | Reproduce HarnessRisk's 128-case benchmark against current Darwin-proposed harness configs; wire "Configuration" as a weighted risk factor into the proof-gate veto rules (ADR-317) | WP9, WP2 | coordinator, security-architect, coder, tester | WP9, WP2 (gates on the harness-evolution and Dream Machine pipelines already existing; the HarnessRisk repro itself can start once WP9's GGUF blocker clears, in parallel with WP9's own remaining scope) |
| WP16 | Implement content-hash-bound versioned views on `ruvector-agent-memory`/RVF artifacts, modeled on StagedWorkspace's SW-AGENT design (no upstream code — build from paper description); benchmark delta measured on an internal OfficeQA-equivalent task set, not the paper's own reported 8.3–12.1pp (ADR-318) | WP3, WP4, WP8 | coordinator, backend-dev, memory-specialist, tester | WP3, WP4 (three-tier memory + TARL ledger must exist first), WP8 (anchoring contract) |
| WP17 | Build a brokered-tool-call shadow-execution sandbox stage for Darwin-proposed skill/harness candidates, modeled on TRUSS's Static-Gate + Controllable-Execution-Environment design (no upstream code — build from paper description); wire it between Darwin's proposal step and Dream Machine's evaluation step (ADR-319) | WP7, WP11, WP2 | coordinator, security-architect, coder ×2, tester | WP7 (quarantine), WP11 (capability gate), WP2 (Dream Machine) |
| WP18 | Add a cross-source causal-graph fusion layer to the continuous-latent-state memory tier, adapting the released implementation ([github.com/Darwin-Agent/Mi-Memory/tree/master/MemFuse](https://github.com/Darwin-Agent/Mi-Memory/tree/master/MemFuse), confirmed live) rather than rebuilding from the paper alone; benchmark against the released MemFuseBench (ADR-320). **Naming**: never ship an npm package, crate, or module literally named `memfuse` — disambiguate from the unrelated, actively-used `memfuse/memfuse` open-source LLM memory layer in every reference | WP3, WP6 | coordinator, memory-specialist, coder, tester | WP3 (memory tiers), WP6 (causal-audit gate — this WP's fusion output feeds the same audit) |
| WP19 | Add synthetic-issue-from-test-coverage skill distillation as a Darwin mutation-candidate source, adapting SkillForge's released implementation ([github.com/cslsolow/SkillForge](https://github.com/cslsolow/SkillForge), confirmed live) rather than rebuilding from the paper alone (ADR-321) | WP9 | coordinator, backend-dev, coder, tester | WP9 (SHAPER-pattern loop must exist to receive this as a candidate source) |
| WP20 | Implement pre-compiled pipeline-sharded multi-node serving in `crates/ruvllm`, adapting the released reproduction package ([github.com/labscommunity/pipeline-sharded-inference-paper](https://github.com/labscommunity/pipeline-sharded-inference-paper), confirmed live); target the verified 1.79× figure on an internal 2-node/8B-class same-family setup, and treat the 4-node/70B interactive-serving result as a separate, non-1.79×-labeled target per `06-wave2-evidence-review.md`'s conflation flag (ADR-323) | WP13 | coordinator, performance-engineer, coder ×2, tester | WP13 (shares `crates/ruvllm` serving surface; no hard blocking dependency — can start immediately in parallel, same pattern as WP13 itself in Wave 1) |

Use `hierarchical` topology, `max-agents 8`, `specialized` strategy per
project config, same as every prior PIR work package.

---

## Repo assignments

All six are scoped to `ruvnet/ruvector` — unlike Wave 1, none of these
papers requires new cross-repo coordination with LatentMesh, rvm,
autogenous, or RuView. Specific surfaces:

- **WP15** (ADR-317): `crates/ruvector-sota-bench/harness` (vetoes.ts extension), `crates/sona/src/darwin_guard.rs`
- **WP16** (ADR-318): `crates/rvf`, `crates/rvm` (extends the ADR-312 anchoring contract), `ruvector-agent-memory`
- **WP17** (ADR-319): new module alongside `crates/rvm/crates/rvm-cap` (capability/tool-broker enforcement) and Darwin's mutation-proposal surface (`harness/src/darwin.ts`)
- **WP18** (ADR-320): `crates/rvf/rvf-adapters/agentdb`, `crates/rvf` (continuous-latent-state tier), coordinates with `latentmesh-align`-consuming code per the existing ADR-310 CI gate
- **WP19** (ADR-321): `harness/src/darwin.ts`, `examples/mragent` `scorePolicy` — same surfaces WP9 already owns
- **WP20** (ADR-323): `crates/ruvllm` (`kv_cache.rs`, `paged_attention.rs`, `serving/kv_cache_manager.rs` — same files WP13 touches, coordinate to avoid merge conflicts)

---

## Security / validation gates (in addition to the ones already governing WP1–WP14)

- **No wholesale adoption without a measured delta**: per the framing above, none of WP16/17/18's mechanisms may be merged on the strength of the paper's own reported numbers alone — `research-gate`'s independent paired-bootstrap recomputation (ADR-322C interop, already adopted via ADR-312) must show a delta over the pre-WP baseline before proof-gate promotion.
- **WP17 is the highest-security-sensitivity item in this wave**: it adds a new tool-call interception point on the Darwin mutation-proposal path. Apply the same "proposer produces untrusted candidates only" separation-of-powers invariant ADR-305 already adopted from ruflo ADR-322B — the shadow-execution broker inspects and can block, but never itself promotes.
- **Name-collision citation discipline**: per `06-wave2-evidence-review.md`, every reference to TRUSS or MemFuse in code comments, ADRs, or issue text must spell out the arXiv ID on first use — TRUSS collides with the live `truss-agent.com` product (agent permission/tool-approval review, uncomfortably close to WP17's own subject matter) and MemFuse collides with the live, actively-forked `memfuse/memfuse` open-source LLM memory layer. HarnessRisk carries a lower-severity naming risk against the unrelated prior paper "Harness-Bench" (arXiv:2605.27922) — cite HarnessRisk's arXiv ID too, as a lighter-touch version of the same discipline.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after WP17 and WP19 land, given both touch mutation-proposal/capability surfaces.

---

## Top risks

1. **Two of six papers have no confirmed upstream code (TRUSS, StagedWorkspace) — both are top-3 priority.** (MemFuse was initially suspected to lack code too; a second verification pass found and confirmed `Darwin-Agent/Mi-Memory/tree/master/MemFuse` is real and live.) WP16 and WP17 are reproduction-from-description efforts, not integration efforts, which is a materially larger scope than WP15/18/19/20's "adapt a released repo" pattern. Size these two WPs accordingly and do not schedule them as if a working reference implementation exists to port.
2. **The preprint-reproduction risk, handled ruv's way**: every one of these six papers reports numbers on its own benchmark, evaluated by its own authors. Per the "candidate mutation" framing above, none of those numbers may be cited as the acceptance bar for a promoted PIR change — only this program's own `research-gate`-recomputed delta counts. The risk is schedule pressure causing a WP to shortcut this and cite the paper's number directly; the mitigation is already structural (ADR-306's existing gate), but it depends on WP15–20 actually routing through it rather than treating it as optional given each paper's own artifact already "proves" the number.
3. **Two sharp name collisions (TRUSS/`truss-agent.com`, MemFuse/memfuse) create real mislabeling risk in fast-moving swarm work.** Unlike Wave 1's low-risk LiveMem/iOS-app collision, both of this wave's collisions are with live products in the *same* problem domain, meaning a hurried commit message, ADR reference, or Slack-style status update reading just "TRUSS" or "MemFuse" is genuinely ambiguous to a reader who doesn't already know this program's citation-discipline rule.
4. **The pipeline-shards headline number requires care to cite correctly.** The 1.79× figure and the 4-node/70B configuration are two separate results in the source paper; WP20 and any ADR-323 text must keep them separate, per the conflation flag in `06-wave2-evidence-review.md`.
5. **The novelty claim is not proven, only not-yet-falsified.** If a full 5-property match surfaces later (the evidence review's caveat explicitly says this search wasn't exhaustive), it would not block this wave's WPs — they're framed as internal candidate mutations either way — but it would change how this program is allowed to describe its own novelty in any public-facing ADR or status update.
6. **The Wave-2 acceptance test (now resolved, see above) is a combined-effect test spanning three ADRs (317/318/319).** WP15–17 issues should reference it explicitly rather than each WP inventing its own narrower success criterion, and no WP should claim the combined test is satisfied until all three of WP15/16/17 have landed and been benchmarked together against a HarnessRisk rerun.

---

## GitHub issue breakdown (ready-to-file specs only — do not create them)

Per the master tracking issue's stated convention, Wave-2 issues are filed
only after the coordinator approves this plan. All six issues are
single-repo (`ruvnet/ruvector`) — no epic-per-repo split is needed the way
Wave 1 required for LatentMesh/rvm/autogenous/RuView. Suggested labels:
`pir`, `wave-2`, `adr`, plus `security` for WP17, `fast-follow` for WP20
(parallels WP13's label).

| WP | Title | Depends on |
|---|---|---|
| WP15 | `[PIR][WP15] HarnessRisk-based lifecycle security benchmark gate on Darwin harness mutations (ADR-317)` | WP9, WP2 |
| WP16 | `[PIR][WP16] Content-hash-bound versioned workspace views (StagedWorkspace pattern, ADR-318)` | WP3, WP4, WP8 |
| WP17 | `[PIR][WP17] Shadow-execution brokered-tool admission gate for generated skills/harness candidates (TRUSS pattern, ADR-319)` | WP7, WP11, WP2 |
| WP18 | `[PIR][WP18] Cross-source causal-graph memory fusion layer (MemFuse pattern, ADR-320)` | WP3, WP6 |
| WP19 | `[PIR][WP19] Self-distilled entity-grounded skill synthesis in the Darwin loop (SkillForge pattern, ADR-321)` | WP9 |
| WP20 | `[PIR][WP20] Pre-compiled pipeline-sharded multi-node ruvLLM serving (pipeline-shards pattern, ADR-323)` | WP13 (soft) |

Each issue body should link `06-wave2-evidence-review.md` for the evidence
grade and artifact-availability status, and this document for the WP/ADR
mapping — same pattern Wave 1's epics used for `03-program-plan.md`.
