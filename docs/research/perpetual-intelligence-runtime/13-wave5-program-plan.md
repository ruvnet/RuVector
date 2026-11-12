# Wave 5 Program Plan — NVIDIA Dynamo Skills, Cordis, OpenAI Monitoring, EMWM

Status: approved for immediate execution (same procedural posture as Wave
4 — see "Approval status"). Compiled 2026-08-23. Depends on
`12-wave5-evidence-review.md` (evidence grades and artifact availability)
and, transitively, on `01`–`11` and the merged ADR-305–321 / ADR-323–334
(Waves 1–4 complete on `origin/main`, confirmed by direct listing of
`docs/adr/` and by INDEX.md's generated header).

## Approval status

ruv's 2026-08-23 briefing supplies the priority order, per-item
difficulty/value scoring, concrete implementation sketches, a verbatim
acceptance test, and the directive "implement upstream, test, validate,
optimize, and publish, merge." That is coordinator approval in advance;
Wave-5 issues are filed together with this plan rather than after a second
approval round, exactly as Wave 4 did. Everything else — the
preprint-reproduction rule, the research-gate promotion bar, the security
gates — is unchanged.

## The framing constraint, carried verbatim from the briefing

> "The biggest failure mode is building more independent primitives
> instead of strengthening the substrate. These four should therefore
> enter existing projects, not become four new repositories."

This is binding on every work package below, and it is the reason WP31 is
sized as an *extension* rather than a build: the scout's audit found
roughly 80% of the priority-1 mechanism already implemented in this repo
(see WP31). **No Wave-5 work package creates a new repository. Only one
creates a new file of substance in `schemas/`.**

## How this wave differs from Waves 1–4

Wave 1 added six bounded contexts; Wave 2 extended six; Waves 3 and 4 were
mixed. **Wave 5 is the most lopsided wave so far, and honesty about that
is the plan's main job:**

1. **Priority 1 is almost entirely already built.** The scout's
   at-source audit of `crates/ruvector-sota-bench/harness/` found the
   objective function, benchmark command, protected invariants,
   one-isolated-variable discipline, cost budget, promotion rule, and
   three-way keep/discard/reject **all present and enforced today**. WP31
   is therefore a *declaration layer* over existing machinery, not new
   machinery. See the WP31 table for the file-by-file mapping.
2. **Priority 2 cannot be implemented in this repo at all.** ADR-333
   (merged Wave 4) already established that RVM work lands in `ruvnet/rvm`
   under maintainer review, USER ACTION for merge. `ruvnet/rvm`'s PRs #38
   and #39 touch the exact files a CapabilityTransaction would extend, and
   #38 is `REVIEW_REQUIRED`. **Wave 5 cannot merge it.** The ruvector-side
   deliverable is a contract and an ADR.
3. **Priority 3's core primitive already shipped in Wave 4.**
   `crates/ruvector-tiny-dancer-core/src/voi.rs` (ADR-331) was written
   with escalation ladders as a named reuse case in its own module
   docstring. WP33 consumes it; it does not reimplement it.
4. **Priority 4 was already deferred by a merged ADR.** ADR-332 says
   verbatim: "Deferred implementation. No code lands in this wave." WP34
   does not re-litigate that.

**Preprint-reproduction rule, unchanged**: every one of these four items is
a **candidate mutation**, run through the existing Darwin →
dream-machine/`research-gate` → proof-gate pipeline (ADR-306, ADR-282,
both merged) like any other proposed change. Source numbers are
hypotheses; **our bar is our own research-gate delta, independently
recomputed, never the source's own reported figure.** This applies with
particular force to NVIDIA's 15–77%, which rests on **two paired runs**
(see `12` §1) — it is a directional anecdote, not a target.

**Invariant 7, carried unchanged and binding on every WP below** (adopted
Wave 2, from ruflo ADR-322B): a proposer produces untrusted candidates
only; it cannot issue promotion decisions or mutate active policy.

## Wave-5 acceptance test — provided verbatim by ruv, 2026-08-23

> "select one RuV repository and run 30 unattended optimization
> experiments. Each experiment must have a fixed objective, one isolated
> mutation, a signed capability envelope, reversible execution, measured
> cost, and reproducible benchmark result. Success means at least 15%
> improvement on the chosen objective, zero capability escapes, complete
> rollback of every rejected experiment, and a cryptographically
> traceable provenance chain for the winning configuration."

**Honesty constraints on measuring it**, applying this program's standing
metric-integrity discipline:

- **"At least 15% improvement" is a research-gate delta, not a
  self-reported number.** It must clear `statistics.ts`'s paired-bootstrap
  decision with `lower95 >= minimumEffect` on held-out confirmation seeds,
  with the holdout and anchor dataset identities disjoint (already asserted
  in `flywheel.ts`). A 15% improvement that fails the paired bootstrap is
  a failed experiment, not a passed one.
- **"Zero capability escapes" is only measurable against a capability
  envelope that actually exists.** The signed-envelope and
  reversible-execution clauses depend on WP32, whose implementation lives
  in `ruvnet/rvm` and is **blocked on maintainer review**. Until that
  lands, a run can demonstrate the *optimization* clauses (fixed
  objective, isolated mutation, measured cost, reproducible benchmark,
  rollback of rejected experiments, provenance chain) against this repo's
  existing machinery, and must **report the capability-envelope clause as
  unmet rather than approximating it**. Reporting a run as passing the
  full test while the envelope clause is stubbed would be exactly the
  metric-integrity failure this program's HarnessRisk gate exists to
  prevent.
- **"Complete rollback of every rejected experiment" is downgrade-only in
  spirit**: a rollback that cannot be verified counts as a failed
  rollback, never as a successful one.
- **30 unattended experiments is a compute commitment, not a code
  deliverable.** WP31 delivers the manifest layer and the loop; the run
  itself is an infrastructure decision. Sizing it honestly up front is
  preferable to redefining the bar afterwards.

**ADR numbering**: INDEX.md's generated header reads "Next available ADR
number: 335" and matches a direct recount of the live tree. **This wave
allocates ADR-335 through ADR-338**, confirmed free at kickoff. ADR-322
remains permanently skipped (it names ruflo's ADR-322 family). The
`adr-numbering` CI job (PR #905) now enforces this mechanically. Re-run
`node scripts/adr-index.mjs` in the PR that lands these ADRs.

**WP numbering**: continues from WP30 (Wave 4's highest). This wave adds
**WP31 through WP34**.

---

## Priority order (ruv's stated sequence, carried verbatim)

> "NVIDIA optimization skills first, RVM reversible capability
> transactions second, adaptive runtime monitoring third, electromagnetic
> world modeling fourth."

Mapped to phases: **Phase W5-1** = WP31 (the only substantial in-repo
implementation). **Phase W5-2** = WP32 (contract + ADR here; code is
cross-repo and blocked). **Phase W5-3** = WP33 (adaptive monitoring, built
on Wave 4's VoI primitive). **Phase W5-4** = WP34 (EMWM, ADR-only,
stretch).

---

## ADR list and mapping

| # | Item | Proposed ADR | Extends (merged) | Genuinely new |
|---|---|---|---|---|
| 1 | NVIDIA Dynamo skills | **ADR-335** — Repo-native optimization manifest and disciplined experiment loop | ADR-282 (research-gate), ADR-306 (dream-machine), ADR-313 (SHAPER), ADR-324 (external-grounding veto) | A **per-repo declarable** optimization contract — objective, protected invariants, benchmark command, tunable levers, cost budget, promotion rules — from which a per-experiment `research-manifest` is *derived*. Today every one of those is hard-coded in TypeScript or declared per-experiment |
| 2 | Cordis + deepseek-harness + Covenant | **ADR-336** — Reversible, signed capability transactions (cross-repo contract) | ADR-312 (witness/anchoring), ADR-315 (capability-expansion gate), ADR-333 (RVM semantic authority above OpenShell) | Temporal composability applied to capability grants: before-state hash, signed authority, effect ledger, **inverse operation**, expiry, provenance, after-state hash. The differentiation is **signed + reversible + evolvable**, not signed alone |
| 3 | OpenAI monitoring | **ADR-337** — Adaptive runtime monitoring with value-of-information escalation | ADR-331 (VoI primitive — consumed as-is, not reimplemented), ADR-317 (HarnessRisk lifecycle gate) | An escalation ladder where each rung is an `EstimatorSpec`, routed by the existing closed-form VoI rule, with an unconditional-inspection floor around privilege escalation, network access, credential use, runtime mutation, and destructive operations |
| 4 | EMWM | **ADR-338** — Electromagnetic world model via privileged-modality distillation (stretch, ADR-only) | ADR-332 (RF sensing modality router — deferral honoured, not re-litigated) | RGB/LiDAR as *training-only privileged evidence* distilled into a **CSI-only student**, so deployment needs no cameras — a deliberate divergence from the paper's architecture, which requires them at inference |

Bounded-context assignment: **Evolution Benchmarking** (335, extending
282/306/313/324's context), **Secure Runtime Interop** (336, extending
333's cross-repo context), **Runtime Governance** (337 — new mechanism,
existing crates), **RF Sensing** (338, stretch, RuView-adjacent).

---

## Work packages

| # | Package | Extends | Team composition | Depends on |
|---|---|---|---|---|
| WP31 | Add `schemas/optimization-manifest-v1.json` and wire it into the harness so a repository declares its objective, protected invariants, benchmark command, tunable levers, cost budget, and promotion rules **once**, and each experiment's `research-manifest` is derived from it. Decide and record whether an in-repo Pareto frontier is owned or the external `@metaharness/darwin` one is depended on (ADR-335) | ADR-282, ADR-306, ADR-313, ADR-324 | coordinator, system-architect, coder ×2, tester | none hard |
| WP32 | *(Cross-repo, contract + ADR only in this repo)* Specify `CapabilityTransaction`: before-state hash, signed authority, effect ledger, inverse operation, expiry, provenance, after-state hash; and the mount → use → invert → verify → retain-or-revoke lifecycle (ADR-336). Any `ruvnet/rvm` code lands under that repo's maintainer-review requirement — **USER ACTION for merge there** | ADR-312, ADR-315, ADR-333 | adr-architect, security-architect | rvm #38/#39 (blocked — USER ACTION) |
| WP33 | Build the escalation ladder in `crates/mcp-gate`: a cheap detector producing a risk score, rungs expressed as `EstimatorSpec`s, routed by `ruvector-tiny-dancer-core`'s existing `voi::decide`, with an unconditional-inspection floor for the five high-risk operation classes and per-rung cost accounting (ADR-337) | ADR-331 (consumed), ADR-317 | coordinator, security-architect, coder, perf-analyzer, tester | WP28 (VoI primitive — merged) |
| WP34 | *(Stretch, ADR-only this wave)* Record the privileged-modality distillation thesis, the honest greenfield assessment, and the data-generation cost; implementation deferred pending RuView coordination (ADR-338) | ADR-332 | adr-architect only | none |

Use `hierarchical` topology, `max-agents 8`, `specialized` strategy per
project config, same as every prior PIR work package.

---

## WP31 — what already exists, file by file

This table is the plan's most important content, because it is the
difference between "build an optimization laboratory" and "declare the one
we already have." Every row was verified at source on `origin/main`.

| Priority-1 element | Status | Where |
|---|---|---|
| Objective function | **Exists, hard-coded** | `harness/src/metrics.ts` `darwinScore()`; `aggregateReports` → `AggregateMetrics.primary`; `flywheel.ts` `toScore()` |
| Benchmark command | **Exists, closed and pinned** | `harness/src/benchmark.ts` `runObservedBenchmark` — isolated native binary, env fingerprint (platform/arch/node/hostname/binary stat), `gitCommit()`, RSS sampling, output caps, cache key |
| Protected invariants | **Exist, with the right semantics** | `harness/src/vetoes.ts` `composeVetoProviders` — "any one provider can block promotion; none can rescue it." Seven providers already shipped |
| One isolated variable | **Exists, enforced twice** | `flywheel.ts` `nextParameter(base, target)` steps **one** lever one notch; `benchmark.ts` `normalizePolicy()` **rejects** unknown levers and range-bounds each |
| Cost budget | **Exists, three places** | `flywheel.ts` `costPerWin`; `ruvectorPromotionRule` emits `resource_cost_worsened` above 1.05×; `darwin.ts` `costBudgetSeconds`/`costCeilingFactor`/`riskBudgetTotal` |
| Promotion rules | **Exist** | `flywheel.ts` `ruvectorPromotionRule` + `gateFingerprint()` + `verifyReplayBundle()` (throws on replay failure) + disjoint holdout/anchor assertion |
| Keep / discard / reject | **Exists as a three-way decision** | `statistics.ts` `pairedBootstrapDecision` — `pass` iff `lower95 >= minimumEffect`, `fail` iff `upper95 <= 0`, else `inconclusive` |
| Adversarial review before spend | **Exists in spirit** | The veto providers run before promotion; `redblue` capability veto and `dreamMachine` veto are adversarial by construction |
| **Per-repo declaration** | **GAP — this is WP31** | Nothing. Everything above is hard-coded in TS or declared per-experiment in `schemas/research-manifest-v1.json` |
| **Pareto frontier** | **GAP — decide in WP31** | `darwin.ts` passes `selection: "pareto"` to `evolve()` from the **external** `@metaharness/darwin@^0.9.1`. Not implemented in this repo. (`environmentDesigner.ts`'s `atFrontier` is a *different* frontier — task difficulty, not Pareto) |

`schemas/research-manifest-v1.json` already requires `schema_version`,
`commit`, `revision`, `phase`, `claim`, `independent_variable`,
`decision_rule`, `datasets`, `embedding_space`, `exploration_seeds`,
`confirmation_seeds`, `topology`, `budget`, `memory_accounting`,
`environment`, `commands`, `evaluator_version`,
`artifact_retention_class`, and `selection` — roughly 70% of the
priority-1 manifest, but **per experiment**. The new
`optimization-manifest-v1.json` is the **per-repo** sibling it derives
from. The consuming gate already exists: `scripts/research-gate/` with
hash-locked dependencies (`--require-hashes`) and offline `$ref`
resolution, so a candidate cannot point validation at a network schema.

---

## Repo assignments

- **WP31** (ADR-335): `schemas/optimization-manifest-v1.json` (new) +
  `crates/ruvector-sota-bench/harness/src/` (derivation + loader) +
  `scripts/research-gate/` (validation wiring).
- **WP32** (ADR-336): docs/contract only in this repo. `ruvnet/rvm` for
  any code — `crates/rvm-context/src/{capability,receipt}.rs` and
  `crates/rvm-witness/src/{log,replay}.rs` are the extension points, all
  arriving via PRs #38/#39.
- **WP33** (ADR-337): `crates/mcp-gate/` (detector + ladder + accounting),
  consuming `crates/ruvector-tiny-dancer-core/src/voi.rs` unchanged.
- **WP34** (ADR-338): docs only this wave.

---

## Security / validation gates (in addition to those governing WP1–WP30)

- **WP33 is this wave's highest-security-sensitivity item.** A monitoring
  ladder that decides *not* to investigate is a gate that can fail open.
  Four binding requirements, all derived from `voi.rs`'s own documented
  caveats (see ADR-337):
  1. **Every rung must have `cost > 0`, or the caller must cap rounds.**
     `EstimatorSpec::validate` permits `cost == 0`, and a free rung is
     bought forever — the module's own docstring records a probe running
     100,000 rounds still returning `Buy`.
  2. **A `noise_std == 0` rung is an oracle: take its verdict and exit the
     protocol.** `decide()` may return `Buy` for such a rung, which
     `observe()` deliberately refuses.
  3. **`value_of_success` must be expressed as the currency value of
     catching a real violation.** VoI is bounded by ≈`0.4σ`; left at a
     nominal 1.0 the gate degenerates into a never-investigate switch
     **that still looks configured** — the identical failure mode ADR-331
     records for routing.
  4. **Rescale utilities to unit range.** The A&S 7.1.26 `norm_cdf`
     approximation biases one-directionally in the deep tail, toward
     worthless purchases.
- **WP33's unconditional-inspection floor is not subject to the
  economics.** Privilege escalation, network access, credential use,
  runtime mutation, and destructive operations are inspected **always**,
  regardless of what the VoI computation returns. The economics govern
  *discretionary* investigation only. This is the monitoring analog of
  ADR-331's escalate-only invariant: the failure direction must be
  conservative by construction.
- **WP31 must not let a repository weaken its own gate.** An
  `optimization-manifest` is **declaration, not authority**: it may
  *narrow* the lever set, *tighten* a budget, or *add* invariants, and it
  may never widen the closed lever set in `benchmark.ts`, lower
  `minimumEffect`, remove a veto provider, or relax the disjoint
  holdout/anchor requirement. Manifest-declared values are intersected
  with the built-in gate, never substituted for it. Invariant 7 applies: a
  manifest is a proposal from an untrusted surface.
- **No wholesale adoption without a measured delta**: none of WP31–WP34's
  mechanisms may be merged on the strength of the source's own numbers.
  This is stated with emphasis for NVIDIA's 15–77% (n=2 pairs) and for
  EMWM's 0.9699 SGCS (no code, no dataset).
- **Name-collision citation discipline**: never write a bare "Cordis" or a
  bare "Covenant" in an ADR, crate, module, or issue title. Write "Cordis
  (`cordiverse/cordis`)" and "Covenant (`open-covenant/covenant`)". The
  Covenant collision is SPADE-class — `cobbr/Covenant` is a 4,729-star
  .NET C2 red-team framework in an adjacent security domain, and
  `csehammad/covenant-layer` is in *our* domain. Never adopt "Cordis",
  "Covenant", "Dynamo", or "EMWM" as a package/crate/module name.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan`
  after WP33 lands (it touches a trust surface).

---

## Top risks

1. **The acceptance test cannot be fully satisfied this wave, and the
   temptation is to report it as if it could.** Its "signed capability
   envelope" and "reversible execution" clauses depend on WP32, which is
   blocked behind `ruvnet/rvm`'s maintainer review (USER ACTION). The
   honest deliverable is the optimization clauses demonstrated against
   real machinery **plus an explicit statement that the envelope clause is
   unmet**. Stubbing the envelope and calling the test passed would be a
   metric-integrity failure of exactly the kind HarnessRisk exists to
   catch.
2. **WP31 looks like a small documentation task and is actually a
   trust-boundary change.** A per-repo manifest that can widen a lever
   set, lower an effect threshold, or drop a veto is a
   privilege-escalation primitive wearing a config file's clothes. The
   intersect-never-substitute rule above is the mitigation and is
   blocking.
3. **NVIDIA's 15–77% will be quoted as a target.** It is two paired runs
   with no variance, no repetition, and no disclosed workload. Every
   citation must carry "one Claude Code pair and one Codex pair." The
   better evidence — independent re-deployment and re-benchmarking of
   every produced recipe on SGLang/GB200 and vLLM/H100 — should be cited
   in preference to it.
4. **The Pareto frontier is not ours.** `darwin.ts` delegates
   `selection: "pareto"` to `@metaharness/darwin@^0.9.1`, an external npm
   package. If Wave 5's promotion story depends on frontier semantics we
   control, that dependency is a gap that WP31 must either close or
   explicitly accept and record.
5. **Cordis's paper is a self-published preprint under active revision.**
   Its README says the content "may change substantially." Cite by commit
   SHA; a claim built on a section that later moves is unverifiable.
6. **EMWM is the expensive kind of reproduction.** No code *and* no
   dataset — the dataset is the authors' own campus digital twin,
   unreleased. WP34 is ADR-only precisely because sizing it as an
   integration would be wrong by an order of magnitude.
7. **The local working copy is stale.** The `ci/nextest-heavy-timeout`
   checkout has 9 harness source files against 22 on `origin/main`. Any
   Wave-5 work package that reads the local tree will conclude Wave-4
   components are absent and rebuild them. Read from `origin/main`.

---

## GitHub issue breakdown

Filed with this plan (see "Approval status"). Labels: `pir`, `adr`,
`wave-5`, plus phase labels `phase-w5-1..4`, `cross-repo` for WP32,
`stretch` for WP34. The `wave-5` and `phase-w5-*` labels do not exist yet
and are created with the issues.

| WP | Title | Depends on |
|---|---|---|
| WP31 | `[PIR][WP31] Repo-native optimization manifest and disciplined experiment loop (ADR-335)` | none |
| WP32 | `[PIR][WP32] Reversible, signed capability transactions (ADR-336, cross-repo, contract-only here)` | rvm #38/#39 (USER ACTION) |
| WP33 | `[PIR][WP33] Adaptive runtime monitoring with VoI escalation (ADR-337)` | WP28 (merged) |
| WP34 | `[PIR][WP34] Electromagnetic world model via privileged-modality distillation (ADR-338, stretch, ADR-only)` | none |

Each issue body links `12-wave5-evidence-review.md` for evidence grades
and this document for the WP/ADR mapping — same pattern as prior waves.
