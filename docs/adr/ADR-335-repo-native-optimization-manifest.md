# ADR-335: Repo-Native Optimization Manifest and Disciplined Experiment Loop

- **Status**: Proposed
- **Date**: 2026-08-23
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-282 (research-gate), ADR-306 (dream-machine consolidation), ADR-313 (SHAPER loop), ADR-324 (external-grounding veto); `crates/ruvector-sota-bench/harness/`; `schemas/research-manifest-v1.json`; `scripts/research-gate/`; **#920** (flywheel `hard_regression` gate stuck closed — field evidence for this ADR); see `docs/research/perpetual-intelligence-runtime/12-wave5-evidence-review.md` and `13-wave5-program-plan.md`
- **Tags**: pir, wave-5, optimization, manifest, benchmark, darwin, research-gate

## Context

NVIDIA merged an agent optimization skillpack into Dynamo
(`ai-dynamo/dynamo`) on **2026-08-21T04:54:39Z** via
[PR #13557](https://github.com/ai-dynamo/dynamo/pull/13557) — 89 files,
Apache-2.0, live under `.agents/skills/`. It forces coding agents to
establish an objective function, construct a benchmark, isolate
experimental variables, pass adversarial review before consuming GPU time,
and apply domain-specific optimization knowledge. Wave-5 evidence review
grades it **A for source fidelity**; every mechanism claim is verbatim
confirmed (see `12` §1).

**The reported throughput range must be cited with its sample size.**
NVIDIA reports "15% to 77% better throughput in our internal A/B tests
(same model, same GPUs, same goal per pair; **one Claude Code pair and one
Codex pair**, measured with AIPerf)." It is a genuine paired A/B with the
skillpack as the declared variable — but **n = 2 pairs**, no variance, no
repetition, no disclosed workload. **15–77% is a directional anecdote, not
a target this program plans against.** The stronger evidence, which the
briefing did not carry, is in the PR body: every recipe the workflow
produced was "independently re-deployed from its shipped files alone and
re-benchmarked on the full workload, reproducing or beating the agents'
claimed numbers" on SGLang/GB200 and vLLM/H100. Cite that in preference.

**What already exists here — checked at source on `origin/main`, and it is
most of the mechanism.** The scout's audit found the priority-1 elements
already implemented and enforced in
`crates/ruvector-sota-bench/harness/`:

- **Objective function**: `metrics.ts` `darwinScore()` →
  `AggregateMetrics.primary` → `flywheel.ts` `toScore()`. Present, but
  **hard-coded**.
- **Benchmark command**: `benchmark.ts` `runObservedBenchmark` — isolated
  native binary, environment fingerprint, `gitCommit()`, RSS sampling,
  output caps, deterministic cache key.
- **Protected invariants**: `vetoes.ts` `composeVetoProviders` — "any one
  provider can block promotion; none can rescue it." Seven providers
  shipped, including adversarial ones (`redblue` capability veto,
  `dreamMachine` veto).
- **One isolated variable, enforced twice**: `flywheel.ts`
  `nextParameter(base, target)` steps exactly one lever one notch, and
  `benchmark.ts` `normalizePolicy()` **rejects unknown levers outright**
  and range-bounds each of the five.
- **Cost budget**: `costPerWin`, `resource_cost_worsened` above 1.05×,
  and `darwin.ts`'s `costBudgetSeconds` / `costCeilingFactor` /
  `riskBudgetTotal`.
- **Promotion rules**: `ruvectorPromotionRule` + `gateFingerprint()` +
  `verifyReplayBundle()` (throws on replay failure) + a hard assertion
  that holdout and anchor identities are disjoint.
- **Keep / discard / reject**: `statistics.ts` `pairedBootstrapDecision`
  is already a three-way decision — `pass` iff `lower95 >= minimumEffect`,
  `fail` iff `upper95 <= 0`, else `inconclusive`.

**So the gap is not the laboratory. It is that the laboratory cannot be
declared per repository.** Every element above is either hard-coded in
TypeScript or specified per-experiment in
`schemas/research-manifest-v1.json`. A second repository adopting this
loop today would have to fork the harness.

**Field evidence for why declaration matters — issue #920.** Running the
Wave-5 acceptance test surfaced that the flywheel's `hard_regression` gate
is **permanently stuck closed**: **120 of 120 candidates rejected across 30
unattended generations**, including **30 candidates that were strictly
better on primary, recall, qps *and* cost.** The cause is that the gate
tests an **absolute threshold** rather than comparing against the
baseline.

This is the argument for this ADR made empirically rather than
rhetorically. A hard-coded absolute is a value that was correct for one
repository at one moment, frozen into TypeScript, silently wrong
thereafter — and wrong in the direction that rejects genuine improvements
without saying so. It is precisely the class of value that belongs in a
per-repo declaration, where it is visible, versioned, and reviewable.
Tracked separately as **#920**; the fix is not in scope here, but the
manifest layer is what stops the next one recurring.

**Second gap: the Pareto frontier is not ours.** `darwin.ts` passes
`selection: "pareto"` to `evolve()` from the **external** npm package
`@metaharness/darwin@^0.9.1`. This repo consumes a frontier
implementation it does not own. (`environmentDesigner.ts`'s `atFrontier`
is a *different* frontier — task difficulty, not Pareto — and must not be
confused with it.)

## Decision

1. **Add `schemas/optimization-manifest-v1.json`**, a per-repository
   sibling to the existing eight schemas. A repository declares **once**:
   its measurable objective, its protected invariants, its benchmark
   command, its tunable levers, its cost budget, and its promotion rules.
2. **Derive, do not duplicate.** Each experiment's
   `research-manifest-v1.json` is *derived* from the repo's optimization
   manifest — the per-experiment schema keeps its existing required
   fields (`independent_variable`, `decision_rule`, `confirmation_seeds`,
   `budget`, `memory_accounting`, …), and the derivation fills the ones
   the repo has already declared. No field moves; nothing is removed.
3. **Declaration, not authority — this is the load-bearing rule.** An
   optimization manifest is a **proposal from an untrusted surface**
   (Invariant 7). Manifest-declared values are **intersected** with the
   built-in gate, never substituted for it. Concretely, a manifest may
   *narrow* the lever set, *tighten* a budget, or *add* an invariant; it
   may **never** widen `benchmark.ts`'s closed lever set, lower
   `minimumEffect`, remove a veto provider, or relax the disjoint
   holdout/anchor requirement. A manifest that attempts any of these is
   rejected at load, not silently clamped.
4. **One isolated mutation per experiment stays enforced where it is.**
   The manifest declares which levers are tunable; `flywheel.ts`'s
   single-lever stepping and `normalizePolicy`'s unknown-lever rejection
   remain the enforcement points. The manifest cannot introduce a lever
   the native runner does not know.
5. **Adversarial review before spend is retained as a veto, not a new
   mechanism.** The existing conjunctive veto composition already runs
   before promotion and already includes adversarial providers; the
   manifest declares *which* providers a repository requires, and may only
   add to the built-in set.
6. **Pareto frontier — decide explicitly.** This ADR records the
   dependency on `@metaharness/darwin@^0.9.1` for `selection: "pareto"` as
   a **known external dependency**, and requires WP31 to either (a)
   implement an in-repo frontier that this program controls, or (b)
   explicitly accept the external dependency with a recorded rationale.
   Silently depending on it while claiming frontier semantics is not an
   option.

## Consequences

### Positive

- A repository can adopt the optimization loop by writing one manifest
  instead of forking the harness — which is the actual content of
  NVIDIA's skillpack idea, applied to machinery this program already has.
- The objective stops being hard-coded, so "what is this repo optimizing?"
  becomes an answerable, versioned, reviewable question.
- Derivation keeps a single source of truth: the per-experiment manifest
  remains the artifact `research-gate` validates, so no gate changes
  shape.

### Negative

- **A config file that can influence a promotion gate is a
  trust-boundary change.** The intersect-never-substitute rule (Decision
  §3) is what keeps it from becoming a privilege-escalation primitive
  wearing a config file's clothes. That rule is blocking, not advisory.
- Two manifest layers is more surface than one. Mitigated by derivation:
  the per-repo layer adds no new validated artifact, it only populates the
  existing one.
- The external Pareto dependency is now explicit, which is honest but does
  not by itself resolve it.

## Security / Validation Gates

- **Intersect, never substitute** (blocking, Decision §3) — a manifest
  may only narrow. Widening the lever set, lowering `minimumEffect`,
  dropping a veto provider, or relaxing the disjoint holdout/anchor
  assertion is rejected at load with a typed error. Regression-tested with
  a manifest that attempts each.
- **Invariant 7** — the manifest is an untrusted proposal; it cannot issue
  promotion decisions or mutate active policy.
- **No lever the runner does not know** — `normalizePolicy`'s rejection of
  unknown levers stays the enforcement point and is not consulted through
  the manifest.
- **Schema validation is offline** — `scripts/research-gate/` resolves
  `$ref` offline with hash-locked dependencies (`--require-hashes`); the
  new schema inherits that posture so a candidate cannot point validation
  at a network schema.
- **The 15% acceptance-test clause is a research-gate delta** — it must
  clear `pairedBootstrapDecision` with `lower95 >= minimumEffect` on
  held-out confirmation seeds. A self-reported 15% that fails the paired
  bootstrap is a failed experiment.
- **Standard repo gate**: harness `npm run check && npm test` plus
  `scripts/research-gate/` tests.

## Affected Repos

- `ruvnet/ruvector` only — `schemas/optimization-manifest-v1.json` (new),
  `crates/ruvector-sota-bench/harness/src/` (loader + derivation),
  `scripts/research-gate/` (validation wiring). Single-repo scope.

## Dependencies

None hard among Wave-5 ADRs. Builds on merged ADR-282/306/313/324.

## Alternatives Considered

- **Vendor NVIDIA's skills directly**: rejected for this wave. They are
  Apache-2.0 and harness-agnostic, so vendoring is *permitted*, but they
  encode Dynamo's domain knowledge (inference-serving levers), not this
  repo's. The transferable part is the discipline, which this repo already
  implements more strictly — `normalizePolicy` rejecting unknown levers is
  a stronger guarantee than a skill instructing an agent to isolate
  variables. Note for any future vendoring: `.agents/skills` is a symlink
  to `skills/` since `ai-dynamo/dynamo` PR #10017.
- **Put the objective in `flywheel.ts` as a parameter**: rejected — it
  would make the objective a function argument rather than a declared,
  versioned, reviewable artifact, and would not survive across repos.
- **Extend `research-manifest-v1.json` with per-repo fields**: rejected —
  it conflates per-experiment and per-repo lifetimes, and would force
  every experiment to re-declare invariants that change once a quarter.
- **Adopt NVIDIA's 15–77% as the WP31 success target**: rejected — n = 2
  pairs. The program's bar is its own recomputed research-gate delta.
- **Name a module "Dynamo"**: rejected — collision discipline ("Dynamo" is
  DynamoDB, Autodesk Dynamo, Dynomite). Cite as "NVIDIA Dynamo
  (`ai-dynamo/dynamo`)".
