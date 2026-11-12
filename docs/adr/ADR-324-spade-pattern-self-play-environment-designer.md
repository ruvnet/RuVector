# ADR-324: SPADE-Pattern Self-Play Environment Designer for Dream Machine

- **Status**: Proposed
- **Date**: 2026-08-21
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-306 (PIR, extends — Dream Machine evaluation core); ADR-313 (PIR, extends — SHAPER frozen-weight harness evolution, WP9); ADR-321 (PIR, extends — SkillForge candidate-source pattern, WP19, same shape); see `docs/research/perpetual-intelligence-runtime/08-wave3-evidence-review.md` and `09-wave3-program-plan.md`
- **Tags**: pir, wave-3, spade, self-play, environment-generation, darwin, research-gate

## Context

Wave-3 evidence review grades this paper **A** —
[arXiv:2608.19197](https://arxiv.org/abs/2608.19197), "Self-Play in Adaptive
Synthetic Executable Environments," submitted 2026-08-19. Confirmed
mechanism, verbatim against the abstract: a single LLM plays both
**Environment Designer** — which "writes complete, long-horizon training
environments as executable code with an OpenAI Gym-style `reset()`/`step()`
interface" — and **Reasoning Agent**. Difficulty is driven by a regret
signal: the gap between reward earned with and without privileged hints,
which the Designer uses to keep environments solvable while still exposing
the Agent's capability gaps (a self-play, learner-relative curriculum, not a
fixed difficulty ladder).

Numbers (verbatim, exact match to the program brief): "+5.3 on average across
eight held-out math, science, code, and reasoning benchmarks," "+5.7 on
BFCL-v4 multi-turn," "+13.9 on ACEBench-Agent," at 30B parameters.

**The paper's own ablation is the load-bearing fact for this ADR's required
invariant.** Verbatim: grounding the Environment Designer "on documents
sampled from a large pretraining corpus, and giving it an accumulated
environment memory" are **"critical to success."** Read plainly, this is the
paper's own authors reporting that an ungrounded, purely self-referential
Designer — one inventing environments from nothing but its own prior
generations — measurably underperforms one anchored to external documents.
**"Closed-epistemic-loop safeguard" is this program's own framing of why
that ablation matters, not the paper's language** — arXiv:2608.19197 never
uses that phrase, and no derived document may attribute it to the paper.

**Artifact availability — checked, not assumed**: the arXiv comments field
states, verbatim, "Work in progress. Project page:
https://spade-rl.github.io ; Code: https://github.com/spade-rl/spade."
Verified directly via the GitHub API: `spade-rl/spade` is public, 1,389 KB,
3 forks, MIT license, last pushed 2026-08-20T05:20:09Z (star count omitted
here — it drifts continuously and is not load-bearing evidence; size,
forks, license, and push date are), containing real training scripts for
4B/8B/30B models, Slime/SGLang + Megatron-LM
integration, and an eval harness — not a placeholder. This ADR adapts the
confirmed-live implementation, the same posture ADR-320 and ADR-321 apply to
MemFuse and SkillForge.

**Name collision — SEVERE, and binding on this ADR's naming decisions.**
"SPADE" is also a real, long-established, widely-taught academic multi-agent
framework — **S**mart **P**ython multi-**A**gent **D**evelopment
**E**nvironment (FIPA messaging, behavior models, a web GUI) — in the *same*
general domain (multi-agent systems), with dozens of university course
repositories on GitHub confirmed via search. This is not a distant-domain
coincidence like ADR-321's "X-Forge" pattern; it is a same-domain collision
at the severity level of ADR-320's MemFuse/`memfuse` conflict. **This ADR,
and every deliverable derived from it, never uses bare "SPADE" as an npm
package, crate, module name, or unqualified prose reference.** Every
reference disambiguates as **"SPADE (arXiv:2608.19197, Self-Play in Adaptive
Synthetic Executable Environments)"**, explicitly distinct from the
pre-existing Smart Python multi-Agent Development Environment.

ADR-306 already adopts `ruvnet/dream-machine`'s engine, wired to `ruvector`
ADR-282's `research-gate` (`crates/ruvector-sota-bench/harness/src/statistics.ts`
paired-bootstrap significance, `src/vetoes.ts` composable hard vetoes,
`src/dreamMachine.ts` the evaluation-stage adapter — WP2, issue #838).
ADR-313 gives Darwin a frozen-weight harness-evolution loop; ADR-321 already
added one self-supplied candidate-generation source (SkillForge-pattern
synthetic issues, `src/skillForge.ts`, WP19) into that same loop. SPADE's
contribution is a **second, structurally different** self-supplied source:
instead of synthesizing point regressions from existing test coverage, it
generates whole **sandboxed executable environments** — training
curricula — that the learner's own regret signal keeps at the frontier of
its capability. Genuinely new: a targeted check of `ruvector`,
`dream-machine`, `metaharness`, and `ruflo` for "self-play," "environment
designer," or "regret signal" found no existing implementation anywhere in
`ruvnet`.

**Preprint-reproduction rule** (applies uniformly across this program, per
`09-wave3-program-plan.md`): SPADE (arXiv:2608.19197) is a **candidate
mutation input**, not adopted prior art whose numbers this program is
entitled to claim. The paper's own "+5.3 average / +5.7 BFCL-v4 / +13.9
ACEBench-Agent" figures are the *source paper's* published numbers, at 30B
parameters, on its own benchmark suite — not a promotion bar this program
may cite as if it were `ruvector`'s own measured result. Every reported gain
is a hypothesis until this program's own `research-gate`-recomputed
paired-bootstrap delta confirms it, exactly as ADR-317 already states for
HarnessRisk.

## Decision

Add an **EnvironmentDesigner** module to the Dream Machine evaluation core
(ADR-306), informed by SPADE's mechanism (arXiv:2608.19197) and adapting the
confirmed-live `spade-rl/spade` implementation:

1. **Environment generation as a new Darwin candidate source.** The
   EnvironmentDesigner generates sandboxed, executable RVF environments —
   each exposing an `reset()`/`step()`/reward/verification interface, per
   SPADE's confirmed Gym-style contract — as one more candidate-mutation
   source feeding ADR-313's SHAPER loop, alongside ADR-321's synthetic-issue
   source (`src/skillForge.ts`) and ADR-313's existing surfaces
   (`harness/src/darwin.ts`, `examples/mragent`'s `scorePolicy`,
   `crates/sona/src/darwin_guard.rs`).
2. **Regret-driven difficulty, following the learner.** Environment
   difficulty tracks the regret signal (reward gap with vs. without
   privileged hints) against the current learner's own measured capability,
   not a fixed schedule — an environment that stops producing regret signal
   (the learner has mastered it) or that the learner cannot make progress in
   at all (regret stays saturated) is a candidate for retirement, not
   indefinite reuse.
3. **Darwin promotes environments that stay solvable while exposing
   capability gaps.** An environment is itself a Darwin-evolved artifact:
   ADR-306's evaluation pipeline scores whether a generated environment is
   (a) solvable in principle and (b) informative — it continues to surface a
   measurable regret signal against the current learner — before that
   environment is retained in the active training pool. An environment that
   fails either property is rejected the same way any other Darwin mutation
   proposal is rejected.
4. **CRITICAL REQUIRED INVARIANT — external-evidence grounding, enforced as
   a hard veto, not a guideline.** Every generated environment MUST trace to
   external evidence: a repository's own tests, a real observed failure, a
   physical observation, or an independently published benchmark. A closed
   self-invented loop — where the EnvironmentDesigner's only input is its
   own prior generations, with no external anchor — is forbidden. This is
   both this program's own safeguard and a direct generalization of SPADE's
   own ablation finding (external corpus grounding "critical to success").
   Concretely: `research-gate`
   (`crates/ruvector-sota-bench/harness/src/statistics.ts`) gains a new
   external-grounding veto, composed via `vetoes.ts`'s existing
   `composeVetoProviders` alongside the paired-bootstrap veto, and it **fires
   before Dream Machine's evaluation stage runs** (`src/dreamMachine.ts`),
   not after — an environment with no traceable external-evidence citation
   never reaches evaluation at all, rather than being evaluated and then
   discarded.
5. **Naming discipline (binding, no exceptions)**: this ADR, its
   implementation, and every derived design document cite the source as
   "SPADE (arXiv:2608.19197, Self-Play in Adaptive Synthetic Executable
   Environments)" in full on every reference; no package, crate, module, CI
   job, or doc heading uses bare "SPADE."
6. **This program's Wave-3 acceptance test governs promotion, not SPADE's
   own reported numbers** (see Consequences and the verbatim acceptance
   criterion below) — the EnvironmentDesigner is validated against this
   program's own held-out task split and `research-gate`-measured delta.

### Acceptance criterion (verbatim, attribution: ruv, 2026-08-20)

Per `09-wave3-program-plan.md`, the following is carried verbatim as this
ADR's acceptance test, attributed to ruv:

> "freeze the underlying model for 30 days; allow only harness, memory,
> topology, tools, and generated environments to evolve; hold out 20% of
> real tasks the evolution system never sees; if held-out success improves
> at least 10% while capability violations remain zero and every promoted
> change has reproducible provenance, you have demonstrated genuine
> substrate-level learning rather than benchmark adaptation."

This composes with the existing WP12 30-day acceptance harness and ADR-315's
capability-expansion gate — "capability violations remain zero" is ADR-315's
zero-unapproved-capability-expansion requirement, already queryable via its
witness log. Two requirements are new on top of what those two ADRs already
build: (a) a genuinely held-out 20% task split the evolution system never
trains or self-evaluates against, and (b) a ≥10% held-out-success threshold
measured via `research-gate`'s existing paired-bootstrap statistics
(`statistics.ts`). No new acceptance-harness infrastructure is required —
this extends WP12's harness rather than replacing it.

## Consequences

### Positive

- Gives Darwin a third distinct self-supplied candidate-generation
  strategy — after ADR-313's manual/incident-driven surfaces and ADR-321's
  synthetic-issue source — that generates entire training curricula rather
  than point regressions, materially widening the mutation-candidate space
  the frozen-weight harness-evolution loop can exercise against.
- The confirmed-live `spade-rl/spade` implementation gives the owning work
  package a genuine port-and-adapt scope (training scripts, eval harness,
  Slime/SGLang + Megatron-LM integration already exist) rather than a
  from-scratch build.
- Making external-evidence grounding a hard veto — fired before evaluation,
  not after — directly operationalizes the paper's own strongest empirical
  finding (grounding is "critical to success") as a structural safety
  property of this program's promotion pipeline, not merely a design
  preference.
- ruv's verbatim acceptance test gives this ADR a concrete, falsifiable
  bar (held-out 20% split, ≥10% improvement, zero capability violations,
  reproducible provenance) that composes with infrastructure this program
  already builds (WP12, ADR-315) rather than requiring new harness work.

### Negative

- A closed self-invented loop is exactly the failure mode a
  reward-hacking-prone self-play system would drift toward under pressure to
  keep producing "solvable" environments cheaply; the external-grounding
  veto is a hard structural defense, but its own correctness (is a cited
  "external evidence" reference actually external and actually load-bearing,
  or a superficial citation of convenience?) is itself a validation surface
  this program must build test coverage for, not assume correct by
  construction.
- The SPADE/Smart Python multi-Agent Development Environment name collision
  is same-domain and severe; every package, module, CI job name, and prose
  reference must be checked against the disambiguation rule, and a single
  missed instance (a commit message, an issue title, a crate name) reintroduces
  exactly the ambiguity this ADR exists to prevent.
- SPADE's own reported numbers (+5.3/+5.7/+13.9) are at 30B parameters on
  its own benchmark suite; this program has no reason to expect the same
  magnitude on `ruvector`'s own tasks and model scale, and must not cite
  those figures as an expected result for its own implementation.
- Environment generation-as-mutation adds a new class of Darwin candidate
  distinct from harness/skill mutations (ADR-313, ADR-321) — the owning work
  package must extend `research-gate`'s veto and scoring surfaces to handle
  environment-shaped candidates, not merely reuse the existing scoring
  functions unmodified.

## Security / Validation Gates

- **External-grounding hard veto (this ADR's core mechanism)**: every
  generated environment must carry a traceable citation to external
  evidence (repo tests, a real observed failure, a physical observation, or
  an independently published benchmark); a candidate with no such citation
  is vetoed by `research-gate` before Dream Machine's evaluation stage runs.
  Blocking, not advisory — composed via `vetoes.ts`'s existing
  `composeVetoProviders` alongside the paired-bootstrap veto.
- **No fine-tuning path**: environment generation and the Reasoning Agent's
  training against generated environments both reuse ADR-313's existing
  structural (CI-enforced) frozen-weights check; this is a harness/curriculum
  mutation, never a weight update.
- **Separation-of-powers invariant** (ADR-305, ADR-313, ADR-315): a
  generated environment is a candidate proposal only; it gains no promotion
  authority and must clear ADR-306's evaluation pipeline and, where it would
  expand capability, ADR-315's constitutional gate, like any other
  Darwin-proposed mutation.
- **Witness-chain requirement**: every environment-generation and
  retirement decision emits an RVM witness record (ruvector ADR-134 schema),
  anchored via this program's shared witness/anchoring contract (ADR-312),
  so the acceptance criterion's "reproducible provenance" requirement is
  queryable end-to-end.
- **Held-out task isolation**: the 20% held-out task split used for the
  acceptance criterion is never visible to the EnvironmentDesigner, the
  Reasoning Agent's training loop, or `research-gate`'s promotion scoring —
  enforced as a hard data-isolation boundary, verified at day 0 and re-verified
  at day 30.
- **Naming discipline (binding, no exceptions)**: never ship an npm package,
  crate, or module literally named `spade`; every reference disambiguates as
  "SPADE (arXiv:2608.19197, Self-Play in Adaptive Synthetic Executable
  Environments)."
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to the environment-generation or veto-composition code, since
  sandboxed executable-environment generation is itself an untrusted-code
  execution surface.

## Affected Repos

- `ruvnet/ruvector` only — `crates/ruvector-sota-bench/harness/src/{darwin.ts,
  dreamMachine.ts, shaperLoop.ts, skillForge.ts, statistics.ts, vetoes.ts}`,
  plus a new `environmentDesigner.ts` module alongside the existing
  candidate-source files (WP19's `skillForge.ts` sibling source) and a new
  external-grounding veto provider composed into `vetoes.ts`'s existing
  `composeVetoProviders` set, alongside `statistics.ts`'s paired-bootstrap
  decision. Single-repo scope, per `09-wave3-program-plan.md`.

## Dependencies

Depends on ADR-306 (the Dream Machine evaluation core this new candidate
source and its hard veto attach to) and ADR-313 (the frozen-weight
harness-evolution loop this source feeds candidates into), directly, the
same dependency shape ADR-321 already established for its own
candidate-source extension. Interacts with ADR-315's capability-expansion
gate for the acceptance criterion's "capability violations remain zero"
requirement, without altering ADR-315's own scope.

## Alternatives Considered

- **Allow environments to be validated purely by internal
  solvability/regret metrics, without an external-evidence requirement**:
  rejected outright — this is precisely the closed self-invented loop the
  paper's own ablation shows underperforms, and precisely the failure mode
  this program's own safeguard exists to prevent structurally, not by
  policy.
- **Fire the external-grounding veto after Dream Machine's evaluation stage
  instead of before**: rejected — evaluating an ungrounded environment first
  wastes evaluation budget on candidates that can never be promoted and
  risks the veto being treated as advisory cleanup rather than a hard gate;
  firing first makes the invariant structurally load-bearing.
- **Adopt "SPADE" as this program's own package or module name, relying on
  the arXiv citation for disambiguation**: rejected outright — the
  same-domain collision with the established Smart Python multi-Agent
  Development Environment framework is exactly the failure mode ADR-320's
  MemFuse naming discipline was written to prevent, and this collision is at
  least as severe.
- **Cite SPADE's own reported improvement figures as this program's expected
  result**: rejected — per the preprint-reproduction rule, those numbers are
  the source paper's own benchmark result at a different scale and task
  suite; only this program's `research-gate`-measured delta may be cited as
  this program's own result.
