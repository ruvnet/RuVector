# ADR-328: AI4AI-Bench Recursive-Improvement Benchmark Adapter

- **Status**: Proposed
- **Date**: 2026-08-22
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-306 (PIR, extends — dream-machine/research-gate promotion pipeline); ADR-313 (PIR — SHAPER frozen-weight loop); ADR-324 (PIR — REQUIRED external-grounding invariant, which this benchmark satisfies by construction); see `docs/research/perpetual-intelligence-runtime/10-wave4-evidence-review.md` and `11-wave4-program-plan.md`
- **Tags**: pir, wave-4, ai4ai-bench, benchmark, darwin, metaharness, recursive-self-improvement

## Context

Wave-4 evidence review grades this paper **A** —
[arXiv:2608.20318](https://arxiv.org/abs/2608.20318), "AI4AI-Bench:
Benchmarking LLM Agents in Algorithmic Design for Recursive
Self-Improvement," submitted 2026-08-20.

Confirmed mechanism, verbatim against the abstract: "10 frozen research
repositories spanning 10 training algorithm families. In each task, an
agent has 4 hours on one B300 to rewrite the training algorithm; its code
is then rerun from scratch for up to 12 hours and scored by a fixed
evaluator hidden from the agent." Numbers, verbatim: "Across 29
configurations of 6 systems on all 10 tasks the mean score is 0.166, and
the best system reaches 0.250," and "More reasoning effort mostly buys the
willingness to go there, taking that minority from 8% of submissions to
64% and the mean score from 0.094 to 0.196."

**The score scale is anchored and must be cited with it**: 0 = an
uninformative model, **0.1 = the algorithm the repository already ships**,
1.0 = the task optimum. "Best 0.250" therefore means the best current
system closes under a fifth of the ship-to-optimum distance — the
benchmark is far from saturated. The abstract further notes most
submissions "never change how the model learns at all, and the minority
that do average 0.226 against 0.126 for the rest" — parameter tuning and
data collection do not count as genuine algorithmic improvement, which is
exactly the "substrate improvement vs. benchmark adaptation" distinction
this program's Wave-3 acceptance test drew.

**Artifact availability — checked, not assumed: code is genuinely
available.** [`Einsia/AI4AI-Bench`](https://github.com/Einsia/AI4AI-Bench)
— Apache-2.0, created on submission day, README badge links
arXiv:2608.20318 directly (confirmed authors' repo per the ADR-305 §6
check-at-source rule), with real content: orchestrator, task suite,
evaluators, 290 released scored trajectories, a Docker image, and a
project page. Adapt-from-repo posture. **Hardware caveat**: official runs
assume one NVIDIA B300 for the 4-hour agent budget plus up to 12 hours of
from-scratch rerun — far beyond CI scale.

**Why this benchmark matters to this program specifically**: ADR-324's
REQUIRED invariant demands that every harness/environment mutation trace
to external evidence and never be validated solely by the loop that
generated it. AI4AI-Bench is external evidence **by construction** —
third-party-frozen repositories, a fixed evaluator hidden from the agent,
and rerun-from-scratch scoring. It is currently the strongest available
external measure of whether the Darwin → dream-machine pipeline performs
genuine algorithmic self-improvement rather than repository optimization.

**Preprint-reproduction rule** (standing, per `11-wave4-program-plan.md`):
the paper's 0.250/0.166 figures are the paper's own measurements of other
systems. Beating 0.250 is a program *target*; nothing about this program's
capability may be claimed except from our own scored runs with
reproducible lineage.

## Decision

Wire AI4AI-Bench (arXiv:2608.20318) into MetaHarness as an
externally-grounded benchmark adapter:

1. **Adapter behind the existing seam, no new registry.** The harness
   deliberately has no plugin registry; benchmark adapters are
   constructor-injected function fields (`RuvectorFlywheelOptions.benchmark`,
   `RuvectorGepaOptions.benchmark`, `runControlledBenchmark`'s runner
   parameter) defaulting to the isolated native runner. The AI4AI adapter
   is a new module (`ai4aiBench.ts`) conforming to that seam — it does not
   modify `benchmark.ts`'s closed lever set.
2. **The research-artifact-emission restriction stays intact.** The
   flywheel hard-refuses research-artifact emission when any injected
   adapter is present ("research artifact emission requires the isolated
   native benchmark runner"). AI4AI runs are exploratory/evaluation runs;
   they cannot mint research-gate confirmation artifacts. This restriction
   is load-bearing and this ADR does not relax it.
3. **Contained execution only.** The AI4AI suite's tasks and evaluators
   are third-party code; the adapter invokes them only in contained
   subprocesses with scrubbed environments — the same posture
   `research-gate` takes toward candidate code and `benchmark.ts` takes
   toward the native binary. Nothing from the suite is imported
   in-process.
4. **Scale honestly split.** This ADR's deliverable is the adapter, its
   contract tests against the released task-suite/evaluator interfaces,
   and smoke-scale local execution of at least one task. Official-scale
   scored runs (B300-class, 4h+12h) are an infrastructure decision
   escalated as USER ACTION, and "beat 0.250" is a program target
   contingent on them.
5. **Registered as external grounding.** AI4AI tasks are registered with
   ADR-324's external-grounding veto as qualifying external evidence, so
   Darwin mutation candidates evaluated against them satisfy the invariant
   by that route.
6. **Reproducible lineage for every mutation.** Every AI4AI-scored
   candidate carries the standard provenance chain (candidate genome,
   commit, seed, suite hash, adapter version) so any reported score can be
   re-derived — the same lineage discipline the flywheel already applies.

## Consequences

### Positive

- The program gains its first externally-authored, externally-evaluated
  measure of genuine algorithmic self-improvement — the exact claim the
  PIR program exists to demonstrate — with the evaluator hidden from the
  agent by the benchmark's own design.
- Adapting a live Apache-2.0 authors' repo is materially cheaper than the
  from-scratch reproductions Wave 4's no-code papers require.
- The adapter slots into an existing, tested seam; no new registry,
  lever, or promotion path is introduced.

### Negative

- Official-scale runs are gated on B300-class hardware this repo's CI
  does not have; until an official run happens, all claims are
  smoke-scale and must be labeled as such.
- The suite is third-party code that will evolve upstream; the adapter
  pins a suite version/hash and must be re-validated on upstream changes.
- A benchmark whose tasks take hours does not fit the harness's existing
  300-second wall-timeout defaults; the adapter needs its own explicit
  budget configuration, kept separate from the native runner's limits.

## Security / Validation Gates

- **Containment**: suite tasks/evaluators run only in contained
  subprocesses with scrubbed env; no in-process import of suite code.
- **No research-artifact emission through the adapter** (flywheel
  restriction preserved, blocking not advisory).
- **Suite pinning**: the adapter refuses to run against an unpinned or
  hash-mismatched suite checkout.
- **Invariant 7**: the adapter and anything it scores are proposers'
  evidence only; promotion decisions remain with the existing
  research-gate/proof-gate path.
- **Standard repo gate**: harness CI (`npm run check && npm test && npm
  run doctor`) plus `cargo check -p ruvector-sota-bench --bin sota-all`.

## Affected Repos

- `ruvnet/ruvector` only — `crates/ruvector-sota-bench/harness/src/`
  (new `ai4aiBench.ts` + tests). Single-repo scope.

## Dependencies

Depends on ADR-306 (the promotion pipeline whose claims this benchmark
externally grounds) and ADR-324 (the external-grounding invariant this
benchmark is registered under). Does not depend on any other Wave-4 ADR.

## Alternatives Considered

- **Build a first-party recursive-improvement benchmark instead**:
  rejected — a self-authored benchmark cannot serve as external grounding
  for the program's own claims (ADR-324's exact failure mode), and the
  authors' suite is live and Apache-2.0.
- **Modify `benchmark.ts` to add AI4AI as a native suite kind**: rejected
  — the native runner's closed lever set and identity-hash discipline are
  load-bearing; a third-party, hours-long, GPU-bound suite belongs behind
  the injection seam, not inside the native runner.
- **Chase the 0.250 headline this wave**: rejected — official-scale runs
  are hardware-gated (USER ACTION), and claiming capability from
  smoke-scale runs would violate the program's own honesty rules.
