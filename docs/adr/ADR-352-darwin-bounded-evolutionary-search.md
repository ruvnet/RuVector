# ADR-352: `ruvector-darwin` — Bounded Evolutionary Parameter Search

## Status

Accepted. New, standalone, zero-impact-on-existing-code crate
(`ruvector-darwin`), plus one opt-in example in `ruvector-gnn-rerank`
(`examples/darwin_alpha_search.rs`) that applies it. No existing public API
changed. (Number assigned from the highest top-level ADR at the start of
this run, ADR-351; `docs/adr` has previously had concurrent-PR numbering
collisions resolved by renumbering on merge — see ADR-347's history note —
so this number may be adjusted at merge time if another in-flight PR has
already claimed ADR-352.)

## Context

Three independent nightly research cycles — 2026-09-05
(`mincut-gated-forgetting`), 2026-09-11 (`mincut-partition-determinism`),
and 2026-09-16 (`witness-signer-agent-memory`) — each explicitly state, in
their own "Future Research" sections, that no Darwin/evolutionary/Flywheel
tooling exists anywhere in the repository, and that whatever hand-tuning
they did (batch sizes, thresholds) was a manual sweep, not a search. Two
earlier PoC-level nightlies separately name the same gap for their own
hand-set constants:

- 2026-05-21 `gnn-rerank`: `GnnMincutReranker`'s `alpha` (self-weight) and
  `coherence_threshold` were hand-set to 0.60 / 0.50; the doc states
  "optimal alpha calibration for production embeddings is unknown" and
  lists "adaptive alpha tuning via a lightweight ruFlo feedback loop" as
  unimplemented future work.
- 2026-06-16 `coherence-hnsw-search`: `AdaptiveCoherenceSearch`'s
  `adaptation_rate`/`max_threshold` were hand-set; the doc lists a "ruFlo
  feedback loop for threshold self-optimization" under "Next", not done.

`CLAUDE.md` names "Darwin" as an ecosystem capability ("Use Darwin to
explore bounded improvements") and the nightly research harness prompt
devotes five steps (18–22) to a Darwin evolution phase with a named default
budget (3 generations × 4 candidates, max 1 promotion) and fitness-function
discipline. No code implementing any of this was found anywhere in
`crates/`, `npm/packages/`, or `docs/research/nightly/` (verified by grep
across the whole tree, and independently confirmed by three nightly docs'
own text, before this run started).

## Hypothesis

```text
Given ruvector-gnn-rerank's GnnMincutReranker with its 2026-05-21 hand-set
defaults (alpha=0.60, coherence_threshold=0.50, k_graph=8), evaluated on
that nightly's own deterministic synthetic benchmark (N=5000, DIM=128,
seed=42), with its 100 queries split into a 70-query fitness set (used only
during search) and a disjoint 30-query held-out set (used only once, for
final acceptance),

when a bounded (3 generations x 4 candidates, elitist, max 1 promotion)
evolutionary search over (alpha, coherence_threshold, k_graph) is run with
fitness = mean recall@10 on the fitness set only,

then the promoted candidate's recall@10 on the HELD-OUT set should exceed
the hand-set baseline's held-out recall@10 by a non-trivial margin
(>= 0.5 percentage points),

subject to: held-out mean rerank latency not regressing more than 50% vs.
baseline, and the search being exactly reproducible (byte-identical
lineage) across independent re-runs with the same seed.
```

Falsification: if held-out recall does not improve by >= 0.5pp, or
improves on the fitness set but not the held-out set (overfitting), or the
search is not reproducible, the hypothesis is rejected or inconclusive —
and either outcome is retained as evidence, not discarded.

## Decision

1. Add `crates/ruvector-darwin`: a generic, dependency-light (`rand`,
   `serde`, `serde_json` only — all already workspace dependencies) elitist
   (1+λ) evolution strategy over a bounded real-valued genome
   (`Genome { genes: Vec<f64> }`, `Bound { min, max }`). Core entry point:
   `run_evolution(parent, bounds, config, fitness_fn) -> EvolutionReport`.
2. Hard constraints enforced in `run_evolution` itself, not left to caller
   discipline:
   - A mutated gene is always clamped into its declared bound before
     evaluation — never evaluated out of bounds.
   - A candidate whose fitness function returns `Evaluation::Rejected(_)`
     or a non-finite fitness can never be promoted.
   - Evaluation count is exactly `1 + generations * candidates_per_generation`,
     independent of outcome — a fitness function cannot make the search
     run away.
   - If nothing strictly beats the parent, the parent is retained
     (`EvolutionReport::promoted == None`), which is a correct, successful
     outcome, not a failure.
3. The full lineage (every candidate evaluated, every generation, the
   parent, and the promotion decision) is `Serialize`/`Deserialize` and
   dumped as JSON — this is the run's Flywheel/witness evidence artifact
   (`docs/research/nightly/2026-10-06-darwin-evolved-gnn-rerank-hyperparameters/evidence/darwin_lineage.json`).
4. Apply it once, end to end, to `GnnMincutReranker`'s three hand-set
   constants via `crates/ruvector-gnn-rerank/examples/darwin_alpha_search.rs`,
   using a fitness/held-out query split to guard against the search
   overfitting its own objective (see "Benchmark evidence" below).
5. `ruvector-darwin` has no dependents yet beyond this one example — it is
   intentionally generic so `coherence-hnsw`'s `adaptation_rate` and
   `memory-admission`'s future conditioned threshold (both separately named
   in prior nightlies as wanting this) can reuse it without modification.

## Evidence

`cargo run --release -p ruvector-gnn-rerank --example darwin_alpha_search`,
3 independent process runs, Linux, Rust 1.97.0, release build:

| | Fitness-set recall@10 | Held-out recall@10 | Held-out mean latency |
|---|---|---|---|
| Baseline (hand-set) | 38.29% | 38.67% | ~330us (±~30us run-to-run) |
| Darwin-promoted | 47.14% | **51.00%** (+12.33pp) | ~290-330us (ratio 0.89-1.0x) |

Promoted genome: `alpha=0.3171, coherence_threshold=0.2923, k_graph=8`
(k_graph unchanged from default; the search moved alpha and
coherence_threshold). Held-out improvement (+12.33pp) exceeds the
fitness-set improvement (+8.86pp) — the opposite of what overfitting would
produce — and is reproducible byte-for-byte across repeated runs (recall
and lineage are deterministic; only wall-clock latency varies with system
noise, as expected). 9 new unit tests in `ruvector-darwin` (determinism,
elitism, bound-clamping, rejection handling, promotion-gate edge cases) and
the existing `ruvector-gnn-rerank` test suite (14 tests) all pass. Full raw
output and JSON lineage are committed under this nightly's `evidence/`
directory.

**Acceptance: ACCEPT** against the hypothesis above (held-out delta
+12.33pp >= 0.5pp threshold; latency ratio 0.89-1.0x, well under the 1.5x
limit; replay verified byte-identical).

## Consequences

- New, reusable, generic infrastructure (`ruvector-darwin`) closes a gap
  named by name in three separate prior nightlies.
- `ruvector-gnn-rerank`'s shipped `GnnMincutReranker::default()` is
  unchanged by this ADR — the evolved parameters are a measured
  recommendation (see "Production path" in the nightly README), not a
  silent behavior change, since changing a public struct's `Default` is a
  larger decision than this one bounded experiment should make unilaterally.
- Other nightlies that named this exact gap (coherence-hnsw,
  memory-admission) now have a ready-made, tested primitive to reuse
  instead of hand-tuning or re-deriving a search loop.

## Alternatives

- **Hand-sweep a small grid** (what every prior nightly did): cheaper to
  write per-run, but produces no reusable infrastructure and each nightly
  re-derives its own ad hoc sweep, which is exactly the repeated gap this
  ADR closes.
- **Full AutoML / surrogate-model search** (Bayesian optimization, CMA-ES):
  more sample-efficient at high dimensionality, but overkill for the
  3-parameter, cheap-to-evaluate problems this repo's nightlies have
  actually hit so far, and harder to audit end to end. Left as a possible
  future upgrade behind the same `run_evolution` call shape.
- **LLM-guided evolutionary code search** (e.g. the RankEvolve pattern,
  arXiv 2026, discussed in the nightly README's SOTA section): mutates
  algorithm *code*, not scalar hyperparameters; a different, larger-scope
  problem than this ADR addresses.

## Security / Governance

`ruvector-darwin` only calls a caller-supplied, pure fitness closure on
caller-supplied bounds — it performs no I/O, no network access, and no
filesystem access itself. It cannot promote a candidate the caller's own
fitness function did not score, and it cannot be pointed at unbounded
parameter ranges by construction (`Bound::new` requires `min <= max`, and
every mutation is clamped). The example applies it only to an in-process,
synthetic benchmark; it does not touch production configuration, a running
index, or any external system. No secrets, network calls, or write
authority are involved.

## Failure modes

- A caller fitness function that is itself overfit to its own evaluation
  set will produce a promoted candidate that looks good in-sample; this is
  exactly why the gnn-rerank application evaluates on a disjoint held-out
  set rather than trusting the search's own fitness score as the final
  number — callers reusing this crate must do the same.
- A fitness function with high run-to-run variance (e.g. small query
  counts) can make `running_best_fitness` track noise rather than signal;
  `EvolutionReport` retains every candidate's raw evaluation so this is
  auditable after the fact, but `run_evolution` itself does not detect it.

## Migration / Rollback

Purely additive: new crate, new example, one new workspace member line.
Rollback is `git revert` with no other repository state to unwind.

## Open questions

- Should `ruvector-gnn-rerank::GnnMincutReranker::default()` actually adopt
  the evolved parameters? Left to a human maintainer decision (see
  "Production path" in the nightly README) rather than decided by this
  bounded experiment.
- Does the (alpha, coherence_threshold) optimum found here generalize past
  this synthetic multi-Gaussian corpus to real embedding distributions?
  Explicitly not claimed — see "What this run does not claim" in the
  nightly README.
