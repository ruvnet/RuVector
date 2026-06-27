# ADR-270: Self-Reconstructing Graph Memory — Beyond MRAgent

**Status**: Accepted
**Date**: 2026-06-27
**Authors**: Claude Code MetaHarness Architect
**Supersedes**: None
**Extends**: ADR-269 (MRAgent Graph Memory over RuVector, Darwin-optimized)
**Related**: ADR-260 (Darwin as Evolutionary Substrate), ADR-266 (Darwin ANN
Integration), ADR-256 (MetaHarness SDK), ADR-150 (MetaHarness Integration Surfaces)

---

## Context

ADR-269 implemented MRAgent ("Memory is Reconstructed, Not Retrieved") on RuVector
and used Darwin Mode to tune the reconstruction parameters. That baseline converged
quickly: once Darwin found `traversalDepth=3` the corpus saturated at 100% accuracy,
leaving only a thin cost-Pareto. A sensitivity sweep showed **4 of 10 genes were
dead** (`hybridAlpha`, `fusion`, `rerank`, `promptStrategy` had Δfit ≈ 0) because
the corpus never exercised them, and the only evaluated dimension was raw accuracy.

Three deeper questions were left open — and they are the questions a graph-memory
system has to answer to still be the right design **25 years out**, when agents
hold lifetime memory and the failure that matters is not "missed a fact" but
"confidently fabricated one":

1. **Calibration.** Raw accuracy rewards a system for guessing. A long-lived agent
   must know *when it does not know* and abstain. What is the harness optimizing —
   helpfulness, or risk-adjusted utility?
2. **Adaptive cost.** A fixed traversal depth spends the same compute on a trivial
   recall and a deep multi-hop reconstruction. Memory access should be
   *adaptive* — cheap when the answer is obvious, deep only when it is not.
3. **Self-organization.** A static graph is a snapshot. Real memory *consolidates*:
   frequently-traversed associations should shorten over time. RuVector already
   advertises a self-learning GNN that "pushes similarities back into the neighbor
   lists" — the harness should exploit it.

**Decision needed**: Evolve the MRAgent harness past the paper to optimize
calibration and adaptive cost, on a benchmark hard enough that the full genome is
load-bearing, while keeping the example deterministic, dependency-free, and
ADR-150-compliant.

---

## Decision

Extend the reference harness (`examples/mragent/`) with three new mechanisms — each
a tunable gene Darwin co-evolves — and harden the corpus so all twelve genes carry
signal. The frozen-model / evolved-harness split (ADR-269) is preserved.

### 1. Calibration: abstention + risk-adjusted utility

A new gene `abstainThreshold` lets the harness answer *"I don't know"* when the
top reconstructed evidence is below threshold. The fitness is no longer accuracy
but a decision-theoretic **risk score**:

```
answerable task:    correct → +1 | abstain → 0 | wrong → −1
unanswerable task:  abstain → +1 | any answer (hallucination) → −1
riskScore = (mean(utility) + 1) / 2   ∈ [0, 1]
```

The corpus now contains **unanswerable** tasks (no correct content exists). A
harness that hallucinates on them is punished; one that abstains is rewarded.

### 2. Adaptive depth

A new gene `haltConfidence` stops traversal once the best content score crosses a
threshold — ACT-style adaptive computation (structurally the same adaptive-depth
idea ADR-260 draws between RDT's ACT loop and the SWE-bench repair loop). Easy
queries halt at hop 1; multi-hop bridge queries run to full depth.

### 3. Self-reorganizing memory: consolidation / replay

`agent/consolidate.mjs` replays successful reconstructions and lays down direct
`Cue→shortcut→Content` edges. This mutates only graph **adjacency** (the store's
own learned index — exactly RuVector's self-learning GNN feature), never the frozen
embeddings or content. A query that needed a 3-hop traversal resolves in 1 hop after
consolidation.

### 4. A benchmark where every gene is load-bearing

`data/eval-set.json` holds **structured signal specs**; `agent/memory.mjs`
synthesizes node texts from them. A concept layer (`agent/concepts.mjs`) projects
synonyms onto shared **concept** dimensions, decoupling dense semantics from
lexical (sparse) overlap — so paraphrases are dense-close with zero shared tokens,
and rare identifiers are sparse-decisive but semantically generic. Six task
classes (semantic, lexical, hybrid, bridge, distractor, unanswerable) each stress
a specific gene.

---

## Mutation Surfaces (12 genes)

`baselineGenome()` in `agent/harness.mjs`. New genes vs ADR-269 in **bold**.

| Gene | Range | Stressed by | RuVector mapping |
|------|-------|-------------|------------------|
| cueK | 1–12 | retrieval breadth | `hybridSearch` top-k |
| efSearch | 16–256 | cost | HNSW search depth |
| hybridAlpha | 0–1 | semantic / lexical | RRF sparse↔dense weight |
| fusion | rrf·linear·dbsf | hybrid | fusion strategy |
| traversalDepth | 1–4 | bridge | Cypher `LINKED_TO*1..N` |
| tagFanout | 1–8 | distractor (corroboration) | tags expanded per node |
| pruneThreshold | 0–0.6 | noise | path-evidence floor |
| maxContent | 1–20 | distractor | content `LIMIT` |
| **haltConfidence** | 0.2–0.9 | adaptive cost | early-stop traversal |
| rerank | gnn·none | distractor | corroboration rerank |
| promptStrategy | terse·evidence-first·prune-explicit | distractor | synthesis prompt |
| **abstainThreshold** | 0–0.6 | unanswerable | calibration / abstention |

### Epistatic interaction (why behavioral diversity matters)

Distractor tasks have **two** disjoint winning basins, confirmed in tests:

```
rerank=none  prompt=terse           → 0/3   (ranking-distractors win)
rerank=gnn   prompt=terse  fanout=1 → 0/3   (no corroborating path reached)
rerank=gnn   prompt=terse  fanout≥2 → 3/3   (corroboration boost rescues)
rerank=none  prompt=evidence-first  → 3/3   (full window finds the answer)
```

This deceptive, multi-basin landscape is exactly the case ADR-260 cites where
greedy score-selection fails and **behavioral-diversity** selection (RuVector ANN
archive) succeeds — motivating the real `@metaharness/darwin` write-layer.

---

## Optimizer: memetic (GA + coordinate descent)

The LLM-free fallback loop is a genetic search (`mapLimit` + `paretoFront`) over
risk-adjusted fitness, followed by **deterministic coordinate-descent polish** over
a per-gene candidate grid. The polish is what reliably finds narrow optima the
blind GA misses — notably the `abstainThreshold ∈ [0.34, 0.38]` band that catches
every hallucination without abstaining on a single correct answer. This makes the
shipped result reproducible. The real Darwin write-layer would propose such leaps
directly from failure traces; the polish is its deterministic stand-in.

```
fitness = 0.40·accuracy + 0.30·riskScore
        + 0.12·latencyTerm + 0.10·contextTerm + 0.08·hopTerm
```

---

## Measured Results (deterministic, zero optional deps)

In-sample, full 60-task corpus (baseline vs a fixed reference evolved genome;
`npm run optimize` reaches higher — see Generalization below):

```
config            accuracy  risk    halluc  latency  hops
baseline           50.0%    0.417   0.17    2.81     1.23
evolved (ref)      70.0%    0.775   0.03    3.09     1.08
evolved+replay     70.0%    0.775   0.03    3.16     1.00
```

- **Calibration** risk 0.417 → 0.775; **hallucination 0.17 → 0.03**.
- **Consolidation** lays graph shortcuts → fewer hops at equal accuracy.
- **`npm run optimize`** (full GA + memetic polish) reaches **+33pt train accuracy
  / risk 0.94**, and **generalizes to a held-out test split** (next section).
- **No dead genes remain** — every gene is load-bearing, several via epistatic
  interaction (above). All proven in `test/harness.test.mjs` (12 gates).

---

## Generalization: train / test / cross-validation (added)

The first cut of this ADR reported 100% accuracy by evolving and scoring on the
**same** corpus — which cannot distinguish a genome that *solves* the task from one
that *memorizes the eval set*. We added a proper generalization protocol, and it
immediately earned its keep by catching a real overfit.

**Protocol.** One memory holds all nodes (full cross-task cue competition). The
corpus is scaled to **60 tasks** (10 per class) via a deterministic generator
(`tools/genCorpus.mjs`) with **varied difficulty** (1-hop *and* 2-hop bridges, 1–3
ranking-distractors). The optimizer evolves on a class-stratified **train** split,
selects via **3-fold cross-validation with a variance penalty** (`mean − ½·range`
across folds, so a knife-edge gene that wins one fold and collapses on another is
rejected), and reports a **held-out test** split it never saw.

**What overfitting looked like (and the fixes).**

1. *Confidence depressed by depth.* Abstention used the decayed path score, so a
   deep-but-relevant 2-hop-bridge answer (confidence ~0.39) looked as weak as an
   unanswerable one (~0.33). A threshold tuned on shallow train bridges (~0.58)
   then wrongly abstained on deep test bridges. **Fix:** derive abstention
   confidence from the answer's **raw query relevance**, independent of traversal
   depth (decay still governs pruning/ranking). Deep bridges now read ~0.79 and
   abstention generalizes across depths.

2. *Under-constrained genes shaved for cost.* With a tiny train set the cost terms
   rewarded `maxContent→1`, `cueK→1` etc. — train-fine, test-fragile. **Fix:** a
   larger, **difficulty-varied** corpus (2-hop bridges and multi-distractor tasks
   in train forbid `depth<3` / `maxContent=1`) plus the CV variance penalty.

3. *Single-split luck.* **Fix:** cross-validated selection + a generalization gate
   that asks whether evolution *improves the unseen split*, not whether it hits an
   absolute bar.

**Result (held-out test, reproducible across runs):**

```
                 accuracy   risk    halluc
baseline (test)   ~30%      ~0.25    0.17
evolved  (test)   ~65%      ~0.81    0.04    →  +35pt acc, +0.56 risk
```

The evolved harness **transfers to unseen tasks**. It does not hit 100% on test —
the synthetic concept embedding has per-instance noise, and a single global
`hybridAlpha` provably cannot serve both dense- and sparse-keyed queries (ceiling
~80%). The honest claim is *generalization*, not a saturated score; chasing 100%
on the held-out split would be teaching to the test.

## The 25-year view (what this prototype is a seed of)

Concrete, implemented-here primitives → where they point:

| Implemented now | 25-year trajectory |
|-----------------|--------------------|
| `abstainThreshold` + risk utility | Memory systems graded on calibrated utility, not accuracy; abstention is a first-class action, not a failure. |
| `haltConfidence` adaptive depth | Per-query compute budgeting; reconstruction depth set by uncertainty, co-scheduled with model inference depth (RDT/ACT). |
| consolidation / replay shortcuts | Memory that continuously rewrites its own topology from workload — sleep/replay consolidation as a standing background process, not a batch job. |
| concept ≠ token embedding | Retrieval that reasons over meaning and surface form independently and fuses them per-query. |
| Darwin co-evolution of the harness | The retrieval *policy itself* is an evolved, versioned, witness-signed artifact that travels with the memory store. |

None of these require a different *model*. They are harness and topology — which is
why "freeze the model, evolve the harness" remains the right frame at this horizon.

---

## ADR-150 Compliance

Unchanged from ADR-269 and re-verified: `@metaharness/darwin` and `ruvector` are
`optionalDependencies` only; every import is `try/catch` guarded; `npm test` (11
gates), `npm run benchmark`, and `npm run optimize` all pass with no optional deps
installed (the CI gate). The memetic polish and consolidation run in the built-in
loop; the real write-layer is a drop-in upgrade.

---

## Consequences

- The example is now a genuine optimization *benchmark* (no dead genes, a deceptive
  multi-basin landscape, a calibration objective), not a toy that saturates.
- Risk-adjusted fitness changes what "best" means: the accepted harness is the one
  that is helpful **and** honest, which is the property that matters at scale.
- **Costs**: the substrate remains a faithful *simulation* of RuVector semantics —
  evolved genomes transfer, absolute latencies do not. The synthesis judge is
  deterministic, so prompt-strategy genes exercise the *shape* of synthesis, not a
  real model's nuance. Validating against a live `.rvf` index (real ONNX
  embeddings, HNSW recall nondeterminism re-activating `efSearch` as an accuracy
  lever, real Cypher) is the next step.

---

## Alternatives Considered

**Keep raw accuracy as the objective.** Rejected — it rewards guessing and makes
abstention strictly dominated, the opposite of what a lifetime-memory agent needs.

**Hand-author English corpus tasks.** Rejected — concept/lexical separation and
ranking-distractors are too fragile to hit by wording. Synthesizing node texts from
structured signal specs guarantees the difficulty and keeps it deterministic.

**Pure GA, no polish.** Rejected — the blind fallback reliably misses the narrow
`abstainThreshold` basin (it converged at risk 0.875 / 0.13 hallucination over 12
generations). Memetic local search finds it deterministically; the real write-layer
finds it from traces.

---

## References

- ADR-269 — MRAgent Graph Memory over RuVector (the baseline this extends)
- ADR-260 — Darwin Mode as Evolutionary Substrate (behavioral diversity, ACT depth)
- ADR-266 — MetaHarness Darwin Integration (scoring policy shape)
- Reference implementation — `examples/mragent/`
- `@metaharness/darwin` — https://github.com/ruvnet/agent-harness-generator
