# ADR-266: MetaHarness Integration for Autonomous ANN Optimization (Darwin Mode)

## Status

Accepted

## Date

2026-06-21

## Authors

Claude Code MetaHarness Architect

## Supersedes

None

## Related

- **ADR-150** — MetaHarness Integration Surfaces (the optional-dependency invariant this ADR obeys)
- **ADR-260** — Darwin Mode as Evolutionary Substrate for MetaHarness (defines the `evolve → score → archive` loop and the `RuvvectorArchive` pattern this ADR mutates)
- **ADR-265** — Benchmark Suite (supplies the `score()` function components consumed here)
- **ADR-267** — SOTA Validation (consumes the evolved configs this ADR produces)

---

## Context

RuVector ships a large surface of ANN tuning knobs — HNSW graph degree (`M`),
construction effort (`efConstruction`), product-quantization bitwidth, RaBitQ
compression strategy, the MLA/SSM hybrid layer ratio, ColBERT token-clustering
`K`, KV-cache eviction policy, and DiskANN robust-pruning `alpha`. Today these are
hand-tuned per workload. The interactions between them are non-linear and
**workload-dependent**: a config that maximizes recall@10 on a 1M-vector OpenAI
embedding set can collapse QPS on a 100M-vector SIFT set. Manual sweeps do not
scale across that surface, and the local optima they find are fragile.

We want **autonomous parameter optimization**: an evolution layer that mutates
index hyperparameters, scores each candidate against a fixed multi-objective
function (recall@10, QPS, memory, p99 latency), and checkpoints the best config
per workload — with zero human in the loop after the baseline is captured.

MetaHarness's Darwin Mode (`@metaharness/darwin`) already implements the evolution
algorithm we need (a genetic + simulated-annealing hybrid; see ADR-260). The
remaining work is **not** to reimplement evolution — it is to define a clean
**integration surface**: what Darwin is allowed to mutate, and how a candidate is
scored.

### Constraints inherited from ADR-150

> [!IMPORTANT]
> **ADR-150 invariant — MetaHarness is OPTIONAL.**
> `@metaharness/darwin` MUST appear only under `optionalDependencies`, never
> under `dependencies`. RuVector's core index path MUST build, test, and run
> with the package absent. Darwin Mode is an *augmentation layer*, never a
> required runtime dependency. A `MODULE_NOT_FOUND` for `@metaharness/darwin`
> is a **gracefully-degraded no-op**, not an error.

This is the same pattern ADR-260 established for `RuvvectorArchive`
(`try { require('@ruvector/ruvector') } catch { /* fall back */ }`,
ADR-260 lines 142–143). Darwin integration follows it exactly.

### Baseline use cases

1. **Per-workload tuning** — evolve a config for a specific corpus + query
   distribution, checkpoint it, ship it as that workload's default.
2. **Regression guard** — when ADR-265's benchmark suite detects a recall/QPS
   regression after a kernel change, re-evolve to recover the lost ground.
3. **SOTA push (ADR-267)** — evolve aggressive configs that trade memory or
   build time for recall to beat published baselines on standard datasets.

---

## Decision

Integrate MetaHarness Darwin Mode as an **optional evolution layer** over
RuVector's index configuration. The integration defines two surfaces and nothing
else:

1. A **mutation surface** — the set of index hyperparameters Darwin may mutate,
   each with a type, a legal range, and semantics (table below).
2. A **scoring function** — a composition over the ADR-265 score components
   (`scorePolicy.ts`), producing a single scalar per candidate.

> [!NOTE]
> **This ADR documents the INTEGRATION SURFACE only.**
> `@metaharness/darwin` owns the evolution algorithm (genetic + simulated
> annealing). RuVector owns (a) the genome schema — what gets mutated — and
> (b) the scoring composition. We do not implement mutation operators,
> selection, crossover, or annealing here.

The evolution loop is run out-of-band (CLI / CI), never on the hot query path.
Evolved configs are persisted as plain JSON and loaded by the index like any
hand-written config — so a workload tuned by Darwin has **no runtime dependency**
on Darwin (re-affirming the ADR-150 invariant: the package can be uninstalled
after evolution and the checkpointed config still loads).

---

## Mutation Surfaces

What Darwin may mutate. Each surface maps to one tunable field in the index
config genome. Ranges are inclusive; mutation operators clamp to range.

| Surface | Module | Type | Range | Semantics |
|---|---|---|---|---|
| `hnsw_M` | HNSW | int | `[4, 32]` | max out-degree per node (graph connectivity) |
| `hnsw_efConstruction` | HNSW | int | `[50, 400]` | candidate-list size during build (construction cost vs graph quality) |
| `pq_bits` | PQ-Search | int | `[4, 8]` | quantization bitwidth per subvector |
| `quant_strategy` | RaBitQ | enum | `[uniform, asymmetric, logarithmic]` | scalar-compression scheme |
| `layer_ratio` | MLA/SSM hybrid | float | `[0.2, 0.8]` | fraction of attention vs SSM in the hybrid stack |
| `colbert_k` | Multi-Vector | int | `[4, 16]` | token-clustering K for late-interaction retrieval |
| `cache_eviction` | KV-Cache | enum | `[H2O, PyramidKV, SlidingWindow]` | eviction policy under cache pressure |
| `diskann_alpha` | DiskANN | float | `[1.0, 1.5]` | robust-pruning strength (graph diversity vs density) |

> [!WARNING]
> **The mutation surface is a closed allowlist.** Darwin MUST NOT mutate any
> field outside this table. Fields that affect correctness rather than the
> recall/speed/memory tradeoff (distance metric, vector dimension, ID space)
> are deliberately excluded — mutating them would change *what* is being
> searched, not *how well*. The genome schema is the enforcement point: any
> field not declared mutable is frozen.

### Genome schema (the enforcement point)

The genome is a flat JSON object with exactly the 8 keys above. The integration
exposes it via a single declaration; Darwin reads this to know its search space.

```json
{
  "genome": {
    "hnsw_M":              { "type": "int",   "min": 4,   "max": 32 },
    "hnsw_efConstruction": { "type": "int",   "min": 50,  "max": 400 },
    "pq_bits":             { "type": "int",   "min": 4,   "max": 8 },
    "quant_strategy":      { "type": "enum",  "values": ["uniform", "asymmetric", "logarithmic"] },
    "layer_ratio":         { "type": "float", "min": 0.2, "max": 0.8 },
    "colbert_k":           { "type": "int",   "min": 4,   "max": 16 },
    "cache_eviction":      { "type": "enum",  "values": ["H2O", "PyramidKV", "SlidingWindow"] },
    "diskann_alpha":       { "type": "float", "min": 1.0, "max": 1.5 }
  }
}
```

A config field absent from `genome` is invisible to Darwin and therefore
immutable by construction — no runtime check needed.

---

## Scoring Function

A candidate config is scored by composing the four ADR-265 benchmark components
into a single scalar. The composition is declared in `scorePolicy.ts`:

```json
{
  "components": {
    "recall_weight":  0.4,
    "qps_weight":     0.3,
    "memory_weight":  0.2,
    "latency_weight": 0.1
  },
  "formula": "0.4*recall@10 + 0.3*log(QPS/baseline_QPS) + 0.2*(1-mem/baseline_mem) + 0.1*(1-p99_ms/baseline_p99_ms)"
}
```

Notes on the composition:

- **`recall@10`** is the dominant term (0.4) — a fast index that returns wrong
  neighbours is worthless. It enters linearly in `[0, 1]`.
- **`QPS`** enters as `log(QPS/baseline_QPS)` so a 2× speedup and a 4× speedup
  are not rewarded linearly — diminishing returns past the baseline, and the log
  is symmetric around regressions (`QPS < baseline` → negative term).
- **`memory`** and **`p99_latency`** are *relief* terms: `1 - ratio`, positive
  when the candidate uses less memory / lower tail latency than baseline,
  negative when worse.
- All four `baseline_*` values come from ADR-265's recorded baseline run for the
  same dataset, so scores are comparable only within a workload.

> [!IMPORTANT]
> **ADR-265 owns the measurements; this ADR owns the weights.** The
> `recall@10`, `QPS`, `mem`, and `p99_ms` numbers are produced by ADR-265's
> benchmark harness. `scorePolicy.ts` only *combines* them. If ADR-265 changes
> how a metric is measured, the weights do not change — but every prior score
> must be recomputed before comparison.

---

## Evolution Loop

A single generation:

```
1. seed     load baseline config (ADR-265 recorded run) as generation-0 genome
2. mutate   Darwin produces N child genomes by mutating surfaces (genetic +
            simulated annealing — @metaharness/darwin internal)
3. score    for each child: build index → run ADR-265 benchmark → scorePolicy.ts
4. rank     sort children by scalar score, descending
5. checkpoint  persist the top genome to configs/evolved/<workload>.json
6. (repeat over G generations; each generation seeds from the prior best)
```

The loop is deliberately **single-objective after composition** — the four
metrics collapse to one scalar at step 3, so ranking is total and the checkpoint
is unambiguous. Multi-objective Pareto fronts are out of scope (a future ADR
could add them by changing only `scorePolicy.ts`).

CLI surface (additive, gated on the package being present):

```bash
ruvector evolve <dataset> \
  --baseline configs/baseline/<workload>.json \
  --generations 5 --children 8 \
  --score-policy configs/scorePolicy.json \
  --out configs/evolved/<workload>.json
```

If `@metaharness/darwin` is not installed, `ruvector evolve` prints a one-line
"MetaHarness not installed — evolution unavailable" notice and exits 0 (it is an
optional capability, not a failed command).

---

## ADR-150 Compliance

How the optional invariant is enforced, line by line:

| Concern | Enforcement |
|---|---|
| Package classification | `@metaharness/darwin` listed under `optionalDependencies` in the CLI `package.json`, never `dependencies`. |
| Missing package | The `evolve` command resolves the module via `try { require('@metaharness/darwin') } catch { return gracefulNoop() }` — the same guard ADR-260 uses for `RuvvectorArchive` (ADR-260 §Component 2, lines 142–143). |
| Hot path isolation | Evolution runs only under the `evolve` subcommand (CLI/CI). No `import '@metaharness/darwin'` appears in the index/query modules. The query path cannot trigger a `MODULE_NOT_FOUND`. |
| Post-evolution independence | Evolved configs are plain JSON loaded by the standard config loader. After evolution, `@metaharness/darwin` can be uninstalled and every checkpointed config still loads — Darwin leaves no runtime artifact. |
| Frozen-field safety | The genome schema is the allowlist; fields absent from it are immutable by construction, so a buggy or adversarial mutator cannot reach correctness-affecting config. |

```typescript
// CLI evolve subcommand — ADR-150 graceful-degradation guard.
let Darwin: typeof import('@metaharness/darwin') | undefined;
try {
  Darwin = require('@metaharness/darwin');     // optionalDependency
} catch {
  console.log('MetaHarness not installed — evolution unavailable. ' +
              'Install with: npm i -O @metaharness/darwin');
  process.exit(0);                             // not an error — optional capability
}
```

### Why MetaHarness stays optional

RuVector is a vector index first. The overwhelming majority of consumers embed
the index and never evolve hyperparameters — they ship a hand-tuned or
Darwin-evolved-then-frozen config. Forcing every consumer to pull a genetic
optimizer (and its transitive deps) onto the install graph would be wrong.
Evolution is a *development-time / CI-time* activity that produces a static
artifact (the JSON config). The invariant keeps the runtime lean and the
dependency surface honest.

---

## Success Criteria

Darwin Mode integration is considered successful when:

- **Primary:** an evolved config beats the ADR-265 baseline on **at least 2 of
  the 4 metrics** (recall@10, QPS, memory, p99) on a standard dataset, with the
  composed score strictly higher than baseline.
- The full RuVector test suite passes with `@metaharness/darwin` **uninstalled**
  (proves the ADR-150 invariant).
- `ruvector evolve` exits 0 with a graceful notice when the package is absent.
- A checkpointed evolved config loads and serves queries after the package is
  uninstalled (proves post-evolution independence).
- Zero index/query-path module imports reference `@metaharness/darwin`
  (greppable check in CI).

---

## Consequences

### Positive

- Autonomous, reproducible per-workload tuning replaces manual sweeps.
- The closed mutation-surface allowlist makes the search space auditable and
  keeps correctness-affecting fields frozen.
- Evolved configs are static JSON — no runtime coupling to the optimizer.
- Composes cleanly with ADR-260 (Darwin is already wired for ruvector) and
  reuses ADR-265's measurement harness verbatim.

### Negative

- Single-scalar scoring hides Pareto tradeoffs; a config that is best-overall
  may be dominated on a metric a specific consumer cares about most.
- Scores are only comparable within a workload (baselines differ), so there is
  no single "best config" across datasets.
- Evolution cost is real (build + benchmark per child × children × generations);
  this is a CI/offline cost, acceptable because it is off the hot path.

### Neutral

- The weights in `scorePolicy.ts` are a policy choice, not a measured fact —
  changing them re-ranks history and requires recomputation.
- Adding a new tunable later means one row in the mutation-surface table plus
  one genome key; the loop and scoring are unaffected.

---

## Options Considered

### Option 1: Reimplement evolution inside RuVector
- **Pros:** no external dependency at all; full control.
- **Cons:** reinvents the genetic + simulated-annealing hybrid `@metaharness/darwin`
  already ships and ADR-260 already wired; large maintenance surface for a
  development-time tool.

### Option 2: MetaHarness Darwin as an optional integration surface (chosen)
- **Pros:** reuses the upstream evolution algorithm; obeys ADR-150; static-config
  output keeps the runtime lean; small, auditable surface (genome + score).
- **Cons:** depends on an external package's API stability for the *evolve*
  workflow (mitigated by the graceful no-op when absent).

### Option 3: Manual grid/random search in CI
- **Pros:** zero dependencies; trivial to reason about.
- **Cons:** does not scale across the 8-dimension surface; finds fragile local
  optima; no behavioural-diversity selection (ADR-260 §3 showed greedy search
  fails on deceptive landscapes 0/5 vs diversity 5/5).

---

## References

- [darwin-mode ADR-074](https://github.com/ruvnet/agent-harness-generator/blob/main/docs/adrs/ADR-074-darwin-ruvector-memory-ruflo-fabric.md) — ruvvector archive design (upstream)
- ADR-260 §Component 2 — `RuvvectorArchive` graceful-degradation pattern (the canonical optional-dependency guard)
