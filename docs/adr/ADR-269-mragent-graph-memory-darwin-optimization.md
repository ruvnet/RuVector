# ADR-269: MRAgent Graph Memory over RuVector, Optimized by Darwin Mode

**Status**: Accepted
**Date**: 2026-06-27
**Authors**: Claude Code MetaHarness Architect
**Supersedes**: None
**Related**: ADR-150 (MetaHarness Integration Surfaces), ADR-256 (MetaHarness SDK
Evaluation), ADR-260 (Darwin Mode as Evolutionary Substrate), ADR-266
(MetaHarness Darwin Integration for ANN), ADR-029 (RVF Canonical Format),
ADR-050 (Graph Transformer Bindings)

---

## Context

**MRAgent** — from *"Memory is Reconstructed, Not Retrieved: Graph Memory for LLM
Agents"* — argues that LLM-agent memory should not be a single flat vector lookup
("retrieve-then-reason"). Instead, memory is a **Cue → Tag → Content** associative
graph, and answering a question is an **active reconstruction**: search for entry
cues, iteratively traverse cue→tag→content associations, prune irrelevant paths as
evidence accumulates, and only then synthesize.

RuVector is an unusually good substrate for this because it already ships every
primitive the paper needs:

- **Cypher hypergraph traversal** — `MATCH (c:Cue)-[:LINKED_TO*1..N]->(t:Tag)-[:REFERENCES]->(m:Content)`
- **Hybrid retrieval (RRF)** — sparse + dense cue search in one call
- **HNSW** with `efSearch` recall control (O(log n))
- **Self-learning GNN rerank** — graph topology tuned from query workload
- **Instant graph/vector mutation** — reconstruction can rewrite associations live

The problem is **not** capability; it is **configuration**. The reconstruction
pipeline exposes ~10 interacting parameters (cue count, `efSearch`, RRF α, fusion
strategy, traversal depth, tag fan-out, prune threshold, content limit, GNN
rerank, prompt strategy). Hand-tuning these against a benchmark is an O(n^k)
search, and the optimum is dataset-dependent (multi-hop corpora reward deeper
traversal; noisy corpora reward aggressive pruning).

**Decision needed**: How do we (a) implement MRAgent on RuVector as a concrete,
testable harness, and (b) optimize its reconstruction parameters automatically
while respecting ADR-150 (MetaHarness must remain removable)?

---

## Decision

Ship a reference MRAgent harness under `examples/mragent/` and drive its
optimization with **Meta-Harness Darwin Mode**, applying the project's standing
invariant: **freeze the model, evolve the harness.**

- **Frozen model** = the RuVector Cue-Tag-Content memory substrate
  (`agent/memory.mjs`): the nodes, embeddings, edges, hybrid-search semantics and
  Cypher traversal semantics. Darwin **never** mutates this.
- **Evolved harness** = the *reconstruction genome* (`agent/harness.mjs`): the
  knobs that govern how memory is reconstructed. Darwin mutates only the
  `DARWIN_MUTABLE_BLOCK` regions.

This mirrors ADR-266 (Darwin evolves ANN index configs) and ADR-260 (Darwin as
evolutionary substrate), but targets the *agentic retrieval* layer rather than the
storage layer. The same `@metaharness/darwin` tooling and the same scorePolicy
shape (`crates/ruvector-sota-bench/harness/scorePolicy.ts`) are reused.

### Why a self-contained example, not a live RuVector binding

The reference harness reimplements the memory substrate **in-process and
deterministically**, with semantics identical to a live RuVector `.rvf` index
(hashed dense embeddings standing in for ONNX MiniLM, term-overlap sparse scores,
RRF fusion, bounded-depth prunable traversal). This makes the example:

- **runnable with zero native deps** (CI-friendly, no model download, no API key),
- **deterministic** (reproducible fitness — a hard requirement for evolution), and
- **transferable** — an evolved genome is just parameters; it drops into a
  production RuVector deployment unchanged.

`ruvector` is declared as an *optional* dependency: if present it can supply real
embeddings, but the example must never require it.

---

## Mutation Surfaces (10 genes)

Defined in `examples/mragent/agent/harness.mjs` → `baselineGenome()`. Each gene
maps to a real RuVector retrieval / Cypher-traversal parameter.

```json
{
  "stage1_hybrid_search": [
    {"gene": "cueK",        "type": "int",   "range": [1, 12],   "default": 5,    "maps_to": "hybridSearch top-k cues"},
    {"gene": "efSearch",    "type": "int",   "range": [16, 256], "default": 64,   "maps_to": "HNSW search depth"},
    {"gene": "hybridAlpha", "type": "float", "range": [0.0, 1.0],"default": 0.5,  "maps_to": "RRF sparse↔dense weight"},
    {"gene": "fusion",      "type": "enum",  "options": ["rrf", "linear", "dbsf"], "default": "rrf"}
  ],
  "stage2_reconstruction": [
    {"gene": "traversalDepth", "type": "int",  "range": [1, 4],     "default": 2,    "maps_to": "Cypher LINKED_TO*1..N"},
    {"gene": "tagFanout",      "type": "int",  "range": [1, 8],     "default": 4,    "maps_to": "tags expanded per node"},
    {"gene": "pruneThreshold", "type": "float","range": [0.0, 0.6], "default": 0.15, "maps_to": "path evidence floor"},
    {"gene": "maxContent",     "type": "int",  "range": [1, 20],    "default": 10,   "maps_to": "content LIMIT to synthesis"}
  ],
  "stage3_synthesis": [
    {"gene": "rerank",         "type": "enum", "options": ["gnn", "none"], "default": "gnn", "maps_to": "GNN rerank layer"},
    {"gene": "promptStrategy", "type": "enum", "options": ["terse", "evidence-first", "prune-explicit"], "default": "evidence-first"}
  ]
}
```

The interactions are the point: `traversalDepth` raises recall on multi-hop
("bridge") tasks but raises cost; `pruneThreshold` and `maxContent` cut cost but
can drop the answer; `efSearch` sets a recall ceiling that `cueK` cannot exceed.

---

## Scoring Policy

`examples/mragent/harness/scorePolicy.ts` (mirrors ADR-266's policy shape).
Accuracy dominates — a fast harness that answers wrong is worthless — while the
remaining weight rewards cheaper reconstruction (the MRAgent "prune irrelevant
paths" objective):

```
fitness = 0.70 × accuracy
        + 0.15 × (1 − avgLatencyMs / BASE_LATENCY).clamp(0,1)
        + 0.10 × (1 − avgContext   / BASE_CONTEXT ).clamp(0,1)
        + 0.05 × (1 − avgHops      / BASE_HOPS    ).clamp(0,1)
```

`scoreVariant()` imports the **current** (Darwin-mutated) harness, evaluates the
genome over the frozen corpus, and returns a value in `[0, 1]`. A mutation that
breaks the harness scores `0` (and is therefore selected out).

Selection itself is **multi-objective**: `optimize.mjs` keeps a Pareto frontier
over `[accuracy, −latency, −hops, −context]` (via `paretoFront`) so the run does
not collapse to a single scalar prematurely; the scalar fitness only ranks
within a generation and drives the acceptance gate.

---

## ADR-150 Compliance (Load-Bearing Invariants)

### Invariant 1 — Removable

The example runs fully without `@metaharness/darwin`. `optimize.mjs`:

```js
async function loadDarwin() {
  try {
    const d = await import("@metaharness/darwin");
    return { mapLimit: d.mapLimit, paretoFront: d.paretoFront, available: true };
  } catch (e) {
    if (e.code !== "ERR_MODULE_NOT_FOUND" && e.code !== "MODULE_NOT_FOUND") throw e;
    console.warn("[darwin] not installed — using built-in evolution loop");
    return { mapLimit: localMapLimit, paretoFront: localParetoFront, available: false };
  }
}
```

The built-in `localMapLimit` / `localParetoFront` honor the exact same contracts,
so evolution still runs — just without the LLM write-layer's smarter mutation
proposals.

### Invariant 2 — Optional in package.json

```json
{
  "optionalDependencies": { "@metaharness/darwin": "^0.3.1", "ruvector": "^2.1.0" },
  "peerDependencies":      { "@metaharness/darwin": "^0.3.1" },
  "peerDependenciesMeta":  { "@metaharness/darwin": { "optional": true } }
}
```

Never in `dependencies`.

### Invariant 3 — Graceful degradation

Every touch of an optional module is wrapped. `probeDarwin.mjs` and `memory.mjs`
(`require("ruvector")`) both `try/catch` `MODULE_NOT_FOUND` and continue.

### Invariant 4 — CI gate without MetaHarness

`npm test`, `npm run benchmark`, and `npm run optimize` all pass with no optional
dependencies installed. This is the daily gate.

---

## Evolution Loop

`examples/mragent/optimize.mjs`:

1. Seed a population from the baseline genome + random mutations.
2. Per generation: evaluate all genomes (`mapLimit`, bounded concurrency),
   compute the Pareto frontier, record the scalar winner.
3. Next generation = elites + mutated children of elites.
4. After `GENERATIONS`, apply the **acceptance gate** over the *entire archive*
   (not just the last generation): accept the highest-scoring variant that does
   not regress accuracy and improves accuracy ≥5pt **or** latency ≥20%.
5. Write `optimize.report.json` (baseline, evolved genome, metrics, history).

---

## CI/CD Workflow (weekly evolution)

```yaml
# .github/workflows/mragent-evolution.yml (sketch)
name: MRAgent Darwin Evolution
on:
  workflow_dispatch:
  schedule:
    - cron: "0 13 * * 3"   # Wednesday 13:00 UTC
jobs:
  evolve:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "22" }
      - name: CI gate — runs WITHOUT MetaHarness
        working-directory: examples/mragent
        run: |
          npm test
          npm run benchmark
          npm run optimize          # built-in fallback loop
      - name: Optional — real Darwin write-layer
        working-directory: examples/mragent
        run: |
          npm i -D @metaharness/darwin || echo "optional; skipping"
          node optimize.mjs
      - name: Archive report
        uses: actions/upload-artifact@v4
        with: { name: mragent-evolution, path: examples/mragent/optimize.report.json }
```

---

## Acceptance Test

```bash
cd examples/mragent
npm test            # 7 deterministic gates, no deps
npm run optimize    # built-in loop, no deps

# Pass criteria (observed on the shipped 12-task corpus):
# ✅ baseline accuracy 83.3% (depth=2 cannot reach 2-hop bridge tasks)
# ✅ evolved  accuracy 100%  (Darwin discovers traversalDepth=3)
# ✅ latency  ~37–58% faster at equal-or-better accuracy
# ✅ context  ~33% smaller (tighter prune threshold + maxContent)
# ✅ zero crashes with @metaharness/darwin absent (ADR-150)
```

The depth gradient is the load-bearing signal that the optimization is real, not
cosmetic: `traversalDepth` 1 → 58.3%, 2 → 83.3%, 3 → 100%. A test asserts that
`depth=1` provably misses a bridge task and `depth≥2` resolves it.

---

## Consequences

| Dimension | Flat RAG baseline | MRAgent + Darwin |
|-----------|-------------------|------------------|
| Memory model | single dense lookup | Cue-Tag-Content graph reconstruction |
| Multi-hop recall | misses (one hop only) | reaches via `LINKED_TO*1..N` |
| Parameter tuning | manual, O(n^k) | autonomous Pareto evolution |
| Reconstruction cost | fixed | minimized at equal accuracy |
| MetaHarness dependency | n/a | optional, removable (ADR-150) |
| Transfer to production | n/a | genome is parameters → drops into live RuVector |

**Costs / risks:**
- The reference substrate is a faithful *simulation* of RuVector semantics, not
  the native engine — genomes transfer, but absolute latency numbers do not.
  Validating evolved genomes against a live `.rvf` index is follow-up work.
- The deterministic synthesis judge is a proxy for an LLM; prompt-strategy genes
  exercise the *shape* of synthesis behavior, not a real model's nuance.

---

## Alternatives Considered

**Bind directly to the native `ruvector` NAPI module in the example.** Rejected
for the reference harness: it would make CI require native builds + a model
download and would make fitness non-deterministic (HNSW build nondeterminism),
breaking evolution reproducibility. The optional `ruvector` hook remains for users
who want real embeddings.

**Optimize with a plain grid/random search instead of Darwin.** Grid search does
not scale to 10 interacting genes; the built-in fallback loop *is* an evolutionary
search and is what runs when Darwin is absent. The Darwin write-layer adds
LLM-proposed mutations grounded in failure traces (per ADR-260), which a blind
search cannot.

**Score with LLM-as-judge only.** Rejected per ADR-266 — Darwin optimizes against
raw execution metrics (accuracy, latency, hops, context), not judge summaries, so
the fitness signal is concrete and gameable-resistant.

---

## References

- MRAgent — *Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents*
- ADR-150 — MetaHarness Integration Surfaces (removability invariants)
- ADR-256 — MetaHarness SDK Evaluation
- ADR-260 — Darwin Mode as Evolutionary Substrate for MetaHarness
- ADR-266 — MetaHarness Darwin Integration for Autonomous ANN Optimization
- Reference implementation — `examples/mragent/` (this ADR)
- `@metaharness/darwin` — https://github.com/ruvnet/agent-harness-generator
