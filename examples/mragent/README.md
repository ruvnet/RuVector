# MRAgent — Graph Memory over RuVector, optimized by Darwin Mode

A runnable reference implementation of **MRAgent** ("Memory is Reconstructed, Not
Retrieved: Graph Memory for LLM Agents") on top of **RuVector**, with a
**Meta-Harness Darwin** loop that *evolves the reconstruction harness* while the
memory substrate stays frozen.

> **Principle:** freeze the model, evolve the harness.
> **Frozen model:** the RuVector Cue-Tag-Content memory graph (`agent/memory.mjs`).
> **Evolved harness:** the reconstruction genome in `agent/harness.mjs`.

See **[ADR-269](../../docs/adr/ADR-269-mragent-graph-memory-darwin-optimization.md)**
for the full design rationale, mutation surfaces, scoring policy, and ADR-150
compliance.

## Why graph memory + Darwin

Standard RAG agents do a single dense search ("retrieve-then-reason"). MRAgent
instead represents memory as a **Cue → Tag → Content** associative graph and
*reconstructs* an answer by:

1. **Hybrid search** for entry **Cues** (sparse + dense, RRF fused).
2. **Active reconstruction** — traverse `LINKED_TO*1..N` from cues to **Tags** to
   **Content**, pruning low-evidence paths along the way.
3. **Synthesis** — hand the surviving content to the LLM with a prompt that
   prunes irrelevant branches.

Every one of those steps has tunable parameters. Hand-tuning them across a
benchmark is a combinatorial search, so we let **Darwin Mode** evolve them.

## The reconstruction genome (what Darwin mutates)

| Gene | Range | RuVector mapping |
|------|-------|------------------|
| `cueK` | 1–12 | # cue vectors from `hybridSearch` |
| `efSearch` | 16–256 | HNSW search depth / candidate pool |
| `hybridAlpha` | 0–1 | RRF sparse↔dense weight |
| `fusion` | rrf · linear · dbsf | hybrid fusion strategy |
| `traversalDepth` | 1–4 | Cypher `LINKED_TO*1..N` hops |
| `tagFanout` | 1–8 | tags expanded per frontier node |
| `pruneThreshold` | 0–0.6 | evidence floor to keep a path |
| `maxContent` | 1–20 | content `LIMIT` to synthesis |
| `rerank` | gnn · none | self-learning GNN rerank toggle |
| `promptStrategy` | terse · evidence-first · prune-explicit | synthesis prompt |

## Run it

```bash
cd examples/mragent

npm test            # deterministic acceptance gates (7 tests, no deps)
npm run benchmark   # baseline vs evolved harness over the corpus
npm run optimize    # Darwin evolution loop -> optimize.report.json
npm run probe       # inspect @metaharness/darwin exports (optional)
```

Nothing above requires network access, an API key, or native bindings — the
memory substrate is a deterministic in-process graph with the **same semantics**
as a live RuVector `.rvf` index (hybrid RRF search + bounded-depth Cypher
traversal). The evolved genome transfers to production unchanged.

### With the real Darwin write-layer (optional)

```bash
npm i -D @metaharness/darwin@latest   # adds the LLM/GA mutation + Pareto layer
npx metaharness evolve . \
  --generations 8 --children 3 --concurrency 3 \
  --eval-cmd "node benchmark.mjs"
```

`harness/scorePolicy.ts` is the fitness function `metaharness evolve` calls after
each mutation — it evaluates the current genome over the frozen corpus and
returns a score in `[0, 1]`.

## What the loop discovers

Out of the box the baseline genome (`traversalDepth: 2`) answers **83.3%** of the
corpus — it cannot reach the two-hop "bridge" tasks whose relevant Tag sits
behind an intermediate hop. A representative Darwin run:

```
baseline:  acc  83.3%  lat 2.52ms  ctx 1.7
evolved:   acc 100.0%  lat 1.59ms  ctx 1.3
           accuracy +16.7pt · latency ~58% faster · context ~33% smaller
```

Darwin reliably finds:
- **`traversalDepth: 3`** — reaches content behind bridge Tags (the
  variable-length-path insight, `MATCH (c)-[:LINKED_TO*1..3]->(m)`).
- **tighter `pruneThreshold` + smaller `maxContent`** — fewer distractor paths
  reach synthesis, so latency and context shrink at no accuracy cost.

## ADR-150 compliance (Meta-Harness is removable)

- `@metaharness/darwin` and `ruvector` are **optionalDependencies** only.
- `optimize.mjs` catches `MODULE_NOT_FOUND` and falls back to a built-in
  evolution loop with the same `mapLimit`/`paretoFront` contracts.
- `npm test`, `npm run benchmark`, and `npm run optimize` all pass with **no
  optional dependencies installed** (this is the CI gate).

## Layout

```
examples/mragent/
├── agent/
│   ├── memory.mjs        # FROZEN: Cue-Tag-Content store (RuVector semantics)
│   └── harness.mjs       # EVOLVED: reconstruction genome + reasoning loop
├── harness/scorePolicy.ts# Darwin fitness function (ADR-269 scoring)
├── data/eval-set.json    # Cue-Tag-Content corpus + multi-hop eval tasks
├── optimize.mjs          # Darwin evolution loop (graceful fallback)
├── benchmark.mjs         # baseline vs evolved comparison
├── probeDarwin.mjs       # probe optional @metaharness/darwin exports
└── test/harness.test.mjs # acceptance gates
```
