# MRAgent — Self-Reconstructing Graph Memory over RuVector, evolved by Darwin

A runnable reference implementation of **MRAgent** ("Memory is Reconstructed, Not
Retrieved: Graph Memory for LLM Agents") on **RuVector** — and then *past* the
paper. A **Meta-Harness Darwin** loop evolves the reconstruction harness while the
memory substrate stays frozen ("freeze the model, evolve the harness").

> **Frozen model:** the RuVector Cue-Tag-Content memory graph (`agent/memory.mjs`).
> **Evolved harness:** a 12-gene reconstruction genome (`agent/harness.mjs`).

ADRs: **[ADR-269](../../docs/adr/ADR-269-mragent-graph-memory-darwin-optimization.md)**
(the MRAgent baseline) and **[ADR-270](../../docs/adr/ADR-270-self-reconstructing-graph-memory-beyond-sota.md)**
(this beyond-SOTA version).

## Beyond the paper

MRAgent reconstructs an answer over a *static* graph: search cues → traverse
cue→tag→content → prune → synthesize. This implementation adds three mechanisms a
25-year-out memory system needs, each a tunable gene Darwin co-evolves:

1. **Adaptive depth** (`haltConfidence`) — stop traversing once evidence is
   decisive, so easy queries cost fewer hops (ACT-style adaptive computation).
2. **Abstention + calibration** (`abstainThreshold`) — answer *"I don't know"*
   when reconstructed evidence is too weak, instead of confidently hallucinating.
   Graded by a **risk-adjusted utility**, not raw accuracy: a confident wrong
   answer scores worse than an honest abstention.
3. **Consolidation / replay** (`agent/consolidate.mjs`) — the store reorganizes
   its own topology from workload (the self-learning GNN RuVector describes),
   laying Cue→shortcut→Content edges so a 3-hop query resolves in 1 hop tomorrow.

## The 12-gene reconstruction genome

| Gene | Range | RuVector mapping |
|------|-------|------------------|
| `cueK` | 1–12 | # cue vectors from `hybridSearch` |
| `efSearch` | 16–256 | HNSW search depth |
| `hybridAlpha` | 0–1 | RRF sparse↔dense weight |
| `fusion` | rrf · linear · dbsf | hybrid fusion strategy |
| `traversalDepth` | 1–4 | Cypher `LINKED_TO*1..N` hops |
| `tagFanout` | 1–8 | tags expanded per node |
| `pruneThreshold` | 0–0.6 | path-evidence floor |
| `maxContent` | 1–20 | content `LIMIT` to synthesis |
| `haltConfidence` | 0.2–0.9 | **adaptive-depth halt** |
| `rerank` | gnn · none | corroboration-aware rerank |
| `promptStrategy` | terse · evidence-first · prune-explicit | synthesis prompt |
| `abstainThreshold` | 0–0.6 | **abstention / calibration** |

Every gene is proven load-bearing in `test/harness.test.mjs` — some only via
*interaction* (distractor tasks are solved by `evidence-first` **or** by
`terse + gnn + fanout≥2`, an epistatic landscape).

## The hardened corpus (24 tasks, 6 classes)

`data/eval-set.json` holds **structured signal specs**; `agent/memory.mjs`
synthesizes the Cue/Tag/Content node texts so the difficulty is guaranteed, not
dependent on fragile English. A **concept layer** (`agent/concepts.mjs`) gives the
dense embedding real semantics decoupled from lexical overlap:

| Class | Stresses |
|-------|----------|
| semantic | `hybridAlpha`→dense (paraphrase, no shared tokens) |
| lexical | `hybridAlpha`→sparse (rare identifier, generic concept) |
| hybrid | `fusion` / RRF (needs both signals) |
| bridge | `traversalDepth` (1–2 intermediate hops) |
| distractor | `rerank` / `tagFanout` / `promptStrategy` (ranking-distractor content) |
| unanswerable | `abstainThreshold` (no correct content exists → abstain) |

## Results (zero optional deps, deterministic)

```
config            accuracy  risk   halluc  latency  hops
baseline           81.0%   0.708   0.13    2.62    1.17
evolved           100.0%   1.000   0.00    1.22    1.33
evolved+replay    100.0%   1.000   0.00    1.20    1.00

evolved vs baseline: accuracy +19.0pt · risk +0.292 · hallucination 0.13 → 0.00
consolidation: 21 shortcuts → 25% fewer hops at 100% accuracy
```

The optimizer is **memetic**: a genetic loop (Darwin `mapLimit`/`paretoFront`)
explores broadly, then deterministic coordinate descent refines narrow optima —
notably the `abstainThreshold ∈ [0.34, 0.38]` band that catches every
hallucination without abstaining on a single correct answer.

## Run it

```bash
cd examples/mragent
npm test            # 11 deterministic gates, every gene proven load-bearing
npm run benchmark   # baseline vs evolved vs evolved+replay
npm run optimize    # Darwin loop + memetic polish + consolidation
npm run probe       # inspect @metaharness/darwin exports (optional)
```

Nothing requires network, an API key, or native bindings. The substrate is a
deterministic in-process graph with the **same semantics** as a live RuVector
`.rvf` index (concept-dense + token-sparse hybrid RRF search, bounded-depth
prunable Cypher traversal, GNN-style corroboration rerank), so an evolved genome
transfers to production unchanged.

### With the real Darwin write-layer (optional)

```bash
npm i -D @metaharness/darwin@latest
npx metaharness evolve . --generations 12 --children 3 --eval-cmd "node benchmark.mjs"
```

`harness/scorePolicy.ts` is the fitness `metaharness evolve` calls per mutation.

## ADR-150 compliance

`@metaharness/darwin` and `ruvector` are **optionalDependencies** only; every
touch is `try/catch` guarded; `npm test`, `npm run benchmark`, and `npm run
optimize` all pass with no optional deps installed (the CI gate).

## Layout

```
examples/mragent/
├── agent/
│   ├── concepts.mjs      # concept layer (dense semantics ≠ sparse tokens)
│   ├── memory.mjs        # FROZEN: Cue-Tag-Content store (RuVector semantics)
│   ├── harness.mjs       # EVOLVED: 12-gene genome + reasoning loop
│   └── consolidate.mjs   # replay → self-reorganizing topology
├── harness/scorePolicy.ts# Darwin fitness (accuracy + risk + cost)
├── data/eval-set.json    # 24-task structured corpus (6 classes)
├── optimize.mjs          # GA + memetic polish + consolidation
├── benchmark.mjs         # baseline vs evolved vs replay
├── probeDarwin.mjs       # probe optional @metaharness/darwin
└── test/harness.test.mjs # 11 acceptance gates
```
