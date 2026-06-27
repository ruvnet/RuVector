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

## The hardened corpus (60 tasks, 6 classes, difficulty-varied)

`data/eval-set.json` is **generated** by `tools/genCorpus.mjs` (`npm run
gen-corpus`) as **structured signal specs**; `agent/memory.mjs` synthesizes the
Cue/Tag/Content node texts so difficulty is guaranteed, not dependent on fragile
English. A **concept layer** (`agent/concepts.mjs`) gives the dense embedding real
semantics decoupled from lexical overlap. 10 instances per class, with varied
difficulty (1-hop AND 2-hop bridges, 1–3 ranking-distractors) so a train/test
split constrains every gene:

| Class | Stresses |
|-------|----------|
| semantic | `hybridAlpha`→dense (paraphrase, no shared tokens) |
| lexical | `hybridAlpha`→sparse (rare identifier, generic concept) |
| hybrid | `fusion` / RRF (needs both signals) |
| bridge | `traversalDepth` (1–2 intermediate hops) |
| distractor | `rerank` / `tagFanout` / `promptStrategy` (ranking-distractor content) |
| unanswerable | `abstainThreshold` (no correct content exists → abstain) |

## Generalization, not overfitting (train / test / CV)

The optimizer **evolves on a train split and reports a held-out test split it
never saw** — proving the genome generalizes rather than memorizing the eval set.
Selection uses **3-fold cross-validation with a variance penalty** (mean − ½·range
across folds) so a knife-edge gene that wins one fold but collapses on another is
rejected. A subtle bug this surfaced — confidence was depressed by `decay^depth`,
making deep-but-relevant answers look weak and breaking abstention across depths —
is fixed by deriving **abstention confidence from the answer's raw relevance, not
its decayed path score** (`agent/memory.mjs`).

```
                 accuracy   risk    halluc
baseline (test)   ~30%      ~0.25    0.17
evolved  (test)   ~65%      ~0.81    0.04      ← held out, never seen in evolution
                  +35pt    +0.56  generalizes
```

(The synthetic toy embedding has per-instance noise, and one global `hybridAlpha`
cannot perfectly serve both dense- and sparse-keyed queries, so the test ceiling
is ~80%, not 100% — the gate asks whether **evolution transfers**, which it does.)

## Results on the full corpus (zero optional deps, deterministic)

```
config            accuracy  risk   halluc  latency  hops
baseline           50.0%   0.417   0.17    2.81    1.23
evolved (ref)      70.0%   0.775   0.03    3.09    1.08
evolved+replay     70.0%   0.775   0.03    3.16    1.00

evolved vs baseline: accuracy +20.0pt · risk +0.358 · hallucination 0.17 → 0.03
consolidation: shortcuts → fewer hops at equal accuracy
```

`npm run optimize` (full GA + memetic polish) reaches **+33pt train accuracy /
risk 0.94** and writes the evolved genome to `optimize.report.json`, which
`npm run benchmark` then picks up. The optimizer is **memetic**: a genetic loop
(Darwin `mapLimit`/`paretoFront`) explores broadly, then deterministic
coordinate descent refines narrow optima (e.g. the abstention band).

## Run it

```bash
cd examples/mragent
npm test            # 12 deterministic gates, every gene proven load-bearing
npm run benchmark   # baseline vs evolved vs evolved+replay
npm run optimize    # Darwin loop + memetic polish + consolidation + held-out test
npm run gen-corpus  # regenerate data/eval-set.json (deterministic)
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
├── data/eval-set.json    # 60-task structured corpus (generated)
├── tools/genCorpus.mjs   # deterministic corpus generator
├── optimize.mjs          # GA + CV + memetic polish + held-out test + consolidation
├── benchmark.mjs         # baseline vs evolved vs replay
├── probeDarwin.mjs       # probe optional @metaharness/darwin
└── test/harness.test.mjs # 12 acceptance gates
```
