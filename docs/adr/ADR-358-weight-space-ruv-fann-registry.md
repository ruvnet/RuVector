# ADR-358: @ruvector/weight-space, a versioned ruv-fann weight registry indexed by RuVector

- **Status**: Accepted
- **Date**: 2026-11-10
- **Deciders**: ruv (owner), RuVector Architecture Team
- **Related**: ADR-356 (distribution and SEO standard), ADR-355 (weaver ledger); PR #1480 (`neural-q`, the first ruv-fann consumer in `ruvector`)
- **Tags**: npm, ruv-fann, wasm, model-registry, transfer-learning, fine-tuning, optional-peer
- **Numbering**: ADR-357 is claimed by open PR #1470; 358 was free on `main` and in every open PR on 2026-11-10.

## Context

The owner asked for a new npm package that pairs `ruv-fann` with RuVector and is
integrated into `ruvector`. The request came with an AI-written description. Most
of that description does not match the real packages, so this ADR records what was
built against what was claimed.

`ruv-fann` 0.1.0 (ESM-only, WASM in `ruv-fann-wasm`, about 1.5 MB) exposes
`loadRuntime()` → `prepare(model)` (`predict`, `predictInto`, `dispose`) and
`train(model, {inputs, targets, rows, epochs, learningRate, optimizer})`, plus
`LIMITS`. Models are plain JSON (`ruv-fann.dense-layers/1`). One synchronous
`train()` call is capped at 1e8 work units, where work is
`rows * (32 * params + 8 * (in + out) + 1) * (epochs + 1) + 64 * params * epochs`.
A 384→16→384 network (12,688 parameters) therefore fits about 100 rows for one
epoch per call. Anything trained on 384-d embeddings has to be mini-batched
across calls, and each call rebuilds the network, so AdamW moments do not carry
over between calls.

Claims in the pasted description that were **not** implemented, because nothing
real backs them:

| Claim | Reality |
|---|---|
| "FannAdapter" dependency | Does not exist. |
| "MetaHarness middleware injected by ruv-fann" | ruv-fann has no such thing. |
| `@ruv-fann/napi` | Does not exist; the package is `ruv-fann` (WASM). |
| WebGPU, CPU-load-adaptive batching | Not present in ruv-fann 0.1.0. |
| "autograd engine", "<0.5 ms" / "sub-millisecond" guarantees | Not established. Only measured latencies are published (below). |
| "Weight-Space Programming breakthrough" | Framing. The honest version is a validated, versioned edit API. |

## Decision

Ship `@ruvector/weight-space` 0.1.0 (`npm/packages/weight-space`), TypeScript,
ESM and CJS builds, with:

1. **Registry** (`WeightRegistry`): `save(model, taskEmbedding, meta)`,
   `nearest(taskEmbedding, k, {compatibleWith})`, `get(id)`, `lineage(id)`,
   `childrenOf(id)`, `recordOutcome(id, score)`. Each snapshot carries topology,
   version, parent, task, sha256 of the canonical model JSON, and an outcome log.
   Ids are content-derived, so saving identical content twice is idempotent.
   State changes are events appended to a JSON-lines log, and opening replays it.
   Replay verifies every hash, id and parent link and rejects a tampered log
   (`EINTEGRITY`); a torn final line from a crash is skipped. Similarity ties
   (a child shares its parent's embedding) resolve to the newer version.
2. **Index**: `@ruvector/core` `VectorDb`, persisted next to the log, used when it
   is installed. A built-in exact flat store is used when it is absent. `ruvector`
   already depends on `@ruvector/core`, so inside `ruvector` the RuVector index is
   used without adding anything.

   Core is **detected at runtime, not declared as a peer.** The plan was an
   exact-pinned optional peer (`0.1.38`). A plain
   `npm i @ruvector/core@0.1.37 ruvector@0.3.24 @ruvector/weight-space`
   then failed with `ERESOLVE`: npm rejects a present optional peer that doesn't
   match. Core published 0.1.35 through 0.1.38 between 2026-11-06 and 2026-11-09,
   so an exact peer would break installs on almost every core release, and the
   no-caret rule for our own packages rules out a range. Instead, the adapter
   checks the `VectorDb` API shape (`insert`, `insertBatch`, `search`, `get`,
   `len`). With `store: 'auto'` it falls back to the flat store; with
   `store: 'ruvector-core'` it throws `ENOCORE`. `0.1.38` is the exact
   devDependency the tests run against.
3. **Editing**: `getLayer`, `setLayer`, `perturb`, `diff`. They are pure,
   validated against ruv-fann `LIMITS` and the row-major `[outputs, inputs]`
   layout, and can never change a layer's shape. Registry variants write a child
   snapshot; stored snapshots are frozen and never mutated.
4. **Outcome fine-tuning**: `learnFromOutcomes(id, trajectories)`. Successes train
   toward the produced output, and failures toward an explicit `correction`;
   failures without one are skipped and counted. A seeded holdout split decides:
   the child is saved only if holdout MSE drops by at least `minImprovement`
   (default 1%). Training goes through `trainWithinBudget`, which packs epochs
   per call when the data fits and mini-batches when it does not.
5. **Query reshaping** (`QueryReshaper`, residual bottleneck MLP, identity at
   init): built and evaluated. It failed its pre-registered gate (*Evaluation*), so
   it is **experimental and off**, available only from `./experimental`.

`ruv-fann` is an exact `0.1.0` dependency (0.1.1 and 0.1.2 were published the
same day; `ruvector` pins its optional peer to 0.1.0 and the two stay aligned).
The ESM-only import uses the runtime-built `import()` from `neural-q.ts`, so the
CJS build loads it too. `loadFann()` refuses a ruv-fann whose `LIMITS` differ
from the mirrored constants.

### Integration into `ruvector` 0.3.24

- `@ruvector/weight-space` is an exact-pinned optional peer. It is not a
  dependency, so a default `ruvector` install does not fetch it (ADR-356 §4).
- CLI: `ruvector weights save|nearest|diff|lineage|get`, and MCP:
  `weights_save|weights_nearest|weights_diff|weights_lineage`. All are lazily
  loaded, path-confined to the working directory (the MCP rule from #1306/#1307),
  and print `npm i @ruvector/weight-space@0.1.0` when the package is absent.

## Evaluation

The numbers below were measured on ruvultra (Linux x64, Node 22, ruv-fann
`wasm-simd`). The scripts and raw JSON are in `npm/packages/weight-space/eval/`
and `.../eval/results/`.

**Warm-start** (synthetic task family y = c·sin(a·x0 + b) + d·x1, [2,16,1] model,
40-task library, 20 new tasks × 3 seeds, target training MSE ≤ 0.01, cap 600 epochs):

| Start | Reached | Median epochs |
|---|---|---|
| random init | 44/60 | 410 |
| `nearest()` snapshot | 59/60 | 105 |
| random library snapshot (control) | 58/60 | 142.5 |

`nearest()` vs random init, paired: 55 wins, 1 loss, 4 ties. `nearest()` vs a random
snapshot: 37 wins, 14 losses, 9 ties. **Shipped.**

**Query reshaping** (real data). The corpus is 325 ADR titles, and the queries are
919 body sentences that never contain their own title. Embeddings are MiniLM-L6-v2.
Splits are 60/20/20 over 3 seeds, with 24 configurations tuned on val. The gate was
pre-registered: pooled test recall@5 must improve with a 95% bootstrap CI that
excludes 0.

| Protocol | recall@5 base → reshaped | Δ (95% CI) |
|---|---|---|
| held-out queries | 0.480 → 0.495 | +0.014 [-0.009, +0.038] |
| held-out documents | 0.492 → 0.479 | -0.013 [-0.031, +0.005] |

**Fails the gate → not shipped as a feature.** It is reachable only through the
unsupported `@ruvector/weight-space/experimental` subpath, and neither the main API
nor `ruvector` uses it. Two other metrics improved on held-out queries: recall@1 by
+0.029 [+0.005, +0.053] and recall@10 by +0.027 [+0.002, +0.054]. Neither was the
pre-registered metric. Most seeds chose the grid's largest learning rate, so the grid
edge was not explored.

**Latency, p50.**

- **`registry.nearest`, k=5:** 1.87 ms over 10k × 384-d snapshots (flat).
- **`WeightRegistry.open` + `nearest`** (log replay with verification): 202 ms at 1k snapshots and 2.17 s at 10k. That is about 0.2 ms per snapshot, mostly JSON parsing, and it is the per-call cost of the `ruvector weights` CLI. Reusing the persisted core index rather than rebuilding it saves 548 → 501 ms at 3k.
- **Index search:**
  - `@ruvector/core` exact: 4.37 ms, against 1.88 ms for the built-in flat store.
  - HNSW at m=16/ef=64: 0.123 ms. Its top-5 matched exact search 100% of the time on 1,244 real MiniLM vectors, but only 43.3% on uniform random vectors, rising to 92.4% at m=32/ef=200. That is why HNSW is opt-in.
- **`predictInto`, reused prepared model:** 0.0136 ms for [64,128,64], down from 2.17 ms when preparing per call.
- **`trainWithinBudget`:** 31.9 ms for [16,32,4] over 256 rows and 50 epochs, down from 66.0 ms with one call per epoch.

**Install size.**

- **`@ruvector/weight-space`:** 31.7 kB packed and 175 kB unpacked, plus about 1.5 MB for `ruv-fann` + `ruv-fann-wasm`.
- **`ruvector` 0.3.24 vs 0.3.23** (`npm pack --dry-run`):
  - packed: +6,498 B (+0.156%)
  - unpacked: +22,043 B (+0.162%)
  - a default install fetches no new packages.

## Consequences

- Agents get a safe, versioned, auditable way to read, edit and retrain small
  networks, and to warm-start from the most similar past task.
- `ruvector`'s default install is unchanged in content; only the CLI/MCP glue was
  added (measured below).
- ruv-fann's per-call work budget makes training 384-d models slow (many calls,
  each with a JSON round-trip). This is a ruv-fann limitation, recorded rather than
  worked around.
