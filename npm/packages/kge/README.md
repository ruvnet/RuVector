# @ruvector/kge

Holographic knowledge-graph embeddings for [ruvector](https://github.com/ruvnet/ruvector):
score a triple `(subject, relation, object)` for plausibility, then answer link
prediction, relation similarity, and 2-hop composition — locally, with no
network and no per-query cost. Native (napi-rs) with a WASM fallback.

## What it is

A trained pair of embedding tables (entities `E`, relations `R`) plus a scorer.
The default scorer is **HolE** (Holographic Embeddings, Nickel, Rosasco &
Poggio, AAAI 2016): it scores `r · (e_s ⋆ e_o)`, a relation vector dotted with
the *circular correlation* of the head and tail entity vectors.

**HolE ≡ ComplEx.** Hayashi & Shimbo (ACL 2017, arXiv:1702.05563) proved HolE is
algebraically the same model class as ComplEx (Trouillon 2016). That is the
load-bearing fact of this package: it lets us ship the real-valued
circular-correlation form — half the parameters of ComplEx's doubled
real/imaginary table — while inheriting ComplEx's accuracy *and* its
inner-product structure, which is what makes ANN candidate retrieval possible
(ADR-001, ADR-002).

The opt-in second scorer is **RotatE** (Sun et al., ICLR 2019,
arXiv:1902.10197), the one family member that represents relation
**composition** `r1 ∘ r2` — the pattern HolE/ComplEx cannot.

## Status (v1)

- **No platform packages yet.** The first release ships the WASM fallback and
  builds the native addon locally; the five `optionalDependencies` platform
  packages are added in a later bump PR (ADR-001 §5).
- **`predict`, `similarRelations`, `compose`, `train`, `eval` and `buildIndex`
  all work today.** `predict` is exhaustive until you `buildIndex`, then
  ANN-accelerated. Before `train`, the tables are the deterministic seed init,
  so scores are structurally valid but not yet meaningful.
- **`optimize` returns `{"error":{"kind":"unavailable"}}`** — the core
  self-optimization `Campaign` needs an `Evaluator` the binding does not provide
  yet (ADR-004).
- **Composition is RotatE-only**; asking a HolE model to `compose` returns
  `{"error":{"kind":"unsupported"}}`.
- **The ANN index and trained weights: growth-safe, index transient.** Adding
  triples grows the tables in place (trained rows are preserved) and drops the
  index. The index is not saved, so after `loadKge` you `buildIndex` again.

## Install

```sh
npm install @ruvector/kge
```

Building from source in this repo:

```sh
bash scripts/build-native.sh   # -> native/kge.<triple>.node
bash scripts/build-wasm.sh     # -> wasm/ (nodejs + bundler), with the ADR-005 import check
npm run build                  # tsc -> dist/
```

`KGE_BACKEND=wasm` forces the fallback; otherwise the native addon loads when
present, else WASM.

## Quick start (TypeScript)

```ts
import { createKge, defineSchema } from '@ruvector/kge';

// Optional: a schema narrows the relation argument to a string-literal union.
const schema = defineSchema({ relations: ['bornIn', 'locatedIn'] as const });

const kge = createKge({ scorer: 'hole', dims: 256, schema });
kge.addTriples([
  { s: 'Ada', r: 'bornIn', o: 'London' },
  { s: 'London', r: 'locatedIn', o: 'England' },
]);

// Link prediction — leave exactly one slot open:
const tails = kge.predict({ s: 'Ada', r: 'bornIn', k: 10 });
tails.candidates[0].entity;   // best-ranked object

// Relation similarity (cosine over relation vectors):
kge.similarRelations({ r: 'bornIn', k: 5 });

// Save / restore (the envelope carries a sha256; load fails closed on a tamper):
const saved = kge.save();
import { loadKge } from '@ruvector/kge';
const restored = loadKge(saved);
```

Errors are thrown as `KgeError` with a `.kind` in
`limit | invalid | unavailable | unsupported | scorer`; success payloads never
throw.

## Quick start (CLI)

```sh
kge import  --triples facts.jsonl --out model.json --scorer hole --dims 256
kge predict --model model.json --s Ada --r bornIn -k 10
kge similar --model model.json --r bornIn -k 5
kge compose --model model.json --r1 bornIn --r2 locatedIn --s Ada -k 10  # RotatE model
kge serve   --model model.json --port 8788
```

`facts.jsonl` is one `{"s","r","o"}` per line. `kge serve` exposes
`POST /v1/predict`, `POST /v1/compose` and `GET /healthz` on 127.0.0.1; it logs
method, path, status and milliseconds only — never the request body.

## The three operators

| Verb | Query | Returns |
|---|---|---|
| `predict` | `{s, r, k}` or `{r, o, k}` | `{candidates:[{entity,score}], exact, ann}` |
| `similarRelations` | `{r, k}` | `{relations:[{relation,score}]}` |
| `compose` | `{r1, r2, s, k}` (RotatE) | `{candidates:[{entity,score}], exact, ann}` |

`exact:true, ann:false` marks the exhaustive path; the
`DistanceMetric::DotProduct` HNSW path (ADR-001 §3) flips these once the ANN
index build lands.

## Training, evaluation, optimization

```ts
await kge.train({ epochs: 50, lr: 0.1 });   // CPU mini-batch training in place
kge.buildIndex();                            // HNSW over entities; predict goes ANN
const r = kge.evaluate({ split: 'test' });   // filtered MRR / Hits@k
r.report.combined.mrr;
```

- `train(config)` / `kge train` — mini-batch training over the tables (ADR-003).
- `evaluate(config)` / `kge eval` — filtered ranking metrics. A given `split`
  partitions the triples 80/10/10; this is a **derived** split for a smoke
  check, not the frozen content-hashed split ADR-006's gates require (that is
  the bench harness's artifact).
- `optimize(campaign)` / `kge optimize` — the self-optimization loop (ADR-004),
  **not yet wired**: the core `Campaign` needs an `Evaluator` the binding does
  not provide yet, so it returns `unavailable`.

## Security promises (ADR-005)

- **No network, no subprocess, no file reads in the scoring path.** The CLI
  reads/writes model files; the library never does. CI greps the built
  artifacts to enforce it (`scripts/check-security.mjs`).
- **The WASM module imports no WASI / fs / net / socket / fetch symbol**
  (`scripts/check-wasm-imports.mjs`, run in the wasm build).
- **Triple and label text is never logged**; errors carry numeric ids only.
- **Saved models are content-hashed** (sha256 envelope); load fails closed on a
  mismatch.
- **Input limits**, rejected with a typed error, never truncated: ≤ 1M
  entities, ≤ 100k relations, label ≤ 1 KiB, `k` ≤ 1000.

## Measured

FFT-vs-direct circular correlation, x86_64, from
[`bench/fft-spike-2026-09-21.json`](bench/fft-spike-2026-09-21.json) (rustfft
6.4.1; median ns/op):

| dimension `d` | FFT correlation speedup vs direct |
|---|---|
| 128 | 71× |
| 256 | 138× |
| 512 | 297× |

Batch scoring against a 2,000-entity table beats per-triple scoring by ~10.5×.
The ADR-002 §3 pass criterion — FFT beats direct at `d=512` and rustfft runs on
wasm32 — holds.

Link-prediction quality (filtered MRR / Hits@k on FB15k-237, WN18RR, CoDEx-M),
ANN recall, and latency are **to be measured by `kge bench`** (ADR-006); no
numbers are quoted here until that harness writes them.

## Design records

- [ADR-001 — architecture](docs/adr/ADR-001-architecture.md)
- [ADR-002 — scorers and math (HolE via FFT)](docs/adr/ADR-002-scorers-and-math.md)
- [ADR-003 — training and evaluation](docs/adr/ADR-003-training-and-evaluation.md)
- [ADR-004 — self-optimization loop](docs/adr/ADR-004-self-optimization-loop.md)
- [ADR-005 — security model](docs/adr/ADR-005-security-model.md)
- [ADR-006 — benchmarks and release gates](docs/adr/ADR-006-benchmarks-and-release-gates.md)

## License

MIT
