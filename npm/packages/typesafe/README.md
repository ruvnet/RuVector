# @ruvector/typesafe

Local typed decisions over sentence embeddings, with the wire contract of Jev
([typesafe.ai](https://typesafe.ai)) "System One". You send one text `state`
plus a batch of questions — `choice`, `score`, `noul` — and get back per-question
answers with confidence, an abstain mass, and a receipt naming
the head, model and temperature that produced each answer.

It is a bounded classifier, not an LLM host: **no network by default, no
per-token cost, no subprocess in the decision path.** A native (napi-rs) core
does the deciding; a WASM build is the portable fallback. Latency targets are
release gates (p95 ≤ 50 ms native, ≤ 150 ms WASM — ADR-006); the engine's own
numbers are measured by `typesafe bench` and recorded under `bench/results/`
(see [Measured](#measured)).

- Drop-in for Jev callers: `POST /v1/systemone` body and response shape are
  accepted and returned unchanged (`typesafe serve`).
- Type-safe questions: the answer to `choice({ billing, fraud })` is typed
  `{ choice: "billing" | "fraud"; probabilities: Record<"billing"|"fraud", number> }`.
- A governed self-optimization loop (ADR-004) that checks proposals against
  frozen validation and transfer splits, and explains every promotion.

Status: **v0.1.0.** The current npm package includes native platform binaries
with the hash test embedder and a WASM fallback. Native ONNX works when built
from source with `scripts/build-native.sh --onnx`; a future ONNX capable npm
release is gated on the frozen real model benchmark. Model weights are separate,
verified against `models/manifest.json`, and are never downloaded during a
decision.

## Install

```sh
npm install @ruvector/typesafe
```

The published package ships **no runtime dependencies**. On a supported
platform it loads the included native binary; otherwise it uses the bundled
WASM fallback. Both currently use the hash test embedder. To evaluate real
sentence embeddings from this source checkout, run `node scripts/fetch-models.mjs`
and `bash scripts/build-native.sh --onnx` from `npm/packages/typesafe`, then
select the model in `createTypesafe` (see [Embedders](#embedders)).

## Quick start (TypeScript)

```ts
import { createTypesafe, choice, score, noul } from '@ruvector/typesafe';

const ts = createTypesafe(); // default embedder: "hash" (see "Embedders")

const r = await ts.decide('my card was charged twice, fix it today', {
  dept: choice({
    billing: 'charges and refunds',
    fraud: { what: 'unauthorised use', not_for: 'duplicate charges' },
  }),
  mood: score(['Calm', 'Irritated', 'Angry'] as const),
  urgent: noul('the sender needs a response soon'),
});

r.dept.choice;            // "billing" | "fraud"   (typed from the criteria keys)
r.dept.probabilities;     // Record<"billing" | "fraud", number>
r.mood.legend;            // "Calm" | "Irritated" | "Angry"
r.urgent.noul;            // number, 0..1
r.dept.confidence;        // confidence; hash test embedder is not calibrated
r.answers.dept.choice;    // same answer, also under .answers
r.usage;                  // { embed_calls, texts_embedded, state_bytes }
```

`decide` batches N questions into one engine call. `decideMany(states, questions)`
runs a list of states (with a small concurrency pool on the async native path).

## Quick start (CLI)

```sh
# questions.json is a Jev-shaped question map
typesafe decide --state "my card was charged twice" --questions questions.json
echo "my card was charged twice" | typesafe decide --questions questions.json
typesafe serve --port 8787            # Jev-compatible HTTP server on 127.0.0.1
typesafe --help
```

## The three question types

- **`choice`** — pick one of up to 255 options. Each option is a description
  string, or `{ what, not_for, examples }`. `not_for` becomes a hard-negative:
  a `state` that fits no option gets low `confidence` and high `abstain` rather
  than a confident wrong answer (ADR-003).
- **`score`** — an ordinal legend, e.g. `['Calm', 'Irritated', 'Angry']`. The
  answer is the expected bucket index plus per-bucket probabilities.
- **`noul`** — a 0–1 predicate ("the sender needs a response soon"). With no
  labeled examples it falls back to a similarity score flagged
  `calibrated: false`; it is never reported as a probability it has not earned.

## Jev compatibility

`systemOne` accepts exactly Jev's `POST /v1/systemone` body
(`{ state, model?, questions: { id: { type, instructions, criteria | legend } } }`)
and returns Jev's response shape plus five additive fields (`abstain`,
`calibrated`, `head`, `model`, `temperature`). Pass `{ jevShapeOnly: true }` to
strip those and get exactly Jev's shape.

```ts
const jev = await ts.systemOne(
  { state, questions: { mood: { type: 'score', criteria: ['Calm', 'Angry'] } } },
  { jevShapeOnly: true },
);
```

The Jev `model` field is accepted for compatibility and ignored: the local
engine selects its own model arm under the loop's governance (ADR-004).

## Train, eval, serve

```sh
# Admit labeled examples for one question (JSONL of {"text","label"})
typesafe train --question dept --examples examples.jsonl

# Score a labeled dataset (a decisions JSON, or a JSONL of {text,label:{qid:..}})
typesafe eval --dataset labeled.jsonl --questions questions.json --split test

# Jev drop-in server
typesafe serve --port 8787
```

`eval` computes accuracy, macro-F1, ECE (10 bins), Brier, mean confidence, mean
abstain, and p50/p95 latency in JS from the responses. Example-bank persistence
depends on the binding exposing it; `train` reports `in-memory only for this run`
when it does not.

### Embedders

`createTypesafe()` defaults to the **`hash`** embedder: a deterministic
bag-of-words test double that needs no weights and runs everywhere. Its answers
are always reported `calibrated: false`. Production accuracy needs the ONNX
embedder:

```ts
const ts = createTypesafe({ embedder: { kind: 'onnx', modelDir: './models/bge', manifest: './models/manifest.json' } });
```

## The self-optimization loop

Improvement is a governed, measured, reversible loop (ADR-004): a proposal must
beat a frozen validation split under a paired anytime-valid test, must not
regress a separate transfer split, respects a per-day budget, and writes an
append-only, hash-chained receipt per decision. Because an argmax-preserving
change (a logit-scale prior, a temperature-floor tweak) carries no paired
*accuracy* information, the gate runs a **second** paired test over per-item
negative log-likelihood: a proposal promotes when the accuracy test rejects, or
when accuracy is non-inferior and the NLL (calibration) test rejects — and the
receipt records which criterion carried it. The promise is "never worse on your
frozen split, and every change explained", not "improves every hour".

**Implemented and measured (2026-09-21):** `train` (bank-backed, append-only),
`optimize` (the gate above), `export`/`import` of the example bank, and the
`typesafe optimize` CLI. `Engine::optimize` / `ts.optimize` run a campaign over
tunable `EngineOptions` for one embedder; the model arm is chosen by
constructing one engine per model. See
[ADR-004](docs/adr/ADR-004-self-optimization-loop.md).

## Security

The engine takes untrusted text and returns typed decisions. Its runtime is
deliberately boring (ADR-005):

- **No network by default.** Models are bundled or loaded from a hash-pinned
  local manifest.
- **No subprocess in the decision path.** The CLI spawns nothing; helpers would
  use argument arrays, never a shell.
- **No logging of inputs.** `state` is never logged; receipts store hashes and
  lengths, not text. The server's access log is method, path, status, ms only.
- **Input limits** are rejected with a typed error, never silently truncated:
  `state` ≤ 16 KiB, ≤ 255 options/choice, ≤ 64 questions/request.

A regression test (`test/security.test.mjs`) asserts the source contains no
`child_process`, no `fetch`/`http` client, and no logging of `state`/`request`/
`body`. See [ADR-005](docs/adr/ADR-005-security-model.md).

## Measured

The 2026-09-21 numbers below are exploratory historical receipts. The corrected
harness trains all three heads and removes one exact training text shared with
a held out split. Its new results must be measured separately and must clear
the strict release gates before claiming a quality or speed improvement.
The old receipts are checked in under `bench/results/`.

**typesafe engine (local, onnx)** — `department` choice question, from
`bench/results/tickets-onnx-bge-2026-09-21.json` (16-shot) and
`bench/results/optimize-tickets-2026-09-21.json` (full-data campaign; per-arm
receipts in `bench/results/optimize-receipts-2026-09-21.jsonl`):

| configuration | accuracy | ECE | p95 ms |
|---|---|---|---|
| Jev replay (reference) | 85.3% | 0.073 | 231 |
| bge-small, 16-shot | 80.0% | 0.075 | 10.0 |
| bge-small-int8, 64-shot | 77.3% | — | 4.1 |
| campaign champion bge-small (full data) | 83.3% | 0.068 | 10 |
| campaign champion bge-small-int8 (full data) | 84.0% | 0.071 | 4 |
| campaign champion MiniLM (full data) | 81.3% | 0.057 | — |

Gates (ADR-006): **`accuracy_vs_jev` passes** with the full-data campaign
champion (83.3% ≥ 82.3%) and **native latency passes** (p95 ≤ 50 ms). Two do
**not** pass: `calibration_ece` (best 0.057 > the 0.05 target) in every regime,
and the accuracy gate in the **16-shot** regime (80.0% < 82.3%). All seven
campaign promotions came through the calibration criterion — the accuracy paired
test alone rejected each (best wealth 4.91 of the 20 threshold).

**Jev (typesafe.ai) baseline**, from `bench/jev-baseline-2026-09-21.json`
(500 synthetic support tickets, 8 departments, frozen 150-item **test** split,
concurrency 4, latency includes the network round trip):

| metric (test split) | Jev baseline | Jev after 8-gen loop |
|---|---|---|
| department accuracy (`choice`) | 85.3% | 90.0% |
| urgent accuracy (`noul`) | 61.3% | 60.0% |
| frustration accuracy (`score`) | 67.3% | 67.3% |
| latency p50 / p95 (ms) | 184.8 / 233.0 | 178.5 / 215.6 |
| ECE | 0.073 | 0.068 |

Jev's `noul` urgency (61.3%) sits **below** a constant "not urgent" baseline of
71.3% (107 of the 150 test tickets are not urgent, from `test_rows.baseline` in
the same JSON), and its confidence is saturated (ECE 0.073) — the design reasons
the abstain bucket and calibration layer exist (ADR-003).

**ruvector substrate (index + query)**, from
`bench/ruvector-router-2026-09-21.json` (5,000 docs, 384-d, 250 test queries,
recall@10):

| implementation | recall@10 | latency p50 / p95 (ms) |
|---|---|---|
| ruvector VectorDb (native) | 1.00 | 5.85 / 6.88 |
| `@ruvector/router` 0.1.28 VectorDb | 0.032 | 0.05 / 0.07 |

The core reuses `ruvector-router-core` directly (ADR-001). The native VectorDb
returns exact neighbours; the `@ruvector/router` 0.1.28 kNN path returns
neighbours only from the most recently inserted region (recall 0.032) — a
documented defect in that file, called out here rather than papered over.

## Architecture (ADRs)

- [ADR-001 — architecture](docs/adr/ADR-001-architecture.md)
- [ADR-002 — inference backend](docs/adr/ADR-002-inference-backend.md)
- [ADR-003 — decision heads and calibration](docs/adr/ADR-003-decision-heads-and-calibration.md)
- [ADR-004 — the self-optimization loop](docs/adr/ADR-004-self-optimization-loop.md)
- [ADR-005 — security model](docs/adr/ADR-005-security-model.md)
- [ADR-006 — benchmarks and release gates](docs/adr/ADR-006-benchmarks-and-release-gates.md)

## License

MIT
