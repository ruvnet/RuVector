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

The source package is version **0.1.1** for the next release. This version is
not published until the native ONNX builds and strict release gates pass.

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

## Known limitations

> **Release gate status (0.2.0).** This release was published with a maintainer
> override of the frozen ONNX ticket gates, which it does not meet. Measured on the
> tickets suite (bge-small-en-v1.5, 8-shot, shipped native binary): accuracy vs
> Jev **0.807** (gate ≥ 0.823), calibration ECE **0.080** (gate ≤ 0.05), urgent vs
> train-majority **0.687** (gate ≥ 0.713), frustration vs train-majority **0.367**
> (gate ≥ 0.527). Native p95 latency (~24 ms, gate ≤ 50 ms) and transfer-regression
> gates pass. The numbers are unchanged from 0.1.x. Validate on your own labelled
> data before relying on `confidence` or these questions.

From an independent evaluation (25–27 Sep 2026, bge-small-en-v1.5, native ONNX
build); details and reproduction in the
[Typed Decisions Lab](https://typesafe-lab-276367410975.europe-west2.run.app).

- **`choice` does not read `instructions` by default.** Only the option
  criteria are embedded, so changing "Which team should *own* this" to "*avoid*
  this" gives the same answer. Put the intent into the option descriptions, or
  opt in with `engine: { choiceInstructions: true }`.
- **Negation.** Embeddings barely separate a predicate from its negation: on one
  urgent message, "needs a response soon" scored 0.84 and "does NOT need a
  response soon" 0.81. Phrase predicates positively and add labelled examples.
- **Untrained `noul` / `score`.** Without examples, urgency on the tickets
  fixture is at chance (AUROC 0.51) and a Low/Medium/High severity question on
  a fresh benchmark scored 33% exact. Train these questions before relying on
  them.
- **Out-of-scope inputs.** `abstain` ranks off-topic states well: on CLINC150
  its AUROC is 0.90 with all 150 intents and 0.94–0.97 on 8-intent subsets. Its
  values are small, though, and shrink as the option count grows (median 0.002
  at 150 options) and after training, so a fixed threshold does not carry
  across questions. An ordinary `other` option caught 17% of off-topic states
  with 24% false alarms. Treat `abstain` as a relative score and tune any
  threshold per question, or use `abstainMode: 'sigmoid'` and a `catchAll`
  option (both below).
- **Calibration needs data.** `calibrated` stays `false` until the calibration
  slice has 20 examples, which at the default split means about 100 labels per
  question. `engine: { crossfitCalibration: true }` needs about 20 labels in
  total instead.

### Out-of-scope scores

`abstain` ranks off-topic inputs well: on CLINC150
(bge-small, zero-shot) its AUROC for the 1,000 out-of-scope test utterances is
0.90 with all 150 intents and 0.94–0.97 on 8-intent subsets. By default it is
the abstain share of a (K+1)-way softmax, so its scale depends on the option
count (median 0.002 at 150 options, 0.03 at 8) and, after training, on the
fitted temperature (about 1e-8 after 8-shot training). To threshold it,
pass `createTypesafe({ engine: { abstainMode: 'sigmoid' } })`: `abstain` is then
`sigmoid` of the same out-of-scope logit, on a fixed 0–1 scale that does not
depend on K or training. A threshold tuned on CLINC150's 150-intent validation
split (0.336) then caught 78% of test out-of-scope items at 14% false alarms, and
92–97% at 11–19% on 8-intent subsets. `choice`, `probabilities` and
`confidence` are the same in both modes.

### Off-topic inputs: a catch-all option

Adding an option such as `other: 'Anything else'` to a `choice` question does
little by default: its text is matched like any other option, and off-topic
states still land on the nearest real option. Declare it as a catch-all
instead:

```ts
const ts = createTypesafe({ engine: { catchAll: 'other', catchAllThreshold: 0.36 } });
```

The catch-all's own text is then ignored. Its probability is the out-of-scope
score over the real options (distance to their nearest prototype, or the best
`not_for` match), the real options share the rest, and `other` is chosen when
its probability reaches the threshold.

The threshold depends on the embedder and on how the options are worded, so
tune it on a few labelled in-scope and off-topic examples for each question.
Measured with bge-small, zero-shot:

| Question | `other` as an ordinary option | Catch-all, tuned threshold |
|---|---|---|
| CLINC150, 150 intents (1,000 off-topic test utterances) | 1.5% caught, 0.1% false alarms | 77.8% caught, 14.4% false alarms (0.336, tuned on validation) |
| Tickets, 8 departments (100 tickets, 106 off-topic states) | 25.5% caught, 0% false alarms | 85–96% caught, 6–8% false alarms (0.358–0.361, tuned on the other half) |

A threshold does not carry between these two questions: CLINC150's 0.336
flags 61% of real tickets.

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

By default a `choice` question's `instructions` are not embedded: only the
criteria shape the answer, so "which asset do they own" and "which asset do they
avoid" score the same. `score` already folds its instructions into each legend
bucket. To do the same for `choice`, pass
`createTypesafe({ engine: { choiceInstructions: true } })`: each option's `what`,
examples and `not_for` are then embedded as `"<instructions>. <text>"`. With
empty instructions the answers are identical to the default.

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
are always reported `calibrated: false`, and on real text they carry no
meaning (the quick start above returns near-even probabilities), so use it for
tests and wiring only. Production accuracy needs the ONNX embedder:

```ts
const ts = createTypesafe({ embedder: { kind: 'onnx', modelDir: './models/bge', manifest: './models/manifest.json' } });
```

On WASM, TypeSafe reads the pinned model and tokenizer from `modelDir` and
passes their bytes to `Engine.fromBytes`. If a manifest contains several
models and `modelDir` does not match an entry name, select one explicitly with
`embedder.model` (or `--model <name>` in the CLI). The model files must already
exist locally; TypeSafe does not download them at runtime. The WASM artifact
must be built with the `wasm-onnx` feature for ONNX inference.
The current WASM `fromBytes` backend does not support `engine` tuning options;
it rejects them rather than silently ignoring them.

Creating an engine on the `hash` embedder emits one process warning
(`TYPESAFE_HASH_EMBEDDER`) per process, so a quick start never silently ships
test-double answers. Tests can pass `{ warnOnHashEmbedder: false }` to silence it.

### Calibration with few labels

`confidence` is calibrated (`calibrated: true`) once the held-out calibration
slice reaches `minCalibration` (20) examples; with the default every-5th split
that takes about 100 labels per question. Below that, pass
`createTypesafe({ engine: { crossfitCalibration: true } })`: the temperature
(or the `noul` Platt layer) is then fitted on 5-fold out-of-fold scores over all
labels, so 20 labels in total are enough. The head that answers is unchanged,
so choices and accuracy are identical. On CFPB product routing (11 classes,
bge-small, 600 test complaints, 3 samples each) it cut ECE from 0.23 to 0.13
with 66 labels and from 0.23 to 0.09 with 88; at 110 labels the held-out slice
is large enough and both settings give the same answers.

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

Receipts are chained with a fast 128-bit FNV-1a checksum by default. It catches
accidental edits and corruption, but anyone who can edit the log can also
recompute it. For an audit trail, pass `receipt_hash: 'sha256'` in the campaign
spec: each receipt is then chained with SHA-256 (stored as `"sha256:<hex>"`),
and publishing or signing the last hash makes every earlier receipt
tamper-evident. Each stored hash names its algorithm, so existing FNV logs keep
verifying, and `verify_chain_requiring(HashAlg::Sha256)` (Rust) rejects a log
rewritten with the weaker hash.

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

At a fixed 0.5 threshold, Jev's `noul` urgency (61.3%) sits **below** a
constant "not urgent" baseline of 71.3% (107 of the 150 test tickets are not
urgent, from `test_rows.baseline` in the same JSON). The replay stores only the
thresholded booleans; an independent live re-run on the same test split
(jev-1.13.0, 25 Sep 2026) kept Jev's continuous `noul`, which ranks urgency
well (AUROC 0.94) and scores 91.3% with a threshold of 0.91 chosen on the
validation split. So the gap is a thresholding choice rather than a ranking
failure; report AUROC alongside accuracy for `noul`.

**ruvector substrate (index + query)**, from
`bench/ruvector-router-2026-09-21.json` (5,000 docs, 384-d, 250 test queries,
recall@10):

| implementation | recall@10 | latency p50 / p95 (ms) |
|---|---|---|
| ruvector VectorDb (native) | 1.00 | 5.85 / 6.88 |
| `@ruvector/router` 0.1.28 VectorDb | 0.032 | 0.05 / 0.07 |

ADR-001 plans for the core to reuse `ruvector-router-core`; the current core
does not import it (retrieval uses its own prototype and probe heads), so it is
not a dependency today. The native VectorDb
returns exact neighbours; the `@ruvector/router` 0.1.28 kNN path returns
neighbours only from the most recently inserted region (recall 0.032) — a
documented defect in that file, called out here rather than papered over.

## ESP32 S3 and C6 firmware

The [native decision firmware](../../../examples/esp32-decision) runs frozen
`choice`, `score`, and `noul` heads on ESP32 S3 and C6 using ESP-IDF 5.4.4.
It accepts numeric feature vectors produced by the same pipeline used during
training. Text embedding remains on the host unless you separately implement
and validate the identical embedder on the device. The included model is a
synthetic numerical fixture, not a trained sensor or language model.

Export a fitted Rust engine with `engine.export_embedded(question_id, &question)`
and serialize the returned snapshot with `serde_json`. This exports real probe
weights, prototypes, hard negatives, temperature and Platt coefficients; it is
independent of the example bank export and RVF format.

```sh
cd examples/esp32-decision  # from the repository root
python3 tools/e2e.py       # actual Rust core tests, export, quantize, C protocol tests
python3 tools/quantize.py build-host/fixtures/probe.json main/model.h \
  --vectors build-host/fixtures/probe.vectors.json --bits 16
# Activate the ESP-IDF 5.4.4 environment first.
idf.py -B build-esp32s3 -DIDF_TARGET=esp32s3 -DSDKCONFIG=sdkconfig.esp32s3 build
idf.py -B build-esp32c6 -DIDF_TARGET=esp32c6 -DSDKCONFIG=sdkconfig.esp32c6 build
idf.py -B build-esp32s3 -p /dev/ttyUSB0 flash
python3 tools/e2e.py --port /dev/ttyUSB0 --model probe --skip-rust
```

Use the board's UART0 USB bridge at 115200 baud. The default build assumes
4 MB flash and requires no PSRAM or network. For native USB-only boards, use
an external UART adapter or explicitly adapt the console transport. Commands
are newline terminated `meta`, `selftest`, `bench`, and `infer` followed by
space separated numeric features. Replies are JSON. Sensor code can call
`rd_init` once and `rd_predict` directly with one workspace per concurrent
caller. No heap allocation occurs in the inference kernel. The envelope is
768 features and 16 class options.

INT16 is the default accuracy profile. INT8 is available with `--bits 8`, but
must pass validation for the specific head: small errors can be amplified by
sharp calibration. Fixed gates require at least 99% decision and acceptance
agreement and at most 0.025 absolute probability error against Rust on unseen
vectors. Quantization does not inherit a calibrated confidence claim: replies
report `calibrated:false`, retaining `source_calibrated` in metadata.

The firmware refuses inference when boot self-tests fail. The hardware runner
checks the model digest, replay parity, heap stability and p99 below 100 ms.
Host or emulator timing is not a physical board benchmark. Device firmware
must be tested with the intended sensors and feature pipeline before deployment.

The optimized kernel deduplicates constant rows, reuses identical negative
similarities and computes only the required abstain softmax output. C6 uses
exact integer activation rounding; S3 retains newlib rounding after it proved
cheaper in instruction-count emulation. Both MCU targets pair INT16 products
before widening the sum. Two products fit INT32 because activations are bounded
to +/-32767; the total remains INT64. Native builds keep the vectorizable loop.

The shipped INT16 parameters occupy 492 bytes, down from 556 (11.5% less).
Inference workspace is 1,732 bytes and MCU application buffers total 2,692
bytes, an 84-byte increase for cached negative scores and model factors.
These counts exclude model descriptors, labels, protocol stack and ESP-IDF.

Reproduce paired measurements against the pinned pre-optimization commit:

```sh
python3 tools/e2e.py --seed 20260926
git fetch --depth=1 origin 739a5621043f9b0f0ffc263c46d1d2f7aeb9495f
python3 tools/benchmark.py --output build-bench/results.json
python3 tools/benchmark.py --verify-only --profile esp32s3
python3 tools/benchmark.py --verify-only --profile esp32c6
python3 tools/benchmark_qemu.py --qemu /path/to/qemu-system-xtensa \
  --baseline-image /path/to/baseline-s3-merged.bin --icount
```

The native benchmark uses 11 alternating paired rounds of 2,048 calls,
warmup, CPU affinity and 48 synthetic shapes, including absent, distinct and
shared negatives. It also measures the exported 32- and 384-feature probes.
`--profile` changes kernel flags only; execution still occurs on the host.
Every profile passes 84,480 bit-for-bit comparisons against the pinned kernel.
Separate Rust parity streams cover 12,000 INT16 decisions, all matching both
decisions and acceptance with worst absolute probability error below 0.001.

In S3 QEMU, the shipped fixture improves from 7.880 to 6.621 virtual microseconds
per call with `-icount shift=0,sleep=off`, about 16% less instruction-clock time.
This clock assigns one virtual nanosecond per emulated instruction; it does
not model silicon cycles, caches or power. Realtime QEMU results are also saved
and fluctuate with host scheduling. An early integer-rounding S3 variant used
6% more virtual time and was rejected. Full samples, hashes and method details
are under `examples/esp32-decision/tests/evidence/benchmark-*.json`.
No physical S3 or C6 performance measurement has been made.

The [sensor benchmark ADR](docs/adr/ADR-007-esp32-measured-sensor-decisions.md)
defines a separate lab for real measurements and physical acceptance. Run its
complete software gate from `examples/esp32-decision`:

```sh
python3 tools/lab.py
```

This downloads the SHA256-pinned UCI Occupancy Detection dataset, trains the
actual Rust engine, tests INT8 and INT16, checks sensor preprocessing and
profiling, and runs 11 paired native trials. It requires Python 3.10+, Rust,
GCC and Git. The dataset is Luis Candanedo (2016), DOI
[10.24432/C5X01N](https://doi.org/10.24432/C5X01N), licensed CC BY 4.0.
No raw dataset is committed. Training uses 7,569 rows, validation uses the last
574 training readings as a separate day, and test uses the supplied separate
12,417 readings. Dates and row IDs are excluded from model features.

The validation-selected INT8 model uses 44 decision-parameter bytes plus 40
bytes of preprocessing coefficients. It scores 98.06% test accuracy versus
98.24% for the 64-byte INT16 head. Quantization parity is 99.80% for decisions
and 99.81% for acceptance, with maximum probability error 0.01652. Both pass
the fixed numerical gates. These are measurements within one office, with no
claim of transfer to other buildings or devices. Confidence acceptance covers
only 51.98% of test readings and 1.57% of validation readings. The model is a
benchmark demonstration and is explicitly marked unsuitable for automatic
deployment pending useful coverage, calibration and physical validation.

`build-sensor/selected/model.h` contains the frozen selected model. Input order
for `sensor` is Celsius, relative humidity percent, lux, CO2 ppm, humidity ratio
kg/kg. For example: `sensor 23.18 27.272 426 721.25 0.004792988`.
The standardizer is fitted on training only and is part of the model digest.
`infer` still accepts already standardized features. Replies expose `parse_us`,
`preprocess_us`, `inference_us` and `capture_us` (null for external readings).

To attach an actual driver, implement strong `rd_sensor_read(float*, size_t)`
and `rd_sensor_name()` functions in a board source file, add it to the main
component, and fill every requested value in the same units/order. The `sample`
command times capture, preprocessing and inference. The default driver returns
`capture_unavailable`; the test driver is explicitly labeled `test_fixture_only`.
No sensor acquisition time has been measured on physical hardware.

`profile N` supports 1 to 2,048 calls over eight golden inputs, reporting
median, p95, p99, maximum, mean cycles, timer overhead, CPU/core and heap.
Profiling adds 16,384 bytes of static sample storage, separate from inference
workspace. Raw overhead is reported rather than silently subtracted. The
existing `bench` command remains available. `energy N` supports 1 to 256 calls,
without per-call timing, with an optional GPIO marker around the whole batch.
The GPIO is disabled by default; select a free board pin with
`CONFIG_RD_BENCH_GPIO` only when connecting a power analyzer.

For production, `CONFIG_RD_PROFILE_SAMPLES=0` removes all 16,384 bytes of
percentile storage while preserving inference, self-tests, `bench` and `energy`.
The `profile` command returns `profile_disabled`; metadata reports actual
capacity and bytes. Build in a separate directory so benchmark settings remain
reproducible (substitute `esp32c6` for C6):

```sh
idf.py -B build-production-esp32s3 -DIDF_TARGET=esp32s3 \
  -DSDKCONFIG=build-production-esp32s3/sdkconfig \
  '-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.production' \
  -DRD_MODEL_DIR="$PWD/build-sensor/selected" build merge-bin
```

Verify the linked storage with `python3 tools/production_check.py
--full-build build-rig/esp32s3-candidate --production-build
build-production-esp32s3 --output build-production-esp32s3/footprint.json`.
Add `--qemu /path/to/qemu-system-xtensa --vectors build-sensor/validation.json`
for an exact S3 replay and warmed heap check. The shipped production builds
recover 16,384 static bytes and reduce application flash by 1,008 bytes on
S3 and 1,040 bytes on C6 relative to the same model with full profiling.

Firmware also defaults to `RD_MODEL_SIZED_WORKSPACE=ON`, sizing workspace
and context to the immutable model. For the five feature occupancy model,
workspace falls from 1,732 to 40 bytes and context from 32 to 20 bytes: another
1,704 static bytes recovered on each MCU. Matching production images save
1,008 additional flash bytes on S3 and 608 on C6. Generic library callers keep
the original 768 feature / 16 class capacities. All firmware consumers receive
identical capacity definitions; changing the model header triggers CMake
reconfiguration. Legacy headers without class metadata keep 16 class slots.
Use `-DRD_MODEL_SIZED_WORKSPACE=OFF` to restore generic firmware capacity.

To reproduce the comparison, build two production images with the same model
and target in separate directories, using `OFF` for `build-generic` and `ON`
for `build-compact`, then run:

```sh
python3 tools/workspace_check.py --generic build-generic --compact build-compact \
  --output build-compact/workspace.json
# For S3, add --qemu /path/to/qemu-system-xtensa --vectors build-sensor/validation.json
```

The checker verifies linked storage and ABI definitions. Optional S3 replay
requires exact replies and stable warmed heap. Run `python3 tools/lab.py` with
GCC, Cargo and CMake on PATH for the complete host acceptance suite. Memory
savings do not establish a physical speed, energy or deployment readiness gain.

`python3 tools/coverage_lab.py` tests three fixed abstention settings using a
separate 1,440-row calibration day. Training and preprocessing exclude this
day. Selection requires useful coverage and accepted accuracy for both classes;
the following validation day can veto it without selecting another candidate.
The initial candidate reached 99.65% calibration accuracy but only 90.59%
validation accuracy and zero validation coverage, so it was rejected. The
default model remains unchanged. This lab also runs in `tools/lab.py`. Its
test data are previously seen regression data, and all results remain limited
to one office. No deployment or statistical calibration claim is made.

Build paired images after activating the ESP-IDF 5.4.4 environment:

```sh
python3 tools/prepare_pair.py --model-dir build-sensor/selected \
  --output build-rig --targets esp32s3 esp32c6
python3 tools/rig.py build-rig/manifest.json --vectors build-sensor/test.json \
  --execution physical --chip esp32s3 --port PORT --flash \
  --metric mean_cycles --rounds 11 --output build-rig/s3.json
# Repeat with --chip esp32c6 and its serial port.
```

The physical runner intentionally flashes both supplied images repeatedly,
alternating their order. Each image contains the same profiling harness and
model; only the pinned kernel differs. The manifest records binary and source
hashes. Fixed CPU frequency, affinity, matching target/model/kernel and exact
paired replay output are checked. Use `--execution native` for host processes
or `--execution emulator --qemu /path/to/qemu-system-xtensa --chip esp32s3`
for S3 emulation. Neither execution mode can satisfy the physical retention
gate. `--limit` defaults to 1,000 reference rows per image per round.
Reports separate protocol round trip from kernel timing and store raw rounds.

Export a power analyzer trace as CSV with `time_s,voltage_v,current_a,marker`.
Include idle samples before and after one complete energy batch. `marker` must
be 0 or 1. Associate it with the corresponding round and arm:

```sh
python3 tools/energy.py trace.csv --run-report build-rig/s3.json \
  --round 0 --arm candidate --output build-rig/energy-candidate-0.json
python3 tools/energy_compare.py pairs.json --rig-report build-rig/s3.json \
  --output build-rig/energy-decision.json
```

`pairs.json` is an array of objects with `baseline` and `candidate` paths to
energy reports. The analyzer integrates voltage times current and reports
gross joules per decision, optional idle-adjusted energy, sample resolution,
scope and provenance. Reused traces, incomplete marker windows and duration
mismatches are rejected. No voltage/current measurement is synthesized from
CPU timing. Retention requires at least 11 independent pairs, median speedup
>=1.10, the bootstrap lower confidence bound >1.0, passing correctness and
stable heap. Energy uses the same gate with gross joules per decision.
Traces need at least ten sample intervals inside the marker window. Both
kernels disable floating point contraction: S3 replay revealed tiny rounding
differences with the compiler default that native tests had not exposed.
Earlier `benchmark-*.json` files retain historical optimization evidence;
`lab-*.json` records this sensor and measurement iteration.

## Architecture (ADRs)

- [ADR-001 — architecture](docs/adr/ADR-001-architecture.md)
- [ADR-002 — inference backend](docs/adr/ADR-002-inference-backend.md)
- [ADR-003 — decision heads and calibration](docs/adr/ADR-003-decision-heads-and-calibration.md)
- [ADR-004 — the self-optimization loop](docs/adr/ADR-004-self-optimization-loop.md)
- [ADR-005 — security model](docs/adr/ADR-005-security-model.md)
- [ADR-006 — benchmarks and release gates](docs/adr/ADR-006-benchmarks-and-release-gates.md)
- [ADR-007 — ESP32 measured sensor decisions](docs/adr/ADR-007-esp32-measured-sensor-decisions.md)

## License

MIT
