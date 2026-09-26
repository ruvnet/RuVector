# @ruvector/typesafe

Local typed decisions over sentence embeddings, with the wire contract of Jev
([typesafe.ai](https://typesafe.ai)) "System One". You send one text `state`
plus a batch of questions — `choice`, `score`, `noul` — and get back per-question
answers with a **calibrated** confidence, an abstain mass, and a receipt naming
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
- A governed self-optimization loop (ADR-004) that never gets worse on your
  frozen split, and explains every change.

Status: **v0.1.0.** The core and the TypeScript API are here. Platform binaries
are not yet published to npm; the native addon must be built locally, and a WASM
fallback ships. INT8 / ONNX embedder availability is per ADR-002's spike.

## Install

```sh
npm install @ruvector/typesafe
```

The first release ships **no dependencies**. If no prebuilt native binary
matches your platform, build the addon locally (`npm run build:napi`) or rely on
the bundled WASM fallback.

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
r.dept.confidence;        // calibrated top-1 probability
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

Measured 2026-09-21 on the frozen tickets fixture (8 departments; the training
pool yields **137 usable examples** after the frozen split). Receipts are
checked in under `bench/results/`.

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
