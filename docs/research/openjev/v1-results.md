# OpenJev v1 — validation results

Design and pre-registered gates: `npm/packages/typesafe/docs/adr/ADR-008-openjev-model.md`
(unchanged by this note). v0 plan: `docs/research/openjev/v0-plan.md`.
Branch `feat/openjev-v1` (from `feat/openjev` @ `4560efa2a`).

**Every number here is on the VALIDATION split** (`run.mjs --no-test`); the
test split was never scored, replayed or passed to training. Validation is
selection-biased for OpenJev (ADR-008 §1b), so none of these numbers is a
cross-arm claim.

## TL;DR

- **The encoder did not need to change.** v1 = the v0 training recipe
  (`configs/openjev-small-v0.toml`, unchanged) × 5 seeds, evaluated under a
  corrected harness configuration. All five seeds pass every gate that is
  evaluable on validation: ECE, urgent, frustration, native p95, CLINC150 OOS
  AUROC (0.966 ≥ 0.85), and no transfer regression vs base (84.6 % → 84.6–96.2 %).
  `accuracy_vs_jev` and the ADR-008 §1b vs-Jev tiers need Jev's test rows and
  are test-only — untouched here.
- **Urgent / frustration "failures" were a harness regime, not the model.**
  The quoted v0 numbers (urgent 49.3 %, frustration 50.0 %) came from the
  harness default `--shots 8`; v0-plan Step 4/5 specifies `--shots 10000`
  (full data). Under full data the same v0 checkpoint scores urgent 84.0 % /
  frustration 80.7 % (engine carve calibration) — both gates pass.
- **ECE needed two fixes**, neither a training change: (1) calibrate on the
  fixture's held-out `calibration` split, as ADR-008 §4 says — the harness
  never did; (2) let the engine's linear probe converge
  (`probeIterations` 400 → 5000). (1) alone made ECE *worse*; see "What failed".
- **Cost: $0.** All training on the local RTX 5080; vast.ai not used.

## Shipped evaluation configuration

```bash
node bench/run.mjs --suite tickets --arm local --embedder onnx \
  --model-dir <seed dir> --model openjev-small-v0 \
  --shots 10000 --engine-options '{"probeIterations":5000}' --no-test
```

- `--shots 10000`: all train examples per class (v0-plan Step 4/5).
- calibration: the fixture `calibration` split (37 rows) is the engine's
  calibration slice by default (new; `--no-calibration-split` reproduces the
  pre-v1 carve). It is never a training row for the engine or the trainer
  (trainer Assertion A / bench Assertion B: intersection 0).
- `probeIterations: 5000` via the existing ADR-004 `EngineOptions` surface;
  `EngineOptions::default()` is unchanged (400), so every old receipt and
  bit-identity test still holds.

## Diagnosis (Task 1)

How the engine answers the two secondary questions
(`crates/ruvector-typesafe-core/src/{heads.rs,engine/fit.rs}`):

- `urgent` (`noul`): a class-balanced binary logistic over the embeddings of
  the trained examples, then Platt scaling on the calibration slice;
  `noul ≥ 0.5` is "urgent". The trainer's urgency head is discarded; only the
  embedding geometry carries over.
- `frustration` (`score`): the linear probe's softmax over the 3 legend
  buckets, temperature-scaled, and the answer is **round(expected bucket)**,
  not the argmax.
- Examples come from `trainTicketQuestions` with a per-class cap of `--shots`.

With the default `--shots 8`, each head sees 8 examples per class — a
class-balanced sample — and the heads are themselves class-balanced. The
train prior (urgent: 104 no / 32 yes; frustration: 82 / 39 / 15) is erased:

- urgent is scored against a threshold tuned for a 50/50 prior on a 71 %-no
  split, and the 16-example logistic has AUROC ≈ 0.54;
- frustration's expected-value rounding over a balanced 3-way posterior
  lands on bucket 1 far too often (bucket 0 is the 58.7 % majority).

With full data the heads see the real prior and 136 examples. Evidence
(seed-1 v0 checkpoint and base bge-small, validation, engine carve):

| arm | regime | dept | ECE | urgent | urgent AUROC | frustration |
|---|---|---|---|---|---|---|
| bge-small (base) | 8-shot | 81.3 | 0.4436 | 52.7 | 0.623 | 38.7 |
| bge-small (base) | full data | 78.0 | 0.0644 | 74.7 | 0.758 | 72.7 |
| OpenJev v0 seed 1 | 8-shot (quoted v0 numbers) | 95.3 | 0.1686 | 49.3 | 0.525 | 50.0 |
| OpenJev v0 seed 1 | full data | 95.3 | 0.0537 | 84.0 | 0.739 | 80.7 |

Gates: urgent ≥ 71.3 %, frustration ≥ 58.7 % (train-majority rates), ECE ≤ 0.05.

**Correction to an ADR-008 premise (not edited in the ADR).** ADR-008
Context point 1 says "frozen general-purpose embeddings carry no urgency
signal (AUROC 0.51)". That was a 16-shot artifact: with full data, *base*
bge-small reaches urgent AUROC 0.758 and passes both secondary gates on
validation. The encoder fine-tune still adds a lot (AUROC 0.89 and
frustration 94 % below), but the premise that no head can learn urgency from
frozen vectors does not hold under the plan's own protocol.

## Calibration (Task 3)

Mechanism of the v0 ECE miss under full data: the engine carved every 5th
*train* row as its calibration slice. Those rows were gradient rows for the
encoder, so they are separated perfectly, the NLL-optimal temperature hits
the floor `T_MIN = 0.1`, and validation gets confidence 0.9996 on 143/150
items at 95.8 % accuracy (ECE 0.0537).

Fix, per ADR-008 §4: fit temperature/Platt on the fixture `calibration`
split. The engine had no way to take an explicit calibration slice, so
`ruvector-typesafe-core` gained `Engine::train_with_calibration` (bank
`Calibration` split; with no explicit slice the carve is bit-identical —
unit-tested, and re-running both arms with `--no-calibration-split`
reproduced the pre-change receipts exactly), exposed as an optional
`calibration` array on `trainJson` (native + WASM). Temperature is never
fitted on validation.

Calibration-slice department accuracy vs validation (v1 config; the slice is
harder than validation for every arm, which is why a temperature fitted on it
is conservative):

| arm | calibration split (n=37) | validation (n=150) | fitted T (dept) |
|---|---|---|---|
| bge-small (base) | 78.4 % | 84.0 % | 0.33 |
| seed 1 | 86.5 % | 94.0 % | 0.57 |
| seed 2 | 91.9 % | 94.7 % | 0.47 |
| seed 3 | 86.5 % | 96.0 % | 0.51 |
| seed 4 | 86.5 % | 95.3 % | 0.55 |
| seed 5 | 91.9 % | 95.3 % | 0.62 |

### Probe convergence (`probeIterations` 5000)

The engine's heads are full-batch gradient descent with a fixed budget
(`lr 0.8`, 400 iterations). That budget does not converge: base bge-small on
Banking77 validation scores 60.1 % at 400 iterations and 80.0 % at 5000. On
tickets the budget was chosen by the **calibration-slice** NLL (fixture
calibration split, post-temperature; no validation row involved), taking the
smallest tested budget within 0.001 of the 10 000-iteration value:

| iterations | seed 1 NLL dept / frus / urgent | bge-small NLL dept / frus / urgent |
|---|---|---|
| 400 | 0.272 / 0.281 / 0.286 | 0.696 / 0.540 / 0.459 |
| 1000 | 0.271 / 0.062 / 0.124 | 0.553 / 0.457 / 0.428 |
| 2000 | 0.271 / 0.016 / 0.075 | 0.522 / 0.426 / 0.405 |
| 5000 | 0.272 / 0.011 / 0.064 | 0.511 / 0.412 / 0.391 |
| 10000 | 0.272 / 0.011 / 0.064 | 0.511 / 0.411 / 0.391 |

Disclosure: the plateau was first *seen* on validation (seed 1: ECE 0.061 →
0.041 from 400 → 2000+ iterations, flat to 10 000; base bge-small the same
shape). The calibration-slice curve above is the val-free criterion that
picks the same value.

## Seed results (Task 4) — validation, v1 configuration

Local RTX 5080, sequential (measured: a v0 training run peaks at ~9–10 GB
of the card's 16 GB per `nvidia-smi`, so two concurrent runs do not fit);
~4.5–5.5 min per seed. Seed 1 is the v0 staged checkpoint
(`runs/staged-seed1`, onnx sha256 `1d632c31…`); seeds 2–5 were trained at
this branch's base with the identical config and binary (trainer source
unchanged since `05066f4`) on byte-identical data: `train.jsonl`
`9c2bd619…` and `val.jsonl` `a6c911bf…` verified unchanged after the run's
own `prep`, whose Assertion A report (`runs/v1-art/data-leakage-report.json`)
shows intersection 0 for every dataset. Transplant parity passed for every seed (cosine
min ≥ 0.99999999999, 256/256 identical decisions).

| seed | dept | ECE | urgent | urgent AUROC | frustration | p95 ms | transfer |
|---|---|---|---|---|---|---|---|
| 1 | 94.0 | 0.0406 | 86.7 | 0.884 | 93.3 | 17.0 | 84.6 |
| 2 | 94.7 | 0.0457 | 86.7 | 0.906 | 94.7 | 16.8 | 88.5 |
| 3 | 96.0 | 0.0427 | 89.3 | 0.896 | 93.3 | 12.8 | 88.5 |
| 4 | 95.3 | 0.0365 | 81.3 | 0.856 | 96.0 | 10.8 | 88.5 |
| 5 | 95.3 | 0.0375 | 89.3 | 0.920 | 94.0 | 9.6 | 96.2 |
| **mean ± sd** | **95.1 ± 0.8** | **0.0406 ± 0.0038** | **86.7 ± 3.3** | **0.892 ± 0.024** | **94.3 ± 1.1** | 13.4 ± 3.4 | 89.2 ± 4.2 |
| gate | ≥ 82.3 (Jev −3 pp); target ≥ 94 | ≤ 0.05 | ≥ 71.3 | — | ≥ 58.7 | ≤ 50 | — |

p95 in this table was measured while other bench jobs loaded the CPU. Idle
re-measurement (load < 3, nothing else running) for the claim seed: **p50 7.1
/ p95 9.4 ms**, and every other metric reproduced bit-for-bit (the engine is
deterministic); base bge-small under the same conditions: 7.1 / 9.4 ms. Base bge-small under the same configuration:
dept 84.0, ECE 0.0387, urgent 78.0 (AUROC 0.831), frustration 76.7.

**v0 → v1, seed 1 (validation):**

| gate | v0 as quoted (`--shots 8`, carve, 400 it.) | v1 (full data, calibration split, 5000 it.) |
|---|---|---|
| department | 95.3 % | 94.0 % |
| ECE | 0.169 FAIL | 0.041 PASS |
| urgent | 49.3 % FAIL (AUROC 0.525) | 86.7 % PASS (AUROC 0.884) |
| frustration | 50.0 % FAIL | 93.3 % PASS |
| p95 | 9.4 ms | 9.4 ms (seed 2, idle; seed 1 17.0 ms under load) |

### Claim checkpoint (ADR-008 §1d)

Rule, fixed before any further scoring: the median of the trainer's
pre-registered in-training selection metric `best_val_selection`
(independent of engine options). Values: s4 0.94870, s5 0.95542,
**s2 0.95614**, s3 0.95643, s1 0.95781 → **seed 2** (onnx sha256
`7bec3f8b…`). Nothing has been scored on test.

### Margins — read before quoting

- ECE: worst seed 0.0457. With 150 items the sampling noise of ECE is
  ~0.02, so "passes on validation" is all this shows; it is not a solved gate.
- Department on seed 1 dropped 95.3 → 94.0 with the converged probe (2
  items); every seed stays ≥ 94.0.

## Ablations — what failed or was not enough (5 seeds, validation)

| configuration | dept | ECE | urgent | frustration | verdict |
|---|---|---|---|---|---|
| `--shots 8` + calibration split + 400 it. | 95.2 ± 0.6 | 0.0371 ± 0.0127 | 71.9 ± 0.6 | 34.8 ± 4.6 | frustration FAIL all seeds; urgent sits on the majority rate |
| full data + calibration split + 400 it. | 95.3 ± 0.5 | 0.0474 ± 0.0088 | 74.9 ± 4.1 | 83.2 ± 0.7 | ECE FAIL on seed 1 (0.061); urgent FAIL on seed 4 (68.0) |
| full data + calibration split + 5000 it. (**v1**) | 95.1 ± 0.8 | 0.0406 ± 0.0038 | 86.7 ± 3.3 | 94.3 ± 1.1 | all gates PASS all seeds |

Seed-1-only probes along the way (validation):

- Calibration split alone (400 it.): ECE 0.0537 → **0.0611**. The fixture
  calibration slice scores 86.5 % vs 95.3 % on validation, so its temperature
  (0.53) over-softens: the top bin moves to 0.983 conf / 96.2 % acc, but 17
  items drop below 0.9 confidence while 15 of them are correct.
- Other `EngineOptions` (one-at-a-time, seed 1, calibration split):
  `probeL2` 0.01 → dept 91.3; `probeL2` 0.03 → dept 89.3 / ECE 0.066;
  `probeClassBalanced` → ECE 0.064; `logitScale` 5 → ECE 0.061; prototype
  head at scale 20 → frustration 29.3. None adopted.
- No trainer change was attempted: once the harness matched the plan, the
  v0 encoder already passed, so Task 2 (a model-level objective for
  urgency/frustration) was not needed. Changing it now would only fit
  validation harder.

## Public suites (Task 5) — validation

`run.mjs --suite <s> --no-test --shots 10000 --engine-options '{"probeIterations":5000}'`:
the engine trains on the suite's train rows minus the validation slice and is
scored on the trainer's validation slice (Banking77 958, CLINC150 2 986
in-scope + 100 OOS, HWU64 1 918). Base = bge-small-en-v1.5, same command.
Accuracy is the engine's own head, as ADR-008 §1c requires.

| suite | base bge-small | OpenJev s1 / s2 / s3 / s4 / s5 | OpenJev mean ± sd | Δ vs base |
|---|---|---|---|---|
| Banking77 acc | 79.96 % | 89.04 / 88.62 / 88.00 / 88.52 / 89.77 | **88.79 ± 0.66 %** | +8.8 pp |
| CLINC150 in-scope acc | 92.16 % | 96.89 / 96.58 / 96.55 / 97.09 / 96.62 | **96.74 ± 0.23 %** | +4.6 pp |
| CLINC150 OOS AUROC | 0.9510 | 0.9641 / 0.9627 / 0.9651 / 0.9705 / 0.9673 | **0.9659 ± 0.0031** | +0.015 (gate ≥ 0.85) |
| HWU64 acc | 55.89 % | 89.36 / 88.63 / 89.10 / 88.89 / 90.04 | **89.21 ± 0.54 %** | see note |

Read with these caveats:

- **Not comparable to the ADR-008 §1c literature targets** (Banking77 ≥ 92.06,
  CLINC150 ≥ 95.31 on *test*). These are validation numbers from a head that
  is still under-converged at 77–150 classes: base bge-small on Banking77 goes
  60.1 % → 80.0 % from 400 → 5000 iterations (CLINC150 91.0 → 92.2 %), so
  absolute numbers understate a converged linear probe. The base-vs-OpenJev
  comparison is apples-to-apples (same command, same budget).
- **HWU64 runs on the prototype head, not the probe.** `cooking_query` has 4
  train rows; the engine's every-5th calibration carve leaves 3 < 4
  (`MIN_EXAMPLES_PER_CLASS`), so `Auto` falls back to nearest-prototype for
  the whole question and the iteration budget is irrelevant (base identical at
  400 and 5000). That is why base is 55.9 % and ECE is ~0.43–0.47 for both
  arms; OpenJev's +33 pp there measures how much better its prototypes are,
  not a probe-vs-probe gap. HWU64 is reported, not claimed (ADR-008 §1c).
- CLINC150 OOS AUROC comes from the engine's abstain mass on the official
  `oos_val` (100 rows) vs in-scope `val`.
- Cost of this harness at full data: one public-suite run peaks at 12–38 GB
  RSS (`Engine::train` makes one `embed()` call for all new rows and
  `OrtEmbedder::embed` runs that as a single ORT batch — no chunking) and CLINC150
  at 5000 iterations takes ~27 min single-threaded. Running 15 at once was
  OOM-killed; the seeds were re-run 3 at a time. Worth fixing (chunked
  embedding in `train`) before the nightly sandbox (4 CPU / 16 GB) tries it.

## Engine / harness changes in this branch

- `ruvector-typesafe-core`: `Engine::train_with_calibration`; class/noul
  splits use an explicit `Calibration` split when present, else the carve
  (unchanged). Tests: `explicit_calibration_replaces_the_positional_carve`,
  `empty_calibration_is_bit_identical_to_plain_train`.
- `ruvector-typesafe-ffi` / `-wasm`: `trainJson` optional `calibration`.
- Semantic side effect: a bank JSON given to `importBankJson` that contains
  `Calibration`-split rows is now honoured by `decide` (those rows become
  the calibration slice); before, `decide` read only `Train` rows and
  ignored them. Banks written by the plain `train` path hold no such rows, so
  their behaviour is unchanged.
- bench: calibration split by default for tickets (`--no-calibration-split`
  to opt out); `--no-test` on public suites scores the exporter's validation
  slice (`lib/public-val.mjs`, row set verified identical to the trainer's
  `val.jsonl`: Banking77 958, CLINC150 3 086 incl. 100 OOS, HWU64 1 918);
  `bench/test/openjev-v1.test.mjs`.
- Pre-existing, unrelated: `test/api.test.mjs`, `test/cli.test.mjs`,
  `test/optimize.test.mjs` and the dist binding test fail without a built
  `dist/` (no TypeScript toolchain here); they fail identically on the
  branch base.

## Reproduce

```bash
# train (local GPU; data/cache from openjev prep/fetch)
RVGR_ARTIFACT_DIR=$PWD/runs/v1-art OPENJEV_BIN=target-cuda/release/openjev OPENJEV_WORK=$PWD/runs \
  bash scripts/openjev/vast-train.sh --seed 2 --seed 3 --seed 4 --seed 5
# native binding with the calibration API
(cd npm/packages/typesafe && bash scripts/build-native.sh --onnx)
# evaluate (validation only)
cd npm/packages/typesafe
node bench/run.mjs --suite tickets --arm local --embedder onnx \
  --model-dir ../../../runs/v1-art/seed-2 --model openjev-small-v0 \
  --shots 10000 --engine-options '{"probeIterations":5000}' --no-test
node bench/run.mjs --suite banking77 --arm local --embedder onnx \
  --model-dir ../../../runs/v1-art/seed-2 --model openjev-small-v0 \
  --shots 10000 --engine-options '{"probeIterations":5000}' --no-test
```

## Frozen test evaluation (one-time, 2026-09-26)

Claim checkpoint **seed 2** and the shipped configuration (`--shots 10000`,
`probeIterations 5000`, calibration on the fixture calibration slice) were fixed
on validation before the test split was scored once. Receipts:
`npm/packages/typesafe/bench/results/openjev/openjev-v1-seed2-{test,vsjev}-2026-09-26.*`.

| gate (test, n=150) | OpenJev v1 seed 2 | Jev baseline | status |
|---|---|---|---|
| department accuracy | 93.3% | 85.3% | PASS (≥ 82.3%) |
| ECE | 0.0543 | 0.0731 | **FAIL** (≤ 0.05) |
| urgent accuracy | 92.7% | 61.3% | PASS (≥ 71.3% train majority) |
| frustration accuracy | 86.0% | 67.3% | PASS (≥ 52.7% train majority) |
| p95 latency (native) | 10.1 ms | 230.7 ms (network) | PASS |
| leakage (Assertion B) | 0 / 41,748 | — | PASS |

Paired sequential test vs Jev (α=0.05, λ=0.5, lexical order):

- urgent: **superior** to both Jev references (56 wins / 9 losses vs baseline).
- frustration: **superior** to both (48 / 20).
- department: **non-inferior** to both (20 / 8 vs baseline, max wealth 14.05 < 20).
- Same tiers on the 119-item novel slice (department 91.6% vs 83.2%).

**Verdict against ADR-008 §1:** secondary claim (non-inferior to Jev champion on
all three) **holds**; primary claim (superior to Jev baseline on all three) is
**not met** (department not significant at n=150), and the calibration gate
fails by 0.0043. Per ADR-008 §6, v1 is **not published** to HF. The test split
has now been used for this claim; v2 must be judged on fresh held-out data
(see v2 plan), not by re-scoring this split.
