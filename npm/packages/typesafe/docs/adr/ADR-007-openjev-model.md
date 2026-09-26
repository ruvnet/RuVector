# ADR-007: OpenJev — an open-weight encoder that clears every frozen gate

## Status
Proposed (2026-09-26). Nothing here has been trained or measured yet; every
number below is either a measurement cited from ADR-006 receipts or a
pre-registered target. Implementation plan: `docs/research/openjev/v0-plan.md`.

## Date
2026-09-26

## Context

ADR-006 measured the local engine against Jev on the frozen tickets fixture.
The embedder arm is the lever with the most headroom left, and it has never
been anything other than an off-the-shelf checkpoint:

| gate (tickets, test n=150) | Jev baseline | Jev champion (gen 8) | best local so far | gate status |
|---|---|---|---|---|
| department accuracy | 85.3 % (128/150) | 90.0 % (135/150) | 84.0 % (bge-small-int8 campaign) | within −3 pp: PASS |
| calibration ECE (10 bins) | 0.073 | 0.068 | 0.057 (MiniLM) / 0.068 (bge-small) | ≤ 0.05: **never passed** |
| urgent accuracy | 61.3 % | 60.0 % | 28.7 %, AUROC 0.51 (16-shot bge) | ≥ train-majority 71.3 %: FAIL |
| frustration accuracy | 67.3 % | 67.3 % | 36.7 % (16-shot bge) | ≥ train-majority (label 0, 52.7 % on test): FAIL |
| native p95 | ~231 ms (network) | — | 10 ms FP32 / 4 ms INT8 | ≤ 50 ms: PASS |

Two facts from the receipts shape this ADR:

1. **Frozen general-purpose embeddings carry no urgency signal.** Urgent AUROC
   0.51 on validation and test with bge-small means no head over those vectors
   can learn it; no amount of probe or temperature tuning fixes that. The
   information has to be put into the encoder.
2. **Temperature scaling is already in the engine and still misses 0.05.**
   Every campaign promotion came through the NLL criterion (ADR-004 gate 2b)
   and the best ECE is 0.057. The residual over-confidence sits on the 21 %
   *ambiguous* tickets (Jev scores 56–62 % on that slice), where a one-hot
   target is itself the wrong target.

Two structural findings from reading the harness:

- **The campaign harness already supports a checkpoint arm.**
  `bench/optimize.mjs` builds one engine per `--models` name, resolved against
  `models/manifest.json` through `ruvector-typesafe-ffi` → `ruvector-embed-core`
  (`OrtEmbedder::from_manifest`, sha256-verified, CLS or mean pooling from the
  manifest, `token_type_ids` optional). `bench/run.mjs --embedder onnx --model
  <name>` does the same for the gated run. A fine-tuned encoder therefore
  needs **no engine change** if it is delivered as an ONNX graph with the same
  inputs/outputs as `bge-small-en-v1.5/model.onnx`. OpenJev v0 is "add a
  checkpoint arm".
- **The engine only loads ONNX** (`ort` 2.0.0-rc.13 native, `tract-onnx` 0.23
  WASM). The training constraint is Rust-only, and `candle` (0.9.2, already in
  the lockfile via `crates/ruvllm`) trains BERT-class models but has no ONNX
  exporter. Section *Export* resolves this.

## Decision

### 1. What "beyond SOTA" means here — pre-registered, not a point estimate

OpenJev v0 is accepted only when all of the following hold on one gated run,
with test scored **once** for the final checkpoint (ADR-004 gate 1).

**1a. Every `gates.json` gate PASSes** (strict mode, native ONNX, not
`--report-only`): `accuracy_vs_jev`, `calibration_ece` ≤ 0.05,
`urgent_vs_train_majority`, `frustration_vs_train_majority`, `oos_auroc` ≥ 0.85
(CLINC150), `latency_native_p95` ≤ 50 ms, and `transfer_regression` ≤ 1 pp.
The transfer baseline is **not** a 2026-09-21 receipt: those are 16-shot and
predate the corrected sampling protocol (`bench/README.md`), so comparing
against them would be a giveaway across protocols. The baseline is a fresh
`bge-small-en-v1.5` full-data run (`--shots 10000`, same host, same commit,
`--gate`) scored in the same campaign; it doubles as the arm-vs-arm
comparison. The tickets transfer split has 26 items, so 1 pp is 0.26 items: on
tickets this gate means "no additional transfer miss". Real transfer evidence
is cross-suite (§1c).

**1b. Paired comparison against Jev, both references, with tiers.** A point
estimate is not a win: on 150 items one ticket is 0.67 pp. We reuse the
campaign's anytime-valid paired test (`PairedSequentialTest`, α = 0.05,
λ = 0.5, threshold 1/α = 20, wealth factors 1.25/0.75 on discordant pairs)
over the 150 test rows, where Jev's per-item rows are `test_rows.baseline` and
`test_rows.champion` in the frozen baseline JSON. α and λ are fixed here and
never tuned. **Pair order is fixed: test ids sorted lexically** (the rejection
latches on the running maximum of wealth, so order matters; this also makes
rejection somewhat easier than the final-wealth arithmetic below implies, and
is why the order is pre-registered rather than chosen). Each reference gets one of three tiers per question
(department, urgent, frustration):

| tier | rule |
|---|---|
| **superior** | the e-process with OpenJev as challenger rejects (wealth ≥ 20) |
| **non-inferior** | the reversed e-process (Jev as challenger) does not reject **and** OpenJev's accuracy ≥ the reference's |
| **loses** | anything else |

Power arithmetic, stated so nobody over-reads the result: rejection needs
`0.223·wins − 0.288·losses ≥ ln 20 = 3.0`, i.e. ≥ 14 discordant wins with no
losses, ≥ 18 with 3 losses. Against Jev baseline (128/150) *superior* needs
roughly ≥ 143/150 (95 %). Against the champion (135/150) *superior* is
arithmetically out of reach on 150 items. The pre-registered claims are
therefore:

- **Primary** (`gates.json` `jev_reference: baseline`): *superior* to Jev
  baseline on department; *superior* on urgent and frustration (Jev is at
  61.3 %/67.3 %, so ≥ 14 net wins there is a realistic bar).
- **Secondary**: *non-inferior* to Jev champion on all three. The champion
  stays informational in `gates.json` (its criteria contain literal train
  texts, ADR-004); we report its tier because the claim invites it.
- **Novel-composition slice.** The generator is templated. Against the
  136-row tickets gradient pool (§2), 31 of 150 test items consist entirely of
  sentences that also occur in it and 141/150 share at least one (sentence =
  split on `[.!?:]+`, `norm` of §2, keep ≥ 3 tokens; measured 2026-09-26).
  A fine-tuned encoder can exploit that exactly as Jev's champion did. Every
  tier is also reported on the **119** test items *not* fully composed of pool
  sentences; if the primary claim holds on the full split but not on this
  slice, the ADR records it as "in-generator only".
- **Validation is selection-biased for OpenJev** (checkpoint, sweep and early
  stopping all use it; the off-the-shelf arms were never selected on it), so
  every cross-arm claim rests on the single test scoring, never on validation
  numbers — including `optimize.mjs`'s val-based arm ranking.

**1c. Public benchmarks against published SOTA, at our latency envelope.**
Jev was never measured on public data, so the reference is the literature.
Full-data test accuracy, published reference (BERT-base-class, ~110 M params):
Banking77 **94.06** (ConvBERT+MLM+Example), CLINC150 in-scope **97.31**,
HWU64 **93.03** (ConvBERT+MLM+Example+Observers; Mehri & Eric,
arXiv:2010.08684, DialoGLUE splits; ConvFiT arXiv:2109.10126 reports the same
band). Pre-registered target: **within N = 2.0 pp** (Banking77 ≥ 92.06,
CLINC150 ≥ 95.31) with a ~33 M-param encoder at native p95 ≤ 50 ms, scored by
the engine's own probe heads (not a dedicated fine-tuned classifier). HWU64 is
reported but **not claimed**: `bench/datasets/hwu64.mjs` uses its own
deterministic 80/20 split of the raw CSV, not the DialoGLUE split, so 93.03 is
not a comparable number until a DialoGLUE-split loader lands. CLINC150 OOS
AUROC ≥ 0.85 is the gate; no literature comparison is claimed for it.

**1d. Seeds.** A checkpoint evaluates deterministically (`run.mjs` has no
sampling seed), so variance comes from training. The claim is made for the
median-validation checkpoint of **5 training seeds**, and that checkpoint is
fixed *before* any test scoring. Each of the five is then scored on test once;
the four non-claim seeds are reported in the model card as variance only. This also satisfies the research-manifest's
`confirmation_seeds` (≥ 5) requirement (§7).

### 2. Leakage policy — train/val only, enforced by hard assertions

- **Gradient steps:** tickets `train` only (136 rows: the carved 137 minus the
  one `excludeHeldOutText` removes); public `train` splits minus a hash-carved
  10 % validation slice (CLINC150 uses its own `val`).
- **Checkpoint selection / early stopping only (never gradients):** tickets
  `validation` (150) and the public validation slices.
- **Never:** tickets `test`, `transfer`, `calibration`; Banking77 `test.csv`;
  CLINC150 `test` and `oos_test`; the HWU64 test bucket of `hwu64.mjs`.
  `calibration` stays reserved for the engine's temperature fit (§4) and
  `transfer` for gate 3.
- **Normalization** (`norm`): Unicode NFKC → lowercase → every run of
  non-alphanumeric code points becomes one space → trim. JS (bench) and Rust
  (trainer) implementations share a golden test-vector file.
- **Assertion A (trainer, before the first step):** compute
  `sha256(norm(text))` for every training and validation row; load the held-out
  hash set exported from the frozen fixtures and the pinned public test files;
  abort non-zero if the intersection is non-empty. Public datasets contain
  exact duplicates across their own train/test (HWU64 especially): such rows
  are **dropped from train** and counted, never kept.
- **Assertion B (bench, before scoring):** the published model ships
  `train-text-hashes.txt` (sorted `sha256(norm(text))`, one per training row).
  `run.mjs` intersects it with the suite's test hashes and refuses to score on
  any match. This is independent of the trainer: anyone can re-verify leakage
  from the artifact alone.
- Both assertions' outputs (`{train_rows, heldout_rows, intersection: 0,
  dropped_duplicates}` per dataset) go into the model card verbatim.
- The vocabulary guard (ADR-006 §4) and fixture hash checks run unchanged.

### 3. Architecture v0 — one lever

**Fine-tune one small encoder; change nothing else.**

- **Base: `BAAI/bge-small-en-v1.5`** (33 M params, 384-d, 12 layers, MIT,
  `model.safetensors` published). Chosen over bge-base (110 M, 768-d) on the
  latency gate: FP32 bge-small measures p95 10 ms on the bench host, and
  bge-base is ~3.3× the FLOPs (≈ 33 ms estimated) — inside 50 ms on a Ryzen
  9950X, but with no headroom on laptop and CI-runner CPUs, and the WASM gate
  (150 ms, ~192 ms/embed FP32 today for small) would fail outright. Keeping
  384-d also keeps every head, bank and RaBitQ index shape unchanged. bge-base
  is the v1 comparison arm if v0 misses the public-accuracy target.
- **Objective** (details and hyperparameters in the plan):
  - supervised contrastive loss over class-balanced batches (P classes × K
    examples), with each label's criteria/description text as an extra
    positive — matching how the engine's prototype head builds prototypes;
  - multi-task heads on the pooled, L2-normalized CLS vector: tickets
    `department` (8), `urgent` (binary, class-weighted), `frustration`
    (3-way), Banking77 (77), CLINC150 (150), HWU64 (64);
  - **soft targets on ambiguous tickets**: train items flagged `ambiguous`
    carry `secondary`; their department target is 0.75 primary / 0.25
    secondary. Train-only information, so protocol-clean; aimed directly at
    the ECE failure;
  - **outlier exposure** on CLINC150 `oos_train` (train split, allowed): a
    uniform-target term on the CLINC head, aimed at OOS AUROC;
  - regularization: weight decay, label smoothing 0.1, early stopping on
    validation. `candle-transformers`' BERT `Dropout` is a no-op in 0.9.2, so
    dropout is not available and is not claimed.
- **Heads are discarded.** OpenJev ships only the encoder. The engine still
  fits its own probe/logistic/temperature heads from train examples at bench
  time, exactly as for any other arm; the auxiliary heads only shape the
  embedding space. The Jev contract (criteria and examples per request) is
  therefore unchanged.
- **Pooling and inputs are frozen to the base:** CLS pooling, `input_ids`,
  `attention_mask`, `token_type_ids`, output `last_hidden_state`, tokenizer
  byte-identical to the pinned `bge-small-en-v1.5/tokenizer.json`.
- **Precision:** FP32 is the v0 deliverable (p95 10 ms already clears the
  gate). INT8 is a stretch entry (§5), not on the acceptance path.
- **Out of scope for v0:** distillation (from Jev, an LLM or a larger
  encoder), a second encoder or ensembling, CORN ordinal heads (ADR-003 v2),
  per-deployment SetFit, criteria mutation (Loop 4), any change to
  `ruvector-typesafe-core`.

### 4. Calibration

Temperature stays where the engine fits it: the **calibration** slice (37
tickets), one scalar per question. Fitting it on validation (150) was
considered and rejected: gate 2b's NLL paired test runs on validation, and a
temperature fitted there would make that test circular. The ECE lever in v0 is
the training objective (soft ambiguous targets, label smoothing, outlier
exposure), measured by the unchanged 10-bin ECE in `bench/lib/metrics.mjs`,
which is the sole arbiter. With 150 test items and several sparse bins, ECE
moves by ~0.01 per few items; the gate is applied as written, and the card
reports the reliability bins.

### 5. Export — Rust-only path from candle to the engine's ONNX

candle cannot write ONNX, so the fine-tuned weights are **transplanted into
the already-pinned base graph** (`models/bge-small-en-v1.5/model.onnx`, sha256
`828e1496…`), which is known to load in both `ort` and `tract` 0.23:

1. Decode the ONNX `ModelProto` with `tract_onnx::pb` (prost-generated,
   already an optional dependency of `ruvector-embed-core` under `wasm`).
2. Map each of its 197 initializers to a parameter of the base
   `model.safetensors`. The graph names 125 of them
   (`embeddings.*`, `encoder.layer.N.*.bias`, all LayerNorms); the 72 linear
   weights are anonymous (`onnx::MatMul_1525`…, stored transposed).
   **Primary mapping: value match** — each initializer must equal exactly one
   base safetensors tensor, directly or transposed, within 1e-6. **Cross-check:
   graph structure** — the consuming node's name
   (`/encoder/layer.0/attention/self/query/MatMul`) must agree. Any
   unmatched float initializer, ambiguous match, or disagreement aborts.
3. Replace each mapped initializer with the fine-tuned tensor (same shape,
   same transpose), leave the graph topology untouched, re-encode.
4. **Parity is the correctness proof:** on a 512-text probe set, `ort` on the
   transplanted graph vs the candle forward pass must agree to cosine
   ≥ 0.9999 per text, and the `department` decisions of an engine built on each
   must be identical. This is the ADR-002 §7 discipline applied to training.

**INT8 (stretch).** The pinned `model_quantized.onnx` is ORT dynamic
quantization: 72 `MatMulInteger` nodes whose weights are named
`<fp32 name>_quantized` / `_scale` / `_zero_point`. The same transplant
applies with the weights re-quantized in Rust per tensor, matching the
template's zero-point dtype. Accepted only if its own gated run passes; else
v0 ships FP32 only.

**Fallback (if transplant parity fails):** add a `CandleEmbedder` backend to
`ruvector-embed-core` behind a `candle` feature that loads safetensors
directly — ADR-002's stated v2 consolidation path — and gate it with the same
cosine/decision parity against `OrtEmbedder` on the base model before trusting
it on OpenJev. This is larger (a new backend, `typesafe-ffi` wiring, WASM
parity) and is only pursued if step 4 fails.

### 6. Deliverable

- **HF repo `ruvnet/openjev-small-v0`**: `onnx/model.onnx` (FP32; optional
  `onnx/model_quantized.onnx`), `model.safetensors` (fine-tuned encoder),
  `tokenizer.json` (byte-identical to the base), `config.json`,
  `train-text-hashes.txt`, `leakage-report.json`, `bench-receipt.json`
  (gated tickets run), `bench-receipt-public.json`, `vs-jev.json` (§1b tiers),
  `training-run.json` (config, seed, git sha, loss curves, hardware,
  wall-clock), `SHA256SUMS`, and `README.md` (model card) embedding the
  receipt tables and leakage-assertion output.
- **Manifest entry** in `models/manifest.json`: `openjev-small-v0` with
  `sha256`, `tokenizer_sha256`, `dims: 384`, `pooling: "cls"`,
  `max_tokens: 256`, `license: "MIT"`, `source_url` pointing at an **immutable
  HF revision** (`/resolve/<commit-sha>/…`, never `main`), `added`,
  `review_by`. `scripts/fetch-models.mjs` has a hard-coded `PLAN` map, so the
  same PR adds the `openjev-small-v0/*` entries there.
- Nothing is published to HF until §1 is met; failed runs keep their receipts
  in `bench/results/` and in the plan's results log.

### 7. Autonomous iteration via `research-nightly-dispatch`

`research-nightly-dispatch.yml` (03:17 UTC daily) dispatches
`research-candidate.yml` for every new head under `research/nightly/*` or
`research/candidate/*`. That workflow checks the manifest, runs the candidate's
`research/run-candidate.sh` inside a `--network none`, read-only, 4-CPU,
16 GB, no-GPU container with an 85-minute kill, evaluates `raw-results.json`
with the trusted research gate, and attests the evidence. OpenJev plugs in as:

- **Training happens outside CI** (GPU, budget-capped; plan §Cost). A nightly
  candidate never trains; it *evaluates* a candidate checkpoint whose weights
  are pinned by sha256 in the candidate's `research-manifest.json`.
- `research/run-candidate.sh` runs the §1 protocol (fixture hashes, both
  leakage assertions, `run.mjs --gate` for tickets and the public suites,
  `vs-jev`) and writes `raw-results.json`/`report.json`, one entry per
  confirmation seed (the 5 trained checkpoints). The ADR-006/§1 gates are the
  acceptance criteria; a candidate that fails any gate is not promotable.
- Iterations may change only hyperparameters, data mixes and manifest pins —
  **automated candidates may not change lockfiles** (`preflight_scan.py`), so
  every dependency change goes through a human PR.

**Prerequisites (blockers for the nightly path, not for v0 itself):**

1. The training crate and its dependencies land on `main` via a normal PR
   (lockfile change); its `research-manifest.json` sets
   `new_crate_justification`.
2. `ort`'s `download-binaries` fetches onnxruntime **at build time**, which
   cannot happen under `--network none`. The research sandbox image
   (`vars.RESEARCH_SANDBOX_IMAGE`, digest-pinned) must ship a pinned
   `libonnxruntime`, and the evaluation build must use `ort`'s `load-dynamic`
   feature — the alternative ADR-002 §2 already documents.
3. The FP32 graph is 133 MB — over GitHub's 100 MB per-file limit — and the
   public datasets are fetched from the network. `research-candidate.yml` needs
   a trusted hydration step (network on, before the offline container, like
   the existing `cargo fetch` step) that downloads model files and dataset files
   and verifies them against sha256 pins, failing closed on any mismatch.
4. The sandbox's 85 minutes on 4 CPUs must fit 5 checkpoints × (tickets + three
   public suites). If they do not, the nightly candidate uses tickets +
   `--limit`ed public slices, and the full public evaluation stays a manual,
   receipted release step.

### 8. Security and licensing

- **Supply chain:** safetensors and ONNX only — no pickle/`pytorch_model.bin`
  is ever downloaded or produced (the trainer refuses `.bin`/`.pt`/`.pkl`).
  Every input (base safetensors, base ONNX template, tokenizer, datasets) and
  every output is sha256-pinned; the manifest loader fails closed (ADR-005).
  HF downloads use immutable revision SHAs. `SHA256SUMS` and the manifest are
  cosign-signed at release (ADR-005). The transplant keeps the *pinned*
  template topology, so no new operators enter the runtime.
- **Licenses:** base model MIT (BAAI/bge-small-en-v1.5 card) → OpenJev weights
  MIT. Banking77 CC-BY-4.0 (PolyAI repo), HWU64 CC-BY-4.0
  (xliuhw/NLU-Evaluation-Data), CLINC150 CC-BY-3.0 (clinc/oos-eval LICENSE);
  the synthetic tickets are this repo's (MIT). CC-BY requires attribution, not
  share-alike: the card attributes all three datasets and links their
  licenses. ADR-006 described Banking77/HWU64 as "research use"; the upstream
  repositories declare CC-BY-4.0, and the fetchers keep downloading from source
  (nothing redistributed except text *hashes*).
- **Naming:** "Jev" is typesafe.ai's product name. The card states that
  OpenJev is an independent, API-compatible reimplementation, not affiliated
  with or endorsed by typesafe.ai; if the name is contested, the HF repo and
  manifest entry are renamed without changing hashes.
- **Data egress / PII:** unchanged — the engine still downloads nothing at
  runtime; the training corpus is public or synthetic; no Jev API responses
  are used as training targets (that would be distillation, and is out of
  scope).

## Consequences

- One new crate and a data-export script; zero changes to the decision engine,
  so every existing parity, security and gate test still applies.
- The transplant ties OpenJev to the base graph's topology. Any architecture
  change (bge-base, a different pooling) needs a new pinned template, not just
  new weights.
- The tiered claim is weaker than "beats Jev" in a headline and stronger in
  substance; a champion-superiority claim is not available on this fixture and
  the ADR says so up front.
- Fine-tuning on a templated synthetic generator improves in-generator
  numbers more than real-world ones; the novel-composition slice and the public
  suites are the counterweight, and both are reported with equal prominence.
- The nightly loop needs three workflow/sandbox changes (§7) before it can
  evaluate an ONNX arm at all.

## Alternatives considered

- **Train in candle, serve through a new candle backend.** Clean, but a second
  runtime to keep in parity with `ort` and `tract`; kept as the fallback in §5.
- **Train in Python (sentence-transformers) and export with optimum.** The
  shortest path, but excluded by the project rule (Rust only).
- **bge-base as the v0 base.** Better public accuracy ceiling, worse latency
  margin and a WASM-gate failure; deferred to a v1 comparison arm.
- **Distill from Jev's outputs.** Would import Jev's saturated confidence and
  its 61 % urgency behaviour, costs API money, and muddies what "open" means.
- **Temperature on validation.** Circular with gate 2b (§4).

## Evidence

- ADR-006 receipts: `bench/results/tickets-onnx-bge-2026-09-21.json`,
  `optimize-tickets-2026-09-21.json`, `optimize-receipts-2026-09-21.jsonl`;
  Jev rows: `bench/jev-baseline-2026-09-21.json` (`test_rows.baseline`,
  `test_rows.champion`, 150 each).
- Harness: `bench/optimize.mjs` (model arm = manifest name),
  `crates/ruvector-typesafe-ffi/src/lib.rs` (`build_onnx`),
  `crates/ruvector-embed-core/src/ort_backend.rs`,
  `crates/ruvector-typesafe-core/src/loop_gate/sequential.rs`.
- ONNX template inventory (2026-09-26): 125 named + 72 `onnx::MatMul_*`
  initializers in `bge-small-en-v1.5/model.onnx`; 72 `MatMulInteger` /
  48 `DynamicQuantizeLinear` in `model_quantized.onnx`.
- Published SOTA: Mehri & Eric, "Example-Driven Intent Prediction with
  Observers", arXiv:2010.08684 (Table 1); DialoGLUE arXiv:2009.13570;
  ConvFiT arXiv:2109.10126.
- `candle-transformers` 0.9.2 `models/bert.rs` (`Dropout::forward` is identity).
- Workflows: `.github/workflows/research-nightly-dispatch.yml`,
  `research-candidate.yml`, `scripts/research-gate/preflight_scan.py`,
  `schemas/research-manifest-v1.json`.
