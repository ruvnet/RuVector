# OpenJev v0 — implementation plan

Design and acceptance criteria: `npm/packages/typesafe/docs/adr/ADR-007-openjev-model.md`.
This plan is the how: files, commands, hyperparameters, compute and cost.
Everything is Rust except thin additions to the existing JS bench harness
(`npm/packages/typesafe/bench/`), which already owns dataset loading and gates.
No Python anywhere.

Paths below are relative to the repo root; `TS=npm/packages/typesafe`.

---

## Step 0 — toolchain smoke tests (no spend, no GPU launch)

| # | Check | Command | Pass condition |
|---|---|---|---|
| 0.1 | candle builds for CUDA on ruvultra's RTX 5080 (Blackwell, sm_120, CUDA 13.0) | `cargo build -p ruvector-typesafe-train --release --features cuda` | compiles; this is a compile, not a GPU job |
| 0.2 | the same crate builds CPU-only | `cargo build -p ruvector-typesafe-train --release` | compiles |
| 0.3 | transplant identity round trip (see Step 3) | `openjev transplant --identity …` then `openjev parity` | cosine = 1.0 − ε (≥ 0.999999) on 512 probes |
| 0.4 | 50-step CPU dry run on a 2 % data slice | `openjev train --config … --max-steps 50 --device cpu` | loss decreases; checkpoint + ONNX + parity written |

If 0.1 fails (candle 0.9 / cudarc 0.19 kernel support for sm_120), training
runs on vast.ai (Step 6). If 0.1 passes, the whole v0 campaign runs locally
for $0 and vast.ai is only used to run seeds in parallel.

---

## Step 1 — harness additions (JS, `TS/bench/`)

These go in one PR ahead of any training, with tests in `TS/bench/test/`.

1. **`bench/lib/norm.mjs`** — `norm(text)`: NFKC → lowercase → every run of
   non-alphanumeric code points (`/[^\p{L}\p{N}]+/gu`) → one space → trim.
   `sha256Norm(text)`. Golden vectors in `bench/test/norm-golden.json`
   (≥ 40 cases: accents, full-width digits, emoji, `can't`, tabs, CJK); the
   Rust `norm` must pass the same file.
2. **`bench/export-openjev-data.mjs`** — reuses `lib/fixture.mjs` and
   `datasets/*.mjs` (so splits cannot diverge from the bench) and writes to
   `bench/.cache/openjev-v0/` (gitignored):
   - `train.jsonl` / `val.jsonl`, rows
     `{dataset, id, text, label, soft?: {label: w}, urgent?, frustration?, oos?}`;
     tickets train = `excludeHeldOutText(bySplit).trainItems` (136 rows —
     the only ticket rows that ever produce gradients), tickets val =
     `validation` (150, selection/early stopping only); Banking77 val = 10 % of train by
     `sha256(id) % 10 == 0`; CLINC150 val = official `val` (+ `oos_val`);
     HWU64 val = 10 % of the loader's train bucket by the same rule;
     CLINC150 `oos_train` rows go to train with `oos: true`.
   - `heldout-hashes.txt` — `sha256Norm` of tickets test/transfer/calibration,
     Banking77 test, CLINC150 test + oos_test, HWU64 test bucket.
   - `labels.json` — label → description text (tickets: gen-0 criteria;
     public: the loaders' `label → description` maps).
   - `data-manifest.json` — row counts, split hashes, the upstream sha256 pins,
     and `sha256` of every file above.
   - Drops any train/val row whose `sha256Norm` is in `heldout-hashes.txt`
     and reports the count per dataset (HWU64 and CLINC150 contain exact
     cross-split duplicates).
3. **`bench/lib/leakage.mjs`** + `run.mjs --train-hashes PATH` — Assertion B
   (ADR-007 §2): intersect the model's `train-text-hashes.txt` with the suite's
   test hashes; refuse to score on any hit; record `{train, test, intersection}`
   in the receipt. `--model openjev-*` makes the flag mandatory.
4. **`run.mjs --no-test`** — score train-side splits only (validation,
   transfer). Used for every selection run; test is scored once, in Step 5.
   **`optimize.mjs` gets the same `--no-test` plus `--model-dir`/`--manifest`**
   (it hardcodes `modelDir: 'models'` and always prints test today), so the
   protocol is enforced by flags, not by "don't look".
5. **`run.mjs --emit-records PATH`** — per-item test records
   `{id, dept_pred, dept_correct, urgent_correct, frustration_correct, confidence}`.
6. **`bench/lib/paired.mjs`** — JS port of `PairedSequentialTest`
   (`crates/ruvector-typesafe-core/src/loop_gate/sequential.rs`), α = 0.05,
   λ = 0.5, with a shared test vector proving identical wealth paths. Pairs are
   fed in **lexically sorted test-id order** (pre-registered, ADR-007 §1b).
7. **`bench/vs-jev.mjs --records PATH [--out PATH]`** — ADR-007 §1b: for each of
   department / urgent / frustration and each reference (`test_rows.baseline`,
   `test_rows.champion`) runs the forward and reversed e-process over the 150
   test ids, assigns *superior / non-inferior / loses*, and repeats on the
   **novel-composition slice**: test items not fully composed of sentences
   from the 136-row tickets gradient pool; sentence = split on `[.!?:]+`,
   `norm`, keep ≥ 3 tokens (`bench/lib/novel-slice.mjs`, computed from the
   pool only; 119 items at the 2026-09-26 freeze, written into the output).
8. **`scripts/fetch-models.mjs`** — `PLAN` entries for `openjev-small-v0/*`
   pointing at `ruvnet/openjev-small-v0/resolve/<hf-commit-sha>/…`.

Tests: norm golden vectors; leakage refuses a planted collision; paired.mjs
matches the Rust wealth path; vs-jev tiers on synthetic rows (all-win,
all-tie, 3 losses + 18 wins); `--no-test` receipt contains no test block.

---

## Step 2 — new crate `crates/ruvector-typesafe-train`

Binary `openjev`. Every file < 500 lines. Workspace member; not a dependency
of any shipped crate (training code never enters the npm package).

```
crates/ruvector-typesafe-train/
  Cargo.toml
  configs/openjev-small-v0.toml      # the v0 hyperparameters below
  src/main.rs                         # clap CLI: prep-check | train | transplant | parity | quantize | card
  src/norm.rs                         # norm + sha256Norm; golden-vector test against TS/bench/test/norm-golden.json
  src/data.rs                         # JSONL rows, per-dataset P×K class-balanced sampler, seeded (rand_chacha)
  src/leakage.rs                      # Assertion A: abort if any train/val hash ∈ heldout-hashes.txt
  src/model.rs                        # candle BertModel (VarMap "enc") + cosine heads (VarMap "heads")
  src/loss.rs                         # SupCon, soft cross-entropy + label smoothing, outlier exposure
  src/train.rs                        # loop, 2×AdamW, warmup/linear decay, grad-clip, eval, early stop
  src/checkpoint.rs                   # encoder-only safetensors with the base's tensor names
  src/export/transplant.rs            # ONNX initializer transplant (ADR-007 §5)
  src/export/quantize.rs              # INT8 stretch: re-quantize into model_quantized.onnx template
  src/parity.rs                       # ort (via ruvector-embed-core/native) vs candle, cosine + decisions
  src/card.rs                         # model card README.md from receipts
  tests/{norm.rs,sampler.rs,leakage.rs,transplant_identity.rs,loss.rs}
```

Dependencies (all already in `Cargo.lock` except where noted; a lockfile
change regardless, so this is a human PR):
`candle-core`/`candle-nn`/`candle-transformers` `=0.9.2` (feature `cuda` →
`candle-*/cuda`), `tokenizers 0.20` (`onig`), `safetensors` (candle's version),
`tract-onnx 0.23` (only for `tract_onnx::pb` — the prost-generated ONNX
protobuf types) + the matching `prost` for `Message::encode/decode`,
`ruvector-embed-core` with `native` (parity via `OrtEmbedder`),
`ruvector-typesafe-core` (engine decisions for parity), `sha2`, `serde`,
`serde_json`, `clap 4.5`, `rand 0.8` + `rand_chacha`, `unicode-normalization`
(check lock; add if absent), `toml`.

Supply-chain rules in code: the trainer loads only `.safetensors`, `.onnx`,
`tokenizer.json`, `.jsonl`; any other extension is a hard error. Every input is
checked against a sha256 in the run config before use.

### CLI

```bash
# Inputs pinned in the config: base safetensors, base ONNX template, tokenizer, data-manifest.
openjev prep-check --config crates/ruvector-typesafe-train/configs/openjev-small-v0.toml
#   verifies all input sha256s, runs Assertion A, reports token-length p50/p99,
#   writes leakage-report.json. Exit 3 on any leakage.

openjev train --config …/openjev-small-v0.toml --seed 1 --device cuda:0 --out runs/v0-s1
#   runs prep-check first (cannot be skipped), trains, keeps the best-val
#   checkpoint: runs/v0-s1/encoder.safetensors, heads.safetensors (not shipped),
#   training-run.json, curves.jsonl, train-text-hashes.txt.

openjev transplant --template $TS/models/bge-small-en-v1.5/model.onnx \
  --base-safetensors <BAAI model.safetensors> --weights runs/v0-s1/encoder.safetensors \
  --out runs/v0-s1/onnx/model.onnx --report runs/v0-s1/transplant-report.json
openjev transplant --identity …   # base weights into template: Step 0.3

openjev parity --onnx runs/v0-s1/onnx/model.onnx --weights runs/v0-s1/encoder.safetensors \
  --tokenizer $TS/models/bge-small-en-v1.5/tokenizer.json --probes 512 --out runs/v0-s1/parity.json
#   cosine ≥ 0.9999 per probe, identical department decisions; exit non-zero otherwise.

openjev quantize --template $TS/models/bge-small-en-v1.5/model_quantized.onnx \
  --fp32 runs/v0-s1/onnx/model.onnx --out runs/v0-s1/onnx/model_quantized.onnx   # stretch

openjev card --run runs/v0-s1 --receipts <dir> --out runs/v0-s1/README.md
```

### Transplant mechanics (ADR-007 §5)

1. Decode template `ModelProto`; collect float initializers (197 expected:
   125 named, 72 `onnx::MatMul_*`); int64 shape constants (e.g.
   `onnx::Slice_214`) are left untouched and listed in the report.
2. Load base safetensors (197 encoder tensors; the pooler, absent from the
   graph, is ignored). For each initializer find base tensors of equal
   element count; accept iff exactly one matches `max|a−b| ≤ 1e-6` directly
   or after transpose. Record `(initializer, tensor, transposed)`.
3. Cross-check: for anonymous initializers, the consuming `MatMul` node name
   must contain the tensor's module path (`encoder.layer.0.attention.self.query`
   ↔ `/encoder/layer.0/attention/self/query/MatMul`). Mismatch → abort.
4. Require a bijection over all 197; write fine-tuned tensors (transposed
   where recorded) into `raw_data`, re-encode, sha256 the output.
5. The identity transplant (base → template) must reproduce the template's
   embeddings exactly; that test runs in CI on every change to the crate.

INT8 stretch: for each of the 72 mapped `MatMul` weights, write
`<name>_quantized` / `_scale` / `_zero_point` using ORT's dynamic per-tensor
scheme, matching the template's zero-point dtype (read from the graph, not
assumed). Verify by re-quantizing the *base* weights and comparing with the
template's stored values (must match exactly or within one quantization step),
then parity vs FP32 at cosine ≥ 0.99 plus its own gated bench run.

---

## Step 3 — v0 hyperparameters (`configs/openjev-small-v0.toml`)

| group | value |
|---|---|
| base | `BAAI/bge-small-en-v1.5` `model.safetensors` at a pinned HF commit; CLS pooling; L2-normalized |
| max_seq_len | 64 (prep-check aborts if tickets/public p99 > 60 tokens; fallback 128) |
| precision | FP32 (no AMP in v0) |
| steps | 3,000 max; eval every 250; early stop after 4 evals without val improvement |
| batch | one dataset per step, sampled by mix weight; P labels × K examples + the P label-description texts as extra positives |
| batch shapes | tickets P=8 (all depts) × K=4; Banking77 / CLINC150 / HWU64 P=32 × K=2; CLINC150 steps add 16 `oos_train` rows |
| mix weights | tickets 0.20, Banking77 0.25, CLINC150 0.30, HWU64 0.25 |
| optimizer | AdamW β=(0.9, 0.999), ε=1e-8; encoder lr 3e-5, heads lr 1e-3; wd 0.01 (none on biases/LayerNorm) |
| schedule | linear warmup 180 steps (6 %), linear decay to 0 |
| grad clip | global norm 1.0 (implemented over candle `GradStore`; candle has no built-in clip) |
| SupCon | temperature τ = 0.05, same-dataset in-batch negatives |
| heads | cosine classifiers, logits = 20·cos(e, w_c) |
| classification loss | CE, label smoothing 0.1; ambiguous tickets use soft target 0.75 primary / 0.25 `secondary` (no extra smoothing) |
| urgent | 2-way head, positive-class weight = n_neg/n_pos computed from the exported `train.jsonl` at `prep-check` (104/32 ≈ 3.25 at the 2026-09-26 freeze) |
| frustration | 3-way head, inverse-frequency weights from the exported `train.jsonl` (82/39/15 at the freeze), normalized to mean 1 |
| outlier exposure | CLINC150 `oos_train`: CE to uniform over the 150 heads, weight 0.5 |
| total loss | SupCon + head CE (+ urgent + frustration on ticket steps) + 0.5·OE (CLINC steps) |
| dropout | none (candle 0.9.2 BERT dropout is a no-op) |
| in-training val metric | mean of: tickets dept acc, urgent AUROC, frustration acc (heads on tickets val); head acc on each public val slice |
| seeds | 1–5 (data order + head init) |

**Sweep (seed 1 only, 5 configs):** encoder lr ∈ {3e-5, 5e-5} × ticket mix ∈
{0.20, 0.35}, plus a tickets-only ablation (no public data). The winning
config is chosen by the **engine's validation metrics** from
`run.mjs --no-test` (department acc, ECE, urgent/frustration acc on
validation; ties → lower validation ECE) — never by test, never by head
metrics alone. Then seeds 2–5 on the winner.

---

## Step 4 — evaluation per checkpoint (CPU, ruvultra; selection only)

Latency gates are CPU gates — evaluate on the CPU box, never on the GPU host.

```bash
cd $TS
# Stage the candidate outside the committed manifest: runs/v0-sN/models/ holds
# openjev-small-v0/{model.onnx,tokenizer.json} and its own manifest.json with the
# sha256s written by `openjev transplant` (the committed models/manifest.json is untouched).
M=runs/v0-sN/models

# Selection runs: validation + transfer only.
node bench/run.mjs --suite tickets --arm local --embedder onnx --model-dir $M --model openjev-small-v0 \
  --shots 10000 --no-test --train-hashes runs/v0-sN/train-text-hashes.txt --out bench/results/openjev-v0-sN-val.json
node bench/optimize.mjs --models openjev-small-v0 --model-dir $M --no-test --budget 64 \
  --out results/optimize-openjev-v0-sN.json      # campaign (department) on the new arm
```

`--shots 10000` means "all train examples per class" (the harness caps at
what exists). Validation numbers are selection-biased for OpenJev and are
never used for cross-arm claims (ADR-007 §1b).

---

## Step 5 — the one gated test run (final checkpoint only)

Order of operations (ADR-007 §1d): the claim checkpoint (median validation of
the 5 seeds) is fixed and recorded in the results log **before** any command
below runs. The transfer baseline is a fresh full-data bge-small run on the
same host and commit — not a 2026-09-21 receipt (16-shot, pre-correction
protocol).

```bash
cd $TS
node bench/run.mjs --suite tickets --arm local --embedder onnx --model bge-small-en-v1.5 \
  --shots 10000 --gate --report-only --emit-records bench/results/bge-small-full-test-records.json \
  --out bench/results/bge-small-full-tickets.json            # baseline arm, same campaign
node bench/run.mjs --suite tickets --arm both --embedder onnx --model openjev-small-v0 \
  --shots 10000 --gate --baseline-receipt bench/results/bge-small-full-tickets.json \
  --train-hashes models/openjev-small-v0/train-text-hashes.txt \
  --emit-records bench/results/openjev-v0-test-records.json --out bench/results/openjev-v0-tickets.json
node bench/vs-jev.mjs --records bench/results/openjev-v0-test-records.json --out bench/results/openjev-v0-vs-jev.json
for s in banking77 clinc150 hwu64; do
  node bench/run.mjs --suite $s --arm local --embedder onnx --model openjev-small-v0 \
    --shots 10000 --gate --train-hashes models/openjev-small-v0/train-text-hashes.txt \
    --out bench/results/openjev-v0-$s.json
done
```

Accept per ADR-007 §1: all gates PASS strict; vs-jev primary claim met;
Banking77 ≥ 92.06 %, CLINC150 ≥ 95.31 %; model card reports all 5 seeds and
the novel-composition slice. If anything fails: record it, do not publish,
do not re-run test with a different checkpoint in the same campaign.

---

## Step 6 — compute, runtime and cost

### Preferred: local RTX 5080 (16 GB) on ruvultra — $0

Conditional on Step 0.1. bge-small at seq 64, batch ≤ 112 texts, FP32 with
AdamW state is < 3 GB of VRAM.

### Fallback / parallel seeds: vast.ai

- **GPU:** 1× RTX 4090 (24 GB) or 1× RTX A6000 (48 GB). 24 GB is far more
  than needed; pick the cheaper verified host with ≥ 8 CPU cores (tokenization
  and the candle build are CPU-bound) and CUDA ≥ 12.4 drivers.
- **Price assumption (verify at booking, marketplace rates move):** RTX 4090
  ~$0.30–0.60/h, A6000 ~$0.40–0.80/h on-demand.
- **Throughput assumption (not measured):** candle FP32 forward+backward for a
  33 M-param BERT at seq 64 on a 4090: 300–1,000 sequences/s (PyTorch AMP
  would be several times faster; candle has no fused attention or AMP here).
- **Per run:** 3,000 steps × ~100 texts ≈ 300 k sequence passes → 5–17 min of
  training; + 12 evals, transplant, parity ≈ 10 min → **15–30 min per run**.
- **Campaign:** 5 sweep runs + 4 seeds = 9 runs ≈ 2.5–4.5 GPU-h, + candle CUDA
  build (15–25 min), data export/upload and idle margin ≈ 1.5 h →
  **4–6 h wall on one GPU**.
- **Cost estimate:** 4–6 h × $0.30–0.80/h ≈ **$1.50–5**. Running seeds on
  2–3 instances in parallel changes wall-clock, not the total.
- **Hard budget cap: $20** for v0 — load at most $20 of vast.ai credit,
  create instances with an explicit auto-destroy after 10 h, and abort the
  campaign (not the cap) if run 1 exceeds 60 min wall-clock: that means the
  throughput assumption is off by > 3× and should be profiled locally first.
- CPU-only training on the 9950X is possible as a last resort (~4.7 PFLOP per
  run at an assumed ~300 GFLOP/s effective ≈ 4–5 h per run), $0 but slow.

### Instance procedure (vast.ai)

1. Image: an official `nvidia/cuda:12.x-devel` digest-pinned tag + `rustup`
   (stable pinned by `rust-toolchain.toml`). No HF write token, GitHub token or
   GCP credentials go onto the instance.
2. `git clone` at the exact commit; `cargo build -p ruvector-typesafe-train --release --features cuda`.
3. `rsync` in `bench/.cache/openjev-v0/` (exported on ruvultra) + the pinned
   base files; `openjev prep-check` verifies every sha256 before training.
4. Run the sweep/seeds; `rsync` artifacts back after each run (below), so an
   instance loss costs at most one run.
5. Destroy the instance; verify `SHA256SUMS` on ruvultra before using anything.

### Artifacts to sync back per run (`runs/v0-sN/`)

| file | purpose |
|---|---|
| `encoder.safetensors` | fine-tuned encoder (base tensor names) |
| `onnx/model.onnx` (+ `model_quantized.onnx` if stretch) | engine-loadable graph |
| `transplant-report.json` | 197-initializer mapping, transposes, output sha256 |
| `parity.json` | cosine distribution, decision identity |
| `training-run.json` | config, seed, git sha, host GPU, wall-clock, best step |
| `curves.jsonl` | per-step losses, per-eval metrics |
| `leakage-report.json` | Assertion A output per dataset |
| `train-text-hashes.txt` | Assertion B input, shipped with the model |
| `SHA256SUMS` | over all of the above |

`heads.safetensors` stays on disk for debugging and is never published.

---

## Step 7 — publish (only after Step 5 passes)

1. `huggingface-cli`-free upload: `git lfs` push to `ruvnet/openjev-small-v0`
   from ruvultra (token stays local), files per ADR-007 §6; model card
   generated by `openjev card` with the gated receipts, `vs-jev.json` tiers,
   all five seed results, the novel-composition slice, leakage output,
   dataset attributions (CC-BY-4.0 Banking77, HWU64; CC-BY-3.0 CLINC150),
   MIT license, and the non-affiliation statement.
2. Record the HF commit sha; replace the local-only manifest entry with the
   pinned `/resolve/<sha>/` entry + `PLAN` entries; `fetch-models.mjs` from a
   clean checkout must reproduce the hashes.
3. PR: manifest + `PLAN` + receipts under `bench/results/` + ADR-007 status →
   "Implemented and measured".

---

## Step 8 — nightly loop wiring (after v0; ADR-007 §7)

Human PRs, in order:

1. Sandbox image with pinned `libonnxruntime`; `ruvector-embed-core` gains an
   `ort/load-dynamic` feature path; evaluation build uses it.
2. Trusted hydration step in `research-candidate.yml`: reads the model and
   dataset pins from the candidate manifest, downloads with network on, verifies
   sha256, mounts read-only into the offline container.
3. `research/run-candidate.sh` template for OpenJev candidates: fixture hash
   check → Assertion B → Step 5 commands (with `--limit` on public suites if the
   85-min / 4-CPU envelope requires it) → `raw-results.json` with one entry per
   confirmation seed → `report.json`.

Nightly candidates then change only config values and manifest pins on a
`research/nightly/openjev-*` branch; training for them runs on the same budget
rules as Step 6, triggered manually.

---

## Risks and open questions

- **candle on sm_120** (Step 0.1) — decides local vs vast.ai.
- **candle BERT training speed** is the least certain number in the cost
  estimate; the 60-minute abort rule bounds the damage.
- **Transplant bijection** — if Xenova's graph folded any constant
  differently from BAAI's safetensors, value matching fails loudly; fallback is
  a `CandleEmbedder` backend (ADR-007 §5).
- **Tickets are tiny** (136 train rows vs ~45 k public); the 0.20/0.35 mix
  sweep and the tickets-only ablation exist to detect over- or under-fitting.
- **Superiority vs the Jev champion is not reachable on 150 items**
  (ADR-007 §1b); non-inferiority is the pre-registered claim.
