# `@ruvector/typesafe` benchmark harness

Measures the local engine against Jev (typesafe.ai) on the same terms
(ADR-006). Publishing is gated by these numbers, not by "tests pass".

## Run

```bash
node bench/run.mjs --suite tickets --arm both            # jev replay + local
node bench/run.mjs --suite tickets --arm jev             # Jev's frozen scorecard
node bench/run.mjs --suite banking77 --limit 1000 --arm local
node bench/run.mjs --suite tickets --arm both --gate --report-only
```

Flags: `--suite tickets|banking77|clinc150|hwu64|all` · `--arm jev|local|both` ·
`--embedder hash|onnx` · `--shots N` (default 8) · `--zero-shot` ·
`--limit N` · `--out PATH` · `--gate` · `--report-only` ·
`--baseline-receipt PATH`. One receipt JSON is written per run to
`bench/results/` (gitignored).

## Arms

- **jev** — replays `jev-baseline-2026-09-21.json`. **No network.** Only the
  `test` split has per-item rows. The headline is Jev's **gen-0 (baseline)**
  number: its `test_rows.baseline` were scored with the same gen-0 criteria the
  local arm sends, so the comparison is apples-to-apples. Jev's champion used
  mutated criteria whose examples are literal ticket texts (memorisation,
  ADR-004) and is reported as informational only. Jev exposes no `noul`
  probability, so urgent AUROC/Brier are not defined for the jev arm.
- **local** — the ruvector binding (`../index.js`: `Engine`, `version`,
  `backend`). Builds one request per item and times `decideJson`. Supports
  SetFit-style few-shot (train on the train split, N/class **per question**)
  and zero-shot, reported separately. The tickets arm trains `department`,
  `urgent`, and `frustration`, mapping the ordinal labels to the exact legend
  strings and binary labels to `yes`/`no`. When the core is not built the arm reports
  "engine unavailable" rather than crashing.

## Guards (the harness refuses to run on failure)

- **Fixture hashes** — every frozen file is checked against
  `fixtures/HASHES.json` (`scripts/hash-fixtures.mjs`). A drifted fixture is
  not the fixture Jev was measured against.
- **Split disjointness** — train / calibration / validation / transfer / test
  are assigned deterministically from item ids and asserted instance-disjoint
  (`assertDisjoint`, ADR-004). For the single-dataset tickets suite the
  calibration and transfer splits are deterministic holdouts carved from the
  native train split; true cross-domain transfer uses a different `--suite`.
  One frozen train row has text identical to a held-out row, despite distinct
  IDs. That train row is excluded before any head samples examples; receipts
  record `training.excludedHeldOutTexts`. Existing 2026-09-21 receipts include
  the old sampling and are not a directly comparable baseline for this corrected
  protocol. Frozen fixtures and Jev/local test IDs are unchanged.
- **Vocabulary disjointness** — generator content words (the retrieval corpus's
  `topics`) must not leak into the gen-0 decision criteria (ADR-006 §4). The one
  reviewed benign overlap — "software", shared between a corpus topic and the
  plain-English `technical` criterion, from two unrelated generators — is
  recorded in `ACCEPTED_OVERLAPS`, so the guard fires on new leaks, not this
  one. A planted corpus-topic word in a criterion is caught.
- **CLINC150 out-of-scope scoring** — both in-scope (`oos: false`) and OOS
  (`oos: true`) rows contribute to abstain AUROC. Intent accuracy, macro-F1,
  ECE, and Brier exclude OOS rows, which have no valid intent label. A declared
  OOS suite with only one class fails its OOS gate.
- **Secondary-head baselines** — the constant urgency and frustration labels
  are selected by frequency in the filtered train pool only, with deterministic
  tie-breaking. Their held-out test accuracy is reported next to each head and
  each head must at least match it. The older `urgent_majority_rate` uses test
  labels and remains informational; it is never used to set a threshold.

## Gates (`gates.json`, ADR-006)

`--gate` evaluates the measurable release gates and prints a PASS/FAIL/SKIP
table; strict mode exits non-zero on any FAIL. `--report-only` keeps the exit
code 0 for advisory PR and push measurements against the native ONNX binary;
manual publishing requires strict mode and a separately validated receipt.
The hash embedder remains a test double and cannot satisfy the ONNX release
gate. A gate whose precondition does not apply to the run (OOS AUROC on tickets;
transfer regression with no `--baseline-receipt`) is SKIP, never a silent pass.

## Datasets

`datasets/{banking77,clinc150,hwu64}.mjs` download from the canonical public
sources into `bench/.cache/` (gitignored), verify a pinned sha256, and convert
to `{id, text, label}` plus a `label → description` criteria map. CLINC150's OOS
slice is the abstain set for the OOS AUROC gate. Nothing is redistributed in the
package; an unreachable source fails loudly with the URL and the suite is marked
`skipped: unavailable`.

## OpenJev harness (ADR-007, plan Step 1)

Interfaces shared with the Rust trainer (`crates/ruvector-typesafe-train`).
The golden files are the contract; change them only together.

- **`norm` / `sha256Norm`** (`lib/norm.mjs`): Unicode NFKC → full Unicode
  lowercase (JS `toLowerCase` / Rust `str::to_lowercase`, final sigma
  included) → every maximal run of code points whose General_Category is not
  `L*` or `N*` becomes one U+0020 → trim. `sha256Norm` = lowercase hex SHA-256
  of the UTF-8 bytes. Rust: `unicode-normalization` `.nfkc()` →
  `to_lowercase()` → `regex` `[^\p{L}\p{N}]+` → `" "` → trim. Not
  `char::is_alphanumeric` (Alphabetic keeps Other_Alphabetic marks such as
  Devanagari vowel signs, which are separators here). Golden vectors:
  `test/norm-golden.json` (52 cases, cross-checked against that Rust recipe).
- **`train-text-hashes.txt`** (shipped with the model, Assertion B,
  `lib/leakage.mjs`): one `sha256Norm(text)` per training row, lowercase 64-hex,
  LF only, sorted ascending, duplicates allowed, optional trailing LF, nothing
  else (no blanks, no comments, no CR). Checked against tickets test + transfer
  + calibration (public suites: test) before anything is scored; any hit
  refuses the run (exit 3). The receipt records `leakage: {file, file_sha256,
  train_rows, train_unique, intersection, splits: {split: {heldout_rows,
  intersection, colliding_ids}}}`.
- **Model dir** (`--model-dir DIR`, `lib/model-dir.mjs`): either `DIR/manifest.json`
  with pinned entries (`sha256` and `tokenizer_sha256` required, optional
  `train_hashes_file` + `train_hashes_sha256`; paths relative to `DIR`) — e.g.
  `DIR/openjev-small-v0/{model.onnx,tokenizer.json,train-text-hashes.txt}` — or
  a bare dir with `model.onnx` (or `onnx/model.onnx`) + `tokenizer.json`, for
  which an entry is synthesized (384-d, CLS, 256 tokens) from hashes computed at
  load. The bench selects the entry by name (never the binding's first-entry
  fallback), verifies both hashes, passes that single entry inline to the
  engine, and records it as `embedder_model` in the receipt. Train hashes:
  `--train-hashes` > `train_hashes_file` > `<model file dir>/train-text-hashes.txt`;
  `openjev*` models must have one.
- **`--no-test`** (`run.mjs`, `optimize.mjs`): selection runs. `run.mjs`
  scores validation + transfer only and skips the Jev replay; `optimize.mjs`
  withholds the test rows from the campaign and records test fields as `null`.
  Incompatible with `--gate` / `--emit-records`.
- **Per-item records**: local-arm receipts carry `item_records.local.<split>`,
  one `{id, predicted, truth, correct, confidence, abstain?, urgent_score?,
  oos?}` per item (keys of `predicted`/`truth`/`correct`: `department`,
  `urgent`, `frustration`; `intent` on public suites). `--emit-records PATH`
  writes the test records as `ruvector-typesafe-bench/item-records@1`.
- **`vs-jev.mjs --receipt|--records PATH [--out PATH] [--strict]`**: ADR-007
  §1b tiers against `test_rows.baseline` and `test_rows.champion` with
  `lib/paired.mjs` (port of `PairedSequentialTest`; `test/paired-golden.json`
  holds wealth paths emitted by the Rust type), α = 0.05, λ = 0.5, test ids in
  lexical order, on the full / novel (119) / template (31) slices. The slice is
  frozen in `fixtures/novel-slice-2026-09-26.json`, pinned under
  `HASHES.json` `derived` (never `files`, which must equal the release receipts'
  fixture pin), and recomputed on every run.
- **`scripts/fetch-models.mjs --add-openjev <40-hex HF commit>`** appends the
  `openjev-small-v0` entry (`hf_repo`, `hf_revision`, empty pins) and
  bootstraps its pins from `ruvnet/openjev-small-v0/resolve/<commit>/…`
  (`onnx/model.onnx`, `tokenizer.json`, `train-text-hashes.txt`). Nothing is
  committed to `models/manifest.json` before publication.
