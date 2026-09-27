# KGE benchmark campaign and brain integration: implementation plan

Plan of record for [ADR-007](../../../npm/packages/kge/docs/adr/ADR-007-sota-training-and-brain-integration.md) (revision 3).

**Worktree:** `/home/ruvultra/projects/ruvector-kge`, branch `feat/kge-bench`.

**Rules.**
- Rust only. No Python, and no new JS beyond minimal edits to the existing harness.
- No test scoring before the signed tag `kge-prereg-v1` exists on `origin` (M5).
- No paid compute before spend gates (a)–(d) (ADR-007 §4); the only earlier launch is the ≤ $2 gate-(d) probe in M7.
- The claim ceiling is "reproduction exceeds published anchor" (REA). No SOTA wording.

**Figure labels used throughout.**
- **MEASURED:** 8.3 ms/pos (HolE, 256 reals, 1-vs-all, |E|=2000); the 494-entity slice receipt; vast.ai prices; /v1/status; RTX 5080 16 GB.
- **EXTRAPOLATED / ASSUMED:** every full-dataset time and every FLOP→time conversion; δ until M4; the W1 speed-up and the W2 threshold until measured.

**Units.** "Rank" is `complex_rank` k (2k reals). HolE figures are in reals.

**Order.** M0 → M1 → M2 → M3 → M4 (→ M4b) → M5 → M6 → M7 → M8 → M9.
- The brain lane (B0–B4) runs in parallel from M0 onward. B3 needs M1 and M2.
- The upstream fixes F1–F3 and the measured wins W1–W2 are **required
  deliverables** inside the milestones named in their section below.

**Roles:** `coder`, `tester`, `reviewer`, `performance-engineer`, `security-auditor`, `researcher`, `release-manager`.

---

## Required upstream fixes and measured wins (F1–F3, W1–W2)

| Id | Lands in | What | Acceptance |
|---|---|---|---|
| **F1** Table-size cap | M0 | Enforce `dims` (1..=`MAX_DIMS`) and an `entities × dims` cap (value ASSUMED until sized against the wasm32 and napi memory limits) in `Tables::new`/`try_new` (`crates/ruvector-kge/src/tables.rs`), `ContinualTrainer::grow` (`optimize/continual.rs:61`) and model load (`crates/ruvector-kge-{ffi,wasm}/src/model.rs`). At planning HEAD `8307c1d4c` only `adversarial::check_limits` existed, called only by its test; a concurrent lane is adding `tables::check_table_size` + `MAX_TABLE_BYTES` (byte cap on `(entities+relations)·dims·4`), which satisfies this row once all three entry points call it. The typed error is `KgeError::Limit`, or a new `KgeError::TableTooLarge{entities,dims,cap}`. | Unit tests: an oversized construct, grow and load each return the typed error with no allocation attempt. A fuzz/proptest over (entities, dims) never aborts. |
| **F2** Snapshot training | M3 | At planning HEAD `8307c1d4c`, ffi `train_json` held the write lock for the whole run (a concurrent lane is moving it to `train_shared`; this row is met only when the acceptance tests pass). Fix: clone the tables and triples under a read lock, train off-lock, then swap under a brief write lock, rejecting the swap if the model changed meanwhile. `predict_json` also takes `write()`: make it read-only by making ANN build explicit (`build_index_json`), not lazy inside predict. Mirror the change in wasm. | Test: a predict issued mid-train returns within 50 ms, using the pre-swap model. Test: a concurrent `add_triples` during training makes the swap fail cleanly. The `Arc<RwLock<KgeModel>>` contract is reused by brain B4. |
| **F3** Publish provenance | M9 | npm `publish --provenance` from CI only (OIDC). All platform packages are built and published by CI, never locally. `SHA256SUMS` over every `.node` and `.wasm` goes in the release. `build-kge.yml` gets a guard that fails if `package.json` version is 0.1.0 or already on the registry. | `npm view @ruvector/kge@0.2.0` shows provenance. SHA256SUMS matches the tarballs. A dispatch at 0.1.0 fails the guard. |
| **W1** SelfAdversarial speed | M1 | MEASURED speed-ups disagree: 8.3 vs 1.13 ms/pos gives ≈ 7.3× (planning maps), and a separate measurement reports ≈ 25×. Re-measure both on one harness (same |E|, d, thread count, warm cache). Add SelfAdversarial as an **arm of the npm-default HolE bench suite** (ADR-006), not one of ADR-007's C1–C8. | A receipt records the reconciled factor. The default switches to SelfAdversarial only if its valid MRR is ≥ the 1-vs-all valid MRR − 0.005 on the slice and on full WN18RR. Otherwise it stays an opt-in arm. |
| **W2** Auto search mode | M2 | `predict` picks exhaustive search below an entity threshold and ANN (`ann.rs`) above it. The threshold is ASSUMED at ~1–2k entities until measured. Extend `ann.rs:142` `recall_at_10_vs_exhaustive` into a sweep over |E| ∈ {500, 1k, 2k, 5k, 15k} at d=256 recording recall@10 and p50/p99 latency. | The threshold is the largest |E| where exhaustive p99 ≤ ANN p99. It is written as a named constant with the receipt path in its doc comment, and a `mode: auto|exhaustive|ann` override is exposed in the napi/wasm API. Below the threshold, recall is 1.0 by construction (test). |

---

## M0: Honesty and correctness fixes (blocks every number)

**Owner:** coder + tester · **Est:** 28 h (includes F1) · $0

**Code changes**
- `crates/ruvector-kge/src/eval/rank.rs`: a non-finite target returns `KgeError::NonFinite`, and a non-finite candidate ranks as worst.
  - Eval computes **Bottom and RANDOM ranks in one pass** (ADR-007 §2.2) and returns both per-query vectors.
  - New tests:
    - an all-NaN scorer must not produce MRR 1;
    - filtered tie-break at |E| ≥ 1000;
    - **RANDOM ≤ Bottom for every query** (CI-asserted).
- `crates/ruvector-kge/src/optimize/trainer_eval.rs:286-292`: stop aliasing `Loss::Bce | Loss::Margin` to `SelfAdversarial`. Implement real BCE or rename the arms, and have receipts record the `LossKind` that actually ran.
- `npm/packages/kge/bench/lib/gates.mjs:91-92`: `pass=false` when the engine is unavailable or zero rows were measured.
- `npm/packages/kge/bench/run.mjs`:
  - add `--help`;
  - add a `--train-config <json>` passthrough (today `:160` passes only scorer/dims/seed/epochs);
  - validate `--baseline-receipt`;
  - run `assertDisjoint` on every suite;
  - run hard-negative scoring as a separate invocation;
  - add a no-decoy retrain control;
  - rename `training.triplesPerSec` to `triple_epochs_per_sec`.
- `npm/packages/kge/bench/datasets/lib.mjs:139,145` and `codexm.mjs:84`: pass `transfer` into `graphCounts`.
- `npm/packages/kge/bench/datasets/yago310.mjs:12`: sha256-pin YAGO3-10, or make it fail closed.
- `npm/packages/kge/bench/datasets/wn18rr.mjs:11-17`: switch `BASE` from villmow
  `WN18RR/text` (41,105 entities, MEASURED) to `WN18RR/original`, the ID-based
  ConvE release the anchors use (ssl-RP README "#Ent 40,943",
  `ComplEx(sizes=[40943,22,40943])`). Same 86,835/3,034/3,134 triples. New
  pins (MEASURED 2026-09-26): train
  `038612e783c215ee5f3ca9fbfca27b8d0739be1028fe4ee7c174aecf0b83d5df`, valid
  `453ce7202afa58094a04d2b1560ee2b02660f1c260b32ce6651c8ccedd1028ab`, test
  `0383bceaaa1096cf3c03ec021ed0048068e2355dbfc0239b292cefdac821cec5`. Update
  the `lib.mjs:155` comment. Unseen-entity test triples become 210 (MEASURED).
- **Entity-count identity assert** (JS loader and the M3 Rust loader, recorded
  in the manifest): FB15k-237 14,541, WN18RR 40,943, CoDEx-M 17,050 (ssl-RP
  README). Any other count fails closed.
- `npm/packages/kge/bench/lib/receipt.mjs`: add the crate git SHA, binary sha256 and receipt-body sha256, and drop the absolute `binding_source`.
- `crates/ruvector-kge-{ffi,wasm}/src/model.rs` (byte-identical): add a zero-init option so `--tie-check` stops SKIPping.
- **F1** (see the table above).

**Acceptance**
- `cargo test -p ruvector-kge` is green, including the NaN, filtered-tie, RANDOM ≤ Bottom and F1 tests.
- An engine-down `--suite synthetic --gate` exits 1.
- FB15k-237 `counts.entities = 14541`; WN18RR = 40943; CoDEx-M = 17050.
- A doctored WN18RR `text` split fails the entity-count assert (test).
- No receipt says `bce` unless BCE ran.
- `--tie-check` gives PASS or FAIL on the napi binding.
- The 0.4995 slice receipt reproduces within ±0.01.

**Rollback:** revert the PR. No artifacts are published.

---

## M1: Recipe ingredients (ComplEx-N3-R), no new kernel yet

**Owner:** coder + tester · **Est:** 40 h (includes W1) · $0

**Code changes**
- `crates/ruvector-kge/src/scorer/complex.rs` + `scorer/mod.rs:40-41`: take ComplEx out of `#[cfg(test)]`.
  - It needs `Scorer` + `Differentiable`, split re/im, `index_vector=[re;im]` and `query_vector` for both sides, and a finite-difference grad test.
  - Keep the HolE≡ComplEx gate.
- `crates/ruvector-kge/src/train/mod.rs` (TrainConfig, `fit` at 122-190, `apply_n3` at :232): add
  - `reciprocal`;
  - `init: Xavier | Normal{scale}`;
  - `reg: N3Moduli{lambda}`;
  - `loss_reduction: Mean`;
  - `rp_weight`;
  - `complex_rank` (recorded in receipts);
  - a per-epoch valid-MRR hook with early stop and a best-on-valid snapshot.
- `crates/ruvector-kge/src/tables.rs:32-53`: normal·scale init, and 2·|R| relation rows under `reciprocal`.
- `crates/ruvector-kge/src/eval/mod.rs` (~138-143): under `reciprocal`, head queries become `(o, r⁻¹, ?)` filtered with `true_heads`.
- `crates/ruvector-kge/src/train/optim.rs:53-54, 93-100`: dense grads and Adagrad/Adam state for 1-N. The sparse path stays for negative sampling.
- `crates/ruvector-kge/src/train/loss.rs:199,222,255`, `scorer/hole.rs`, `scorer/fft.rs`: grads go into caller buffers, and FFT scratch is reused.
- `crates/ruvector-kge-{ffi,wasm}/src/{model.rs,ops.rs}`: reciprocal-aware serialisation and the `predictJson` head side.
- **W1** (see the table above).

**Acceptance**
- The ComplEx FD grad test passes within 1e-3 relative.
- The HolE≡ComplEx gate is green.
- With tied reciprocal rows (conj(r) as r⁻¹), reciprocal eval equals non-reciprocal eval.
- The W1 receipt reconciles 7.3× vs 25×.
- The wasm32 build and import audit are green.

**Rollback:** revert the PR.

---

## M2: Batched 1-N GEMM kernel plus `parallel` feature

**Owner:** performance-engineer + coder · **Est:** 46 h (includes W2) · $0

**Code changes**
- `crates/ruvector-kge/src/scorer/mod.rs`: `trait Bilinear: Scorer { query_batch, query_backward }`, for ComplEx and for HolE (via its frequency view).
- `crates/ruvector-kge/src/train/kernel.rs` (new). Per batch:
  - `Q` (B×2k), `logits = Q·Eᵀ`;
  - softmax-CE, or BCE with label smoothing;
  - `grad_E = dLᵀ·Q`, `grad_Q = dL·E`;
  - the RP auxiliary loss via a GEMM over relations.
- `crates/ruvector-kge/src/train/loss.rs:172`: route `OneVsAll` to `kernel.rs` for `Bilinear` scorers. The per-triple path is kept as the test oracle.
- `crates/ruvector-kge/Cargo.toml` (**shared file**):
  - `matrixmultiply` (or `gemm`) after a `deny.toml` licence check;
  - optional `rayon` under `[features] parallel`;
  - the default build stays wasm-safe.
- `crates/ruvector-kge/src/eval/mod.rs:~128,140`: `BatchScorer` mat-vec plus rayon over queries (per-query `qseed` exists).
- Fixed-order reductions, so results are deterministic per thread count.
- A microbench reports ms/batch at complex_rank 1000 on |E|=14,541 synthetic tables.
- **W2** (see the table above).

**Acceptance**
- The batched loss and grads equal the per-triple oracle within 1e-4.
- Single-threaded 1-N is ≥ 20× faster per positive than 8.3 ms (HolE, 256 reals, |E|=2000).
- On the slice, ComplEx-N3-R **valid** MRR ≥ HolE slice valid − 0.02.
- Two same-seed runs are bitwise identical at the same thread count.
- The wasm32 build is green with `parallel` off.
- A *projected* full FB15k-237 epoch time is recorded, **informational only**. Spend gate (a) is decided in M4 on a measured epoch.

**Rollback:** revert the PR. The `parallel` feature is off by default.

---

## M3: Native Rust bench binary with `--final` and `--verify-final`

**Owner:** coder + tester · **Est:** 46 h (includes F2) · $0

**Code changes**
- `crates/ruvector-kge-bench/` (new binary crate, **`publish = false`**, joins the workspace; the root `Cargo.toml` is a **shared file**). The core stays fs-free.
  - **Loaders.** FB15k-237, WN18RR (`original`), CoDEx-M and YAGO3-10 loaders use the same sha256 pins and transfer carve as `bench/datasets/*.mjs`, plus the M0 entity-count identity assert.
  - **Training.** `max_epochs` per run (HPO ≤ 100 with early stop; final 500 with best-on-valid and no early stop, ADR-007 §3).
  - **Checkpoints.** A tables-only checkpoint every epoch, resumable from `$RVGR_ARTIFACT_DIR`.
  - **Leakage asserts.** By triple hash, train ∩ (valid ∪ transfer ∪ test) = ∅, and the filter set equals the four-split union. All four split hashes plus the sorted entity-vocabulary-order hash go in the receipt and the manifest.
  - **Default mode.** Scores **valid only**, and also emits per-query valid RRs (used by M4 to size δ).
  - **`--final`** (ADR-007 §2.6). It reads the record of truth from **origin's** protected branch `kge-final-ledger` (`git fetch origin kge-final-ledger`), never the working tree. It refuses unless all of these hold:
    - `git ls-remote origin refs/tags/kge-prereg-v1` resolves, `git verify-tag` passes against the signer fingerprint committed at the tag, and the tag is an ancestor of HEAD;
    - origin's `selection.json` exists and the run's `config_hash` equals its entry for the dataset;
    - the seed is in {0,1,2,3,4};
    - origin's `ledger.jsonl` has no line for (dataset, seed).

    Before scoring test, it appends an `intent` line (dataset, seed, config_hash,
    tag SHA, checkpoint sha256, UTC) and pushes it as a fast-forward. If the push
    is rejected or fails, it scores nothing. It then scores test once, writes the
    receipt with the tag SHA and the **per-query Bottom and RANDOM rank vectors
    plus their sha256**, and pushes a `result` line (receipt sha256). A retry of
    an `intent` with no `result` pushes **no new intent**: it refuses unless the
    checkpoint sha256 equals origin's intent, then scores and pushes only the
    `result` (scoring is deterministic; RANDOM is seeded per query).
  - **`--verify-final <receipt>`.** Loads the exported tables, recomputes the
    per-query ranks, and compares their sha256 with the receipt. The result is
    recorded as `verification`, never `final`, and no ledger line is written.
  - **Export.** Rust `export_tables`/`load_tables`: an f32 blob plus a manifest (scorer, complex_rank, reciprocal, seed, vocab-order hash, pins, receipt hash). No triples or labels.
- CI job (`.github/workflows/kge.yml`, **shared file**) on `kge-final-ledger`. It fails if:
  - a ledger line is modified or deleted (append-only);
  - a dataset has more than one `config_hash`, or one that differs from `selection.json`;
  - a seed is outside {0..4}, or a (dataset, seed) has a second `intent` or `result`;
  - a `result` has no `intent`, or a tag SHA differs from the tag;
  - `selection.json` changes after the first `intent`.
- **F2** (see the table above).

**Acceptance**
- The receipt validates against `receipt@1`.
- The slice valid MRR is within 1e-6 of the napi path.
- Resume is bitwise-identical to the uninterrupted run.
- The export has zero triple rows.
- The leakage and entity-count asserts fire on a doctored split (test).
- Tests against a local bare "origin": `--final` refuses without the tag, with an unsigned or unknown-signer tag, with no `selection.json`, with a non-selected `config_hash`, with seed 5 or 100, when (dataset, seed) is already on origin, when the key exists only in origin but not the working tree, and when the intent push is rejected (a concurrent push). A working-tree-only ledger edit does not unlock a key.
- `--verify-final` passes on its own output and fails on a perturbed table.

**Rollback:** revert the PR. The ledger branch is created only in M5.

---

## M4: Local $0 measurement (valid only), then the go/no-go

**Owner:** performance-engineer + researcher · **Est:** 16 h · $0

**Deliverables**
- **Measured** 1-epoch runs on full FB15k-237, WN18RR and CoDEx-M at complex_rank 1000 and 2000, at each dataset's C1 batch size (32 threads, and GPU if M4b ran), recording s/epoch and peak RSS.
- Full WN18RR ComplEx-N3-R at its C1 (k=1000, lr 0.1, batch 100, λ 0.10, w_rel 0.05; ssl-RP `wn18rr.md`), seed 100, ≤ 100 epochs, **valid only**.
- **δ sizing.** sd(RR) from per-query **valid** RRs on each dataset, then δ = sd(RR)/√(n_test_queries). Placeholders +0.002 / +0.006 / +0.003 are replaced by these values in the M5 ADR.
- **HPO wiring.** The HPO over each dataset's C1–C8 (ADR-007 §3 table) goes through the ADR-004 campaign as `--optimize` (`bench/lib/arms.mjs:172` is currently never called).
  - Seed 100, ≤ 100 epochs, early stop (declared deviation; affects selection only).
  - α/N = 0.05/8 in `gate.rs:57`.
  - Selection reads valid only.
- **72 h rule.** Projected wall time from the measured s/epoch for 3 × 8 × ≤ 100 = 2,400 HPO epochs plus 3 × 5 × 500 = 7,500 final epochs (9,900 dataset-epochs; about 26 s/epoch mean to fit 72 h). Reference: the anchors' own reported 500-epoch times (FB 11.5 h, WN18RR 7 h, CoDEx-M 9.5 h) give about 185 h, so the rule is expected to fail unless the kernel is about 2.6× faster than that.

**Acceptance (spend gates)**
- **Gate (a):** a measured full FB15k-237 epoch at complex_rank 1000 takes ≤ 15 min on ruvultra. Committed as a receipt. This is a kernel sanity floor only (at 15 min the campaign is about 2,500 h).
- **Gate (b):** WN18RR valid MRR ≥ 0.47 at its C1 within 100 epochs. If it falls short, stop and debug, with no rental.
- The 72 h verdict is recorded and reported to the user.
- The δ values are committed with their valid-RR receipt.

**Rollback:** none needed (measurement only).

## M4b (conditional): `cuda` feature in the bench crate

**Owner:** performance-engineer · **Est:** 24 h · $0 · **Trigger:** the M4 CPU measurement misses the 72 h rule.

- `crates/ruvector-kge-bench`: optional `cuda` feature (cudarc or candle, Rust-only) for the `Bilinear` batch forward/backward. The core crate is untouched.
- Images:
  - the local RTX 5080 is Blackwell sm_120 and needs **CUDA ≥ 12.8**;
  - the runner's CUDA 12.4 image is kept for rented 4090s;
  - both are pinned by digest.

**Acceptance**
- Logits and grads match the CPU kernel within 1e-3.
- Measured FB15k-237 s/epoch is recorded.
- Same-seed MRR is within 1e-3 of the CPU run.

---

## M5: Pre-registration (ADR-007 → Accepted, signed tag)

**Owner:** researcher + reviewer + release-manager · **Est:** 12 h · $0

**Changes**
- **ADR-007.** Set Status to Accepted, and freeze:
  - recipe, the per-dataset C1–C8 table (C1 = ssl-RP "Best Run" per dataset), HPO seed 100 and the 100-epoch HPO cap;
  - final seeds {0..4} at 500 epochs;
  - anchors, n_anchor, the measured δ, Holm across 3 datasets, and the Bottom verdict;
  - the WN18RR `original` variant and the entity-count identity values.

  Resolve every **TO-VERIFY**, or drop the figure:
  - CoDEx paper table for 0.337;
  - Lacroix 2018 Table 2 WN18RR 0.48 vs README 0.49;
  - kbc YAGO3-10 0.58.
- **ADR-006:25.** Cite "LibKGE RotatE reproduction 0.478", and amend Protocol 1 so that test is read only by `--final`.
- **`npm/packages/kge/bench/gates.json`.** Add `link_prediction_codexm` ≥ 0.307, and informational `rea_*` rows holding the anchors, n_anchor, δ and the bound rule.
- **`codexm.mjs:89`:** `baselineMrr: 0.337`.
- **CI:** gates.json anchors must equal the ADR table.
- **Signer.** Commit the signer fingerprint file (e.g. `npm/packages/kge/bench/results/final/SIGNERS`).
- **Ledger branch.** Create `kge-final-ledger` on origin holding an empty `npm/packages/kge/bench/results/final/ledger.jsonl` (no `selection.json` yet), with branch protection: no force-push, no deletion, the M3 ledger CI required.
- **Tag.** Create the signed tag `kge-prereg-v1` on the Accepted commit and **push it to `origin` (ruvnet/ruvector)**.
- **Licence decision** (blocking; owner **release-manager**). Decide per dataset whether HF weights may ship:
  - FB15k-237 is derived from Freebase MIDs and released by Microsoft;
  - WN18RR is derived from WordNet.

  Record each decision in the ADR.

**Acceptance**
- `git ls-remote origin refs/tags/kge-prereg-v1` resolves and `git verify-tag` passes.
- `kge-final-ledger` exists on origin, is protected, and a force-push to it is rejected.
- No `--final` receipt exists anywhere yet (the ledger is empty).
- The gates run without `--report-only`.

**Rollback:** before any `--final`, a mistaken tag is superseded by `kge-prereg-v2` with a written reason in the ADR. Tags are never moved or deleted. After any `--final`, there is no rollback: a protocol change needs a new ADR and new seeds.

---

## M6: Runner checkpoint safety (always) + CPU mode (conditional)

**Owner:** coder + security-auditor

**M6a, checkpoint safety (required whichever runner mode wins).** Est 10 h · $0
- `crates/ruvector-gpu-runner/src/job.rs:110-145` today tars and uploads artifacts once, after `main` exits. Add:
  - **periodic upload.** A per-epoch (or every N minutes) upload of `$RVGR_ARTIFACT_DIR` to a rolling artifact slot (signed URL or GCS object), so preemption or the budget watchdog (`budget.rs:51`) loses at most one interval;
  - **`--resume-from <artifact>`.** Fetches the artifact into `$RVGR_ARTIFACT_DIR` on the new instance **before** the job command runs.
- Tests:
  - an upload happens mid-run (mocked clock and store);
  - resume places the files before `cmd` starts;
  - a watchdog-destroyed job leaves the last checkpoint uploaded.

**M6b, CPU mode (conditional: only if the CPU kernel won and rental is justified).** Est 16 h · $0
- `crates/ruvector-gpu-runner/src/offer.rs`:
  - `Offer` (:20) gets the CPU fields (cores, RAM, CPU name);
  - `OfferFilter` (:105) gets `min_cpu_cores`/`min_cpu_ram_gb`, and an empty `gpu_names` means any GPU;
  - `query_json` (:117) omits the GPU keys in CPU mode;
  - `reject_reason` (:140) re-checks the CPU fields;
  - `rank` (:186) orders by $/core-hour.
- `main.rs`: `--cpu-mode`, `--min-cpu-cores`, `--min-cpu-ram-gb` (these drop the `:74` GPU defaults), an overridable `--max-dph` (:83), and the image pinned by digest. Fixture offers get the CPU fields.
- The job script (modelled on `scripts/openjev/vast-train.sh`) builds `ruvector-kge-bench --features parallel` with `target-cpu=native`.

**Acceptance**
- The M6a tests are green.
- `--dry-run` lists offers without launching.
- The existing GPU safety tests are unchanged and green.
- The runner refuses to launch past `--max-usd`.

**Rollback:** revert the PR. The runner is `publish = false` and internal.

---

## M7: Training campaign (local first, vast.ai only per the M4 verdict)

**Owner:** performance-engineer · **Est:** 30 h · **≤ $150** (hard cap $250 total, per-job ledger)

**Deliverables**
- **Gate (d), before any rental** (only if the M4 72 h verdict failed; needs gates (a)–(c)): one probe job (one FB15k-237 epoch at C1 on the candidate offer type, ≤ $2, counted against the cap) measures $/epoch. Projected cost = $/epoch × the remaining epochs, scaled per dataset by the M4 local s/epoch ratios. Launch only if ≤ $150; otherwise report the figure to the user and stop.
- FB15k-237, WN18RR and CoDEx-M:
  - HPO over **exactly that dataset's C1–C8** (ADR-007 §3) on valid, seed 100, ≤ 100 epochs with early stop;
  - **selection record:** push `selection.json` (one `config_hash` per dataset, the valid-MRR argmax, with the HPO receipt hashes) to origin `kge-final-ledger` before M8;
  - the selected config × seeds {0..4} × **500 epochs** (the anchors' count), best-on-valid checkpoint, no early stop.
- YAGO3-10 only after pinning: ≤ 1 seed, last, informational, and only if budget remains.
- Rented jobs:
  - run under 11.5 h with M6a checkpoint upload, resumable via `--resume-from`;
  - use `--require-datacenter` where possible;
  - log $/h, hours and receipt hash to the audit log.

**Acceptance**
- Every HPO run's config hash is one of that dataset's C1–C8; every final run's hash equals `selection.json`.
- `selection.json` is on origin before any final-seed run is scored.
- Total spend ≤ cap.
- No test scoring happens in this milestone.

**Rollback:** stop launches, keep the checkpoints. Spend is not recoverable, and the cap bounds it.

---

## M8: One-time test scoring and write-up

**Owner:** researcher + reviewer · **Est:** 8 h · $0

**Deliverables**
- `ruvector-kge-bench --final` on each best-on-valid checkpoint of the **selected config only**, **exactly once per (dataset, seed ∈ {0..4})**, each pushing an `intent` line to origin before scoring and a `result` line after.
- Results table:
  - Tier A, "matches" and **REA** verdicts per ADR-007 §1, with the verdict on **Bottom** and RANDOM shown alongside;
  - L = m − t·s·√(1/n + 1/n_anchor), n_anchor (FB 1, WN 3, CoDEx-M 1), δ, and Holm-adjusted p across the 3 datasets;
  - mean ± std, 95% CI, H@1/3/10, head/tail;
  - the NBFNet context row and the lower class rows from the ADR table;
  - MEASURED vs EXTRAPOLATED labels.
- Negative results are published as-is.
- The paired per-query bootstrap and seed-paired t-test are **not applicable** (REA has no in-house comparison arm; ADR-007 §1). They are required only in an ADR-008 ingredient hypothesis.

**Acceptance**
- Every number traces to a ledger line, and its receipt descends from `kge-prereg-v1`.
- CI ledger uniqueness is green, and each claimed dataset has exactly five `result` lines, seeds {0,1,2,3,4}.

**Rollback:** errata policy (ADR-007 §7). Append to `bench/results/final/ERRATA.md` and mark the table row retracted. Never delete ledger lines.

---

## M9: Publish and merge

**Owner:** release-manager · **Est:** 26 h (includes F3) · $0

**Changes**
- `.github/workflows/build-kge.yml`:
  - fix the stale comments (3-8; `:312-314` "NO platform-package fallback" is wrong, since `index.js:51` tries `@ruvector/kge-<platform>`);
  - add the version guard (F3);
  - make publish depend on a green full-suite receipt.
- **wasm fallback.** `index.js:53-56` falls back to `./wasm/ruvector_kge_wasm.js`, but `npm/packages/kge/wasm/` is never built. Either add a wasm-pack step plus a tarball assertion for `wasm/`, or remove the fallback and document the supported platforms.
- **Binary provenance.** The bundled binaries must be built from the same SHA as the receipts (ComplEx/reciprocal features).
- `.github/workflows/kge.yml`: a scheduled/dispatch full-suite job with non-report-only gates at a reduced epoch count.
- **npm.** `npm/packages/kge/package.json` → 0.2.0 (**F3**: `--provenance`, CI-built platform packages, `SHA256SUMS`). Dispatch with `dry_run=true`, then `false`. `build-kge.yml:311-321` already bundles all five `.node` binaries, so no `optionalDependencies` follow-up is needed for 0.2.0.
- **Crates.** Per-crate versions past 2.3.0 (e.g. 2.3.1). Publish `ruvector-kge` first, then `-ffi` and `-wasm`, with their `ruvector-kge = { version = ... }` requirement updated. `ruvector-kge-bench` stays `publish = false`. Use the GCP `CARGO_REGISTRY_TOKEN` path.
- **Hugging Face** `ruvnet/ruvector-kge-{fb15k237,wn18rr,codexm}`, only where the M5 licence decision allows:
  - tables-only weights plus receipts;
  - a model card with the receipts, the Bottom/RANDOM tie-break, the valid-carve deviation and the vocab-order spec;
  - uploaded with a Rust uploader in the bench crate (token: GCP secret `huggingface-token`).

  Otherwise receipts only.
- PR `feat/kge-bench` → `main` with the results table.

**Acceptance**
- `npm view` shows 0.2.0 with provenance and the platform packages. SHA256SUMS is published.
- crates.io shows the new versions, published in order.
- **`ruvector-kge-bench --verify-final <receipt>`** passes on every published table. This replaces the old "re-score test" check: test is not re-scored.
- CI is green and the PR is merged after review.
- No SOTA wording appears. "Exceeds the published anchor" wording appears only for datasets where M8's REA verdict passed.

**Rollback (ADR-007 §7)**
- npm: `npm deprecate @ruvector/kge@0.2.0` and the platform packages, then ship a fixed 0.2.1. Never plan on unpublish.
- crates.io: `cargo yank --version` per crate, then a patch release.
- Hugging Face: revert the commit and mark the card RETRACTED.
- Results: erratum as in M8.

---

## Brain lane (pi.ruv.io): B0–B4, own gate, runs in parallel

**Data residency (ADR-007 §4).** Brain data never leaves ruvultra or the
brain's own Cloud Run project. The lane has **no rental budget ($0)**.

| Stage | Owner | Est | $ | Deliverables | Acceptance |
|---|---|---|---|---|---|
| B0 Measurement | researcher + security-auditor | 6 h | 0 | Tag histogram and Zipf shape, Custom category count, contributor mix, weight>0.9 edge share, `dp_proof` null rate, `RVF_DP_ENABLED` on Cloud Run, any visibility or deletion flags. | Numbers recorded. **Go only if ≥ ~1k tags have ≥ 3 uses.** The DP finding is reported to the user. |
| B1 Merge queue | coder | 8 h | 0 | Near-duplicate queue from edges with weight > 0.9 plus a title/content hash, sent to moderators. No model. | Served. Never auto-merges. |
| B2 Export + triples | coder + security-auditor | 12 h | 0 | **Prerequisite:** a separate fix or issue for the `verify_system_key` fail-open (`routes.rs:8216-8221`). Fail-closed `GET /internal/kge/export`: snapshot under the read lock; ids, category, normalised tags (min-count 3, PiiStripper re-run), salted contributor hashes. No content, and no hidden or deleted memories. It refuses if production DP is on (KGE training is not DP). Triples v1 per ADR-007 §6. Temporal splitter in `crates/ruvector-kge/src/data.rs`. | 401/503 with the key unset (test). No content text (test). Temporal-split test passes. The worker `create_router` is verified writer-free. |
| B3 Held-out eval | researcher + tester | 12 h | **0** | Train ComplEx-N3 (M1/M2 path) **on ruvultra or Cloud Run only**. Run the 5 pre-fixed baselines and human P@20 on 200 items. | **Ship only if KGE ≥ best baseline + 0.02 filtered MRR (temporal test) AND P@20 ≥ 0.6.** Otherwise kill, and keep B1. |
| B4 Serve (only if B3 ships) | coder + reviewer | 8 h | 0 | F2-style snapshot-trained `Arc<RwLock<KgeModel>>` with a pinned model version; `KGE_ENABLED` env kill-switch; `/v1/kge/suggest-tags`, `/v1/reclassify` flags, MCP `brain_suggest_tags` and `brain_similar_tags`; queue items tagged with the model version; retrain within 7 days of any `brain_delete`, with serve-time filtering of deleted ids until then. | KGE never takes the graph write lock (test). `predict` never returns contributor or voter entities (test). k-anonymity: no suggested tag is used by fewer than 3 distinct contributors (test). `KGE_ENABLED=false` disables every route and tool (test). Brain tables stay private. |

**Brain rollback:** set `KGE_ENABLED=false` (an env-only Cloud Run revision).
Hot-swap the previous pinned model version, and purge queue items by model
version. B1 is independent and stays up.

---

## Totals (milestone sum; this is the estimate)

| Block | Hours | $ |
|---|---|---|
| M0–M3 (honesty + F1 + WN18RR variant, recipe + W1, kernel + W2, bench binary + F2 + origin ledger) | 28 + 40 + 46 + 46 = 160 | 0 |
| M4 + M4b (measure; CUDA if triggered) | 16 + 24 | 0 |
| M5 pre-registration and signed tag | 12 | 0 |
| M6a checkpoint safety + M6b CPU mode (conditional) | 10 + 16 | 0 |
| M7 campaign | 30 | ≤ 150 |
| M8 final scoring and write-up | 8 | 0 |
| M9 publish (+ F3) | 26 | 0 |
| B0–B4 brain lane | 46 | **0** |
| **Total** | **348 h (~44 engineer-days); 308 h if neither conditional triggers** | **≤ $150 planned; $250 hard cap** |

**Before any rental:** M0–M5 is 188 h, or 212 h with M4b.

The cap is enforced by runner `--max-usd` and the per-job ledger. The roughly
$100 between the planned spend and the cap needs explicit user sign-off.
