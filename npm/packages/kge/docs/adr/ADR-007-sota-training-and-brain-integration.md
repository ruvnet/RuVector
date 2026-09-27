# ADR-007: Pre-registered KGE benchmark campaign and pi.ruv.io brain integration

## Status
Proposed (pre-registration draft, revision 3 after the completeness critique and the anchor-config review).
The targets and protocol here are frozen by the signed tag `kge-prereg-v1`
pushed to `origin` (§2.6), not by a commit date. The M5 commit that flips this
status to **Accepted** is the one that tag points at. Every `--final`
(test-scoring) receipt must descend from that tag.

## Date
2026-09-26

## Context

Two requests drive this ADR:
1. **Benchmark and publish (the OpenJev pattern).** Pre-register targets on
   FB15k-237 and WN18RR, train on the existing vast.ai runner, and publish
   honestly. The user's "about 0.34 / 0.48" is the LibKGE ComplEx reproduction
   (0.348 / 0.475). That is our release gate, not the state of the art.
2. **Make pi.ruv.io reason over its graph.** The requested outputs are link
   prediction as a curation queue, relation dedup, and multi-hop.

What the code shows today. Plan M0/M1 hold the full defect list with file:line.
- **The trainer cannot run the literature recipe.** ComplEx is `#[cfg(test)]`
  only (`scorer/mod.rs:40-41`), there are no reciprocal relations
  (`eval/mod.rs:~138-143`), N3 is summed elementwise, and init is Xavier.
- **The 1-N path is too slow.** `one_vs_all_step` (`train/loss.rs:172`) calls
  `scorer.grad` per candidate, optimizer state lives in `BTreeMap`s, and
  everything runs on one thread. MEASURED: 8.3 ms per positive (HolE, 256 reals,
  |E|=2000, 9950X); EXTRAPOLATED: about 4.6 h per full FB15k-237 epoch. The
  494-entity slice receipt (test MRR 0.4995) is not comparable to a full dataset.
- **Honesty defects (plan M0).** A NaN target ranks 1; `bce`/`margin` both run
  `SelfAdversarial`; an engine-down `--gate` prints PASS; `graphCounts` omits
  transfer; CoDEx-M has no baseline; YAGO3-10 is unpinned; the WN18RR loader
  uses the text variant (§2.4); ADR-006 Protocol 1 scores test twice.
- **Library defects (plan F1-F3), at planning HEAD `8307c1d4c`.** Nothing caps
  `entities × dims` (`adversarial::check_limits` is called only by its test);
  train and predict both take the model write lock (ffi `train_json`,
  `predict_json`); npm 0.1.0 shipped without provenance or checksums.
- **Compute.** The runner is GPU-only (`offer.rs:105,117,140`) and uploads
  artifacts once, after exit (`job.rs:110-145`). MEASURED: vast.ai has no
  CPU-only offers; ≥128-core hosts cost $0.67-1.00/h; ruvultra has 32 threads,
  about 111 GB free RAM and an RTX 5080 (16 GB, sm_120).
- **The brain** (/v1/status 2026-09-26): 59,762 memories and 1,200,842 untyped
  `GraphEdge` edges at cosine ≥ 0.55 over the lexical 128-dim `HashEmbedder`
  (`graph.rs:73-84`); search already blends cosine with PPR.
  `verify_system_key` fails **open** with the key unset (`routes.rs:8216-8221`),
  and DP defaults to off (`types.rs:1343-1345`).

## Decision

### 1. Claim class, anchors and tiers

**Claim class.** *Tensor-factorisation / bilinear embedding models trained on
the triples alone.* That excludes GNN and path models, text/LLM features and
ensembles. The model is **ComplEx-N3 with reciprocal relations**, optionally
with the relation-prediction auxiliary loss (RP). The published number names the
scorer that produced it. The npm HolE default does not inherit it.

**Tier B decision (P0-1): option (b). Tier B is renamed "reproduction exceeds
published anchor", Tier B (REA), and no "beyond SOTA" wording is used anywhere.**

Why. A Tier B ingredient must pass three filters: (i) it is not already inside
a published anchor recipe; (ii) it is an MRR hypothesis; (iii) there is a
mechanistic reason to expect it to move MRR by at least δ. No candidate passes.
- **RP, N3, DURA and IVR** fail (i): the anchor *is* ComplEx-N3-RP, and DURA
  and IVR are published.
- **HolE's "half-parameter real form"** fails (ii). HolE with d reals equals
  ComplEx with a conjugate-symmetric spectrum, about d/2 free complex
  coordinates (Hayashi & Shimbo 2017, arXiv:1702.05563). It is a rank and
  parameter claim, so it stays an efficiency footnote, never an MRR claim.
- **Codebase features** (self-adversarial negatives, ANN, `continual.rs`,
  hard-negative mining) have no accuracy mechanism against full 1-vs-All CE.

A genuine ingredient found after M4 gets its own pre-registered ADR-008,
tested against an in-house 5-seed reproduction of the anchor recipe with the
same evaluator and seeds (paired per-query bootstrap plus seed-paired t-test).
Those paired tests do not apply to REA: anchors publish neither per-query ranks
nor per-seed values. The in-house 5-seed anchor recipe *is* our headline model.

**Anchor search rule (P0-4).** One rule applies to every dataset:
- Search date 2026-09-26. arXiv queries: ("knowledge graph completion" OR "link prediction") AND ("FB15k-237" OR "WN18RR" OR "CoDEx"), cs.LG/cs.AI/cs.CL, 2018-01 to 2026-09-26.
- Sources: arXiv plus the kbc, ssl-relation-prediction (ssl-RP), IVR, LibKGE
  and CoDEx repositories.
- Class as above, filtered MRR, standard splits. Anchor = the class maximum
  among the rows found, cited as paper/table/row or repo/file/line. A figure not
  re-read from its primary source is **TO-VERIFY** and is dropped before Accepted.

| Dataset | Tier A release gate (frozen `gates.json`) | "Matches" line | REA anchor (class max) | Other class rows found (all lower) | Context, never claimed |
|---|---|---|---|---|---|
| FB15k-237 | MRR ≥ **0.318** (LibKGE ComplEx 0.348 − 3pp) | ≥ 0.37 (kbc README, rank 1000) | **0.388**: ComplEx-N3-RP, ssl-RP README FB15K-237 table, RP=Yes (Chen et al. 2021, arXiv:2110.02834) | ComplEx-DURA .371 (DURA Table 2); ComplEx-IVR .370, TuckER-IVR .368 (IVR Table 1); RESCAL-DURA .368 | NBFNet 0.415 (arXiv:2106.06935 Table 3) |
| WN18RR | MRR ≥ **0.445** (LibKGE ComplEx 0.475 − 3pp) | ≥ 0.49 (kbc README, rank 2000) | **0.501**: TuckER-IVR, arXiv:2506.02749 Table 1 | RESCAL-DURA .498, ComplEx-DURA .491 (DURA Table 2); ComplEx-IVR .494; ComplEx-N3-RP .488 (ssl-RP) | NBFNet 0.551 (arXiv:2106.06935 Table 3) |
| CoDEx-M | MRR ≥ **0.307** (CoDEx ComplEx 0.337 − 3pp), **new gate** | ≥ 0.35 (ComplEx-N3 0.351, ssl-RP) | **0.352**: ComplEx-N3-RP (1000 dim), ssl-RP README CoDEx-M table | TuckER 0.328, RESCAL 0.317 (CoDEx README) | none |
| YAGO3-10 | informational only: 0.521 (LibKGE ComplEx 0.551 − 3pp) | ≥ 0.58 (kbc README, **TO-VERIFY**) | **0.586**: ComplEx-IVR, arXiv:2506.02749 Table 1 | ComplEx-DURA .584 (DURA Table 2; IVR's re-run shows .585) | none |

IVR and DURA do not report CoDEx-M. Every cell without TO-VERIFY was re-read
from the cited source on 2026-09-26 (see Evidence).

**TO-VERIFY before Accepted:**
- the CoDEx paper table number for 0.337 (arXiv:2009.07810; a summariser said Table 4);
- Lacroix 2018 (arXiv:1806.07297) Table 2 WN18RR 0.48 vs the README's 0.49 (the README value is used);
- kbc YAGO3-10 0.58 (kbc README results, or Lacroix 2018 arXiv:1806.07297 Table 2).

**Anchor tie handling, from code.** Every anchor is worst-rank or rounded mean
rank: kbc `kbc/models.py:68-69`; ssl-RP `src/models.py:107-108`; IVR
`main.py:96-99` @ `adeb78e` (filter includes the target, so `>=` is worst rank);
LibKGE `config-default.yaml` `tie_handling: rounded_mean_rank` @ `9bca31f`.

**`n_anchor`.** IVR is 3, as the paper states. Chen 2021 and DURA state no seed
count, so both get the conservative value 1.

**Tier REA verdict rule (P0-2, P0-3).** Applies to FB15k-237, WN18RR and CoDEx-M.
1. **Tie-break.** The verdict uses `TieBreak::Bottom` (worst rank, `eval/rank.rs:80`).
   RANDOM is reported alongside. An M0 CI test asserts Random ≤ Bottom per query.
2. **Seeds.** n = 5 pre-registered seeds {0,1,2,3,4}, with seed mean m and std s.
3. **Anchor-noise-aware bound.** L = m − t_{1−α_d}(n−1) · s · √(1/n + 1/n_anchor).
   The anchor's seed variance is unknown, so it is taken equal to ours.
4. **Minimum effect δ.** REA holds only if L > anchor + δ_d, with δ_d one
   test-set standard error of MRR, sd(RR)/√(n_test_queries). ASSUMED
   placeholders (sd(RR) ≈ 0.4): FB15k-237 +0.002 (40,932 queries), WN18RR
   +0.006 (6,268), CoDEx-M +0.003 (20,622). M4 measures sd(RR) on **valid**
   per-query RRs (test cannot be read before `--final`); δ is frozen at M5.
5. **Holm correction** across the 3 datasets at family α = 0.05. The one-sided p
   for dataset d comes from T_d = (m − anchor − δ_d) / (s·√(1/n + 1/n_anchor))
   with n−1 df. Sort the p values and compare them in turn with α/3, α/2, α/1.
   α_d in item 3 is the Holm level each dataset is tested at. YAGO3-10 is
   informational and excluded.
6. **Outcome and fallback.** A dataset that fails is reported, as measured, as
   "matches" (at or above the matches line) or "Tier A", never "exceeds".
   Negative results are published. Said in advance: a +0.001 to +0.005 margin
   will almost always land as "matches"; with n_anchor = 1 the bound is about
   2.4× wider than the seed-only CI; REA on CoDEx-M (RP adds +0.001) is
   expected to be out of reach.

### 2. Evaluation protocol (frozen by the tag)

1. **Filtered ranking** over all entities, per side. The filter is
   train ∪ valid ∪ transfer ∪ test (LibKGE `filter_with_test: True`, kbc). The
   target itself is never filtered.
2. **Ties.** Both worst-rank (Bottom, used for the verdict) and RANDOM
   (`rank = 1 + #greater + U{0..tied}`) are computed from the same scores in one
   pass. Non-finite scores are an error. A NaN target returns
   `KgeError::NonFinite`. A NaN candidate ranks as worst, and a test shows an
   all-NaN scorer cannot reach MRR 1.
3. **Full splits only.** `--limit` runs are never gated or published against
   these targets.
4. **Declared deviations.**
   - "valid" is the post-carve 70% (about 30% is sha256-bucketed into
     `transfer`); HPO and checkpoint selection read it. It can only change
     *our* model selection, not the anchors' numbers. Test is untouched.
   - HPO runs stop at 100 epochs; final runs use the anchors' 500 (§3).
   - Test triples with unseen entities are **kept and ranked** (MEASURED: 28 in
     FB15k-237, 210 in WN18RR, 0 in CoDEx-M), as LibKGE and PyKEEN do.
   - **WN18RR variant.** The anchors (ssl-RP README "#Ent 40,943" and
     `ComplEx(sizes=[40943,22,40943])`; kbc, IVR, DURA, LibKGE) use the
     ID-based ConvE release. The pre-M1 loader read villmow `WN18RR/text`:
     same 86,835/3,034/3,134 triples, in the same order with the same
     relations, but 41,105 entities (MEASURED). ConvE ids are WordNet offsets,
     unique only per part of speech, so 161 ids each denote 2–3 synsets that
     the text release names apart. That is a different entity set and
     candidate count; no WN18RR REA or "matches" claim is valid on it. The
     loader now reads the ConvE release (40,943 entities, 11 relations).
   - **Sources (M1).** FB15k-237, WN18RR and YAGO3-10 are fetched from
     `DeepGraphLearning/KnowledgeGraphEmbedding` pinned at commit `2e440e0f`,
     and CoDEx-M from `tsafavi/codex` at `3132e426`. The URLs name a commit,
     never a branch, and every file is sha256-pinned. The first three are
     byte-identical to the members of the `TimDettmers/ConvE@f3c0eb28`
     tarballs (MEASURED). ConvE's FB15k-237 files use CRLF line endings. The
     earlier villmow FB15k-237 pins hash the same triples with LF endings, and
     the parsed `splits_hash` is identical (full `e800ac1d…`; `--limit 500`
     `57cd8b03…`, which reproduces the committed slice receipt). A reported
     FB15k-237 "sha256 mismatch, got `b664af6c…`" was not upstream drift:
     `b664af6c…` is sha256 of `a\tr\tb\n`, the tampered cache file written by
     the M0 fail-closed test. The mismatch error now says whether the bytes
     came from the cache or the network.
5. **Leakage and identity.** The Rust bench asserts, by triple hash, that
   train ∩ (valid ∪ transfer ∪ test) = ∅ and that the filter set equals the
   four-split union. It asserts the entity count equals the anchor's
   (FB15k-237 14,541; WN18RR 40,943; CoDEx-M 17,050; ssl-RP README) and refuses
   otherwise. The JS loaders already assert this, together with split sizes and
   relation counts (FB15k-237 272,115/17,535/20,466 and 237; WN18RR
   86,835/3,034/3,134 and 11; CoDEx-M 185,584/10,310/10,311 and 51, plus
   10,310/10,311 hard negatives; YAGO3-10 1,079,040/5,000/5,000, 123,182
   entities and 37 relations). It records all four split hashes and a hash of the sorted
   entity-vocabulary order.
6. **Test is read once, checked against origin (P0-5).**
   - Selection reads valid only. This amends ADR-006 Protocol 1 for benchmark claims.
   - The record of truth is on `origin`, on the protected branch
     `kge-final-ledger` (no force-push, no deletion), under
     `npm/packages/kge/bench/results/final/`:
     - `selection.json`: exactly one `config_hash` per dataset (valid-MRR argmax
       over C1-C8, seed 100) with its HPO receipt hashes. It is pushed once,
       after M7 HPO and before the first `--final`, and never changes after that;
     - `ledger.jsonl`: append-only `intent` and `result` lines keyed by
       (dataset, seed).
   - `ruvector-kge-bench --final` refuses unless: tag `kge-prereg-v1` exists on
     origin (`git ls-remote`), passes `git verify-tag` against the signer
     fingerprint committed at the tag, and is an ancestor of HEAD; after
     `git fetch origin kge-final-ledger`, **origin's** files (never the working
     tree) show that the config_hash equals the dataset's `selection.json` entry,
     the seed is in {0,1,2,3,4}, and there is no line for (dataset, seed).
   - Before scoring test, it appends an `intent` line (key, config_hash, tag SHA,
     checkpoint sha256, UTC) and pushes it as a fast-forward. If the push is
     rejected or fails, nothing is scored. The `result` line (receipt sha256) is
     pushed after scoring. A retry of an intent with no result pushes **no new
     intent**: it refuses unless the checkpoint sha256 matches origin's intent,
     then pushes only the `result`. Scoring is deterministic (RANDOM is seeded
     per query), so the retry cannot shop.
   - CI on the branch fails if a line is modified or deleted, a dataset has more
     than one config_hash or one that differs from `selection.json`, a seed is
     outside {0..4}, a (dataset, seed) has a second intent or result, a result
     has no intent, a tag SHA differs, or `selection.json` changes after the
     first intent. A REA or "matches" claim needs exactly five results, seeds
     {0,1,2,3,4}.
   - Scope. This binds the bench tool and the published record. It cannot stop
     someone scoring test with their own code. Such numbers have no ledger line
     and are never published.
7. **Receipts** (`receipt@1`) add the tag SHA, crate git SHA, binary and
   receipt-body sha256, host CPU/GPU, wall time, $ cost, the full TrainConfig
   with `complex_rank` (§3), and **the per-query Bottom and RANDOM rank vectors
   plus their sha256**. The absolute `binding_source` path is dropped.
8. **Post-hoc verification never re-scores test (P0-6).** `--verify-final
   <receipt>` recomputes per-query ranks from the exported tables and passes if
   their sha256 matches the receipt. It is logged as `verification`, never
   `final`, and writes no ledger line.

### 3. Training recipe and pre-registered search (P1-10, P1-11)

**Rank unit.** "Rank" means `complex_rank` k: complex coordinates per embedding,
2k reals. kbc and ssl-RP "rank 1000" is complex. Receipts record `complex_rank`,
and HolE figures are always labelled in reals.

**Base recipe (ComplEx-N3-R; Lacroix 2018 / Chen 2021), in Rust in `ruvector-kge`:**
- reciprocal relations (2·|R| rows; head queries answered as `(o, r⁻¹, ?)`);
- 1-vs-All full-softmax cross-entropy;
- N3 over complex moduli Σ|z_j|³, with loss and regulariser batch-averaged;
- Adagrad, lr 0.1, init N(0,1)·1e-3;
- **epochs.** Final seeds run the anchor's own **500 epochs** on all three
  datasets (ssl-RP best-run docs: FB 500 epochs "about 11.5 hours", WN18RR
  "about 7 hours", CoDEx-M "about 9.5 hours"), keeping the best-on-valid
  checkpoint with no early stop. HPO runs (seed 100) are capped at 100 epochs
  with early stop on valid MRR. That cap is a **declared deviation** that
  affects only which config is selected, never the final runs.

**Pre-registered configs: exactly N = 8 per dataset.**

C1 is the anchor's own best run for that dataset, re-read from ssl-RP
`doc/hyper-parameters/{fb15k237,wn18rr,codex}.md` "Best Run". All three use
k=1000 and lr 0.1. C2-C8 change one or two axes from **that dataset's** C1.

| Config | FB15k-237 | WN18RR | CoDEx-M |
|---|---|---|---|
| **C1** (anchor) | batch 1000, λ 0.05, w_rel 4 | batch 100, λ 0.10, w_rel 0.05 | batch 500, λ 0.01, w_rel 0.125 |
| C2 | w_rel 0 (RP off) | w_rel 0 | w_rel 0 |
| C3 | λ 0.01 | λ 0.05 | λ 0.005 |
| C4 | λ 0.1 | λ 0.5 | λ 0.05 |
| C5 | w_rel 1 | w_rel 0.5 | w_rel 0.0625 (below the grid edge the doc flags) |
| C6 | batch 100 | batch 1000 | batch 1000 |
| C7 | batch 100, λ 0.1 | batch 500, λ 0.05 | batch 100, λ 0.005 |
| C8 | k 2000 | k 2000 | k 2000 |

- **HPO and selection.** Seed 100 (disjoint from final seeds {0..4}), valid MRR
  only. The ADR-004 campaign gate uses α/N = 0.05/8 = 0.00625 per proposal
  (replacing the fresh α at `gate.rs:57`).
- **What goes to `--final`.** Only the config in `selection.json` (§2.6).
- **Removed.** KvsAll and DURA are no longer arms. They would exceed N and are
  not needed for REA.
- **Out of scope.** Faithful RotatE is out of scope. The current RotatE
  (`rotate.rs:145-148`, overall L2) is excluded from every claim.

### 4. Compute plan and budget

**Local first.** The kernel becomes a batched GEMM (logits = Q·Eᵀ):
- a pure-Rust `matrixmultiply`/`gemm` core, with rayon behind a `parallel`
  feature that is off for wasm32;
- optional `cuda` (cudarc or candle, Rust-only) **in the bench crate only**. A
  local RTX 5080 needs CUDA ≥ 12.8 (sm_120). The runner's 12.4 image is fine for
  rented 4090s. Both images are pinned by digest.

ASSUMED FLOPs per epoch at k=1000 (reciprocal, forward plus two backward
GEMMs): FB15k-237 about 1.0e14, WN18RR about 0.9e14, CoDEx-M about 0.8e14,
YAGO3-10 about 3e15. Campaign volume is 3 × 8 × ≤ 100 = 2,400 HPO epochs
plus 3 × 5 × 500 = 7,500 final epochs, 9,900 dataset-epochs in all, or about
9e17 FLOPs (C8 at k=2000 adds about 3%). M4 measurements replace these.

**Spend gates.** No rental beyond the (d) probe starts until all four hold:
- **(a)** **M4 measures** a full FB15k-237 epoch at complex_rank 1000 on ruvultra
  (CPU or local GPU) at ≤ 15 min. This is a kernel sanity floor only; at 15
  min the campaign is about 2,500 h. The M2 projection is informational only.
- **(b)** M4 reaches WN18RR **valid** MRR ≥ 0.47 at its C1 within 100 epochs.
- **(c)** The tag `kge-prereg-v1` exists on origin.
- **(d)** $/epoch, measured by one ≤ $2 probe job after (a)-(c) (counted
  against the cap), × the remaining epochs is ≤ $150. If not, the figure goes
  to the user and nothing else launches.

**72-hour rule.** Everything runs locally at $0 if the 9,900 dataset-epochs
fit in about 72 h on ruvultra, which needs a measured mean of about 26 s per
epoch. For reference, the anchors' own reported speed (FB 83 s, WN18RR 50 s,
CoDEx-M 68 s per epoch) gives about 140 h for the final seeds alone and about
185 h in all, so the rule is **expected to fail** unless our kernel is about
2.6× faster than the anchors' hardware. vast.ai is then used under gate (d),
and the measurement that justified each rental goes in the ledger. The user
asked for vast.ai, so this trade-off is reported to them, not decided silently.

**Runner.** If CUDA wins, the GPU runner is used unchanged; a CPU mode is built
only if the CPU kernel wins and rental is still justified. Either way M6 adds
checkpoint safety (P1-8): per-epoch checkpoint upload to a rolling artifact slot,
and `--resume-from <artifact>`, which fetches into `$RVGR_ARTIFACT_DIR` before
the job command runs. Jobs run under 11.5 h (a 500-epoch run spans jobs via
resume). `--require-datacenter` is set where offers allow it.

**YAGO3-10.** It is pinned first (sha256), then gets at most 1 seed, last, and
only if budget remains.

**Hard cap: $250 total vast.ai spend**, enforced by runner `--max-usd` and a
per-job $ ledger: ≤ $150 for benchmark runs under gate (d); the remaining ~$100
is reserve that needs explicit user sign-off. Crossing the cap stops all
launches; a run in flight finishes its current checkpoint and halts.
**The brain lane has no rental budget (P1-9):** brain data never leaves
ruvultra or the brain's own Cloud Run project, and B3 trains there at $0.

### 5. Publish rules

- **No "SOTA", "beyond SOTA" or "state of the art" wording.** The strongest
  allowed wording is "exceeds the published <anchor> figure (<citation>) under
  worst-rank evaluation, n=5 seeds, Holm-corrected". It is used only for
  datasets where REA passed on `--final` receipts that descend from the tag.
  Otherwise the wording is "matches" or "Tier A".
- **Results table.** Tier verdicts, both tie-breaks, n_anchor, δ and L per
  dataset, mean ± std, 95% CI, H@1/3/10, head/tail, the NBFNet context row,
  MEASURED vs EXTRAPOLATED labels.
- **Weights.** Tables-only export: an f32 blob plus a manifest (scorer,
  complex_rank, reciprocal, seed, vocab-order hash, dataset pins, receipt hash),
  **no triples or labels** (the FFI envelope embeds triples and is never
  published). Hugging Face only after the per-dataset id-licence decision (a
  blocking M5 item with a named owner), with a model card; else receipts only.
- **Packages** (plan M9, F3). npm 0.2.0 with `--provenance`, CI-built platform
  packages and `SHA256SUMS`; `build-kge.yml` refuses to republish 0.1.0. Crates
  in order (`ruvector-kge`, then `-ffi`/`-wasm`), per-crate versions past 2.3.0.
  Publishing depends on a green, non-report-only full-suite job.
- **Brain numbers are never presented as benchmark numbers.**

### 6. pi.ruv.io brain integration (separate lane, separate gate)

**The user's idea, restated from the code.** (1) *"Should be linked but
aren't"* has no ground truth (the only memory→memory relation is cosine ≥ 0.55);
it is **deferred** until a non-embedding relation is persisted (the dead
`LinkEdge`/`store_link_edge`, or accepted curation decisions). (2) *Relation
dedup* has about 4-5 relation types; the real population is **tags**, so it
becomes tag-synonym merging. (3) *Untyped multi-hop* already ships (cosine +
PPR); only typed composition is new, and it must beat co-tag joins.

**Stages**, each gated by the one before.
- **B0: measure (read-only).** Tag counts and Zipf shape, Custom category
  count, contributor mix, weight > 0.9 edge share, `dp_proof` null rate,
  `RVF_DP_ENABLED`, visibility/deletion flags. Go if ≥ ~1k tags have ≥ 3 uses.
- **B1: near-duplicate merge queue, no model**, from weight > 0.9 edges plus a
  title/content hash. It ships even if B3 is killed.
- **B2: typed triple export.** A new fail-**closed** `GET /internal/kge/export`
  (401/503 with the key unset; it does not reuse `verify_system_key`, whose
  fail-open defect is fixed or filed first). It snapshots under the graph
  **read** lock and emits ids, category, normalised tags (min-count 3, re-run
  through `PiiStripper`) and salted contributor hashes, with **no content** and
  no hidden or deleted memories. It refuses if production DP is on (KGE training
  is **not** DP). Schema v1: `(memory, in_category, category)`,
  `(memory, has_tag, tag_norm)`, and `(memory, similar_to, memory)` top-10 as
  context only. `contributed_by`/voter edges are added only if B3 fails without
  them, and are never served.
- **B3: held-out evaluation, on ruvultra or Cloud Run only.** A temporal
  splitter by `created_at` in `ruvector-kge/src/data.rs`; targets are
  has_tag/in_category edges of test-period memories; a held-out `similar_to`
  pair drops both orientations, and has_tag gets no inverse. Baselines:
  popularity, HashEmbedder kNN vote, co-tag Adamic-Adar, label propagation,
  semantic re-embedding kNN. **Ship gate:** KGE ≥ best baseline **+0.02
  filtered MRR** on the temporal test **and** human P@20 on 200 items ≥ 0.6.
  Otherwise **kill**, keeping B1.
- **B4: serve (only if B3 ships).** A separate `Arc<RwLock<KgeModel>>`, trained
  on a snapshot and swapped (F2), never taking the graph write lock (tested).
  `/v1/kge/suggest-tags`, MCP `brain_suggest_tags`, `/v1/reclassify` flags and
  `brain_similar_tags` all feed a moderator queue; nothing auto-writes.
  `ruvbrain_worker` is verified writer-free before it hosts training.

**Privacy.** `predict` never returns contributor or voter entities (tested).
Suggest-tags never returns a tag used by fewer than k = 3 distinct contributors
(tested). Exports stay internal; brain tables stay private (RVF `.rvgraph`).

**Flag for the user (not edited here).** The project CLAUDE.md says brain
"embeddings are noised". The code defaults DP off, and 1.2M edges at 0.55 is
inconsistent with σ ≈ 4.84 noise. B0 verifies this.

### 7. Rollback (P0-7)

- **npm.** `npm deprecate @ruvector/kge@0.2.0 "<reason>"` (platform packages in
  step) plus a fixed 0.2.1. Never plan on `npm unpublish` (72 h, no dependents).
- **crates.io.** `cargo yank --version <v>` per affected crate, then a patch
  release. Yanking does not delete; lockfile-pinned dependents keep working.
- **Hugging Face.** Revert the weights commit; mark the card **RETRACTED** with
  a link to the erratum.
- **Errata.** The release-manager owns retraction: append to
  `bench/results/final/ERRATA.md` and mark the table and PR/README row
  "retracted". Ledger lines are never deleted.
- **Brain.** Env kill-switch `KGE_ENABLED=false` (an env-only Cloud Run
  revision) disables every KGE route and MCP tool. Served models are pinned by
  version; rollback hot-swaps the previous `Arc<RwLock<KgeModel>>` and purges
  queue items by model version.
- **Brain deletions.** `fit_delta` cannot remove a deleted memory's influence,
  so the SLA is a **full retrain within 7 days of any `brain_delete`** (model
  version bumped); until then, suggestions touching a deleted id are filtered.

### 8. Required upstream fixes and measured wins (specified in the plan)

These are deliverables of this campaign, not optional clean-ups.
- **F1 table-size cap:** `dims` and `entities × dims` enforced in the
  constructor, `grow` and load, with a typed error.
- **F2 snapshot training:** train on a snapshot, swap under a brief write lock;
  predict is never blocked by training.
- **F3 publish provenance:** npm `--provenance` from CI, CI-built platform
  packages, `SHA256SUMS`, no republish of 0.1.0.
- **W1 SelfAdversarial:** an arm of the npm-default (HolE) bench suite, not one
  of C1-C8; default only if valid MRR holds; speed-up reconciled in M1.
- **W2 auto search mode:** exhaustive below a measured entity threshold, ANN above.

## Consequences

- No benchmark number exists until M0-M5 land: 188 engineer-hours of $0 work
  before any rental, or 212 with M4b (plan Totals).
- Published numbers are tag-pre-registered, use one pushed-before-scoring
  selected config per dataset, are scored once per (dataset, seed) against
  origin's ledger, use n = 5 seeds, and carry an anchor-noise-aware, Holm-corrected bound.
- The strongest possible claim is "exceeds the published anchor". A "beyond
  SOTA" claim needs a new ingredient and ADR-008.
- REA may well be missed; "matches" and Tier A wording is pre-committed.
- The core stays wasm-safe and fs-free, with CUDA only in the bench crate.
- The brain lane may end at B1. That is an accepted outcome.

## Alternatives considered

- **Tier B option (a) with HolE half-parameters, or RP + DURA/IVR stacking.**
  Rejected (§1). The first is not an accuracy hypothesis. The second only
  recombines published ingredients, with no mechanism aimed at δ. Either would
  make "beyond" a relabelled seed or HPO win.
- **WN18RR anchor 0.494 (ComplEx-IVR) vs 0.501 (TuckER-IVR).** The class-max
  rule gives 0.501. Narrowing the class to the ComplEx family only makes the
  claim easier (anchor-shopping). The same rule gives 0.388 on FB15k-237,
  because no class row found exceeds Chen 2021.
- **Rejected statistics and procedure.** Our RANDOM tie-break against the
  anchors' worst rank (handicaps the anchors). A bare "mean > anchor", a fixed
  ≤ 0.005 margin, or a seed-only CI (each treats a single-run anchor as exact).
  A commit date, a local counter or a working-tree ledger keyed on config (a
  fresh checkout forges or resets them, and per-config keys allow config and
  seed shopping). Test scored twice per campaign (ADR-004/006). A 100-epoch cap
  on final runs (the anchors trained 500).
- **Rejected compute and scope.** Unconditional rental (gated by the 72-hour
  rule and gate (d)). YAGO3-10 in the paid campaign (about 30× FB15k-237,
  informational). A $10 rental for brain B3 (brain data to a third party).
  Training on the brain's similarity edges as phrased (no ground truth).

## Evidence

- **Literature, re-read 2026-09-26.**
  - ssl-RP README: FB .366 / .388 (RP); WN .487 / .488; CoDEx-M .351 / .352;
    "#Ent" 14,541 / 40,943 / 17,050.
  - ssl-RP `doc/hyper-parameters/*.md` "Best Run", all rank 1000, lr 0.1,
    500 epochs: `fb15k237.md` batch 1000, λ 0.05, w_rel 4 (11.5 h);
    `wn18rr.md` batch 100, λ 0.10, w_rel 0.05 (7 h, `sizes=[40943,22,40943]`);
    `codex.md` CoDEx-M batch 500, λ 0.01, w_rel 0.125 (9.5 h).
  - IVR arXiv:2506.02749 Table 1 (3-seed mean); DURA arXiv:2011.05816 Table 2;
    NBFNet arXiv:2106.06935 Table 3.
  - LibKGE README (.348 / .475 / .551); CoDEx README.
  - kbc README: FB .37 at rank 1000; WN .49 at rank 500/2000, batch 100.
  - Hayashi & Shimbo 2017, arXiv:1702.05563; Sun et al. 2020, arXiv:1911.03903.
- **Code verified.** Anchor tie handling (§1) and `rank.rs:80`; `Tables::new`,
  `adversarial::check_limits` (F1) and ffi `train_json`/`predict_json` (F2) at
  `8307c1d4c`; runner `job.rs:110-145`.
- **MEASURED.** 8.3 ms/positive (HolE, 256 reals, 1-vs-all, |E|=2000); 54 vs
  605 ns/score (BatchScorer); slice receipt
  `bench/results/fb15k237-500-hole-2026-09-21.json`; split sizes and
  unseen-entity counts (WN18RR on `original`); villmow WN18RR entities
  `original` 40,943 vs `text` 41,105 (161 merged offset ids); dataset source
  byte-identity with the ConvE tarballs, and the FB15k-237 LF/CRLF
  `splits_hash` equality; vast.ai prices; /v1/status; RTX 5080
  16,303 MiB.
- **EXTRAPOLATED / ASSUMED.** Every full-dataset s/epoch figure and FLOP→time
  conversion of ours; δ (until M4); the W2 threshold; the W1 speed-up (M1).
- **Plan of record:** `docs/research/kge/plan.md`.
- **Fix outside this ADR:** ADR-006:25 should cite "LibKGE RotatE reproduction 0.478" (plan M5).
