# ADR-272: `ruvector-tabfm` — A Zero-Shot Tabular Foundation Model, and Pointing It Inward at the DB's Own Tuning

- **Status**: Proposed — **gated, with a viable pivot found**. Gate #1 (weight license) resolved 2026-07-01: TabFM's own weights are **non-commercial / non-redistributable → NO-GO to ship**. But **`TabPFN-v2` weights (clf + reg) are Apache-2.0-with-attribution ("Prior Labs License", id `priorlabs-1-1`) → commercially shippable and redistributable.** So the product path is *port TabPFN-v2 timesfm-style* + adopt TabFM's Apache-2.0 architectural ideas, not "train from scratch." See "Gate #1 resolution" below and `docs/research/tabfm/PLAN.md`.
- **Date**: 2026-07-01
- **External anchor**: Google Research, "Introducing TabFM: a zero-shot foundation model for tabular data" (research.google, 2026)
- **Mirrors**: ADR-191 line — `ruvector-timesfm` (candle port of TimesFM, wired inward into `sweep::EarlyStopper` / `rebuild` / `anomaly`)
- **Touches**: SONA (ADR-271), IVF/PQ tuning line (ADR-205/206), `ruvector-postgres` (143 SQL functions), `ruvector-attention`

---

## Context

**TabFM** is a foundation model for tabular classification/regression that predicts **zero-shot via in-context learning (ICL)**: hand it a table of labeled training rows + unlabeled test rows as one prompt, and it predicts in a **single forward pass** — no per-dataset training, hyper-parameter search, or feature engineering. Its architecture is three stages: (1) **alternating row↔column attention** over the raw grid (captures feature interactions natively, replacing manual feature crafting); (2) **row compression** — each contextualized row collapses to one dense vector; (3) an **ICL transformer over the compressed row-vectors** (cheap; scales to larger tables). It is pretrained on hundreds of millions of **synthetic tables** from structural causal models, is order-invariant, beats heavily-tuned XGBoost zero-shot on TabArena (38 classification + 13 regression datasets, 700–150k rows), and Google ships it in BigQuery as an `AI.PREDICT` SQL call.

Two facts make this a high-leverage fit for RuVector, not a novelty:

1. **We already did this for time series.** `crates/ruvector-timesfm` is a parity-validated candle port of TimesFM, and it is not merely a user feature — it is wired into the DB's *own* machinery: `sweep::EarlyStopper` (ADR-191) kills doomed tuning runs, `rebuild` forecasts recall-drift to time HNSW rebuilds, `anomaly` watches telemetry. **TabFM is the tabular sibling of TimesFM**, and the same "port it *and* point it inward" playbook applies.
2. **Every ingredient exists and no tabular work does.** A grep of `crates/`, `npm/`, `docs/` finds zero TabPFN/TabFM/tabular-ICL work — this is greenfield. Meanwhile the stack is already here: candle transformers (6 crates), a modular attention SDK (`ruvector-attention`: graph/sparse/MoE/training), quantization (RaBitQ/PQ), WASM builds, and a 143-function Postgres surface that is the natural home for a local `AI.PREDICT`.

The open question this ADR answers: *which* TabFM-derived capabilities are worth building first, and in what order.

## Decision

Create **`crates/ruvector-tabfm`**, structured as a deliberate mirror of `ruvector-timesfm` (feature-gated `candle` numeric path; load-weights-once predictor; JSON-in/JSON-out CLI = MCP tool; WASM build). Scope the first release to **inference against published weights only** — porting the architecture, *not* training our own model. Prioritize the four highest-impact surfaces below; explicitly defer the two long-tail items.

### 1. Core inference port (foundation — everything depends on it)
Port TabFM's three-stage forward pass into candle inside `ruvector-tabfm`:
- `predictor.rs`: `TabFm::load(weights, device)` → `predict(train_rows, train_labels, test_rows)` → class probabilities (classification) / point + p10..p90 quantiles (regression, reusing the quantile-band idiom already in `ruvector-timesfm::forecast_types`).
- The alternating row↔column attention block is the real engineering. Assess candle op coverage first; if a fused kernel is needed, add it under `ruvector-attention` (which already houses sparse/graph/custom attention) rather than in-crate, so it is reusable.
- `bin/predict.rs`: JSON-in/JSON-out CLI, registered as the `tab_predict` MCP tool (parity with `time_series_forecast`).
- Feature gating identical to timesfm: numeric path behind `candle` (+ `cuda`/`metal`); a stock `cargo build` stays light.

### 2. Row-compression as a native tabular-row embedder (largest DB synergy)
TabFM's stage-2 already maps a heterogeneous row (mixed numeric/categorical) to **one dense vector**. That is exactly the "how do I embed a structured record?" problem a vector DB faces. Expose stage-2 as a standalone `encode_row(row) -> Vec<f32>` so a user can `INSERT` a SQL row and get an **indexable embedding** — no external embedding model, no feature engineering — that flows straight into the existing HNSW/DiskANN index **untouched**. This is a genuinely new retrieval modality (structured-record search) for near-zero marginal cost once §1 exists.

### 3. Point TabFM inward — zero-shot index/SONA autotuner (the "make the DB smarter" play)
Choosing HNSW `M`/`efConstruction`, IVF `nprobe`, and quantization level from workload features **is a tabular prediction problem**. SONA (ADR-271) learns this online but cold-starts blind; the IVF/PQ tuning line (ADR-205/206) established that these knobs are workload-dependent and non-differentiable. Feed TabFM a small table of `(workload features → best config)` from tuning history and take its **zero-shot warm-start** prediction — the exact analog of `timesfm::sweep::EarlyStopper` gating optimization runs. This is the differentiator: a self-optimizing DB that cold-starts *near-optimal* instead of blind.

### 4. Postgres `ruvector_tab_predict()` — a local, free `AI.PREDICT` (headline feature)
`SELECT ruvector_tab_predict('churn', ROW(...))` inside Postgres is BigQuery's `AI.PREDICT`, but local, free, and offline — a headline that fits the README's "local AI, PostgreSQL built in" story. A thin SQL wrapper over §1; low effort, high narrative value.

### Explicitly deferred (out of scope for the first release)
- **Structured/tabular reranker** alongside `ruvector-gnn-rerank` (row-column attention over `(query, candidate-metadata)`): promising and complementary to the GNN, but adds an ICL pass to the per-query hot path — revisit after the core lands and latency is measured.

### Gate #1 resolution (2026-07-01) — the split license reshapes everything above

Verified against the primary sources:
- **Code** — `github.com/google-research/tabfm` is **Apache-2.0** ("not an officially supported Google product"; JAX + PyTorch backends; weights auto-download from HF).
- **Weights** — `google/tabfm-1.0.0-pytorch` ship under **TabFM Non-Commercial License v1.0**. Verbatim-confirmed terms: *"You may only access, use, or create Derivatives of the TabFM Model for Non-Commercial Purposes"*; commercial = *"any revenue-generating activity … interactions with end users or production systems … or to train, fine-tune, or distill other models for commercial use"*; *"You will not … Distribute the TabFM Model or a Derivative"*; *"Any restrictions set forth herein … also apply to any Derivative."*

**Consequence for §1–§4 as written:** a candle port of the published weights is a *Derivative*, so it inherits the non-commercial + no-redistribution terms. That is incompatible with RuVector's identity (MIT/Apache, "free forever," bundled in `npx ruvector`, powering the commercial Cognitum product). Therefore the naive "port the weights" plan **cannot ship**. This splits the work into two tracks:

- **Track A — Research (allowed now, published weights):** use TabFM strictly for **internal benchmarking/research** (the license's own non-commercial examples) to *quantify the ceiling* — does §3's TabFM-predicted config beat SONA? is §2's row embedding competitive? — **provided results do not feed commercial decision-making, paid products, or production**. No bundling, no redistribution, no production inference. This is a measurement exercise, not a feature.
- **Track B — Product (the shippable path):** **port `TabPFN-v2` weights** — the classifier and regressor are both under the **Prior Labs License = Apache-2.0 + attribution** (verbatim obligations: display *"Built with PriorLabs-TabPFN"* in docs/UI; if we distill/fine-tune into a *distributed* model, prefix its name with *"TabPFN"*; internal benchmarking needs no attribution). These weights are **commercial-OK and redistributable**, so we can bundle them in `npx ruvector`, Postgres, WASM, and Cognitum. Port them timesfm-style into candle, and separately adopt TabFM's **Apache-2.0 architectural** ideas (row compression, scale-friendly ICL-over-compressed-rows) as our own optimization on top.

**This resurrects §1–§4 as shippable features** — just built on TabPFN-v2 weights (Apache-2.0+attr) instead of TabFM's non-commercial weights. "Train our own weights from scratch on a synthetic-SCM engine" reverts to a *genuinely optional* research bet (only if we later need to beat TabPFN-v2 or shed the attribution clause), **not** the critical path.

**Licensing scorecard (weights):**

| Model | Weights license | Ship in RuVector? |
|---|---|---|
| TabFM 1.0 | TabFM Non-Commercial v1.0 | ❌ research/benchmark only |
| TabPFN-2.5 / 2.6 / 3 | `tabpfn-2.5-license-v1.1` (non-commercial) | ❌ research/benchmark only |
| **TabPFN-v2 (clf + reg)** | **Prior Labs License (Apache-2.0 + attribution)** | ✅ **yes — Track B base** |

**Revised sequencing:** Track A first (cheap — benchmark TabFM *and* the newer TabPFN weights internally to know the accuracy ceiling we'd forgo by shipping v2) → Track B (port TabPFN-v2 into `ruvector-tabfm`, timesfm-style; layer TabFM's Apache-2.0 architecture ideas) → §2/§4/§3 on the shipped v2 weights.

## Consequences

**Positive**
- Adds a whole capability class (zero-shot tabular ML + structured-record embedding) with **no new heavy runtime** — candle is already a workspace dependency, and the timesfm crate is a proven template.
- §3 turns TabFM into infrastructure the DB uses on itself, compounding the SONA/BET-line investment rather than sitting beside it.
- §4 gives a concrete, demoable, README-worthy feature (`AI.PREDICT` in local Postgres) that no pgvector competitor has.

**Negative / risks**
- **Weight availability & license — RESOLVED, and it's a blocker for the naive path.** The published weights are **non-commercial + non-redistributable**, and derivatives inherit those terms (see "Gate #1 resolution"). We cannot bundle or ship them. Track B (Apache-2.0 architecture + our own weights) is the only commercial path; parity-validation would then be against TabFM's *behavior*, not a redistribution of its weights.
- **Candle op coverage.** The alternating row-column attention may need a custom kernel; budget for a `ruvector-attention` addition.
- **ICL context scaling.** Inference cost/memory grows with the number of in-context training rows; §3's tuning-history tables are small (safe), but §4 user tables need a documented row cap + a subsampling story.
- **§3 needs labeled history.** The autotuner is only as good as the `(features → best config)` table; it is strongest *after* the tuning-history logging exists, so it lands last.

## Implementation status

Not started. This ADR and `docs/research/tabfm/PLAN.md` are the research + plan deliverable.

Gate #1 (weight license) is now **resolved with a viable pivot**: TabFM's own weights can't ship, but **TabPFN-v2 (Apache-2.0 + attribution)** can.

**Track A ran (2026-07-01) → GO.** Empirical benchmark (license-clean; details in `docs/research/tabfm/PLAN.md` §5b): shippable TabPFN-v2, zero-shot on CPU, is **top-or-tied on 5/6** standard OpenML datasets vs **tuned** XGBoost/HistGBM (mean rank 1.33 vs 2.83), with one large multiclass win (`vehicle` +0.034 AUC / +8.5 pts acc) and binary margins small-but-consistent. The gated v2.5/v3 weights refused to load without license acceptance — corroborating the license finding. **Conclusion: the Apache-2.0 model we can ship is good enough to own → proceed to Track B.** Remaining steps:
1. **Track A (research):** benchmark harness to measure the accuracy ceiling of TabFM + newer TabPFN (non-commercial) weights, so we know exactly what we trade away by shipping v2. Cheap; de-risks the bet.
2. **Track B (product):** port TabPFN-v2 (clf + reg) into `crates/ruvector-tabfm` timesfm-style; satisfy the attribution clause ("Built with PriorLabs-TabPFN" in docs; name any distilled distributed model "TabPFN…"); layer TabFM's Apache-2.0 row-compression/ICL ideas.

Naming note: the crate is provisionally `ruvector-tabfm` (the initiative's origin), but since the shipped base is TabPFN-v2, `ruvector-tabpfn` may be the more accurate name — and the attribution clause makes a "TabPFN" association *beneficial*, not just permitted. Decide at scaffold time.

## References

- Google Research — "Introducing TabFM: a zero-shot foundation model for tabular data" (research.google/blog, 2026). Builds on TabPFN and TabICL.
- Code (Apache-2.0): `github.com/google-research/tabfm`. Weights (non-commercial): `huggingface.co/google/tabfm-1.0.0-pytorch`; license text: `.../blob/main/LICENSE` = **TabFM Non-Commercial License v1.0**.
- **Track B base (Apache-2.0 + attribution):** `huggingface.co/Prior-Labs/TabPFN-v2-clf` + `TabPFN-v2-reg`; license = Prior Labs License (`priorlabs-1-1`), text mirrored at `github.com/PriorLabs/TabPFN/blob/main/LICENSE`. Attribution: display "Built with PriorLabs-TabPFN"; prefix any distilled/distributed derivative model name with "TabPFN"; internal benchmarking exempt.
- TabPFN newer weights (**non-commercial**, research-only): `Prior-Labs/tabpfn_2_5` etc., license `tabpfn-2.5-license-v1.1`. Commercial via Prior Labs Enterprise License (`sales@priorlabs.ai`).
- `crates/ruvector-timesfm` — the mirror template (ADR-191 line): `predictor`/`forecaster`, `sweep::EarlyStopper`, `rebuild`, `anomaly`, JSON CLI = MCP tool.
- ADR-271 — Metaharness-Darwin for SONA self-improvement (the online-tuning substrate §3 warm-starts).
- ADR-205/206 — IVF `nprobe` / IVFPQ tuning line (the workload-dependent knobs §3 predicts).
- `crates/ruvector-attention` — home for any custom row-column attention kernel.
- `crates/ruvector-postgres` — surface for §4's `ruvector_tab_predict()`.
- Full landscape, benchmarks, and reimplementation notes: `docs/research/tabfm/PLAN.md`.
