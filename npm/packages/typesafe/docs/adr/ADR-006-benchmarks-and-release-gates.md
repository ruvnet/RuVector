# ADR-006: Benchmarks and release gates

## Status
Implemented and measured (2026-09-21)

## Date
2026-09-21

## Context

This package exists because of numbers measured against Jev (ADR-001). It
must keep measuring, on the same terms, or its claims decay into marketing.
The Jev scorecard run also showed how a benchmark can flatter itself: a loop
that mutates criteria from train examples memorised phrasing that recurred in
the test split, so part of a +4.7 pt gain was not better criteria. Gates must
be designed against that.

## Decision

### Datasets

| Dataset | Size | Source / licence | Used for | Gate |
|---|---|---|---|---|
| Banking77 | 13,083 ex., 77 intents | PolyAI, arXiv:2003.04807 (research use) | `choice`, single domain, many options | accuracy ≥ Jev − 3 pp; macro-F1 reported |
| CLINC150 (+ OOS) | 22,500 in-scope + 1,200 out-of-scope, 150 intents | Larson et al. 2019 (HF hub) | `choice` at high option count; **abstain/OOS** | accuracy ≥ Jev − 3 pp; OOS AUROC ≥ 0.85 |
| HWU64 | 25,716 ex., 64 intents, 21 domains | Liu et al. 2019 (research use) | `choice` across domains | accuracy ≥ Jev − 3 pp |
| CLINC150 OOS + HWU64 scope | as above | as above | `noul` ("is this in scope") | AUROC ≥ 0.85; must beat the majority-class rate |
| tweet_eval sentiment | ~50 k | CC-BY | `score` (ordinal proxy) | informational in v1; gating once CORN lands |
| MASSIVE | ~1 M, 51 languages | CC-BY-4.0, arXiv:2204.08582 | multilingual slice | informational |
| TREC-6/50 | ~5.5 k | public domain | small label sets | informational |
| Synthetic tickets (seeded) | 500, 8 departments, 3 question types | this repo, `bench/data` | the Jev-parity fixture with all three question types | parity with the 2026-09-21 scorecard |

### Protocol

1. **One frozen item set per dataset, one criteria text per option, one
   question batch** — generated once, committed with a content hash, and never
   regenerated between arms. Both arms (Jev live API; local engine) score the
   *identical* JSON.
2. **Few-shot regime follows SetFit's protocol** (8 examples/class unless the
   dataset defines a split); zero-shot is reported separately and never
   compared to Jev's few-shot number.
3. **Splits:** train / calibration / validation / transfer / test, all
   instance-ID disjoint (`assert_train_eval_disjoint`). Transfer is a different
   dataset or domain than validation (ADR-004 gate 3). Test is scored twice per
   campaign: baseline and champion.
4. **Vocabulary-disjointness check** between any synthetic generator and the
   gen-0 criteria; the bench refuses to run if a generator content word appears
   in the starting criteria (the guard that caught six leaks on 2026-09-21).
5. **Calibration is measured, not assumed:** ECE (10 equal-width bins) and
   Brier on `probabilities`/`confidence`; reliability bins stored in the
   receipt; sparse bins flagged.
6. **Latency** is wall-clock at the public API: p50/p95 per decision at
   concurrency 1 and 8, native and WASM, plus embeds/s and decisions/s.
   Jev's numbers include the network round trip and are labelled so.
7. **Cost:** Jev from `usage` tokens at its listed input rate (output rate
   unlisted — stated); local as CPU-seconds per 1k decisions and a $ figure at a
   stated cloud vCPU price.
8. **Adversarial set:** for each dataset, 200 items with another option's
   example phrases stuffed into `state` (ADR-005 steering). Reported as
   accuracy *and* mean `confidence` on the flipped items — the gate is that
   confidence drops, not that accuracy is perfect.

### Release gates (all must pass to publish)

| Gate | Threshold |
|---|---|
| Accuracy vs Jev (mean of Banking77, CLINC150, HWU64, few-shot) | ≥ Jev − 3 pp |
| Calibration | ECE ≤ 0.05 on test; `calibrated: false` never reported as a probability |
| Out-of-scope / `noul` | AUROC ≥ 0.85 on CLINC150 OOS; beats majority baseline on every predicate |
| Latency, native | p95 ≤ 50 ms per decision at concurrency 1 (bge-small, INT8 or FP32) |
| Latency, WASM | p95 ≤ 150 ms |
| WASM ↔ native parity | embeddings cosine ≥ 0.9999 on the probe set; decisions identical |
| Loop safety | no regression on the transfer split beyond 1 pp; control-arm drift alarm exercised in CI |
| Security | ADR-005 CI assertions (no net symbols, no WASI fs/net imports, no shell) green |
| Provenance | platform packages resolvable on npm before the meta-package bump (`optional-deps-resolvable-on-npm`) |

### Measured 2026-09-21

On the frozen tickets fixture (`department` choice; 137 usable training
examples). Receipts: `bench/results/tickets-onnx-bge-2026-09-21.json` (16-shot),
`bench/results/optimize-tickets-2026-09-21.json` + `optimize-receipts-2026-09-21.jsonl`
(full-data campaign, four model arms).

| configuration | accuracy | ECE | p95 ms | gate status |
|---|---|---|---|---|
| Jev replay (reference) | 85.3% | 0.073 | 231 | — |
| bge-small, 16-shot | 80.0% | 0.075 | 10.0 | accuracy FAIL, ECE FAIL |
| campaign champion bge-small (full data) | 83.3% | 0.068 | 10 | accuracy PASS, ECE FAIL |
| campaign champion bge-small-int8 (full data) | 84.0% | 0.071 | 4 | accuracy PASS, ECE FAIL |
| campaign champion MiniLM (full data) | 81.3% | 0.057 | — | ECE FAIL |

`accuracy_vs_jev` passes with the full-data campaign champion (83.3 % ≥ 82.3 %);
native latency passes (p95 ≤ 50 ms). `calibration_ece` does not pass in any
regime (best 0.057 > 0.05), nor does accuracy in the 16-shot regime. All seven
campaign promotions came through the calibration criterion (gate 2b); the
accuracy paired test alone rejected each (best wealth 4.91 of the 20 threshold).

### Where it lives

`npm/packages/typesafe/bench/`: dataset fetchers (hash-verified), fixtures,
the harness (`typesafe bench` / `typesafe eval`), the Jev baseline JSON from
2026-09-21, and one receipt per run (signed in v2). CI runs the synthetic
fixture and a 1 k-item slice of Banking77/CLINC150 on every PR; the full suite
runs nightly and before any publish, and the metaharness `bench --op verify`
suite hash is pinned so evolution and evaluation cannot drift apart.

## Consequences

- Publishing is gated by measurements that take minutes, not by "tests pass".
  That is the intended friction.
- Jev numbers age: the baseline JSON is dated and re-run quarterly (a paid
  call); gates compare against the most recent dated baseline.
- Some datasets are research-use licensed; the fetchers download from the
  canonical sources at bench time and nothing is redistributed in the package.

## Alternatives considered

- **Accuracy-only gates.** Would have passed the Jev scorecard's memorised
  gain; the transfer split and vocabulary check exist precisely to fail it.
- **Synthetic data only.** Reproducible but self-referential; public intent
  datasets keep the claims comparable to the literature and to Jev.

## Evidence

- SetFit protocol (arXiv:2209.11055, github.com/huggingface/setfit); DNNC for
  threshold-based OOS (arXiv:2010.13009).
- OOD/AUROC framing: arXiv:2206.09387. ECE/Brier definitions: Guo et al. 2017.
- Datasets: Banking77 arXiv:2003.04807; CLINC150 Larson et al. 2019; HWU64 Liu
  et al. 2019; MASSIVE arXiv:2204.08582.
- The memorisation finding: `bench/jev-baseline-2026-09-21.json` and the
  scorecard report.
