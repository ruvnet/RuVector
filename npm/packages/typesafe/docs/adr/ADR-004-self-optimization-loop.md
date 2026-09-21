# ADR-004: The self-optimization loop — governed, measured, reversible

## Status
Proposed

## Date
2026-09-21

## Context

"Self-improving" is easy to claim and easy to fake. ruvector's own ADR-276
(accepted 2026-08-01) reviewed the 2026 literature on memory/trajectory
learning and found: gains are often confound-sized (the embedder choice alone
swings accuracy ±6.2 pp, MemDelta arXiv:2606.29914); utility follows an
inverted U and can drop below baseline after consolidation (arXiv:2605.12978);
and greedy "accept if better" commits 30–42 % false edits (PACE
arXiv:2606.08106). The Jev scorecard run showed the same shape in miniature:
val accuracy rose 87.3 → 92.0 % over 8 generations, but part of the +4.7 pt test
gain was the loop re-introducing generator phrasing that recurs in the test set.

This engine has an advantage ADR-276's subject (an LLM agent) did not: given a
fixed (embedding model, example bank, criteria text), its decisions are
deterministic, so a "measured win" is cheap and repeatable.

## Decision

Four loops, each a *proposal* that must pass one promotion gate.

### Loop 1 — example bank growth (active learning)

- Candidates come from production decisions with low `confidence` or high
  `abstain`, ranked by **margin on cosine-to-centroid** (uncertainty) and
  **core-set / k-center on the embedding** (diversity) — BADGE without the
  gradient step. Iterative: representative sampling on the first round,
  uncertainty afterwards.
- A candidate enters the bank only with a label from trust tier A or B (below)
  and after **noisy-label filtering** (agreement with the current head's
  neighbourhood; disagreement quarantines rather than rejects).
- The bank is **append-only and immutable** (ADR-276 two-tier storage);
  prototypes, probes, and temperatures are *derived artifacts* that point back
  at the examples that produced them.

### Loop 2 — model-arm selection (bandit)

- Arms are embedding models (`bge-small-en-v1.5`, `gte-small`, `e5-small-v2`,
  `all-MiniLM-L6-v2`, `nomic-embed-text-v1.5@256/384`) and, per question, the
  head (prototype vs probe).
- **Thompson sampling** with a cost-aware reward `accuracy − λ·(latency_ms + $)`,
  seeded by a one-time offline evaluation on the deployment's labeled slice so
  cold start is not uniform.
- Start **global**; split per question/task category only after a minimum
  sample count — ADR-271 measured that a per-category router beats a global
  choice by ~2 % but *reverses to hurt* under scarce per-category data.

### Loop 3 — speed self-tuning

- INT8 ONNX by default where the runtime loads it (ADR-002 spike).
- RaBitQ-quantized HNSW (`ruvector-rabitq`) once the bank exceeds a size
  threshold; multi-bit RaBitQ reaches ~96 % recall without a refine pass.
- **Per-query adaptive `ef`** against a declared recall target from offline
  density statistics (arXiv:2512.06636), instead of one global `efSearch`.
- These change latency, not decisions; they are still gated on
  "decisions unchanged on the frozen split" before promotion.

### Loop 4 — criteria mutation (v2)

- Description text (`what`, `not_for`) is the genome; the embedding model and
  head are the frozen scorer. EvoPrompt-style mutation and crossover, each
  variant re-embedded and scored — the metaharness-Darwin pattern ADR-271
  validated for SONA (GA + coordinate descent, held-out gains with 98.6 % less
  forgetting). DSPy/MIPRO, OPRO and ProTeGi do not apply directly: there is no
  LLM in the loop to hold prior candidates in context.
- Off by default in v1; when on, it is the highest-risk loop and gets the
  strictest budget.

### The promotion gate (all loops)

1. **Frozen splits.** Validation and test are fixed at deployment time;
   `assert_train_eval_disjoint` (ruvector `crates/sona/src/darwin_guard.rs`)
   runs on every proposal. Test is scored only for the baseline and the final
   champion of a campaign, never inside the loop.
2. **Sequential, not greedy.** A proposal is promoted only when a **paired
   anytime-valid test** (e-process / testing-by-betting, arXiv:2606.00878,
   2501.03982) on the validation split rejects "no improvement" — never on a
   single accept-if-better comparison. PACE reports this both removes false
   commits and cuts evaluation cost ~18 %.
3. **Transfer holdout.** A second, different-distribution split (ADR-276 §4)
   must not regress beyond a stated tolerance, so the loop cannot overfit the
   validation set's phrasing — the exact failure the Jev scorecard exhibited.
4. **Permanent control arm.** A slice of traffic (default 5 %) always runs the
   pre-mutation configuration. Drift between control and champion is the
   regression alarm; ADR-276 records this catching a 54 % regression that was
   otherwise invisible.
5. **Budget.** Each loop has a per-day evaluation budget (ADR-282 pre-PR gate
   pattern); exhaustion pauses proposals, never lowers the bar.
6. **Trust tiers.** A: programmatic verifiers (held-out accuracy, parity tests).
   B: LLM-judge labels, quarantined until confirmed by A. C: user-supplied text
   (`state`, criteria) — never influences a promotion decision directly.

### Receipts

Every promotion writes an append-only receipt: proposal, parent, mutation,
val/transfer metrics with the test statistic, decision, model manifest hash,
head, temperature, budget consumed, and — in v2 — an RVF lineage link
(`rvf_derive`) and an Ed25519 signature compatible with metaharness's
flywheel receipts, so `typesafe eval --history` and rollback are exact.

## Consequences

- The loop is slower to promote than a naive one, by design. The user-facing
  promise is "never gets worse on your frozen split, and every change is
  explained", not "improves every hour".
- Two extra splits (validation, transfer) and a control arm consume examples
  and traffic; ADR-006 sets minimum sizes below which loops stay in
  propose-only mode.
- Determinism of the engine makes the sequential test cheap; the expensive
  part is labels, which is why Loop 1 is ranked first.

## Alternatives considered

- **Continuous online learning (update prototypes on every decision).** The
  inverted-U evidence and the false-commit rate argue against it; it also
  destroys reproducibility of a given decision.
- **LLM-driven criteria rewriting on each failure (ProTeGi-style).** Needs an
  LLM in the loop and reintroduces per-call cost; kept as an optional mutation
  *proposer* for Loop 4, never as the promoter.
- **Per-tenant model arms from day one.** ADR-271's data says start global.

## Evidence

- ruvector ADR-276 (learning-loop gating, trust tiers, sequential tests,
  control arm), ADR-271 (metaharness-Darwin genome evolution, per-category
  router finding), ADR-282 (pre-PR quality gate), ADR-288 (immutable base +
  deltas); `crates/sona/src/darwin_guard.rs`.
- Active learning: BADGE; NAACL 2025 "Active Few-Shot Learning for Text
  Classification"; noise-robust AL (arXiv:2504.02901); support-set noise
  sensitivity (arXiv:2204.05494).
- Bandits: PromptWise (arXiv:2505.18901), BayesianRouter (arXiv:2510.02850).
- Quantization/autotuning: RaBitQ (SIGMOD 2024, doi:10.1145/3654970);
  INT8 ONNX (ONNX Runtime docs); adaptive ef (arXiv:2512.06636).
- Sequential testing: PACE (arXiv:2606.08106); e-processes (arXiv:2606.00878,
  2501.03982).
