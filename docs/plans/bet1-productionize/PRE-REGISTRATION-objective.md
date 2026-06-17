# BET 1 generality check — is the 40% holding ceiling objective-dependent?

**Status:** Pre-registered (frozen before the run) · **Date:** 2026-06-04 ·
**Research line:** SepRAG (ruvnet/RuVector issue #534) · **Tests an ADR-202 caveat** ·
**Self-contained:** `ruvector-diskann` + `ruvector-gnn` · **Outcome:** ADR-202 addendum.

> ADR-202 established its 40% top-10 churn holding ceiling on **one** learned objective
> (contrastive link-prediction). Its named caveat: "the holding ceiling is objective-dependent."
> This check tests that directly with a *different* objective — **node classification** (real
> ogbn-arxiv 40-class subject labels, cross-entropy on a linear head, embeddings as the
> trainable params). CE-toward-class-separability reorganizes the embedding geometry differently
> from citation-neighbour contrastive learning, so it is a genuine second objective, not a
> reparametrization.

## Thesis (one claim, one number)

> The ADR-202 holding ceiling (reuse within 2% recall@10 of full rebuild) is a property of
> **reuse-under-drift**, not of the link-prediction objective: under a node-classification
> trajectory of comparable churn, reuse holds to a **≥ 30% churn ceiling** and `Periodic{k}`
> recovers the high-churn tail.

## Method

Identical harness, contenders, and 2% gate as ADR-202 (`diskann_real_trajectory.rs`, selected via
an `objective=nodeclass` arg) — **only the trajectory objective changes**. n=20k; recall@10; 200
queries; production Vamana R=32/L=64/α=1.2. Embeddings on the unit sphere (L2 ranking ≡ the metric
the GNN shapes). Precondition (teeth): churn ≥ 15% and the stale control degrades materially —
else VOID.

## Pre-registered outcome criteria (frozen)

- **CONFIRM (generality):** reuse holding ceiling **≥ 30% churn** (within ~10 pts of the 40%
  link-prediction ceiling) **and** `Periodic{k}` recovers the tail within ADR-202's bar (within
  1% of full rebuild at ≤ 50% cost). → ADR-202's objective-dependence caveat is **resolved**; the
  result generalizes across two learned objectives.
- **CAVEAT (objective-dependent — the honest negative):** holding ceiling **< 20% churn**, or
  reuse behaves materially differently (e.g. does not decay, or decays from step 1). → the ceiling
  is objective-specific; reported as a sharpened caveat on ADR-202, not a silent omission.
- **Reported regardless:** the node-class holding ceiling vs the link-prediction 40%, and the
  per-step recall/churn curves.

A CAVEAT outcome is acceptable and reportable (the prove-not-hype stance): it would mean "reuse
transfers for citation-structure drift but the safe-reuse window depends on what the GNN learns."

> **OUTCOME: CONFIRM (with a degeneracy caveat)** (2026-06-04) — see
> [ADR-202 addendum](../../adr/ADR-202-reuse-under-drift-real-gnn-trajectory.md#addendum-2026-06-04-objective-dependence--generality-confirmed-with-a-degeneracy-caveat).
> Node-class holding ceiling = **54% churn** (≥ 30%, *above* link-prediction's 40%) → generality
> confirmed across two objectives. Surfaced a real finding: past ~60% churn node-classification
> collapses embeddings into ~40 class blobs where recall@10 is ill-posed and the *rebuild baseline
> itself* destabilizes — so the trajectory-wide "reuse > rebuild" is a degeneracy artifact, not a
> claim. Reported as such, not as a flattering headline.
